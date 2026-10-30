"""Camada documents: arxiv.gold -> datalake-documents.

Baixa o PDF de cada artigo da gold no dia (particao datestamp) e grava em
arxiv/<tema>/AAAA/MM/DD/<titulo sem subtitulo>.pdf -- por exemplo, o artigo
"CRISS: A Retrieval-Augmented AI Chatbot for ..." vira
arxiv/rag/2026/09/23/CRISS.pdf.

A data da chave e a da v1 (data_v1), nao o datestamp: a v1 nunca muda, entao
a chave de um artigo e sempre a mesma. Isso e o que permite pular o que ja foi
baixado com um HEAD so. Com o datestamp, o mesmo artigo visto de novo num dia
posterior iria para outra pasta e seria baixado outra vez. Separar por tema ja
deixa os PDFs agrupados pela collection do Qdrant em que vao entrar.

Titulo nao e unico ("RAG: ..." e "RAG: ..." dariam o mesmo nome), entao o
arxiv_id vai nos metadados do objeto (x-amz-meta-arxiv-id). Achou o nome ja
ocupado: se o metadado e o mesmo artigo, pula; se e outro, este ganha o id no
nome -- "RAG (2609.12345).pdf". Titulo corrigido depois no arXiv muda o nome e
baixa de novo; e raro o bastante para nao valer uma tabela de controle.

E a camada que faz N chamadas externas por execucao, entao e a que mais falha
por motivo alheio ao codigo -- PDF ainda nao gerado, rede oscilando, arXiv
lento. O laco sobrevive a isso: falha de um artigo nao para os outros, e o
proximo run tenta de novo so os que faltam. O bucket e versionado e o usuario
de servico nao tem DeleteObject: PDF gravado nao se perde.

Uso (da raiz do projeto): python -m src.load_documents
"""

import datetime as dt
import re
import time
from collections import Counter

import requests

from src.load_gold import TABELA as TABELA_GOLD
from src.utils.load_catalog import NAMESPACE, load_catalog
from src.utils.load_logger import get_logger, setup_logging
from src.utils.load_time import dia_de_referencia
from src.utils.oai_connect import (
    ESPERA_INICIAL_S,
    FATOR_DE_BACKOFF,
    PAUSA_ENTRE_CHAMADAS_S,
    TENTATIVAS,
    pausa_ate_a_proxima,
    vale_repetir,
)
from src.utils.storage_functions import StorageFuncions

logger = get_logger(__name__)

BUCKET = "datalake-documents"

# Teto do nome: a chave do S3 aguenta 1024 bytes, mas sistema de arquivo e
# ferramenta de download sofrem bem antes disso.
TAMANHO_MAXIMO_NOME = 150


def nome_de_arquivo(titulo: str | None) -> str:
    """Titulo sem subtitulo, seguro como nome de arquivo. Vazio se nao sobrar nada.

    O subtitulo e o que vem depois do primeiro ":" ou do primeiro "?" que
    ainda tem texto depois -- "Return or Revise? Learning When..." vira
    "Return or Revise". Titulo que termina em "?" fica inteiro. Saem tambem os
    caracteres que quebram caminho ("/" viraria pasta no S3) ou nome de
    arquivo no Windows (<>:"\\|?*), e a marcacao LaTeX que o arXiv deixa no
    titulo ("$k$-NN", "\\textbf{...}").
    """
    titulo = re.split(r":|\?(?=\s*\S)", titulo or "", maxsplit=1)[0]
    titulo = re.sub(r"\\[a-zA-Z]+", "", titulo)       # comandos LaTeX
    titulo = re.sub(r'[$\\{}<>"|?*]', "", titulo)
    titulo = titulo.replace("/", "-")
    titulo = " ".join(titulo.split()).strip(" .")
    if len(titulo) > TAMANHO_MAXIMO_NOME:
        titulo = titulo[:TAMANHO_MAXIMO_NOME].rsplit(" ", 1)[0]
    return titulo


def chave_documento(tema: str, data_v1: dt.datetime, nome: str) -> str:
    """Chave do PDF: arxiv/<tema>/AAAA/MM/DD/<nome>.pdf, com a data da v1."""
    return f"arxiv/{tema}/{data_v1.strftime('%Y/%m/%d')}/{nome}.pdf"


def chaves_candidatas(artigo: dict) -> list[str]:
    """As chaves que o PDF do artigo pode ter, na ordem em que sao tentadas.

    Primeiro o titulo; "titulo (arxiv_id)" so quando o titulo ja era de outro
    artigo. Artigo precisa de arxiv_id, tema, titulo e data_v1.
    """
    # arxiv_id antigo tem barra (cs/0112017) e viraria pasta
    id_seguro = artigo["arxiv_id"].replace("/", "_")
    nome = nome_de_arquivo(artigo["titulo"]) or id_seguro
    return [chave_documento(artigo["tema"], artigo["data_v1"], candidato)
            for candidato in (nome, f"{nome} ({id_seguro})")]


def localizar_pdf(storage: StorageFuncions, artigo: dict) -> str | None:
    """Chave do PDF ja baixado do artigo, ou None se ele ainda nao esta no bucket.

    Confere o metadado arxiv-id: o nome sozinho pode ser de outro artigo com o
    mesmo titulo. E como as etapas seguintes (texto, chunks) acham o PDF.
    """
    for key in chaves_candidatas(artigo):
        metadados = storage.head(BUCKET, key)
        if metadados is not None and metadados.get("arxiv-id") == artigo["arxiv_id"]:
            return key
    return None


class PDFIndisponivel(Exception):
    """O arXiv respondeu, mas nao com um PDF.

    Artigo recem-anunciado as vezes ainda nao tem PDF gerado e a URL devolve
    uma pagina HTML com status 200. Nao e erro de rede -- repetir agora nao
    adianta, o proximo run pega.
    """


class LoadDocuments:
    """Download dos PDFs de um dia da gold.

    Uma instancia trata um unico dia. Sem argumento, e o dia_de_referencia();
    para reprocessar outra data, passe `dia` -- e o que o orquestrador usa.
    """

    # mesma pausa do OAI, controlada pela classe: vale entre artigos e
    # entre instancias
    _ultima_chamada = 0.0

    def __init__(self, dia: dt.date | None = None):
        self.storage = StorageFuncions()
        self.dia_alvo = dia or dia_de_referencia()

    def ler_gold(self) -> list[dict]:
        """Os artigos da particao do dia na arxiv.gold."""
        gold = load_catalog().load_table(f"{NAMESPACE}.{TABELA_GOLD}")
        filtro = f"datestamp = '{self.dia_alvo.isoformat()}'"
        colunas = ("arxiv_id", "tema", "titulo", "data_v1", "pdf_url")
        return gold.scan(row_filter=filtro, selected_fields=colunas).to_arrow().to_pylist()

    @classmethod
    def _esperar_a_vez(cls) -> None:
        falta = PAUSA_ENTRE_CHAMADAS_S - (time.monotonic() - cls._ultima_chamada)
        if falta > 0:
            time.sleep(falta)
        cls._ultima_chamada = time.monotonic()

    def baixar_pdf(self, url: str) -> bytes:
        """Baixa o PDF, com o mesmo retry do OAI no que for transitorio."""
        espera = ESPERA_INICIAL_S
        for tentativa in range(1, TENTATIVAS + 1):
            self._esperar_a_vez()
            try:
                resposta = requests.get(url, timeout=60)
                resposta.raise_for_status()
                break
            except requests.RequestException as erro:
                if tentativa == TENTATIVAS or not vale_repetir(erro):
                    raise
                descanso = pausa_ate_a_proxima(erro, espera)
                logger.warning("Download de %s falhou (%s). Tentativa %d de %d -- "
                               "nova tentativa em %.0fs.",
                               url, erro, tentativa, TENTATIVAS, descanso)
                time.sleep(descanso)
                espera *= FATOR_DE_BACKOFF

        if not resposta.content.startswith(b"%PDF"):
            raise PDFIndisponivel(
                f"{url} respondeu {resposta.headers.get('Content-Type')}, nao PDF")
        return resposta.content

    def chave_livre(self, artigo: dict) -> str | None:
        """Chave onde gravar o PDF do artigo, ou None se ele ja foi baixado.

        Tenta o titulo; se o nome ja e de outro artigo (metadado arxiv-id
        diferente), tenta "titulo (arxiv_id)", que e unico.
        """
        for key in chaves_candidatas(artigo):
            metadados = self.storage.head(BUCKET, key)
            if metadados is None:
                return key
            if metadados.get("arxiv-id") == artigo["arxiv_id"]:
                return None
            logger.info("Nome %s ja e de %s; usando o id no nome",
                        key, metadados.get("arxiv-id"))
        return None  # "titulo (id)" ocupado so pode ser o proprio artigo

    def run(self, limite: int | None = None) -> dict[str, int]:
        """Baixa o que falta do dia. Devolve a contagem por situacao.

        `limite` corta a quantidade de artigos -- para teste. Falha em um
        artigo nao para os outros; so o dia inteiro falhando vira excecao.
        """
        artigos = self.ler_gold()[:limite]
        situacao: Counter[str] = Counter()

        for artigo in artigos:
            key = self.chave_livre(artigo)
            if key is None:
                situacao["ja_existia"] += 1
                continue
            try:
                pdf = self.baixar_pdf(artigo["pdf_url"])
                self.storage.save_data(
                    data=pdf, bucket_name=BUCKET, key=key,
                    content_type="application/pdf",
                    metadata={"arxiv-id": artigo["arxiv_id"], "tema": artigo["tema"]},
                )
            except PDFIndisponivel as erro:
                situacao["indisponivel"] += 1
                logger.warning("%s -- tenta de novo no proximo run", erro)
                continue
            except Exception:
                situacao["falha"] += 1
                logger.exception("PDF de %s falhou -- seguindo para o proximo",
                                 artigo["arxiv_id"])
                continue
            situacao["baixado"] += 1
            logger.info("Gravado %s (%d KB)", key, len(pdf) // 1024)

        logger.info("Documents %s: %d artigos na gold -> %s",
                    self.dia_alvo, len(artigos), dict(situacao))
        if artigos and situacao["falha"] == len(artigos):
            raise RuntimeError(
                f"nenhum dos {len(artigos)} PDFs foi gravado -- ver os erros acima")
        return dict(situacao)


if __name__ == "__main__":
    setup_logging()
    document_loader = LoadDocuments()
    document_loader.run()
