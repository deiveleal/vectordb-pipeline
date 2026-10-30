"""Camada de texto: PDFs do datalake-documents -> TEI XML ao lado de cada PDF.

Para cada artigo da gold no dia (particao datestamp), manda o PDF ao GROBID e
grava a resposta como <mesmo nome>.tei.xml, na mesma pasta:

    arxiv/rag/2026/09/23/The Fellowship of the Query.pdf
    arxiv/rag/2026/09/23/The Fellowship of the Query.tei.xml

O TEI traz a estrutura do artigo -- titulo, abstract, secoes, paragrafos
separados em frases (<s>), legendas e, a parte, agradecimentos e referencias.
E dele que o chunker parte. Extrair e a etapa mais cara do pipeline (~4s por
PDF no GROBID em CPU), por isso o resultado fica guardado: mudar o chunker nao
exige extrair de novo. Fica no datalake-documents porque tem a mesma natureza
do PDF: caro de produzir e imutavel (bucket versionado, sem DeleteObject).

Artigo que ja tem .tei.xml e pulado. Artigo sem PDF (download falhou ou ainda
nao rodou) tambem, contado a parte -- o proximo run pega.

Uso (da raiz do projeto): python -m src.load_text
"""

import datetime as dt
import os
import time
from collections import Counter

import requests

from src.load_documents import BUCKET, localizar_pdf
from src.load_gold import TABELA as TABELA_GOLD
from src.utils.load_catalog import NAMESPACE, load_catalog
from src.utils.load_env import load_env
from src.utils.load_logger import get_logger, setup_logging
from src.utils.load_time import dia_de_referencia
from src.utils.oai_connect import (
    ESPERA_INICIAL_S,
    FATOR_DE_BACKOFF,
    TENTATIVAS,
    pausa_ate_a_proxima,
    vale_repetir,
)
from src.utils.storage_functions import StorageFuncions

logger = get_logger(__name__)

# Endereco visto do HOST; em container da rede do compose, http://grobid:8070.
GROBID_PADRAO = "http://localhost:8070"

# consolidate*=0: nao consultar CrossRef e afins. Mantem a extracao local,
# rapida e sem depender de servico externo -- os metadados certos ja vem da
# silver. segmentSentences=1: frases em <s>, o que o chunker usa para nunca
# cortar no meio de uma frase.
PARAMETROS_GROBID = {
    "consolidateHeader": "0",
    "consolidateCitations": "0",
    "segmentSentences": "1",
}


def chave_tei(chave_pdf: str) -> str:
    """O TEI mora ao lado do PDF, com o mesmo nome."""
    return chave_pdf.removesuffix(".pdf") + ".tei.xml"


class LoadText:
    """Extracao de texto dos PDFs de um dia da gold.

    Uma instancia trata um unico dia. Sem argumento, e o dia_de_referencia();
    para reprocessar outra data, passe `dia` -- e o que o orquestrador usa.
    """

    def __init__(self, dia: dt.date | None = None):
        load_env()
        self.storage = StorageFuncions()
        self.dia_alvo = dia or dia_de_referencia()
        self.url = os.environ.get("GROBID_URL", GROBID_PADRAO).rstrip("/")

    def ler_gold(self) -> list[dict]:
        """Os artigos da particao do dia na arxiv.gold."""
        gold = load_catalog().load_table(f"{NAMESPACE}.{TABELA_GOLD}")
        filtro = f"datestamp = '{self.dia_alvo.isoformat()}'"
        colunas = ("arxiv_id", "tema", "titulo", "data_v1")
        return gold.scan(row_filter=filtro, selected_fields=colunas).to_arrow().to_pylist()

    def extrair(self, pdf: bytes) -> bytes:
        """Manda o PDF ao GROBID e devolve o TEI XML.

        O GROBID responde 503 quando a fila dele esta cheia -- transitorio,
        entra no mesmo retry do OAI. Timeout largo: PDF grande pode levar
        dezenas de segundos em CPU.
        """
        espera = ESPERA_INICIAL_S
        for tentativa in range(1, TENTATIVAS + 1):
            try:
                resposta = requests.post(
                    f"{self.url}/api/processFulltextDocument",
                    files={"input": ("artigo.pdf", pdf, "application/pdf")},
                    data=PARAMETROS_GROBID,
                    timeout=300,
                )
                resposta.raise_for_status()
                return resposta.content
            except requests.RequestException as erro:
                if tentativa == TENTATIVAS or not vale_repetir(erro):
                    raise
                descanso = pausa_ate_a_proxima(erro, espera)
                logger.warning("GROBID recusou (%s). Tentativa %d de %d -- "
                               "nova tentativa em %.0fs.",
                               erro, tentativa, TENTATIVAS, descanso)
                time.sleep(descanso)
                espera *= FATOR_DE_BACKOFF
        raise RuntimeError("laco de tentativas terminou sem resposta nem erro")

    def run(self, limite: int | None = None) -> dict[str, int]:
        """Extrai o que falta do dia. Devolve a contagem por situacao.

        `limite` corta a quantidade de artigos -- para teste. Falha em um
        artigo nao para os outros; so o dia inteiro falhando vira excecao.
        """
        artigos = self.ler_gold()[:limite]
        situacao: Counter[str] = Counter()

        for artigo in artigos:
            chave_pdf = localizar_pdf(self.storage, artigo)
            if chave_pdf is None:
                situacao["sem_pdf"] += 1
                logger.warning("%s sem PDF no bucket -- fica para o proximo run",
                               artigo["arxiv_id"])
                continue
            key = chave_tei(chave_pdf)
            if self.storage.exists(BUCKET, key):
                situacao["ja_existia"] += 1
                continue
            try:
                t0 = time.monotonic()
                pdf = self.storage.load_data(BUCKET, chave_pdf)["Body"].read()
                tei = self.extrair(pdf)
                self.storage.save_data(
                    data=tei, bucket_name=BUCKET, key=key,
                    content_type="application/xml",
                    metadata={"arxiv-id": artigo["arxiv_id"], "tema": artigo["tema"]},
                )
            except Exception:
                situacao["falha"] += 1
                logger.exception("Extracao de %s falhou -- seguindo para o proximo",
                                 artigo["arxiv_id"])
                continue
            situacao["extraido"] += 1
            logger.info("Gravado %s (%.1fs)", key, time.monotonic() - t0)

        logger.info("Text %s: %d artigos na gold -> %s",
                    self.dia_alvo, len(artigos), dict(situacao))
        if artigos and situacao["falha"] == len(artigos):
            raise RuntimeError(
                f"nenhum dos {len(artigos)} textos foi extraido -- ver os erros acima")
        return dict(situacao)


if __name__ == "__main__":
    setup_logging()
    text_loader = LoadText()
    text_loader.run()
