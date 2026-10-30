"""Camada de chunks: TEI XML do datalake-documents -> tabela Iceberg arxiv.chunks.

Para cada artigo da gold no dia, le o .tei.xml que o GROBID gerou e quebra o
texto em trechos para o embedding. A tabela fica em
s3://datalake-chunks/iceberg/arxiv, particionada pela data da v1 (dia_v1) --
a mesma data da pasta do PDF, e que nunca muda para um artigo.

Estrategia (versao_chunker "secoes-v1"), guiada pela estrutura do artigo:

- chunk 0 e o abstract: o melhor resumo do artigo, e o que mais casa com busca
  generica. Abstract longo vira mais de um chunk.
- o corpo e quebrado por secao: paragrafos juntados ate ALVO_TOKENS, nunca
  cortando frase (o GROBID separa as frases em <s>). Paragrafo maior que o alvo
  e quebrado por frase; entre dois chunks da mesma secao repete-se a ultima
  frase, para o trecho nao comecar sem contexto.
- subsecoes sem numero ("Mitigating reward hacking." como titulo de
  paragrafo) contam como parte da secao numerada anterior: podem dividir chunk.
  Sem isso, artigo com 47 cabecalhos desses viraria 47 chunks minusculos.
- legendas de figura e tabela viram chunks proprios (secao "Figures and
  tables"): costumam resumir os resultados.
- ficam de fora referencias, agradecimentos, anexos e declaracoes
  (financiamento, uso de IA, conflito de interesse): so poluem a busca.

Cada chunk guarda `texto` (o trecho limpo, que a API mostra) e
`texto_embedding` (o mesmo com um cabecalho "Title: ... / Section: ...", que
e o que vira vetor -- o trecho sozinho perde contexto). Tokens contados com o
tokenizador do modelo de embedding (MODELO_TOKENIZADOR): ele corta sem aviso o
que passa de 512, entao o limite e verificado aqui, nao estimado.

Rodar de novo um artigo substitui todos os chunks dele num commit so
(overwrite filtrado por arxiv_id): se ele passar de 20 para 18 chunks, os 2 a
mais somem, o que um upsert por chunk_id nao faria.

Uso (da raiz do projeto): python -m src.load_chunks
"""

import datetime as dt
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

import pyarrow as pa
from tokenizers import Tokenizer

from src.load_documents import BUCKET as BUCKET_DOCUMENTS
from src.load_documents import localizar_pdf
from src.load_gold import TABELA as TABELA_GOLD
from src.load_text import chave_tei
from src.utils.load_catalog import NAMESPACE, abrir_ou_criar_tabela, load_catalog
from src.utils.load_logger import get_logger, setup_logging
from src.utils.load_time import dia_de_referencia
from src.utils.storage_functions import StorageFuncions

logger = get_logger(__name__)

BUCKET = "datalake-chunks"
TABELA = "chunks"  # arxiv.chunks no catalogo
# v1 -> v2 (2026-09-26): corta os apendices, que o GROBID deixa no corpo com
# titulo sem numero e que a v1 pendurava na conclusao (ate 134 chunks num
# artigo so, com prompts e provas).
VERSAO_CHUNKER = "secoes-v2"

# O tokenizador tem que ser o do modelo de embedding: trocar o modelo pede
# trocar aqui e refazer os chunks (versao_chunker nova).
MODELO_TOKENIZADOR = "BAAI/bge-small-en-v1.5"
LIMITE_MODELO = 512           # tokens que o modelo aceita, com [CLS] e [SEP]
ALVO_TOKENS = 350             # tamanho-alvo do trecho, sem o cabecalho
MAX_SOBREPOSICAO_TOKENS = 60  # frase repetida entre chunks so se for curta

NS = {"t": "http://www.tei-c.org/ns/1.0"}

# Secoes que nao entram. Casa com o titulo da secao, sem caixa.
SECOES_DESCARTADAS = re.compile(
    r"acknowledg|funding|author contributions?|competing interests?|"
    r"conflicts? of interest|ethics statement|data availability|"
    r"genai|generative ai|use of (ai|llms?)|ai use|llm usage|disclosure|"
    r"references|bibliography",
    re.IGNORECASE,
)

# O GROBID nem sempre separa os apendices (<back type="annex">): muitos ficam
# no corpo, com titulo sem numero. Medido em 25/09: provas ("EC.1.3. Proof of
# Lemma 3"), prompts ("Listing 5: ..."), detalhes ("A HARNESS DETAILS"). O
# corpo acaba no primeiro titulo com cara de apendice ...
INICIO_DE_APENDICE = re.compile(
    r"^(appendix|appendices|supplementary|supplemental|listing \d)", re.IGNORECASE)
# So vale em titulo curto: o GROBID ja tomou por titulo a frase "Appendix G
# describes the additional ...".
MAX_PALAVRAS_TITULO_APENDICE = 8
# Letra de apendice, COM caixa: sem ela, [A-Z] casaria "We ..." e "In ...".
# Sem ponto logo depois da letra: "A. URL Allowlist" e subsecao no estilo
# IEEE, nao apendice.
LETRA_DE_APENDICE = re.compile(
    r"^[A-Z]\s+[A-Z]"                     # "A HARNESS", "B MODEL", "G Tool Declarations"
    r"|^[A-Z]{1,2}(\.\d+)+\.?\s"          # "A.5 RUN", "EC.1.2. Proof of Lemma 3"
)
# Secao IEEE ("I. INTRODUCTION", "IV. RESULTS") que o GROBID deixa sem numero:
# o numeral vira o numero da secao. Sem isto, o "I" parecia letra de apendice
# e artigos inteiros ficavam sem corpo (12 de 180 em 25/09).
NUMERAL_ROMANO = re.compile(r"^([IVX]{1,4})\.\s+(.*)$")
# ... ou no primeiro titulo sem numero depois da conclusao -- menos estes, que
# em artigo da ACL vem sem numero depois dela e sao parte do texto.
CONCLUSAO = re.compile(r"conclu", re.IGNORECASE)
PERMITIDAS_APOS_CONCLUSAO = re.compile(r"^limitations?\b", re.IGNORECASE)

# Titulo gigante (prompt que o GROBID tomou por cabecalho) estouraria o limite
# de tokens do chunk pelo cabecalho de contexto.
MAX_CHARS_SECAO = 100

SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string(), nullable=False),   # <arxiv_id>#<ordem>
    pa.field("arxiv_id", pa.string(), nullable=False),
    pa.field("tema", pa.string()),
    pa.field("titulo", pa.string()),
    pa.field("secao", pa.string()),
    pa.field("ordem", pa.int32()),
    pa.field("texto", pa.string()),
    pa.field("texto_embedding", pa.string()),
    pa.field("n_tokens", pa.int32()),                    # de texto_embedding, com [CLS]/[SEP]
    pa.field("dia_v1", pa.date32(), nullable=False),     # particao
    pa.field("chave_pdf", pa.string()),
    pa.field("chave_tei", pa.string()),
    pa.field("versao_chunker", pa.string()),
])


def _texto(elemento: ET.Element | None) -> str:
    """Texto do elemento com espaco normalizado."""
    return " ".join("".join(elemento.itertext()).split()) if elemento is not None else ""


TOKEN_NUMERICO = re.compile(r"[\d.,()%±+\-–/:;\[\]]*\d[\d.,()%±+\-–/:;\[\]]*")
MIN_SEQUENCIA_NUMERICA = 4


def _sem_celulas_de_tabela(frase: str) -> str:
    """Tira sequencias de MIN_SEQUENCIA_NUMERICA+ numeros seguidos.

    Em PDF de duas colunas o GROBID as vezes joga celulas de tabela dentro do
    paragrafo vizinho: "ASVspoof2021 DF 7.02 (6.40,7.50) 19.20 (18.96,19.52)
    ... addition, we trained". Numero solto ("5,000 attempted rollouts") fica;
    so a rajada sai. Heuristica: sobra lixo curto como "ASVspoof2021 DF".
    """
    palavras = frase.split()
    manter = [True] * len(palavras)
    inicio = None
    for i, palavra in enumerate(palavras + [""]):  # "" fecha a ultima sequencia
        if palavra and TOKEN_NUMERICO.fullmatch(palavra):
            inicio = i if inicio is None else inicio
            continue
        if inicio is not None and i - inicio >= MIN_SEQUENCIA_NUMERICA:
            manter[inicio:i] = [False] * (i - inicio)
        inicio = None
    return " ".join(p for p, m in zip(palavras, manter) if m)


def _frases(paragrafo: ET.Element) -> list[str]:
    """Frases do paragrafo, sem rajadas de celula de tabela.

    Juntar o texto do <p> direto colaria uma frase na outra ("split.For"): o
    GROBID nao poe espaco entre os <s>. Frase que fica com menos de 3 palavras
    depois da limpeza e descartada.
    """
    frases = [_texto(s) for s in paragrafo.findall("t:s", NS)] or [_texto(paragrafo)]
    limpas = (_sem_celulas_de_tabela(f) for f in frases if f)
    return [f for f in limpas if len(f.split()) >= 3]


@dataclass
class Bloco:
    """Um paragrafo (ou legenda), ja em frases, com a secao a que pertence."""
    grupo: str          # secao de primeiro nivel: chunks nao atravessam grupos
    secao: str          # rotulo completo, com subsecao quando houver
    frases: list[str]


def ler_tei(tei: bytes) -> tuple[list[str], list[Bloco]]:
    """Frases do abstract e blocos do corpo + legendas, na ordem do artigo."""
    raiz = ET.fromstring(tei)

    abstract = [f for p in raiz.findall(".//t:profileDesc/t:abstract//t:p", NS)
                for f in _frases(p)]

    blocos: list[Bloco] = []
    corpo = raiz.find(".//t:text/t:body", NS)
    topo: str | None = None          # secao numerada corrente
    descartando = False
    passou_da_conclusao = False
    for div in (corpo.findall("t:div", NS) if corpo is not None else []):
        cabecalho = div.find("t:head", NS)
        numero = cabecalho.get("n") if cabecalho is not None else None
        titulo = _texto(cabecalho)[:MAX_CHARS_SECAO]
        if not numero and (romano := NUMERAL_ROMANO.match(titulo)):
            numero, titulo = f"{romano.group(1)}.", romano.group(2)

        if not numero and titulo and (
                (INICIO_DE_APENDICE.search(titulo)
                 and len(titulo.split()) <= MAX_PALAVRAS_TITULO_APENDICE)
                or LETRA_DE_APENDICE.search(titulo)
                or (passou_da_conclusao and not PERMITIDAS_APOS_CONCLUSAO.search(titulo))):
            break  # daqui para baixo e apendice
        if numero and CONCLUSAO.search(titulo):
            passou_da_conclusao = True

        if numero:
            topo = f"{numero} {titulo}".strip()
            grupo = secao = topo
            descartando = bool(SECOES_DESCARTADAS.search(titulo))
        elif topo and titulo:
            # titulo de paragrafo dentro da secao numerada corrente
            grupo, secao = topo, f"{topo} › {titulo}"
            if SECOES_DESCARTADAS.search(titulo):
                continue
        else:
            # artigo sem numeracao: cada cabecalho e uma secao
            grupo = secao = titulo or "Body"
            topo = None
            descartando = bool(SECOES_DESCARTADAS.search(titulo))
        if descartando:
            continue

        for paragrafo in div.findall("t:p", NS):
            if frases := _frases(paragrafo):
                blocos.append(Bloco(grupo, secao, frases))

    for figura in raiz.findall(".//t:text/t:body//t:figure", NS):
        rotulo = _texto(figura.find("t:head", NS))
        descricao = _texto(figura.find("t:figDesc", NS))
        # o GROBID manda "Figure 1 :" no rotulo e "Figure 1: ..." na descricao;
        # comparar so letras e numeros evita "Figure 1 : Figure 1: ..."
        so_alnum = lambda s: re.sub(r"\W", "", s).lower()  # noqa: E731
        if descricao and so_alnum(descricao).startswith(so_alnum(rotulo)):
            legenda = descricao
        else:
            legenda = f"{rotulo} {descricao}".strip()
        if legenda:
            blocos.append(Bloco("Figures and tables", "Figures and tables", [legenda]))

    return abstract, blocos


@dataclass
class Trecho:
    """Chunk em construcao."""
    grupo: str
    secoes: list[str] = field(default_factory=list)
    frases: list[str] = field(default_factory=list)
    tokens: int = 0


class Chunker:
    """Quebra um artigo em chunks, contando tokens com o tokenizador do modelo."""

    def __init__(self) -> None:
        self.tokenizador = Tokenizer.from_pretrained(MODELO_TOKENIZADOR)

    def tokens(self, texto: str) -> int:
        return len(self.tokenizador.encode(texto, add_special_tokens=False).ids)

    def _quebrar_frase_longa(self, frase: str) -> list[str]:
        """Frase maior que o alvo (tabela achatada, lista sem ponto) vira pedacos
        por palavra. Raro, mas sem isto o modelo cortaria o final sem aviso."""
        pedacos: list[str] = []
        atual: list[str] = []
        for palavra in frase.split():
            if atual and self.tokens(" ".join(atual + [palavra])) > ALVO_TOKENS:
                pedacos.append(" ".join(atual))
                atual = []
            atual.append(palavra)
        if atual:
            pedacos.append(" ".join(atual))
        return pedacos

    def empacotar(self, blocos: list[Bloco]) -> list[Trecho]:
        """Junta os blocos em trechos de ate ALVO_TOKENS, sem atravessar grupo."""
        trechos: list[Trecho] = []
        atual: Trecho | None = None

        def fechar() -> None:
            if atual is not None and atual.frases:
                trechos.append(atual)

        for bloco in blocos:
            frases = [p for f in bloco.frases
                      for p in ([f] if self.tokens(f) <= ALVO_TOKENS
                                else self._quebrar_frase_longa(f))]
            tokens_bloco = self.tokens(" ".join(frases))

            if atual is None or atual.grupo != bloco.grupo:
                fechar()
                atual = Trecho(bloco.grupo)

            if atual.tokens + tokens_bloco <= ALVO_TOKENS:
                # o paragrafo inteiro cabe
                atual.frases.extend(frases)
                atual.tokens += tokens_bloco
                if bloco.secao not in atual.secoes:
                    atual.secoes.append(bloco.secao)
                continue

            # nao cabe: vai frase a frase, abrindo trecho novo quando enche
            for frase in frases:
                tokens_frase = self.tokens(frase)
                if atual.frases and atual.tokens + tokens_frase > ALVO_TOKENS:
                    ultima = atual.frases[-1]
                    fechar()
                    atual = Trecho(bloco.grupo)
                    # repete a ultima frase do trecho anterior, se for curta e
                    # se couber junto com a frase nova no alvo
                    t = self.tokens(ultima)
                    if t <= MAX_SOBREPOSICAO_TOKENS and t + tokens_frase <= ALVO_TOKENS:
                        atual.frases.append(ultima)
                        atual.tokens += t
                atual.frases.append(frase)
                atual.tokens += tokens_frase
                if bloco.secao not in atual.secoes:
                    atual.secoes.append(bloco.secao)
        fechar()
        return trechos

    def chunks_do_artigo(self, artigo: dict, tei: bytes,
                         chave_pdf: str, chave_t: str) -> list[dict]:
        """Os chunks de um artigo, prontos para a tabela."""
        abstract, blocos = ler_tei(tei)
        trechos = []
        if abstract:
            trechos = self.empacotar([Bloco("Abstract", "Abstract", abstract)])
        trechos += self.empacotar(blocos)

        linhas = []
        for ordem, trecho in enumerate(trechos):
            # varias subsecoes num trecho: o rotulo e a secao de primeiro nivel
            secao = trecho.secoes[0] if len(trecho.secoes) == 1 else trecho.grupo
            texto = " ".join(trecho.frases)
            texto_embedding = f"Title: {artigo['titulo']}\nSection: {secao}\n\n{texto}"
            n_tokens = self.tokens(texto_embedding) + 2  # [CLS] e [SEP]
            if n_tokens > LIMITE_MODELO:
                logger.warning("%s#%d com %d tokens (> %d): o modelo vai cortar o fim",
                               artigo["arxiv_id"], ordem, n_tokens, LIMITE_MODELO)
            linhas.append({
                "chunk_id": f"{artigo['arxiv_id']}#{ordem:04d}",
                "arxiv_id": artigo["arxiv_id"],
                "tema": artigo["tema"],
                "titulo": artigo["titulo"],
                "secao": secao,
                "ordem": ordem,
                "texto": texto,
                "texto_embedding": texto_embedding,
                "n_tokens": n_tokens,
                "dia_v1": artigo["data_v1"].date(),
                "chave_pdf": chave_pdf,
                "chave_tei": chave_t,
                "versao_chunker": VERSAO_CHUNKER,
            })
        return linhas


class LoadChunks:
    """Chunks dos artigos de um dia da gold para a tabela Iceberg arxiv.chunks.

    Uma instancia trata um unico dia. Sem argumento, e o dia_de_referencia();
    para reprocessar outra data, passe `dia` -- e o que o orquestrador usa.
    """

    def __init__(self, dia: dt.date | None = None):
        self.storage = StorageFuncions()
        self.dia_alvo = dia or dia_de_referencia()
        self.catalog = load_catalog()

    def ler_gold(self) -> list[dict]:
        """Os artigos da particao do dia na arxiv.gold."""
        gold = self.catalog.load_table(f"{NAMESPACE}.{TABELA_GOLD}")
        filtro = f"datestamp = '{self.dia_alvo.isoformat()}'"
        colunas = ("arxiv_id", "tema", "titulo", "data_v1")
        return gold.scan(row_filter=filtro, selected_fields=colunas).to_arrow().to_pylist()

    def run(self, limite: int | None = None) -> dict[str, int]:
        """Gera e grava os chunks do dia. Devolve a contagem.

        `limite` corta a quantidade de artigos -- para teste. Artigo sem TEI
        (extracao ainda nao rodou ou falhou) e pulado e contado a parte.
        """
        artigos = self.ler_gold()[:limite]
        chunker = Chunker()
        situacao: Counter[str] = Counter()
        linhas: list[dict] = []

        for artigo in artigos:
            chave_pdf = localizar_pdf(self.storage, artigo)
            if chave_pdf is None:
                situacao["sem_texto"] += 1
                continue
            chave_t = chave_tei(chave_pdf)
            if not self.storage.exists(BUCKET_DOCUMENTS, chave_t):
                situacao["sem_texto"] += 1
                continue
            try:
                tei = self.storage.load_data(BUCKET_DOCUMENTS, chave_t)["Body"].read()
                do_artigo = chunker.chunks_do_artigo(artigo, tei, chave_pdf, chave_t)
            except Exception:
                situacao["falha"] += 1
                logger.exception("Chunks de %s falharam -- seguindo para o proximo",
                                 artigo["arxiv_id"])
                continue
            linhas.extend(do_artigo)
            situacao["artigos"] += 1

        situacao["chunks"] = len(linhas)
        if linhas:
            tabela = abrir_ou_criar_tabela(
                self.catalog, TABELA, SCHEMA, BUCKET, particao="dia_v1")
            ids = sorted({linha["arxiv_id"] for linha in linhas})
            filtro = "arxiv_id IN (" + ", ".join(f"'{i}'" for i in ids) + ")"
            tabela.overwrite(pa.Table.from_pylist(linhas, schema=SCHEMA),
                             overwrite_filter=filtro)

        logger.info("Chunks %s: %d artigos na gold -> %s",
                    self.dia_alvo, len(artigos), dict(situacao))
        if artigos and situacao["falha"] == len(artigos):
            raise RuntimeError(
                f"nenhum dos {len(artigos)} artigos virou chunk -- ver os erros acima")
        return dict(situacao)


if __name__ == "__main__":
    setup_logging()
    chunks_loader = LoadChunks()
    chunks_loader.run()
