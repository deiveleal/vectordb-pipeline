"""Camada gold: arxiv.silver -> arxiv.gold (s3://datalake-gold/iceberg/arxiv).

Aplica as regras de negocio sobre a silver de um dia:

1. So artigos novos: uma versao so (v1) e v1 submetida nos ultimos
   JANELA_ARTIGO_NOVO_DIAS dias. O OAI devolve tudo que mudou no dia --
   inclusive artigo antigo revisado ou com metadado corrigido -- e isso fica
   de fora.
2. So os temas de interesse, por padroes em titulo + abstract, na ordem de
   prioridade RAG > Agents > LLM: artigo de RAG que tambem fala de LLM fica so
   em RAG. Artigo sem tema nenhum fica de fora.

A tabela e uma so, com upsert por arxiv_id: o mesmo artigo visto em dois dias
(v1 anunciada num dia, metadado corrigido no outro) continua sendo uma linha.
Rodar o mesmo dia de novo nao muda nada. `datestamp` e o dia mais recente em
que o artigo apareceu no OAI, e particiona a tabela como na silver
(iceberg/arxiv/AAAA/MM/DD/data/) -- um artigo revisto num dia posterior muda
de particao.

A classificacao e heuristica, calibrada em 2026-09-26 sobre os 392 artigos
novos de 25/09 (RAG 7, Agents 62, LLM 170 antes da prioridade). Mudou um
padrao? Rode a gold de novo para os dias afetados: a silver guarda tudo.

Uso (da raiz do projeto): python -m src.load_gold
"""

import datetime as dt
import re
from collections import Counter

import pyarrow as pa

from src.load_silver import TABELA as TABELA_SILVER
from src.utils.load_catalog import NAMESPACE, abrir_ou_criar_tabela, load_catalog
from src.utils.load_logger import get_logger, setup_logging
from src.utils.load_time import dia_de_referencia

logger = get_logger(__name__)

BUCKET = "datalake-gold"
TABELA = "gold"  # arxiv.gold no catalogo

JANELA_ARTIGO_NOVO_DIAS = 7

# Siglas em maiusculas sao comparadas com caixa (RAG, LLM, GPT); palavras, sem
# caixa -- (?i:...) liga o ignore-case so no trecho.
TERMOS_LLM = re.compile(
    r"(?i:\blarge language models?\b|\blanguage models?\b|\bfoundation models?\b"
    r"|\bvision[- ]language models?\b|\bchatgpt\b)"
    r"|\bL?LMs?\b|\bVLMs?\b|\bGPT[- ]?\d*|\bLLaMA\b"
)
TERMOS_RAG = re.compile(r"(?i:retrieval[- ]augment\w*|\bgraphrag\b)|\bRAG\b")
TERMOS_AGENTIC = re.compile(r"(?i:\bagentic\b)")
TERMOS_AGENTE = re.compile(r"(?i:\bagents?\b|\bmulti[- ]agent\b)")


def classificar_tema(texto: str) -> tuple[str, str] | None:
    """Tema do artigo e o trecho que decidiu, ou None se nao for de interesse.

    Agents: "agentic" basta; "agent(s)" e "multi-agent" so contam junto com um
    termo de LLM -- sem isso, em cs.LG, e quase sempre aprendizado por reforco
    classico, nao agente de LLM.
    """
    if m := TERMOS_RAG.search(texto):
        return "rag", m.group(0)
    if m := TERMOS_AGENTIC.search(texto):
        return "agents", m.group(0)
    llm = TERMOS_LLM.search(texto)
    if (m := TERMOS_AGENTE.search(texto)) and llm:
        return "agents", f"{m.group(0)} + {llm.group(0)}"
    if llm:
        return "llm", llm.group(0)
    return None


SCHEMA = pa.schema([
    pa.field("arxiv_id", pa.string(), nullable=False),
    pa.field("tema", pa.string(), nullable=False),
    pa.field("evidencia", pa.string()),  # trecho que decidiu o tema, para auditar
    pa.field("titulo", pa.string()),
    pa.field("abstract", pa.string()),
    pa.field("autores_lista", pa.list_(pa.string())),
    pa.field("categorias", pa.list_(pa.string())),
    pa.field("categoria_principal", pa.string()),
    pa.field("data_v1", pa.timestamp("us", tz="UTC")),
    pa.field("datestamp", pa.date32(), nullable=False),
    pa.field("pdf_url", pa.string()),
])


class LoadGold:
    """Filtro de um dia da silver para a tabela Iceberg arxiv.gold.

    Uma instancia trata um unico dia. Sem argumento, e o dia_de_referencia();
    para reprocessar outra data, passe `dia` -- e o que o orquestrador usa.
    """

    def __init__(self, dia: dt.date | None = None):
        self.dia_alvo = dia or dia_de_referencia()
        self.catalog = load_catalog()

    def ler_silver(self) -> list[dict]:
        """A particao do dia na arxiv.silver."""
        silver = self.catalog.load_table(f"{NAMESPACE}.{TABELA_SILVER}")
        filtro = f"datestamp = '{self.dia_alvo.isoformat()}'"
        return silver.scan(row_filter=filtro).to_arrow().to_pylist()

    def e_artigo_novo(self, artigo: dict) -> bool:
        """So v1, submetida ate JANELA_ARTIGO_NOVO_DIAS antes do dia."""
        if artigo["n_versoes"] != 1 or artigo["data_v1"] is None:
            return False
        idade = (self.dia_alvo - artigo["data_v1"].date()).days
        return 0 <= idade <= JANELA_ARTIGO_NOVO_DIAS

    def run(self) -> dict[str, int]:
        """Filtra, classifica e faz upsert. Devolve quantos artigos por tema.

        Dia sem artigo de interesse devolve {} -- vazio, entao falso para quem
        so quer saber se ha o que baixar.
        """
        silver = self.ler_silver()
        novos = [a for a in silver if self.e_artigo_novo(a)]

        linhas = []
        for artigo in novos:
            texto = f"{artigo['titulo'] or ''} {artigo['abstract'] or ''}"
            if (classificacao := classificar_tema(texto)) is None:
                continue
            tema, evidencia = classificacao
            linhas.append({
                **{campo: artigo[campo] for campo in SCHEMA.names if campo in artigo},
                "tema": tema,
                "evidencia": evidencia,
            })

        por_tema = Counter(linha["tema"] for linha in linhas)
        logger.info("Gold %s: silver %d -> novos %d -> de interesse %d %s",
                    self.dia_alvo, len(silver), len(novos), len(linhas), dict(por_tema))
        if not linhas:
            logger.warning("Nenhum artigo de interesse em %s -- nada gravado.", self.dia_alvo)
            return {}

        tabela = abrir_ou_criar_tabela(
            self.catalog, TABELA, SCHEMA, BUCKET, particao="datestamp", chave="arxiv_id")
        resultado = tabela.upsert(pa.Table.from_pylist(linhas, schema=SCHEMA))
        logger.info("Upsert na arxiv.%s: %d inseridos, %d atualizados",
                    TABELA, resultado.rows_inserted, resultado.rows_updated)
        return dict(por_tema)


if __name__ == "__main__":
    setup_logging()
    gold_loader = LoadGold()
    gold_loader.run()
