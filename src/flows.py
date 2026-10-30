"""Orquestracao do pipeline com Prefect.

Uso (da raiz do projeto):
    uv run prefect server start                  # terminal 1: UI em :4200
    export PREFECT_API_URL=http://127.0.0.1:4200/api
    uv run python -m src.flows                   # terminal 2: agenda e espera

Reprocessar um dia, ou testar com poucos PDFs:
    uv run prefect deployment run 'arxiv_data_pipeline/diaria' -p dia=2026-09-22
    uv run prefect deployment run 'arxiv_data_pipeline/diaria' -p limite_pdfs=5

Cada execucao deixa na UI um artifact "resumo-arxiv" com o que cada camada fez.
"""

import datetime as dt
import logging
import os

# Antes de importar o Prefect: e na importacao que ele le a configuracao. Sem
# isto, os logs das camadas (logger "src") nao chegam a UI -- so os do proprio
# Prefect. setdefault: um valor no ambiente continua valendo.
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "src")

from prefect import flow, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.schedules import Cron  # noqa: E402

from src.backup import Backup  # noqa: E402
from src.load_bronze import LoadBronze  # noqa: E402
from src.load_chunks import LoadChunks  # noqa: E402
from src.load_documents import LoadDocuments  # noqa: E402
from src.load_embeddings import LoadEmbeddings  # noqa: E402
from src.load_gold import LoadGold  # noqa: E402
from src.load_silver import LoadSilver  # noqa: E402
from src.load_text import LoadText  # noqa: E402
from src.utils.load_time import dia_de_referencia  # noqa: E402

# O Prefect encaminha o logger "src" para a UI, mas nao mexe no nivel dele: sem
# isto fica o WARNING herdado do root, e os INFO das camadas se perdem. O
# setup_logging() nao serve aqui -- so roda no entry point de cada camada, e o
# flow roda num subprocesso do .serve() que nao passa por ele.
logging.getLogger("src").setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


# O OAIConnect ja repete 4x em ~65s. O retry daqui cobre queda mais longa do
# arXiv (instabilidade): tenta de novo 10 min depois.
@task(retries=1, retry_delay_seconds=600)
def run_load_bronze(dia: dt.date | None) -> list[str]:
    """Executa a coleta e gravacao do bronze. Devolve as categorias com registros."""
    return LoadBronze(dia).run()


# Sem retry: le e grava no RustFS local. Erro de parse ou manifesto ausente
# nao muda com o tempo, e repetir so atrasaria a falha.
@task
def run_load_silver(dia: dt.date | None, categorias: list[str]) -> int:
    """Executa o parse do bronze para o silver. Devolve o numero de artigos."""
    return LoadSilver(dia).run(categorias)


# Sem retry, pelo mesmo motivo da silver.
@task
def run_load_gold(dia: dt.date | None) -> dict[str, int]:
    """Filtra artigos novos dos temas de interesse. Devolve quantos por tema."""
    return LoadGold(dia).run()


# Cada artigo ja tem retry proprio e falha isolada; o retry daqui cobre queda
# geral (rede, RustFS). Retry e barato: PDF ja gravado e pulado.
@task(retries=2, retry_delay_seconds=[60, 300])
def run_load_documents(dia: dt.date | None, limite: int | None) -> dict[str, int]:
    """Baixa os PDFs dos artigos da gold no dia. Devolve a contagem por situacao."""
    return LoadDocuments(dia).run(limite)


# Mesma logica do documents: falha isolada por artigo, retry barato (TEI ja
# gravado e pulado). O retry cobre o GROBID fora do ar ou reiniciando.
@task(retries=2, retry_delay_seconds=[60, 300])
def run_load_text(dia: dt.date | None, limite: int | None) -> dict[str, int]:
    """Extrai o texto (TEI) dos PDFs do dia com o GROBID. Devolve a contagem."""
    return LoadText(dia).run(limite)


# Sem retry: le e grava no RustFS local, nada transitorio.
@task
def run_load_chunks(dia: dt.date | None, limite: int | None) -> dict[str, int]:
    """Quebra os textos do dia em chunks na arxiv.chunks. Devolve a contagem."""
    return LoadChunks(dia).run(limite)


# O retry cobre o servidor de embeddings ou o Qdrant reiniciando. Barato:
# artigo ja indexado com a mesma versao do chunker e pulado.
@task(retries=2, retry_delay_seconds=[60, 300])
def run_load_embeddings(dia: dt.date | None, limite: int | None) -> dict[str, int]:
    """Gera os vetores dos chunks do dia e indexa no Qdrant. Devolve a contagem."""
    return LoadEmbeddings(dia).run(limite)


# Logo depois de indexar: o backup pega o Qdrant ja com o dia novo. Retry
# barato -- cada execucao grava uma copia nova, e a retencao apaga as velhas.
@task(retries=1, retry_delay_seconds=120)
def run_backup() -> dict[str, int]:
    """Backup do catalogo Iceberg e das collections do Qdrant no datalake-backup."""
    return Backup().run()


def publicar_resumo(dia: dt.date, resumo: dict) -> None:
    """Artifact "resumo-arxiv" na UI: uma tabela com o que cada camada fez.

    Com a mesma key em toda execucao, a UI mostra o historico dele como uma
    serie -- da para comparar os dias sem abrir log.
    """
    def fmt(valor) -> str:
        if isinstance(valor, dict):
            return ", ".join(f"{k}: {v}" for k, v in valor.items()) or "-"
        if isinstance(valor, list):
            return ", ".join(valor) or "-"
        return str(valor)

    linhas = "\n".join(f"| {camada} | {fmt(valor)} |" for camada, valor in resumo.items())
    create_markdown_artifact(
        key="resumo-arxiv",
        markdown=f"### arXiv {dia.isoformat()}\n\n| camada | resultado |\n|---|---|\n{linhas}\n",
        description=f"Resumo da ingestao de {dia.isoformat()}",
    )


@flow(name="arxiv_data_pipeline",
      description="Pipeline de coleta e gravacao de dados do arXiv")
def arxiv_data_pipeline(dia: dt.date | None = None,
                        limite_pdfs: int | None = None) -> dict:
    """Coleta do arXiv: bronze -> silver -> gold -> documents -> text -> chunks
    -> embeddings (Qdrant) -> backup.

    Sem `dia`, processa o dia_de_referencia(). `limite_pdfs` corta a
    quantidade de artigos nas etapas por artigo (documents, text, chunks) --
    para teste; sem ele, processa todos os artigos da gold no dia.

    Devolve e publica o resumo; etapa que nao rodou fica como "nao rodou".
    """
    dia_efetivo = dia or dia_de_referencia()
    resumo: dict = {etapa: "nao rodou" for etapa in
                    ("bronze", "silver", "gold", "documents", "text", "chunks", "embeddings",
                     "backup")}
    try:
        categorias = run_load_bronze(dia)
        resumo["bronze"] = categorias or "fim de semana: nenhum registro"
        if not categorias:
            return resumo
        resumo["silver"] = f"{run_load_silver(dia, categorias)} artigos"
        temas = run_load_gold(dia)
        resumo["gold"] = temas or "nenhum artigo de interesse"
        if not temas:
            return resumo
        resumo["documents"] = run_load_documents(dia, limite_pdfs)
        resumo["text"] = run_load_text(dia, limite_pdfs)
        resumo["chunks"] = run_load_chunks(dia, limite_pdfs)
        resumo["embeddings"] = run_load_embeddings(dia, limite_pdfs)
        resumo["backup"] = run_backup()
        return resumo
    finally:
        # no finally: falha numa etapa ainda publica o que as anteriores fizeram
        publicar_resumo(dia_efetivo, resumo)


if __name__ == "__main__":
    # Todo dia as 21:00 de Brasilia (= 00:00 UTC), o fuso da data de negocio
    arxiv_data_pipeline.serve(
        name="diaria",
        schedules=[Cron("0 21 * * *", timezone="America/Sao_Paulo")],
        # uma execucao por vez: duas em paralelo (carga historica, reexecucao
        # manual) teriam cada uma sua pausa de 4s para o arXiv -- juntas,
        # passariam do limite de 1 chamada a cada 3s -- e disputariam a CPU
        # no GROBID e nos embeddings. As demais esperam na fila.
        global_limit=1,
    )
