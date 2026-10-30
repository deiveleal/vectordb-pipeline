"""API de busca semantica nos artigos do arXiv.

Camada fina sobre o src.retrieve: valida a entrada, chama o Retriever e
serializa. Nenhuma logica de busca mora aqui -- traducao, embedding e
agrupamento por artigo sao do retrieve, e sao os mesmos que a avaliacao
(src.avaliacao) mede.

    GET /search?q=...&topic=rag&k=10   artigos mais proximos da pergunta
    GET /articles/{arxiv_id}/chunks    trechos de um artigo, na ordem do texto
    GET /health                        estado do Qdrant, embeddings e tradutor
    GET /docs                          documentacao interativa (OpenAPI)

Nao ha score_minimo por padrao: na avaliacao de 2026-09-26 as consultas sem
resposta no indice chegaram a score 0,805, acima de varias respostas certas
(minimo 0,739). O corte existe como parametro opcional.

Uso (da raiz do projeto):
    uv run uvicorn src.api:app --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import requests
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from pydantic import BaseModel
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.load_embeddings import TEMAS
from src.retrieve import Artigo, Busca, Retriever
from src.utils.load_logger import get_logger, setup_logging

logger = get_logger(__name__)

Tema = Literal["rag", "agents", "llm"]

# Reranking desligado por padrao ate a avaliacao com o indice maior (3 dias)
# mostrar se compensa o custo em CPU. Quem quiser, pede com rerank=true.
RERANK_PADRAO = True

# Falha de um servico de que a busca depende: vira 503, nao 500 -- o problema
# e de disponibilidade, nao da requisicao.
ERROS_DE_DEPENDENCIA = (requests.RequestException, UnexpectedResponse,
                        ResponseHandlingException, ConnectionError, TimeoutError)


# --- modelos de resposta ----------------------------------------------------

class TrechoOut(BaseModel):
    secao: str
    texto: str
    score: float
    ordem: int


class ResultadoOut(BaseModel):
    arxiv_id: str
    titulo: str
    tema: str
    url: str
    pdf_url: str
    data_v1: str
    score: float
    score_vetor: float | None
    trechos: list[TrechoOut]


class BuscaOut(BaseModel):
    pergunta: str
    pergunta_busca: str
    idioma: str | None
    traduzida: bool
    reranqueada: bool
    total: int
    resultados: list[ResultadoOut]


class TrechoArtigoOut(BaseModel):
    secao: str
    texto: str
    ordem: int


class ArtigoOut(BaseModel):
    arxiv_id: str
    titulo: str
    tema: str
    url: str
    pdf_url: str
    data_v1: str
    total_trechos: int
    trechos: list[TrechoArtigoOut]


def serializar_artigo(artigo: Artigo) -> ArtigoOut:
    return ArtigoOut(
        arxiv_id=artigo.arxiv_id, titulo=artigo.titulo, tema=artigo.tema, url=artigo.url,
        pdf_url=f"https://arxiv.org/pdf/{artigo.arxiv_id}", data_v1=artigo.data_v1,
        total_trechos=artigo.total_trechos,
        trechos=[TrechoArtigoOut(secao=t.secao, texto=t.texto, ordem=t.ordem)
                 for t in artigo.trechos],
    )


class HealthOut(BaseModel):
    status: Literal["ok", "degradado", "indisponivel"]
    dependencias: dict[str, str]
    pontos_por_tema: dict[str, int]


def serializar(busca: Busca) -> BuscaOut:
    return BuscaOut(
        pergunta=busca.pergunta,
        pergunta_busca=busca.pergunta_busca,
        idioma=busca.idioma,
        traduzida=busca.traduzida,
        reranqueada=busca.reranqueada,
        total=len(busca.resultados),
        resultados=[
            ResultadoOut(
                arxiv_id=r.arxiv_id, titulo=r.titulo, tema=r.tema, url=r.url,
                pdf_url=f"https://arxiv.org/pdf/{r.arxiv_id}",
                data_v1=r.data_v1, score=r.score, score_vetor=r.score_vetor,
                trechos=[TrechoOut(secao=t.secao, texto=t.texto, score=t.score, ordem=t.ordem)
                         for t in r.trechos],
            )
            for r in busca.resultados
        ],
    )


# --- aplicacao --------------------------------------------------------------

@asynccontextmanager
async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
    # um Retriever por processo: guarda os clientes HTTP e do Qdrant
    setup_logging()
    app.state.retriever = Retriever()
    yield


app = FastAPI(
    title="arXiv Radar -- busca semantica",
    description="Busca por significado nos artigos de RAG, Agents e LLM do arXiv. "
                "Perguntas em portugues sao traduzidas para ingles antes da busca.",
    version="0.1.0",
    lifespan=ciclo_de_vida,
)


def obter_retriever(request: Request) -> Retriever:
    """Dependencia: os testes trocam por um Retriever falso."""
    return request.app.state.retriever


RetrieverDep = Annotated[Retriever, Depends(obter_retriever)]


# Handlers com `def`, nao `async def`: o Retriever e sincrono (requests e o
# cliente sincrono do Qdrant), e o FastAPI roda `def` num threadpool sem
# bloquear o event loop.

@app.get("/search", response_model=BuscaOut, summary="Busca semantica")
def buscar(
    retriever: RetrieverDep,
    q: Annotated[str, Query(min_length=1, max_length=500,
                            description="Pergunta, em portugues ou ingles")],
    topic: Annotated[Tema | None, Query(description="Tema; sem ele, busca nos tres")] = None,
    k: Annotated[int, Query(ge=1, le=12, description="Quantos artigos")] = 10,
    trechos: Annotated[int, Query(ge=1, le=10, description="Trechos por artigo")] = 3,
    score_minimo: Annotated[float | None, Query(ge=-1, le=1,
                            description="Corta artigos abaixo deste score (cosseno)")] = None,
    traduzir: Annotated[bool, Query(description="Traduzir pergunta que nao e ingles")] = True,
    rerank: Annotated[bool, Query(description="Reordenar com o cross-encoder")] = RERANK_PADRAO,
) -> BuscaOut:
    try:
        busca = retriever.buscar(q, tema=topic, k=k, trechos_por_artigo=trechos,
                                 score_minimo=score_minimo, traduzir=traduzir, rerank=rerank)
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro
    except ERROS_DE_DEPENDENCIA as erro:
        logger.exception("Busca falhou por dependencia indisponivel")
        raise HTTPException(status_code=503,
                            detail="servico de busca indisponivel no momento") from erro
    return serializar(busca)


@app.get("/articles/{arxiv_id}/chunks", response_model=ArtigoOut,
         summary="Trechos de um artigo")
def trechos_do_artigo(
    retriever: RetrieverDep,
    arxiv_id: Annotated[str, Path(pattern=r"^[\w.\-/]+$", max_length=40,
                                  description="Ex.: 2609.28653")],
    secao: Annotated[str | None, Query(max_length=100,
                     description="Filtra pela secao (trecho do nome, sem caixa)")] = None,
) -> ArtigoOut:
    """Todos os trechos indexados do artigo, na ordem do texto -- para ler
    alem dos trechos que a busca devolveu."""
    try:
        artigo = retriever.artigo(arxiv_id, secao=secao)
    except ERROS_DE_DEPENDENCIA as erro:
        logger.exception("Leitura de artigo falhou por dependencia indisponivel")
        raise HTTPException(status_code=503,
                            detail="servico de busca indisponivel no momento") from erro
    if artigo is None:
        raise HTTPException(status_code=404, detail=f"artigo {arxiv_id} nao indexado")
    return serializar_artigo(artigo)


@app.get("/health", response_model=HealthOut, summary="Estado das dependencias")
def saude(retriever: RetrieverDep, response: Response) -> HealthOut:
    """ok: tudo no ar. degradado: so o tradutor fora -- a busca funciona, mas
    pergunta em portugues perde qualidade. indisponivel (503): Qdrant ou
    embeddings fora -- nao ha busca."""
    dependencias: dict[str, str] = {}
    pontos: dict[str, int] = {}

    try:
        for tema in TEMAS:
            pontos[tema] = retriever.qdrant.count(tema, exact=False).count
        dependencias["qdrant"] = "ok"
    except Exception as erro:  # noqa: BLE001 -- qualquer falha aqui e "fora do ar"
        dependencias["qdrant"] = f"erro: {type(erro).__name__}"

    for nome, url in (("embeddings", f"{retriever.url_embeddings}/health"),
                      ("tradutor", f"{retriever.tradutor.url}/languages")):
        try:
            requests.get(url, timeout=5).raise_for_status()
            dependencias[nome] = "ok"
        except requests.RequestException as erro:
            dependencias[nome] = f"erro: {type(erro).__name__}"

    status: Literal["ok", "degradado", "indisponivel"]
    if dependencias["qdrant"] != "ok" or dependencias["embeddings"] != "ok":
        status = "indisponivel"
        response.status_code = 503
    elif dependencias["tradutor"] != "ok":
        status = "degradado"
    else:
        status = "ok"
    return HealthOut(status=status, dependencias=dependencias, pontos_por_tema=pontos)
