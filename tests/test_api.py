"""Testes da API com um Retriever falso: sem Qdrant, embeddings nem tradutor.

O que se testa aqui e a camada HTTP -- validacao, formato da resposta,
traducao de erro em status. A qualidade da busca e medida em src.avaliacao.
"""

from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

import src.api as api
from src.retrieve import Artigo, Busca, Resultado, Trecho, TrechoDoArtigo


class RetrieverFalso:
    """Guarda os argumentos da ultima busca e devolve um resultado fixo."""

    def __init__(self, erro: Exception | None = None, contagem=None):
        self.erro = erro
        self.chamada: dict = {}
        self.url_embeddings = "http://embeddings.falso"
        self.tradutor = SimpleNamespace(url="http://tradutor.falso")

        def contar(tema, exact=False):
            if contagem is None:
                raise ConnectionError("qdrant fora")
            return SimpleNamespace(count=contagem[tema])
        self.qdrant = SimpleNamespace(count=contar)

    def artigo(self, arxiv_id, secao=None):
        self.chamada = dict(arxiv_id=arxiv_id, secao=secao)
        if self.erro:
            raise self.erro
        if arxiv_id != "2609.28653":
            return None
        return Artigo(arxiv_id=arxiv_id, titulo="The Fellowship of the Query", tema="rag",
                      url=f"https://arxiv.org/abs/{arxiv_id}", data_v1="2026-09-23",
                      total_trechos=19,
                      trechos=[TrechoDoArtigo(secao="3.3 RQ3", texto="We evaluate...", ordem=13)])

    def buscar(self, pergunta, tema=None, k=10, trechos_por_artigo=3,
               score_minimo=None, traduzir=True, rerank=False):
        self.chamada = dict(pergunta=pergunta, tema=tema, k=k,
                            trechos_por_artigo=trechos_por_artigo,
                            score_minimo=score_minimo, traduzir=traduzir, rerank=rerank)
        if self.erro:
            raise self.erro
        return Busca(
            pergunta=pergunta, pergunta_busca="fact checking", idioma="pt", traduzida=True,
            resultados=[Resultado(
                arxiv_id="2609.30227", titulo="To Trust or Not to Trust", tema="rag",
                url="https://arxiv.org/abs/2609.30227", data_v1="2026-09-24", score=0.83,
                trechos=[Trecho(secao="Abstract", texto="Online misinformation...",
                                score=0.83, ordem=0)],
            )],
        )


@pytest.fixture
def cliente():
    """TestClient com o Retriever trocado pelo falso da vez."""
    def fabricar(retriever):
        api.app.dependency_overrides[api.obter_retriever] = lambda: retriever
        return TestClient(api.app)
    yield fabricar
    api.app.dependency_overrides.clear()


def test_search_devolve_artigos_com_trechos(cliente):
    falso = RetrieverFalso()
    r = cliente(falso).get("/search", params={"q": "checagem de fatos", "topic": "rag",
                                              "k": 5, "trechos": 2})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["pergunta_busca"] == "fact checking" and corpo["traduzida"] is True
    assert corpo["total"] == 1
    artigo = corpo["resultados"][0]
    assert artigo["pdf_url"] == "https://arxiv.org/pdf/2609.30227"
    assert artigo["trechos"][0]["secao"] == "Abstract"
    # os parametros chegam ao retriever com os nomes dele; sem `rerank` na
    # requisicao, vale o padrao da API
    assert falso.chamada == dict(pergunta="checagem de fatos", tema="rag", k=5,
                                 trechos_por_artigo=2, score_minimo=None, traduzir=True,
                                 rerank=api.RERANK_PADRAO)
    # o falso devolve Busca sem reranking
    assert corpo["reranqueada"] is False


@pytest.mark.parametrize("valor, esperado", [("true", True), ("false", False)])
def test_search_repassa_rerank(cliente, valor, esperado):
    falso = RetrieverFalso()
    cliente(falso).get("/search", params={"q": "rag", "rerank": valor})
    assert falso.chamada["rerank"] is esperado


@pytest.mark.parametrize("params", [
    {"q": ""},                      # pergunta vazia
    {"q": "x", "topic": "vision"},  # tema fora da lista
    {"q": "x", "k": 51},            # k acima do teto
    {"q": "x", "trechos": 0},
    {"q": "x", "score_minimo": 2},
    {},                             # sem pergunta
])
def test_search_valida_parametros(cliente, params):
    assert cliente(RetrieverFalso()).get("/search", params=params).status_code == 422


def test_search_erro_de_entrada_vira_400(cliente):
    r = cliente(RetrieverFalso(erro=ValueError("pergunta vazia"))).get("/search", params={"q": "  "})
    assert r.status_code == 400
    assert r.json()["detail"] == "pergunta vazia"


def test_search_dependencia_fora_vira_503(cliente):
    falso = RetrieverFalso(erro=requests.ConnectionError("embeddings fora"))
    r = cliente(falso).get("/search", params={"q": "rag"})
    assert r.status_code == 503


@pytest.mark.parametrize("fora, esperado, http", [
    (set(), "ok", 200),
    ({"tradutor.falso"}, "degradado", 200),
    ({"embeddings.falso"}, "indisponivel", 503),
])
def test_health(cliente, monkeypatch, fora, esperado, http):
    def get_falso(url, timeout):
        if any(host in url for host in fora):
            raise requests.ConnectionError(url)
        return SimpleNamespace(raise_for_status=lambda: None)
    monkeypatch.setattr(api.requests, "get", get_falso)

    falso = RetrieverFalso(contagem={"rag": 227, "agents": 2798, "llm": 3317})
    r = cliente(falso).get("/health")
    assert r.status_code == http
    assert r.json()["status"] == esperado
    assert r.json()["pontos_por_tema"]["rag"] == 227


def test_health_qdrant_fora_e_indisponivel(cliente, monkeypatch):
    monkeypatch.setattr(api.requests, "get",
                        lambda url, timeout: SimpleNamespace(raise_for_status=lambda: None))
    r = cliente(RetrieverFalso(contagem=None)).get("/health")
    assert r.status_code == 503
    assert r.json()["dependencias"]["qdrant"].startswith("erro")


def test_artigo_devolve_trechos_em_ordem(cliente):
    falso = RetrieverFalso()
    r = cliente(falso).get("/articles/2609.28653/chunks", params={"secao": "rq3"})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["total_trechos"] == 19 and corpo["trechos"][0]["ordem"] == 13
    assert corpo["pdf_url"] == "https://arxiv.org/pdf/2609.28653"
    assert falso.chamada == dict(arxiv_id="2609.28653", secao="rq3")


def test_artigo_inexistente_e_404(cliente):
    assert cliente(RetrieverFalso()).get("/articles/9999.99999/chunks").status_code == 404


def test_artigo_com_dependencia_fora_e_503(cliente):
    falso = RetrieverFalso(erro=requests.ConnectionError("qdrant fora"))
    assert cliente(falso).get("/articles/2609.28653/chunks").status_code == 503


def test_artigo_id_invalido_e_422(cliente):
    assert cliente(RetrieverFalso()).get("/articles/a b;c/chunks").status_code == 422
