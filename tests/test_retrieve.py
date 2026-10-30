"""Testes da preparacao da pergunta (idioma, glossario, traducao), sem servicos."""

import requests

from src.retrieve import Retriever, aplicar_glossario


class TradutorFalso:
    def __init__(self, idioma="pt", confianca=100.0, fora=False):
        self.idioma, self.confianca, self.fora = idioma, confianca, fora
        self.traduzido: str | None = None
        self.url = "http://tradutor.falso"

    def detectar(self, texto):
        if self.fora:
            raise requests.ConnectionError("tradutor fora")
        return self.idioma, self.confianca

    def para_ingles(self, texto, origem):
        self.traduzido = texto  # o que chegou ao tradutor, ja com glossario
        return f"EN({texto})"


def retriever_com(tradutor) -> Retriever:
    return Retriever(qdrant_url="http://qdrant.falso", embeddings_url="http://emb.falso",
                     tradutor=tradutor)


def test_pergunta_curta_com_confianca_baixa_e_traduzida():
    # regressao: "memoria de longo prazo para agentes" sai como pt com 29% de
    # confianca, e um corte em 50% deixava a pergunta passar sem traducao
    pergunta, idioma, traduzida = retriever_com(TradutorFalso(confianca=29.0)).preparar_pergunta(
        "memória de longo prazo para agentes")
    assert traduzida is True and idioma == "pt"
    assert pergunta == "EN(memória de longo prazo para agentes)"


def test_pergunta_em_ingles_nao_e_traduzida():
    pergunta, idioma, traduzida = retriever_com(TradutorFalso(idioma="en")).preparar_pergunta(
        "long-term memory for agents")
    assert (pergunta, idioma, traduzida) == ("long-term memory for agents", "en", False)


def test_glossario_chega_antes_da_traducao():
    tradutor = TradutorFalso()
    retriever_com(tradutor).preparar_pergunta("checagem com geração aumentada por recuperação")
    assert tradutor.traduzido == "checagem com retrieval-augmented generation"


def test_tradutor_fora_segue_com_a_pergunta_original():
    pergunta, idioma, traduzida = retriever_com(TradutorFalso(fora=True)).preparar_pergunta(
        "memória de agentes")
    assert (pergunta, idioma, traduzida) == ("memória de agentes", None, False)


def test_glossario():
    assert aplicar_glossario("Saídas antecipadas em modelos de linguagem") == \
        "early exit em language models"
    assert aplicar_glossario("grandes modelos de linguagem") == "large language models"


# --- reranking ----------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402


def ponto(arxiv_id, ordem, score, texto):
    return SimpleNamespace(score=score, payload={
        "arxiv_id": arxiv_id, "titulo": f"titulo {arxiv_id}", "tema": "rag",
        "url": f"https://arxiv.org/abs/{arxiv_id}", "data_v1": "2026-09-24",
        "secao": "Abstract", "texto": texto, "ordem": ordem})


class QdrantFalso:
    """A por vetor vem na frente de B; o agrupamento devolve a mesma ordem."""
    pontos = [ponto("A", 0, 0.90, "texto de A"), ponto("B", 0, 0.80, "texto de B")]

    def query_points(self, collection_name, query, limit, score_threshold, with_payload):
        return SimpleNamespace(points=list(self.pontos))

    def query_points_groups(self, collection_name, query, group_by, limit, group_size,
                            score_threshold, with_payload):
        return SimpleNamespace(groups=[SimpleNamespace(id=p.payload["arxiv_id"], hits=[p])
                                       for p in self.pontos])


class RerankerFalso:
    def __init__(self, fora=False):
        self.fora = fora

    def pontuar(self, pergunta, textos):
        if self.fora:
            raise requests.ConnectionError("reranker fora")
        return [0.1 if "texto de A" in t else 0.95 for t in textos]  # prefere B


def retriever_para_rerank(reranker) -> Retriever:
    r = Retriever(qdrant_url="http://qdrant.falso", embeddings_url="http://emb.falso",
                  tradutor=TradutorFalso(idioma="en"), reranker=reranker)
    r.qdrant = QdrantFalso()
    r.vetorizar_pergunta = lambda pergunta: [0.0]  # type: ignore[method-assign]
    return r


def test_rerank_reordena_os_artigos():
    busca = retriever_para_rerank(RerankerFalso()).buscar("q", tema="rag", rerank=True)
    assert busca.reranqueada is True
    assert [r.arxiv_id for r in busca.resultados] == ["B", "A"]
    assert busca.resultados[0].score == 0.95 and busca.resultados[0].score_vetor == 0.80


def test_reranker_fora_volta_para_ordem_por_vetor():
    busca = retriever_para_rerank(RerankerFalso(fora=True)).buscar("q", tema="rag", rerank=True)
    assert busca.reranqueada is False
    assert [r.arxiv_id for r in busca.resultados] == ["A", "B"]


def test_sem_rerank_nao_chama_o_reranker():
    busca = retriever_para_rerank(RerankerFalso(fora=True)).buscar("q", tema="rag")
    assert busca.reranqueada is False and busca.resultados[0].arxiv_id == "A"
