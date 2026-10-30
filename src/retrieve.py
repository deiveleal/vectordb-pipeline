"""Busca semantica nos artigos indexados: pergunta -> artigos com os melhores trechos.

E o mecanismo que a API expoe; aqui nao ha HTTP. O caminho de uma pergunta:

1. idioma: o bge-small-en-v1.5 e so ingles, e pergunta em portugues errava o
   artigo certo em 3 de 3 testes (2026-09-26). Pergunta que nao e ingles e
   traduzida no LibreTranslate -- antes, os termos tecnicos passam pelo
   GLOSSARIO, porque o tradutor os verte ao pe da letra ("geracao aumentada
   por recuperacao" -> "increased generation by recovery"). Tradutor fora do
   ar nao derruba a busca: segue com a pergunta original e loga o aviso;
2. a pergunta (em ingles) ganha o prefixo de consulta do bge
   (QUERY_INSTRUCTION) e vira vetor no MESMO servidor de embeddings da
   indexacao;
3. o Qdrant busca na collection do tema (ou nas tres) AGRUPANDO por arxiv_id:
   sem isso, o top-10 podia trazer 6 chunks do mesmo artigo;
4. com mais de um tema, os grupos sao juntados pelo score -- comparavel entre
   as collections porque modelo e distancia (cosseno) sao os mesmos;
5. opcional (rerank=True): em vez de agrupar direto, o Qdrant traz os
   CANDIDATOS melhores trechos, um cross-encoder (ms-marco-MiniLM-L-6-v2, no
   servico reranker) le cada par pergunta+trecho e da a relevancia, e os
   trechos sao agrupados por artigo nessa nova ordem. Mais preciso em tese e
   mais caro (a pergunta passa pelo modelo junto com cada trecho). Reranker
   fora do ar: volta para a ordem por vetor e loga o aviso.

O resultado e por ARTIGO: titulo, link, tema, o score do melhor trecho e os
melhores trechos dele, cada um com a secao de onde veio.

Uso (da raiz do projeto):
    python -m src.retrieve "checagem de fatos com geração aumentada por recuperação"
    python -m src.retrieve "reward hacking in autonomous agents" --tema agents -k 5
"""

import argparse
import os
import re
from dataclasses import dataclass, field

import requests
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import ScoredPoint

from src.load_embeddings import (
    QUERY_INSTRUCTION,
    TEMAS,
    gerar_embeddings,
    url_embeddings,
    url_qdrant,
)
from src.utils.load_env import load_env
from src.utils.load_logger import get_logger, setup_logging

logger = get_logger(__name__)

# Enderecos vistos do HOST; em container da rede do compose,
# http://translate:5000 e http://reranker:80.
TRADUTOR_PADRAO = "http://localhost:5010"
RERANKER_PADRAO = "http://localhost:8082"

# Quantos trechos o Qdrant manda para o reranker: pelo menos este numero, ou
# 4 por artigo pedido. So entra no resultado artigo com trecho entre eles.
MIN_CANDIDATOS = 52

# Termos tecnicos em portugues -> o termo em ingles que os artigos usam.
# Aplicado ANTES da traducao: o tradutor preserva o que ja esta em ingles.
# Ordem importa -- o mais especifico primeiro ("modelos de linguagem de
# grande porte" antes de "modelos de linguagem").
GLOSSARIO: list[tuple[str, str]] = [
    (r"gera[cç][aã]o aumentada (por|via|com) (recupera[cç][aã]o|busca)",
     "retrieval-augmented generation"),
    (r"recupera[cç][aã]o de informa[cç][aã]o", "information retrieval"),
    (r"(grandes modelos de linguagem|modelos? de linguagem de grande porte|"
     r"modelos? de linguagem grandes?)", "large language models"),
    (r"modelos? de linguagem (de|para) vis[aã]o", "vision-language models"),
    (r"modelos? de linguagem", "language models"),
    (r"agentes aut[oô]nomos", "autonomous agents"),
    (r"multi-?agentes?", "multi-agent"),
    (r"ag[eê]ntic[oa]s?", "agentic"),
    (r"uso de ferramentas", "tool use"),
    (r"ajuste fino", "fine-tuning"),
    (r"aprendizado por refor[cç]o", "reinforcement learning"),
    (r"cadeia de pensamento", "chain of thought"),
    (r"engenharia de prompts?", "prompt engineering"),
    (r"janela de contexto", "context window"),
    (r"base de conhecimento", "knowledge base"),
    (r"busca sem[aâ]ntica", "semantic search"),
    (r"bancos? de dados vetoriais?|bancos? vetoriais?", "vector database"),
    (r"reordena[cç][aã]o|reranqueamento", "reranking"),
    # o tradutor verte "saida" como "output" (avaliacao de 2026-09-26)
    (r"sa[ií]das? antecipadas?", "early exit"),
    (r"recuperador(es)?", "retriever"),
    (r"alucina[cç](ão|ões)", "hallucination"),
]
_GLOSSARIO = [(re.compile(rf"\b{pt}\b", re.IGNORECASE), en) for pt, en in GLOSSARIO]


def aplicar_glossario(texto: str) -> str:
    for padrao, termo in _GLOSSARIO:
        texto = padrao.sub(termo, texto)
    return texto


@dataclass
class Trecho:
    """Um chunk que casou com a pergunta."""
    secao: str
    texto: str
    score: float
    ordem: int


@dataclass
class Resultado:
    """Um artigo, com os trechos que casaram, do melhor para o pior."""
    arxiv_id: str
    titulo: str
    tema: str
    url: str
    data_v1: str
    score: float                      # do melhor trecho (do reranker, se houve)
    trechos: list[Trecho] = field(default_factory=list)
    score_vetor: float | None = None  # cosseno do melhor trecho, com ou sem rerank


@dataclass
class Busca:
    """A resposta de uma busca: o que foi perguntado, o que foi buscado e o achado."""
    pergunta: str                     # como o usuario escreveu
    pergunta_busca: str               # o que virou vetor (traduzida, se foi o caso)
    idioma: str | None                # detectado; None se a deteccao falhou
    traduzida: bool
    resultados: list[Resultado]
    reranqueada: bool = False


@dataclass
class TrechoDoArtigo:
    """Um chunk de um artigo, sem score -- leitura, nao busca."""
    secao: str
    texto: str
    ordem: int


@dataclass
class Artigo:
    """Os trechos indexados de um artigo, na ordem do texto."""
    arxiv_id: str
    titulo: str
    tema: str
    url: str
    data_v1: str
    total_trechos: int                # do artigo inteiro, antes do filtro de secao
    trechos: list[TrechoDoArtigo] = field(default_factory=list)


class Tradutor:
    """Cliente do LibreTranslate: deteccao de idioma e traducao para ingles."""

    def __init__(self, url: str | None = None):
        load_env()
        self.url = (url or os.environ.get("TRANSLATE_URL", TRADUTOR_PADRAO)).rstrip("/")

    def detectar(self, texto: str) -> tuple[str, float]:
        resposta = requests.post(f"{self.url}/detect", json={"q": texto}, timeout=30)
        resposta.raise_for_status()
        melhor = resposta.json()[0]
        return melhor["language"], float(melhor["confidence"])

    def para_ingles(self, texto: str, origem: str) -> str:
        resposta = requests.post(
            f"{self.url}/translate",
            json={"q": texto, "source": origem, "target": "en", "format": "text"},
            timeout=60,
        )
        resposta.raise_for_status()
        return resposta.json()["translatedText"]


class Reranker:
    """Cliente do cross-encoder (TEI /rerank)."""

    def __init__(self, url: str | None = None):
        load_env()
        self.url = (url or os.environ.get("RERANKER_URL", RERANKER_PADRAO)).rstrip("/")

    def pontuar(self, pergunta: str, textos: list[str]) -> list[float]:
        """Relevancia de cada texto para a pergunta, na ordem dos textos.

        O TEI devolve ordenado por score, com o indice original; aqui volta
        para a ordem de entrada.
        """
        resposta = requests.post(
            f"{self.url}/rerank",
            json={"query": pergunta, "texts": textos, "truncate": True},
            timeout=120,
        )
        resposta.raise_for_status()
        scores = [0.0] * len(textos)
        for item in resposta.json():
            scores[item["index"]] = float(item["score"])
        return scores


class Retriever:
    """Cliente de busca: um por processo, reaproveitado entre perguntas."""

    def __init__(self, qdrant_url: str | None = None, embeddings_url: str | None = None,
                 tradutor: Tradutor | None = None, reranker: Reranker | None = None):
        self.qdrant = QdrantClient(url=qdrant_url or url_qdrant())
        self.url_embeddings = embeddings_url or url_embeddings()
        self.tradutor = tradutor or Tradutor()
        self.reranker = reranker or Reranker()

    def preparar_pergunta(self, pergunta: str) -> tuple[str, str | None, bool]:
        """(pergunta em ingles, idioma detectado, se foi traduzida).

        Falha do tradutor nao e fatal: a busca segue com a pergunta original,
        pior mas funcionando.
        """
        try:
            # Sem corte por confianca: o tradutor so carrega pt e en, entao a
            # deteccao so escolhe entre os dois. Pergunta curta sai com
            # confianca baixa mesmo estando certa ("memoria de longo prazo
            # para agentes": pt, 29%) -- um corte em 50% deixava passar sem
            # traduzir, e a busca em portugues degradava calada. Traduzir por
            # engano uma frase em ingles quase nao a altera.
            idioma, confianca = self.tradutor.detectar(pergunta)
            if idioma == "en":
                return pergunta, idioma, False
            traduzida = self.tradutor.para_ingles(aplicar_glossario(pergunta), idioma)
            logger.info("Pergunta traduzida (%s, confianca %.0f%%): %r -> %r",
                        idioma, confianca, pergunta, traduzida)
            return traduzida, idioma, True
        except (requests.RequestException, KeyError, IndexError, ValueError) as erro:
            logger.warning("Tradutor indisponivel (%s) -- buscando com a pergunta original",
                           erro)
            return pergunta, None, False

    def vetorizar_pergunta(self, pergunta: str) -> list[float]:
        return gerar_embeddings(self.url_embeddings, [QUERY_INSTRUCTION + pergunta])[0]

    def buscar(self, pergunta: str, tema: str | None = None, k: int = 10,
               trechos_por_artigo: int = 3, score_minimo: float | None = None,
               traduzir: bool = True, rerank: bool = False) -> Busca:
        """Os k artigos mais proximos da pergunta.

        tema: "rag", "agents" ou "llm"; sem ele, busca nos tres e junta.
        trechos_por_artigo: quantos chunks de cada artigo voltam.
        score_minimo: corta trechos abaixo deste cosseno, antes de tudo.
            Sem ele, voltam sempre k artigos, mesmo que fracos.
        traduzir: False busca a pergunta como veio -- para comparar na avaliacao.
        rerank: reordena os candidatos com o cross-encoder.
        """
        pergunta = pergunta.strip()
        if not pergunta:
            raise ValueError("pergunta vazia")
        if tema is not None and tema not in TEMAS:
            raise ValueError(f"tema {tema!r} invalido; use um de {TEMAS}")

        if traduzir:
            pergunta_busca, idioma, traduzida = self.preparar_pergunta(pergunta)
        else:
            pergunta_busca, idioma, traduzida = pergunta, None, False

        vetor = self.vetorizar_pergunta(pergunta_busca)
        collections = [tema] if tema else list(TEMAS)
        if rerank:
            try:
                resultados = self._buscar_com_rerank(
                    pergunta_busca, vetor, collections, k, trechos_por_artigo, score_minimo)
                return Busca(pergunta, pergunta_busca, idioma, traduzida, resultados,
                             reranqueada=True)
            except (requests.RequestException, KeyError, IndexError, ValueError) as erro:
                logger.warning("Reranker indisponivel (%s) -- ordem por vetor", erro)
        resultados = self._buscar_por_vetor(vetor, collections, k, trechos_por_artigo,
                                            score_minimo)
        return Busca(pergunta, pergunta_busca, idioma, traduzida, resultados)

    def _buscar_com_rerank(self, pergunta: str, vetor: list[float], collections: list[str],
                           k: int, trechos_por_artigo: int,
                           score_minimo: float | None) -> list[Resultado]:
        """Candidatos por vetor, reordenados pelo cross-encoder, agrupados por artigo."""
        n_candidatos = max(MIN_CANDIDATOS, 4 * k)
        candidatos = []
        for collection in collections:
            candidatos += self.qdrant.query_points(
                collection_name=collection, query=vetor, limit=n_candidatos,
                score_threshold=score_minimo, with_payload=True,
            ).points
        # com mais de uma collection, fica o melhor conjunto geral por vetor
        candidatos = sorted(candidatos, key=lambda p: p.score, reverse=True)[:n_candidatos]
        if not candidatos:
            return []

        # o titulo junto do trecho: o cross-encoder le os dois, e o trecho
        # sozinho perde o contexto (como no cabecalho do embedding)
        textos = [f"{(p.payload or {}).get('titulo', '')}\n{(p.payload or {}).get('texto', '')}"
                  for p in candidatos]
        scores = self.reranker.pontuar(pergunta, textos)

        por_artigo: dict[str, list[tuple[float, ScoredPoint]]] = {}
        for ponto, score in zip(candidatos, scores):
            por_artigo.setdefault(str((ponto.payload or {}).get("arxiv_id")), []).append((score, ponto))

        resultados = []
        for arxiv_id, pares in por_artigo.items():
            pares.sort(key=lambda par: par[0], reverse=True)
            melhores = pares[:trechos_por_artigo]
            payload = melhores[0][1].payload or {}
            resultados.append(Resultado(
                arxiv_id=arxiv_id,
                titulo=payload.get("titulo", ""),
                tema=payload.get("tema", ""),
                url=payload.get("url", ""),
                data_v1=payload.get("data_v1", ""),
                score=melhores[0][0],
                score_vetor=max(p.score for _, p in pares),
                trechos=[Trecho(secao=(p.payload or {}).get("secao", ""),
                                texto=(p.payload or {}).get("texto", ""),
                                score=s,
                                ordem=(p.payload or {}).get("ordem", -1))
                         for s, p in melhores],
            ))
        resultados.sort(key=lambda r: r.score, reverse=True)
        return resultados[:k]

    def artigo(self, arxiv_id: str, secao: str | None = None) -> Artigo | None:
        """Os trechos de um artigo ja indexado, na ordem do texto. None se nao existe.

        Le do Qdrant pelo arxiv_id (payload indexado), sem vetor. secao filtra
        pelo rotulo da secao, sem caixa e por trecho do nome ("result" pega
        "5 Results and Discussion"). E o que deixa um cliente -- a ferramenta
        ler_artigo do MCP -- aprofundar num artigo que a busca achou.
        """
        filtro = models.Filter(must=[
            models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id))])
        pontos: list = []
        for tema in TEMAS:
            # um artigo esta em uma collection so; o maior tem ~100 chunks
            pontos, _ = self.qdrant.scroll(tema, scroll_filter=filtro, limit=1000,
                                           with_payload=True, with_vectors=False)
            if pontos:
                break
        if not pontos:
            return None

        payloads = sorted((p.payload or {} for p in pontos), key=lambda pl: pl.get("ordem", 0))
        primeiro = payloads[0]
        trechos = [TrechoDoArtigo(secao=pl.get("secao", ""), texto=pl.get("texto", ""),
                                  ordem=pl.get("ordem", -1)) for pl in payloads]
        if secao:
            trechos = [t for t in trechos if secao.lower() in t.secao.lower()]
        return Artigo(arxiv_id=arxiv_id, titulo=primeiro.get("titulo", ""),
                      tema=primeiro.get("tema", ""), url=primeiro.get("url", ""),
                      data_v1=primeiro.get("data_v1", ""), total_trechos=len(payloads),
                      trechos=trechos)

    def _buscar_por_vetor(self, vetor: list[float], collections: list[str], k: int,
                          trechos_por_artigo: int,
                          score_minimo: float | None) -> list[Resultado]:
        """Busca agrupada por artigo no Qdrant, ordem pelo cosseno."""
        resultados: list[Resultado] = []
        for collection in collections:
            grupos = self.qdrant.query_points_groups(
                collection_name=collection,
                query=vetor,
                group_by="arxiv_id",
                limit=k,
                group_size=trechos_por_artigo,
                score_threshold=score_minimo,
                with_payload=True,
            ).groups
            for grupo in grupos:
                pontos = sorted(grupo.hits, key=lambda p: p.score, reverse=True)
                payload = pontos[0].payload or {}
                resultados.append(Resultado(
                    arxiv_id=str(grupo.id),
                    titulo=payload.get("titulo", ""),
                    tema=payload.get("tema", collection),
                    url=payload.get("url", ""),
                    data_v1=payload.get("data_v1", ""),
                    score=pontos[0].score,
                    score_vetor=pontos[0].score,
                    trechos=[Trecho(secao=(p.payload or {}).get("secao", ""),
                                    texto=(p.payload or {}).get("texto", ""),
                                    score=p.score,
                                    ordem=(p.payload or {}).get("ordem", -1))
                             for p in pontos],
                ))

        # com mais de uma collection, cada uma devolveu ate k: fica o top-k geral
        resultados.sort(key=lambda r: r.score, reverse=True)
        return resultados[:k]


def _imprimir(busca: Busca, tamanho_trecho: int = 220) -> None:
    if busca.traduzida:
        print(f"({busca.idioma}) {busca.pergunta!r} -> {busca.pergunta_busca!r}")
    if busca.reranqueada:
        print("(reordenado pelo cross-encoder; entre colchetes, score do reranker e cosseno)")
    for i, r in enumerate(busca.resultados, 1):
        vetor = f" / {r.score_vetor:.3f}" if busca.reranqueada and r.score_vetor is not None else ""
        print(f"\n{i:2}. [{r.score:.3f}{vetor}] ({r.tema}) {r.titulo}")
        print(f"    {r.url} · v1 {r.data_v1}")
        for t in r.trechos:
            texto = t.texto if len(t.texto) <= tamanho_trecho else t.texto[:tamanho_trecho] + "..."
            print(f"      - [{t.score:.3f}] {t.secao[:60]}: {texto}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Busca semantica nos artigos do arXiv")
    parser.add_argument("pergunta")
    parser.add_argument("--tema", choices=TEMAS)
    parser.add_argument("-k", type=int, default=10, help="artigos no resultado")
    parser.add_argument("--trechos", type=int, default=2, help="trechos por artigo")
    parser.add_argument("--score-minimo", type=float)
    parser.add_argument("--sem-traducao", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="reordena com o cross-encoder")
    args = parser.parse_args()
    _imprimir(Retriever().buscar(args.pergunta, tema=args.tema, k=args.k,
                                 trechos_por_artigo=args.trechos,
                                 score_minimo=args.score_minimo,
                                 traduzir=not args.sem_traducao,
                                 rerank=args.rerank))
