"""Camada de indice: arxiv.chunks -> Qdrant, uma collection por tema.

Para cada artigo da gold no dia (particao datestamp), pega os chunks dele na
arxiv.chunks, gera os vetores no servidor de embeddings (TEI, bge-small-en-v1.5,
384 dimensoes) e grava na collection do tema: `rag`, `agents` ou `llm`.

O Qdrant e indice derivado, nao fonte: tudo aqui sai da arxiv.chunks, e
recriar uma collection e rodar esta camada de novo. Por isso:

- o id do ponto e um UUID5 do chunk_id: gravar de novo substitui, nao duplica;
- antes de gravar um artigo, os pontos antigos dele saem das TRES collections
  -- ele pode ter mudado de tema ou de numero de chunks (versao nova do
  chunker);
- artigo ja indexado com a mesma versao_chunker e o mesmo numero de chunks e
  pulado: embedding e a etapa mais cara (~7 chunks/s nesta CPU, medido em
  2026-09-26), e rodar o flow de novo nao deve refazer o dia inteiro.

O payload leva o necessario para a API mostrar o resultado sem outra consulta:
texto do chunk, titulo, secao, link do arXiv e a chave do PDF.

Na busca, o bge-small-en-v1.5 pede o prefixo QUERY_INSTRUCTION na consulta
(nao nos trechos indexados aqui) -- quem montar a API precisa usar o mesmo.

Uso (da raiz do projeto): python -m src.load_embeddings
"""

import datetime as dt
import os
import time
import uuid
from collections import Counter

import requests
from qdrant_client import QdrantClient, models

from src.load_chunks import TABELA as TABELA_CHUNKS
from src.load_chunks import VERSAO_CHUNKER
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

logger = get_logger(__name__)

# Uma collection por tema da gold.
TEMAS = ("rag", "agents", "llm")

# Enderecos vistos do HOST; em container da rede do compose,
# http://qdrant:6333 e http://embeddings:80.
QDRANT_PADRAO = "http://localhost:6666"
EMBEDDINGS_PADRAO = "http://localhost:8081"

# Ate 64 pelo --max-client-batch-size do servico; 32 deixa folga de memoria.
LOTE_EMBEDDING = 32

# Prefixo que o bge-small-en-v1.5 recomenda na CONSULTA de busca.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Namespace fixo do UUID5: o mesmo chunk_id gera sempre o mesmo id de ponto.
NAMESPACE_PONTOS = uuid.uuid5(uuid.NAMESPACE_URL, "arxiv-chunks")


def id_do_ponto(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE_PONTOS, chunk_id))


def url_embeddings() -> str:
    """Endereco do servidor de embeddings: EMBEDDINGS_URL, ou o do host."""
    load_env()
    return os.environ.get("EMBEDDINGS_URL", EMBEDDINGS_PADRAO).rstrip("/")


def url_qdrant() -> str:
    """Endereco do Qdrant: QDRANT_URL, ou o do host."""
    load_env()
    return os.environ.get("QDRANT_URL", QDRANT_PADRAO)


def gerar_embeddings(url: str, textos: list[str]) -> list[list[float]]:
    """Vetores normalizados, com o mesmo retry do OAI no que for transitorio.

    Fonte unica para indexar (aqui) e para consultar (src/retrieve): trecho e
    pergunta tem que passar pelo mesmo modelo e pela mesma normalizacao.

    truncate=True e rede de seguranca: o chunker ja garante <= 512 tokens,
    mas se um dia falhar, o servidor corta em vez de recusar o lote.
    """
    espera = ESPERA_INICIAL_S
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            resposta = requests.post(
                f"{url}/embed",
                json={"inputs": textos, "truncate": True, "normalize": True},
                timeout=600,
            )
            resposta.raise_for_status()
            return resposta.json()
        except requests.RequestException as erro:
            if tentativa == TENTATIVAS or not vale_repetir(erro):
                raise
            descanso = pausa_ate_a_proxima(erro, espera)
            logger.warning("Embeddings recusou (%s). Tentativa %d de %d -- "
                           "nova tentativa em %.0fs.", erro, tentativa, TENTATIVAS, descanso)
            time.sleep(descanso)
            espera *= FATOR_DE_BACKOFF
    raise RuntimeError("laco de tentativas terminou sem resposta nem erro")


class LoadEmbeddings:
    """Indexacao no Qdrant dos chunks dos artigos de um dia da gold.

    Uma instancia trata um unico dia. Sem argumento, e o dia_de_referencia();
    para reprocessar outra data, passe `dia` -- e o que o orquestrador usa.
    """

    def __init__(self, dia: dt.date | None = None):
        load_env()
        self.dia_alvo = dia or dia_de_referencia()
        self.catalog = load_catalog()
        self.qdrant = QdrantClient(url=url_qdrant())
        self.url_embeddings = url_embeddings()

    # --- embeddings -----------------------------------------------------------

    def embed(self, textos: list[str]) -> list[list[float]]:
        return gerar_embeddings(self.url_embeddings, textos)

    # --- qdrant ---------------------------------------------------------------

    def garantir_collections(self) -> None:
        """Cria as collections que faltam, com o tamanho do vetor do modelo.

        O tamanho vem do proprio servico (um vetor de teste), nao de uma
        constante: trocar o modelo sem recriar as collections falharia aqui,
        com a mensagem certa, em vez de no meio do upsert.
        """
        dimensao = len(self.embed(["dimensao"])[0])
        for tema in TEMAS:
            if self.qdrant.collection_exists(tema):
                atual = self.qdrant.get_collection(tema).config.params.vectors
                tamanho = getattr(atual, "size", None)
                if tamanho != dimensao:
                    raise ValueError(f"collection {tema} tem vetor de {tamanho}, o modelo "
                                     f"gera {dimensao}: recrie as collections")
                continue
            logger.info("Criando collection %s (%d dimensoes, cosseno)", tema, dimensao)
            self.qdrant.create_collection(
                tema,
                vectors_config=models.VectorParams(size=dimensao, distance=models.Distance.COSINE),
            )
            # arxiv_id: apagar os pontos de um artigo e agrupar resultado por
            # artigo na busca. versao_chunker: saber o que ja esta indexado.
            for campo in ("arxiv_id", "versao_chunker"):
                self.qdrant.create_payload_index(tema, campo, models.PayloadSchemaType.KEYWORD)
            self.qdrant.create_payload_index(tema, "data_v1", models.PayloadSchemaType.DATETIME)

    def ja_indexado(self, arxiv_id: str, tema: str, n_chunks: int) -> bool:
        """Se o artigo ja esta no tema, com esta versao do chunker e os mesmos chunks."""
        filtro = models.Filter(must=[
            models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id)),
            models.FieldCondition(key="versao_chunker", match=models.MatchValue(value=VERSAO_CHUNKER)),
        ])
        return self.qdrant.count(tema, count_filter=filtro, exact=True).count == n_chunks

    def remover_artigo(self, arxiv_id: str) -> None:
        """Tira os pontos do artigo das tres collections."""
        filtro = models.Filter(must=[
            models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id))])
        for tema in TEMAS:
            self.qdrant.delete(tema, points_selector=models.FilterSelector(filter=filtro), wait=True)

    def indexar_artigo(self, chunks: list[dict]) -> None:
        """Embeddings + upsert de todos os chunks de um artigo."""
        vetores: list[list[float]] = []
        for i in range(0, len(chunks), LOTE_EMBEDDING):
            vetores += self.embed([c["texto_embedding"] for c in chunks[i:i + LOTE_EMBEDDING]])

        tema, arxiv_id = chunks[0]["tema"], chunks[0]["arxiv_id"]
        pontos = [
            models.PointStruct(
                id=id_do_ponto(c["chunk_id"]),
                vector=v,
                payload={
                    "chunk_id": c["chunk_id"],
                    "arxiv_id": arxiv_id,
                    "tema": tema,
                    "titulo": c["titulo"],
                    "secao": c["secao"],
                    "ordem": c["ordem"],
                    "texto": c["texto"],
                    "data_v1": c["dia_v1"].isoformat(),
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "chave_pdf": c["chave_pdf"],
                    "versao_chunker": c["versao_chunker"],
                },
            )
            for c, v in zip(chunks, vetores)
        ]
        self.remover_artigo(arxiv_id)
        self.qdrant.upsert(tema, points=pontos, wait=True)

    # --- execucao -------------------------------------------------------------

    def ler_chunks_do_dia(self) -> dict[str, list[dict]]:
        """Chunks dos artigos da gold no dia, agrupados por artigo e em ordem."""
        gold = self.catalog.load_table(f"{NAMESPACE}.{TABELA_GOLD}")
        filtro_gold = f"datestamp = '{self.dia_alvo.isoformat()}'"
        ids = [r["arxiv_id"] for r in gold.scan(
            row_filter=filtro_gold, selected_fields=("arxiv_id",)).to_arrow().to_pylist()]
        if not ids:
            return {}

        chunks = self.catalog.load_table(f"{NAMESPACE}.{TABELA_CHUNKS}")
        filtro = "arxiv_id IN (" + ", ".join(f"'{i}'" for i in ids) + ")"
        por_artigo: dict[str, list[dict]] = {}
        for c in chunks.scan(row_filter=filtro).to_arrow().to_pylist():
            por_artigo.setdefault(c["arxiv_id"], []).append(c)
        for lista in por_artigo.values():
            lista.sort(key=lambda c: c["ordem"])
        # na ordem da gold, para o limite de teste ser previsivel
        return {i: por_artigo[i] for i in ids if i in por_artigo}

    def run(self, limite: int | None = None) -> dict[str, int]:
        """Indexa o que falta do dia. Devolve a contagem.

        `limite` corta a quantidade de artigos -- para teste. Falha em um
        artigo nao para os outros; so o dia inteiro falhando vira excecao.
        """
        self.garantir_collections()
        artigos = list(self.ler_chunks_do_dia().items())[:limite]
        situacao: Counter[str] = Counter()
        t0 = time.monotonic()

        for arxiv_id, chunks in artigos:
            if self.ja_indexado(arxiv_id, chunks[0]["tema"], len(chunks)):
                situacao["ja_indexado"] += 1
                continue
            try:
                self.indexar_artigo(chunks)
            except Exception:
                situacao["falha"] += 1
                logger.exception("Indexacao de %s falhou -- seguindo para o proximo", arxiv_id)
                continue
            situacao["indexado"] += 1
            situacao["chunks"] += len(chunks)

        segundos = time.monotonic() - t0
        logger.info("Embeddings %s: %d artigos -> %s em %.0fs (%.1f chunks/s)",
                    self.dia_alvo, len(artigos), dict(situacao), segundos,
                    situacao["chunks"] / segundos if segundos else 0)
        if artigos and situacao["falha"] == len(artigos):
            raise RuntimeError(
                f"nenhum dos {len(artigos)} artigos foi indexado -- ver os erros acima")
        return dict(situacao)


if __name__ == "__main__":
    setup_logging()
    embeddings_loader = LoadEmbeddings()
    embeddings_loader.run()
