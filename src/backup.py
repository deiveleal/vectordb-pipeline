"""Backup do que nao se reconstroi sozinho: catalogo Iceberg e collections do Qdrant.

Roda no flow logo depois de indexar os dados novos no Qdrant. Grava no bucket
datalake-backup, com o mesmo recorte de data das outras camadas:

    catalogo/AAAA/MM/DD/catalogo_<AAAAMMDD_HHMMSS>.db
    qdrant/<tema>/AAAA/MM/DD/<tema>_<AAAAMMDD_HHMMSS>.snapshot

Por que estes dois:

- o catalogo (SQLite) e o unico lugar que sabe qual metadata.json e o atual de
  cada tabela. Os dados continuam no bucket se ele se perde, mas as tabelas
  ficam inacessiveis. A copia usa a API de backup do SQLite, consistente mesmo
  com o arquivo aberto -- copiar o arquivo direto podia pegar escrita pela
  metade.
- o Qdrant da para reconstruir a partir da arxiv.chunks, mas isso e refazer
  os embeddings de tudo (~15 min por dia de artigos nesta CPU). O snapshot
  volta em segundos.

Bronze, documents e as tabelas Iceberg nao entram: ja estao no RustFS, com
versionamento (bronze, documents) ou historico em snapshots (Iceberg).

Retencao: ficam os RETENCAO backups mais recentes de cada item; os mais
antigos sao apagados. Limite: o backup mora no MESMO RustFS -- protege contra
arquivo corrompido ou apagado e contra volume do Qdrant perdido, nao contra a
maquina inteira. Para isso, copiar o datalake-backup para fora.

Restauracao:
    catalogo:  baixar o .db para data/iceberg/catalogo.db (pipeline parado).
    Qdrant:    POST /collections/<tema>/snapshots/upload?priority=snapshot com
               o arquivo .snapshot -- recria a collection com os pontos. Ver
               restaurar_qdrant(), testada em 2026-09-26.

Uso (da raiz do projeto): python -m src.backup
"""

import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import requests
from qdrant_client import QdrantClient

from src.load_embeddings import TEMAS, url_qdrant
from src.utils.load_catalog import uri_catalogo
from src.utils.load_logger import get_logger, setup_logging
from src.utils.load_time import agora
from src.utils.storage_functions import StorageFuncions

logger = get_logger(__name__)

BUCKET = "datalake-backup"
RETENCAO = 7  # backups mantidos por item (catalogo, cada collection)


class Backup:
    def __init__(self) -> None:
        self.storage = StorageFuncions()
        self.url_qdrant = url_qdrant().rstrip("/")
        self.qdrant = QdrantClient(url=self.url_qdrant)
        self.momento = agora()

    def _chave(self, prefixo: str, nome: str, extensao: str) -> str:
        dia = self.momento.strftime("%Y/%m/%d")
        marca = self.momento.strftime("%Y%m%d_%H%M%S")
        return f"{prefixo}/{dia}/{nome}_{marca}.{extensao}"

    # --- catalogo -------------------------------------------------------------

    def catalogo(self, pasta: Path) -> str | None:
        """Copia consistente do SQLite do catalogo. None se nao for SQLite."""
        uri = uri_catalogo()
        if not uri.startswith("sqlite:///"):
            # Postgres (ou outro) tem backup proprio -- pg_dump, nao arquivo
            logger.warning("Catalogo nao e SQLite (%s) -- backup do catalogo pulado",
                           uri.split(":", 1)[0])
            return None
        origem = uri.removeprefix("sqlite:///")
        copia = pasta / "catalogo.db"
        with sqlite3.connect(origem) as fonte, sqlite3.connect(copia) as destino:
            fonte.backup(destino)
        key = self._chave("catalogo", "catalogo", "db")
        self.storage.save_file(copia, BUCKET, key)
        return key

    # --- qdrant ---------------------------------------------------------------

    def collection(self, tema: str, pasta: Path) -> str:
        """Snapshot de uma collection: cria no Qdrant, baixa, sobe, apaga do Qdrant.

        Apagar do Qdrant depois de subir: o snapshot ocupa o disco do proprio
        Qdrant, e a copia que vale e a do bucket.
        """
        snapshot = self.qdrant.create_snapshot(tema, wait=True)
        if snapshot is None:
            raise RuntimeError(f"Qdrant nao devolveu snapshot de {tema}")
        arquivo = pasta / snapshot.name
        try:
            with requests.get(f"{self.url_qdrant}/collections/{tema}/snapshots/{snapshot.name}",
                              stream=True, timeout=600) as resposta:
                resposta.raise_for_status()
                with open(arquivo, "wb") as saida:
                    for bloco in resposta.iter_content(chunk_size=1 << 20):
                        saida.write(bloco)
            key = self._chave(f"qdrant/{tema}", tema, "snapshot")
            self.storage.save_file(arquivo, BUCKET, key)
        finally:
            self.qdrant.delete_snapshot(tema, snapshot.name, wait=True)
        return key

    # --- retencao -------------------------------------------------------------

    def aplicar_retencao(self, prefixo: str) -> int:
        """Apaga o que passa dos RETENCAO mais recentes sob o prefixo.

        A marca de data e hora no nome faz a ordem alfabetica ser a
        cronologica -- nao precisa de LastModified.
        """
        chaves = sorted(self.storage.list_keys(BUCKET, prefixo),
                        key=lambda k: k.rsplit("/", 1)[1])
        antigas = chaves[:-RETENCAO] if len(chaves) > RETENCAO else []
        for key in antigas:
            self.storage.delete(BUCKET, key)
        return len(antigas)

    # --- execucao -------------------------------------------------------------

    def run(self) -> dict[str, int]:
        """Backup do catalogo e das tres collections. Devolve a contagem.

        Falha de um item nao impede os outros; so tudo falhando vira excecao.
        """
        situacao: Counter[str] = Counter()
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp)
            itens = [("catalogo", lambda: self.catalogo(pasta))] + [
                (f"qdrant/{tema}", lambda t=tema: self.collection(t, pasta)) for tema in TEMAS]
            for prefixo, fazer in itens:
                try:
                    key = fazer()
                except Exception:
                    situacao["falha"] += 1
                    logger.exception("Backup de %s falhou", prefixo)
                    continue
                if key is None:
                    continue
                situacao["gravado"] += 1
                situacao["apagado_por_retencao"] += self.aplicar_retencao(prefixo + "/")
                logger.info("Backup gravado em %s/%s", BUCKET, key)

        logger.info("Backup: %s", dict(situacao))
        if situacao["falha"] and not situacao["gravado"]:
            raise RuntimeError("nenhum backup foi gravado -- ver os erros acima")
        return dict(situacao)


def restaurar_qdrant(key: str, collection: str) -> int:
    """Recria `collection` a partir de um snapshot do bucket. Devolve os pontos.

    Sobrescreve a collection se ela existir. Para testar sem risco, restaure
    num nome novo e compare.
    """
    storage = StorageFuncions()
    url = url_qdrant().rstrip("/")
    with tempfile.TemporaryDirectory() as tmp:
        arquivo = Path(tmp) / "restaurar.snapshot"
        storage.download_file(BUCKET, key, arquivo)
        with open(arquivo, "rb") as f:
            resposta = requests.post(
                f"{url}/collections/{collection}/snapshots/upload",
                params={"priority": "snapshot", "wait": "true"},
                files={"snapshot": (arquivo.name, f)}, timeout=1800)
        resposta.raise_for_status()
    return QdrantClient(url=url).count(collection, exact=True).count


if __name__ == "__main__":
    setup_logging()
    Backup().run()
