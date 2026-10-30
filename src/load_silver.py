"""Camada silver: datalake-bronze -> datalake-silver.

Le o XML arXivRaw que o bronze gravou para o dia e grava na tabela Iceberg
`arxiv.silver` (s3://datalake-silver/iceberg/arxiv), uma linha por artigo,
particionada por datestamp. Os arquivos do dia ficam em
iceberg/arxiv/AAAA/MM/DD/data/, como o recorte de data do bronze.

Reprocessar um dia substitui so a particao dele, num commit atomico
(dynamic_partition_overwrite): quem le ve o dia inteiro antes ou inteiro
depois, nunca pela metade, e os outros dias nao sao tocados.

So faz parse e limpeza tecnica -- nenhuma regra de negocio:

- entrada pelo manifesto de cada categoria, lendo cada pagina na versao exata
  que ele registra. Paginas de coletas anteriores que sobraram no bucket ficam
  de fora (ver o docstring do load_bronze);
- espaco em branco normalizado em titulo e abstract (o XML traz quebra de linha
  e indentacao no meio do texto);
- datas das versoes convertidas para datetime UTC;
- duplicata entre categorias removida: artigo listado em cs.CL e cs.LG vem nas
  duas coletas e e o mesmo artigo.

Artigo revisado (v2+) e artigo novo (so v1) ficam os dois aqui, com n_versoes e
data_v1 para distinguir. Decidir o que interessa -- so artigos novos, so os
temas RAG / Agents / LLM -- e papel da gold.

Uso (da raiz do projeto): python -m src.load_silver
"""

import datetime as dt
import json
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pyarrow as pa

from src.load_bronze import BUCKET as BUCKET_BRONZE
from src.load_bronze import CATEGORIAS, chave_manifesto
from src.utils.load_catalog import abrir_ou_criar_tabela, load_catalog
from src.utils.load_logger import get_logger, setup_logging
from src.utils.load_time import dia_de_referencia
from src.utils.oai_connect import NS_OAI
from src.utils.storage_functions import StorageFuncions

logger = get_logger(__name__)

BUCKET = "datalake-silver"
TABELA = "silver"  # arxiv.silver no catalogo
NS = {**NS_OAI, "r": "http://arxiv.org/OAI/arXivRaw/"}

# Schema explicito, e nao inferido dos dados: inferencia muda de tipo quando
# um dia vem com coluna toda nula, e o Iceberg recusa append com schema
# diferente do da tabela. nullable=False vira coluna required no Iceberg.
SCHEMA = pa.schema([
    pa.field("arxiv_id", pa.string(), nullable=False),
    pa.field("titulo", pa.string()),
    pa.field("abstract", pa.string()),
    pa.field("autores", pa.string()),
    pa.field("autores_lista", pa.list_(pa.string())),
    pa.field("submitter", pa.string()),
    pa.field("categorias", pa.list_(pa.string())),
    pa.field("categoria_principal", pa.string()),
    pa.field("n_versoes", pa.int32()),
    pa.field("data_v1", pa.timestamp("us", tz="UTC")),
    pa.field("data_ultima_versao", pa.timestamp("us", tz="UTC")),
    pa.field("datestamp", pa.date32(), nullable=False),
    pa.field("doi", pa.string()),
    pa.field("journal_ref", pa.string()),
    pa.field("comments", pa.string()),
    pa.field("license", pa.string()),
    pa.field("pdf_url", pa.string()),
])


def limpar(texto: str | None) -> str | None:
    """Colapsa quebras de linha e espacos repetidos que vem do XML."""
    return " ".join(texto.split()) if texto else None


def separar_autores(autores: str) -> list[str]:
    """Lista de nomes a partir do texto livre do arXivRaw.

    O campo e uma string so, as vezes com afiliacao entre parenteses -- ate
    aninhados: "Ana (1 and 2), Bia (2) ((1) USP, (2) Unicamp)". Os parenteses
    saem primeiro, de dentro para fora, e o resto e separado por virgula e
    " and ". E aproximado: um nome com virgula quebraria. A string original
    fica na coluna `autores` para quando isso importar.
    """
    anterior = None
    while anterior != autores:
        anterior, autores = autores, re.sub(r"\([^()]*\)", "", autores)
    nomes = re.split(r",|\band\b", autores)
    return [n for n in (limpar(n) for n in nomes) if n]


def ler_registro(record: ET.Element) -> dict | None:
    """Um <record> do OAI vira um dicionario. None para registro deletado."""
    header = record.find("o:header", NS)
    if header is None or header.get("status") == "deleted":
        return None
    raw = record.find("o:metadata/r:arXivRaw", NS)
    if raw is None:
        return None

    def campo(nome: str) -> str | None:
        return limpar(raw.findtext(f"r:{nome}", namespaces=NS))

    versoes = [
        parsedate_to_datetime(data)
        for v in raw.findall("r:version", NS)
        if (data := v.findtext("r:date", default="", namespaces=NS))
    ]
    categorias = (campo("categories") or "").split()
    autores = campo("authors") or ""
    arxiv_id = campo("id")

    return {
        "arxiv_id": arxiv_id,
        "titulo": campo("title"),
        "abstract": campo("abstract"),
        "autores": autores,
        "autores_lista": separar_autores(autores),
        "submitter": campo("submitter"),
        "categorias": categorias,
        # a primeira da lista e a principal -- convencao do arXiv
        "categoria_principal": categorias[0] if categorias else None,
        "n_versoes": len(versoes),
        "data_v1": min(versoes) if versoes else None,
        "data_ultima_versao": max(versoes) if versoes else None,
        "datestamp": dt.date.fromisoformat(
            header.findtext("o:datestamp", default="", namespaces=NS)),
        "doi": campo("doi"),
        "journal_ref": campo("journal-ref"),
        "comments": campo("comments"),
        "license": campo("license"),
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


class LoadSilver:
    """Parse de um dia do bronze para a tabela Iceberg arxiv.silver.

    Uma instancia trata um unico dia. Sem argumento, e o dia_de_referencia();
    para reprocessar outra data, passe `dia` -- e o que o orquestrador usa.
    """

    def __init__(self, dia: dt.date | None = None):
        self.storage = StorageFuncions()
        self.dia_alvo = dia or dia_de_referencia()

    def ler_manifesto(self, categoria: str) -> dict:
        """Manifesto da categoria no dia.

        NoSuchKey aqui quer dizer que o bronze nao rodou para esta data --
        dia coletado e vazio tem manifesto, com paginas: [].
        """
        key = chave_manifesto(categoria, self.dia_alvo)
        return json.loads(self.storage.load_data(BUCKET_BRONZE, key)["Body"].read())

    def ler_categoria(self, categoria: str) -> list[dict]:
        """Registros de todas as paginas do manifesto, na versao registrada."""
        manifesto = self.ler_manifesto(categoria)
        registros = []
        deletados = 0
        for pagina in manifesto["paginas"]:
            conteudo = self.storage.load_data(
                BUCKET_BRONZE, pagina["chave"], version_id=pagina["version_id"]
            )["Body"].read()
            for record in ET.fromstring(conteudo).findall(".//o:record", NS):
                linha = ler_registro(record)
                if linha is None:
                    deletados += 1
                else:
                    registros.append(linha)

        if len(registros) + deletados != manifesto["registros"]:
            # o manifesto conta no bronze, na hora da coleta; divergir aqui e
            # sinal de pagina trocada ou parse perdendo registro
            logger.warning("%s: manifesto diz %d registros, parse leu %d (+%d deletados)",
                           categoria, manifesto["registros"], len(registros), deletados)
        if deletados:
            logger.info("%s: %d registros deletados ignorados", categoria, deletados)
        logger.info("%s: %d registros lidos", categoria, len(registros))
        return registros

    def save_to_silver(self, tabela_dia: pa.Table) -> None:
        """Substitui a particao do dia na tabela arxiv.silver.

        dynamic_partition_overwrite troca so as particoes presentes em
        `tabela_dia` -- aqui, uma: o datestamp do dia. Delete e append vao no
        mesmo commit.
        """
        tabela = abrir_ou_criar_tabela(
            load_catalog(), TABELA, SCHEMA, BUCKET, particao="datestamp")
        tabela.dynamic_partition_overwrite(tabela_dia)

    def run(self, categorias: list[str] | None = None) -> int:
        """Le as categorias do dia, junta, deduplica e grava. Devolve o n. de artigos.

        `categorias` sao as que o bronze devolveu com registros; sem ela, le
        todas as de CATEGORIAS. Dia sem registro nenhum nao grava nada e
        devolve 0 -- a particao do dia, se existir, fica como esta.
        """
        linhas = []
        for categoria in categorias or CATEGORIAS:
            linhas.extend(self.ler_categoria(categoria))

        if not linhas:
            logger.warning("Nenhum registro no bronze para %s -- nada gravado.",
                           self.dia_alvo)
            return 0

        # artigo em mais de uma categoria coletada vem repetido, com o mesmo
        # conteudo -- qualquer copia serve
        por_id: dict[str, dict] = {}
        for linha in linhas:
            por_id.setdefault(linha["arxiv_id"], linha)
        unicos = list(por_id.values())

        fora_do_dia = [u["arxiv_id"] for u in unicos if u["datestamp"] != self.dia_alvo]
        if fora_do_dia:
            # o overwrite e por datestamp: registro de outro dia apagaria a
            # particao dele. O OAI filtra por datestamp, entao isto nao deveria
            # acontecer -- se acontecer, e melhor parar do que apagar dado.
            raise ValueError(f"{len(fora_do_dia)} registros com datestamp diferente de "
                             f"{self.dia_alvo}, ex.: {fora_do_dia[:3]}")

        self.save_to_silver(pa.Table.from_pylist(unicos, schema=SCHEMA))
        logger.info("Silver arxiv.%s, particao %s: %d artigos (%d antes de deduplicar)",
                    TABELA, self.dia_alvo, len(unicos), len(linhas))
        return len(unicos)


if __name__ == "__main__":
    setup_logging()
    silver_loader = LoadSilver()
    silver_loader.run()
