"""Catalogo Iceberg das camadas silver e gold.

SQL catalog do PyIceberg sobre SQLite: um arquivo local guarda so o ponteiro
para o metadata.json atual de cada tabela; dados e metadados ficam no RustFS.
Nao ha servico a subir -- mas o arquivo e a fonte da verdade de "qual e a
versao atual" e precisa de lugar persistente. Padrao: data/iceberg/catalogo.db,
na raiz do projeto (data/ esta no .gitignore). Em container, aponte
ICEBERG_CATALOG_URI para um volume.

SQLite aceita um escritor por vez. Serve enquanto as camadas rodam em
sequencia no mesmo flow; com escritores concorrentes, troque por Postgres
(ICEBERG_CATALOG_URI=postgresql+psycopg2://...) sem mudar mais nada.

Testado em 2026-09-26 com PyIceberg 0.12 sobre o RustFS: append, overwrite de
particao atomico, upsert por chave e time travel funcionando com o usuario de
servico.
"""

import datetime as dt
import os
from pathlib import Path

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.partitioning import PartitionKey
from pyiceberg.table import Table
from pyiceberg.table.locations import LocationProvider
from pyiceberg.types import DateType

from src.utils.load_boto_client import endpoint_s3
from src.utils.load_env import load_env
from src.utils.load_logger import get_logger

logger = get_logger(__name__)

NAMESPACE = "arxiv"
RAIZ = Path(__file__).resolve().parents[2]
CATALOGO_PADRAO = RAIZ / "data" / "iceberg" / "catalogo.db"

class LocationPorDia(LocationProvider):
    """Arquivos de dados em <tabela>/AAAA/MM/DD/data/, o recorte de data do bronze.

    Com a tabela em s3://<bucket>/iceberg/arxiv, um arquivo da silver de 25/09
    fica em datalake-silver/iceberg/arxiv/2026/09/25/data/<arquivo>.parquet.
    Sem isto o Iceberg gravaria em <tabela>/data/datestamp=2026-09-25/.

    So o caminho muda: a particao continua identity(<coluna date>), entao
    filtrar por ela continua pulando os arquivos dos outros dias. Os metadados
    seguem em <tabela>/metadata/.

    Ligado por tabela na propriedade write.py-location-provider.impl. Vale so
    para escrita pelo PyIceberg; leitura usa o caminho gravado no manifesto e
    nao precisa desta classe.
    """

    def new_data_location(self, data_file_name: str,
                          partition_key: PartitionKey | None = None) -> str:
        dia = _dia_da_particao(partition_key)
        if dia is None:
            return f"{self.data_path}/{data_file_name}"
        return f"{self.table_location}/{dia.strftime('%Y/%m/%d')}/data/{data_file_name}"


def _dia_da_particao(partition_key: PartitionKey | None) -> dt.date | None:
    """O dia da particao: o valor da primeira coluna de particao do tipo date.

    Vale para qualquer coluna date -- o datestamp na silver e na gold, a data
    da v1 nos chunks. Tabela sem particao por data grava em data/ direto.
    """
    if partition_key is None:
        return None
    for valor in partition_key.field_values:
        coluna = partition_key.schema.find_field(valor.field.source_id)
        if not isinstance(coluna.field_type, DateType):
            continue
        # o Iceberg representa date como dias desde 1970-01-01
        if isinstance(valor.value, int):
            return dt.date(1970, 1, 1) + dt.timedelta(days=valor.value)
        if isinstance(valor.value, dt.date):
            return valor.value
    return None


PROPRIEDADES = {
    "write.py-location-provider.impl": f"{__name__}.{LocationPorDia.__name__}",
}


def uri_catalogo() -> str:
    """URI do banco do catalogo: ICEBERG_CATALOG_URI, ou o SQLite padrao.

    Fonte unica para quem abre o catalogo e para o backup, que copia o arquivo.
    """
    load_env()
    uri = os.environ.get("ICEBERG_CATALOG_URI")
    if not uri:
        CATALOGO_PADRAO.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{CATALOGO_PADRAO}"
    return uri


def load_catalog() -> SqlCatalog:
    """Abre o catalogo e garante o namespace `arxiv`."""
    load_env()
    uri = uri_catalogo()

    catalog = SqlCatalog(
        "arxiv",
        uri=uri,
        # so o local padrao de tabela nova; cada tabela daqui diz o proprio
        warehouse="s3://datalake-silver/iceberg",
        **{
            "s3.endpoint": endpoint_s3(),
            # [] e nao .get(): credencial ausente falha aqui, com o nome da
            # variavel, em vez de virar AccessDenied na primeira escrita
            "s3.access-key-id": os.environ["S3_ACCESS_KEY"],
            "s3.secret-access-key": os.environ["S3_SECRET_KEY"],
            # o RustFS ignora regiao, mas o cliente S3 do pyarrow exige uma
            "s3.region": "us-east-1",
            "s3.path-style-access": "true",
        },
    )
    catalog.create_namespace_if_not_exists(NAMESPACE)
    return catalog


def abrir_ou_criar_tabela(
    catalog: SqlCatalog,
    nome: str,
    schema: pa.Schema,
    bucket: str,
    particao: str | None = None,
    chave: str | None = None,
) -> Table:
    """Abre `arxiv.<nome>`; na primeira vez, cria com schema, particao e chave.

    A tabela fica em s3://<bucket>/iceberg/arxiv -- o nome da camada nao entra
    no caminho, o bucket ja diz qual e. Os dados vao para AAAA/MM/DD/data/
    (LocationPorDia) e os metadados para metadata/. Por isso cabe UMA tabela
    por bucket: uma segunda no mesmo bucket dividiria a pasta metadata/.

    `particao` e uma coluna com particionamento identity (o datestamp);
    `chave` e o identifier field do upsert (o arxiv_id da gold) e precisa ser
    uma coluna nao nula no schema.

    Schema de tabela existente nao e alterado aqui: evolucao de schema e
    decisao explicita, nao efeito colateral de rodar a camada.
    """
    identificador = f"{NAMESPACE}.{nome}"
    try:
        return catalog.load_table(identificador)
    except NoSuchTableError:
        pass

    local = f"s3://{bucket}/iceberg/{NAMESPACE}"
    ocupado = [t for t in catalog.list_tables(NAMESPACE)
               if catalog.load_table(t).location().rstrip("/") == local]
    if ocupado:
        raise ValueError(f"{local} ja e a tabela {ocupado[0]}; uma tabela por bucket")

    logger.info("Criando tabela Iceberg %s em %s", identificador, local)
    tabela = catalog.create_table(
        identificador,
        schema=schema,
        location=local,
        properties=PROPRIEDADES,
    )
    if particao:
        with tabela.update_spec() as spec:
            spec.add_identity(particao)
    if chave:
        with tabela.update_schema() as update:
            update.set_identifier_fields(chave)
    return tabela
