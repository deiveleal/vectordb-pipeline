"""Leitura e escrita no S3.

Unico ponto do projeto que fala com o bucket. As camadas do pipeline nao
conhecem boto3: trocar RustFS por AWS, ou por outro armazenamento, e mudar esta
classe e mais nada.

O nome da classe tem um erro de digitacao ("Funcions"). Esta assim em tres
modulos e nos testes; renomear e seguro, mas ainda nao foi feito.
"""

from botocore.exceptions import ClientError

from src.utils.load_boto_client import BotoClient
from src.utils.load_logger import get_logger, setup_logging

logger = get_logger(__name__)


class StorageFuncions:
    """Operacoes de objeto no datalake.

    Cada instancia abre o proprio cliente boto3. Sao baratos, mas uma instancia
    por camada e suficiente -- nao vale criar uma por chamada.
    """

    def __init__(self):
        self.s3 = BotoClient().get_boto_client()

    def save_data(self, data, bucket_name, key, content_type: str | None = None,
                  metadata: dict[str, str] | None = None):
        """Grava um objeto. `data` vai crua: bytes, str ou file-like.

        Nao serializa nem valida nada -- quem chama decide o formato. Com o
        versionamento ligado, gravar sobre uma chave existente cria versao
        nova em vez de substituir. `content_type` e opcional: sem ele o objeto
        fica como binario generico. `metadata` vira x-amz-meta-* no objeto --
        so ASCII, e limite do S3.

        Devolve a resposta crua do boto3: o VersionId dela e o que permite
        apontar para exatamente esta gravacao depois (o manifesto do bronze).
        """
        extra: dict = {}
        if content_type:
            extra["ContentType"] = content_type
        if metadata:
            extra["Metadata"] = metadata
        try:
            response = self.s3.put_object(
                Body=data,
                Bucket=bucket_name,
                Key=key,
                **extra
            )
            logger.info(
                "Data saved to S3 bucket '%s' successfully.", bucket_name)
            return response
        except Exception:
            logger.exception(
                "Error occurred while saving data to S3 bucket '%s'", bucket_name)
            raise

    def head(self, bucket_name, key) -> dict | None:
        """Metadados do objeto (x-amz-meta-*) sem baixar o conteudo, ou None.

        So 404 vira None. AccessDenied, rede e o resto propagam: tratar
        qualquer erro como "nao existe" faria quem chama gravar de novo algo
        que talvez ja esteja la.
        """
        try:
            return self.s3.head_object(Bucket=bucket_name, Key=key)["Metadata"]
        except ClientError as erro:
            if erro.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def exists(self, bucket_name, key) -> bool:
        """Se o objeto existe, sem baixar o conteudo (HEAD). Ver head()."""
        return self.head(bucket_name, key) is not None

    def load_data(self, bucket_name, key, version_id: str | None = None):
        """Busca um objeto e devolve a resposta crua do boto3.

        Nao e o conteudo: quem chama faz ["Body"].read(). Devolver o dicionario
        inteiro preserva ContentLength, LastModified e VersionId, que servem
        para diagnostico -- e que uma assinatura que so devolvesse bytes
        jogaria fora.

        Sem `version_id`, vem a versao mais recente. Com ele, exatamente aquela
        gravacao -- e como o silver le as paginas listadas no manifesto.

        O LastModified vem sempre em UTC, tanto na AWS quanto no RustFS; para
        exibir no fuso do projeto, use .astimezone(FUSO) do load_time.
        """
        try:
            logger.info(
                "Loading data from S3 bucket '%s' with key '%s'.", bucket_name, key)
            extra = {"VersionId": version_id} if version_id else {}
            response = self.s3.get_object(Bucket=bucket_name, Key=key, **extra)
            logger.info(
                "Data loaded from S3 bucket '%s' successfully.", bucket_name)
            return response
        except Exception:
            logger.exception(
                "Error occurred while loading data from S3 bucket '%s'", bucket_name)
            raise

    def save_file(self, caminho, bucket_name, key) -> None:
        """Grava um arquivo do disco. Para arquivo grande (snapshot do Qdrant):
        o boto3 manda em partes, sem carregar tudo na memoria."""
        self.s3.upload_file(str(caminho), bucket_name, key)
        logger.info("File %s saved to S3 bucket '%s' as '%s'.", caminho, bucket_name, key)

    def download_file(self, bucket_name, key, caminho) -> None:
        """Baixa um objeto para um arquivo no disco, em partes."""
        self.s3.download_file(bucket_name, key, str(caminho))

    def list_keys(self, bucket_name, prefix) -> list[str]:
        """Todas as chaves sob o prefixo, paginando o que passar de 1000."""
        chaves: list[str] = []
        for pagina in self.s3.get_paginator("list_objects_v2").paginate(
                Bucket=bucket_name, Prefix=prefix):
            chaves += [o["Key"] for o in pagina.get("Contents", [])]
        return chaves

    def delete(self, bucket_name, key) -> None:
        """Apaga um objeto. So funciona nos buckets em que o usuario de servico
        tem DeleteObject (silver, gold, chunks e backup)."""
        self.s3.delete_object(Bucket=bucket_name, Key=key)
        logger.info("Deleted '%s' from S3 bucket '%s'.", key, bucket_name)
