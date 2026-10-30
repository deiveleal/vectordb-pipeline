"""Construcao do cliente boto3 apontado para o RustFS.

Isolado do StorageFuncions de proposito: credencial e endereco sao assunto de
ambiente, nao de operacao de objeto, e separar permite testar a montagem do
cliente sem abrir conexao nenhuma.
"""

import os

import boto3

from src.utils.load_env import load_env
from src.utils.load_logger import get_logger

logger = get_logger(__name__)

# Endereco visto do HOST. De dentro de um container da rede do compose, use
# S3_ENDPOINT_URL=http://rustfs:9000.
ENDPOINT_PADRAO = "http://localhost:9010"


def endpoint_s3() -> str:
    """Endereco do RustFS: S3_ENDPOINT_URL, ou o do host.

    Fonte unica para o boto3 e para o catalogo Iceberg -- os dois precisam
    apontar para o mesmo lugar.
    """
    load_env()
    return os.environ.get("S3_ENDPOINT_URL", ENDPOINT_PADRAO)


class BotoClient:
    """Cliente S3 configurado para o RustFS local."""

    def __init__(self, endpoint_url: str | None = None):
        """Monta o cliente com as credenciais do ambiente.

        Sem `endpoint_url`, usa endpoint_s3().

        load_env() roda antes da leitura das variaveis; invertida, a ordem
        faria as credenciais sairem vazias sem aviso.

        Credencial ausente vira None em vez de erro, e o boto3 cai no
        encadeamento padrao da AWS. O sintoma aparece depois, como AccessDenied
        na primeira operacao, sem apontar a causa.
        """
        load_env()
        endpoint_url = endpoint_url or endpoint_s3()
        try:
            logger.info(
                "Initializing Boto3 S3 client with endpoint: %s", endpoint_url)
            self.boto_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
                aws_secret_access_key=os.environ.get("S3_SECRET_KEY")
            )
        except Exception:
            logger.exception(
                "Error occurred while initializing Boto3 S3 client")
            raise

    def get_boto_client(self):
        """Devolve o cliente ja construido -- sempre a mesma instancia."""
        return self.boto_client
