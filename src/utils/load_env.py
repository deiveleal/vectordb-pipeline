"""Carga do .env da raiz do projeto."""

from pathlib import Path
from dotenv import load_dotenv


def load_env():
    """Carrega o .env que fica na raiz do repositorio, acima de src/.

    O caminho e derivado da posicao deste arquivo, nao do diretorio de
    trabalho: assim funciona igual rodando de src/, da raiz ou de dentro
    de um container.

    Variavel ja presente no ambiente vence a do arquivo -- comportamento
    padrao do python-dotenv, que e o que permite sobrepor na linha de comando.

    Atencao: o load_logger NAO se beneficia disto. O setup_logging roda no
    inicio do entry point, antes de qualquer BotoClient, que e quem chama esta
    funcao -- entao LOG_LEVEL e LOG_JSON so valem como variavel de ambiente de
    verdade, nao pelo .env.
    """
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")