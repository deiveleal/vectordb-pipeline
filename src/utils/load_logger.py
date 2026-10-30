"""Logging do projeto.

Regra: modulos pegam logger, a aplicacao configura. Por isso nada aqui roda
no import -- quem chama setup_logging e o entry point de cada camada.
Configurar dentro de uma classe utilitaria faria a primeira instancia
definir o formato de todo o processo, e as chamadas seguintes seriam
ignoradas em silencio (basicConfig so tem efeito uma vez).
"""

import json
import logging
import os
import sys
from datetime import datetime

from src.utils.load_time import FUSO

FORMATO = "%(asctime)s | %(levelname)-8s | %(name)-34s | %(message)s"
# O %z no fim e o que torna a linha comparavel com o LastModified do S3, que e
# sempre UTC: sem ele "22:02:54" nao diz de qual fuso e.
DATA_FORMATO = "%Y-%m-%d %H:%M:%S%z"

# Em DEBUG estas bibliotecas emitem ~50 registros por chamada S3 -- medido.
# Fixa-las em WARNING preserva o DEBUG para o codigo do projeto. O pyiceberg
# loga em INFO "Loaded LocationProvider" a cada arquivo gravado.
BIBLIOTECAS_RUIDOSAS = ("boto3", "botocore", "urllib3", "s3transfer", "pyiceberg")


class HoraBrasilia(logging.Formatter):
    """Base dos formatadores: timestamp em Brasilia, com o offset na linha.

    Sobrescrever formatTime -- e nao trocar o `converter` da classe, como se
    costuma fazer -- e o que faz o %z sair certo: o converter devolve um
    struct_time, que nao carrega o offset, entao o strftime cairia no fuso do
    container e imprimiria +0000 num horario que e -0300. Um offset errado
    engana mais que nenhum.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        momento = datetime.fromtimestamp(record.created, FUSO)
        if datefmt:
            return momento.strftime(datefmt)
        # ISO-8601 -- o formato que Athena, CloudWatch e Grafana leem sozinhos.
        return momento.isoformat(timespec="milliseconds")


class FormatadorJson(HoraBrasilia):
    """Uma linha JSON por registro, para agregadores de log em container."""

    def format(self, record: logging.LogRecord) -> str:
        dado = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "file": f"{record.filename}:{record.lineno}",
        }
        if record.exc_info:
            dado["exc"] = self.formatException(record.exc_info)
        return json.dumps(dado, ensure_ascii=False)


def setup_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    """Configura a saida de log. Chamar UMA vez, no entry point.

    Sem argumentos, le LOG_LEVEL e LOG_JSON do ambiente -- assim da para
    subir para DEBUG em producao sem editar codigo nem refazer a imagem.
    """
    # o getenv fica fora do `or` de proposito: dentro dele o mypy resolve a
    # sobrecarga com o contexto do lado esquerdo e o tipo vira str | None
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    nivel = level.upper()
    if json_logs is None:
        json_logs = os.getenv("LOG_JSON", "").lower() in ("1", "true", "yes")

    # stdout, nao stderr: em container o padrao do logging separaria os
    # registros em dois streams sem motivo.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(FormatadorJson() if json_logs
                         else HoraBrasilia(FORMATO, datefmt=DATA_FORMATO))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(nivel)

    for nome in BIBLIOTECAS_RUIDOSAS:
        logging.getLogger(nome).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Logger do modulo chamador. Uso: logger = get_logger(__name__).

    Passar __name__ e o que faz cada linha dizer de onde veio
    (src.load_bronze, src.utils.oai_connect, ...) e o que
    permite ligar DEBUG de um modulo so.

    No modulo executado como entry point o __name__ vale "__main__", o que
    apagaria justamente a origem da camada que esta rodando. Sob `python -m`
    o nome real fica em __spec__.name, entao recuperamos dali.
    """
    if name == "__main__":
        spec = getattr(sys.modules["__main__"], "__spec__", None)
        if spec is not None:
            name = spec.name
    return logging.getLogger(name)
