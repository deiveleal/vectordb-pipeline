"""Cliente OAI-PMH do arXiv: coleta em massa de metadados por categoria e dia.

Substituiu a API de busca (export.arxiv.org/api/query) na camada bronze. A API
de busca passou a responder 406 na CDN para este IP desde 2026-09-24; o
OAI-PMH e a interface que o proprio arXiv indica para coleta em volume, e
responde normalmente.

Como o OAI se comporta -- medido em 2026-09-26, set cs:cs:CL:

- O filtro from/until e o `datestamp`: data da ULTIMA alteracao do registro,
  nao da submissao. Artigos antigos revisados no dia aparecem junto; separar o
  que e novo e regra de negocio, e fica para a gold.
- Sabado e domingo nao tem registro (nao ha anuncio). O OAI responde 200 com
  <error code='noRecordsMatch'>, que aqui vira "nenhuma pagina", nao excecao.
- O dia D ja esta completo na manha de D+1.
- Resultado grande vem paginado com <resumptionToken>; a pagina seguinte e
  pedida so com o token, sem repetir set/data.
"""

import datetime as dt
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import requests

from src.utils.load_logger import get_logger

logger = get_logger(__name__)

NS_OAI = {"o": "http://www.openarchives.org/OAI/2.0/"}

# arXivRaw, e nao o formato "arXiv": so ele traz a data de cada versao. No
# "arXiv" o <created> nao e a data da v1 -- 2005.11401 foi submetido em
# 2020-05-22 e vem com created=2021-04-12.
METADATA_PREFIX = "arXivRaw"

# Os termos de uso do arXiv pedem no maximo uma chamada a cada 3s, uma conexao
# por vez. 4s da margem: um excesso ja custou o bloqueio da API de busca. Vale
# entre paginas, categorias e tentativas.
PAUSA_ENTRE_CHAMADAS_S = 4.0

# O arXiv ja cortou excesso de chamadas com 429 e corpo "Rate exceeded." --
# medido na API de busca em 2026-09-21, sem cabecalho Retry-After. Quando o
# servidor nao diz quanto esperar, a espera e nossa: 5s, 15s, 45s. Passar
# disso em um pipeline diario nao compensa.
TENTATIVAS = 4
ESPERA_INICIAL_S = 5.0
FATOR_DE_BACKOFF = 3

# 429 e os 5xx sao transitorios: a mesma requisicao tende a funcionar depois.
# Um 400 ou 404 nao muda com insistencia, entao falha na primeira.
STATUS_TRANSITORIOS = frozenset({429, 500, 502, 503, 504})


def _resposta_de(erro: requests.RequestException) -> requests.Response | None:
    """Extrai a Response anexada ao erro, quando houve uma.

    Timeout e recusa de conexao nao tem resposta: a falha aconteceu antes de o
    servidor dizer qualquer coisa.
    """
    resposta = getattr(erro, "response", None)
    return resposta if isinstance(resposta, requests.Response) else None


def vale_repetir(erro: requests.RequestException) -> bool:
    """Se a falha e do tipo que costuma passar sozinha.

    Separa o que muda com o tempo (429, 5xx, rede) do que nao muda (400, 404).
    Repetir um erro permanente so atrasaria o pipeline em ~65s antes de falhar
    do mesmo jeito.
    """
    resposta = _resposta_de(erro)
    if resposta is None:
        # sem resposta: conexao recusada, timeout, DNS. Transitorio por natureza.
        return isinstance(erro, (requests.ConnectionError, requests.Timeout))
    return resposta.status_code in STATUS_TRANSITORIOS


def pausa_ate_a_proxima(erro: requests.RequestException, padrao: float) -> float:
    """Respeita o Retry-After quando existir; cai no backoff quando nao.

    O OAI do arXiv usa 503 + Retry-After para controle de carga, e proxies e
    CDNs na frente dele tambem mandam -- obedecer ao servidor e melhor que
    adivinhar.
    """
    resposta = _resposta_de(erro)
    if resposta is None:
        return padrao
    cabecalho = resposta.headers.get("Retry-After")
    if not cabecalho:
        return padrao
    try:
        return float(cabecalho)
    except ValueError:
        # a RFC tambem permite uma data HTTP; nesse caso o backoff resolve
        return padrao


def set_da_categoria(categoria: str) -> str:
    """cs.CL -> cs:cs:CL, o setSpec que o OAI do arXiv usa por subcategoria."""
    arquivo = categoria.split(".")[0]
    return f"{arquivo}:{categoria.replace('.', ':')}"


def contar_registros(pagina: bytes) -> int:
    """Quantos <record> vieram na pagina."""
    return len(ET.fromstring(pagina).findall(".//o:record", NS_OAI))


class ErroOAI(Exception):
    """Erro de protocolo OAI (badArgument, badResumptionToken...).

    Vem com HTTP 200, entao raise_for_status nao pega: e preciso ler o XML.
    """


class OAIConnect:
    """Coleta de um set num dia, pagina a pagina.

    O intervalo entre chamadas e controlado pela classe, nao pela instancia:
    o bronze cria uma instancia por categoria, e a pausa tem que valer entre
    elas tambem.
    """

    URL_OAI = "https://oaipmh.arxiv.org/oai"
    _ultima_chamada = 0.0

    def __init__(self, categoria: str, dia: dt.date):
        self.categoria = categoria
        self.dia = dia

    @classmethod
    def _esperar_a_vez(cls) -> None:
        falta = PAUSA_ENTRE_CHAMADAS_S - (time.monotonic() - cls._ultima_chamada)
        if falta > 0:
            time.sleep(falta)
        cls._ultima_chamada = time.monotonic()

    def _get(self, params: dict[str, str]) -> bytes:
        """Uma chamada, com retry no que for transitorio: 5s, 15s, 45s ou o
        Retry-After do servidor."""
        espera = ESPERA_INICIAL_S
        for tentativa in range(1, TENTATIVAS + 1):
            self._esperar_a_vez()
            try:
                resposta = requests.get(self.URL_OAI, params=params, timeout=60)
                resposta.raise_for_status()
                return resposta.content
            except requests.RequestException as erro:
                if tentativa == TENTATIVAS or not vale_repetir(erro):
                    logger.exception("Falha no OAI do arXiv (%s)", self.categoria)
                    raise
                descanso = pausa_ate_a_proxima(erro, espera)
                logger.warning(
                    "OAI recusou a chamada (%s). Tentativa %d de %d -- "
                    "nova tentativa em %.0fs.",
                    erro, tentativa, TENTATIVAS, descanso)
                time.sleep(descanso)
                espera *= FATOR_DE_BACKOFF
        raise RuntimeError("laco de tentativas terminou sem resposta nem erro")

    def paginas(self) -> Iterator[bytes]:
        """Devolve o XML cru de cada pagina do dia, na ordem.

        Dia sem registro (fim de semana, categoria parada) nao devolve nada.
        """
        dia = self.dia.isoformat()
        params = {
            "verb": "ListRecords",
            "metadataPrefix": METADATA_PREFIX,
            "set": set_da_categoria(self.categoria),
            "from": dia,
            "until": dia,
        }
        while True:
            pagina = self._get(params)
            raiz = ET.fromstring(pagina)

            erro = raiz.find("o:error", NS_OAI)
            if erro is not None:
                if erro.get("code") == "noRecordsMatch":
                    return
                raise ErroOAI(f"{erro.get('code')}: {(erro.text or '').strip()}")

            yield pagina

            # Na ultima pagina o token vem vazio (<resumptionToken .../>) ou
            # nem vem.
            token = raiz.find(".//o:resumptionToken", NS_OAI)
            valor = (token.text or "").strip() if token is not None else ""
            if not valor:
                return
            params = {"verb": "ListRecords", "resumptionToken": valor}
