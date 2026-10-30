"""Camada bronze: arXiv (OAI-PMH) -> datalake-bronze.

Grava o XML exatamente como o OAI respondeu, sem transformacao nenhuma. E o
que torna todo o resto reprocessavel: se o parser do silver ou o filtro de
temas da gold mudarem, o dado original continua aqui para rodar de novo.

Coleta TUDO que mudou no dia em cada categoria -- nao ha busca por termo no
OAI. Separar artigos novos e classificar os temas (RAG, Agents, LLM) e papel
da gold.

Uma chave por pagina, particionada por categoria, ano, mes e dia:
arxiv/<categoria>/AAAA/MM/DD/<categoria>_AAAAMMDD_pNN.xml -- por exemplo
arxiv/cs.CL/2026/09/25/cs.CL_20260925_p01.xml. O nome repete categoria e data
de proposito: o arquivo continua identificavel fora do bucket. O bucket tem
versionamento e o usuario de servico nao tem DeleteObject, entao uma
reexecucao empilha versao em vez de sobrescrever.

Ao fim de cada categoria vai um manifesto, ao lado das paginas:
arxiv/<categoria>/AAAA/MM/DD/<categoria>_AAAAMMDD_manifest.json. Ele e a porta de
entrada do silver, que le so as paginas listadas ali, na versao exata. Sem
isso, um dia reprocessado que encolhesse (artigo revisado depois muda de
datestamp) deixaria para tras paginas da coleta anterior -- sem DeleteObject,
elas nunca somem -- e o silver misturaria as duas coletas.

O manifesto e gravado por ultimo: se a coleta cair no meio, o manifesto
anterior continua apontando para as versoes da coleta completa anterior.
Dia vazio tambem ganha manifesto (paginas: []), o que separa "coletado e sem
registro" de "nunca coletado" (NoSuchKey).

Uso (da raiz do projeto): python -m src.load_bronze
"""

import datetime as dt
import json

from src.utils.load_logger import get_logger, setup_logging
from src.utils.load_time import agora, dia_de_referencia
from src.utils.oai_connect import METADATA_PREFIX, OAIConnect, contar_registros
from src.utils.storage_functions import StorageFuncions

logger = get_logger(__name__)

# Categorias coletadas. Artigo listado em mais de uma aparece em todas; quem
# tira a duplicata e o silver.
CATEGORIAS = ["cs.CL", "cs.AI", "cs.LG"]

BUCKET = "datalake-bronze"


def _prefixo_do_dia(categoria: str, dia: dt.date) -> str:
    """arxiv/<categoria>/AAAA/MM/DD/<categoria>_AAAAMMDD -- base das chaves do dia."""
    return f"arxiv/{categoria}/{dia.strftime('%Y/%m/%d')}/{categoria}_{dia.strftime('%Y%m%d')}"


def chave_bronze(categoria: str, dia: dt.date, pagina: int) -> str:
    """Chave de uma pagina. Fonte unica para bronze e silver."""
    return f"{_prefixo_do_dia(categoria, dia)}_p{pagina:02d}.xml"


def chave_manifesto(categoria: str, dia: dt.date) -> str:
    """Chave do manifesto de uma categoria num dia, ao lado das paginas."""
    return f"{_prefixo_do_dia(categoria, dia)}_manifest.json"


class LoadBronze:
    """Coleta de um dia do OAI do arXiv para o bucket bronze.

    Uma instancia trata um unico dia. Sem argumento, e o dia_de_referencia();
    para reprocessar outra data, passe `dia` -- e o que o orquestrador usa.
    """

    def __init__(self, dia: dt.date | None = None):
        self.storage = StorageFuncions()
        # dia_de_referencia ja desconta a defasagem e sai do fuso de Brasilia
        # -- as camadas leem dessa mesma fonte.
        self.dia_alvo = dia or dia_de_referencia()

    def coletar_categoria(self, categoria: str) -> int:
        """Coleta e grava as paginas e o manifesto de uma categoria.

        Devolve quantos registros. As tentativas, o backoff e a pausa entre
        chamadas ficam no OAIConnect; o que chega aqui como excecao ja esgotou
        as repeticoes -- e o manifesto nao e gravado.
        """
        logger.info("Coletando %s em %s via OAI.", categoria, self.dia_alvo)
        paginas = []
        for n, pagina in enumerate(OAIConnect(categoria, self.dia_alvo).paginas(), 1):
            registros = contar_registros(pagina)
            key = chave_bronze(categoria, self.dia_alvo, n)
            resposta = self.storage.save_data(data=pagina, bucket_name=BUCKET, key=key)
            paginas.append({
                "chave": key,
                "version_id": resposta.get("VersionId"),
                "registros": registros,
            })
            logger.info("Gravados %d registros em %s", registros, key)

        total = sum(p["registros"] for p in paginas)
        self.gravar_manifesto(categoria, paginas, total)

        if not total:
            # Fim de semana ou categoria parada: o OAI responde noRecordsMatch
            # e so o manifesto vazio e gravado.
            logger.warning("Nenhum registro para %s em %s -- so o manifesto gravado.",
                           categoria, self.dia_alvo)
        return total

    def gravar_manifesto(self, categoria: str, paginas: list[dict], total: int) -> None:
        """Grava o manifesto da coleta: o que o silver deve ler, e em que versao."""
        manifesto = {
            "categoria": categoria,
            "dia": self.dia_alvo.isoformat(),
            "metadata_prefix": METADATA_PREFIX,
            "coletado_em": agora().isoformat(timespec="seconds"),
            "registros": total,
            "paginas": paginas,
        }
        key = chave_manifesto(categoria, self.dia_alvo)
        self.storage.save_data(
            data=json.dumps(manifesto, ensure_ascii=False, indent=2).encode(),
            bucket_name=BUCKET,
            key=key,
        )
        logger.info("Manifesto gravado em %s (%d paginas)", key, len(paginas))

    def run(self) -> list[str]:
        """Coleta e grava cada categoria. Devolve as que tinham registros.

        O retorno existe para o orquestrador: lista vazia quer dizer dia sem
        registro nenhum (fim de semana), e as camadas seguintes nao devem rodar.
        Uma categoria vazia nao impede as outras.
        """
        return [c for c in CATEGORIAS if self.coletar_categoria(c)]


if __name__ == "__main__":
    setup_logging()
    loader = LoadBronze()
    loader.run()
