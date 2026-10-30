"""Fuso horario do projeto.

Regra: data de negocio (chave de objeto, particao, janela de consulta) sai
daqui, nunca de dt.date.today() ou dt.datetime.now() sem fuso -- esses leem o
relogio do container, que em imagem padrao e UTC, e entre 21h e meia-noite de
Brasilia ja gravariam o dia seguinte no nome do arquivo.

O S3 nao entra nessa conta: o LastModified dele e sempre UTC, definido pelo
servidor da AWS e sem opcao de configuracao. Compatibilidade com ele se faz na
leitura, com `momento.astimezone(FUSO)`, nao tentando mudar a origem.
"""

import datetime as dt
from zoneinfo import ZoneInfo

# Nome IANA, nao offset fixo: assim o horario de verao volta a ser tratado
# sozinho caso seja reinstituido.
FUSO = ZoneInfo("America/Sao_Paulo")


def agora() -> dt.datetime:
    """Momento atual em Brasilia, com fuso embutido (aware)."""
    return dt.datetime.now(FUSO)


def hoje() -> dt.date:
    """Data corrente em Brasilia -- a data de negocio do pipeline."""
    return agora().date()


# A busca do arXiv indexa o submittedDate com atraso. Medido em 2026-09-20 com
# o termo amplo all:"agents", uma janela de um dia devolvia:
#
#   D-1  0 artigos        D-4  239        D-7  196
#   D-2  66 (parcial)     D-5  222        D-8  114
#   D-3  183              D-6  231
#
# Nao e efeito de fim de semana (D-7 e domingo e tem 196). D-3 e o primeiro dia
# estavel, entao e dele que o pipeline trata. Perguntar por "ontem" traria zero.
DIAS_DE_DEFASAGEM = 1


def dia_de_referencia() -> dt.date:
    """Dia que o pipeline processa: o mais recente ja indexado pelo arXiv.

    Fonte unica para as tres camadas. Se bronze, silver e documents calculassem
    a propria data, bastaria uma divergir para o silver procurar um objeto que
    o bronze gravou com outro nome.
    """
    return hoje() - dt.timedelta(days=DIAS_DE_DEFASAGEM)
