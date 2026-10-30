"""Servidor MCP "arxiv-radar": a busca semantica como ferramentas para modelos.

Adaptador fino sobre a API HTTP (src/api.py) -- nao importa nada de src/. A
API continua sendo a unica porta de entrada: traducao, reranking e tratamento
de erro sao os dela. Aqui so se descreve cada operacao para o modelo e se
formata a resposta para ele citar.

    ferramentas   buscar_artigos      pergunta -> artigos numerados [n] com trechos
                  ler_artigo          mais trechos de um artigo ja encontrado
                  temas_disponiveis   temas e quantos trechos ha em cada um
    prompt        responder_com_citacoes
    recurso       arxiv://artigo/{arxiv_id}

A descricao de cada ferramenta e o que o modelo le para decidir quando e
como usa-la: tratar como prompt.

Cada ferramenta devolve as duas formas que o MCP permite: texto compacto e
numerado em `content`, para o modelo ler e citar, e o JSON da API em
`structured_content`, para cliente que quer os dados -- a interface Streamlit
monta os cartoes de resultado a partir dele.

Uso (da raiz do projeto):
    uv run python mcp_server/server.py                 # stdio (Claude Code, Claude Desktop, IDEs)
    uv run python mcp_server/server.py --http          # streamable HTTP em 127.0.0.1:8765/mcp
Variavel: API_URL (padrao http://localhost:8000).
"""

import argparse
import os
from typing import Literal

import requests
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

# Trecho inteiro tem ~1500 caracteres; cortar mantem o resultado da busca
# cabendo no contexto de modelo pequeno. ler_artigo devolve o texto inteiro.
MAX_CHARS_TRECHO_BUSCA = 900

SO_LEITURA = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                             idempotent_hint=True, open_world_hint=False)

mcp = MCPServer(
    name="arxiv-radar",
    title="arXiv Radar",
    description="Busca semantica em artigos recentes do arXiv sobre RAG, Agents e LLM.",
    instructions=(
        "Use buscar_artigos para achar artigos sobre a pergunta do usuario; pode "
        "chamar mais de uma vez com formulacoes ou temas diferentes. Use ler_artigo "
        "para aprofundar num artigo encontrado. Responda so com base no que as "
        "ferramentas devolveram e cite as fontes pelo numero [n]."
    ),
    # stdio: o log vai para o stderr do cliente; INFO registraria cada requisicao
    log_level="WARNING",
)


def _get(caminho: str, params: dict | None = None, aceitar: tuple[int, ...] = ()) -> dict:
    """GET na API, com erro legivel para o modelo em vez de traceback.

    aceitar: status de erro cujo corpo e resposta valida -- o /health responde
    503 com o diagnostico quando a busca esta fora, e isso e o que se quer ler.
    """
    try:
        resposta = requests.get(f"{API_URL}{caminho}", params=params, timeout=180)
    except requests.RequestException as erro:
        raise ToolError(f"API de busca inacessivel em {API_URL} ({type(erro).__name__})") from erro
    if resposta.status_code in aceitar:
        return resposta.json()
    if resposta.status_code == 404:
        raise ToolError(resposta.json().get("detail", "nao encontrado"))
    if resposta.status_code >= 400:
        raise ToolError(f"API respondeu {resposta.status_code}: {resposta.text[:200]}")
    return resposta.json()


def _cortar(texto: str, limite: int) -> str:
    return texto if len(texto) <= limite else texto[:limite].rsplit(" ", 1)[0] + " [...]"


def _resultado(texto: str, dados: dict) -> CallToolResult:
    """Texto para o modelo + dados para o cliente, na mesma resposta."""
    return CallToolResult(content=[TextContent(type="text", text=texto)],
                          structured_content=dados)


def _texto_busca(busca: dict) -> str:
    cabecalho = f'Busca: "{busca["pergunta_busca"]}"'
    if busca.get("traduzida"):
        cabecalho += f' (traduzida de "{busca["pergunta"]}")'
    cabecalho += f" · {busca['total']} artigos"
    if not busca["resultados"]:
        return cabecalho + "\nNenhum artigo encontrado. Tente outra formulacao ou outro tema."

    blocos = [cabecalho]
    for i, r in enumerate(busca["resultados"], 1):
        linhas = [f"[{i}] {r['titulo']}",
                  f"    arxiv_id: {r['arxiv_id']} · tema: {r['tema']} · v1: {r['data_v1']} "
                  f"· score: {r['score']:.3f}",
                  f"    {r['url']}"]
        for t in r["trechos"]:
            linhas.append(f"    - ({t['secao'] or 'sem secao'}) "
                          f"{_cortar(t['texto'], MAX_CHARS_TRECHO_BUSCA)}")
        blocos.append("\n".join(linhas))
    return "\n\n".join(blocos)


def _texto_artigo(artigo: dict, secao: str | None) -> str:
    trechos = artigo["trechos"]
    linhas = [f"{artigo['titulo']}",
              f"arxiv_id: {artigo['arxiv_id']} · tema: {artigo['tema']} · v1: {artigo['data_v1']}",
              f"{artigo['url']} · PDF: {artigo['pdf_url']}",
              f"{len(trechos)} de {artigo['total_trechos']} trechos"
              + (f' (filtro de secao: "{secao}")' if secao else "")]
    if not trechos:
        linhas.append("Nenhum trecho nessa secao. Tente sem filtro ou outro nome de secao.")
    for t in trechos:
        linhas.append(f"\n({t['ordem']}) {t['secao'] or 'sem secao'}\n{t['texto']}")
    return "\n".join(linhas)


def _ler_artigo(arxiv_id: str, secao: str | None, max_trechos: int) -> dict:
    params = {"secao": secao} if secao else None
    artigo = _get(f"/articles/{arxiv_id}/chunks", params)
    artigo["trechos"] = artigo["trechos"][:max(1, min(max_trechos, 20))]
    return artigo


@mcp.tool(annotations=SO_LEITURA)
def buscar_artigos(
    pergunta: str,
    tema: Literal["rag", "agents", "llm"] | None = None,
    k: int = 5,
    trechos_por_artigo: int = 2,
    rerank: bool | None = None,
) -> CallToolResult:
    """Busca por significado em artigos recentes do arXiv sobre RAG
    (retrieval-augmented generation), agentes de LLM e LLMs em geral.

    Aceita pergunta em portugues ou ingles (e traduzida para ingles antes da
    busca). Devolve os artigos mais relevantes numerados [1], [2], ..., com
    arxiv_id, link e os trechos que casaram com a pergunta.

    Args:
        pergunta: o que procurar, em linguagem natural. Termos tecnicos
            especificos funcionam melhor que perguntas vagas.
        tema: restringe a "rag", "agents" ou "llm"; sem ele, busca nos tres.
        k: quantos artigos (1 a 10).
        trechos_por_artigo: quantos trechos de cada artigo (1 a 3).
        rerank: reordenar com o cross-encoder; sem ele, vale o padrao da API.
    """
    k = max(1, min(k, 10))
    trechos_por_artigo = max(1, min(trechos_por_artigo, 3))
    params: dict = {"q": pergunta, "k": k, "trechos": trechos_por_artigo}
    if tema:
        params["topic"] = tema
    if rerank is not None:
        params["rerank"] = str(rerank).lower()
    busca = _get("/search", params)
    return _resultado(_texto_busca(busca), busca)


@mcp.tool(annotations=SO_LEITURA)
def ler_artigo(arxiv_id: str, secao: str | None = None, max_trechos: int = 8) -> CallToolResult:
    """Le os trechos de um artigo ja encontrado por buscar_artigos, na ordem do
    texto -- para aprofundar alem dos trechos que a busca devolveu.

    Args:
        arxiv_id: o id do artigo, ex.: "2609.28653" (vem no resultado da busca).
        secao: filtra pelo nome da secao, por parte do nome e sem diferenciar
            maiusculas ("result", "method", "conclusion"). Sem ele, comeca pelo
            abstract.
        max_trechos: quantos trechos devolver (1 a 20).
    """
    artigo = _ler_artigo(arxiv_id, secao, max_trechos)
    return _resultado(_texto_artigo(artigo, secao), artigo)


@mcp.tool(annotations=SO_LEITURA)
def temas_disponiveis() -> CallToolResult:
    """Os temas indexados (rag, agents, llm), quantos trechos ha em cada um e se
    a busca esta funcionando. Util para saber o que da para perguntar."""
    saude = _get("/health", aceitar=(503,))
    pontos = saude.get("pontos_por_tema", {})
    linhas = [f"status da busca: {saude['status']}"]
    linhas += [f"- {tema}: {n} trechos indexados" for tema, n in pontos.items()]
    linhas += [f"- servico {nome}: {estado}" for nome, estado in saude.get("dependencias", {}).items()
               if estado != "ok"]
    return _resultado("\n".join(linhas), saude)


@mcp.prompt()
def responder_com_citacoes(pergunta: str) -> str:
    """Pede uma resposta baseada so nos artigos encontrados, com citacoes [n]."""
    return (
        f"Pergunta: {pergunta}\n\n"
        "Use a ferramenta buscar_artigos (mais de uma vez, se precisar) e, se for "
        "util, ler_artigo para aprofundar. Responda no idioma da pergunta, usando "
        "APENAS o que as ferramentas devolveram, e cite cada afirmacao com o numero "
        "do artigo entre colchetes, como [1]. Se os artigos nao bastarem, diga isso."
    )


@mcp.resource("arxiv://artigo/{arxiv_id}", mime_type="text/plain",
              description="Metadados e abstract de um artigo indexado")
def recurso_artigo(arxiv_id: str) -> str:
    return _texto_artigo(_ler_artigo(arxiv_id, None, 1), None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor MCP arxiv-radar")
    parser.add_argument("--http", action="store_true",
                        help="streamable HTTP em vez de stdio")
    # localhost por padrao: sem autenticacao, nao deve ficar exposto na rede
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.http:
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run("stdio")
