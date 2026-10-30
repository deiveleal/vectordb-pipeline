"""Interface de teste da API de busca (Streamlit).

So conversa com a API por HTTP -- nao importa nada de src/, como qualquer
cliente faria. Roda fora do Docker, apontando para a API em container.

Os dados de busca vem de um de dois caminhos, escolhido na barra lateral:

- API direta: GET /search e /health, com todas as opcoes da API;
- MCP: as ferramentas do servidor arxiv-radar (mcp_server/server.py), que
  chamam a mesma API por dentro. As ferramentas devolvem, alem do texto para o
  modelo, o JSON da API (structured_content) -- e com ele que os cartoes sao
  montados. Expoem menos opcoes (k ate 10, trechos ate 3; traducao e score
  minimo ficam no padrao da API) e sobem um processo a cada chamada.

Opcionalmente, responde a pergunta com um LLM local (Ollama). Dois modos:

- fluxo fixo: a interface busca uma vez e entrega os trechos ao modelo, que
  responde citando [n]. Previsivel -- o que funciona bem com modelo de 3B;
- agente (MCP): o modelo recebe as ferramentas do servidor MCP arxiv-radar
  (mcp_server/server.py) e decide sozinho quando buscar, reformular ou ler um
  artigo. Mais flexivel, mais lento, e depende de o modelo usar ferramentas
  direito.

E um recurso DA INTERFACE: a API continua so de busca semantica.

Uso (da raiz do projeto):
    uv run streamlit run ui/app.py
    API_URL=http://outro-host:8000 OLLAMA_URL=http://host:11434 uv run streamlit run ui/app.py
"""

import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

API_PADRAO = os.environ.get("API_URL", "http://localhost:8000")
OLLAMA_PADRAO = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# 3B quantizado: cabe inteiro nos 4 GB da GTX 1650 e responde bem em portugues
MODELO_PADRAO = "qwen2.5:3b-instruct-q4_K_M"
# o contexto do Ollama e curto por padrao; 6 trechos de ~350 tokens + prompt
# cabem com folga em 8k
NUM_CTX = 8192

PROMPT_SISTEMA = """Você responde perguntas sobre artigos científicos do arXiv usando \
APENAS os trechos fornecidos, numerados como [1], [2], ...

Regras:
- Responda no mesmo idioma da pergunta.
- Cite as fontes de cada afirmação com o número entre colchetes, ex.: [2].
- Se os trechos não bastarem para responder, diga isso claramente -- não \
complete com conhecimento próprio.
- Seja direto: poucos parágrafos."""

# --- modo agente (MCP) ---
SERVIDOR_MCP = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"
MAX_PASSOS_AGENTE = 5
# Em ingles: e o idioma em que o modelo segue melhor instrucao. A citacao e
# pelo arxiv_id, nao por [n]: cada busca recomeca a numeracao em [1], e com
# duas buscas o [1] ficaria ambiguo.
PROMPT_AGENTE = """You answer questions about recent arXiv papers on RAG \
(retrieval-augmented generation), LLM agents and LLMs.

- Use the tool buscar_artigos to find papers; call it again with another phrasing \
or topic if the first results are weak. Use ler_artigo to read more of a paper.
- Base the answer ONLY on what the tools returned. If it is not enough, say so.
- Cite every claim with the paper's arxiv_id in brackets, e.g. [2609.28653].
- Answer in the same language as the user's question."""
# Modelo de 3B cita de jeitos variados -- "[2609.27334]", "[arxiv_id: 2609.27334]"
# ou o id solto. Qualquer id no texto vira link; o que ja esta dentro de um
# link (precedido de "[" ou "/") fica como esta.
ID_ARXIV = re.compile(r"(?<![\[/\w.])(\d{4}\.\d{4,5})(?![\d])")
ID_ENTRE_COLCHETES = re.compile(r"\[(\d{4}\.\d{4,5})\]")
QUALQUER_ID = re.compile(r"(?<![\w.])\d{4}\.\d{4,5}(?!\d)")


def linkar_ids(texto: str) -> tuple[str, list[str]]:
    """Troca ids do arXiv por links. Devolve (texto, ids citados, sem repetir).

    Duas passadas: "[id]" vira "[id](url)"; depois, id solto vira link -- o que
    a primeira passada ja fez fica de fora, por estar depois de "[" ou "/".
    """
    ids = list(dict.fromkeys(QUALQUER_ID.findall(re.sub(r"\(https?://[^)]*\)", "", texto))))
    texto = ID_ENTRE_COLCHETES.sub(r"[\1](https://arxiv.org/abs/\1)", texto)
    return ID_ARXIV.sub(r"[\1](https://arxiv.org/abs/\1)", texto), ids

TEMAS = {"todos": None, "RAG": "rag", "Agents": "agents", "LLM": "llm"}
ACESSOS = {"API direta": "api", "MCP (servidor arxiv-radar)": "mcp"}
# limites das ferramentas do servidor MCP (ver buscar_artigos)
MCP_MAX_K, MCP_MAX_TRECHOS = 10, 3
CORES_TEMA = {"rag": "blue", "agents": "green", "llm": "violet"}
EXEMPLOS = [
    "checagem de fatos com geração aumentada por recuperação",
    "memória de longo prazo para agentes",
    "reward hacking in autonomous research agents",
    "escolher automaticamente o rank de cada adaptador LoRA",
]

st.set_page_config(page_title="arXiv Radar · busca", page_icon="🔎", layout="wide")


# --- chamadas a API -----------------------------------------------------------

def chamar_api(url: str, caminho: str, params: dict | None = None,
               timeout: int = 120) -> tuple[dict | None, str | None, float]:
    """(corpo, erro, segundos). Erro legivel em vez de traceback na tela."""
    t0 = time.monotonic()
    try:
        resposta = requests.get(f"{url.rstrip('/')}{caminho}", params=params, timeout=timeout)
    except requests.RequestException as erro:
        return None, f"API inacessível em {url}: {type(erro).__name__}", time.monotonic() - t0
    segundos = time.monotonic() - t0
    try:
        corpo = resposta.json()
    except ValueError:
        return None, f"HTTP {resposta.status_code}: resposta não é JSON", segundos
    if resposta.status_code >= 400 and caminho != "/health":
        detalhe = corpo.get("detail") if isinstance(corpo, dict) else corpo
        return None, f"HTTP {resposta.status_code}: {detalhe}", segundos
    return corpo, None, segundos


# --- Ollama ---------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def modelos_ollama(url: str) -> tuple[list[str], dict[str, list[str]], str | None]:
    """(modelos de texto, capacidades por modelo, erro).

    Fica de fora o que nao gera texto: embeddings (capacidade "embedding"),
    audio (whisper aparece como "completion", mas de familia "unknown") e
    modelos ":cloud", que rodam fora da maquina.
    """
    try:
        tags = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5).json().get("models", [])
    except (requests.RequestException, ValueError) as erro:
        return [], {}, f"Ollama inacessível em {url}: {type(erro).__name__}"
    modelos, capacidades = [], {}
    for m in tags:
        nome = m["name"]
        if nome.endswith(":cloud"):
            continue
        try:
            info = requests.post(f"{url.rstrip('/')}/api/show", json={"model": nome},
                                 timeout=10).json()
        except (requests.RequestException, ValueError):
            continue
        caps = info.get("capabilities") or []
        familia = (info.get("details") or {}).get("family", "unknown")
        if "completion" in caps and "embedding" not in caps and familia != "unknown":
            modelos.append(nome)
            capacidades[nome] = caps
    return sorted(modelos), capacidades, None


def modelos_na_memoria(url: str) -> list[dict]:
    """Modelos carregados no Ollama agora (/api/ps), de qualquer cliente."""
    try:
        return requests.get(f"{url.rstrip('/')}/api/ps", timeout=5).json().get("models", [])
    except (requests.RequestException, ValueError):
        return []


def descarregar_modelo(url: str, modelo: str) -> str | None:
    """Tira o modelo da memoria (RAM/VRAM): keep_alive=0 e o jeito do Ollama."""
    try:
        requests.post(f"{url.rstrip('/')}/api/generate",
                      json={"model": modelo, "keep_alive": 0}, timeout=60).raise_for_status()
        return None
    except requests.RequestException as erro:
        return f"{type(erro).__name__}: {erro}"


def esperar_sair_da_memoria(url: str, modelos: set[str], limite_s: float = 20) -> bool:
    """Espera o Ollama tirar os modelos da memoria. True se sairam a tempo.

    O keep_alive=0 responde antes de a VRAM ser liberada. Carregar o novo
    modelo logo em seguida fazia o Ollama ver a GPU ainda cheia e mandar tudo
    para a CPU (medido: 0 de 37 camadas na GPU, com 521 MB livres naquele
    instante e 3,8 GB livres um segundo depois).
    """
    fim = time.monotonic() + limite_s
    while time.monotonic() < fim:
        if not modelos & {m["name"] for m in modelos_na_memoria(url)}:
            time.sleep(1)  # o processo do modelo sai; a VRAM volta logo depois
            return True
        time.sleep(0.5)
    return False


def carregar_modelo(url: str, modelo: str) -> str | None:
    """Carrega o modelo sem gerar nada (generate sem prompt): a primeira
    resposta depois nao paga o tempo de carga."""
    try:
        requests.post(f"{url.rstrip('/')}/api/generate",
                      json={"model": modelo}, timeout=300).raise_for_status()
        return None
    except requests.RequestException as erro:
        return f"{type(erro).__name__}: {erro}"


def ao_trocar_modelo(url: str) -> None:
    """Callback da lista de modelos: descarrega o modelo que ESTA interface
    estava usando e carrega o novo.

    So o que a interface usou sai da memoria -- nunca os modelos que outros
    clientes do mesmo Ollama carregaram. Com a GPU de 4 GB, dois modelos
    juntos empurram parte do novo para a CPU.
    """
    anterior = st.session_state.get("modelo_em_uso")
    novo = st.session_state.modelo
    avisos = []
    if anterior and anterior != novo:
        erro = descarregar_modelo(url, anterior)
        if not erro and not esperar_sair_da_memoria(url, {anterior}):
            erro = "ainda na memória depois de 20s"
        avisos.append(f"⚠️ não descarregou {anterior}: {erro}" if erro
                      else f"🧹 {anterior} descarregado")
    erro = carregar_modelo(url, novo)
    avisos.append(f"⚠️ não carregou {novo}: {erro}" if erro else f"✅ {novo} carregado")
    st.session_state.modelo_em_uso = novo
    st.session_state.avisos_modelo = avisos


def liberar_memoria(url: str) -> None:
    """Callback do botao: descarrega TODOS os modelos do Ollama e recarrega o
    selecionado.

    Acao explicita, nunca automatica: afeta outros clientes do mesmo Ollama.
    Nao e destrutivo -- quem usar um modelo descarregado so paga de novo o tempo
    de carga --, mas e a unica forma de liberar a GPU de um modelo que esta
    interface nao carregou (medido: com chat:latest de outro cliente ocupando a
    GPU, o qwen3:1.7b carregou so 7% nela).
    """
    avisos = []
    carregados = {m["name"] for m in modelos_na_memoria(url)}
    for nome in sorted(carregados):
        erro = descarregar_modelo(url, nome)
        avisos.append(f"⚠️ não descarregou {nome}: {erro}" if erro else f"🧹 {nome} descarregado")
    if carregados and not esperar_sair_da_memoria(url, carregados):
        avisos.append("⚠️ a memória não foi liberada em 20s -- o modelo pode carregar na CPU")
    modelo = st.session_state.get("modelo")
    if modelo:
        erro = carregar_modelo(url, modelo)
        avisos.append(f"⚠️ não carregou {modelo}: {erro}" if erro else f"✅ {modelo} carregado")
        st.session_state.modelo_em_uso = modelo
    st.session_state.avisos_modelo = avisos or ["Nada na memória do Ollama"]


def montar_contexto(busca: dict, max_trechos: int) -> tuple[str, list[dict]]:
    """Os melhores trechos, na ordem da busca, numerados como fontes."""
    fontes = []
    for r in busca.get("resultados", []):
        for t in r["trechos"]:
            fontes.append({"titulo": r["titulo"], "url": r["url"],
                           "secao": t["secao"], "texto": t["texto"]})
    fontes = fontes[:max_trechos]
    contexto = "\n\n".join(
        f"[{i}] {f['titulo']} — seção: {f['secao'] or 'n/d'}\n{f['texto']}"
        for i, f in enumerate(fontes, 1))
    return contexto, fontes


# A instrucao final vai NO IDIOMA da resposta: escrita em portugues, "responda
# no idioma da pergunta" fazia o modelo de 3B responder em portugues a
# pergunta em ingles. O idioma vem da deteccao que a API ja fez.
INSTRUCAO_FINAL = {
    "pt": "Responda em português e cite a fonte de cada afirmação com o número do "
          "trecho entre colchetes, como [1] ou [2].",
    "en": "Answer in English and cite the source of each statement with the passage "
          "number in brackets, like [1] or [2].",
}


def responder(url: str, modelo: str, pergunta: str, contexto: str, temperatura: float,
              pensa: bool, idioma: str | None) -> Iterator[str]:
    """Resposta do Ollama em streaming (/api/chat), pedaco a pedaco."""
    instrucao = INSTRUCAO_FINAL.get(idioma or "pt", INSTRUCAO_FINAL["pt"])
    corpo: dict[str, Any] = {
        "model": modelo,
        "stream": True,
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            # a regra das citacoes repetida no fim: modelo de 3B segue melhor a
            # ultima instrucao que a do system (sem isto, respondeu sem nenhum [n])
            {"role": "user", "content": f"Trechos:\n\n{contexto}\n\nPergunta: {pergunta}"
                                        f"\n\n{instrucao}"},
        ],
        "options": {"temperature": temperatura, "num_ctx": NUM_CTX},
    }
    if pensa:
        # modelo com "thinking" escreveria o raciocinio antes da resposta
        corpo["think"] = False
    with requests.post(f"{url.rstrip('/')}/api/chat", json=corpo, stream=True,
                       timeout=600) as resposta:
        if resposta.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {resposta.status_code}: {resposta.text[:200]}")
        for linha in resposta.iter_lines():
            if not linha:
                continue
            parte = json.loads(linha)
            if parte.get("error"):
                raise RuntimeError(parte["error"])
            yield parte.get("message", {}).get("content", "")


# --- agente via MCP --------------------------------------------------------------

async def rodar_agente(pergunta: str, modelo: str, ollama: str, api: str, temperatura: float,
                       pensa: bool, registrar: Callable[[str, dict, str, bool], None]) -> tuple[str, int]:
    """Laco do agente: o Ollama pede ferramentas, o MCP executa. (resposta, n. de chamadas).

    O servidor MCP sobe como subprocesso (stdio) a cada pergunta, apontando para
    a mesma API da interface.
    """
    params = StdioServerParameters(command=sys.executable, args=[str(SERVIDOR_MCP)],
                                   env={**os.environ, "API_URL": api})
    with open(os.devnull, "w") as silencio:
        async with stdio_client(params, errlog=silencio) as (leitura, escrita):
            async with ClientSession(leitura, escrita) as sessao:
                await sessao.initialize()
                # o esquema de ferramenta do MCP ja e JSON Schema: vai direto para o Ollama
                ferramentas = [{"type": "function",
                                "function": {"name": t.name, "description": t.description or "",
                                             "parameters": t.input_schema}}
                               for t in (await sessao.list_tools()).tools]
                mensagens: list[dict] = [{"role": "system", "content": PROMPT_AGENTE},
                                         {"role": "user", "content": pergunta}]
                chamadas_feitas = 0
                for _ in range(MAX_PASSOS_AGENTE):
                    mensagem = await asyncio.to_thread(
                        chat_ollama, ollama, modelo, mensagens, temperatura, pensa, ferramentas)
                    mensagens.append(mensagem)
                    chamadas = mensagem.get("tool_calls") or []
                    if not chamadas:
                        return mensagem.get("content", ""), chamadas_feitas
                    for chamada in chamadas:
                        nome = chamada["function"]["name"]
                        args = chamada["function"].get("arguments") or {}
                        if isinstance(args, str):  # alguns modelos mandam JSON em texto
                            try:
                                args = json.loads(args)
                            except ValueError:
                                args = {}
                        resultado = await sessao.call_tool(nome, args)
                        texto = "\n".join(c.text for c in resultado.content
                                          if isinstance(c, TextContent))
                        chamadas_feitas += 1
                        registrar(nome, args, texto, bool(resultado.is_error))
                        # erro de ferramenta volta para o modelo: ele pode corrigir
                        mensagens.append({"role": "tool", "tool_name": nome, "content": texto})
                # passou do limite de passos: responde com o que tem, sem ferramentas
                mensagens.append({"role": "user", "content":
                                  "Answer now using only what the tools returned so far."})
                final = await asyncio.to_thread(
                    chat_ollama, ollama, modelo, mensagens, temperatura, pensa, None)
                return final.get("content", ""), chamadas_feitas


def chat_ollama(url: str, modelo: str, mensagens: list[dict], temperatura: float, pensa: bool,
                ferramentas: list[dict] | None) -> dict:
    """Uma chamada /api/chat sem streaming; devolve a mensagem do assistente."""
    corpo: dict = {"model": modelo, "messages": mensagens, "stream": False,
                   "options": {"temperature": temperatura, "num_ctx": NUM_CTX}}
    if ferramentas:
        corpo["tools"] = ferramentas
    if pensa:
        corpo["think"] = False
    resposta = requests.post(f"{url.rstrip('/')}/api/chat", json=corpo, timeout=600)
    if resposta.status_code >= 400:
        raise RuntimeError(f"Ollama HTTP {resposta.status_code}: {resposta.text[:200]}")
    return resposta.json()["message"]


# --- acesso aos dados: API direta ou MCP -----------------------------------------

async def _chamar_mcp(api: str, ferramenta: str, args: dict):
    params = StdioServerParameters(command=sys.executable, args=[str(SERVIDOR_MCP)],
                                   env={**os.environ, "API_URL": api})
    with open(os.devnull, "w") as silencio:
        async with stdio_client(params, errlog=silencio) as (leitura, escrita):
            async with ClientSession(leitura, escrita) as sessao:
                await sessao.initialize()
                return await sessao.call_tool(ferramenta, args)


def chamar_mcp(api: str, ferramenta: str, args: dict) -> tuple[dict | None, str | None, float]:
    """(dados, erro, segundos) de uma ferramenta MCP -- mesmo contrato de chamar_api.

    Os dados sao o structured_content, que o servidor preenche com o JSON da
    API; o texto (content) e o que um modelo leria.
    """
    t0 = time.monotonic()
    try:
        resultado = asyncio.run(_chamar_mcp(api, ferramenta, args))
    except Exception as erro:  # noqa: BLE001 -- processo, protocolo ou rede: mostrar
        return None, f"Servidor MCP falhou: {type(erro).__name__}: {erro}", time.monotonic() - t0
    segundos = time.monotonic() - t0
    if resultado.is_error:
        texto = " ".join(c.text for c in resultado.content if isinstance(c, TextContent))
        return None, f"MCP: {texto}", segundos
    return resultado.structured_content, None, segundos


@st.cache_data(ttl=60, show_spinner=False)
def limites_api(api: str) -> dict[str, int]:
    """Maximo de k e trechos que a API aceita, lido do proprio OpenAPI dela.

    Copiar os numeros para ca deixaria a interface oferecendo o que a API
    recusa (422) quando alguem muda o limite em src/api.py.
    """
    limites = {"k": 50, "trechos": 10}
    try:
        esquema = requests.get(f"{api.rstrip('/')}/openapi.json", timeout=5).json()
        for p in esquema["paths"]["/search"]["get"]["parameters"]:
            if p["name"] in limites and "maximum" in p.get("schema", {}):
                limites[p["name"]] = int(p["schema"]["maximum"])
    except (requests.RequestException, ValueError, KeyError):
        pass
    return limites


@st.cache_data(ttl=30, show_spinner=False)
def saude_via_mcp(api: str) -> tuple[dict | None, str | None, float]:
    """Cache curto: cada chamada MCP sobe um processo, e a barra lateral redesenha
    a cada interacao."""
    return chamar_mcp(api, "temas_disponiveis", {})


def parametros_busca(q: str, rerank: bool) -> dict:
    # .get com padrao: controle que sumiu da tela (modo MCP) perde o valor guardado
    params = {"q": q, "k": st.session_state.k, "trechos": st.session_state.trechos,
              "traduzir": str(st.session_state.get("traduzir", True)).lower(),
              "rerank": str(rerank).lower()}
    if TEMAS[st.session_state.tema]:
        params["topic"] = TEMAS[st.session_state.tema]
    if st.session_state.get("usar_score_minimo"):
        params["score_minimo"] = st.session_state.score_minimo
    return params


def buscar_dados(api: str, pergunta: str, rerank: bool) -> tuple[dict | None, str | None, float]:
    """A busca pelo caminho escolhido. Os dois devolvem o mesmo JSON da API."""
    if acesso == "mcp":
        args: dict = {"pergunta": pergunta, "k": min(st.session_state.k, MCP_MAX_K),
                      "trechos_por_artigo": min(st.session_state.trechos, MCP_MAX_TRECHOS),
                      "rerank": rerank}
        if TEMAS[st.session_state.tema]:
            args["tema"] = TEMAS[st.session_state.tema]
        return chamar_mcp(api, "buscar_artigos", args)
    return chamar_api(api, "/search", parametros_busca(pergunta, rerank))


# --- barra lateral --------------------------------------------------------------

with st.sidebar:
    st.header("Dados de busca")
    acesso = ACESSOS[st.radio(
        "Acesso aos dados", list(ACESSOS), key="acesso_rotulo",
        help="API direta: chama /search. MCP: usa as ferramentas do servidor "
             "arxiv-radar, que chamam a mesma API por dentro.")]
    api_url = st.text_input("Endereço da API", value=API_PADRAO,
                            help="No modo MCP, e a API que o servidor MCP usa.")

    if acesso == "mcp":
        saude, erro_saude, _ = saude_via_mcp(api_url)
    else:
        saude, erro_saude, _ = chamar_api(api_url, "/health", timeout=15)
    if erro_saude:
        st.error(erro_saude)
    elif saude:
        icone = {"ok": "🟢", "degradado": "🟡", "indisponivel": "🔴"}.get(saude["status"], "⚪")
        st.markdown(f"**Status:** {icone} {saude['status']}")
        for servico, estado in saude.get("dependencias", {}).items():
            st.caption(f"{'✅' if estado == 'ok' else '❌'} {servico}: {estado}")
        pontos = saude.get("pontos_por_tema", {})
        if pontos:
            st.caption("Trechos indexados: " + " · ".join(f"{t} {n:,}".replace(",", ".")
                                                          for t, n in pontos.items()))
    if st.button("Atualizar status", use_container_width=True):
        saude_via_mcp.clear()
        st.rerun()

    st.header("Busca")
    st.selectbox("Tema", list(TEMAS), key="tema")
    limites = limites_api(api_url)
    max_k, max_trechos = limites["k"], limites["trechos"]
    if acesso == "mcp":
        max_k, max_trechos = min(max_k, MCP_MAX_K), min(max_trechos, MCP_MAX_TRECHOS)
    # O valor mora so no session_state (o slider nao recebe `value`): passar os
    # dois gera aviso do Streamlit. E trocar para MCP com k acima do teto daria
    # "valor fora do intervalo" -- entao o valor e ajustado antes do slider.
    for chave, padrao, teto in (("k", 10, max_k), ("trechos", 3, max_trechos)):
        st.session_state[chave] = min(st.session_state.get(chave, padrao), teto)
    st.slider("Artigos (k)", 1, max_k, key="k")
    st.slider("Trechos por artigo", 1, max_trechos, key="trechos")
    st.toggle("Reranking (cross-encoder)", value=True, key="rerank",
              help="Reordena os candidatos do Qdrant lendo pergunta e trecho juntos. "
                   "Mais preciso em tese, mais lento.")
    if acesso == "mcp":
        st.caption("No MCP, tradução e score mínimo seguem o padrão da API "
                   "(traduz sempre, sem corte): as ferramentas não expõem essas opções.")
    else:
        st.toggle("Traduzir pergunta para inglês", value=True, key="traduzir")
        st.toggle("Usar score mínimo", value=False, key="usar_score_minimo",
                  help="Na avaliação, nenhum corte separou bem respostas certas de "
                       "perguntas sem resposta -- use para explorar, não como filtro.")
        if st.session_state.usar_score_minimo:
            st.slider("Score mínimo (cosseno)", -1.0, 1.0, 0.75, 0.01, key="score_minimo")
    st.divider()
    st.toggle("Comparar com e sem reranking", value=False, key="comparar",
              help="Faz a mesma busca duas vezes e mostra lado a lado.")

    st.header("Resposta com LLM")
    st.toggle("Gerar resposta (Ollama)", value=True, key="usar_llm",
              help="Responde a partir dos trechos encontrados, citando as fontes. "
                   "Não vale no modo de comparação.")
    if st.session_state.usar_llm:
        ollama_url = st.text_input("Endereço do Ollama", value=OLLAMA_PADRAO)
        modelos, capacidades, erro_ollama = modelos_ollama(ollama_url)
        if erro_ollama:
            st.error(erro_ollama)
        elif not modelos:
            st.warning("Nenhum modelo de texto no Ollama. Ex.: `ollama pull qwen2.5:3b-instruct`")
        else:
            st.selectbox("Modelo", modelos, key="modelo",
                         index=modelos.index(MODELO_PADRAO) if MODELO_PADRAO in modelos else 0,
                         on_change=ao_trocar_modelo, args=(ollama_url,),
                         help="Ao trocar, o modelo que a interface estava usando sai da "
                              "memória e o novo é carregado.")
            for aviso in st.session_state.pop("avisos_modelo", []):
                st.toast(aviso)
            carregados = modelos_na_memoria(ollama_url)
            if carregados:
                em_uso = st.session_state.get("modelo_em_uso")
                st.caption("Na memória do Ollama: " + " · ".join(
                    f"{'**' if m['name'] == em_uso else ''}{m['name']}"
                    f"{'**' if m['name'] == em_uso else ''} "
                    f"({100 * m.get('size_vram', 0) / max(m['size'], 1):.0f}% GPU)"
                    for m in carregados))
                if any(m["name"] != st.session_state.get("modelo_em_uso") for m in carregados):
                    st.button("Liberar a memória do Ollama", use_container_width=True,
                              on_click=liberar_memoria, args=(ollama_url,),
                              help="Descarrega TODOS os modelos -- inclusive os de outros "
                                   "programas que usam este Ollama -- e recarrega o "
                                   "selecionado, para ele ficar inteiro na GPU. Quem usar "
                                   "um modelo descarregado só espera ele carregar de novo.")
            st.toggle("Modo agente (ferramentas via MCP)", value=False, key="modo_agente",
                      disabled=acesso != "mcp",
                      help="O modelo decide quando buscar, reformular ou ler um artigo, "
                           "pelas ferramentas do servidor MCP arxiv-radar. Mais lento, e "
                           "modelo pequeno erra mais ao usar ferramentas. Só com o "
                           "acesso aos dados via MCP.")
            if not (st.session_state.modo_agente and acesso == "mcp"):
                st.slider("Trechos no contexto", 1, 15, 6, key="max_trechos",
                          help="Os melhores trechos da busca, na ordem do resultado.")
            st.slider("Temperatura", 0.0, 1.0, 0.2, 0.05, key="temperatura")
    else:
        ollama_url, capacidades = OLLAMA_PADRAO, {}


# --- resultados -------------------------------------------------------------------

def mostrar_cabecalho(busca: dict, segundos: float) -> None:
    partes = []
    if busca.get("traduzida"):
        partes.append(f"traduzida de `{busca.get('idioma')}` → **{busca['pergunta_busca']}**")
    elif busca.get("idioma") == "en":
        partes.append("em inglês, sem tradução")
    elif busca.get("idioma") is None and st.session_state.get("traduzir", True):
        partes.append("⚠️ tradutor indisponível: buscada como veio")
    partes.append("reranqueada" if busca.get("reranqueada") else "ordem por vetor")
    partes.append(f"{busca.get('total', 0)} artigos em {segundos:.1f}s "
                  f"via {'MCP' if acesso == 'mcp' else 'API'}")
    st.caption(" · ".join(partes))


def mostrar_artigo(pos: int, r: dict, reranqueada: bool, compacto: bool = False) -> None:
    cor = CORES_TEMA.get(r["tema"], "gray")
    score = f"score **{r['score']:.3f}**"
    if reranqueada and r.get("score_vetor") is not None:
        score += f" · vetor {r['score_vetor']:.3f}"
    with st.container(border=True):
        st.markdown(f"**{pos}. [{r['titulo']}]({r['url']})**")
        st.markdown(f":{cor}[{r['tema']}] · v1 {r['data_v1']} · {score} · "
                    f"[arXiv]({r['url']}) · [PDF]({r['pdf_url']})")
        if compacto:
            return
        with st.expander(f"Trechos ({len(r['trechos'])})", expanded=pos == 1):
            for t in r["trechos"]:
                st.markdown(f"**{t['secao'] or '—'}** · score {t['score']:.3f}")
                st.write(t["texto"])


def mostrar_resposta_llm(pergunta: str, busca: dict) -> None:
    """Resposta do LLM em streaming, com as fontes numeradas logo abaixo."""
    modelo = st.session_state.get("modelo")
    if not modelo or not busca.get("resultados"):
        return
    st.session_state.modelo_em_uso = modelo
    contexto, fontes = montar_contexto(busca, st.session_state.max_trechos)
    with st.container(border=True):
        st.markdown(f"**Resposta** · `{modelo}` · {len(fontes)} trechos no contexto")
        t0 = time.monotonic()
        try:
            st.write_stream(responder(ollama_url, modelo, pergunta, contexto,
                                      st.session_state.temperatura,
                                      "thinking" in capacidades.get(modelo, []),
                                      busca.get("idioma")))
        except (requests.RequestException, RuntimeError, ValueError) as erro:
            st.error(f"Falha ao gerar a resposta: {erro}")
            return
        st.caption(f"gerada em {time.monotonic() - t0:.1f}s · fontes: " + " · ".join(
            f"[{i}] [{f['titulo'][:60]}]({f['url']})" for i, f in enumerate(fontes, 1)))


def mostrar_agente(pergunta: str) -> None:
    """Roda o agente e mostra cada chamada de ferramenta e a resposta final."""
    modelo = st.session_state.get("modelo")
    if not modelo:
        return
    st.session_state.modelo_em_uso = modelo
    t0 = time.monotonic()
    with st.status(f"Agente `{modelo}` consultando o servidor MCP...", expanded=True) as status:
        def registrar(nome: str, args: dict, texto: str, erro: bool) -> None:
            argumentos = ", ".join(f"{k}={v!r}" for k, v in args.items())
            st.markdown(f"{'⚠️' if erro else '🔧'} `{nome}({argumentos})`")
            st.code(texto[:600] + (" [...]" if len(texto) > 600 else ""), language=None)
        try:
            resposta, n_chamadas = asyncio.run(rodar_agente(
                pergunta, modelo, ollama_url, api_url, st.session_state.temperatura,
                "thinking" in capacidades.get(modelo, []), registrar))
        except Exception as erro:  # noqa: BLE001 -- MCP, Ollama e rede: mostrar, nao quebrar
            status.update(label="Agente falhou", state="error")
            st.error(f"Falha no agente: {type(erro).__name__}: {erro}")
            return
        status.update(label=f"Agente: {n_chamadas} chamada(s) de ferramenta em "
                            f"{time.monotonic() - t0:.1f}s", state="complete", expanded=False)
    with st.container(border=True):
        st.markdown(f"**Resposta (agente)** · `{modelo}`")
        if n_chamadas == 0:
            st.warning("O modelo respondeu sem consultar a busca: a resposta pode vir do "
                       "conhecimento dele, e não dos artigos.")
        texto, citados = linkar_ids(resposta)
        st.markdown(texto)
        if citados:
            st.caption(f"{len(citados)} artigo(s) citado(s) pelo arxiv_id")
        elif n_chamadas:
            st.caption("⚠️ a resposta não citou nenhum artigo pelo arxiv_id -- confira as "
                       "fontes nas chamadas de ferramenta acima")


def mostrar_busca(busca: dict, segundos: float, compacto: bool = False) -> None:
    mostrar_cabecalho(busca, segundos)
    if not busca.get("resultados"):
        st.info("Nenhum artigo encontrado.")
    for i, r in enumerate(busca.get("resultados", []), 1):
        mostrar_artigo(i, r, busca.get("reranqueada", False), compacto)


st.title("🔎 arXiv Radar · busca semântica")
st.caption("Artigos de RAG, Agents e LLM do arXiv. Pergunte em português ou inglês. "
           "A resposta do LLM usa só os trechos encontrados.")

with st.form("busca"):
    pergunta = st.text_input("Pergunta", placeholder=EXEMPLOS[0], label_visibility="collapsed")
    buscar = st.form_submit_button("Buscar", type="primary")
st.caption("Exemplos: " + " · ".join(f"_{e}_" for e in EXEMPLOS))

if buscar:
    if not pergunta.strip():
        st.warning("Escreva uma pergunta.")
    elif st.session_state.comparar:
        col_rerank, col_vetor = st.columns(2)
        for coluna, rerank, titulo in ((col_rerank, True, "Com reranking"),
                                       (col_vetor, False, "Sem reranking (vetor)")):
            with coluna:
                st.subheader(titulo)
                with st.spinner("Buscando..."):
                    busca, erro, segundos = buscar_dados(api_url, pergunta, rerank)
                if erro:
                    st.error(erro)
                elif busca:
                    mostrar_busca(busca, segundos, compacto=True)
    elif (st.session_state.usar_llm and st.session_state.get("modo_agente")
          and acesso == "mcp"):
        # o agente faz as proprias buscas pelo MCP; as chamadas aparecem no status
        mostrar_agente(pergunta)
    else:
        with st.spinner("Buscando..."):
            busca, erro, segundos = buscar_dados(api_url, pergunta, st.session_state.rerank)
        if erro:
            st.error(erro)
        elif busca:
            if st.session_state.usar_llm:
                mostrar_resposta_llm(pergunta, busca)
            mostrar_busca(busca, segundos)
            with st.expander("Dados completos da busca (JSON, via "
                             f"{'MCP structured_content' if acesso == 'mcp' else 'API'})"):
                st.json(busca)
