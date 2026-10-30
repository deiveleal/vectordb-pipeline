"""Avaliacao da busca: known-item search sobre eval/consultas.json.

Cada consulta tem UM artigo esperado; a metrica e em que posicao ele aparece
na busca (resultado por artigo, k=10):

    acerto@1, @5, @10  -- fracao das consultas com o esperado ate a posicao N
    MRR@10             -- media de 1/posicao (0 quando fica fora do top-10)

Variacoes, para isolar o efeito de cada peca:

    en            pergunta em ingles, nas tres collections
    en_com_tema   pergunta em ingles, so na collection do tema certo
    pt            pergunta em portugues, com traducao (o caminho real)
    pt_sem_trad   pergunta em portugues, sem traducao (o que era antes)
    en_rerank     ingles, reordenado pelo cross-encoder
    pt_rerank     portugues com traducao, reordenado pelo cross-encoder

Mede tambem o tempo medio por consulta de cada variacao: o reranking so
compensa se o ganho de qualidade pagar o custo em CPU.

As consultas negativas (assunto fora do indice) mostram o score do melhor
resultado quando NAO ha resposta certa -- e o que um score_minimo teria de
cortar sem derrubar as consultas boas.

Uso (da raiz do projeto): python -m src.avaliacao [--consultas eval/consultas.json]
Grava o relatorio em eval/resultados/<data>.json.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from src.retrieve import Retriever
from src.utils.load_logger import get_logger, setup_logging
from src.utils.load_time import agora

logger = get_logger(__name__)

RAIZ = Path(__file__).resolve().parents[1]
K = 10


def posicao(ids: list[str], esperado: str) -> int | None:
    return ids.index(esperado) + 1 if esperado in ids else None


def metricas(posicoes: list[int | None]) -> dict[str, float]:
    n = len(posicoes)
    return {
        "acerto@1": sum(p == 1 for p in posicoes if p) / n,
        "acerto@5": sum(p <= 5 for p in posicoes if p) / n,
        "acerto@10": sum(p <= 10 for p in posicoes if p) / n,
        "mrr@10": sum(1 / p for p in posicoes if p) / n,
    }


def avaliar(caminho: Path) -> dict:
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    consultas, negativas = dados["consultas"], dados["negativas"]
    retriever = Retriever()

    variacoes = {
        "en": lambda c: retriever.buscar(c["en"], k=K, trechos_por_artigo=1),
        "en_com_tema": lambda c: retriever.buscar(c["en"], tema=c["tema"], k=K, trechos_por_artigo=1),
        "pt": lambda c: retriever.buscar(c["pt"], k=K, trechos_por_artigo=1),
        "pt_sem_trad": lambda c: retriever.buscar(c["pt"], k=K, trechos_por_artigo=1, traduzir=False),
        "en_rerank": lambda c: retriever.buscar(c["en"], k=K, trechos_por_artigo=1, rerank=True),
        "pt_rerank": lambda c: retriever.buscar(c["pt"], k=K, trechos_por_artigo=1, rerank=True),
    }
    relatorio: dict = {"consultas": str(caminho.relative_to(RAIZ)), "k": K, "variacoes": {}}
    scores_top1_certo: dict[str, list[float]] = {}

    for nome, buscar in variacoes.items():
        detalhes, posicoes, certos, tempos = [], [], [], []
        for c in consultas:
            t0 = time.monotonic()
            busca = buscar(c)
            tempos.append(time.monotonic() - t0)
            ids = [r.arxiv_id for r in busca.resultados]
            p = posicao(ids, c["esperado"])
            posicoes.append(p)
            if p == 1:
                certos.append(busca.resultados[0].score)
            detalhes.append({
                "esperado": c["esperado"], "tema": c["tema"], "posicao": p,
                "pergunta_busca": busca.pergunta_busca,
                "top1": busca.resultados[0].titulo if busca.resultados else None,
                "score_top1": busca.resultados[0].score if busca.resultados else None,
            })
        relatorio["variacoes"][nome] = {
            "metricas": {**metricas(posicoes), "ms_por_consulta": 1000 * statistics.mean(tempos)},
            "detalhes": detalhes,
        }
        scores_top1_certo[nome] = certos

    negs: dict[str, list[dict]] = {}
    for idioma in ("en", "pt"):
        negs[idioma] = []
        for n in negativas:
            busca = retriever.buscar(n[idioma], k=1, trechos_por_artigo=1)
            r = busca.resultados[0]
            negs[idioma].append({"pergunta": n[idioma], "pergunta_busca": busca.pergunta_busca,
                                 "score_top1": r.score, "top1": r.titulo})
    relatorio["negativas"] = negs
    relatorio["scores_top1_certo"] = scores_top1_certo
    return relatorio


def imprimir(relatorio: dict) -> None:
    print(f"\n{'variacao':14} {'acerto@1':>9} {'acerto@5':>9} {'acerto@10':>10} {'MRR@10':>8} {'ms/consulta':>12}")
    for nome, v in relatorio["variacoes"].items():
        m = v["metricas"]
        print(f"{nome:14} {m['acerto@1']:9.0%} {m['acerto@5']:9.0%} {m['acerto@10']:10.0%} "
              f"{m['mrr@10']:8.3f} {m['ms_por_consulta']:12.0f}")

    for nome in ("pt", "en", "pt_rerank"):
        erros = [d for d in relatorio["variacoes"][nome]["detalhes"] if d["posicao"] != 1]
        print(f"\nfora do 1o lugar em '{nome}' ({len(erros)}):")
        for d in erros:
            print(f"  pos {str(d['posicao']):>4} | {d['esperado']} ({d['tema']}) | busca: {d['pergunta_busca'][:70]!r}")
            print(f"           | veio em 1o: {str(d['top1'])[:80]}")

    print("\nscore do 1o resultado:")
    for nome in ("en", "pt"):
        s = relatorio["scores_top1_certo"][nome]
        print(f"  consultas certas ({nome}): min {min(s):.3f} | mediana {statistics.median(s):.3f} | max {max(s):.3f}")
    for idioma, lista in relatorio["negativas"].items():
        s = [n["score_top1"] for n in lista]
        print(f"  negativas ({idioma}):       min {min(s):.3f} | mediana {statistics.median(s):.3f} | max {max(s):.3f}")
    for n in relatorio["negativas"]["pt"]:
        print(f"    [{n['score_top1']:.3f}] {n['pergunta_busca'][:45]!r} -> {n['top1'][:60]}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Avaliacao da busca (known-item)")
    parser.add_argument("--consultas", default=str(RAIZ / "eval" / "consultas.json"))
    args = parser.parse_args()

    relatorio = avaliar(Path(args.consultas))
    relatorio["executado_em"] = agora().isoformat(timespec="seconds")
    imprimir(relatorio)

    saida = RAIZ / "eval" / "resultados" / f"{agora().strftime('%Y-%m-%d_%H%M')}.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrelatorio: {saida.relative_to(RAIZ)}")
