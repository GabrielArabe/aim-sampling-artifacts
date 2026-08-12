"""
Tabela descritiva por jogador, para um demo.

    python src/rank.py <demo.dem>

O que este script É: estatística descritiva das janelas de mira de um
demo, quebrada por jogador, com o tamanho de amostra e a incerteza SEMPRE
visíveis ao lado de cada número.

O que este script NÃO é, e não deve virar: um placar. Não há pontuação,
não há limiar, não há classificação, e a ordenação é alfabética justamente
para não sugerir uma. Ordenar dez pessoas por uma feature cria um "primeiro
lugar" que a estatística aqui não sustenta -- com uma dezena de janelas por
jogador, quem aparece no topo de qualquer coluna é largamente sorteio
amostral. Ver `--sort n` se o interesse for tamanho de amostra.

TUDO que este módulo emite é MEDIÇÃO, não saída de modelo: quantis
empíricos e intervalos por estatística de ordem, sem premissa distribucional
e sem extrapolação. A distinção precisa ser mantida porque contaminação a
taxa ε corrompe os quantis acima de 1−ε — a ~2% de prevalência em
matchmaking, nada acima do percentil 98 de uma amostra não verificada é
mensurável, porque os cheaters são a própria cauda. Estimar FPR nessa região
exige modelo de cauda (GPD/valores extremos), e o resultado passa a ser
dominado pelo parâmetro de forma, não pelo dado. Se algum dia entrar
extrapolação aqui, ela tem que sair rotulada como modelo. Ver README,
seção "Medição vs. saída de modelo".

Por que a incerteza é obrigatória na saída: 12 janelas e 26 janelas não
são comparáveis, e a diferença entre duas medianas some dentro do
intervalo de confiança muito antes de significar alguma coisa. O IC é
não-paramétrico (estatística de ordem, cobertura binomial exata) -- não
assume normalidade, que seria falsa aqui, e não desmonta com n pequeno.
Quando a amostra não alcança um dos limites, a saída mostra "--", que é a
informação honesta: o dado não chega lá.
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest import process_demo  # noqa: E402

# (coluna, rótulo, casas decimais)
FEATURES = [
    ("snap_fraction", "snap_fraction", 3),
    ("path_ratio", "path_ratio", 3),
    ("settle_ms", "settle_ms", 1),
    ("peak_speed_dps", "peak_speed_dps", 1),
]

CONF = 0.95


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) para X ~ Binomial(n, p). Exato; n aqui é da ordem de dezenas."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(sum(comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k + 1)))


def quantile_ci(
    x: np.ndarray, q: float, conf: float = CONF
) -> tuple[float, float | None, float | None, int]:
    """
    Estimativa e IC não-paramétrico do quantil q.

    Devolve (estimativa, limite_inferior, limite_superior, n).
    Um limite é None quando a amostra é pequena demais para alcançá-lo --
    caso comum no p90 com poucas janelas, e exatamente o que precisa ficar
    visível em vez de ser preenchido com um número inventado.

    Método: seja K = #{amostras <= quantil verdadeiro} ~ Binomial(n, q).
    Então x_(l) <= quantil <= x_(u) com probabilidade cdf(u-1) - cdf(l-1).
    Escolhe-se o l mais alto e o u mais baixo que ainda dão cobertura
    >= conf. Não há premissa sobre a distribuição -- só que as amostras
    são trocáveis.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(x[np.isfinite(x)])
    n = len(x)
    if n == 0:
        return float("nan"), None, None, 0

    est = float(np.quantile(x, q))
    alpha = 1.0 - conf

    lo_i = None
    for l in range(1, n + 1):
        if _binom_cdf(l - 1, n, q) <= alpha / 2:
            lo_i = l
        else:
            break

    hi_i = None
    for u in range(1, n + 1):
        if _binom_cdf(u - 1, n, q) >= 1.0 - alpha / 2:
            hi_i = u
            break

    lo = float(x[lo_i - 1]) if lo_i is not None else None
    hi = float(x[hi_i - 1]) if hi_i is not None else None
    return est, lo, hi, n


def min_n_for_ci(q: float, conf: float = CONF) -> int:
    """
    Menor n em que o limite SUPERIOR do IC do quantil q existe dentro da amostra.

    O limite superior só existe se cdf(n-1) >= 1-alpha/2, e cdf(n-1) = 1-q^n.
    Logo q^n <= alpha/2. Para p90 a 95%: n >= 36 (0.9^35 = 0.02503, ainda
    acima de 0.025 -- por pouco, e por isso calculado e não estimado de
    cabeça). Para mediana: n >= 6.

    Existe para poder ser dito na saída. Uma coluna inteira de "--" no p90
    parece defeito de formatação; é o contrário -- é a amostra informando
    que não chega lá, e o número de janelas por jogador num único demo é
    uma ordem de grandeza menor que 35.
    """
    alpha = 1.0 - conf
    n = 1
    while q**n > alpha / 2:
        n += 1
        if n > 10_000:
            raise ValueError("n requerido é grande demais")
    return n


def _fmt(v: float | None, dec: int) -> str:
    if v is None:
        return "--"
    if not np.isfinite(v):
        return "n/d"
    return f"{v:.{dec}f}"


def build_table(res, sort: str = "name") -> pl.DataFrame:
    """
    Monta a tabela por jogador.

    O elenco vem de res.roster, NÃO das janelas. Um jogador cujas janelas
    foram todas descartadas (poucas kills, movimento de mira abaixo do
    mínimo) precisa aparecer com n=0 em vez de sumir da tabela: a ausência
    também é um resultado, e uma tabela que só mostra quem sobreviveu aos
    filtros mente sobre a cobertura da amostra.
    """
    w = res.windows
    rows = []
    for r in res.roster.iter_rows(named=True):
        sid = r["steamid"]
        sub = w.filter(pl.col("steamid") == sid) if w.height else w
        row = {"player": r["name"], "steamid": str(sid), "n": sub.height}
        for col, _label, _dec in FEATURES:
            vals = sub[col].to_numpy() if sub.height else np.array([])
            for q, tag in ((0.5, "med"), (0.9, "p90")):
                est, lo, hi, _ = quantile_ci(vals, q)
                row[f"{col}__{tag}"] = est
                row[f"{col}__{tag}_lo"] = lo
                row[f"{col}__{tag}_hi"] = hi
        rows.append(row)

    df = pl.DataFrame(rows)
    if sort == "n":
        df = df.sort(["n", "player"], descending=[True, False])
    else:
        df = df.sort("player")
    return df


def render(res, df: pl.DataFrame, *, conf: float = CONF) -> str:
    out: list[str] = []
    d = res.diagnostics
    out.append(f"demo            : {res.windows['demo'][0] if res.windows.height else '(sem janelas)'}")
    out.append(f"tickrate medido : {res.tickrate:.4f} Hz   (dt = {res.dt:.9f} s)")
    out.append(
        f"janelas         : {d['windows_kept']} de {d['kills_total']} kills"
        f"   |  jogadores: {df.height}"
    )
    descartes = [f"{k.removeprefix('dropped_')}={v}" for k, v in d.items()
                 if k.startswith("dropped_") and v]
    out.append(f"descartes       : {', '.join(descartes) if descartes else 'nenhum'}")
    out.append("")
    out.append(
        f"Estatística descritiva de UM demo. IC {conf:.0%} não-paramétrico por"
        " estatística de ordem;"
    )
    out.append(
        '"--" = a amostra não alcança esse limite. Não existe baseline medida'
        " para comparar"
    )
    out.append(
        "estes valores, e com n desta ordem a diferença entre dois jogadores fica"
        " dentro do IC."
    )
    out.append(
        f"O limite superior do IC exige n >= {min_n_for_ci(0.5)} na mediana e"
        f" n >= {min_n_for_ci(0.9)} no p90;"
    )
    out.append(
        "abaixo disso a amostra não alcança e a coluna mostra \"--\" em vez de"
        " um número."
    )
    out.append("")

    name_w = max(8, min(24, max(len(str(p)) for p in df["player"].to_list())))
    for col, label, dec in FEATURES:
        out.append(f"--- {label} ---")
        out.append(
            f"{'jogador':<{name_w}}  {'n':>3}  "
            f"{'mediana':>9} {'[IC95]':>19}  {'p90':>9} {'[IC95]':>19}"
        )
        for r in df.iter_rows(named=True):
            med = _fmt(r[f"{col}__med"], dec)
            med_ci = f"[{_fmt(r[f'{col}__med_lo'], dec)}, {_fmt(r[f'{col}__med_hi'], dec)}]"
            p90 = _fmt(r[f"{col}__p90"], dec)
            p90_ci = f"[{_fmt(r[f'{col}__p90_lo'], dec)}, {_fmt(r[f'{col}__p90_hi'], dec)}]"
            out.append(
                f"{str(r['player'])[:name_w]:<{name_w}}  {r['n']:>3}  "
                f"{med:>9} {med_ci:>19}  {p90:>9} {p90_ci:>19}"
            )
        out.append("")

    out.append(
        "Leitura: um demo é n=1. A taxa de erro destas features nunca foi medida"
        " contra"
    )
    out.append(
        "amostra rotulada (README, itens 5 e 7), então nenhuma linha desta tabela"
        " diz algo"
    )
    out.append("sobre a pessoa que jogou -- só sobre 750 ms de trajetória de mira.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tabela descritiva por jogador das janelas de mira de um demo."
    )
    ap.add_argument("demo", type=Path)
    ap.add_argument(
        "--sort",
        choices=["name", "n"],
        default="name",
        help="ordem das linhas: nome (padrão) ou tamanho de amostra. "
             "Não há ordenação por feature: ver docstring do módulo.",
    )
    ap.add_argument("--tickrate", type=float, default=None,
                    help="força o tickrate em vez de medir de game_time")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="salva a tabela em .csv além de imprimir")
    args = ap.parse_args()

    res = process_demo(args.demo, tickrate=args.tickrate)
    df = build_table(res, sort=args.sort)
    print(render(res, df))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(args.out)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
