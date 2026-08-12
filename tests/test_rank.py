"""
Testes de rank.py.

Cobrem três coisas: o IC não-paramétrico estar correto (inclusive a
cobertura, por Monte Carlo), a tabela nunca perder um jogador, e a saída
não conter vocabulário de veredito. O último é um teste de conteúdo, não
de código -- está aqui porque o escopo declarado do repositório é medir
baseline, e uma palavra como "anômalo" na saída transforma estatística
descritiva em acusação sem que ninguém tenha decidido isso.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest import DemoResult  # noqa: E402
from rank import (  # noqa: E402
    FEATURES,
    _binom_cdf,
    build_table,
    min_n_for_ci,
    quantile_ci,
    render,
)


# --------------------------------------------------------------------------
# intervalo de confiança
# --------------------------------------------------------------------------

def test_binom_cdf_matches_hand_computation():
    assert _binom_cdf(-1, 10, 0.5) == 0.0
    assert _binom_cdf(10, 10, 0.5) == 1.0
    assert _binom_cdf(0, 10, 0.5) == pytest.approx(1 / 1024)
    assert _binom_cdf(1, 10, 0.5) == pytest.approx(11 / 1024)


def test_median_ci_matches_known_order_statistics():
    """
    n=10, mediana, 95%: o IC clássico é [x_(2), x_(9)], cobertura 97.9%.
    Valor tabelado em qualquer texto de estatística não-paramétrica.
    """
    x = np.arange(1.0, 11.0)  # x_(k) = k
    est, lo, hi, n = quantile_ci(x, 0.5, conf=0.95)
    assert n == 10
    assert lo == 2.0
    assert hi == 9.0
    assert est == pytest.approx(5.5)


def test_ci_brackets_the_estimate():
    rng = np.random.default_rng(7)
    x = rng.lognormal(0, 1, 40)
    est, lo, hi, _ = quantile_ci(x, 0.5)
    assert lo is not None and hi is not None
    assert lo <= est <= hi


def test_median_ci_coverage_is_at_least_nominal():
    """
    Monte Carlo: o IC tem que cobrir a mediana verdadeira >= 95% das vezes.

    Distribuição assimétrica de propósito -- é o caso em que um IC
    normal-aproximado erra, e as features aqui (path_ratio, peak_speed) são
    todas de cauda pesada.
    """
    rng = np.random.default_rng(11)
    true_median = 1.0  # mediana de lognormal(0, 1) = e^0
    hits = 0
    trials = 2000
    for _ in range(trials):
        x = rng.lognormal(0.0, 1.0, 15)
        _, lo, hi, _ = quantile_ci(x, 0.5, conf=0.95)
        if lo is not None and hi is not None and lo <= true_median <= hi:
            hits += 1
    assert hits / trials >= 0.95


def test_ci_widens_as_sample_shrinks():
    rng = np.random.default_rng(3)
    pool = rng.normal(100, 20, 400)
    _, lo_big, hi_big, _ = quantile_ci(pool, 0.5)
    _, lo_small, hi_small, _ = quantile_ci(pool[:20], 0.5)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_p90_upper_bound_unreachable_with_small_n():
    """
    Com 12 janelas não existe limite superior de 95% para o p90.

    Isto não é falha: é a amostra dizendo que não alcança. O valor certo a
    mostrar é "--", não um número que passe a impressão de precisão.
    """
    x = np.arange(1.0, 13.0)
    _, lo, hi, n = quantile_ci(x, 0.9, conf=0.95)
    assert n == 12
    assert lo is not None
    assert hi is None


def test_median_ci_unreachable_with_tiny_n():
    _, lo, hi, _ = quantile_ci(np.array([1.0, 2.0, 3.0]), 0.5)
    assert lo is None and hi is None


def test_min_n_for_ci():
    assert min_n_for_ci(0.5, 0.95) == 6
    # 0.9^35 = 0.02503 > 0.025, então 35 NÃO basta. Passa raspando em 36.
    assert min_n_for_ci(0.9, 0.95) == 36
    # e o limiar tem que casar com o comportamento real de quantile_ci
    n = min_n_for_ci(0.9, 0.95)
    _, _, hi_ok, _ = quantile_ci(np.arange(float(n)), 0.9)
    _, _, hi_no, _ = quantile_ci(np.arange(float(n - 1)), 0.9)
    assert hi_ok is not None and hi_no is None


def test_quantile_ci_on_empty_sample():
    est, lo, hi, n = quantile_ci(np.array([]), 0.5)
    assert n == 0 and lo is None and hi is None and np.isnan(est)


def test_quantile_ci_ignores_non_finite():
    x = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
    _, _, _, n = quantile_ci(x, 0.5)
    assert n == 3


# --------------------------------------------------------------------------
# tabela
# --------------------------------------------------------------------------

def _result(n_per_player: dict[int, int], names: dict[int, str] | None = None):
    names = names or {sid: f"p{sid}" for sid in n_per_player}
    rng = np.random.default_rng(1)
    rows = []
    for sid, n in n_per_player.items():
        for _ in range(n):
            rows.append({
                "demo": "d.dem", "steamid": sid, "player": names[sid],
                "victim": "v", "weapon": "ak47", "headshot": True, "tick": 1,
                "snap_fraction": float(rng.uniform(0.05, 0.4)),
                "path_ratio": float(rng.uniform(1.0, 3.0)),
                "settle_ms": float(rng.uniform(0, 700)),
                "peak_speed_dps": float(rng.uniform(30, 900)),
            })
    schema = {"demo": pl.String, "steamid": pl.UInt64, "player": pl.String,
              "victim": pl.String, "weapon": pl.String, "headshot": pl.Boolean,
              "tick": pl.Int64, "snap_fraction": pl.Float64,
              "path_ratio": pl.Float64, "settle_ms": pl.Float64,
              "peak_speed_dps": pl.Float64}
    windows = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    roster = pl.DataFrame(
        {"steamid": list(names), "name": [names[s] for s in names]},
        schema={"steamid": pl.UInt64, "name": pl.String},
    )
    return DemoResult(
        windows=windows, roster=roster, tickrate=64.0, dt=1 / 64.0,
        n_kills=sum(n_per_player.values()),
        diagnostics={"kills_total": sum(n_per_player.values()),
                     "windows_kept": windows.height,
                     "dropped_below_min_net_deg": 0},
    )


def test_table_has_all_ten_players():
    res = _result({i: 10 + i for i in range(1, 11)})
    assert build_table(res).height == 10


def test_player_with_zero_windows_still_appears():
    """
    Nunca filtre. Um jogador sem janela nenhuma sai com n=0 e "n/d" nas
    estatísticas -- some da tabela, não.
    """
    res = _result({1: 12, 2: 0, 3: 8})
    df = build_table(res)
    assert df.height == 3
    zero = df.filter(pl.col("n") == 0)
    assert zero.height == 1
    assert np.isnan(zero["snap_fraction__med"][0])
    assert zero["snap_fraction__med_lo"][0] is None


def test_default_sort_is_alphabetical_not_by_feature():
    """
    A ordenação padrão não pode depender de nenhuma feature.

    Ordenar dez pessoas por snap_fraction cria um primeiro lugar, e um
    primeiro lugar é lido como resultado mesmo debaixo de um IC que o
    desmente.
    """
    res = _result({1: 12, 2: 20, 3: 5},
                  names={1: "carlos", 2: "ana", 3: "bruno"})
    df = build_table(res)
    assert df["player"].to_list() == ["ana", "bruno", "carlos"]

    by_n = build_table(res, sort="n")
    assert by_n["n"].to_list() == [20, 12, 5]

    for col, _label, _dec in FEATURES:
        vals = df[f"{col}__med"].to_list()
        assert vals != sorted(vals) or vals != sorted(vals, reverse=True)


def test_n_is_reported_per_player():
    res = _result({1: 12, 2: 26})
    df = build_table(res)
    assert sorted(df["n"].to_list()) == [12, 26]


def test_all_four_features_present():
    res = _result({1: 12})
    df = build_table(res)
    for col, _label, _dec in FEATURES:
        for tag in ("med", "p90"):
            assert f"{col}__{tag}" in df.columns
            assert f"{col}__{tag}_lo" in df.columns
            assert f"{col}__{tag}_hi" in df.columns
    assert [c for c, _, _ in FEATURES] == [
        "snap_fraction", "path_ratio", "settle_ms", "peak_speed_dps"
    ]


# --------------------------------------------------------------------------
# linguagem da saída
# --------------------------------------------------------------------------

VETADAS = [
    "suspeit", "anômal", "anomal", "flag", "cheat", "hack", "aimbot",
    "score", "pontuaç", "ranking", "classific", "detect", "alerta",
    "outlier", "acusa", "culpad", "limiar", "threshold",
]


def test_render_has_no_verdict_language():
    res = _result({i: 10 + i for i in range(1, 11)})
    txt = render(res, build_table(res)).lower()
    encontradas = [v for v in VETADAS if v in txt]
    assert not encontradas, f"vocabulário de veredito na saída: {encontradas}"


def test_render_shows_every_player_and_their_n():
    res = _result({i: 10 + i for i in range(1, 11)},
                  names={i: f"jogador_{i}" for i in range(1, 11)})
    txt = render(res, build_table(res))
    for i in range(1, 11):
        assert f"jogador_{i}" in txt
    # cada feature aparece com seu bloco
    for _col, label, _dec in FEATURES:
        assert label in txt


def test_render_marks_unreachable_bounds():
    res = _result({1: 12, 2: 13})
    txt = render(res, build_table(res))
    assert "--" in txt  # p90 sem limite superior com n desta ordem


def test_render_states_the_no_baseline_caveat():
    """A ressalva metodológica é parte da saída, não do README."""
    res = _result({1: 12})
    txt = render(res, build_table(res)).lower()
    assert "baseline" in txt
    assert "n=1" in txt
