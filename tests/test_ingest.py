"""
Testes de ingest.py.

Dois grupos. O primeiro cobre a inferência de tickrate -- a armadilha nº3
do README, a que escala toda velocidade angular por um fator constante sem
parecer bug. O segundo congela o contrato de nomes de coluna do awpy 2.0.2,
verificado contra um demo real de matchmaking; se uma versão futura
renomear qualquer coisa, o teste quebra em vez de a coluna virar null em
silêncio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest import (  # noqa: E402
    MIN_SAMPLES,
    build_windows,
    infer_tickrate,
    nearest_standard_tickrate,
    sample_step,
)


# --------------------------------------------------------------------------
# tickrate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rate", [32.0, 64.0, 100.0, 128.0])
def test_infer_tickrate_recovers_rate(rate):
    tick = np.arange(1, 5000)
    game_time = 43.28 + tick / rate
    assert infer_tickrate(tick, game_time) == pytest.approx(rate, rel=1e-9)


def test_infer_tickrate_64_not_128_on_realistic_input():
    """
    O caso concreto do demo de teste: 64 Hz, e o awpy afirmando 128.

    Se esta asserção inverter, toda velocidade angular do estudo sai
    multiplicada por 2 e as distribuições parecem plausíveis assim mesmo.
    """
    tick = np.arange(1, 142397)
    game_time = 43.28125 + tick * 0.015625
    assert infer_tickrate(tick, game_time) == pytest.approx(64.0, rel=1e-9)


def test_infer_tickrate_survives_tick_gaps():
    """
    A tabela do awpy vem filtrada para in_play_ticks: freezetime, warmup e
    timeout saem e deixam buracos. No demo real havia um buraco de 7319
    ticks e mais 12 pequenos. A mediana precisa atravessar isso intacta.
    """
    a = np.arange(1, 40_000)
    b = np.arange(47_319, 90_000)  # buraco de 7319 ticks
    tick = np.concatenate([a, b])
    game_time = 43.28 + tick / 64.0
    assert infer_tickrate(tick, game_time) == pytest.approx(64.0, rel=1e-6)


def test_infer_tickrate_unsorted_input():
    tick = np.arange(1, 5000)
    game_time = 43.28 + tick / 64.0
    rng = np.random.default_rng(0)
    p = rng.permutation(len(tick))
    assert infer_tickrate(tick[p], game_time[p]) == pytest.approx(64.0, rel=1e-9)


def test_infer_tickrate_tolerates_f32_quantization():
    """game_time chega como f32; a quantização não pode mover o resultado."""
    tick = np.arange(1, 100_000)
    game_time = np.float32(43.28125) + np.float32(1.0 / 64.0) * tick.astype(np.float32)
    assert infer_tickrate(tick, game_time.astype(np.float64)) == pytest.approx(64.0, rel=1e-4)


def test_infer_tickrate_raises_when_estimators_disagree():
    """
    game_time resetando no meio (demo concatenado, troca de mapa).

    A base longa vê uma coisa, a mediana vê outra. Duas leituras
    incompatíveis do mesmo arquivo é motivo para parar, não para escolher
    uma: o custo de escolher errado é um estudo inteiro com escala errada.
    """
    tick = np.arange(1, 20_000)
    game_time = 43.28 + tick / 64.0
    game_time[10_000:] -= 150.0  # reset parcial: span total ainda positivo
    with pytest.raises(ValueError, match="discordam"):
        infer_tickrate(tick, game_time)


def test_infer_tickrate_raises_on_backwards_span():
    """Reset grande o bastante para o demo terminar antes de começar."""
    tick = np.arange(1, 20_000)
    game_time = 43.28 + tick / 64.0
    game_time[10_000:] -= 500.0
    with pytest.raises(ValueError, match="degenerado"):
        infer_tickrate(tick, game_time)


def test_infer_tickrate_raises_on_frozen_clock():
    tick = np.arange(1, 1000)
    game_time = np.full(len(tick), 43.28)
    with pytest.raises(ValueError):
        infer_tickrate(tick, game_time)


def test_infer_tickrate_raises_on_too_few_samples():
    with pytest.raises(ValueError):
        infer_tickrate(np.array([1, 2]), np.array([0.0, 0.015625]))


def test_infer_tickrate_ignores_duplicate_ticks():
    """Dez jogadores por tick: a tabela tem o mesmo tick dez vezes."""
    base = np.arange(1, 2000)
    tick = np.repeat(base, 10)
    game_time = 43.28 + tick / 64.0
    assert infer_tickrate(tick, game_time) == pytest.approx(64.0, rel=1e-9)


def test_nearest_standard_tickrate():
    assert nearest_standard_tickrate(64.0) == (64.0, pytest.approx(0.0))
    assert nearest_standard_tickrate(127.9)[0] == 128.0
    assert nearest_standard_tickrate(50.0)[1] > 0.2  # longe de tudo


def test_sample_step_is_median_not_mean():
    """Um buraco gigante não pode mover o passo de amostragem."""
    ticks = pl.DataFrame({"tick": list(range(1, 1000)) + [8319, 8320, 8321]})
    assert sample_step(ticks) == 1


# --------------------------------------------------------------------------
# contrato de colunas e recorte de janelas
# --------------------------------------------------------------------------

TICKRATE = 64.0
SPAN = 48  # 0.75 s


def _ticks(players, n=200, start=1000):
    """players: {steamid: (name, yaw_fn)}. Gera tabela no formato do awpy."""
    rows = []
    for t in range(start, start + n):
        for sid, (name, yaw_fn) in players.items():
            rows.append(
                {"tick": t, "steamid": sid, "name": name,
                 "yaw": float(yaw_fn(t - start)), "pitch": 0.0}
            )
    return pl.DataFrame(rows)


def _kills(entries):
    """entries: lista de (attacker_steamid, attacker_name, tick, victim_name)."""
    return pl.DataFrame(
        [
            {"attacker_steamid": sid, "attacker_name": name, "tick": t,
             "victim_name": victim, "weapon": "ak47", "headshot": True}
            for sid, name, t, victim in entries
        ]
    )


def test_victim_comes_from_victim_name_not_user_name():
    """
    dem.kills do awpy 2.0.2 usa victim_name. Não existe user_name.

    O código antigo lia kill.get("user_name"), que devolve None sem erro --
    a coluna inteira sairia nula e o parquet ficaria pronto para o lote.
    """
    ticks = _ticks({1: ("atirador", lambda i: i * 0.5)})
    kills = _kills([(1, "atirador", 1100, "alvo")])
    w, _, _ = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    assert w.height == 1
    assert w["victim"][0] == "alvo"


def test_kills_missing_victim_name_would_fail_loudly():
    """Se um dia a coluna sumir, é KeyError -- não uma coluna de None."""
    ticks = _ticks({1: ("atirador", lambda i: i * 0.5)})
    kills = _kills([(1, "atirador", 1100, "alvo")]).drop("victim_name")
    with pytest.raises(Exception):
        build_windows(ticks, kills, "d.dem", TICKRATE, 1)


def test_join_is_by_steamid_not_name():
    """
    Dois jogadores com o mesmo nick. Casar por nome pegaria os ticks
    errados, ou os dois. Nick não é chave: dá para trocar no meio da
    partida e dá para copiar o do adversário.
    """
    ticks = _ticks({
        1: ("mesmo_nick", lambda i: i * 0.5),    # 24 graus na janela
        2: ("mesmo_nick", lambda i: i * 0.05),   # 2.4 graus, abaixo do mínimo
    })
    kills = _kills([(1, "mesmo_nick", 1100, "alvo")])
    w, _, _ = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    assert w.height == 1
    assert w["steamid"][0] == 1
    assert w["net_disp_deg"][0] == pytest.approx(23.5, abs=0.6)


def test_window_with_tick_gap_is_dropped():
    """
    Janela atravessando um buraco de amostragem.

    Sem esta guarda o passo angular através do buraco é grande, o dt
    assumido é pequeno, e sai uma velocidade enorme -- um "snap" que é
    artefato de amostragem, não movimento de mira. É o falso positivo mais
    caro possível: ele aparece exatamente na feature que o estudo mede.
    """
    ticks = _ticks({1: ("p", lambda i: i * 0.5)})
    furado = ticks.filter(pl.col("tick") != 1080)  # buraco dentro da janela
    kills = _kills([(1, "p", 1100, "alvo")])

    w_ok, _, d_ok = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    w_bad, _, d_bad = build_windows(furado, kills, "d.dem", TICKRATE, 1)

    assert w_ok.height == 1 and d_ok["dropped_tick_gap"] == 0
    assert w_bad.height == 0 and d_bad["dropped_tick_gap"] == 1


def test_window_spans_exactly_075s_of_ticks():
    ticks = _ticks({1: ("p", lambda i: i * 0.5)})
    kills = _kills([(1, "p", 1100, "alvo")])
    w, _, d = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    assert d["window_ticks"] == SPAN
    assert w["n_samples"][0] == SPAN
    assert w["duration_s"][0] == pytest.approx((SPAN - 1) / TICKRATE)


def test_window_ticks_scale_with_tickrate():
    """A 128 Hz a mesma janela de 0.75 s tem o dobro de amostras."""
    ticks = _ticks({1: ("p", lambda i: i * 0.25)}, n=400)
    kills = _kills([(1, "p", 1200, "alvo")])
    _, _, d = build_windows(ticks, kills, "d.dem", 128.0, 1)
    assert d["window_ticks"] == 96


def test_roster_keeps_players_with_zero_windows():
    """
    O elenco vem dos ticks, não das kills.

    Quem não matou ninguém, ou cujas janelas foram todas descartadas,
    continua no elenco. Uma tabela que só lista quem sobreviveu aos filtros
    esconde a cobertura real da amostra.
    """
    ticks = _ticks({
        1: ("mata", lambda i: i * 0.5),
        2: ("nao_mata", lambda i: 0.0),
        3: ("parado", lambda i: 0.0),
    })
    kills = _kills([(1, "mata", 1100, "alvo")])
    w, roster, _ = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    assert roster.height == 3
    assert set(roster["steamid"].to_list()) == {1, 2, 3}
    assert w["steamid"].n_unique() == 1


def test_below_min_net_deg_is_counted_not_silently_lost():
    ticks = _ticks({1: ("p", lambda i: i * 0.01)})  # ~0.5 grau total
    kills = _kills([(1, "p", 1100, "alvo")])
    w, _, d = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    assert w.height == 0
    assert d["dropped_below_min_net_deg"] == 1


def test_short_window_at_start_of_data_is_dropped():
    ticks = _ticks({1: ("p", lambda i: i * 0.5)}, n=200, start=1000)
    kills = _kills([(1, "p", 1004, "alvo")])  # só 5 ticks antes
    w, _, d = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    assert w.height == 0
    assert d["dropped_too_short"] == 1
    assert MIN_SAMPLES == 8


def test_unknown_attacker_is_counted():
    ticks = _ticks({1: ("p", lambda i: i * 0.5)})
    kills = _kills([(99, "fantasma", 1100, "alvo")])
    w, _, d = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    assert w.height == 0
    assert d["dropped_no_attacker"] == 1


def test_diagnostics_account_for_every_kill():
    """Toda kill entra em exatamente um balde. Nada some sem registro."""
    ticks = _ticks({1: ("p", lambda i: i * 0.5), 2: ("q", lambda i: 0.0)})
    kills = _kills([
        (1, "p", 1100, "a"),      # ok
        (2, "q", 1100, "b"),      # sem movimento
        (99, "x", 1100, "c"),     # atacante desconhecido
        (1, "p", 1002, "d"),      # curta demais
    ])
    _, _, d = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    total = d["windows_kept"] + sum(v for k, v in d.items() if k.startswith("dropped_"))
    assert total == d["kills_total"] == 4


def test_empty_kills_gives_empty_frame_with_schema():
    ticks = _ticks({1: ("p", lambda i: i * 0.5)})
    kills = _kills([]) if False else pl.DataFrame(
        schema={"attacker_steamid": pl.UInt64, "attacker_name": pl.String,
                "tick": pl.Int64, "victim_name": pl.String,
                "weapon": pl.String, "headshot": pl.Boolean}
    )
    w, roster, d = build_windows(ticks, kills, "d.dem", TICKRATE, 1)
    assert w.height == 0
    assert "steamid" in w.columns  # schema presente para concat de lote
    assert roster.height == 1
