"""
Ingestão: demo .dem -> tabela de features por kill.

Uso:
    python src/ingest.py data/algum.dem --inspect
    python src/ingest.py data/*.dem -o out/windows.parquet

Validado contra demo real de matchmaking CS2 (awpy 2.0.2 / demoparser2).
Ver NOTAS DE PARSING no fim do arquivo para o que foi verificado e o que
continua sendo suposição.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import extract_window  # noqa: E402

# O awpy já força ["last_place_name","X","Y","Z","health","team_name"] e faz
# union com o que a gente passa. pitch/yaw NÃO vêm por padrão -- sem passar
# aqui, as colunas simplesmente não existem na tabela. É a pegadinha nº1.
# Note que 'team_name' chega renomeado para 'side' (awpy.parsers.utils.
# fix_common_names) e 'last_place_name' para 'place'. Pedir 'team_name' e
# procurar 'team_name' depois não acha nada.
NEEDED_PROPS = ["pitch", "yaw", "X", "Y", "Z", "health", "team_name", "team_clan_name"]

WINDOW_S = 0.75    # janela antes do kill
MIN_NET_DEG = 5.0  # ignora kills sem movimento real de mira
MIN_SAMPLES = 8    # janela curta demais não sustenta feature de trajetória

# Só para sanidade: se o tickrate medido não bater com nenhum destes, o
# número provavelmente está errado e vale investigar antes de confiar.
STANDARD_TICKRATES = (32.0, 64.0, 100.0, 128.0)


def infer_tickrate(
    tick: np.ndarray,
    game_time: np.ndarray,
    *,
    rel_tol: float = 0.01,
) -> float:
    """
    Mede o tickrate real, em Hz, a partir do par (tick, game_time).

    Função PURA -- não conhece awpy. É onde mora a decisão que o README
    chama de armadilha nº3.

    Por que isto não é opcional: o awpy tem DEFAULT_SERVER_TICKRATE = 128 e
    NUNCA infere nada -- `Demo(path)` guarda 128 sem olhar para o arquivo.
    Muito demo de matchmaking CS2 é 64. Usar o default do awpy num demo 64
    escala TODA velocidade angular por 2x. O erro é multiplicativo e
    uniforme, então não aparece como bug: aparece como descoberta.

    Dois estimadores independentes, e os dois têm que concordar:

      base longa  -- (game_time[-1]-game_time[0]) / (tick[-1]-tick[0]).
                     Imune a ruído de quantização do f32 em game_time.
      mediana     -- mediana de d(game_time)/d(tick) entre amostras.
                     Imune a buraco no meio (pausa, timeout, troca de lado).

    Cada um é cego para o que o outro enxerga. Se discordam, alguma
    premissa quebrou -- game_time resetou, o demo foi concatenado, os
    ticks não são monotônicos -- e aí a resposta certa é falhar alto, não
    escolher um dos dois. Um tickrate errado em silêncio contamina o
    estudo inteiro sem deixar rastro.
    """
    tick = np.asarray(tick, dtype=np.float64)
    game_time = np.asarray(game_time, dtype=np.float64)

    if tick.shape != game_time.shape:
        raise ValueError("tick e game_time com tamanhos diferentes")
    if len(tick) < 3:
        raise ValueError(f"amostras insuficientes para inferir tickrate (n={len(tick)})")

    order = np.argsort(tick, kind="stable")
    tick, game_time = tick[order], game_time[order]

    keep = np.concatenate(([True], np.diff(tick) > 0))
    tick, game_time = tick[keep], game_time[keep]
    if len(tick) < 3:
        raise ValueError("ticks distintos insuficientes para inferir tickrate")

    span_tick = tick[-1] - tick[0]
    span_time = game_time[-1] - game_time[0]
    if span_tick <= 0 or span_time <= 0:
        raise ValueError(
            f"span degenerado: {span_tick:.0f} ticks / {span_time:.3f} s"
        )
    s_span = span_time / span_tick

    ratios = np.diff(game_time) / np.diff(tick)
    ratios = ratios[ratios > 0]
    if len(ratios) == 0:
        raise ValueError("game_time não avança em nenhum passo")
    s_med = float(np.median(ratios))

    if abs(s_med - s_span) > rel_tol * s_med:
        raise ValueError(
            "estimadores de tickrate discordam: "
            f"base longa {1.0 / s_span:.4f} Hz vs mediana {1.0 / s_med:.4f} Hz. "
            "game_time pode ter resetado ou o demo pode estar concatenado. "
            "NÃO adivinhe o tickrate -- investigue."
        )

    return 1.0 / s_med


def nearest_standard_tickrate(tickrate: float) -> tuple[float, float]:
    """Devolve (tickrate padrão mais próximo, desvio relativo)."""
    nearest = min(STANDARD_TICKRATES, key=lambda r: abs(r - tickrate))
    return nearest, abs(tickrate - nearest) / nearest


def sample_step(ticks: pl.DataFrame) -> int:
    """
    Passo de amostragem, em ticks, da tabela de análise.

    Normalmente 1. Mediana e não média porque a tabela do awpy vem filtrada
    para in_play_ticks -- freezetime, warmup e timeout saem, deixando
    buracos. Num demo de teste real: 133.704 passos de 1 tick, um buraco
    único de 7.319 ticks e mais 12 buracos pequenos. A média seria mentira,
    a mediana é 1.
    """
    t = ticks.select("tick").unique().sort("tick").to_series().to_numpy()
    if len(t) < 3:
        raise ValueError(f"ticks insuficientes (n={len(t)})")
    return int(np.median(np.diff(t)))


# Contrato de colunas do awpy 2.0.2, verificado contra demo real. Toda
# coluna que o código LÊ está listada aqui, inclusive as que são só
# metadado. O motivo é o modo de falha, não a importância: pl.DataFrame
# .iter_rows(named=True) devolve dicts, e dict.get() de uma coluna que não
# existe devolve None sem levantar nada. Foi assim que 'user_name' -- nome
# do CSGO, inexistente no awpy 2.0.2 -- passou despercebido: a coluna
# 'victim' saía inteira nula e o parquet ficava pronto para o lote. Uma
# renomeação em versão futura tem que quebrar aqui, alto, e não virar uma
# coluna de None três etapas adiante.
TICK_COLS = ("tick", "steamid", "name", "yaw", "pitch")
KILL_COLS = ("tick", "attacker_steamid", "attacker_name", "victim_name",
             "weapon", "headshot")


def _require_columns(df: pl.DataFrame, cols: tuple[str, ...], what: str) -> None:
    faltando = [c for c in cols if c not in df.columns]
    if faltando:
        raise KeyError(
            f"{what}: colunas ausentes {faltando}. Presentes: {df.columns}. "
            "Nomes de coluna do awpy/demoparser2 mudam entre versões e entre "
            "GOTV/POV -- rode --inspect antes de processar o lote."
        )


@dataclass
class DemoResult:
    """Saída de um demo: janelas, elenco completo e diagnóstico do parsing."""

    windows: pl.DataFrame
    roster: pl.DataFrame          # os 10, sempre -- venham ou não a ter janela
    tickrate: float
    dt: float
    n_kills: int
    diagnostics: dict = field(default_factory=dict)


def load_demo(path: Path, *, verbose: bool = False):
    from awpy import Demo

    dem = Demo(path, verbose=verbose)
    dem.parse(player_props=NEEDED_PROPS)
    return dem


def demo_tickrate(dem) -> float:
    """
    Extrai o tickrate real do demo.

    Precisa de um passe extra de parse_ticks porque o awpy pede game_time
    para montar in_play_ticks (Demo.parse, linha ~241) e depois RECONSTRÓI
    self.ticks sem ele (linha ~257). O game_time existe no demo; ele só não
    sobrevive até a tabela final. Passar other_props=["game_time"] para
    Demo.parse() também não resolve: lá o argumento só desce para
    parse_events, não para o parse_ticks que gera self.ticks.
    """
    gt = dem.parse_ticks(other_props=["game_time"])
    if "game_time" not in gt.columns:
        raise KeyError(
            f"game_time indisponível (colunas: {gt.columns}). Sem referência "
            "temporal absoluta não dá para medir o tickrate."
        )
    g = gt.select(["tick", "game_time"]).unique(subset=["tick"]).sort("tick").drop_nulls()
    return infer_tickrate(g["tick"].to_numpy(), g["game_time"].to_numpy())


def build_windows(
    ticks: pl.DataFrame,
    kills: pl.DataFrame,
    demo_name: str,
    tickrate: float,
    step: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """
    Recorta as janelas. Devolve (janelas, elenco, diagnóstico).

    Separada de process_demo de propósito: recebe DataFrames, não um objeto
    Demo, então dá para testar todo o contrato de nomes de coluna e todos os
    filtros com tabelas sintéticas, sem precisar de um .dem no disco. É o
    único jeito de ter teste de regressão para os nomes do awpy.
    """
    _require_columns(ticks, TICK_COLS, "dem.ticks")
    _require_columns(kills, KILL_COLS, "dem.kills")

    dt = step / tickrate
    window_ticks = int(round(WINDOW_S * tickrate))

    # Elenco a partir dos TICKS, não das kills. Quem não matou ninguém e
    # quem não gerou janela nenhuma continua existindo e tem que aparecer
    # na tabela final com n=0.
    roster = (
        ticks.select(["steamid", "name"])
        .unique()
        .group_by("steamid")
        .agg(pl.col("name").last().alias("name"))
        .sort("steamid")
    )

    # Índice por steamid, não por nome: nome não é chave. Jogador pode
    # trocar de nick no meio da partida, dois jogadores podem usar o mesmo
    # nick, e nick tem unicode que não sobrevive a todo caminho de I/O.
    per_player: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sid in roster["steamid"].to_list():
        sub = ticks.filter(pl.col("steamid") == sid).sort("tick")
        per_player[sid] = (
            sub["tick"].to_numpy(),
            sub["yaw"].to_numpy(),
            sub["pitch"].to_numpy(),
        )

    rows: list[dict] = []
    n_no_attacker = n_short = n_gap = n_below_min = n_failed = 0

    for kill in kills.iter_rows(named=True):
        sid = kill.get("attacker_steamid")
        tick = kill.get("tick")
        if sid is None or tick is None or sid not in per_player:
            n_no_attacker += 1
            continue

        t_arr, yaw_arr, pitch_arr = per_player[sid]
        sel = (t_arr <= tick) & (t_arr > tick - window_ticks)
        n_sel = int(np.count_nonzero(sel))
        if n_sel < MIN_SAMPLES:
            n_short += 1
            continue

        w_ticks = t_arr[sel]
        # Guarda de contiguidade. A tabela do awpy vem sem freezetime nem
        # warmup, então uma janela que encoste num desses limites tem um
        # salto de tempo real que o dt escalar não representa. O passo
        # angular medido através do buraco é grande, o dt assumido é
        # pequeno, e o resultado é uma velocidade enorme -- ou seja, um
        # "snap" que é artefato de amostragem, não movimento de mira.
        # Neste demo isto nunca disparou (173/173 contíguas); é seguro
        # descartar e é o descarte que mantém dt escalar honesto.
        if not np.all(np.diff(w_ticks) == step):
            n_gap += 1
            continue

        try:
            feat = extract_window(yaw_arr[sel], pitch_arr[sel], dt)
        except ValueError:
            n_failed += 1
            continue

        if feat.net_disp_deg < MIN_NET_DEG:
            n_below_min += 1
            continue

        rows.append(
            {
                "demo": demo_name,
                "steamid": sid,
                "player": kill.get("attacker_name"),
                # dem.kills usa victim_*, NÃO user_*. O nome antigo do CSGO
                # era user_name; no awpy 2.0.2 ele não existe e .get()
                # devolveria None em silêncio para a coluna inteira.
                "victim": kill.get("victim_name"),
                "weapon": kill.get("weapon"),
                "headshot": kill.get("headshot"),
                "tick": tick,
                **feat.to_dict(),
            }
        )

    schema = {
        "demo": pl.String, "steamid": pl.UInt64, "player": pl.String,
        "victim": pl.String, "weapon": pl.String, "headshot": pl.Boolean,
        "tick": pl.Int64,
    }
    windows = pl.DataFrame(rows) if rows else pl.DataFrame(schema=schema)

    diagnostics = {
        "sample_step_ticks": step,
        "window_ticks": window_ticks,
        "kills_total": kills.height,
        "windows_kept": len(rows),
        "dropped_no_attacker": n_no_attacker,
        "dropped_too_short": n_short,
        "dropped_tick_gap": n_gap,
        "dropped_below_min_net_deg": n_below_min,
        "dropped_feature_error": n_failed,
    }
    return windows, roster, diagnostics


def process_demo(path: Path, tickrate: float | None = None) -> DemoResult:
    dem = load_demo(path)
    ticks, kills = dem.ticks, dem.kills

    for col in ("pitch", "yaw"):
        if col not in ticks.columns:
            raise KeyError(
                f"coluna '{col}' ausente. Colunas disponíveis: {ticks.columns}. "
                "Passou player_props no parse()?"
            )

    tickrate = tickrate or demo_tickrate(dem)
    step = sample_step(ticks)
    windows, roster, diagnostics = build_windows(
        ticks, kills, path.name, tickrate, step
    )
    return DemoResult(
        windows=windows,
        roster=roster,
        tickrate=tickrate,
        dt=step / tickrate,
        n_kills=kills.height,
        diagnostics=diagnostics,
    )


def windows_from_demo(path: Path, tickrate: float | None = None) -> pl.DataFrame:
    return process_demo(path, tickrate).windows


def inspect(path: Path) -> None:
    """Diagnóstico de parsing. Rode isto antes de confiar em qualquer lote."""
    dem = load_demo(path, verbose=True)
    print("\nTICKS:", dem.ticks.columns)
    print("\nKILLS:", dem.kills.columns)
    print("\n", dem.ticks.select(["tick", "steamid", "name", "pitch", "yaw"]).head(5))

    from awpy.constants import DEFAULT_SERVER_TICKRATE

    tickrate = demo_tickrate(dem)
    nearest, dev = nearest_standard_tickrate(tickrate)
    print("\n--- TICKRATE ---")
    print(f"  medido de game_time : {tickrate:.4f} Hz  (dt = {1 / tickrate:.9f} s)")
    print(f"  padrão mais próximo : {nearest:.0f} Hz  (desvio {dev * 100:.4f}%)")
    print(f"  awpy acha que é     : {dem.tickrate} Hz  (default={DEFAULT_SERVER_TICKRATE}, nunca inferido)")
    if abs(dem.tickrate - tickrate) > 1e-6:
        fator = dem.tickrate / tickrate
        print(f"  >> usar o valor do awpy escalaria toda velocidade por {fator:.2f}x <<")
    if dev > 0.01:
        print("  >> AVISO: não bate com nenhum tickrate padrão. Investigue. <<")

    step = sample_step(dem.ticks)
    t = dem.ticks.select("tick").unique().sort("tick").to_series().to_numpy()
    d = np.diff(t)
    print("\n--- AMOSTRAGEM ---")
    print(f"  passo mediano: {step} tick(s)   ticks distintos: {len(t)}")
    print(f"  buracos (passo != {step}): {int(np.count_nonzero(d != step))}"
          f"  maior: {int(d.max())} ticks")

    print("\n--- ELENCO ---")
    print(dem.ticks.select(["steamid", "name"]).unique().sort("steamid"))
    print(f"\nkills: {dem.kills.height}  |  atacantes distintos: "
          f"{dem.kills['attacker_steamid'].n_unique()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("demos", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out/windows.parquet"))
    ap.add_argument("--inspect", action="store_true", help="só mostra diagnóstico e sai")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.demos[0])
        return

    frames = []
    for p in args.demos:
        try:
            res = process_demo(p)
            print(f"{p.name}: {res.windows.height} janelas de {res.n_kills} kills "
                  f"@ {res.tickrate:.2f} Hz")
            for k, v in res.diagnostics.items():
                if k.startswith("dropped_") and v:
                    print(f"    {k}: {v}")
            frames.append(res.windows)
        except Exception as exc:  # noqa: BLE001
            print(f"{p.name}: FALHOU ({exc})", file=sys.stderr)

    frames = [f for f in frames if f.height]
    if not frames:
        sys.exit("nenhuma janela extraída")

    out = pl.concat(frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    print(f"-> {args.out}  ({out.height} janelas, {out['steamid'].n_unique()} jogadores)")


if __name__ == "__main__":
    main()


# NOTAS DE PARSING -- awpy 2.0.2 / demoparser2, verificado contra demo real
# de matchmaking CS2 (de_dust2, 64 tick, 173 kills, 10 jogadores).
#
# VERIFICADO:
#   - pitch/yaw só existem se passados em player_props. Confirmado.
#   - dem.kills usa attacker_* / victim_*. NÃO existe user_name.
#   - team_name volta como 'side'; last_place_name volta como 'place'.
#   - game_time não está em dem.ticks; precisa de parse_ticks() separado.
#   - dem.tickrate é o default 128 e nunca é inferido do arquivo.
#   - dem.ticks já vem sem warmup/freezetime (filtro in_play_ticks).
#
# AINDA SUPOSIÇÃO (não verificável com n=1 demo):
#   - que demo GOTV de torneio use os mesmos nomes de coluna. README item 6.
#   - que 128 tick passe pelo mesmo caminho. Não houve demo 128 para testar;
#     a inferência é genérica, mas genérico não é o mesmo que testado.
#   - subtick continua aberto (README item 4): o ângulo no tick do kill não
#     é o ângulo no disparo. Afeta pouco feature de trajetória de 750 ms.
