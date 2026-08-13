"""
Features de uma janela de mira anterior a um evento (tiro ou kill).

A hipótese que estas features testam é a de controle motor: um humano
move a mira em SUBMOVIMENTOS. Uma fase balística rápida que erra o alvo,
seguida de uma ou mais correções pequenas (Woodworth, 1899; e a lei de
Fitts em geral). Isso deixa três marcas:

  1. o caminho percorrido é MAIOR que a distância líquida (overshoot);
  2. a velocidade tem múltiplos picos, não um só;
  3. existe um tempo de acomodação antes do tiro.

Assistência de mira, na forma ingênua, produz o oposto: caminho ótimo,
pico único, acomodação zero. Cheat "humanizado" injeta ruído pra imitar
isso -- e aí a pergunta vira se o ruído injetado tem a MESMA ESTRUTURA
do ruído humano, o que é uma pergunta bem mais interessante e bem mais
difícil de falsear pro lado do atacante.

IMPORTANTE -- limitação conhecida: CS2 usa subtick. O tiro acontece
ENTRE ticks e o servidor interpola. O view angle no tick N não é
exatamente o ângulo no instante do disparo. Para tudo que envolve o
instante exato do tiro (ex.: distância à hitbox no disparo), isso é um
erro de amostragem real que precisa ser tratado. Para as features de
TRAJETÓRIA abaixo, que olham centenas de ms, o efeito é desprezível.

--------------------------------------------------------------------------
FEATURES APOSENTADAS
--------------------------------------------------------------------------
Continuam sendo CALCULADAS -- remover mudaria a matemática testada e
quebraria a suíte sem necessidade -- mas NÃO devem entrar em análise por
jogador nem em modelo. Ver tests/test_sampling_invariance.py.

(1) n_peaks, settle_ms, settle_disp_deg -- aposentadas.

    n_peaks conta máximos locais da velocidade AMOSTRADA, e em mira real a
    contagem acompanha a taxa de amostragem: 7.0 a 64 Hz contra 4.0 a 32 Hz
    nas 263 janelas DE KILL dos três demos (razão das medianas; janelas
    amostradas do fluxo de ticks dão 0.33x -- ver README), ou seja 0.57x
    para metade da taxa nesta população de janelas. Está
    medindo a grade, não submovimento. O sintoma já era visível antes do
    teste de decimação: mediana de 7.5 picos numa janela de 750 ms é um a
    cada 100 ms, e correção sob feedback visual tem latência de 100-200 ms,
    então o teto plausível é ~4-7 e realisticamente menos. O excedente é
    quantização de sensor e tremor.

    settle_ms e settle_disp_deg: componente de variância entre jogadores
    estimado em ZERO (IC95 [0.000, 0.014]) sobre 28 jogadores em 3 partidas,
    e teste de permutação p=0.56. Não carregam sinal individual. Como são
    definidas a partir do índice do pico de velocidade, herdam de quebra a
    dependência de grade de peak_speed_dps.

(2) snap_fraction -- utilizável, mas SÓ dentro de um tickrate fixo.

    É O(dt): o maior passo único é ~v_max*dt, então snap_fraction tende a
    ZERO conforme a taxa de amostragem sobe. Não existe quantidade em tempo
    contínuo que ela estime. Nos demos: 1.69x indo de 64 para 32 Hz. Como
    matchmaking é 64 tick e torneio/FACEIT são 128, comparar as duas
    populações nesta feature produz diferença sistemática para mira
    idêntica. Dentro de um tickrate a ordenação se preserva (Spearman 0.97).

Restam como admissíveis para comparação entre tickrates: net_disp_deg,
path_deg, path_ratio, mean_speed_dps, duration_s.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from kinematics import (
    angular_accel,
    angular_speed,
    angular_step,
    net_displacement,
    unwrap_yaw,
)


@dataclass
class AimWindow:
    """Features de uma única janela de mira."""

    n_samples: int
    duration_s: float
    net_disp_deg: float
    path_deg: float
    path_ratio: float          # path / net. Humano > 1. Ótimo = 1.0
    peak_speed_dps: float
    mean_speed_dps: float
    peak_accel_dps2: float
    snap_fraction: float       # maior passo único / desloc. líquido -- ver (2)
    n_peaks: int               # submovimentos detectados -- APOSENTADA, ver (1)
    settle_ms: float           # pico -> acomodação -- APOSENTADA, ver (1)
    settle_disp_deg: float     # deslocamento depois do pico -- APOSENTADA, (1)

    def to_dict(self) -> dict:
        return asdict(self)


def _count_submovements(speed: np.ndarray, rel_height: float = 0.15) -> int:
    """
    Conta picos locais de velocidade acima de rel_height * pico global.

    Proxy grosseiro para submovimentos corretivos. Grosseiro de propósito:
    a alternativa é decomposição de submovimento por ajuste de curva, que
    é mais correta e MUITO mais lenta. Trocar depois se a feature pagar.
    """
    if len(speed) < 3:
        return 0
    thr = rel_height * float(np.max(speed)) if np.max(speed) > 0 else 0.0
    interior = (speed[1:-1] > speed[:-2]) & (speed[1:-1] >= speed[2:]) & (speed[1:-1] > thr)
    return int(np.sum(interior))


def extract_window(
    yaw: np.ndarray,
    pitch: np.ndarray,
    dt: float,
    *,
    settle_frac: float = 0.10,
) -> AimWindow:
    """
    Calcula as features de uma janela já recortada.

    yaw/pitch: arrays na ordem cronológica, terminando NO evento.
    dt: segundos entre amostras (1/tickrate do demo).
    """
    yaw = unwrap_yaw(np.asarray(yaw, dtype=np.float64))
    pitch = np.asarray(pitch, dtype=np.float64)

    if len(yaw) < 3:
        raise ValueError("janela curta demais (< 3 amostras)")

    steps = angular_step(yaw, pitch)
    speed = angular_speed(yaw, pitch, dt)
    accel = angular_accel(speed, dt)

    path = float(np.sum(steps))
    net = net_displacement(yaw, pitch)
    peak = float(np.max(speed))
    i_peak = int(np.argmax(speed))

    # Acomodação: a partir do pico, quando a velocidade cai abaixo de
    # settle_frac do pico e NÃO volta a subir acima dele.
    tail = speed[i_peak:]
    below = tail < settle_frac * peak if peak > 0 else np.ones_like(tail, dtype=bool)
    settle_idx = len(tail) - 1
    for i in range(len(tail)):
        if below[i:].all():
            settle_idx = i
            break
    settle_ms = settle_idx * dt * 1000.0
    settle_disp = float(np.sum(steps[i_peak : i_peak + settle_idx])) if settle_idx > 0 else 0.0

    return AimWindow(
        n_samples=len(yaw),
        duration_s=(len(yaw) - 1) * dt,
        net_disp_deg=net,
        path_deg=path,
        path_ratio=path / net if net > 1e-9 else float("nan"),
        peak_speed_dps=peak,
        mean_speed_dps=float(np.mean(speed)),
        peak_accel_dps2=float(np.max(np.abs(accel))) if len(accel) else 0.0,
        snap_fraction=float(np.max(steps)) / net if net > 1e-9 else float("nan"),
        n_peaks=_count_submovements(speed),
        settle_ms=settle_ms,
        settle_disp_deg=settle_disp,
    )
