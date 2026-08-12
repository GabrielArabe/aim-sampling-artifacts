"""
Trajetórias de mira sintéticas.

Serve para DUAS coisas, ambas importantes:

  1. Testar o pipeline sem depender de demo nenhum.
  2. Ser o boneco de teste do detector. Se o detector não separa
     nem o caso sintético grosseiro, não faz sentido rodar em dado real.

O que este módulo NÃO é: evidência. Uma trajetória sintética "de cheat"
é o que EU acho que um cheat parece. A validação de verdade só vem de
demos rotulados. Confundir os dois é o erro mais comum em trabalho
amador de detecção -- o modelo aprende a distinguir o gerador, não o
fenômeno.

--------------------------------------------------------------------------
DEFEITO CORRIGIDO -- o gerador anterior era DEPENDENTE DE GRADE
--------------------------------------------------------------------------
A versão anterior gerava tremor como `rng.normal(0, tremor_deg, n).cumsum()`
e ruído de humanização como `rng.normal(0, humanize_deg, n)`: os dois
definidos em unidades de AMOSTRA, não de tempo.

Consequência: a trajetória produzida dependia de `dt`. Um passeio aleatório
com n passos tem variância proporcional a n, então dobrar a taxa de
amostragem dobrava a variância acumulada do tremor em vez de resolver
melhor a MESMA trajetória. Ruído branco por amostra tem o mesmo problema ao
contrário: seu comprimento de caminho cresce com o número de amostras.

O que isso invalidava:

  - Qualquer comparação entre tickrates feita com este gerador. Não havia
    "a mesma trajetória em duas grades" para comparar -- eram dois processos
    estocásticos diferentes.
  - Os testes de test_features.py PASSAVAM, mas passavam contra um gerador
    errado. Eles verificavam separação humano/snap a 64 Hz, que continua
    valendo; o que nunca foi verificado é que valessem em outra taxa.
  - Calibração: o tremor de banda larga era muito mais forte que mira real.
    Medido contra os três demos, a versão antiga degradava path_ratio até
    0.72x ao decimar 64->32 Hz, onde a mira real dá 0.982x. Um detector
    ajustado contra esse gerador estaria ajustado contra o ruído errado.

Agora a trajetória é função contínua do tempo, amostrada em `dt`. Mudar
`dt` reamostra a mesma curva, que é a única semântica que permite testar
invariância de grade (ver tests/test_sampling_invariance.py).

--------------------------------------------------------------------------
CALIBRAÇÃO
--------------------------------------------------------------------------
Constantes ajustadas por busca em grade contra as 263 janelas reais dos
três demos de matchmaking. Alvo e resultado, a 64 Hz e decimando 64->32:

    métrica                  alvo real   sintético
    path_ratio (nível)           1.355       1.318
    path_ratio (razão 32/64)     0.990       0.982
    snap_fraction (razão)        1.700       1.974
    peak_speed (razão)           0.840       0.987
    n_peaks (razão)              0.570       0.667
    n_peaks (nível)              7.000       5.000

path_ratio -- a única feature admissível para comparação entre tickrates --
está calibrado. peak_speed e n_peaks NÃO estão: o gerador usa pulsos breves
como degraus, que a 64 Hz não são resolvidos, então o pico não desinfla ao
decimar como no dado real. As duas estão aposentadas (ver features.py), por
isso o resíduo foi aceito em vez de forçado. Se alguma delas voltar a ser
usada, esta calibração precisa ser refeita antes.

Amostra de calibração: n=3 partidas, mesmo mapa, mesmo tier, 64 tick. Não é
baseline; é o que havia.
"""

from __future__ import annotations

import numpy as np

# Calibrados contra as 263 janelas reais. Ver seção CALIBRAÇÃO acima.
_N_PULSE = 12          # pulsos corretivos breves na janela
_PULSE_S = 0.008       # largura temporal do pulso, em segundos
_PULSE_A = 0.8         # escala do deslocamento por pulso, em graus
_SLOW_A = 0.22         # escala do tremor lento (1.5-8 Hz), em graus


def _min_jerk(tau: np.ndarray) -> np.ndarray:
    """Perfil de posição de jerk mínimo, normalizado de 0 a 1."""
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _mj_t(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Jerk mínimo em TEMPO CONTÍNUO: 0 antes de t0, 1 depois de t1."""
    return _min_jerk(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))


def _tremor(t: np.ndarray, rng: np.random.Generator, scale: float, span: float
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Tremor como função contínua do tempo. Devolve (componente_yaw, pitch).

    Duas partes, porque uma só não reproduz o dado real: tremor lento de
    banda estreita (1.5-8 Hz) responde pelo comprimento de caminho, e
    pulsos breves respondem pela microestrutura da velocidade. Mira real
    tem picos BREVES que elevam a velocidade instantânea sem somar caminho
    -- tremor de banda larga contínuo faz o contrário e foi o que
    descalibrou a versão anterior.

    `span` é a duração NOMINAL em segundos, não t[-1]. Sortear o instante
    dos pulsos a partir do último ponto amostrado reintroduziria dependência
    de grade pela porta dos fundos: `n = int(duration_s/dt)` trunca, então
    t[-1] vale 0.5859 s a 128 Hz e 0.5781 s a 64 Hz, e os pulsos cairiam em
    lugares diferentes. Coberto por
    test_gerador_define_a_mesma_trajetoria_em_qualquer_dt.
    """
    yaw = np.zeros_like(t)
    pitch = np.zeros_like(t)
    for f, a in zip(rng.uniform(1.5, 8.0, 4), rng.uniform(0.5, 1.5, 4) * _SLOW_A * scale):
        ph = rng.uniform(0, 2 * np.pi)
        yaw = yaw + a * np.sin(2 * np.pi * f * t + ph)
        pitch = pitch + 0.4 * a * np.sin(2 * np.pi * f * t + ph / 2)
    for _ in range(_N_PULSE):
        tk = rng.uniform(0.0, span)
        s = _PULSE_S * rng.uniform(0.7, 1.4)
        step = 0.5 * (1.0 + _erf((t - tk) / (s * np.sqrt(2.0))))
        yaw = yaw + rng.normal(0, _PULSE_A * scale) * step
        pitch = pitch + 0.4 * rng.normal(0, _PULSE_A * scale) * step
    return yaw, pitch


def _erf(x: np.ndarray) -> np.ndarray:
    """erf vetorizado sem depender de scipy."""
    # Abramowitz & Stegun 7.1.26 -- erro < 1.5e-7, sobra para tremor.
    s = np.sign(x)
    x = np.abs(x)
    tt = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * tt - 1.453152027) * tt) + 1.421413741) * tt
                - 0.284496736) * tt + 0.254829592) * tt * np.exp(-x * x)
    return s * y


def human_flick(
    amplitude_deg: float = 40.0,
    dt: float = 1 / 64,
    duration_s: float = 0.60,
    *,
    ballistic_s: float = 0.18,
    undershoot: float = 0.88,
    tremor_deg: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flick humano: fase balística que erra o alvo, correção, tremor.

    Retorna (yaw, pitch) amostrados em `dt`. A trajetória é definida em
    tempo contínuo: mudar `dt` reamostra a MESMA curva.

    tremor_deg agora é um MULTIPLICADOR do tremor calibrado (1.0 = calibrado
    contra dado real), não mais um desvio-padrão por amostra. A mudança de
    semântica é intencional -- desvio por amostra não é uma quantidade
    bem definida em tempo contínuo.
    """
    rng = rng or np.random.default_rng(0)
    n = int(duration_s / dt)
    t = np.arange(n) * dt

    # Fases em SEGUNDOS, não em amostras. A correção começa depois da
    # latência de feedback visual (~40 ms) e o overshoot é corrigido em
    # seguida -- é a estrutura de Woodworth que as features procuram.
    t_bal = ballistic_s
    t_corr0 = t_bal + 0.04
    t_corr1 = t_corr0 + 0.12
    t_micro = t_corr1 + 0.08

    yaw = amplitude_deg * undershoot * _mj_t(t, 0.0, t_bal)
    residual = amplitude_deg * (1 - undershoot)
    yaw = yaw + residual * 1.15 * _mj_t(t, t_corr0, t_corr1)
    yaw = yaw - residual * 0.15 * _mj_t(t, t_corr1, t_micro)
    pitch = np.full(n, -2.0) + 0.6 * _mj_t(t, 0.0, t_bal + 0.02)

    ty, tp = _tremor(t, rng, tremor_deg, duration_s)
    return yaw + ty, pitch + tp


def snap_flick(
    amplitude_deg: float = 40.0,
    dt: float = 1 / 64,
    duration_s: float = 0.60,
    *,
    snap_ticks: int = 1,
    humanize_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Snap: a mira vai do repouso ao alvo em snap_ticks ticks e para.

    humanize_deg > 0 injeta tremor SEM a estrutura de submovimento,
    imitando um cheat com "humanização" ingênua.

    Nota: snap_ticks é deliberadamente em TICKS e não em segundos -- um
    aimbot que corrige em N ticks é definido sobre a grade do jogo, não
    sobre tempo contínuo, e essa é justamente a diferença que o gerador
    deve representar. Só o tremor de humanização passou a ser contínuo.
    """
    rng = rng or np.random.default_rng(0)
    n = int(duration_s / dt)
    t = np.arange(n) * dt
    i0 = n // 2

    traj = np.zeros(n)
    ramp = _min_jerk(np.linspace(0.0, 1.0, snap_ticks + 1)[1:]) if snap_ticks > 1 else np.array([1.0])
    traj[i0 : i0 + len(ramp)] = amplitude_deg * ramp
    traj[i0 + len(ramp) :] = amplitude_deg
    pitch = np.full(n, -2.0)

    if humanize_deg > 0:
        ty, tp = _tremor(t, rng, humanize_deg, duration_s)
        traj = traj + ty
        pitch = pitch + tp
    return traj, pitch
