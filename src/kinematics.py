"""
Cinemática angular do view angle (pitch/yaw) em CS2.

Este módulo é PURO: recebe arrays numpy, devolve arrays numpy.
Nenhuma dependência de demo, parser ou I/O. Isso é proposital --
é a única parte do pipeline que dá pra testar sem dados reais,
então ela precisa ser trivialmente testável.

Convenções do Source 2:
  yaw   in (-180, 180], gira no plano horizontal, WRAPA.
  pitch in [-89, 89],   negativo = olhando pra cima, NÃO wrapa.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wrap_deg",
    "unwrap_yaw",
    "angular_step",
    "angular_speed",
    "angular_accel",
    "cumulative_path",
    "net_displacement",
]


def wrap_deg(a: np.ndarray) -> np.ndarray:
    """Normaliza ângulos para (-180, 180]."""
    a = np.asarray(a, dtype=np.float64)
    return -((-a + 180.0) % 360.0 - 180.0)


def unwrap_yaw(yaw: np.ndarray) -> np.ndarray:
    """
    Remove as descontinuidades de +-180 do yaw.

    Sem isso, um jogador cruzando a fronteira de 180 graus vira um
    "flick" de 360 graus em 1 tick -- que é exatamente a assinatura
    que a gente procura. Falso positivo garantido. Este é o bug nº1
    de análise de view angle.
    """
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.rad2deg(np.unwrap(np.deg2rad(yaw)))


def angular_step(
    yaw: np.ndarray,
    pitch: np.ndarray,
    *,
    great_circle: bool = True,
) -> np.ndarray:
    """
    Deslocamento angular entre amostras consecutivas, em graus.

    Retorna array de tamanho len(yaw) - 1.

    great_circle=True usa a distância real na esfera de visão. Importa:
    a 60 graus de pitch, 1 grau de yaw move a mira metade do que move
    a 0 grau de pitch. Tratar (yaw, pitch) como plano euclidiano infla
    a velocidade angular de quem está olhando pra cima/baixo -- de novo,
    falso positivo sistemático, e correlacionado com estilo de jogo
    (jogador de AWP em ângulo alto vs. rifle em plano).

    Usa a fórmula de Vincenty (bem condicionada para distâncias pequenas),
    não o arccos ingênuo, que perde precisão perto de zero -- e quase todo
    passo entre ticks é perto de zero.
    """
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)

    dyaw = wrap_deg(np.diff(yaw))
    dpitch = np.diff(pitch)

    if not great_circle:
        return np.hypot(dyaw, dpitch)

    # Trata pitch como latitude, yaw como longitude.
    lat1 = np.deg2rad(pitch[:-1])
    lat2 = np.deg2rad(pitch[1:])
    dlon = np.deg2rad(dyaw)

    num = np.hypot(
        np.cos(lat2) * np.sin(dlon),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )
    den = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.rad2deg(np.arctan2(num, den))


def angular_speed(
    yaw: np.ndarray,
    pitch: np.ndarray,
    dt: float | np.ndarray,
    *,
    great_circle: bool = True,
) -> np.ndarray:
    """Velocidade angular em graus/segundo. Tamanho len(yaw) - 1."""
    step = angular_step(yaw, pitch, great_circle=great_circle)
    dt = np.asarray(dt, dtype=np.float64)
    if dt.ndim == 0:
        return step / float(dt)
    if len(dt) == len(step) + 1:
        dt = np.diff(dt)  # foi passado um vetor de tempo, não de deltas
    return step / dt


def angular_accel(speed: np.ndarray, dt: float | np.ndarray) -> np.ndarray:
    """Aceleração angular em graus/s^2. Tamanho len(speed) - 1."""
    speed = np.asarray(speed, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    d = np.diff(speed)
    return d / (float(dt) if dt.ndim == 0 else dt[: len(d)])


def cumulative_path(yaw: np.ndarray, pitch: np.ndarray) -> float:
    """Comprimento total do caminho percorrido pela mira, em graus."""
    return float(np.sum(angular_step(yaw, pitch)))


def net_displacement(yaw: np.ndarray, pitch: np.ndarray) -> float:
    """Distância angular direta entre o primeiro e o último ponto, em graus."""
    return float(
        angular_step(
            np.array([yaw[0], yaw[-1]]),
            np.array([pitch[0], pitch[-1]]),
        )[0]
    )
