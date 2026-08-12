import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinematics import (  # noqa: E402
    angular_speed,
    angular_step,
    net_displacement,
    unwrap_yaw,
    wrap_deg,
)


def test_wrap_deg_basico():
    assert wrap_deg(np.array([0.0]))[0] == pytest.approx(0.0)
    assert wrap_deg(np.array([190.0]))[0] == pytest.approx(-170.0)
    assert wrap_deg(np.array([-190.0]))[0] == pytest.approx(170.0)
    assert wrap_deg(np.array([540.0]))[0] == pytest.approx(180.0)


def test_cruzar_fronteira_180_nao_vira_flick_gigante():
    """
    O bug nº1. Jogador gira 4 graus atravessando +-180.
    Sem tratamento, isso vira um passo de 356 graus.
    """
    yaw = np.array([178.0, 179.0, 180.0, -179.0, -178.0])
    pitch = np.zeros(5)
    steps = angular_step(yaw, pitch)
    assert np.max(steps) < 1.5, f"passo espúrio detectado: {steps}"
    assert np.sum(steps) == pytest.approx(4.0, abs=1e-6)


def test_unwrap_yaw_preserva_deltas():
    yaw = np.array([170.0, 175.0, -180.0, -175.0])
    un = unwrap_yaw(yaw)
    assert np.allclose(np.diff(un), [5.0, 5.0, 5.0], atol=1e-6)


def test_great_circle_encolhe_yaw_em_pitch_alto():
    """
    10 graus de yaw a 80 graus de pitch percorrem MENOS na esfera
    de visão do que 10 graus de yaw no horizonte.
    """
    plano = angular_step(np.array([0.0, 10.0]), np.array([0.0, 0.0]))[0]
    alto = angular_step(np.array([0.0, 10.0]), np.array([80.0, 80.0]))[0]
    assert plano == pytest.approx(10.0, abs=1e-6)
    assert alto < plano * 0.25
    # e o euclidiano ingênuo erraria feio
    ingenuo = angular_step(
        np.array([0.0, 10.0]), np.array([80.0, 80.0]), great_circle=False
    )[0]
    assert ingenuo == pytest.approx(10.0, abs=1e-6)


def test_precisao_em_passos_minusculos():
    """arccos ingênuo perde precisão perto de zero; Vincenty não."""
    yaw = np.array([0.0, 1e-4])
    pitch = np.array([0.0, 0.0])
    assert angular_step(yaw, pitch)[0] == pytest.approx(1e-4, rel=1e-6)


def test_velocidade_angular_unidades():
    yaw = np.array([0.0, 10.0, 20.0])
    pitch = np.zeros(3)
    v = angular_speed(yaw, pitch, dt=1 / 64)
    assert np.allclose(v, 640.0, rtol=1e-6)


def test_net_displacement_ignora_o_caminho():
    yaw = np.array([0.0, 50.0, 10.0])
    pitch = np.zeros(3)
    assert net_displacement(yaw, pitch) == pytest.approx(10.0, abs=1e-6)
    assert np.sum(angular_step(yaw, pitch)) == pytest.approx(90.0, abs=1e-6)
