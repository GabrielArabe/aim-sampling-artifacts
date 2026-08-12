import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import extract_window  # noqa: E402
from synthetic import human_flick, snap_flick  # noqa: E402

DT = 1 / 64


def test_humano_tem_caminho_maior_que_a_distancia():
    yaw, pitch = human_flick(rng=np.random.default_rng(1))
    w = extract_window(yaw, pitch, DT)
    # Limiar calibrado no gerador (~1.04), NÃO em dado humano real.
    # Em demo real esse número precisa ser remedido do zero -- é a
    # primeira coisa que o estudo de baseline tem que responder.
    assert w.path_ratio > 1.02, w.path_ratio


def test_snap_tem_caminho_quase_otimo():
    yaw, pitch = snap_flick(rng=np.random.default_rng(1))
    w = extract_window(yaw, pitch, DT)
    assert w.path_ratio == pytest.approx(1.0, abs=0.02), w.path_ratio


def test_snap_concentra_o_movimento_em_um_tick():
    yaw, pitch = snap_flick(snap_ticks=1, rng=np.random.default_rng(1))
    w = extract_window(yaw, pitch, DT)
    assert w.snap_fraction > 0.95


def test_humano_distribui_o_movimento():
    yaw, pitch = human_flick(rng=np.random.default_rng(1))
    w = extract_window(yaw, pitch, DT)
    assert w.snap_fraction < 0.35, w.snap_fraction


def test_humano_tem_tempo_de_acomodacao():
    yaw, pitch = human_flick(rng=np.random.default_rng(1))
    w = extract_window(yaw, pitch, DT)
    assert w.settle_ms > 40, w.settle_ms


def test_separacao_e_robusta_a_seed_e_amplitude():
    """Sem isso, o teste acima só prova que uma seed específica funciona."""
    for seed in range(20):
        rng = np.random.default_rng(seed)
        amp = float(rng.uniform(15, 90))
        h = extract_window(*human_flick(amplitude_deg=amp, rng=rng), DT)
        s = extract_window(*snap_flick(amplitude_deg=amp, rng=rng), DT)
        assert h.snap_fraction < s.snap_fraction, (seed, amp)
        assert h.path_ratio > s.path_ratio, (seed, amp)


def test_humanizacao_por_tremor_injetado_nao_reproduz_path_ratio():
    """
    Um cheat que injeta tremor CONSEGUE inflar o path_ratio -- tremor
    aumenta o caminho sem mudar o destino. Ou seja, path_ratio sozinho é
    falseável de forma barata. O que o tremor injetado NÃO reproduz é a
    estrutura temporal: submovimento único e ordenado. Este teste documenta
    a fraqueza, não a força.

    (Antes chamado "..._por_ruido_branco_...": o gerador injetava ruído
    branco por amostra, que era dependente de grade. Ver a nota de defeito
    corrigido em synthetic.py. O ponto do teste não mudou.)
    """
    rng = np.random.default_rng(7)
    s = extract_window(*snap_flick(humanize_deg=2.0, rng=rng), DT)
    h = extract_window(*human_flick(rng=rng), DT)
    # o tremor inflou o path_ratio do cheat acima do humano:
    assert s.path_ratio > h.path_ratio
    # mas a concentração do movimento continua entregando:
    assert s.snap_fraction > h.snap_fraction


# --------------------------------------------------------------------------
# Regressão do defeito de grade em synthetic.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("gerador", [human_flick, snap_flick])
def test_gerador_define_a_mesma_trajetoria_em_qualquer_dt(gerador):
    """
    REGRESSÃO. O gerador anterior definia tremor em unidades de AMOSTRA
    (`rng.normal(...).cumsum()`), então mudar `dt` mudava a trajetória em
    vez de reamostrar a mesma curva. Com isso, "a mesma mira em dois
    tickrates" não existia, e qualquer conclusão sobre tickrate tirada do
    gerador era sobre o gerador.

    Agora a trajetória é função contínua do tempo: amostrar a 128 Hz e
    decimar por 2 tem que dar EXATAMENTE a amostragem a 64 Hz.
    """
    kw = dict(humanize_deg=2.0) if gerador is snap_flick else {}
    y128, p128 = gerador(dt=1 / 128, rng=np.random.default_rng(3), **kw)
    y64, p64 = gerador(dt=1 / 64, rng=np.random.default_rng(3), **kw)
    n = len(y64)
    np.testing.assert_allclose(y128[::2][:n], y64, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p128[::2][:n], p64, rtol=1e-12, atol=1e-12)


def test_tremor_nao_cresce_com_a_taxa_de_amostragem():
    """
    A assinatura específica do bug antigo: passeio aleatório em unidades de
    amostra tem variância proporcional a n, então dobrar a taxa dobrava a
    excursão do tremor. Aqui a amplitude do tremor tem que ser a mesma nas
    duas taxas, porque é a mesma função do tempo.
    """
    def excursao(dt):
        yaw, _ = human_flick(amplitude_deg=0.0, dt=dt, rng=np.random.default_rng(5))
        return float(np.ptp(yaw))

    assert excursao(1 / 64) == pytest.approx(excursao(1 / 128), rel=0.05)
    assert excursao(1 / 32) == pytest.approx(excursao(1 / 128), rel=0.10)
