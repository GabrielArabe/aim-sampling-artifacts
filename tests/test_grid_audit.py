"""
Testes da bateria de aceitação de grade.

O controle POSITIVO (dado cru real passa) roda contra os três demos e está
registrado em src/grid_audit.py, na origem das bandas -- não dá para rodar
aqui porque exige os .dem no disco.

O que roda aqui é o controle NEGATIVO, que é o que de fato valida a
bateria: dado corrompido de forma conhecida tem que REPROVAR. Uma bateria
que só sabe dizer "ok" não serve para julgar fonte de terceiro; ela precisa
ter poder de rejeição demonstrado, e contra a corrupção específica que a
gente teme (reamostragem, interpolação, suavização).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_audit import (  # noqa: E402
    BANDAS,
    auditar,
    autocorr_passo,
    completude,
    quantizacao,
    relatorio,
)

# Rede do Source 2 medida nos demos reais: os valores vivem numa rede muito
# mais fina que 360/2^16, com menor passo observado 0.000335693 graus.
REDE = 360.0 / 2**20


def _quantiza(x: np.ndarray) -> np.ndarray:
    return np.round(x / REDE) * REDE


def frame_cru(n_jogadores: int = 6, n_ticks: int = 6000, seed: int = 0) -> pl.DataFrame:
    """
    Fluxo de ticks com as propriedades do dado cru: ângulo na rede fina,
    trajetória suave o bastante para a autocorrelação do passo cair na faixa
    observada, um valor por tick.
    """
    rng = np.random.default_rng(seed)
    linhas = []
    for p in range(n_jogadores):
        t = np.arange(n_ticks) / 64.0
        yaw = np.zeros(n_ticks)
        pitch = np.full(n_ticks, -2.0)
        for f, a in zip(rng.uniform(0.05, 3.0, 26), rng.uniform(0.4, 14.0, 26)):
            ph = rng.uniform(0, 2 * np.pi)
            yaw = yaw + a * np.sin(2 * np.pi * f * t + ph)
            pitch = pitch + 0.25 * a * np.sin(2 * np.pi * f * t + ph / 2)
        # Ruído por amostra calibrado para a autocorrelação do passo cair na
        # faixa observada. Movimento suave puro dá autocorr ~0.97; mira real
        # dá 0.40 a 0.51 porque tem estrutura fina que o modelo liso não tem.
        yaw = yaw + rng.normal(0, 2.0, n_ticks)
        pitch = pitch + rng.normal(0, 0.6, n_ticks)
        linhas.append(pl.DataFrame({
            "tick": np.arange(n_ticks, dtype=np.int64),
            "steamid": np.full(n_ticks, 1000 + p, dtype=np.int64),
            "yaw": _quantiza(yaw),
            "pitch": _quantiza(pitch),
        }))
    return pl.concat(linhas)


def _corrompe(frame: pl.DataFrame, modo: str) -> pl.DataFrame:
    partes = []
    for sid in frame["steamid"].unique().to_list():
        s = frame.filter(pl.col("steamid") == sid).sort("tick")
        t = s["tick"].to_numpy()
        y = s["yaw"].to_numpy()
        p = s["pitch"].to_numpy()
        if modo == "interpolado":
            # reamostra para metade e devolve ao passo original por
            # interpolação linear -- exatamente o que um pipeline faz ao
            # normalizar taxas diferentes para uma grade comum
            ti = t[::2]
            y = np.interp(t, ti, y[::2])
            p = np.interp(t, ti, p[::2])
        elif modo == "suavizado":
            k = np.ones(3) / 3.0
            y = np.convolve(y, k, mode="same")
            p = np.convolve(p, k, mode="same")
        elif modo == "reamostrado":
            t, y, p = t[::2], y[::2], p[::2]
        else:
            raise ValueError(modo)
        partes.append(pl.DataFrame({
            "tick": t.astype(np.int64),
            "steamid": np.full(len(t), sid, dtype=np.int64),
            "yaw": y, "pitch": p,
        }))
    return pl.concat(partes)


# --------------------------------------------------------------------------
# controle positivo sintético
# --------------------------------------------------------------------------

def test_frame_cru_passa_em_A_B_C():
    f = frame_cru()
    menor, razao, _ = quantizacao(f, "yaw")
    lo, hi = BANDAS["A_menor_passo"]
    assert lo <= menor <= hi, menor
    lo, hi = BANDAS["A_razao_dobra"]
    assert lo <= razao <= hi, razao

    med, _ = autocorr_passo(f)
    lo, hi = BANDAS["B_autocorr_mediana"]
    assert lo <= med <= hi, med

    frac, passo, _, _ = completude(f)
    assert passo == 1
    assert frac >= BANDAS["C_frac_passo_igual"][0]


# --------------------------------------------------------------------------
# controle negativo -- é isto que valida a bateria
# --------------------------------------------------------------------------

def test_A_reprova_dado_interpolado():
    """
    O teste mais forte da bateria. Interpolação linear coloca os valores
    entre pontos da rede: a média de dois pontos da rede quase nunca está na
    rede, então a razão de dobra desaba.
    """
    menor, razao, fracoes = quantizacao(_corrompe(frame_cru(), "interpolado"), "yaw")
    lo, hi = BANDAS["A_menor_passo"]
    assert not (lo <= menor <= hi), (
        f"interpolação passou despercebida: menor passo {menor:.9f}"
    )
    # e A2 sozinha NÃO pega este caso -- a média de dois pontos da rede
    # continua numa rede diádica, só que duas vezes mais fina. É por isso
    # que A1 existe.
    lo2, hi2 = BANDAS["A_razao_dobra"]
    assert lo2 <= razao <= hi2, (
        f"A2 passou a detectar interpolação (razão {razao:.3f}); se isso "
        "virou verdade, o comentário sobre a fraqueza de A2 está obsoleto"
    )


def test_A_reprova_dado_suavizado():
    menor, razao, fracoes = quantizacao(_corrompe(frame_cru(), "suavizado"), "yaw")
    lo, hi = BANDAS["A_menor_passo"]
    assert not (lo <= menor <= hi), (menor, razao, fracoes)


def test_A_aprova_reamostragem_pura_e_isso_e_esperado():
    """
    Reamostragem UNIFORME não mexe nos valores, só descarta linhas. A rede
    continua intacta e A passa -- corretamente, porque A testa o valor, não
    o espaçamento. Documenta o limite do teste: quem pega esse caso é D e a
    comparação de tickrate, não A.
    """
    menor, razao, _ = quantizacao(_corrompe(frame_cru(), "reamostrado"), "yaw")
    assert BANDAS["A_menor_passo"][0] <= menor <= BANDAS["A_menor_passo"][1]
    assert BANDAS["A_razao_dobra"][0] <= razao <= BANDAS["A_razao_dobra"][1]


def test_B_sobe_com_suavizacao():
    """
    Direcional, que é o que este teste sustenta. Filtro injeta memória no
    passo. A magnitude depende do filtro, então a asserção é sobre a
    direção, não sobre um valor.
    """
    base, _ = autocorr_passo(frame_cru())
    suav, _ = autocorr_passo(_corrompe(frame_cru(), "suavizado"))
    assert suav > base + 0.1, (base, suav)


def test_C_nao_pega_reamostragem_uniforme():
    """
    Limite conhecido e documentado: descartar 1 tick em cada 2 mantém o
    espaçamento constante, só que igual a 2. C mede regularidade, não
    completude absoluta -- não há como saber, só olhando a coluna de tick,
    se o passo 2 é o passo nativo da fonte.
    """
    frac, passo, _, _ = completude(_corrompe(frame_cru(), "reamostrado"))
    assert passo == 2
    assert frac >= BANDAS["C_frac_passo_igual"][0]


def test_C_pega_buraco_irregular():
    f = frame_cru()
    f = f.filter(~pl.col("tick").is_in(list(range(1000, 1400, 3))))
    frac, passo, buracos, _ = completude(f)
    assert passo == 1
    assert frac < BANDAS["C_frac_passo_igual"][0]
    assert buracos > 100


# --------------------------------------------------------------------------
# integração
# --------------------------------------------------------------------------

def test_auditar_exige_as_colunas():
    with pytest.raises(KeyError, match="colunas ausentes"):
        auditar(frame_cru().drop("pitch"), 64.0)


def test_auditar_pula_D_fora_de_64hz():
    """
    Não há demo 128-tick no controle positivo, então a banda de D não existe
    para 128. O certo é pular e dizer que pulou -- adaptar a banda no olho
    seria inventar um controle que não foi medido.
    """
    checks = auditar(frame_cru(n_ticks=3000), 128.0)
    d = [c for c in checks if c.nome.startswith("D decima")]
    assert len(d) == 1 and d[0].status == "PULADO"
    assert "128" in d[0].nome


def test_auditar_roda_D_em_64hz():
    checks = auditar(frame_cru(n_ticks=3000), 64.0)
    d = [c for c in checks if c.nome.startswith("D decimação ")]
    assert len(d) == 4


def test_relatorio_diz_reprovado_quando_ha_falha():
    checks = auditar(_corrompe(frame_cru(), "interpolado"), 64.0)
    txt = relatorio(checks, "teste")
    assert "REPROVADO" in txt
    assert any(c.status == "FALHA" for c in checks)


def test_relatorio_nao_promete_prova_de_crueza():
    """
    Aprovar na bateria é compatibilidade com dado cru, não prova de que é
    cru. O texto tem que dizer isso, porque é a diferença entre uma
    verificação e um carimbo.

    Monta os Check à mão em vez de auditar um frame: `frame_cru` é calibrado
    para A/B/C e NÃO para a assinatura D, que depende da estrutura fina da
    mira real. O controle positivo de D são os três demos, registrado nas
    bandas de grid_audit.py.
    """
    from grid_audit import Check

    txt = relatorio([Check("teste", 1.0, (0.5, 1.5))], "controle")
    assert "não é o mesmo que provar" in txt
    assert "REPROVADO" not in txt
