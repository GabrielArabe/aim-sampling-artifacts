"""
Invariância de grade: quais features sobrevivem a uma mudança de tickrate.

CRITÉRIO DE PROJETO (ver README): feature nova só entra no estudo se tiver
LIMITE BEM DEFINIDO quando dt -> 0, e já estiver perto desse limite nas
taxas em uso. Este arquivo é a forma executável da regra: ADMISSIVEIS e
REPROVADAS classificam cada campo de AimWindow, e
`test_toda_feature_esta_classificada` quebra se alguém adicionar uma
feature sem decidir de que lado ela cai.

Por que importa: matchmaking CS2 é 64 tick, torneio e FACEIT são 128. Uma
feature sem limite contínuo produz diferença sistemática entre essas duas
populações PARA MIRA IDÊNTICA -- e o viés se alinha exatamente com a
comparação matchmaking vs. profissional que o README propõe. Acertar o `dt`
não resolve: `dt` correto dá a unidade certa, não cria um limite que não
existe.

Há DOIS modos de falha distintos, e a distinção importa para classificar
feature nova:

  (a) ESTRUTURAL -- o estimador é definido em termos do espaçamento da
      grade e não converge. `snap_fraction` = maior passo único / líquido:
      para trajetória suave o maior passo é ~v_max*dt, então snap_fraction
      é O(dt) e tende a ZERO conforme a taxa sobe. Não existe valor
      contínuo que ela esteja estimando. Falha mesmo com sinal perfeitamente
      suave e limitado em banda.
  (b) DE BANDA -- o estimador tem limite, mas o sinal real tem energia
      acima de Nyquist e por isso ainda não convergiu nas taxas em uso.
      `peak_speed_dps` e `n_peaks` são assim: numa sintética limitada a 8 Hz
      eles são invariantes; em mira real, não.

(a) é insalvável. (b) é quantificável e em princípio corrigível, mas exige
saber o espectro do movimento real -- que ninguém mediu.

Medido nas 263 janelas reais dos três demos, decimando 64 -> 32 Hz
(mesmas janelas, mesma trajetória, metade das amostras):

    feature            p50 @64Hz   p50 @32Hz    razão   modo de falha
    snap_fraction          0.129       0.218     1.69x   estrutural
    n_peaks                7.000       4.000     0.57x   de banda
    peak_speed_dps        91.018      76.068     0.84x   de banda
    path_ratio             1.355       1.339     0.99x   --
    net_disp_deg          10.135      10.103     1.00x   --

Spearman entre as duas medições da mesma janela: 0.97 a 0.996. Dentro de um
tickrate fixo a ordenação se preserva; o que quebra é comparar ENTRE taxas.

NOTA sobre path_ratio: a invariância dele NÃO vem de ser razão. A
discretização encurta o caminho (numerador) e não mexe na distância líquida
(denominador), então a razão não se protege sozinha. Ela sobrevive porque o
comprimento de caminho da mira humana já convergiu a 32 Hz -- fato empírico
sobre o espectro do movimento, não garantia matemática. Em taxa muito mais
alta pode deixar de valer, e aí é medir de novo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import extract_window  # noqa: E402

GRID_TOL = 0.05  # variação máxima tolerada ao halvar a taxa

ADMISSIVEIS = ("duration_s", "net_disp_deg", "path_deg", "path_ratio", "mean_speed_dps")
REPROVADAS = ("peak_speed_dps", "peak_accel_dps2", "snap_fraction", "n_peaks",
              "settle_ms", "settle_disp_deg")
# n_samples é contagem de amostras: depende da grade por definição e não é
# candidata a feature. Fora das duas listas de propósito.
NAO_E_FEATURE = ("n_samples",)

ESTRUTURAIS = ("snap_fraction",)


def _mj(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Jerk mínimo em TEMPO CONTÍNUO: 0 antes de t0, 1 depois de t1."""
    u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def _aim(t: np.ndarray, amplitude: float = 30.0, seed: int = 0, hf: bool = False,
         hf_scale: float = 1.0):
    """
    Trajetória de mira como função contínua do tempo.

    Precisa ser contínua, e não definida por amostra, senão o teste é
    circular: o tremor de `synthetic.human_flick` é passeio aleatório em
    unidades de AMOSTRA (`rng.normal(...).cumsum()`), cuja variância por
    segundo muda com a taxa. Amostrar isso em duas grades compara dois
    processos diferentes, não a mesma trajetória, e QUALQUER feature
    pareceria depender da grade.

    hf=False: tremor limitado a 8 Hz, bem abaixo de Nyquist de todas as
      taxas testadas. Isola o modo de falha ESTRUTURAL.
    hf=True: acrescenta energia de 15 a 45 Hz, imitando o fato de que mira
      real não é limitada em banda nessas taxas. Ativa o modo DE BANDA.
    """
    rng = np.random.default_rng(seed)
    yaw = (
        amplitude * 0.88 * _mj(t, 0.05, 0.23)      # balística, erra curto
        + amplitude * 0.15 * _mj(t, 0.27, 0.39)    # correção, passa um pouco
        - amplitude * 0.03 * _mj(t, 0.39, 0.47)    # micro-correção
    )
    pitch = -2.0 + 0.6 * _mj(t, 0.05, 0.25)
    for f, a in zip(rng.uniform(1.5, 8.0, 5), rng.uniform(0.03, 0.12, 5)):
        ph = rng.uniform(0, 2 * np.pi)
        yaw = yaw + a * np.sin(2 * np.pi * f * t + ph)
        pitch = pitch + 0.4 * a * np.sin(2 * np.pi * f * t + ph / 2)
    if hf:
        for f, a in zip(rng.uniform(15.0, 45.0, 6),
                        rng.uniform(0.02, 0.06, 6) * hf_scale):
            ph = rng.uniform(0, 2 * np.pi)
            yaw = yaw + a * np.sin(2 * np.pi * f * t + ph)
            pitch = pitch + 0.5 * a * np.sin(2 * np.pi * f * t + ph / 3)
    return yaw, pitch


def _amostra(rate_hz: float, dur: float = 0.75, **kw):
    n = int(round(dur * rate_hz))
    if n % 2 == 0:
        n += 1
    t = np.arange(n) / rate_hz
    yaw, pitch = _aim(t, **kw)
    return extract_window(yaw, pitch, 1.0 / rate_hz)


def _par(rate_hz: float = 128.0, dur: float = 0.75, **kw):
    """
    Mesma trajetória em duas grades: rate_hz e rate_hz/2.

    A grade grossa é subconjunto exato da fina (t[::2]), então as duas
    compartilham primeira e última amostra e `net_disp_deg` sai idêntico --
    checagem de que o pareamento está certo, não do fenômeno.
    """
    n = int(round(dur * rate_hz))
    if n % 2 == 0:
        n += 1
    t = np.arange(n) / rate_hz
    yaw, pitch = _aim(t, **kw)
    fina = extract_window(yaw, pitch, 1.0 / rate_hz)
    grossa = extract_window(yaw[::2], pitch[::2], 2.0 / rate_hz)
    return fina, grossa


def _razao(nome: str, **kw) -> float:
    fina, grossa = _par(**kw)
    a, b = getattr(fina, nome), getattr(grossa, nome)
    if a == 0:
        return 1.0 if b == 0 else float("inf")
    return b / a


# --------------------------------------------------------------------------
# pareamento
# --------------------------------------------------------------------------

def test_grades_compartilham_extremos():
    """Sem isto, qualquer diferença poderia ser só recorte diferente."""
    fina, grossa = _par()
    assert fina.net_disp_deg == pytest.approx(grossa.net_disp_deg, rel=1e-12)
    assert fina.duration_s == pytest.approx(grossa.duration_s, rel=1e-12)
    assert grossa.n_samples == (fina.n_samples + 1) // 2


# --------------------------------------------------------------------------
# admissíveis: têm limite e já convergiram
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nome", ADMISSIVEIS)
def test_feature_admissivel_e_invariante_a_grade(nome):
    r = _razao(nome)
    assert abs(r - 1.0) <= GRID_TOL, (
        f"{nome} mudou {100 * (r - 1):+.1f}% ao halvar a taxa"
    )


def test_path_ratio_invariante_em_varias_amplitudes_e_sementes():
    for seed in range(12):
        for amp in (8.0, 20.0, 45.0, 90.0):
            r = _razao("path_ratio", amplitude=amp, seed=seed)
            assert abs(r - 1.0) <= GRID_TOL, (seed, amp, r)


def test_path_ratio_invariante_no_dado_real_em_toda_a_faixa():
    """
    Registro do que foi MEDIDO, não do que a sintética faz.

    Razão 32Hz/64Hz de path_ratio nas 507 janelas dos três demos, por faixa
    de deslocamento líquido -- inclusive abaixo do corte de MIN_NET_DEG:

        0-2 deg  (n=120): 0.982x       10-20 deg (n= 79): 0.992x
        2-5 deg  (n=124): 0.990x       20-45 deg (n= 41): 0.994x
        5-10 deg (n=130): 0.992x        >45 deg  (n= 13): 0.995x

    Invariante em toda a faixa: a admissibilidade do path_ratio NÃO depende
    do filtro de amplitude. A leve melhora com o deslocamento (0.982 ->
    0.995) é a única dependência, e é pequena demais para importar.

    LIMITAÇÃO DA SINTÉTICA, registrada porque afeta synthetic.py inteiro:
    com hf=True este arquivo degrada path_ratio até 0.72x em amplitude 5,
    muito pior que os 0.982x reais. O tremor sintético é banda larga e forte
    demais. Mira real tem picos BREVES -- que derrubam peak_speed sem somar
    comprimento de caminho, e por isso preservam path_ratio. Enquanto
    synthetic.py não for calibrado contra espectro real, ele não serve de
    boneco de teste para nada que dependa de alta frequência.
    """
    reais = {(0, 2): 0.982, (2, 5): 0.990, (5, 10): 0.992,
             (10, 20): 0.992, (20, 45): 0.994, (45, None): 0.995}
    assert all(abs(v - 1.0) <= GRID_TOL for v in reais.values())


def test_path_ratio_converge_ao_refinar_a_grade():
    """
    O teste de admissão de verdade: o valor estabiliza conforme dt -> 0.

    Não basta duas grades concordarem -- é preciso que a sequência convirja,
    senão a concordância pode ser coincidência de duas taxas vizinhas.
    """
    vals = [_amostra(r).path_ratio for r in (64, 128, 256, 512, 1024)]
    passos = [abs(vals[i + 1] - vals[i]) / vals[i] for i in range(len(vals) - 1)]
    assert passos[-1] < passos[0], f"não está convergindo: {vals}"
    assert abs(vals[-1] - vals[-2]) / vals[-2] < 0.01, vals


# --------------------------------------------------------------------------
# reprovadas: modo ESTRUTURAL
# --------------------------------------------------------------------------

def test_snap_fraction_nao_tem_limite_continuo():
    """
    snap_fraction é O(dt): tende a ZERO conforme a taxa sobe.

    Para trajetória suave o maior passo único é ~v_max*dt, e o
    deslocamento líquido não depende da grade. Logo snap_fraction ~ dt.
    Não existe quantidade em tempo contínuo que ela estime -- é propriedade
    da amostragem, não do movimento. Este é o modo de falha insalvável: não
    há correção de tickrate possível porque não há alvo para o qual corrigir.
    """
    taxas = np.array([32.0, 64.0, 128.0, 256.0, 512.0])
    vals = np.array([_amostra(r).snap_fraction for r in taxas])
    assert np.all(np.diff(vals) < 0), f"deveria decrescer monotonamente: {vals}"
    assert vals[-1] < vals[0] / 8, f"deveria tender a zero: {vals}"
    # e o produto snap_fraction * taxa é aproximadamente constante,
    # que é a assinatura de ser O(dt):
    prod = vals * taxas
    assert prod.max() / prod.min() < 2.0, f"esperado ~O(dt): {prod}"


def test_snap_fraction_infla_mesmo_com_sinal_suave():
    """Falha ESTRUTURAL: não precisa de alta frequência para quebrar."""
    r = _razao("snap_fraction", hf=False)
    assert r > 1.4, f"esperado inflar substancialmente, veio {r:.2f}x"


@pytest.mark.parametrize("nome", ESTRUTURAIS)
def test_reprovada_estrutural_quebra_sem_alta_frequencia(nome):
    rs = [_razao(nome, amplitude=a, seed=s, hf=False)
          for a in (12.0, 40.0) for s in range(4)]
    assert all(abs(r - 1.0) > GRID_TOL for r in rs), (
        f"{nome} está marcada como falha estrutural mas foi invariante com "
        f"sinal suave: {[round(r, 3) for r in rs]}"
    )


# --------------------------------------------------------------------------
# reprovadas: modo DE BANDA
# --------------------------------------------------------------------------

# Configuração com energia acima de Nyquist da grade grossa. Serve só para
# ATIVAR o modo de falha de banda -- a amplitude do tremor aqui é maior que
# a da mira real (ver test_path_ratio_invariante_no_dado_real_em_toda_a_faixa),
# então os números destes testes não são estimativas de nada. As magnitudes
# reais estão no docstring do módulo; aqui só se afirma a DIREÇÃO.
BANDA = dict(rate_hz=64.0, hf=True, hf_scale=3.0)


def test_peak_speed_desinfla_com_conteudo_de_alta_frequencia():
    """
    Grade grossa suaviza o pico: a máxima instantânea some entre amostras.

    É falha DE BANDA, não estrutural -- peak_speed tem limite bem definido,
    só não convergiu nas taxas em uso. A contrapositiva abaixo é o que
    sustenta esse diagnóstico: com sinal limitado em banda, é invariante.
    Nos demos reais: 0.84x indo de 64 para 32 Hz.
    """
    assert _razao("peak_speed_dps", **BANDA) < 1.0 - GRID_TOL
    assert abs(_razao("peak_speed_dps", rate_hz=64.0, hf=False) - 1.0) <= GRID_TOL


def test_n_peaks_encolhe_com_conteudo_de_alta_frequencia():
    """
    n_peaks conta máximos locais da velocidade AMOSTRADA. Com energia acima
    de Nyquist, a contagem acompanha a taxa: 0.57x nos demos reais indo de
    64 para 32 Hz, quase exatamente metade. Ver nota de aposentadoria em
    features.py.
    """
    assert _razao("n_peaks", **BANDA) < 1.0 - GRID_TOL


@pytest.mark.parametrize("nome", REPROVADAS)
def test_feature_reprovada_de_fato_depende_da_grade(nome):
    """
    Espelho do teste das admissíveis. Uma feature aqui que se tornasse
    invariante seria notícia boa -- e este teste falharia, forçando revisão
    da classificação em vez de deixar a lista apodrecer.
    """
    rs = [_razao(nome, amplitude=a, seed=s, **BANDA)
          for a in (12.0, 40.0) for s in range(4)]
    assert any(abs(r - 1.0) > GRID_TOL for r in rs), (
        f"{nome} está classificada como reprovada mas se comportou como "
        f"invariante: razões {[round(r, 3) for r in rs]}"
    )


# --------------------------------------------------------------------------
# a regra
# --------------------------------------------------------------------------

def test_toda_feature_esta_classificada():
    """
    Feature nova obriga decisão explícita sobre invariância de grade.

    É o critério do README com dentes: adicionar campo em AimWindow sem
    classificá-lo aqui quebra a suíte. Sem isso a regra vira comentário, e
    comentário não impede ninguém de misturar 64 com 128 tick seis meses
    depois.
    """
    fina, _ = _par()
    campos = set(fina.to_dict())
    classificadas = set(ADMISSIVEIS) | set(REPROVADAS) | set(NAO_E_FEATURE)
    faltando = campos - classificadas
    assert not faltando, (
        f"feature(s) sem classificação de invariância de grade: {sorted(faltando)}. "
        "Rode _razao('<nome>', hf=True) e adicione a ADMISSIVEIS ou REPROVADAS."
    )
    assert not (classificadas - campos), (
        f"classificadas mas inexistentes em AimWindow: {sorted(classificadas - campos)}"
    )


def test_comparar_reprovada_entre_tickrates_inventa_diferenca():
    """
    Demonstração numérica do dano, caso 5% pareça tolerância frouxa.

    Duas populações com mira IDÊNTICA, uma a 128 Hz e outra a 64: a
    diferença aparente em snap_fraction é maior que quase qualquer efeito
    real que o estudo espera medir.
    """
    fina, grossa = _par(rate_hz=64.0)
    assert grossa.snap_fraction / fina.snap_fraction > 1.4
    # medido na mesma trajetória, path_ratio não inventa diferença nenhuma:
    assert abs(grossa.path_ratio / fina.path_ratio - 1.0) <= GRID_TOL
    # e no dado real a diferença fabricada é da mesma ordem: 1.70x em
    # snap_fraction contra 0.99x em path_ratio (ver docstring do módulo).
