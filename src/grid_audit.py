"""
Bateria de aceitação de grade: o dado é cru, ou passou por um pipeline?

    python src/grid_audit.py <demo.dem>
    python src/grid_audit.py <demo.dem> --calibrate    # imprime medidas sem julgar

Por que isto existe. Todo o critério de admissibilidade de feature deste
repositório (ver README) pressupõe conhecer a grade de amostragem: `dt`
constante, um valor por tick do servidor, ângulo como o cliente mandou.
Dado de terceiro pode ter sido reamostrado, interpolado ou suavizado em
qualquer ponto do pipeline dele, normalmente sem documentar. Se isso
acontecer, herdamos uma grade desconhecida e o critério cai junto --
`path_ratio` deixa de ser comparável, e nada avisa, porque dado suavizado
parece dado limpo.

Quatro testes, do mais difícil de falsificar para o mais circunstancial:

  A. REDE DE QUANTIZAÇÃO. O ângulo no Source 2 vive numa rede fina, e a
     rede é impressão digital do pipeline. Duas medidas:

       A1 MENOR PASSO entre valores distintos. É a medida forte. Nos três
          demos deu 0.000335693 graus, idêntico nos três e em yaw e pitch.
          Qualquer média de pontos da rede refina a rede e o menor passo
          cai: interpolação em ponto médio leva a rede/2, média móvel de 3
          leva a rede/3. Verificado.
       A2 RAZÃO DE DOBRA. A fração de valores que cai numa rede de 360/2^k
          dobra a cada bit de k, porque a rede real é mais fina que todas
          as testadas. Detecta valores fora de QUALQUER rede diádica.

     A2 sozinha NÃO basta, e o teste negativo mostrou isso: interpolação em
     ponto médio produz (a+b)/2, que continua numa rede diádica -- só que
     duas vezes mais fina -- e a razão continua 2.0. É A1 que pega esse
     caso. As duas juntas cobrem refino de rede e saída da rede.
  B. AUTOCORRELAÇÃO LAG-1 DO PASSO ANGULAR. Filtro de média móvel ou
     interpolação injeta memória no passo. Teste fraco: mira humana já é
     autocorrelacionada por ser suave, e a dispersão entre jogadores é
     grande. Serve para pegar suavização forte, não sutil.
  C. COMPLETUDE DE TICK. Espaçamento constante fora de fronteira de round.
     Pega buraco irregular. NÃO pega reamostragem UNIFORME: se a fonte
     entrega 1 tick em cada 2, o espaçamento continua constante e C passa
     tranquilo. Quem pega esse caso é D, mais a comparação entre o tickrate
     inferido e o declarado no cabeçalho.
  D. ASSINATURA DE DECIMAÇÃO. Decimar por 2 e medir como cada feature se
     move. Usa o próprio trabalho de invariância como instrumento: a
     assinatura é impressão digital da grade. Só vale comparando grades de
     mesmo tickrate -- ver AVISO em BANDAS.

Um controle positivo é obrigatório antes de julgar qualquer fonte: se a
bateria não passa em dado que sabemos ser cru, ela não serve para julgar
dado de terceiro. As bandas abaixo vieram desse controle.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import extract_window  # noqa: E402

WINDOW_S = 0.75
MIN_NET_DEG = 5.0
MIN_SAMPLES = 8

# --------------------------------------------------------------------------
# BANDAS DE ACEITAÇÃO
# --------------------------------------------------------------------------
# Origem: controle positivo em 3 demos de matchmaking CS2 (de_dust2, 64 tick,
# 30 jogadores, ~435 mil ticks), lidos por awpy 2.0.2 / demoparser2 e
# portanto crus por construção -- nenhum reprocessamento entre o arquivo e a
# medida. Medido em 2026-08.
#
# Cada banda declara o observado e a margem, porque n=3 demos é uma
# estimativa ruim da variação real entre partidas. A margem é deliberadamente
# generosa: o custo de reprovar dado bom é refazer a análise, o custo de
# aprovar dado processado é um estudo inteiro sobre a grade de outra pessoa.
#
# AVISO sobre D: as bandas valem para decimação 64 -> 32 Hz. Não há demo
# 128-tick nesta amostra, então para dado 128 (torneio, FACEIT) a banda de D
# é DESCONHECIDA e o teste tem que ser pulado, não adaptado no olho.
BANDAS = {
    # A1: menor passo entre valores distintos. Observado 0.000335693 nos
    # 3 demos, em yaw e pitch, sem variação. A banda cobre o valor real e
    # rejeita rede/2 (0.000168, interpolação) e rede/3 (0.000114, média
    # móvel de 3), que são as corrupções testadas em test_grid_audit.py.
    # Exige amostra grande: com poucos valores distintos pode não haver
    # dois pontos adjacentes da rede e o mínimo sai inflado.
    "A_menor_passo": (0.00025, 0.00045),
    # A2: teoria diz exatamente 2.0 -- se a rede real é mais fina que todas
    # as testadas, cada bit de refino dobra a fração que cai nela.
    # Observado: 1.9676 a 2.0115 (6 medidas, 3 demos x yaw/pitch).
    # Banda ancorada na teoria, não no observado.
    "A_razao_dobra": (1.85, 2.15),
    # B: mediana ENTRE jogadores. Observado 0.4047 / 0.4609 / 0.5089.
    # A dispersão POR JOGADOR é muito maior (0.274 a 0.770 entre os 30
    # jogadores), por isso a estatística é a mediana e a banda é larga.
    # Teste fraco de propósito: pega suavização forte, não sutil.
    "B_autocorr_mediana": (0.25, 0.70),
    # C: observado 0.9998 a 0.9999. Banda folgada porque outra fonte pode
    # legitimamente ter mais fronteiras de round ou cortes de timeout.
    "C_frac_passo_igual": (0.995, 1.0),
    # D: observado sobre 1421 a 1591 janelas por demo, amostradas do fluxo
    # de ticks (não de kills). Faixas observadas nos 3 demos:
    #   path_ratio     0.9940 a 0.9959   -> banda 0.96 a 1.02
    #   snap_fraction  1.7288 a 1.7568   -> banda 1.55 a 1.95
    #   peak_speed     0.8466 a 0.8638   -> banda 0.75 a 0.95
    #   n_peaks        0.3333 a 0.4000   -> banda 0.25 a 0.50
    # A faixa observada é estreita (path_ratio varia 0.2% entre demos), mas
    # a banda é larga: 3 demos do mesmo mapa e tier não estimam a variação
    # entre fontes diferentes.
    "D_path_ratio": (0.96, 1.02),
    "D_snap_fraction": (1.55, 1.95),
    "D_peak_speed_dps": (0.75, 0.95),
    "D_n_peaks": (0.25, 0.50),
}

TICKRATE_D_VALIDO = (64.0,)


@dataclass
class Check:
    nome: str
    medido: float | None
    banda: tuple[float, float] | None
    detalhe: str = ""

    @property
    def status(self) -> str:
        if self.medido is None:
            return "PULADO"
        if self.banda is None:
            return "MEDIDO"
        if not np.isfinite(self.medido):
            return "FALHA"
        return "ok" if self.banda[0] <= self.medido <= self.banda[1] else "FALHA"


def quantizacao(frame: pl.DataFrame, col: str, ks=(13, 14, 15, 16)
                ) -> tuple[float, float, list[float]]:
    """
    Devolve (menor_passo, razao_de_dobra, frações por k).

    menor_passo é a medida forte: é o espaçamento da rede em que os valores
    vivem. Qualquer média de pontos da rede refina a rede e derruba esse
    número -- interpolação em ponto médio dá rede/2, média móvel de 3 dá
    rede/3.

    razao_de_dobra: se os valores vivem numa rede mais fina que 360/2^k, a
    fração que cai na rede grossa é 2^-(bits de diferença), e portanto DOBRA
    a cada bit que se refina o teste. Detecta valores fora de qualquer rede
    diádica, mas NÃO detecta refino da rede -- por isso as duas medidas.
    """
    v = np.unique(np.asarray(frame[col].to_numpy(), dtype=np.float64))
    v = v[np.isfinite(v)]
    if len(v) < 1000:
        raise ValueError(f"amostra pequena demais para testar rede ({len(v)} valores)")

    d = np.diff(v)
    d = d[d > 1e-12]
    menor = float(d.min()) if len(d) else float("nan")

    fr = []
    for k in ks:
        passo = 360.0 / 2**k
        r = np.abs(np.round(v / passo) - v / passo)
        fr.append(float(np.mean(r < 1e-4)))
    if min(fr) <= 0:
        return menor, float("nan"), fr
    razoes = [fr[i + 1] / fr[i] for i in range(len(fr) - 1)]
    return menor, float(np.median(razoes)), fr


def autocorr_passo(frame: pl.DataFrame) -> tuple[float, list[float]]:
    """Mediana, entre jogadores, da autocorrelação lag-1 do passo de yaw."""
    acs = []
    for sid in frame["steamid"].unique().to_list():
        y = frame.filter(pl.col("steamid") == sid).sort("tick")["yaw"].to_numpy()
        s = np.diff(np.asarray(y, dtype=np.float64))
        s = s[np.abs(s) < 180.0]  # descarta cruzamento de fronteira
        if len(s) < 1000 or np.std(s) == 0:
            continue
        acs.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))
    if not acs:
        raise ValueError("nenhum jogador com amostra suficiente")
    return float(np.median(acs)), sorted(acs)


def completude(frame: pl.DataFrame) -> tuple[float, int, int, int]:
    """Fração de espaçamentos iguais ao passo mediano, e o maior buraco."""
    t = frame.select("tick").unique().sort("tick").to_series().to_numpy()
    if len(t) < 100:
        raise ValueError("ticks insuficientes")
    d = np.diff(t)
    passo = int(np.median(d))
    return float(np.mean(d == passo)), passo, int((d != passo).sum()), int(d.max())


def _janelas(frame: pl.DataFrame, tickrate: float, passo: int, max_por_jogador: int = 400):
    """
    Recorta janelas contíguas do fluxo de ticks.

    Deliberadamente NÃO usa kills: o esquema de eventos muda entre fontes, e
    a auditoria é da grade, não do parsing de evento. Janelas amostradas do
    fluxo também dão uma ordem de magnitude mais de dados.
    """
    n_w = int(round(WINDOW_S * tickrate))
    for sid in frame["steamid"].unique().to_list():
        s = frame.filter(pl.col("steamid") == sid).sort("tick")
        t = s["tick"].to_numpy()
        yaw = np.asarray(s["yaw"].to_numpy(), dtype=np.float64)
        pit = np.asarray(s["pitch"].to_numpy(), dtype=np.float64)
        quebras = np.flatnonzero(np.diff(t) != passo)
        inicios = np.concatenate(([0], quebras + 1))
        fins = np.concatenate((quebras + 1, [len(t)]))
        emitidas = 0
        for a, b in zip(inicios, fins):
            for i in range(a, b - n_w + 1, n_w):
                if emitidas >= max_por_jogador:
                    break
                yield yaw[i : i + n_w], pit[i : i + n_w]
                emitidas += 1


def assinatura_decimacao(frame: pl.DataFrame, tickrate: float, passo: int) -> dict:
    """Razão mediana de cada feature ao decimar por 2, sobre janelas pareadas."""
    dt = passo / tickrate
    fina, grossa = [], []
    for yaw, pit in _janelas(frame, tickrate, passo):
        if len(yaw) < MIN_SAMPLES:
            continue
        try:
            a = extract_window(yaw, pit, dt)
            b = extract_window(yaw[::2], pit[::2], 2 * dt)
        except ValueError:
            continue
        if a.net_disp_deg < MIN_NET_DEG:
            continue
        fina.append(a)
        grossa.append(b)
    if len(fina) < 30:
        raise ValueError(f"janelas insuficientes para assinatura ({len(fina)})")
    out = {"_n": len(fina)}
    for f in ("path_ratio", "snap_fraction", "peak_speed_dps", "n_peaks"):
        av = np.nanmedian([getattr(x, f) for x in fina])
        bv = np.nanmedian([getattr(x, f) for x in grossa])
        out[f] = float(bv / av) if av else float("nan")
    return out


def auditar(frame: pl.DataFrame, tickrate: float, *, julgar: bool = True) -> list[Check]:
    faltando = [c for c in ("tick", "steamid", "yaw", "pitch") if c not in frame.columns]
    if faltando:
        raise KeyError(f"colunas ausentes para auditoria: {faltando}")

    B = BANDAS if julgar else {}
    checks: list[Check] = []

    for col in ("yaw", "pitch"):
        menor, razao, fr = quantizacao(frame, col)
        checks.append(Check(
            f"A1 menor passo da rede ({col})", menor, B.get("A_menor_passo"),
            f"rede real observada nos demos: 0.000335693",
        ))
        checks.append(Check(
            f"A2 razão de dobra ({col})", razao, B.get("A_razao_dobra"),
            "frações k=13..16: " + ", ".join(f"{x:.4f}" for x in fr),
        ))

    med, acs = autocorr_passo(frame)
    checks.append(Check(
        "B autocorr lag-1 do passo", med, B.get("B_autocorr_mediana"),
        f"por jogador: {acs[0]:+.3f} a {acs[-1]:+.3f} (n={len(acs)})",
    ))

    frac, passo, buracos, maior = completude(frame)
    checks.append(Check(
        "C completude de tick", frac, B.get("C_frac_passo_igual"),
        f"passo mediano={passo}, buracos={buracos}, maior={maior}",
    ))

    if tickrate in TICKRATE_D_VALIDO:
        d = assinatura_decimacao(frame, tickrate, passo)
        for f in ("path_ratio", "snap_fraction", "peak_speed_dps", "n_peaks"):
            checks.append(Check(
                f"D decimação {f}", d[f], B.get(f"D_{f}"), f"n={d['_n']} janelas",
            ))
    else:
        checks.append(Check(
            f"D decimação (tickrate {tickrate:.0f})", None, None,
            "banda medida só para 64 Hz; para outro tickrate o controle "
            "positivo não existe e o teste é pulado em vez de adaptado",
        ))
    return checks


def relatorio(checks: list[Check], rotulo: str) -> str:
    def num(v: float) -> str:
        # precisão adaptativa: a rede de quantização vive na 6a casa e
        # some com formatação fixa
        return f"{v:.4f}" if abs(v) >= 0.01 else f"{v:.8f}"

    linhas = [f"AUDITORIA DE GRADE -- {rotulo}", "=" * 78]
    for c in checks:
        banda = f"[{num(c.banda[0])}, {num(c.banda[1])}]" if c.banda else "--"
        med = num(c.medido) if c.medido is not None and np.isfinite(c.medido) else "n/d"
        linhas.append(f"  {c.status:>6}  {c.nome:<34} {med:>9}  banda {banda:>18}")
        if c.detalhe:
            linhas.append(f"          {c.detalhe}")
    falhas = [c for c in checks if c.status == "FALHA"]
    linhas.append("=" * 78)
    if falhas:
        linhas.append(f"REPROVADO em {len(falhas)} teste(s): "
                      + ", ".join(c.nome for c in falhas))
        linhas.append("A grade desta fonte não é a grade do controle positivo. "
                      "Não use antes de entender a diferença.")
    else:
        linhas.append("Aprovado. Compatível com dado cru nos testes disponíveis --")
        linhas.append("o que não é o mesmo que provar que é cru. Ver limitações no topo.")
    return "\n".join(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auditoria de grade de amostragem.")
    ap.add_argument("demo", type=Path)
    ap.add_argument("--calibrate", action="store_true",
                    help="mede sem julgar, para fixar bandas em controle positivo")
    args = ap.parse_args()

    from ingest import demo_tickrate, load_demo

    dem = load_demo(args.demo)
    tickrate = demo_tickrate(dem)
    checks = auditar(dem.ticks, tickrate, julgar=not args.calibrate)
    print(relatorio(checks, f"{args.demo.name}  ({tickrate:.1f} Hz)"))


if __name__ == "__main__":
    main()
