# cs2-aim-forensics

Caracterização da assinatura cinemática da mira humana em Counter-Strike 2,
a partir de demos GOTV.

**Escopo desta fase: baseline. Não há classificador aqui, e é de propósito.**
Antes de dizer o que é anômalo é preciso medir o que é normal. Pular essa
etapa é o motivo de a maioria dos "detectores de cheat" amadores não
sobreviver ao contato com dado real.

## Pergunta

Como é a distribuição da velocidade angular, do overshoot e do tempo de
acomodação da mira nos ~750 ms anteriores a um kill, em jogo profissional?

Jogo profissional é a melhor amostra disponível de "mira humana de altíssimo
nível, sob escrutínio máximo, quase certamente limpa". É o limite superior do
que a mão humana faz. Qualquer detector precisa colocar esses jogadores
confortavelmente do lado limpo -- se um s1mple dispara o alarme, o detector
está errado, não o s1mple.

## Limitação estrutural: o tamanho do efeito é escolha do adversário

Esta é a limitação mais séria do projeto, e não é uma questão de amostra,
de método ou de orçamento. Não some com mais dado.

**Não existe um "tamanho de efeito verdadeiro" a estimar.** Cheat não é
fenômeno natural com um parâmetro fixo esperando medição — é software
escrito por gente que otimiza contra detecção. O que existe é uma
distribuição sobre implementações, e a ponta sofisticada dela é
*deliberadamente escolhida* para ser pequena.

Concretamente, na única feature que sobrevive ao critério de grade:

- aimbot ingênuo → `path_ratio` ≈ 1,0 contra ≈ 1,35 humano. Efeito enorme,
  ~5,3 desvios-padrão entre jogadores. Trivial de detectar.
- cheat "humanizado" → `path_ratio` calibrado para bater com o humano.
  Efeito ≈ 0 **por construção**, porque é exatamente para isso que a
  humanização existe.

Três consequências que mudam o que este repositório pode prometer:

1. **Limite superior é datado.** O argumento de mistura (ver abaixo) limita
   o efeito dos cheats *implantados hoje*. Não limita os de amanhã, e o
   prazo de validade é o ciclo de desenvolvimento do adversário, não o do
   estudo.
2. **Publicar limiar é publicar a especificação de como escapar dele.**
   Publicar a *baseline* é seguro: é descrição de movimento humano, e
   humano não se adapta para deixar de parecer humano. Publicar o limiar
   não é.
3. **Sensibilidade apodrece, especificidade não.** O TPR de qualquer
   detector decai conforme os cheats se adaptam. O FPR não, porque a
   distribuição humana é estável. A assimetria é permanente.

A consequência (3) é o argumento mais forte a favor do escopo atual: o lado
durável do problema é justamente o que a baseline mede. Um detector
publicado hoje tem sensibilidade perecível; uma caracterização de mira
humana, não.

## Medição vs. saída de modelo

**Regra: todo número publicado tem que declarar de qual dos dois lados vem.**

Contaminação a taxa ε corrompe os quantis acima de 1−ε. Com ~2% de
prevalência em matchmaking, tudo **acima do percentil 98** de uma amostra
não verificada é inutilizável, e nenhum N conserta — os cheaters *são* a
cauda que se quer medir. Abaixo disso o corpo da distribuição está limpo.

Portanto:

| Faixa | Origem | Estatuto |
|---|---|---|
| até ~p98 | quantil empírico | **medição** |
| acima de ~p98 | extrapolação de cauda (GPD/valores extremos) | **saída de modelo** |

FPR a 1e-3 ou 1e-4 é dominado pelo parâmetro de forma da GPD, não pelo
dado. É legítimo publicar, é ilegítimo publicar sem o rótulo. `rank.py` só
emite quantis empíricos e IC por estatística de ordem, ou seja só medição;
qualquer código futuro de extrapolação de cauda tem que marcar a saída.

## Estado

| Componente | Status |
|---|---|
| `src/kinematics.py` | testado (7 testes) |
| `src/features.py` | testado via sintético (7 testes) |
| `src/synthetic.py` | testado |
| `src/ingest.py` | testado (27 testes) + rodado contra 1 demo de MM 64-tick |
| `src/rank.py` | testado (19 testes) |

`ingest.py` deixou de ser código não exercitado, mas "roda em 1 demo de
matchmaking" não é "roda em demo de torneio": ver armadilha 6.

## Instalação

```bash
pip install awpy polars numpy pytest
python -m pytest tests/ -q
```

Requer Python >= 3.11 (exigência do awpy).

## Uso

```bash
# 1. sempre comece por aqui, num demo só
python src/ingest.py data/algum.dem --inspect

# 2. depois o lote
python src/ingest.py data/*.dem -o out/windows.parquet

# 3. tabela descritiva por jogador de um demo (os 10, sempre)
python src/rank.py data/algum.dem
python src/rank.py data/algum.dem --sort n    # por tamanho de amostra
```

`rank.py` é descritivo e nada mais: sem pontuação, sem limiar, sem
ordenação por feature. A ordem padrão é alfabética de propósito — ordenar
dez pessoas por uma feature fabrica um primeiro lugar, e um primeiro lugar
é lido como resultado mesmo debaixo de um IC que o desmente. Cada número
sai com n e com intervalo de confiança não-paramétrico ao lado.

Demos profissionais: HLTV publica o `.rar` de cada partida na página do match.

## Features

| Feature | Hipótese | Grade |
|---|---|---|
| `path_ratio` | caminho / distância líquida. Humano erra e corrige, então > 1 | ✅ |
| `net_disp_deg` | deslocamento líquido da janela | ✅ |
| `mean_speed_dps` | velocidade angular média | ✅ |
| `snap_fraction` | maior passo único / deslocamento. Aimbot concentra em 1 tick | ⚠️ só dentro de um tickrate |
| `peak_speed_dps` | velocidade angular de pico | ⚠️ não convergiu a 64 Hz |
| `peak_accel_dps2` | aceleração de pico -- limitada por biomecânica | ⚠️ idem |
| `n_peaks` | submovimentos | ❌ **aposentada** |
| `settle_ms` | tempo entre pico de velocidade e mira estável | ❌ **aposentada** |

### Critério de admissão de feature

**Feature nova só entra se tiver limite bem definido quando `dt` → 0, e já
estiver perto desse limite nas taxas em uso.**

Isto é regra, não acidente. Matchmaking CS2 é 64 tick; torneio e FACEIT são
128. Uma feature sem limite contínuo produz diferença sistemática entre
essas duas populações **para mira idêntica** -- e o viés se alinha
exatamente com a comparação matchmaking vs. profissional que este README
propõe. Acertar o `dt` não salva: `dt` correto dá a unidade certa, não cria
um limite que não existe.

Dois modos de falha, e vale distinguir ao classificar:

- **Estrutural** -- o estimador é definido em termos do espaçamento da grade.
  `snap_fraction` é O(`dt`): o maior passo único é ~`v_max·dt`, então ela
  tende a zero conforme a taxa sobe. Insalvável, porque não há alvo contínuo
  para o qual corrigir.
- **De banda** -- o estimador tem limite, mas o sinal real tem energia acima
  de Nyquist e ainda não convergiu. `peak_speed_dps`, `n_peaks`. Em princípio
  corrigível, mas exige conhecer o espectro do movimento real, que ninguém
  mediu.

Medido nas 263 **janelas de kill** dos três demos, decimando 64 → 32 Hz, como
razão das medianas: `snap_fraction` 1,69x, `n_peaks` 0,57x, `peak_speed`
0,84x, contra `path_ratio` 0,99x e `net_disp` 1,00x. A invariância do
`path_ratio` vale em toda a faixa de deslocamento, de <2° a >45°.

Dois cuidados ao comparar estes números com os de `src/grid_audit.py`, que
parecem os mesmos e não são:

- **População de janelas.** Aqui são janelas de kill (263). Lá são janelas
  amostradas do fluxo de ticks (~4.500), que incluem períodos parados e têm
  medianas diferentes — `n_peaks`, por exemplo, dá 0,57x aqui e 0,33x lá.
- **Estimador.** Ambos são razão das medianas. O estimador pareado (mediana
  das razões janela a janela) dá 1,74 / 0,56 / 0,86 / 0,99 nas janelas de
  kill. É o estimador mais defensável — cada janela é seu próprio controle —
  e a conclusão não muda, mas os números sim.

**Provenência destes números.** Vêm de `yaw`/`pitch` reais lidos por
demoparser2 e decimados na própria janela — mesma trajetória, metade das
amostras. Nenhum passa por `src/synthetic.py`, e nenhum teste de
`test_sampling_invariance.py` importa aquele módulo: ele define sua própria
trajetória em tempo contínuo e importa só `extract_window`. Portanto o
defeito de grade que `synthetic.py` teve (documentado no cabeçalho daquele
arquivo) **não contamina esta tabela**; os valores foram reproduzidos sem
alteração depois da correção do gerador. A distinção importa porque o
argumento sobre `snap_fraction` tem duas partes com estatuto diferente: a
direção é analítica (passo único cresce se o passo cobre mais tempo, e a
feature é O(`dt`)), o fator 1,69x é empírico e vale para 64→32 Hz nesta
amostra — não é constante universal e não se extrapola para 64→128.

Cuidado com a explicação errada: `path_ratio` **não** é invariante por ser
razão. A discretização encurta o caminho (numerador) e não mexe na distância
líquida (denominador), então a razão não se protege sozinha. Ela sobrevive
porque o comprimento de caminho da mira humana já convergiu a 32 Hz -- fato
empírico sobre o espectro do movimento, não garantia matemática.

`tests/test_sampling_invariance.py` é a forma executável desta regra:
adicionar campo em `AimWindow` sem classificá-lo quebra a suíte.

Base teórica: modelo de dois componentes de Woodworth e a lei de Fitts.
Movimento humano dirigido a alvo tem uma fase balística rápida que
sistematicamente erra, seguida de correção sob feedback visual, com
latência de ~100-200 ms.

## Armadilhas já tratadas no código

1. **Wrap de yaw em ±180.** Um giro de 4° cruzando a fronteira vira um passo
   de 356° se você fizer `diff` ingênuo. Falso positivo garantido. Coberto
   por teste.
2. **Distância euclidiana em (yaw, pitch).** Não é a distância na esfera de
   visão. Infla a velocidade de quem joga em ângulo alto de pitch --
   viés sistemático correlacionado com estilo de jogo e com arma.
3. **Tickrate presumido.** O awpy assume 128. Muito demo de CS2 é 64. Errar
   isso escala toda velocidade por 2x de forma uniforme, então não parece
   bug, parece descoberta.
   Confirmado no demo de teste: `awpy.constants.DEFAULT_SERVER_TICKRATE`
   é 128, `Demo.__init__` guarda esse valor **sem nunca olhar o arquivo**, e
   o demo era 64. Agora `infer_tickrate()` mede de `game_time` com dois
   estimadores independentes (base longa e mediana de passos) que precisam
   concordar; se discordarem, levanta em vez de escolher um. `game_time`
   não está em `dem.ticks` — o awpy pede o campo, usa para `in_play_ticks`
   e reconstrói a tabela sem ele, então é preciso um `parse_ticks()` extra.
4. **Buraco de amostragem dentro da janela.** `dem.ticks` já vem filtrado
   para `in_play_ticks`: freezetime, warmup e timeout somem e deixam
   buracos na sequência de ticks (no demo de teste, um de 7319 ticks e mais
   12 pequenos). Uma janela que atravesse um buraco tem passo angular
   grande sob um `dt` pequeno — velocidade enorme, ou seja, um "snap" que é
   artefato de amostragem e cai bem na feature que o estudo mede. Janelas
   não contíguas são descartadas e contadas. Neste demo o descarte foi 0/173,
   mas 0 não é nunca.
5. **Nome como chave de junção.** As janelas casam por `steamid`, não por
   `name`. Nick não é chave: muda no meio da partida, repete entre
   jogadores, e o nick do demo de teste tem unicode fora do BMP
   (`um nick com hieróglifo 𓂀`) que nem todo caminho de I/O preserva.

## Armadilhas ainda ABERTAS

6. **Subtick.** No CS2 o tiro acontece entre ticks. O ângulo amostrado no
   tick não é o ângulo no disparo. Afeta qualquer feature no instante do
   tiro. Features de trajetória longa quase não sofrem.
7. **Ausência de rótulo.** Não há amostra de cheater confirmado aqui. Sem
   isso não existe taxa de falso positivo medida, e sem taxa de falso
   positivo medida não existe detector -- existe opinião com gráfico.
8. **Demo POV vs GOTV.** Taxas de erro de parsing diferentes. Misturar os
   dois contamina a baseline. O contrato de colunas em `ingest.py` foi
   verificado contra **um** GOTV de matchmaking; `_require_columns()` falha
   alto se outro tipo de demo trouxer nomes diferentes, o que é melhor que
   descobrir depois, mas não é o mesmo que ter testado.
9. **Viés de seleção do `MIN_NET_DEG`.** O filtro de 5° descartou **85 de
   173 kills (49%)** no demo de teste. As janelas que sobram são, por
   construção, as kills que exigiram movimento de mira — kill de espera em
   ângulo, spray e alvo que anda para a mira ficam de fora. A baseline
   resultante é a de "kills com deslocamento de mira", não a de "kills". O
   número é defensável; o que não é defensável é esquecer que ele existe ao
   comparar com qualquer distribuição publicada com outro critério.
10. **n por jogador em um demo é pequeno demais para o p90.** Entre 3 e 13
   janelas por jogador aqui. O IC 95% da mediana só existe com n >= 6 e o
   limite superior do p90 só existe com n >= 36 (estatística de ordem), de
   modo que a coluna do p90 sai com limite superior vazio para todo mundo.
   O p90 por jogador em um único demo não é uma quantidade estimável — é
   praticamente o máximo amostral com outro nome.
11. **Tickrate 128 nunca foi exercitado.** `infer_tickrate()` é genérico e
   testado com 32/64/100/128 sintéticos, mas não houve demo 128 real. Demo
   de torneio é o caso que importa e continua não testado de ponta a ponta.

## Próximo passo honesto

Rodar em 50 demos de HLTV, publicar as distribuições marginais e conjuntas,
e nada além disso. Só a baseline já é uma contribuição, e é verificável --
que é mais do que se pode dizer da maioria das afirmações públicas sobre
detecção de cheat.

## Licença / escopo ético

Ferramenta de análise de demos públicos. Não interage com o cliente do jogo,
não lê memória, não roda junto com o CS2. Um resultado aqui é uma hipótese
estatística sobre uma partida, nunca uma acusação sobre uma pessoa.
