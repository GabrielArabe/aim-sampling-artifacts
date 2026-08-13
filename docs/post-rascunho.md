# Três resultados negativos e um limite estrutural ao tentar medir mira humana em CS2

*Rascunho. Números de n=3 demos de matchmaking; ver "Limitações" no fim.*

Comecei querendo caracterizar a assinatura cinemática da mira humana em
Counter-Strike 2 — a distribuição da velocidade angular, do overshoot e do
tempo de acomodação nos ~750 ms antes de um kill. A ideia era: antes de
dizer o que é anômalo, medir o que é normal.

Não cheguei a um detector, e este texto não propõe um. Cheguei a três
resultados negativos e a um limite que não é resultado de medição nenhuma —
é estrutural, e vale para qualquer um que tente. Acho que valem mais do que
a baseline valeria, porque os três primeiros invalidam coisas que a
literatura amadora de detecção de cheat faz rotineiramente sem verificar.

Tudo é reprodutível: código aberto, e cada afirmação abaixo tem teste.

---

## Resultado 1 — metade das features media a grade de amostragem, não a mira

Matchmaking do CS2 roda a 64 tick. Torneio e FACEIT rodam a 128. Se você
quer comparar as duas populações — e a comparação óbvia é "profissional é a
melhor amostra disponível de mira humana limpa" — precisa de features que
não mudem só porque a taxa de amostragem mudou.

Testei decimando as mesmas janelas de 64 para 32 Hz. Mesma trajetória,
metade das amostras — 263 janelas pareadas, cada uma terminando num kill:

| Feature | mediana @64 Hz | mediana @32 Hz | razão |
|---|---|---|---|
| `snap_fraction` | 0,129 | 0,218 | **1,69x** |
| `n_peaks` | 7,00 | 4,00 | **0,57x** |
| `peak_speed_dps` | 91,02 | 76,07 | 0,84x |
| `path_ratio` | 1,355 | 1,339 | 0,99x |
| `net_disp_deg` | 10,135 | 10,103 | 1,00x |

A coluna de razão é a razão das duas colunas anteriores — divida e confira.
Existe um estimador melhor para isso: a mediana das razões janela a janela,
em que cada janela é seu próprio controle e a assimetria da distribuição não
entra. Ele dá 1,74 / 0,56 / 0,86 / 0,99 / 1,00 sobre as mesmas 263 janelas.
Mesma conclusão, e nada aqui depende da escolha — mas se você for comparar
com outro trabalho, confira qual dos dois ele usou, porque em distribuição
assimétrica os dois divergem.

`snap_fraction` — "maior passo único dividido pelo deslocamento", uma das
features mais usadas em análise amadora de aimbot — **não tem limite quando
dt → 0**. Para trajetória suave, o maior passo é ~`v_max·dt`, então a
feature é O(dt) e tende a zero conforme a taxa sobe. Não existe quantidade
em tempo contínuo que ela esteja estimando: ela mede a grade.

Consequência prática: um demo 64-tick e um 128-tick dão `snap_fraction`
sistematicamente diferentes **para mira idêntica**, e o viés se alinha
exatamente com a comparação matchmaking × profissional. Acertar o `dt` não
resolve — `dt` correto dá a unidade certa, não cria um limite que não existe.

`n_peaks` é pior de um jeito mais bobo: a contagem acompanha a taxa quase
1:1. A mediana de 7 picos numa janela de 750 ms é um a cada 100 ms, e
correção sob feedback visual tem latência de 100–200 ms. Estava contando
quantização de sensor, não submovimento.

Adotei um critério de admissão: **feature nova só entra se tiver limite bem
definido quando dt → 0 e já estiver perto dele nas taxas em uso.** Sobram
`path_ratio`, `net_disp_deg`, `mean_speed_dps`, `path_deg`.

Um cuidado, porque a explicação intuitiva está errada: `path_ratio` **não**
é invariante por ser razão. A discretização encurta o caminho (numerador) e
não mexe na distância líquida (denominador) — a razão não se protege
sozinha. Ela sobrevive porque o comprimento de caminho da mira humana já
convergiu a 32 Hz. É fato empírico sobre o espectro do movimento, não
garantia matemática, e vale em toda a faixa de deslocamento (0,982x abaixo
de 2°, 0,995x acima de 45°).

---

## Interlúdio — o que sobrou, e por que ainda dá para confiar no resto

Se metade das features media a grade, é justo perguntar se sobra alguma
medida de movimento humano de verdade. Sobra uma, e é a mais forte das que
eu tinha.

O modelo de dois componentes do controle motor (Woodworth, 1899; lei de
Fitts) prevê que movimento dirigido a alvo tem fase balística rápida que
erra sistematicamente, seguida de correção sob feedback visual. A marca mais
direta disso é que o caminho percorrido excede a distância líquida. Nas 88
janelas de um demo, `path_ratio > 1` em **100%** delas, mediana 1,35.

Cem por cento é o número que mais tranquiliza, porque `path_ratio < 1` é
geometricamente impossível: se o unwrap de yaw ou a distância de grande
círculo estivessem errados, apareceriam violações. Não apareceu nenhuma.

Há também estrutura que não é circular: movimentos grandes são
proporcionalmente mais retos (Spearman entre deslocamento e `path_ratio` =
**−0,347**, ambas features admissíveis). Faz sentido — a fase corretiva tem
tamanho aproximadamente fixo, então pesa mais numa correção curta.

**E o que eu não posso mais afirmar.** O modelo prevê outras duas marcas:
múltiplos submovimentos, e uma fase de acomodação antes do disparo. Eu tinha
as duas confirmadas — ≥2 picos em 94,3% das janelas, acomodação não nula em
98,9%. Só que a primeira foi medida com `n_peaks` e a segunda com
`settle_ms`, que são exatamente as duas features que o Resultado 1 acabou de
aposentar. Uma predição confirmada por um instrumento que mede a grade não
está confirmada.

Deixo isso à vista porque é o padrão de erro: a validação parecia
tripla e era simples. Duas das três "confirmações" mediam o relógio da
amostragem, não a mão de ninguém.

---

## Resultado 2 — o gerador sintético mudava de física com a taxa de amostragem

Eu tinha um gerador de trajetórias sintéticas para testar o pipeline sem
depender de demo. O tremor era assim:

```python
traj += rng.normal(0, tremor_deg, n).cumsum() * 0.3
```

Um passeio aleatório em unidades de **amostra**. Variância proporcional a
n. Dobrar a taxa de amostragem dobrava a excursão acumulada do tremor, em
vez de resolver melhor a mesma curva.

Ou seja: não existia "a mesma trajetória em duas grades". Eram dois
processos estocásticos diferentes. Qualquer conclusão sobre tickrate tirada
desse gerador era sobre o gerador.

Os testes passavam. Todos. Eles verificavam separação humano/aimbot a 64 Hz,
o que continua válido — mas nunca verificaram que valesse em outra taxa, e
era exatamente isso que eu precisava.

Havia também erro de calibração: o tremor era banda larga e muito mais forte
que mira real. Decimando 64→32, o gerador antigo degradava `path_ratio` até
0,72x, onde mira real dá 0,982x. Um detector ajustado contra ele estaria
ajustado contra o ruído errado.

Ao corrigir, apareceu um segundo caso da mesma classe: os pulsos corretivos
sorteavam instante em `uniform(t[0], t[-1])`, e `t[-1]` depende do
truncamento de `int(duration_s/dt)` — 0,5859 s a 128 Hz contra 0,5781 s a
64 Hz. Dependência de grade pela porta dos fundos.

A lição não é "cuidado com bug". É que **gerador sintético é hipótese
disfarçada de dado**, e um teste verde contra um gerador errado é
indistinguível de um teste verde contra um gerador certo.

---

## Resultado 3 — contaminação come a cauda por construção

Para saber se um limiar produz falso positivo, você precisa da cauda da
distribuição limpa. A tentação é medir isso em matchmaking, onde há volume.

Não funciona, e não é questão de amostra. Contaminação a taxa ε corrompe os
quantis **acima de 1−ε**. Com ~2% de cheaters, tudo acima do percentil 98 de
uma amostra não verificada é inutilizável — os cheaters *são* a cauda que se
quer medir. Nenhum N conserta: o quantil observado converge para o valor
contaminado.

Abaixo de p98, o corpo da distribuição está limpo e é mensurável.

As duas saídas óbvias falham de formas simétricas:

- **Fonte verificada** (LAN, profissional): compra pureza ao custo de
  validade populacional. Profissional é o extremo superior de habilidade;
  uma baseline tirada dali não dá o FPR de matchmaking, que é quem você
  rastrearia. "Conta antiga sem ban" é verificação fraca — ausência de ban
  ≠ limpo é a premissa do problema.
- **Método robusto**: robustez funciona *descontando a cauda*, que é
  exatamente o que se quer medir. Estimador aparado remove cheater e cauda
  limpa sem distinguir.

A saída que sobra é ajustar o corpo, extrapolar a cauda com modelo explícito
(valores extremos), e tratar a diferença entre cauda observada e extrapolada
como o estimando — o que converte contaminação de estorvo em sinal. Com a
ressalva obrigatória: **FPR a 1e-4 passa a ser dominado pelo parâmetro de
forma do modelo, não pelo dado.** É legítimo publicar; é ilegítimo publicar
sem o rótulo de "saída de modelo".

### E há menos sinal individual do que parece

Decompondo a variância de `path_ratio` em partida / jogador / janela, sobre
28 jogadores em 3 partidas:

| | partida | jogador | janela |
|---|---|---|---|
| `path_ratio` | 0,0% | 1,9% | 98,1% |
| `snap_fraction` | 0,0% | 8,8% | 91,2% |
| `settle_ms` | 0,9% | 0,0% | 99,1% |

O componente de partida é indistinguível de zero (negativo antes de truncar)
— boa notícia: sem drift entre partidas, um limiar não se desloca. Mas o
componente de **jogador** é 2–9%, e os intervalos de confiança dos três
incluem zero. Mais de 90% da variância é janela a janela dentro do mesmo
jogador.

Com ~9 janelas por jogador numa partida, **mais da metade da dispersão
aparente entre jogadores é ruído de medida** (84% no caso do `path_ratio`).
Qualquer tabela que ordene jogadores por essas features está ordenando
principalmente sorteio amostral.

---

## O limite estrutural — o tamanho do efeito é escolha do adversário

Este não é resultado de medição nenhuma, e é por isso que não numerei junto
com os outros três. É uma propriedade do problema, não do meu dado, e não
some com mais coleta.

**Não existe um "tamanho de efeito verdadeiro" a estimar.** Cheat não é
fenômeno natural com um parâmetro fixo esperando medição — é software
escrito por gente que otimiza contra detecção. O que existe é uma
distribuição sobre implementações, e a ponta sofisticada dela é
deliberadamente escolhida para ser pequena.

Em `path_ratio`:

- aimbot ingênuo → ≈1,0 contra ≈1,35 humano. **5,3 desvios-padrão** entre
  jogadores. Trivial.
- cheat humanizado → calibrado para bater com o humano. Efeito ≈ 0 **por
  construção**, porque é para isso que a humanização existe.

Três consequências:

1. **Todo limite superior é datado.** Dá para limitar o efeito dos cheats
   implantados hoje; não os de amanhã. O prazo de validade é o ciclo de
   desenvolvimento do adversário.
2. **Publicar limiar é publicar a especificação de como escapar dele.**
   Publicar a *baseline* é seguro — é descrição de movimento humano, e
   humano não se adapta para deixar de parecer humano. Publicar o limiar
   não é. É por isso que este texto não traz nenhum.
3. **Sensibilidade apodrece; especificidade não.** O TPR de qualquer
   detector decai conforme os cheats se adaptam. O FPR não, porque a
   distribuição humana é estável.

A consequência (3) é o argumento mais forte que encontrei a favor de fazer
só a baseline: o lado durável do problema é exatamente o que ela mede.

---

## O que sobra: um método reutilizável

Se a maior parte do resultado é negativa, o que fica de utilizável?

**Uma bateria de aceitação de grade de amostragem.** Todo o critério de
admissibilidade acima pressupõe conhecer a grade. Dado de terceiro pode ter
sido reamostrado, interpolado ou suavizado em qualquer ponto do pipeline,
normalmente sem documentar — e dado suavizado parece dado limpo. Quatro
testes — A com duas medidas —, do mais forte ao mais circunstancial:

- **A1 — menor passo da rede de quantização.** O ângulo do Source 2 vive numa
  rede fina: menor passo entre valores distintos = 0,000335693°, idêntico
  em três demos e em yaw e pitch. Qualquer média de pontos da rede a refina:
  interpolação em ponto médio leva a rede/2, média móvel de 3 a rede/3.
- **A2 — razão de dobra.** A fração de valores que cai numa rede 360/2^k
  dobra a cada bit de k. Detecta valores fora de qualquer rede diádica.
- **B — autocorrelação lag-1 do passo angular.** Filtro injeta memória.
  Teste fraco: mira humana já é autocorrelacionada (mediana 0,40–0,51 entre
  jogadores, com dispersão individual de 0,27 a 0,77).
- **C — completude de tick.** Espaçamento constante fora de fronteira de
  round. Pega buraco irregular, e só isso: reamostragem *uniforme* passa
  ileso, porque 1 tick em cada 2 continua sendo espaçamento constante. Quem
  pega esse caso é D, mais a comparação entre tickrate inferido e declarado.
- **D — assinatura de decimação.** Decimar por 2 e conferir se as razões
  batem com as medidas em dado cru. Usa o próprio trabalho de invariância
  como instrumento.

O desenho de A veio de um teste negativo que me corrigiu: **A2 sozinha não
pega interpolação em ponto médio**, porque `(a+b)/2` continua numa rede
diádica, só que duas vezes mais fina, e a razão segue 2,0. Foi preciso A1.

Uma bateria que só sabe dizer "ok" não serve para julgar fonte de terceiro —
precisa de poder de rejeição demonstrado contra a corrupção específica que
se teme. E aprovar significa *compatibilidade* com dado cru, não prova de
crueza.

---

## O que seria preciso para ir adiante

Sem rótulo, dá para estimar o eixo da especificidade inteiro e nada do eixo
da sensibilidade. Existe um caminho que usa a contaminação como sinal —
procurar componente de mistura na cauda de uma amostra grande não rotulada —
mas o custo depende de ε, que ninguém sabe.

Para 80% de poder contra um aimbot grosseiro, com ~3 partidas por jogador:

| ε | jogadores | partidas |
|---|---|---|
| 0,5% | 5.209 | **1.480** |
| 1% | 1.468 | 417 |
| 2% | 451 | 128 |
| 3% | 240 | **68** |

**22x de variação** sobre uma faixa de ε inteiramente plausível. Não dá para
orçar isso de antemão; a coleta tem que ser sequencial.

---

## Limitações

Três demos, mesmo mapa, mesmo tickrate, mesma faixa de habilidade, um
jogador em comum. É a amostra mais estreita possível — "variância de partida
≈ 0" significa "≈0 entre três partidas muito parecidas", não que mapa,
tickrate ou tier não importem, e com 2 graus de liberdade um efeito moderado
passaria despercebido. Nenhum demo 128-tick foi testado de ponta a ponta.
Nenhum cheater confirmado existe na amostra, então nenhum número aqui tem
taxa de falso positivo medida.

Nada neste texto identifica jogador, e não deve ser usado para isso.

---

*Código e testes: [link do repositório]. Todo número tem teste; a bateria de
aceitação e suas bandas são abertas.*
