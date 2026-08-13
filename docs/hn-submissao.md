# Submissão ao Hacker News

## Título

O título faz quase todo o trabalho aqui, e a decisão é liderar pela
afirmação transferível, não por CS2. Quem clicaria em "detecção de cheat em
CS2" é o público que vai achatar o texto em "cheat é indetectável"; quem
clica em artefato de amostragem é o público que o texto serve.

Opções, em ordem de preferência:

1. **Half my behavioral features were measuring the sampling rate, not the behavior**
   Concreto, primeira pessoa, admite erro. Não menciona jogo nenhum, o que é
   a intenção. É o mais forte.

2. **A feature with no limit as dt→0 is measuring your clock, not your user**
   Mais afiado e mais estreito. Bom se você quiser filtrar para leitor
   técnico, ruim porque `dt→0` no título afasta quem não é da área.

3. **How to tell if your data vendor resampled your telemetry**
   Lidera pela bateria de auditoria em vez do resultado negativo. Público
   maior (qualquer um que compre dado), mas entrega o item menos original.

Evite: qualquer título com "cheat", "aimbot" ou "Counter-Strike". Eles
sequestram a discussão.

## Primeiro comentário do autor

No HN, o comentário do autor logo após a submissão define o enquadramento.
Deve dar contexto e delimitar o escopo antes que alguém o faça por você.

---

Author here. Some context on what this is and is not.

I set out to build a baseline of human aim kinematics in CS2, on the premise
that you cannot say what is anomalous before measuring what is normal. I did
not get a baseline worth publishing. I got four reasons why the obvious
approach does not work, and those turned out to be the transferable part.

The one I would flag for people outside gaming: if you compute behavioral
features from an event stream whose sampling rate you do not control, check
that each feature converges as the sampling interval goes to zero. Several
standard ones do not. "Largest single step divided by total displacement" is
O(dt), so it tends to zero as you sample faster, which means it has no
continuous-time limit at all: it is a property of your sampling grid, not of
the person. Mouse-movement bot detection on the web has exactly this problem,
because event rate varies with the client's hardware and browser. So does
keystroke dynamics, and eye tracking across 60/120/250 Hz devices.

Two things I want to be upfront about. The sample is three matches on one
map, so the specific numbers are illustrative and nobody should cite them as
constants. And there is no detector at the end: I have no labelled cheating
data, so no false positive rate was measured, and I deliberately publish no
threshold. Publishing a baseline is a description of how humans move.
Publishing a threshold is a specification for evading it.

Happy to answer questions about the audit battery in particular. The
quantisation-lattice test generalises to any quantised sensor stream and it
is the part I would reuse.

---

## Mecânica

- Submeta o link do post, não do repositório. O repositório entra no
  primeiro comentário e no rodapé do post.
- Poste o comentário do autor imediatamente após submeter.
- Melhor janela: manhã de dia útil no horário do Pacífico, mais ou menos
  entre 13h e 16h em Brasília.
- Não peça upvote a ninguém. É a única coisa que o HN pune de verdade.
- Se afundar sem comentários, isso é o normal e não diz nada sobre o texto.
  Ressubmeter uma vez, semanas depois, é aceito.

## O comentário que vai aparecer, e a resposta

Alguém vai dizer alguma variação de "então é impossível pegar cheater".
Vale ter a resposta pronta:

> Not what I claim. I claim that the specific features I tried are not
> usable across sampling rates, and that with three matches and no labels I
> cannot measure a false positive rate. Anti-cheat systems that work do not
> rely on this kind of trajectory statistic in isolation, and they have
> labelled data I do not have.
