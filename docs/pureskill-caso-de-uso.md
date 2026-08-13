# Texto para a solicitação de assinatura — PureSkill.gg CS2 Gameplay (AWS Data Exchange)

Cole no campo de caso de uso. Ajuste nome/afiliação antes de enviar.
Está em inglês porque a revisão é deles.

---

**Use case: measurement of human aim kinematics in CS2 (baseline study, no
player-level output)**

I am building an open, reproducible baseline of how human aim moves in the
~750 ms before a kill in CS2 — angular path length, overshoot, and the
amplitude–velocity relationship — grounded in the two-component model of
motor control (Woodworth; Fitts). The goal is a published description of the
normal distribution of these quantities, not a cheat detector, and not any
per-player assessment.

**What I need from the dataset.** The `player_vector` channel, specifically
`phi_ang` and `theta_ang` at tick resolution, plus the `tick` and `header`
channels to establish the sampling grid and server tickrate. I do not need
economy, positional, or outcome data beyond what is required to segment
windows.

**Why this dataset.** Valve matchmaking demos are gated per account: share
codes and auth codes are per-player, so a single researcher can only obtain
their own matches. That yields a sample restricted to one skill tier and a
handful of maps, which is exactly the range restriction that makes a
baseline useless. Your dataset is the only source I have found that provides
tick-resolution view angles across many players without requiring per-player
consent collection.

**First step, before any volume.** I will pull approximately 50 matches to
run a sampling-grid audit — verifying that `phi_ang`/`theta_ang` are
per-server-tick and have not been resampled, interpolated, or smoothed
anywhere in the pipeline. My analysis depends on knowing the sampling grid
exactly: several standard aim metrics turn out to measure the grid rather
than the movement, and a filtered or resampled angle stream silently
invalidates them. The audit code and its acceptance thresholds are open
source and were calibrated against raw demos parsed locally. If the audit
does not pass, I will report why and will not proceed to volume.

**What I will publish.** Aggregate distributions, variance decompositions,
and methodology. Code and thresholds are public.

**What I will not publish or attempt.** No per-player statistics, no
rankings, no identification or scoring of individuals, and no cheat
detection claims. There is no labelled cheating data in this study, so no
false-positive rate can be measured, and I treat any per-player inference as
unsupported by construction. Player identifiers will be pseudonymised with a
salted hash in anything that leaves my machine.

**Volume estimate.** ~50 matches for the grid audit, then on the order of
150–1500 matches depending on what the audit and an initial power analysis
support. I understand standard AWS transfer costs apply.

---

## Notas para você antes de enviar

- Este texto **compromete o projeto publicamente com o escopo de baseline**.
  Está alinhado com o README, mas leia antes de assinar embaixo.
- O parágrafo sobre auditoria de grade não é diplomacia: é o gate real. Se
  A1/A2 reprovarem, o dado deles não serve para este estudo e o certo é
  dizer isso a eles.
- Não prometi prazo. **Não ofereça coautoria** — eu havia levantado isso
  como possível ajuda na aprovação e estava errado. Coautoria é obrigação
  de verdade: revisão, direito de veto sobre o texto, e depois o desconforto
  de publicar resultado que possa não agradar a quem forneceu o dado. Num
  estudo cuja contribuição pode ser justamente um resultado negativo, isso
  é um risco editorial real.
- O que dá para oferecer sem se amarrar já está no texto: **atribuição da
  fonte** e **uma cópia do resultado antes da publicação**. É generoso,
  honesto e não cria direito de veto.
