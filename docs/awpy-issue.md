# Rascunho de issue para o awpy

Repositório: https://github.com/pnxenopoulos/awpy
Verificado contra awpy 2.0.2 + demoparser2, três demos de matchmaking CS2.

Título sugerido:

> `Demo.tickrate` defaults to 128 and is never inferred, silently scaling all
> time-derived quantities on 64-tick demos

---

**Corpo:**

## What happens

`Demo.__init__` takes `tickrate: int = awpy.constants.DEFAULT_SERVER_TICKRATE`
(=128) and stores it without ever inspecting the file:

```python
# awpy/demo.py
def __init__(self, path, ..., tickrate: int = awpy.constants.DEFAULT_SERVER_TICKRATE, ...):
    ...
    self.tickrate = tickrate
```

CS2 matchmaking demos are 64-tick. Tournament and FACEIT demos are 128. So on
any matchmaking demo, `dem.tickrate` is off by a factor of 2, and anything a
user derives from it (angular velocity, time-to-event, rates per second) is
scaled by 2 without any warning.

The failure mode is what makes this worth reporting: the error is
multiplicative and uniform, so it does not look like a bug. Everything stays
self-consistent and plausible. On my three test demos, using `dem.tickrate`
would have doubled every angular velocity in the study.

## The information is already parsed, then discarded

`Demo.parse` requests `game_time` to build `in_play_ticks`:

```python
# awpy/demo.py, in parse()
self.in_play_ticks = awpy.parsers.ticks.get_valid_ticks(
    self.parse_ticks(other_props=["game_time", ...])
)
```

and then rebuilds `self.ticks` without it:

```python
self.ticks = self.parse_ticks(player_props=player_props)
```

So `game_time` is read from the demo and dropped. Recovering it needs a second
`parse_ticks` pass. Note that passing `other_props=["game_time"]` to `parse()`
does not help either, because that argument only reaches `parse_events`, not
the `parse_ticks` call that produces `self.ticks`.

## Reproducing

```python
from awpy import Demo
import numpy as np

dem = Demo("match730_....dem")
dem.parse(player_props=["pitch", "yaw"])
print(dem.tickrate)                       # 128

g = (dem.parse_ticks(other_props=["game_time"])
        .select(["tick", "game_time"]).unique(subset=["tick"]).sort("tick"))
t = g["tick"].to_numpy().astype(float)
s = g["game_time"].to_numpy().astype(float)
print(1.0 / np.median(np.diff(s) / np.diff(t)))   # 64.0000
```

On all three demos I tested, both a long-baseline estimate
`(s[-1]-s[0])/(t[-1]-t[0])` and the median of per-tick ratios agree on exactly
`0.015625 s/tick` = 64 Hz, over ~142k ticks.

## Suggestions, in increasing order of intrusiveness

1. **Document it.** A line in the `Demo` docstring saying `tickrate` is a
   user-supplied assumption, not a property read from the file, would be
   enough to stop people from trusting it.
2. **Warn.** If `tickrate` was left at the default, log a warning once.
3. **Infer it.** Since `game_time` is already being parsed, the tickrate could
   be derived in `parse()` at no extra I/O cost. Two independent estimators
   that must agree (long baseline and median of per-step ratios) is cheap and
   catches the pathological cases: the first is immune to f32 quantisation in
   `game_time`, the second is immune to gaps from pauses or timeouts. If they
   disagree, raising is better than picking one.

I have (3) implemented and tested against 32/64/100/128 synthetic cases plus
three real 64-tick demos, and I am happy to open a PR if you would like it.
It is about 40 lines.

I have not tested against a 128-tick demo end to end, so I would not want the
inference merged without someone confirming on tournament data.

---

## Notas para você antes de abrir

- O tom está deliberadamente sem acusação: isto é um default documentado que
  vira armadilha, não um bug. Vale manter assim.
- A oferta de PR é real e barata: o código já existe em `src/ingest.py`
  (`infer_tickrate`), com testes. Se eles aceitarem, é copiar e adaptar.
- A ressalva final sobre não ter testado 128-tick é importante. Sem ela você
  estaria oferecendo algo que não verificou no caso que mais importa para
  eles, que é demo de torneio.
