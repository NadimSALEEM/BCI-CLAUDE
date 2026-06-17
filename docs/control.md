# Decision, safety & control

Phase 5 turns model predictions into **safe, traceable commands** and drives
an on-screen demo. Predictions never become actions directly — they pass
through a decision layer and a single safety-gated router.

```
model scores ─► decision logic ─► Command ─► CommandRouter ─► adapter ─► effect
                (selection /        │              │  ▲
                 streaming)         │        SafetyMonitor + rate limit
                                    └─► history + online metrics
```

## Decision logic

Two complementary styles (paradigm chooses):

**Selection (P300)** — `control/selection.py`. Accumulates per-item target
evidence over flashes and decides **only** when:

- every item flashed ≥ `min_repetitions`,
- the best item's mean score ≥ `min_confidence`,
- and it beats the runner-up by ≥ `confidence_margin`.

Meeting these early stops early (fewer flashes). Hitting `max_repetitions`
or `timeout_s` first → **abstain** (no command). The default response to
uncertainty is always *no action*.

**Streaming (active BCI)** — `control/decision.py`. For a per-window class
stream: a confidence threshold (low-confidence ⇒ neutral), sliding-window
**majority vote** needing `min_agree`, a **refractory period** after each
command, and a **neutral/no-command** default.

## Safety gating

`control/safety.py` + the router enforce, for **every** command:

| Gate | Blocks when |
|------|-------------|
| Emergency stop | the e-stop latch is active |
| Connection | stream not connected/stable (`require_connected`) |
| Confidence | below `min_confidence` |
| Signal quality | quality is unacceptable (`require_quality`) |
| External control | command targets external HW and `external_control_enabled` is False (**default**) |
| Rate limit | more than `max_commands_per_min` in the last minute |
| Test mode | `test_mode` on ⇒ predict but never execute |

External/OS control is **off by default** and no adapter executes arbitrary
system commands. Because all commands funnel through `CommandRouter.submit`,
these guarantees hold on every path.

## Traceability

Every executed or rejected command is stored as a `CommandRecord` with the
`Command` (action, item, confidence, margin) and its `trace` (model
version, per-item evidence, flash count, decision rule), plus whether it
executed, the rejection reason, latency and correctness (vs intended).

## Online metrics

`control/metrics.py`: executed / rejected / abstained counts, **accuracy**
(when intent is known), **commands per minute**, **abstention rate** and
mean latency — the quantities that actually characterise a BCI, not just
accuracy.

## On-screen demo (Control / BCI tab)

Items sit on a ring; the cursor moves toward the selected item. *Run
selection* (or *Run 6 trials*) runs the real stack on simulated stimuli for
a chosen intended item; the history and metrics update; **Test mode** and
the toolbar **Emergency Stop** visibly gate execution. Requires a model
from the Calibration tab and a running (simulated) stream.

> Only the *stimulus source* is simulated. The decision logic, safety
> gating, adapter, history and metrics are the production components. A
> live on-screen flashing task wired to real-time LSL markers is the
> remaining integration on top of this stack.

## Adding a control adapter

Subclass `ControlAdapter`, set `is_external` appropriately, implement
`execute(command)`. The decision/safety/history/metrics machinery is reused
unchanged — and external adapters stay disabled until explicitly enabled.
