"""Drive a selection trial from the simulator through the full stack.

The on-screen demo and the integration tests both use this: it generates a
P300 selection run for a chosen *intended* item, feeds the model's
per-flash target scores into a :class:`SelectionController` with early
stopping, then routes the outcome (a selection command, or a safe
abstention) through the :class:`CommandRouter`.

Only the *stimulus source* is simulated here; the decision logic, safety
gating, adapter, history and metrics are the real production components.
"""

from __future__ import annotations

import numpy as np

from neurobci.bci.models import ParadigmModel
from neurobci.bci.online import OnlineP300Decoder
from neurobci.config.schema import SelectionConfig
from neurobci.control.decision import Decision, DecisionLayer
from neurobci.control.router import CommandRouter
from neurobci.control.selection import SelectionController
from neurobci.control.types import Command, CommandRecord, SelectionOutcome
from neurobci.paradigms.base import get_paradigm
from neurobci.paradigms.synthetic import make_p300_selection_run
from neurobci.paradigms.synthetic_paradigms import make_mi_dataset

# Representative stimulus-onset asynchrony (s) used to estimate latency.
_SOA_S = 0.15


def simulate_selection(
    model: ParadigmModel,
    controller: SelectionController,
    intended_item: int,
    channels,
    sfreq: float,
    p300_amp_uv: float = 6.0,
    noise_uv: float = 4.0,
    seed: int = 0,
) -> tuple[SelectionOutcome, float]:
    """Run one selection with early stopping; return (outcome, latency_s)."""

    window = get_paradigm(model.paradigm).window
    ds, item_ids, _ = make_p300_selection_run(
        n_items=controller.n_items,
        n_repetitions=controller.config.max_repetitions,
        target_item=intended_item, channels=channels, sfreq=sfreq,
        p300_amp_uv=p300_amp_uv, noise_uv=noise_uv, window=window, seed=seed,
    )
    decoder = OnlineP300Decoder(model)
    scores = decoder.score_epochs(ds.X)

    controller.reset()
    outcome: SelectionOutcome | None = None
    for item, score in zip(item_ids, scores):
        outcome = controller.add(int(item), float(score), now=0.0)  # ignore wall-clock
        if outcome.status.value != "pending":
            break
    latency = (outcome.n_flashes if outcome else 0) * _SOA_S
    return outcome, latency


def run_selection_trial(
    model: ParadigmModel,
    router: CommandRouter,
    sel_config: SelectionConfig,
    intended_item: int,
    channels,
    sfreq: float,
    p300_amp_uv: float = 6.0,
    noise_uv: float = 4.0,
    emergency_stop: bool = False,
    connected: bool = True,
    quality_ok: bool | None = None,
    seed: int = 0,
) -> tuple[SelectionOutcome, CommandRecord]:
    """Full trial: simulate -> decide -> route (command or abstention)."""

    controller = SelectionController(sel_config, n_items=router.config.n_items)
    # Disable the wall-clock timeout for deterministic simulation; rely on
    # max_repetitions for the abstain condition.
    outcome, latency = simulate_selection(
        model, controller, intended_item, channels, sfreq,
        p300_amp_uv, noise_uv, seed,
    )

    if outcome.decided:
        cmd = Command(
            action=f"select:{outcome.item}",
            item=outcome.item,
            confidence=outcome.confidence,
            margin=outcome.margin,
            trace={
                "model": f"{model.paradigm}/{model.name}",
                "n_flashes": outcome.n_flashes,
                "per_item_mean": outcome.per_item_mean,
                "rule": outcome.reason,
            },
        )
        rec = router.submit(
            cmd, emergency_stop=emergency_stop, connected=connected,
            quality_ok=quality_ok, intended=intended_item, latency_s=latency,
        )
    else:
        rec = router.record_abstention(
            confidence=outcome.confidence, intended=intended_item
        )
    return outcome, rec


def simulate_mi_stream(
    model: ParadigmModel,
    decision_layer: DecisionLayer,
    true_class: int,
    channels,
    sfreq: float,
    n_windows: int = 15,
    mu_amp_uv: float = 8.0,
    noise_uv: float = 3.0,
    seed: int = 0,
) -> list[Decision]:
    """Stream motor-imagery windows of one class through the decision layer.

    Returns the per-window :class:`Decision` records. Demonstrates the active
    (continuous) BCI path: noisy per-window class predictions are stabilised
    into commands by majority voting + refractory in the decision layer.
    """

    paradigm = get_paradigm(model.paradigm)
    labels = paradigm.class_labels
    # Generate enough trials to harvest n_windows epochs of the target class.
    ds = make_mi_dataset(
        n_trials=max(4 * n_windows, 40), sfreq=sfreq, channels=channels,
        mu_amp_uv=mu_amp_uv, noise_uv=noise_uv, window=paradigm.window, seed=seed,
    )
    idx = np.where(ds.y == true_class)[0][:n_windows]
    decision_layer.reset()
    decisions: list[Decision] = []
    for t, i in enumerate(idx):
        ep = ds.X[i][None]
        pred = int(model.predict(ep)[0])
        conf = float(np.max(model.predict_proba(ep)[0]))
        decisions.append(decision_layer.push(labels[pred], conf, now=float(t)))
    return decisions
