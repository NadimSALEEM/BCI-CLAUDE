"""P300 selection decision logic with early stopping.

Accumulates per-item target evidence as flashes arrive and decides which
item the user is attending *only* when the evidence is strong enough:

* every item must have flashed at least ``min_repetitions`` times,
* the best item's mean score must exceed ``min_confidence``,
* and beat the runner-up by at least ``confidence_margin``.

If those are met early, it stops early (fewer flashes = faster selection).
If ``max_repetitions`` or ``timeout_s`` is reached first, it **abstains**
(the safe default: no command), never guessing.
"""

from __future__ import annotations

import time

import numpy as np

from neurobci.config.schema import SelectionConfig
from neurobci.control.types import OutcomeStatus, SelectionOutcome


class SelectionController:
    def __init__(self, config: SelectionConfig, n_items: int) -> None:
        self.config = config
        self.n_items = n_items
        self._start = time.time()
        self._sums = np.zeros(n_items)
        self._counts = np.zeros(n_items)
        self._flashes = 0

    def reset(self) -> None:
        self._start = time.time()
        self._sums[:] = 0.0
        self._counts[:] = 0.0
        self._flashes = 0

    @property
    def n_flashes(self) -> int:
        return self._flashes

    def _means(self) -> np.ndarray:
        return np.divide(
            self._sums, self._counts,
            out=np.full(self.n_items, -np.inf), where=self._counts > 0,
        )

    def add(self, item: int, target_score: float, now: float | None = None) -> SelectionOutcome:
        """Add one flash's score and return the current outcome."""
        self._sums[item] += float(target_score)
        self._counts[item] += 1.0
        self._flashes += 1
        return self._evaluate(now)

    def _evaluate(self, now: float | None) -> SelectionOutcome:
        now = time.time() if now is None else now
        means = self._means()
        finite = means[np.isfinite(means)]
        order = np.argsort(means)[::-1]
        best = int(order[0])
        best_mean = float(means[best])
        second_mean = float(means[order[1]]) if self.n_items > 1 and np.isfinite(means[order[1]]) else -np.inf
        margin = best_mean - second_mean if np.isfinite(second_mean) else best_mean
        per_item = [float(m) if np.isfinite(m) else 0.0 for m in means]

        c = self.config
        enough_reps = self._counts.min() >= c.min_repetitions
        confident = best_mean >= c.min_confidence and margin >= c.confidence_margin

        if enough_reps and confident:
            return SelectionOutcome(
                OutcomeStatus.DECIDED, best, best_mean, margin, self._flashes,
                per_item, reason=f"margin {margin:.2f}>= {c.confidence_margin} "
                                 f"& reps>= {c.min_repetitions}",
            )

        timed_out = (now - self._start) >= c.timeout_s
        maxed = self._counts.max() >= c.max_repetitions
        if timed_out or maxed:
            why = "timeout" if timed_out else "max repetitions"
            return SelectionOutcome(
                OutcomeStatus.ABSTAINED, None, best_mean, margin, self._flashes,
                per_item, reason=f"abstained ({why}); evidence insufficient",
            )

        return SelectionOutcome(
            OutcomeStatus.PENDING, None, best_mean, margin, self._flashes,
            per_item, reason="collecting evidence",
        )
