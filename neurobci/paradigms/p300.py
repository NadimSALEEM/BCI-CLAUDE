"""P300 / ERP-based selection paradigm.

The user attends to one item among several; each item flashes repeatedly.
A flash of the attended item elicits a P300 (a centro-parietal positive
deflection ~300 ms post-stimulus); other flashes do not. The classifier
discriminates target vs non-target flashes; aggregating the target scores
across repetitions for each item and taking the maximum yields the
selected item.

Chosen as the first paradigm because the P300 is large, stereotyped and
needs little user training, so it exercises the whole pipeline
(calibration -> epoching -> spatial filtering -> classifier -> decision)
with the most reliable signal.
"""

from __future__ import annotations

import numpy as np

from neurobci.paradigms.base import EpochWindow, Paradigm, register_paradigm

TARGET = "target"
NONTARGET = "nontarget"


@register_paradigm
class P300Paradigm(Paradigm):
    name = "p300"

    @property
    def class_labels(self) -> list[str]:
        # index 0 = nontarget, index 1 = target
        return [NONTARGET, TARGET]

    @property
    def positive_label(self) -> str:
        return TARGET

    @property
    def window(self) -> EpochWindow:
        return EpochWindow(tmin=-0.1, tmax=0.6, baseline=(-0.1, 0.0), reject_uv=150.0)

    @property
    def model_names(self) -> list[str]:
        return ["vec_lda", "riemann_lr"]

    # ----- selection aggregation ---------------------------------------- #

    def aggregate_selection(
        self, item_ids: np.ndarray, target_scores: np.ndarray, n_items: int
    ) -> int:
        """Return the selected item index.

        ``item_ids[i]`` is the item that flashed for epoch ``i``;
        ``target_scores[i]`` is that epoch's target probability/score. The
        selected item is the one whose flashes accumulate the most target
        evidence.
        """

        sums = np.zeros(n_items, dtype=float)
        counts = np.zeros(n_items, dtype=float)
        for item, score in zip(item_ids, target_scores):
            sums[int(item)] += float(score)
            counts[int(item)] += 1.0
        means = np.divide(sums, counts, out=np.full(n_items, -np.inf), where=counts > 0)
        return int(np.argmax(means))
