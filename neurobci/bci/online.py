"""Online P300 decoding.

Scores incoming stimulus epochs with a trained model and aggregates the
target evidence per item across repetitions to make a *selection*. An
``ItemAccumulator` supports the streaming case (scores arrive one flash at
a time); :func:`decide_selection` handles a complete run at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurobci.bci.models import ParadigmModel
from neurobci.paradigms.base import get_paradigm


@dataclass
class ItemAccumulator:
    """Accumulates per-item target evidence during a selection run."""

    n_items: int
    _sums: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _counts: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        self._sums = np.zeros(self.n_items)
        self._counts = np.zeros(self.n_items)

    def add(self, item: int, target_score: float) -> None:
        self._sums[item] += target_score
        self._counts[item] += 1.0

    def reset(self) -> None:
        self._sums[:] = 0.0
        self._counts[:] = 0.0

    @property
    def means(self) -> np.ndarray:
        return np.divide(
            self._sums, self._counts,
            out=np.full(self.n_items, -np.inf), where=self._counts > 0,
        )

    def best_item(self) -> int:
        return int(np.argmax(self.means))

    @property
    def total_flashes(self) -> int:
        return int(self._counts.sum())


class OnlineP300Decoder:
    """Applies a trained P300 model to epochs and decides selections."""

    def __init__(self, model: ParadigmModel) -> None:
        self.model = model
        self.paradigm = get_paradigm(model.paradigm)

    def score_epochs(self, X: np.ndarray) -> np.ndarray:
        """Per-epoch target probability."""
        if X.shape[0] == 0:
            return np.zeros(0)
        return self.model.target_scores(X)

    def decide_selection(
        self, X: np.ndarray, item_ids: np.ndarray, n_items: int
    ) -> tuple[int, np.ndarray]:
        """Decide which item was attended over a whole selection run.

        Returns ``(selected_item, per_item_mean_score)``.
        """
        scores = self.score_epochs(X)
        acc = ItemAccumulator(n_items)
        for item, score in zip(item_ids, scores):
            acc.add(int(item), float(score))
        return acc.best_item(), acc.means
