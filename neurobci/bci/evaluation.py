"""Scientifically-valid model evaluation.

Guards against the classic ways EEG decoding results get inflated:

* the **entire** pipeline (spatial filters, scalers, classifier) is refit
  inside each cross-validation fold -- nothing is fit on held-out data;
* **block/group-aware** splitting is supported so temporally adjacent
  trials don't leak across folds;
* results are always reported **against chance** (0.5 balanced accuracy and
  the majority-class accuracy), with class counts, so a chance-level model
  can never look successful;
* class-imbalance-robust metrics (balanced accuracy, ROC-AUC, kappa, MCC)
  are primary -- plain accuracy is reported but never used alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

from neurobci.bci.models import build_pipeline

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    model_name: str
    balanced_accuracy: float
    balanced_accuracy_std: float
    accuracy: float
    roc_auc: float
    kappa: float
    mcc: float
    confusion: list[list[int]]
    per_fold_balacc: list[float]
    n_folds: int
    class_counts: dict[int, int]
    majority_accuracy: float
    chance_balanced: float = 0.5
    oof_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    oof_true: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def beats_chance(self) -> bool:
        # Above chance by more than one fold-to-fold std.
        return self.balanced_accuracy - self.balanced_accuracy_std > self.chance_balanced

    def summary(self) -> str:
        return (
            f"{self.model_name}: bal.acc {self.balanced_accuracy:.3f}"
            f"+/-{self.balanced_accuracy_std:.3f}  AUC {self.roc_auc:.3f}  "
            f"kappa {self.kappa:.3f}  (chance {self.chance_balanced:.2f}, "
            f"majority {self.majority_accuracy:.3f})"
        )


def _majority_accuracy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def cross_validate(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    positive_index: int = 1,
    n_folds: int = 5,
    groups: np.ndarray | None = None,
    seed: int = 0,
) -> EvaluationResult:
    """Out-of-fold cross-validation for one model. Refits per fold."""

    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_folds = int(min(n_folds, counts.min())) if counts.min() >= 2 else 2
    n_folds = max(n_folds, 2)

    if groups is not None:
        n_groups = len(np.unique(groups))
        splitter = GroupKFold(n_splits=min(n_folds, n_groups))
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)

    oof_scores = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)
    per_fold: list[float] = []

    base = build_pipeline(model_name, sfreq)
    for train_idx, test_idx in split_iter:
        pipe = clone(base)
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        proba = pipe.predict_proba(X[test_idx])
        col = list(pipe.classes_).index(positive_index)
        oof_scores[test_idx] = proba[:, col]
        oof_pred[test_idx] = pred
        per_fold.append(balanced_accuracy_score(y[test_idx], pred))

    mask = oof_pred >= 0
    yt, yp, ys = y[mask], oof_pred[mask], oof_scores[mask]

    try:
        auc = float(roc_auc_score(yt == positive_index, ys))
    except ValueError:
        auc = float("nan")

    return EvaluationResult(
        model_name=model_name,
        balanced_accuracy=float(np.mean(per_fold)),
        balanced_accuracy_std=float(np.std(per_fold)),
        accuracy=float(accuracy_score(yt, yp)),
        roc_auc=auc,
        kappa=float(cohen_kappa_score(yt, yp)),
        mcc=float(matthews_corrcoef(yt, yp)),
        confusion=confusion_matrix(yt, yp).tolist(),
        per_fold_balacc=[float(f) for f in per_fold],
        n_folds=len(per_fold),
        class_counts={int(k): int(v) for k, v in zip(classes, counts)},
        majority_accuracy=_majority_accuracy(y),
        oof_scores=ys,
        oof_true=(yt == positive_index).astype(int),
    )


def compare_models(
    model_names: list[str],
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    positive_index: int = 1,
    n_folds: int = 5,
    groups: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[dict[str, EvaluationResult], str | None]:
    """Evaluate several models; return (results, best_model_name).

    A model that raises (e.g. an optional dependency problem) is skipped
    with a log message rather than aborting the whole comparison.
    """

    results: dict[str, EvaluationResult] = {}
    for name in model_names:
        try:
            results[name] = cross_validate(
                name, X, y, sfreq, positive_index, n_folds, groups, seed
            )
        except Exception:  # noqa: BLE001
            logger.exception("Model %r failed during evaluation; skipping.", name)

    best = None
    if results:
        best = max(results, key=lambda n: results[n].balanced_accuracy)
    return results, best
