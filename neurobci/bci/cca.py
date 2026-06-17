"""Canonical Correlation Analysis decoder for SSVEP.

For each candidate stimulus frequency a reference set of sine/cosine pairs
(fundamental + harmonics) is built; the predicted class is the frequency
whose reference best correlates (max canonical correlation) with the epoch.
This needs no per-class training, which is why SSVEP can be used with a very
short (or no) calibration.

The decoder is scikit-learn-compatible so it slots into the same
``ParadigmModel`` / evaluation machinery as the trained classifiers.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import CCA


class CCADecoder(BaseEstimator, ClassifierMixin):
    def __init__(self, frequencies, sfreq: float, n_harmonics: int = 2) -> None:
        self.frequencies = tuple(float(f) for f in frequencies)
        self.sfreq = float(sfreq)
        self.n_harmonics = int(n_harmonics)

    # ----- sklearn API --------------------------------------------------- #

    def fit(self, X, y=None):
        # Calibration-free: just record the class set.
        self.classes_ = np.arange(len(self.frequencies))
        return self

    def _references(self, n_times: int) -> list[np.ndarray]:
        t = np.arange(n_times) / self.sfreq
        refs = []
        for f in self.frequencies:
            comps = []
            for h in range(1, self.n_harmonics + 1):
                comps.append(np.sin(2 * np.pi * h * f * t))
                comps.append(np.cos(2 * np.pi * h * f * t))
            refs.append(np.column_stack(comps))      # (n_times, 2*n_harmonics)
        return refs

    def _epoch_scores(self, epoch: np.ndarray, refs) -> np.ndarray:
        """Max canonical correlation of one epoch vs each frequency reference."""
        x = epoch.T                                   # (n_times, n_channels)
        x = x - x.mean(axis=0, keepdims=True)
        scores = np.zeros(len(refs))
        for k, ref in enumerate(refs):
            try:
                cca = CCA(n_components=1)
                xc, rc = cca.fit_transform(x, ref)
                r = np.corrcoef(xc[:, 0], rc[:, 0])[0, 1]
                scores[k] = 0.0 if not np.isfinite(r) else abs(r)
            except Exception:  # noqa: BLE001 - degenerate epoch
                scores[k] = 0.0
        return scores

    def decision_function(self, X):
        X = np.asarray(X)
        refs = self._references(X.shape[2])
        return np.vstack([self._epoch_scores(ep, refs) for ep in X])

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        # Softmax over correlations -> pseudo-probabilities.
        z = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(z * 5.0)                            # temperature sharpens
        return e / e.sum(axis=1, keepdims=True)


def evaluate_cca(decoder: CCADecoder, X, y) -> dict:
    """Simple decode accuracy report (CCA needs no train/test split)."""
    pred = decoder.fit(X, y).predict(X)
    y = np.asarray(y)
    acc = float(np.mean(pred == y))
    n_classes = len(decoder.frequencies)
    return {
        "accuracy": acc,
        "chance": 1.0 / n_classes,
        "n_trials": int(len(y)),
        "n_classes": n_classes,
        "confusion": _confusion(y, pred, n_classes),
    }


def _confusion(y, pred, n) -> list[list[int]]:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        m[int(t), int(p)] += 1
    return m.tolist()
