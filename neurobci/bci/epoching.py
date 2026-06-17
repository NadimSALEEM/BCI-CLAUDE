"""Epoch extraction from continuous data.

Turns continuous samples plus stimulus onsets into a labelled epoch array
``X`` of shape ``(n_epochs, n_channels, n_times)`` -- the canonical layout
expected by the feature/model code (and by pyriemann/MNE). Handles
baseline correction and peak-to-peak amplitude rejection, and reports
exactly which onsets were dropped and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurobci.paradigms.base import EpochWindow


@dataclass
class EpochSet:
    X: np.ndarray                    # (n_epochs, n_channels, n_times)
    y: np.ndarray                    # (n_epochs,) int labels
    times: np.ndarray                # (n_times,) seconds relative to onset
    sfreq: float
    kept_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    n_rejected: int = 0
    n_out_of_bounds: int = 0

    @property
    def n_epochs(self) -> int:
        return self.X.shape[0]

    def class_counts(self) -> dict[int, int]:
        u, c = np.unique(self.y, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}


def onsets_from_timestamps(
    sample_timestamps: np.ndarray, event_times: np.ndarray
) -> np.ndarray:
    """Map event timestamps to nearest sample indices (monotonic search)."""
    sample_timestamps = np.asarray(sample_timestamps, dtype=np.float64)
    idx = np.searchsorted(sample_timestamps, np.asarray(event_times, float))
    return np.clip(idx, 0, len(sample_timestamps) - 1).astype(int)


def extract_epochs(
    data: np.ndarray,
    sfreq: float,
    onsets: np.ndarray,
    labels: np.ndarray,
    window: EpochWindow,
    reject: bool = True,
) -> EpochSet:
    """Extract epochs around ``onsets`` (sample indices into ``data``).

    ``data`` is ``(n_samples, n_channels)``. Returns an :class:`EpochSet`
    with out-of-bounds and amplitude-rejected epochs removed.
    """

    data = np.asarray(data, dtype=np.float64)
    onsets = np.asarray(onsets, dtype=int)
    labels = np.asarray(labels)
    n_samples, n_channels = data.shape

    start_off = int(round(window.tmin * sfreq))
    n_times = window.n_times(sfreq)
    times = window.tmin + np.arange(n_times) / sfreq

    b0 = b1 = None
    if window.baseline is not None:
        bmask = (times >= window.baseline[0]) & (times <= window.baseline[1])
    else:
        bmask = None

    epochs, kept, ys = [], [], []
    n_oob = n_rej = 0
    for onset, label in zip(onsets, labels):
        s = onset + start_off
        e = s + n_times
        if s < 0 or e > n_samples:
            n_oob += 1
            continue
        ep = data[s:e].T.copy()                      # (n_channels, n_times)
        if bmask is not None and bmask.any():
            ep -= ep[:, bmask].mean(axis=1, keepdims=True)
        if reject:
            ptp = np.ptp(ep, axis=1)
            if np.max(ptp) > window.reject_uv:
                n_rej += 1
                continue
        epochs.append(ep)
        ys.append(int(label))
        kept.append(int(onset))

    if epochs:
        X = np.stack(epochs, axis=0)
        y = np.array(ys, dtype=int)
    else:
        X = np.empty((0, n_channels, n_times), dtype=np.float64)
        y = np.empty(0, dtype=int)

    return EpochSet(
        X=X, y=y, times=times, sfreq=sfreq,
        kept_indices=np.array(kept, dtype=int),
        n_rejected=n_rej, n_out_of_bounds=n_oob,
    )
