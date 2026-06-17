"""Synthetic ERP data with known ground truth.

Generates P300 datasets where target epochs contain a centro-parietal
positive deflection (~300 ms) and non-targets do not. Used to (a) unit-test
the classification pipeline against a known answer and (b) drive the GUI's
*simulated calibration* so the whole paradigm is usable without hardware.

Amplitudes are microvolts. ``seed`` makes everything reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurobci.acquisition.default_montages import region_of
from neurobci.config.schema import ChannelConfig
from neurobci.core.stream_info import KIND_EEG, KIND_EOG
from neurobci.paradigms.base import EpochWindow

# Centro-parietal P300 topography (relative weights by scalp region).
_P300_REGION_WEIGHT = {
    "parietal": 1.0, "central": 0.95, "occipital": 0.6,
    "temporal": 0.45, "frontal": 0.35, "other": 0.5,
}


@dataclass
class SyntheticDataset:
    X: np.ndarray            # (n_trials, n_channels, n_times)
    y: np.ndarray            # int labels (0 nontarget, 1 target)
    times: np.ndarray        # (n_times,) seconds relative to stimulus
    sfreq: float
    channel_names: list[str]
    channel_kinds: list[str]


def _spatial_weights(channels: ChannelConfig) -> np.ndarray:
    w = []
    eog = set(channels.eog_channels)
    for name in channels.all_channels:
        if name in eog:
            w.append(0.0)                      # eye channel carries no P300
        else:
            w.append(_P300_REGION_WEIGHT[region_of(name)])
    return np.array(w, dtype=float)


def _p300_waveform(times: np.ndarray) -> np.ndarray:
    """Unit-peak ERP: small N200 trough then a P300 peak at ~300 ms."""
    p300 = np.exp(-((times - 0.30) / 0.06) ** 2)
    n200 = -0.35 * np.exp(-((times - 0.20) / 0.04) ** 2)
    return p300 + n200


def make_p300_dataset(
    n_trials: int = 240,
    sfreq: float = 500.0,
    channels: ChannelConfig | None = None,
    target_ratio: float = 0.25,
    p300_amp_uv: float = 6.0,
    noise_uv: float = 4.0,
    window: EpochWindow | None = None,
    seed: int = 0,
) -> SyntheticDataset:
    """Create a labelled epoch dataset with a known P300 in target trials."""

    channels = channels or ChannelConfig()
    window = window or EpochWindow()
    rng = np.random.default_rng(seed)

    n_ch = channels.n_channels
    n_times = window.n_times(sfreq)
    times = window.tmin + np.arange(n_times) / sfreq

    weights = _spatial_weights(channels)
    erp = _p300_waveform(times)               # (n_times,)

    n_target = int(round(n_trials * target_ratio))
    y = np.array([1] * n_target + [0] * (n_trials - n_target))
    rng.shuffle(y)

    X = np.empty((n_trials, n_ch, n_times), dtype=np.float64)
    for i in range(n_trials):
        # Background: white + a little low-frequency wander per channel.
        noise = noise_uv * rng.standard_normal((n_ch, n_times))
        noise += 0.5 * noise_uv * np.cumsum(
            rng.standard_normal((n_ch, n_times)), axis=1
        ) / np.sqrt(n_times)
        trial = noise
        if y[i] == 1:
            amp = p300_amp_uv * rng.uniform(0.7, 1.3)   # trial-to-trial jitter
            trial = trial + amp * np.outer(weights, erp)
        X[i] = trial

    # Baseline correction.
    if window.baseline is not None:
        b0, b1 = window.baseline
        bmask = (times >= b0) & (times <= b1)
        if bmask.any():
            X -= X[:, :, bmask].mean(axis=2, keepdims=True)

    return SyntheticDataset(
        X=X, y=y, times=times, sfreq=sfreq,
        channel_names=channels.all_channels,
        channel_kinds=[
            KIND_EOG if n in set(channels.eog_channels) else KIND_EEG
            for n in channels.all_channels
        ],
    )


def make_p300_selection_run(
    n_items: int = 6,
    n_repetitions: int = 10,
    target_item: int = 2,
    sfreq: float = 500.0,
    channels: ChannelConfig | None = None,
    p300_amp_uv: float = 6.0,
    noise_uv: float = 4.0,
    window: EpochWindow | None = None,
    seed: int = 1,
) -> tuple[SyntheticDataset, np.ndarray, int]:
    """A single 'selection' run: each item flashes ``n_repetitions`` times.

    Returns ``(dataset, item_ids, target_item)`` where ``item_ids[i]`` is the
    item that flashed for epoch ``i`` (in random order). Flashes of
    ``target_item`` carry the P300.
    """

    channels = channels or ChannelConfig()
    window = window or EpochWindow()
    rng = np.random.default_rng(seed)

    item_ids = np.repeat(np.arange(n_items), n_repetitions)
    rng.shuffle(item_ids)
    y = (item_ids == target_item).astype(int)

    # Reuse the single-trial generator by building trials directly.
    n_ch = channels.n_channels
    n_times = window.n_times(sfreq)
    times = window.tmin + np.arange(n_times) / sfreq
    weights = _spatial_weights(channels)
    erp = _p300_waveform(times)

    X = np.empty((len(item_ids), n_ch, n_times), dtype=np.float64)
    for i, is_t in enumerate(y):
        noise = noise_uv * rng.standard_normal((n_ch, n_times))
        trial = noise
        if is_t:
            amp = p300_amp_uv * rng.uniform(0.7, 1.3)
            trial = trial + amp * np.outer(weights, erp)
        X[i] = trial
    if window.baseline is not None:
        b0, b1 = window.baseline
        bmask = (times >= b0) & (times <= b1)
        if bmask.any():
            X -= X[:, :, bmask].mean(axis=2, keepdims=True)

    ds = SyntheticDataset(
        X=X, y=y, times=times, sfreq=sfreq,
        channel_names=channels.all_channels,
        channel_kinds=[
            KIND_EOG if n in set(channels.eog_channels) else KIND_EEG
            for n in channels.all_channels
        ],
    )
    return ds, item_ids, target_item
