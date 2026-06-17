"""Synthetic ground-truth data for the Phase-6 paradigms.

Each generator embeds a known, decodable signal so the pipeline can be
tested without hardware and the GUI can run simulated calibration:

* motor imagery -- contralateral mu/beta ERD (band-power lateralisation),
* SSVEP -- occipital oscillation at the attended frequency (+ harmonic),
* ErrP -- a fronto-central ERN/Pe complex on error trials.

Amplitudes are microvolts; ``seed`` makes everything reproducible.
"""

from __future__ import annotations

import numpy as np

from neurobci.acquisition.default_montages import hemisphere_of, region_of
from neurobci.config.schema import ChannelConfig
from neurobci.core.stream_info import KIND_EEG, KIND_EOG
from neurobci.paradigms.base import EpochWindow
from neurobci.paradigms.synthetic import SyntheticDataset


def _kinds(channels: ChannelConfig) -> list[str]:
    eog = set(channels.eog_channels)
    return [KIND_EOG if n in eog else KIND_EEG for n in channels.all_channels]


# --------------------------------------------------------------------------- #
# Motor imagery (active): contralateral mu/beta ERD
# --------------------------------------------------------------------------- #


def make_mi_dataset(
    n_trials: int = 160,
    sfreq: float = 500.0,
    channels: ChannelConfig | None = None,
    mu_amp_uv: float = 8.0,
    beta_amp_uv: float = 4.0,
    noise_uv: float = 3.0,
    erd: float = 0.6,                # fractional power drop contralaterally
    window: EpochWindow | None = None,
    seed: int = 0,
) -> SyntheticDataset:
    channels = channels or ChannelConfig()
    window = window or EpochWindow(tmin=0.5, tmax=2.5, baseline=None)
    rng = np.random.default_rng(seed)
    names = channels.all_channels
    n_ch = len(names)
    n_times = window.n_times(sfreq)
    times = window.tmin + np.arange(n_times) / sfreq
    hemi = np.array([hemisphere_of(n) for n in names])
    eog = set(channels.eog_channels)
    is_eog = np.array([n in eog for n in names])
    amp_drop = np.sqrt(max(1.0 - erd, 0.0))  # amplitude factor for ERD

    y = np.array([0, 1] * (n_trials // 2) + [0] * (n_trials % 2))[:n_trials]
    rng.shuffle(y)

    X = np.empty((n_trials, n_ch, n_times), dtype=np.float64)
    for i in range(n_trials):
        # Independent mu/beta oscillation per channel (random phase).
        ph = rng.uniform(0, 2 * np.pi, (n_ch, 1))
        sig = mu_amp_uv * np.sin(2 * np.pi * 11.0 * times[None, :] + ph)
        sig += beta_amp_uv * np.sin(2 * np.pi * 20.0 * times[None, :] + ph + 0.7)
        # ERD: left-hand imagery (0) suppresses the RIGHT hemisphere; right (1)
        # suppresses the LEFT hemisphere.
        suppress = "R" if y[i] == 0 else "L"
        sig[(hemi == suppress) & ~is_eog] *= amp_drop
        sig += noise_uv * rng.standard_normal((n_ch, n_times))
        sig[is_eog] = noise_uv * rng.standard_normal((int(is_eog.sum()), n_times))
        X[i] = sig

    return SyntheticDataset(X=X, y=y, times=times, sfreq=sfreq,
                            channel_names=names, channel_kinds=_kinds(channels))


# --------------------------------------------------------------------------- #
# SSVEP (reactive): occipital oscillation at the attended frequency
# --------------------------------------------------------------------------- #


def make_ssvep_dataset(
    frequencies: tuple[float, ...] = (8.0, 10.0, 12.0, 15.0),
    n_per_class: int = 12,
    sfreq: float = 500.0,
    channels: ChannelConfig | None = None,
    amp_uv: float = 6.0,
    noise_uv: float = 4.0,
    n_harmonics: int = 2,
    window: EpochWindow | None = None,
    seed: int = 0,
) -> SyntheticDataset:
    channels = channels or ChannelConfig()
    window = window or EpochWindow(tmin=0.0, tmax=2.0, baseline=None)
    rng = np.random.default_rng(seed)
    names = channels.all_channels
    n_ch = len(names)
    n_times = window.n_times(sfreq)
    times = window.tmin + np.arange(n_times) / sfreq

    # Occipital/parietal channels carry the SSVEP; others mostly noise.
    occ_weight = np.array([
        1.0 if region_of(n) == "occipital" else 0.5 if region_of(n) == "parietal"
        else 0.1 for n in names
    ])
    eog = set(channels.eog_channels)
    for j, n in enumerate(names):
        if n in eog:
            occ_weight[j] = 0.0

    y = np.repeat(np.arange(len(frequencies)), n_per_class)
    rng.shuffle(y)
    X = np.empty((len(y), n_ch, n_times), dtype=np.float64)
    for i, cls in enumerate(y):
        f = frequencies[cls]
        wave = np.zeros(n_times)
        for h in range(1, n_harmonics + 1):
            ph = rng.uniform(0, 2 * np.pi)
            wave += (amp_uv / h) * np.sin(2 * np.pi * h * f * times + ph)
        sig = occ_weight[:, None] * wave[None, :]
        sig += noise_uv * rng.standard_normal((n_ch, n_times))
        X[i] = sig

    return SyntheticDataset(X=X, y=y, times=times, sfreq=sfreq,
                            channel_names=names, channel_kinds=_kinds(channels))


# --------------------------------------------------------------------------- #
# ErrP (reactive): fronto-central ERN/Pe on error trials
# --------------------------------------------------------------------------- #

_ERRP_REGION_WEIGHT = {
    "frontal": 0.9, "central": 1.0, "parietal": 0.5,
    "occipital": 0.2, "temporal": 0.4, "other": 0.5,
}


def _errp_waveform(times: np.ndarray) -> np.ndarray:
    ern = -1.0 * np.exp(-((times - 0.10) / 0.04) ** 2)   # error-related negativity
    pe = 0.8 * np.exp(-((times - 0.32) / 0.07) ** 2)     # error positivity
    return ern + pe


def make_errp_dataset(
    n_trials: int = 260,
    sfreq: float = 500.0,
    channels: ChannelConfig | None = None,
    error_ratio: float = 0.3,
    amp_uv: float = 6.0,
    noise_uv: float = 4.0,
    window: EpochWindow | None = None,
    seed: int = 0,
) -> SyntheticDataset:
    channels = channels or ChannelConfig()
    window = window or EpochWindow(tmin=-0.1, tmax=0.8, baseline=(-0.1, 0.0))
    rng = np.random.default_rng(seed)
    names = channels.all_channels
    n_ch = len(names)
    n_times = window.n_times(sfreq)
    times = window.tmin + np.arange(n_times) / sfreq
    eog = set(channels.eog_channels)
    weights = np.array([
        0.0 if n in eog else _ERRP_REGION_WEIGHT[region_of(n)] for n in names
    ])
    erp = _errp_waveform(times)

    n_err = int(round(n_trials * error_ratio))
    y = np.array([1] * n_err + [0] * (n_trials - n_err))
    rng.shuffle(y)
    X = np.empty((n_trials, n_ch, n_times), dtype=np.float64)
    for i in range(n_trials):
        noise = noise_uv * rng.standard_normal((n_ch, n_times))
        trial = noise
        if y[i] == 1:
            a = amp_uv * rng.uniform(0.7, 1.3)
            trial = trial + a * np.outer(weights, erp)
        X[i] = trial
    if window.baseline is not None:
        b0, b1 = window.baseline
        bmask = (times >= b0) & (times <= b1)
        if bmask.any():
            X -= X[:, :, bmask].mean(axis=2, keepdims=True)

    return SyntheticDataset(X=X, y=y, times=times, sfreq=sfreq,
                            channel_names=names, channel_kinds=_kinds(channels))
