"""Power spectral density and band power.

Everything here is plain, inspectable numerics on a window of shape
``(n_samples, n_channels)``: Welch PSD, absolute and relative band power,
custom bands, individual alpha frequency and per-region aggregation. No
cognitive interpretation happens at this level -- that is layered on top in
:mod:`neurobci.spectral.indices`, with full transparency.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from neurobci.acquisition.default_montages import region_of
from neurobci.core.stream_info import KIND_EEG, StreamInfo

# Canonical EEG bands (Hz). Users may pass their own dict anywhere ``bands``
# is accepted.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_psd(
    data: np.ndarray, sfreq: float, nperseg_s: float = 1.0, fmax: float = 45.0
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD. Returns ``(freqs, psd)`` with ``psd`` of shape
    ``(n_freqs, n_channels)`` in uV^2/Hz."""
    n = data.shape[0]
    nperseg = int(min(n, max(64, nperseg_s * sfreq)))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=0)
    keep = freqs <= fmax
    return freqs[keep], psd[keep]


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Absolute power in ``band`` (integrated PSD) per channel."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return np.zeros(psd.shape[1])
    return np.trapezoid(psd[mask], freqs[mask], axis=0)


def total_power(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    return np.trapezoid(psd, freqs, axis=0)


def band_powers(
    freqs: np.ndarray, psd: np.ndarray, bands: dict | None = None
) -> dict[str, np.ndarray]:
    """Absolute power per band: ``{band_name: (n_channels,)}``."""
    bands = bands or BANDS
    return {name: band_power(freqs, psd, b) for name, b in bands.items()}


def relative_band_powers(
    freqs: np.ndarray, psd: np.ndarray, bands: dict | None = None
) -> dict[str, np.ndarray]:
    """Band power as a fraction of total power (per channel)."""
    bands = bands or BANDS
    total = total_power(freqs, psd) + 1e-20
    return {name: band_power(freqs, psd, b) / total for name, b in bands.items()}


def individual_alpha_frequency(
    freqs: np.ndarray, psd: np.ndarray, info: StreamInfo,
    search: tuple[float, float] = (7.0, 13.0),
) -> float:
    """Peak alpha frequency over posterior (parietal/occipital) EEG channels."""
    post = [
        i for i in range(info.n_channels)
        if info.channel_kinds[i] == KIND_EEG
        and region_of(info.channel_names[i]) in ("parietal", "occipital")
    ]
    if not post:
        post = [i for i in range(info.n_channels) if info.channel_kinds[i] == KIND_EEG]
    if not post:
        return float("nan")
    mean_psd = psd[:, post].mean(axis=1)
    lo, hi = search
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return float("nan")
    band_freqs = freqs[mask]
    return float(band_freqs[np.argmax(mean_psd[mask])])


def regional_band_power(
    powers: np.ndarray, info: StreamInfo, exclude: set[int] | None = None
) -> dict[str, float]:
    """Mean of a per-channel power vector within each scalp region (EEG only)."""
    exclude = exclude or set()
    regions: dict[str, list[float]] = {}
    for i in range(info.n_channels):
        if info.channel_kinds[i] != KIND_EEG or i in exclude:
            continue
        regions.setdefault(region_of(info.channel_names[i]), []).append(float(powers[i]))
    return {r: float(np.mean(v)) for r, v in regions.items() if v}
