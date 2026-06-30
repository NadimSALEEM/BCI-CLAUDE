"""Trial-level feature extraction for statistics and tabular ML.

Reduces an :class:`~neurobci.analysis.datasource.EpochBundle` to one number per
trial (ROI-averaged) for univariate tests, or to a per-channel matrix for ML.
Covers time-domain ERP features (mean amplitude, peak amplitude/latency, area,
GFP) and spectral band power, plus named ERP-component windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid

# Canonical ERP-component windows (seconds). User-editable in the UI.
ERP_COMPONENTS: dict[str, tuple[float, float]] = {
    "P100": (0.080, 0.130),
    "N170": (0.130, 0.200),
    "N200": (0.180, 0.280),
    "P300": (0.250, 0.500),
    "N400": (0.300, 0.500),
    "ERN":  (0.000, 0.100),
    "Pe":   (0.200, 0.400),
    "FRN":  (0.200, 0.350),
    "CNV":  (-0.500, 0.000),
}

# Common scalp ROIs by 10-20 name (intersection with the montage is used).
STANDARD_ROIS: dict[str, list[str]] = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "Central": ["C3", "C4", "Cz", "FC1", "FC2", "CP1", "CP2"],
    "Parietal": ["P3", "P4", "Pz", "P7", "P8"],
    "Occipital": ["O1", "O2", "Oz"],
    "Temporal": ["T7", "T8", "TP7", "TP8", "FT7", "FT8"],
    "Midline": ["Fz", "Cz", "Pz", "Oz"],
}

# Canonical EEG frequency bands (Hz).
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
}


@dataclass
class ERPFeatureSpec:
    """Time-domain ERP feature over a window, averaged across an ROI."""

    feature: str = "mean_amplitude"   # mean_amplitude|peak_amplitude|peak_latency|auc|gfp
    tmin: float = 0.25
    tmax: float = 0.45
    channels: list[str] | None = None  # None -> all EEG channels
    peak_sign: str = "abs"             # abs|pos|neg
    rectify_auc: bool = False

    def label(self) -> str:
        roi = "ROI" if self.channels else "allEEG"
        return f"{self.feature}[{self.tmin:.0f}-{self.tmax:.0f}s,{roi}]" \
            .replace("0-", f"{self.tmin:g}-")


@dataclass
class BandPowerSpec:
    """Spectral band power over the epoch, averaged across an ROI."""

    fmin: float = 4.0
    fmax: float = 8.0
    channels: list[str] | None = None
    relative: bool = False             # divide by total in `total_band`
    log: bool = False                  # 10*log10
    total_band: tuple[float, float] = (1.0, 45.0)
    nperseg: int | None = None

    def label(self) -> str:
        kind = "relpow" if self.relative else ("logpow" if self.log else "pow")
        return f"{kind}[{self.fmin:g}-{self.fmax:g}Hz]"


def window_mask(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    m = (times >= tmin) & (times <= tmax)
    if not m.any():                       # degenerate window -> nearest sample
        m = np.zeros_like(times, dtype=bool)
        m[int(np.argmin(np.abs(times - 0.5 * (tmin + tmax))))] = True
    return m


# ----- ERP / time-domain ------------------------------------------------- #

def _erp_per_channel(X: np.ndarray, times: np.ndarray, spec: ERPFeatureSpec,
                     ch_idx: list[int]) -> np.ndarray:
    """Per-trial, per-selected-channel feature -> (n_trials, n_sel)."""
    m = window_mask(times, spec.tmin, spec.tmax)
    seg = X[:, ch_idx][:, :, m]                       # (n, n_sel, n_win)
    twin = times[m]
    f = spec.feature
    if f == "mean_amplitude":
        return seg.mean(axis=2)
    if f == "auc":
        s = np.abs(seg) if spec.rectify_auc else seg
        return trapezoid(s, twin, axis=2)
    if f == "gfp":
        # spatial std across selected channels per timepoint, mean over window
        return np.repeat(seg.std(axis=1).mean(axis=1, keepdims=True),
                         seg.shape[1], axis=1)
    if f in ("peak_amplitude", "peak_latency"):
        if spec.peak_sign == "pos":
            arg = np.argmax(seg, axis=2)
        elif spec.peak_sign == "neg":
            arg = np.argmin(seg, axis=2)
        else:
            arg = np.argmax(np.abs(seg), axis=2)
        if f == "peak_latency":
            return twin[arg]
        return np.take_along_axis(seg, arg[:, :, None], axis=2)[:, :, 0]
    raise ValueError(f"Unknown ERP feature {f!r}")


def erp_feature(bundle, spec: ERPFeatureSpec) -> np.ndarray:
    """Per-trial scalar (ROI-averaged) -> (n_trials,)."""
    ch_idx = bundle.roi_indices(spec.channels)
    per_ch = _erp_per_channel(bundle.X, bundle.times, spec, ch_idx)
    return per_ch.mean(axis=1)


def erp_feature_per_channel(bundle, spec: ERPFeatureSpec):
    """Per-trial, per-channel features for ML -> (values, names)."""
    ch_idx = bundle.roi_indices(spec.channels)
    per_ch = _erp_per_channel(bundle.X, bundle.times, spec, ch_idx)
    names = [f"{bundle.channel_names[i]}:{spec.feature}" for i in ch_idx]
    return per_ch, names


# ----- spectral ---------------------------------------------------------- #

def _band_power_per_channel(X: np.ndarray, sfreq: float, spec: BandPowerSpec,
                            ch_idx: list[int]) -> np.ndarray:
    seg = X[:, ch_idx]                                 # (n, n_sel, n_times)
    nperseg = spec.nperseg or min(seg.shape[2], int(sfreq))
    nperseg = max(16, min(nperseg, seg.shape[2]))
    freqs, psd = signal.welch(seg, fs=sfreq, nperseg=nperseg, axis=2)
    band = (freqs >= spec.fmin) & (freqs <= spec.fmax)
    if not band.any():
        band[np.argmin(np.abs(freqs - 0.5 * (spec.fmin + spec.fmax)))] = True
    power = trapezoid(psd[:, :, band], freqs[band], axis=2)
    if spec.relative:
        tot = (freqs >= spec.total_band[0]) & (freqs <= spec.total_band[1])
        denom = trapezoid(psd[:, :, tot], freqs[tot], axis=2)
        power = power / np.where(denom > 0, denom, np.nan)
    if spec.log:
        power = 10.0 * np.log10(np.clip(power, 1e-20, None))
    return power


def band_power_feature(bundle, spec: BandPowerSpec) -> np.ndarray:
    ch_idx = bundle.roi_indices(spec.channels)
    return _band_power_per_channel(bundle.X, bundle.sfreq, spec, ch_idx).mean(axis=1)


def band_power_feature_per_channel(bundle, spec: BandPowerSpec):
    ch_idx = bundle.roi_indices(spec.channels)
    per_ch = _band_power_per_channel(bundle.X, bundle.sfreq, spec, ch_idx)
    names = [f"{bundle.channel_names[i]}:{spec.label()}" for i in ch_idx]
    return per_ch, names


# ----- dispatch + grouping ----------------------------------------------- #

def compute_feature(bundle, spec) -> np.ndarray:
    """Per-trial scalar for an ERP or band-power spec -> (n_trials,)."""
    if isinstance(spec, ERPFeatureSpec):
        return erp_feature(bundle, spec)
    if isinstance(spec, BandPowerSpec):
        return band_power_feature(bundle, spec)
    raise TypeError(f"Unsupported feature spec: {type(spec).__name__}")


def metadata_feature(bundle, key: str) -> np.ndarray:
    """A behavioural/metadata column as a per-trial feature."""
    if key not in bundle.metadata:
        raise KeyError(f"No metadata column {key!r}. "
                       f"Available: {sorted(bundle.metadata)}")
    return np.asarray(bundle.metadata[key], dtype=float)


def feature_by_condition(values: np.ndarray, bundle) -> dict[str, np.ndarray]:
    """Split a per-trial feature vector into ``{condition_name: values}``."""
    return {name: values[bundle.y == i]
            for i, name in enumerate(bundle.condition_names)}
