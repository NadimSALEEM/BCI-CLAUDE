"""Live signal-quality metrics.

Everything here is *explainable*: each channel gets numeric metrics plus a
list of human-readable reasons for its rating. We never invent impedance
values (the Enobio LSL stream does not provide them); quality is inferred
only from the signal itself.

The metrics operate on a recent window copied out of the ring buffer and
are deliberately cheap so they can run every UI refresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal

from neurobci.core.stream_info import KIND_EOG, StreamInfo


class QualityRating(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    BAD = "bad"          # unusable (flat / railing / disconnected)
    NODATA = "nodata"


@dataclass
class QualityThresholds:
    """Tunable, documented limits. Microvolt units unless noted."""

    flat_std_uv: float = 0.7          # below this std => flatline / disconnected
    high_std_uv: float = 50.0         # above this std => excessively noisy
    extreme_std_uv: float = 120.0     # gross contamination
    rail_uv: float = 200.0            # |x| above this counts as railing
    rail_fraction: float = 0.02       # fraction railing to flag
    line_ratio_warn: float = 0.20     # line power / total to warn
    line_ratio_bad: float = 0.45
    hf_ratio_warn: float = 0.45       # >40 Hz power / total (muscle/EMG)
    line_freq_hz: float = 50.0
    hf_cutoff_hz: float = 40.0


@dataclass
class ChannelQuality:
    name: str
    kind: str
    rating: QualityRating
    rms_uv: float
    std_uv: float
    ptp_uv: float
    line_ratio: float
    hf_ratio: float
    rail_fraction: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    channels: list[ChannelQuality]
    n_samples: int
    sfreq: float

    @property
    def overall_rating(self) -> QualityRating:
        if not self.channels:
            return QualityRating.NODATA
        order = [
            QualityRating.GOOD,
            QualityRating.FAIR,
            QualityRating.POOR,
            QualityRating.BAD,
        ]
        worst = QualityRating.GOOD
        # Overall reflects the EEG channels' worst common case, but a single
        # bad channel shouldn't condemn the whole montage -> use the median.
        ratings = [c.rating for c in self.channels if c.rating in order]
        if not ratings:
            return QualityRating.NODATA
        idxs = sorted(order.index(r) for r in ratings)
        median = idxs[len(idxs) // 2]
        return order[median]

    @property
    def n_bad_channels(self) -> int:
        return sum(c.rating == QualityRating.BAD for c in self.channels)


def _band_ratio(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    total = float(np.sum(psd)) + 1e-12
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sum(psd[mask])) / total


def dc_robust(data: np.ndarray, sfreq: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """Remove DC offset + slow drift so amplitude/railing reflect the *AC*
    signal. Raw EEG (especially dry Enobio) carries huge electrode offsets and
    drift -- tens of thousands of microvolts -- that are **not** faults and are
    removed by the mandatory high-pass anyway. Judging "railing" on absolute
    amplitude without this falsely condemns perfectly good raw data.
    """
    if data.ndim != 2 or data.shape[0] < 32:
        return data - np.mean(data, axis=0, keepdims=True)
    wc = max(cutoff_hz / (sfreq / 2.0), 1e-4)
    b, a = signal.butter(2, wc, btype="highpass")
    try:
        return signal.filtfilt(b, a, data, axis=0)
    except Exception:  # noqa: BLE001 - short/odd windows: fall back to mean removal
        return data - np.mean(data, axis=0, keepdims=True)


def compute_quality(
    data: np.ndarray,
    info: StreamInfo,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """Compute a :class:`QualityReport` from a window of samples.

    ``data`` is ``(n_samples, n_channels)`` in microvolts.
    """

    th = thresholds or QualityThresholds()
    n = 0 if data.ndim != 2 else data.shape[0]
    if n < 8:
        return QualityReport(channels=[], n_samples=n, sfreq=info.sfreq)

    sfreq = info.sfreq
    # Assess amplitude/railing/spectrum on the DC+drift-removed signal so a
    # normal raw electrode offset is never mistaken for a railing electrode.
    ac = dc_robust(data, sfreq)
    # One Welch estimate for the whole array (per-column), if long enough.
    nperseg = int(min(n, max(64, sfreq)))  # ~1 s or the whole window
    freqs, psd = signal.welch(ac, fs=sfreq, nperseg=nperseg, axis=0)

    channels: list[ChannelQuality] = []
    for ci in range(info.n_channels):
        x = ac[:, ci]
        std = float(np.std(x))
        rms = float(np.sqrt(np.mean(x**2)))
        ptp = float(np.ptp(x))
        rail_frac = float(np.mean(np.abs(x) > th.rail_uv))
        line = _band_ratio(
            psd[:, ci], freqs, th.line_freq_hz - 2, th.line_freq_hz + 2
        )
        hf = _band_ratio(psd[:, ci], freqs, th.hf_cutoff_hz, sfreq / 2)

        reasons: list[str] = []
        rating = QualityRating.GOOD
        is_eog = info.channel_kinds[ci] == KIND_EOG

        if std < th.flat_std_uv:
            rating = QualityRating.BAD
            reasons.append(f"flatline (std {std:.2f} <{th.flat_std_uv} uV)")
        elif rail_frac > th.rail_fraction:
            rating = QualityRating.BAD
            reasons.append(f"railing ({rail_frac*100:.0f}% > {th.rail_uv} uV)")
        else:
            if std > th.extreme_std_uv:
                rating = QualityRating.BAD
                reasons.append(f"extreme noise (std {std:.0f} uV)")
            elif std > th.high_std_uv:
                rating = _worse(rating, QualityRating.POOR)
                reasons.append(f"high noise (std {std:.0f} uV)")
            if line > th.line_ratio_bad:
                rating = _worse(rating, QualityRating.POOR)
                reasons.append(f"strong line noise ({line*100:.0f}%)")
            elif line > th.line_ratio_warn:
                rating = _worse(rating, QualityRating.FAIR)
                reasons.append(f"line noise ({line*100:.0f}%)")
            # EOG is *expected* to have large low-freq content; don't penalise
            # it for HF the same way, and never for amplitude.
            if not is_eog and hf > th.hf_ratio_warn:
                rating = _worse(rating, QualityRating.FAIR)
                reasons.append(f"high-frequency/EMG ({hf*100:.0f}%)")

        if not reasons:
            reasons.append("ok")
        channels.append(
            ChannelQuality(
                name=info.channel_names[ci],
                kind=info.channel_kinds[ci],
                rating=rating,
                rms_uv=rms,
                std_uv=std,
                ptp_uv=ptp,
                line_ratio=line,
                hf_ratio=hf,
                rail_fraction=rail_frac,
                reasons=reasons,
            )
        )

    return QualityReport(channels=channels, n_samples=n, sfreq=sfreq)


_SEVERITY = {
    QualityRating.GOOD: 0,
    QualityRating.FAIR: 1,
    QualityRating.POOR: 2,
    QualityRating.BAD: 3,
}


def _worse(a: QualityRating, b: QualityRating) -> QualityRating:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b
