"""Artifact detection.

Detection here is *transparent*: it produces a list of events, each with
the channel, a numeric value and a human-readable reason, so the user can
always inspect *why* something was flagged. Detection never silently
modifies the signal -- removal/correction (ASR/ICA) is a separate, opt-in
concern handled with fitted transforms in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal

from neurobci.core.electrodes import is_frontal
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    REJECT = "reject"


@dataclass
class ArtifactThresholds:
    """Documented, tunable detection limits (microvolts unless noted)."""

    flat_std_uv: float = 0.7          # below => flat / disconnected
    amplitude_uv: float = 120.0       # peak amplitude => likely artifact
    rail_uv: float = 200.0            # |x| above => clipping/railing
    rail_fraction: float = 0.02       # fraction railing to reject
    gradient_uv: float = 40.0         # |x[n]-x[n-1]| jump => electrode pop/shift
    muscle_hf_ratio: float = 0.5      # >40 Hz power fraction => EMG
    line_ratio: float = 0.35          # line-band power fraction => mains
    blink_uv: float = 60.0            # EOG excursion counted as a blink
    line_freq_hz: float = 50.0
    hf_cutoff_hz: float = 40.0


@dataclass
class ArtifactEvent:
    scope: str               # "channel" | "global"
    kind: str
    severity: Severity
    value: float
    message: str
    channel: str | None = None


@dataclass
class ArtifactReport:
    events: list[ArtifactEvent] = field(default_factory=list)
    n_samples: int = 0
    sfreq: float = 0.0

    @property
    def bad_channels(self) -> list[str]:
        return sorted(
            {e.channel for e in self.events
             if e.scope == "channel" and e.severity is Severity.REJECT and e.channel}
        )

    @property
    def n_blinks(self) -> int:
        for e in self.events:
            if e.kind == "blinks" and e.scope == "global":
                return int(e.value)
        return sum(int(e.value) for e in self.events if e.kind == "blinks")

    @property
    def blink_events(self) -> list[ArtifactEvent]:
        """Eye-blink detections (per-channel counts and the global total)."""
        return [e for e in self.events if e.kind == "blinks"]

    def for_channel(self, name: str) -> list[ArtifactEvent]:
        return [e for e in self.events if e.channel == name]


def _band_ratio(psd, freqs, lo, hi):
    total = float(np.sum(psd)) + 1e-12
    return float(np.sum(psd[(freqs >= lo) & (freqs < hi)])) / total


def _count_blinks(x: np.ndarray, thr: float) -> int:
    """Count threshold-exceeding excursions (rising edges of |x| > thr)."""
    above = np.abs(x) > thr
    if above.size == 0:
        return 0
    rising = np.logical_and(above[1:], ~above[:-1])
    return int(np.count_nonzero(rising)) + int(above[0])


def detect_artifacts(
    data: np.ndarray,
    info: StreamInfo,
    thresholds: ArtifactThresholds | None = None,
) -> ArtifactReport:
    """Scan a window of samples and return an :class:`ArtifactReport`."""

    th = thresholds or ArtifactThresholds()
    report = ArtifactReport(n_samples=0 if data.ndim != 2 else data.shape[0],
                            sfreq=info.sfreq)
    if data.ndim != 2 or data.shape[0] < 8:
        return report

    n = data.shape[0]
    # Missing samples (NaN) -> global.
    n_nan = int(np.count_nonzero(~np.isfinite(data)))
    if n_nan:
        report.events.append(ArtifactEvent(
            "global", "missing_samples", Severity.WARN, n_nan,
            f"{n_nan} non-finite samples in window."))

    # Remove each channel's DC offset before amplitude/rail/blink checks: a
    # large baseline (common in unfiltered recordings — electrode half-cell
    # potentials of hundreds/thousands of uV) is not an artifact, and without
    # this every channel would falsely trip the clipping/amplitude limits.
    clean = np.nan_to_num(data)
    clean = clean - clean.mean(axis=0, keepdims=True)
    nperseg = int(min(n, max(64, info.sfreq)))
    freqs, psd = signal.welch(clean, fs=info.sfreq, nperseg=nperseg, axis=0)

    for ci in range(info.n_channels):
        name = info.channel_names[ci]
        is_eog = info.channel_kinds[ci] == KIND_EOG
        x = clean[:, ci]
        std = float(np.std(x))
        maxabs = float(np.max(np.abs(x)))
        maxgrad = float(np.max(np.abs(np.diff(x)))) if n > 1 else 0.0
        rail_frac = float(np.mean(np.abs(x) > th.rail_uv))

        if std < th.flat_std_uv:
            report.events.append(ArtifactEvent(
                "channel", "flat", Severity.REJECT, std,
                f"flat/disconnected (std {std:.2f} uV)", name))
            continue  # other metrics meaningless on a flat channel

        if rail_frac > th.rail_fraction:
            report.events.append(ArtifactEvent(
                "channel", "clipping", Severity.REJECT, rail_frac,
                f"clipping/railing ({rail_frac*100:.0f}% > {th.rail_uv} uV)", name))
        if maxgrad > th.gradient_uv:
            report.events.append(ArtifactEvent(
                "channel", "gradient", Severity.WARN, maxgrad,
                f"sudden jump {maxgrad:.0f} uV/sample (electrode pop/shift)", name))
        if not is_eog and maxabs > th.amplitude_uv:
            report.events.append(ArtifactEvent(
                "channel", "amplitude", Severity.WARN, maxabs,
                f"high amplitude {maxabs:.0f} uV", name))

        hf = _band_ratio(psd[:, ci], freqs, th.hf_cutoff_hz, info.sfreq / 2)
        line = _band_ratio(psd[:, ci], freqs, th.line_freq_hz - 2, th.line_freq_hz + 2)
        if not is_eog and hf > th.muscle_hf_ratio:
            report.events.append(ArtifactEvent(
                "channel", "muscle", Severity.WARN, hf,
                f"muscle/EMG (HF {hf*100:.0f}%)", name))
        if line > th.line_ratio:
            report.events.append(ArtifactEvent(
                "channel", "line_noise", Severity.WARN, line,
                f"line noise ({line*100:.0f}%)", name))

    # Eye-blinks: count per source channel, then a global total. Prefer the
    # dedicated EOG channel(s); when a montage has none, fall back to frontal
    # scalp electrodes (Fp/AF/F…) where blinks project most strongly, so the
    # detector still works on EOG-less caps.
    blink_idx = list(info.eog_indices)
    source = "EOG"
    if not blink_idx:
        blink_idx = [i for i in range(info.n_channels)
                     if info.channel_kinds[i] == KIND_EEG
                     and is_frontal(info.channel_names[i])]
        source = "frontal EEG"

    if blink_idx:
        blink_total = 0
        for ci in blink_idx:
            c = _count_blinks(clean[:, ci], th.blink_uv)
            blink_total += c
            if c:
                report.events.append(ArtifactEvent(
                    "channel", "blinks", Severity.INFO, c,
                    f"{c} blink(s) (|x| > {th.blink_uv:.0f} uV)",
                    info.channel_names[ci]))
        suffix = "" if source == "EOG" else " (no EOG channel; frontal EEG)"
        report.events.append(ArtifactEvent(
            "global", "blinks", Severity.INFO, blink_total,
            f"{blink_total} eye-blink(s) on {source}{suffix}."))

    return report
