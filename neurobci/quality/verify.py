"""Is this data trustworthy? An honest, explainable verdict.

``verify_window`` / ``verify_recording`` inspect a window (or a whole recorded
session) and return a :class:`VerifyReport` with a blunt verdict --
**TRUST**, **CAUTION** or **UNTRUSTWORTHY** -- plus the *reasons*. Nothing is
hidden and no result is invented: every downgrade names the metric that caused
it. This is the tool to answer "did my Enobio recording actually work, and can
I believe the indices/decoding computed from it?".

It deliberately overlaps with :mod:`neurobci.quality.metrics` (per-channel
ratings) and adds whole-recording checks: saturation, non-finite samples,
montage-wide amplitude plausibility and a coarse PSD sanity check (does the
spectrum look like EEG -- 1/f falloff -- or like broadband/line junk?).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal

from neurobci.core.stream_info import KIND_EEG, StreamInfo
from neurobci.quality.metrics import (
    QualityRating,
    QualityThresholds,
    compute_quality,
    dc_robust,
)


class DataTrust(str, Enum):
    TRUST = "trust"
    CAUTION = "caution"
    UNTRUSTWORTHY = "untrustworthy"
    NODATA = "nodata"


_ORDER = {DataTrust.TRUST: 0, DataTrust.CAUTION: 1,
          DataTrust.UNTRUSTWORTHY: 2, DataTrust.NODATA: 0}


@dataclass
class VerifyReport:
    verdict: DataTrust
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    bad_channels: list[str] = field(default_factory=list)
    n_samples: int = 0
    sfreq: float = 0.0

    def summary(self) -> str:
        head = f"DATA VERDICT: {self.verdict.value.upper()}  " \
               f"({self.n_samples:,} samples @ {self.sfreq:.0f} Hz)"
        lines = [head, "-" * len(head)]
        for r in self.reasons:
            lines.append(f"  - {r}")
        if self.bad_channels:
            lines.append(f"  bad channels: {', '.join(self.bad_channels)}")
        return "\n".join(lines)


@dataclass
class VerifyThresholds:
    rail_uv: float = 200.0             # |x| above this counts as saturation
    sat_fraction_caution: float = 0.002
    sat_fraction_bad: float = 0.02
    nan_fraction_bad: float = 0.0005
    bad_channel_fraction_caution: float = 0.15
    bad_channel_fraction_bad: float = 0.40
    eeg_median_std_lo: float = 1.5     # plausible EEG amplitude band (uV)
    eeg_median_std_hi: float = 60.0
    onef_ratio_caution: float = 1.0    # low-band/high-band power; <1 is suspicious


def _stream_info_from(channel_names, channel_kinds, sfreq) -> StreamInfo:
    return StreamInfo(name="verify", sfreq=sfreq,
                      channel_names=list(channel_names),
                      channel_kinds=list(channel_kinds), source_kind="replay")


def verify_window(
    data: np.ndarray,
    info: StreamInfo,
    qthresh: QualityThresholds | None = None,
    vthresh: VerifyThresholds | None = None,
) -> VerifyReport:
    """Verdict for one ``(n_samples, n_channels)`` window (microvolts)."""
    vt = vthresh or VerifyThresholds()
    n = 0 if data.ndim != 2 else data.shape[0]
    if n < 16:
        return VerifyReport(DataTrust.NODATA, ["Not enough samples."],
                            n_samples=n, sfreq=info.sfreq)

    eeg = np.array(info.eeg_indices, dtype=int)
    reasons: list[str] = []
    verdict = DataTrust.TRUST

    def _downgrade(to: DataTrust):
        nonlocal verdict
        if _ORDER[to] > _ORDER[verdict]:
            verdict = to

    # 1) Non-finite samples -------------------------------------------------
    nan_frac = float(np.mean(~np.isfinite(data)))
    if nan_frac > 0:
        msg = f"{nan_frac*100:.3f}% non-finite (NaN/Inf) samples"
        if nan_frac > vt.nan_fraction_bad:
            _downgrade(DataTrust.UNTRUSTWORTHY); reasons.append("UNTRUSTWORTHY: " + msg)
        else:
            _downgrade(DataTrust.CAUTION); reasons.append("CAUTION: " + msg)

    finite = np.nan_to_num(data)
    # Judge amplitude/saturation/spectrum on the AC signal (raw EEG carries a
    # large DC offset + drift that the high-pass removes and is not a fault).
    ac = dc_robust(finite, info.sfreq)

    # 2) Saturation / railing on EEG ---------------------------------------
    sat_frac = float(np.mean(np.abs(ac[:, eeg]) > vt.rail_uv)) if eeg.size else 0.0
    if sat_frac > vt.sat_fraction_bad:
        _downgrade(DataTrust.UNTRUSTWORTHY)
        reasons.append(f"UNTRUSTWORTHY: {sat_frac*100:.2f}% of EEG samples saturate "
                       f"(>|{vt.rail_uv:.0f}| uV) -- electrodes railing")
    elif sat_frac > vt.sat_fraction_caution:
        _downgrade(DataTrust.CAUTION)
        reasons.append(f"CAUTION: {sat_frac*100:.2f}% of EEG samples saturate "
                       f"(>|{vt.rail_uv:.0f}| uV)")

    # 3) Per-channel quality -----------------------------------------------
    qrep = compute_quality(finite, info, qthresh)
    bad = [c.name for c in qrep.channels
           if c.kind == KIND_EEG and c.rating == QualityRating.BAD]
    n_eeg = max(eeg.size, 1)
    bad_frac = len(bad) / n_eeg
    if bad_frac > vt.bad_channel_fraction_bad:
        _downgrade(DataTrust.UNTRUSTWORTHY)
        reasons.append(f"UNTRUSTWORTHY: {len(bad)}/{n_eeg} EEG channels unusable")
    elif bad_frac > vt.bad_channel_fraction_caution:
        _downgrade(DataTrust.CAUTION)
        reasons.append(f"CAUTION: {len(bad)}/{n_eeg} EEG channels unusable")

    # 4) Amplitude plausibility (median EEG std) ---------------------------
    med_std = float(np.median(np.std(ac[:, eeg], axis=0))) if eeg.size else 0.0
    if med_std < vt.eeg_median_std_lo:
        _downgrade(DataTrust.UNTRUSTWORTHY)
        reasons.append(f"UNTRUSTWORTHY: median EEG std {med_std:.1f} uV is implausibly "
                       f"low (flat / disconnected)")
    elif med_std > vt.eeg_median_std_hi:
        _downgrade(DataTrust.CAUTION)
        reasons.append(f"CAUTION: median EEG std {med_std:.0f} uV is high "
                       f"(noisy / contaminated)")

    # 5) Spectrum sanity: does it fall off like 1/f? -----------------------
    onef = float("nan")
    if eeg.size:
        nperseg = int(min(n, max(64, info.sfreq)))
        f, psd = signal.welch(ac[:, eeg], fs=info.sfreq, nperseg=nperseg, axis=0)
        mean_psd = psd.mean(axis=1)
        lo = mean_psd[(f >= 2) & (f < 7)].mean()
        hi = mean_psd[(f >= 20) & (f < 40)].mean() + 1e-12
        onef = float(lo / hi)
        if onef < vt.onef_ratio_caution:
            _downgrade(DataTrust.CAUTION)
            reasons.append(f"CAUTION: spectrum does not fall off like EEG "
                           f"(low/high power ratio {onef:.2f} < 1) -- broadband/EMG?")

    if not reasons:
        reasons.append("All checks passed: finite, non-saturating, plausible "
                       "amplitudes and a 1/f-like spectrum.")

    return VerifyReport(
        verdict=verdict, reasons=reasons,
        checks={"nan_fraction": nan_frac, "saturation_fraction": sat_frac,
                "median_eeg_std_uv": med_std, "onef_ratio": onef,
                "overall_quality": qrep.overall_rating.value},
        bad_channels=bad, n_samples=n, sfreq=info.sfreq,
    )


def verify_recording(
    data: np.ndarray,
    channel_names: list[str],
    channel_kinds: list[str],
    sfreq: float,
    window_s: float = 4.0,
    qthresh: QualityThresholds | None = None,
    vthresh: VerifyThresholds | None = None,
) -> VerifyReport:
    """Verify a whole recording by aggregating over non-overlapping windows.

    The overall verdict is the **worst** window verdict; reasons summarise how
    often each problem occurred so a brief glitch is not conflated with a
    pervasive one.
    """
    info = _stream_info_from(channel_names, channel_kinds, sfreq)
    n = data.shape[0]
    win = max(int(window_s * sfreq), 16)
    if n < win:
        return verify_window(data, info, qthresh, vthresh)

    verdicts: list[DataTrust] = []
    sat = []
    bad_counter: dict[str, int] = {}
    nan_total = 0.0
    reports = []
    for s in range(0, n - win + 1, win):
        rep = verify_window(data[s:s + win], info, qthresh, vthresh)
        reports.append(rep)
        verdicts.append(rep.verdict)
        sat.append(rep.checks.get("saturation_fraction", 0.0))
        nan_total += rep.checks.get("nan_fraction", 0.0)
        for b in rep.bad_channels:
            bad_counter[b] = bad_counter.get(b, 0) + 1

    nwin = len(reports)
    worst = max(verdicts, key=lambda v: _ORDER[v])
    n_flagged = sum(v != DataTrust.TRUST for v in verdicts)
    persistent_bad = [b for b, c in sorted(bad_counter.items(), key=lambda kv: -kv[1])
                      if c >= 0.5 * nwin]

    reasons = [f"{nwin} windows of {window_s:.0f}s checked; "
               f"{n_flagged} flagged ({100*n_flagged/nwin:.0f}%)."]
    reasons.append(f"Mean EEG saturation: {np.mean(sat)*100:.2f}% of samples.")
    if persistent_bad:
        reasons.append(f"Persistently-bad channels (>=50% of windows): "
                       f"{', '.join(persistent_bad)}.")
    # Surface the dominant reason from the worst windows.
    for rep in reports:
        if rep.verdict == worst and rep.reasons:
            reasons.append("Example: " + rep.reasons[0])
            break

    return VerifyReport(
        verdict=worst, reasons=reasons,
        checks={"n_windows": nwin, "flagged_windows": n_flagged,
                "mean_saturation_fraction": float(np.mean(sat)),
                "total_nan_fraction": nan_total / max(nwin, 1)},
        bad_channels=persistent_bad, n_samples=n, sfreq=sfreq,
    )
