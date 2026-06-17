"""Exploratory cognitive-state indices.

These are *proxies*, not measurements. The platform's contract (section 16
of the spec) is enforced here: every index exposes its mathematical
definition, the exact channels used, the raw band-power components behind
the ratio, a validity flag (contaminated/insufficient channels), an
optional baseline-relative value, and a standing caveat that generic
indices require experimental validation for a given task.

A supervised cognitive-state *classifier* (which would need labelled data)
is deliberately NOT provided here -- only transparent, inspectable ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurobci.acquisition.default_montages import region_of
from neurobci.core.stream_info import KIND_EEG, StreamInfo

CAVEAT = ("Exploratory proxy, not a direct measure of mental state. "
          "Requires experimental validation for your task and participant.")


@dataclass
class CognitiveIndex:
    name: str
    value: float
    definition: str
    channels: list[str]
    components: dict[str, float]      # raw band powers behind the ratio
    valid: bool
    note: str = CAVEAT
    baseline: float | None = None

    @property
    def relative(self) -> float | None:
        if self.baseline is None or not np.isfinite(self.baseline) or self.baseline == 0:
            return None
        return self.value / self.baseline


def _channels_in(info: StreamInfo, regions: set[str], bad: set[int]) -> list[int]:
    return [
        i for i in range(info.n_channels)
        if info.channel_kinds[i] == KIND_EEG and i not in bad
        and region_of(info.channel_names[i]) in regions
    ]


def _mean_power(powers: dict[str, np.ndarray], band: str, idx: list[int]) -> float:
    if not idx:
        return float("nan")
    return float(np.mean(powers[band][idx]))


def _names(info: StreamInfo, idx: list[int]) -> list[str]:
    return [info.channel_names[i] for i in idx]


def compute_indices(
    powers: dict[str, np.ndarray],
    info: StreamInfo,
    bad_channels: set[int] | None = None,
    baseline: dict[str, float] | None = None,
) -> list[CognitiveIndex]:
    """Compute the exploratory indices from absolute band powers."""

    bad = bad_channels or set()
    baseline = baseline or {}
    out: list[CognitiveIndex] = []

    # ----- engagement: beta / (alpha + theta)  (Pope et al., 1995) ------- #
    eng_idx = _channels_in(info, {"frontal", "central", "parietal"}, bad)
    a = _mean_power(powers, "alpha", eng_idx)
    t = _mean_power(powers, "theta", eng_idx)
    b = _mean_power(powers, "beta", eng_idx)
    eng_val = b / (a + t) if np.isfinite(a + t) and (a + t) > 0 else float("nan")
    out.append(CognitiveIndex(
        name="engagement",
        value=eng_val,
        definition="beta / (alpha + theta)   [Pope et al. 1995]",
        channels=_names(info, eng_idx),
        components={"alpha": a, "theta": t, "beta": b},
        valid=len(eng_idx) >= 3 and np.isfinite(eng_val),
        baseline=baseline.get("engagement"),
    ))

    # ----- workload: frontal theta / parietal alpha (Gevins) ------------- #
    front = _channels_in(info, {"frontal"}, bad)
    pari = _channels_in(info, {"parietal", "occipital"}, bad)
    tf = _mean_power(powers, "theta", front)
    ap = _mean_power(powers, "alpha", pari)
    wl_val = tf / ap if np.isfinite(ap) and ap > 0 else float("nan")
    out.append(CognitiveIndex(
        name="workload",
        value=wl_val,
        definition="theta(frontal) / alpha(parietal)   [Gevins et al.]",
        channels=_names(info, front) + _names(info, pari),
        components={"theta_frontal": tf, "alpha_parietal": ap},
        valid=len(front) >= 1 and len(pari) >= 1 and np.isfinite(wl_val),
        baseline=baseline.get("workload"),
    ))

    # ----- drowsiness / fatigue: (theta + alpha) / beta ------------------ #
    allc = _channels_in(info, {"frontal", "central", "parietal", "occipital", "temporal"}, bad)
    ta = _mean_power(powers, "theta", allc)
    aa = _mean_power(powers, "alpha", allc)
    ba = _mean_power(powers, "beta", allc)
    dr_val = (ta + aa) / ba if np.isfinite(ba) and ba > 0 else float("nan")
    out.append(CognitiveIndex(
        name="drowsiness",
        value=dr_val,
        definition="(theta + alpha) / beta",
        channels=_names(info, allc),
        components={"theta": ta, "alpha": aa, "beta": ba},
        valid=len(allc) >= 3 and np.isfinite(dr_val),
        baseline=baseline.get("drowsiness"),
    ))

    return out


def index_values(indices: list[CognitiveIndex]) -> dict[str, float]:
    """Convenience: ``{name: value}`` for baseline capture / history."""
    return {ix.name: ix.value for ix in indices}
