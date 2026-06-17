"""Spectral analyzer: ties PSD, band power, indices and baselines together.

One :class:`SpectralAnalyzer` instance (held by the GUI) is fed the latest
window each refresh. It produces a :class:`SpectralReport`, keeps a short
time-history of band powers and indices for the temporal plot, and supports
capturing a baseline so later windows can be shown as change-from-baseline.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from neurobci.config.schema import SpectralConfig
from neurobci.core.stream_info import StreamInfo
from neurobci.spectral import psd as _psd
from neurobci.spectral.indices import CognitiveIndex, compute_indices, index_values


@dataclass
class SpectralReport:
    freqs: np.ndarray
    psd: np.ndarray                          # (n_freqs, n_channels)
    abs_powers: dict[str, np.ndarray]
    rel_powers: dict[str, np.ndarray]
    regional: dict[str, dict[str, float]]    # band -> region -> mean abs power
    iaf: float
    indices: list[CognitiveIndex]
    bad_channels: list[int]
    info: StreamInfo
    sfreq: float


class SpectralAnalyzer:
    def __init__(self, info: StreamInfo, config: SpectralConfig | None = None,
                 bands: dict | None = None, history: int = 300) -> None:
        self.info = info
        self.config = config or SpectralConfig()
        self.bands = bands or _psd.BANDS
        self._baseline_indices: dict[str, float] | None = None
        self._baseline_abs: dict[str, np.ndarray] | None = None
        self._history: deque = deque(maxlen=history)
        self._t0 = time.time()

    # ----- analysis ------------------------------------------------------ #

    def analyze(self, data: np.ndarray, bad_names: list[str] | None = None) -> SpectralReport:
        bad_idx = {self.info.channel_names.index(n)
                   for n in (bad_names or []) if n in self.info.channel_names}

        freqs, psd = _psd.compute_psd(
            data, self.info.sfreq, self.config.nperseg_s, self.config.fmax_hz)
        abs_p = _psd.band_powers(freqs, psd, self.bands)
        rel_p = _psd.relative_band_powers(freqs, psd, self.bands)
        regional = {
            band: _psd.regional_band_power(abs_p[band], self.info, bad_idx)
            for band in self.bands
        }
        iaf = _psd.individual_alpha_frequency(freqs, psd, self.info)
        indices = compute_indices(abs_p, self.info, bad_idx, self._baseline_indices)

        self._record_history(indices, abs_p, bad_idx)
        return SpectralReport(
            freqs=freqs, psd=psd, abs_powers=abs_p, rel_powers=rel_p,
            regional=regional, iaf=iaf, indices=indices,
            bad_channels=sorted(bad_idx), info=self.info, sfreq=self.info.sfreq,
        )

    # ----- baseline ------------------------------------------------------ #

    def set_baseline(self, report: SpectralReport) -> None:
        self._baseline_indices = index_values(report.indices)
        self._baseline_abs = {b: report.abs_powers[b].copy() for b in report.abs_powers}

    def clear_baseline(self) -> None:
        self._baseline_indices = None
        self._baseline_abs = None

    @property
    def has_baseline(self) -> bool:
        return self._baseline_abs is not None

    # ----- topomap values ------------------------------------------------ #

    def topomap_values(self, report: SpectralReport, band: str, mode: str) -> np.ndarray:
        """Return per-channel values for a scalp map.

        mode: 'absolute' | 'relative' | 'baseline' (dB change vs baseline).
        Bad channels are set to NaN so they are excluded from interpolation.
        """
        if mode == "relative":
            vals = report.rel_powers[band].copy()
        elif mode == "baseline" and self._baseline_abs is not None:
            base = self._baseline_abs[band]
            vals = 10.0 * np.log10((report.abs_powers[band] + 1e-20) / (base + 1e-20))
        else:
            vals = report.abs_powers[band].copy()
        for i in report.bad_channels:
            vals[i] = np.nan
        return vals

    # ----- history ------------------------------------------------------- #

    def _record_history(self, indices, abs_p, bad_idx) -> None:
        good = [i for i in range(self.info.n_channels) if i not in bad_idx]
        band_means = {b: float(np.mean(abs_p[b][good])) if good else float("nan")
                      for b in abs_p}
        self._history.append({
            "t": time.time() - self._t0,
            "indices": index_values(indices),
            "bands": band_means,
        })

    def history_series(self, kind: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t, values)`` for a history series.

        kind: 'indices' or 'bands'; key: the index/band name.
        """
        t = np.array([h["t"] for h in self._history])
        v = np.array([h[kind].get(key, np.nan) for h in self._history])
        return t, v
