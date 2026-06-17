"""Configurable artifact injection for replay / testing.

Overlays controllable artifacts (line noise, eye-blinks, amplitude bursts,
flat channels, drift) onto a sample chunk. Used by replay to stress-test the
preprocessing / quality / artifact-detection path on otherwise-clean
recorded data, and available as a general utility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurobci.acquisition.default_montages import region_of
from neurobci.core.stream_info import KIND_EOG, StreamInfo


@dataclass
class ArtifactInjectionConfig:
    enabled: bool = False
    line_uv: float = 0.0
    line_freq_hz: float = 50.0
    blink_rate_hz: float = 0.0
    blink_amp_uv: float = 80.0
    burst_rate_hz: float = 0.0
    burst_amp_uv: float = 150.0
    drift_uv: float = 0.0
    flat_channels: list[str] = field(default_factory=list)


class ArtifactInjector:
    def __init__(self, info: StreamInfo, config: ArtifactInjectionConfig, seed: int = 0) -> None:
        self.info = info
        self.config = config
        self._rng = np.random.default_rng(seed)
        names = info.channel_names
        self._eog = set(info.eog_indices)
        self._frontal = [i for i, n in enumerate(names)
                         if region_of(n) == "frontal" and i not in self._eog]
        idx = {n: i for i, n in enumerate(names)}
        self._flat = [idx[c] for c in config.flat_channels if c in idx]

    def inject(self, chunk: np.ndarray, start_global: int) -> np.ndarray:
        c = self.config
        if not c.enabled or chunk.shape[0] == 0:
            return chunk
        out = chunk.astype(np.float64, copy=True)
        m, nch = out.shape
        sf = self.info.sfreq
        t = (start_global + np.arange(m)) / sf

        if c.line_uv > 0:
            out += c.line_uv * np.sin(2 * np.pi * c.line_freq_hz * t)[:, None]

        if c.drift_uv > 0:
            out += c.drift_uv * np.sin(2 * np.pi * 0.1 * t)[:, None]

        if c.blink_rate_hz > 0:
            self._add_events(out, m, sf, c.blink_rate_hz,
                             self._blink_targets(), c.blink_amp_uv, dur=0.25)

        if c.burst_rate_hz > 0:
            target = [int(self._rng.integers(0, nch))]
            self._add_events(out, m, sf, c.burst_rate_hz, target, c.burst_amp_uv,
                             dur=0.1, noisy=True)

        for i in self._flat:
            out[:, i] = 0.3 * self._rng.standard_normal(m)
        return out.astype(chunk.dtype)

    def _blink_targets(self) -> list[int]:
        return sorted(self._eog) + self._frontal

    def _add_events(self, out, m, sf, rate, channels, amp, dur, noisy=False) -> None:
        if not channels:
            return
        expected = rate * (m / sf)
        n_events = self._rng.poisson(expected)
        klen = max(int(dur * sf), 2)
        x = np.linspace(0, 1, klen)
        kernel = (x ** 2) * np.exp(-6 * x)
        kernel /= kernel.max() if kernel.max() > 0 else 1.0
        for _ in range(int(n_events)):
            onset = int(self._rng.integers(0, m))
            end = min(onset + klen, m)
            seg = kernel[: end - onset]
            for ch in channels:
                w = 1.0 if ch in self._eog else 0.5
                contrib = amp * w * seg
                if noisy:
                    contrib = amp * self._rng.standard_normal(end - onset)
                out[onset:end, ch] += contrib
