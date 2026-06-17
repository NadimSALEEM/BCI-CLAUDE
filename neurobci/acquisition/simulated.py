"""Synthetic EEG source.

Generates continuous, spatially-structured multichannel EEG so the whole
application can run without hardware. The signal contains posterior alpha,
theta, beta, 1/f-ish background, line noise, slow drift and eye-blinks,
plus optional per-channel faults (flat / noisy) for robustness testing.

Design note -- *content* vs *pacing* are separated:

* :meth:`generate` produces exactly ``n`` samples and advances internal
  state deterministically (given the seed). Tests use it for known
  ground truth.
* :meth:`read` is the real-time interface: it works out how many samples
  *should* exist by now from the wall clock and calls :meth:`generate`.

All amplitudes are microvolts. Continuity across chunks is maintained via
stored IIR filter state and an absolute sample counter, so there are no
discontinuities at chunk boundaries.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from scipy import signal

from neurobci.acquisition.base import EEGSource
from neurobci.acquisition.default_montages import (
    ALPHA_REGION_WEIGHT,
    BLINK_REGION_WEIGHT,
    build_stream_info,
    region_of,
)
from neurobci.config.schema import ChannelConfig, SimulationConfig
from neurobci.core.stream_info import KIND_EOG, StreamInfo

# Economy pink-noise approximation IIR (filters white noise to ~1/f).
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])

# Centro-parietal weighting for an injected ERP (P300) by scalp region.
_ERP_REGION_WEIGHT = {
    "parietal": 1.0, "central": 0.95, "occipital": 0.6,
    "temporal": 0.45, "frontal": 0.35, "other": 0.5,
}


class SimulatedSource(EEGSource):
    def __init__(
        self,
        channels: ChannelConfig,
        sim: SimulationConfig,
        sfreq: float = 500.0,
        realtime: bool = True,
    ) -> None:
        self._channels = channels
        self._sim = sim
        self._sfreq = float(sfreq)
        self._realtime = realtime
        self._info = build_stream_info(channels, sfreq, source_kind="simulated")
        self._rng = np.random.default_rng(sim.seed)
        self._started = False

        nch = self._info.n_channels
        names = self._info.channel_names

        # Per-channel spatial weights.
        self._alpha_w = np.array(
            [ALPHA_REGION_WEIGHT[region_of(n)] for n in names], dtype=np.float64
        )
        blink_w = np.array(
            [BLINK_REGION_WEIGHT[region_of(n)] for n in names], dtype=np.float64
        )
        # Blink projects fully onto EOG, partially onto scalp (frontal-max).
        self._blink_proj = 0.5 * blink_w
        for i in self._info.eog_indices:
            self._blink_proj[i] = 1.0

        # Fault-injection channel index sets.
        name_to_idx = {n: i for i, n in enumerate(names)}
        self._flat_idx = [name_to_idx[c] for c in sim.flat_channels if c in name_to_idx]
        self._noisy_idx = [
            name_to_idx[c] for c in sim.noisy_channels if c in name_to_idx
        ]

        # Streaming filter state (continuity across chunks).
        self._pink_zi = np.zeros((max(len(_PINK_A), len(_PINK_B)) - 1, nch))
        self._drift_zi = np.zeros((1, nch))
        self._pink_gain = self._measure_gain(_PINK_B, _PINK_A)
        self._drift_gain = self._measure_gain([1.0], [1.0, -0.999])

        # Blink scheduling (absolute-sample aligned future contribution).
        self._blink_kernel = self._make_blink_kernel()
        self._blink_future = np.zeros(0, dtype=np.float64)
        self._next_blink = self._schedule_next_blink(0)

        # Externally-triggered ERP (P300) injection, for live calibration.
        self._erp_proj = np.array(
            [
                0.0 if i in set(self._info.eog_indices)
                else _ERP_REGION_WEIGHT[region_of(names[i])]
                for i in range(nch)
            ],
            dtype=np.float64,
        )
        self._erp_kernel = self._make_erp_kernel()
        self._erp_future = np.zeros(0, dtype=np.float64)
        self._erp_lock = threading.Lock()
        self._pending_erps: list[tuple[int, float]] = []

        self._n0 = 0                # absolute samples generated so far
        self._t0 = 0.0              # wall-clock start (set in start())

    # ----- EEGSource interface ------------------------------------------ #

    @property
    def info(self) -> StreamInfo:
        return self._info

    def start(self) -> None:
        if self._started:
            return
        self._t0 = time.time()
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._started:
            return self._empty()
        if not self._realtime:
            # Free-run: emit a modest fixed block each call.
            return self.generate(int(self._sfreq * 0.02) or 1)
        # Real-time: emit exactly enough to match elapsed wall-clock time.
        elapsed = time.time() - self._t0
        target = int(elapsed * self._sfreq)
        m = target - self._n0
        if m <= 0:
            return self._empty()
        # Guard against huge catch-ups after a stall (cap at 1 s).
        m = min(m, int(self._sfreq))
        return self.generate(m)

    # ----- deterministic generation ------------------------------------- #

    def generate(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate exactly ``n`` samples; advance state. Deterministic."""

        if n <= 0:
            return self._empty()
        nch = self._info.n_channels
        s = self._sim
        idx = np.arange(self._n0, self._n0 + n, dtype=np.float64)
        t = idx / self._sfreq

        # Shared oscillations (common across channels -> spatial correlation).
        alpha = np.sin(2 * np.pi * 10.0 * t)
        theta = np.sin(2 * np.pi * 6.0 * t + 0.5)
        beta = np.sin(2 * np.pi * 20.0 * t + 1.0)
        line = np.sin(2 * np.pi * s.line_freq_hz * t)

        data = (
            s.alpha_amp_uv * self._alpha_w[None, :] * alpha[:, None]
            + s.theta_amp_uv * theta[:, None]
            + s.beta_amp_uv * beta[:, None]
            + s.line_noise_uv * line[:, None]
        )

        # 1/f-ish background.
        white = self._rng.standard_normal((n, nch))
        pink, self._pink_zi = signal.lfilter(
            _PINK_B, _PINK_A, white, axis=0, zi=self._pink_zi
        )
        data += (s.pink_noise_uv / self._pink_gain) * pink
        data += s.white_noise_uv * self._rng.standard_normal((n, nch))

        # Slow drift (leaky integrator).
        drift_in = self._rng.standard_normal((n, nch))
        drift, self._drift_zi = signal.lfilter(
            [1.0], [1.0, -0.999], drift_in, axis=0, zi=self._drift_zi
        )
        data += (s.drift_uv / self._drift_gain) * drift

        # Eye-blinks.
        if s.blink_rate_hz > 0 and s.blink_amp_uv > 0:
            blink_unit = self._render_blinks(n)
            data += s.blink_amp_uv * blink_unit[:, None] * self._blink_proj[None, :]

        # Externally-injected ERPs (P300) for live calibration.
        erp_unit = self._render_erps(n)
        if erp_unit is not None:
            data += erp_unit[:, None] * self._erp_proj[None, :]

        # Fault injection.
        for i in self._flat_idx:
            data[:, i] = 0.5 * self._rng.standard_normal(n)   # near-flat
        for i in self._noisy_idx:
            data[:, i] += 40.0 * self._rng.standard_normal(n)  # very noisy

        ts = self._t0 + idx / self._sfreq if self._t0 else idx / self._sfreq
        self._n0 += n
        return data.astype(np.float32), ts.astype(np.float64)

    # ----- helpers ------------------------------------------------------- #

    def _empty(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.empty((0, self._info.n_channels), dtype=np.float32),
            np.empty(0, dtype=np.float64),
        )

    def _measure_gain(self, b, a) -> float:
        """Std-dev gain of an IIR applied to unit-variance white noise."""
        probe = np.random.default_rng(0).standard_normal(8000)
        out = signal.lfilter(b, a, probe)
        g = float(np.std(out))
        return g if g > 1e-9 else 1.0

    def inject_erp(self, amplitude_uv: float = 6.0) -> None:
        """Schedule a P300-like ERP to begin at the next generated sample.

        Used by live calibration: call this when a *target* stimulus is
        shown. Thread-safe (may be called from the UI/stimulus thread while
        the acquisition thread is generating).
        """
        with self._erp_lock:
            self._pending_erps.append((self._n0, float(amplitude_uv)))

    def _make_erp_kernel(self) -> np.ndarray:
        """Unit-amplitude P300 waveform over ~0.6 s (peak ~300 ms)."""
        n = max(int(0.6 * self._sfreq), 8)
        t = np.arange(n) / self._sfreq
        p300 = np.exp(-((t - 0.30) / 0.06) ** 2)
        n200 = -0.35 * np.exp(-((t - 0.20) / 0.04) ** 2)
        return p300 + n200

    def _render_erps(self, n: int) -> np.ndarray | None:
        """Return a length-``n`` ERP timeline (unit amplitude), or None."""
        with self._erp_lock:
            pending = self._pending_erps
            self._pending_erps = [p for p in pending if p[0] >= self._n0 + n]
            due = [p for p in pending if p[0] < self._n0 + n]
        if not due and self._erp_future.shape[0] == 0:
            return None
        future = self._erp_future
        if future.shape[0] < n:
            future = np.concatenate([future, np.zeros(n - future.shape[0])])
        klen = self._erp_kernel.shape[0]
        for onset, amp in due:
            offset = max(onset - self._n0, 0)
            need = offset + klen
            if future.shape[0] < need:
                future = np.concatenate([future, np.zeros(need - future.shape[0])])
            future[offset : offset + klen] += amp * self._erp_kernel
        out = future[:n].copy()
        self._erp_future = future[n:]
        return out

    def _make_blink_kernel(self) -> np.ndarray:
        """A smooth ~300 ms positive deflection (unit peak)."""
        dur = 0.3
        k = max(int(dur * self._sfreq), 4)
        x = np.linspace(0, 1, k)
        # Asymmetric bump: fast rise, slower decay.
        kernel = (x**2) * np.exp(-6.0 * x)
        peak = kernel.max()
        return kernel / peak if peak > 0 else kernel

    def _schedule_next_blink(self, from_sample: int) -> int:
        rate = self._sim.blink_rate_hz
        if rate <= 0:
            return np.iinfo(np.int64).max
        gap = self._rng.exponential(1.0 / rate)
        return from_sample + max(int(gap * self._sfreq), 1)

    def _render_blinks(self, n: int) -> np.ndarray:
        """Return a length-``n`` unit-amplitude blink timeline.

        Future blink tails are carried in ``self._blink_future`` aligned to
        the current absolute position, guaranteeing continuity.
        """
        future = self._blink_future
        if future.shape[0] < n:
            future = np.concatenate([future, np.zeros(n - future.shape[0])])
        klen = self._blink_kernel.shape[0]
        while self._next_blink < self._n0 + n:
            offset = self._next_blink - self._n0
            need = offset + klen
            if future.shape[0] < need:
                future = np.concatenate([future, np.zeros(need - future.shape[0])])
            future[offset : offset + klen] += self._blink_kernel
            self._next_blink = self._schedule_next_blink(self._next_blink)
        out = future[:n].copy()
        self._blink_future = future[n:]
        return out
