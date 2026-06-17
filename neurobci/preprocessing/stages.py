"""Preprocessing stages.

Each stage transforms a ``(n_samples, n_channels)`` block. Two execution
paths are kept strictly separate, because conflating them is a classic
source of bad BCI results:

* **Causal / online** (:meth:`process_chunk`, :meth:`apply`) -- never uses
  future samples. IIR filters carry their state across chunks, so feeding
  the signal in pieces yields *exactly* the same result as filtering it
  whole. This is what the live system uses.
* **Offline** (:meth:`apply_offline`) -- may use the whole window, e.g.
  zero-phase ``filtfilt`` or linear detrending. Only for analysis of
  already-recorded data, never for real-time decisions.

Every stage exposes :meth:`describe` (parameter docs for the UI) and
:meth:`validate` (scientifically questionable settings -> warnings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from neurobci.core.stream_info import KIND_EEG


@dataclass
class ParamSpec:
    name: str
    kind: str            # "float" | "int" | "choice"
    default: Any
    doc: str
    min: float | None = None
    max: float | None = None
    choices: tuple = ()


class ProcessingStage:
    """Base class for all stages."""

    type_name: str = "base"
    realtime_safe: bool = True
    # Stages that learn from data (ICA, ASR, bad-channel detection) set this.
    # They must pass the signal through unchanged until :meth:`fit` is called,
    # so an un-calibrated stage can never silently corrupt the live stream.
    requires_fit: bool = False
    PARAM_SPECS: tuple[ParamSpec, ...] = ()

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        self.enabled = enabled
        self.params: dict[str, Any] = {s.name: s.default for s in self.PARAM_SPECS}
        if params:
            self.params.update({k: v for k, v in params.items() if k in self.params})
        self._sfreq = 0.0
        self._ch_kinds: list[str] = []
        self._ch_names: list[str] = []
        self.fitted: bool = False
        self.fit_summary: str = ""

    # ----- lifecycle ----------------------------------------------------- #

    def fit(self, data: np.ndarray) -> None:
        """Learn parameters from a calibration window (no-op by default).

        ``data`` is the signal *as it reaches this stage* (i.e. already
        transformed by earlier stages). Override in fit-requiring stages.
        """

    def prepare(
        self, sfreq: float, ch_kinds: list[str], ch_names: list[str] | None = None
    ) -> None:
        """Design filters / cache indices for the given stream.

        ``ch_names`` is optional: only montage-aware stages (e.g. the
        Laplacian) need electrode names. Stages that do not receive names
        simply degrade to a pass-through rather than failing.
        """
        self._sfreq = sfreq
        self._ch_kinds = list(ch_kinds)
        self._ch_names = list(ch_names) if ch_names else []

    def reset(self) -> None:
        """Clear any streaming state."""

    # ----- processing (override as needed) ------------------------------- #

    def process_chunk(self, data: np.ndarray) -> np.ndarray:
        """Stateful causal processing across consecutive chunks."""
        return self.apply(data)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Stateless causal processing of a standalone window."""
        return data

    def apply_offline(self, data: np.ndarray) -> np.ndarray:
        """Non-causal processing of a standalone window (default: causal)."""
        return self.apply(data)

    # ----- introspection ------------------------------------------------- #

    def describe(self) -> dict:
        return {
            "type": self.type_name,
            "enabled": self.enabled,
            "realtime_safe": self.realtime_safe,
            "requires_fit": self.requires_fit,
            "fitted": self.fitted,
            "doc": (self.__doc__ or "").strip().split("\n")[0],
            "params": [
                {
                    "name": s.name, "kind": s.kind, "value": self.params[s.name],
                    "default": s.default, "min": s.min, "max": s.max,
                    "choices": s.choices, "doc": s.doc,
                }
                for s in self.PARAM_SPECS
            ],
        }

    def to_dict(self) -> dict:
        return {"type": self.type_name, "enabled": self.enabled, "params": dict(self.params)}

    def validate(self, sfreq: float) -> list[str]:
        return []


# --------------------------------------------------------------------------- #
# IIR filter base (Butterworth / notch) with cross-chunk state
# --------------------------------------------------------------------------- #


class _IIRStage(ProcessingStage):
    """Shared machinery for IIR filters with continuous streaming state."""

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        super().__init__(enabled, params)
        self._b = np.array([1.0])
        self._a = np.array([1.0])
        self._zi_unit = np.zeros(0)
        self._state: np.ndarray | None = None

    def _design(self, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def prepare(
        self, sfreq: float, ch_kinds: list[str], ch_names: list[str] | None = None
    ) -> None:
        super().prepare(sfreq, ch_kinds, ch_names)
        self._b, self._a = self._design(sfreq)
        self._zi_unit = signal.lfilter_zi(self._b, self._a)
        self._state = None

    def reset(self) -> None:
        self._state = None

    def apply(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0:
            return data
        zi = self._zi_unit[:, None] * data[0][None, :]
        out, _ = signal.lfilter(self._b, self._a, data, axis=0, zi=zi)
        return out

    def process_chunk(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0:
            return data
        if self._state is None:
            self._state = self._zi_unit[:, None] * data[0][None, :]
        out, self._state = signal.lfilter(
            self._b, self._a, data, axis=0, zi=self._state
        )
        return out

    def apply_offline(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] <= 3 * (max(len(self._a), len(self._b))):
            return self.apply(data)
        return signal.filtfilt(self._b, self._a, data, axis=0)


# --------------------------------------------------------------------------- #
# Concrete stages
# --------------------------------------------------------------------------- #


class HighPass(_IIRStage):
    """High-pass Butterworth filter (removes drift / DC offset)."""

    type_name = "highpass"
    PARAM_SPECS = (
        ParamSpec("cutoff_hz", "float", 0.5, "Cutoff frequency (Hz).", 0.01, 100.0),
        ParamSpec("order", "int", 4, "Filter order.", 1, 8),
    )

    def _design(self, sfreq):
        nyq = sfreq / 2.0
        wc = min(self.params["cutoff_hz"] / nyq, 0.99)
        return signal.butter(int(self.params["order"]), wc, btype="highpass")

    def validate(self, sfreq):
        msgs = []
        if self.params["cutoff_hz"] >= sfreq / 2:
            msgs.append("High-pass cutoff is at/above Nyquist.")
        if self.params["cutoff_hz"] <= 0:
            msgs.append("High-pass cutoff must be positive.")
        return msgs


class LowPass(_IIRStage):
    """Low-pass Butterworth filter (anti-aliasing / EMG suppression)."""

    type_name = "lowpass"
    PARAM_SPECS = (
        ParamSpec("cutoff_hz", "float", 40.0, "Cutoff frequency (Hz).", 1.0, 1000.0),
        ParamSpec("order", "int", 4, "Filter order.", 1, 8),
    )

    def _design(self, sfreq):
        nyq = sfreq / 2.0
        wc = min(self.params["cutoff_hz"] / nyq, 0.99)
        return signal.butter(int(self.params["order"]), wc, btype="lowpass")

    def validate(self, sfreq):
        if self.params["cutoff_hz"] >= sfreq / 2:
            return ["Low-pass cutoff is at/above Nyquist."]
        return []


class BandPass(_IIRStage):
    """Band-pass Butterworth filter."""

    type_name = "bandpass"
    PARAM_SPECS = (
        ParamSpec("low_hz", "float", 1.0, "Low edge (Hz).", 0.1, 1000.0),
        ParamSpec("high_hz", "float", 40.0, "High edge (Hz).", 1.0, 1000.0),
        ParamSpec("order", "int", 4, "Filter order (per edge).", 1, 8),
    )

    def _design(self, sfreq):
        nyq = sfreq / 2.0
        lo = max(self.params["low_hz"] / nyq, 1e-4)
        hi = min(self.params["high_hz"] / nyq, 0.99)
        return signal.butter(int(self.params["order"]), [lo, hi], btype="bandpass")

    def validate(self, sfreq):
        msgs = []
        if self.params["low_hz"] >= self.params["high_hz"]:
            msgs.append("Band-pass low edge >= high edge.")
        if self.params["high_hz"] >= sfreq / 2:
            msgs.append("Band-pass high edge is at/above Nyquist.")
        return msgs


class Notch(_IIRStage):
    """IIR notch filter (removes mains line noise at a single frequency)."""

    type_name = "notch"
    PARAM_SPECS = (
        ParamSpec("freq_hz", "float", 50.0, "Notch frequency, e.g. 50/60 Hz.", 1.0, 1000.0),
        ParamSpec("quality", "float", 30.0, "Quality factor Q (sharpness).", 1.0, 100.0),
    )

    def _design(self, sfreq):
        return signal.iirnotch(self.params["freq_hz"], self.params["quality"], fs=sfreq)

    def validate(self, sfreq):
        if self.params["freq_hz"] >= sfreq / 2:
            return ["Notch frequency is at/above Nyquist."]
        return []


class CommonAverageReference(ProcessingStage):
    """Common Average Reference (subtract the mean of EEG channels).

    EOG / non-EEG channels are excluded from the average and left
    untouched. Has no temporal state, so it is trivially causal.
    """

    type_name = "car"

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        super().__init__(enabled, params)
        self._eeg_idx: np.ndarray = np.zeros(0, dtype=int)

    def prepare(
        self, sfreq: float, ch_kinds: list[str], ch_names: list[str] | None = None
    ) -> None:
        super().prepare(sfreq, ch_kinds, ch_names)
        self._eeg_idx = np.array(
            [i for i, k in enumerate(ch_kinds) if k == KIND_EEG], dtype=int
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0 or self._eeg_idx.size == 0:
            return data
        out = data.copy()
        ref = data[:, self._eeg_idx].mean(axis=1, keepdims=True)
        out[:, self._eeg_idx] -= ref
        return out

    def validate(self, sfreq):
        if self._sfreq and self._eeg_idx.size < 2:
            return ["CAR needs >=2 EEG channels to be meaningful."]
        return []


class Detrend(ProcessingStage):
    """Linear detrend (offline only -- needs the whole window, not causal)."""

    type_name = "detrend"
    realtime_safe = False

    def apply(self, data: np.ndarray) -> np.ndarray:
        # In a causal context we cannot truly detrend; pass through. The
        # pipeline skips non-realtime-safe stages in causal mode anyway.
        return data

    def apply_offline(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] < 2:
            return data
        return signal.detrend(data, axis=0, type="linear")


class BandStop(_IIRStage):
    """Band-stop (reject) Butterworth filter.

    Removes a *band* of frequencies (a wider alternative to the
    single-frequency notch, e.g. to reject a noisy sub-band).
    """

    type_name = "bandstop"
    PARAM_SPECS = (
        ParamSpec("low_hz", "float", 48.0, "Low edge of rejected band (Hz).", 0.1, 1000.0),
        ParamSpec("high_hz", "float", 52.0, "High edge of rejected band (Hz).", 1.0, 1000.0),
        ParamSpec("order", "int", 4, "Filter order (per edge).", 1, 8),
    )

    def _design(self, sfreq):
        nyq = sfreq / 2.0
        lo = max(self.params["low_hz"] / nyq, 1e-4)
        hi = min(self.params["high_hz"] / nyq, 0.99)
        return signal.butter(int(self.params["order"]), [lo, hi], btype="bandstop")

    def validate(self, sfreq):
        msgs = []
        if self.params["low_hz"] >= self.params["high_hz"]:
            msgs.append("Band-stop low edge >= high edge.")
        if self.params["high_hz"] >= sfreq / 2:
            msgs.append("Band-stop high edge is at/above Nyquist.")
        return msgs


class MovingAverage(_IIRStage):
    """Moving-average smoother (FIR low-pass over a short window).

    A simple linear-phase smoother implemented as an FIR filter, so chunked
    streaming matches whole-signal filtering exactly (same machinery as the
    IIR stages). Complements the steeper Butterworth low-pass.
    """

    type_name = "moving_average"
    PARAM_SPECS = (
        ParamSpec("window_ms", "float", 20.0, "Averaging window length (ms).", 1.0, 2000.0),
    )

    def _window_samples(self, sfreq: float) -> int:
        return max(1, int(round(self.params["window_ms"] * sfreq / 1000.0)))

    def _design(self, sfreq):
        n = self._window_samples(sfreq)
        return np.ones(n) / n, np.array([1.0])

    def validate(self, sfreq):
        if sfreq and self._window_samples(sfreq) < 2:
            return ["Moving-average window < 2 samples: no smoothing applied."]
        return []


class AmplitudeClamp(ProcessingStage):
    """Hard amplitude limiter -- clamp samples to +/- a threshold (uV).

    Bounds extreme transients (e.g. movement spikes) so a few huge samples
    cannot dominate downstream scaling or feature statistics. Pointwise, so
    it is trivially causal and has no temporal state.
    """

    type_name = "clamp"
    PARAM_SPECS = (
        ParamSpec("limit_uv", "float", 200.0, "Clamp magnitude (uV).", 1.0, 100000.0),
    )

    def apply(self, data: np.ndarray) -> np.ndarray:
        lim = float(self.params["limit_uv"])
        return np.clip(data, -lim, lim)

    def validate(self, sfreq):
        if self.params["limit_uv"] <= 0:
            return ["Clamp limit must be positive."]
        return []


class RobustReference(ProcessingStage):
    """Robust (median) re-reference across EEG channels.

    Like CAR but subtracts the per-sample *median* of the EEG channels, so a
    few high-amplitude/bad channels do not bias the reference. EOG /
    non-EEG channels are excluded and left untouched. No temporal state.
    """

    type_name = "robust_ref"

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        super().__init__(enabled, params)
        self._eeg_idx: np.ndarray = np.zeros(0, dtype=int)

    def prepare(
        self, sfreq: float, ch_kinds: list[str], ch_names: list[str] | None = None
    ) -> None:
        super().prepare(sfreq, ch_kinds, ch_names)
        self._eeg_idx = np.array(
            [i for i, k in enumerate(ch_kinds) if k == KIND_EEG], dtype=int
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0 or self._eeg_idx.size == 0:
            return data
        out = data.copy()
        ref = np.median(data[:, self._eeg_idx], axis=1, keepdims=True)
        out[:, self._eeg_idx] -= ref
        return out

    def validate(self, sfreq):
        if self._sfreq and self._eeg_idx.size < 2:
            return ["Robust reference needs >=2 EEG channels to be meaningful."]
        return []


class Laplacian(ProcessingStage):
    """Surface Laplacian (Hjorth) spatial filter.

    Re-references each scalp EEG channel against a distance-weighted average
    of its nearest neighbours, sharpening focal activity and suppressing
    spatially-broad signals -- a montage-aware alternative to CAR. Needs
    10-20 channel positions: channels without a known position (EOG, or when
    names are unavailable) pass through unchanged. No temporal state.
    """

    type_name = "laplacian"
    PARAM_SPECS = (
        ParamSpec("neighbors", "int", 4, "Number of nearest neighbours.", 1, 8),
    )

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        super().__init__(enabled, params)
        self._W: np.ndarray = np.zeros((0, 0))
        self._lap_idx: np.ndarray = np.zeros(0, dtype=int)

    def prepare(
        self, sfreq: float, ch_kinds: list[str], ch_names: list[str] | None = None
    ) -> None:
        super().prepare(sfreq, ch_kinds, ch_names)
        self._build_weights(ch_kinds, self._ch_names)

    def _build_weights(self, ch_kinds: list[str], ch_names: list[str]) -> None:
        n = len(ch_kinds)
        self._W = np.zeros((n, n))
        self._lap_idx = np.zeros(0, dtype=int)
        if not ch_names or len(ch_names) != n:
            return
        # Lazy import keeps the preprocessing package self-contained at import
        # time and avoids pulling in the spectral package unless a Laplacian
        # stage is actually used.
        from neurobci.spectral.topo import channel_positions_2d

        pos, found = channel_positions_2d(list(ch_names))
        usable = np.array(
            [bool(found[i]) and ch_kinds[i] == KIND_EEG for i in range(n)], dtype=bool
        )
        idx = np.where(usable)[0]
        if idx.size < 2:
            return
        k = max(1, int(self.params["neighbors"]))
        lap = []
        for i in idx:
            others = idx[idx != i]
            d = np.linalg.norm(pos[others] - pos[i], axis=1)
            order = np.argsort(d)[:k]
            sel = others[order]
            w = 1.0 / np.maximum(d[order], 1e-6)
            self._W[i, sel] = w / w.sum()
            lap.append(i)
        self._lap_idx = np.array(lap, dtype=int)

    def apply(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0 or self._lap_idx.size == 0:
            return data
        out = data.copy()
        neighbour = data @ self._W.T               # (n_samples, n_channels)
        out[:, self._lap_idx] = data[:, self._lap_idx] - neighbour[:, self._lap_idx]
        return out

    def validate(self, sfreq):
        if self._sfreq and self._lap_idx.size == 0:
            return ["Laplacian inactive: 10-20 channel positions unavailable "
                    "(needs electrode names); passing through unchanged."]
        return []


class Standardize(ProcessingStage):
    """Per-channel amplitude standardisation (z-score).

    Causal mode uses an exponential moving mean/variance (only past & present
    samples); offline mode uses the whole-window mean/std. Puts channels on a
    comparable scale before some ML features. NOTE: the output is in
    standard-deviation units, *not* microvolts.
    """

    type_name = "standardize"
    PARAM_SPECS = (
        ParamSpec("halflife_s", "float", 2.0,
                  "Causal EWMA half-life (s); smaller = more adaptive.", 0.05, 60.0),
    )
    _EPS = 1e-8

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        super().__init__(enabled, params)
        self._b = np.array([1.0])
        self._a = np.array([1.0])
        self._zi = np.zeros(0)
        self._m_state: np.ndarray | None = None
        self._sq_state: np.ndarray | None = None

    def prepare(
        self, sfreq: float, ch_kinds: list[str], ch_names: list[str] | None = None
    ) -> None:
        super().prepare(sfreq, ch_kinds, ch_names)
        halflife_samples = max(self.params["halflife_s"] * sfreq, 1e-6)
        alpha = 1.0 - 0.5 ** (1.0 / halflife_samples)
        self._b = np.array([alpha])
        self._a = np.array([1.0, -(1.0 - alpha)])
        self._zi = signal.lfilter_zi(self._b, self._a)
        self._m_state = None
        self._sq_state = None

    def reset(self) -> None:
        self._m_state = None
        self._sq_state = None

    def _ewma(self, data, state):
        if state is None:
            state = self._zi[:, None] * data[0][None, :]
        out, state = signal.lfilter(self._b, self._a, data, axis=0, zi=state)
        return out, state

    def _zscore(self, data, m, ms):
        std = np.sqrt(np.clip(ms - m**2, 0.0, None))
        return (data - m) / (std + self._EPS)

    def apply(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0:
            return data
        m, _ = self._ewma(data, None)
        ms, _ = self._ewma(data**2, None)
        return self._zscore(data, m, ms)

    def process_chunk(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0:
            return data
        m, self._m_state = self._ewma(data, self._m_state)
        ms, self._sq_state = self._ewma(data**2, self._sq_state)
        return self._zscore(data, m, ms)

    def apply_offline(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] < 2:
            return data
        m = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        return (data - m) / (std + self._EPS)


class SavitzkyGolay(ProcessingStage):
    """Savitzky-Golay smoothing (offline only).

    Fits a low-order polynomial in a sliding *centred* window -- smooths
    while preserving peak shape better than a moving average. Because the
    window is centred (uses future samples) it is offline-only and is
    skipped in causal mode.
    """

    type_name = "savgol"
    realtime_safe = False
    PARAM_SPECS = (
        ParamSpec("window_ms", "float", 50.0, "Window length (ms; forced odd).", 3.0, 2000.0),
        ParamSpec("polyorder", "int", 3, "Polynomial order.", 1, 6),
    )

    def apply(self, data: np.ndarray) -> np.ndarray:
        return data  # not causal; skipped in causal mode anyway

    def apply_offline(self, data: np.ndarray) -> np.ndarray:
        t = data.shape[0]
        if t < 5:
            return data
        n = int(round(self.params["window_ms"] * self._sfreq / 1000.0))
        n = max(n, 5)
        if n % 2 == 0:
            n += 1
        max_odd = t if t % 2 == 1 else t - 1
        n = min(n, max_odd)
        poly = min(int(self.params["polyorder"]), n - 1)
        if poly < 1:
            return data
        return signal.savgol_filter(data, n, poly, axis=0)

    def validate(self, sfreq):
        if self.params["polyorder"] < 1:
            return ["Savitzky-Golay polyorder must be >= 1."]
        return []


# Registry: stage type name -> class.
STAGE_REGISTRY: dict[str, type[ProcessingStage]] = {
    cls.type_name: cls
    for cls in (
        HighPass, LowPass, BandPass, BandStop, Notch,
        CommonAverageReference, RobustReference, Laplacian,
        AmplitudeClamp, MovingAverage, Standardize, Detrend, SavitzkyGolay,
    )
}


def make_stage(spec: dict) -> ProcessingStage:
    """Instantiate a stage from a ``{type, enabled, params}`` dict."""
    stype = spec.get("type")
    if stype not in STAGE_REGISTRY:
        raise ValueError(f"Unknown preprocessing stage type: {stype!r}")
    return STAGE_REGISTRY[stype](
        enabled=spec.get("enabled", True), params=spec.get("params", {})
    )
