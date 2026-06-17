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
    PARAM_SPECS: tuple[ParamSpec, ...] = ()

    def __init__(self, enabled: bool = True, params: dict | None = None) -> None:
        self.enabled = enabled
        self.params: dict[str, Any] = {s.name: s.default for s in self.PARAM_SPECS}
        if params:
            self.params.update({k: v for k, v in params.items() if k in self.params})
        self._sfreq = 0.0
        self._ch_kinds: list[str] = []

    # ----- lifecycle ----------------------------------------------------- #

    def prepare(self, sfreq: float, ch_kinds: list[str]) -> None:
        """Design filters / cache indices for the given stream."""
        self._sfreq = sfreq
        self._ch_kinds = list(ch_kinds)

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

    def prepare(self, sfreq: float, ch_kinds: list[str]) -> None:
        super().prepare(sfreq, ch_kinds)
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

    def prepare(self, sfreq: float, ch_kinds: list[str]) -> None:
        super().prepare(sfreq, ch_kinds)
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


# Registry: stage type name -> class.
STAGE_REGISTRY: dict[str, type[ProcessingStage]] = {
    cls.type_name: cls
    for cls in (HighPass, LowPass, BandPass, Notch, CommonAverageReference, Detrend)
}


def make_stage(spec: dict) -> ProcessingStage:
    """Instantiate a stage from a ``{type, enabled, params}`` dict."""
    stype = spec.get("type")
    if stype not in STAGE_REGISTRY:
        raise ValueError(f"Unknown preprocessing stage type: {stype!r}")
    return STAGE_REGISTRY[stype](
        enabled=spec.get("enabled", True), params=spec.get("params", {})
    )
