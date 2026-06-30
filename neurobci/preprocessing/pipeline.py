"""Ordered preprocessing pipeline.

Builds a list of :class:`ProcessingStage` objects from the JSON-friendly
configuration (``list[{type, enabled, params}]``) and runs them in order.

Two run modes:

* ``causal`` -- the real-time path. Non-real-time-safe stages (e.g. linear
  detrend) are **skipped** and reported, and IIR filters keep state across
  chunks via :meth:`process_chunk`.
* ``offline`` -- analysis path. All enabled stages run, using their
  non-causal implementations (zero-phase filtering, detrending, ...).

The pipeline never mixes the two: a real-time consumer calls
:meth:`process_chunk`; an offline/visualisation consumer calls
:meth:`apply_window`.
"""

from __future__ import annotations

import logging

import numpy as np

from neurobci.preprocessing.stages import ProcessingStage, make_stage

logger = logging.getLogger(__name__)

MODE_CAUSAL = "causal"
MODE_OFFLINE = "offline"


class Pipeline:
    def __init__(
        self,
        stages: list[ProcessingStage],
        sfreq: float,
        ch_kinds: list[str],
        enabled: bool = True,
        mode: str = MODE_CAUSAL,
        ch_names: list[str] | None = None,
    ) -> None:
        self.stages = stages
        self.enabled = enabled
        self.mode = mode
        self.sfreq = sfreq
        self.ch_kinds = list(ch_kinds)
        self.ch_names = list(ch_names) if ch_names else []
        for st in self.stages:
            st.prepare(sfreq, ch_kinds, self.ch_names)

    # ----- construction -------------------------------------------------- #

    @classmethod
    def from_config(
        cls,
        pre_config,
        sfreq: float,
        ch_kinds: list[str],
        ch_names: list[str] | None = None,
    ) -> "Pipeline":
        stages = [make_stage(s) for s in pre_config.stages]
        return cls(
            stages=stages,
            sfreq=sfreq,
            ch_kinds=ch_kinds,
            enabled=pre_config.enabled,
            mode=pre_config.mode,
            ch_names=ch_names,
        )

    def to_config_stages(self) -> list[dict]:
        return [st.to_dict() for st in self.stages]

    def matches_montage(
        self, sfreq: float, ch_names: list[str], *, tol: float = 1e-3
    ) -> bool:
        """True if this pipeline was prepared for this exact stream layout.

        A calibrated stage (ICA/ASR/bad-channel) learns a transform tied to a
        specific sampling rate and channel set, so a fitted pipeline may only
        be reused on data with the same montage. Used by the offline ERP tab
        to decide whether the live, calibrated pipeline applies to a loaded
        session (otherwise it rebuilds from config, uncalibrated).
        """
        if abs(float(self.sfreq) - float(sfreq)) > tol:
            return False
        return list(self.ch_names) == list(ch_names)

    # ----- active-stage selection ---------------------------------------- #

    def _active(self, mode: str) -> list[ProcessingStage]:
        out = []
        for st in self.stages:
            if not st.enabled:
                continue
            if mode == MODE_CAUSAL and not st.realtime_safe:
                continue  # skipped in real time (reported via skipped_in_causal)
            out.append(st)
        return out

    @property
    def skipped_in_causal(self) -> list[str]:
        return [
            st.type_name for st in self.stages
            if st.enabled and not st.realtime_safe
        ]

    # ----- execution ----------------------------------------------------- #

    def reset(self) -> None:
        for st in self.stages:
            st.reset()

    # ----- calibration (fit-requiring stages: ICA / ASR / bad channels) --- #

    @property
    def requires_fit(self) -> bool:
        """True if any enabled stage needs calibration before it does anything."""
        return any(st.enabled and st.requires_fit for st in self.stages)

    @property
    def fitted(self) -> bool:
        """True if every enabled fit-requiring stage has been calibrated."""
        return all(
            st.fitted for st in self.stages if st.enabled and st.requires_fit
        )

    def fit(self, data: np.ndarray) -> list[str]:
        """Calibrate every fit-requiring stage on a window of raw samples.

        Each fit-requiring stage is fit on the signal *as it reaches it* --
        i.e. earlier stages are applied (causally) first -- so e.g. ICA is
        calibrated on filtered data, exactly as it will run online. Returns a
        per-stage summary of what was learned.
        """
        x = np.asarray(data, dtype=np.float64)
        summaries: list[str] = []
        for st in self.stages:
            if not st.enabled:
                continue
            if not st.realtime_safe:          # offline-only stage: skip in this chain
                continue
            if st.requires_fit:
                try:
                    st.fit(x)
                    summaries.append(f"{st.type_name}: {st.fit_summary or 'fitted'}")
                except Exception as exc:       # noqa: BLE001 - report, do not crash
                    logger.exception("Stage %s fit failed.", st.type_name)
                    summaries.append(f"{st.type_name}: FIT FAILED ({exc})")
            x = st.apply(x)
        return summaries

    def process_chunk(self, data: np.ndarray) -> np.ndarray:
        """Real-time path: stateful, causal, never uses future samples."""
        if not self.enabled:
            return data
        x = np.asarray(data, dtype=np.float64)
        for st in self._active(MODE_CAUSAL):
            x = st.process_chunk(x)
        return x

    def apply_window(self, data: np.ndarray, mode: str | None = None) -> np.ndarray:
        """Standalone-window path (visualisation / offline). Stateless."""
        if not self.enabled:
            return np.asarray(data, dtype=np.float64)
        mode = mode or self.mode
        x = np.asarray(data, dtype=np.float64)
        for st in self._active(mode):
            x = st.apply_offline(x) if mode == MODE_OFFLINE else st.apply(x)
        return x

    # ----- editing ------------------------------------------------------- #

    def move(self, index: int, delta: int) -> None:
        j = index + delta
        if 0 <= index < len(self.stages) and 0 <= j < len(self.stages):
            self.stages[index], self.stages[j] = self.stages[j], self.stages[index]

    def set_enabled(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self.stages):
            self.stages[index].enabled = enabled

    def insert(self, index: int, stage: ProcessingStage) -> None:
        """Insert an already-built stage, preparing it for this stream."""
        stage.prepare(self.sfreq, self.ch_kinds, self.ch_names)
        index = max(0, min(index, len(self.stages)))
        self.stages.insert(index, stage)

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.stages):
            self.stages.pop(index)

    # ----- introspection ------------------------------------------------- #

    def describe(self) -> list[dict]:
        return [st.describe() for st in self.stages]

    def validate(self) -> list[str]:
        """Aggregate per-stage warnings plus cross-stage sanity checks."""
        warnings: list[str] = []
        for st in self.stages:
            if st.enabled:
                warnings.extend(st.validate(self.sfreq))

        # Cross-stage: a high-pass cutoff above a low-pass cutoff removes
        # everything -- a classic mistake.
        hp = self._first_enabled("highpass")
        lp = self._first_enabled("lowpass")
        if hp and lp and hp.params["cutoff_hz"] >= lp.params["cutoff_hz"]:
            warnings.append(
                f"High-pass cutoff ({hp.params['cutoff_hz']} Hz) >= low-pass "
                f"cutoff ({lp.params['cutoff_hz']} Hz): the pass-band is empty."
            )
        return warnings

    def _first_enabled(self, type_name: str) -> ProcessingStage | None:
        for st in self.stages:
            if st.enabled and st.type_name == type_name:
                return st
        return None
