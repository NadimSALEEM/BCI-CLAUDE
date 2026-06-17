"""SSVEP paradigm (reactive, frequency-tagged).

Each selectable target flickers at a distinct frequency; attending a target
entrains an occipital oscillation at that frequency (and harmonics). Decoded
with Canonical Correlation Analysis against reference sinusoids -- a
training-light approach that needs no per-class classifier fit.
"""

from __future__ import annotations

from neurobci.paradigms.base import EpochWindow, Paradigm, register_paradigm


@register_paradigm
class SSVEPParadigm(Paradigm):
    name = "ssvep"
    family = "reactive"

    frequencies: tuple[float, ...] = (8.0, 10.0, 12.0, 15.0)
    n_harmonics: int = 2

    @property
    def class_labels(self) -> list[str]:
        return [f"{f:g}Hz" for f in self.frequencies]

    @property
    def positive_label(self) -> str:
        return self.class_labels[0]

    @property
    def window(self) -> EpochWindow:
        # A few seconds of steady-state response; no baseline / rejection off.
        return EpochWindow(tmin=0.0, tmax=2.0, baseline=None, reject_uv=1e9)

    @property
    def model_names(self) -> list[str]:
        return ["cca"]
