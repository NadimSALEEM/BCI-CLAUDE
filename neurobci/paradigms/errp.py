"""Error-related potential paradigm (supervisory error correction).

After the system performs (or previews) an action, the user's EEG may carry
an error-related potential (a fronto-central ERN/Pe complex) if they
perceive the action as wrong. An ErrP classifier estimates this, enabling
the platform to reject/undo/repeat an action. Calibrated and evaluated
separately from the primary BCI; not every user produces detectable ErrPs.
"""

from __future__ import annotations

from neurobci.paradigms.base import EpochWindow, Paradigm, register_paradigm

CORRECT = "correct"
ERROR = "error"


@register_paradigm
class ErrPParadigm(Paradigm):
    name = "errp"
    family = "reactive"

    @property
    def class_labels(self) -> list[str]:
        return [CORRECT, ERROR]

    @property
    def positive_label(self) -> str:
        return ERROR

    @property
    def window(self) -> EpochWindow:
        return EpochWindow(tmin=-0.1, tmax=0.8, baseline=(-0.1, 0.0), reject_uv=150.0)

    @property
    def model_names(self) -> list[str]:
        return ["vec_lda", "riemann_lr"]
