"""Motor-imagery paradigm (active BCI).

The user imagines left- vs right-hand movement, producing a contralateral
sensorimotor mu/beta power decrease (event-related desynchronisation).
Decoded from band-power spatial patterns (CSP / Riemannian) rather than
ERPs, and driven continuously through the streaming decision layer.
"""

from __future__ import annotations

from neurobci.paradigms.base import EpochWindow, Paradigm, register_paradigm

LEFT = "left"
RIGHT = "right"


@register_paradigm
class MotorImageryParadigm(Paradigm):
    name = "motor_imagery"
    family = "active"

    @property
    def class_labels(self) -> list[str]:
        return [LEFT, RIGHT]

    @property
    def positive_label(self) -> str:
        return RIGHT

    @property
    def window(self) -> EpochWindow:
        # An imagery window well after cue onset; no baseline (band power).
        return EpochWindow(tmin=0.5, tmax=2.5, baseline=None, reject_uv=200.0)

    @property
    def model_names(self) -> list[str]:
        return ["csp_lda", "cov_ts_lr"]
