"""Error-related-potential supervisory correction.

After the primary BCI performs (or previews) an action, the EEG response is
classified by an ErrP model. If an error potential is detected the action is
recommended for rejection (undo / repeat / confirm) instead of being
committed. This is a *secondary, optional* layer -- not every user produces
detectable ErrPs, so it must be calibrated and evaluated separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurobci.bci.models import ParadigmModel


@dataclass
class CorrectionDecision:
    is_error: bool
    error_prob: float
    action: str            # "accept" | "reject"
    reason: str


class ErrPCorrector:
    def __init__(self, model: ParadigmModel, threshold: float = 0.5) -> None:
        if model.paradigm != "errp":
            raise ValueError("ErrPCorrector requires an 'errp' model.")
        self.model = model
        self.threshold = threshold

    def assess(self, errp_epoch: np.ndarray) -> tuple[bool, float]:
        """Return (is_error, error_probability) for one post-action epoch."""
        X = errp_epoch[None] if errp_epoch.ndim == 2 else errp_epoch
        prob = float(self.model.target_scores(X)[0])   # P(error)
        return prob >= self.threshold, prob

    def review(self, errp_epoch: np.ndarray) -> CorrectionDecision:
        """Recommend whether to accept or reject the just-performed action."""
        is_error, prob = self.assess(errp_epoch)
        if is_error:
            return CorrectionDecision(
                True, prob, "reject",
                f"error potential detected (p={prob:.2f} >= {self.threshold})")
        return CorrectionDecision(
            False, prob, "accept",
            f"no error potential (p={prob:.2f} < {self.threshold})")
