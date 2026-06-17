"""Model registry and a thin paradigm-aware wrapper.

Models are scikit-learn ``Pipeline`` objects that fit on epoch arrays
``(n_epochs, n_channels, n_times)``. Two baselines suited to small,
short-calibration EEG datasets:

* ``vec_lda`` -- decimate+flatten -> standardise -> shrinkage LDA. Simple,
  fast, interpretable; a strong default for ERP.
* ``riemann_lr`` -- xDAWN covariances -> tangent space -> logistic
  regression. A robust Riemannian approach that often wins on P300.

Deep learning is deliberately *not* a default here: with a few hundred
calibration epochs it would overfit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neurobci.bci.features import EpochVectorizer

logger = logging.getLogger(__name__)

# ERP models take epochs as time series; oscillatory models use covariance.
MODEL_NAMES = ["vec_lda", "riemann_lr", "csp_lda", "cov_ts_lr"]


def _decim_for(sfreq: float, target_hz: float = 32.0) -> int:
    return max(int(round(sfreq / target_hz)), 1)


def build_pipeline(name: str, sfreq: float) -> Pipeline:
    """Construct an unfitted sklearn pipeline by name."""

    if name == "vec_lda":
        return Pipeline([
            ("vec", EpochVectorizer(decim=_decim_for(sfreq))),
            ("scale", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])

    if name == "riemann_lr":
        # ERP Riemannian model (xDAWN-enhanced covariances). Lazy import.
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace

        return Pipeline([
            ("xdawn", XdawnCovariances(nfilter=3, estimator="lwf")),
            ("ts", TangentSpace()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    if name == "csp_lda":
        # Oscillatory (motor-imagery) model: Common Spatial Patterns + LDA.
        import mne
        from mne.decoding import CSP

        mne.set_log_level("ERROR")
        return Pipeline([
            ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True,
                        norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])

    if name == "cov_ts_lr":
        # Oscillatory Riemannian model (plain covariances + tangent space).
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        return Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    raise ValueError(f"Unknown model: {name!r} (have {MODEL_NAMES})")


@dataclass
class ParadigmModel:
    """A fitted model plus the metadata needed to use and persist it."""

    name: str
    paradigm: str
    pipeline: object
    channel_names: list[str]
    channel_kinds: list[str]
    sfreq: float
    window: object                       # EpochWindow
    positive_index: int
    threshold: float = 0.5               # decision threshold on target score
    metrics: dict = field(default_factory=dict)
    trained: bool = False

    # ----- training / inference ----------------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ParadigmModel":
        self.pipeline.fit(X, y)
        self.trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def target_scores(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive (target) class per epoch."""
        proba = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.classes_)
        col = classes.index(self.positive_index) if self.positive_index in classes else -1
        return proba[:, col]

    def decide(self, X: np.ndarray) -> np.ndarray:
        """Boolean target/non-target decisions using ``threshold``."""
        return self.target_scores(X) >= self.threshold


def make_model(name: str, paradigm, channel_names, channel_kinds, sfreq) -> ParadigmModel:
    return ParadigmModel(
        name=name,
        paradigm=paradigm.name,
        pipeline=build_pipeline(name, sfreq),
        channel_names=list(channel_names),
        channel_kinds=list(channel_kinds),
        sfreq=sfreq,
        window=paradigm.window,
        positive_index=paradigm.positive_index,
    )
