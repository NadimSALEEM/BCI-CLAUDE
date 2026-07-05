"""Leakage-safe scikit-learn pipeline assembly.

Every data-dependent step (scaling, variance/feature selection, PCA) is placed
*inside* the estimator pipeline, so under cross-validation it is fit on the
training fold only -- never on the whole dataset. Spatial EEG pipelines
(CSP/xDAWN/Riemannian) already encapsulate their fitted spatial filters the
same way. Resampling-based balancing (SMOTE/over/under) is intentionally not
applied here because it requires imbalanced-learn and is easy to leak; we use
class weights and stratified CV instead and surface that choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


@dataclass
class PreprocOptions:
    scaler: str = "standard"            # standard | robust | minmax | none
    variance_threshold: float | None = None
    select_k: int | None = None         # SelectKBest(f_classif/f_regression)
    pca: object = None                  # int components, float variance, or None
    class_weight_balanced: bool = False


def _scaler(name: str):
    from sklearn.preprocessing import (MinMaxScaler, RobustScaler,
                                       StandardScaler)
    return {"standard": StandardScaler, "robust": RobustScaler,
            "minmax": MinMaxScaler}.get(name, lambda: None)()


class UniformPriorClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a priors-based classifier (LDA/QDA/GaussianNB) to fit with uniform
    class priors, so "Balance classes" has effect for models that have no
    ``class_weight``. Priors are set from the *training fold's* classes at fit
    time, so it stays leakage-safe under cross-validation.
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        self.base_ = clone(self.base).set_params(priors=[1.0 / k] * k)
        self.base_.fit(X, y)
        return self

    def predict(self, X):
        return self.base_.predict(X)

    def predict_proba(self, X):
        return self.base_.predict_proba(X)


def _balance_estimator(est):
    """Return an estimator that honours class balancing, or ``est`` unchanged.

    Prefers native ``class_weight='balanced'``; falls back to uniform priors for
    priors-based classifiers; leaves models with neither untouched (the UI warns
    that balancing does not apply to those, e.g. kNN / boosting).
    """
    params = est.get_params()
    if "class_weight" in params:
        est.set_params(class_weight="balanced")
        return est
    if "priors" in params:
        return UniformPriorClassifier(base=est)
    return est


def _maybe_balance(estimator):
    """Apply class balancing to the final predictive step (or the estimator)."""
    if hasattr(estimator, "steps"):
        name, last = estimator.steps[-1]
        estimator.steps[-1] = (name, _balance_estimator(last))
        return estimator
    return _balance_estimator(estimator)


def balancing_capability(spec) -> str | None:
    """How a model can be balanced: ``'class_weight'``, ``'priors'`` or ``None``.

    ``None`` means "Balance classes" has no effect for this model (e.g. kNN,
    Gradient Boosting, AdaBoost) -- the UI surfaces that rather than silently
    ignoring the checkbox.
    """
    try:
        est = spec.build(dict(spec.default_params))
    except Exception:  # noqa: BLE001
        return None
    target = est.steps[-1][1] if hasattr(est, "steps") else est
    params = target.get_params()
    if "class_weight" in params:
        return "class_weight"
    if "priors" in params:
        return "priors"
    return None


def build_pipeline(spec, params: dict | None = None,
                   opts: PreprocOptions | None = None):
    """Compose a leakage-safe pipeline for one :class:`ModelSpec`."""
    from sklearn.pipeline import Pipeline
    params = params or {}
    opts = opts or PreprocOptions()
    estimator = spec.build(params)
    if opts.class_weight_balanced:
        estimator = _maybe_balance(estimator)

    # Spatial/tensor models consume raw epochs; no tabular preprocessing.
    if spec.data == "tensor":
        if hasattr(estimator, "steps"):
            return estimator
        return Pipeline([("model", estimator)])

    steps = []
    sc = _scaler(opts.scaler)
    if sc is not None:
        steps.append(("scaler", sc))
    if opts.variance_threshold is not None:
        from sklearn.feature_selection import VarianceThreshold
        steps.append(("variance", VarianceThreshold(opts.variance_threshold)))
    if opts.select_k:
        from sklearn.feature_selection import (SelectKBest, f_classif,
                                               f_regression)
        score = f_regression if spec.task == "regression" else f_classif
        steps.append(("select", SelectKBest(score, k=opts.select_k)))
    if opts.pca is not None:
        from sklearn.decomposition import PCA
        steps.append(("pca", PCA(n_components=opts.pca, random_state=0)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def describe_pipeline(pipe) -> str:
    """Human-readable one-line description of the final fitted pipeline."""
    if hasattr(pipe, "steps"):
        return " -> ".join(f"{name}({type(step).__name__})"
                           for name, step in pipe.steps)
    return type(pipe).__name__
