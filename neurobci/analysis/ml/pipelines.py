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


def _maybe_balance(estimator):
    """Set ``class_weight='balanced'`` on the estimator (or its last step)."""
    try:
        last = estimator
        if hasattr(estimator, "steps"):
            last = estimator.steps[-1][1]
        if "class_weight" in last.get_params():
            last.set_params(class_weight="balanced")
    except Exception:  # noqa: BLE001 - estimator simply may not support it
        pass
    return estimator


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
