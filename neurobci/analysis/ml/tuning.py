"""Hyperparameter tuning (grid / randomized search), leakage-safe.

The search itself runs cross-validated on the data it is given, so when used as
the *inner* loop of nested CV the outer test fold is never seen during tuning.
For a single-level search the returned estimator must still be reported as
optimistically biased (no held-out set) -- the UI says so.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TuningResult:
    best_params: dict
    best_score: float
    estimator: object
    cv_results: dict


def tune(spec, X, y, *, opts=None, method="grid", cv=None, groups=None,
         scoring="balanced_accuracy", n_iter=20, seed=42, n_jobs=1,
         param_grid=None) -> TuningResult:
    """Grid or randomized search over ``spec.param_grid`` (or ``param_grid``)."""
    from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV)

    from neurobci.analysis.ml.pipelines import build_pipeline

    grid = param_grid if param_grid is not None else dict(spec.param_grid)
    if not grid:
        raise ValueError(f"Model {spec.key!r} has no tunable parameters.")
    pipe = build_pipeline(spec, {}, opts)
    # Address parameters at the final estimator step of the pipeline.
    prefix = "model__" if hasattr(pipe, "steps") and pipe.steps[-1][0] == "model" else ""
    grid = {f"{prefix}{k}": v for k, v in grid.items()}

    if method == "random":
        search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring=scoring,
                                    cv=cv, random_state=seed, n_jobs=n_jobs,
                                    error_score="raise")
    else:
        search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=n_jobs,
                              error_score="raise")
    search.fit(X, y, groups=groups) if groups is not None else search.fit(X, y)
    best = {k.replace(prefix, ""): v for k, v in search.best_params_.items()}
    return TuningResult(best_params=best, best_score=float(search.best_score_),
                        estimator=search.best_estimator_,
                        cv_results=search.cv_results_)
