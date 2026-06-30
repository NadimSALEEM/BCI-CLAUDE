"""Model evaluation: cross-validation, metrics, chance level, diagnostics.

This is where the suite's ML safety rules are enforced: cross-validation is the
single source of truth for performance, confusion matrices and aggregate
metrics come from *out-of-fold* predictions, every classification result is
compared against a chance/dummy baseline, and class imbalance / subject leakage
/ overfitting are surfaced as warnings rather than hidden.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as st

from neurobci.analysis.ml.pipelines import describe_pipeline
from neurobci.analysis.shared.validation import (
    Diagnostic, check_class_balance, check_overfit, check_sample_size,
    check_subject_leakage)

GROUPED_CV = ("group_kfold", "logo")


# --------------------------------------------------------------------------- #
# Cross-validation factory
# --------------------------------------------------------------------------- #

def make_cv(strategy="stratified_kfold", *, n_splits=5, n_repeats=5, seed=42):
    """Return ``(cv, is_grouped, description)`` for a named CV strategy."""
    from sklearn.model_selection import (GroupKFold, KFold,
                                         LeaveOneGroupOut, LeaveOneOut,
                                         RepeatedStratifiedKFold,
                                         StratifiedKFold)
    if strategy == "stratified_kfold":
        return (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
                False, f"{n_splits}-fold stratified CV")
    if strategy == "repeated_stratified":
        return (RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                        random_state=seed),
                False, f"{n_repeats}x{n_splits} repeated stratified CV")
    if strategy == "kfold":
        return (KFold(n_splits=n_splits, shuffle=True, random_state=seed),
                False, f"{n_splits}-fold CV")
    if strategy == "loo":
        return LeaveOneOut(), False, "leave-one-out CV"
    if strategy == "group_kfold":
        return (GroupKFold(n_splits=n_splits), True,
                f"{n_splits}-fold group CV (no subject leakage)")
    if strategy == "logo":
        return (LeaveOneGroupOut(), True,
                "leave-one-subject/session-out CV")
    raise ValueError(f"Unknown CV strategy {strategy!r}.")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def classification_metrics(y_true, y_pred, y_proba=None, labels=None) -> dict:
    from sklearn import metrics as M
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = labels if labels is not None else np.unique(y_true)
    out = {
        "accuracy": float(M.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(M.balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(M.precision_score(y_true, y_pred, average="macro",
                                                   zero_division=0)),
        "recall_macro": float(M.recall_score(y_true, y_pred, average="macro",
                                             zero_division=0)),
        "f1_macro": float(M.f1_score(y_true, y_pred, average="macro",
                                     zero_division=0)),
        "f1_weighted": float(M.f1_score(y_true, y_pred, average="weighted",
                                        zero_division=0)),
        "cohen_kappa": float(M.cohen_kappa_score(y_true, y_pred)),
        "mcc": float(M.matthews_corrcoef(y_true, y_pred)),
    }
    if len(classes) == 2:
        cm = M.confusion_matrix(y_true, y_pred, labels=classes)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
            out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    if y_proba is not None:
        try:
            if len(classes) == 2:
                p1 = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                out["roc_auc"] = float(M.roc_auc_score(y_true, p1))
                out["pr_auc"] = float(M.average_precision_score(y_true, p1))
                out["brier"] = float(M.brier_score_loss(y_true, p1))
            else:
                out["roc_auc_ovr"] = float(M.roc_auc_score(
                    y_true, y_proba, multi_class="ovr"))
            out["log_loss"] = float(M.log_loss(y_true, y_proba))
        except Exception:  # noqa: BLE001 - proba metrics are best-effort
            pass
    return out


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn import metrics as M
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rmse = float(np.sqrt(M.mean_squared_error(y_true, y_pred)))
    out = {
        "r2": float(M.r2_score(y_true, y_pred)),
        "mae": float(M.mean_absolute_error(y_true, y_pred)),
        "mse": float(M.mean_squared_error(y_true, y_pred)),
        "rmse": rmse,
        "explained_variance": float(M.explained_variance_score(y_true, y_pred)),
    }
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        out["pearson_r"] = float(st.pearsonr(y_true, y_pred).statistic)
        out["spearman_r"] = float(st.spearmanr(y_true, y_pred).statistic)
    return out


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class ModelEvaluation:
    key: str
    label: str
    task: str
    scoring: str
    fold_scores: np.ndarray
    mean_score: float
    std_score: float
    ci95: tuple
    train_mean: float
    fit_time: float
    predict_time: float
    n_features: int
    cv_desc: str
    params: dict = field(default_factory=dict)
    pipeline_desc: str = ""
    metrics: dict = field(default_factory=dict)
    confusion: object = None
    class_labels: list = field(default_factory=list)
    y_true: object = None
    y_pred: object = None
    y_proba: object = None
    chance_score: float | None = None
    chance_p: float | None = None
    dummy_score: float | None = None
    diagnostics: list = field(default_factory=list)

    def summary_row(self) -> dict:
        return {
            "model": self.label,
            "cv_mean": round(self.mean_score, 4),
            "cv_std": round(self.std_score, 4),
            "ci95": f"[{self.ci95[0]:.3f}, {self.ci95[1]:.3f}]",
            "train": round(self.train_mean, 4),
            "chance": None if self.chance_score is None else round(self.chance_score, 4),
            "p_vs_chance": None if self.chance_p is None else self.chance_p,
            "fit_s": round(self.fit_time, 3),
            "n_feat": self.n_features,
        }


def _ci95(scores: np.ndarray) -> tuple:
    scores = np.asarray(scores, float)
    n = scores.size
    if n < 2:
        return (float(scores.mean()), float(scores.mean()))
    se = scores.std(ddof=1) / np.sqrt(n)
    h = float(st.t.ppf(0.975, n - 1) * se)
    m = float(scores.mean())
    return (m - h, m + h)


# --------------------------------------------------------------------------- #
# Evaluation drivers
# --------------------------------------------------------------------------- #

def evaluate_classification(
    pipeline, X, y, *, key="", label="", scoring="balanced_accuracy",
    cv=None, is_grouped=False, groups=None, cv_desc="", params=None,
    class_names=None, n_jobs=1,
) -> ModelEvaluation:
    """Cross-validate a classifier and assemble a full :class:`ModelEvaluation`."""
    from sklearn.model_selection import cross_val_predict, cross_validate
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    class_names = class_names or [str(c) for c in classes]

    cv_res = cross_validate(pipeline, X, y, cv=cv, groups=groups, scoring=scoring,
                            return_train_score=True, n_jobs=n_jobs,
                            error_score="raise")
    fold_scores = np.asarray(cv_res["test_score"], float)

    # Out-of-fold predictions for confusion matrix + aggregate metrics.
    confusion = y_pred = y_proba = None
    metrics: dict = {}
    try:
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, groups=groups, n_jobs=n_jobs)
        from sklearn.metrics import confusion_matrix
        confusion = confusion_matrix(y, y_pred, labels=classes)
        try:
            y_proba = cross_val_predict(pipeline, X, y, cv=cv, groups=groups,
                                        method="predict_proba", n_jobs=n_jobs)
        except Exception:  # noqa: BLE001 - estimator may lack predict_proba
            y_proba = None
        metrics = classification_metrics(y, y_pred, y_proba, labels=classes)
    except Exception:  # noqa: BLE001 - e.g. repeated CV: predictions overlap
        pass

    diags = list(check_class_balance(y, class_names))
    diags += check_sample_size(len(y), floor=2 * len(classes) * 5,
                               what="trials")
    diags += check_subject_leakage(groups, is_grouped)
    diags += check_overfit(float(np.mean(cv_res["train_score"])),
                           float(fold_scores.mean()))

    return ModelEvaluation(
        key=key, label=label, task="classification", scoring=scoring,
        fold_scores=fold_scores, mean_score=float(fold_scores.mean()),
        std_score=float(fold_scores.std(ddof=1)) if fold_scores.size > 1 else 0.0,
        ci95=_ci95(fold_scores), train_mean=float(np.mean(cv_res["train_score"])),
        fit_time=float(np.mean(cv_res["fit_time"])),
        predict_time=float(np.mean(cv_res["score_time"])),
        n_features=int(X.reshape(X.shape[0], -1).shape[1]), cv_desc=cv_desc,
        params=params or {}, pipeline_desc=describe_pipeline(pipeline),
        metrics=metrics, confusion=confusion, class_labels=class_names,
        y_true=y, y_pred=y_pred, y_proba=y_proba, diagnostics=diags)


def add_chance_level(evaluation, pipeline, X, y, *, cv=None, groups=None,
                     n_permutations=200, scoring="balanced_accuracy", seed=42,
                     n_jobs=1):
    """Attach a permutation chance level and p-value to an evaluation."""
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import cross_val_score, permutation_test_score
    X, y = np.asarray(X), np.asarray(y)
    dummy = cross_val_score(DummyClassifier(strategy="prior"),
                            X.reshape(X.shape[0], -1), y, cv=cv, groups=groups,
                            scoring=scoring, n_jobs=n_jobs)
    evaluation.dummy_score = float(np.mean(dummy))
    _, perm_scores, pvalue = permutation_test_score(
        pipeline, X, y, groups=groups, cv=cv, scoring=scoring,
        n_permutations=n_permutations, random_state=seed, n_jobs=n_jobs)
    evaluation.chance_score = float(np.mean(perm_scores))
    evaluation.chance_p = float(pvalue)
    if evaluation.chance_p > 0.05:
        evaluation.diagnostics.append(Diagnostic(
            "warning",
            f"Performance is not distinguishable from chance "
            f"(permutation p={evaluation.chance_p:.3f})."))
    return evaluation


def evaluate_regression(
    pipeline, X, y, *, key="", label="", scoring="r2", cv=None,
    is_grouped=False, groups=None, cv_desc="", params=None, n_jobs=1,
) -> ModelEvaluation:
    from sklearn.model_selection import cross_val_predict, cross_validate
    X, y = np.asarray(X), np.asarray(y, float)
    cv_res = cross_validate(pipeline, X, y, cv=cv, groups=groups, scoring=scoring,
                            return_train_score=True, n_jobs=n_jobs,
                            error_score="raise")
    fold_scores = np.asarray(cv_res["test_score"], float)
    metrics, y_pred = {}, None
    try:
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, groups=groups, n_jobs=n_jobs)
        metrics = regression_metrics(y, np.ravel(y_pred))
    except Exception:  # noqa: BLE001
        pass
    diags = check_sample_size(len(y), floor=20, what="trials")
    diags += check_subject_leakage(groups, is_grouped)
    return ModelEvaluation(
        key=key, label=label, task="regression", scoring=scoring,
        fold_scores=fold_scores, mean_score=float(fold_scores.mean()),
        std_score=float(fold_scores.std(ddof=1)) if fold_scores.size > 1 else 0.0,
        ci95=_ci95(fold_scores), train_mean=float(np.mean(cv_res["train_score"])),
        fit_time=float(np.mean(cv_res["fit_time"])),
        predict_time=float(np.mean(cv_res["score_time"])),
        n_features=int(X.reshape(X.shape[0], -1).shape[1]), cv_desc=cv_desc,
        params=params or {}, pipeline_desc=describe_pipeline(pipeline),
        metrics=metrics, y_true=y, y_pred=y_pred, diagnostics=diags)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

@dataclass
class ClusterEvaluation:
    key: str
    label: str
    labels: np.ndarray
    n_clusters: int
    metrics: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)


def evaluate_clustering(estimator, X, y_true=None, *, key="", label="") -> ClusterEvaluation:
    from sklearn import metrics as M
    X = np.asarray(X)
    X = X.reshape(X.shape[0], -1)
    labels = estimator.fit_predict(X)
    valid = labels[labels >= 0]
    n_clusters = int(len(np.unique(valid)))
    out: dict = {}
    diags: list[Diagnostic] = []
    if n_clusters >= 2 and len(np.unique(labels)) < len(labels):
        try:
            out["silhouette"] = float(M.silhouette_score(X, labels))
            out["calinski_harabasz"] = float(M.calinski_harabasz_score(X, labels))
            out["davies_bouldin"] = float(M.davies_bouldin_score(X, labels))
        except Exception:  # noqa: BLE001
            pass
    else:
        diags.append(Diagnostic("warning",
                     "Clustering produced <2 clusters; internal metrics undefined."))
    if y_true is not None:
        out["adjusted_rand"] = float(M.adjusted_rand_score(y_true, labels))
        out["nmi"] = float(M.normalized_mutual_info_score(y_true, labels))
        out["purity"] = _cluster_purity(y_true, labels)
    return ClusterEvaluation(key=key, label=label, labels=labels,
                             n_clusters=n_clusters, metrics=out, diagnostics=diags)


def _cluster_purity(y_true, labels) -> float:
    y_true, labels = np.asarray(y_true), np.asarray(labels)
    total = 0
    for c in np.unique(labels):
        members = y_true[labels == c]
        if members.size:
            _, counts = np.unique(members, return_counts=True)
            total += counts.max()
    return float(total / len(y_true)) if len(y_true) else float("nan")


def rank_models(evaluations: list[ModelEvaluation]) -> list[ModelEvaluation]:
    """Sort evaluations by mean CV score (descending)."""
    return sorted(evaluations, key=lambda e: e.mean_score, reverse=True)
