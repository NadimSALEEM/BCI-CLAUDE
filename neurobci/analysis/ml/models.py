"""Model registry with hyperparameters and optional-dependency gating.

Each :class:`ModelSpec` knows its task (classification / regression /
clustering), the data shape it consumes (``tabular`` 2-D features or ``tensor``
raw epochs for spatial pipelines), a builder that returns a fresh
scikit-learn-compatible estimator from a parameter dict, a small tuning grid,
and which optional dependency (if any) it needs. Builders import heavy/optional
libraries lazily, so importing this module never requires them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from neurobci.analysis.shared import deps


@dataclass
class ModelSpec:
    key: str
    label: str
    task: str                       # classification | regression | clustering
    data: str                       # tabular | tensor
    build: Callable                 # (params: dict) -> estimator
    default_params: dict = field(default_factory=dict)
    param_grid: dict = field(default_factory=dict)
    needs: str | None = None        # optional dependency module
    proba: bool = False             # supports predict_proba (for ROC/PR)

    @property
    def available(self) -> bool:
        return self.needs is None or deps.have(self.needs)

    @property
    def install_hint(self) -> str:
        return "" if self.available else deps.install_hint(self.needs)


# --------------------------------------------------------------------------- #
# Tabular classifiers
# --------------------------------------------------------------------------- #

def _clf_specs() -> list[ModelSpec]:
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
        RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    s = []
    s.append(ModelSpec("dummy", "Dummy (chance baseline)", "classification",
                       "tabular", lambda p: DummyClassifier(strategy="most_frequent"),
                       proba=True))
    s.append(ModelSpec("logreg", "Logistic Regression", "classification", "tabular",
                       lambda p: LogisticRegression(max_iter=1000, **p),
                       {"C": 1.0}, {"C": [0.01, 0.1, 1.0, 10.0]}, proba=True))
    s.append(ModelSpec("elasticnet_logreg", "Elastic-Net Logistic", "classification",
                       "tabular",
                       lambda p: LogisticRegression(penalty="elasticnet", solver="saga",
                                                    l1_ratio=p.get("l1_ratio", 0.5),
                                                    C=p.get("C", 1.0), max_iter=2000),
                       {"l1_ratio": 0.5, "C": 1.0},
                       {"l1_ratio": [0.2, 0.5, 0.8], "C": [0.1, 1.0]}, proba=True))
    s.append(ModelSpec("lda", "LDA", "classification", "tabular",
                       lambda p: LinearDiscriminantAnalysis(
                           solver="lsqr", shrinkage="auto"), proba=True))
    s.append(ModelSpec("qda", "QDA", "classification", "tabular",
                       lambda p: QuadraticDiscriminantAnalysis(), proba=True))
    s.append(ModelSpec("gnb", "Gaussian Naive Bayes", "classification", "tabular",
                       lambda p: GaussianNB(), proba=True))
    s.append(ModelSpec("knn", "k-Nearest Neighbours", "classification", "tabular",
                       lambda p: KNeighborsClassifier(**p),
                       {"n_neighbors": 7}, {"n_neighbors": [3, 5, 7, 11]}, proba=True))
    s.append(ModelSpec("dtree", "Decision Tree", "classification", "tabular",
                       lambda p: DecisionTreeClassifier(random_state=0, **p),
                       {"max_depth": 5}, {"max_depth": [3, 5, 10, None]}, proba=True))
    s.append(ModelSpec("svm_linear", "SVM (linear)", "classification", "tabular",
                       lambda p: SVC(kernel="linear", probability=True,
                                     C=p.get("C", 1.0)),
                       {"C": 1.0}, {"C": [0.1, 1.0, 10.0]}, proba=True))
    s.append(ModelSpec("svm_rbf", "SVM (RBF)", "classification", "tabular",
                       lambda p: SVC(kernel="rbf", probability=True,
                                     C=p.get("C", 1.0), gamma=p.get("gamma", "scale")),
                       {"C": 1.0, "gamma": "scale"},
                       {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
                       proba=True))
    s.append(ModelSpec("rf", "Random Forest", "classification", "tabular",
                       lambda p: RandomForestClassifier(random_state=0, **p),
                       {"n_estimators": 300},
                       {"n_estimators": [200, 400], "max_depth": [None, 8]},
                       proba=True))
    s.append(ModelSpec("extratrees", "Extra Trees", "classification", "tabular",
                       lambda p: ExtraTreesClassifier(random_state=0, **p),
                       {"n_estimators": 300}, proba=True))
    s.append(ModelSpec("gradboost", "Gradient Boosting", "classification", "tabular",
                       lambda p: GradientBoostingClassifier(random_state=0, **p),
                       {}, proba=True))
    s.append(ModelSpec("adaboost", "AdaBoost", "classification", "tabular",
                       lambda p: AdaBoostClassifier(random_state=0, **p),
                       {"n_estimators": 100}, proba=True))

    # Optional gradient-boosting libraries
    s.append(ModelSpec("xgboost", "XGBoost", "classification", "tabular",
                       _build_xgb, {"n_estimators": 300}, needs="xgboost", proba=True))
    s.append(ModelSpec("lightgbm", "LightGBM", "classification", "tabular",
                       _build_lgbm, {"n_estimators": 300}, needs="lightgbm", proba=True))
    s.append(ModelSpec("catboost", "CatBoost", "classification", "tabular",
                       _build_catboost, {}, needs="catboost", proba=True))
    return s


def _build_xgb(p):
    from xgboost import XGBClassifier
    return XGBClassifier(eval_metric="logloss", random_state=0, **p)


def _build_lgbm(p):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=0, **p)


def _build_catboost(p):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(verbose=0, random_state=0, **p)


# --------------------------------------------------------------------------- #
# EEG-specific spatial pipelines (tensor input)
# --------------------------------------------------------------------------- #

def _build_csp_lda(p):
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    return Pipeline([("csp", CSP(n_components=p.get("n_components", 6))),
                     ("lda", LinearDiscriminantAnalysis(solver="lsqr",
                                                        shrinkage="auto"))])


def _build_csp_svm(p):
    from mne.decoding import CSP
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    return Pipeline([("csp", CSP(n_components=p.get("n_components", 6))),
                     ("svm", SVC(kernel="linear", probability=True))])


def _build_xdawn_lr(p):
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([("xdawn", XdawnCovariances(nfilter=p.get("nfilter", 4))),
                     ("ts", TangentSpace()),
                     ("lr", LogisticRegression(max_iter=1000))])


def _build_riemann_lr(p):
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([("cov", Covariances(estimator="oas")),
                     ("ts", TangentSpace()),
                     ("lr", LogisticRegression(max_iter=1000))])


def _build_mdm(p):
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from sklearn.pipeline import Pipeline
    return Pipeline([("cov", Covariances(estimator="oas")), ("mdm", MDM())])


def _eeg_specs() -> list[ModelSpec]:
    return [
        ModelSpec("csp_lda", "CSP + LDA", "classification", "tensor",
                  _build_csp_lda, {"n_components": 6},
                  {"n_components": [4, 6, 8]}, needs="mne", proba=True),
        ModelSpec("csp_svm", "CSP + SVM", "classification", "tensor",
                  _build_csp_svm, {"n_components": 6}, needs="mne", proba=True),
        ModelSpec("xdawn_lr", "xDAWN + Logistic Regression", "classification",
                  "tensor", _build_xdawn_lr, {"nfilter": 4},
                  {"nfilter": [2, 4, 6]}, needs="pyriemann", proba=True),
        ModelSpec("riemann_lr", "Riemann tangent space + LR", "classification",
                  "tensor", _build_riemann_lr, needs="pyriemann", proba=True),
        ModelSpec("mdm", "Riemann MDM", "classification", "tensor",
                  _build_mdm, needs="pyriemann", proba=False),
    ]


# --------------------------------------------------------------------------- #
# Regressors
# --------------------------------------------------------------------------- #

def _reg_specs() -> list[ModelSpec]:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR

    return [
        ModelSpec("dummy_reg", "Dummy (mean baseline)", "regression", "tabular",
                  lambda p: DummyRegressor(strategy="mean")),
        ModelSpec("linreg", "Linear Regression", "regression", "tabular",
                  lambda p: LinearRegression()),
        ModelSpec("ridge", "Ridge", "regression", "tabular",
                  lambda p: Ridge(**p), {"alpha": 1.0}, {"alpha": [0.1, 1.0, 10.0]}),
        ModelSpec("lasso", "Lasso", "regression", "tabular",
                  lambda p: Lasso(**p), {"alpha": 0.1}, {"alpha": [0.01, 0.1, 1.0]}),
        ModelSpec("elasticnet", "Elastic Net", "regression", "tabular",
                  lambda p: ElasticNet(**p), {"alpha": 0.1, "l1_ratio": 0.5}),
        ModelSpec("svr", "SVR", "regression", "tabular",
                  lambda p: SVR(**p), {"C": 1.0}, {"C": [0.1, 1.0, 10.0]}),
        ModelSpec("rf_reg", "Random Forest Regressor", "regression", "tabular",
                  lambda p: RandomForestRegressor(random_state=0, **p),
                  {"n_estimators": 300}),
        ModelSpec("gb_reg", "Gradient Boosting Regressor", "regression", "tabular",
                  lambda p: GradientBoostingRegressor(random_state=0, **p)),
        ModelSpec("knn_reg", "kNN Regressor", "regression", "tabular",
                  lambda p: KNeighborsRegressor(**p), {"n_neighbors": 7}),
        ModelSpec("pls", "PLS Regression", "regression", "tabular",
                  lambda p: PLSRegression(n_components=p.get("n_components", 2)),
                  {"n_components": 2}),
    ]


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

def _cluster_specs() -> list[ModelSpec]:
    from sklearn.cluster import (DBSCAN, AgglomerativeClustering, KMeans,
                                 MiniBatchKMeans, SpectralClustering)
    from sklearn.mixture import GaussianMixture

    def _build_hdbscan(p):
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=p.get("min_cluster_size", 10))

    return [
        ModelSpec("kmeans", "K-Means", "clustering", "tabular",
                  lambda p: KMeans(n_clusters=p.get("n_clusters", 3),
                                   n_init=10, random_state=0), {"n_clusters": 3}),
        ModelSpec("minibatch_kmeans", "MiniBatch K-Means", "clustering", "tabular",
                  lambda p: MiniBatchKMeans(n_clusters=p.get("n_clusters", 3),
                                            n_init=10, random_state=0),
                  {"n_clusters": 3}),
        ModelSpec("gmm", "Gaussian Mixture", "clustering", "tabular",
                  lambda p: GaussianMixture(n_components=p.get("n_clusters", 3),
                                            random_state=0), {"n_clusters": 3}),
        ModelSpec("agglomerative", "Agglomerative", "clustering", "tabular",
                  lambda p: AgglomerativeClustering(n_clusters=p.get("n_clusters", 3)),
                  {"n_clusters": 3}),
        ModelSpec("dbscan", "DBSCAN", "clustering", "tabular",
                  lambda p: DBSCAN(eps=p.get("eps", 0.5),
                                   min_samples=p.get("min_samples", 5)),
                  {"eps": 0.5, "min_samples": 5}),
        ModelSpec("spectral", "Spectral Clustering", "clustering", "tabular",
                  lambda p: SpectralClustering(n_clusters=p.get("n_clusters", 3),
                                               random_state=0), {"n_clusters": 3}),
        ModelSpec("hdbscan", "HDBSCAN", "clustering", "tabular",
                  _build_hdbscan, {"min_cluster_size": 10}, needs="hdbscan"),
    ]


# --------------------------------------------------------------------------- #
# Registry access
# --------------------------------------------------------------------------- #

def all_specs() -> list[ModelSpec]:
    return _clf_specs() + _eeg_specs() + _reg_specs() + _cluster_specs()


def registry(task: str | None = None, *, only_available: bool = False
             ) -> dict[str, ModelSpec]:
    specs = all_specs()
    if task is not None:
        specs = [s for s in specs if s.task == task]
    if only_available:
        specs = [s for s in specs if s.available]
    return {s.key: s for s in specs}


def get_spec(key: str) -> ModelSpec:
    reg = registry()
    if key not in reg:
        raise KeyError(f"Unknown model {key!r}.")
    return reg[key]
