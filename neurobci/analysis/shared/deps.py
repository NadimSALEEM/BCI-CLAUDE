"""Optional-dependency detection.

The analysis suite degrades gracefully: a feature that needs an optional
package is *disabled with an install hint* when that package is missing, never
crashed. UI code calls :func:`have` to enable/disable controls; compute code
calls :func:`require` to fetch a module or raise a readable error.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


class MissingDependencyError(RuntimeError):
    """Raised when an analysis needs an optional package that is not installed."""


@dataclass(frozen=True)
class Dependency:
    module: str
    purpose: str
    pip: str


# Everything the suite can *optionally* use. numpy/scipy/scikit-learn/matplotlib
# are hard requirements of the platform and intentionally not listed here.
OPTIONAL: tuple[Dependency, ...] = (
    Dependency("mne", "Cluster-permutation stats, xDAWN, TFR, MNE objects", "mne"),
    Dependency("statsmodels", "ANOVA, OLS/logistic regression, mixed models", "statsmodels"),
    Dependency("pingouin", "Convenient ANOVA / RM-ANOVA / post-hoc / effect sizes", "pingouin"),
    Dependency("pandas", "Tabular metadata and CSV/HTML export", "pandas"),
    Dependency("pyriemann", "Riemannian covariance / tangent space / MDM", "pyriemann"),
    Dependency("xgboost", "XGBoost gradient boosting", "xgboost"),
    Dependency("lightgbm", "LightGBM gradient boosting", "lightgbm"),
    Dependency("catboost", "CatBoost gradient boosting", "catboost"),
    Dependency("shap", "SHAP model explanations", "shap"),
    Dependency("umap", "UMAP embedding", "umap-learn"),
    Dependency("hdbscan", "HDBSCAN clustering", "hdbscan"),
    Dependency("torch", "Deep learning models (EEGNet, ConvNets, TCN)", "torch"),
    Dependency("braindecode", "Braindecode deep EEG models", "braindecode"),
)

_BY_MODULE = {d.module: d for d in OPTIONAL}


def have(module: str) -> bool:
    """True if ``module`` can be imported (cached by the import system)."""
    try:
        importlib.import_module(module)
        return True
    except Exception:  # noqa: BLE001 - a broken optional dep is "not available"
        return False


def status() -> dict[str, bool]:
    """Availability of every known optional dependency, ``{module: bool}``."""
    return {d.module: have(d.module) for d in OPTIONAL}


def install_hint(module: str) -> str:
    dep = _BY_MODULE.get(module)
    return f"pip install {dep.pip}" if dep else f"pip install {module}"


def require(module: str):
    """Import and return ``module`` or raise :class:`MissingDependencyError`."""
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        purpose = _BY_MODULE[module].purpose if module in _BY_MODULE else module
        raise MissingDependencyError(
            f"This analysis needs '{module}' ({purpose}), which is not "
            f"installed. Install it with:  {install_hint(module)}"
        ) from exc
