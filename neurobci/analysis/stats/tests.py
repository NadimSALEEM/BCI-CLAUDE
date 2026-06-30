"""Parametric, non-parametric and robust statistical tests.

Each test returns a uniform :class:`TestResult` carrying the statistic, df,
p-value, effect size(s), a confidence interval, assumption diagnostics, group
sizes and a cautious plain-language interpretation. The wording deliberately
says "the analysis suggests", never "proves".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as st

from neurobci.analysis.shared.validation import (
    Diagnostic, check_finite, check_paired, check_sample_size)


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    df: object = None
    effect_size: dict = field(default_factory=dict)
    ci: tuple | None = None
    ci_label: str = ""
    assumptions: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)
    alternative: str = "two-sided"
    extra: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    interpretation: str = ""

    def to_summary(self) -> dict:
        es = ", ".join(f"{k}={v:.3f}" for k, v in self.effect_size.items()
                       if v is not None and np.isfinite(v))
        return {
            "test": self.name,
            "statistic": round(float(self.statistic), 4),
            "df": self.df,
            "p": float(self.p_value),
            "effect_size": es,
            "n": self.n,
        }


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #

def cohen_d_one_sample(x, mu=0.0):
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float((x.mean() - mu) / sd) if sd > 0 else np.nan


def cohen_d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.nan


def cohen_d_independent(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = a.size, b.size
    sp2 = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    sp = np.sqrt(sp2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def hedges_g(d, n_total):
    if not np.isfinite(d):
        return np.nan
    return float(d * (1.0 - 3.0 / (4.0 * n_total - 9.0)))


def eta_squared_oneway(groups):
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((allv - grand) ** 2).sum())
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


# --------------------------------------------------------------------------- #
# Assumption checks
# --------------------------------------------------------------------------- #

def shapiro_p(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or x.size > 5000:
        return np.nan
    try:
        return float(st.shapiro(x).pvalue)
    except Exception:  # noqa: BLE001
        return np.nan


def levene_p(*groups):
    try:
        return float(st.levene(*[np.asarray(g, float) for g in groups]).pvalue)
    except Exception:  # noqa: BLE001
        return np.nan


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #

def _es_label(d):
    a = abs(d)
    if not np.isfinite(a):
        return "undefined"
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def interpret(test_name, p, alpha, primary_es=None, es_name="d"):
    sig = p < alpha
    verdict = "suggests a statistically detectable" if sig else \
        "does not provide evidence for an"
    txt = (f"At alpha={alpha:g}, the {test_name} {verdict} effect "
           f"(p={p:.3g}).")
    if primary_es is not None and np.isfinite(primary_es):
        txt += (f" Estimated effect size {es_name}={primary_es:.2f} "
                f"({_es_label(primary_es)}).")
    txt += (" This is an estimate from the available data, not proof; "
            "consider sample size, assumptions and multiple testing.")
    return txt


def _base_diagnostics(*arrays, floor=10, what="observations"):
    diags: list[Diagnostic] = []
    for arr in arrays:
        diags += check_finite(arr, "feature")
    diags += check_sample_size(min(a.size for a in arrays), floor=floor, what=what)
    return diags


# --------------------------------------------------------------------------- #
# Parametric tests
# --------------------------------------------------------------------------- #

def one_sample_t(x, popmean=0.0, alternative="two-sided", alpha=0.05, ci=0.95):
    x = np.asarray(x, float)
    res = st.ttest_1samp(x, popmean, alternative=alternative)
    d = cohen_d_one_sample(x, popmean)
    interval = res.confidence_interval(confidence_level=ci)
    out = TestResult(
        name="One-sample t-test", statistic=float(res.statistic),
        p_value=float(res.pvalue), df=float(getattr(res, "df", x.size - 1)),
        effect_size={"cohen_d": d, "hedges_g": hedges_g(d, x.size)},
        ci=(float(interval.low), float(interval.high)),
        ci_label=f"{int(ci*100)}% CI on mean",
        assumptions={"shapiro_p": shapiro_p(x)},
        n={"n": int(x.size)}, alternative=alternative)
    out.diagnostics = _base_diagnostics(x)
    out.interpretation = interpret(out.name, out.p_value, alpha, d, "Cohen's d")
    return out


def paired_t(a, b, alternative="two-sided", alpha=0.05, ci=0.95):
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = st.ttest_rel(a, b, alternative=alternative)
    d = cohen_d_paired(a, b)
    interval = res.confidence_interval(confidence_level=ci)
    out = TestResult(
        name="Paired-samples t-test", statistic=float(res.statistic),
        p_value=float(res.pvalue), df=float(getattr(res, "df", a.size - 1)),
        effect_size={"cohen_d": d, "hedges_g": hedges_g(d, a.size)},
        ci=(float(interval.low), float(interval.high)),
        ci_label=f"{int(ci*100)}% CI on mean difference",
        assumptions={"shapiro_diff_p": shapiro_p(a - b)},
        n={"n_pairs": int(a.size)}, alternative=alternative)
    out.diagnostics = check_paired(a, b) + check_finite(a - b, "difference")
    out.interpretation = interpret(out.name, out.p_value, alpha, d, "Cohen's d")
    return out


def independent_t(a, b, *, welch=False, alternative="two-sided",
                  alpha=0.05, ci=0.95):
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = st.ttest_ind(a, b, equal_var=not welch, alternative=alternative)
    d = cohen_d_independent(a, b)
    interval = res.confidence_interval(confidence_level=ci)
    name = "Welch t-test" if welch else "Independent-samples t-test"
    out = TestResult(
        name=name, statistic=float(res.statistic), p_value=float(res.pvalue),
        df=float(getattr(res, "df", a.size + b.size - 2)),
        effect_size={"cohen_d": d, "hedges_g": hedges_g(d, a.size + b.size)},
        ci=(float(interval.low), float(interval.high)),
        ci_label=f"{int(ci*100)}% CI on mean difference",
        assumptions={"shapiro_a_p": shapiro_p(a), "shapiro_b_p": shapiro_p(b),
                     "levene_p": levene_p(a, b)},
        n={"n_a": int(a.size), "n_b": int(b.size)}, alternative=alternative)
    out.diagnostics = _base_diagnostics(a, b)
    out.interpretation = interpret(name, out.p_value, alpha, d, "Cohen's d")
    return out


def oneway_anova(groups, names=None, alpha=0.05):
    groups = [np.asarray(g, float) for g in groups]
    res = st.f_oneway(*groups)
    eta2 = eta_squared_oneway(groups)
    k, n = len(groups), sum(g.size for g in groups)
    out = TestResult(
        name="One-way ANOVA", statistic=float(res.statistic),
        p_value=float(res.pvalue), df=(k - 1, n - k),
        effect_size={"eta_squared": eta2},
        assumptions={"levene_p": levene_p(*groups)},
        n={(names[i] if names else f"g{i}"): int(g.size)
           for i, g in enumerate(groups)})
    out.diagnostics = _base_diagnostics(*groups, what="observations/group")
    out.interpretation = interpret(out.name, out.p_value, alpha, eta2, "eta^2")
    return out


# --------------------------------------------------------------------------- #
# Non-parametric tests
# --------------------------------------------------------------------------- #

def _wilcoxon_rank_biserial(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    ranks = st.rankdata(np.abs(d))
    t_pos, t_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    tot = t_pos + t_neg
    return float((t_pos - t_neg) / tot) if tot > 0 else np.nan


def wilcoxon(a, b, alternative="two-sided", alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = st.wilcoxon(a, b, alternative=alternative)
    r = _wilcoxon_rank_biserial(a, b)
    out = TestResult(
        name="Wilcoxon signed-rank test", statistic=float(res.statistic),
        p_value=float(res.pvalue), effect_size={"rank_biserial": r},
        n={"n_pairs": int(a.size)}, alternative=alternative)
    out.diagnostics = check_paired(a, b)
    out.interpretation = interpret(out.name, out.p_value, alpha, r, "r")
    return out


def mann_whitney(a, b, alternative="two-sided", alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = st.mannwhitneyu(a, b, alternative=alternative)
    u = float(res.statistic)
    r = 2.0 * u / (a.size * b.size) - 1.0       # rank-biserial (x relative to y)
    out = TestResult(
        name="Mann-Whitney U test", statistic=u, p_value=float(res.pvalue),
        effect_size={"rank_biserial": r},
        n={"n_a": int(a.size), "n_b": int(b.size)}, alternative=alternative)
    out.diagnostics = _base_diagnostics(a, b)
    out.interpretation = interpret(out.name, out.p_value, alpha, r, "r")
    return out


def kruskal(groups, names=None, alpha=0.05):
    groups = [np.asarray(g, float) for g in groups]
    res = st.kruskal(*groups)
    out = TestResult(
        name="Kruskal-Wallis test", statistic=float(res.statistic),
        p_value=float(res.pvalue), df=len(groups) - 1,
        n={(names[i] if names else f"g{i}"): int(g.size)
           for i, g in enumerate(groups)})
    out.diagnostics = _base_diagnostics(*groups, what="observations/group")
    out.interpretation = interpret(out.name, out.p_value, alpha)
    return out


def friedman(groups, alpha=0.05):
    """Repeated-measures non-parametric test; ``groups`` are aligned columns."""
    groups = [np.asarray(g, float) for g in groups]
    res = st.friedmanchisquare(*groups)
    out = TestResult(
        name="Friedman test", statistic=float(res.statistic),
        p_value=float(res.pvalue), df=len(groups) - 1,
        n={"n_subjects": int(groups[0].size), "k": len(groups)})
    out.interpretation = interpret(out.name, out.p_value, alpha)
    return out


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #

def correlation(x, y, method="pearson", alpha=0.05, ci=0.95):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if method == "pearson":
        res = st.pearsonr(x, y)
    elif method == "spearman":
        res = st.spearmanr(x, y)
    elif method == "kendall":
        res = st.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown correlation method {method!r}.")
    r = float(res.statistic if hasattr(res, "statistic") else res[0])
    p = float(res.pvalue if hasattr(res, "pvalue") else res[1])
    interval = None
    if method in ("pearson", "spearman") and x.size > 3 and abs(r) < 1:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(x.size - 3)
        zc = st.norm.ppf(1 - (1 - ci) / 2)
        interval = (float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se)))
    out = TestResult(
        name=f"{method.capitalize()} correlation", statistic=r, p_value=p,
        df=x.size - 2, effect_size={"r": r}, ci=interval,
        ci_label=f"{int(ci*100)}% CI on r", n={"n": int(x.size)})
    out.diagnostics = _base_diagnostics(x, y)
    out.interpretation = interpret(out.name, p, alpha, r, "r")
    return out


# --------------------------------------------------------------------------- #
# Robust / resampling
# --------------------------------------------------------------------------- #

def permutation_ttest(a, b, *, paired=False, n_permutations=5000,
                      alternative="two-sided", seed=None, alpha=0.05):
    """Permutation test on the mean difference (sign-flip if paired)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        d = a - b
        obs = d.mean()
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, d.size))
        null = (signs * d).mean(axis=1)
    else:
        obs = a.mean() - b.mean()
        pooled = np.concatenate([a, b])
        na = a.size
        null = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(pooled)
            null[i] = perm[:na].mean() - perm[na:].mean()
    if alternative == "greater":
        p = (np.sum(null >= obs) + 1) / (n_permutations + 1)
    elif alternative == "less":
        p = (np.sum(null <= obs) + 1) / (n_permutations + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_permutations + 1)
    d_eff = cohen_d_paired(a, b) if paired else cohen_d_independent(a, b)
    out = TestResult(
        name="Permutation t-test", statistic=float(obs), p_value=float(p),
        effect_size={"cohen_d": d_eff},
        n={"n_a": int(a.size), "n_b": int(b.size),
           "n_permutations": int(n_permutations)}, alternative=alternative,
        extra={"seed": seed})
    out.interpretation = interpret(out.name, out.p_value, alpha, d_eff, "Cohen's d")
    return out


def bootstrap_ci(x, statistic=np.mean, *, n_boot=5000, ci=0.95, seed=None):
    """Percentile bootstrap CI of ``statistic`` over a 1-D sample."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    boots = np.array([statistic(rng.choice(x, x.size, replace=True))
                      for _ in range(n_boot)])
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 - (1 - ci) / 2)))
    return float(statistic(x)), (lo, hi)


def assumptions_suggest_nonparametric(assumptions: dict, alpha=0.05):
    """Advisory only -- never auto-switches. Returns ``(suggest, reason)``."""
    reasons = []
    for k, v in assumptions.items():
        if v is None or not np.isfinite(v):
            continue
        if "shapiro" in k and v < alpha:
            reasons.append(f"normality rejected ({k}={v:.3f})")
        if "levene" in k and v < alpha:
            reasons.append(f"unequal variances ({k}={v:.3f})")
    return (bool(reasons), "; ".join(reasons))
