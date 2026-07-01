"""ANOVA family and post-hoc comparisons (design-aware).

Repeated-measures / mixed / factorial ANOVA and ANCOVA via pingouin, plus
parametric and rank-based post-hoc pairwise tests. Two design frames are built
from an :class:`~neurobci.analysis.datasource.EpochBundle`:

* **trial-level** (one observation per trial) for one-way / factorial / ANCOVA
  -- exploratory within a single subject;
* **subject x condition means** for repeated-measures / mixed ANOVA -- the
  standard group-level EEG design, where the repeated unit is a subject
  (``groups``) or a chosen metadata field (block / repetition).

pandas and pingouin are imported lazily, so importing this module never
requires them; the UI gates the feature on their availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurobci.analysis.shared.deps import require
from neurobci.analysis.shared.validation import Diagnostic, WARNING

# Map the UI's correction names to pingouin's ``padjust`` codes.
_PADJUST = {"none": "none", "bonferroni": "bonf", "holm": "holm",
            "fdr_bh": "fdr_bh", "fdr_by": "fdr_by"}


@dataclass
class AnovaResult:
    name: str
    table: list = field(default_factory=list)          # rows (dicts)
    columns: list = field(default_factory=list)
    posthoc: list = field(default_factory=list)         # rows (dicts)
    posthoc_columns: list = field(default_factory=list)
    sphericity: dict | None = None
    diagnostics: list = field(default_factory=list)
    interpretation: str = ""
    extra: dict = field(default_factory=dict)

    def to_summary(self) -> dict:
        eff = self.extra.get("significant", [])
        return {"test": self.name, "significant_effects": ", ".join(eff) or "none",
                "min_p": self.extra.get("min_p")}


# --------------------------------------------------------------------------- #
# Design frames
# --------------------------------------------------------------------------- #

def trial_frame(bundle, values, *, factor2=None, factor2_name="factor2",
                covariate=None, covariate_name="covar"):
    """One row per trial: ``dv``, ``condition`` and optional 2nd factor/covariate.

    ``factor2`` may be a per-trial array (already categorical) or the string
    name of a numeric metadata column to median-split into ``low``/``high``.
    """
    pd = require("pandas")
    cond = np.array([bundle.condition_names[i] for i in bundle.y])
    data = {"dv": np.asarray(values, float), "condition": cond}
    if factor2 is not None:
        if isinstance(factor2, str):
            raw = np.asarray(bundle.metadata[factor2], float)
            med = np.nanmedian(raw)
            data[factor2_name] = np.where(raw >= med, "high", "low")
        else:
            data[factor2_name] = np.asarray(factor2)
    if covariate is not None:
        col = bundle.metadata[covariate] if isinstance(covariate, str) else covariate
        data[covariate_name] = np.asarray(col, float)
    return pd.DataFrame(data)


def subject_condition_frame(bundle, values, *, subject=None):
    """Aggregate to one mean ``dv`` per (subject, condition).

    ``subject`` is a metadata key or per-trial array; defaults to ``groups``
    (session/subject id). Subjects missing any condition are dropped (RM/mixed
    ANOVA require a balanced, fully-crossed design), and reported.
    """
    pd = require("pandas")
    if subject is None:
        subj = bundle.groups
        subj_name = "subject"
    elif isinstance(subject, str):
        subj = np.asarray(bundle.metadata[subject])
        subj_name = subject
    else:
        subj = np.asarray(subject)
        subj_name = "subject"
    cond = np.array([bundle.condition_names[i] for i in bundle.y])
    df = pd.DataFrame({"subject": subj, "condition": cond,
                       "dv": np.asarray(values, float)})
    agg = df.groupby(["subject", "condition"], as_index=False)["dv"].mean()
    n_cond = agg["condition"].nunique()
    counts = agg.groupby("subject")["condition"].nunique()
    complete = counts[counts == n_cond].index
    dropped = int((counts < n_cond).sum())
    agg = agg[agg["subject"].isin(complete)]
    return agg, dropped, subj_name


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #

def _p_of(row: dict):
    for key in ("p-GG-corr", "p-corr", "p-unc"):
        if key in row and row[key] is not None and np.isfinite(row[key]):
            return float(row[key])
    return None


def _es_of(row: dict):
    for key in ("np2", "ng2", "n2"):
        if key in row and row[key] is not None and np.isfinite(row[key]):
            return f"{key}={row[key]:.3f}"
    return ""


def _finalize(table, name, alpha, *, exclude=("Error", "Residual", "Within"),
              posthoc=None, sphericity=None, diagnostics=None):
    rows = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            for r in table.to_dict("records")]
    sig, min_p = [], None
    for r in rows:
        if str(r.get("Source", "")) in exclude:
            continue
        p = _p_of(r)
        if p is None:
            continue
        min_p = p if min_p is None else min(min_p, p)
        if p < alpha:
            sig.append(str(r.get("Source", "")))
    verdict = ("suggests significant effect(s) of " + ", ".join(sig)) if sig \
        else "does not provide evidence for the tested effect(s)"
    interp = (f"At alpha={alpha:g}, the {name} {verdict} (smallest "
              f"p={min_p:.3g}). Effect sizes are partial/generalised eta-squared; "
              f"interpret as estimates, not proof, and mind assumptions and "
              f"multiple testing.") if min_p is not None else \
        f"The {name} could not be evaluated."
    ph_rows, ph_cols = [], []
    if posthoc is not None:
        ph_rows = [{k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in r.items()} for r in posthoc.to_dict("records")]
        ph_cols = list(posthoc.columns)
    return AnovaResult(
        name=name, table=rows, columns=list(table.columns), posthoc=ph_rows,
        posthoc_columns=ph_cols, sphericity=sphericity,
        diagnostics=list(diagnostics or []), interpretation=interp,
        extra={"significant": sig, "min_p": min_p})


# --------------------------------------------------------------------------- #
# ANOVA variants (each takes a long DataFrame)
# --------------------------------------------------------------------------- #

def one_way(df, *, dv="dv", between="condition", alpha=0.05, posthoc=True,
            correction="holm", parametric=True):
    pg = require("pingouin")
    if parametric:
        table = pg.anova(data=df, dv=dv, between=between, detailed=True)
    else:
        table = pg.kruskal(data=df, dv=dv, between=between)
    ph = pg.pairwise_tests(data=df, dv=dv, between=between, parametric=parametric,
                           padjust=_PADJUST.get(correction, "holm")) \
        if posthoc else None
    name = "One-way ANOVA" if parametric else "Kruskal-Wallis"
    return _finalize(table, name, alpha, posthoc=ph)


def factorial(df, *, dv="dv", factors=("condition", "factor2"), alpha=0.05,
              posthoc=True, correction="holm"):
    pg = require("pingouin")
    table = pg.anova(data=df, dv=dv, between=list(factors), detailed=True)
    ph = pg.pairwise_tests(data=df, dv=dv, between=list(factors),
                           padjust=_PADJUST.get(correction, "holm")) \
        if posthoc else None
    return _finalize(table, f"Factorial ANOVA ({' x '.join(factors)})", alpha,
                     posthoc=ph)


def repeated_measures(df, *, dv="dv", within="condition", subject="subject",
                      alpha=0.05, posthoc=True, correction="holm",
                      parametric=True):
    pg = require("pingouin")
    diagnostics = []
    if parametric:
        table = pg.rm_anova(data=df, dv=dv, within=within, subject=subject,
                            detailed=True, correction=True)
        sph = None
        row0 = table.to_dict("records")[0]
        if "sphericity" in row0:
            sph = {"sphericity_ok": bool(row0.get("sphericity", True)),
                   "W": row0.get("W-spher"), "p": row0.get("p-spher"),
                   "eps": row0.get("eps")}
            if sph["sphericity_ok"] is False:
                diagnostics.append(Diagnostic(
                    WARNING, "Sphericity is violated; Greenhouse-Geisser "
                    "corrected p (p-GG-corr) is used."))
        name = "Repeated-measures ANOVA"
    else:
        table = pg.friedman(data=df, dv=dv, within=within, subject=subject)
        sph = None
        name = "Friedman test"
    ph = pg.pairwise_tests(data=df, dv=dv, within=within, subject=subject,
                           parametric=parametric,
                           padjust=_PADJUST.get(correction, "holm")) \
        if posthoc else None
    return _finalize(table, name, alpha, posthoc=ph, sphericity=sph,
                     diagnostics=diagnostics)


def mixed(df, *, dv="dv", within="condition", between="group", subject="subject",
          alpha=0.05, posthoc=True, correction="holm"):
    pg = require("pingouin")
    table = pg.mixed_anova(data=df, dv=dv, within=within, between=between,
                           subject=subject)
    ph = pg.pairwise_tests(data=df, dv=dv, within=within, between=between,
                           subject=subject,
                           padjust=_PADJUST.get(correction, "holm")) \
        if posthoc else None
    return _finalize(table, "Mixed ANOVA", alpha, posthoc=ph)


def ancova(df, *, dv="dv", between="condition", covar="covar", alpha=0.05):
    pg = require("pingouin")
    table = pg.ancova(data=df, dv=dv, between=between, covar=covar)
    return _finalize(table, f"ANCOVA (covariate: {covar})", alpha)
