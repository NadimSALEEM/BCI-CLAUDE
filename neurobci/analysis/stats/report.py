"""Automatic statistical reports (HTML).

Assembles a cautious, scientific report from a :class:`TestResult` or a cluster
result: data provenance, sample sizes, feature settings, the test, assumption
checks, the statistic / p / corrected-p / effect size / CI, warnings and a
plain-language interpretation that never overclaims.
"""

from __future__ import annotations

import numpy as np

from neurobci.analysis.shared.export import HTMLReport


def _fmt_p(p) -> str:
    return "< 1e-4" if p is not None and p < 1e-4 else f"{p:.4g}"


def _data_section(rep: HTMLReport, config, bundle) -> None:
    rep.add_heading("Data source")
    src = dict(config.data_source)
    src["conditions"] = ", ".join(f"{k} (n={v})"
                                  for k, v in bundle.class_counts().items())
    src["preprocessing"] = bundle.preprocessing
    rep.add_keyvalue(src)


def report_for_test(config, bundle, result, figures=None) -> HTMLReport:
    rep = HTMLReport(f"Statistics report — {result.name}")
    rep.add_paragraph(
        f"Analysis level: {config.level} · design: {config.design} · "
        f"seed: {config.seed}", css="muted")
    _data_section(rep, config, bundle)

    rep.add_heading("Feature")
    rep.add_keyvalue(config.features or {"feature": "(per-trial scalar)"})

    rep.add_heading("Test result")
    es = ", ".join(f"{k}={v:.3f}" for k, v in result.effect_size.items()
                   if v is not None and np.isfinite(v))
    kv = {"Test": result.name, "Statistic": f"{result.statistic:.4f}",
          "df": result.df, "p-value": _fmt_p(result.p_value),
          "Alternative": result.alternative, "Effect size": es or "—"}
    if result.ci is not None:
        kv[result.ci_label or "CI"] = f"[{result.ci[0]:.4g}, {result.ci[1]:.4g}]"
    if "p_corrected" in result.extra:
        kv["Corrected p (%s)" % result.extra.get("correction", "")] = \
            _fmt_p(result.extra["p_corrected"])
    kv["n"] = result.n
    rep.add_keyvalue(kv)

    if result.assumptions:
        rep.add_heading("Assumption checks")
        rep.add_keyvalue({k: ("—" if v is None or not np.isfinite(v) else f"{v:.4f}")
                          for k, v in result.assumptions.items()})

    if result.diagnostics:
        rep.add_heading("Warnings & diagnostics")
        rep.add_diagnostics(result.diagnostics)

    rep.add_heading("Interpretation")
    rep.add_paragraph(result.interpretation)

    for fig, cap in (figures or []):
        rep.add_figure(fig, cap)
    return rep


def report_for_anova(config, bundle, result, figures=None) -> HTMLReport:
    rep = HTMLReport(f"Statistics report — {result.name}")
    rep.add_paragraph(
        f"Analysis level: {config.level} · design: {config.design}", css="muted")
    _data_section(rep, config, bundle)

    rep.add_heading("Feature")
    rep.add_keyvalue(config.features or {"feature": "(per-trial scalar)"})

    rep.add_heading("ANOVA table")
    rep.add_table(result.columns,
                  [[row.get(c, "") for c in result.columns] for row in result.table])
    if result.sphericity:
        rep.add_heading("Sphericity")
        rep.add_keyvalue(result.sphericity)

    if result.posthoc:
        rep.add_heading("Post-hoc pairwise comparisons")
        rep.add_table(result.posthoc_columns,
                      [[row.get(c, "") for c in result.posthoc_columns]
                       for row in result.posthoc])

    if result.diagnostics:
        rep.add_heading("Warnings & diagnostics")
        rep.add_diagnostics(result.diagnostics)

    rep.add_heading("Interpretation")
    rep.add_paragraph(result.interpretation)
    for fig, cap in (figures or []):
        rep.add_figure(fig, cap)
    return rep


def report_for_cluster(config, bundle, cluster, figures=None) -> HTMLReport:
    rep = HTMLReport("Statistics report — cluster-based permutation test")
    rep.add_paragraph(
        f"Analysis level: {config.level} · seed: {cluster.seed} · "
        f"engine: {cluster.engine}", css="muted")
    _data_section(rep, config, bundle)

    rep.add_heading("Mass-univariate / cluster settings")
    rep.add_keyvalue({
        "Permutations": cluster.n_permutations,
        "Cluster-forming threshold (|t|)": f"{cluster.threshold:.3f}",
        "Tail": cluster.tail, "alpha": cluster.alpha, "paired": cluster.paired})

    rep.add_heading("Clusters")
    sig = cluster.significant()
    rows = [[i + 1, c.extent, f"{c.mass:.2f}", _fmt_p(c.p_value),
             "yes" if c.p_value <= cluster.alpha else "no"]
            for i, c in enumerate(cluster.clusters)]
    rep.add_table(["#", "extent", "mass", "p (perm)", "significant"], rows)
    rep.add_paragraph(
        f"{len(sig)} of {len(cluster.clusters)} candidate cluster(s) survived "
        f"permutation correction at alpha={cluster.alpha:g}. Cluster extent and "
        f"location are not precisely localised — interpret as evidence that the "
        f"conditions differ somewhere within the cluster, not at a single point.",
        css="muted")

    if cluster.diagnostics:
        rep.add_diagnostics(cluster.diagnostics)
    for fig, cap in (figures or []):
        rep.add_figure(fig, cap)
    return rep
