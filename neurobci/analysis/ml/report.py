"""Automatic machine-learning reports (HTML).

Summarises a model comparison: data and classes, feature/preprocessing/CV
settings, the ranked model table with chance-level results, per-model
diagnostics, and a cautious interpretation that separates predictive
performance from neuroscientific explanation.
"""

from __future__ import annotations

from neurobci.analysis.shared.export import HTMLReport


def report_for_models(config, bundle, evaluations, figures=None) -> HTMLReport:
    rep = HTMLReport("Machine-learning report — model comparison")
    rep.add_paragraph(
        f"Level: {config.level} · design: {config.design} · seed: {config.seed}",
        css="muted")

    rep.add_heading("Data source")
    src = dict(config.data_source)
    src["classes"] = ", ".join(f"{k} (n={v})"
                               for k, v in bundle.class_counts().items())
    src["preprocessing"] = bundle.preprocessing
    rep.add_keyvalue(src)

    rep.add_heading("Setup")
    rep.add_keyvalue({
        "Features": ", ".join(config.features.get("families", [])) or "—",
        "Scaling/selection": config.params.get("preproc", "—"),
        "Cross-validation": config.params.get("cv_desc", "—"),
        "Scoring": config.params.get("scoring", "—")})

    if not evaluations:
        rep.add_paragraph("No models were evaluated.")
        return rep

    rep.add_heading("Ranked models")
    headers = ["model", "CV mean", "CV std", "95% CI", "train", "chance",
               "p vs chance", "fit (s)", "n feat"]
    rows = []
    for e in evaluations:
        r = e.summary_row()
        rows.append([r["model"], r["cv_mean"], r["cv_std"], r["ci95"],
                     r["train"], r["chance"], r["p_vs_chance"], r["fit_s"],
                     r["n_feat"]])
    rep.add_table(headers, rows)

    best = evaluations[0]
    rep.add_heading("Best model")
    rep.add_keyvalue({
        "Model": best.label, "Pipeline": best.pipeline_desc,
        "Scoring": best.scoring,
        f"{best.scoring} (CV)": f"{best.mean_score:.3f} "
                                f"[{best.ci95[0]:.3f}, {best.ci95[1]:.3f}]",
        "Chance": "—" if best.chance_score is None else f"{best.chance_score:.3f}",
        "p vs chance": "—" if best.chance_p is None else f"{best.chance_p:.4g}"})
    if best.metrics:
        rep.add_keyvalue({k: round(v, 4) for k, v in best.metrics.items()})

    # Per-model diagnostics
    diags = [d for e in evaluations for d in e.diagnostics
             if getattr(d, "level", "info") != "info"]
    if diags:
        rep.add_heading("Warnings & diagnostics")
        rep.add_diagnostics(diags)

    rep.add_heading("Interpretation")
    verdict = ("above chance" if best.chance_p is not None and best.chance_p < 0.05
               else "not clearly above chance")
    rep.add_paragraph(
        f"The analysis suggests the best model ({best.label}) performs "
        f"{verdict} under {config.params.get('cv_desc', 'cross-validation')} "
        f"(mean {best.scoring}={best.mean_score:.3f}). Cross-validated scores "
        f"estimate generalisation, not mechanism: a high score shows the chosen "
        f"features are predictive, not which brain process drives them. Inspect "
        f"spatial patterns / importances for interpretation, and validate on "
        f"independent data before drawing conclusions. The 95% CI is a "
        f"t-interval over CV folds (a spread indicator, not a significance "
        f"test); significance comes from the permutation p / corrected p.")
    for fig, cap in (figures or []):
        rep.add_figure(fig, cap)
    return rep
