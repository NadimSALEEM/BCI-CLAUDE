"""End-to-end demo workflow for the analysis suite (headless).

Mirrors the documented demo: load -> select two events -> epoch -> ERP
mean-amplitude paired t-test -> cluster-permutation test -> train an ERP
classifier -> compare LDA/SVM/RF -> confusion matrix + chance test -> export
report. Exercises the same cores the Statistics and ML tabs drive.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurobci.analysis.datasource import bundle_from_session
from neurobci.analysis.ml import evaluation as ev
from neurobci.analysis.ml import models as ml_models
from neurobci.analysis.ml import report as ml_report
from neurobci.analysis.ml.features import MLFeatureConfig, extract_features
from neurobci.analysis.ml.pipelines import PreprocOptions, build_pipeline
from neurobci.analysis.shared.config import AnalysisConfig
from neurobci.analysis.stats import mass_univariate as mu
from neurobci.analysis.stats import report as stats_report
from neurobci.analysis.stats import tests as stats_tests
from neurobci.analysis.stats.features import (ERPFeatureSpec, compute_feature,
                                              feature_by_condition)
from neurobci.bci.erp_analysis import Condition
from neurobci.paradigms.base import EpochWindow
from neurobci.recording.exporter import LoadedSession

SF = 128.0
NAMES = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]


def _session(per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    step = int(SF)
    n = (2 * per_class + 2) * step
    data = rng.standard_normal((n, len(NAMES))).astype(np.float32)
    markers = []
    for i in range(2 * per_class):
        onset = (i + 1) * step
        label = "target" if i % 2 == 0 else "nontarget"
        markers.append({"label": label, "sample": int(onset), "t": onset / SF})
        if label == "target":
            c = onset + int(0.3 * SF)
            idx = np.arange(c - 8, c + 8)
            bump = 4.0 * np.exp(-0.5 * ((idx - c) / 3.0) ** 2)
            data[idx, 1] += bump.astype(np.float32)
            data[idx, 2] += (0.7 * bump).astype(np.float32)
    meta = {"channel_names": list(NAMES), "channel_kinds": ["eeg"] * len(NAMES),
            "sfreq_nominal": SF, "n_channels": len(NAMES)}
    return LoadedSession(path=Path("mem"), meta=meta, data=data,
                         timestamps=np.arange(n) / SF, markers=markers)


class TestDemoWorkflow(unittest.TestCase):
    def test_full_demo(self):
        # 1-3: load, select two events, epoch
        window = EpochWindow(tmin=-0.1, tmax=0.5, baseline=(-0.1, 0.0), reject_uv=1e9)
        bundle = bundle_from_session(
            _session(), [Condition("target", ["target"]),
                         Condition("nontarget", ["nontarget"])], window)
        self.assertEqual(set(bundle.condition_names), {"target", "nontarget"})

        # 4: ERP mean-amplitude paired t-test at Cz
        spec = ERPFeatureSpec("mean_amplitude", 0.25, 0.45, channels=["Cz"])
        by = feature_by_condition(compute_feature(bundle, spec), bundle)
        t_res = stats_tests.paired_t(by["target"], by["nontarget"])
        self.assertLess(t_res.p_value, 0.01)
        cfg = AnalysisConfig(kind="stats", analysis_type="paired_ttest",
                             selection={"conditions": bundle.condition_names})
        rep = stats_report.report_for_test(cfg, bundle, t_res)
        self.assertIn("Interpretation", rep.to_html())

        # 5: cluster-based permutation test on the Cz ERP
        ci = bundle.channel_index("Cz")
        A = bundle.condition_trials("target")[:, ci, :]
        B = bundle.condition_trials("nontarget")[:, ci, :]
        clust = mu.cluster_permutation_time(A, B, n_permutations=300, seed=0)
        self.assertTrue(any(c.p_value < 0.1 for c in clust.clusters))

        # 6-8: train ERP classifier, compare LDA/SVM/RF, confusion + chance
        X, _ = extract_features(bundle, MLFeatureConfig(
            [ERPFeatureSpec("mean_amplitude", 0.2, 0.45, channels=["Cz", "Pz"])]))
        cv, grouped, desc = ev.make_cv("stratified_kfold", n_splits=5, seed=0)
        evals = []
        for key in ("lda", "svm_linear", "rf"):
            spec_m = ml_models.get_spec(key)
            pipe = build_pipeline(spec_m, dict(spec_m.default_params),
                                  PreprocOptions())
            res = ev.evaluate_classification(
                pipe, X, bundle.y, key=key, label=spec_m.label, cv=cv,
                cv_desc=desc, class_names=bundle.condition_names)
            evals.append(res)
        ev.add_chance_level(evals[0], build_pipeline(ml_models.get_spec("lda"), {},
                                                     PreprocOptions()),
                            X, bundle.y, cv=cv, n_permutations=50, seed=0)
        ranked = ev.rank_models(evals)
        self.assertGreater(ranked[0].mean_score, 0.7)
        self.assertEqual(ranked[0].confusion.shape, (2, 2))

        # 9: export ML report
        mlcfg = AnalysisConfig(kind="ml", analysis_type="classification",
                               features={"families": ["ERP window"]},
                               params={"cv_desc": desc, "scoring": "balanced_accuracy"})
        ml_rep = ml_report.report_for_models(mlcfg, bundle, ranked)
        with tempfile.TemporaryDirectory() as d:
            path = ml_rep.save(Path(d) / "ml.html")
            self.assertTrue(path.exists())
            self.assertIn("Ranked models", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
