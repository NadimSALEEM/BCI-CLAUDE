"""Unit tests for the statistics + machine-learning analysis suite.

Synthetic sessions/epochs with a known condition difference exercise the data
bridge, feature extraction, statistical tests, multiple-comparison correction,
mass-univariate / cluster permutation, leakage-safe ML pipelines, evaluation
and chance-level testing, plus the shared config/history/export/validation
infrastructure and optional-dependency gating.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurobci.analysis.datasource import bundle_from_session, concat_bundles
from neurobci.analysis.ml import evaluation as ev
from neurobci.analysis.ml import models as ml_models
from neurobci.analysis.ml.features import (MLFeatureConfig, RawEpochsSpec,
                                           extract_features, tensor_X)
from neurobci.analysis.ml.pipelines import PreprocOptions, build_pipeline
from neurobci.analysis.shared import deps
from neurobci.analysis.shared.config import AnalysisConfig
from neurobci.analysis.shared.export import HTMLReport, save_json, save_table_csv
from neurobci.analysis.shared.history import AnalysisHistory
from neurobci.analysis.shared.validation import check_class_balance
from neurobci.analysis.stats import correction, mass_univariate
from neurobci.analysis.stats import tests as stats_tests
from neurobci.analysis.stats.features import (BandPowerSpec, ERPFeatureSpec,
                                              band_power_feature, erp_feature,
                                              feature_by_condition)
from neurobci.bci.erp_analysis import Condition
from neurobci.paradigms.base import EpochWindow
from neurobci.recording.exporter import LoadedSession

SF = 128.0
NAMES = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]
WINDOW = EpochWindow(tmin=-0.1, tmax=0.5, baseline=(-0.1, 0.0), reject_uv=1e9)


def make_session(per_class=40, seed=0, group=0):
    """Two conditions; condition A has a 0.3 s positive deflection at Cz/Pz."""
    rng = np.random.default_rng(seed)
    n_ch = len(NAMES)
    step = int(SF)                       # 1 s between onsets
    n_onsets = 2 * per_class
    n_samples = (n_onsets + 2) * step
    data = rng.standard_normal((n_samples, n_ch)).astype(np.float32)
    markers = []
    for i in range(n_onsets):
        onset = (i + 1) * step
        label = "A" if i % 2 == 0 else "B"
        markers.append({"label": label, "sample": int(onset),
                        "t": onset / SF, "rt": float(rng.uniform(0.4, 0.9))})
        if label == "A":
            c = onset + int(0.3 * SF)
            idx = np.arange(c - 8, c + 8)
            bump = 4.0 * np.exp(-0.5 * ((idx - c) / 3.0) ** 2)
            data[idx, 1] += bump.astype(np.float32)        # Cz
            data[idx, 2] += (0.7 * bump).astype(np.float32)  # Pz
    meta = {"channel_names": list(NAMES), "channel_kinds": ["eeg"] * n_ch,
            "sfreq_nominal": SF, "n_channels": n_ch}
    ts = np.arange(n_samples) / SF
    session = LoadedSession(path=Path("mem"), meta=meta, data=data,
                            timestamps=ts, markers=markers)
    return session


def make_bundle(per_class=40, seed=0, group=0):
    session = make_session(per_class, seed)
    return bundle_from_session(
        session, [Condition("A", ["A"]), Condition("B", ["B"])], WINDOW,
        group=group)


# --------------------------------------------------------------------------- #
class TestDataSource(unittest.TestCase):
    def test_bundle_shapes_and_labels(self):
        b = make_bundle(per_class=20)
        self.assertEqual(b.n_trials, 40)
        self.assertEqual(b.condition_names, ["A", "B"])
        self.assertEqual(b.X.shape[1], len(NAMES))
        self.assertEqual(b.class_counts(), {"A": 20, "B": 20})
        self.assertIn("rt", b.metadata)                  # behavioural metadata
        self.assertEqual(b.metadata["rt"].shape[0], 40)

    def test_subset_and_concat(self):
        b = make_bundle(per_class=15)
        sub = b.subset(["A", "B"])
        self.assertEqual(sub.n_trials, b.n_trials)
        b2 = make_bundle(per_class=15, seed=1, group=1)
        merged = concat_bundles([b, b2])
        self.assertEqual(merged.n_trials, b.n_trials + b2.n_trials)
        self.assertEqual(set(np.unique(merged.groups)), {0, 1})


# --------------------------------------------------------------------------- #
class TestStatsFeatures(unittest.TestCase):
    def test_erp_mean_amplitude_recovers_difference(self):
        b = make_bundle(per_class=40)
        spec = ERPFeatureSpec("mean_amplitude", 0.25, 0.4, channels=["Cz"])
        vals = erp_feature(b, spec)
        self.assertEqual(vals.shape, (80,))
        by = feature_by_condition(vals, b)
        self.assertGreater(by["A"].mean(), by["B"].mean() + 0.5)

    def test_band_power_positive_shape(self):
        b = make_bundle(per_class=10)
        p = band_power_feature(b, BandPowerSpec(8.0, 13.0, channels=["Cz", "Pz"]))
        self.assertEqual(p.shape, (20,))
        self.assertTrue(np.all(p >= 0))

    def test_peak_latency_in_window(self):
        b = make_bundle(per_class=10)
        lat = erp_feature(b, ERPFeatureSpec("peak_latency", 0.2, 0.45,
                                            channels=["Cz"], peak_sign="pos"))
        self.assertTrue(np.all((lat >= 0.2) & (lat <= 0.45)))


# --------------------------------------------------------------------------- #
class TestStatsTests(unittest.TestCase):
    def setUp(self):
        self.b = make_bundle(per_class=40)
        spec = ERPFeatureSpec("mean_amplitude", 0.25, 0.4, channels=["Cz"])
        by = feature_by_condition(erp_feature(self.b, spec), self.b)
        self.a, self.bb = by["A"], by["B"]

    def test_independent_t_detects_effect(self):
        r = stats_tests.independent_t(self.a, self.bb)
        self.assertLess(r.p_value, 0.01)
        self.assertGreater(abs(r.effect_size["cohen_d"]), 0.4)
        self.assertEqual(len(r.ci), 2)
        self.assertIn("shapiro_a_p", r.assumptions)

    def test_paired_and_nonparametric(self):
        rp = stats_tests.paired_t(self.a, self.bb)
        self.assertLess(rp.p_value, 0.05)
        rw = stats_tests.mann_whitney(self.a, self.bb)
        self.assertLess(rw.p_value, 0.05)
        self.assertIn("rank_biserial", rw.effect_size)

    def test_permutation_and_correlation(self):
        rperm = stats_tests.permutation_ttest(self.a, self.bb, n_permutations=2000,
                                              seed=0)
        self.assertLess(rperm.p_value, 0.05)
        rc = stats_tests.correlation(self.a, np.arange(self.a.size), "pearson")
        self.assertIn("r", rc.effect_size)

    def test_interpretation_is_cautious(self):
        r = stats_tests.independent_t(self.a, self.bb)
        self.assertIn("suggests", r.interpretation)
        self.assertNotIn("proves", r.interpretation.lower())


# --------------------------------------------------------------------------- #
class TestCorrection(unittest.TestCase):
    def test_methods_monotone_and_bounds(self):
        p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
        for method in correction.METHODS:
            rej, pc = correction.correct(p, method=method, alpha=0.05)
            self.assertTrue(np.all((pc >= 0) & (pc <= 1)))
            self.assertEqual(rej.shape, p.shape)
        # Bonferroni is the most conservative of the listed methods here.
        _, bonf = correction.correct(p, "bonferroni")
        _, bh = correction.correct(p, "fdr_bh")
        self.assertTrue(np.all(bonf >= bh - 1e-9))

    def test_handles_nan(self):
        p = np.array([0.01, np.nan, 0.2])
        rej, pc = correction.correct(p, "holm")
        self.assertTrue(np.isnan(pc[1]))
        self.assertFalse(rej[1])


# --------------------------------------------------------------------------- #
class TestMassUnivariate(unittest.TestCase):
    def test_pointwise_and_correction_separation(self):
        b = make_bundle(per_class=40)
        A = b.condition_trials("A")[:, b.channel_index("Cz"), :]
        B = b.condition_trials("B")[:, b.channel_index("Cz"), :]
        res = mass_univariate.mass_univariate_test(A, B, correction="fdr_bh")
        self.assertEqual(res.t_values.shape, (b.n_times,))
        # The deflection window should survive correction; uncorrected >= corrected.
        self.assertGreaterEqual(res.n_significant_uncorrected,
                                res.n_significant_corrected)
        self.assertGreater(res.n_significant_corrected, 0)

    def test_cluster_permutation_time_finds_cluster(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((30, 60))
        B = rng.standard_normal((30, 60))
        A[:, 20:35] += 1.2
        res = mass_univariate.cluster_permutation_time(
            A, B, paired=False, n_permutations=300, seed=0)
        self.assertTrue(any(c.p_value < 0.1 for c in res.clusters))
        self.assertTrue(res.significance_mask().any())

    def test_cluster_permutation_null_is_quiet(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((25, 50))
        B = rng.standard_normal((25, 50))
        res = mass_univariate.cluster_permutation_time(
            A, B, paired=True, n_permutations=300, seed=0)
        self.assertFalse(any(c.p_value < 0.05 for c in res.clusters))


# --------------------------------------------------------------------------- #
class TestMLPipelinesAndEval(unittest.TestCase):
    def _features(self, b):
        cfg = MLFeatureConfig(families=[
            ERPFeatureSpec("mean_amplitude", 0.2, 0.45, channels=["Cz", "Pz"])])
        return extract_features(b, cfg)

    def test_pipeline_is_leakage_safe_structure(self):
        spec = ml_models.get_spec("lda")
        pipe = build_pipeline(spec, {}, PreprocOptions(scaler="standard"))
        names = [s[0] for s in pipe.steps]
        self.assertEqual(names[0], "scaler")     # scaler precedes model, fit per fold
        self.assertEqual(names[-1], "model")

    def test_classification_above_chance(self):
        b = make_bundle(per_class=40)
        X, _ = self._features(b)
        spec = ml_models.get_spec("lda")
        pipe = build_pipeline(spec, {}, PreprocOptions())
        cv, grouped, desc = ev.make_cv("stratified_kfold", n_splits=5, seed=0)
        res = ev.evaluate_classification(pipe, X, b.y, key="lda", label="LDA",
                                         cv=cv, is_grouped=grouped, cv_desc=desc,
                                         class_names=b.condition_names)
        self.assertGreater(res.mean_score, 0.75)
        self.assertIsNotNone(res.confusion)
        self.assertIn("balanced_accuracy", res.metrics)
        self.assertEqual(res.confusion.shape, (2, 2))

    def test_chance_level_permutation(self):
        b = make_bundle(per_class=30)
        X, _ = self._features(b)
        spec = ml_models.get_spec("lda")
        pipe = build_pipeline(spec, {}, PreprocOptions())
        cv, grouped, desc = ev.make_cv("stratified_kfold", n_splits=5, seed=0)
        res = ev.evaluate_classification(pipe, X, b.y, cv=cv, cv_desc=desc,
                                         class_names=b.condition_names)
        ev.add_chance_level(res, pipe, X, b.y, cv=cv, n_permutations=60, seed=0)
        self.assertLess(abs(res.chance_score - 0.5), 0.15)
        self.assertLess(res.chance_p, 0.1)

    def test_grouped_cv_runs_with_groups(self):
        b1 = make_bundle(per_class=20, seed=0, group=0)
        b2 = make_bundle(per_class=20, seed=3, group=1)
        b = concat_bundles([b1, b2])
        X, _ = self._features(b)
        spec = ml_models.get_spec("lda")
        pipe = build_pipeline(spec, {}, PreprocOptions())
        cv, grouped, desc = ev.make_cv("logo")
        res = ev.evaluate_classification(pipe, X, b.y, cv=cv, is_grouped=grouped,
                                         groups=b.groups, cv_desc=desc,
                                         class_names=b.condition_names)
        self.assertTrue(grouped)
        self.assertEqual(res.fold_scores.size, 2)        # one fold per subject

    def test_tensor_features_for_spatial_models(self):
        b = make_bundle(per_class=10)
        X = tensor_X(b)
        self.assertEqual(X.shape, (20, len(NAMES), b.n_times))


# --------------------------------------------------------------------------- #
class TestModelRegistryGating(unittest.TestCase):
    def test_core_models_available_optional_gated(self):
        reg = ml_models.registry("classification")
        self.assertTrue(reg["lda"].available)
        self.assertTrue(reg["rf"].available)
        # xgboost is not installed in this environment -> gated, with a hint.
        self.assertFalse(reg["xgboost"].available)
        self.assertIn("xgboost", reg["xgboost"].install_hint)

    def test_only_available_filter(self):
        avail = ml_models.registry("classification", only_available=True)
        self.assertNotIn("xgboost", avail)
        self.assertIn("lda", avail)


# --------------------------------------------------------------------------- #
class TestSharedInfra(unittest.TestCase):
    def test_config_roundtrip_and_fingerprint(self):
        cfg = AnalysisConfig(kind="stats", analysis_type="paired_ttest",
                             selection={"conditions": ["A", "B"]}, seed=7)
        again = AnalysisConfig.from_json(cfg.to_json())
        self.assertEqual(again.analysis_type, "paired_ttest")
        self.assertEqual(cfg.fingerprint(), again.fingerprint())
        cfg2 = AnalysisConfig(kind="stats", analysis_type="paired_ttest",
                              selection={"conditions": ["A", "C"]}, seed=7)
        self.assertNotEqual(cfg.fingerprint(), cfg2.fingerprint())

    def test_history_add_get_persist(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "hist.json"
            h = AnalysisHistory(path)
            rec = h.add(AnalysisConfig(kind="ml", analysis_type="classification"),
                        summary={"best": "LDA"})
            self.assertEqual(len(h), 1)
            h2 = AnalysisHistory(path)               # reload from disk
            self.assertEqual(len(h2), 1)
            self.assertEqual(h2.get(rec.id).summary["best"], "LDA")

    def test_validation_balance(self):
        diags = check_class_balance(np.array([0] * 20 + [1] * 3), ["A", "B"])
        self.assertTrue(any(d.level == "warning" for d in diags))

    def test_exports(self):
        with tempfile.TemporaryDirectory() as d:
            p_json = save_json({"a": 1}, Path(d) / "c.json")
            p_csv = save_table_csv({"model": ["LDA"], "score": [0.9]},
                                   Path(d) / "t.csv")
            rep = HTMLReport("Demo").add_heading("X").add_keyvalue({"k": "v"})
            p_html = rep.save(Path(d) / "r.html")
            self.assertTrue(p_json.exists() and p_csv.exists() and p_html.exists())
            self.assertIn("Demo", p_html.read_text(encoding="utf-8"))

    def test_deps_status(self):
        s = deps.status()
        self.assertIn("mne", s)
        self.assertTrue(deps.have("numpy") if "numpy" in s else True)


if __name__ == "__main__":
    unittest.main()
