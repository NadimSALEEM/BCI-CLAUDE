"""Regression tests for the ML-tab audit fixes (determinism, balancing, ...).

Synthetic null (random features + random labels) and signal (known mean shift)
generators guard both false-positive control and true-positive sensitivity.
"""

from __future__ import annotations

import unittest

import numpy as np

from neurobci.analysis.ml import evaluation as ev
from neurobci.analysis.ml import models as M
from neurobci.analysis.ml.pipelines import (PreprocOptions, UniformPriorClassifier,
                                            balancing_capability, build_pipeline)


def _null(n=160, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, n)
    return X, y


def _signal(n=160, d=12, shift=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, d))
    X[y == 1, 0] += shift            # class-1 mean shift on feature 0
    return X, y


class TestRejectionStats(unittest.TestCase):
    """Item 0: rejection is ptp on the epoched preprocessed signal and its
    count is now surfaced on the bundle and depends on that signal."""

    def _session(self):
        from pathlib import Path
        from neurobci.recording.exporter import LoadedSession
        rng = np.random.default_rng(0)
        sf, nch = 250.0, 4
        step = int(sf)
        n = 42 * step
        data = (rng.standard_normal((n, nch)) * 5).astype(np.float32)
        markers = []
        for i in range(40):
            onset = (i + 1) * step
            markers.append({"label": "A" if i % 2 else "B", "sample": onset,
                            "t": onset / sf})
            if i % 4 == 0:               # slow linear drift -> big ptp, low freq
                s0, e0 = onset - int(0.2 * sf), onset + int(1.0 * sf)
                data[s0:e0, 0] += np.linspace(0, 400, e0 - s0).astype(np.float32)
        meta = {"channel_names": [f"E{i}" for i in range(nch)],
                "channel_kinds": ["eeg"] * nch, "sfreq_nominal": sf,
                "n_channels": nch}
        return LoadedSession(path=Path("mem"), meta=meta, data=data,
                             timestamps=np.arange(n) / sf, markers=markers)

    def test_reject_count_surfaced_and_signal_dependent(self):
        from neurobci.analysis.datasource import bundle_from_session
        from neurobci.bci.erp_analysis import Condition
        from neurobci.config.schema import PreprocessingConfig
        from neurobci.paradigms.base import EpochWindow
        from neurobci.preprocessing.pipeline import Pipeline
        s = self._session()
        conds = [Condition("A", ["A"]), Condition("B", ["B"])]
        win = EpochWindow(tmin=-0.2, tmax=1.0, baseline=(-0.2, 0.0), reject_uv=250.0)
        bp = Pipeline.from_config(PreprocessingConfig(enabled=True, mode="offline",
            stages=[{"type": "highpass", "enabled": True, "params": {"cutoff_hz": 4.0}},
                    {"type": "lowpass", "enabled": True, "params": {"cutoff_hz": 30.0}}]),
            s.sfreq, ["eeg"] * 4, s.channel_names)
        raw = bundle_from_session(s, conds, win, preprocess=None, reject=True)
        band = bundle_from_session(s, conds, win, preprocess=bp, reject=True)
        total = bundle_from_session(s, conds, win, preprocess=None, reject=False).n_trials
        # rejection count is now reported and is internally consistent
        self.assertEqual(raw.n_rejected + raw.n_trials, total)
        # the 0.2 Hz drift is rejected on raw but removed by the 4-30 band ->
        # fewer rejections -> proves rejection runs on the *preprocessed* signal
        self.assertGreater(raw.n_rejected, band.n_rejected)


class TestDownsampleAntiAlias(unittest.TestCase):
    def test_decimation_attenuates_above_new_nyquist(self):
        from neurobci.analysis.datasource import EpochBundle
        from neurobci.analysis.ml.features import (MLFeatureConfig, RawEpochsSpec,
                                                   extract_features)
        sf, T = 100.0, 120
        t = np.arange(T) / sf
        # 40 Hz tone: above the post-decimation (q=4 -> 25 Hz) Nyquist of 12.5 Hz
        tone = np.sin(2 * np.pi * 40 * t)
        X = np.tile(tone, (8, 4, 1))
        b = EpochBundle(X=X, y=np.array([0, 1] * 4), condition_names=["A", "B"],
                        times=t, sfreq=sf, channel_names=[f"E{i}" for i in range(4)],
                        channel_kinds=["eeg"] * 4, groups=np.zeros(8, int))
        Xd, names = extract_features(b, MLFeatureConfig([RawEpochsSpec(downsample=4)]))
        self.assertEqual(Xd.shape[1], len(names))
        # anti-alias low-pass should kill the 40 Hz tone (naive [::4] would alias it)
        self.assertLess(np.var(Xd), 0.05 * np.var(tone))


class TestDeterminism(unittest.TestCase):
    def _run(self, key, X, y):
        spec = M.get_spec(key)
        pipe = build_pipeline(spec, dict(spec.default_params), PreprocOptions())
        cv, _, desc = ev.make_cv("stratified_kfold", n_splits=5, seed=42)
        return ev.evaluate_classification(pipe, X, y, key=key, cv=cv, cv_desc=desc)

    def test_svm_rbf_bit_identical(self):
        X, y = _signal(seed=1)
        a = self._run("svm_rbf", X, y)
        b = self._run("svm_rbf", X, y)
        np.testing.assert_array_equal(a.fold_scores, b.fold_scores)
        # proba-derived metric now reproducible too (SVC seeded)
        self.assertEqual(a.metrics.get("roc_auc"), b.metrics.get("roc_auc"))

    def test_svm_linear_bit_identical(self):
        X, y = _signal(seed=2)
        a = self._run("svm_linear", X, y)
        b = self._run("svm_linear", X, y)
        np.testing.assert_array_equal(a.fold_scores, b.fold_scores)
        self.assertEqual(a.metrics.get("roc_auc"), b.metrics.get("roc_auc"))


class TestBalancing(unittest.TestCase):
    def test_capability_map(self):
        cap = {k: balancing_capability(M.get_spec(k)) for k in
               ["logreg", "svm_linear", "rf", "dtree", "lda", "qda", "gnb",
                "knn", "gradboost", "adaboost"]}
        self.assertEqual(cap["logreg"], "class_weight")
        self.assertEqual(cap["svm_linear"], "class_weight")
        self.assertEqual(cap["lda"], "priors")
        self.assertEqual(cap["qda"], "priors")
        self.assertEqual(cap["gnb"], "priors")
        for k in ("knn", "gradboost", "adaboost"):
            self.assertIsNone(cap[k], f"{k} should report no balancing")

    def test_lda_balanced_sets_uniform_priors(self):
        spec = M.get_spec("lda")
        pipe = build_pipeline(spec, {}, PreprocOptions(class_weight_balanced=True))
        self.assertIsInstance(pipe.steps[-1][1], UniformPriorClassifier)
        X, y = _null(n=240, seed=5)
        y = np.array([0] * 40 + [1] * 200)          # imbalanced
        pipe.fit(X[:240], y)
        priors = pipe.steps[-1][1].base_.priors_
        np.testing.assert_allclose(priors, [0.5, 0.5])

    def test_balancing_changes_lda_predictions(self):
        # Mild signal, strong imbalance -> balancing should raise minority recall.
        X, y = _signal(n=240, shift=1.2, seed=6)
        y[:200] = 1                                 # make class 1 dominate
        y[200:] = 0
        X[y == 0, 0] += 1.2
        from sklearn.metrics import recall_score
        from sklearn.model_selection import cross_val_predict
        spec = M.get_spec("lda")
        base = build_pipeline(spec, {}, PreprocOptions())
        bal = build_pipeline(spec, {}, PreprocOptions(class_weight_balanced=True))
        cv, _, _ = ev.make_cv("stratified_kfold", n_splits=5, seed=1)
        r_base = recall_score(y, cross_val_predict(base, X, y, cv=cv), pos_label=0)
        r_bal = recall_score(y, cross_val_predict(bal, X, y, cv=cv), pos_label=0)
        self.assertGreaterEqual(r_bal, r_base)      # minority recall not hurt

    def test_evaluate_runs_with_balanced_priors(self):
        X, y = _signal(seed=7)
        spec = M.get_spec("gnb")
        pipe = build_pipeline(spec, {}, PreprocOptions(class_weight_balanced=True))
        cv, _, desc = ev.make_cv("stratified_kfold", n_splits=5, seed=42)
        res = ev.evaluate_classification(pipe, X, y, cv=cv, cv_desc=desc)
        self.assertIn("roc_auc", res.metrics)       # proba still flows through wrapper


class TestOmnibus(unittest.TestCase):
    def _panel(self):
        return [(k, build_pipeline(M.get_spec(k), {}, PreprocOptions()))
                for k in ("lda", "gnb", "knn")]

    def test_null_not_significant_and_selection_aware(self):
        X, y = _null(n=160, seed=3)
        cv, _, _ = ev.make_cv("stratified_kfold", n_splits=5, seed=42)
        r = ev.omnibus_permutation(self._panel(), X, y, cv=cv,
                                   n_permutations=150, seed=42)
        self.assertGreater(r.p_omnibus, 0.05)          # random labels -> chance
        self.assertEqual(r.corrected_p[r.best_key], r.p_omnibus)
        self.assertEqual(len(r.observed), 3)

    def test_signal_is_significant(self):
        X, y = _signal(n=200, shift=1.6, seed=4)
        cv, _, _ = ev.make_cv("stratified_kfold", n_splits=5, seed=42)
        r = ev.omnibus_permutation(self._panel(), X, y, cv=cv,
                                   n_permutations=150, seed=42)
        self.assertLess(r.p_omnibus, 0.05)

    def test_correction_is_more_conservative_than_naive(self):
        # Weak/no signal: the best-of-panel omnibus p should be >= the naive
        # single-best permutation p (this is the SVM-RBF false-positive guard).
        from sklearn.model_selection import permutation_test_score
        X, y = _null(n=160, seed=8)
        cv, _, _ = ev.make_cv("stratified_kfold", n_splits=5, seed=42)
        panel = self._panel()
        r = ev.omnibus_permutation(panel, X, y, cv=cv, n_permutations=200, seed=42)
        best_pipe = dict(panel)[r.best_key]
        _, _, naive_p = permutation_test_score(
            best_pipe, X, y, cv=cv, scoring="balanced_accuracy",
            n_permutations=200, random_state=42)
        self.assertGreaterEqual(r.p_omnibus + 1e-9, naive_p)


if __name__ == "__main__":
    unittest.main()
