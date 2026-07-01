"""Tests for the ANOVA family + post-hoc and the design-frame builders."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from neurobci.analysis.datasource import bundle_from_session, concat_bundles
from neurobci.analysis.stats import anova
from neurobci.bci.erp_analysis import Condition
from neurobci.paradigms.base import EpochWindow
from neurobci.recording.exporter import LoadedSession

_SF = 128.0
_NAMES = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]
_WIN = EpochWindow(tmin=-0.1, tmax=0.5, baseline=(-0.1, 0.0), reject_uv=1e9)


def _pandas():
    import pandas as pd
    return pd


def _one_bundle(per_class, seed, group):
    rng = np.random.default_rng(seed)
    step = int(_SF)
    n = (2 * per_class + 2) * step
    data = rng.standard_normal((n, len(_NAMES))).astype(np.float32)
    markers = []
    for i in range(2 * per_class):
        onset = (i + 1) * step
        label = "A" if i % 2 == 0 else "B"
        markers.append({"label": label, "sample": int(onset), "t": onset / _SF})
        if label == "A":                      # per-subject deflection at Cz
            c = onset + int(0.3 * _SF)
            idx = np.arange(c - 8, c + 8)
            data[idx, 1] += (4.0 * np.exp(-0.5 * ((idx - c) / 3.0) ** 2)).astype(np.float32)
    meta = {"channel_names": list(_NAMES), "channel_kinds": ["eeg"] * len(_NAMES),
            "sfreq_nominal": _SF, "n_channels": len(_NAMES)}
    session = LoadedSession(path=Path("mem"), meta=meta, data=data,
                            timestamps=np.arange(n) / _SF, markers=markers)
    return bundle_from_session(session, [Condition("A", ["A"]),
                                         Condition("B", ["B"])], _WIN, group=group)


def make_multisubject_bundle(n_subjects=6, per_class=12):
    return concat_bundles([_one_bundle(per_class, seed=s, group=s)
                           for s in range(n_subjects)])


def _rm_frame(n_subjects=14, seed=0):
    pd = _pandas()
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        base = rng.normal(0, 1)
        for c in ("A", "B", "C"):
            shift = 1.5 if c == "C" else 0.0
            rows.append({"subject": s, "condition": c,
                         "dv": base + shift + rng.normal(0, 0.4)})
    return pd.DataFrame(rows)


def _mixed_frame(n=14, seed=1):
    pd = _pandas()
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        grp = "old" if s % 2 else "young"
        base = rng.normal(0, 1)
        for c in ("pre", "post"):
            eff = (1.2 if c == "post" else 0.0) + (0.8 if grp == "old" else 0.0)
            rows.append({"subject": s, "group": grp, "condition": c,
                         "dv": base + eff + rng.normal(0, 0.4)})
    return pd.DataFrame(rows)


class TestAnovaCore(unittest.TestCase):
    def test_repeated_measures_detects_effect(self):
        res = anova.repeated_measures(_rm_frame(), alpha=0.05)
        self.assertIn("condition", res.extra["significant"])
        self.assertTrue(res.posthoc)                      # pairwise rows present
        self.assertIn("suggests", res.interpretation)

    def test_one_way_and_nonparametric(self):
        df = _rm_frame()
        r_param = anova.one_way(df, alpha=0.05)
        self.assertIn("condition", r_param.extra["significant"])
        r_np = anova.one_way(df, parametric=False, alpha=0.05)
        self.assertEqual(r_np.name, "Kruskal-Wallis")

    def test_factorial_two_factors(self):
        pd = _pandas()
        rng = np.random.default_rng(2)
        rows = []
        for _ in range(120):
            cond = rng.choice(["A", "B"])
            load = rng.choice(["low", "high"])
            dv = (1.0 if cond == "B" else 0) + (0.8 if load == "high" else 0) \
                + rng.normal(0, 0.5)
            rows.append({"condition": cond, "factor2": load, "dv": dv})
        res = anova.factorial(pd.DataFrame(rows), alpha=0.05)
        self.assertTrue(set(res.extra["significant"]) & {"condition", "factor2"})

    def test_mixed_runs(self):
        res = anova.mixed(_mixed_frame(), within="condition", between="group",
                          alpha=0.05)
        self.assertEqual(res.name, "Mixed ANOVA")
        self.assertTrue(res.table)

    def test_ancova_runs(self):
        pd = _pandas()
        rng = np.random.default_rng(3)
        n = 90
        cond = rng.choice(["A", "B"], n)
        cov = rng.normal(size=n)
        dv = np.where(cond == "B", 1.0, 0.0) + 0.5 * cov + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"condition": cond, "covar": cov, "dv": dv})
        res = anova.ancova(df, alpha=0.05)
        self.assertTrue(res.table)


class TestDesignFrames(unittest.TestCase):
    def test_subject_condition_aggregation(self):
        bundle = make_multisubject_bundle(n_subjects=6, per_class=12)
        vals = bundle.X[:, bundle.channel_index("Cz"), :].mean(axis=1)
        agg, dropped, subj_name = anova.subject_condition_frame(bundle, vals)
        # 6 subjects x 2 conditions = 12 rows, all complete
        self.assertEqual(len(agg), 12)
        self.assertEqual(dropped, 0)
        res = anova.repeated_measures(agg, within="condition", subject="subject")
        self.assertIn("condition", res.extra["significant"])

    def test_trial_frame_median_split(self):
        bundle = make_multisubject_bundle(n_subjects=2, per_class=20)
        vals = np.arange(bundle.n_trials, dtype=float)
        df = anova.trial_frame(bundle, vals, factor2=vals, factor2_name="tercile")
        self.assertEqual(set(df["condition"].unique()), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
