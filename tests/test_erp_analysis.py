"""Offline ERP analysis: marker grouping, averaging, counts, preprocessing."""

import unittest
from pathlib import Path

import numpy as np

from neurobci.bci.erp_analysis import (
    Condition,
    combine_markers,
    compute_erp,
    global_field_power,
    latency_index,
    marker_label_counts,
    select_channels,
)
from neurobci.config.schema import PreprocessingConfig
from neurobci.paradigms.base import EpochWindow
from neurobci.preprocessing.pipeline import Pipeline
from neurobci.recording.exporter import LoadedSession

SF = 100.0


def make_session(markers, n_samples=1000, n_channels=2):
    data = np.zeros((n_samples, n_channels), dtype=np.float32)
    # Stamp a per-epoch constant (= sample/100) on channel 0 around each onset
    # so an epoch's average is exactly that constant.
    for m in markers:
        o = m["sample"]
        data[o - 10:o + 41, 0] = o / 100.0
    meta = {
        "channel_names": [f"Ch{i+1}" for i in range(n_channels)],
        "channel_kinds": ["eeg"] * n_channels,
        "sfreq_nominal": SF,
        "n_channels": n_channels,
    }
    ts = np.arange(n_samples) / SF
    return LoadedSession(path=Path("mem"), meta=meta, data=data,
                         timestamps=ts, markers=markers)


WIN = EpochWindow(tmin=-0.1, tmax=0.4, baseline=None, reject_uv=1e12)


class TestErpAnalysis(unittest.TestCase):
    def setUp(self):
        self.markers = [
            {"label": "target", "sample": 200, "t": 2.0},
            {"label": "target", "sample": 400, "t": 4.0},
            {"label": "nontarget", "sample": 700, "t": 7.0},
        ]
        self.session = make_session(self.markers)

    def test_label_counts(self):
        self.assertEqual(marker_label_counts(self.session.markers),
                         {"nontarget": 1, "target": 2})

    def test_combine_markers_unions_onsets(self):
        combined = combine_markers(self.markers, ["target", "nontarget"], "all")
        self.assertEqual([m["label"] for m in combined], ["all", "all", "all"])
        self.assertEqual([m["sample"] for m in combined], [200, 400, 700])

    def test_combine_markers_dedupes_and_sorts(self):
        markers = [
            {"label": "a", "sample": 400},
            {"label": "b", "sample": 400},   # same onset -> one event
            {"label": "a", "sample": 100},
        ]
        combined = combine_markers(markers, ["a", "b"], "m")
        self.assertEqual([m["sample"] for m in combined], [100, 400])

    def test_combined_marker_epochs_like_a_group(self):
        # A derived "all" marker should average identically to grouping labels.
        derived = combine_markers(self.markers, ["target", "nontarget"], "all")
        markers = self.markers + derived
        session = make_session(self.markers)        # data is the same stamps
        session.markers = markers
        res = compute_erp(session, [Condition("all", ["all"])], WIN,
                          preprocess=None)
        c = res.condition("all")
        self.assertEqual(c.n_epochs, 3)
        np.testing.assert_allclose(
            c.average[0], np.full(WIN.n_times(SF), (2.0 + 4.0 + 7.0) / 3.0))

    def test_average_per_condition(self):
        conds = [Condition("A", ["target"]), Condition("B", ["nontarget"])]
        res = compute_erp(self.session, conds, WIN, preprocess=None)
        a = res.condition("A")
        b = res.condition("B")
        self.assertEqual(a.n_epochs, 2)
        self.assertEqual(b.n_epochs, 1)
        # A averages the two target constants (2.0, 4.0) -> 3.0 everywhere.
        np.testing.assert_allclose(a.average[0], np.full(WIN.n_times(SF), 3.0))
        np.testing.assert_allclose(b.average[0], np.full(WIN.n_times(SF), 7.0))
        # Untouched channel stays flat at zero.
        np.testing.assert_allclose(a.average[1], 0.0)

    def test_label_grouping(self):
        conds = [Condition("all", ["target", "nontarget"])]
        res = compute_erp(self.session, conds, WIN, preprocess=None)
        c = res.condition("all")
        self.assertEqual(c.n_epochs, 3)
        self.assertEqual(c.n_onsets, 3)
        np.testing.assert_allclose(
            c.average[0], np.full(WIN.n_times(SF), (2.0 + 4.0 + 7.0) / 3.0))

    def test_unmatched_labels_ignored(self):
        conds = [Condition("only_t", ["target"])]
        res = compute_erp(self.session, conds, WIN, preprocess=None)
        self.assertEqual(len(res.conditions), 1)
        self.assertEqual(res.condition("only_t").n_onsets, 2)

    def test_out_of_bounds_counted(self):
        markers = [{"label": "x", "sample": 5}, {"label": "x", "sample": 500}]
        session = make_session(markers)
        res = compute_erp(session, [Condition("x", ["x"])], WIN, preprocess=None)
        c = res.condition("x")
        self.assertEqual(c.n_onsets, 2)
        self.assertEqual(c.n_epochs, 1)
        self.assertEqual(c.n_out_of_bounds, 1)

    def test_preprocessing_reused_offline(self):
        # A CAR-only configured pipeline, reused unchanged, must run and be
        # reported; CAR centres channels so the per-sample channel mean is ~0.
        pre = PreprocessingConfig(
            enabled=True, mode="offline",
            stages=[{"type": "car", "enabled": True, "params": {}}])
        pipe = Pipeline.from_config(
            pre, sfreq=SF,
            ch_kinds=self.session.channel_kinds,
            ch_names=self.session.channel_names)
        res = compute_erp(self.session, [Condition("A", ["target"])], WIN,
                          preprocess=pipe)
        self.assertIn("Configured pipeline (offline)", res.preprocessing)
        self.assertIn("car", res.preprocessing)
        # After CAR the two channels are mirror images -> their sum is ~0.
        avg = res.condition("A").average
        np.testing.assert_allclose(avg[0] + avg[1], 0.0, atol=1e-6)

    def test_select_channels_drops_columns(self):
        session = make_session(self.markers, n_channels=4)
        out = select_channels(session, [0, 2], ["A", "C"], ["eeg", "eog"])
        self.assertEqual(out.data.shape[1], 2)
        self.assertEqual(out.channel_names, ["A", "C"])
        self.assertEqual(out.channel_kinds, ["eeg", "eog"])
        self.assertEqual(out.meta["n_channels"], 2)
        np.testing.assert_array_equal(out.data[:, 0], session.data[:, 0])
        np.testing.assert_array_equal(out.data[:, 1], session.data[:, 2])
        # The original session is left untouched.
        self.assertEqual(session.data.shape[1], 4)

    def test_times_and_helpers(self):
        res = compute_erp(self.session, [Condition("A", ["target"])], WIN,
                          preprocess=None)
        self.assertAlmostEqual(float(res.times[0]), -0.1, places=6)
        self.assertEqual(latency_index(res.times, 0.0), 10)
        gfp = global_field_power(res.condition("A").average, res.eeg_indices)
        self.assertEqual(gfp.shape[0], WIN.n_times(SF))


if __name__ == "__main__":
    unittest.main()
