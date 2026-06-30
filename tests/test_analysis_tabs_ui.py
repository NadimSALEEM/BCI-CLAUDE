"""Offscreen UI-path smoke for the Statistics and ML tabs.

Feeds a synthetic epoch bundle straight into each tab (bypassing the Replay
load) and drives a synchronous analysis, so the widget wiring -- signals,
feature specs, plotting, report rendering -- is exercised without a live
session. Skips cleanly if Qt is unavailable.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from neurobci.analysis.datasource import bundle_from_session  # noqa: E402
from neurobci.analysis.shared.history import AnalysisHistory  # noqa: E402
from neurobci.bci.erp_analysis import Condition  # noqa: E402
from neurobci.paradigms.base import EpochWindow  # noqa: E402
from neurobci.recording.exporter import LoadedSession  # noqa: E402

SF = 128.0
NAMES = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]


def _bundle(per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    step = int(SF)
    n = (2 * per_class + 2) * step
    data = rng.standard_normal((n, len(NAMES))).astype(np.float32)
    markers = []
    for i in range(2 * per_class):
        onset = (i + 1) * step
        label = "A" if i % 2 == 0 else "B"
        markers.append({"label": label, "sample": int(onset), "t": onset / SF})
        if label == "A":
            c = onset + int(0.3 * SF)
            idx = np.arange(c - 8, c + 8)
            data[idx, 1] += (4.0 * np.exp(-0.5 * ((idx - c) / 3.0) ** 2)).astype(np.float32)
    meta = {"channel_names": list(NAMES), "channel_kinds": ["eeg"] * len(NAMES),
            "sfreq_nominal": SF, "n_channels": len(NAMES)}
    session = LoadedSession(path=Path("mem"), meta=meta, data=data,
                            timestamps=np.arange(n) / SF, markers=markers)
    win = EpochWindow(tmin=-0.1, tmax=0.5, baseline=(-0.1, 0.0), reject_uv=1e9)
    return bundle_from_session(session, [Condition("A", ["A"]),
                                         Condition("B", ["B"])], win)


def _stub_controller():
    return SimpleNamespace(
        replay_ws=SimpleNamespace(_session_path=None),
        engine=SimpleNamespace(pipeline=None, stream_info=None,
                               native_stream_info=None, keep_indices=None),
        config=SimpleNamespace(),
        analysis_history=AnalysisHistory(Path(tempfile.mkdtemp()) / "h.json"))


class TestAnalysisTabsUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from PyQt5 import QtWidgets
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"Qt unavailable: {exc}")
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_statistics_two_group_run(self):
        from neurobci.ui.workspaces.statistics_ws import StatisticsWorkspace
        ws = StatisticsWorkspace(_stub_controller())
        ws._on_bundle(_bundle())
        ws.mode.setCurrentText("Two-group test")
        ws.feature.setCurrentText("mean_amplitude")
        ws.test.setCurrentText("Independent t-test")
        ws.tmin.setValue(0.25)
        ws.tmax.setValue(0.45)
        ws._run()
        self.assertIn("Interpretation", ws.summary.toPlainText())

    def test_statistics_cluster_run(self):
        from neurobci.ui.workspaces.statistics_ws import StatisticsWorkspace
        ws = StatisticsWorkspace(_stub_controller())
        ws._on_bundle(_bundle())
        ws.mode.setCurrentText("Cluster permutation (time)")
        ws.nperm.setValue(200)
        ws._run()
        self.assertIn("cluster", ws.summary.toPlainText().lower())

    def test_ml_tab_features_and_models(self):
        from neurobci.ui.workspaces.machine_learning_ws import \
            MachineLearningWorkspace
        ws = MachineLearningWorkspace(_stub_controller())
        ws._on_bundle(_bundle())
        ws.feature_set.setCurrentText("ERP window (per channel)")
        ws._refresh_models()
        self.assertGreater(ws.model_list.count(), 0)
        X, names = ws._features()
        self.assertEqual(X.shape[0], ws._bundle.n_trials)
        self.assertEqual(X.shape[1], len(names))


if __name__ == "__main__":
    unittest.main()
