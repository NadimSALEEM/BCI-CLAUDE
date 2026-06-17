"""Pipeline build/order/enable/mode/validate behaviour."""

import unittest

import numpy as np

from neurobci.config.schema import PreprocessingConfig
from neurobci.core.stream_info import KIND_EEG, KIND_EOG
from neurobci.preprocessing.pipeline import MODE_CAUSAL, MODE_OFFLINE, Pipeline

KINDS = [KIND_EEG, KIND_EEG, KIND_EOG]


def _pipe(stages, mode=MODE_CAUSAL, enabled=True):
    cfg = PreprocessingConfig(enabled=enabled, mode=mode, stages=stages)
    return Pipeline.from_config(cfg, sfreq=500.0, ch_kinds=KINDS)


class TestPipeline(unittest.TestCase):
    def test_build_from_config_dicts(self):
        p = _pipe([
            {"type": "highpass", "enabled": True, "params": {"cutoff_hz": 1.0}},
            {"type": "car", "enabled": True, "params": {}},
        ])
        self.assertEqual([s.type_name for s in p.stages], ["highpass", "car"])

    def test_disabled_pipeline_is_passthrough(self):
        p = _pipe([{"type": "highpass", "enabled": True, "params": {}}], enabled=False)
        x = np.random.default_rng(0).standard_normal((100, 3))
        np.testing.assert_allclose(p.process_chunk(x), x)

    def test_disabled_stage_skipped(self):
        x = np.random.default_rng(1).standard_normal((200, 3))
        on = _pipe([{"type": "highpass", "enabled": True, "params": {"cutoff_hz": 5.0}}])
        off = _pipe([{"type": "highpass", "enabled": False, "params": {"cutoff_hz": 5.0}}])
        self.assertFalse(np.allclose(on.process_chunk(x), x))
        np.testing.assert_allclose(off.process_chunk(x), x)

    def test_detrend_skipped_in_causal_but_runs_offline(self):
        p = _pipe([{"type": "detrend", "enabled": True, "params": {}}], mode=MODE_OFFLINE)
        self.assertEqual(p.skipped_in_causal, ["detrend"])
        # Causal: passthrough (skipped).
        x = (np.linspace(0, 10, 300)[:, None] + np.zeros((300, 3))).copy()
        np.testing.assert_allclose(p.process_chunk(x), x)
        # Offline: linear trend removed -> mean ~ 0.
        y = p.apply_window(x, mode=MODE_OFFLINE)
        self.assertLess(abs(float(y.mean())), 1e-6)

    def test_move_reorders(self):
        p = _pipe([
            {"type": "highpass", "enabled": True, "params": {}},
            {"type": "lowpass", "enabled": True, "params": {}},
        ])
        p.move(0, 1)
        self.assertEqual([s.type_name for s in p.stages], ["lowpass", "highpass"])

    def test_validate_warns_on_empty_passband(self):
        p = _pipe([
            {"type": "highpass", "enabled": True, "params": {"cutoff_hz": 40.0}},
            {"type": "lowpass", "enabled": True, "params": {"cutoff_hz": 10.0}},
        ])
        warnings = p.validate()
        self.assertTrue(any("pass-band is empty" in w for w in warnings))

    def test_default_pipeline_runs(self):
        cfg = PreprocessingConfig()  # default stages
        p = Pipeline.from_config(cfg, 500.0, KINDS)
        x = np.random.default_rng(2).standard_normal((500, 3))
        y = p.process_chunk(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
