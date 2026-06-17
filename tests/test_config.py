"""Config defaults, (de)serialisation, profile round-trip."""

import tempfile
import unittest
from pathlib import Path

from neurobci.config.manager import ConfigManager
from neurobci.config.schema import AppConfig


class TestConfigSchema(unittest.TestCase):
    def test_defaults(self):
        cfg = AppConfig()
        self.assertEqual(cfg.acquisition.source_type, "simulated")
        self.assertEqual(cfg.acquisition.expected_sfreq, 500.0)
        # 19 EEG + 1 EOG = 20 channels for the Enobio default.
        self.assertEqual(cfg.channels.n_channels, 20)
        self.assertIn("EOG", cfg.channels.eog_channels)
        self.assertNotIn("EOG", cfg.channels.eeg_channels)

    def test_roundtrip_dict(self):
        cfg = AppConfig()
        cfg.acquisition.expected_sfreq = 250.0
        cfg.ui.theme = "light"
        restored = AppConfig.from_dict(cfg.to_dict())
        self.assertEqual(restored.acquisition.expected_sfreq, 250.0)
        self.assertEqual(restored.ui.theme, "light")

    def test_partial_dict_uses_defaults(self):
        cfg = AppConfig.from_dict({"acquisition": {"expected_sfreq": 128.0}})
        self.assertEqual(cfg.acquisition.expected_sfreq, 128.0)
        self.assertEqual(cfg.acquisition.source_type, "simulated")  # default kept
        self.assertEqual(cfg.channels.n_channels, 20)               # default kept

    def test_unknown_keys_ignored(self):
        cfg = AppConfig.from_dict({"made_up": 1, "ui": {"nonsense": True}})
        self.assertIsInstance(cfg, AppConfig)
        self.assertEqual(cfg.ui.theme, "dark")

    def test_preprocessing_roundtrip(self):
        cfg = AppConfig()
        self.assertTrue(len(cfg.preprocessing.stages) >= 1)
        cfg.preprocessing.mode = "offline"
        cfg.preprocessing.stages = [
            {"type": "bandpass", "enabled": True, "params": {"low_hz": 1.0, "high_hz": 30.0}},
        ]
        restored = AppConfig.from_dict(cfg.to_dict())
        self.assertEqual(restored.preprocessing.mode, "offline")
        self.assertEqual(restored.preprocessing.stages[0]["type"], "bandpass")
        self.assertEqual(restored.preprocessing.stages[0]["params"]["high_hz"], 30.0)


class TestConfigManager(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = ConfigManager(Path(d))
            cfg = AppConfig(profile_name="exp1")
            cfg.simulation.alpha_amp_uv = 20.0
            mgr.save(cfg)
            self.assertIn("exp1", mgr.list_profiles())
            loaded = mgr.load("exp1")
            self.assertEqual(loaded.simulation.alpha_amp_uv, 20.0)

    def test_load_missing_creates_default(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = ConfigManager(Path(d))
            cfg = mgr.load("brand_new")
            self.assertEqual(cfg.profile_name, "brand_new")
            self.assertTrue(mgr.exists("brand_new"))

    def test_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = ConfigManager(Path(d))
            mgr.save(AppConfig(profile_name="base"))
            mgr.duplicate("base", "copy")
            self.assertIn("copy", mgr.list_profiles())


if __name__ == "__main__":
    unittest.main()
