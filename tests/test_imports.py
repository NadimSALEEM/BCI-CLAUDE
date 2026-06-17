"""Package imports, paradigm registry and full-config round-trip."""

import importlib
import unittest

from neurobci.config.schema import AppConfig
from neurobci.paradigms.base import available_paradigms

_SUBPACKAGES = [
    "neurobci",
    "neurobci.config",
    "neurobci.core",
    "neurobci.acquisition",
    "neurobci.preprocessing",
    "neurobci.quality",
    "neurobci.paradigms",
    "neurobci.bci",
    "neurobci.control",
    "neurobci.spectral",
    "neurobci.recording",
]


class TestImports(unittest.TestCase):
    def test_all_subpackages_import(self):
        for mod in _SUBPACKAGES:
            importlib.import_module(mod)

    def test_ui_entry_imports_without_display(self):
        # Lazy Qt import means importing the entry module needs no display.
        importlib.import_module("neurobci.ui.app")
        importlib.import_module("neurobci.__main__")

    def test_paradigm_registry(self):
        self.assertEqual(
            available_paradigms(), ["errp", "motor_imagery", "p300", "ssvep"])

    def test_full_config_roundtrip(self):
        cfg = AppConfig()
        cfg.control.safety.external_control_enabled = True
        cfg.spectral.window_s = 6.0
        cfg.acquisition.replay_speed = 2.0
        restored = AppConfig.from_dict(cfg.to_dict())
        self.assertTrue(restored.control.safety.external_control_enabled)
        self.assertEqual(restored.spectral.window_s, 6.0)
        self.assertEqual(restored.acquisition.replay_speed, 2.0)
        # All major sections present.
        for section in ("channels", "acquisition", "simulation", "preprocessing",
                        "control", "spectral", "recording", "ui", "logging"):
            self.assertTrue(hasattr(restored, section))


if __name__ == "__main__":
    unittest.main()
