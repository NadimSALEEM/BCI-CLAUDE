"""Codifies the GUI smoke: the main window constructs offscreen with all tabs.

Skips cleanly if Qt / the offscreen platform is unavailable.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestAppConstruction(unittest.TestCase):
    def test_main_window_builds_with_all_tabs(self):
        try:
            from PyQt5 import QtWidgets
            from neurobci.acquisition.engine import AcquisitionEngine
            from neurobci.config.schema import AppConfig
            from neurobci.ui.main_window import MainWindow
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Qt unavailable: {exc}")

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        engine = AcquisitionEngine(AppConfig())
        win = MainWindow(engine, AppConfig())
        try:
            # All workspaces present and each exposes update_view
            # (11 core tabs + Statistics, Machine Learning, Analysis History).
            self.assertEqual(win.tabs.count(), 14)
            for i in range(win.tabs.count()):
                w = win.tabs.widget(i)
                self.assertTrue(hasattr(w, "update_view"))
        finally:
            win.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
