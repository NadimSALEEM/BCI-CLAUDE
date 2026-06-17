"""Application entry point.

Wires logging, configuration, the acquisition engine and the main window,
then hands control to the Qt event loop. Importing this module has no side
effects; everything happens inside :func:`main`.
"""

from __future__ import annotations

import sys

from neurobci.acquisition.engine import AcquisitionEngine
from neurobci.config.manager import ConfigManager
from neurobci.core.app_state import AppState
from neurobci.core.events import EventBus
from neurobci.core.logging_setup import configure_logging


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    # Optional first positional arg selects a configuration profile.
    profile = "default"
    for a in argv[1:]:
        if not a.startswith("-"):
            profile = a
            break

    cfg = ConfigManager().load(profile)
    configure_logging(
        level=cfg.logging.level,
        to_file=cfg.logging.to_file,
        directory=cfg.logging.directory,
    )

    # Import Qt lazily so the headless engine/tests never require a display.
    from PyQt5 import QtWidgets
    from neurobci.ui.main_window import MainWindow
    from neurobci.ui.theme import DARK_QSS

    app = QtWidgets.QApplication(argv)
    if cfg.ui.theme == "dark":
        app.setStyleSheet(DARK_QSS)

    state = AppState()
    bus = EventBus()
    engine = AcquisitionEngine(cfg, app_state=state, event_bus=bus)

    window = MainWindow(engine, cfg)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
