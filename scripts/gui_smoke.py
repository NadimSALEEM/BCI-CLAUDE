"""Headless GUI smoke test.

Builds the real main window with a simulated source on Qt's *offscreen*
platform, lets data flow, exercises every workspace's refresh path, and
performs a full recording round-trip (record -> stop -> load back) --
without requiring a visible display or modal dialogs.

    QT_QPA_PLATFORM=offscreen python scripts/gui_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5 import QtWidgets                                       # noqa: E402

from neurobci.acquisition.engine import AcquisitionEngine        # noqa: E402
from neurobci.config.schema import AppConfig                     # noqa: E402
from neurobci.core.logging_setup import configure_logging        # noqa: E402
from neurobci.recording.exporter import load_session             # noqa: E402
from neurobci.ui.main_window import MainWindow                   # noqa: E402


def _pump(app, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)


def main() -> int:
    configure_logging(level="WARNING", to_file=False)
    tmp = tempfile.mkdtemp(prefix="neurobci_smoke_")
    cfg = AppConfig()
    cfg.recording.directory = tmp
    cfg.simulation.flat_channels = ["Cz"]
    cfg.simulation.noisy_channels = ["F8"]

    app = QtWidgets.QApplication(sys.argv)
    engine = AcquisitionEngine(cfg)
    win = MainWindow(engine, cfg)
    win.show()

    win.start_acquisition("simulated")
    assert engine.running, "engine did not start"

    # Cycle through every tab so each update_view() runs.
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        _pump(app, 0.25)
        win._on_refresh()

    # Recording round-trip (use engine API directly: no modal dialogs).
    rec = engine.start_recording(participant_id="smoke", notes="gui smoke")
    _pump(app, 0.6)
    engine.push_marker("target")
    _pump(app, 0.3)
    engine.push_marker("nontarget")
    _pump(app, 0.3)
    stats = engine.stop_recording()

    loaded = load_session(rec.path)

    # Exercise the preprocessing tab explicitly (before/after + pipeline).
    win.tabs.setCurrentWidget(win.preprocessing_ws)
    _pump(app, 0.3)
    win._on_refresh()
    before_ch = win.preprocessing_ws.before_plot.n_channels
    after_ch = win.preprocessing_ws.after_plot.n_channels
    pipe_ok = win.preprocessing_ws._pipe is not None

    # Exercise the spectral / cognitive-state tab (throttled -> call repeatedly).
    win.tabs.setCurrentWidget(win.spectral_ws)
    for _ in range(15):
        win.spectral_ws.update_view()
        app.processEvents()
    srep = getattr(win.spectral_ws, "_last_report", None)
    spectral_ok = srep is not None and len(srep.indices) == 3
    win.spectral_ws._capture_baseline()
    baseline_ok = win.spectral_ws._analyzer.has_baseline

    # Exercise the calibration tab: run a small simulated P300 calibration.
    cfg.recording.directory = tmp  # already set; models go under ./models
    win.tabs.setCurrentWidget(win.calibration_ws)
    p_idx = win.calibration_ws.paradigm_combo.findData("p300")
    win.calibration_ws.paradigm_combo.setCurrentIndex(p_idx)
    win.calibration_ws.n_trials.setValue(200)
    if "riemann_lr" in win.calibration_ws._model_checks:
        win.calibration_ws._model_checks["riemann_lr"].setChecked(False)
    win.calibration_ws._run()
    app.processEvents()
    calib_model_ok = win.model is not None and win.model.trained
    calib_usable = win.calibration_result.is_usable if win.calibration_result else False

    # Also verify a non-P300 paradigm calibrates through the GUI dispatch.
    mi_idx = win.calibration_ws.paradigm_combo.findData("motor_imagery")
    win.calibration_ws.paradigm_combo.setCurrentIndex(mi_idx)
    win.calibration_ws.n_trials.setValue(80)
    win.calibration_ws._run()
    app.processEvents()
    mi_ok = win.model is not None and win.model.paradigm == "motor_imagery"
    # Restore a P300 model for the control demo below.
    win.calibration_ws.paradigm_combo.setCurrentIndex(p_idx)
    win.calibration_ws.n_trials.setValue(200)
    if "riemann_lr" in win.calibration_ws._model_checks:
        win.calibration_ws._model_checks["riemann_lr"].setChecked(False)
    win.calibration_ws._run()
    app.processEvents()

    # Exercise the control/BCI demo: run selections through the full stack.
    win.tabs.setCurrentWidget(win.control_ws)
    win.control_ws.update_view()
    for _ in range(4):
        win.control_ws._run_one(intended=2)
        app.processEvents()
    router = win.control_ws._router
    ctrl_executed = any(r.executed for r in router.history)
    board_moved = win.control_ws._board.last_item is not None
    # Emergency stop must block a decided command (deterministic router check).
    from neurobci.control.types import Command
    blk = router.submit(Command(action="select:1", item=1, confidence=0.9),
                        emergency_stop=True, connected=True, quality_ok=True)
    estop_blocked = (not blk.executed) and "emergency" in (blk.rejected_reason or "")

    total = engine.buffer.total_written if engine.buffer else 0
    raw_curves = len(win.raw_ws._curves)
    quality_rows = win.quality_ws.table.rowCount()
    channel_rows = win.channels_ws.table.rowCount()

    win._toggle_estop()
    estop_on = engine.state.snapshot().emergency_stop

    win.close()
    app.processEvents()

    print(f"samples written : {total}")
    print(f"raw curves      : {raw_curves}")
    print(f"quality rows    : {quality_rows}")
    print(f"channel rows    : {channel_rows}")
    print(f"preproc before  : {before_ch} ch")
    print(f"preproc after   : {after_ch} ch")
    print(f"pipeline built  : {pipe_ok}")
    print(f"spectral ok     : {spectral_ok}")
    print(f"baseline ok     : {baseline_ok}")
    print(f"calib model ok  : {calib_model_ok}")
    print(f"calib usable    : {calib_usable}")
    print(f"MI dispatch ok  : {mi_ok}")
    print(f"ctrl executed   : {ctrl_executed}")
    print(f"board moved     : {board_moved}")
    print(f"estop blocked   : {estop_blocked}")
    print(f"recorded samples: {stats.n_samples}")
    print(f"loaded samples  : {loaded.n_samples}")
    print(f"loaded markers  : {len(loaded.markers)}")
    print(f"e-stop latched  : {estop_on}")

    ok = (
        total > 0
        and raw_curves == 20
        and quality_rows == 20
        and channel_rows == 20
        and before_ch == 20
        and after_ch == 20
        and pipe_ok
        and spectral_ok
        and baseline_ok
        and calib_model_ok
        and calib_usable
        and mi_ok
        and ctrl_executed
        and board_moved
        and estop_blocked
        and stats.n_samples > 0
        and loaded.n_samples == stats.n_samples
        and len(loaded.markers) == 2
        and estop_on
    )
    print("GUI SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
