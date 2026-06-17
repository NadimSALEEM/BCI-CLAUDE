"""Preprocessing workspace.

Edit the ordered pipeline (enable/disable, reorder, edit parameters),
switch between causal (real-time) and offline (zero-phase) modes, and see
the effect live as a *before / after* comparison. Validation warnings and
detected artifacts are shown so questionable settings and contaminated
data are never hidden.
"""

from __future__ import annotations

import logging

from PyQt5 import QtCore, QtWidgets

from neurobci.preprocessing.artifacts import Severity, detect_artifacts
from neurobci.preprocessing.pipeline import MODE_CAUSAL, MODE_OFFLINE, Pipeline
from neurobci.ui.widgets.stacked_trace import StackedTracePlot

logger = logging.getLogger(__name__)


class StageParamDialog(QtWidgets.QDialog):
    """A parameter editor generated from a stage's ``describe()`` output."""

    def __init__(self, stage, parent=None) -> None:
        super().__init__(parent)
        self.stage = stage
        self.setWindowTitle(f"Edit '{stage.type_name}'")
        layout = QtWidgets.QFormLayout(self)
        desc = stage.describe()
        doc = QtWidgets.QLabel(desc["doc"])
        doc.setWordWrap(True)
        doc.setStyleSheet("color:#8fa3b0;")
        layout.addRow(doc)

        self._editors: dict[str, QtWidgets.QWidget] = {}
        for p in desc["params"]:
            w = self._make_editor(p)
            w.setToolTip(p["doc"])
            label = QtWidgets.QLabel(f"{p['name']}:")
            label.setToolTip(p["doc"])
            layout.addRow(label, w)
            self._editors[p["name"]] = w

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _make_editor(self, p) -> QtWidgets.QWidget:
        if p["kind"] == "int":
            w = QtWidgets.QSpinBox()
            w.setRange(int(p["min"] or 0), int(p["max"] or 1000))
            w.setValue(int(p["value"]))
            return w
        if p["kind"] == "choice":
            w = QtWidgets.QComboBox()
            w.addItems([str(c) for c in p["choices"]])
            w.setCurrentText(str(p["value"]))
            return w
        w = QtWidgets.QDoubleSpinBox()
        w.setDecimals(3)
        w.setRange(float(p["min"] if p["min"] is not None else 0.0),
                   float(p["max"] if p["max"] is not None else 1e6))
        w.setValue(float(p["value"]))
        return w

    def values(self) -> dict:
        out = {}
        for name, w in self._editors.items():
            if isinstance(w, QtWidgets.QSpinBox):
                out[name] = w.value()
            elif isinstance(w, QtWidgets.QComboBox):
                out[name] = w.currentText()
            else:
                out[name] = w.value()
        return out


class PreprocessingWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._pipe: Pipeline | None = None
        self._pipe_key = None
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        # --- top controls ------------------------------------------------ #
        ctrl = QtWidgets.QHBoxLayout()
        self.enabled_chk = QtWidgets.QCheckBox("Pipeline enabled")
        self.enabled_chk.setChecked(self._ctl.config.preprocessing.enabled)
        self.enabled_chk.stateChanged.connect(self._on_enabled)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Causal (real-time)", MODE_CAUSAL)
        self.mode_combo.addItem("Offline (zero-phase)", MODE_OFFLINE)
        idx = self.mode_combo.findData(self._ctl.config.preprocessing.mode)
        self.mode_combo.setCurrentIndex(max(idx, 0))
        self.mode_combo.currentIndexChanged.connect(self._on_mode)

        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(1.0, 20.0)
        self.window_spin.setValue(5.0)
        self.window_spin.setSuffix(" s")
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(5.0, 1000.0)
        self.scale_spin.setValue(self._ctl.config.ui.default_scale_uv)
        self.scale_spin.setSuffix(" uV")
        self.scale_spin.valueChanged.connect(self._on_scale)

        ctrl.addWidget(self.enabled_chk)
        ctrl.addSpacing(10)
        ctrl.addWidget(QtWidgets.QLabel("Mode:"))
        ctrl.addWidget(self.mode_combo)
        ctrl.addStretch(1)
        ctrl.addWidget(QtWidgets.QLabel("Window:"))
        ctrl.addWidget(self.window_spin)
        ctrl.addWidget(QtWidgets.QLabel("Scale:"))
        ctrl.addWidget(self.scale_spin)
        root.addLayout(ctrl)

        # --- main split: stage editor | before/after plots --------------- #
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        left = QtWidgets.QWidget()
        left_l = QtWidgets.QVBoxLayout(left)
        left_l.addWidget(QtWidgets.QLabel("Pipeline stages (order = top→bottom):"))
        self.stage_list = QtWidgets.QListWidget()
        self.stage_list.itemChanged.connect(self._on_item_changed)
        left_l.addWidget(self.stage_list, stretch=1)
        btns = QtWidgets.QHBoxLayout()
        for text, slot in (
            ("↑", lambda: self._move(-1)), ("↓", lambda: self._move(1)),
            ("Edit…", self._edit), ("Reset", self._reset),
        ):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            btns.addWidget(b)
        self.save_btn = QtWidgets.QPushButton("Save profile")
        self.save_btn.clicked.connect(self._save_profile)
        btns.addWidget(self.save_btn)
        left_l.addLayout(btns)
        self.warnings = QtWidgets.QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet("color:#e0c040;")
        left_l.addWidget(self.warnings)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        right_l = QtWidgets.QVBoxLayout(right)
        self.before_plot = StackedTracePlot("Raw (before)", self.scale_spin.value())
        self.after_plot = StackedTracePlot("Processed (after)", self.scale_spin.value())
        right_l.addWidget(self.before_plot, stretch=1)
        right_l.addWidget(self.after_plot, stretch=1)
        split.addWidget(right)
        split.setSizes([320, 760])
        root.addWidget(split, stretch=1)

        # --- artifacts --------------------------------------------------- #
        root.addWidget(QtWidgets.QLabel("Detected artifacts (raw, last window):"))
        self.artifacts = QtWidgets.QListWidget()
        self.artifacts.setMaximumHeight(120)
        root.addWidget(self.artifacts)

    # ----- pipeline lifecycle ------------------------------------------- #

    def _ensure_pipeline(self, info) -> bool:
        key = (info.sfreq, tuple(info.channel_kinds))
        if self._pipe is not None and key == self._pipe_key:
            return False
        self._pipe = Pipeline.from_config(
            self._ctl.config.preprocessing, info.sfreq, info.channel_kinds
        )
        self._pipe.enabled = self.enabled_chk.isChecked()
        self._pipe.mode = self.mode_combo.currentData()
        self._pipe_key = key
        self._populate_stage_list()
        return True

    def _populate_stage_list(self) -> None:
        self.stage_list.blockSignals(True)
        self.stage_list.clear()
        if self._pipe:
            for st in self._pipe.stages:
                summary = self._stage_summary(st)
                item = QtWidgets.QListWidgetItem(summary)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.Checked if st.enabled else QtCore.Qt.Unchecked
                )
                if not st.realtime_safe:
                    item.setForeground(QtCore.Qt.gray)
                self.stage_list.addItem(item)
        self.stage_list.blockSignals(False)
        self._refresh_warnings()

    def _stage_summary(self, st) -> str:
        params = ", ".join(f"{k}={v}" for k, v in st.params.items())
        rt = "" if st.realtime_safe else "  [offline-only]"
        return f"{st.type_name}{rt}" + (f"  ({params})" if params else "")

    # ----- edits --------------------------------------------------------- #

    def _on_enabled(self) -> None:
        en = self.enabled_chk.isChecked()
        self._ctl.config.preprocessing.enabled = en
        if self._pipe:
            self._pipe.enabled = en

    def _on_mode(self) -> None:
        mode = self.mode_combo.currentData()
        self._ctl.config.preprocessing.mode = mode
        if self._pipe:
            self._pipe.mode = mode

    def _on_scale(self, v) -> None:
        self.before_plot.set_scale(v)
        self.after_plot.set_scale(v)

    def _on_item_changed(self, item) -> None:
        if self._pipe is None:
            return
        row = self.stage_list.row(item)
        self._pipe.set_enabled(row, item.checkState() == QtCore.Qt.Checked)
        self._sync_config()
        self._refresh_warnings()

    def _move(self, delta: int) -> None:
        if self._pipe is None:
            return
        row = self.stage_list.currentRow()
        if row < 0:
            return
        self._pipe.move(row, delta)
        self._sync_config()
        self._populate_stage_list()
        self.stage_list.setCurrentRow(min(max(row + delta, 0), self.stage_list.count() - 1))

    def _edit(self) -> None:
        if self._pipe is None:
            return
        row = self.stage_list.currentRow()
        if row < 0:
            return
        stage = self._pipe.stages[row]
        dlg = StageParamDialog(stage, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            stage.params.update(dlg.values())
            stage.prepare(self._pipe.sfreq, self._pipe.ch_kinds)  # redesign filters
            self._sync_config()
            self._populate_stage_list()
            self.stage_list.setCurrentRow(row)

    def _reset(self) -> None:
        from neurobci.config.schema import default_pipeline
        self._ctl.config.preprocessing.stages = default_pipeline()
        self._pipe = None  # force rebuild on next refresh
        self._pipe_key = None

    def _save_profile(self) -> None:
        from neurobci.config.manager import ConfigManager
        try:
            self._sync_config()
            path = ConfigManager().save(self._ctl.config)
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Preprocessing profile saved to:\n{path}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to save profile.")
            QtWidgets.QMessageBox.critical(self, "Save error", str(exc))

    def _sync_config(self) -> None:
        if self._pipe:
            self._ctl.config.preprocessing.stages = self._pipe.to_config_stages()

    def _refresh_warnings(self) -> None:
        if self._pipe is None:
            self.warnings.setText("")
            return
        msgs = list(self._pipe.validate())
        if self._pipe.skipped_in_causal and self.mode_combo.currentData() == MODE_CAUSAL:
            msgs.append(
                "Skipped in causal mode (not real-time safe): "
                + ", ".join(self._pipe.skipped_in_causal)
            )
        self.warnings.setText("⚠ " + "\n⚠ ".join(msgs) if msgs else "")

    # ----- refresh ------------------------------------------------------- #

    def update_view(self) -> None:
        engine = self._ctl.engine
        info = engine.stream_info
        if info is None or engine.buffer is None or self._ctl.config is None:
            return
        self._ensure_pipeline(info)
        self.before_plot.set_channels(info.channel_names, info.channel_kinds)
        self.after_plot.set_channels(info.channel_names, info.channel_kinds)

        raw, _ = engine.latest_seconds(self.window_spin.value())
        if raw.shape[0] < 2:
            return
        self.before_plot.update_data(raw, info.sfreq)
        processed = self._pipe.apply_window(raw) if self._pipe else raw
        self.after_plot.update_data(processed, info.sfreq)

        self._update_artifacts(raw, info)

    def _update_artifacts(self, raw, info) -> None:
        report = detect_artifacts(raw, info)
        self.artifacts.clear()
        sev_color = {
            Severity.REJECT: "❌", Severity.WARN: "⚠", Severity.INFO: "ℹ",
        }
        for e in report.events:
            tag = sev_color.get(e.severity, "")
            where = e.channel or "global"
            self.artifacts.addItem(f"{tag} [{where}] {e.message}")
        if self.artifacts.count() == 0:
            self.artifacts.addItem("No artifacts detected.")
