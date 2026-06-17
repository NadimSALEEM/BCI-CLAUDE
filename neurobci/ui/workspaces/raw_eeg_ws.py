"""Raw EEG visualisation workspace.

Stacked multichannel traces drawn with pyqtgraph (via the reusable
:class:`StackedTracePlot`). Rendering is fully decoupled from acquisition:
this widget only *pulls* the most recent window from the ring buffer each
refresh, so a slow redraw can never affect data collection.
"""

from __future__ import annotations

from PyQt5 import QtWidgets

from neurobci.ui.widgets.stacked_trace import StackedTracePlot


class RawEEGWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        ctrl = QtWidgets.QHBoxLayout()
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(1.0, 30.0)
        self.window_spin.setValue(self._ctl.config.ui.raw_window_s)
        self.window_spin.setSuffix(" s")

        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(5.0, 1000.0)
        self.scale_spin.setValue(self._ctl.config.ui.default_scale_uv)
        self.scale_spin.setSuffix(" uV")
        self.scale_spin.valueChanged.connect(
            lambda v: self.plot.set_scale(v)
        )

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("Raw", False)
        self.source_combo.addItem("Preprocessed", True)
        self.source_combo.setToolTip(
            "Show the raw stream or the shared preprocessing output.")

        self.pause_chk = QtWidgets.QCheckBox("Pause")

        ctrl.addWidget(QtWidgets.QLabel("Window:"))
        ctrl.addWidget(self.window_spin)
        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel("Scale (±):"))
        ctrl.addWidget(self.scale_spin)
        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel("Show:"))
        ctrl.addWidget(self.source_combo)
        ctrl.addSpacing(12)
        ctrl.addWidget(self.pause_chk)
        ctrl.addStretch(1)
        root.addLayout(ctrl)

        self.plot = StackedTracePlot(scale_uv=self.scale_spin.value())
        root.addWidget(self.plot, stretch=1)

    # Kept for backwards-compatible smoke checks.
    @property
    def _curves(self):
        return self.plot._curves

    def update_view(self) -> None:
        if self.pause_chk.isChecked():
            return
        engine = self._ctl.engine
        info = engine.stream_info
        if info is None or engine.buffer is None:
            return
        self.plot.set_channels(info.channel_names, info.channel_kinds)
        if self.source_combo.currentData():
            data, _ = engine.latest_processed_seconds(self.window_spin.value())
        else:
            data, _ = engine.latest_seconds(self.window_spin.value())
        self.plot.update_data(data, info.sfreq)
