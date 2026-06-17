"""Reusable stacked multichannel trace plot (pyqtgraph).

Used by the Raw EEG workspace and by the Preprocessing workspace (twice:
before / after). Channels are drawn as vertically-offset traces; EOG is
drawn in a distinct colour and never confused with scalp EEG.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from neurobci.core.stream_info import KIND_EOG

pg.setConfigOptions(antialias=False, background="#101418", foreground="#b0b8c0")

_EEG_PEN = pg.mkPen("#cfe6ff", width=1)
_EOG_PEN = pg.mkPen("#ffd070", width=1)


class StackedTracePlot(QtWidgets.QWidget):
    def __init__(self, title: str | None = None, scale_uv: float = 75.0) -> None:
        super().__init__()
        self._curves: list[pg.PlotDataItem] = []
        self._names: list[str] = []
        self._kinds: list[str] = []
        self._scale = scale_uv

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if title:
            lbl = QtWidgets.QLabel(title)
            lbl.setStyleSheet("font-weight:600; padding:2px;")
            layout.addWidget(lbl)
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.getPlotItem().setDownsampling(auto=True, mode="peak")
        self.plot.getPlotItem().setClipToView(True)
        layout.addWidget(self.plot, stretch=1)

    # ----- configuration ------------------------------------------------- #

    def set_channels(self, names: list[str], kinds: list[str]) -> None:
        if names == self._names and kinds == self._kinds:
            return
        self._names = list(names)
        self._kinds = list(kinds)
        self.plot.clear()
        self._curves = []
        for kind in kinds:
            pen = _EOG_PEN if kind == KIND_EOG else _EEG_PEN
            self._curves.append(self.plot.plot(pen=pen))
        self._rebuild_axis()

    def set_scale(self, scale_uv: float) -> None:
        self._scale = scale_uv
        self._rebuild_axis()

    @property
    def n_channels(self) -> int:
        return len(self._curves)

    def _spacing(self) -> float:
        return 2.0 * self._scale

    def _rebuild_axis(self) -> None:
        n = len(self._names)
        if n == 0:
            return
        spacing = self._spacing()
        ticks = [((n - 1 - i) * spacing, self._names[i]) for i in range(n)]
        self.plot.getAxis("left").setTicks([ticks])
        self.plot.setYRange(-spacing, (n - 1) * spacing + spacing, padding=0.02)

    # ----- data ---------------------------------------------------------- #

    def update_data(self, data: np.ndarray, sfreq: float) -> None:
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != len(self._curves):
            return
        n_samples = data.shape[0]
        t = (np.arange(n_samples) - (n_samples - 1)) / sfreq
        spacing = self._spacing()
        n = len(self._curves)
        for ci in range(n):
            baseline = (n - 1 - ci) * spacing
            self._curves[ci].setData(t, data[:, ci] + baseline)

    def clear_data(self) -> None:
        for c in self._curves:
            c.clear()
