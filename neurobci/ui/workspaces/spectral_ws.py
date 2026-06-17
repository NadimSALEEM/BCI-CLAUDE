"""Spectral & cognitive-state monitoring workspace.

Shows the PSD, a band-power scalp topomap, a temporal trace of an
exploratory index, and a fully transparent index panel (value, formula,
channels, raw band powers, validity) -- with the standing caveat that these
indices are proxies requiring validation. A baseline can be captured so maps
and indices are shown relative to it.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from neurobci.quality.metrics import QualityRating, compute_quality
from neurobci.spectral.analysis import SpectralAnalyzer
from neurobci.spectral.indices import CAVEAT
from neurobci.spectral.psd import BANDS
from neurobci.spectral.topo import channel_positions_2d, interpolate_topomap

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False


class SpectralWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._analyzer: SpectralAnalyzer | None = None
        self._info_key: list[str] | None = None
        self._pos = None
        self._found = None
        self._tick = 0
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        # --- controls ---------------------------------------------------- #
        ctrl = QtWidgets.QHBoxLayout()
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(1.0, 10.0)
        self.window_spin.setValue(self._ctl.config.spectral.window_s)
        self.window_spin.setSuffix(" s")
        self.band_combo = QtWidgets.QComboBox()
        self.band_combo.addItems(list(BANDS.keys()))
        self.band_combo.setCurrentText("alpha")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["absolute", "relative", "baseline"])
        self.index_combo = QtWidgets.QComboBox()
        self.index_combo.addItems(["engagement", "workload", "drowsiness"])
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("Preprocessed", True)
        self.source_combo.addItem("Raw", False)
        self.source_combo.setToolTip(
            "Run spectral analysis on the shared preprocessing output or on "
            "the raw stream.")
        self.baseline_btn = QtWidgets.QPushButton("Capture baseline")
        self.baseline_btn.clicked.connect(self._capture_baseline)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_baseline)

        for label, w in (("Window:", self.window_spin), ("Band:", self.band_combo),
                         ("Map:", self.mode_combo), ("Track:", self.index_combo),
                         ("Source:", self.source_combo)):
            ctrl.addWidget(QtWidgets.QLabel(label))
            ctrl.addWidget(w)
        ctrl.addStretch(1)
        ctrl.addWidget(self.baseline_btn)
        ctrl.addWidget(self.clear_btn)
        root.addLayout(ctrl)

        # --- main split -------------------------------------------------- #
        split = QtWidgets.QSplitter()

        # Left: PSD + temporal index trace.
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        self.psd_plot = pg.PlotWidget(title="Mean PSD (EEG)")
        self.psd_plot.setLogMode(x=False, y=True)
        self.psd_plot.setLabel("bottom", "Frequency", units="Hz")
        self.psd_plot.setLabel("left", "PSD (uV^2/Hz)")
        self.psd_curve = self.psd_plot.plot(pen=pg.mkPen("#3a9bd0", width=1))
        ll.addWidget(self.psd_plot)
        self.temporal_plot = pg.PlotWidget(title="Index over time")
        self.temporal_plot.setLabel("bottom", "Time", units="s")
        self.temporal_curve = self.temporal_plot.plot(pen=pg.mkPen("#ffd070", width=1))
        ll.addWidget(self.temporal_plot)
        split.addWidget(left)

        # Centre: topomap.
        if _HAVE_MPL:
            self.fig = Figure(figsize=(3.2, 3.2), facecolor="#14181c")
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.ax = self.fig.add_subplot(111)
            split.addWidget(self.canvas)
        else:
            self.canvas = None
            split.addWidget(QtWidgets.QLabel("matplotlib unavailable — no topomap"))

        # Right: indices panel.
        self.indices_text = QtWidgets.QTextEdit()
        self.indices_text.setReadOnly(True)
        split.addWidget(self.indices_text)
        split.setSizes([420, 360, 380])
        root.addWidget(split, stretch=1)

        caveat = QtWidgets.QLabel("⚠ " + CAVEAT)
        caveat.setWordWrap(True)
        caveat.setStyleSheet("color:#e0c040; font-style:italic; padding:3px;")
        root.addWidget(caveat)

    # ----- analyzer lifecycle ------------------------------------------- #

    def _ensure_analyzer(self, info) -> None:
        if self._analyzer is not None and info.channel_names == self._info_key:
            return
        self._analyzer = SpectralAnalyzer(info, self._ctl.config.spectral)
        self._info_key = list(info.channel_names)
        self._pos, self._found = channel_positions_2d(info.channel_names)

    def _capture_baseline(self) -> None:
        if self._analyzer is not None and getattr(self, "_last_report", None) is not None:
            self._analyzer.set_baseline(self._last_report)

    def _clear_baseline(self) -> None:
        if self._analyzer is not None:
            self._analyzer.clear_baseline()

    # ----- refresh ------------------------------------------------------- #

    def update_view(self) -> None:
        engine = self._ctl.engine
        info = engine.stream_info
        if info is None or engine.buffer is None:
            return
        # Throttle heavy spectral work to ~3 Hz.
        self._tick += 1
        if self._tick % max(1, int(self._ctl.config.ui.refresh_hz / 3)) != 0:
            return
        self._ensure_analyzer(info)

        if self.source_combo.currentData():
            data, _ = engine.latest_processed_seconds(self.window_spin.value())
        else:
            data, _ = engine.latest_seconds(self.window_spin.value())
        if data.shape[0] < int(info.sfreq):     # need ~1 s minimum
            return
        bad = [c.name for c in compute_quality(data, info).channels
               if c.rating == QualityRating.BAD]
        report = self._analyzer.analyze(data, bad_names=bad)
        self._last_report = report

        self._update_psd(report, info)
        self._update_topomap(report)
        self._update_temporal()
        self._update_indices(report)

    def _update_psd(self, report, info) -> None:
        good = [i for i in info.eeg_indices if i not in report.bad_channels]
        if not good:
            return
        mean_psd = report.psd[:, good].mean(axis=1)
        self.psd_curve.setData(report.freqs, np.clip(mean_psd, 1e-6, None))

    def _update_topomap(self, report) -> None:
        if self.canvas is None:
            return
        band = self.band_combo.currentText()
        mode = self.mode_combo.currentData() or self.mode_combo.currentText()
        vals = self._analyzer.topomap_values(report, band, mode)
        gx, gy, gz = interpolate_topomap(vals, self._pos, self._found, res=48)
        self.ax.clear()
        self.ax.set_facecolor("#14181c")
        if np.isfinite(gz).any():
            cmap = "RdBu_r" if mode == "baseline" else "viridis"
            self.ax.contourf(gx, gy, gz, levels=14, cmap=cmap)
        self.ax.add_artist(Circle((0, 0), 1.0, fill=False, color="#aaaaaa", lw=1.2))
        self.ax.plot([-0.12, 0, 0.12], [1.0, 1.13, 1.0], color="#aaaaaa", lw=1.2)
        m = self._found
        self.ax.scatter(self._pos[m, 0], self._pos[m, 1], c="#dddddd", s=10, zorder=3)
        self.ax.set_xlim(-1.25, 1.25)
        self.ax.set_ylim(-1.25, 1.3)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title(f"{band} ({mode})", color="#cdd4da", fontsize=9)
        self.canvas.draw_idle()

    def _update_temporal(self) -> None:
        key = self.index_combo.currentText()
        t, v = self._analyzer.history_series("indices", key)
        if t.size:
            self.temporal_curve.setData(t, np.nan_to_num(v, nan=0.0))

    def _update_indices(self, report) -> None:
        rows = []
        for ix in report.indices:
            val = f"{ix.value:.3f}" if np.isfinite(ix.value) else "n/a"
            color = "#36c24a" if ix.valid else "#ff6b6b"
            rel = ""
            if ix.relative is not None:
                rel = f" &nbsp; <i>(× {ix.relative:.2f} vs baseline)</i>"
            comps = ", ".join(f"{k}={v:.2f}" for k, v in ix.components.items())
            rows.append(
                f'<p style="margin:4px 0"><b style="color:{color}">{ix.name}'
                f"</b> = <b>{val}</b>{rel}<br>"
                f'<span style="color:#8fa3b0">def: {ix.definition}</span><br>'
                f'<span style="color:#8fa3b0">raw: {comps}</span><br>'
                f'<span style="color:#8fa3b0">channels: {", ".join(ix.channels) or "—"}'
                f"</span></p>")
        iaf = report.iaf
        head = f'<p><b>Individual alpha freq:</b> {iaf:.1f} Hz</p>' if np.isfinite(iaf) else ""
        self.indices_text.setHtml(head + "".join(rows))
