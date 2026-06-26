"""ERP / Epoch-Average analysis workspace (offline).

Analyses the session currently loaded in the **Replay** tab: it reads the
event markers that session carries, lets the analyst group/combine markers
into conditions, and computes trial-averaged evoked responses -- without
changing how the platform preprocesses data. The configured preprocessing
pipeline is reused verbatim (run in offline/zero-phase mode), or the raw
signal is epoched as-is.

Markers can be *combined* into new derived events (e.g. pool ``stim/left`` and
``stim/right`` into ``stimulus``) before epoching. Outputs: per-channel ERP
overlays (with SEM), a butterfly view, global field power, and a scalp topomap
(adaptive to the present electrodes) at a chosen latency. Averages can be
exported to a portable ``.npz``. This tab never touches acquisition or the
live preprocessing config; the session comes from the Replay tab.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from neurobci.bci.erp_analysis import (
    Condition,
    combine_markers,
    compute_erp,
    global_field_power,
    latency_index,
    marker_label_counts,
    select_channels,
)
from neurobci.core.stream_info import KIND_EEG, KIND_EOG
from neurobci.paradigms.base import EpochWindow
from neurobci.preprocessing.pipeline import Pipeline
from neurobci.recording.external import load_session_any
from neurobci.spectral.topo import channel_positions_2d, interpolate_topomap

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False

# A fixed, colour-blind-friendly palette cycled across conditions.
_PALETTE = ["#4f9fe0", "#e0823a", "#36c24a", "#d8508a",
            "#b07cf0", "#e0c040", "#46c7c0", "#d85050"]

_MARKER_COLS = ["Use", "Label", "Count", "Condition"]


class AnalysisWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._session = None
        self._markers: list[dict] = []          # session markers + derived
        self._result = None
        self._cond_colors: dict[str, str] = {}
        self._loaded_path: str | None = None
        self._montage_note = ""
        self._build()

    # ----- construction -------------------------------------------------- #

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        # --- session source (Replay tab only) --------------------------- #
        sel = QtWidgets.QHBoxLayout()
        self.use_replay_btn = QtWidgets.QPushButton("Load from Replay tab")
        self.use_replay_btn.clicked.connect(self._use_replay)
        self.path_label = QtWidgets.QLabel(
            "No session. Load one in the Replay tab, then click ‘Load from "
            "Replay tab’.")
        self.path_label.setWordWrap(True)
        sel.addWidget(self.use_replay_btn)
        sel.addWidget(self.path_label, stretch=1)
        root.addLayout(sel)

        split = QtWidgets.QSplitter()

        # --- left: markers + parameters --------------------------------- #
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)

        ll.addWidget(QtWidgets.QLabel(
            "Markers → conditions (tick to include; edit Condition to group):"))

        # --- search + bulk (de)selection -------------------------------- #
        tools = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("search event…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_markers)
        self.select_all_btn = QtWidgets.QPushButton("Select all")
        self.select_all_btn.setToolTip("Tick every visible marker.")
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self.unselect_all_btn = QtWidgets.QPushButton("Unselect all")
        self.unselect_all_btn.setToolTip("Untick every visible marker.")
        self.unselect_all_btn.clicked.connect(lambda: self._set_all_checked(False))
        tools.addWidget(self.search, stretch=1)
        tools.addWidget(self.select_all_btn)
        tools.addWidget(self.unselect_all_btn)
        ll.addLayout(tools)

        self.marker_table = QtWidgets.QTableWidget(0, len(_MARKER_COLS))
        self.marker_table.setHorizontalHeaderLabels(_MARKER_COLS)
        self.marker_table.verticalHeader().setVisible(False)
        self.marker_table.horizontalHeader().setStretchLastSection(True)
        ll.addWidget(self.marker_table, stretch=1)

        # --- combine markers into a new derived event -------------------- #
        combine = QtWidgets.QHBoxLayout()
        self.combine_name = QtWidgets.QLineEdit()
        self.combine_name.setPlaceholderText("new marker name")
        self.combine_btn = QtWidgets.QPushButton("Combine ticked →")
        self.combine_btn.setToolTip(
            "Pool the onsets of the ticked markers into a new derived marker "
            "you can then epoch like any other.")
        self.combine_btn.clicked.connect(self._combine_markers)
        self.reset_markers_btn = QtWidgets.QPushButton("Reset")
        self.reset_markers_btn.setToolTip("Drop all derived markers.")
        self.reset_markers_btn.clicked.connect(self._reset_markers)
        combine.addWidget(self.combine_name, stretch=1)
        combine.addWidget(self.combine_btn)
        combine.addWidget(self.reset_markers_btn)
        ll.addLayout(combine)

        params = QtWidgets.QGroupBox("Epoch window")
        form = QtWidgets.QFormLayout(params)
        self.tmin = self._dspin(-2.0, 0.0, -0.1, " s")
        self.tmax = self._dspin(0.05, 5.0, 0.6, " s")
        self.baseline_chk = QtWidgets.QCheckBox("Baseline correct")
        self.baseline_chk.setChecked(True)
        self.bmin = self._dspin(-2.0, 0.0, -0.1, " s")
        self.bmax = self._dspin(-2.0, 2.0, 0.0, " s")
        self.reject = self._dspin(10.0, 1000.0, 150.0, " uV p-p")
        self.reject_chk = QtWidgets.QCheckBox("Reject by amplitude")
        self.reject_chk.setChecked(True)
        self.preproc_combo = QtWidgets.QComboBox()
        self.preproc_combo.addItem("Configured pipeline (offline)", True)
        self.preproc_combo.addItem("Raw (no preprocessing)", False)
        self.preproc_combo.setToolTip(
            "Reuse the platform's configured preprocessing (run zero-phase) "
            "or epoch the raw signal. This never changes the live pipeline.")
        form.addRow("tmin:", self.tmin)
        form.addRow("tmax:", self.tmax)
        form.addRow(self.baseline_chk)
        form.addRow("baseline start:", self.bmin)
        form.addRow("baseline end:", self.bmax)
        form.addRow(self.reject_chk)
        form.addRow("reject:", self.reject)
        form.addRow("Preprocessing:", self.preproc_combo)
        ll.addWidget(params)

        actions = QtWidgets.QHBoxLayout()
        self.compute_btn = QtWidgets.QPushButton("Compute averages")
        self.compute_btn.clicked.connect(self._compute)
        self.compute_btn.setEnabled(False)
        self.export_btn = QtWidgets.QPushButton("Export averages (.npz)")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        actions.addWidget(self.compute_btn)
        actions.addWidget(self.export_btn)
        ll.addLayout(actions)

        self.summary = QtWidgets.QLabel("Load a session to begin.")
        self.summary.setWordWrap(True)
        ll.addWidget(self.summary)
        split.addWidget(left)

        # --- right: plots ----------------------------------------------- #
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)

        ctrl = QtWidgets.QHBoxLayout()
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.currentIndexChanged.connect(self._redraw_traces)
        self.butterfly_chk = QtWidgets.QCheckBox("Butterfly (all channels)")
        self.butterfly_chk.stateChanged.connect(self._redraw_traces)
        self.topo_cond = QtWidgets.QComboBox()
        self.topo_cond.currentIndexChanged.connect(self._redraw_topomap)
        self.latency = self._dspin(-2.0, 5.0, 0.3, " s")
        self.latency.valueChanged.connect(self._redraw_topomap)
        ctrl.addWidget(QtWidgets.QLabel("Channel:"))
        ctrl.addWidget(self.channel_combo)
        ctrl.addWidget(self.butterfly_chk)
        ctrl.addStretch(1)
        ctrl.addWidget(QtWidgets.QLabel("Topo condition:"))
        ctrl.addWidget(self.topo_cond)
        ctrl.addWidget(QtWidgets.QLabel("at"))
        ctrl.addWidget(self.latency)
        rl.addLayout(ctrl)

        self.erp_plot = pg.PlotWidget(title="Evoked response")
        self.erp_plot.setLabel("bottom", "Time", units="s")
        self.erp_plot.setLabel("left", "Amplitude", units="uV")
        self.erp_plot.addLegend()
        self.erp_plot.showGrid(x=True, y=True, alpha=0.2)
        rl.addWidget(self.erp_plot, stretch=2)

        bottom = QtWidgets.QSplitter()
        self.gfp_plot = pg.PlotWidget(title="Global field power")
        self.gfp_plot.setLabel("bottom", "Time", units="s")
        self.gfp_plot.setLabel("left", "GFP", units="uV")
        self.gfp_plot.addLegend()
        bottom.addWidget(self.gfp_plot)
        if _HAVE_MPL:
            self.fig = Figure(figsize=(3.0, 3.0), facecolor="#14181c")
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.ax = self.fig.add_subplot(111)
            bottom.addWidget(self.canvas)
        else:
            self.canvas = None
            bottom.addWidget(QtWidgets.QLabel("matplotlib unavailable — no topomap"))
        bottom.setSizes([520, 360])
        rl.addWidget(bottom, stretch=2)

        split.addWidget(right)
        split.setSizes([430, 760])
        root.addWidget(split, stretch=1)

    def _dspin(self, lo, hi, val, suffix):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(3)
        s.setSingleStep(0.05)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    # ----- loading (Replay tab only) ------------------------------------ #

    def _use_replay(self) -> None:
        path = getattr(self._ctl.replay_ws, "_session_path", None)
        if not path:
            QtWidgets.QMessageBox.information(
                self, "No replay session",
                "Load a session in the Replay tab first, then come back here.")
            return
        try:
            session = load_session_any(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load session for analysis.")
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))
            return
        session, self._montage_note = self._adopt_montage(session)
        self._session = session
        self._markers = [dict(m) for m in session.markers]   # working copy
        self._loaded_path = path
        self._result = None
        self.path_label.setText(path)
        self._populate_markers()
        self.export_btn.setEnabled(False)
        self._refresh_summary()

    def _adopt_montage(self, session):
        """Relabel (and, if channels were deleted, drop) the file's channels to
        match the montage you curated. Returns ``(session, note)``.

        An XDF/FIF often carries generic names (``Ch1``…); the real montage
        lives in the Channels tab. In order of preference, as long as widths
        line up: replicate the **live Replay stream's selection** (drops +
        renames the Channels tab made), then adopt the live stream's names, then
        the configured montage. This is what keeps the channel picker, the
        topomap and the averages on the same channels as everything else.
        """
        n = len(session.channel_names)
        engine = self._ctl.engine
        info = engine.stream_info
        native = engine.native_stream_info
        keep = engine.keep_indices

        # Channels were deleted live: replicate the exact column drop + labels.
        if (info is not None and keep is not None and native is not None
                and native.n_channels == n):
            session = select_channels(session, keep, info.channel_names,
                                      info.channel_kinds)
            return session, (f"montage from the live Replay stream — "
                             f"{len(keep)} of {n} channels kept (Channels tab)")

        # No deletion, just renames/kinds from the live stream.
        if info is not None and info.n_channels == n:
            session.meta["channel_names"] = list(info.channel_names)
            session.meta["channel_kinds"] = list(info.channel_kinds)
            return session, "names adopted from the live Replay stream (Channels tab)"

        cfg = self._ctl.config.channels
        if cfg.n_channels == n:
            names = list(cfg.all_channels)
            eog = set(cfg.eog_channels)
            session.meta["channel_names"] = names
            session.meta["channel_kinds"] = [
                KIND_EOG if x in eog else KIND_EEG for x in names]
            return session, "names adopted from the configured montage (Channels tab)"

        return session, (
            f"using the file's own channel names — the Channels-tab montage "
            f"has a different channel count ({cfg.n_channels} vs {n}), so it "
            f"was not applied")

    def _refresh_summary(self) -> None:
        session = self._session
        if session is None:
            return
        counts = marker_label_counts(self._markers)
        self.compute_btn.setEnabled(bool(self._markers))
        self.summary.setText(
            f"{session.n_samples} samples @ {session.sfreq:.0f} Hz "
            f"({session.n_samples / session.sfreq:.1f} s), "
            f"{len(session.channel_names)} channels, "
            f"{len(self._markers)} markers across {len(counts)} labels — "
            f"{self._montage_note}. Tick labels (or combine them), then Compute.")

    def _populate_markers(self) -> None:
        counts = marker_label_counts(self._markers)
        derived = {m["label"] for m in self._markers if m.get("derived")}
        self.marker_table.setRowCount(len(counts))
        for row, (label, count) in enumerate(counts.items()):
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(QtCore.Qt.Checked)
            self.marker_table.setItem(row, 0, chk)
            lab_item = QtWidgets.QTableWidgetItem(
                f"{label} ⊕" if label in derived else label)
            lab_item.setFlags(QtCore.Qt.ItemIsEnabled)
            lab_item.setData(QtCore.Qt.UserRole, label)        # true label
            if label in derived:
                lab_item.setForeground(QtGui.QColor("#8fd0a0"))
            self.marker_table.setItem(row, 1, lab_item)
            cnt_item = QtWidgets.QTableWidgetItem(str(count))
            cnt_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.marker_table.setItem(row, 2, cnt_item)
            # Default: each label is its own condition (editable to group).
            self.marker_table.setItem(row, 3, QtWidgets.QTableWidgetItem(label))
        self.marker_table.resizeColumnsToContents()
        self.marker_table.horizontalHeader().setStretchLastSection(True)
        self._filter_markers(self.search.text())     # keep any active filter

    def _ticked_labels(self) -> list[str]:
        out = []
        for row in range(self.marker_table.rowCount()):
            if self.marker_table.item(row, 0).checkState() == QtCore.Qt.Checked:
                out.append(self.marker_table.item(row, 1).data(QtCore.Qt.UserRole))
        return out

    # ----- search / bulk selection -------------------------------------- #

    def _filter_markers(self, text: str) -> None:
        """Show only rows whose label contains ``text`` (case-insensitive)."""
        needle = (text or "").strip().lower()
        for row in range(self.marker_table.rowCount()):
            label = (self.marker_table.item(row, 1).data(QtCore.Qt.UserRole) or "")
            self.marker_table.setRowHidden(row, needle not in label.lower())

    def _set_all_checked(self, checked: bool) -> None:
        """Tick/untick every *visible* marker (respects the search filter)."""
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        for row in range(self.marker_table.rowCount()):
            if not self.marker_table.isRowHidden(row):
                self.marker_table.item(row, 0).setCheckState(state)

    # ----- combine markers ---------------------------------------------- #

    def _combine_markers(self) -> None:
        if self._session is None:
            return
        labels = self._ticked_labels()
        name = self.combine_name.text().strip()
        if len(labels) < 2:
            QtWidgets.QMessageBox.information(
                self, "Combine markers", "Tick at least two markers to combine.")
            return
        if not name:
            QtWidgets.QMessageBox.information(
                self, "Combine markers", "Give the new marker a name.")
            return
        existing = {m["label"] for m in self._markers}
        if name in existing:
            QtWidgets.QMessageBox.warning(
                self, "Combine markers", f"A marker named {name!r} already exists.")
            return
        new = combine_markers(self._markers, labels, name)
        for m in new:
            m["derived"] = True
        self._markers.extend(new)
        self.combine_name.clear()
        self._populate_markers()
        self._refresh_summary()

    def _reset_markers(self) -> None:
        if self._session is None:
            return
        self._markers = [dict(m) for m in self._session.markers]
        self._populate_markers()
        self._refresh_summary()

    # ----- compute ------------------------------------------------------- #

    def _collect_conditions(self) -> list[Condition]:
        """Group the ticked marker labels by their (editable) condition name."""
        grouped: dict[str, list[str]] = {}
        for row in range(self.marker_table.rowCount()):
            if self.marker_table.item(row, 0).checkState() != QtCore.Qt.Checked:
                continue
            label = self.marker_table.item(row, 1).data(QtCore.Qt.UserRole)
            cond = (self.marker_table.item(row, 3).text() or label).strip()
            grouped.setdefault(cond, []).append(label)
        return [Condition(name=n, labels=labs) for n, labs in grouped.items()]

    def _build_window(self) -> EpochWindow:
        baseline = None
        if self.baseline_chk.isChecked():
            baseline = (self.bmin.value(), self.bmax.value())
        return EpochWindow(
            tmin=self.tmin.value(), tmax=self.tmax.value(),
            baseline=baseline, reject_uv=self.reject.value())

    def _build_pipeline(self) -> Pipeline | None:
        if not self.preproc_combo.currentData():
            return None
        return Pipeline.from_config(
            self._ctl.config.preprocessing,
            sfreq=self._session.sfreq,
            ch_kinds=list(self._session.channel_kinds),
            ch_names=list(self._session.channel_names),
        )

    def _compute(self) -> None:
        if self._session is None:
            return
        conditions = self._collect_conditions()
        if not conditions:
            QtWidgets.QMessageBox.warning(
                self, "No conditions", "Tick at least one marker label.")
            return
        window = self._build_window()
        if window.tmax <= window.tmin:
            QtWidgets.QMessageBox.warning(
                self, "Bad window", "tmax must be greater than tmin.")
            return

        # Epoch against the working marker set (session markers + any derived).
        session = dataclasses.replace(self._session, markers=self._markers)

        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.compute_btn.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        try:
            result = compute_erp(
                session, conditions, window,
                preprocess=self._build_pipeline(),
                reject=self.reject_chk.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ERP analysis failed.")
            QtWidgets.QMessageBox.critical(self, "Analysis error", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.compute_btn.setEnabled(True)

        self._result = result
        self._cond_colors = {
            c.name: _PALETTE[i % len(_PALETTE)]
            for i, c in enumerate(result.conditions)}
        self._refresh_controls(result)
        self._show_summary(result)
        self._redraw_traces()
        self._redraw_topomap()
        self.export_btn.setEnabled(any(c.n_epochs for c in result.conditions))

    def _refresh_controls(self, result) -> None:
        prev = self.channel_combo.currentText()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(result.channel_names)
        idx = result.channel_names.index(prev) if prev in result.channel_names else \
            (result.eeg_indices[-1] if result.eeg_indices else 0)
        self.channel_combo.setCurrentIndex(idx)
        self.channel_combo.blockSignals(False)

        self.topo_cond.blockSignals(True)
        self.topo_cond.clear()
        self.topo_cond.addItems([c.name for c in result.conditions])
        self.topo_cond.blockSignals(False)

    def _show_summary(self, result) -> None:
        rows = []
        for c in result.conditions:
            color = self._cond_colors.get(c.name, "#cccccc")
            rows.append(
                f'<span style="color:{color}">■</span> <b>{c.name}</b> '
                f"({', '.join(c.labels)}): {c.n_epochs} epochs kept "
                f"of {c.n_onsets} (rejected {c.n_rejected}, "
                f"out-of-bounds {c.n_out_of_bounds})")
        warn = ""
        if result.warnings:
            warn = ('<br><span style="color:#e0c040">⚠ '
                    + "; ".join(result.warnings) + "</span>")
        self.summary.setText(
            f'<span style="color:#8fa3b0">{result.preprocessing}</span><br>'
            + "<br>".join(rows) + warn)

    # ----- plotting ------------------------------------------------------ #

    def _redraw_traces(self) -> None:
        if self._result is None:
            return
        self._draw_erp()
        self._draw_gfp()

    def _draw_erp(self) -> None:
        result = self._result
        self._clear_plot(self.erp_plot)
        t = result.times
        if self.butterfly_chk.isChecked():
            cond = self._current_topo_condition() or result.conditions[0]
            self.erp_plot.setTitle(f"Butterfly — {cond.name}")
            for i in result.eeg_indices:
                self.erp_plot.plot(t, cond.average[i],
                                   pen=pg.mkPen("#6fa8d0", width=1))
        else:
            ch = self.channel_combo.currentIndex()
            name = self.channel_combo.currentText()
            self.erp_plot.setTitle(f"Evoked response — {name}")
            for c in result.conditions:
                if not c.n_epochs:
                    continue
                color = self._cond_colors.get(c.name, "#cccccc")
                mean = c.average[ch]
                sem = c.sem[ch]
                upper = self.erp_plot.plot(t, mean + sem, pen=None)
                lower = self.erp_plot.plot(t, mean - sem, pen=None)
                fill = pg.FillBetweenItem(upper, lower,
                                          brush=pg.mkBrush(self._rgba(color, 50)))
                self.erp_plot.addItem(fill)
                self.erp_plot.plot(t, mean, pen=pg.mkPen(color, width=2),
                                   name=f"{c.name} (n={c.n_epochs})")
        self._mark_zero(self.erp_plot)

    def _draw_gfp(self) -> None:
        result = self._result
        self._clear_plot(self.gfp_plot)
        t = result.times
        for c in result.conditions:
            if not c.n_epochs:
                continue
            color = self._cond_colors.get(c.name, "#cccccc")
            gfp = global_field_power(c.average, result.eeg_indices)
            self.gfp_plot.plot(t, gfp, pen=pg.mkPen(color, width=2), name=c.name)
        self._mark_zero(self.gfp_plot)

    def _redraw_topomap(self) -> None:
        if self._result is None or self.canvas is None:
            return
        result = self._result
        cond = self._current_topo_condition()
        self.ax.clear()
        self.ax.set_facecolor("#14181c")
        if cond is not None and cond.n_epochs:
            idx = latency_index(result.times, self.latency.value())
            # Anchor only on scalp EEG channels so a non-scalp channel that
            # happens to share a 10-20-like name can never skew the map.
            vals = cond.average[:, idx].copy()
            eeg = set(result.eeg_indices)
            for i in range(len(result.channel_names)):
                if i not in eeg:
                    vals[i] = np.nan
            pos, found = channel_positions_2d(result.channel_names)
            gx, gy, gz = interpolate_topomap(vals, pos, found, res=120)
            if np.isfinite(gz).any():
                im = self.ax.imshow(
                    np.ma.masked_invalid(gz), origin="lower",
                    extent=(gx.min(), gx.max(), gy.min(), gy.max()),
                    cmap="RdBu_r", interpolation="bilinear", aspect="equal",
                    zorder=1)
                im.set_clip_path(Circle((0, 0), 1.0, transform=self.ax.transData))
            self.ax.add_artist(Circle((0, 0), 1.0, fill=False, color="#aaaaaa",
                                      lw=1.2, zorder=2))
            self.ax.plot([-0.12, 0, 0.12], [1.0, 1.13, 1.0], color="#aaaaaa",
                         lw=1.2, zorder=2)
            self.ax.scatter(pos[found, 0], pos[found, 1], c="#dddddd", s=10,
                            zorder=3)
            self.ax.set_title(
                f"{cond.name} @ {result.times[idx] * 1000:.0f} ms",
                color="#cdd4da", fontsize=9)
        self.ax.set_xlim(-1.25, 1.25)
        self.ax.set_ylim(-1.25, 1.3)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.canvas.draw_idle()

    def _current_topo_condition(self):
        if self._result is None:
            return None
        return self._result.condition(self.topo_cond.currentText())

    def _clear_plot(self, plot) -> None:
        legend = plot.getPlotItem().legend
        if legend is not None and hasattr(legend, "clear"):
            legend.clear()
        plot.clear()

    def _mark_zero(self, plot) -> None:
        plot.addLine(x=0.0, pen=pg.mkPen("#777777", style=QtCore.Qt.DashLine))

    @staticmethod
    def _rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)

    # ----- export -------------------------------------------------------- #

    def _export(self) -> None:
        if self._result is None:
            return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export averages", "erp_averages.npz", "NumPy archive (*.npz)")
        if not f:
            return
        result = self._result
        payload = {
            "times": result.times,
            "sfreq": result.sfreq,
            "channel_names": np.array(result.channel_names),
            "channel_kinds": np.array(result.channel_kinds),
            "preprocessing": result.preprocessing,
        }
        for c in result.conditions:
            payload[f"avg__{c.name}"] = c.average
            payload[f"sem__{c.name}"] = c.sem
            payload[f"n__{c.name}"] = c.n_epochs
        try:
            np.savez_compressed(f, **payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Export failed.")
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "Exported", f"Saved to:\n{f}")

    # ----- refresh ------------------------------------------------------- #

    def update_view(self) -> None:
        # Offline, on-demand analysis: nothing to refresh continuously.
        pass
