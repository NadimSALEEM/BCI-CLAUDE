"""Persistent status strip shown at the bottom of the main window.

Surfaces, at all times, the safety- and trust-critical facts the spec
asks for: operating mode, connection health, stream rate, recording
state, signal quality, current prediction/command and the emergency-stop
flag. It is a pure *view* -- it only renders snapshots handed to it.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from neurobci.core.app_state import ConnectionStatus
from neurobci.quality.metrics import QualityRating

# Colour per connection / quality status (dark-theme friendly).
_CONN_COLOR = {
    ConnectionStatus.DISCONNECTED: "#888888",
    ConnectionStatus.CONNECTING: "#d0a000",
    ConnectionStatus.CONNECTED: "#36c24a",
    ConnectionStatus.STALE: "#d08000",
    ConnectionStatus.LOST: "#d83030",
    ConnectionStatus.SIMULATED: "#3a9bd0",
    ConnectionStatus.REPLAYED: "#9b59b6",
}
_QUALITY_COLOR = {
    QualityRating.GOOD: "#36c24a",
    QualityRating.FAIR: "#c8c020",
    QualityRating.POOR: "#d08000",
    QualityRating.BAD: "#d83030",
    QualityRating.NODATA: "#888888",
}


class _Pill(QtWidgets.QLabel):
    """A small rounded label used as a status indicator."""

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumWidth(90)
        self._set_style("#888888")

    def _set_style(self, color: str) -> None:
        self.setStyleSheet(
            f"QLabel {{ background:{color}; color:#0e0e0e; "
            f"border-radius:7px; padding:2px 8px; font-weight:600; }}"
        )

    def set(self, text: str, color: str) -> None:
        self.setText(text)
        self._set_style(color)


class StatusBar(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(10)

        self._mode = _Pill("idle")
        self._conn = _Pill("disconnected")
        self._quality = _Pill("quality: -")
        self._rate = QtWidgets.QLabel("-- Hz")
        self._samples = QtWidgets.QLabel("0 samples")
        self._rec = _Pill("not recording")
        self._pred = QtWidgets.QLabel("pred: -")
        self._estop = _Pill("E-STOP OK")

        for w in (self._mode, self._conn, self._quality):
            layout.addWidget(w)
        layout.addWidget(self._sep())
        layout.addWidget(self._rate)
        layout.addWidget(self._samples)
        layout.addWidget(self._sep())
        layout.addWidget(self._pred)
        layout.addStretch(1)
        layout.addWidget(self._rec)
        layout.addWidget(self._estop)

    def _sep(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setStyleSheet("color:#444;")
        return line

    def update_status(self, snap, quality_rating: QualityRating | None) -> None:
        self._mode.set(snap.mode.value, "#5a6acb")
        conn = snap.connection
        self._conn.set(conn.value, _CONN_COLOR.get(conn, "#888888"))

        if quality_rating is not None:
            self._quality.set(
                f"quality: {quality_rating.value}",
                _QUALITY_COLOR.get(quality_rating, "#888888"),
            )

        nominal = snap.stream_info.sfreq if snap.stream_info else 0.0
        self._rate.setText(f"{snap.measured_sfreq:5.1f} / {nominal:.0f} Hz")
        self._samples.setText(f"{snap.samples_received:,} samples")
        self._pred.setText(f"pred: {snap.last_prediction}")

        if snap.recording:
            self._rec.set("● recording", "#d83030")
        else:
            self._rec.set("not recording", "#666666")

        if snap.emergency_stop:
            self._estop.set("E-STOP ACTIVE", "#d83030")
        else:
            self._estop.set("E-STOP OK", "#36c24a")
