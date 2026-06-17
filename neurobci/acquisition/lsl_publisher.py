"""Publish a source as a virtual LSL stream.

Re-broadcasts any :class:`EEGSource` (simulated or replayed) onto an LSL
outlet so external tools -- or a second instance of this application -- can
consume it exactly like a real device. ``pylsl`` is imported lazily; the
outlet advertises proper channel labels/types so receivers can validate the
montage.
"""

from __future__ import annotations

import logging
import threading
import time

from neurobci.acquisition.base import EEGSource
from neurobci.core.stream_info import KIND_EOG

logger = logging.getLogger(__name__)


class LSLPublisher:
    def __init__(
        self,
        source: EEGSource,
        stream_name: str = "NeuroBCI-Virtual",
        source_id: str = "neurobci-virtual",
        interval_s: float = 0.05,
    ) -> None:
        self._source = source
        self._name = stream_name
        self._source_id = source_id
        self._interval = interval_s
        self._outlet = None
        self._marker_outlet = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        import pylsl  # lazy

        info = self._source.info
        lsl_info = pylsl.StreamInfo(
            self._name, "EEG", info.n_channels, info.sfreq, "float32", self._source_id)
        chans = lsl_info.desc().append_child("channels")
        for name, kind in zip(info.channel_names, info.channel_kinds):
            ch = chans.append_child("channel")
            ch.append_child_value("label", name)
            ch.append_child_value("type", "EOG" if kind == KIND_EOG else "EEG")
            ch.append_child_value("unit", "microvolts")
        self._outlet = pylsl.StreamOutlet(lsl_info)

        # Optional marker outlet for replay marker re-publication.
        if hasattr(self._source, "poll_markers"):
            m_info = pylsl.StreamInfo(
                f"{self._name}-markers", "Markers", 1, 0, "string",
                f"{self._source_id}-markers")
            self._marker_outlet = pylsl.StreamOutlet(m_info)

        self._source.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lsl-publisher",
                                        daemon=True)
        self._thread.start()
        logger.info("Publishing virtual LSL stream %r (%d ch @ %.1f Hz).",
                    self._name, info.n_channels, info.sfreq)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._source.read()
                if data.shape[0]:
                    self._outlet.push_chunk(data.astype("float32").tolist())
                if self._marker_outlet is not None:
                    for m in self._source.poll_markers():
                        self._marker_outlet.push_sample([m["label"]])
            except Exception:  # noqa: BLE001
                logger.exception("Virtual LSL publish step failed.")
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._source.stop()
        except Exception:  # noqa: BLE001
            pass
        self._outlet = None
        self._marker_outlet = None
        logger.info("Stopped virtual LSL stream %r.", self._name)
