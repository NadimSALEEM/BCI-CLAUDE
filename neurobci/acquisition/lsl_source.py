"""Lab Streaming Layer acquisition backend.

``pylsl`` (and its native ``liblsl``) is imported lazily inside
:meth:`start`, so simulation/replay users never need the native library
installed. Channel names and types are read from the stream's XML
metadata when present, falling back to the configured montage otherwise
-- and the EOG channel is never silently treated as scalp EEG.
"""

from __future__ import annotations

import logging

import numpy as np

from neurobci.acquisition.base import EEGSource
from neurobci.config.schema import AcquisitionConfig, ChannelConfig
from neurobci.core.electrodes import classify_channel
from neurobci.core.stream_info import KIND_EOG, StreamInfo

logger = logging.getLogger(__name__)


class LSLError(RuntimeError):
    pass


class LSLSource(EEGSource):
    def __init__(self, acq: AcquisitionConfig, channels: ChannelConfig) -> None:
        self._acq = acq
        self._channels = channels
        self._inlet = None
        self._info: StreamInfo | None = None
        self._started = False

    @property
    def info(self) -> StreamInfo:
        if self._info is None:
            raise LSLError("Stream not started; info unavailable.")
        return self._info

    def start(self) -> None:
        if self._started:
            return
        try:
            import pylsl  # lazy: only needed for live acquisition
        except Exception as exc:  # noqa: BLE001
            raise LSLError(f"pylsl/liblsl unavailable: {exc}") from exc

        streams = self._resolve(pylsl)
        if not streams:
            raise LSLError(
                "No matching LSL stream found "
                f"(name={self._acq.lsl_stream_name!r}, type={self._acq.lsl_stream_type!r})."
            )
        lsl_info = streams[0]
        self._inlet = pylsl.StreamInlet(
            lsl_info, max_buflen=60, recover=True
        )
        self._info = self._build_info(self._inlet.info())
        self._started = True
        logger.info(
            "Connected to LSL stream %r (%d ch @ %.1f Hz).",
            self._info.name, self._info.n_channels, self._info.sfreq,
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._started or self._inlet is None:
            return self._empty()
        samples, timestamps = self._inlet.pull_chunk(timeout=0.0)
        if not samples:
            return self._empty()
        data = np.asarray(samples, dtype=np.float32)
        ts = np.asarray(timestamps, dtype=np.float64)
        return data, ts

    def stop(self) -> None:
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing LSL inlet.", exc_info=True)
        self._inlet = None
        self._started = False

    # ----- helpers ------------------------------------------------------- #

    def _resolve(self, pylsl, timeout: float = 3.0):
        name = self._acq.lsl_stream_name.strip()
        stype = self._acq.lsl_stream_type.strip()
        if name:
            return pylsl.resolve_byprop("name", name, timeout=timeout)
        if stype:
            return pylsl.resolve_byprop("type", stype, timeout=timeout)
        return pylsl.resolve_streams(wait_time=timeout)

    def _build_info(self, lsl_info) -> StreamInfo:
        n = lsl_info.channel_count()
        sfreq = lsl_info.nominal_srate() or self._acq.expected_sfreq
        names, kinds = self._parse_channels(lsl_info, n)
        return StreamInfo(
            name=lsl_info.name(),
            sfreq=float(sfreq),
            channel_names=names,
            channel_kinds=kinds,
            source_kind="lsl",
            units="uV",
        )

    def _parse_channels(self, lsl_info, n: int) -> tuple[list[str], list[str]]:
        """Read channel labels/types from LSL XML metadata if available."""
        names: list[str] = []
        types: list[str] = []
        try:
            ch = lsl_info.desc().child("channels").child("channel")
            while not ch.empty():
                label = ch.child_value("label")
                ctype = (ch.child_value("type") or "").lower()
                names.append(label or f"ch{len(names)}")
                types.append(ctype)
                ch = ch.next_sibling()
        except Exception:  # noqa: BLE001
            logger.debug("No usable channel metadata in LSL stream.", exc_info=True)

        if len(names) != n or not any(names):
            # Fall back to the configured montage (manual correction path).
            logger.warning(
                "LSL stream metadata incomplete; using configured montage names."
            )
            cfg_names = self._channels.all_channels
            names = (cfg_names + [f"ch{i}" for i in range(n)])[:n]
            types = [""] * n

        # Auto-detect each channel's kind from its name + declared type, but
        # let the configured montage's explicit EOG list win (manual override).
        eog_set = {c.lower() for c in self._channels.eog_channels}
        kinds = [
            KIND_EOG if label.lower() in eog_set
            else classify_channel(label, ctype)
            for label, ctype in zip(names, types)
        ]
        return names, kinds

    def _empty(self) -> tuple[np.ndarray, np.ndarray]:
        nch = self._info.n_channels if self._info else 0
        return (
            np.empty((0, nch), dtype=np.float32),
            np.empty(0, dtype=np.float64),
        )
