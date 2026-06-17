"""The source contract shared by every acquisition backend.

A *source* produces multichannel samples in (approximately) real time.
The three Phase-1+ backends -- :class:`SimulatedSource`, ``LSLSource`` and
``ReplaySource`` -- all implement this interface, so the acquisition
thread and everything downstream are completely backend-agnostic.

Pull model: the acquisition thread repeatedly calls :meth:`read`, which
returns however many samples have become available since the last call
(possibly zero). This keeps backpressure simple and lets a single thread
service any source.
"""

from __future__ import annotations

import abc

import numpy as np

from neurobci.core.stream_info import StreamInfo


class EEGSource(abc.ABC):
    """Abstract base class for all sample sources."""

    @property
    @abc.abstractmethod
    def info(self) -> StreamInfo:
        """Stream metadata. Must be valid after :meth:`start` returns."""

    @abc.abstractmethod
    def start(self) -> None:
        """Open the stream / begin generation. Idempotent."""

    @abc.abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(samples, timestamps)`` available since the last call.

        ``samples`` has shape ``(m, n_channels)`` and ``timestamps`` shape
        ``(m,)``; ``m`` may be 0. Implementations must be non-blocking
        (return quickly even if no data is available).
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Close the stream / stop generation. Idempotent."""

    # Optional context-manager sugar.
    def __enter__(self) -> "EEGSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
