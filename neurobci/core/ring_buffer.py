"""Thread-safe fixed-capacity ring buffer for multichannel samples.

This is the component that decouples acquisition from everything else.
The acquisition thread is the *single writer*; any number of readers
(GUI, quality monitor, future processing threads) call :meth:`latest`
to copy out a recent window. Readers never block the writer for longer
than a single ``memcpy``-style numpy copy, so a slow plot can never
stall data collection.

Layout: samples are stored as ``(capacity, n_channels)`` float32. A
monotonically increasing ``_total`` counter records how many samples
have ever been written, which lets readers detect overruns/gaps.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int, n_channels: int, dtype=np.float32) -> None:
        if capacity <= 0 or n_channels <= 0:
            raise ValueError("capacity and n_channels must be positive")
        self._capacity = int(capacity)
        self._n_channels = int(n_channels)
        self._data = np.zeros((self._capacity, self._n_channels), dtype=dtype)
        self._ts = np.zeros(self._capacity, dtype=np.float64)
        self._write = 0            # next write position
        self._total = 0            # total samples ever written
        self._lock = threading.Lock()

    # ----- properties ---------------------------------------------------- #

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total

    @property
    def n_available(self) -> int:
        with self._lock:
            return min(self._total, self._capacity)

    # ----- write --------------------------------------------------------- #

    def append(self, chunk: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Append ``chunk`` of shape ``(m, n_channels)``.

        If a chunk is larger than the whole buffer (pathological), only its
        last ``capacity`` samples are retained.
        """

        chunk = np.asarray(chunk, dtype=self._data.dtype)
        if chunk.ndim != 2 or chunk.shape[1] != self._n_channels:
            raise ValueError(
                f"chunk must be (m, {self._n_channels}), got {chunk.shape}"
            )
        m = chunk.shape[0]
        if m == 0:
            return
        if timestamps is None:
            timestamps = np.full(m, np.nan, dtype=np.float64)
        else:
            timestamps = np.asarray(timestamps, dtype=np.float64)
            if timestamps.shape[0] != m:
                raise ValueError("timestamps length must match chunk rows")

        with self._lock:
            if m >= self._capacity:
                # Keep only the most recent `capacity` samples.
                self._data[:] = chunk[-self._capacity:]
                self._ts[:] = timestamps[-self._capacity:]
                self._write = 0
                self._total += m
                return
            end = self._write + m
            if end <= self._capacity:
                self._data[self._write:end] = chunk
                self._ts[self._write:end] = timestamps
            else:                                  # wrap-around
                first = self._capacity - self._write
                self._data[self._write:] = chunk[:first]
                self._ts[self._write:] = timestamps[:first]
                self._data[: m - first] = chunk[first:]
                self._ts[: m - first] = timestamps[first:]
            self._write = end % self._capacity
            self._total += m

    # ----- read ---------------------------------------------------------- #

    def latest(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``n`` most recent samples in chronological order.

        Returns ``(data, timestamps)`` where ``data`` is ``(k, n_channels)``
        with ``k = min(n, n_available)``. The arrays are fresh copies, safe
        to use without holding the lock.
        """

        with self._lock:
            avail = min(self._total, self._capacity)
            k = min(int(n), avail)
            if k == 0:
                return (
                    np.empty((0, self._n_channels), dtype=self._data.dtype),
                    np.empty(0, dtype=np.float64),
                )
            start = (self._write - k) % self._capacity
            end = start + k
            if end <= self._capacity:
                data = self._data[start:end].copy()
                ts = self._ts[start:end].copy()
            else:
                first = self._capacity - start
                data = np.empty((k, self._n_channels), dtype=self._data.dtype)
                ts = np.empty(k, dtype=np.float64)
                data[:first] = self._data[start:]
                data[first:] = self._data[: k - first]
                ts[:first] = self._ts[start:]
                ts[first:] = self._ts[: k - first]
            return data, ts

    def clear(self) -> None:
        with self._lock:
            self._write = 0
            self._total = 0
            self._data.fill(0)
            self._ts.fill(0)
