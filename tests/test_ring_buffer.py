"""Ring-buffer correctness: ordering, wrap-around, overrun, empties."""

import unittest

import numpy as np

from neurobci.core.ring_buffer import RingBuffer


class TestRingBuffer(unittest.TestCase):
    def _chunk(self, start, m, nch):
        # Each row r has value start+r in every channel; easy to verify order.
        col = np.arange(start, start + m, dtype=np.float32)
        return np.repeat(col[:, None], nch, axis=1)

    def test_basic_append_and_latest(self):
        rb = RingBuffer(capacity=10, n_channels=3)
        rb.append(self._chunk(0, 4, 3))
        data, ts = rb.latest(4)
        self.assertEqual(data.shape, (4, 3))
        np.testing.assert_array_equal(data[:, 0], [0, 1, 2, 3])
        self.assertEqual(rb.n_available, 4)
        self.assertEqual(rb.total_written, 4)

    def test_latest_more_than_available(self):
        rb = RingBuffer(capacity=10, n_channels=2)
        rb.append(self._chunk(0, 3, 2))
        data, _ = rb.latest(100)
        self.assertEqual(data.shape, (3, 2))

    def test_wraparound_preserves_order(self):
        rb = RingBuffer(capacity=5, n_channels=1)
        rb.append(self._chunk(0, 4, 1))   # [0,1,2,3]
        rb.append(self._chunk(4, 4, 1))   # wraps; newest 5 => [3,4,5,6,7]
        data, _ = rb.latest(5)
        np.testing.assert_array_equal(data[:, 0], [3, 4, 5, 6, 7])
        self.assertEqual(rb.total_written, 8)
        self.assertEqual(rb.n_available, 5)

    def test_oversized_chunk_keeps_tail(self):
        rb = RingBuffer(capacity=4, n_channels=1)
        rb.append(self._chunk(0, 10, 1))  # only last 4 kept => [6,7,8,9]
        data, _ = rb.latest(4)
        np.testing.assert_array_equal(data[:, 0], [6, 7, 8, 9])

    def test_timestamps_follow_data(self):
        rb = RingBuffer(capacity=6, n_channels=1)
        rb.append(self._chunk(0, 3, 1), timestamps=np.array([10.0, 11.0, 12.0]))
        _, ts = rb.latest(3)
        np.testing.assert_array_equal(ts, [10.0, 11.0, 12.0])

    def test_empty_read(self):
        rb = RingBuffer(capacity=4, n_channels=2)
        data, ts = rb.latest(3)
        self.assertEqual(data.shape, (0, 2))
        self.assertEqual(ts.shape, (0,))

    def test_clear(self):
        rb = RingBuffer(capacity=4, n_channels=1)
        rb.append(self._chunk(0, 3, 1))
        rb.clear()
        self.assertEqual(rb.n_available, 0)
        self.assertEqual(rb.total_written, 0)

    def test_bad_shape_raises(self):
        rb = RingBuffer(capacity=4, n_channels=3)
        with self.assertRaises(ValueError):
            rb.append(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
