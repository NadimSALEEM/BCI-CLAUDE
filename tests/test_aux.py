"""Auxiliary-stream QC: alignment, motion, quality->bad-channels, disconnects.

Synthetic tests pin the maths; a real-file test (skipped if the recording is
absent) proves it end-to-end on an actual Enobio XDF with ACC + Quality.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from neurobci.quality import aux
from neurobci.recording.exporter import AuxStream, LoadedSession

_REAL = Path("recordings/sub-P001_ses-S001_task-Default_run-123withACCandQLT_eeg.xdf")


def _session(data, sfreq=500.0, aux_streams=None, names=None):
    n, nch = data.shape
    names = names or [f"E{i}" for i in range(nch)]
    meta = {"channel_names": names, "channel_kinds": ["eeg"] * nch,
            "sfreq_nominal": sfreq, "n_channels": nch}
    return LoadedSession(path=Path("mem"), meta=meta, data=data.astype(np.float32),
                         timestamps=np.arange(n) / sfreq,
                         aux_streams=aux_streams or {})


class TestAlignment(unittest.TestCase):
    def test_previous_hold(self):
        aux_ts = np.array([0.0, 1.0, 2.0])
        vals = np.array([[-1.0], [0.9], [-1.0]])
        eeg_ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.9])
        out = aux.align_to_eeg(vals, aux_ts, eeg_ts, method="previous")
        np.testing.assert_array_equal(out[:, 0], [-1, -1, 0.9, 0.9, -1, -1])

    def test_linear(self):
        aux_ts = np.array([0.0, 2.0])
        vals = np.array([[0.0], [10.0]])
        eeg_ts = np.array([0.0, 1.0, 2.0])
        out = aux.align_to_eeg(vals, aux_ts, eeg_ts, method="linear")
        np.testing.assert_allclose(out[:, 0], [0.0, 5.0, 10.0])


class TestMotion(unittest.TestCase):
    def test_gravity_removed_and_burst_detected(self):
        sf = 100.0
        n = 2000
        acc = np.zeros((n, 3))
        acc[:, 2] = 11000.0                     # gravity on Z (~1 g)
        acc[900:1000, 0] += 3000.0              # a motion burst on X
        m = aux.motion_index(acc, sf, as_g=True)
        self.assertAlmostEqual(aux.gravity_g(acc), 11000.0, delta=50)
        self.assertLess(np.median(m), 0.05)     # mostly still
        self.assertGreater(m[950], 0.1)         # burst well above baseline

    def test_reported_in_g(self):
        sf = 100.0
        acc = np.zeros((500, 3))
        acc[:, 2] = 9800.0
        m = aux.motion_index(acc, sf, as_g=True)
        self.assertTrue(np.all(m < 0.01))       # pure gravity -> ~0 motion


class TestQuality(unittest.TestCase):
    def test_enobio_minus1_09(self):
        # 20 s at 1 Hz: ch0 mostly good, ch1 mostly bad, ch2 half/half
        q = np.full((20, 3), -1.0)
        q[:18, 0] = 0.9                          # 90% good
        q[:2, 1] = 0.9                           # 10% good
        q[:8, 2] = 0.9                           # 40% good (< 50%)
        bad, frac = aux.quality_bad_channels(q, threshold=0.0, good_when="high",
                                             min_good_frac=0.5)
        self.assertEqual(bad, [1, 2])           # ch0 good, ch1/ch2 below 50%
        self.assertAlmostEqual(frac[0], 0.9)

    def test_good_when_low(self):
        q = np.array([[0, 5], [0, 5], [0, 5]], float)  # 0 good, 5 bad
        bad, _ = aux.quality_bad_channels(q, threshold=1.0, good_when="low",
                                          min_good_frac=0.5)
        self.assertEqual(bad, [1])


class TestDisconnect(unittest.TestCase):
    def test_railed_and_flat(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1000, 4)) * 10
        x[:, 1] = -400000.0                      # railed
        x[:, 2] = 0.0                            # dead flat
        bad = aux.disconnected_channels(x)
        self.assertIn(1, bad)
        self.assertIn(2, bad)
        self.assertNotIn(0, bad)
        self.assertNotIn(3, bad)


class TestEpochReduce(unittest.TestCase):
    def test_shapes_and_values(self):
        sf = 100.0
        sig = np.zeros(1000)
        sig[500:520] = 5.0
        m = aux.epoch_reduce(sig, [500, 100], sf, tmin=0.0, tmax=0.1, reduce="max")
        self.assertEqual(m.shape, (2,))
        self.assertAlmostEqual(m[0], 5.0)        # window over the burst
        self.assertAlmostEqual(m[1], 0.0)
        q2d = np.tile(sig[:, None], (1, 3))
        r = aux.epoch_reduce(q2d, [500], sf, 0.0, 0.1, "max")
        self.assertEqual(r.shape, (1, 3))


class TestChannelReport(unittest.TestCase):
    def test_report_combines_quality_and_motion(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((1000, 4)) * 8
        q = np.full((10, 4), -1.0)
        q[:, 0] = 0.9                            # only ch0 in contact
        acc = np.zeros((200, 3))
        acc[:, 2] = 10500.0
        streams = {
            "Q": AuxStream("Q", "Quality", q, np.arange(10) / 1.0, 1.0),
            "A": AuxStream("A", "Accelerometer", acc, np.arange(200) / 100.0, 100.0),
        }
        rep = aux.channel_report(_session(data, aux_streams=streams))
        self.assertEqual(rep.quality_bad, [1, 2, 3])
        self.assertAlmostEqual(rep.gravity_g_raw, 10500.0, delta=50)
        self.assertLess(rep.median_motion_g, 0.05)


class TestRealFile(unittest.TestCase):
    def test_loader_captures_aux_and_flags_unplugged(self):
        if not _REAL.exists():
            self.skipTest(f"real recording not present: {_REAL}")
        try:
            import pyxdf  # noqa: F401
        except Exception:
            self.skipTest("pyxdf not installed")
        from neurobci.recording.external import load_xdf
        s = load_xdf(_REAL)
        # aux streams are now captured, not discarded
        types = {a.stype for a in s.aux_streams.values()}
        self.assertIn("Accelerometer", types)
        self.assertIn("Quality", types)
        rep = aux.channel_report(s)
        # electrodes were not plugged in -> quality flags (nearly) all channels
        self.assertGreaterEqual(len(rep.quality_bad), 18)
        # accelerometer sat still on a desk: ~1 g gravity, negligible motion
        self.assertTrue(10000 < rep.gravity_g_raw < 12000)
        self.assertLess(rep.median_motion_g, 0.05)


if __name__ == "__main__":
    unittest.main()
