"""Correctness of the additional preprocessing stages."""

import unittest

import numpy as np
from scipy import signal

from neurobci.core.stream_info import KIND_EEG, KIND_EOG
from neurobci.preprocessing.stages import (
    STAGE_REGISTRY,
    AmplitudeClamp,
    BandStop,
    Laplacian,
    MovingAverage,
    RobustReference,
    SavitzkyGolay,
    Standardize,
    make_stage,
)

SF = 500.0
# A small, real 10-20 subset (names matter for the Laplacian).
ENOBIO = ["Fz", "Cz", "Pz", "C3", "C4", "F3", "F4", "EOG"]
KINDS = [KIND_EEG] * 7 + [KIND_EOG]


def _sine(freq, n=2000, amp=1.0, sf=SF):
    t = np.arange(n) / sf
    return amp * np.sin(2 * np.pi * freq * t)


def _power(x, freq, sf=SF):
    f, p = signal.welch(x, fs=sf, nperseg=min(len(x), 1024))
    return float(p[np.argmin(np.abs(f - freq))])


class TestBandStop(unittest.TestCase):
    def test_rejects_band_keeps_outside(self):
        st = BandStop(params={"low_hz": 18.0, "high_hz": 22.0})
        st.prepare(SF, [KIND_EEG])
        x = (_sine(10.0) + _sine(20.0) + _sine(35.0))[:, None]
        y = st.apply(x)
        self.assertLess(_power(y[:, 0], 20.0), 0.1 * _power(x[:, 0], 20.0))
        self.assertGreater(_power(y[:, 0], 10.0), 0.5 * _power(x[:, 0], 10.0))
        self.assertGreater(_power(y[:, 0], 35.0), 0.5 * _power(x[:, 0], 35.0))

    def test_streaming_equals_whole(self):
        st = BandStop(params={"low_hz": 48.0, "high_hz": 52.0})
        st.prepare(SF, [KIND_EEG, KIND_EEG])
        x = np.random.default_rng(0).standard_normal((1500, 2))
        whole = st.apply(x)
        st.reset()
        out = [st.process_chunk(x[s:s + 137]) for s in range(0, 1500, 137)]
        np.testing.assert_allclose(np.vstack(out), whole, rtol=1e-9, atol=1e-9)


class TestMovingAverage(unittest.TestCase):
    def test_smooths_high_freq(self):
        st = MovingAverage(params={"window_ms": 20.0})
        st.prepare(SF, [KIND_EEG])
        x = (_sine(2.0) + _sine(80.0))[:, None]
        y = st.apply(x)
        self.assertLess(_power(y[:, 0], 80.0), 0.5 * _power(x[:, 0], 80.0))
        self.assertGreater(_power(y[:, 0], 2.0), 0.5 * _power(x[:, 0], 2.0))

    def test_streaming_equals_whole(self):
        st = MovingAverage(params={"window_ms": 30.0})
        st.prepare(SF, [KIND_EEG, KIND_EEG, KIND_EEG])
        x = np.random.default_rng(1).standard_normal((1200, 3))
        whole = st.apply(x)
        st.reset()
        out = [st.process_chunk(x[s:s + 101]) for s in range(0, 1200, 101)]
        np.testing.assert_allclose(np.vstack(out), whole, rtol=1e-9, atol=1e-9)


class TestAmplitudeClamp(unittest.TestCase):
    def test_clamps_extremes(self):
        st = AmplitudeClamp(params={"limit_uv": 100.0})
        st.prepare(SF, [KIND_EEG, KIND_EEG])
        x = np.array([[10.0, -250.0], [500.0, 50.0]])
        y = st.apply(x)
        np.testing.assert_allclose(y, [[10.0, -100.0], [100.0, 50.0]])


class TestRobustReference(unittest.TestCase):
    def test_subtracts_median_ignores_eog(self):
        st = RobustReference()
        st.prepare(SF, [KIND_EEG, KIND_EEG, KIND_EEG, KIND_EOG])
        common = _sine(10.0, n=500)
        data = np.column_stack([
            common + _sine(3.0, n=500),
            common + _sine(7.0, n=500),
            common - _sine(5.0, n=500),
            _sine(1.0, n=500),                       # EOG must be untouched
        ])
        y = st.apply(data)
        # Per-sample median across EEG channels is ~0 after referencing.
        self.assertLess(np.abs(np.median(y[:, :3], axis=1)).max(), 1e-9)
        np.testing.assert_allclose(y[:, 3], data[:, 3])

    def test_robust_to_one_huge_channel(self):
        # A single wild channel must not swamp the reference (unlike the mean).
        st = RobustReference()
        st.prepare(SF, [KIND_EEG] * 4)
        base = np.random.default_rng(3).standard_normal((300, 4))
        base[:, 0] += 1000.0                          # one bad channel
        y = st.apply(base)
        # The three good channels stay close to their de-medianed selves.
        self.assertLess(np.abs(y[:, 1:]).mean(), 5.0)


class TestLaplacian(unittest.TestCase):
    def test_attenuates_common_signal_eog_untouched(self):
        st = Laplacian(params={"neighbors": 3})
        st.prepare(SF, KINDS, ENOBIO)
        common = _sine(10.0, n=400)
        data = np.tile(common[:, None], (1, len(ENOBIO)))
        data[:, -1] = _sine(1.0, n=400)               # EOG distinct
        y = st.apply(data)
        # A spatially-uniform signal is strongly suppressed on EEG channels.
        self.assertLess(np.abs(y[:, :7]).max(), 0.2 * np.abs(data[:, :7]).max())
        # EOG (no position) passes through unchanged.
        np.testing.assert_allclose(y[:, -1], data[:, -1])

    def test_passthrough_without_names(self):
        st = Laplacian()
        st.prepare(SF, KINDS)                          # no ch_names
        self.assertIn("positions unavailable", " ".join(st.validate(SF)))
        x = np.random.default_rng(4).standard_normal((100, len(ENOBIO)))
        np.testing.assert_allclose(st.apply(x), x)


class TestStandardize(unittest.TestCase):
    def test_offline_zero_mean_unit_std(self):
        st = Standardize()
        st.prepare(SF, [KIND_EEG, KIND_EEG])
        x = np.random.default_rng(5).standard_normal((600, 2)) * 7.0 + 3.0
        y = st.apply_offline(x)
        np.testing.assert_allclose(y.mean(axis=0), [0, 0], atol=1e-6)
        np.testing.assert_allclose(y.std(axis=0), [1, 1], atol=1e-3)

    def test_causal_streaming_matches_whole(self):
        st = Standardize(params={"halflife_s": 0.5})
        st.prepare(SF, [KIND_EEG, KIND_EEG])
        x = np.random.default_rng(6).standard_normal((1000, 2)) * 4.0
        whole = st.apply(x)
        st.reset()
        out = [st.process_chunk(x[s:s + 73]) for s in range(0, 1000, 73)]
        np.testing.assert_allclose(np.vstack(out), whole, rtol=1e-7, atol=1e-7)

    def test_causal_reaches_unit_variance(self):
        st = Standardize(params={"halflife_s": 0.3})
        st.prepare(SF, [KIND_EEG])
        x = (np.random.default_rng(7).standard_normal((4000, 1)) * 20.0)
        y = st.apply(x)
        # After the transient, the standardised signal is ~unit scale.
        self.assertLess(abs(y[2000:, 0].std() - 1.0), 0.25)


class TestSavitzkyGolay(unittest.TestCase):
    def test_offline_smooths_causal_is_passthrough(self):
        st = SavitzkyGolay(params={"window_ms": 40.0, "polyorder": 3})
        st.prepare(SF, [KIND_EEG])
        x = (_sine(3.0, n=800) + _sine(90.0, n=800, amp=0.5))[:, None]
        causal = st.apply(x)
        np.testing.assert_allclose(causal, x)         # skipped in causal use
        smoothed = st.apply_offline(x)
        self.assertLess(_power(smoothed[:, 0], 90.0), 0.3 * _power(x[:, 0], 90.0))
        self.assertFalse(st.realtime_safe)


class TestRegistry(unittest.TestCase):
    def test_all_new_types_registered_and_roundtrip(self):
        for name in ("bandstop", "robust_ref", "laplacian", "clamp",
                     "moving_average", "standardize", "savgol"):
            self.assertIn(name, STAGE_REGISTRY)
            st = make_stage({"type": name, "enabled": True, "params": {}})
            d = st.to_dict()
            self.assertEqual(d["type"], name)
            # Rebuilding from the serialised dict yields the same params.
            again = make_stage(d)
            self.assertEqual(again.params, st.params)


if __name__ == "__main__":
    unittest.main()
