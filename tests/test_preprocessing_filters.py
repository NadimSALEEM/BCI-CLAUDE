"""Filter correctness: frequency response, causal streaming, zero-phase."""

import unittest

import numpy as np
from scipy import signal

from neurobci.core.stream_info import KIND_EEG, KIND_EOG
from neurobci.preprocessing.stages import (
    BandPass,
    CommonAverageReference,
    HighPass,
    LowPass,
    Notch,
)

SF = 500.0


def _sine(freq, n=2000, amp=1.0, sf=SF):
    t = np.arange(n) / sf
    return amp * np.sin(2 * np.pi * freq * t)


def _power(x, freq, sf=SF):
    f, p = signal.welch(x, fs=sf, nperseg=min(len(x), 1024))
    return float(p[np.argmin(np.abs(f - freq))])


class TestFilters(unittest.TestCase):
    def _prep(self, stage, nch=1):
        stage.prepare(SF, [KIND_EEG] * nch)
        return stage

    def test_highpass_removes_low_freq(self):
        st = self._prep(HighPass(params={"cutoff_hz": 5.0}))
        x = (_sine(1.0) + _sine(20.0))[:, None]
        y = st.apply(x)
        # 1 Hz strongly attenuated, 20 Hz preserved.
        self.assertLess(_power(y[:, 0], 1.0), 0.1 * _power(x[:, 0], 1.0))
        self.assertGreater(_power(y[:, 0], 20.0), 0.5 * _power(x[:, 0], 20.0))

    def test_lowpass_removes_high_freq(self):
        st = self._prep(LowPass(params={"cutoff_hz": 15.0}))
        x = (_sine(5.0) + _sine(60.0))[:, None]
        y = st.apply(x)
        self.assertLess(_power(y[:, 0], 60.0), 0.1 * _power(x[:, 0], 60.0))
        self.assertGreater(_power(y[:, 0], 5.0), 0.5 * _power(x[:, 0], 5.0))

    def test_notch_removes_line(self):
        st = self._prep(Notch(params={"freq_hz": 50.0, "quality": 30.0}))
        x = (_sine(10.0) + _sine(50.0, amp=3.0))[:, None]
        y = st.apply(x)
        self.assertLess(_power(y[:, 0], 50.0), 0.05 * _power(x[:, 0], 50.0))
        self.assertGreater(_power(y[:, 0], 10.0), 0.5 * _power(x[:, 0], 10.0))

    def test_bandpass_keeps_passband(self):
        st = self._prep(BandPass(params={"low_hz": 8.0, "high_hz": 12.0}))
        x = (_sine(2.0) + _sine(10.0) + _sine(40.0))[:, None]
        y = st.apply(x)
        self.assertGreater(_power(y[:, 0], 10.0), _power(y[:, 0], 2.0) * 5)
        self.assertGreater(_power(y[:, 0], 10.0), _power(y[:, 0], 40.0) * 5)

    def test_streaming_equals_whole_signal_exactly(self):
        # The key real-time property: filtering in chunks (with carried
        # state) must equal filtering the whole signal in one shot.
        st = self._prep(HighPass(params={"cutoff_hz": 1.0}), nch=3)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1500, 3))
        whole = st.apply(x)

        st.reset()
        out = []
        for start in range(0, 1500, 137):     # uneven chunk sizes
            out.append(st.process_chunk(x[start:start + 137]))
        streamed = np.vstack(out)
        np.testing.assert_allclose(streamed, whole, rtol=1e-9, atol=1e-9)

    def test_offline_is_zero_phase(self):
        # filtfilt has no phase delay; a causal IIR does. Compare peak lag.
        st = self._prep(LowPass(params={"cutoff_hz": 20.0}))
        x = np.zeros((600, 1))
        x[300, 0] = 1.0  # impulse
        causal = st.apply(x)[:, 0]
        zero_phase = st.apply_offline(x)[:, 0]
        # Zero-phase peak stays near the impulse; causal peak is delayed.
        self.assertLessEqual(abs(np.argmax(zero_phase) - 300), 3)
        self.assertGreater(np.argmax(causal), 300)

    def test_car_removes_common_signal(self):
        st = CommonAverageReference()
        st.prepare(SF, [KIND_EEG, KIND_EEG, KIND_EEG, KIND_EOG])
        common = _sine(10.0, n=500)
        data = np.column_stack([
            common + _sine(3.0, n=500),
            common + _sine(7.0, n=500),
            common - _sine(5.0, n=500),
            _sine(1.0, n=500),  # EOG, must be untouched
        ])
        y = st.apply(data)
        # Mean across EEG channels per sample ~ 0 after CAR.
        self.assertLess(np.abs(y[:, :3].mean(axis=1)).max(), 1e-9)
        # EOG unchanged.
        np.testing.assert_allclose(y[:, 3], data[:, 3])


if __name__ == "__main__":
    unittest.main()
