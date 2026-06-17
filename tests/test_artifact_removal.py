"""Calibrated artifact removal: bad-channel repair, ICA and ASR.

Each method is validated on synthetic data with a *known* artifact so we can
check it removes the artifact while preserving the underlying signal -- and
that it is an honest pass-through until calibrated.
"""

import unittest

import numpy as np

from neurobci.core.stream_info import KIND_EEG, KIND_EOG
from neurobci.preprocessing.artifact_removal import ASR, ICARemoval, InterpolateBad
from neurobci.preprocessing.stages import STAGE_REGISTRY, make_stage

SF = 500.0
ENOBIO = ["Fz", "Cz", "Pz", "C3", "C4", "F3", "F4", "P3", "P4", "EOG"]
KINDS = [KIND_EEG] * 9 + [KIND_EOG]


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return abs(float(a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


class TestInterpolateBad(unittest.TestCase):
    def test_detects_and_repairs_saturated_channel(self):
        rng = np.random.default_rng(0)
        T = 1500
        data = rng.standard_normal((T, len(ENOBIO))) * 10.0
        bad = ENOBIO.index("Pz")
        data[:, bad] += 2000.0 * np.sign(rng.standard_normal(T))  # railing electrode
        st = InterpolateBad()
        st.prepare(SF, KINDS, ENOBIO)

        # Honest pass-through before calibration.
        np.testing.assert_allclose(st.apply(data), data)
        self.assertTrue(any("not calibrated" in w for w in st.validate(SF)))

        st.fit(data)
        self.assertIn("Pz", st.bad_names)
        out = st.apply(data)
        # The repaired channel is no longer railing and is now neighbour-scaled.
        self.assertLess(np.abs(out[:, bad]).max(), 200.0)
        # Other channels are untouched.
        good = ENOBIO.index("C3")
        np.testing.assert_allclose(out[:, good], data[:, good])


class TestICARemoval(unittest.TestCase):
    def test_removes_eog_component_preserves_brain(self):
        rng = np.random.default_rng(1)
        T = 5000
        t = np.arange(T) / SF
        s1 = np.sin(2 * np.pi * 10 * t)               # alpha-like
        s2 = np.sin(2 * np.pi * 6 * t)                # theta-like
        s3 = rng.standard_normal(T) * 0.5
        blink = np.zeros(T)                           # sparse, spiky (super-Gaussian)
        for c in rng.choice(T, 25, replace=False):
            blink[max(0, c - 10):c + 10] += rng.standard_normal() * 6.0
        src = np.column_stack([s1, s2, s3, blink])
        A = rng.standard_normal((9, 4))
        A[:, 3] = np.abs(A[:, 3]) + 0.5               # blink loads on all EEG
        eeg = src @ A.T
        eog = blink * 5.0 + rng.standard_normal(T) * 0.1
        data = np.column_stack([eeg, eog])

        st = ICARemoval(params={"max_remove": 2})
        st.prepare(SF, KINDS, ENOBIO)
        np.testing.assert_allclose(st.apply(data), data)   # pass-through pre-fit

        st.fit(data)
        self.assertGreaterEqual(st.removed, 1)
        out = st.apply(data)

        eeg_i = list(range(9))
        raw_blink = np.mean([_corr(eeg[:, c], blink) for c in range(9)])
        out_blink = np.mean([_corr(out[:, c], blink) for c in eeg_i])
        self.assertLess(out_blink, 0.75 * raw_blink)       # artifact clearly reduced
        # Brain content (10 Hz) largely preserved.
        raw_a = np.mean([_corr(eeg[:, c], s1) for c in range(9)])
        out_a = np.mean([_corr(out[:, c], s1) for c in eeg_i])
        self.assertGreater(out_a, 0.6 * raw_a)
        # EOG channel left untouched.
        np.testing.assert_allclose(out[:, 9], data[:, 9])


class TestASR(unittest.TestCase):
    def _clean(self, T, n, rng):
        mix = rng.standard_normal((n, n))
        return rng.standard_normal((T, n)) @ mix.T

    def test_reconstructs_burst_preserves_clean(self):
        rng = np.random.default_rng(2)
        n = 6
        kinds = [KIND_EEG] * n
        names = ["Fz", "Cz", "Pz", "C3", "C4", "P3"]
        clean = self._clean(8000, n, rng)
        st = ASR(params={"cutoff": 5.0, "window_ms": 500.0})
        st.prepare(SF, kinds, names)
        seg = clean[:1500].copy()
        np.testing.assert_allclose(st.apply(seg), seg)     # pass-through pre-fit

        st.fit(clean)
        # Inject a big transient in one spatial direction.
        d = rng.standard_normal(n)
        d /= np.linalg.norm(d)
        seg[600:700] += np.outer(rng.standard_normal(100) * 40.0, d)
        out = st.apply(seg)

        burst_in = float(np.var(seg[600:700]))
        burst_out = float(np.var(out[600:700]))
        self.assertLess(burst_out, 0.6 * burst_in)         # transient attenuated
        # A clean stretch is largely preserved.
        keep = np.mean([_corr(out[:500, c], seg[:500, c]) for c in range(n)])
        self.assertGreater(keep, 0.9)

    def test_streaming_chunks_pass_through_when_unfitted(self):
        st = ASR()
        st.prepare(SF, [KIND_EEG] * 4, ["Fz", "Cz", "Pz", "C3"])
        x = np.random.default_rng(3).standard_normal((137, 4))
        np.testing.assert_allclose(st.process_chunk(x), x)


class TestRegistry(unittest.TestCase):
    def test_registered_and_constructible(self):
        for name in ("interpolate_bad", "ica", "asr"):
            self.assertIn(name, STAGE_REGISTRY)
            st = make_stage({"type": name, "enabled": True, "params": {}})
            self.assertTrue(st.requires_fit)
            self.assertFalse(st.fitted)


if __name__ == "__main__":
    unittest.main()
