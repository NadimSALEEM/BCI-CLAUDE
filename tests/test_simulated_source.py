"""Synthetic EEG: shapes, determinism, known spectral ground truth, faults."""

import unittest

import numpy as np
from scipy import signal

from neurobci.acquisition.simulated import SimulatedSource
from neurobci.config.schema import ChannelConfig, SimulationConfig


def _make_source(**sim_kwargs):
    channels = ChannelConfig()
    sim = SimulationConfig(**sim_kwargs)
    return SimulatedSource(channels=channels, sim=sim, sfreq=500.0, realtime=False)


def _bandpower(x, sfreq, lo, hi):
    freqs, psd = signal.welch(x, fs=sfreq, nperseg=min(len(x), 1024))
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.trapezoid(psd[mask], freqs[mask]))


class TestSimulatedSource(unittest.TestCase):
    def test_generate_shape_and_kinds(self):
        src = _make_source()
        data, ts = src.generate(500)
        self.assertEqual(data.shape, (500, src.info.n_channels))
        self.assertEqual(ts.shape, (500,))
        self.assertEqual(data.dtype, np.float32)
        # Default Enobio montage => 20 channels, exactly one EOG.
        self.assertEqual(len(src.info.eog_indices), 1)

    def test_determinism_same_seed(self):
        a = _make_source(seed=99).generate(1000)[0]
        b = _make_source(seed=99).generate(1000)[0]
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self):
        a = _make_source(seed=1).generate(1000)[0]
        b = _make_source(seed=2).generate(1000)[0]
        self.assertFalse(np.array_equal(a, b))

    def test_alpha_is_posterior(self):
        # Occipital channels should carry more alpha than frontal ones.
        src = _make_source(blink_rate_hz=0.0)  # remove blink confound
        data, _ = src.generate(2000)  # 4 s
        info = src.info
        o1 = data[:, info.index_of("O1")]
        fp1 = data[:, info.index_of("Fp1")]
        alpha_o1 = _bandpower(o1, 500.0, 8, 12)
        alpha_fp1 = _bandpower(fp1, 500.0, 8, 12)
        self.assertGreater(alpha_o1, alpha_fp1)

    def test_flat_channel_injection(self):
        src = _make_source(flat_channels=["Cz"])
        data, _ = src.generate(1000)
        cz = data[:, src.info.index_of("Cz")]
        other = data[:, src.info.index_of("O1")]
        self.assertLess(np.std(cz), 2.0)          # near-flat
        self.assertGreater(np.std(other), np.std(cz) * 3)

    def test_noisy_channel_injection(self):
        src = _make_source(noisy_channels=["Pz"])
        data, _ = src.generate(1000)
        pz = data[:, src.info.index_of("Pz")]
        clean = data[:, src.info.index_of("O1")]
        self.assertGreater(np.std(pz), np.std(clean))

    def test_inject_erp_produces_parietal_deflection(self):
        # Isolate the ERP by suppressing other components.
        src = _make_source(
            blink_rate_hz=0.0, white_noise_uv=0.3, pink_noise_uv=0.3,
            line_noise_uv=0.0, drift_uv=0.0, alpha_amp_uv=0.0,
            theta_amp_uv=0.0, beta_amp_uv=0.0,
        )
        src.generate(50)              # advance a little
        src.inject_erp(amplitude_uv=30.0)
        data, _ = src.generate(300)   # 0.6 s -> contains the ERP
        pz = data[:, src.info.index_of("Pz")]
        eog = data[:, src.info.index_of("EOG")]
        # Positive parietal peak near 300 ms (~sample 150 at 500 Hz).
        self.assertGreater(pz.max(), 12.0)
        self.assertLess(abs(int(np.argmax(pz)) - 150), 40)
        # EOG carries no ERP, so its excursion stays far below the parietal peak.
        self.assertLess(abs(eog).max(), pz.max())

    def test_deterministic_signal_is_chunk_continuous(self):
        # With all stochastic components disabled, the signal is a pure
        # function of the absolute sample index, so chunking must not matter
        # (verifies phase/filter-state continuity at chunk boundaries).
        kw = dict(
            pink_noise_uv=0.0, white_noise_uv=0.0, drift_uv=0.0,
            blink_rate_hz=0.0, seed=7,
        )
        whole = _make_source(**kw).generate(500)[0]
        src = _make_source(**kw)
        part = np.vstack([src.generate(250)[0], src.generate(250)[0]])
        np.testing.assert_allclose(whole, part, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
