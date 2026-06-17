"""PSD / band power / IAF / regional on signals with known spectra."""

import unittest

import numpy as np

from neurobci.acquisition.default_montages import build_stream_info
from neurobci.config.schema import ChannelConfig
from neurobci.spectral import psd as P

SF = 500.0
N = 2000


def _info():
    return build_stream_info(ChannelConfig(), SF, "simulated")


def _sine_all(freq, n_channels, amp=10.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SF
    base = amp * np.sin(2 * np.pi * freq * t)
    data = np.tile(base[:, None], (1, n_channels))
    data += noise * rng.standard_normal((N, n_channels))
    return data


class TestPSD(unittest.TestCase):
    def setUp(self):
        self.info = _info()

    def test_band_power_peaks_in_correct_band(self):
        data = _sine_all(10.0, self.info.n_channels)      # alpha
        freqs, psd = P.compute_psd(data, SF)
        bp = P.band_powers(freqs, psd)
        ch0 = {b: bp[b][0] for b in bp}
        self.assertGreater(ch0["alpha"], ch0["theta"])
        self.assertGreater(ch0["alpha"], ch0["beta"])
        self.assertGreater(ch0["alpha"], ch0["delta"])

    def test_theta_signal(self):
        data = _sine_all(6.0, self.info.n_channels)
        freqs, psd = P.compute_psd(data, SF)
        bp = P.band_powers(freqs, psd)
        self.assertGreater(bp["theta"][0], bp["alpha"][0])

    def test_relative_power_fraction(self):
        data = _sine_all(10.0, self.info.n_channels, amp=15.0, noise=0.5)
        freqs, psd = P.compute_psd(data, SF)
        rel = P.relative_band_powers(freqs, psd)
        self.assertGreater(rel["alpha"][0], 0.5)
        self.assertLessEqual(rel["alpha"][0], 1.0)

    def test_individual_alpha_frequency(self):
        # 11 Hz alpha stronger over occipital channels.
        data = _sine_all(11.0, self.info.n_channels, amp=4.0, noise=1.0)
        for name in ("O1", "O2"):
            data[:, self.info.index_of(name)] += 20 * np.sin(
                2 * np.pi * 11.0 * np.arange(N) / SF)
        freqs, psd = P.compute_psd(data, SF)
        iaf = P.individual_alpha_frequency(freqs, psd, self.info)
        self.assertAlmostEqual(iaf, 11.0, delta=1.0)

    def test_regional_band_power(self):
        data = _sine_all(10.0, self.info.n_channels, amp=2.0, noise=1.0)
        # Boost occipital alpha.
        for name in ("O1", "O2"):
            data[:, self.info.index_of(name)] += 20 * np.sin(
                2 * np.pi * 10.0 * np.arange(N) / SF)
        freqs, psd = P.compute_psd(data, SF)
        bp = P.band_powers(freqs, psd)
        regional = P.regional_band_power(bp["alpha"], self.info)
        self.assertGreater(regional["occipital"], regional["frontal"])


if __name__ == "__main__":
    unittest.main()
