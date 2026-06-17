"""Topographic layout/interpolation and the SpectralAnalyzer."""

import unittest

import numpy as np

from neurobci.acquisition.default_montages import build_stream_info
from neurobci.config.schema import ChannelConfig, SpectralConfig
from neurobci.spectral.analysis import SpectralAnalyzer
from neurobci.spectral.topo import channel_positions_2d, interpolate_topomap

SF = 500.0


def _info():
    return build_stream_info(ChannelConfig(), SF, "simulated")


class TestTopo(unittest.TestCase):
    def test_positions_found_for_eeg(self):
        info = _info()
        pos, found = channel_positions_2d(info.channel_names)
        # All 19 scalp channels have positions; EOG does not.
        self.assertEqual(found.sum(), 19)
        self.assertFalse(found[info.index_of("EOG")])
        # Positions lie within the head disc.
        r = np.hypot(pos[found][:, 0], pos[found][:, 1])
        self.assertTrue(np.all(r <= 1.01))

    def test_interpolation_shape_and_mask(self):
        info = _info()
        pos, found = channel_positions_2d(info.channel_names)
        values = np.arange(info.n_channels, dtype=float)
        gx, gy, gz = interpolate_topomap(values, pos, found, res=40)
        self.assertEqual(gz.shape, (40, 40))
        # Outside the head circle is NaN; somewhere inside is finite.
        self.assertTrue(np.isnan(gz[0, 0]))
        self.assertTrue(np.isfinite(gz[20, 20]))


class TestAnalyzer(unittest.TestCase):
    def _data(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        info = _info()
        t = np.arange(n) / SF
        base = 10 * np.sin(2 * np.pi * 10 * t)
        data = np.tile(base[:, None], (1, info.n_channels))
        data += rng.standard_normal((n, info.n_channels))
        return data, info

    def test_report_fields(self):
        data, info = self._data()
        an = SpectralAnalyzer(info, SpectralConfig())
        rep = an.analyze(data)
        self.assertIn("alpha", rep.abs_powers)
        self.assertEqual(rep.abs_powers["alpha"].shape[0], info.n_channels)
        self.assertEqual(len(rep.indices), 3)
        self.assertTrue(np.isfinite(rep.iaf))

    def test_bad_channel_excluded(self):
        data, info = self._data()
        an = SpectralAnalyzer(info, SpectralConfig())
        rep = an.analyze(data, bad_names=["Cz"])
        self.assertIn(info.index_of("Cz"), rep.bad_channels)
        vals = an.topomap_values(rep, "alpha", "absolute")
        self.assertTrue(np.isnan(vals[info.index_of("Cz")]))

    def test_baseline_change_near_zero_db(self):
        data, info = self._data()
        an = SpectralAnalyzer(info, SpectralConfig())
        rep = an.analyze(data)
        an.set_baseline(rep)
        self.assertTrue(an.has_baseline)
        rep2 = an.analyze(data)  # same data -> ~0 dB change
        vals = an.topomap_values(rep2, "alpha", "baseline")
        finite = vals[np.isfinite(vals)]
        self.assertLess(np.nanmax(np.abs(finite)), 1.0)

    def test_history_accumulates(self):
        data, info = self._data()
        an = SpectralAnalyzer(info, SpectralConfig())
        for _ in range(3):
            an.analyze(data)
        t, v = an.history_series("indices", "engagement")
        self.assertEqual(len(t), 3)


if __name__ == "__main__":
    unittest.main()
