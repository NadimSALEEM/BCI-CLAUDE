"""Configurable artifact injection overlays."""

import unittest

import numpy as np
from scipy import signal

from neurobci.acquisition.artifacts_inject import (
    ArtifactInjectionConfig,
    ArtifactInjector,
)
from neurobci.acquisition.default_montages import build_stream_info
from neurobci.config.schema import ChannelConfig

SF = 500.0


def _info():
    return build_stream_info(ChannelConfig(), SF, "replay")


def _clean(n, nch):
    return np.zeros((n, nch), dtype=np.float32)


class TestArtifactInjection(unittest.TestCase):
    def test_disabled_is_noop(self):
        info = _info()
        inj = ArtifactInjector(info, ArtifactInjectionConfig(enabled=False))
        x = _clean(500, info.n_channels)
        np.testing.assert_array_equal(inj.inject(x, 0), x)

    def test_line_noise_added(self):
        info = _info()
        cfg = ArtifactInjectionConfig(enabled=True, line_uv=20.0, line_freq_hz=50.0)
        inj = ArtifactInjector(info, cfg)
        out = inj.inject(_clean(1000, info.n_channels), 0)
        f, p = signal.welch(out[:, 0], fs=SF, nperseg=512)
        peak = f[np.argmax(p)]
        self.assertAlmostEqual(peak, 50.0, delta=2.0)

    def test_flat_channel(self):
        info = _info()
        cfg = ArtifactInjectionConfig(enabled=True, flat_channels=["Cz"])
        inj = ArtifactInjector(info, cfg)
        x = 10 * np.ones((500, info.n_channels), dtype=np.float32)
        out = inj.inject(x, 0)
        self.assertLess(np.std(out[:, info.index_of("Cz")]), 2.0)

    def test_blinks_hit_eog(self):
        info = _info()
        cfg = ArtifactInjectionConfig(enabled=True, blink_rate_hz=5.0, blink_amp_uv=100.0)
        inj = ArtifactInjector(info, cfg, seed=1)
        out = inj.inject(_clean(2000, info.n_channels), 0)
        eog = out[:, info.eog_indices[0]]
        self.assertGreater(np.max(np.abs(eog)), 50.0)


if __name__ == "__main__":
    unittest.main()
