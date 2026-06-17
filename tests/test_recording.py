"""Recording round-trip: write -> read back -> export, incl. crash recovery."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurobci.config.schema import AppConfig
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo
from neurobci.recording.exporter import load_session, to_mne_raw
from neurobci.recording.writer import EEG_FILE, SessionRecorder


def _info():
    return StreamInfo(
        name="sim",
        sfreq=100.0,
        channel_names=["F3", "F4", "EOG"],
        channel_kinds=[KIND_EEG, KIND_EEG, KIND_EOG],
        source_kind="simulated",
    )


class TestRecording(unittest.TestCase):
    def _record(self, root, chunks, markers=()):
        rec = SessionRecorder(root, _info(), AppConfig(), participant_id="p01")
        rec.start()
        sample = 0
        for ch in chunks:
            ts = np.arange(sample, sample + ch.shape[0], dtype=np.float64) / 100.0
            rec.write(ch, ts)
            sample += ch.shape[0]
        for label, s in markers:
            rec.push_marker(label, timestamp=s / 100.0, sample=s)
        return rec

    def test_roundtrip_preserves_data(self):
        with tempfile.TemporaryDirectory() as d:
            c1 = np.random.default_rng(0).standard_normal((50, 3)).astype(np.float32)
            c2 = np.random.default_rng(1).standard_normal((30, 3)).astype(np.float32)
            rec = self._record(d, [c1, c2], markers=[("target", 10), ("nontarget", 60)])
            rec.stop()

            loaded = load_session(rec.path)
            self.assertEqual(loaded.n_samples, 80)
            self.assertEqual(loaded.data.shape, (80, 3))
            np.testing.assert_allclose(loaded.data[:50], c1, rtol=1e-6)
            np.testing.assert_allclose(loaded.data[50:], c2, rtol=1e-6)
            self.assertEqual(len(loaded.markers), 2)
            self.assertEqual(loaded.markers[0]["label"], "target")

    def test_metadata_finalised(self):
        with tempfile.TemporaryDirectory() as d:
            c = np.zeros((40, 3), dtype=np.float32)
            rec = self._record(d, [c])
            rec.stop()
            meta = json.loads((rec.path / "metadata.json").read_text())
            self.assertTrue(meta["finalised"])
            self.assertEqual(meta["n_samples"], 40)
            self.assertEqual(meta["channel_kinds"][2], KIND_EOG)
            self.assertIn("config", meta)  # full reproducibility snapshot

    def test_crash_recovery_from_file_size(self):
        # Simulate a crash: never call stop(), so metadata stays unfinalised.
        with tempfile.TemporaryDirectory() as d:
            c = np.random.default_rng(2).standard_normal((25, 3)).astype(np.float32)
            rec = self._record(d, [c])
            try:
                meta = json.loads((rec.path / "metadata.json").read_text())
                self.assertFalse(meta["finalised"])
                self.assertNotIn("n_samples", meta)
                # Loader must recover sample count from the .f32 file size.
                loaded = load_session(rec.path)
                self.assertEqual(loaded.n_samples, 25)
                size = (rec.path / EEG_FILE).stat().st_size
                self.assertEqual(size, 25 * 3 * 4)
            finally:
                # A real crash would release handles by terminating the
                # process; here we close them so the temp dir can be removed
                # on Windows (open files can't be unlinked).
                for f in (rec._eeg_f, rec._ts_f, rec._markers_f):
                    f.close()

    def test_mne_export(self):
        with tempfile.TemporaryDirectory() as d:
            c = (10 * np.random.default_rng(3).standard_normal((200, 3))).astype(np.float32)
            rec = self._record(d, [c], markers=[("evt", 100)])
            rec.stop()
            loaded = load_session(rec.path)
            raw = to_mne_raw(loaded)
            self.assertEqual(raw.info["sfreq"], 100.0)
            self.assertEqual(len(raw.ch_names), 3)
            # EOG channel typed correctly.
            self.assertEqual(raw.get_channel_types()[2], "eog")
            # Microvolt -> volt scaling applied.
            self.assertLess(np.abs(raw.get_data()).max(), 1e-2)
            self.assertEqual(len(raw.annotations), 1)


if __name__ == "__main__":
    unittest.main()
