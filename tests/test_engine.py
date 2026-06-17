"""End-to-end (headless) acquisition: simulated source -> buffer -> state."""

import tempfile
import time
import unittest

from neurobci.acquisition.engine import AcquisitionEngine
from neurobci.config.schema import AppConfig
from neurobci.core.app_state import ConnectionStatus, OperatingMode
from neurobci.recording.exporter import load_session


class TestAcquisitionEngine(unittest.TestCase):
    def _wait_for(self, predicate, timeout=3.0, interval=0.02):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return False

    def test_simulated_pipeline(self):
        cfg = AppConfig()
        cfg.acquisition.source_type = "simulated"
        cfg.acquisition.buffer_seconds = 5.0
        engine = AcquisitionEngine(cfg)
        engine.start()
        try:
            got_data = self._wait_for(
                lambda: engine.buffer is not None and engine.buffer.total_written > 100
            )
            self.assertTrue(got_data, "no samples flowed within timeout")

            snap = engine.state.snapshot()
            self.assertEqual(snap.mode, OperatingMode.SIMULATION)
            self.assertEqual(snap.connection, ConnectionStatus.SIMULATED)
            self.assertIsNotNone(snap.stream_info)
            self.assertEqual(snap.stream_info.n_channels, 20)

            data, ts = engine.latest_seconds(1.0)
            self.assertGreater(data.shape[0], 0)
            self.assertEqual(data.shape[1], 20)
            # Effective rate should be in a sane neighbourhood of 500 Hz.
            self.assertGreater(snap.measured_sfreq, 100.0)
        finally:
            engine.stop()

        self.assertFalse(engine.running)
        self.assertEqual(engine.state.snapshot().mode, OperatingMode.IDLE)

    def test_recording_integration(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = AppConfig()
            cfg.recording.directory = d
            engine = AcquisitionEngine(cfg)
            engine.start()
            try:
                self.assertTrue(
                    self._wait_for(
                        lambda: engine.buffer and engine.buffer.total_written > 50
                    )
                )
                rec = engine.start_recording(participant_id="subjX", notes="unit test")
                self.assertTrue(engine.is_recording)
                self.assertTrue(engine.state.snapshot().recording)
                engine.push_marker("hello")
                self.assertTrue(self._wait_for(lambda: rec.n_samples > 50))
                engine.push_marker("world")
                stats = engine.stop_recording()
            finally:
                engine.stop()

            self.assertFalse(engine.is_recording)
            self.assertGreater(stats.n_samples, 50)
            loaded = load_session(rec.path)
            self.assertEqual(loaded.data.shape[1], 20)
            self.assertGreater(loaded.n_samples, 50)
            self.assertEqual(len(loaded.markers), 2)
            self.assertEqual(loaded.meta["participant_id"], "subjX")


if __name__ == "__main__":
    unittest.main()
