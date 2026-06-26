"""End-to-end (headless) acquisition: simulated source -> buffer -> state."""

import tempfile
import time
import unittest

import numpy as np

from neurobci.acquisition.engine import AcquisitionEngine
from neurobci.config.schema import AppConfig
from neurobci.core.app_state import ConnectionStatus, OperatingMode
from neurobci.preprocessing.stages import make_stage
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

    def test_processed_buffer_is_filled_and_differs_from_raw(self):
        cfg = AppConfig()
        cfg.acquisition.source_type = "simulated"
        cfg.acquisition.buffer_seconds = 5.0
        engine = AcquisitionEngine(cfg)
        engine.start()
        try:
            self.assertTrue(self._wait_for(
                lambda: engine.processed_buffer is not None
                and engine.processed_buffer.total_written > 600
            ))
            raw, _ = engine.latest_seconds(1.0)
            proc, _ = engine.latest_processed_seconds(1.0)
            # Aligned length, same channel count.
            self.assertEqual(proc.shape[1], raw.shape[1])
            self.assertGreater(proc.shape[0], 0)
            # The default pipeline (high-pass + CAR + ...) must actually change
            # the signal: processed != raw.
            n = min(raw.shape[0], proc.shape[0])
            self.assertFalse(np.allclose(raw[-n:], proc[-n:]))
            # CAR makes the per-sample EEG mean ~0 in the processed stream.
            eeg = engine.stream_info.eeg_indices
            self.assertLess(np.abs(proc[-n:][:, eeg].mean(axis=1)).max(), 1.0)
        finally:
            engine.stop()
        # After stop the processed buffer is released.
        self.assertIsNone(engine.processed_buffer)

    def test_rebuild_preprocessing_takes_effect_live(self):
        cfg = AppConfig()
        cfg.acquisition.source_type = "simulated"
        engine = AcquisitionEngine(cfg)
        engine.start()
        try:
            self.assertTrue(self._wait_for(
                lambda: engine.processed_buffer is not None
                and engine.processed_buffer.total_written > 300))
            # Replace the pipeline with a single hard clamp and confirm the
            # live processed stream respects it.
            with engine.pipeline_lock:
                engine.pipeline.stages = [
                    make_stage({"type": "clamp", "params": {"limit_uv": 5.0}})
                ]
                engine.pipeline.stages[0].prepare(
                    engine.pipeline.sfreq, engine.pipeline.ch_kinds,
                    engine.pipeline.ch_names)
                engine.pipeline.reset()
            base = engine.processed_buffer.total_written
            self.assertTrue(self._wait_for(
                lambda: engine.processed_buffer.total_written > base + 300))
            proc, _ = engine.latest_processed_seconds(0.5)
            self.assertLessEqual(np.abs(proc).max(), 5.0 + 1e-4)
        finally:
            engine.stop()

    def test_calibrate_artifacts_fits_pipeline_stages(self):
        cfg = AppConfig()
        cfg.acquisition.source_type = "simulated"
        cfg.preprocessing.stages = [
            {"type": "highpass", "enabled": True, "params": {"cutoff_hz": 0.5}},
            {"type": "interpolate_bad", "enabled": True, "params": {}},
            {"type": "ica", "enabled": True, "params": {"max_remove": 1}},
        ]
        engine = AcquisitionEngine(cfg)
        engine.start()
        try:
            self.assertTrue(engine.pipeline.requires_fit)
            self.assertFalse(engine.pipeline.fitted)
            self.assertTrue(self._wait_for(
                lambda: engine.buffer is not None
                and engine.buffer.total_written > int(2 * cfg.acquisition.expected_sfreq)))
            summaries = engine.calibrate_artifacts(seconds=3.0)
            self.assertTrue(any("interpolate_bad" in s for s in summaries))
            self.assertTrue(any("ica" in s for s in summaries))
            self.assertTrue(engine.pipeline.fitted)
        finally:
            engine.stop()

    def test_apply_channel_selection_narrows_live_stream(self):
        cfg = AppConfig()
        cfg.acquisition.source_type = "simulated"
        cfg.acquisition.buffer_seconds = 5.0
        engine = AcquisitionEngine(cfg)
        engine.start()
        try:
            self.assertTrue(self._wait_for(
                lambda: engine.buffer is not None and engine.buffer.total_written > 100))
            native = engine.native_stream_info
            keep = list(range(native.n_channels - 3))      # drop the last three
            names = [native.channel_names[i] for i in keep]
            kinds = [native.channel_kinds[i] for i in keep]
            info = engine.apply_channel_selection(keep, names, kinds)

            self.assertEqual(info.n_channels, len(keep))
            self.assertEqual(engine.stream_info.channel_names, names)
            self.assertEqual(engine.keep_indices, keep)
            # The native source is unchanged; only the active montage narrows.
            self.assertEqual(engine.native_stream_info.n_channels, native.n_channels)
            # New (reduced-width) buffers fill and read back at the kept width.
            self.assertTrue(self._wait_for(
                lambda: engine.buffer.total_written > 200))
            data, _ = engine.latest_seconds(0.5)
            self.assertEqual(data.shape[1], len(keep))
            proc, _ = engine.latest_processed_seconds(0.5)
            self.assertEqual(proc.shape[1], len(keep))
        finally:
            engine.stop()

    def test_channel_selection_refused_while_recording(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = AppConfig()
            cfg.recording.directory = d
            engine = AcquisitionEngine(cfg)
            engine.start()
            try:
                self.assertTrue(self._wait_for(
                    lambda: engine.buffer and engine.buffer.total_written > 50))
                engine.start_recording(participant_id="x")
                native = engine.native_stream_info
                keep = list(range(native.n_channels - 2))
                with self.assertRaises(RuntimeError):
                    engine.apply_channel_selection(
                        keep,
                        [native.channel_names[i] for i in keep],
                        [native.channel_kinds[i] for i in keep],
                    )
            finally:
                engine.stop()

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
