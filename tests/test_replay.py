"""Replay source: determinism, transport, markers, and decode regression."""

import shutil
import tempfile
import unittest

import numpy as np

from neurobci.acquisition.replay_source import ReplaySource
from neurobci.bci.calibration import epochs_from_recording, train_and_select
from neurobci.config.schema import AppConfig
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo
from neurobci.paradigms.p300 import P300Paradigm
from neurobci.recording.exporter import load_session
from neurobci.recording.synthetic_session import record_p300_session
from neurobci.recording.writer import SessionRecorder


class _TmpMixin(unittest.TestCase):
    def tmpdir(self) -> str:
        # mkdtemp + ignore_errors cleanup avoids a Windows race where the OS
        # briefly holds a just-closed file (handles are closed by stop()).
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d


def _make_session(d, n=400, nch=4):
    info = StreamInfo("rec", 100.0, [f"c{i}" for i in range(nch)],
                      [KIND_EEG] * (nch - 1) + [KIND_EOG])
    rec = SessionRecorder(d, info, AppConfig(), participant_id="t")
    rec.start()
    data = np.arange(n * nch, dtype=np.float32).reshape(n, nch)
    ts = np.arange(n) / 100.0
    rec.write(data, ts)
    m1, m2 = n // 8, n // 2
    rec.push_marker("a", timestamp=ts[m1], sample=m1)
    rec.push_marker("b", timestamp=ts[m2], sample=m2)
    rec.stop()
    return rec.path, data


class TestReplay(_TmpMixin):
    def test_read_all_matches_recording(self):
        path, data = _make_session(self.tmpdir())
        rs = ReplaySource(load_session(path), realtime=False)
        replayed, markers = rs.read_all()
        np.testing.assert_array_equal(replayed, data)
        self.assertEqual(len(markers), 2)
        self.assertEqual({m["label"] for m in markers}, {"a", "b"})
        self.assertTrue(rs.finished)

    def test_seek_and_position(self):
        path, data = _make_session(self.tmpdir())
        rs = ReplaySource(load_session(path), realtime=False)
        rs.start()
        rs.seek(2.0)                       # 2 s @ 100 Hz -> sample 200
        self.assertAlmostEqual(rs.position_s, 2.0, places=2)
        chunk, _ = rs._emit(10)
        np.testing.assert_array_equal(chunk[0], data[200])

    def test_loop_does_not_finish(self):
        path, _ = _make_session(self.tmpdir(), n=50)
        rs = ReplaySource(load_session(path), realtime=False, loop=True)
        rs.start()
        total = 0
        for _ in range(20):
            c, _ = rs._emit(10)
            total += c.shape[0]
        self.assertFalse(rs.finished)
        self.assertGreater(total, 50)      # wrapped around

    def test_pause_emits_nothing(self):
        path, _ = _make_session(self.tmpdir())
        rs = ReplaySource(load_session(path), realtime=True)
        rs.start()
        rs.pause()
        data, _ = rs.read()
        self.assertEqual(data.shape[0], 0)

    def test_info_is_replay_kind(self):
        path, _ = _make_session(self.tmpdir())
        rs = ReplaySource(load_session(path))
        self.assertEqual(rs.info.source_kind, "replay")


class TestReplayRegression(_TmpMixin):
    def test_simulate_record_replay_decode(self):
        path = record_p300_session(self.tmpdir(), n_stimuli=200,
                                   p300_amp_uv=12.0, seed=0)
        session = load_session(path)

        # Replay reproduces the recording exactly.
        rs = ReplaySource(session, realtime=False)
        replayed, markers = rs.read_all()
        np.testing.assert_allclose(replayed, session.data, rtol=1e-5)
        self.assertEqual(len(markers), 200)

        # Decode the planted P300 from the (replayed == recorded) data.
        para = P300Paradigm()
        es = epochs_from_recording(
            session.data, session.timestamps, session.markers,
            session.sfreq, para, {"target": 1, "nontarget": 0})
        self.assertGreater(es.n_epochs, 150)
        res = train_and_select(
            para, es.X, es.y, session.channel_names, session.channel_kinds,
            session.sfreq, model_names=["vec_lda"], n_folds=4, seed=0)
        self.assertTrue(res.is_usable, "planted P300 not recovered above chance")


if __name__ == "__main__":
    unittest.main()
