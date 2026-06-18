"""Import XDF / FIF recordings as replay sources and re-publish them.

These cover the file-import path that lets a ``.xdf`` (LabRecorder/LSL) or
``.fif`` (MNE/BIDS) recording drive replay and virtual-LSL republication
exactly like a native session directory.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurobci.acquisition.replay_source import ReplaySource
from neurobci.config.schema import AppConfig
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, KIND_MISC, StreamInfo
from neurobci.recording.exporter import export_fif, load_session
from neurobci.recording.external import (
    _session_from_xdf_streams,
    load_fif,
    load_session_any,
)
from neurobci.recording.writer import SessionRecorder

# Real recordings shipped in the repo (skipped gracefully if absent).
_XDF_SAMPLE = Path(__file__).resolve().parents[1] / (
    "rien/physiodata_n/sub-P025/ses-S001/eeg/"
    "sub-P025_ses-S001_task-oddball_run-001_eeg.xdf"
)


class _TmpMixin(unittest.TestCase):
    def tmpdir(self) -> str:
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d


def _make_session(d, n=400, nch=4):
    info = StreamInfo("rec", 100.0, [f"c{i}" for i in range(nch)],
                      [KIND_EEG] * (nch - 1) + [KIND_EOG])
    rec = SessionRecorder(d, info, AppConfig(), participant_id="t")
    rec.start()
    data = (np.arange(n * nch, dtype=np.float32).reshape(n, nch)) * 0.01
    ts = np.arange(n) / 100.0
    rec.write(data, ts)
    rec.push_marker("a", timestamp=ts[50], sample=50)
    rec.push_marker("b", timestamp=ts[200], sample=200)
    rec.stop()
    return rec.path, data


def _fake_xdf_streams(sfreq=100.0, n=300, nch=4, t0=1000.0):
    """A minimal two-stream (EEG + Markers) structure, like pyxdf output."""
    ts = t0 + np.arange(n) / sfreq
    data_uv = (np.arange(n * nch, dtype=np.float32).reshape(n, nch)) * 0.1
    data_v = data_uv * 1e-6  # stored in volts -> loader must rescale to uV
    channels = [
        {"label": [f"c{i}"], "type": ["EEG" if i < nch - 1 else "EOG"],
         "unit": ["V"]}
        for i in range(nch)
    ]
    eeg = {
        "info": {
            "name": ["EEG"], "type": ["EEG"], "channel_count": [str(nch)],
            "nominal_srate": [str(sfreq)], "channel_format": ["float32"],
            "desc": [{"channels": [{"channel": channels}]}],
        },
        "time_series": data_v.astype(np.float32),
        "time_stamps": ts.astype(np.float64),
    }
    markers = {
        "info": {
            "name": ["events"], "type": ["Markers"], "channel_count": ["1"],
            "nominal_srate": ["0.0"], "channel_format": ["string"],
            "desc": [None],
        },
        "time_series": [["go"], ["stop"]],
        "time_stamps": np.array([ts[50], ts[200]], dtype=np.float64),
    }
    return [markers, eeg], data_uv


class TestFifImport(_TmpMixin):
    def test_fif_roundtrip_and_replay(self):
        path, data = _make_session(self.tmpdir())
        session = load_session(path)
        fif = export_fif(session, Path(self.tmpdir()) / "imported_raw.fif")

        loaded = load_fif(fif)
        self.assertEqual(loaded.channel_names, session.channel_names)
        self.assertEqual(loaded.channel_kinds, session.channel_kinds)  # EOG kept
        self.assertAlmostEqual(loaded.sfreq, session.sfreq)
        self.assertEqual(loaded.meta["imported_from"], "fif")
        self.assertEqual({m["label"] for m in loaded.markers}, {"a", "b"})
        self.assertEqual(sorted(m["sample"] for m in loaded.markers), [50, 200])

        # Round-trips microvolts -> volts -> microvolts within float tolerance.
        replayed, markers = ReplaySource(loaded, realtime=False).read_all()
        np.testing.assert_allclose(replayed, data, rtol=1e-4, atol=1e-2)
        self.assertEqual(len(markers), 2)

    def test_dispatch_by_extension(self):
        path, _ = _make_session(self.tmpdir())
        fif = export_fif(load_session(path), Path(self.tmpdir()) / "s_raw.fif")
        # Directory -> native loader; .fif -> fif loader.
        self.assertEqual(load_session_any(path).meta.get("imported_from"), None)
        self.assertEqual(load_session_any(fif).meta["imported_from"], "fif")


class TestXdfImport(_TmpMixin):
    def test_parses_streams_scales_and_maps_markers(self):
        streams, data_uv = _fake_xdf_streams()
        session = _session_from_xdf_streams(streams, Path("rec.xdf"))

        self.assertEqual(session.meta["imported_from"], "xdf")
        self.assertEqual(session.meta["stream_name"], "EEG")
        self.assertEqual(session.channel_kinds[-1], KIND_EOG)
        self.assertEqual(session.meta["units"], "uV")
        self.assertAlmostEqual(session.sfreq, 100.0)
        # Volts were rescaled to microvolts.
        np.testing.assert_allclose(session.data, data_uv, rtol=1e-4, atol=1e-3)
        # String stream mapped onto EEG sample indices by timestamp.
        self.assertEqual([(m["label"], m["sample"]) for m in session.markers],
                         [("go", 50), ("stop", 200)])

        replayed, markers = ReplaySource(session, realtime=False).read_all()
        np.testing.assert_allclose(replayed, data_uv, rtol=1e-4, atol=1e-3)
        self.assertEqual(len(markers), 2)

    def test_auto_detects_montage_from_blanket_eeg_types(self):
        # A common real-world case: every channel typed "EEG", but names
        # reveal an ocular and a cardiac lead -> classify by name on import.
        streams, _ = _fake_xdf_streams(nch=4)
        chans = streams[1]["info"]["desc"][0]["channels"][0]["channel"]
        chans[0]["label"] = ["Fp1"]
        chans[1]["label"] = ["Cz"]
        chans[2]["label"] = ["EOG"]
        chans[3]["label"] = ["ECG"]
        for c in chans:
            c["type"] = ["EEG"]  # blanket type
        session = _session_from_xdf_streams(streams, Path("rec.xdf"))
        self.assertEqual(session.channel_kinds,
                         [KIND_EEG, KIND_EEG, KIND_EOG, KIND_MISC])

    def test_picks_widest_eeg_stream_and_estimates_srate(self):
        # nominal_srate 0 must be recovered from the timestamps.
        streams, _ = _fake_xdf_streams(sfreq=128.0)
        streams[1]["info"]["nominal_srate"] = ["0.0"]
        session = _session_from_xdf_streams(streams, Path("rec.xdf"))
        self.assertAlmostEqual(session.sfreq, 128.0, places=3)

    @unittest.skipUnless(_XDF_SAMPLE.exists(), "sample .xdf not present")
    def test_real_sample_file_loads_and_publishes(self):
        from neurobci.acquisition.lsl_publisher import LSLPublisher

        session = load_session_any(_XDF_SAMPLE)
        self.assertEqual(session.meta["imported_from"], "xdf")
        self.assertGreater(session.n_samples, 0)
        self.assertEqual(session.meta["participant_id"], "P025")
        self.assertEqual(len(session.channel_names), session.data.shape[1])
        self.assertGreater(len(session.markers), 0)

        # The imported session drives a virtual LSL publisher like any source.
        try:
            import pylsl  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"pylsl unavailable: {exc}")
        pub = LSLPublisher(ReplaySource(session, loop=True),
                           stream_name="NeuroBCI-Test")
        try:
            pub.start()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"LSL outlet unavailable: {exc}")
        finally:
            pub.stop()


class TestDispatchErrors(_TmpMixin):
    def test_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            load_session_any(Path(self.tmpdir()) / "nope.xdf")

    def test_unsupported_extension(self):
        bad = Path(self.tmpdir()) / "data.csv"
        bad.write_text("not a recording")
        with self.assertRaises(ValueError):
            load_session_any(bad)


if __name__ == "__main__":
    unittest.main()
