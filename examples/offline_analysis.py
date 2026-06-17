"""Offline analysis: record a session, then preprocess, analyse and decode.

    python examples/offline_analysis.py

Generates a ground-truth P300 session, loads it back, applies the *offline*
(zero-phase) preprocessing pipeline, runs spectral analysis (band powers +
cognitive indices), and decodes the planted P300 -- demonstrating that the
same modules used live also work on recorded data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurobci.acquisition.default_montages import build_stream_info  # noqa: E402
from neurobci.bci.calibration import epochs_from_recording, train_and_select  # noqa: E402
from neurobci.config.schema import AppConfig, ChannelConfig         # noqa: E402
from neurobci.core.logging_setup import configure_logging           # noqa: E402
from neurobci.paradigms.base import get_paradigm                    # noqa: E402
from neurobci.preprocessing.pipeline import MODE_OFFLINE, Pipeline  # noqa: E402
from neurobci.recording.exporter import load_session               # noqa: E402
from neurobci.recording.synthetic_session import record_p300_session  # noqa: E402
from neurobci.spectral.analysis import SpectralAnalyzer             # noqa: E402


def main() -> int:
    configure_logging(level="WARNING", to_file=False)
    tmp = tempfile.mkdtemp(prefix="neurobci_example_")

    print("1) Recording a synthetic P300 session...")
    path = record_p300_session(tmp, n_stimuli=200, p300_amp_uv=12.0, seed=0)
    session = load_session(path)
    info = build_stream_info(ChannelConfig(), session.sfreq, "replay")
    print(f"   {session.n_samples} samples, {len(session.markers)} markers")

    print("\n2) Offline preprocessing (zero-phase) on the first 10 s...")
    cfg = AppConfig()
    pipe = Pipeline.from_config(
        cfg.preprocessing, session.sfreq,
        session.channel_kinds, session.channel_names,
    )
    window = session.data[: int(10 * session.sfreq)]
    clean = pipe.apply_window(window, mode=MODE_OFFLINE)
    print(f"   processed window shape {clean.shape}")

    print("\n3) Spectral analysis...")
    analyzer = SpectralAnalyzer(info, cfg.spectral)
    report = analyzer.analyze(window)
    print(f"   individual alpha frequency: {report.iaf:.1f} Hz")
    for ix in report.indices:
        print(f"   {ix.name:11s} = {ix.value:.3f}  ({ix.definition})")

    print("\n4) Decode the planted P300...")
    para = get_paradigm("p300")
    es = epochs_from_recording(session.data, session.timestamps, session.markers,
                               session.sfreq, para, {"target": 1, "nontarget": 0})
    res = train_and_select(para, es.X, es.y, session.channel_names,
                           session.channel_kinds, session.sfreq,
                           model_names=["vec_lda"], n_folds=4)
    print("   " + res.results[res.best_model_name].summary())
    print("   USABLE" if res.is_usable else "   NOT USABLE")
    print("\n(Exploratory indices are proxies requiring validation - see docs.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
