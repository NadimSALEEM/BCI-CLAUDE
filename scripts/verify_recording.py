"""Check whether a recorded session is trustworthy.

    python scripts/verify_recording.py <session_dir> [--window 4.0]
                                       [--preprocess [profile.json]]

Loads the session and prints a blunt verdict -- TRUST / CAUTION /
UNTRUSTWORTHY -- with the reasons. With ``--preprocess`` it also runs the
(offline, zero-phase) preprocessing pipeline first and verifies the cleaned
data, so you can see whether preprocessing actually rescued the recording.

This is the honest "did my Enobio data work?" check. It never alters or
re-saves your recording.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurobci.config.manager import ConfigManager                    # noqa: E402
from neurobci.config.schema import AppConfig                         # noqa: E402
from neurobci.preprocessing.pipeline import MODE_OFFLINE, Pipeline   # noqa: E402
from neurobci.quality.verify import verify_recording                 # noqa: E402
from neurobci.recording.exporter import load_session                 # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    session_dir = Path(argv[1])
    window_s = 4.0
    if "--window" in argv:
        window_s = float(argv[argv.index("--window") + 1])

    session = load_session(session_dir)
    print(f"Loaded {session.n_samples:,} samples @ {session.sfreq:.0f} Hz, "
          f"{len(session.channel_names)} channels, {len(session.markers)} markers.\n")

    print("=== RAW DATA ===")
    raw_report = verify_recording(
        session.data, session.channel_names, session.channel_kinds,
        session.sfreq, window_s=window_s,
    )
    print(raw_report.summary())

    if "--preprocess" in argv:
        i = argv.index("--preprocess")
        profile = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else None
        cfg = ConfigManager().load(profile) if profile else AppConfig()
        pipe = Pipeline.from_config(
            cfg.preprocessing, session.sfreq,
            session.channel_kinds, session.channel_names,
        )
        clean = pipe.apply_window(session.data, mode=MODE_OFFLINE)
        print("\n=== AFTER OFFLINE PREPROCESSING ===")
        if pipe.requires_fit and not pipe.fitted:
            print("(note: ICA/ASR/bad-channel stages are present but were not "
                  "calibrated here; they passed through.)")
        clean_report = verify_recording(
            clean, session.channel_names, session.channel_kinds,
            session.sfreq, window_s=window_s,
        )
        print(clean_report.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
