"""Export a recorded session to MNE .fif and/or .npz.

    python scripts/export_session.py <session_dir> [--fif] [--npz]

With no format flag, both are written next to the session directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurobci.recording.exporter import export_fif, export_npz, load_session  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    session_dir = Path(argv[1])
    want_fif = "--fif" in argv
    want_npz = "--npz" in argv
    if not want_fif and not want_npz:
        want_fif = want_npz = True

    session = load_session(session_dir)
    print(f"Loaded {session.n_samples} samples, {len(session.markers)} markers.")
    if want_fif:
        out = export_fif(session, session_dir.parent / f"{session_dir.name}_raw.fif")
        print(f"Wrote {out}")
    if want_npz:
        out = export_npz(session, session_dir.parent / f"{session_dir.name}.npz")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
