"""Run the whole validation suite: unit tests + headless smoke scripts.

    python scripts/run_all.py

Aggregates the stdlib unittest run, the engine smoke test and the offscreen
GUI smoke test into a single PASS/FAIL.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, args: list[str], env: dict | None = None) -> bool:
    print(f"\n=== {label} ===")
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(args, cwd=ROOT, env=full_env)
    ok = result.returncode == 0
    print(f"--- {label}: {'PASS' if ok else 'FAIL'} (exit {result.returncode}) ---")
    return ok


def main() -> int:
    py = sys.executable
    offscreen = {"QT_QPA_PLATFORM": "offscreen"}
    results = {
        "unit tests": _run("unit tests",
                           [py, "-m", "unittest", "discover", "-s", "tests", "-t", "."]),
        "engine smoke": _run("engine smoke", [py, "scripts/engine_smoke.py", "2"]),
        "gui smoke": _run("gui smoke", [py, "scripts/gui_smoke.py"], offscreen),
    }
    print("\n================ SUMMARY ================")
    for name, ok in results.items():
        print(f"  {name:<14} {'PASS' if ok else 'FAIL'}")
    overall = all(results.values())
    print("OVERALL:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
