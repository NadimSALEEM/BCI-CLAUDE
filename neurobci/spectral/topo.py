"""Topographic layout and scalp interpolation.

Uses a standard normalised 10-20 layout (nose up, unit head radius) so a
per-channel value vector can be rendered as a scalp map. Channels without a
known position (e.g. EOG) or flagged as bad are simply not used as
interpolation anchors -- artifact-aware by construction.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

# Normalised 10-20 positions: x = left(-)/right(+), y = back(-)/front(+),
# unit head radius. Approximate but standard topographic layout.
POS_1020: dict[str, tuple[float, float]] = {
    "Fp1": (-0.27, 0.91), "Fp2": (0.27, 0.91),
    "F7": (-0.81, 0.59), "F3": (-0.40, 0.67), "Fz": (0.0, 0.72),
    "F4": (0.40, 0.67), "F8": (0.81, 0.59),
    "T7": (-1.0, 0.0), "C3": (-0.5, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.5, 0.0), "T8": (1.0, 0.0),
    "P7": (-0.81, -0.59), "P3": (-0.40, -0.67), "Pz": (0.0, -0.72),
    "P4": (0.40, -0.67), "P8": (0.81, -0.59),
    "O1": (-0.27, -0.91), "O2": (0.27, -0.91),
    # Common aliases.
    "T3": (-1.0, 0.0), "T4": (1.0, 0.0), "T5": (-0.81, -0.59), "T6": (0.81, -0.59),
}


def channel_positions_2d(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(positions (n,2), found_mask (n,))`` for the given channels."""
    pos = np.full((len(names), 2), np.nan)
    found = np.zeros(len(names), dtype=bool)
    for i, n in enumerate(names):
        key = n if n in POS_1020 else n.capitalize()
        if key in POS_1020:
            pos[i] = POS_1020[key]
            found[i] = True
    return pos, found


def interpolate_topomap(
    values: np.ndarray, positions: np.ndarray, found: np.ndarray, res: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate channel values onto a head-disc grid.

    Returns ``(grid_x, grid_y, grid_z)`` where ``grid_z`` is NaN outside the
    unit head circle. Only channels with a known position and a finite value
    are used as anchors.
    """
    use = found & np.isfinite(values)
    gx, gy = np.meshgrid(np.linspace(-1.1, 1.1, res), np.linspace(-1.1, 1.1, res))
    if use.sum() < 3:
        return gx, gy, np.full_like(gx, np.nan)

    pts = positions[use]
    vals = np.asarray(values)[use]
    grid = griddata(pts, vals, (gx, gy), method="cubic")
    # Fill cubic NaNs with nearest, then mask outside the head circle.
    nan = np.isnan(grid)
    if nan.any():
        grid[nan] = griddata(pts, vals, (gx[nan], gy[nan]), method="nearest")
    grid[(gx**2 + gy**2) > 1.05**2] = np.nan
    return gx, gy, grid
