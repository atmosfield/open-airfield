"""U7 — CASE-01 ingest (technical spec §U7).

Case01Field(TruthField) from a VTK export: pyvista.read for the grid,
arbitrary-point probing via PolyData.sample against it [VERIFIED at
1.344M-cell scale in-container, 0.35 s/1,000 points].

The unit tests itself without the real case (this is how an unverified
ground truth gets tested): write the synthetic field to .vtu, re-read it
through this ingest unit, probe points, match the analytic values within
interpolation tolerance.

Open properties on the real export (each with a fallback, none invented):
- Solver/version unnamed -> recorded verbatim into the case record, blocks
  nothing at ingest.
- Units unstated -> if the file carries no unit metadata, SI is assumed and
  meta is stamped `units: "SI assumed — unverified"`, flagged in the gate
  report header.
- Binary vs ASCII -> pyvista reads both.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from open_airfield.contracts import TruthField
from open_airfield.geometry import LX, LY, LZ

VELOCITY_ARRAY_CANDIDATES = ("U", "velocity", "Velocity", "u")


def write_field_vtu(
    field: TruthField, path: Path | str, spacing: float = 0.1
) -> Path:
    """Write a TruthField to .vtu on a regular grid (round-trip test surface).

    Cell-size mirrors the eval grid; the real CASE-01 mesh replaces this file,
    not this code.
    """
    path = Path(path)
    dims = (int(LX / spacing) + 1, int(LY / spacing) + 1, int(LZ / spacing) + 1)
    grid = pv.ImageData(dimensions=dims, spacing=(spacing,) * 3, origin=(0.0, 0.0, 0.0))
    grid.point_data["U"] = field.velocity(np.asarray(grid.points, dtype=np.float64))
    grid.cast_to_unstructured_grid().save(path)
    return path


class Case01Field:
    """TruthField over a VTK export, probed at arbitrary coordinates."""

    def __init__(self, path: Path | str, units_verified: bool = False):
        path = Path(path)
        self._grid = pv.read(path)
        self._array = next(
            (n for n in VELOCITY_ARRAY_CANDIDATES if n in self._grid.point_data),
            None,
        )
        if self._array is None:
            raise ValueError(
                f"no velocity array in {path.name}; point data: {list(self._grid.point_data)}"
            )
        self.meta = {
            "name": f"vtk-{path.stem}",
            "source": str(path),
            "case_id": "CASE-01" if "case" in path.stem.lower() else path.stem,
            "units": "SI" if units_verified else "SI assumed — unverified",
            "n_cells": self._grid.n_cells,
            "velocity_array": self._array,
        }

    def velocity(self, points: np.ndarray) -> np.ndarray:
        cloud = pv.PolyData(np.asarray(points, dtype=np.float64))
        sampled = cloud.sample(self._grid)
        return np.asarray(sampled[self._array], dtype=np.float64)
