"""U7 — CASE-01 ingest (technical spec §U7).

Case01Field(TruthField) from a VTK export: pyvista.read for the grid,
arbitrary-point probing via PolyData.sample against it [VERIFIED at
1.344M-cell scale in-container, 0.35 s/1,000 points].

The unit tests itself without the real case (this is how an unverified
ground truth gets tested): write the synthetic field to .vtu, re-read it
through this ingest unit, probe points, match the analytic values within
interpolation tolerance. That round-trip stays as the regression now the
real case has landed.

CASE-01 as delivered (26 Aug 2026), and what it settled:
- Solver/version, units and format are all declared in the `meta.yml` sidecar
  shipped alongside, so none of the three fallbacks is needed.
- Velocity is stored at CELL CENTRES, with point-interpolated data explicitly
  excluded from the export. Cell arrays are converted to point data on load;
  measured at 0.1 s / 0.36 GB on the 1.36M-cell mesh.
- Probing the raw cell data (piecewise-constant, nearest cell) differs from
  the interpolated field by 8.6% rel_l2, max 0.32 m/s, on the eval grid, so
  the choice is not cosmetic. Interpolated is used for BOTH the sparse
  observations and the eval grid: a piecewise-constant truth is discontinuous
  at every cell face, which a continuity-constrained model cannot represent,
  so nearest-cell sampling would manufacture an error floor that is an
  artefact of sampling rather than a property of the method.
- pyvista reads both binary and ASCII; the delivery is binary.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from open_airfield.contracts import TruthField
from open_airfield.geometry import LX, LY, LZ

VELOCITY_ARRAY_CANDIDATES = ("U", "velocity", "Velocity", "u")

CASE_ID = "CASE-01"  # settled 26 Aug 2026; the author's own label is A01
BOUNDS_TOL = 1e-6  # m, on the domain-extent assertion


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

    def __init__(
        self,
        path: Path | str,
        units_verified: bool = False,
        case_id: str = CASE_ID,
        check_bounds: bool = True,
    ):
        path = Path(path)
        grid = pv.read(path)
        if isinstance(grid, pv.MultiBlock):
            raise ValueError(
                f"{path.name} is a MultiBlock ({grid.n_blocks} blocks); pass the "
                "single internal-volume mesh, not the whole foamToVTK directory"
            )

        # CASE-01 ships U as cell data. Convert once, on load: probing cell data
        # is nearest-cell, which is a different field (8.6% rel_l2 apart).
        if not any(n in grid.point_data for n in VELOCITY_ARRAY_CANDIDATES) and any(
            n in grid.cell_data for n in VELOCITY_ARRAY_CANDIDATES
        ):
            grid = grid.cell_data_to_point_data()
            self.interpolation = "cell->point averaged, then linear within cells"
        else:
            self.interpolation = "point data as delivered, linear within cells"

        self._grid = grid
        self._array = next(
            (n for n in VELOCITY_ARRAY_CANDIDATES if n in self._grid.point_data),
            None,
        )
        if self._array is None:
            raise ValueError(
                f"no velocity array in {path.name}; point data: "
                f"{list(self._grid.point_data)}; cell data: {list(self._grid.cell_data)}"
            )

        # The domain must be the room the benchmark is defined on. A mismatch
        # here is silent otherwise: out-of-bounds probes return [0,0,0], which
        # is indistinguishable from still air, and would score cleanly against
        # nothing at all.
        if check_bounds:
            got = tuple(round(float(b), 6) for b in self._grid.bounds)
            want = (0.0, LX, 0.0, LY, 0.0, LZ)
            if any(abs(g - w) > BOUNDS_TOL for g, w in zip(got, want)):
                raise ValueError(
                    f"{path.name} bounds {got} do not match the CASE-01 domain "
                    f"{want} from geometry.py — wrong case, wrong frame, or an "
                    "offset origin"
                )

        self.meta = {
            "name": f"vtk-{case_id.lower()}",
            "source": str(path),
            "case_id": case_id,
            "units": "SI" if units_verified else "SI assumed — unverified",
            "n_cells": self._grid.n_cells,
            "velocity_array": self._array,
            "interpolation": self.interpolation,
        }

    def velocity(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        sampled = pv.PolyData(points).sample(self._grid)
        # sample() returns [0,0,0] for points outside the mesh and records it
        # only here. Unchecked, a frame error reads as a room full of still air.
        mask = np.asarray(sampled["vtkValidPointMask"])
        if not mask.all():
            bad = int((~mask.astype(bool)).sum())
            first = points[~mask.astype(bool)][0]
            raise ValueError(
                f"{bad} of {len(points)} probe points fell outside the mesh "
                f"(first: {np.round(first, 4).tolist()}); sampling would have "
                "returned zeros for them"
            )
        return np.asarray(sampled[self._array], dtype=np.float64)
