"""U7 acceptance without the real case (technical spec §U7).

Round-trip: write the synthetic field to .vtu via pyvista, re-read through
the ingest unit, probe 1,000 points, match the analytic values within
interpolation tolerance (rel err < 2% [ASSUMED — falsifier: the tolerance
reflects cell size, retune to the mesh actually delivered]).
"""

import numpy as np

from open_airfield.field_synthetic import SyntheticField
from open_airfield.geometry import LX, LY, LZ
from open_airfield.ingest_vtk import Case01Field, write_field_vtu

FIELD = SyntheticField()
N_PROBE = 1_000


def _probe_points() -> np.ndarray:
    rng = np.random.default_rng(7)
    inset = 0.1
    lo = np.array([inset] * 3)
    hi = np.array([LX, LY, LZ]) - inset
    return lo + rng.random((N_PROBE, 3)) * (hi - lo)


def test_roundtrip_within_interpolation_tolerance(tmp_path):
    vtu = write_field_vtu(FIELD, tmp_path / "synthetic-roundtrip.vtu")
    reread = Case01Field(vtu)
    pts = _probe_points()
    analytic = FIELD.velocity(pts)
    probed = reread.velocity(pts)
    rel_err = np.linalg.norm(probed - analytic) / np.linalg.norm(analytic)
    assert rel_err < 0.02


def test_units_stamped_unverified_by_default(tmp_path):
    vtu = write_field_vtu(FIELD, tmp_path / "synthetic-roundtrip.vtu")
    field = Case01Field(vtu)
    assert field.meta["units"] == "SI assumed — unverified"
    assert Case01Field(vtu, units_verified=True).meta["units"] == "SI"


def test_probe_shape_and_dtype(tmp_path):
    vtu = write_field_vtu(FIELD, tmp_path / "synthetic-roundtrip.vtu")
    out = Case01Field(vtu).velocity(_probe_points()[:13])
    assert out.shape == (13, 3)
    assert out.dtype == np.float64


# --- CASE-01 as delivered: the cell-data path the round-trip above never took ---
#
# The round-trip tolerance stays at 2%: it measures write -> read -> probe on a
# 0.1 m regular grid, and that grid did not change when the real case arrived.
# What changed is that the delivered export stores U at CELL CENTRES, a path the
# original test never exercised. It gets its own test rather than a retuned number.

import pytest
import pyvista as pv

from open_airfield.geometry import LX as _LX
from open_airfield.ingest_vtk import CASE_ID


def _write_cell_data_vtu(path, spacing=0.1):
    """Same field, stored as cell data — how foamToVTK exports volume fields."""
    dims = (int(LX / spacing) + 1, int(LY / spacing) + 1, int(LZ / spacing) + 1)
    grid = pv.ImageData(dimensions=dims, spacing=(spacing,) * 3, origin=(0.0, 0.0, 0.0))
    centres = np.asarray(grid.cell_centers().points, dtype=np.float64)
    grid.cell_data["U"] = FIELD.velocity(centres)
    grid.cast_to_unstructured_grid().save(path)
    return path


def test_cell_data_export_is_converted_and_probes_sanely(tmp_path):
    vtu = _write_cell_data_vtu(tmp_path / "cell-data.vtu")
    field = Case01Field(vtu)
    assert "cell->point" in field.meta["interpolation"]
    pts = _probe_points()
    rel_err = np.linalg.norm(field.velocity(pts) - FIELD.velocity(pts)) / np.linalg.norm(
        FIELD.velocity(pts)
    )
    # Looser than the point-data round-trip: cell-centre storage plus node
    # averaging is two lossy steps, not one.
    assert rel_err < 0.05


def test_out_of_bounds_probe_raises_instead_of_returning_zeros(tmp_path):
    """The failure this guards: sample() returns [0,0,0] outside the mesh, which
    is indistinguishable from still air and would score cleanly against nothing."""
    field = Case01Field(write_field_vtu(FIELD, tmp_path / "rt.vtu"))
    with pytest.raises(ValueError, match="outside the mesh"):
        field.velocity(np.array([[99.0, 99.0, 99.0]]))


def test_wrong_domain_is_rejected(tmp_path):
    grid = pv.ImageData(dimensions=(5, 5, 5), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    grid.point_data["U"] = np.zeros((125, 3))
    path = tmp_path / "wrong-room.vtu"
    grid.cast_to_unstructured_grid().save(path)
    with pytest.raises(ValueError, match="do not match the CASE-01 domain"):
        Case01Field(path)


def test_case_id_is_settled_not_derived_from_the_filename(tmp_path):
    """Nishan's label is A01 and his file is internal_00010000; ours is CASE-01."""
    vtu = write_field_vtu(FIELD, tmp_path / "internal_00010000.vtu")
    assert Case01Field(vtu).meta["case_id"] == CASE_ID == "CASE-01"
    assert Case01Field(vtu).meta["name"] == "vtk-case-01"
