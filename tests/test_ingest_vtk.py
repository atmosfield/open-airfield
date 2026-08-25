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
