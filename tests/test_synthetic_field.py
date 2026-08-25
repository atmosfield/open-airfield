"""U1 acceptance criteria (technical spec §U1), runnable without Tino.

1. max |div u| over 10k random points: exactly 0 symbolically (sympy .diff sum
   simplifies to 0) and < 1e-6 numerically via central finite differences.
2. Velocity magnitude at the supply centre within [0.5, 1.5] m/s; spatial
   variation: std of |u| over the room > 0.05 m/s (not a constant field).
3. Mean |u . n| over wall sample points < 5% of peak speed.

Wall sampling note: the ceiling openings are vents, not walls — normal flow
THROUGH them is the design intent — so ceiling samples exclude a disc of
radius WALL_EXCLUSION_RADIUS around each diffuser centre (the jet's Gaussian
influence footprint, ~4 sigma). Everything else on all six faces is sampled.
"""

import numpy as np
import sympy as sp

from open_airfield.contracts import TruthField
from open_airfield.field_synthetic import SyntheticField
from open_airfield.geometry import EXTRACT, LX, LY, LZ, SUPPLY

RNG_SEED = 0  # deterministic everywhere (design invariant 5)
N_INTERIOR = 10_000
FD_H = 1e-4  # central-difference step: truncation ~1e-7 at jet scales, roundoff ~1e-12
INSET = 0.01  # keep FD stencils strictly inside the domain
WALL_EXCLUSION_RADIUS = 1.2  # m, ~4 sigma of the jet footprint
N_PER_FACE = 2_000

FIELD = SyntheticField()


def _interior_points(n: int) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    lo = np.array([INSET, INSET, INSET])
    hi = np.array([LX - INSET, LY - INSET, LZ - INSET])
    return lo + rng.random((n, 3)) * (hi - lo)


def _wall_samples() -> tuple[np.ndarray, np.ndarray]:
    """Sample points on all six faces with their outward unit normals.

    Ceiling points inside the diffuser influence discs are excluded (vents
    are not walls).
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    points, normals = [], []
    faces = [
        (0, 0.0, [-1, 0, 0]),
        (0, LX, [1, 0, 0]),
        (1, 0.0, [0, -1, 0]),
        (1, LY, [0, 1, 0]),
        (2, 0.0, [0, 0, -1]),
        (2, LZ, [0, 0, 1]),
    ]
    for axis, value, normal in faces:
        pts = rng.random((N_PER_FACE, 3)) * np.array([LX, LY, LZ])
        pts[:, axis] = value
        if axis == 2 and value == LZ:  # ceiling: cut the vent discs
            for d in (SUPPLY, EXTRACT):
                r = np.hypot(pts[:, 0] - d.x, pts[:, 1] - d.y)
                pts = pts[r > WALL_EXCLUSION_RADIUS]
        points.append(pts)
        normals.append(np.tile(normal, (len(pts), 1)))
    return np.vstack(points), np.vstack(normals).astype(float)


def test_divergence_zero_symbolically():
    """sympy .diff sum simplifies to exactly 0 — divergence-free identically."""
    x, y, z = FIELD.symbols
    ux, uy, uz = FIELD.u_sym
    div = sp.diff(ux, x) + sp.diff(uy, y) + sp.diff(uz, z)
    assert sp.simplify(sp.expand(div)) == 0


def test_divergence_below_1e6_by_finite_differences():
    pts = _interior_points(N_INTERIOR)
    div = np.zeros(len(pts))
    for axis in range(3):
        fwd, bwd = pts.copy(), pts.copy()
        fwd[:, axis] += FD_H
        bwd[:, axis] -= FD_H
        du = FIELD.velocity(fwd)[:, axis] - FIELD.velocity(bwd)[:, axis]
        div += du / (2.0 * FD_H)
    assert np.max(np.abs(div)) < 1e-6


def test_supply_centre_speed_in_band():
    centre = np.array([[SUPPLY.x, SUPPLY.y, SUPPLY.z]])
    speed = np.linalg.norm(FIELD.velocity(centre))
    assert 0.5 <= speed <= 1.5


def test_supply_jet_points_down():
    """The jet under the supply is downward (-z), per the declared condition."""
    below = np.array([[SUPPLY.x, SUPPLY.y, SUPPLY.z - 0.2]])
    assert FIELD.velocity(below)[0, 2] < 0.0


def test_extract_drawdown_points_up():
    below = np.array([[EXTRACT.x, EXTRACT.y, EXTRACT.z - 0.2]])
    assert FIELD.velocity(below)[0, 2] > 0.0


def test_spatial_variation():
    speeds = np.linalg.norm(FIELD.velocity(_interior_points(N_INTERIOR)), axis=1)
    assert np.std(speeds) > 0.05


def test_wall_normal_flow_below_5pct_of_peak():
    peak = np.max(np.linalg.norm(FIELD.velocity(_interior_points(N_INTERIOR)), axis=1))
    pts, normals = _wall_samples()
    u_dot_n = np.abs(np.sum(FIELD.velocity(pts) * normals, axis=1))
    assert np.mean(u_dot_n) < 0.05 * peak


def test_truthfield_contract():
    assert isinstance(FIELD, TruthField)
    assert FIELD.meta["source"] == "synthetic"
    out = FIELD.velocity(_interior_points(7))
    assert out.shape == (7, 3)
    assert out.dtype == np.float64
