"""CASE-01 geometry and flow condition.

Single source of truth for the room the benchmark is defined on. Every value
here was declared by the CFD author (Nishan Jain, AeroSHILA) on 23 Aug 2026 and
is now VERIFIED against the delivered export (26 Aug 2026): the mesh reads
bounds exactly (0,0,0)-(8,7,3), the supply column holds 1.00 m/s downward and
the extract column +1.01 m/s upward. The synthetic stand-in field is built to
these numbers so that swapping in the real case is a change of data source, not
a change of coordinate frame.

Two things the case as RUN differs from the case as declared, both recorded by
the author in his writeup §3.3 and neither affecting the interior field we
reconstruct:

- **No wall grading.** Declared as "graded near walls"; run flat at the 0.05 m
  base resolution, because omegaWallFunction covers the full y+ range (measured
  0.59 to 127) and the deliverable is the interior velocity field, not wall
  shear.
- **Plain rectangular ceiling openings, not diffusers.** Hence CeilingPatch
  below rather than Diffuser: the published applicability statement has to say
  what was run.

Coordinate frame, as declared: origin at a floor corner, x along the 8 m wall,
y along the 7 m wall, z vertical and positive upwards. Domain (0,0,0)-(8,7,3).
"""

from dataclasses import dataclass

# Room extent in metres.
LX, LY, LZ = 8.0, 7.0, 3.0
VOLUME = LX * LY * LZ  # 168 m^3


@dataclass(frozen=True)
class CeilingPatch:
    """A ceiling supply or extract opening.

    A plain rectangular patch, which is what was run — not a diffuser with vanes
    or a throw pattern. The distinction is in the published claim, so it is in
    the type name too.
    """

    x: float
    y: float
    z: float
    width: float
    depth: float

    @property
    def area(self) -> float:
        return self.width * self.depth


# Ceiling supply: 1 m/s downward through 0.6 x 0.6 m.
SUPPLY = CeilingPatch(x=4.0, y=1.5, z=3.0, width=0.6, depth=0.6)
SUPPLY_VELOCITY = 1.0  # m/s, downward (-z)

# Matching ceiling extract, ~4 m away along the y axis.
EXTRACT = CeilingPatch(x=4.0, y=5.5, z=3.0, width=0.6, depth=0.6)

# Derived flow condition. These are the numbers that go into the published
# applicability statement, so they are computed here rather than transcribed.
VOLUMETRIC_FLOW = SUPPLY.area * SUPPLY_VELOCITY  # m^3/s
AIR_CHANGES_PER_HOUR = VOLUMETRIC_FLOW * 3600.0 / VOLUME

# Regime, as declared: steady-state RANS, k-omega SST, purely forced.
# No buoyancy, no heat loads, no occupancy, no obstruction.
# The absence of thermal drive is what makes the incompressible RANS mean
# field EXACTLY divergence-free, which is the closure the reconstruction
# rests on. It is a property of this case, not an approximation.
BUOYANCY_DRIVEN = False
STEADY_STATE = True


# --- Boundary conditions for the real case (U4/U6 consume these) ------------
#
# Technical spec §U4: "CASE-01 -> no-slip walls + declared supply velocity".
# Read literally, that is three classes of surface, not two:
#
#   walls    -> no-slip, u = 0                     (declared, and true)
#   supply   -> u = (0, 0, -SUPPLY_VELOCITY)       (declared)
#   extract  -> NOT declared and NOT a wall        (inletOutlet in his case;
#               measured ~+1.01 m/s upward)
#
# Two defects this replaces, both measured 26 Aug 2026 and both invisible on
# the synthetic field (where BCs came from truth everywhere, so they were
# self-consistent by construction):
#
#   1. The extract patch was being forced to no-slip along with the walls,
#      pinning 3 of 4,096 points to zero exactly where the field's second
#      largest feature sits. It is now EXCLUDED from the BC set rather than
#      asserted wrongly: we do not know the outflow profile a priori, and
#      taking it from the truth field would leak truth into the constraint.
#   2. Uniform sampling over the six faces put 4 of 4,096 points inside the
#      supply patch (0.098%), because it is 0.36 m^2 of a 202 m^2 surface.
#      The "declared supply velocity" was therefore statistically nil, on the
#      one boundary that drives the entire flow. The supply is now sampled as
#      its own stratum.
#
# SUPPLY_FRACTION [ASSUMED 1:7 — falsifier: if the density-20 fit shows the
# jet reconstructed but the near-wall field degraded, the supply stratum is
# over-weighted; if the jet is missed entirely with the walls clean, it is
# under-weighted. Either way, retune and record it.] The reasoning for a
# non-area-proportional split: the supply is one of two boundary CONDITIONS,
# not one of two areas, and area-weighting silently deletes it.

SUPPLY_FRACTION = 0.125
PATCH_TOL = 1e-9


def _in_patch(points, patch) -> "np.ndarray":
    import numpy as np

    return (
        (np.abs(points[:, 2] - LZ) < PATCH_TOL)
        & (np.abs(points[:, 0] - patch.x) <= patch.width / 2)
        & (np.abs(points[:, 1] - patch.y) <= patch.depth / 2)
    )


def case01_boundary_conditions(n: int = 4_096, seed: int = 99):
    """No-slip wall points + declared supply points. Extract is excluded.

    Returns (points, values), both (m, 3) float64, m <= n.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n_supply = int(round(n * SUPPLY_FRACTION))
    n_wall = n - n_supply
    box = np.array([LX, LY, LZ])

    # Walls: uniform over the six faces, then drop anything landing on either
    # ceiling patch — the supply has its own stratum and the extract is not a
    # boundary condition we hold.
    pts = rng.random((n_wall, 3)) * box
    face = rng.integers(0, 6, size=n_wall)
    axis, side = face % 3, face // 3
    pts[np.arange(n_wall), axis] = side * box[axis]
    keep = ~(_in_patch(pts, SUPPLY) | _in_patch(pts, EXTRACT))
    wall_pts = pts[keep]
    wall_u = np.zeros_like(wall_pts)

    # Supply: uniform over the patch itself, on the ceiling plane.
    sup = np.empty((n_supply, 3))
    sup[:, 0] = SUPPLY.x + (rng.random(n_supply) - 0.5) * SUPPLY.width
    sup[:, 1] = SUPPLY.y + (rng.random(n_supply) - 0.5) * SUPPLY.depth
    sup[:, 2] = SUPPLY.z
    sup_u = np.tile([0.0, 0.0, -SUPPLY_VELOCITY], (n_supply, 1))

    return np.vstack([wall_pts, sup]), np.vstack([wall_u, sup_u])
