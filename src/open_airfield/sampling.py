"""U2 — sampling harness (technical spec §U2).

Turns any TruthField into seeded sparse observation sets plus a fixed held-out
evaluation grid.

Spec constants: densities 5/10/20/50 [frozen spec §5]; 8 placements per
density [ASSUMED — falsifier: if placement variance at density 20 exceeds the
model-vs-baseline margin, raise to 16]; placements uniform random in the room
inset 0.1 m from walls (sensors sample the field, not the wall); seed law
seed = 1000*density + placement_id, documented in the README.

Eval grid: regular 0.1 m grid, 80 x 70 x 30 = 168,000 points, cell-centred
(0.05, 0.15, ...) so no point sits on a wall. Computed once per field and
cached as .npz keyed by the field's meta name.
"""

from pathlib import Path

import numpy as np

from open_airfield.contracts import ObservationSet, TruthField
from open_airfield.geometry import LX, LY, LZ

DENSITIES = (5, 10, 20, 50)  # frozen spec §5
N_PLACEMENTS = 8
WALL_INSET = 0.1  # m
EVAL_GRID_SPACING = 0.1  # m
EVAL_GRID_SHAPE = (80, 70, 30)


def observation_seed(density: int, placement_id: int) -> int:
    """Seed law, documented in the README: seed = 1000*density + placement_id."""
    return 1000 * density + placement_id


def sample_observations(field: TruthField, density: int, placement_id: int) -> ObservationSet:
    """Seeded uniform-random sensor placement, inset from the walls."""
    seed = observation_seed(density, placement_id)
    rng = np.random.default_rng(seed)
    lo = np.array([WALL_INSET, WALL_INSET, WALL_INSET])
    hi = np.array([LX, LY, LZ]) - WALL_INSET
    points = lo + rng.random((density, 3)) * (hi - lo)
    return ObservationSet(
        points=points,
        u=field.velocity(points),
        density=density,
        placement_id=placement_id,
        seed=seed,
    )


def eval_grid() -> np.ndarray:
    """The fixed held-out evaluation grid: cell-centred, (168000, 3)."""
    axes = [
        (np.arange(n) + 0.5) * EVAL_GRID_SPACING
        for n in EVAL_GRID_SHAPE
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 3)


def load_or_build_eval_cache(
    field: TruthField, cache_dir: Path | str = "outputs"
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Probe the truth field on the eval grid once, then serve from .npz.

    Returns (cache_path, points, u). The cache is keyed by the field's meta
    name, so synthetic and CASE-01 caches coexist.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"eval-grid-{field.meta['name']}.npz"
    if path.exists():
        data = np.load(path)
        return path, data["points"], data["u"]
    points = eval_grid()
    u = field.velocity(points)
    np.savez_compressed(path, points=points, u=u)
    return path, points, u


# --- Informed placement (28 Aug 2026) ---------------------------------------
#
# The counterfactual to the uniform-random placements every gate figure was
# measured on. The measured finding is that random sensors miss the jet: at
# density 20 the median layout sees NOTHING above 0.25 m/s. This asks what
# happens when a practitioner places the same budget deliberately.
#
# 🔴 CONSTRAINT THAT MAKES IT PUBLISHABLE: placement uses DECLARED GEOMETRY
# ONLY — supply patch, extract patch, room extent, and the textbook expectation
# that a ceiling jet falls, impinges, spreads along the floor and returns. It
# never reads the truth field. Placing sensors by looking at the answer would
# be the same class of error as the oracle-BC diagnostic and could not be
# entered. This is exactly the knowledge an engineer holds before measuring,
# which is also why it is a fair demonstration of the proprietary
# "what to measure next" layer rather than a rigged comparison.
#
# Strata, as fractions of the budget:
#   jet      0.30  vertical column under the supply, ceiling to floor
#   floor    0.30  impingement ring, two radii, low z
#   return   0.20  under the extract and along the supply->extract line
#   control  0.20  far field, so still air is still represented
#
# The control stratum is not padding. Deleting it would let informed placement
# win on the jet by losing the quiescent 95%, and the headline metric is
# global rel_l2.

INFORMED_STRATA = (("jet", 0.30), ("floor", 0.30), ("return", 0.20), ("control", 0.20))
INFORMED_SEED_BASE = 900_000


def informed_seed(density: int, placement_id: int) -> int:
    """Distinct from observation_seed's space so the two never collide."""
    return INFORMED_SEED_BASE + 1000 * density + placement_id


def _allocate(density: int) -> dict:
    """Largest-remainder allocation, so the strata always sum to density."""
    raw = {name: density * frac for name, frac in INFORMED_STRATA}
    base = {k: int(v) for k, v in raw.items()}
    short = density - sum(base.values())
    for name, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if short <= 0:
            break
        base[name] += 1
        short -= 1
    return base


def informed_points(density: int, placement_id: int = 0, jitter: float = 0.15) -> np.ndarray:
    """Deliberate sensor positions from declared geometry. No truth is read."""
    from open_airfield.geometry import EXTRACT, LX, LY, LZ, SUPPLY

    rng = np.random.default_rng(informed_seed(density, placement_id))
    n = _allocate(density)
    box = np.array([LX, LY, LZ])
    pts = []

    # Jet: vertical column under the supply centre, ceiling side to near floor.
    for z in np.linspace(LZ - 0.3, 0.4, max(n["jet"], 1))[: n["jet"]]:
        pts.append([SUPPLY.x, SUPPLY.y, z])

    # Floor: impingement ring at two radii around the supply footprint, low z.
    if n["floor"]:
        ang = np.linspace(0, 2 * np.pi, n["floor"], endpoint=False)
        radii = np.where(np.arange(n["floor"]) % 2 == 0, 1.0, 2.0)
        for a, r in zip(ang, radii):
            pts.append([SUPPLY.x + r * np.cos(a), SUPPLY.y + r * np.sin(a), 0.25])

    # Return: under the extract, then along the supply->extract line at mid height.
    if n["return"]:
        half = n["return"] // 2
        for z in np.linspace(LZ - 0.3, LZ - 1.2, max(half, 1))[:half]:
            pts.append([EXTRACT.x, EXTRACT.y, z])
        for f in np.linspace(0.3, 0.8, n["return"] - half):
            pts.append(
                [
                    SUPPLY.x + f * (EXTRACT.x - SUPPLY.x),
                    SUPPLY.y + f * (EXTRACT.y - SUPPLY.y),
                    LZ * 0.5,
                ]
            )

    # Control: far field, seeded uniform, so the quiescent room stays represented.
    if n["control"]:
        lo, hi = np.full(3, WALL_INSET), box - WALL_INSET
        pts.extend((lo + rng.random((n["control"], 3)) * (hi - lo)).tolist())

    out = np.asarray(pts, dtype=np.float64)
    # Seeded jitter: the strategy is a rule, not a hand-tuned point set, and a
    # result that only holds for exact coordinates would not be a result.
    out += rng.normal(0.0, jitter, out.shape)
    return np.clip(out, WALL_INSET, box - WALL_INSET)


def sample_informed(field, density: int, placement_id: int = 0) -> ObservationSet:
    """Informed counterpart to sample_observations. Same ObservationSet contract."""
    points = informed_points(density, placement_id)
    return ObservationSet(
        points=points,
        u=field.velocity(points),
        density=density,
        placement_id=placement_id,
        seed=informed_seed(density, placement_id),
    )
