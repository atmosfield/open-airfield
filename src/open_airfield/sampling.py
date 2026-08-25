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
