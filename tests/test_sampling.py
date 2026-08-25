"""U2 acceptance criteria (technical spec §U2), runnable without Tino.

1. Same seed -> byte-identical ObservationSet.
2. All observation points inside the domain (0.1 m wall inset).
3. Eval-grid cache round-trips.
"""

import numpy as np

from open_airfield.field_synthetic import SyntheticField
from open_airfield.geometry import LX, LY, LZ
from open_airfield.sampling import (
    DENSITIES,
    EVAL_GRID_SHAPE,
    N_PLACEMENTS,
    WALL_INSET,
    eval_grid,
    load_or_build_eval_cache,
    observation_seed,
    sample_observations,
)

FIELD = SyntheticField()


def test_seed_law():
    assert observation_seed(density=20, placement_id=3) == 20_003
    for d in DENSITIES:
        for p in range(N_PLACEMENTS):
            assert observation_seed(d, p) == 1000 * d + p


def test_same_seed_byte_identical():
    a = sample_observations(FIELD, density=20, placement_id=5)
    b = sample_observations(FIELD, density=20, placement_id=5)
    assert a.points.tobytes() == b.points.tobytes()
    assert a.u.tobytes() == b.u.tobytes()
    assert (a.density, a.placement_id, a.seed) == (b.density, b.placement_id, b.seed)


def test_placements_differ():
    a = sample_observations(FIELD, density=20, placement_id=0)
    b = sample_observations(FIELD, density=20, placement_id=1)
    assert a.points.tobytes() != b.points.tobytes()


def test_all_points_inside_inset_domain():
    for d in DENSITIES:
        for p in range(N_PLACEMENTS):
            obs = sample_observations(FIELD, density=d, placement_id=p)
            assert obs.points.shape == (d, 3)
            assert obs.u.shape == (d, 3)
            lo = np.array([WALL_INSET] * 3)
            hi = np.array([LX, LY, LZ]) - WALL_INSET
            assert np.all(obs.points >= lo) and np.all(obs.points <= hi)


def test_eval_grid_shape_and_bounds():
    pts = eval_grid()
    assert pts.shape == (int(np.prod(EVAL_GRID_SHAPE)), 3)
    assert pts.shape[0] == 168_000
    assert np.all(pts > 0.0) and np.all(pts < np.array([LX, LY, LZ]))


def test_eval_cache_roundtrip(tmp_path):
    path1, pts1, u1 = load_or_build_eval_cache(FIELD, cache_dir=tmp_path)
    assert path1.exists()
    # Second call must hit the cache and return byte-identical arrays.
    path2, pts2, u2 = load_or_build_eval_cache(FIELD, cache_dir=tmp_path)
    assert path1 == path2
    assert pts1.tobytes() == pts2.tobytes()
    assert u1.tobytes() == u2.tobytes()
    # And the truth values must match a fresh probe of the field.
    assert np.array_equal(u1, FIELD.velocity(pts1))
