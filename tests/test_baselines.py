"""U3 acceptance criteria (technical spec §U3), runnable without Tino.

Spec §U3 originally asserted IDW rel_l2 < mean rel_l2 < zero rel_l2. The
mean-vs-zero half was FALSIFIED on 25 Aug 2026: in a sealed recirculating
room the volume-mean velocity is ~0, so the observation mean is a small
noise vector and scores marginally WORSE than zero (1.002-1.034 vs 1.0
across all 8 placements at density 50). The revised sanity ordering is
IDW strictly below both trivial baselines, with mean pinned near 1.0.
Recorded in the spec §U3.
"""

import numpy as np

from open_airfield.baselines import IDWBaseline, MeanBaseline, ZeroBaseline
from open_airfield.field_synthetic import SyntheticField
from open_airfield.sampling import eval_grid, sample_observations

FIELD = SyntheticField()


def _rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def test_baseline_sanity_ordering_at_density_50():
    obs = sample_observations(FIELD, density=50, placement_id=0)
    pts = eval_grid()
    truth = FIELD.velocity(pts)

    scores = {}
    for name, recon in [
        ("zero", ZeroBaseline()),
        ("mean", MeanBaseline()),
        ("idw", IDWBaseline()),
    ]:
        recon.fit(obs)
        scores[name] = _rel_l2(recon.predict(pts), truth)

    assert scores["idw"] < scores["zero"]
    assert scores["idw"] < scores["mean"]
    # Near-zero volume-mean field: the mean baseline sits at ~1.0, not below it.
    assert 0.95 < scores["mean"] < 1.10


def test_zero_baseline_scores_exactly_one():
    obs = sample_observations(FIELD, density=50, placement_id=0)
    pts = eval_grid()
    zero = ZeroBaseline()
    zero.fit(obs)
    assert _rel_l2(zero.predict(pts), FIELD.velocity(pts)) == 1.0


def test_predict_shapes():
    obs = sample_observations(FIELD, density=10, placement_id=0)
    pts = eval_grid()[:97]
    for recon in (ZeroBaseline(), MeanBaseline(), IDWBaseline()):
        recon.fit(obs)
        out = recon.predict(pts)
        assert out.shape == (97, 3)


def test_idw_interpolates_observations_exactly():
    """At an observation point the IDW weight diverges: prediction == observation."""
    obs = sample_observations(FIELD, density=10, placement_id=2)
    idw = IDWBaseline()
    idw.fit(obs)
    assert np.allclose(idw.predict(obs.points), obs.u, atol=1e-9)
