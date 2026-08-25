"""U5 acceptance criteria (technical spec §U5), runnable without Tino.

1. Metrics of truth-vs-truth are exactly 0.
2. Metrics of the zero-field are exactly (1.0, ...) — rel_l2 1.0, frac_above 1.0.
3. A recursive search for Rich's forbidden word over src/ tests/ notebooks/
   returns nothing (the pattern is assembled at runtime so this test cannot
   itself trip the check).
"""

import subprocess
from pathlib import Path

import numpy as np

from open_airfield.field_synthetic import SyntheticField
from open_airfield.metrics import VOXEL_SIZE, component_metrics, evaluate
from open_airfield.sampling import eval_grid

FIELD = SyntheticField()
REPO_ROOT = Path(__file__).resolve().parents[1]


def _pts_truth():
    pts = eval_grid()
    return pts, FIELD.velocity(pts)


def test_truth_vs_truth_is_exactly_zero():
    pts, truth = _pts_truth()
    r = evaluate(truth, truth, pts)
    assert r.rel_l2 == 0.0
    assert r.p95_abs == 0.0
    assert r.worst_voxel_err == 0.0
    assert r.frac_above == 0.0


def test_zero_field_scores_exactly_one():
    pts, truth = _pts_truth()
    r = evaluate(np.zeros_like(truth), truth, pts)
    assert r.rel_l2 == 1.0
    assert r.frac_above == 1.0
    assert r.p95_abs > 0.0  # 95th percentile of |truth| itself


def test_worst_voxel_localises_an_injected_error():
    pts, truth = _pts_truth()
    pred = truth.copy()
    # Corrupt every eval point inside one known 0.5 m voxel.
    target = (4, 9, 2)  # voxel indices: x in [2.0,2.5), y in [4.5,5.0), z in [1.0,1.5)
    idx = np.floor(pts / VOXEL_SIZE).astype(int)
    in_voxel = np.all(idx == target, axis=1)
    assert in_voxel.sum() > 0
    pred[in_voxel] += 5.0
    r = evaluate(pred, truth, pts)
    assert r.worst_voxel == target
    assert r.worst_voxel_err > 1.0


def test_component_metrics_shape():
    pts, truth = _pts_truth()
    m = component_metrics(np.zeros_like(truth), truth)
    assert set(m) == {"u", "v", "w", "magnitude"}
    assert m["magnitude"]["rel_l2"] == 1.0


def test_forbidden_word_absent_from_codebase():
    """Rich's binding constraint, as CI. Pattern assembled at runtime."""
    word = "".join(["exp", "osure"])
    dirs = [d for d in ("src", "tests", "notebooks") if (REPO_ROOT / d).is_dir()]
    out = subprocess.run(
        ["grep", "-ri", word, *dirs],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # grep exits 1 when nothing matches — exactly what we require.
    assert out.returncode == 1, f"forbidden word found:\n{out.stdout}"
