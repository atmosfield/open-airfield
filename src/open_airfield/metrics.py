"""U5 — metrics (technical spec §U5, frozen spec §5).

One EvalResult per (reconstructor, density, placement):

- rel_l2: ||pred - truth||_2 / ||truth||_2 over the eval grid, all components.
- p95_abs: 95th percentile of per-point error magnitude |pred - truth| (m/s).
- worst_voxel / worst_voxel_err: the 0.5 m voxel with the highest mean
  per-point error magnitude — the worst-served region, reported as (i,j,k)
  indices plus the value.
- frac_above: fraction of eval points with pointwise relative error
  |err| / |truth| > 0.25 [ASSUMED theta — falsifier: if the synthetic run
  saturates it at 0 or 1, retune before Friday and record the change].
  Points where the error is exactly zero count as 0 regardless of |truth|
  (this makes truth-vs-truth exactly 0); points with |truth| = 0 and a
  non-zero error count as above-threshold.

v1 reconstructs velocity. Metric names say what they measure, nothing more.
"""

from dataclasses import asdict

import numpy as np

from open_airfield.contracts import EvalResult

VOXEL_SIZE = 0.5  # m
THETA = 0.25  # pointwise relative-error threshold for frac_above


def evaluate(pred: np.ndarray, truth: np.ndarray, points: np.ndarray) -> EvalResult:
    err = pred - truth
    err_mag = np.linalg.norm(err, axis=1)
    truth_mag = np.linalg.norm(truth, axis=1)

    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(truth))
    p95_abs = float(np.percentile(err_mag, 95))

    # Worst-served 0.5 m voxel: mean error magnitude per voxel, argmax.
    idx = np.floor(points / VOXEL_SIZE).astype(int)
    keys = idx[:, 0] * 10_000 + idx[:, 1] * 100 + idx[:, 2]
    uniq, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=err_mag)
    counts = np.bincount(inverse)
    means = sums / counts
    worst = int(np.argmax(means))
    key = int(uniq[worst])
    worst_voxel = (key // 10_000, (key // 100) % 100, key % 100)

    # Pointwise relative error with the exact-zero conventions documented above.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(err_mag == 0.0, 0.0, err_mag / truth_mag)
    rel = np.where((truth_mag == 0.0) & (err_mag > 0.0), np.inf, rel)
    frac_above = float(np.mean(rel > THETA))

    return EvalResult(
        rel_l2=rel_l2,
        p95_abs=p95_abs,
        worst_voxel=worst_voxel,
        worst_voxel_err=float(means[worst]),
        frac_above=frac_above,
    )


def component_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Componentwise + magnitude rel_l2 and p95, supplementary to EvalResult."""
    out = {}
    for i, name in enumerate(("u", "v", "w")):
        e = pred[:, i] - truth[:, i]
        out[name] = {
            "rel_l2": float(np.linalg.norm(e) / np.linalg.norm(truth[:, i])),
            "p95_abs": float(np.percentile(np.abs(e), 95)),
        }
    err_mag = np.linalg.norm(pred - truth, axis=1)
    out["magnitude"] = {
        "rel_l2": float(np.linalg.norm(pred - truth) / np.linalg.norm(truth)),
        "p95_abs": float(np.percentile(err_mag, 95)),
    }
    return out


def as_row(result: EvalResult, **labels) -> dict:
    """Flatten an EvalResult into a CSV-ready row (U6 consumes this)."""
    row = dict(labels)
    row.update(asdict(result))
    row["worst_voxel"] = "/".join(map(str, result.worst_voxel))
    return row


def moving_air_metrics(
    pred: np.ndarray, truth: np.ndarray, theta_speed: float = 0.25
) -> dict:
    """rel_l2 restricted to the part of the room that is actually moving.

    Added 28 Aug 2026 for the informed-placement counterfactual. Global rel_l2
    is dominated by the ~95% of CASE-01 that is near-still air, where predicting
    zero scores 1.0 and any smooth field scores well: it cannot say whether the
    jet was recovered, which is the whole question the counterfactual asks.
    Without this a null result is uninterpretable.

    theta_speed is the same 0.25 m/s (a quarter of supply velocity) used to
    define "moving air" in the placement finding, so the numbers compare.

    Selecting points by truth is a property of the METRIC, not of the fit — no
    truth reaches the reconstructor.
    """
    truth_mag = np.linalg.norm(truth, axis=1)
    moving = truth_mag > theta_speed
    n_moving = int(moving.sum())
    if n_moving == 0:
        return {"theta_speed": theta_speed, "frac_moving": 0.0, "rel_l2_moving": float("nan")}
    err = pred[moving] - truth[moving]
    return {
        "theta_speed": theta_speed,
        "frac_moving": float(moving.mean()),
        "n_moving": n_moving,
        "rel_l2_moving": float(np.linalg.norm(err) / np.linalg.norm(truth[moving])),
        "rel_l2_still": float(
            np.linalg.norm(pred[~moving] - truth[~moving])
            / max(np.linalg.norm(truth[~moving]), 1e-12)
        ),
        "peak_speed_pred": float(np.linalg.norm(pred, axis=1).max()),
        "peak_speed_truth": float(truth_mag.max()),
    }
