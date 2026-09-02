"""P4: does the peak under-prediction close as sensors are added?

Nishan Jain's diagnostic (2 Sep 2026, physics sense-check reply): a third-low
peak from 20 random sensors is the expected failure mode of sparse
reconstruction -- BUT "if the gap persists as you add sensors, I would look at
the jet prior rather than the sensor count. If the analytic model
under-represents the confined jet, the sensors have to do more correction work
and 20 may not be enough for the peak region."

This runs that test. For each density we select the plain and jet models on p0
by log marginal likelihood over the OBSERVATIONS ONLY -- no eval-grid truth
reaches selection -- then report the held-out median over p1-p7, with the peak
speed alongside. The prior alone (amp = 1.0, zero sensors) is the reference.

Usage:
    uv run python scripts/p4_peak_density_diagnostic.py \
        --vtk data/case-01/data/internal_00010000.vtu
"""

import argparse
import json
from pathlib import Path

import numpy as np

from open_airfield.baselines import DivFreeGP
from open_airfield.metrics import evaluate, moving_air_metrics
from open_airfield.sampling import N_PLACEMENTS, load_or_build_eval_cache

from p1_gp_fit import AMP_GRID, ANISO, DENSITIES, ELL_GRID, build_obs_cache, obs_from_cache


def select_on_p0(cache, d):
    """Best (ell, amp, aniso) by logML on p0 observations, plain and jet."""
    grid = []
    for aniso in (False, True):
        scale = ANISO if aniso else 1.0
        for a in AMP_GRID:
            for l in ELL_GRID:
                r = DivFreeGP(ell=l * scale, prior_amp=a)
                r.fit(obs_from_cache(cache, d, 0))
                grid.append((r.log_marginal_, l, a, aniso))
    plain = max((g for g in grid if g[2] == 0.0), key=lambda g: g[0])
    jet = max(grid, key=lambda g: g[0])
    return plain, jet


def held_out_peaks(ell, amp, aniso, cache, d, eval_pts, truth):
    scale = ANISO if aniso else 1.0
    scores, peaks, moving, still = [], [], [], []
    for p in range(1, N_PLACEMENTS):
        r = DivFreeGP(ell=ell * scale, prior_amp=amp)
        r.fit(obs_from_cache(cache, d, p))
        pred = r.predict(eval_pts)
        mv = moving_air_metrics(pred, truth)
        scores.append(evaluate(pred, truth, eval_pts).rel_l2)
        peaks.append(mv["peak_speed_pred"])
        moving.append(mv["rel_l2_moving"])
        still.append(mv["rel_l2_still"])
    return {
        "median_rel_l2": float(np.median(scores)),
        "median_peak_pred": float(np.median(peaks)),
        "min_peak_pred": float(np.min(peaks)),
        "max_peak_pred": float(np.max(peaks)),
        "median_rel_l2_moving": float(np.median(moving)),
        "median_rel_l2_still": float(np.median(still)),
        "all_rel_l2": [float(s) for s in scores],
        "all_peak_pred": [float(p) for p in peaks],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/2026-09-02-peak-diagnostic"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from open_airfield.ingest_vtk import Case01Field

    field = Case01Field(args.vtk, units_verified=True)
    _, eval_pts, truth = load_or_build_eval_cache(field, cache_dir="outputs")
    cache = build_obs_cache(field, Path("outputs/obs-cache-case01.npz"))
    peak_truth = float(np.linalg.norm(truth, axis=1).max())

    prior = DivFreeGP(ell=1.0, prior_amp=1.0)
    prior_pred = prior._prior.velocity(eval_pts)
    prior_mv = moving_air_metrics(prior_pred, truth)
    report = {
        "question": "does the peak gap close as sensor count rises? (Nishan, 2 Sep 2026)",
        "selector": "log marginal likelihood on p0 observations only; no eval truth",
        "peak_truth": peak_truth,
        "prior_only": {
            "rel_l2": float(evaluate(prior_pred, truth, eval_pts).rel_l2),
            "peak_pred": prior_mv["peak_speed_pred"],
            "peak_ratio": prior_mv["peak_speed_pred"] / peak_truth,
            "rel_l2_moving": prior_mv["rel_l2_moving"],
            "rel_l2_still": prior_mv["rel_l2_still"],
        },
        "by_density": {},
    }
    print(f"peak_truth = {peak_truth:.4f} m/s")
    print(f"prior only (0 sensors, amp 1.0): peak {prior_mv['peak_speed_pred']:.4f} "
          f"({100*prior_mv['peak_speed_pred']/peak_truth:.1f}% of truth), "
          f"rel_l2 {report['prior_only']['rel_l2']:.4f}")

    for d in DENSITIES:
        plain, jet = select_on_p0(cache, d)
        row = {
            "sel_plain": {"logML": plain[0], "ell": plain[1], "amp": plain[2], "aniso": plain[3]},
            "sel_jet": {"logML": jet[0], "ell": jet[1], "amp": jet[2], "aniso": jet[3]},
            "plain": held_out_peaks(plain[1], plain[2], plain[3], cache, d, eval_pts, truth),
            "jet": held_out_peaks(jet[1], jet[2], jet[3], cache, d, eval_pts, truth),
        }
        for k in ("plain", "jet"):
            row[k]["peak_ratio"] = row[k]["median_peak_pred"] / peak_truth
        report["by_density"][str(d)] = row
        print(f"\n=== d={d} ===")
        for k in ("plain", "jet"):
            s = row[f"sel_{k}"]
            r = row[k]
            print(f"  {k:5s} amp={s['amp']:.2f} ell={s['ell']:.3f} aniso={s['aniso']}  "
                  f"rel_l2={r['median_rel_l2']:.4f}  peak={r['median_peak_pred']:.4f} "
                  f"({100*r['peak_ratio']:.1f}% of truth)  "
                  f"moving={r['median_rel_l2_moving']:.4f} still={r['median_rel_l2_still']:.4f}")

    (args.out / "peak-density.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwritten: {args.out/'peak-density.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
