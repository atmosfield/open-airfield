"""U4 acceptance run (technical spec §U4), executed in-container on the T4.

On synthetic truth, density 50, one placement: model rel_l2 < IDW rel_l2
AND < 0.15. Loss curve decreases; NaN guard trips the run red.

Usage (from the repo root inside the container):
    PYTHONPATH=src python scripts/u4_train_synthetic.py [--steps N] [--momentum]

Writes outputs/u4-loss-log.csv and outputs/u4-result.json.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from open_airfield.baselines import IDWBaseline
from open_airfield.field_synthetic import SyntheticField
from open_airfield.geometry import LX, LY, LZ
from open_airfield.model import PINNReconstructor
from open_airfield.sampling import load_or_build_eval_cache, sample_observations

N_BC = 4_096
BC_SEED = 99


def wall_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples over all six faces (the vent values live in the truth field).

    Synthetic only. CASE-01 uses geometry.case01_boundary_conditions instead,
    which excludes the extract patch and stratifies the supply.
    """
    pts = rng.random((n, 3)) * np.array([LX, LY, LZ])
    face = rng.integers(0, 6, size=n)
    axis, side = face % 3, face // 3
    pts[np.arange(n), axis] = side * np.array([LX, LY, LZ])[axis]
    return pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--density", type=int, default=50)
    ap.add_argument("--placement", type=int, default=0)
    ap.add_argument("--momentum", action="store_true")
    args = ap.parse_args()

    field = SyntheticField()
    obs = sample_observations(field, density=args.density, placement_id=args.placement)

    rng = np.random.default_rng(BC_SEED)
    bc_pts = wall_points(N_BC, rng)
    bc_u = field.velocity(bc_pts)  # synthetic: data-consistent truth BC

    model = PINNReconstructor(
        bc_points=bc_pts, bc_values=bc_u, steps=args.steps, momentum=args.momentum
    )
    model.fit(obs)

    _, eval_pts, truth = load_or_build_eval_cache(field, cache_dir="outputs")
    idw = IDWBaseline()
    idw.fit(obs)

    def rel_l2(pred):
        return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))

    model_score = rel_l2(model.predict(eval_pts))
    idw_score = rel_l2(idw.predict(eval_pts))

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    with (out / "u4-loss-log.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(model.log[0].keys()))
        writer.writeheader()
        writer.writerows(model.log)

    verdict = model_score < idw_score and model_score < 0.15
    result = {
        "density": args.density,
        "placement": args.placement,
        "steps": args.steps,
        "momentum": args.momentum,
        "model_rel_l2": round(model_score, 4),
        "idw_rel_l2": round(idw_score, 4),
        "bar": 0.15,
        "acceptance": "PASS" if verdict else "FAIL",
        "train_wall_clock_s": model.log[-1]["elapsed_s"],
    }
    (out / "u4-result.json").write_text(json.dumps(result, indent=2))

    # The verdict line prints the numbers next to the thresholds (spec §U6 style).
    print(
        f"\nU4 acceptance: model rel_l2 {model_score:.4f} "
        f"vs IDW {idw_score:.4f} and bar 0.15 -> {result['acceptance']}"
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
