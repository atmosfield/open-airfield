"""U4 — a single PINN fit, on either truth field (technical spec §U4).

Was synthetic-only; generalised 26 Aug when CASE-01 landed, because the one
question worth ten minutes of GPU is whether the model transfers off the
synthetic stand-in it was proven on.

    PYTHONPATH=src python scripts/u4_train.py                       # synthetic
    PYTHONPATH=src python scripts/u4_train.py --vtk PATH -d 20      # CASE-01

Two different acceptance tests, because the fields are not comparable:

- **Synthetic** keeps the original §U4 bar: model rel_l2 < IDW AND < 0.15.
- **CASE-01** is measured against the PRE-COMMITTED GATE instead, computed
  from the baseline actually observed at that density rather than a number
  carried over from another field: model <= ratio_max x best baseline
  (configs/gate.yaml). A single placement is not the gate, which takes the
  median over 8 — this is the go/no-go on whether the sweep is worth five
  hours, not the verdict.

Writes outputs/u4-loss-log-<field>.csv and outputs/u4-result-<field>.json.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import yaml

from open_airfield.baselines import IDWBaseline, MeanBaseline, ZeroBaseline
from open_airfield.field_synthetic import SyntheticField
from open_airfield.geometry import LX, LY, LZ, case01_boundary_conditions
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
    ap.add_argument("--density", "-d", type=int, default=50)
    ap.add_argument("--placement", "-p", type=int, default=0)
    ap.add_argument("--momentum", action="store_true")
    ap.add_argument("--vtk", type=Path, default=None, help="CASE-01 export; default synthetic")
    ap.add_argument(
        "--extract-bc",
        action="store_true",
        help="CASE-01: add the conservation-derived extract stratum (closes the mass balance)",
    )
    ap.add_argument(
        "--oracle-bc",
        action="store_true",
        help="DIAGNOSTIC ONLY, NEVER PUBLISHABLE: CASE-01 BCs read from the truth field",
    )
    args = ap.parse_args()
    if args.oracle_bc and not args.vtk:
        ap.error("--oracle-bc is a CASE-01 diagnostic; the synthetic path already uses truth BCs")

    if args.vtk:
        from open_airfield.ingest_vtk import Case01Field

        # units_verified: the meta.yml sidecar declares SI explicitly.
        field = Case01Field(args.vtk, units_verified=True)
        if args.oracle_bc:
            # 🔴 TRUTH LEAKAGE BY DESIGN. This run answers ONE question and its
            # number may never be published, quoted to Rich as a result, or put
            # in the benchmark: with a CONSISTENT boundary condition set, i.e.
            # exactly the setup the synthetic field had when it PASSED at 0.124,
            # does the model still score ~0.92 on the real field?
            #   ~0.92 -> information limit. The 26 Aug NO-GO is sound.
            #   much lower -> the NO-GO is a BC-construction artefact.
            # Points are inset 1 mm so the VTK probe does not reject them for
            # sitting exactly on the mesh surface; the cell size is 50 mm.
            rng = np.random.default_rng(BC_SEED)
            bc_pts = wall_points(N_BC, rng)
            box = np.array([LX, LY, LZ])
            bc_pts = np.clip(bc_pts, 1e-3, box - 1e-3)
            bc_u = field.velocity(bc_pts)
        else:
            # No-slip walls + declared supply, supply stratified. --extract-bc
            # adds the conservation-derived extract; neither reads truth.
            bc_pts, bc_u = case01_boundary_conditions(include_extract=args.extract_bc)
    else:
        field = SyntheticField()
        rng = np.random.default_rng(BC_SEED)
        bc_pts = wall_points(N_BC, rng)
        bc_u = field.velocity(bc_pts)  # synthetic: data-consistent truth BC

    obs = sample_observations(field, density=args.density, placement_id=args.placement)

    model = PINNReconstructor(
        bc_points=bc_pts, bc_values=bc_u, steps=args.steps, momentum=args.momentum
    )
    model.fit(obs)

    _, eval_pts, truth = load_or_build_eval_cache(field, cache_dir="outputs")

    def rel_l2(pred):
        return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))

    model_score = rel_l2(model.predict(eval_pts))
    scores = {}
    for name, B in (("zero", ZeroBaseline()), ("mean", MeanBaseline()), ("idw", IDWBaseline())):
        B.fit(obs)
        scores[name] = rel_l2(B.predict(eval_pts))
    idw_score = scores["idw"]
    best_name = min(scores, key=scores.get)
    best_score = scores[best_name]

    tag = "case01" if args.vtk else "synthetic"
    if args.oracle_bc:
        tag += "-ORACLE-BC-DIAGNOSTIC-NOT-PUBLISHABLE"
    elif args.extract_bc:
        tag += "-extractbc"
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    with (out / f"u4-loss-log-{tag}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(model.log[0].keys()))
        writer.writeheader()
        writer.writerows(model.log)

    if args.vtk:
        # The gate's own rule, on the baseline actually observed here.
        cfg = yaml.safe_load(Path("configs/gate.yaml").read_text())
        ratio_max = cfg["gate"]["ratio_max"]
        bar = ratio_max * best_score
        verdict = model_score <= bar
        bar_label = f"{ratio_max} x best baseline ({best_name} {best_score:.4f})"
    else:
        bar = 0.15
        verdict = model_score < idw_score and model_score < bar
        bar_label = f"IDW {idw_score:.4f} and bar {bar}"

    result = {
        "field": field.meta["name"],
        "density": args.density,
        "placement": args.placement,
        "steps": args.steps,
        "momentum": args.momentum,
        "bc_mode": (
            "ORACLE (truth-leaking diagnostic, NOT PUBLISHABLE)"
            if args.oracle_bc
            else ("walls+supply+derived-extract" if args.extract_bc else "walls+supply")
        ),
        "n_bc_points": int(len(bc_pts)),
        "model_rel_l2": round(model_score, 4),
        "baselines": {k: round(v, 4) for k, v in scores.items()},
        "best_baseline": best_name,
        "bar": round(bar, 4),
        "acceptance": "PASS" if verdict else "FAIL",
        "train_wall_clock_s": model.log[-1]["elapsed_s"],
        # Divergence guard (28 Aug 2026): if best_step is well short of the last
        # step, or final/best is large, the run went unstable and the score is
        # from the retained best weights rather than from where training ended.
        "best_step": model.best_step,
        "best_loss": model.best_loss,
        "final_loss": model.final_loss,
        "final_over_best": round(model.final_loss / model.best_loss, 3),
    }
    (out / f"u4-result-{tag}.json").write_text(json.dumps(result, indent=2))

    # The verdict line prints the numbers next to the thresholds (spec §U6 style).
    print(
        f"\nU4 single fit [{tag}, d={args.density} p={args.placement}]: "
        f"model rel_l2 {model_score:.4f} vs {bar_label} = {bar:.4f} "
        f"-> {result['acceptance']}"
    )
    if args.vtk:
        print("NOTE: one placement, not the gate. The gate takes the median over 8.")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
