"""U6 — one command -> the Monday 31 decision input (technical spec §U6).

Usage:
    PYTHONPATH=src python scripts/u6_gate_report.py                # synthetic, baselines only
    PYTHONPATH=src python scripts/u6_gate_report.py --with-model   # + PINN fits (needs the T4)
    PYTHONPATH=src python scripts/u6_gate_report.py --vtk PATH ... # CASE-01 field

Writes outputs/gate-report/{results.csv, verdict.txt, *.png} and prints the
verdict line. Baselines-only runs are a plumbing dry run: the gate itself is
not evaluated without model rows.
"""

import argparse
import csv
from pathlib import Path

from open_airfield.baselines import IDWBaseline, MeanBaseline, ZeroBaseline
from open_airfield.field_synthetic import SyntheticField
from open_airfield.gate import decide, load_config, run_sweep, write_charts


def _fitted(cls):
    def factory(obs):
        recon = cls()
        recon.fit(obs)
        return recon

    return factory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk", type=Path, default=None, help="CASE-01 export; default synthetic")
    ap.add_argument("--with-model", action="store_true", help="include PINN fits (GPU)")
    ap.add_argument(
        "--extract-bc",
        action="store_true",
        help="CASE-01: add the conservation-derived extract stratum (see geometry.py)",
    )
    ap.add_argument("--steps", type=int, default=None, help="override sweep.model_steps")
    ap.add_argument("--out", type=Path, default=Path("outputs/gate-report"))
    args = ap.parse_args()

    cfg = load_config()

    if args.vtk:
        from open_airfield.ingest_vtk import Case01Field

        # units_verified: CASE-01 ships a meta.yml sidecar declaring SI explicitly,
        # so the claim is evidenced here rather than assumed in the reader.
        field = Case01Field(args.vtk, units_verified=True)
        provisional = False
    else:
        field = SyntheticField()
        provisional = True  # synthetic evidence -> PROVISIONAL stamp

    factories = {
        "zero": _fitted(ZeroBaseline),
        "mean": _fitted(MeanBaseline),
        "idw": _fitted(IDWBaseline),
    }
    if args.with_model:
        # Imported lazily: torch/physicsnemo exist only in the container.
        import numpy as np

        from open_airfield.geometry import LX, LY, LZ, case01_boundary_conditions
        from open_airfield.model import PINNReconstructor

        steps = args.steps or cfg["sweep"]["model_steps"]

        if args.vtk:
            # CASE-01: no-slip walls + declared supply, extract excluded, supply
            # stratified. See geometry.case01_boundary_conditions for why.
            bc_pts, bc_u = case01_boundary_conditions(include_extract=args.extract_bc)
        else:
            # Synthetic: BCs come from truth on all six faces, which is
            # data-consistent by construction and needs no stratum.
            rng = np.random.default_rng(99)
            n_bc = 4_096
            bc_pts = rng.random((n_bc, 3)) * np.array([LX, LY, LZ])
            face = rng.integers(0, 6, size=n_bc)
            axis, side = face % 3, face // 3
            bc_pts[np.arange(n_bc), axis] = side * np.array([LX, LY, LZ])[axis]
            bc_u = field.velocity(bc_pts)

        def model_factory(obs):
            recon = PINNReconstructor(bc_points=bc_pts, bc_values=bc_u, steps=steps)
            recon.fit(obs)
            return recon

        factories["model"] = model_factory

    rows = run_sweep(field, factories, cfg)
    outcome = decide(rows, cfg, provisional=provisional)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    charts = write_charts(rows, cfg, args.out)
    (args.out / "verdict.txt").write_text(outcome["verdict"] + "\n")

    print(f"\n{outcome['verdict']}")
    print(f"artefacts: results.csv, verdict.txt, {', '.join(p.name for p in charts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
