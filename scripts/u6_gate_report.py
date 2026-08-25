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
    ap.add_argument("--steps", type=int, default=None, help="override sweep.model_steps")
    ap.add_argument("--out", type=Path, default=Path("outputs/gate-report"))
    args = ap.parse_args()

    cfg = load_config()

    if args.vtk:
        from open_airfield.ingest_vtk import Case01Field

        field = Case01Field(args.vtk)
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

        from open_airfield.geometry import LX, LY, LZ, SUPPLY, SUPPLY_VELOCITY
        from open_airfield.model import PINNReconstructor

        steps = args.steps or cfg["sweep"]["model_steps"]
        rng = np.random.default_rng(99)
        n_bc = 4_096
        bc_pts = rng.random((n_bc, 3)) * np.array([LX, LY, LZ])
        face = rng.integers(0, 6, size=n_bc)
        axis, side = face % 3, face // 3
        bc_pts[np.arange(n_bc), axis] = side * np.array([LX, LY, LZ])[axis]

        if args.vtk:
            # CASE-01: no-slip walls + the declared supply velocity.
            bc_u = np.zeros_like(bc_pts)
            on_supply = (
                (np.abs(bc_pts[:, 2] - LZ) < 1e-9)
                & (np.abs(bc_pts[:, 0] - SUPPLY.x) <= SUPPLY.width / 2)
                & (np.abs(bc_pts[:, 1] - SUPPLY.y) <= SUPPLY.depth / 2)
            )
            bc_u[on_supply] = [0.0, 0.0, -SUPPLY_VELOCITY]
        else:
            bc_u = field.velocity(bc_pts)  # synthetic: data-consistent truth BC

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
