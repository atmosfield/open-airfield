"""v3 plan P1-P3: fit the divergence-free GP on CASE-01 and report d20.

Protocol, decision D3 (locked): SELECT on placement p0, REPORT the median over
p1-p7, and say so. Selection here is by log marginal likelihood on p0's
OBSERVATIONS ONLY -- no eval-grid truth reaches the selection at all, which is
stronger than D3 requires.

Usage:
    uv run python scripts/p1_gp_fit.py --vtk data/case-01/data/internal_00010000.vtu
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from open_airfield.baselines import (
    DivFreeGP,
    GPRBaseline,
    IDWBaseline,
    MeanBaseline,
    ZeroBaseline,
)
from open_airfield.contracts import ObservationSet
from open_airfield.geometry import LX, LY, LZ
from open_airfield.metrics import evaluate, moving_air_metrics
from open_airfield.sampling import N_PLACEMENTS, load_or_build_eval_cache, observation_seed

DENSITIES = (10, 20, 50)
ELL_GRID = np.geomspace(0.25, 5.0, 15)
AMP_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
# Room-proportional anisotropy (v3 plan P2): the room is 8 x 7 x 3, so an
# isotropic length scale is a compromise no axis wants.
_GM = (LX * LY * LZ) ** (1.0 / 3.0)
ANISO = np.array([LX, LY, LZ]) / _GM


def build_obs_cache(field, path: Path) -> dict:
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    out = {}
    for d in DENSITIES:
        for p in range(N_PLACEMENTS):
            rng = np.random.default_rng(observation_seed(d, p))
            lo = np.full(3, 0.1)
            hi = np.array([LX, LY, LZ]) - 0.1
            pts = lo + rng.random((d, 3)) * (hi - lo)
            out[f"p_{d}_{p}"] = pts
            out[f"u_{d}_{p}"] = field.velocity(pts)
            print(f"  probed d={d} p={p}")
    np.savez_compressed(path, **out)
    return out


def obs_from_cache(cache, d, p) -> ObservationSet:
    return ObservationSet(
        points=cache[f"p_{d}_{p}"],
        u=cache[f"u_{d}_{p}"],
        density=d,
        placement_id=p,
        seed=observation_seed(d, p),
    )


def held_out(factory, cache, d, eval_pts, truth) -> dict:
    """Median rel_l2 over p1-p7. p0 is NEVER in the reported number (D3)."""
    scores, extras = [], []
    for p in range(1, N_PLACEMENTS):
        r = factory(obs_from_cache(cache, d, p))
        pred = r.predict(eval_pts)
        res = evaluate(pred, truth, eval_pts)
        scores.append(res.rel_l2)
        extras.append((res, moving_air_metrics(pred, truth)))
    scores = np.array(scores)
    best = int(np.argsort(scores)[len(scores) // 2])
    res, mv = extras[best]
    return {
        "median_rel_l2": float(np.median(scores)),
        "spread_rel_l2": float(scores.max() - scores.min()),
        "sd_rel_l2": float(scores.std(ddof=1)),
        "all_rel_l2": [float(s) for s in scores],
        "p95_abs": res.p95_abs,
        "frac_above": res.frac_above,
        "rel_l2_moving": mv["rel_l2_moving"],
        "rel_l2_still": mv["rel_l2_still"],
        "peak_speed_pred": mv["peak_speed_pred"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/2026-09-01-gp"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from open_airfield.ingest_vtk import Case01Field

    t0 = time.time()
    field = Case01Field(args.vtk, units_verified=True)
    _, eval_pts, truth = load_or_build_eval_cache(field, cache_dir="outputs")
    cache = build_obs_cache(field, Path("outputs/obs-cache-case01.npz"))
    print(f"field + caches ready in {time.time()-t0:.1f}s; eval grid {eval_pts.shape}")

    report = {"protocol": "select on p0 by log marginal likelihood (observations "
                          "only); report median rel_l2 over p1-p7 (decision D3)"}

    # --- Stage 1: selection on p0, no eval-grid truth touched ---------------
    sel = {}
    for name, mk in (
        ("gp_iso", lambda l, a: DivFreeGP(ell=l, prior_amp=a)),
        ("gp_aniso", lambda l, a: DivFreeGP(ell=l * ANISO, prior_amp=a)),
    ):
        for d in DENSITIES:
            o0 = obs_from_cache(cache, d, 0)
            grid = []
            for a in AMP_GRID:
                for l in ELL_GRID:
                    r = mk(l, a)
                    r.fit(o0)
                    grid.append((r.log_marginal_, l, a))
            # plain (amp = 0) and jet-prior (amp free) selected separately, so
            # P1-vs-plain is a like-for-like comparison of two selected models.
            plain = max((g for g in grid if g[2] == 0.0), key=lambda g: g[0])
            jet = max(grid, key=lambda g: g[0])
            sel[f"{name}_plain_d{d}"] = {"logML": plain[0], "ell": plain[1], "amp": plain[2]}
            sel[f"{name}_jet_d{d}"] = {"logML": jet[0], "ell": jet[1], "amp": jet[2]}

    for d in DENSITIES:
        o0 = obs_from_cache(cache, d, 0)
        best = max(
            ((GPRBaseline(ell=l * ANISO), l) for l in ELL_GRID),
            key=lambda ri: (ri[0].fit(o0), ri[0].log_marginal_)[1],
        )
        sel[f"gpr_d{d}"] = {"logML": best[0].log_marginal_, "ell": best[1], "amp": 0.0}
    report["selection_on_p0"] = sel
    print(json.dumps(sel, indent=1, default=float))

    # --- Stage 2: held-out evaluation on p1-p7 ------------------------------
    results = {}
    for d in DENSITIES:
        print(f"\n=== density {d} ===")
        specs = {
            "zero": lambda o: _fit(ZeroBaseline(), o),
            "mean": lambda o: _fit(MeanBaseline(), o),
            "idw": lambda o: _fit(IDWBaseline(), o),
            "gpr": lambda o, d=d: _fit(GPRBaseline(ell=sel[f"gpr_d{d}"]["ell"] * ANISO), o),
            "divfree_gp_iso": lambda o, d=d: _fit(
                DivFreeGP(ell=sel[f"gp_iso_plain_d{d}"]["ell"]), o
            ),
            "divfree_gp_aniso": lambda o, d=d: _fit(
                DivFreeGP(ell=sel[f"gp_aniso_plain_d{d}"]["ell"] * ANISO), o
            ),
            "divfree_gp_jet": lambda o, d=d: _fit(
                DivFreeGP(
                    ell=sel[f"gp_aniso_jet_d{d}"]["ell"] * ANISO,
                    prior_amp=sel[f"gp_aniso_jet_d{d}"]["amp"],
                ),
                o,
            ),
            "prior_only": lambda o: _fit(DivFreeGP(ell=1.0, prior_amp=1.0), o, prior_only=True),
        }
        for name, fac in specs.items():
            t = time.time()
            results[f"{name}_d{d}"] = held_out(fac, cache, d, eval_pts, truth)
            r = results[f"{name}_d{d}"]
            print(
                f"  {name:>17} d={d:<3} median(p1-p7)={r['median_rel_l2']:.4f} "
                f"spread={r['spread_rel_l2']:.4f}  [{time.time()-t:.1f}s]"
            )
    report["held_out_p1_p7"] = results
    (args.out / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwritten: {args.out/'results.json'}")
    return 0


def _fit(recon, obs, prior_only: bool = False):
    if prior_only:
        # The analytic prior with NO data at all: the honest measure of how much
        # the declared ventilation condition alone explains.
        class _P:
            def predict(self, pts):
                return recon._prior.velocity(pts)

        return _P()
    recon.fit(obs)
    return recon


if __name__ == "__main__":
    raise SystemExit(main())
