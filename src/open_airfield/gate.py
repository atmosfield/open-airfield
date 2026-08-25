"""U6 — gate report (technical spec §U6).

One command -> the Monday 31 decision input: run {model, 3 baselines} x
densities x placements on the active truth field; write a CSV, two charts and
an auto-evaluated verdict line.

The pre-committed gate (Tino, 24 Aug, written before any run):
- GO iff, on the active field: median rel_l2(model, 20 obs) <= 0.5 x median
  rel_l2(best baseline, 20 obs), AND median rel_l2(model) at 50 < 20 < 10.
- "Best baseline" is selected PER DENSITY (25 Aug finding: IDW at density 5
  is worse than the zero baseline).
- Synthetic truth -> the verdict is stamped PROVISIONAL — synthetic evidence.
- The verdict line prints the numbers next to the thresholds. No
  interpretation in code.

Thresholds live in configs/gate.yaml, nowhere else.
"""

from pathlib import Path
from statistics import median

import numpy as np
import yaml

from open_airfield.contracts import TruthField
from open_airfield.metrics import as_row, evaluate
from open_airfield.sampling import load_or_build_eval_cache, sample_observations

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "gate.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def run_sweep(
    field: TruthField,
    reconstructor_factories: dict,
    cfg: dict,
    cache_dir: Path | str = "outputs",
    log=print,
) -> list[dict]:
    """Fit + evaluate every (reconstructor, density, placement) combination.

    reconstructor_factories: {name: callable(obs) -> fitted Reconstructor}.
    The model is just another factory; omit it for a baselines-only dry run.
    """
    _, eval_pts, truth = load_or_build_eval_cache(field, cache_dir=cache_dir)
    rows = []
    for density in cfg["densities"]:
        for placement in range(cfg["placements"]):
            obs = sample_observations(field, density=density, placement_id=placement)
            for name, factory in reconstructor_factories.items():
                recon = factory(obs)
                result = evaluate(recon.predict(eval_pts), truth, eval_pts)
                rows.append(
                    as_row(
                        result,
                        reconstructor=name,
                        density=density,
                        placement=placement,
                        field=field.meta["name"],
                    )
                )
                log(
                    f"  {name:>6} d={density:<3} p={placement} "
                    f"rel_l2={result.rel_l2:.4f}"
                )
    return rows


def median_rel_l2(rows: list[dict], reconstructor: str, density: int) -> float:
    vals = [
        r["rel_l2"]
        for r in rows
        if r["reconstructor"] == reconstructor and r["density"] == density
    ]
    return median(vals) if vals else float("nan")


def decide(rows: list[dict], cfg: dict, provisional: bool) -> dict:
    """The pre-committed gate, as a pure function of the sweep rows."""
    g = cfg["gate"]
    baselines = sorted(
        {r["reconstructor"] for r in rows} - {"model"}
    )
    if not any(r["reconstructor"] == "model" for r in rows):
        return {
            "verdict": "DRY RUN (baselines only) — no model rows, gate not evaluated",
            "go": None,
        }

    d0 = g["at_density"]
    model_at = median_rel_l2(rows, "model", d0)
    per_baseline = {b: median_rel_l2(rows, b, d0) for b in baselines}
    best_name = min(per_baseline, key=per_baseline.get)
    best_at = per_baseline[best_name]
    ratio_ok = model_at <= g["ratio_max"] * best_at

    med = {d: median_rel_l2(rows, "model", d) for d in g["monotone_densities"]}
    ordered = sorted(g["monotone_densities"])
    monotone_ok = all(
        med[ordered[i + 1]] < med[ordered[i]] for i in range(len(ordered) - 1)
    )

    go = ratio_ok and monotone_ok
    stamp = " [PROVISIONAL — synthetic evidence]" if provisional else ""
    verdict = (
        f"{'GO' if go else 'NO-GO'}{stamp}: "
        f"model median rel_l2 @ {d0} obs = {model_at:.4f} vs "
        f"{g['ratio_max']} x best baseline ({best_name} {best_at:.4f}) = "
        f"{g['ratio_max'] * best_at:.4f} -> {'PASS' if ratio_ok else 'FAIL'}; "
        f"monotone {' > '.join(str(d) for d in ordered)}: "
        + " > ".join(f"{med[d]:.4f}" for d in ordered)
        + f" -> {'PASS' if monotone_ok else 'FAIL'}"
    )
    return {
        "verdict": verdict,
        "go": go,
        "ratio_ok": ratio_ok,
        "monotone_ok": monotone_ok,
        "model_at_gate_density": model_at,
        "best_baseline": best_name,
        "best_baseline_at_gate_density": best_at,
        "model_medians": med,
    }


def write_charts(rows: list[dict], cfg: dict, out_dir: Path) -> list[Path]:
    """Chart 1: median rel_l2 vs density per reconstructor.
    Chart 2: model improvement over IDW vs density (skipped without model rows).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    densities = cfg["densities"]
    names = sorted({r["reconstructor"] for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in names:
        ax.plot(
            densities,
            [median_rel_l2(rows, name, d) for d in densities],
            marker="o",
            label=name,
        )
    ax.set_xlabel("observations")
    ax.set_ylabel("median rel_l2")
    ax.set_yscale("log")
    ax.set_title(f"Reconstruction error vs sensor density ({rows[0]['field']})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p1 = out_dir / "median_rel_l2_vs_density.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p1)

    if "model" in names and "idw" in names:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            densities,
            [
                median_rel_l2(rows, "idw", d) / median_rel_l2(rows, "model", d)
                for d in densities
            ],
            marker="o",
            color="tab:green",
        )
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("observations")
        ax.set_ylabel("IDW median rel_l2 / model median rel_l2  (x better)")
        ax.set_title("Model improvement over IDW vs sensor density")
        ax.grid(True, alpha=0.3)
        p2 = out_dir / "improvement_over_idw.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p2)
    return written
