"""U6 gate-logic tests. The decide() function is pure, so the gate's three
outcomes (GO, ratio-fail, monotone-fail) are tested from constructed rows;
the sweep plumbing is tested baselines-only at reduced placements.
"""

import numpy as np

from open_airfield.baselines import IDWBaseline, MeanBaseline, ZeroBaseline
from open_airfield.field_synthetic import SyntheticField
from open_airfield.gate import decide, load_config, median_rel_l2, run_sweep, write_charts


def _rows(model_by_density: dict, baseline_by_density: dict) -> list[dict]:
    rows = []
    for density, val in model_by_density.items():
        rows.append({"reconstructor": "model", "density": density, "rel_l2": val, "field": "synthetic-v1"})
    for density, val in baseline_by_density.items():
        rows.append({"reconstructor": "idw", "density": density, "rel_l2": val, "field": "synthetic-v1"})
        rows.append({"reconstructor": "zero", "density": density, "rel_l2": 1.0, "field": "synthetic-v1"})
    return rows


CFG = load_config()


def test_go_when_both_criteria_hold():
    rows = _rows({10: 0.5, 20: 0.3, 50: 0.12}, {10: 0.9, 20: 0.8, 50: 0.8})
    out = decide(rows, CFG, provisional=True)
    assert out["go"] is True
    assert "GO" in out["verdict"] and "PROVISIONAL" in out["verdict"]


def test_no_go_on_ratio():
    rows = _rows({10: 0.6, 20: 0.5, 50: 0.4}, {10: 0.9, 20: 0.8, 50: 0.8})
    out = decide(rows, CFG, provisional=False)
    assert out["go"] is False and out["ratio_ok"] is False
    assert "PROVISIONAL" not in out["verdict"]


def test_no_go_on_monotonicity():
    rows = _rows({10: 0.3, 20: 0.2, 50: 0.25}, {10: 0.9, 20: 0.8, 50: 0.8})
    out = decide(rows, CFG, provisional=True)
    assert out["go"] is False and out["monotone_ok"] is False


def test_best_baseline_selected_per_gate_density():
    # IDW is terrible at the gate density -> zero (1.0) must be chosen as best.
    rows = _rows({10: 0.5, 20: 0.3, 50: 0.12}, {10: 2.0, 20: 2.3, 50: 2.0})
    out = decide(rows, CFG, provisional=True)
    assert out["best_baseline"] == "zero"
    assert out["best_baseline_at_gate_density"] == 1.0


def test_dry_run_without_model_rows():
    rows = _rows({}, {10: 0.9, 20: 0.8, 50: 0.8})
    out = decide(rows, CFG, provisional=True)
    assert out["go"] is None
    assert "DRY RUN" in out["verdict"]


def _fitted(cls):
    def factory(obs):
        r = cls()
        r.fit(obs)
        return r

    return factory


def test_sweep_plumbing_baselines_only(tmp_path):
    cfg = dict(CFG)
    cfg["placements"] = 2  # keep the test fast; full sweep uses gate.yaml
    field = SyntheticField()
    rows = run_sweep(
        field,
        {"zero": _fitted(ZeroBaseline), "mean": _fitted(MeanBaseline), "idw": _fitted(IDWBaseline)},
        cfg,
        cache_dir=tmp_path,
        log=lambda *_: None,
    )
    assert len(rows) == 3 * len(cfg["densities"]) * 2
    assert median_rel_l2(rows, "zero", 20) == 1.0
    assert np.isfinite(median_rel_l2(rows, "idw", 50))
    charts = write_charts(rows, cfg, tmp_path / "charts")
    assert len(charts) == 1  # no model rows -> improvement chart skipped
    assert charts[0].exists() and charts[0].stat().st_size > 0
