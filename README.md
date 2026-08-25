# Open AirField

Sparse-reconstruction benchmark: how well can indoor airflow be reconstructed
from a handful of point measurements? A physics-constrained coordinate network
is measured against dumb-but-honest baselines on a ventilated-room velocity
field, at sensor densities 5/10/20/50.

Work in progress; the full applicability statement, container digest and run
instructions land with the gate report.

## Reproducibility

- **Seed law**: every observation set is generated with
  `seed = 1000 * density + placement_id` (e.g. density 20, placement 3 →
  seed 20003). Same seed → byte-identical observations.
- Densities: 5 / 10 / 20 / 50. Placements per density: 8. Sensors are placed
  uniformly at random, inset 0.1 m from the walls.
- Evaluation grid: fixed, held out, cell-centred 0.1 m grid,
  80 × 70 × 30 = 168,000 points, cached per truth field as `.npz`.
- Python 3.12 (matches the training container); `uv sync` then
  `uv run pytest` runs every unit's acceptance criteria.
