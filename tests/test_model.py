"""U4 contract tests. Skip cleanly where torch/physicsnemo are absent (the
Mac); the full acceptance run is scripts/u4_train_synthetic.py on the T4.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("physicsnemo")


@pytest.fixture(scope="module")
def tiny_model():
    from open_airfield.field_synthetic import SyntheticField
    from open_airfield.model import PINNReconstructor
    from open_airfield.sampling import sample_observations

    field = SyntheticField()
    rng = np.random.default_rng(1)
    bc = rng.random((64, 3)) * np.array([8.0, 7.0, 3.0])
    model = PINNReconstructor(
        bc_points=bc,
        bc_values=field.velocity(bc),
        steps=20,
        collocation=256,
        log_every=10,
    )
    model.fit(sample_observations(field, density=10, placement_id=0))
    return model


def test_fit_logs_and_loss_finite(tiny_model):
    assert len(tiny_model.log) >= 2
    assert all(np.isfinite(row["loss"]) for row in tiny_model.log)


def test_predict_shape_dtype(tiny_model):
    out = tiny_model.predict(np.random.default_rng(2).random((11, 3)) * 2.0)
    assert out.shape == (11, 3)
    assert out.dtype == np.float64


def test_momentum_flag_off_by_default(tiny_model):
    assert tiny_model.momentum is False
