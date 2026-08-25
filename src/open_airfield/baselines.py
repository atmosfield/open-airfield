"""U3 — baselines (technical spec §U3).

The dumb-but-honest reconstructors the gate measures against; also the
broken-loss detector. Three, each a Reconstructor:

- ZeroBaseline: predicts 0 everywhere. Its rel_l2 is exactly 1.0 by
  definition, which anchors the metric.
- MeanBaseline: predicts the mean of the observed velocities everywhere.
- IDWBaseline: inverse-distance weighting (p=2, all observations). At an
  observation point the prediction equals the observation exactly.

[ASSUMED IDW is the strongest of the three — falsifier: the Friday table.]
"""

import numpy as np

from open_airfield.contracts import ObservationSet


class ZeroBaseline:
    def fit(self, obs: ObservationSet) -> None:
        pass

    def predict(self, points: np.ndarray) -> np.ndarray:
        return np.zeros((len(points), 3))


class MeanBaseline:
    def fit(self, obs: ObservationSet) -> None:
        self._mean = obs.u.mean(axis=0)

    def predict(self, points: np.ndarray) -> np.ndarray:
        return np.tile(self._mean, (len(points), 1))


class IDWBaseline:
    """Inverse-distance weighting, power p=2, over all observations."""

    def __init__(self, power: float = 2.0):
        self.power = power

    def fit(self, obs: ObservationSet) -> None:
        self._points = obs.points
        self._u = obs.u

    def predict(self, points: np.ndarray) -> np.ndarray:
        # (M, n) pairwise distances from query points to observations.
        d = np.linalg.norm(points[:, None, :] - self._points[None, :, :], axis=2)
        exact = d < 1e-12  # query coincides with an observation
        with np.errstate(divide="ignore"):
            w = 1.0 / d**self.power
        w[exact.any(axis=1)] = 0.0
        w[exact] = 1.0
        return (w / w.sum(axis=1, keepdims=True)) @ self._u
