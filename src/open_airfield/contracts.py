"""Shared unit contracts, verbatim from the technical spec ("Shared types").

Every downstream unit consumes these. The synthetic field (U1) and CASE-01
(U7) both implement TruthField, so swapping in the real case is one
constructor call (design invariant 1).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TruthField(Protocol):
    """The swap point (invariant 1)."""

    meta: dict  # {"name", "source", "params"|"case_id", ...}

    def velocity(self, points: np.ndarray) -> np.ndarray:  # (N,3) m -> (N,3) m/s
        ...


@dataclass(frozen=True)
class ObservationSet:
    points: np.ndarray  # (n,3) float64, room frame, metres
    u: np.ndarray  # (n,3) float64, m/s
    density: int  # 5|10|20|50
    placement_id: int  # 0..P-1
    seed: int  # derived: seed = 1000*density + placement_id


class Reconstructor(Protocol):
    def fit(self, obs: ObservationSet) -> None: ...

    def predict(self, points: np.ndarray) -> np.ndarray: ...  # (M,3) -> (M,3)


@dataclass(frozen=True)
class EvalResult:
    rel_l2: float  # ||pred-truth||_2 / ||truth||_2 over eval grid
    p95_abs: float  # 95th percentile of per-point |error| (m/s)
    worst_voxel: tuple  # (i,j,k) argmax of 0.5 m-voxel mean error
    worst_voxel_err: float  # that voxel's mean abs error (m/s)
    frac_above: float  # fraction of eval points with rel err > 0.25
