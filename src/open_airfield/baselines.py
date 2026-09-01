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


# --- U3b: Gaussian-process reconstructors (v3 plan Tier 1 + Tier 2) ---------
#
# Added 1 Sep 2026. The divergence-free GP is the entry's PRIMARY PUBLISHED
# METHOD, not a baseline; it lives here because it implements the same
# Reconstructor protocol and the gate sweep consumes it identically. Per-
# component GPR (ordinary kriging) is the comparator the v2 table was missing.
#
# Provenance of the direction: the 29 Aug adversarial technical review (F5)
# measured a divergence-free GP beating the v1 PINN at densities 20 and 50,
# out of repo. Everything below is the in-repo implementation that decision
# D2 makes conditional on reproducing 0.8848 at d20.

import numpy as _np

from open_airfield.jet_prior import VentilationPrior

_JITTER = 1e-10  # kernel is normalised to unit prior variance, so this is relative


def _as_ell(ell) -> "_np.ndarray":
    """Scalar -> isotropic; length-3 -> anisotropic (v3 plan P2)."""
    e = _np.atleast_1d(_np.asarray(ell, dtype=float))
    return _np.broadcast_to(e, (3,)).astype(float)


def divfree_kernel(X: _np.ndarray, Y: _np.ndarray, ell) -> _np.ndarray:
    """Matrix-valued divergence-free squared-exponential kernel.

    K = (grad grad^T - I laplacian) phi, with phi the (optionally anisotropic)
    squared exponential. Any field drawn from this GP is EXACTLY divergence-
    free, because div K = grad(lap phi) - grad(lap phi) = 0 identically. That
    is the whole point: incompressibility is imposed by the function space, not
    by a penalty term that trades off against the data.

    Isotropic form, for the record (verified to reduce to this):

        K_ij(r) = (l^2/2) exp(-|r|^2/2l^2) [ 2 d_ij/l^2 - d_ij |r|^2/l^4
                                             + r_i r_j / l^4 ]

    Returned BLOCKED, (3n, 3m): rows/cols ordered [all x, then all y, then all
    z], matching u.T.ravel(). Normalised so the mean prior variance at r = 0
    is exactly 1, which makes the signal variance separable and lets the
    marginal likelihood be concentrated analytically.
    """
    ell = _as_ell(ell)
    l2 = ell**2
    d = X[:, None, :] - Y[None, :, :]  # (n, m, 3)
    phi = _np.exp(-0.5 * _np.sum(d**2 / l2, axis=2))
    q = _np.sum(d**2 / l2**2, axis=2)  # sum_k r_k^2 / l_k^4
    s_inv = float(_np.sum(1.0 / l2))  # sum_k 1 / l_k^2

    n, m = len(X), len(Y)
    K = _np.empty((3 * n, 3 * m))
    for i in range(3):
        for j in range(3):
            block = d[:, :, i] * d[:, :, j] / (l2[i] * l2[j])
            if i == j:
                block = block - 1.0 / l2[i] - (q - s_inv)
            K[i * n : (i + 1) * n, j * m : (j + 1) * m] = phi * block
    return K / ((2.0 / 3.0) * s_inv)


def se_kernel(X: _np.ndarray, Y: _np.ndarray, ell) -> _np.ndarray:
    """Plain (anisotropic) squared exponential, (n, m). Used by GPRBaseline."""
    ell = _as_ell(ell)
    d = (X[:, None, :] - Y[None, :, :]) / ell
    return _np.exp(-0.5 * _np.sum(d**2, axis=2))


def _cho_fit(K: _np.ndarray, y: _np.ndarray):
    """Cholesky solve + log-determinant. Sizes here are <= 150 x 150."""
    L = _np.linalg.cholesky(K)
    alpha = _np.linalg.solve(L.T, _np.linalg.solve(L, y))
    logdet = 2.0 * float(_np.sum(_np.log(_np.diag(L))))
    return alpha, logdet


def _log_marginal(y: _np.ndarray, alpha: _np.ndarray, logdet: float) -> float:
    """Log marginal likelihood, signal variance concentrated out.

    Selection uses the OBSERVATIONS ONLY -- no eval-grid truth is read at any
    point, which is stronger than decision D3 requires (D3 permits selecting
    on p0's score). The headline stays the median over p1-p7 regardless.
    """
    n = len(y)
    s2 = float(y @ alpha) / n
    return -0.5 * n * _np.log(s2) - 0.5 * logdet - 0.5 * n * (1.0 + _np.log(2.0 * _np.pi))


class DivFreeGP:
    """Divergence-free GP with an optional analytic ventilation prior mean.

    prior_amp = 0.0  -> the plain divergence-free GP (v3 plan Tier 1 headline)
    prior_amp > 0.0  -> P1: the declared ceiling jet + extract as the prior
                        mean, GP models the residual. Nested, so the P1-vs-plain
                        comparison is a single parameter moving off zero.

    Both the mean function and the residual are divergence-free, so the TOTAL
    predicted field is too. That is load-bearing: a mean function with
    divergence would silently falsify the published claim.
    """

    def __init__(self, ell=1.0, prior_amp: float = 0.0, jitter: float = _JITTER):
        self.ell = ell
        self.prior_amp = float(prior_amp)
        self.jitter = jitter
        self._prior = VentilationPrior(amp=self.prior_amp)

    def fit(self, obs) -> None:
        self._points = _np.asarray(obs.points, dtype=float)
        resid = _np.asarray(obs.u, dtype=float) - self._prior.velocity(self._points)
        y = resid.T.ravel()  # [all x, all y, all z], matching the blocked kernel
        K = divfree_kernel(self._points, self._points, self.ell)
        K[_np.diag_indices_from(K)] += self.jitter
        self._alpha, logdet = _cho_fit(K, y)
        self.log_marginal_ = _log_marginal(y, self._alpha, logdet)

    def predict(self, points: _np.ndarray, chunk: int = 8192) -> _np.ndarray:
        points = _np.asarray(points, dtype=float)
        out = _np.empty((len(points), 3))
        for a in range(0, len(points), chunk):
            blk = points[a : a + chunk]
            ks = divfree_kernel(blk, self._points, self.ell)
            out[a : a + chunk] = (ks @ self._alpha).reshape(3, len(blk)).T
        return out + self._prior.velocity(points)


class GPRBaseline:
    """Per-component GP regression -- ordinary kriging, the honest comparator.

    Three independent scalar GPs, one per velocity component, sharing a kernel.
    It has NO incompressibility constraint, which is exactly the contrast the
    divergence-free kernel is claiming to be worth something.

    [MEASURED 29 Aug: collapses at density 5 (1.4270 at p0, worse than
    predicting zero). Reported restricted to d >= 10, with the collapse stated
    rather than hidden -- v3 plan Tier 1, F6.]
    """

    def __init__(self, ell=1.0, jitter: float = _JITTER):
        self.ell = ell
        self.jitter = jitter

    def fit(self, obs) -> None:
        self._points = _np.asarray(obs.points, dtype=float)
        K = se_kernel(self._points, self._points, self.ell)
        K[_np.diag_indices_from(K)] += self.jitter
        L = _np.linalg.cholesky(K)
        u = _np.asarray(obs.u, dtype=float)
        self._alpha = _np.linalg.solve(L.T, _np.linalg.solve(L, u))  # (n, 3)
        logdet = 2.0 * float(_np.sum(_np.log(_np.diag(L))))
        self.log_marginal_ = sum(
            _log_marginal(u[:, k], self._alpha[:, k], logdet) for k in range(3)
        )

    def predict(self, points: _np.ndarray, chunk: int = 8192) -> _np.ndarray:
        points = _np.asarray(points, dtype=float)
        out = _np.empty((len(points), 3))
        for a in range(0, len(points), chunk):
            blk = points[a : a + chunk]
            out[a : a + chunk] = se_kernel(blk, self._points, self.ell) @ self._alpha
        return out
