"""Analytic, divergence-free ventilation prior (v3 plan P1).

The mean function for the divergence-free GP: an analytic model of what the
DECLARED ventilation condition does to the room, built from `geometry.py` and
`meta.yml` alone. No truth field is read, at any point. That is what keeps it
inside Rich's 1 Sep boundary ("standard interpolation using ventilation
geometry as a prior") and out of the truth-leaking class the oracle diagnostic
fell into.

WHY A STREAM FUNCTION, and not an axial profile
------------------------------------------------
The published claim is that the method enforces incompressibility ANALYTICALLY.
The divergence-free GP kernel guarantees that for the RESIDUAL only. If the
mean function carries divergence, the total field mean + residual does not, and
the headline claim is false — a defect a judge would find before we did.

So both vents are built as axisymmetric Stokes stream functions about a
vertical axis:

    u_s = (1/r) dpsi/dr        u_r = -(1/r) dpsi/ds

which satisfies (1/r) d(r u_r)/dr + du_s/ds = 0 IDENTICALLY, for ANY psi. The
consequence is worth stating because it drives the design: we may shape psi
however the physics asks — potential core, spreading, floor impingement, a
finite extract draw depth — and the field stays exactly divergence-free. The
shaping is free; the constraint is structural.

THE MODEL
---------
Both the supply jet and the extract are the same object: a vertical
axisymmetric column hung from a ceiling patch, parametrised by s, the distance
BELOW the ceiling, and r, the radial distance from the patch centre.

    sigma(s) = sigma0 + c*s          Gaussian half-width, linear spreading
    U_c(s)   = U0 * sigma0/sigma(s)  momentum-conserving centreline decay
    T(s)                             smooth taper, ends the column
    psi      = amp * T(s) * U_c(s) * sigma(s)^2 * (1 - exp(-r^2/2 sigma^2))

sigma0 is DERIVED, not guessed: the Gaussian that carries the declared
volumetric flow at the declared face velocity, U0 * 2 pi sigma0^2 = Q, so
sigma0 = sqrt(Q / (2 pi U0)) = 0.239 m for CASE-01. c = 0.082 is the standard
round-free-jet Gaussian width growth (velocity half-width 0.0965 s, converted
to a Gaussian sigma by the sqrt(2 ln 2) factor).

The taper is where the floor comes in. As T -> 0 near the floor, psi collapses,
u_s -> 0 and u_r grows outward: the axial jet converts into a radial wall jet.
That is impingement, and it falls out of the stream function rather than being
imposed. The extract uses the same machinery with a negative amplitude (flow
toward the ceiling) and a taper on a ~1 m draw depth, which is what makes a
sink bounded instead of 1/R^2 singular at the patch.

WHAT THIS IS NOT
----------------
Not a solution of the Navier-Stokes equations, not calibrated against CASE-01,
and not claimed to be either. It is the textbook expectation an engineer holds
BEFORE measuring, expressed in a form the GP can use as a prior mean. Its
amplitude is selected on the held-out p0 placement (v3 plan P3) precisely
because the profile model is approximate; amp = 0 recovers the plain
divergence-free GP exactly, so P1-vs-plain is a nested comparison.
"""

import numpy as np

from open_airfield.geometry import (
    EXTRACT,
    EXTRACT_VELOCITY,
    LZ,
    SUPPLY,
    SUPPLY_VELOCITY,
    VOLUMETRIC_FLOW,
)

# Round-free-jet Gaussian spreading rate. Velocity half-width b_1/2 = 0.0965 s
# (standard), converted to a Gaussian sigma by b_1/2 / sqrt(2 ln 2).
SPREAD_RATE = 0.0965 / np.sqrt(2.0 * np.log(2.0))  # ~= 0.0820

# Jet taper: the column ends 0.35 m above the floor over a 0.15 m scale.
JET_TAPER_S = LZ - 0.35
JET_TAPER_W = 0.15

# Extract draw depth: a ceiling extract does not reach the far field.
EXTRACT_TAPER_S = 1.0
EXTRACT_TAPER_W = 0.25

_R_FLOOR = 1e-12  # m, removes the 0/0 in u_r on the axis (the limit is 0)


def flux_matched_sigma0(flow_rate: float, face_velocity: float) -> float:
    """Gaussian width carrying `flow_rate` at centreline `face_velocity`.

    Integral of U0 exp(-r^2/2 sigma^2) over the plane = U0 * 2 pi sigma^2 = Q.
    Derived from the declared flow condition, so it reads no truth.
    """
    return float(np.sqrt(flow_rate / (2.0 * np.pi * face_velocity)))


class VerticalColumn:
    """One divergence-free axisymmetric column hung from a ceiling patch.

    amp > 0 drives flow DOWNWARD (increasing s, away from the ceiling): the
    supply jet. amp < 0 drives flow toward the ceiling: the extract.
    """

    def __init__(
        self,
        centre: tuple,
        u0: float,
        sigma0: float,
        amp: float,
        taper_s: float,
        taper_w: float,
        spread: float = SPREAD_RATE,
    ):
        self.cx, self.cy, self.cz = centre
        self.u0 = u0
        self.sigma0 = sigma0
        self.amp = amp
        self.taper_s = taper_s
        self.taper_w = taper_w
        self.spread = spread

    # --- shape functions, and their s-derivatives, analytic ----------------

    def _sigma(self, s):
        return self.sigma0 + self.spread * s

    def _taper(self, s):
        return 0.5 * (1.0 - np.tanh((s - self.taper_s) / self.taper_w))

    def _dtaper(self, s):
        return -0.5 / (self.taper_w * np.cosh((s - self.taper_s) / self.taper_w) ** 2)

    def _G(self, s):
        """psi = G(s) * (1 - exp(-r^2/2 sigma^2)).

        G = amp * T * U_c * sigma^2, and U_c * sigma^2 = U0 * sigma0 * sigma
        exactly (momentum-conserving decay), so G is linear in sigma and G'
        is closed form rather than differenced.
        """
        return self.amp * self._taper(s) * self.u0 * self.sigma0 * self._sigma(s)

    def _dG(self, s):
        return (
            self.amp
            * self.u0
            * self.sigma0
            * (self._dtaper(s) * self._sigma(s) + self._taper(s) * self.spread)
        )

    def velocity(self, points: np.ndarray) -> np.ndarray:
        """(N,3) m -> (N,3) m/s. Exactly divergence-free by construction."""
        p = np.asarray(points, dtype=np.float64)
        dx = p[:, 0] - self.cx
        dy = p[:, 1] - self.cy
        s = self.cz - p[:, 2]  # distance below the ceiling patch
        r = np.sqrt(dx * dx + dy * dy)
        r_safe = np.maximum(r, _R_FLOOR)

        sig = self._sigma(s)
        G, dG = self._G(s), self._dG(s)
        x = r * r / (2.0 * sig * sig)
        E = np.exp(-x)
        one_minus_E = -np.expm1(-x)  # accurate as r -> 0, where 1-E ~ r^2/2sig^2

        # u_s = (1/r) dpsi/dr = G * E / sigma^2, which equals amp*T*U_c*E:
        # a Gaussian axial profile peaking at the centreline velocity.
        u_s = G * E / (sig * sig)
        # u_r = -(1/r) dpsi/ds
        u_r = -(dG * one_minus_E / r_safe - G * E * r * self.spread / sig**3)

        # s increases DOWNWARD, so +u_s is -z. Radial unit vector from the axis.
        out = np.zeros_like(p)
        out[:, 0] = u_r * dx / r_safe
        out[:, 1] = u_r * dy / r_safe
        out[:, 2] = -u_s
        return out


class VentilationPrior:
    """CASE-01's declared supply jet + ceiling extract, superposed.

    A superposition of divergence-free fields is divergence-free, so the sum
    keeps the property each part has.
    """

    def __init__(self, amp: float = 1.0, spread: float = SPREAD_RATE):
        self.amp = amp
        sigma0 = flux_matched_sigma0(VOLUMETRIC_FLOW, SUPPLY_VELOCITY)
        self.sigma0 = sigma0
        self.jet = VerticalColumn(
            centre=(SUPPLY.x, SUPPLY.y, SUPPLY.z),
            u0=SUPPLY_VELOCITY,
            sigma0=sigma0,
            amp=amp,
            taper_s=JET_TAPER_S,
            taper_w=JET_TAPER_W,
            spread=spread,
        )
        self.extract = VerticalColumn(
            centre=(EXTRACT.x, EXTRACT.y, EXTRACT.z),
            u0=EXTRACT_VELOCITY,
            sigma0=flux_matched_sigma0(VOLUMETRIC_FLOW, EXTRACT_VELOCITY),
            amp=-amp,
            taper_s=EXTRACT_TAPER_S,
            taper_w=EXTRACT_TAPER_W,
            spread=spread,
        )

    def velocity(self, points: np.ndarray) -> np.ndarray:
        if self.amp == 0.0:
            return np.zeros((len(points), 3))
        return self.jet.velocity(points) + self.extract.velocity(points)


def divergence(field_fn, points: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Central-difference divergence of any (N,3)->(N,3) field. Test surface."""
    p = np.asarray(points, dtype=np.float64)
    div = np.zeros(len(p))
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        div += (field_fn(p + step)[:, k] - field_fn(p - step)[:, k]) / (2.0 * h)
    return div
