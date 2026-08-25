"""U1 — synthetic truth field (technical spec §U1).

A closed-form, exactly divergence-free, ventilation-like velocity field on the
CASE-01 domain, so U2-U6 are built and debugged against known truth.

Construction: u = curl(Psi) for a smooth vector potential Psi — divergence-free
IDENTICALLY, for any Psi (div curl = 0). Psi is composed of three
Gaussian-modulated terms:

1. Supply jet — a poloidal potential P_s = -A_s * s(z) * G_s(x, y), where G_s
   is a Gaussian footprint over the supply diffuser and s(z) = (z/LZ)^2
   vanishes at the floor. Contributes a downward jet through the supply centre
   with compensating radial return flow (mass balance forces recirculation in
   a sealed box).
2. Extract drawdown — the mirror term +A_e * s(z) * G_e(x, y) over the
   extract, drawing air upward into the terminal.
3. Interior circulation — a toroidal component Psi_x = B * gx(x) * q(y) * c(z)
   carrying air along the room between supply and extract. q(y) = sin^2(pi
   y/LY) and c(z) = sin^2(pi z/LZ) vanish at every wall they are normal to, so
   the cell contributes no wall-normal flow at all.

Poloidal terms enter as Psi = curl(P zhat) = (dP/dy, -dP/dx, 0), giving
u = (d2P/dxdz, d2P/dydz, -laplacian_xy P). Amplitudes are calibrated so the
speed at the supply centre is exactly SUPPLY_VELOCITY: |u_z| there is
2*A/sigma^2 * s(LZ), hence A = SUPPLY_VELOCITY * sigma^2 / 2.

Wall behaviour by construction: normal flow is identically zero on the floor
(s(0) = c(0) = 0) and on the y-walls (q(0) = q(LY) = 0); Gaussian tails make
it negligible on the x-walls; the ceiling carries normal flow only inside the
diffuser footprints, which are vents, not walls.

Implemented in sympy, derived symbolically, lambdified to numpy [VERIFIED
sympy 1.14 in local repo; same construction is torch-portable].
"""

import numpy as np
import sympy as sp

from open_airfield.geometry import EXTRACT, LY, LZ, SUPPLY, SUPPLY_VELOCITY

# Jet footprint: sigma at half the diffuser half-width, so ~95% of the jet
# passes inside the 0.6 x 0.6 m opening.
SIGMA_JET = 0.3  # m
# Interior circulation cell: peak tangential speed (m/s) and x-extent (m).
CIRC_AMPLITUDE = 0.3
CIRC_SIGMA_X = 2.0


class SyntheticField:
    """TruthField implementation with a closed-form solenoidal velocity."""

    def __init__(
        self,
        sigma_jet: float = SIGMA_JET,
        circ_amplitude: float = CIRC_AMPLITUDE,
        circ_sigma_x: float = CIRC_SIGMA_X,
    ) -> None:
        x, y, z = sp.symbols("x y z", real=True)
        self.symbols = (x, y, z)

        # Exact rationals throughout the symbolic construction, so the
        # divergence cancels ALGEBRAICALLY (float coefficients leave ~1e-19
        # residues that break the symbolic zero check).
        def R(v: float) -> sp.Rational:
            return sp.Rational(str(v))

        # Amplitude such that |u_z| at a diffuser centre equals SUPPLY_VELOCITY.
        amp = R(SUPPLY_VELOCITY) * R(sigma_jet) ** 2 / 2

        def footprint(d):
            return sp.exp(-((x - R(d.x)) ** 2 + (y - R(d.y)) ** 2) / (2 * R(sigma_jet) ** 2))

        s = (z / R(LZ)) ** 2  # vertical jet profile, zero at the floor
        poloidal = amp * s * (-footprint(SUPPLY) + footprint(EXTRACT))

        # Toroidal circulation, wall-normal flow identically zero.
        circ = (
            R(circ_amplitude)
            * (R(LZ) / sp.pi)  # normalise so peak u_y ~= circ_amplitude
            * sp.exp(-((x - R(SUPPLY.x)) ** 2) / (2 * R(circ_sigma_x) ** 2))
            * sp.sin(sp.pi * y / R(LY)) ** 2
            * sp.sin(sp.pi * z / R(LZ)) ** 2
        )

        # Psi = (dP/dy + circ, -dP/dx, 0); u = curl(Psi).
        psi_x = sp.diff(poloidal, y) + circ
        psi_y = -sp.diff(poloidal, x)
        u_sym = (
            -sp.diff(psi_y, z),
            sp.diff(psi_x, z),
            sp.diff(psi_y, x) - sp.diff(psi_x, y),
        )
        self.u_sym = u_sym
        self._u_num = [sp.lambdify((x, y, z), c, "numpy") for c in u_sym]

        self.meta = {
            "name": "synthetic-v1",
            "source": "synthetic",
            "params": {
                "sigma_jet": sigma_jet,
                "jet_amplitude": float(amp),
                "circ_amplitude": circ_amplitude,
                "circ_sigma_x": circ_sigma_x,
                "construction": "u = curl(Psi); poloidal supply/extract + toroidal cell",
            },
        }

    def velocity(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        out = np.empty_like(pts)
        for i, f in enumerate(self._u_num):
            out[:, i] = f(pts[:, 0], pts[:, 1], pts[:, 2])
        return out
