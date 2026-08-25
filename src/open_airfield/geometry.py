"""CASE-01 geometry and flow condition.

Single source of truth for the room the benchmark is defined on. Every value
here is declared by the CFD author (Nishan Jain, AeroSHILA) on 23 Aug 2026 and
is NOT yet verified against a delivered export. The synthetic stand-in field is
built to these numbers so that swapping in the real case is a change of data
source, not a change of coordinate frame.

Coordinate frame, as declared: origin at a floor corner, x along the 8 m wall,
y along the 7 m wall, z vertical and positive upwards. Domain (0,0,0)-(8,7,3).
"""

from dataclasses import dataclass

# Room extent in metres.
LX, LY, LZ = 8.0, 7.0, 3.0
VOLUME = LX * LY * LZ  # 168 m^3


@dataclass(frozen=True)
class Diffuser:
    """A ceiling supply or extract terminal."""

    x: float
    y: float
    z: float
    width: float
    depth: float

    @property
    def area(self) -> float:
        return self.width * self.depth


# Ceiling supply: 1 m/s downward through 0.6 x 0.6 m.
SUPPLY = Diffuser(x=4.0, y=1.5, z=3.0, width=0.6, depth=0.6)
SUPPLY_VELOCITY = 1.0  # m/s, downward (-z)

# Matching ceiling extract, ~4 m away along the y axis.
EXTRACT = Diffuser(x=4.0, y=5.5, z=3.0, width=0.6, depth=0.6)

# Derived flow condition. These are the numbers that go into the published
# applicability statement, so they are computed here rather than transcribed.
VOLUMETRIC_FLOW = SUPPLY.area * SUPPLY_VELOCITY  # m^3/s
AIR_CHANGES_PER_HOUR = VOLUMETRIC_FLOW * 3600.0 / VOLUME

# Regime, as declared: steady-state RANS, k-omega SST, purely forced.
# No buoyancy, no heat loads, no occupancy, no obstruction.
# The absence of thermal drive is what makes the incompressible RANS mean
# field EXACTLY divergence-free, which is the closure the reconstruction
# rests on. It is a property of this case, not an approximation.
BUOYANCY_DRIVEN = False
STEADY_STATE = True
