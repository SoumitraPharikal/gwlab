"""Physical and astronomical constants for gravitational-wave inference.

All constants are specified in standard SI units (m, kg, s) unless explicitly
noted otherwise.
"""

from __future__ import annotations

import math

# Speed of light in vacuum (m / s) - SI exact definition
C_SI: float = 299_792_458.0

# Newtonian constant of gravitation (m^3 / (kg * s^2)) - CODATA
G_SI: float = 6.67430e-11

# Standard solar mass (kg)
M_SUN_SI: float = 1.9884099e30

# 1 Megaparsec in meters (m)
MPC_SI: float = 3.085677581491367e22

# Euler-Mascheroni constant (dimensionless)
GAMMA_E: float = 0.577215664901532860606512090

# Schwarzschild ISCO frequency-mass constant (Hz * M_sun)
# f_ISCO = C_ISCO / (M_total / M_sun)
# Frozen specification constant from Stage 1.3 Phase B Revision 9 (Group C)
C_ISCO_SI: float = 4397.17
