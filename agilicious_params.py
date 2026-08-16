"""Kingfisher vehicle parameters mapped from the public Agilicious/NeuroBEM data.

RotorPy uses a quadratic thrust/torque model plus compact aerodynamic terms.
The rigid-body, geometry, motor, propeller and frame-drag values below come from
the public Kingfisher and NeuroBEM parameter files. RotorPy's rotor-drag terms
are scaled Hummingbird coefficients, because the full nonlinear NeuroBEM blade
element model is not part of RotorPy.
"""

from __future__ import annotations

import numpy as np


MASS_KG = 0.752
GRAVITY = 9.81
PROPELLER_RADIUS_M = 6.477e-2

# NeuroBEM's public quadratic propeller fit.
THRUST_COEFF = 1.562522e-6  # N / (rad/s)^2
YAW_MOMENT_COEFF = 1.908873e-8  # Nm / (rad/s)^2
MAX_THRUST_PER_MOTOR_N = 10.0
MAX_ROTOR_SPEED_RAD_S = float(np.sqrt(MAX_THRUST_PER_MOTOR_N / THRUST_COEFF))

# Exact motor offsets in the NeuroBEM simulator (front-left-up convention in
# RotorPy; the z offset does not change the moment from axial thrust).
DX, DY, DZ = 0.078, 0.100, 0.027

# NeuroBEM frame drag is -0.5*rho*C*A*|v|v. RotorPy expects the combined
# coefficient multiplying -|v|v.
AIR_DENSITY = 1.204
C_DX = 0.5 * AIR_DENSITY * (0.06 * 0.09)
C_DY = 0.5 * AIR_DENSITY * (0.10 * 0.09)
C_DZ = 0.5 * AIR_DENSITY * (0.10 * 0.06)

# RotorPy's compact rotor-aero terms do not have a direct NeuroBEM equivalent.
# Scale the RotorPy Hummingbird reference by disk area and keep these values
# visibly separate from the measured/mapped values above.
_DISK_SCALE = (PROPELLER_RADIUS_M / 0.10) ** 2

KINGFISHER_PARAMS = {
    "mass": MASS_KG,
    "Ixx": 2.54e-3,
    "Iyy": 2.14e-3,
    "Izz": 4.36e-3,
    "Ixy": 0.0,
    "Iyz": 0.0,
    "Ixz": 0.0,
    "num_rotors": 4,
    "rotor_radius": PROPELLER_RADIUS_M,
    "rotor_pos": {
        "r1": np.array([-DX, DY, DZ]),
        "r2": np.array([DX, DY, DZ]),
        "r3": np.array([-DX, -DY, DZ]),
        "r4": np.array([DX, -DY, DZ]),
    },
    "rotor_directions": np.array([1, -1, -1, 1]),
    "rI": np.zeros(3),
    "c_Dx": C_DX,
    "c_Dy": C_DY,
    "c_Dz": C_DZ,
    "k_eta": THRUST_COEFF,
    "k_m": YAW_MOMENT_COEFF,
    "k_d": 1.19e-4 * _DISK_SCALE,
    "k_z": 2.32e-4 * _DISK_SCALE,
    "k_h": 3.39e-3 * _DISK_SCALE,
    "k_flap": 0.0,
    "tau_m": 0.033,
    "rotor_speed_min": 0.0,
    "rotor_speed_max": MAX_ROTOR_SPEED_RAD_S,
    "motor_noise_std": 0.0,
    # Fast inner-loop body-rate response, representing Agilicious's low-level
    # controller while PPO supplies collective thrust and desired body rates.
    "k_w": 8.0,
    "k_v": 10.0,
    "kp_att": 544.0,
    "kd_att": 46.64,
}


def parameter_summary() -> str:
    hover_speed = np.sqrt(MASS_KG * GRAVITY / (4 * THRUST_COEFF))
    return (
        f"mass={MASS_KG:.3f} kg, inertia="
        f"[{KINGFISHER_PARAMS['Ixx']:.5f}, {KINGFISHER_PARAMS['Iyy']:.5f}, "
        f"{KINGFISHER_PARAMS['Izz']:.5f}] kg m^2, prop radius="
        f"{PROPELLER_RADIUS_M:.4f} m, hover speed={hover_speed:.1f} rad/s, "
        f"max thrust={MAX_THRUST_PER_MOTOR_N:.1f} N/motor"
    )
