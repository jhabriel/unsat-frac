import numpy as np
import porepy as pp


def capillary_threshold(sat_hydraulic_cond: float):
    """
    Capillary threshold as a function of the saturated hydraulic conductivity [1].

    Parameters:
    -----------
        sat_hydraulic_cond (float): Saturated hydraulic conductivity in [m/s].

    Returns:
    --------
        psi (float): Capillary threshold pressure head [m]

    Reference:
    ----------
        [1] T. K. Tokunaga and J. Wan, “Approximate boundaries between different flow regimes
            in fractured rocks,” Water Resour. Res., vol. 37, no. 8, pp. 2103–2111, 2001.
    """

    # Constants
    DENSITY: float = 1000.0  # [kg/m^3]
    GRAVITY: float = 9.81  # [m/s^2]
    DYNAMIC_VISCOSITY: float = 0.001  # [Pa s]

    # Intrinsic permeability
    intrinsic_perm = (sat_hydraulic_cond * DYNAMIC_VISCOSITY) / (
        DENSITY * GRAVITY
    )  # [m^2]

    # Pressure
    p = -(0.5**0.64) * intrinsic_perm ** (-0.32)  # [Pa]

    # Pressure head
    psi = p / (DENSITY * GRAVITY)  # [m]

    return psi
