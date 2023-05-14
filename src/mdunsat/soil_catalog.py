"""
Catalog of unsaturated hydraulic properties for different soils.

Data source:
    [1] : https://doi.org/10.1029/WR024i005p00755
    [2] : https://doi.org/10.1029/WR026i007p01483

"""

from dataclasses import dataclass


@dataclass
class Soil:
    """Class for representing the hydraulic properties of a soil."""

    K_s: float
    """Saturated hydraulic conductivity in [cm/h]."""

    theta_sat: float
    """Saturated water content."""

    theta_r: float
    """Residual water content."""

    alpha: float
    """vanGenuchten-Mualem ``alpha`` parameter in [1/cm]."""

    n: float
    """vanGenuchten-Mualem ``n`` parameter."""


soil_catalog: dict[str, Soil] = {
    "clay": Soil(0.20, 0.38, 0.068, 0.008, 1.09),  # [1]
    "clay_loam": Soil(0.26, 0.41, 0.095, 0.019, 1.31),  # [1]
    "loam": Soil(1.04, 0.43, 0.078, 0.036, 1.56),  # [1]
    "loamy_sand": Soil(14.59, 0.41, 0.057, 0.124, 2.28),  # [1]
    "silt": Soil(0.25, 0.46, 0.034, 0.016, 1.37),  # [1]
    "silt_loam": Soil(0.45, 0.45, 0.067, 0.020, 1.41),  # [1]
    "silty_clay": Soil(0.02, 0.36, 0.070, 0.005, 1.09),  # [1]
    "silty_clay_loam": Soil(0.07, 0.43, 0.089, 0.010, 1.23),  # [1]
    "sand": Soil(29.70, 0.43, 0.045, 0.145, 2.68),  # [1]
    "sandy_clay": Soil(0.12, 0.38, 0.100, 0.027, 1.23),  # [1]
    "sandy_clay_loam": Soil(1.31, 0.39, 0.100, 0.059, 1.48),  # [1]
    "sandy_loam": Soil(4.42, 0.41, 0.065, 0.075, 1.89),  # [1]
    "new_mexico": Soil(33.1920, 0.368, 0.102, 0.0335, 2.0),  # [2]
}
