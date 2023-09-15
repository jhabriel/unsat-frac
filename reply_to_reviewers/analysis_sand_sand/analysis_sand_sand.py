import numpy as np

from mdunsat.soil_catalog import soil_catalog
from equidimensional_model import equidim_model

#%% PARAMETERS
MATRIX_SOIL = "new_mexico"
BLOCK_SOIL = "sandy_clay_loam"

matrix_parameters = {
    "K_sat": soil_catalog[MATRIX_SOIL].K_s / 3600,  # [cm/s]
    "theta_sat": soil_catalog[MATRIX_SOIL].theta_sat,  # [-]
    "theta_res": soil_catalog[MATRIX_SOIL].theta_r,  # [-]
    "alpha_vg": soil_catalog[MATRIX_SOIL].alpha,  # [1/cm]
    "n_vg": soil_catalog[MATRIX_SOIL].n,  # [-]
}

block_parameters = {
    "K_sat": soil_catalog[BLOCK_SOIL].K_s / 3600,  # [cm/s]
    "theta_sat": soil_catalog[BLOCK_SOIL].theta_sat,  # [-]
    "theta_res": soil_catalog[BLOCK_SOIL].theta_r,  # [-]
    "alpha_vg": soil_catalog[BLOCK_SOIL].alpha,  # [1/cm]
    "n_vg": soil_catalog[BLOCK_SOIL].n,  # [-]
}

# Check the different simulations using the same material in the fracture
fracture_parameters = matrix_parameters.copy()
fracture_parameters["K_sat"] = 1E2

#%% Prepare to
simulations = 6  # 6 simulations in total
increase_factor = 10  # increase conductivity in the fracture cells by a factor of 10
for i in range(simulations):

    equidim_model(
        matrix_params=matrix_parameters,
        block_params=block_parameters,
        fracture_params=fracture_parameters,
        out_fname="KSAT_1E" + str(int(np.log10(fracture_parameters["K_sat"])))
    )

    # Increase hydraulic conductivity in the fracture cells
    fracture_parameters["K_sat"] *= increase_factor