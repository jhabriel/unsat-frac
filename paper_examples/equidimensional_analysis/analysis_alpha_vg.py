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

# Use gravel-like parameters in the fracture cells
# https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2001WR001242

fracture_parameters = {
    "K_sat": 1E5,  # [cm/s]
    "theta_sat": 0.4,  # [-]
    "theta_res": 0,  # [-]
    "alpha_vg": 0.2,  # [1/cm]
    "n_vg": 3.0,  # [-]
}

#%% Simulation loop
simulations = 1  # 3 simulations in total
for i in range(simulations):

    # Vary the alpha vanGenuchten parameter in each simulation
    if i == 1:
        fracture_parameters["alpha_vg"] = 1  # [1/cm]
    elif i == 2:
        fracture_parameters["alpha_vg"] = 10  # [1/cm]

    # Run the model
    equidim_model(
        matrix_params=matrix_parameters,
        block_params=block_parameters,
        fracture_params=fracture_parameters,
        out_fname="alpha_" + str(fracture_parameters["alpha_vg"])
    )
