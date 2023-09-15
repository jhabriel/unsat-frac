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

# Check the different simulations using gravel-like parameters in the fracture cells

"""
Values of SWRC for gravel.

References:
-----------

alpha_vg, S_res, and n_vg were taken from
https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2001WR001242

Porosity was obtained from Wu and Wang (2006) relation based on median size particle
D_50 [mm]:

porosity = 0.13 + (0.21 / (D_50 + 0.002) ** 0.21)

https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010WR009690
https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%282006%29132%3A8%28858%29

########################################################################################

Gravel 2 mm
-----------
theta_sat = 0.3115 [-]
theta_res = 0.05296 [-]  (S_res = 0.17)
alpha = 0.2545 [1/cm]  (1/alpha = 39.3 mm)
n = 4.35 [-]

Gravel 5 mm
-----------
theta_sat = 0.2798 [-]
theta_res = 0.06099 [-]  (S_res = 0.218)
alpha = 0.7874 [1/cm]  (1/alpha =  12.7 mm)
n = 3.99 [-]

Gravel 9 mm
-----------
theta_sat = 0.2624 [-]
theta_res = 0.01155 [-]  (S_res = 0.044)
alpha = 2.222 [1/cm]  (1/alpha = 4.5 mm)
n = 2.13 [-]

"""

# We start with gravel of 2 mm
fracture_parameters = {
    "K_sat": 1E5,  # [cm/s]
    "theta_sat": 0.3115,  # [-]
    "theta_res": 0.05296,  # [-]
    "alpha_vg": 0.2545,  # [1/cm]
    "n_vg": 4.35,  # [-]
    "name": "2mm",
}

#%% Run simulations
simulations = 3  # 3 simulations in total
for i in range(simulations):

    # Update fracture cells hydraulic parameters
    if i == 1:
        fracture_parameters["theta_sat"] = 0.2798  # [-]
        fracture_parameters["theta_res"] = 0.06099  # [-]
        fracture_parameters["alpha_vg"] = 0.7874  # [1/cm]
        fracture_parameters["n_vg"] = 3.99  # [-]
        fracture_parameters["name"] = "5mm"
    elif i == 2:
        fracture_parameters["theta_sat"] = 0.2624  # [-]
        fracture_parameters["theta_res"] = 0.01155  # [-]
        fracture_parameters["alpha_vg"] = 2.222  # [1/cm]
        fracture_parameters["n_vg"] = 2.13  # [-]
        fracture_parameters["name"] = "9mm"

    equidim_model(
        matrix_params=matrix_parameters,
        block_params=block_parameters,
        fracture_params=fracture_parameters,
        out_fname="gravel_" + str(fracture_parameters["name"])
    )
