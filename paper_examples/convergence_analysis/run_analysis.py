import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc(
    "font",
    family="Times New Roman",
)
plt.rcParams.update(
    {
        "text.usetex": True,
    }
)
from model import manufactured_model

#%% Prepare for the analysis
mesh_sizes = [0.05, 0.025, 0.0125, 0.00625, 0.003125]
error_h_bulk = []
error_q_bulk = []
error_q_intf = []
error_h_frac = []
error_vol_frac = []

errors = {
    "error_h_bulk": [],
    "error_q_bulk": [],
    "error_q_intf": [],
    "error_h_frac": [],
    "error_vol_frac": [],
}

#%% Run simulations and retrieve errors
for mesh_size in mesh_sizes:
    out = manufactured_model(mesh_size=mesh_size)
    errors["error_h_bulk"].append(out["error_h_bulk"])
    errors["error_q_bulk"].append(out["error_q_bulk"])
    errors["error_q_intf"].append(out["error_q_intf"])
    errors["error_h_frac"].append(out["error_h_frac"])
    errors["error_vol_frac"].append(out["error_vol_frac"])

#%% Save errors to reproduce results

# Create the directory if it does not exist
if not os.path.exists("out"):
    os.makedirs("out")

# Dump into the pickle file
with open("out/l2_errors.pickle", "wb") as handle:
    pickle.dump([errors, mesh_sizes], handle, protocol=pickle.HIGHEST_PROTOCOL)
