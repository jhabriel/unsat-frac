# Importing modules
import numpy as np
import porepy as pp
import itertools

from time import time
from model import model

#%% Functions
def make_constrained_mesh(mesh_size=0.2):
    """
    Creates an unstructured 3D mesh for a given target mesh size for the case
    of a  single 2D vertical fracture embedded in a 3D domain

    Parameters
    ----------
    mesh_size : float, optional
        Target mesh size. The default is 0.2.

    Returns
    -------
    gb : PorePy Object
        Porepy grid bucket object.

    """
    # Load fracture network: Fracture + Ghost Fractures
    network_3d = pp.fracture_importer.network_3d_from_csv("network.csv")

    # Create mesh_arg dictionary
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_bound": mesh_size,
        "mesh_size_min": mesh_size / 10,
    }

    # Construct grid bucket
    ghost_fracs = list(np.arange(1, 25))  # 1 to 24
    gb = network_3d.mesh(mesh_args, constraints=ghost_fracs)

    return gb


#%% Defining mesh targets, numerical methods, and dictionary fields
mesh_targets = np.array([0.2, 0.15, 0.10, 0.07, 0.04])
num_methods = ["TPFA", "MPFA", "RT0", "MVEM"]

#%% Obtain grid buckets for each mesh size
print("Assembling grid buckets...", end="")
tic = time()
grid_buckets = []
for h in mesh_targets:
    grid_buckets.append(make_constrained_mesh(h))
print(f"\u2713 Time {time() - tic}.\n")

#%% Create dictionary and initialize fields
d = {k: {} for k in num_methods}
for method in num_methods:
    d[method] = {
        "mesh_size": [],
        "error_estimate_3d": [],
        "true_error_pressure_3d": [],
        "true_error_velocity_3d": [],
        "num_cells_3d": [],
        "error_estimate_2d": [],
        "true_error_pressure_2d": [],
        "true_error_velocity_2d": [],
        "num_cells_2d": [],
        "error_estimate_mortar": [],
        "true_error_pressure_mortar": [],
        "true_error_velocity_mortar": [],
        "num_cells_mortar": [],
        "majorant": [],
        "true_error_pressure": [],
        "true_error_velocity": [],
        "I_eff_pressure": [],
        "I_eff_velocity": [],
        "I_eff_combined": [],
    }

#%% Populate fields (Warning: This loop may take considerable time)
for i in itertools.product(num_methods, grid_buckets):

    # Print info in the console
    print("Solving with", i[0], "for mesh size:", i[1].diameter())

    # Get hold of errors
    tic = time()
    (
        h_max,
        error_estimate_3d,
        true_error_pressure_3d,
        true_error_velocity_3d,
        num_cells_3d,
        error_estimate_2d,
        true_error_pressure_2d,
        true_error_velocity_2d,
        num_cells_2d,
        error_estimates_mortar,
        true_error_pressure_mortar,
        true_error_velocity_mortar,
        num_cells_mortar,
        majorant,
        true_error_pressure,
        true_error_velocity,
        I_eff_pressure,
        I_eff_velocity,
        I_eff_combined,
    ) = model(i[1], i[0])
    print(f"Done. Time {time() - tic}\n")

    # Store errors in the dictionary
    d[i[0]]["mesh_size"].append(h_max)
    d[i[0]]["error_estimate_3d"].append(error_estimate_3d)
    d[i[0]]["true_error_pressure_3d"].append(true_error_pressure_3d)
    d[i[0]]["true_error_velocity_3d"].append(true_error_velocity_3d)
    d[i[0]]["num_cells_3d"].append(num_cells_3d)
    d[i[0]]["error_estimate_2d"].append(error_estimate_2d)
    d[i[0]]["true_error_pressure_2d"].append(true_error_pressure_2d)
    d[i[0]]["true_error_velocity_2d"].append(true_error_velocity_2d)
    d[i[0]]["num_cells_2d"].append(num_cells_2d)
    d[i[0]]["error_estimate_mortar"].append(error_estimates_mortar)
    d[i[0]]["true_error_pressure_mortar"].append(true_error_pressure_mortar)
    d[i[0]]["true_error_velocity_mortar"].append(true_error_velocity_mortar)
    d[i[0]]["num_cells_mortar"].append(num_cells_mortar)
    d[i[0]]["majorant"].append(majorant)
    d[i[0]]["true_error_pressure"].append(true_error_pressure)
    d[i[0]]["true_error_velocity"].append(true_error_velocity)
    d[i[0]]["I_eff_pressure"].append(I_eff_pressure)
    d[i[0]]["I_eff_velocity"].append(I_eff_velocity)
    d[i[0]]["I_eff_combined"].append(I_eff_combined)


#%% Exporting

# Permutations
rows = len(num_methods) * len(mesh_targets)

# Initialize lists
num_method_name = []
h_max = []
col_3d_estimate = []
col_2d_estimate = []
col_mortar_estimate = []
col_majorant = []
col_true_error_pressure = []
col_true_error_velocity = []
I_eff_pressure = []
I_eff_velocity = []
I_eff_combined = []

# Populate lists
for i in itertools.product(num_methods, range(len(grid_buckets))):
    num_method_name.append(i[0])
    h_max.append(d[i[0]]["mesh_size"][i[1]])
    col_3d_estimate.append(d[i[0]]["error_estimate_3d"][i[1]])
    col_2d_estimate.append(d[i[0]]["error_estimate_2d"][i[1]])
    col_mortar_estimate.append(d[i[0]]["error_estimate_mortar"][i[1]])
    col_majorant.append(d[i[0]]["majorant"][i[1]])
    col_true_error_pressure.append(d[i[0]]["true_error_pressure"][i[1]])
    col_true_error_velocity.append(d[i[0]]["true_error_velocity"][i[1]])
    I_eff_pressure.append(d[i[0]]["I_eff_pressure"][i[1]])
    I_eff_velocity.append(d[i[0]]["I_eff_velocity"][i[1]])
    I_eff_combined.append(d[i[0]]["I_eff_combined"][i[1]])

# Prepare for exporting
export = np.zeros(
    rows,
    dtype=[
        ("var2", "U6"),
        ("var3", float),
        ("var4", float),
        ("var5", float),
        ("var6", float),
        ("var7", float),
        ("var8", float),
        ("var9", float),
        ("var10", float),
        ("var11", float),
        ("var12", float),
    ],
)

# Declare column variables
export["var2"] = num_method_name
export["var3"] = h_max
export["var4"] = col_3d_estimate
export["var5"] = col_2d_estimate
export["var6"] = col_mortar_estimate
export["var7"] = col_majorant
export["var8"] = col_true_error_pressure
export["var9"] = col_true_error_velocity
export["var10"] = I_eff_pressure
export["var11"] = I_eff_velocity
export["var12"] = I_eff_combined


# Formatting string
fmt = "%6s %2.3f %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e %2.2f %2.2f %2.2f"

# Headers
header = "num_method h_max eta_3d eta_2d eta_mortar majorant true_error_p "
header += "true_error_u  I_eff_p I_eff_u I_eff_pu"

# Writing into txt
np.savetxt("eff_analysis.txt", export, delimiter=",", fmt=fmt, header=header)
