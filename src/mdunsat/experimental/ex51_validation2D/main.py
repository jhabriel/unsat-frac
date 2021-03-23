# Importing modules
import numpy as np
import porepy as pp
import itertools

from time import time
from model import model

#%% Functions
def make_constrained_mesh(h=0.1):
    """
    Creates unstructured mesh for a given target mesh size for the case of a
    single vertical fracture embedded in the domain

    Parameters
    ----------
    h : float, optional
        Target mesh size. The default is 0.1.

    Returns
    -------
    gb : PorePy Object
        Porepy grid bucket object.

    """

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network_2d = pp.fracture_importer.network_2d_from_csv("network.csv", domain=domain)

    # Target lengths
    target_h_bound = h
    target_h_fract = h
    mesh_args = {"mesh_size_bound": target_h_bound, "mesh_size_frac": target_h_fract}
    # Construct grid bucket
    gb = network_2d.mesh(mesh_args, constraints=[1, 2])

    return gb


def create_non_matching_gridbucket(h_2d, h_1d, h_mortar):
    """
    Generates a gridbucket containing non-matching grids

    Parameters
    ----------
    h_2d : Float
        Mesh size of the higher-dimensional grid
    h_1d : Float
        Mesh size of the lower-dimensional grid
    h_mortar : Float
        Mesh size of the mortar grid

    Raises
    ------
    Warning
        If the subdomain cells are smaller than the mortar cell

    Returns
    -------
    gb : PorePy object
        Grid bucket

    """

    # Sanity check
    if (h_2d > h_mortar) or (h_1d > h_mortar):
        warning_msg = "Subdomain cell are smaller than mortar cells "
        warning_msg += "and this may lead to inconsistent results."
        raise Warning(warning_msg)

    # NOTE: The easiest way to construct the non-matching gridbucket is to
    # replace the lower-dimensional grid and the mortar grids into the
    # higher-dimensional grid

    # Create a grid bucket using h_2d as target mesh size
    gb_h = make_constrained_mesh(h_2d)
    gl_old = gb_h.grids_of_dimension(1)[0]  # extract 1d-grid
    mg_old = gb_h.get_mortar_grids()[0]  # extract mortar-grid

    # Obtain fracture and mortar grids to be replaced into
    gl_new = make_constrained_mesh(h_1d).grids_of_dimension(1)[0]
    mg_new = make_constrained_mesh(h_mortar).get_mortar_grids()[0]

    # Create the mapping dictionaries
    g_map = {gl_old: gl_new}
    mg_map = {mg_old: mg_new.side_grids}

    # Replace grids
    gb = gb_h.copy()
    gb.replace_grids(g_map=g_map)
    gb.replace_grids(mg_map=mg_map)

    return gb


#%% Defining numerical methods, and obtaining grid buckets
num_methods = ["TPFA", "MPFA", "RT0", "MVEM"]

levels = 5  # coarsening levels
coarsening_factor = 2
h_2d_ref = 0.003125  # reference 2D mesh size
h_1d_ref = h_2d_ref * 1.5  # reference 1D mesh size
h_mortar_ref = h_2d_ref * 2.0  # reference mortar mesh size
h_2d = coarsening_factor ** np.arange(levels) * h_2d_ref
h_1d = coarsening_factor ** np.arange(levels) * h_1d_ref
h_mortar = coarsening_factor ** np.arange(levels) * h_mortar_ref
grid_buckets = []

tic = time()
print("Assembling non-matching grid buckets...", end="")
for counter in range(levels):
    grid_buckets.append(
        create_non_matching_gridbucket(h_2d[counter], h_1d[counter], h_mortar[counter])
    )
grid_buckets = grid_buckets[::-1]
print(f"\u2713 Time {time() - tic}\n")

#%% Create dictionary and initialize fields
d = {k: {} for k in num_methods}
for method in num_methods:
    d[method] = {
        "mesh_size": [],
        "error_estimate_2d": [],
        "true_error_pressure_2d": [],
        "true_error_velocity_2d": [],
        "mesh_size_2d": [],
        "error_estimate_1d": [],
        "true_error_pressure_1d": [],
        "true_error_velocity_1d": [],
        "mesh_size_1d": [],
        "error_estimate_mortar": [],
        "true_error_pressure_mortar": [],
        "true_error_velocity_mortar": [],
        "mesh_size_mortar": [],
        "majorant": [],
        "true_error_pressure": [],
        "true_error_velocity": [],
        "I_eff_pressure": [],
        "I_eff_velocity": [],
        "I_eff_combined": [],
    }

#%% Populate fields (NOTE: This loop may take considerable time)
for i in itertools.product(num_methods, grid_buckets):

    # Print info in the console
    print("Solving with", i[0], "for refinement level", grid_buckets.index(i[1]) + 1)

    # Get hold of errors
    tic = time()
    (
        h_max,
        error_estimate_2d,
        true_error_pressure_2d,
        true_error_velocity_2d,
        mesh_size_2d,
        error_estimate_1d,
        true_error_pressure_1d,
        true_error_velocity_1d,
        mesh_size_1d,
        error_estimates_mortar,
        true_error_pressure_mortar,
        true_error_velocity_mortar,
        mesh_size_mortar,
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
    d[i[0]]["error_estimate_2d"].append(error_estimate_2d)
    d[i[0]]["true_error_pressure_2d"].append(true_error_pressure_2d)
    d[i[0]]["true_error_velocity_2d"].append(true_error_velocity_2d)
    d[i[0]]["mesh_size_2d"].append(mesh_size_2d)
    d[i[0]]["error_estimate_1d"].append(error_estimate_1d)
    d[i[0]]["true_error_pressure_1d"].append(true_error_pressure_1d)
    d[i[0]]["true_error_velocity_1d"].append(true_error_velocity_1d)
    d[i[0]]["mesh_size_1d"].append(mesh_size_1d)
    d[i[0]]["error_estimate_mortar"].append(error_estimates_mortar)
    d[i[0]]["true_error_pressure_mortar"].append(true_error_pressure_mortar)
    d[i[0]]["true_error_velocity_mortar"].append(true_error_velocity_mortar)
    d[i[0]]["mesh_size_mortar"].append(mesh_size_mortar)
    d[i[0]]["majorant"].append(majorant)
    d[i[0]]["true_error_pressure"].append(true_error_pressure)
    d[i[0]]["true_error_velocity"].append(true_error_velocity)
    d[i[0]]["I_eff_pressure"].append(I_eff_pressure)
    d[i[0]]["I_eff_velocity"].append(I_eff_velocity)
    d[i[0]]["I_eff_combined"].append(I_eff_combined)

#%% Exporting

# Permutations
rows = len(num_methods) * len(grid_buckets)

# Intialize lists
num_method_name = []
diam_2d = []
diam_1d = []
diam_mortar = []
col_2d_estimate = []
col_1d_estimate = []
col_mortar_estimate = []
col_majorant = []
col_true_error_pressure = []
col_true_error_velocity = []
I_eff_pressure = []
I_eff_velocity = []
I_eff_combined = []

# Populate lists
for i in itertools.product(num_methods, range(levels)):
    num_method_name.append(i[0])
    diam_2d.append(d[i[0]]["mesh_size_2d"][i[1]])
    diam_1d.append(d[i[0]]["mesh_size_1d"][i[1]])
    diam_mortar.append(d[i[0]]["mesh_size_mortar"][i[1]])
    col_2d_estimate.append(d[i[0]]["error_estimate_2d"][i[1]])
    col_1d_estimate.append(d[i[0]]["error_estimate_1d"][i[1]])
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
        ("var13", float),
        ("var14", float),
    ],
)

# Declaring column variables
export["var2"] = num_method_name
export["var3"] = diam_2d
export["var4"] = diam_1d
export["var5"] = diam_mortar
export["var6"] = col_2d_estimate
export["var7"] = col_1d_estimate
export["var8"] = col_mortar_estimate
export["var9"] = col_majorant
export["var10"] = col_true_error_pressure
export["var11"] = col_true_error_velocity
export["var12"] = I_eff_pressure
export["var13"] = I_eff_velocity
export["var14"] = I_eff_combined


# Formatting string
fmt = "%6s %2.5f %2.5f %2.5f %2.2e %2.2e "
fmt += "%2.2e %2.2e %2.2e %2.2e %2.2f %2.2f %2.2f"

# Headers
header = "num_method h_2d, h_1d, h_mortar, eta_2d eta_1d eta_mortar "
header += "majorant true_error_p true_error_u I_eff_p I_eff_u I_eff_pu"

# Writing into txt
np.savetxt("eff_analysis.txt", export, delimiter=",", fmt=fmt, header=header)
