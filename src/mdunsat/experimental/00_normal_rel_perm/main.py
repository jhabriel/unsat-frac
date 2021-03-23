 # Importing modules
import numpy as np
import porepy as pp
import itertools
import matplotlib.pyplot as plt

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
    network_2d = pp.fracture_importer.network_2d_from_csv(
        "network.csv", domain=domain
    )
    
    # Target lengths
    target_h_bound = h
    target_h_fract = h
    mesh_args = {
        "mesh_size_bound": target_h_bound,
        "mesh_size_frac": target_h_fract,
    }
    # Construct grid bucket
    gb = network_2d.mesh(mesh_args, constraints=[1, 2])

    return gb

#%% Defining numerical methods, and obtaining grid buckets
#num_methods = ['TPFA', 'MPFA', 'RT0', 'MVEM']
num_methods = ['MPFA']
mesh_targets = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
krw_orders = [0, 1, 2]

grid_buckets = []
tic = time()
print("Assembling grid buckets...", end="")
for mesh_target in mesh_targets:
    grid_buckets.append(make_constrained_mesh(mesh_target))
print(f"\u2713 Time {time() - tic}\n")

#%% Create dictionary and initialize fields
d = {k: {} for k in num_methods}
for order in krw_orders:
    for method in num_methods:
        d[method][order] = {}
        
for i in itertools.product(num_methods, krw_orders):
    d[i[0]][i[1]] = {
        #"mesh_size": [ ],
        #"error_estimate_2d": [ ], 
        #"true_error_pressure_2d": [ ], 
        #"true_error_velocity_2d": [ ], 
        #"mesh_size_2d": [ ],
        #"error_estimate_1d": [ ], 
        #"true_error_pressure_1d": [ ],
        #"true_error_velocity_1d": [ ],
        #"mesh_size_1d": [ ],
        "error_estimate_mortar": [ ], 
        "true_error_pressure_mortar": [ ], 
        "true_error_velocity_mortar": [ ], 
        "mesh_size_mortar": [ ],
        #"majorant": [ ], 
        #"true_error_pressure": [ ], 
        #"true_error_velocity": [ ],
        #"I_eff_pressure": [ ], 
        #"I_eff_velocity": [ ],
        #"I_eff_combined": [ ],
        }

#%% Populate fields (NOTE: This loop may take considerable time)
for i in itertools.product(grid_buckets, num_methods, krw_orders):
    
    # Print info in the console
    print('Solving with', i[1], 'for mesh size', i[0].diameter(), 'and order', i[2])
    
    # Get hold of errors
    tic = time()
    (h_max, 
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
     I_eff_combined
     ) = model(i[0], i[1], i[2])
    print(f'Done. Time {time() - tic}\n')
    
    # Store errors in the dictionary
    # d[i[1]][i[2]]["mesh_size"].append(h_max)
    # d[i[1]][i[2]]['error_estimate_2d'].append(error_estimate_2d)
    # d[i[1]][i[2]]["true_error_pressure_2d"].append(true_error_pressure_2d)
    # d[i[1]][i[2]]["true_error_velocity_2d"].append(true_error_velocity_2d)
    # d[i[1]][i[2]]['mesh_size_2d'].append(mesh_size_2d)
    # d[i[1]][i[2]]['error_estimate_1d'].append(error_estimate_1d)
    # d[i[1]][i[2]]["true_error_pressure_1d"].append(true_error_pressure_1d)
    # d[i[1]][i[2]]["true_error_velocity_1d"].append(true_error_velocity_1d)
    # d[i[1]][i[2]]["mesh_size_1d"].append(mesh_size_1d)
    d[i[1]][i[2]]['error_estimate_mortar'].append(error_estimates_mortar)
    d[i[1]][i[2]]["true_error_pressure_mortar"].append(true_error_pressure_mortar)
    d[i[1]][i[2]]["true_error_velocity_mortar"].append(true_error_velocity_mortar)
    d[i[1]][i[2]]["mesh_size_mortar"].append(mesh_size_mortar)
    # d[i[1]][i[2]]['majorant'].append(majorant)
    # d[i[1]][i[2]]['true_error_pressure'].append(true_error_pressure)
    # d[i[1]][i[2]]['true_error_velocity'].append(true_error_velocity)
    # d[i[1]][i[2]]["I_eff_pressure"].append(I_eff_pressure)
    # d[i[1]][i[2]]["I_eff_velocity"].append(I_eff_velocity)
    # d[i[1]][i[2]]["I_eff_combined"].append(I_eff_combined)
    
    
#%% Plotting for MPFA

x1 = np.log(np.max(1/(np.asarray(d['MPFA'][0]['mesh_size_mortar']))))
y1 = np.log(1e-5)
y2 = np.log(3e-4)
x2_lin = x1 - (y2 - y1)
x2_qua = x1 - 0.5*(y2 - y1)

error_0 = np.asarray(d['MPFA'][0]["true_error_velocity_mortar"])
error_1 = np.asarray(d['MPFA'][1]["true_error_velocity_mortar"])
error_2 = np.asarray(d['MPFA'][2]["true_error_velocity_mortar"])
mesh_size = np.asarray(d['MPFA'][0]["mesh_size_mortar"])

fig, ax = plt.subplots()
# ax.loglog(
#     [np.exp(x1), np.exp(x2_lin)], [np.exp(y1), np.exp(y2)], 
#     color='k', linewidth=2, label="Linear"
#     )
# ax.loglog(
#     [np.exp(x1), np.exp(x2_qua)], [np.exp(y1), np.exp(y2)], 
#     color='m', linewidth=2, label="Quadratic"
#     )
ax.loglog(1/mesh_size, error_0, linewidth=2, marker='s', label=r'$\kappa=1$')
ax.loglog(1/mesh_size, error_1, linewidth=2, marker='s', label=r'$\kappa=h$')
ax.loglog(1/mesh_size, error_2, linewidth=2, marker='s', label=r'$\kappa=h^2$')
ax.set(xlabel=r'$1/h_\Gamma$', ylabel=r'$||\lambda - \lambda_h||_{L^2(\Gamma)}$',
       title=r'Convergence of mortars in the $L^2$-norm',
       xlim=(5, 400))
ax.grid()
ax.legend(loc='lower left')
fig.tight_layout()
fig.save_fig('mpfa_convergence.pdf')

#%% Plotting for MPFA

x1 = np.log(np.max(1/(np.asarray(d['MPFA'][0]['mesh_size_mortar']))))
y1 = np.log(1e-5)
y2 = np.log(3e-4)
x2_lin = x1 - (y2 - y1)
x2_qua = x1 - 0.5*(y2 - y1)

error_0 = np.asarray(d['MPFA'][0]["true_error_pressure_mortar"])
error_1 = np.asarray(d['MPFA'][1]["true_error_pressure_mortar"])
error_2 = np.asarray(d['MPFA'][2]["true_error_pressure_mortar"])
mesh_size = np.asarray(d['MPFA'][0]["mesh_size_mortar"])

fig, ax = plt.subplots()
# ax.loglog(
#     [np.exp(x1), np.exp(x2_lin)], [np.exp(y1), np.exp(y2)], 
#     color='k', linewidth=2, label="Linear"
#     )
# ax.loglog(
#     [np.exp(x1), np.exp(x2_qua)], [np.exp(y1), np.exp(y2)], 
#     color='m', linewidth=2, label="Quadratic"
#     )
ax.loglog(1/mesh_size, error_0, linewidth=2, marker='s', label=r'$\kappa=1$')
ax.loglog(1/mesh_size, error_1, linewidth=2, marker='s', label=r'$\kappa=h$')
ax.loglog(1/mesh_size, error_2, linewidth=2, marker='s', label=r'$\kappa=h^2$')
ax.set(xlabel=r'$1/h_\Gamma$', ylabel=r'$||\lambda - \lambda_h||_{L^2(\Gamma)}$',
       title=r'Convergence of mortars in the $\kappa$-weighted norm',
       xlim=(5, 400))
ax.grid()
ax.legend(loc='lower left')
fig.tight_layout()
fig.save_fig('mpfa_convergence_kappa_weighted.pdf')


#%% Exporting

# # Permutations
# rows = len(num_methods) * len(grid_buckets)

# # Intialize lists
# num_method_name = []
# diam_2d = []
# diam_1d = []
# diam_mortar = []
# col_2d_estimate = []
# col_1d_estimate = []
# col_mortar_estimate = []
# col_majorant = []
# col_true_error_pressure = []
# col_true_error_velocity = []
# I_eff_pressure = []
# I_eff_velocity = []
# I_eff_combined = []

# # Populate lists
# for i in itertools.product(num_methods, range(len(grid_buckets))):
#     num_method_name.append(i[0])
#     diam_2d.append(d[i[0]]["mesh_size_2d"][i[1]])
#     diam_1d.append(d[i[0]]["mesh_size_1d"][i[1]])
#     diam_mortar.append(d[i[0]]["mesh_size_mortar"][i[1]])
#     col_2d_estimate.append(d[i[0]]["error_estimate_2d"][i[1]])
#     col_1d_estimate.append(d[i[0]]["error_estimate_1d"][i[1]])
#     col_mortar_estimate.append(d[i[0]]["error_estimate_mortar"][i[1]])
#     col_majorant.append(d[i[0]]["majorant"][i[1]])
#     col_true_error_pressure.append(d[i[0]]["true_error_pressure"][i[1]])
#     col_true_error_velocity.append(d[i[0]]["true_error_velocity"][i[1]])
#     I_eff_pressure.append(d[i[0]]["I_eff_pressure"][i[1]])
#     I_eff_velocity.append(d[i[0]]["I_eff_velocity"][i[1]])
#     I_eff_combined.append(d[i[0]]["I_eff_combined"][i[1]])

# # Prepare for exporting
# export = np.zeros(rows, 
#               dtype=[ ('var2', 'U6'), 
#                       ('var3', float), ('var4', float), 
#                       ('var5', float), ('var6', float), 
#                       ('var7', float), ('var8', float),
#                       ('var9', float), ('var10', float),
#                       ('var11', float), ('var12', float),
#                       ('var13', float), ('var14', float)
#                       ])

# # Declaring column variables
# export['var2'] = num_method_name
# export['var3'] = diam_2d
# export['var4'] = diam_1d
# export['var5'] = diam_mortar
# export['var6'] = col_2d_estimate
# export['var7'] = col_1d_estimate
# export['var8'] = col_mortar_estimate
# export['var9'] = col_majorant
# export['var10'] = col_true_error_pressure
# export['var11'] = col_true_error_velocity
# export['var12'] = I_eff_pressure
# export['var13'] = I_eff_velocity
# export['var14'] = I_eff_combined


# # Formatting string
# fmt = "%6s %2.5f %2.5f %2.5f %2.2e %2.2e "
# fmt += "%2.2e %2.2e %2.2e %2.2e %2.2f %2.2f %2.2f"

# # Headers
# header = "num_method h_2d, h_1d, h_mortar, eta_2d eta_1d eta_mortar "
# header += "majorant true_error_p true_error_u I_eff_p I_eff_u I_eff_pu"

# # Writing into txt
# np.savetxt('eff_analysis.txt', export, delimiter=',', fmt=fmt, header=header)

# #%% Exporting to Latex

# # Permutations
# rows = len(num_methods) * len(grid_buckets)

# # Intialize lists
# ampersend = []
# for i in range(rows): ampersend.append('&')
# num_method_name = []
# diam_2d = []
# diam_1d = []
# diam_mortar = []
# col_2d_estimate = []
# col_1d_estimate = []
# col_mortar_estimate = []
# col_majorant = []
# col_true_error_pressure = []
# col_true_error_velocity = []
# I_eff_pressure = []
# I_eff_velocity = []
# I_eff_combined = []

# # Populate lists
# for i in itertools.product(num_methods, range(len(grid_buckets))):
#     num_method_name.append(i[0])
#     diam_2d.append(d[i[0]]["mesh_size_2d"][i[1]])
#     diam_1d.append(d[i[0]]["mesh_size_1d"][i[1]])
#     diam_mortar.append(d[i[0]]["mesh_size_mortar"][i[1]])
#     col_2d_estimate.append(d[i[0]]["error_estimate_2d"][i[1]])
#     col_1d_estimate.append(d[i[0]]["error_estimate_1d"][i[1]])
#     col_mortar_estimate.append(d[i[0]]["error_estimate_mortar"][i[1]])
#     col_majorant.append(d[i[0]]["majorant"][i[1]])
#     col_true_error_pressure.append(d[i[0]]["true_error_pressure"][i[1]])
#     col_true_error_velocity.append(d[i[0]]["true_error_velocity"][i[1]])
#     I_eff_pressure.append(d[i[0]]["I_eff_pressure"][i[1]])
#     I_eff_velocity.append(d[i[0]]["I_eff_velocity"][i[1]])
#     I_eff_combined.append(d[i[0]]["I_eff_combined"][i[1]])


# # Prepare for exporting
# export = np.zeros(rows, 
#               dtype=[ ('var2', 'U6'), 
#                       ('var3', float), ('var4', float), 
#                       ('var5', float), ('var6', float), 
#                       ('amp1', 'U6'), ('var7', float), 
#                       ('amp2', 'U6'), ('var8', float),
#                       ('amp3', 'U6'), ('var9', float), 
#                       ('amp4', 'U6'), ('var10', float),
#                       ('amp5', 'U6'), ('var11', float),
#                       ('amp6', 'U6'), ('var12', float),
#                       ('amp7', 'U6'), ('var13', float),
#                       ('amp8', 'U6'), ('var14', float)
#                       ])

# # Prepare for exporting
# export['var2'] = num_method_name
# export['var3'] = diam_2d
# export['var4'] = diam_1d
# export['var5'] = diam_mortar
# export['var6'] = col_2d_estimate
# export['amp1'] = ampersend
# export['var7'] = col_1d_estimate
# export['amp2'] = ampersend
# export['var8'] = col_mortar_estimate
# export['amp3'] = ampersend
# export['var9'] = col_majorant
# export['amp4'] = ampersend
# export['var10'] = col_true_error_pressure
# export['amp5'] = ampersend
# export['var11'] = col_true_error_velocity
# export['amp6'] = ampersend
# export['var12'] = I_eff_pressure
# export['amp7'] = ampersend
# export['var13'] = I_eff_velocity
# export['amp8'] = ampersend
# export['var14'] = I_eff_combined


# # Formatting string
# fmt = "%6s %2.5f %2.5f %2.5f %2.2e %1s %2.2e %1s %2.2e "
# fmt += "%1s %2.2e %1s %2.2e %1s %2.2e %1s %2.2f %1s %2.2f %1s %2.2f"

# # Headers
# header = "num_method h_2d h_1d h_mortar eta_2d & eta_1d & eta_mortar & "
# header += "majorant & true_error_p & true_error_u & I_eff_p & I_eff_u & I_eff_pu"

# np.savetxt('eff_analysis_tex.txt', 
#             export,
#             delimiter=',',
#             fmt=fmt,
#             header=header
#             )