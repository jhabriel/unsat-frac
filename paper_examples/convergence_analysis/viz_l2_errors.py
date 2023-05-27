import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc(
    "font",
    family="Times New Roman",
)
plt.rcParams.update(
    {
        "text.usetex": True,
    }
)

#%% Open pickle file
with open('out/l2_errors.pickle', 'rb') as handle:
    input_list = pickle.load(handle)

errors: dict[str, np.ndarray] = input_list[0]
mesh_sizes: list[float] = input_list[1]

#%% Plotting
fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})

# First order line
x0 = np.log2(1 / np.asarray(mesh_sizes[0]))
x1 = np.log2(1 / np.asarray(mesh_sizes[-1]))
y0 = -12
y1 = y0 - (x1 - x0)

ax[0].plot(
    [x0, x1],
    [y0, y1],
    linewidth=2,
    linestyle="-",
    color="black",
)

ax[1].plot([], [], linewidth=2, linestyle="-", color="black", label="First order")

# Bulk hydraulic head
ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_h_bulk"]),
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    color="orange",
)

ax[1].plot(
    [],
    [],
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    label="Bulk hydraulic head",
    color="orange",
)

# Interface fluxes
ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_q_intf"]),
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    color="green",
)

ax[1].plot(
    [],
    [],
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    label="Interface fluxes",
    color="green",
)

# Fracture hydraulic head
ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_h_frac"]),
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    color="blue",
)

ax[1].plot(
    [],
    [],
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    label="Fracture hydraulic head",
    color="blue",
)

# Bulk flux
ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_q_bulk"]),
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    color="red",
)

ax[1].plot(
    [],
    [],
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    label="Bulk flux",
    color="red",
)

# Fracture volume
ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_vol_frac"]),
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    color="magenta",
)

ax[1].plot(
    [],
    [],
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize="6",
    label="Fracture volume",
    color="magenta",
)

ax[0].set_xlabel(r"$\log_2\left(1/\mathrm{mesh~size}\right)$", fontsize=16)
ax[0].set_ylabel(r"$\log_2\left(\mathrm{error}\right)$", fontsize=16)
ax[1].legend(fontsize=14, loc="center")
ax[1].axis("off")

plt.tight_layout()

plt.savefig("out/convergence_analysis.pdf")
