import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc(
    "font", family="Times New Roman",
)
plt.rcParams.update({
    "text.usetex": True,
})
from model import manufactured_model

#%% Prepare for the analysis
mesh_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
error_h_bulk = []
error_q_bulk = []
error_q_intf = []
error_h_frac = []
error_vol_frac = []

errors = {
    "Bulk hydraulic head": [],
    "Bulk flux": [],
    "Interface flux": [],
    "Fracture hydraulic head": [],
    "Fracture volume": [],
}

#%% Run simulations and retrieve errors
for mesh_size in mesh_sizes:
    out = manufactured_model(mesh_size=mesh_size)
    errors["Bulk hydraulic head"].append(out["error_h_bulk"])
    errors["Bulk flux"].append(out["error_q_bulk"])
    errors["Interface flux"].append(out["error_q_intf"])
    errors["Fracture hydraulic head"].append(out["error_h_frac"])
    errors["Fracture volume"].append(out["error_vol_frac"])

#%% Plotting
fig, ax = plt.subplots(1, 2)

# First order line
x0 = np.log2(1 / np.asarray(mesh_sizes[0]))
x1 = np.log2(1 / np.asarray(mesh_sizes[-1]))
y0 = -9.0
y1 = y0 - (x1 - x0)
ax[0].plot(
    [x0, x1],
    [y0, y1],
    linewidth=2,
    linestyle="--",
    color="black",
)
ax[1].plot(
    [],
    [],
    linewidth=2,
    linestyle="--",
    color="black",
    label="First order convergence",
)

# Second order line
# x0 = np.log2(1 / np.asarray(mesh_sizes[0]))
# x1 = np.log2(1 / np.asarray(mesh_sizes[-1]))
# y0 = -10.5
# y1 = y0 - 2 * (x1 - x0)
# ax[0].plot(
#     [x0, x1],
#     [y0, y1],
#     linewidth=3,
#     linestyle="--",
#     color="black",
# )
# ax[1].plot(
#     [],
#     [],
#     linewidth=3,
#     linestyle="--",
#     color="black",
#     label="Second order",
# )

for key in errors.keys():

    # Rate
    ax[0].plot(
        np.log2(1 / np.asarray(mesh_sizes)),
        np.log2(errors[key]),
        linewidth=2,
        marker="s",
        markersize="6"
    )

    # Legend
    ax[1].plot(
        [],
        [],
        label=key,
        linewidth=2,
        marker="s",
        markersize="6",
    )
    ax[1].legend(loc="center", fontsize=13)
    ax[1].axis("off")

ax[0].grid()
ax[0].set_xlabel(r"$\log_2\left(1/\mathrm{mesh~size}\right)$", fontsize=15)
ax[0].set_ylabel(r"$\log_2\left(||\mathrm{error}||_{L^2}\right)$", fontsize=15)

plt.savefig("convergence_rates.pdf")