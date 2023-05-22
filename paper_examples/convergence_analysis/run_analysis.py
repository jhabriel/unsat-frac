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

#%% Plotting
fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})

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

# Rate
ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_h_bulk"]),
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    color="orange",
)

ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_q_intf"]),
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    color="green",
)

ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_h_frac"]),
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    color="blue",
)

ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_q_bulk"]),
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    color="red",
)

ax[0].plot(
    np.log2(1 / np.asarray(mesh_sizes)),
    np.log2(errors["error_vol_frac"]),
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    color="magenta",
)

ax[0].set_xlabel(r"$\log_2\left(1/\mathrm{mesh~size}\right)$", fontsize=15)
ax[0].set_ylabel(r"$\log_2\left(||\mathrm{error}||_{L^2}\right)$", fontsize=15)

ax[1].plot(
    [],
    [],
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    label=r"$h_b$",
    color="orange",
)

ax[1].plot(
    [],
    [],
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    label=r"$\Lambda_{\pm}$",
    color="green",
)

ax[1].plot(
    [],
    [],
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    label=r"$h_f$",
    color="blue",
)

ax[1].plot(
    [],
    [],
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    label=r"$Q_b$",
    color="red",
)

ax[1].plot(
    [],
    [],
    linewidth=1,
    linestyle="--",
    marker="*",
    markersize="6",
    label=r"$V_f$",
    color="magenta",
)

ax[1].legend(fontsize=14, loc="center")
ax[1].axis("off")

plt.savefig("convergence_rates.pdf")


#%% Save to pickle to reproduce results
# See: Blender's answer -> https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
import pickle

with open("out/l2_errors.pickle", "wb") as handle:
    pickle.dump([errors, mesh_sizes], handle, protocol=pickle.HIGHEST_PROTOCOL)
