import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rc(
    "font",
    family="Times New Roman",
)
plt.rcParams.update(
    {
        "text.usetex": True,
    }
)

# %% Open pickle file
with open("out/water_volume.pickle", "rb") as handle:
    input_list = pickle.load(handle)

times: np.ndarray = np.asarray(input_list[0])
water_vol_left: np.ndarray = np.asarray(input_list[1])
water_vol_right: np.ndarray = np.asarray(input_list[2])
water_vol_top: np.ndarray = np.asarray(input_list[3])


# %% Plot water volume
fig, ax = plt.subplots(1, 1)
ax.plot(times, water_vol_right, color="red", label=r"$V_{2}(t)$")
ax.plot(times, water_vol_left, color="green", label=r"$V_{3}(t)$")
ax.plot(times, water_vol_top, color="black", label=r"$V_{4}(t)$")
ax.set_xlabel(r"$\mathrm{Time}~[\mathrm{s}]$", fontsize=17)
ax.set_ylabel(
    r"$\mathrm{Fracture~water~volume}~\left[\mathrm{cm^3}\right]$", fontsize=17
)
ax.legend(fontsize=17)
fig.tight_layout()
fig.savefig("out/water_volume.pdf")
