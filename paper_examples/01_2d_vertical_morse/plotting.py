import numpy as np
import matplotlib
from scipy.interpolate import UnivariateSpline
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import pickle

from matplotlib import rc

rc('font', **{
    "family": "serif",
    "serif": "Computer Modern Roman",
    "sans-serif": "Computer Modern Roman serif",
    "cursive": "Zap Chancery",
    "monospace": "Computer Modern Typewriter",
    })
rc('text', usetex=True)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
rc('font', size=SMALL_SIZE)          # controls default text sizes
rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

# %% Load files
file_name = "out.plk"
open_file = open(file_name, "rb")
d = pickle.load(open_file)
open_file.close()

# %% Water volume
spl = UnivariateSpline(d["time"], d["water_volume"])
spl.set_smoothing_factor(0.2)
xs = np.linspace(0.19, 4, 1000)
fig, ax = plt.subplots(1, 1)
ax.plot(d["time"], d["water_volume"])
ax.plot(xs, spl(xs), 'r', lw=8, alpha=0.4)
ax.set_xlabel(r"$\mathrm{Time}~[\mathrm{hours}]$")
ax.set_ylabel(r"$\mathrm{Water~volume}~\left[\mathrm{cm^3}\right]$")
fig.tight_layout()
plt.show()
fig.savefig("out/numex1_water_volume.pdf")



#%% Time step and number of itertations
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time [hours]')
ax1.set_ylabel('Time step [s]', color=color)
ax1.plot(d["time"], d["time_step"], color=color, alpha=0.75)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Iterations', color=color)  # we already handled the x-label with ax1
ax2.plot(d["time"], d["iterations"], color=color, alpha=1)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#fig.savefig("out/numex1_dt_and_iter.pdf")

# %% Water table height
fig, ax = plt.subplots(1, 1)
ax.plot(d["time"], d["water_table"])
ax.set_xlabel(r"$Time [h]$")
ax.set_ylabel("Water table height in the fracture [cm]")
plt.show()
# fig.savefig("out/hydraulic_head_evolution.pdf")

# %% Time step
fig, ax = plt.subplots(1, 1)
ax.plot(d["time"], d["time_step"])
ax.set_xlabel("Time [h]")
ax.set_ylabel("Time step [s]")
#plt.show()
# fig.savefig("out/time_steps.pdf")

# %% Iterations
fig, ax = plt.subplots(1, 1)
ax.plot(d["time"], d["iterations"])
ax.set_xlabel("Time [h]")
ax.set_ylabel("Number of iterations")
#plt.show()
# fig.savefig("out/number_of_iters.pdf")
