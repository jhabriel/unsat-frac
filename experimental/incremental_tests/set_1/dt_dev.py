import numpy as np
import matplotlib.pyplot as plt
from _class import TimeSteppingControl

# Create time object
tsc = TimeSteppingControl(
    time_init_final=(0, 10),
    dt_init=0.1,
    dt_min_max=(0.01, 2),
    iter_max=10,
    iter_optimal_range=(4, 7),
    iter_lowupp_factor=(1.3, 0.7),
    recomp_factor=0.5,
    recomp_max=10
)

# Consider the logistic equation:
#   u'(t) = u(t) (1 - u(t)); u(0) = 0.1
# and its approximation using BE with a Netwon solver

# Residual and its derivative
def eq(u, u0, dt):
    a = dt
    b = 1 - dt
    c = -u0
    val = a * u ** 2 + b * u + c
    jac = 2 * a * u + b
    return val, jac

# Initial condition
u_n = 0.1
ts = []
us = []

# Time loop
while tsc.time < tsc.time_final:

    tsc.time += tsc.dt
    iters = 0
    tol = 1e-12
    error = 1e+8
    u_m = u_n

    # Newton loop
    while iters <= tsc.iter_max and error > tol:
        res, der = eq(u_m, u_n, tsc.dt)
        x = -res/der
        u_m += x
        error = np.abs(res)
        print(f"time: {tsc.time}, dt: {tsc.dt}, error: {error}, iter: {iters}")
        iters += 1

    # Get new time step
    tsc.next_time_step(iters-1)
    if tsc.recomp_sol:
        u_m = u_n
        continue

    u_n = u_m
    ts.append(tsc.time)
    us.append(u_m)

    print("")

#%%Plot
plot_fig = True
if plot_fig:
    fig, ax = plt.subplots(1, 1)
    ax.plot(ts, us, marker="s", linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    plt.show()