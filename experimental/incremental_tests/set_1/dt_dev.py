"""
This script is meant as MWE of how to use the time-stepping control algorithm. To this purpose
we consider the following nonlinear ODE:
    u'(t) = u(t) (1 - u(t))
    u(0) = 0.1
and its approximation using backward Euler

"""
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

# Create time object
tsc = pp.TimeSteppingControl(
    schedule=[0, 2, 5, 6, 10],
    dt_init=0.1,
    dt_min_max=(0.01, 2),
    iter_max=10,
    iter_optimal_range=(4, 7),
    iter_lowupp_factor=(1.3, 0.7),
    recomp_factor=0.5,
    recomp_max=10,
    print_info=True,
)


# Exact solution
def u_ex(u0, t):
    return (u0 * np.exp(t)) / (u0 * (np.exp(t) - 1) + 1)


# Residual and its derivative
def eq(u, u0, dt):
    a = dt
    b = 1 - dt
    c = -u0
    val = a * u ** 2 + b * u + c
    jac = 2 * a * u + b
    return val, jac


# Initial condition
u0 = 0.1
u_n = u0
ts = []
us = []

# Time loop
while tsc.time < tsc.time_final:

    tsc.time += tsc.dt
    iters = 0
    tol = 1e-12
    error = 1e8
    u_m = u_n

    # Newton loop
    while iters <= tsc.iter_max and error > tol:
        res, der = eq(u_m, u_n, tsc.dt)
        x = -res / der
        u_m += x
        error = np.abs(res)
        print(f"time: {tsc.time}, dt: {tsc.dt}, error: {error}, iter: {iters}")
        iters += 1
    recomp_sol = error > tol  # conditions for recomputing solution can be more general

    # Get new time step
    tsc.next_time_step(recompute_solution=recomp_sol, iterations=iters - 1)
    # If the solution has been recomputed, recover previous time conditions and return to
    # the time loop. Otherwise, continue as normal
    if recomp_sol:
        u_m = u_n
        continue
    else:
        u_n = u_m
        ts.append(tsc.time)
        us.append(u_m)

    print()

#%% Plot
plot_fig = True
if plot_fig:
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        np.linspace(tsc.time_init, tsc.time_final, 1000),
        u_ex(u0, np.linspace(tsc.time_init, tsc.time_final, 1000)),
        label="exact",
    )
    ax.plot(ts, us, marker=".", markersize=7, label="approximated")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    plt.show()
