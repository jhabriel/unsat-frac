#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of compressible single phase flow equations in an unfractured 
porous media using automatic differentiation
@author: jv
"""

#%% Import modules
import porepy as pp
import numpy as np
import sympy as sym
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from porepy.numerics.ad.grid_operators import DirBC
from ad_utils import ArithmeticAverageAd, UpwindFluxBasedAd

#%% Error functions
def pressure_error(g, num_pressure, true_pressure):
    """Computes discrete L2-error norm for the cell-centered pressures"""

    V = g.cell_volumes.copy()
    error = (V * (true_pressure - num_pressure) ** 2).sum() ** 0.5 / (
        V * true_pressure ** 2
    ).sum() ** 0.5

    return error


def flux_error(g, num_flux, true_flux):
    """Computes discrete L2-error norm for the cell-centered pressures"""

    A = g.face_areas.copy()
    error = (A * (true_flux - num_flux) ** 2).sum() ** 0.5 / (
        A * true_flux ** 2
    ).sum() ** 0.5

    return error


#%% Create unstructured grid
def make_2d_grid(mesh_size=0.1, L=[1.0, 1.0], plot_grid=False):

    domain = {"xmin": 0.0, "xmax": L[0], "ymin": 0.0, "ymax": L[1]}
    network_2d = pp.FractureNetwork2d(None, None, domain)
    mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
    gb = network_2d.mesh(mesh_args)

    if plot_grid:
        pp.plot_grid(gb.grids_of_dimension(2)[0], plot_2d=True)

    return gb


gb = make_2d_grid(mesh_size=0.2, plot_grid=False)
g = gb.grids_of_dimension(2)[0]

#%% Physical parameters
K = 1  # intrinsic permeability
c = 0.1  # compressibility
phi = 0.4  # porosity
rho_ref = 1 # reference density
p_ref = 1 # reference pressure

#%% Time parameters
final_time = 1
num_time_steps = 1
time = 0
dt = final_time / num_time_steps

#%% Exact solution
x = sym.symbols("x", real=True)
y = sym.symbols("y", real=True)
t = sym.symbols("t", real=True)

exact_sol = "parabolic"

# Exact pressure distribution
if exact_sol == "parabolic":
    p_sym = t * (1 - x) * x * (1 - y) * y
elif exact_sol == "trigonometric":
    p_sym = sym.sin(np.pi * t) * sym.sin(np.pi * x) * sym.cos(np.pi * y)
else:
    raise ValueError("Exact solution not available")

p_grad_sym = [sym.diff(p_sym, x), sym.diff(p_sym, y)]  # pressure gradient
q_sym = [-K * p_grad_sym[0], -K * p_grad_sym[1]]  # darcy flux
q_div_sym = sym.diff(q_sym[0], x) + sym.diff(q_sym[1], y)  # flux divergence
rho_sym = rho_ref * sym.exp(c * (p_sym - p_ref))  # rho = rho(p)
accum_sym = phi * sym.diff(rho_sym, t)  # phi * drho/dt
f_sym = accum_sym + rho_sym * q_div_sym  # source term

# Convert symbolic expressions to functions for fast evaluation
p_ex = sym.lambdify((x, y, t), p_sym, "numpy")
q_ex = sym.lambdify((x, y, t), q_sym, "numpy")
f_ex = sym.lambdify((x, y, t), f_sym, "numpy")

#%% Define primary variable and assign data
param_key = "flow"

pressure_var = "pressure"
for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}


def assign_parameters(current_time):
    for g, d in gb:

        # Permeability tensor
        perm = pp.SecondOrderTensor(K * np.ones(g.num_cells))

        # Boundary conditions (Dirichlet only)
        bc_faces = g.get_boundary_faces()
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_val = np.zeros(g.num_faces)
        xf = g.face_centers[0]
        yf = g.face_centers[1]
        bc_val[bc_faces] = p_ex(xf[bc_faces], yf[bc_faces], current_time)

        # (Integrated) source terms
        xc = g.cell_centers[0]
        yc = g.cell_centers[1]
        source_term = f_ex(xc, yc, current_time) * g.cell_volumes

        # Initialize data dictionary
        specified_data = {
            "second_order_tensor": perm,
            "bc": bc,
            "bc_values": bc_val,
            "source": source_term,
            "mass_weight": phi * np.ones(g.num_cells),
        }
        d = pp.initialize_data(g, d, param_key, specified_data)


#%% Set initial states
for g, d in gb:
    cc = g.cell_centers
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][pressure_var] = p_ex(cc[0], cc[1], time * np.ones_like(cc[0]))
    d[pp.STATE][pp.ITERATE][pressure_var] = p_ex(
        cc[0], cc[1], time * np.ones_like(cc[0])
    )

#%% AD variables and manager
grid_list = [g for g, _ in gb]
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
p = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])

#%% We let the density to be a non-linear function of the pressure
def rho(p):
    return rho_ref * np.exp(c * (p - p_ref))


rho_ad = pp.ad.Function(rho, name="density")

#%% Initialize exporter
exporter = pp.Exporter(gb, "compressible_1p", "out")
d[pp.STATE]["p_ex"] = p_ex(g.cell_centers[0], g.cell_centers[1], time)
exporter.write_vtu(["p_ex", pressure_var], time_step=0)

#%% Time loop

total_iteration_counter = 0
for n in range(1, num_time_steps + 1):
    tol = 1e-8
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1
    time += dt

    print("Current time: ", np.round(time, decimals=1))

    assign_parameters(time)  # update parameters accordingly

    while iteration_counter <= 30 and not (rel_res < 1e-12 or residual_norm < 1e-12):

        # Declare equations
        div_ad = pp.ad.Divergence(grid_list)  # discrete diveregence
        bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)  # boundary vals
        dir_bound_ad = DirBC(bound_ad, grid_list)

        source_ad = pp.ad.ParameterArray(
            param_key, "source", grids=grid_list
        )  # sources
        mass_ad = pp.ad.MassMatrixAd(param_key, grid_list)  # mass matrix
        mpfa_ad = pp.ad.MpfaAd(param_key, grid_list)  # mpfa discretization
        flux_ad = mpfa_ad.flux * p + mpfa_ad.bound_flux * bound_ad  # darcy flux

        arithmetic_avg = ArithmeticAverageAd(g, d, param_key)
        upwind = UpwindFluxBasedAd(g, d, param_key)

        # avg_method = "arithmetic"
        avg_method = "upwind"
        if avg_method == "arithmetic":
            rho_faces_ad = arithmetic_avg(rho_ad(p.previous_iteration()), rho_ad(dir_bound_ad))
        elif avg_method == "upwind":
            rho_faces_ad = upwind(
                rho_ad(p.previous_iteration()), rho_ad(dir_bound_ad), flux_ad
            )
        else:
            raise ValueError("Averaging method not implemented")

        advective_flux_ad = rho_faces_ad * flux_ad   # advective flux
        # The following expression results in an error
        # advective_flux_ad =  rho_faces_ad * flux_ad
        # This suggests that there is a preferrable place where the operations must be added to

        continuity_ad = (
            mass_ad.mass * (rho_ad(p) - rho_ad(p.previous_timestep()))
            + dt * div_ad * advective_flux_ad
            - dt * source_ad
        )

        # Assemble the system of equations
        eqs = pp.ad.Expression(continuity_ad, dof_manager)  # convert to expression
        equation_manager.equations.clear()
        equation_manager.equations.append(eqs)  # feed eq to the equation manager
        equation_manager.discretize(gb)  # discretize problem

        # Solve for pressure increment and update pressure
        A, b = equation_manager.assemble_matrix_rhs()
        pressure_increment = spla.spsolve(A, b)
        d[pp.STATE][pp.ITERATE][pressure_var] += pressure_increment

        # Compute 'error' as norm of the residual
        residual_norm = np.linalg.norm(b, 2)
        if iteration_counter == 0:
            initial_residual_norm = residual_norm
        else:
            initial_residual_norm = max(residual_norm, initial_residual_norm)
        rel_res = residual_norm / initial_residual_norm
        print(
            "iteration",
            iteration_counter,
            "abs res",
            residual_norm,
            "rel res",
            residual_norm / initial_residual_norm,
        )

        # Prepare next iteration
        iteration_counter += 1
        total_iteration_counter += 1

    print()

    # Update next time step solution
    d[pp.STATE][pressure_var] = d[pp.STATE][pp.ITERATE][pressure_var].copy()

    # Export to PARAVIEW
    d[pp.STATE]["p_ex"] = p_ex(g.cell_centers[0], g.cell_centers[1], time)
    exporter.write_vtu(["p_ex", pressure_var], time_step=n)

#%% Compute errors
num_pressure = d[pp.STATE][pressure_var]
true_pressure = d[pp.STATE]["p_ex"]
p_error = pressure_error(g, num_pressure, true_pressure)
print("L2-error for the pressure:", np.round(p_error, decimals=4))

flux_ad_eval = pp.ad.Expression(flux_ad, dof_manager)
flux_ad_eval.discretize(gb)
num_flux = flux_ad_eval.to_ad(gb)
xf = g.face_centers[0]
yf = g.face_centers[1]
xn = g.face_normals[0]
yn = g.face_normals[1]
true_flux = (
    xn * q_ex(xf, yf, time * np.ones(g.num_faces))[0]
    + yn * q_ex(xf, yf, time * np.ones(g.num_faces))[1]
)
q_error = flux_error(g, num_flux.val, true_flux)
print("L2-error for the flux:", np.round(q_error, decimals=4))