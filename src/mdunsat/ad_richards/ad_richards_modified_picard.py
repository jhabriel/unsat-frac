#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the incompressible version of Richards' equation using
the modified Picard iteration scheme within the AD framework.
@author: jv
"""

#%% Import modules
import porepy as pp
import numpy as np
import sympy as sym
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import mdunsat as mdu

from porepy.numerics.ad.grid_operators import DirBC
from porepy.numerics.ad.functions import heaviside
from mdunsat.ad_utils.ad_utils import ( 
    ArithmeticAverageAd, UpwindFluxBasedAd, vanGenuchten
    )
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
k = 1  # intrinsic permeability
mu = 1 # dynamic viscosity
rho = 1 # density
grav = 1 # gravity 
K_sat = (k/mu) / (rho*grav) # Saturated hydraulic conductivity

theta_r = 0.1 # residual water content
theta_s = 0.4 # saturated water content, a.k.a porosity 
alpha_vG = 0.4 # van Genuchten parameter
n_vG = 2 # van Genuchten parameter
m_vG = 1 - 1/n_vG # van Genuchten paramete

#%% Assign parameters
param_key = "flow"
pressure_var = "pressure_head"
for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}


def assign_parameters(current_time):
    for g, d in gb:

        # Permeability tensor
        perm = pp.SecondOrderTensor(k * np.ones(g.num_cells))

        # Boundary conditions (Dirichlet only)
        bc_faces = g.get_boundary_faces()
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_val = np.zeros(g.num_faces)
        xf = g.face_centers[0]
        yf = g.face_centers[1]
        bc_val[bc_faces] = 0 # p_ex(xf[bc_faces], yf[bc_faces], current_time)

        # (Integrated) source terms
        xc = g.cell_centers[0]
        yc = g.cell_centers[1]
        source_term = 0 # f_ex(xc, yc, current_time) * g.cell_volumes

        # Initialize data dictionary
        specified_data = {
            "second_order_tensor": perm,
            "bc": bc,
            "bc_values": bc_val,
            "source": source_term,
            "mass_weight": np.ones(g.num_cells),
            "theta_r": theta_r,
            "theta_s": theta_s,
            "alpha_vG": alpha_vG,
            "n_vG": n_vG,
            "m_vG": 1 - 1/n_vG
        }
        d = pp.initialize_data(g, d, param_key, specified_data)

#%% Set initial states
for g, d in gb:
    cc = g.cell_centers
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][pressure_var] = 0 * np.ones(g.num_cells)
    d[pp.STATE][pp.ITERATE][pressure_var] = 0 * np.ones(g.num_cells)

#%% AD variables and manager
grid_list = [g for g, _ in gb]
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
psi = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])

#%% Soil water retention curves
assign_parameters(0)

#%%
swrc = vanGenuchten(g, d, param_key)
theta_ad = pp.ad.Function(swrc.water_content, "water content")
krw_ad = pp.ad.Function(swrc.relative_permeability, "relative permeability")
C_ad = pp.ad.Function(swrc.moisture_capacity, "moisture capacity")

theta_eval = pp.ad.Expression(theta_ad(psi), dof_manager)
theta_eval.discretize(gb)
num_theta = theta_eval.to_ad(gb)
print(num_theta)

#%% Time parameters
final_time = 1
num_time_steps = 1
time = 0
dt = final_time / num_time_steps

#%% Exact solution
# x = sym.symbols("x", real=True)
# y = sym.symbols("y", real=True)
# t = sym.symbols("t", real=True)

# exact_sol = "parabolic"

# # Exact pressure distribution
# if exact_sol == "parabolic":
#     p_sym = t * (1 - x) * x * (1 - y) * y - 1
# elif exact_sol == "trigonometric":
#     p_sym = sym.sin(np.pi * t) * sym.sin(np.pi * x) * sym.cos(np.pi * y) - 1
# else:
#     raise ValueError("Exact solution not available")

# p_grad_sym = [sym.diff(p_sym, x), sym.diff(p_sym, y)]  # pressure gradient
# q_sym = [-K * p_grad_sym[0], -K * p_grad_sym[1]]  # darcy flux
# q_div_sym = sym.diff(q_sym[0], x) + sym.diff(q_sym[1], y)  # flux divergence
# rho_sym = rho_ref * sym.exp(c * (p_sym - p_ref))  # rho = rho(p)
# accum_sym = phi * sym.diff(rho_sym, t)  # phi * drho/dt
# f_sym = accum_sym + rho_sym * q_div_sym  # source term

# # Convert symbolic expressions to functions for fast evaluation
# p_ex = sym.lambdify((x, y, t), p_sym, "numpy")
# q_ex = sym.lambdify((x, y, t), q_sym, "numpy")
# f_ex = sym.lambdify((x, y, t), f_sym, "numpy")



