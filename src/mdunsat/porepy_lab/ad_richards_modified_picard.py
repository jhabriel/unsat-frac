"""
Implementation of the incompressible version of Richards' equation.

Grids: Cartesian (structured) or triangular (unstructured)
Water retention curves: van Genuchten-Mualem
Linearization scheme: Modified Picard Iteration (Celia et. al. 1990)
Face averaging: Upwind or Arithmetic
Exact manufactured solution: Parabolic or Trigonometric
Boundary conditions: Mixed. Neumann top and bottom. Dirichlet left and right.

@author: jv
"""

#%% Import modules
import porepy as pp
import numpy as np
import sympy as sym
import scipy.sparse.linalg as spla

from porepy.numerics.ad.grid_operators import DirBC
from mdunsat.ad_utils.ad_utils import (
    ArithmeticAverageAd,
    UpwindFluxBasedAd,
    vanGenuchten,
)

#%% Utility functions
def l2_error(g, num_array, true_array, array_sol):
    """Computes discrete L2-error norm for the cell-centered pressures"""

    if array_sol == "pressure":
        V = g.cell_volumes.copy()
        error = (V * (true_array - num_array) ** 2).sum() ** 0.5 / (
            V * true_array ** 2
        ).sum() ** 0.5
    elif array_sol == "flux":
        A = g.face_areas.copy()
        error = (A * (true_array - num_array) ** 2).sum() ** 0.5 / (
            A * true_array ** 2
        ).sum() ** 0.5
    else:
        raise ValueError("Solution array not recognized. Use pressure or flux")

    return error


def make_grid(mesh_size, grid_type):
    """Creates grid bucket given the element size and mesh type"""

    if grid_type == "cartesian":
        n = int(1 / mesh_size)
        gb = pp.meshing.cart_grid([], nx=[n, n], physdims=[1.0, 1.0])
    elif grid_type == "triangular":
        domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
        network_2d = pp.FractureNetwork2d(None, None, domain)
        mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
        gb = network_2d.mesh(mesh_args)
    else:
        raise ValueError("Solution array not recognized. Use pressure or flux")

    return gb


#%% Establish model parameters and create grid
solution = "parabolic"  # trigonometric
avg_method = "upwind"  # arithmetic

grid_type = "cartesian"  # triangular
mesh_size = 0.25
gb = make_grid(mesh_size, grid_type)
g = gb.grids_of_dimension(2)[0]
d = gb.node_props(g)

#%% Physical parameters
k = 1  # intrinsic permeability
mu = 1  # dynamic viscosity
rho = 1  # density
grav = 1  # gravity
K_sat = (k / mu) / (rho * grav)  # Saturated hydraulic conductivity

theta_r = 0.1  # residual water content
theta_s = 0.4  # saturated water content, a.k.a porosity
alpha_vG = 0.04  # van Genuchten parameter
n_vG = 2  # van Genuchten parameter
m_vG = 1 - 1 / n_vG  # van Genuchten parameter

#%% Time parameters
final_time = 1
num_time_steps = 5
time = 0
dt = final_time / num_time_steps

#%% Exact solution
x = sym.symbols("x", real=True)
y = sym.symbols("y", real=True)
t = sym.symbols("t", real=True)

# Exact pressure distribution
if solution == "parabolic":
    p_sym = t * (1 - x) * x * (1 - y) * y - 1
elif solution == "trigonometric":
    p_sym = sym.cos(sym.pi * t) * sym.sin(sym.pi * x) * sym.cos(sym.pi * y) - 2
else:
    raise ValueError("Exact solution not available")

# Exact SWRC (Soil Water Retention Curves)
theta_sym = (theta_s - theta_r) / (
    1 + (alpha_vG * (p_sym ** 0.5) ** 2.0) ** n_vG
) ** m_vG + theta_r
S_eff = (theta_sym - theta_r) / (theta_s - theta_r)
krw_sym = S_eff ** 0.5 * (1 - (1 - S_eff ** (1 / m_vG)) ** m_vG) ** 2

# Pressure gradient
p_grad_sym = [sym.diff(p_sym, x), sym.diff(p_sym, y)]

# Accumulation term
theta_t_sym = sym.diff(theta_sym, t)

# Darcy flux
q_sym = [-K_sat * krw_sym * p_grad_sym[0], -K_sat * krw_sym * p_grad_sym[1]]

# Flux divergence
q_div_sym = sym.diff(q_sym[0], x) + sym.diff(q_sym[1], y)

# Source term (rhs)
f_sym = theta_t_sym + q_div_sym

# Convert symbolic expressions to functions for fast evaluation
p_ex = sym.lambdify((x, y, t), p_sym, "numpy")
q_ex = sym.lambdify((x, y, t), q_sym, "numpy")
f_ex = sym.lambdify((x, y, t), f_sym, "numpy")

#%% Assign parameters
param_key = "flow"
pressure_var = "pressure_head"
d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}


def initialize_parameters(g, d, param_key, time):

    nf = g.num_faces
    nc = g.num_cells
    fn = g.face_normals
    fc = g.face_centers
    cc = g.cell_centers
    V = g.cell_volumes

    perm = pp.SecondOrderTensor(k * np.ones(nc))

    top = np.where(np.abs(fc[1] - 1) < 1e-5)[0]
    bottom = np.where(np.abs(fc[1]) < 1e-5)[0]
    left = np.where(np.abs(fc[0]) < 1e-5)[0]
    right = np.where(np.abs(fc[0] - 1) < 1e-5)[0]

    bc_faces = g.get_boundary_faces()
    bc_type = np.array(bc_faces.size * [None])
    bc_type[np.in1d(bc_faces, top)] = "neu"
    bc_type[np.in1d(bc_faces, bottom)] = "neu"
    bc_type[np.in1d(bc_faces, left)] = "dir"
    bc_type[np.in1d(bc_faces, right)] = "dir"
    bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

    bc_values = np.zeros(g.num_faces)
    pf = p_ex(fc[0], fc[1], time * np.ones(nf))
    qf = q_ex(fc[0], fc[1], time * np.ones(nf))
    Qf = qf[0] * fn[0] + qf[1] * fn[1]

    bc_values[top] = np.abs(Qf[top])  # outflow flux
    bc_values[bottom] = np.abs(Qf[bottom])  # outflow flux
    bc_values[left] = pf[left]
    bc_values[right] = pf[right]

    source_term = f_ex(cc[0], cc[1], time * np.ones(nc)) * V

    specified_parameters = {
        "second_order_tensor": perm,
        "bc": bc,
        "bc_values": bc_values,
        "source": source_term,
        "mass_weight": np.ones(nc),
        "theta_r": theta_r,
        "theta_s": theta_s,
        "alpha_vG": alpha_vG,
        "n_vG": n_vG,
        "m_vG": m_vG,
    }

    pp.initialize_data(g, d, param_key, specified_parameters)


def update_parameters(g, d, param_key, time):

    nf = g.num_faces
    nc = g.num_cells
    fn = g.face_normals
    fc = g.face_centers
    V = g.cell_volumes

    top = np.where(np.abs(fc[1] - 1) < 1e-5)[0]
    bottom = np.where(np.abs(fc[1]) < 1e-5)[0]
    left = np.where(np.abs(fc[0]) < 1e-5)[0]
    right = np.where(np.abs(fc[0] - 1) < 1e-5)[0]

    bc_values = np.zeros(g.num_faces)
    pf = p_ex(fc[0], fc[1], time * np.ones(nf))
    qf = q_ex(fc[0], fc[1], time * np.ones(nf))
    Qf = qf[0] * fn[0] + qf[1] * fn[1]

    bc_values[top] = np.abs(Qf[top])  # outflow flux
    bc_values[bottom] = np.abs(Qf[bottom])  # outflow flux
    bc_values[left] = pf[left]
    bc_values[right] = pf[right]

    cc = g.cell_centers
    source_term = f_ex(cc[0], cc[1], time * np.ones(nc)) * V

    d[pp.PARAMETERS][param_key]["bc_values"] = bc_values
    d[pp.PARAMETERS][param_key]["source"] = source_term


#%% Set initial states
for g, d in gb:
    cc = g.cell_centers
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][pressure_var] = p_ex(cc[0], cc[1], time * np.ones_like(cc[0]))
    d[pp.STATE][pp.ITERATE][pressure_var] = d[pp.STATE][pressure_var].copy()

#%% AD variables and manager
grid_list = [g for g, _ in gb]
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
psi = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])
# psi_m = psi.previous_iteration()
# psi_n = psi.previous_timestep()

#%% AD operators and discrete expressions
initialize_parameters(g, d, param_key, time)

# Grid operators
div_ad = pp.ad.Divergence(grid_list)  # discrete diveregence
bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)  # boundary vals
dir_bound_ad = DirBC(bound_ad, grid_list)  # dirichlet bc vals

# Soil water retention curves
swrc = vanGenuchten(g, d, param_key)
theta_ad = pp.ad.Function(swrc.water_content, "water content")
krw_ad = pp.ad.Function(swrc.relative_permeability, "relative permeability")
C_ad = pp.ad.Function(swrc.moisture_capacity, "moisture capacity")

# MPFA discretization
mpfa_ad = pp.ad.MpfaAd(param_key, grid_list)
# Obtain single phase flow to compute directionality of upwind scheme
flux_1p_ad = mpfa_ad.flux * psi.previous_iteration() + mpfa_ad.bound_flux * bound_ad

# Face-averaging of relative permeabilities
if avg_method == "arithmetic":
    arithmetic_avg = ArithmeticAverageAd(g, d, param_key)
    krw_faces_ad = arithmetic_avg(
        krw_ad(psi.previous_iteration()), krw_ad(dir_bound_ad)
    )
elif avg_method == "upwind":
    upwind = UpwindFluxBasedAd(g, d, param_key)
    krw_faces_ad = upwind(
        krw_ad(psi.previous_iteration()), krw_ad(dir_bound_ad), flux_1p_ad
    )
else:
    raise ValueError("Averaging method not implemented")

# Darcy fluxes
flux_scaling = 1 / (mu * rho * grav)
flux_ad = (
    flux_scaling * krw_faces_ad * (mpfa_ad.flux * psi + mpfa_ad.bound_flux * bound_ad)
)

# Source and accumulation terms (Linearization: Modified Picard iteration)
# Note: The expression containing the active ad variable (psi) must
# be placed at the left of the (final) expression. Otherwise, numpy
# will broadcast the expression leading to and undesirable ouput format
source_ad = pp.ad.ParameterArray(param_key, "source", grids=grid_list)
mass_ad = pp.ad.MassMatrixAd(param_key, grid_list)
accum_active = mass_ad.mass * psi * C_ad(psi.previous_iteration())
accum_inactive = mass_ad.mass * (
    theta_ad(psi.previous_iteration())
    - C_ad(psi.previous_iteration()) * psi.previous_iteration()
    - theta_ad(psi.previous_timestep())
)
accumulation_ad = accum_active + accum_inactive

# Continuity equation
continuity_ad = accumulation_ad + dt * div_ad * flux_ad - dt * source_ad

#%% Assemble the system of equations
eqs = pp.ad.Expression(continuity_ad, dof_manager)  # convert to expression
equation_manager.equations.clear()
equation_manager.equations.append(eqs)  # feed eq to the equation manager
equation_manager.discretize(gb)  # discretize problem

#%% Time loop
total_iteration_counter = 0
for n in range(1, num_time_steps + 1):
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1
    time += dt

    print("Current time: ", np.round(time, decimals=1))

    update_parameters(g, d, param_key, time)

    while iteration_counter <= 30 and not (rel_res < 1e-12 or residual_norm < 1e-12):

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
    d[pp.STATE]["p_ex"] = p_ex(
        g.cell_centers[0], g.cell_centers[1], time * np.ones(g.num_cells)
    )

#%% Compute errors
num_pressure = d[pp.STATE][pressure_var].copy()
true_pressure = d[pp.STATE]["p_ex"]
p_error = l2_error(g, num_pressure, true_pressure, "pressure")

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
q_error = l2_error(g, num_flux.val, true_flux, "flux")

# return p_error, q_error
print("Pressure error:", p_error, "\t | \t", "Flux error:", q_error)
