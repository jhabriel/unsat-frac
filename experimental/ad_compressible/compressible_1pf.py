"""
Implementation of compressible single phase flow in (unfractured) porous media 
using automatic differentiation.

    Domain: Unit square (2D)
    Grids: Structured Cartesian or Unstructured Triangular
    Boundary conditions: Neumann (top and bottom), Dirichlet (left and right).
    Manufactured solutions: Parabolic or Trigonometric
    Face averaging of densities: Upwinding or Arithmetic Average
    Discretization: Mpfa

@author: Jhabriel Varela
"""

#%% Import modules
import porepy as pp
import numpy as np
import sympy as sym
import scipy.sparse.linalg as spla

from porepy.numerics.ad.grid_operators import DirBC
from porepy.numerics.fv.generaltpfaad import ArithmeticAverageAd, FluxBasedUpwindAD

#%% Error functions
def pressure_error(g, num_pressure, true_pressure):
    """Computes discrete L2-error norm for the cell-centered pressures"""

    V = g.cell_volumes
    error = (V * (true_pressure - num_pressure) ** 2).sum() ** 0.5 / (
        V * true_pressure ** 2
    ).sum() ** 0.5

    return error


def flux_error(g, num_flux, true_flux):
    """Computes discrete L2-error norm for the face-centered fluxes"""

    A = g.face_areas
    error = (A * (true_flux - num_flux) ** 2).sum() ** 0.5 / (
        A * true_flux ** 2
    ).sum() ** 0.5

    return error


#%% Create unstructured grid
def make_grid(mesh_size=0.1, grid_type="cartesian", plot_grid=False):

    if grid_type == "cartesian":
        n = int(1 / mesh_size)
        gb = pp.meshing.cart_grid([], nx=[n, n], physdims=[1.0, 1.0])
    elif grid_type == "triangular":
        domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
        network_2d = pp.FractureNetwork2d(None, None, domain)
        mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
        gb = network_2d.mesh(mesh_args)
    else:
        raise ValueError("Grid type not available.")

    if plot_grid:
        pp.plot_grid(g, plot_2d=True)

    return gb


#%% Set model parameters
grid_type = "cartesian"  # "triangular"
face_avg_method = "upwind"  # "arithmetic"
solution = "parabolic"  # "trigonometric"

#%% Create grid and define physical parameters
gb = make_grid(mesh_size=0.1, grid_type=grid_type)
g = gb.grids_of_dimension(gb.dim_max())[0]
d = gb.node_props(g)

K = 5  # intrinsic permeability
c = 0.1  # compressibility
phi = 0.4  # porosity
rho_ref = 1  # reference density
p_ref = 1  # reference pressure

#%% Time parameters
final_time = 0.5
num_time_steps = 10
time = 0
dt = final_time / num_time_steps

#%% Exact solution
x = sym.symbols("x", real=True)
y = sym.symbols("y", real=True)
t = sym.symbols("t", real=True)

# Exact pressure distribution
if solution == "parabolic":
    p_sym = t * (1 - x) * x * (1 - y) * y
elif solution == "trigonometric":
    p_sym = sym.sin(np.pi * t) * sym.cos(np.pi * x) * sym.cos(np.pi * y)
else:
    raise ValueError("Exact solution not available")

p_grad_sym = [sym.diff(p_sym, x), sym.diff(p_sym, y)]  # pressure gradient
q_sym = [-K * p_grad_sym[0], -K * p_grad_sym[1]]  # darcy flux
rho_sym = rho_ref * sym.exp(c * (p_sym - p_ref))  # rho = rho(p)
advec_sym = [rho_sym * q_sym[0], rho_sym * q_sym[1]]  # advective flux
advec_div_sym = sym.diff(advec_sym[0], x) + sym.diff(advec_sym[1], y)  # divergence
accum_sym = phi * sym.diff(rho_sym, t)  # phi * d(rho)/dt
f_sym = accum_sym + advec_div_sym  # source term

# Convert symbolic expressions to functions for fast evaluation
p_ex = sym.lambdify((x, y, t), p_sym, "numpy")
q_ex = sym.lambdify((x, y, t), q_sym, "numpy")
advec_ex = sym.lambdify((x, y, t), advec_sym, "numpy")
f_ex = sym.lambdify((x, y, t), f_sym, "numpy")

#%% Define primary variable and assign data
param_key = "flow"
pressure_var = "pressure"
d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}


def assign_parameters(time):

    nf = g.num_faces
    nc = g.num_cells
    fn = g.face_normals
    fc = g.face_centers
    cc = g.cell_centers
    V = g.cell_volumes

    # Permeability tensor
    perm = pp.SecondOrderTensor(K * np.ones(nc))

    # Boundary condtions
    top = np.where(np.abs(fc[1] - 1) < 1e-5)[0]
    bottom = np.where(np.abs(fc[1]) < 1e-5)[0]
    left = np.where(np.abs(fc[0]) < 1e-5)[0]
    right = np.where(np.abs(fc[0] - 1) < 1e-5)[0]

    bc_faces = g.get_boundary_faces()
    bc_type = np.array(bc_faces.size * ["neu"])
    bc_type[np.in1d(bc_faces, left)] = "dir"
    bc_type[np.in1d(bc_faces, right)] = "dir"
    bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

    bc_values = np.zeros(g.num_faces)
    pf = p_ex(fc[0], fc[1], time * np.ones(nf))  # exact face pressures
    adv_f = advec_ex(fc[0], fc[1], time * np.ones(nf))  # exact advective velocities
    Adv_f = adv_f[0] * fn[0] + adv_f[1] * fn[1]  # exact advective fluxes

    bc_values[top] = np.abs(Adv_f[top])
    bc_values[bottom] = np.abs(Adv_f[bottom])
    bc_values[left] = pf[left]
    bc_values[right] = pf[right]

    source_term = f_ex(cc[0], cc[1], time * np.ones(nc)) * V

    # Initialize data dictionary
    specified_data = {
        "second_order_tensor": perm,
        "bc": bc,
        "bc_values": bc_values,
        "source": source_term,
        "mass_weight": phi * np.ones(g.num_cells),
    }

    # Assing (or update) parameters
    if time == 0:
        pp.initialize_data(g, d, param_key, specified_data)
    else:
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
p = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])
p_m = p.previous_iteration()
p_n = p.previous_timestep()

#%% We let the density to be a non-linear function of the pressure
def rho(p):
    if isinstance(p, pp.ad.Ad_array):
        return rho_ref * pp.ad.exp(c * (p - p_ref))
    else:
        return rho_ref * np.exp(c * (p - p_ref))


rho_ad = pp.ad.Function(rho, name="density")

#%% Initialize exporter
exporter = pp.Exporter(gb, "compressible_1p", "out")
d[pp.STATE]["p_ex"] = p_ex(cc[0], cc[1], time * np.ones_like(cc[0]))
exporter.write_vtu(["p_ex", pressure_var], time_step=0)

#%% Declare AD operators and equations
assign_parameters(time)

div_ad = pp.ad.Divergence(grid_list)  # discrete diveregence
bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)  # boundary vals
dir_bound_ad = DirBC(bound_ad, grid_list)

source_ad = pp.ad.ParameterArray(param_key, "source", grids=grid_list)
mass_ad = pp.ad.MassMatrixAd(param_key, grid_list)
mpfa_ad = pp.ad.MpfaAd(param_key, grid_list)
flux_inactive = mpfa_ad.flux * p_m + mpfa_ad.bound_flux * bound_ad
flux_active = mpfa_ad.flux * p + mpfa_ad.bound_flux * bound_ad

if face_avg_method == "arithmetic":
    arithmetic_avg = ArithmeticAverageAd(g, d, param_key)
    rho_faces_ad = arithmetic_avg(rho_ad(p_m), rho_ad(dir_bound_ad))
elif face_avg_method == "upwind":
    upwind = FluxBasedUpwindAD(g, d, param_key)
    rho_faces_ad = upwind(rho_ad(p_m), rho_ad(dir_bound_ad), flux_inactive)
else:
    raise ValueError("Face averaging method not implemented")

advective_flux_ad = rho_faces_ad * flux_active
continuity_ad = (
    mass_ad.mass * (rho_ad(p) - rho_ad(p_n))
    + dt * div_ad * advective_flux_ad
    - dt * source_ad
)

#%% Assemble and discretize model
eqs = pp.ad.Expression(continuity_ad, dof_manager)
equation_manager.equations.clear()
equation_manager.equations.append(eqs)
equation_manager.discretize(gb)

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

        # Solve for pressure increment and update pressure
        A, b = equation_manager.assemble_matrix_rhs()
        solution = spla.spsolve(A, b)

        # # Distribute variable to local data dictionaries
        dof_manager.distribute_variable(solution, additive=True, to_iterate=True)

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

    # Update next time step solution. Note that additive should be False here
    dof_manager.distribute_variable(solution, additive=False)

    # Export to PARAVIEW
    d[pp.STATE]["p_ex"] = p_ex(cc[0], cc[1], time * np.ones_like(cc[0]))
    exporter.write_vtu(["p_ex", pressure_var], time_step=n)

#%% Compute errors
num_pressure = d[pp.STATE][pressure_var]
true_pressure = d[pp.STATE]["p_ex"]
p_error = pressure_error(g, num_pressure, true_pressure)
print("L2-error for the pressure:", np.round(p_error, decimals=8))

adv_flux_ad_eval = pp.ad.Expression(advective_flux_ad, dof_manager)
adv_flux_ad_eval.discretize(gb)
num_adv_flux = adv_flux_ad_eval.to_ad(gb)
xf = g.face_centers[0]
yf = g.face_centers[1]
xn = g.face_normals[0]
yn = g.face_normals[1]
true_adv_flux = (
    xn * advec_ex(xf, yf, time * np.ones(g.num_faces))[0]
    + yn * advec_ex(xf, yf, time * np.ones(g.num_faces))[1]
)
adv_error = flux_error(g, num_adv_flux.val, true_adv_flux)
print("L2-error for the advective flux:", np.round(adv_error, decimals=8))
