"""
 -- Compressible single phase flow with MPFA on a unit square using AD --

Grid types: Cartesian (structured) / Triangular (unstructured)
Face averaging of densities: Upwind / Arithmetic
Exact manufactured solution type: Parabolic / Trigonometric
Boundary conditions: Neumann top and bottom, Dirichlet left and right.

@author: jv
"""

#%% Importing modules
import porepy as pp
import numpy as np
import sympy as sym
import scipy.sparse.linalg as spla

from porepy.numerics.ad.grid_operators import DirBC
from mdunsat.ad_utils.ad_utils import ArithmeticAverageAd, UpwindFluxBasedAd

from porepy.numerics.fv.generaltpfaad import GeneralTpfaAd
from porepy.numerics.fv.generaltpfaad import UpwindAd

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
        n = int(1/mesh_size)
        gb = pp.meshing.cart_grid([], nx=[n, n], physdims=[1.0, 1.0])
    elif grid_type == "triangular":
        domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
        network_2d = pp.FractureNetwork2d(None, None, domain)
        mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
        gb = network_2d.mesh(mesh_args) 
    else:
        raise ValueError("Solution array not recognized. Use pressure or flux")
    
    return gb

#%% Model specifications
solution = "parabolic" # trigonometric
grid_type = "cartesian" # triangular
avg_method = "arithmetic" # artihmetic
refine = 4

#%% Make grid
# mesh_size = 0.1
# gb = make_grid(mesh_size, grid_type)
# g = gb.grids_of_dimension(2)[0]
# d = gb.node_props(g)
gb = pp.meshing.cart_grid([], nx=[4 * 2 ** refine, 4 * 2 ** refine], physdims=[1, 1])
g = gb.grids_of_dimension(2)[0]
d = gb.node_props(g)

#%% Time parameters
final_time = 0.5
num_time_steps = 4 * 2 ** (2 * refine)
time = 0
dt = final_time / num_time_steps

#%% Physical parameters
K = 1  # intrinsic permeability
c = 0.1  # compressibility
phi = 0.4  # porosity
rho_ref = 1 # reference density
p_ref = 1 # reference pressure

#%% Exact solution
x = sym.symbols("x", real=True)
y = sym.symbols("y", real=True)
t = sym.symbols("t", real=True)

# Exact pressure distribution
if solution == "parabolic":
    p_sym = t * (1 - x) * x * (1 - y) * y
elif solution == "trigonometric":
    p_sym = sym.sin(np.pi * t) * sym.sin(np.pi * x) * sym.cos(np.pi * y)
else:
    raise ValueError("Exact solution not available")

p_grad_sym = [sym.diff(p_sym, x), sym.diff(p_sym, y)]  # pressure gradient
rho_sym = rho_ref * sym.exp(c * (p_sym - p_ref))  # rho = rho(p)
q_sym = [-K * p_grad_sym[0], -K * p_grad_sym[1]]  # darcy flux
advec_sym = [rho_sym * q_sym[0], rho_sym * q_sym[1]] # advective flux
advec_div_sym = sym.diff(advec_sym[0], x) + sym.diff(advec_sym[1], y) # div advection
accum_sym = phi * sym.diff(rho_sym, t)  # phi drho/dt
f_sym = accum_sym + advec_div_sym  # source term

# Convert symbolic expressions to functions for fast evaluation
p_ex = sym.lambdify((x, y, t), p_sym, "numpy")
q_ex = sym.lambdify((x, y, t), q_sym, "numpy")
f_ex = sym.lambdify((x, y, t), f_sym, "numpy")

#%% Assign data
param_key = "flow"
pressure_variable = "pressure"
d[pp.PRIMARY_VARIABLES] = {pressure_variable: {"cells": 1}}

def assign_data(g, d, param_key, time):
    
    nf = g.num_faces
    nc = g.num_cells
    fn = g.face_normals
    fc = g.face_centers
    cc = g.cell_centers
    V = g.cell_volumes

    perm = pp.SecondOrderTensor(K * np.ones(nc))

    top = np.where(np.abs(fc[1] - 1) < 1e-5)[0]
    bottom = np.where(np.abs(fc[1]) < 1e-5)[0]
    left = np.where(np.abs(fc[0]) < 1e-5)[0]
    right = np.where(np.abs(fc[0] - 1) < 1e-5)[0]
    
    bc_faces = g.get_boundary_faces()
    bc_type = np.array(bc_faces.size * ["neu"])
    #bc_type = np.array(bc_faces.size * ["dir"])
    bc_type[np.in1d(bc_faces, left)] = "dir"
    bc_type[np.in1d(bc_faces, right)] = "dir"
    bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
                    
    bc_values = np.zeros(g.num_faces)
    pf = p_ex(fc[0], fc[1], time * np.ones(nf))
    qf = q_ex(fc[0], fc[1], time * np.ones(nf))
    Qf = qf[0] * fn[0] + qf[1] * fn[1]
    
    bc_values[top] = np.abs(Qf[top]) # outflow flux 
    bc_values[bottom] = np.abs(Qf[bottom]) # outflow flux
    #bc_values[top] = pf[top] # outflow flux 
    #bc_values[bottom] = pf[bottom] # outflow flux
    bc_values[left] = pf[left]
    bc_values[right] = pf[right]
    
    source_term = f_ex(cc[0], cc[1], time * np.ones(nc)) * V

    specified_parameters = {
        "second_order_tensor": perm,
        "bc": bc,
        "bc_values": bc_values,
        "source": source_term,
        "mass_weight": phi * np.ones(nc),
    }
    
    if time == 0.0:
        pp.initialize_data(g, d, param_key, specified_parameters)
    else:
        d[pp.PARAMETERS][param_key]["bc_values"] = bc_values
        d[pp.PARAMETERS][param_key]["source"] = source_term
 

#%% Set initial states
cc = g.cell_centers
pp.set_state(d)
pp.set_iterate(d)
d[pp.STATE][pressure_variable] = p_ex(cc[0], cc[1], time * np.ones_like(cc[0]))
d[pp.STATE][pp.ITERATE][pressure_variable] = d[pp.STATE][pressure_variable].copy()

#%% AD variables and manager
grid_list = [g for g, _ in gb]
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
p = equation_manager.merge_variables([(g, pressure_variable) for g in grid_list])
p_m = p.previous_iteration()
p_n = p.previous_timestep()

#%% AD denisity function
def rho(p):
    return rho_ref * pp.ad.exp(c * (p - p_ref))

rho_ad = pp.ad.Function(rho, name="density")

#%% AD operators and discrete expressions/equations
assign_data(g, d, param_key, time)

# Grid operators
div_ad = pp.ad.Divergence(grid_list)  # discrete diveregence
bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)  # boundary vals
dir_bound_ad = DirBC(bound_ad, grid_list)  # dirichlet bc vals
   
# MPFA discretization
mpfa_ad = pp.ad.MpfaAd(param_key, grid_list)
# Obtain single phase flow to compute directionality of upwind scheme
flux_1p_ad = mpfa_ad.flux * p_m + mpfa_ad.bound_flux * bound_ad
   
# Face-averaging of densities
if avg_method == "arithmetic":
    arithmetic_avg = ArithmeticAverageAd(g, d, param_key)
    rho_faces_ad = arithmetic_avg(rho_ad(p_m), rho_ad(dir_bound_ad))
elif avg_method == "upwind":
    upwind = UpwindFluxBasedAd(g, d, param_key)
    rho_faces_ad = upwind(rho_ad(p_m), rho_ad(dir_bound_ad), flux_1p_ad)
else:
    raise ValueError("Averaging method not implemented")
   
# Fluxes
darcy_flux = mpfa_ad.flux * p + mpfa_ad.bound_flux * bound_ad
advective_flux_ad =  rho_faces_ad * darcy_flux

# Source and accumulation terms
source_ad = pp.ad.ParameterArray(param_key, "source", grids=grid_list)
mass_ad = pp.ad.MassMatrixAd(param_key, grid_list)
accumulation_ad = mass_ad.mass * (rho_ad(p) - rho_ad(p_n))

# Continuity equation
continuity_ad = accumulation_ad + dt * div_ad * advective_flux_ad - dt * source_ad

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
    
    
    print("Current time: ", np.round(time, decimals=3))
    assign_data(g, d, param_key, time)

    while iteration_counter <= 30 and not (rel_res < 1e-12 or residual_norm < 1e-12):

        # Solve for pressure increment and update pressure
        A, b = equation_manager.assemble_matrix_rhs()
        pressure_increment = spla.spsolve(A, b)
        d[pp.STATE][pp.ITERATE][pressure_variable] += pressure_increment

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
    d[pp.STATE][pressure_variable] = d[pp.STATE][pp.ITERATE][pressure_variable].copy()
    d[pp.STATE]["p_ex"] = p_ex(
        g.cell_centers[0], g.cell_centers[1], time * np.ones(g.num_cells)
    )
    
#%% Compute discrete L2-errors
num_pressure = d[pp.STATE][pressure_variable].copy()
true_pressure = d[pp.STATE]["p_ex"]
p_error = l2_error(g, num_pressure, true_pressure, "pressure")

fn = g.face_normals
fc = g.face_centers
tf = time * np.ones(g.num_faces)
flux_ad_eval = pp.ad.Expression(darcy_flux, dof_manager)
flux_ad_eval.discretize(gb)
num_flux = flux_ad_eval.to_ad(gb)
true_flux = fn[0] * q_ex(fc[0], fc[1], tf)[0] + fn[1] * q_ex(fc[0], fc[1], tf)[1]
q_error = l2_error(g, num_flux.val, true_flux, "flux")
print(f"Pressure error: {p_error} \t | \t Flux error: {q_error}")