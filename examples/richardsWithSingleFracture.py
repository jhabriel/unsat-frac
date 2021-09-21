"""
        Water infiltration in a initially dry soil column

Grids: Cartesian (structured) or triangular (unstructured)
Water retention curves: van Genuchten-Mualem
Linearization scheme: Modified Picard Iteration (Celia et. al. 1990)
Face averaging: Upwind or Arithmetic
Exact manufactured solution: Parabolic or Trigonometric
Boundary conditions: Mixed. Dirichlet top and bottom. Neumann left and right.

@author: jhabriel.varela@uib.no
"""

#%% Import modules
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla

from porepy.numerics.ad.grid_operators import DirBC
from mdunsat.ad_utils.ad_utils import (
    ArithmeticAverageAd,
    UpwindFluxBasedAd,
    vanGenuchten,
)

#%% Model specifications
avg_method = "upwind" # artihmetic

#%% Make grid
domain = {"xmin": 0.0, "xmax": 100.0, "ymin": 0.0, "ymax": 100.0}
points = np.array([ [25, 75], [75, 25] ]).T
fracs = np.array([[0 , 1]]).T
network_2d = pp.FractureNetwork2d(points, fracs, domain)
mesh_size = 10
mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
gb = network_2d.mesh(mesh_args) 

g_2d = gb.grids_of_dimension(2)[0]
dim = gb.dim_max()
z_cc = g_2d.cell_centers[dim-1]
z_fc = g_2d.face_centers[dim-1]

#%% Physical parameters
K_sat = 0.00922  # [cm/s] Saturated hydraulic conductivity
theta_r = 0.102  # [-] residual water content
theta_s = 0.368  # [-] saturated water content, a.k.a porosity
alpha_vG = 0.0335  # [1/cm] van Genuchten parameter
n_vG = 2  # van Genuchten parameter
m_vG = 1 - 1 / n_vG  # van Genuchten parameter

#%% Time parameters
final_time = 72 * pp.HOUR
num_time_steps = 250
time = 0
dt = final_time / num_time_steps

#%% Assign parameters
param_key = "flow"
pressure_var = "pressure_head"
mortar_var = "mortar_flux"

for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}

for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {mortar_var: {"cells": 1}}


def assign_data(g, d, param_key, time):

    # Assign node parameters
    for g, d in gb:    

        if g.dim == 2:        

            nc = g.num_cells
            fc = g.face_centers
            dim = gb.dim_max()
            z_fc = g.face_centers[dim-1]
            Ly = g.bounding_box()[1][1]
        
            perm = pp.SecondOrderTensor(K_sat * np.ones(nc))
        
            top = np.where(np.abs(fc[1] - Ly) < 1e-5)[0]
            bottom = np.where(np.abs(fc[1]) < 1e-5)[0]
            
            bc_faces = g.get_boundary_faces()
            bc_type = np.array(bc_faces.size * ["neu"])
            bc_type[np.in1d(bc_faces, top)] = "dir"
            bc_type[np.in1d(bc_faces, bottom)] = "dir"
            bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
                            
            bc_values = np.zeros(g.num_faces)
            bc_values[top] = -75
            bc_values[bottom] = -1_000
            
            # Add gravity contribution to Dirichlet faces
            bc_values[bc.is_dir] += z_fc[bc.is_dir]
            
            source_term = np.zeros(nc)
        
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
            
        else:
             
            perm = pp.SecondOrderTensor(K_sat * np.ones(nc))
            specified_parameters = {
                "second_order_tensor": perm,
            }
            
        pp.initialize_data(g, d, param_key, specified_parameters)
    
    
    # Assign interface parameters
    for e, d in gb.edges():
        
        data = {"normal_diffusivity": 0}
        mg = d["mortar_grid"]
        pp.initialize_data(mg, d, param_key, data)
    
#%% Set initial states
for g, d in gb:
    pp.set_state(d)
    pp.set_iterate(d)
    if g.dim == 2:
        d[pp.STATE][pressure_var] = -1_000 * np.ones(g.num_cells)
    else:
        d[pp.STATE][pressure_var] = np.zeros(g.num_cells)
    d[pp.STATE][pp.ITERATE][pressure_var] = d[pp.STATE][pressure_var].copy()

for e, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][mortar_var] = np.zeros(mg.num_cells)
    d[pp.STATE][pp.ITERATE][mortar_var] = d[pp.STATE][mortar_var].copy()

#%% AD variables and manager
grid_list = [g for g, _ in gb]
edge_list = [e for e, _ in gb.edges()]
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
psi = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])
lmbda = equation_manager.merge_variables([(e, mortar_var) for e in edge_list])

psi_m = psi.previous_iteration()
psi_n = psi.previous_timestep()
lmbda_m = lmbda.previous_iteration()
lmbda_n = lmbda.previous_timestep()

#%% AD operators and discrete expressions
assign_data(g, d, param_key, time)

# Grid operators
div_ad = pp.ad.Divergence(grid_list)  # discrete diveregence
mortar_proj = pp.ad.MortarProjections(gb=gb)
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
flux_1p_ad = mpfa_ad.flux * (psi_m + z_cc) + mpfa_ad.bound_flux * bound_ad

# Face-averaging of relative permeabilities
if avg_method == "arithmetic":
    arithmetic_avg = ArithmeticAverageAd(g, d, param_key)
    krw_faces_ad = arithmetic_avg(krw_ad(psi_m), krw_ad(dir_bound_ad))
elif avg_method == "upwind":
    upwind = UpwindFluxBasedAd(g, d, param_key)
    krw_faces_ad = upwind(krw_ad(psi_m), krw_ad(dir_bound_ad), flux_1p_ad)
else:
    raise ValueError("Averaging method not implemented")

# Darcy fluxes
flux_ad = krw_faces_ad * (mpfa_ad.flux * (psi  + z_cc) + mpfa_ad.bound_flux * bound_ad)

# Source and accumulation terms (Linearization: Modified Picard iteration)
# Note: The expression containing the active ad variable (psi) must
# be placed at the left of the (final) expression. Otherwise, numpy
# will broadcast the expression leading to and undesirable ouput format
source_ad = pp.ad.ParameterArray(param_key, "source", grids=grid_list)
mass_ad = pp.ad.MassMatrixAd(param_key, grid_list)
accum_active = mass_ad.mass * psi * C_ad(psi_m)
accum_inactive = mass_ad.mass * (
    theta_ad(psi_m)
    - C_ad(psi_m) * psi_m
    - theta_ad(psi_n)
)
accumulation_ad = accum_active + accum_inactive

# Continuity equation
continuity_ad = accumulation_ad + dt * div_ad * flux_ad - dt * source_ad

#%% Assemble the system of equations
eqs = pp.ad.Expression(continuity_ad, dof_manager)  # convert to expression
equation_manager.equations.clear()
equation_manager.equations.append(eqs)  # feed eq to the equation manager
equation_manager.discretize(gb)  # discretize problem

#%% Initialize exporter
exporter = pp.Exporter(gb, "new_mexico", "out")
exporter.write_vtu([pressure_var], time_step=0)

#%% Time loop
total_iteration_counter = 0
for n in range(1, num_time_steps + 1):
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1
    time += dt
    
    print("Current time: ", np.round(time, decimals=1))
    
    while iteration_counter <= 30 and not (rel_res < 1e-6 or residual_norm < 1e-6):

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
    if np.mod(n, 10) == 0:
        exporter.write_vtu([pressure_var], time_step=n)