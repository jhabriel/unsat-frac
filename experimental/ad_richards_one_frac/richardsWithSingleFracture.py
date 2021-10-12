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
from matplotlib.pyplot import spy as sparsity

from porepy.numerics.ad.grid_operators import DirBC
from mdunsat.ad_utils.ad_utils import (
    ArithmeticAverageAd,
    UpwindFluxBasedAd,
    vanGenuchten,
)

#%% Model specifications
avg_method = "upwind"  # artihmetic

#%% Make grid
vert_frac = np.array([[50, 50], [0, 100]])
nx = np.array([2, 2])
L = np.array([100, 100])
gb = pp.meshing.cart_grid([vert_frac], nx=nx, physdims=L)
exporter = pp.Exporter(gb, "new_mexico", "out")
exporter.write_vtu()

dim_max = gb.dim_max()
g_bulk = gb.grids_of_dimension(dim_max)[0]
g_frac = gb.grids_of_dimension(dim_max - 1)[0]

d_bulk = gb.node_props(g_bulk)
d_frac = gb.node_props(g_frac)
d_edge = gb.edge_props((g_bulk, g_frac))

z_cc = g_bulk.cell_centers[dim_max - 1]
z_fc = g_bulk.face_centers[dim_max - 1]

#%% Physical parameters
K_sat = 0.00922  # [cm/s] Saturated hydraulic conductivity
theta_r = 0.102  # [-] residual water content
theta_s = 0.368  # [-] saturated water content, a.k.a porosity
alpha_vG = 0.0335  # [1/cm] van Genuchten parameter
n_vG = 2  # van Genuchten parameter
m_vG = 1 - 1 / n_vG  # van Genuchten parameter

#%% Time parameters
final_time = 3600 * 4
num_time_steps = 100
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


def assign_data(param_key, time):

    # Assign node parameters
    for g, d in gb:

        # Parameters for the bulk
        if g.dim == gb.dim_max():
            nc = g.num_cells
            fc = g.face_centers
            Lx = g.bounding_box()[1][0]

            perm = pp.SecondOrderTensor(K_sat * np.ones(nc))

            left = np.where(np.abs(fc[0]) < 1e-5)[0]
            right = np.where(np.abs(fc[0] - Lx) < 1e-5)[0]

            bc_faces = g.get_boundary_faces()
            bc_type = np.array(bc_faces.size * ["neu"])
            bc_type[np.in1d(bc_faces, left)] = "dir"
            bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

            bc_values = np.zeros(g.num_faces)
            bc_values[left] = -75
            # Add gravity contribution to Dirichlet faces
            # bc_values[bc.is_dir] += z_fc[bc.is_dir]

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

            pp.initialize_data(g, d, param_key, specified_parameters)

        # Parameters for the fracture
        else:
            perm = pp.SecondOrderTensor(K_sat * np.ones(g.num_cells))
            # No-flow Neumann conditions
            bc = pp.BoundaryCondition(g)
            bc_val = np.zeros(g.num_faces)
            specified_data = {
                "second_order_tensor": perm,
                "bc": bc,
                "bc_values": bc_val,
            }
            d = pp.initialize_data(g, d, param_key, specified_data)

    # Assign interface parameters
    for e, d in gb.edges():

        data = {"normal_diffusivity": K_sat}
        mg = d["mortar_grid"]
        pp.initialize_data(mg, d, param_key, data)


#%% Set initial states
for g, d in gb:
    pp.set_state(d)
    pp.set_iterate(d)
    if g.dim == 2:
        d[pp.STATE][pressure_var] = -1_000 * np.ones(g.num_cells)
    else:
        d[pp.STATE][pressure_var] = -1_000 * np.ones(g.num_cells)
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

#%% Assign data
assign_data(param_key, time)

#%% Declare equations for the bulk

# Grid operators
div_bulk = pp.ad.Divergence([g_bulk])
mortar_proj = pp.ad.MortarProjections(gb=gb)
bound_bulk = pp.ad.BoundaryCondition(param_key, grids=[g_bulk])  # boundary vals
dirbc_bulk = DirBC(bound_bulk, [g_bulk])  # dirichlet bc vals
subdomain_proj_scalar = pp.ad.SubdomainProjections(gb=gb)
bulk_cell_proj = subdomain_proj_scalar.cell_restriction(g_bulk)
bulk_face_proj = subdomain_proj_scalar.face_restriction(g_bulk)
frac_cell_proj = subdomain_proj_scalar.cell_restriction(g_frac)

# Soil water retention curves
swrc = vanGenuchten(g_bulk, d_bulk, param_key)
theta_ad = pp.ad.Function(swrc.water_content, "water content")
krw_ad = pp.ad.Function(swrc.relative_permeability, "relative permeability")
C_ad = pp.ad.Function(swrc.moisture_capacity, "moisture capacity")

# MPFA discretization
mpfa_bulk = pp.ad.MpfaAd(param_key, [g_bulk])

# Face-averaging of relative permeabilities
if avg_method == "arithmetic":
    arithmetic_avg = ArithmeticAverageAd(g_bulk, d_bulk, param_key)
    krw_faces_ad = arithmetic_avg(krw_ad(psi_m), krw_ad(dirbc_bulk))
elif avg_method == "upwind":
    upwind = UpwindFluxBasedAd(g_bulk, d_bulk, param_key)
    flux1p_bulk = (
        mpfa_bulk.flux * bulk_cell_proj * psi_m
        + mpfa_bulk.bound_flux * bound_bulk
        # + mpfa_bulk.bound_flux * bulk_face_rest * mortar_proj.mortar_to_primary_int * lmbda_m
    )
    krw_faces_ad = upwind(
        krw_ad(bulk_cell_proj * psi_m), krw_ad(dirbc_bulk), flux1p_bulk
    )
else:
    raise ValueError("Averaging method not implemented")

# Darcy fluxes
flux_bulk = (
    krw_faces_ad * mpfa_bulk.flux * bulk_cell_proj * psi
    + krw_faces_ad * mpfa_bulk.bound_flux * bound_bulk
    # + krw_faces_ad * mpfa_bulk.bound_flux * bulk_face_rest * mortar_proj.mortar_to_primary_int * lmbda
)

# Source and accumulation terms (Linearization: Modified Picard iteration)
# Note: The expression containing the active ad variable (psi) must
# be placed at the left of the (final) expression. Otherwise, numpy
# will broadcast the expression leading to and undesirable ouput format
source_bulk = pp.ad.ParameterArray(param_key, "source", grids=[g_bulk])
mass_bulk = pp.ad.MassMatrixAd(param_key, [g_bulk])
accum_bulk_active = mass_bulk.mass * bulk_cell_proj * psi * C_ad(bulk_cell_proj * psi_m)
accum_bulk_inactive = mass_bulk.mass * (
    theta_ad(bulk_cell_proj * psi_m)
    - C_ad(bulk_cell_proj * psi_m) * (bulk_cell_proj * psi_m)
    - theta_ad(bulk_cell_proj * psi_n)
)
accumulation_bulk = accum_bulk_active + accum_bulk_inactive
sources_from_mortar = mortar_proj.mortar_to_secondary_int * lmbda

conserv_bulk_eq = accumulation_bulk + dt * div_bulk * flux_bulk - dt * source_bulk
conserv_bulk_eval = pp.ad.Expression(conserv_bulk_eq, dof_manager)
conserv_bulk_eval.discretize(gb)
conserv_bulk_num = conserv_bulk_eval.to_ad(gb)
print(f">> Conservation bulk: \n {conserv_bulk_num.jac.A} \n")

# #%% Declare equations for the fracture
# conserv_frac_eq = (
#     0 * frac_cell_rest * (psi - psi_n)
#     + dt * frac_cell_rest * mortar_proj.mortar_to_secondary_int * lmbda
#     )
# conserv_frac_eval = pp.ad.Expression(conserv_frac_eq, dof_manager)
# conserv_frac_eval.discretize(gb)

# #%% Declare equations for the interface
# mpfa_global = pp.ad.MpfaAd(kw, grid_list)
# pressure_trace_from_high = (
#     mortar_proj.primary_to_mortar_avg * mpfa_global.bound_pressure_cell * psi
#     + mortar_proj.primary_to_mortar_avg * mpfa_global.bound_pressure_face * mortar_proj.mortar_to_primary_int * lmbda
#     )
# robin = pp.ad.RobinCouplingAd(kw, edge_list)

# interface_flux_eq = (
#     1e-30 * robin.mortar_scaling * (pressure_trace_from_high - mortar_proj.secondary_to_mortar_avg * psi)
#     + robin.mortar_discr * lmbda
#     )
# interface_flux_eval = pp.ad.Expression(interface_flux_eq, dof_manager)
# interface_flux_eval.discretize(gb)

# #%% Assemble the system of equations
# eqs = [
#        pp.ad.Expression(conserv_bulk_eq, dof_manager, name="conservation bulk"),
#        pp.ad.Expression(conserv_frac_eq, dof_manager, name="conservation fracture"),
#        pp.ad.Expression(interface_flux_eq, dof_manager, name="interface flux")
#        ]
# equation_manager.equations += eqs

#%% Initialize exporter
d_bulk[pp.STATE]["S_eff"] = swrc.effective_saturation(d_bulk[pp.STATE][pressure_var])
d_frac[pp.STATE]["S_eff"] = np.zeros(g_frac.num_cells)

exporter = pp.Exporter(gb, "new_mexico", "out")
exporter.write_vtu([pressure_var, "S_eff"], time_step=0)

#%% Time loop
total_iteration_counter = 0
for n in range(1, num_time_steps + 1):
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1
    time += dt

    print("Current time: ", np.round(time, decimals=1))

    while iteration_counter <= 30 and not (rel_res < 1e-6 or residual_norm < 1e-6):

        conserv_bulk_eval = pp.ad.Expression(conserv_bulk_eq, dof_manager)
        conserv_bulk_eval.discretize(gb)
        conserv_bulk_num = conserv_bulk_eval.to_ad(gb)

        b = -conserv_bulk_num.val
        A = conserv_bulk_num.jac
        y = spla.spsolve(A[: g_bulk.num_cells, : g_bulk.num_cells], b)
        d_bulk[pp.STATE][pp.ITERATE][pressure_var] += y

        # # Solve for pressure increment and update pressure
        # A, b = equation_manager.assemble_matrix_rhs()
        # solution = spla.spsolve(A, b)
        # assert False

        # # Distribute variable to local data dictionaries
        # dof_manager.distribute_variable(solution, additive=True)

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
    d_bulk[pp.STATE][pressure_var] = d_bulk[pp.STATE][pp.ITERATE][pressure_var].copy()
    d_bulk[pp.STATE]["S_eff"] = swrc.effective_saturation(
        d_bulk[pp.STATE][pressure_var]
    )

    # Export to PARAVIEW
    if np.mod(n, 10) == 0:
        exporter.write_vtu([pressure_var, "S_eff"], time_step=n)
