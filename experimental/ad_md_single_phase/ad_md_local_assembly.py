import numpy as np
import porepy as pp

import scipy.sparse.linalg as spla
from matplotlib.pyplot import spy as sparsity
from porepy.numerics.ad.grid_operators import DirBC


def make_triangular_mesh(mesh_size=0.1):

    domain = {"xmin": 0.0, "xmax": 1, "ymin": 0.0, "ymax": 1}
    points = np.array([[0.5, 0], [0.5, 1]]).T
    fractures = np.array([[0, 1]]).T
    network_2d = pp.FractureNetwork2d(points, fractures, domain)
    mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
    gb = network_2d.mesh(mesh_args)

    return gb


def make_toy_grid():
    vert_frac = np.array([[0.5, 0.5], [0, 1.0]])
    nx = np.array([2, 2])
    L = np.array([1.0, 1.0])
    gb = pp.meshing.cart_grid([vert_frac], nx=nx, physdims=L)
    return gb


#%% Create mixed-dimensional grid
# gb = make_triangular_mesh(mesh_size=0.5)
gb = make_toy_grid()

g_2d = gb.grids_of_dimension(2)[0]
g_1d = gb.grids_of_dimension(1)[0]
for _, d in gb.edges():
    mg = d["mortar_grid"]
d_2d = gb.node_props(g_2d)
d_1d = gb.node_props(g_1d)
d_e = gb.edge_props([g_1d, g_2d])

# String representations of the variables.
pressure_var = "pressure"
mortar_var = "mortar_flux"

#%% Define variables on subdomains and interfaces
# Loop over all subdomains
for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}

# Also loop over interfaces
for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {mortar_var: {"cells": 1}}

#%% Assign data
param_key = "flow"
matrix_perm = 1
fracture_perm = 1
interface_diffusivity = 1

for g, d in gb:
    if g.dim == 2:
        perm = pp.SecondOrderTensor(matrix_perm * np.ones(g.num_cells))

        # Dirichlet conditions on right and left
        left = np.where(np.abs(g.face_centers[0] - gb.bounding_box()[0][0]) < 1e-6)[0]
        right = np.where(np.abs(g.face_centers[0] - gb.bounding_box()[1][0]) < 1e-6)[0]
        bc_cond = ["dir"] * (left.size + right.size)
        bc = pp.BoundaryCondition(g, np.hstack((left, right)), bc_cond)

        bc_val = np.zeros(g.num_faces)
        bc_val[left] = 1
        specified_data = {"second_order_tensor": perm, "bc": bc, "bc_values": bc_val}
        d = pp.initialize_data(g, d, param_key, specified_data)

    else:
        perm = pp.SecondOrderTensor(fracture_perm * np.ones(g.num_cells))

        # No-flow Neumann conditions
        bc = pp.BoundaryCondition(g)
        bc_val = np.zeros(g.num_faces)
        specified_data = {"second_order_tensor": perm, "bc": bc, "bc_values": bc_val}
        d = pp.initialize_data(g, d, param_key, specified_data)

# Initialize data for interfaces as well
for e, d in gb.edges():
    mg = d["mortar_grid"]
    kn = interface_diffusivity * np.ones(mg.num_cells)
    pp.initialize_data(mg, d, param_key, {"normal_diffusivity": kn})

#%% Give some numerical values
for g, d in gb:
    pp.set_state(d)
    d[pp.STATE][pressure_var] = np.zeros(g.num_cells)

for e, d in gb.edges():
    pp.set_state(d)
    d[pp.STATE][mortar_var] = np.zeros(d["mortar_grid"].num_cells)

#%% Grid and edge lists
grid_list = [g for g, _ in gb]
edge_list = [e for e, _ in gb.edges()]

#%% Mixed-dimensional AD variables
dof_manager = pp.DofManager(gb)  # note: no pp.ad here
equation_manager = pp.ad.EquationManager(gb, dof_manager)

# Define AD variables
p = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])
lmbda = equation_manager.merge_variables([(e, mortar_var) for e in edge_list])

#%% Grid operators
mortar_proj = pp.ad.MortarProjections(gb=gb)
subdomain_proj_scalar = pp.ad.SubdomainProjections(gb=gb)

bulk_cell_rest = subdomain_proj_scalar.cell_restriction(g_2d)
bulk_face_rest = subdomain_proj_scalar.face_restriction(g_2d)
bulk_cell_prol = subdomain_proj_scalar.cell_prolongation(g_2d)

frac_cell_rest = subdomain_proj_scalar.cell_restriction(g_1d)
frac_face_rest = subdomain_proj_scalar.face_restriction(g_1d)
frac_cell_prol = subdomain_proj_scalar.cell_prolongation(g_1d)

#%% Equations for the bulk
div_bulk = pp.ad.Divergence([g_2d])
bound_bulk = pp.ad.BoundaryCondition(param_key, grids=[g_2d])
mpfa_bulk = pp.ad.MpfaAd(param_key, [g_2d])

full_flux_bulk = (
    mpfa_bulk.flux * bulk_cell_rest * p
    + mpfa_bulk.bound_flux * bound_bulk
    + mpfa_bulk.bound_flux * bulk_face_rest * mortar_proj.mortar_to_primary_int * lmbda
)

continuity_bulk = div_bulk * full_flux_bulk

# # Evaluate continuity in the bulk
cons_bulk_eval = pp.ad.Expression(continuity_bulk, dof_manager)
cons_bulk_eval.discretize(gb)
# cons_bulk_num = cons_bulk_eval.to_ad(gb)
# print(f'>> Conservation in the bulk:\n {cons_bulk_num.jac.toarray()} \n')

#%% Equations for the fracture
div_frac = pp.ad.Divergence([g_1d])
bound_frac = pp.ad.BoundaryCondition(param_key, grids=[g_1d])
mpfa_frac = pp.ad.MpfaAd(param_key, [g_1d])

full_flux_frac = mpfa_frac.flux * frac_cell_rest * p + mpfa_frac.bound_flux * bound_frac

sources_from_mortar = frac_cell_rest * mortar_proj.mortar_to_secondary_int * lmbda
continuity_frac = div_frac * full_flux_frac + sources_from_mortar

# # Evaluate continuity in the fracture
cons_frac_eval = pp.ad.Expression(continuity_frac, dof_manager)
cons_frac_eval.discretize(gb)
# cons_frac_num = cons_frac_eval.to_ad(gb)
# print(f'>> Conservation in the fracture: \n {cons_frac_num.jac.toarray()} \n')

#%% Equations for the interface
mpfa = pp.ad.MpfaAd(param_key, grid_list)
pressure_trace_from_high = (
    mortar_proj.primary_to_mortar_avg * mpfa.bound_pressure_cell * p
    + mortar_proj.primary_to_mortar_avg
    * mpfa.bound_pressure_face
    * mortar_proj.mortar_to_primary_int
    * lmbda
)
robin = pp.ad.RobinCouplingAd(param_key, edge_list)

interface_flux_eq = (
    robin.mortar_scaling
    * (pressure_trace_from_high - mortar_proj.secondary_to_mortar_avg * p)
    + robin.mortar_discr * lmbda
)

interface_eval = pp.ad.Expression(interface_flux_eq, dof_manager)
interface_eval.discretize(gb)
# interface_num= interface_eval.to_ad(gb)
# print(f'>> Interface flux: \n {interface_num.jac.toarray()} \n')


#%% Assemble the system of equations

eqs = [
    pp.ad.Expression(continuity_bulk, dof_manager),
    pp.ad.Expression(continuity_frac, dof_manager),
    pp.ad.Expression(interface_flux_eq, dof_manager),
]
equation_manager.equations += eqs

# Don't use global discretization
# equation_manager.discretize(gb)

# # next assemble the equations
A, b = equation_manager.assemble_matrix_rhs()

# # Solve, system, note the minus sign on the right hand side
# solution = spla.spsolve(A, b)

# # Distribute variable to local data dictionaries
# dof_manager.distribute_variable(solution, additive=True)
