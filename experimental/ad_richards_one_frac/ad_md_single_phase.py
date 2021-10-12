import numpy as np
import porepy as pp

import scipy.sparse.linalg as spla
from matplotlib.pyplot import spy as sparsity


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

#%% Define grid-related operators
grid_list = [g for g, _ in gb]
edge_list = [e for e, _ in gb.edges()]

div = pp.ad.Divergence(grid_list)
mortar_proj = pp.ad.MortarProjections(gb=gb)
bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)

#%% Mixed-dimensional AD variables
dof_manager = pp.DofManager(gb)  # note: no pp.ad here
equation_manager = pp.ad.EquationManager(gb, dof_manager)

# Define AD variables
p = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])
lmbda = equation_manager.merge_variables([(e, mortar_var) for e in edge_list])

#%% Mixed-dimensional AD equations
mpfa = pp.ad.MpfaAd(param_key, grid_list)
interior_flux = mpfa.flux * p
eval_flux = pp.ad.Expression(interior_flux, dof_manager)
eval_flux.discretize(gb)
num_flux = eval_flux.to_ad(gb=gb)
print(num_flux)

full_flux = (
    mpfa.flux * p
    + mpfa.bound_flux * bound_ad
    + mpfa.bound_flux * mortar_proj.mortar_to_primary_int * lmbda
)

vals = pp.ad.Expression(full_flux, dof_manager).to_ad(gb)
print(f"Size of value array: {vals.val.shape}")
print(f"Size of Jacobian matrix: {vals.jac.shape}")

sources_from_mortar = mortar_proj.mortar_to_secondary_int * lmbda

conservation = div * full_flux + sources_from_mortar

pressure_trace_from_high = (
    mortar_proj.primary_to_mortar_avg * mpfa.bound_pressure_cell * p
    + mortar_proj.primary_to_mortar_avg
    * mpfa.bound_pressure_face
    * mortar_proj.mortar_to_primary_int
    * lmbda
)

robin = pp.ad.RobinCouplingAd(param_key, edge_list)

# add heaviside function to the definition of the mortar flux
interface_flux_eq = (
    robin.mortar_scaling
    * (pressure_trace_from_high - mortar_proj.secondary_to_mortar_avg * p)
    + robin.mortar_discr * lmbda
)

#%% Checking the equations
# Bulk and fracture
conservation_eval = pp.ad.Expression(conservation, dof_manager)
conservation_eval.discretize(gb)
conservation_num = conservation_eval.to_ad(gb)
print(f">> Conservation equation: {conservation_num} \n")

# Interface
interface_flux = pp.ad.Expression(interface_flux_eq, dof_manager)
interface_flux.discretize(gb)
interface_flux_num = interface_flux.to_ad(gb)
print(f">> Interface flux: {interface_flux_num} \n")

#%% Assemble the system of equations

eqs = [
    pp.ad.Expression(conservation, dof_manager),
    pp.ad.Expression(interface_flux_eq, dof_manager),
]
equation_manager.equations += eqs

# first discretize
equation_manager.discretize(gb)
# next assemble the equations
A, b = equation_manager.assemble_matrix_rhs()

# Solve, system, note the minus sign on the right hand side
solution = spla.spsolve(A, b)

# Distribute variable to local data dictionaries
dof_manager.distribute_variable(solution, additive=True)

exporter = pp.Exporter(gb, "ad_test")
exporter.write_vtu([pressure_var])
