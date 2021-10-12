#%% Importing modules
import porepy as pp
import numpy as np


gb = pp.meshing.cart_grid(fracs=[], nx=np.array([2, 2]), physdims=[1, 1])
gb.compute_geometry()
g = gb.grids_of_dimension(2)[0]
d = gb.node_props(g)
pressure_variable = "pressure"

keyword = "flow"
discr = pp.Mpfa(keyword)

# Assign data
nf = g.num_faces
nc = g.num_cells
fn = g.face_normals
fc = g.face_centers
V = g.cell_volumes

# Permeability tensor
perm = pp.SecondOrderTensor(np.ones(nc))

# Boundary conditions
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
bc_values[left] = np.array([5, 40])
bc_values[right] = np.array([5, 50])
bc_values[top] = np.array([-1, +1])
bc_values[bottom] = np.array([+1, -1])
specified_parameters = {
    "second_order_tensor": perm,
    "bc": bc,
    "bc_values": bc_values,
}

# Initialize data
pp.initialize_default_data(g, d, keyword, specified_parameters)

# Declare grid primary variable
d[pp.PRIMARY_VARIABLES] = {pressure_variable: {"cells": 1}}

# Assign discretization
d[pp.DISCRETIZATION] = {pressure_variable: {"diff": discr}}

# Initialize state
d[pp.STATE] = {
    pressure_variable: np.array([30, 10, 20, 40]),
    pp.ITERATE: {pressure_variable: np.array([30, 10, 20, 40])},
}

dof_manager = pp.DofManager(gb)
assembler = pp.Assembler(gb, dof_manager)
assembler.discretize()

# Reference discretization
A_ref, b_ref = assembler.assemble_matrix_rhs()
manager = pp.ad.EquationManager(gb, dof_manager)
grid_list = [g]

node_discr = pp.ad.MpfaAd(keyword, grid_list)
bc_val = pp.ad.BoundaryCondition(keyword, grid_list)
div = pp.ad.Divergence(grids=grid_list)
p = manager.merge_variables([(g, pressure_variable) for g in grid_list])

flux = node_discr.flux * p + node_discr.bound_flux * bc_val
flow_eq = div * flux

flow_eq_ad = pp.ad.Expression(flow_eq, dof_manager, "flow on nodes")
flow_eq_ad.discretize(gb)
manager.equations.clear()
manager.equations.append(flow_eq_ad)


# Evaluate flow
flux_eval = pp.ad.Expression(flux, dof_manager)
flux_eval.discretize(gb)
flux_num = flux_eval.to_ad(gb)


state = np.zeros(g.num_cells)
A, b = manager.assemble_matrix_rhs(state=state)
diff = A - A_ref
if diff.data.size > 0:
    assert np.max(np.abs(diff.data)) < 1e-10
assert np.max(np.abs(b - b_ref)) < 1e-10
