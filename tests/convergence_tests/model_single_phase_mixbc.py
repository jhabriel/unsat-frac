#%% Importing modules
import porepy as pp
import numpy as np
import sympy as sym
import scipy.sparse.linalg as spla

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


#%% Model parameters
solution = "parabolic"  # trigonometric
grid_type = "cartesian"  # triangular
# mesh_size = 0.5
# gb = make_grid(mesh_size, grid_type)
# g = gb.grids_of_dimension(2)[0]
# d = gb.node_props(g)

refine = 1
gb = pp.meshing.cart_grid([], nx=[4 * 2 ** refine, 4 * 2 ** refine], physdims=[1, 1])
g = gb.grids_of_dimension(2)[0]
d = gb.node_props(g)

#%% Time step size
final_time = 0.5
num_time_steps = 4 * 2 ** (2 * refine)
time = 0
dt = final_time / num_time_steps


#%% Exact solution
x = sym.symbols("x", real=True)
y = sym.symbols("y", real=True)

# Exact pressure distribution
if solution == "parabolic":
    p_sym = (1 - x) * x * (1 - y) * y
elif solution == "trigonometric":
    p_sym = sym.sin(sym.pi * x) * sym.cos(sym.pi * y)
else:
    raise ValueError("Exact solution not available")
p_grad_sym = [sym.diff(p_sym, x), sym.diff(p_sym, y)]
q_sym = [-p_grad_sym[0], -p_grad_sym[1]]
q_div_sym = sym.diff(q_sym[0], x) + sym.diff(q_sym[1], y)
f_sym = q_div_sym

# Convert symbolic expressions to functions for fast evaluation
p_ex = sym.lambdify((x, y), p_sym, "numpy")
q_ex = sym.lambdify((x, y), q_sym, "numpy")
f_ex = sym.lambdify((x, y), f_sym, "numpy")

# Assign data
keyword = "flow"
pressure_variable = "pressure"

nf = g.num_faces
nc = g.num_cells
fn = g.face_normals
fc = g.face_centers
cc = g.cell_centers
V = g.cell_volumes

perm = pp.SecondOrderTensor(np.ones(nc))

top = np.where(np.abs(fc[1] - 1) < 1e-5)[0]
bottom = np.where(np.abs(fc[1]) < 1e-5)[0]
left = np.where(np.abs(fc[0]) < 1e-5)[0]
right = np.where(np.abs(fc[0] - 1) < 1e-5)[0]

bc_faces = g.get_boundary_faces()
bc_type = np.array(bc_faces.size * ["neu"])
# bc_type[np.in1d(bc_faces, top)] = "neu"
# bc_type[np.in1d(bc_faces, bottom)] = "neu"
bc_type[np.in1d(bc_faces, left)] = "dir"
bc_type[np.in1d(bc_faces, right)] = "dir"
bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

bc_values = np.zeros(g.num_faces)
pf = p_ex(fc[0], fc[1])
qf = q_ex(fc[0], fc[1])
Qf = qf[0] * fn[0] + qf[1] * fn[1]
bc_values[bc.is_dir] = pf[bc.is_dir]
bc_values[bc.is_neu] = np.abs(Qf[bc.is_neu])  # outflow must be positive

source = f_ex(cc[0], cc[1]) * V

specified_parameters = {
    "second_order_tensor": perm,
    "bc": bc,
    "bc_values": bc_values,
    "source": source,
}

pp.initialize_default_data(g, d, keyword, specified_parameters)

# Declare grid primary variable
d[pp.PRIMARY_VARIABLES] = {pressure_variable: {"cells": 1}}

# Initialize state
d[pp.STATE] = {
    pressure_variable: np.zeros(g.num_cells),
    pp.ITERATE: {pressure_variable: np.zeros(g.num_cells)},
}

grid_list = [g for g, _ in gb]
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
p = equation_manager.merge_variables([(g, pressure_variable) for g in grid_list])

node_discr = pp.ad.MpfaAd(keyword, grid_list)
bc_val = pp.ad.BoundaryCondition(keyword, grid_list)
div = pp.ad.Divergence(grids=grid_list)
flux = node_discr.flux * p + node_discr.bound_flux * bc_val
source_ad = pp.ad.ParameterArray(keyword, "source", grids=grid_list)
flow_eq = div * flux - source_ad

flow_eq_ad = pp.ad.Expression(flow_eq, dof_manager, "flow on nodes")
flow_eq_ad.discretize(gb)
equation_manager.equations.clear()
equation_manager.equations.append(flow_eq_ad)

state = np.zeros(g.num_cells)
A, b = equation_manager.assemble_matrix_rhs()
pressure_increment = spla.spsolve(A, b)
d[pp.STATE][pp.ITERATE][pressure_variable] += pressure_increment
d[pp.STATE][pressure_variable] = d[pp.STATE][pp.ITERATE][pressure_variable].copy()

d[pp.STATE]["p_ex"] = p_ex(cc[0], cc[1])

# Evaluate flow
flux_eval = pp.ad.Expression(flux, dof_manager)
flux_eval.discretize(gb)
flux_num = flux_eval.to_ad(gb)

#%% Compute errors
num_pressure = d[pp.STATE][pressure_variable].copy()
true_pressure = d[pp.STATE]["p_ex"]
p_error = l2_error(g, num_pressure, true_pressure, "pressure")
print("L2-error for the pressure:", p_error)

flux_ad_eval = pp.ad.Expression(flux, dof_manager)
flux_ad_eval.discretize(gb)
num_flux = flux_ad_eval.to_ad(gb)
true_flux = fn[0] * q_ex(fc[0], fc[1])[0] + fn[1] * q_ex(fc[0], fc[1])[1]
q_error = l2_error(g, num_flux.val, true_flux, "flux")
print("L2-error for the flux:", q_error)
