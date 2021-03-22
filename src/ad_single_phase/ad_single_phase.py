"""
Implementation of single phase flow in an unfractured domain using
automatic differentiation
"""

#%% Import modules
import numpy as np
import porepy as pp
import sympy as sym

import scipy.sparse.linalg as spla

#%% Create grid
def make_gb(mesh_size=0.05, L=[1.0, 1.0]):

    domain = {"xmin": 0.0, "xmax": L[0], "ymin": 0.0, "ymax": L[1]}
    network_2d = pp.FractureNetwork2d(None, None, domain)
    target_h_bound = target_h_fracture = target_h_min = mesh_size
    mesh_args = {
        "mesh_size_bound": target_h_bound,
        "mesh_size_frac": target_h_fracture,
        "mesh_size_min": target_h_min,
    }
    gb = network_2d.mesh(mesh_args)

    return gb

gb = make_gb()
g = gb.grids_of_dimension(2)[0]
#pp.plot_grid(g, plot_2d=True)

#%% Create mixed-dimensional grid
pressure_var = 'pressure'
mortar_var = 'mortar_flux'

#%% Physical parameters
k = 1 # intrinsic permeability

#%% Exact solution
x = sym.symbols("x", real=True)
y = sym.symbols("y", real=True)

exact_sol = "parabolic"

if exact_sol == "parabolic":
    p_sym = (1 - x) * x * (1 - y) * y
elif exact_sol == "trigonometric":
    p_sym = sym.sin(np.pi * x) * sym.cos(np.pi * y)
else:
    raise ValueError("Exact solution not available")

# Exact pressure gradient
p_grad_sym = [sym.diff(p_sym, x), sym.diff(p_sym, y)]

# Exact Darcy flux
q_sym = [-k * p_grad_sym[0], -k * p_grad_sym[1]]

# Exact divergence of the flux
q_div_sym = sym.diff(q_sym[0], x) + sym.diff(q_sym[1], y)

# Exact source term
f_sym = q_div_sym

# Convert symbolic expressions to functions for fast evaluation
p_ex = sym.lambdify((x, y), p_sym, "numpy")
q_ex = sym.lambdify((x, y), q_sym, "numpy")
f_ex = sym.lambdify((x, y), f_sym, "numpy")

#%% Define variables on subdomains and interfaces
# Loop over all subdomains
for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {'cells': 1}}
    
# Also loop over interfaces
for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {mortar_var: {'cells': 1}}
    
#%% Assign data    
param_key = 'flow'

matrix_perm = 1
fracture_perm = 1e2
interface_diffusivity = 1e2

for g, d in gb:
    if g.dim == 2:
        perm = pp.SecondOrderTensor(k * np.ones(g.num_cells))

        # Dirichlet conditions on right and left
        #left = np.where(np.abs(g.face_centers[0] - gb.bounding_box()[0][0]) < 1e-6)[0]
        #right = np.where(np.abs(g.face_centers[0] - gb.bounding_box()[1][0]) < 1e-6)[0]
        #bc_cond = ['dir'] * (left.size + right.size)
        #bc = pp.BoundaryCondition(g, np.hstack((left, right)), bc_cond)
        
        #bc_val = np.zeros(g.num_faces)
        #bc_val[left] = 1
        
        bc_faces = g.get_boundary_faces()
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        bc_val = np.zeros(g.num_faces)
        bc_val[bc_faces] = p_ex(g.face_centers[0][bc_faces], g.face_centers[1][bc_faces])
        
        source_term = f_ex(g.cell_centers[0], g.cell_centers[1]) * g.cell_volumes
        mass_weights = np.ones_like(g.cell_volumes)
        mass = np.ones_like(g.cell_volumes)
        
        specified_data = {'second_order_tensor': perm,
                         'bc': bc,
                         'bc_values': bc_val,
                         "source": source_term,
                         "mass_weight": mass_weights,
                         "mass": mass}
        d = pp.initialize_data(g, d, param_key, specified_data)
        
    else:
        perm = pp.SecondOrderTensor(fracture_perm * np.ones(g.num_cells))
        
        # No-flow Neumann conditions
        bc = pp.BoundaryCondition(g)
        bc_val = np.zeros(g.num_faces)
        specified_data = {'second_order_tensor': perm,
                         'bc': bc,
                         'bc_values': bc_val}
        d = pp.initialize_data(g, d, param_key, specified_data)
        
# Initialize data for interfaces as well        
for e, d in gb.edges():
    mg = d['mortar_grid']
    kn = interface_diffusivity * np.ones(mg.num_cells)
    pp.initialize_data(mg, d, param_key, {'normal_diffusivity': kn})
    
#%% Give some numerical values
for g, d in gb:
    pp.set_state(d)
    d[pp.STATE][pressure_var] = np.random.rand(g.num_cells)

#%% Define grid-related operators
grid_list = [g for g, _ in gb]
#edge_list = [e for e, _ in gb.edges()]    

div = pp.ad.Divergence(grid_list)
bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)
source_ad = pp.ad.ScalarSource(param_key, grids=grid_list)

#%% Mixed-dimensional AD variables
dof_manager = pp.DofManager(gb)  # note: no pp.ad here
equation_manager = pp.ad.EquationManager(gb, dof_manager)

# Define AD variables
p = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])
#lmbda = equation_manager.merge_variables([(e, mortar_var) for e in edge_list])

#%% Mixed-dimensional AD equations
mpfa = pp.ad.MpfaAd(param_key, grid_list)
mass_ad = pp.ad.MassMatrixAd(param_key, grid_list)
flux = mpfa.flux * p + mpfa.bound_flux * bound_ad
eval_flux = pp.ad.Expression(flux, dof_manager)
eval_flux.discretize(gb)
num_flux = eval_flux.to_ad(gb=gb)
print(num_flux)

#%%
  
vals = pp.ad.Expression(flux, dof_manager).to_ad(gb)
print(f'Size of value array: {vals.val.shape}')
print(f'Size of Jacobian matrix: {vals.jac.shape}')

pressure_eq = div * flux - source_ad

#%% Assemble the system of equations
    
eqs = [pp.ad.Expression(pressure_eq, dof_manager)]
equation_manager.equations += eqs

# first discretize
equation_manager.discretize(gb)
# next assemble the equations
A, b = equation_manager.assemble_matrix_rhs()

# Solve, system, note the minus sign on the right hand side
solution = spla.spsolve(A, b)

# Distribute variable to local data dictionaries
dof_manager.distribute_variable(solution, additive=True)

#%% Export to PARAVIEW

exporter = pp.Exporter(gb, '1p_unfrac_ad', 'out_single_phase')
exporter.write_vtu([pressure_var])

#%% 

