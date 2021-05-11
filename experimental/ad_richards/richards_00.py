"""
Convergence analysis for the incompressible version of Richards' equation using
the modified Picard iteration on a unit square

author: @jv
"""

import porepy as pp
import numpy as np
import sympy as sym

#%% Create unfractured 2d unstructured grid
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
pp.plot_grid(g, plot_2d=True)


#%% Populate data dictionaries with state
# for _, d in gb:
#     pp.set_state(d)
# for _, d in gb.edges():
#     pp.set_state(d)

#%% Physical parameters
theta_r = 0.1 # residual water content
theta_s = 0.4 # saturated water content
alpha_vG = 0.4 # van Genuchten parameter
n_vG = 2 # van Genuchten parameter
m_vG = 1 - 1 / n_vG # van Genuchten parameter
K_sat = 1 # saturated hydraulic conductivity

#%% Time levels
dt = 0.1  # time step
t0 = 0  # initial time
tf = 1  # final simulation time
ttol = 1e-6

#%% Exact solution
x = sym.symbols("x", real=True)
y = sym.symbols("y", real=True)
t = sym.symbols("t", real=True)

#exact_solution = "parabolic"  
exact_solution = "trigonometric"

# Exact pressure head
if exact_solution == "parabolic":
    psi_sym = t * (1 - x) * x * (1 - y) * y - 1
elif exact_solution == "trigonometric":
    psi_sym = sym.sin(np.pi * t) * sym.sin(np.pi * x) * sym.cos(np.pi * y) - 2
else:
    raise ValueError("Exact solution not available")

# Exact water content
theta_sym = (theta_s - theta_r) / (
    1 + (alpha_vG * sym.Abs(psi_sym)) ** n_vG
) ** m_vG - theta_r

# Exact relative permeability
krw_sym = (
    1
    - (alpha_vG * sym.Abs(psi_sym)) ** (n_vG - 1)
    * (1 + (alpha_vG * sym.Abs(psi_sym)) ** (n_vG)) ** (-m_vG)
) ** 2 / (1 + (alpha_vG * sym.Abs(psi_sym) ** n_vG) ** (m_vG / 2))

# Exact time derivative of the water content
theta_t_sym = sym.diff(theta_sym, t)

# Exact pressure head gradient
psi_grad_sym = [sym.diff(psi_sym, x), sym.diff(psi_sym, y)]

# Exact Darcy flux
q_sym = [-K_sat * krw_sym * psi_grad_sym[0], -K_sat * krw_sym * psi_grad_sym[1]]

# Exact divergence of the flux
q_div_sym = sym.diff(q_sym[0], x) + sym.diff(q_sym[1], y)

# Exact source term
f_sym = theta_t_sym + q_div_sym

# Convert symbolic expressions to functions for fast evaluation
psi_ex = sym.lambdify((x, y, t), psi_sym, "numpy")
theta_ex = sym.lambdify((x, y, t), theta_sym, "numpy")
q_ex = sym.lambdify((x, y, t), q_sym, "numpy")
f_ex = sym.lambdify((x, y, t), f_sym, "numpy")

# Create exporter and export initial data
# exporter = pp.Exporter(gb, "richards", "out")
# d[pp.STATE]["psi_ex"] = psi_ex(g.cell_centers[0], g.cell_centers[1], 0)
# d[pp.STATE]["theta_ex"] = theta_ex(g.cell_centers[0], g.cell_centers[1], 0)
# exporter.write_vtu(["psi_ex", "theta_ex"], time_step=0)

#%% Assign parameters
keyword = "flow"
pressure_variable = "psi"
flux_variable = "mortar_flux" # NB! Currently unused

discr = pp.Mpfa(keyword)

tt = 0  # current time
level = 0  # current time level
while tt + ttol < tf:

    # Assign parameters
    for g, d in gb:

        if g.dim == gb.dim_max():

            # Get hold of  boundary faces and assign Dirichlet bc to the bulk
            bc_faces = g.get_boundary_faces()
            bc_type = bc_faces.size * ["dir"]
            bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

            # Set the values, specified as a vector of size g.num_faces
            bc_vals = np.zeros(g.num_faces)
            bc_vals[bc_faces] = psi_ex(
                g.face_centers[0][bc_faces], 
                g.face_centers[1][bc_faces], 
                tt * np.ones_like(bc_faces)
                )

            # Integrated source terms are given by the exact solution
            integrated_source = g.cell_volumes * f_ex(
                g.cell_centers[0], 
                g.cell_centers[1], 
                tt * np.ones_like(g.num_cells)
            )
            
            # Create dictionary of parameters
            specified_parameters = {
                "bc": bc, "bc_values": bc_vals, "source": integrated_source
                }

            # Initialize default data
            pp.initialize_default_data(g, d, keyword, specified_parameters)
            
            # Declare primary variables and initialize states
            d[pp.PRIMARY_VARIABLES] = {pressure_variable: {"cells": 1}}
            d[pp.DISCRETIZATION] = {pressure_variable: {"diffusion": discr}}
            d[pp.STATE] = {
                pressure_variable: np.zeros(g.num_cells),
                pp.ITERATE: {pressure_variable: np.zeros(g.num_cells)},
            }

    for e, d in gb.edges():

        # Set the normal diffusivity
        data = {"normal_diffusivity": 1.0}

        # Add parameters: We again use keywords to identify sets of parameters
        mg = d["mortar_grid"]
        pp.initialize_data(mg, d, keyword, data)

    # Manager
    dof_manager = pp.DofManager(gb) # new way
    assembler = pp.Assembler(gb, dof_manager) # old way
    assembler.discretize() # old way
    manager = pp.ad.EquationManager(gb, dof_manager) # ad equation manager

    # Create grid and edge lists
    grid_list = [g for g, _ in gb]
    edge_list = [e for e, _ in gb.edges()] # empty for now

    # Discretize the problem using AD
    node_discr = pp.ad.MpfaAd(keyword, grid_list) # matrix operators are here
    bc_val = pp.ad.BoundaryCondition(keyword, grid_list)
    div = pp.ad.Divergence(grids=grid_list)
    
    # Declare AD variables
    psi = manager.merge_variables([(g, pressure_variable) for g in grid_list])

    # Declare discrete equations in a mixed-dimensional way
    flux = node_discr.flux * psi + node_discr.bound_flux * bc_val
    flow_eq = div * flux 
    
    flow_eq_ad = pp.ad.Expression(flow_eq, dof_manager, "flow on nodes")
    flow_eq_ad.discretize(gb)
    
    # Print info
    print("Time:", np.round(tt + dt, decimals=1))

    # Increase time step
    tt += dt
    level += 1

    # # Export exact cell-based quantities to Paraview
    # d[pp.STATE]["psi_ex"] = psi_ex(g.cell_centers[0], g.cell_centers[1], tt)
    # d[pp.STATE]["theta_ex"] = theta_ex(g.cell_centers[0], g.cell_centers[1], tt)
    # exporter.write_vtu(["psi_ex", "theta_ex"], time_step=level)
