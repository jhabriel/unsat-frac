"""
Water infiltration of an initially dry soil column with seepage boundary conditions at the right side of the domain

This is an implementation of the so-called seepage boundary conditions for variable saturated domains.
Common situations where this type of boundary condition is relevant include the stability analysis of dykes and
and the transport on riverbeds.

In this study case, we assume that a sand sample [200 cm x 100 cm] is initially at 0.64 of saturation. In the beginning,
all sides of the sample are impervious to water, except the top boundary where a constant pressure head of -25 [cm] is
imposed. As time progresses, saturation (and therefore pressure head) increases, until the bottom of the soil column
becomes saturated. At this point, water is able to exit the domain through the saturated faces on the right side
of the domain. Initially, only a few faces from the bottom of the column are saturated, but later, several, and
eventually all the right side of the domain becomes saturated.

Because the amount of faces that become saturated are not known a priori, the type and values of the boundary conditions
needed to impose at the righ side of the domain become part of the solution. Arguably, the simplest way to handle
this issue, is to solve the problem by considering it as a static obstacle problem. Mathematically, the boundary
condition is then given by a variational inequality of the type:

                                q_w . n = 0,      if      \psi < 0  on \partial\Omega_right
                                psi_w   = 0,      if      \psi >=0  on \partial\Omega_right

In terms of implementation, what we do is to assume, initially, that all sides are no-flow (except the top side),
solve the flow problem, and then check the values of the pressure at the faces of the right boundary. If the faces
pressure are greater than 0, it means that water "wants" to escape, but it is not able to, due to the no flow condition.
Thus, we go back to the previous state, and recompute the solution, this time, imposing a Dirichlet boundary condition
on the faces that became saturated. Since water is connected to the atmosphere, a value of 0 is set at these faces.

At the next time step, we again assume that all sides are no-flow, solve the problem, and update the amount
of boundary faces that will become saturated. This is needed, since the exterior conditions may change with time,
and thus we cannot assume that a previous saturated face will remain saturated in the next time step. Imagine a sudden
interruption in the water recharge at the top, or alternative, a sudden evaporation due to intense sunlight radiation.

@author: jhabriel.varela@uib.no
"""

#%% Import modules
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla

from porepy.numerics.ad.grid_operators import DirBC, EdgeParameterArray

from mdunsat.ad_utils.ad_utils import (
    ArithmeticAverageAd,
    UpwindFluxBasedAd,
    vanGenuchten,
)


def eval_ad_expression(
        ad_expression: pp.ad.Operator,
        grid_bucket: pp.GridBucket,
        dof_manager: pp.DofManager,
        name: str = None,
        print_expression: bool = False) -> tuple:

    """
    Utility function for rapid evaluation of ad expressions.

    NOTE:
    -----
    This function is meant mostly for debugging purposes, use it with caution otherwise.

    PARAMETERS:
    ----------
    ad_expression: pp.ad.Operator
        (Active) ad expression to be evaluated. Note that if atomic ad variables are passed, it will result in an error
    grid_bucket: pp.GridBucket
        Grid bucket containing the mixed-dimensional geometry and necessary discretization matrices in the respective
        data dictionaries
    dof_manager: pp.DofManager
        Degree of freedom manager of the ad problem
    name: (optional) String
        Name of the ad expression passed by the user. Default is None.
    print_expression: (optional) Bool
       Prints the value and the Jacobian of the resulting ad expression in the console. Default is False.

    RAISES:
    ------
    TypeError:
        If ad_expression is not of the type: pp.ad.Operator

    RETURNS:
    --------
    expression_num.val: np.nd_array
        Values of the evaluated expression
    expression_num.jac : sps.spmatrix
        Jacobian of the evaluated expression
    """

    # Sanity check: Make sure ad expression is of the type pp.ad.Operator
    # In particular, we want to avoid evaluating atomic ad variables and numpy-like objects
    if not isinstance(ad_expression, pp.ad.Operator):
        raise TypeError("Ad expression can only be of the type pp.ad.Operator")

    # Evaluate ad expression
    expression_eval = pp.ad.Expression(ad_expression, dof_manager)

    # Discretize ad expression
    expression_eval.discretize(grid_bucket)

    # Parse expression to retrieve the numerical values
    expression_num = expression_eval.to_ad(grid_bucket)

    # Print if necessary: Meant only for small arrays and matrices, a.k.a. debugging.
    if print_expression:
        if name is None:
            print('Evaluation of ad expression: \n')
            print(f'Array with values: \n {expression_num.val} \n')
            print(f'Jacobian with values: \n {expression_num.jac.A} \n')
        else:
            print(f'Evaluation of ad expression: {name} \n')
            print(f'Array with values: \n {expression_num.val} \n')
            print(f'Jacobian with values: \n {expression_num.jac.A} \n')

    return expression_num.val, expression_num.jac


#%% Model specifications
avg_method = "upwind"

#%% Make grid
gb = pp.meshing.cart_grid([], nx=[50, 100], physdims=[200, 100])
g = gb.grids_of_dimension(2)[0]
d = gb.node_props(g)

dim = gb.dim_max()
z_cc = g.cell_centers[dim-1]
z_fc = g.face_centers[dim-1]

#%% Physical parameters
K_sat = 0.00922  # [cm/s] Saturated hydraulic conductivity
theta_r = 0.102  # [-] residual water content
theta_s = 0.368  # [-] saturated water content, a.k.a porosity
alpha_vG = 0.0335  # [1/cm] van Genuchten parameter
n_vG = 2  # van Genuchten parameter
m_vG = 1 - 1 / n_vG  # van Genuchten parameter

#%% Time parameters
final_time = 6 * pp.HOUR
num_time_steps = 120
time = 0
dt = final_time / num_time_steps

#%% Assign parameters
param_key = "flow"
pressure_var = "pressure_head"
d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}

Lx = g.bounding_box()[1][0]
Ly = g.bounding_box()[1][1]
fc = g.face_centers
top = np.where(np.abs(fc[1] - Ly) < 1e-5)[0]
bottom = np.where(np.abs(fc[1]) < 1e-5)[0]
left = np.where(np.abs(fc[0]) < 1e-5)[0]
right = np.where(np.abs(fc[0] - Lx) < 1e-5)[0]

def assign_data(g, d, param_key, sat_faces=[]):

    nc = g.num_cells
    perm = pp.SecondOrderTensor(K_sat * np.ones(nc))

    bc_faces = g.get_boundary_faces()
    bc_type = np.array(bc_faces.size * ["neu"])  # bc faces are initially neu
    bc_type[np.in1d(bc_faces, top)] = "dir"
    bc_type[np.in1d(bc_faces, sat_faces)] = "dir"  # we set seepage face as dirichlet boundaries
    bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

    bc_values = np.zeros(g.num_faces)
    bc_values[top] = -25
    bc_values[sat_faces] = 0

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

    pp.initialize_data(g, d, param_key, specified_parameters)

#%% Set initial states
cc = g.cell_centers
pp.set_state(d)
pp.set_iterate(d)
d[pp.STATE][pressure_var] = -50 * np.ones(g.num_cells)
d[pp.STATE][pp.ITERATE][pressure_var] = d[pp.STATE][pressure_var].copy()

#%% AD variables and manager
grid_list = [g for g, _ in gb]
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
psi = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])
psi_m = psi.previous_iteration()
psi_n = psi.previous_timestep()

#%% AD operators and discrete expressions
assign_data(g, d, param_key)

# Grid operators
div_ad = pp.ad.Divergence(grid_list)  # discrete diveregence
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
h_m = psi_m + z_cc
flux_1p_ad = mpfa_ad.flux * h_m + mpfa_ad.bound_flux * bound_ad

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
h = psi + z_cc
flux_ad = krw_faces_ad * (mpfa_ad.flux * h + mpfa_ad.bound_flux * bound_ad)

# Source and accumulation terms (Linearization: Modified Picard iteration)
# Note: The expression containing the active ad variable (psi) must
# be placed at the left of the (final) expression. Otherwise, numpy
# will broadcast the expression leading to and undesirable ouput format
source_ad = pp.ad.ParameterArray(param_key, "source", grids=grid_list)
mass_ad = pp.ad.MassMatrixAd(param_key, grid_list)
accum_active = mass_ad.mass * psi * C_ad(psi_m)
accum_inactive = mass_ad.mass * (theta_ad(psi_m) - C_ad(psi_m) * psi_m - theta_ad(psi_n))
accumulation_ad = accum_active + accum_inactive

# Continuity equation
continuity_ad = accumulation_ad + dt * div_ad * flux_ad - dt * source_ad

# We need to keep track of the pressure traces
h_trace = mpfa_ad.bound_pressure_cell * h + mpfa_ad.bound_pressure_face * bound_ad
psi_trace = h_trace - z_fc

#%% Assemble the system of equations
eqs = pp.ad.Expression(continuity_ad, dof_manager)  # convert to expression
equation_manager.equations.clear()
equation_manager.equations.append(eqs)  # feed eq to the equation manager

#%% Initialize exporter
exporter = pp.Exporter(gb, "new_mexico", "out")
exporter.write_vtu([pressure_var], time_step=0)

#%% Time loop
total_iteration_counter = 0

# Time loop
for n in range(1, num_time_steps + 1):
    time += dt
    recompute_solution = True
    sat_faces = []
    control_faces = []
    d[pp.STATE]["water_table"] = np.zeros(g.num_cells)
    print("Current time: ", np.round(time, decimals=1))

    # Control Loop
    while recompute_solution:

        iteration_counter = 0
        residual_norm = 1
        rel_res = 1
        assign_data(g, d, param_key, sat_faces)
        equation_manager.discretize(gb)  # we need to re-discretize the problem in the control loop

        # Solver loop
        while iteration_counter <= 100 and not (rel_res < 1e-5 or residual_norm < 1e-5):

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
            # end of iteration loop

        # Seepage control
        trace_val, _ = eval_ad_expression(psi_trace, gb, dof_manager, print_expression=False)
        sat_faces = right[trace_val[right] >= 0]  # retrieve saturated faces
        if not list(sat_faces) == list(control_faces):  # avoid infinite loop
            control_faces = sat_faces  # make a copy of the sat faces to compare in the next iteration
            print(f"The faces {sat_faces} are saturated. Recomputing solution...")
            d[pp.STATE][pp.ITERATE][pressure_var] = d[pp.STATE][pressure_var].copy() # go back to the previous state
        else:
            recompute_solution = False
        # end of control loop

    # Update next time step solution
    d[pp.STATE][pressure_var] = d[pp.STATE][pp.ITERATE][pressure_var].copy()
    print()

    # Export to PARAVIEW
    if np.mod(n, 1) == 0:
        exporter.write_vtu([pressure_var, "water_table"], time_step=n)
