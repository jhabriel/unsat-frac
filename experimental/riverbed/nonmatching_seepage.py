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
import scipy.sparse as sps
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.serif": "Computer Modern Roman",
        "font.sans-serif": "Computer Modern Sans Serif",
        "font.cursive": "Zapf Chancery",
    }
)

from porepy.numerics.ad.grid_operators import DirBC
from mdunsat.ad_utils import (
    InterfaceUpwindAd,
    UpwindFluxBasedAd,
    vanGenuchten,
)


def eval_ad_expression(
    ad_expression: pp.ad.Operator,
    grid_bucket: pp.GridBucket,
    dof_manager: pp.DofManager,
    name: str = None,
    print_expression: bool = False,
) -> tuple:

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
            print("Evaluation of ad expression: \n")
            print(f"Array with values: \n {expression_num.val} \n")
            print(f"Jacobian with values: \n {expression_num.jac.A} \n")
        else:
            print(f"Evaluation of ad expression: {name} \n")
            print(f"Array with values: \n {expression_num.val} \n")
            print(f"Jacobian with values: \n {expression_num.jac.A} \n")

    return expression_num.val, expression_num.jac


#%% Make grid
fracs = np.array([[200, 200], [0, 100]])
domain_size = np.array([400, 100])

nx_ghost = np.array([40, 40])
nx_mono = np.array([2, 1])

# Ghost bucket... This gb should not be touched. We will not really use it, but rather
# retrieve mortar projections that we will use later
gb_ghost = pp.meshing.cart_grid([fracs], nx=nx_ghost, physdims=domain_size)
g_bulk_ghost = gb_ghost.grids_of_dimension(2)[0]
d_bulk_ghost = gb_ghost.node_props(g_bulk_ghost)
g_frac_ghost = gb_ghost.grids_of_dimension(1)[0]
d_frac_ghost = gb_ghost.node_props(g_frac_ghost)
d_edge_ghost = gb_ghost.edge_props((g_bulk_ghost, g_frac_ghost))
mg_ghost = d_edge_ghost["mortar_grid"]
pp.set_state(
    d_bulk_ghost, state={"pressure_head": -25 * np.ones(g_bulk_ghost.num_cells)}
)
pp.set_state(d_frac_ghost, state={"pressure_head": np.zeros(g_frac_ghost.num_cells)})

# Physical bucket... This gb will be the one reflects the actual number of dof that we
# will use for solving the problem. Initially, it is the same as the ghost bucket, but
# then it will be modified. In particular, fracture grids will be replaced with grids with
# single cells
gb = pp.meshing.cart_grid([fracs], nx=nx_ghost, physdims=domain_size)

# Grid bucket with a single fracture cell. Note that we don't really care about anything
# else here, the bulk and mortar grids won't be used, only the fracture grid with its
# single cell. So, for computational efficiency, we should use the minimum amount of cells
gb_mono = pp.meshing.cart_grid([fracs], nx=nx_mono, physdims=domain_size)

# Retrieve fracture grids
g_frac_old = gb.grids_of_dimension(1)[0]
g_frac_new = gb_mono.grids_of_dimension(1)[0]

# Create map dictionary to feed the replace_grids method. Key is the old grid,
# value is the new grid
g_map = {g_frac_old: g_frac_new}

# Replace grids
gb.replace_grids(g_map=g_map)

dim_max = gb.dim_max()
g_bulk = gb.grids_of_dimension(dim_max)[0]
g_frac = gb.grids_of_dimension(dim_max - 1)[0]
g_frac_ghost = gb_ghost.grids_of_dimension(dim_max - 1)[0]

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
final_time = 10 * pp.HOUR
num_time_steps = 400
time = 0
dt = final_time / num_time_steps

#%% Assign parameters
param_key = "flow"
pressure_var = "pressure_head"
bulk_var = "pressure_head"
fracture_var = "water_volume"
mortar_var = "mortar_flux"

# Declare primary variables
d_bulk[pp.PRIMARY_VARIABLES] = {bulk_var: {"cells": 1}}
d_frac[pp.PRIMARY_VARIABLES] = {fracture_var: {"cells": 1}}
for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {mortar_var: {"cells": 1}}

Lx = g_bulk.bounding_box()[1][0]
Ly = g_bulk.bounding_box()[1][1]

fc = g_bulk.face_centers
cc = g_bulk.cell_centers

bc_top = np.where(np.abs(fc[1] - Ly) < 1e-5)[0]
bc_bottom = np.where(np.abs(fc[1]) < 1e-5)[0]
bc_left = np.where(np.abs(fc[0]) < 1e-5)[0]
bc_right = np.where(np.abs(fc[0] - Lx) < 1e-5)[0]
bc_infilt = bc_top[np.where(np.abs(fc[0][bc_top] < (Lx / 2)))[0]]

cells_left = np.where(np.abs(cc[0] < (Lx / 2)))[0]
cells_right = np.where(np.abs(cc[0] > (Lx / 2)))[0]


def assign_data(param_key, is_conductive):

    # Assign parameters for the bulk
    perm = pp.SecondOrderTensor(K_sat * np.ones(g_bulk.num_cells))

    bc_faces = g_bulk.get_boundary_faces()
    bc_type = np.array(bc_faces.size * ["neu"])
    bc_type[np.in1d(bc_faces, bc_infilt)] = "dir"
    bc = pp.BoundaryCondition(g_bulk, faces=bc_faces, cond=bc_type)

    bc_values = np.zeros(g_bulk.num_faces)
    bc_values[bc_infilt] = -25  # pressure head on the left-top bc faces
    # Add gravity contribution to Dirichlet faces
    bc_values[bc.is_dir] += z_fc[bc.is_dir]

    source_term = np.zeros(g_bulk.num_cells)

    specified_parameters = {
        "second_order_tensor": perm,
        "bc": bc,
        "bc_values": bc_values,
        "source": source_term,
        "mass_weight": np.ones(g_bulk.num_cells),
        "theta_r": theta_r,
        "theta_s": theta_s,
        "alpha_vG": alpha_vG,
        "n_vG": n_vG,
        "m_vG": m_vG,
    }

    pp.initialize_data(g_bulk, d_bulk, param_key, specified_parameters)

    # Assign fracture parameters
    pp.initialize_data(g_frac, d_frac, param_key, specified_parameters={})

    # Assign interface parameters
    for _, d in gb.edges():
        mg = d["mortar_grid"]
        aperture = 0.1
        k_perp_val = K_sat / (2 * aperture)
        normal_diffusivity = k_perp_val * np.ones(mg.num_cells)
        data = {
            "normal_diffusivity": normal_diffusivity,
            "is_conductive": is_conductive,
        }
        pp.initialize_data(mg, d, param_key, data)


#%% Set initial states
for g, d in gb:
    pp.set_state(d)
    pp.set_iterate(d)
    if g.dim == dim_max:
        d[pp.STATE][bulk_var] = np.zeros(g.num_cells)
        d[pp.STATE][bulk_var][cells_left] = -25
        d[pp.STATE][bulk_var][cells_right] = -25
        d[pp.STATE][pp.ITERATE][bulk_var] = d[pp.STATE][bulk_var].copy()
    else:
        d[pp.STATE][fracture_var] = np.zeros(g.num_cells)
        d[pp.STATE][pp.ITERATE][fracture_var] = d[pp.STATE][fracture_var].copy()

d_frac[pp.STATE]["pressure_head"] = np.zeros(g_frac.num_cells)

for _, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][mortar_var] = 0 * np.ones(mg.num_cells)
    d[pp.STATE][pp.ITERATE][mortar_var] = d[pp.STATE][mortar_var].copy()

#%% AD variables and manager
bulk_grid_list = [g_bulk]
frac_grid_list = [g_frac]
edge_list = [e for e, _ in gb.edges()]

# Construct degree of freedom manager and equations manager
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

# Assign primary variables to their corresponding grids
psi_bulk = equation_manager.merge_variables([(g, bulk_var) for g in bulk_grid_list])
vol_frac = equation_manager.merge_variables([(g, fracture_var) for g in frac_grid_list])
lmbda = equation_manager.merge_variables([(e, mortar_var) for e in edge_list])

# Shorthands...
psi_bulk_m = psi_bulk.previous_iteration()
psi_bulk_n = psi_bulk.previous_timestep()
vol_frac_m = vol_frac.previous_iteration()
vol_frac_n = vol_frac.previous_timestep()
lmbda_m = lmbda.previous_iteration()
lmbda_n = lmbda.previous_timestep()

#%% AD operators and discrete expressions
assign_data(param_key, np.zeros(gb.num_mortar_cells(), dtype=np.int8))

# Grid operators
div_bulk = pp.ad.Divergence([g_bulk])
bound_bulk = pp.ad.BoundaryCondition(param_key, grids=[g_bulk])
dirbc_bulk = DirBC(bound_bulk, [g_bulk])
mortar_proj = pp.ad.MortarProjections(gb=gb)
mortar_proj_ghost = pp.ad.MortarProjections(gb=gb_ghost)
subdomain_proj_scalar = pp.ad.SubdomainProjections(gb=gb)
bulk_cell_rest = subdomain_proj_scalar.cell_restriction(g_bulk)
bulk_face_rest = subdomain_proj_scalar.face_restriction(g_bulk)
bulk_face_prol = subdomain_proj_scalar.face_prolongation(g_bulk)
frac_cell_rest = subdomain_proj_scalar.cell_restriction(g_frac)
frac_cell_prol = subdomain_proj_scalar.cell_prolongation(g_frac)

# Soil water retention curves
swrc = vanGenuchten(g_bulk, d_bulk, param_key)
theta_ad = pp.ad.Function(swrc.water_content, "water content")
krw_ad = pp.ad.Function(swrc.relative_permeability, "relative permeability")
C_ad = pp.ad.Function(swrc.moisture_capacity, "moisture capacity")

# MPFA discretization
mpfa_bulk = pp.ad.MpfaAd(param_key, [g_bulk])

# Obtain single phase flow to compute directionality of upwind scheme
h_bulk_m = psi_bulk_m + z_cc
flux1p_bulk = (
    mpfa_bulk.flux * h_bulk_m
    + mpfa_bulk.bound_flux * bound_bulk
    + mpfa_bulk.bound_flux
    * bulk_face_rest
    * mortar_proj.mortar_to_primary_int
    * lmbda_m
)

# Upwinding of relative permeabilities
upwind = UpwindFluxBasedAd(g_bulk, d_bulk, param_key)
krw_faces_ad = upwind(krw_ad(psi_bulk_m), krw_ad(dirbc_bulk), flux1p_bulk)

# Darcy fluxes
h_bulk = psi_bulk + z_cc
flux_bulk = (
    krw_faces_ad * mpfa_bulk.flux * h_bulk
    + krw_faces_ad * mpfa_bulk.bound_flux * bound_bulk
    + krw_faces_ad
    * mpfa_bulk.bound_flux
    * bulk_face_rest
    * mortar_proj.mortar_to_primary_int
    * lmbda
)

# Treatment of source and accumulation terms
# NOTE: The expression containing the active ad variable (psi_bulk) must be placed at the left of the (final) expression.
# Otherwise, numpy will broadcast the expression leading to an undesirable output format
source_bulk = pp.ad.ParameterArray(param_key, "source", grids=[g_bulk])
mass_bulk = pp.ad.MassMatrixAd(param_key, [g_bulk])

linearization = "modified_picard"
if linearization == "newton":
    accum_bulk_active = mass_bulk.mass * theta_ad(psi_bulk)
    accum_bulk_inactive = mass_bulk.mass * theta_ad(psi_bulk_n) * (-1)
elif linearization == "modified_picard":
    accum_bulk_active = mass_bulk.mass * psi_bulk * C_ad(psi_bulk_m)
    accum_bulk_inactive = mass_bulk.mass * (
        theta_ad(psi_bulk_m) - C_ad(psi_bulk_m) * psi_bulk_m - theta_ad(psi_bulk_n)
    )
elif linearization == "l_scheme":
    L = 0.0025
    accum_bulk_active = L * mass_bulk.mass * psi_bulk
    accum_bulk_inactive = mass_bulk.mass * (
        theta_ad(psi_bulk_m) - L * psi_bulk_m - theta_ad(psi_bulk_n)
    )
else:
    raise NotImplementedError(
        "Linearization scheme not implemented. Use 'newton', 'modified_picard', or 'l_scheme'."
    )

accumulation_bulk = accum_bulk_active + accum_bulk_inactive
conserv_bulk_eq = accumulation_bulk + dt * div_bulk * flux_bulk - dt * source_bulk

# Discretize and evaluate
conserv_bulk_eval = pp.ad.Expression(conserv_bulk_eq, dof_manager)
conserv_bulk_eval.discretize(gb)
conserv_bulk_num = conserv_bulk_eval.to_ad(gb)
# print(f'>> Conservation bulk: \n {conserv_bulk_num.jac.A} \n')

# %% Declare equations for the fracture


def fracture_pressure(water_volume):
    """
    This function computes the pressure head in the fracture given a volume of water.
    """

    aperture = 0.1  # [cm]
    inv_aperture = 1 / aperture
    datum = 0  # [cm]
    cc = g_frac_ghost.cell_centers[1]

    broadcaster = sps.csr_matrix(np.ones_like(cc)).reshape((-1, 1))
    airwater_interface = inv_aperture * water_volume + datum
    pressure_head = broadcaster * airwater_interface - cc

    # If the pressure is negative, it means that the cell is still dry. In that case, we set the pressure
    # in the fracture equal to the atmospheric pressure.
    if isinstance(pressure_head, pp.ad.Ad_array):
        pressure_head.val[pressure_head.val < 0] = 0
    else:
        pressure_head[pressure_head < 0] = 0

    return pressure_head


fracture_presure_adFun = pp.ad.Function(
    fracture_pressure, name="Fracture Pressure in ghost grid"
)
psi_fracture = fracture_presure_adFun(vol_frac)
psi_fracture_m = fracture_presure_adFun(vol_frac_m)

# Conservation equation in the fracture: This is a water volume balance
aperture = 0.1
sources_from_mortar = frac_cell_rest * mortar_proj.mortar_to_secondary_int * lmbda
conserv_frac_eq = vol_frac - sources_from_mortar * aperture * dt - vol_frac_n

conserv_frac_eval = pp.ad.Expression(conserv_frac_eq, dof_manager)
conserv_frac_eval.discretize(gb)
conserva_frac_num = conserv_frac_eval.to_ad(gb)

# %% Declare equations for the interface
mpfa_global = pp.ad.MpfaAd(param_key, [g_bulk, g_frac])
robin = pp.ad.RobinCouplingAd(param_key, edge_list)

# Projected bulk pressure traces onto the mortar space
mortar_trace_psi_bulk = (
    mortar_proj.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_cell
    * psi_bulk
    + mortar_proj.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_face
    * bulk_face_rest
    * mortar_proj.mortar_to_primary_int
    * lmbda
)

mortar_trace_psi_bulk_m = (
    mortar_proj.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_cell
    * psi_bulk_m
    + mortar_proj.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_face
    * bulk_face_rest
    * mortar_proj.mortar_to_primary_int
    * lmbda_m
)


def fracture_pressure_onto_mortar(psi_frac):

    # Get Secondary -> Mortar (for intensive properties)
    secondary_to_mortar = mortar_proj_ghost.secondary_to_mortar_avg.parse(gb=gb_ghost)

    # Get sparse matrix corresponding to the prolongation of fracture cells
    projections = pp.ad.SubdomainProjections(gb=gb_ghost)
    ghost_frac_cell_prol = projections.cell_prolongation(g_frac_ghost).parse(
        gb=gb_ghost
    )

    if isinstance(psi_frac, pp.ad.Ad_array):
        psi_frac_to_mortar = secondary_to_mortar * ghost_frac_cell_prol * psi_frac
    else:
        psi_frac_to_mortar = secondary_to_mortar * ghost_frac_cell_prol * psi_frac.val

    return psi_frac_to_mortar


fracture_pressure_onto_mortar_adFun = pp.ad.Function(
    fracture_pressure_onto_mortar, name="Projection of fracture pressure onto mortars"
)
mortar_psi_frac = fracture_pressure_onto_mortar_adFun(psi_fracture)
mortar_psi_frac_m = fracture_pressure_onto_mortar_adFun(psi_fracture_m)

# Upwinding of relative permeability on the interfaces
upwind_interface = InterfaceUpwindAd()
krw_interface_ad = upwind_interface(
    mortar_trace_psi_bulk_m,
    krw_ad(mortar_trace_psi_bulk_m),
    mortar_psi_frac_m,
    krw_ad(mortar_psi_frac_m),
)

is_conductive = pp.ad.ParameterArray(param_key, "is_conductive", edges=edge_list)

# Interface flux
mortar_flux = (
    robin.mortar_scaling * (mortar_trace_psi_bulk - mortar_psi_frac) * is_conductive
)
interface_flux_eq = mortar_flux + robin.mortar_discr * lmbda

interface_flux_eval = pp.ad.Expression(interface_flux_eq, dof_manager)
interface_flux_eval.discretize(gb)
interface_flux_num = interface_flux_eval.to_ad(gb)


def get_conductive_faces(psi_trace_primary, psi_secondary):

    is_mortar_conductive = np.zeros_like(psi_trace_primary, dtype=np.int8)
    for cell in range(1, psi_trace_primary.size):
        if psi_secondary[cell] > 0:
            is_mortar_conductive[cell] = 1
        else:
            if psi_trace_primary[cell] >= 0:
                is_mortar_conductive[cell] = 1

    return is_mortar_conductive


# %% Assemble the system of equations
eqs = [
    pp.ad.Expression(conserv_bulk_eq, dof_manager, name="conservation bulk"),
    pp.ad.Expression(conserv_frac_eq, dof_manager, name="conservation fracture"),
    pp.ad.Expression(interface_flux_eq, dof_manager, name="interface flux"),
]
equation_manager.equations += eqs
equation_manager.discretize(gb)

#%% Initialize exporter
exporter = pp.Exporter(gb, "new_mexico", "out")
exporter_ghost = pp.Exporter(gb_ghost, "double_dyke", "out1")
exporter.write_vtu([pressure_var], time_step=0)
exporter_ghost.write_vtu([pressure_var], time_step=0)

#%% Time loop
total_iteration_counter = 0

# Time loop
for n in range(1, num_time_steps + 1):
    time += dt
    recompute_solution = True
    is_mortar_conductive = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
    control_faces = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
    print("Current time: ", np.round(time, decimals=1))

    # Control Loop
    while recompute_solution:

        iteration_counter = 0
        residual_norm = 1
        rel_res = 1
        assign_data(param_key, is_mortar_conductive)

        # Solver loop
        while iteration_counter <= 100 and not (rel_res < 1e-5 or residual_norm < 1e-5):

            # Solve for pressure increment and update pressure
            A, b = equation_manager.assemble_matrix_rhs()
            solution = spla.spsolve(A, b)

            # # Distribute variable to local data dictionaries
            dof_manager.distribute_variable(solution, additive=True, to_iterate=True)

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
        proj_trace_psi_bulk, _ = eval_ad_expression(
            mortar_trace_psi_bulk, gb, dof_manager, print_expression=False
        )
        proj_psi_frac, _ = eval_ad_expression(
            mortar_psi_frac, gb, dof_manager, print_expression=False
        )
        is_mortar_conductive = get_conductive_faces(proj_trace_psi_bulk, proj_psi_frac)
        if control_faces.sum() == 0 and is_mortar_conductive.sum() > 0:
            control_faces = is_mortar_conductive  # make a copy of the sat faces to compare in the next iteration
            print(
                f"The faces {np.where(is_mortar_conductive)[0]} are saturated. Solution will be recomputed."
            )
            # Return pp.ITERATE to previous pp.STATE
            pp.set_iterate(data=d_bulk, iterate={bulk_var: d_bulk[pp.STATE][bulk_var]})
            pp.set_iterate(
                data=d_frac, iterate={fracture_var: d_frac[pp.STATE][fracture_var]}
            )
            pp.set_iterate(
                data=d_edge, iterate={mortar_var: d_edge[pp.STATE][mortar_var]}
            )
        else:
            recompute_solution = False
        # end of control loop

    # Update next time step solution. Note that additive should be False here
    print(f"Water volume: {d_frac[pp.STATE][fracture_var][0]}")
    print(
        f"Water pressure: {eval_ad_expression(psi_fracture, gb, dof_manager, print_expression=False)[0]}"
    )
    print(f"Left mortar fluxes: {d_edge[pp.STATE][pp.ITERATE][mortar_var][10:].sum()}")
    print(f"Right mortar fluxes: {d_edge[pp.STATE][pp.ITERATE][mortar_var][:10].sum()}")
    dof_manager.distribute_variable(solution, additive=False)
    # pp.plot_grid(g_bulk, cell_value=d_bulk[pp.STATE][bulk_var], plot_2d=True, color_map=[-25, 75])
    print()

    # Export to PARAVIEW
    pp.set_state(
        data=d_bulk_ghost, state={"pressure_head": d_bulk[pp.STATE]["pressure_head"]}
    )
    pp.set_state(
        data=d_frac_ghost,
        state={
            "pressure_head": eval_ad_expression(
                psi_fracture, gb, dof_manager, print_expression=False
            )[0]
        },
    )
    exporter.write_vtu([pressure_var], time_step=n)
    exporter_ghost.write_vtu(["pressure_head"], time_step=n)
