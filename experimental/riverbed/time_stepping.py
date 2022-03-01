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

# %% Import modules
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "DejaVu Sans",
        "font.serif": "Computer Modern Roman",
        "font.sans-serif": "Computer Modern Sans Serif",
        "font.cursive": "Zapf Chancery",
    }
)

from porepy.numerics.ad.grid_operators import DirBC
from mdunsat.ad_utils import (
    get_conductive_mortar_cells,
    is_water_volume_negative,
    set_iterate_to_state,
    GhostProjection,
    HydrostaticFracturePressure,
    UpwindFluxBasedAd,
    vanGenuchten,
    TimeStepping,
    ParameterUpdate,
    ParameterScalar,
)

# %% Make grid
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


# Physical bucket... This gb will be the one reflects the actual number of dof that we
# will use for solving the problem. Initially, it is the same as the ghost bucket, but
# later it will be modified. In particular, fracture grids will be replaced with grids with
# single cells.
# NOTE: Making a copy of the grid bucket, as in
# gb = gb_ghost.copy()
# is not an option, since this is not a hard but rather a shallow copy of the object.
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

# %% Physical parameters
K_sat = 0.00922  # [cm/s] Saturated hydraulic conductivity
theta_r = 0.102  # [-] residual water content
theta_s = 0.368  # [-] saturated water content, a.k.a porosity
alpha_vG = 0.0335  # [1/cm] van Genuchten parameter
n_vG = 2  # van Genuchten parameter
m_vG = 1 - 1 / n_vG  # van Genuchten parameter

# %% Time parameters
time_stepping = TimeStepping(final_time=20 * pp.HOUR, dt_min_max=(5, 5000))
time = 0
dt = time_stepping.dt_min
times = [time]
dts = [dt]
export_counter = 0

# %% Assign parameters
param_key = "flow"
pressure_var = "pressure_head"
bulk_var = "pressure_head"
fracture_var = "water_volume"
mortar_var = "mortar_flux"
param_update = ParameterUpdate(gb, param_key)

# Declare primary variables
d_bulk[pp.PRIMARY_VARIABLES] = {bulk_var: {"cells": 1}}
d_frac[pp.PRIMARY_VARIABLES] = {fracture_var: {"cells": 1}}
for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {mortar_var: {"cells": 1}}

# Assign parameters for the bulk
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
perm = pp.SecondOrderTensor(K_sat * np.ones(g_bulk.num_cells))
bc_faces = g_bulk.get_boundary_faces()
bc_type = np.array(bc_faces.size * ["neu"])
bc_type[np.in1d(bc_faces, bc_infilt)] = "dir"
bc = pp.BoundaryCondition(g_bulk, faces=bc_faces, cond=bc_type)
bc_values = np.zeros(g_bulk.num_faces)
bc_values[bc_infilt] = 0  # pressure head on the left-top bc faces
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
    "time_step": time_stepping.dt_min,
}
pp.initialize_data(g_bulk, d_bulk, param_key, specified_parameters)

# Parameters for the fracture
specified_parameters = {"aperture": 0.1}
pp.initialize_data(g_frac, d_frac, param_key, specified_parameters=specified_parameters)

# Parameters for the mortar
for e, d in gb.edges():
    mg = d["mortar_grid"]
    g_sec, _ = gb.nodes_of_edge(e)
    d_sec = gb.node_props(g_sec)
    aperture = d_sec[pp.PARAMETERS][param_key]["aperture"]
    k_perp_val = K_sat / (2 * aperture)
    normal_diffusivity = k_perp_val * np.ones(mg.num_cells)
    is_conductive = np.zeros(mg.num_cells, dtype=np.int8)
    data = {"normal_diffusivity": normal_diffusivity, "is_conductive": is_conductive}
    pp.initialize_data(mg, d, param_key, data)

# %% Set initial states
for g, d in gb:
    pp.set_state(d)
    pp.set_iterate(d)
    if g.dim == dim_max:
        d[pp.STATE][bulk_var] = np.zeros(g.num_cells)
        d[pp.STATE][bulk_var][cells_left] = -1000
        d[pp.STATE][bulk_var][cells_right] = -1000
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

pp.set_state(d_bulk_ghost, state={"pressure_head": d_bulk[pp.STATE][bulk_var]})
pp.set_state(d_frac_ghost, state={"pressure_head": np.zeros(g_frac_ghost.num_cells)})

# %% AD variables and manager
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

# %% AD operators and discrete expressions

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
h_bulk = psi_bulk + pp.ad.Array(z_cc)
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
# NOTE: The expression containing the active ad variable (psi_bulk)
# must be placed at the left of the (final) expression.
# Otherwise, numpy will broadcast the expression leading to an undesirable output format
dt_ad = ParameterScalar(param_key, "time_step", grid=g_bulk)
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
conserv_bulk_eq = accumulation_bulk + dt_ad * div_bulk * flux_bulk - dt_ad * source_bulk

# Discretize and evaluate
conserv_bulk_eval = pp.ad.Expression(conserv_bulk_eq, dof_manager)
conserv_bulk_eval.discretize(gb)
conserv_bulk_num = conserv_bulk_eval.to_ad(gb)
print()
# print(f'>> Conservation bulk: \n {conserv_bulk_num.jac.A} \n')

# %% Declare equations for the fracture
hydrostatic = HydrostaticFracturePressure(
    ghost_grid=g_frac_ghost, data=d_frac, param_key=param_key
)
psi_fracture = pp.ad.Function(
    hydrostatic.get_pressure_head, name="hydrostatic pressure head"
)

# Conservation equation in the fracture: This is a water volume balance
aperture_ad = ParameterScalar(param_key, "aperture", grid=g_frac)
sources_from_mortar = frac_cell_rest * mortar_proj.mortar_to_secondary_int * lmbda
conserv_frac_eq = vol_frac - dt_ad * aperture_ad * sources_from_mortar - vol_frac_n

# Evaluate and discretize
conserv_frac_eval = pp.ad.Expression(conserv_frac_eq, dof_manager)
conserv_frac_eval.discretize(gb)
conserva_frac_num = conserv_frac_eval.to_ad(gb)

# %% Declare equations for the interface
mpfa_global = pp.ad.MpfaAd(param_key, [g_bulk, g_frac])
robin = pp.ad.RobinCouplingAd(param_key, edge_list)

# Projected bulk pressure traces onto the mortar grid
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

# Projected fracture pressure (from the ghost grid) onto the mortar grid
ghost_proj = GhostProjection(gb_ghost=gb_ghost, g_fracture=g_frac_ghost)
frac_to_mortar = pp.ad.Function(
    ghost_proj.secondary_to_mortar, name="Secondary to mortar projection"
)
mortar_psi_frac = frac_to_mortar(psi_fracture(vol_frac))

# Array parameter that keeps track of conductive (1) and blocking (0) mortar cells
# Note that if it is blocking, the whole discrete equation is removed for that mortar cell
is_conductive = pp.ad.ParameterArray(param_key, "is_conductive", edges=edge_list)

# Interface flux
mortar_flux = (
    robin.mortar_scaling * (mortar_trace_psi_bulk - mortar_psi_frac) * is_conductive
)
interface_flux_eq = mortar_flux + robin.mortar_discr * lmbda

interface_flux_eval = pp.ad.Expression(interface_flux_eq, dof_manager)
interface_flux_eval.discretize(gb)
interface_flux_num = interface_flux_eval.to_ad(gb)

# %% Assemble the system of equations
eqs = [
    pp.ad.Expression(conserv_bulk_eq, dof_manager, name="conservation bulk"),
    pp.ad.Expression(conserv_frac_eq, dof_manager, name="conservation fracture"),
    pp.ad.Expression(interface_flux_eq, dof_manager, name="interface flux"),
]
equation_manager.equations += eqs
equation_manager.discretize(gb)

# %% Initialize exporter
water_volume = [0]
exporter_ghost = pp.Exporter(gb_ghost, "double_dyke", "out1")
exporter_ghost.write_vtu([pressure_var], time_step=0)

# %% Time loop
total_iteration_counter = 0
iters = []
# Time loop
while time < time_stepping.final_time:
    time += dt
    times.append(time)
    dts.append(dt)
    recompute_solution = True
    is_mortar_conductive = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
    control_faces = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
    print("Current time: ", np.round(time, decimals=1))

    # Control Loop
    while recompute_solution:

        iteration_counter = 0
        residual_norm = 1
        rel_res = 1
        param_update.update_mortar_conductivity_state(is_mortar_conductive, edge_list)

        # Solver loop
        while iteration_counter <= 30 and not (rel_res < 1e-5 or residual_norm < 1e-5):

            # Solve system of equations and distribute variables to pp.ITERATE
            A, b = equation_manager.assemble_matrix_rhs()
            solution = spla.spsolve(A, b)
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
                rel_res,
            )

            # Prepare next iteration
            iteration_counter += 1
            total_iteration_counter += 1
            # end of iteration loop

        # Recompute solution if negative water volume is encountered
        if is_water_volume_negative(gb, fracture_var, [g_frac]):
            dt /= 2
            param_update.update_time_step(dt)
            print(
                f"Encountered negative volume. Reducing time step and recomputing solution."
            )
            set_iterate_to_state(
                gb, bulk_var, fracture_var, mortar_var
            )  # set iterate to state
            continue  # go back to control loop

        # Recompute solution is capillary barrier is overcome
        is_mortar_conductive = get_conductive_mortar_cells(
            gb, dof_manager, mortar_trace_psi_bulk, mortar_psi_frac
        )
        if control_faces.sum() == 0 and is_mortar_conductive.sum() > 0:
            control_faces = is_mortar_conductive  # make a copy of the sat faces to compare in the next iteration
            print(
                f"The faces {np.where(is_mortar_conductive)[0]} are saturated. Solution will be recomputed."
            )
            set_iterate_to_state(
                gb, bulk_var, fracture_var, mortar_var
            )  # return to previous state
        else:
            recompute_solution = False
        # end of control loop

    # Save number of iterations
    iters.append(iteration_counter - 1)

    # Compute next time step based on the current amount of iterations
    dt = time_stepping.next_time_step(time, dt_old=dt, iters=iteration_counter - 1)
    param_update.update_time_step(dt)

    # Update next time step solution. Note that additive should be False here
    print(f"Water volume: {d_frac[pp.STATE][fracture_var][0]}")
    water_volume.append(d_frac[pp.STATE][fracture_var][0])
    # print(f"Water pressure: {hydrostatic.get_pressure_head(d_frac[pp.STATE][fracture_var])}")
    # print(f"Left mortar fluxes: {d_edge[pp.STATE][pp.ITERATE][mortar_var][10:].sum()}")
    # print(f"Right mortar fluxes: {d_edge[pp.STATE][pp.ITERATE][mortar_var][:10].sum()}")
    dof_manager.distribute_variable(solution, additive=False)
    print()

    # Export to PARAVIEW
    pp.set_state(data=d_bulk_ghost, state={"pressure_head": d_bulk[pp.STATE][bulk_var]})
    pp.set_state(
        data=d_frac_ghost,
        state={
            "pressure_head": hydrostatic.get_pressure_head(
                d_frac[pp.STATE][fracture_var]
            )
        },
    )
    if (np.mod(len(times), 10) == 0) or (time == time_stepping.final_time):
        export_counter += 1
        exporter_ghost.write_vtu(["pressure_head"], time_step=export_counter)

# %% Plotting
plot_volume = True
plot_dt = True
plot_iters = True

if plot_volume:
    _, ax = plt.subplots(1, 1)
    ax.plot(np.array(times) / 3600, np.array(water_volume), linewidth=2)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Volume [cc]")
    plt.show()

if plot_dt:
    _, ax = plt.subplots(1, 1)
    ax.plot(np.array(times) / 3600, np.array(dts), linewidth=2)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Time step [s]")
    plt.show()

if plot_iters:
    _, ax = plt.subplots(1, 1)
    ax.plot(np.array(times[1:]) / 3600, np.array(iters), linewidth=2)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Number of iterations")
    plt.show()
