# %% Import modules
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib.pyplot import spy as sparsity

from mdunsat.ad_utils.ad_utils import (
    SoilWaterRetentionCurves,
    get_conductive_mortars,
    is_water_volume_negative,
    set_iterate_as_state,
    GhostProjection,
    GhostFractureHydraulicHead,
    FractureVolume,
    InterfaceUpwindAd,
    UpwindFluxBasedAd,
    ParameterUpdate,
    ParameterScalar,
)

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "DejaVu Sans",
        "font.serif": "Computer Modern Roman",
        "font.sans-serif": "Computer Modern Sans Serif",
        "font.cursive": "Zapf Chancery",
    }
)

#%% Global simulation parameters
blocking_block: bool = True
linearization: str = "newton"

#%% Make grid
x_min: float = 0.0
x_max: float = 100.0
y_min: float = 0.0
y_max: float = 100.0
domain: dict = {"xmin": x_min, "xmax": x_max, "ymin": y_min, "ymax": y_max}
# Create fracture newtork from csv
if blocking_block:
    network_2d = pp.fracture_importer.network_2d_from_csv(
        "net_with_block.csv", domain=domain
    )
else:
    network_2d = pp.fracture_importer.network_2d_from_csv(
        "vertfrac_embedded_80cm.csv", domain=domain
    )
# Mesh arguments
target_h_bound: float = 5.0
target_h_frac: float = 5.0
mesh_args: dict = {"mesh_size_frac": target_h_frac, "mesh_size_bound": target_h_bound}

# Create the physical fracture as a one-dimensional grid with one cell and length physdims
# NOTE: By default, this will create a horizontal fracture
g1d_mono: pp.Grid = pp.CartGrid(np.array([1]), physdims=80)
# Perturb the grid nodes to get to set the actual position of the fracture
g1d_mono.nodes = np.array([[50, 50], [10, 90], [0, 0]])
g1d_mono.compute_geometry()

# Ghost bucket... This gb should not be touched. We will not really use it, but rather
# retrieve mortar projections that we will use later
if blocking_block:
    gb_ghost: pp.GridBucket = network_2d.mesh(mesh_args, constraints=[1, 2, 3, 4])
else:
    gb_ghost: pp.GridBucket = network_2d.mesh(mesh_args)

# Extract useful information from the ghost GridBucket
g_bulk_ghost: pp.Grid = gb_ghost.grids_of_dimension(2)[0]
d_bulk_ghost: dict = gb_ghost.node_props(g_bulk_ghost)
g_frac_ghost: pp.Grid = gb_ghost.grids_of_dimension(1)[0]
d_frac_ghost: dict = gb_ghost.node_props(g_frac_ghost)
d_edge_ghost: dict = gb_ghost.edge_props((g_bulk_ghost, g_frac_ghost))
mg_ghost: pp.MortarGrid = d_edge_ghost["mortar_grid"]
ghost_grid_list: list = [g for g, _ in gb_ghost]
ghost_edge_list: list = [e for e, _ in gb_ghost.edges()]

# Physical bucket... This gb will be the one reflects the actual number of dof that we
# will use for solving the problem. Initially, it is the same as the ghost bucket, but
# later it will be modified. In particular, fracture grids will be replaced with grids with
# single cells.
# WARNING: Making a copy of the grid bucket, as in gb = gb_ghost.copy() does not work,
# since this is not a hard but rather a shallow copy of the object.
if blocking_block:
    gb: pp.GridBucket = network_2d.mesh(mesh_args, constraints=[1, 2, 3, 4])
else:
    gb: pp.GridBucket = network_2d.mesh(mesh_args)

# Retrieve fracture grids
g_frac_old: pp.Grid = gb.grids_of_dimension(1)[0]

# Create map dictionary to feed the replace_grids method. Key is old grid, value is new grid.
g_map: dict = {g_frac_old: g1d_mono}
gb.replace_grids(g_map=g_map)

dim_max: int = gb.dim_max()
g_bulk: pp.Grid = gb.grids_of_dimension(dim_max)[0]
g_frac: pp.Grid = gb.grids_of_dimension(dim_max - 1)[0]

d_bulk: dict = gb.node_props(g_bulk)
d_frac: dict = gb.node_props(g_frac)
d_edge: dict = gb.edge_props((g_bulk, g_frac))

# %% Time parameters
schedule: list = list(np.linspace(0, 4 * pp.HOUR, 60, dtype=np.int32))
tsc = pp.TimeSteppingControl(
    schedule=schedule,
    dt_init=1.0,
    dt_min_max=(0.01, 0.25 * pp.HOUR),
    iter_max=50,
    iter_optimal_range=(25 - 3, 25 + 3),
    iter_lowupp_factor=(1.1, 0.9),
    recomp_factor=0.5,
    recomp_max=12,
    print_info=True,
)
times: list = [tsc.time]
dts: list = [tsc.dt]
export_counter: int = 0

# %% Assign parameters
# Keywords
param_key: str = "flow"
node_var: str = "hydraulic_head"
edge_var: str = "mortar_flux"

# Declare primary variables
for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {node_var: {"cells": 1}}
for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {edge_var: {"cells": 1}}

# Parameter assignment
param_update = ParameterUpdate(gb, param_key)  # object to update parameters

# PARAMETERS FOR THE BULK
# Face indices of the boundaries
fc: np.ndarray = g_bulk.face_centers
top: np.ndarray = np.where(np.abs(fc[1] - y_max) < 1e-5)[0]
bottom: np.ndarray = np.where(np.abs(fc[1]) < 1e-5)[0]
left: np.ndarray = np.where(np.abs(fc[0]) < 1e-5)[0]
right: np.ndarray = np.where(np.abs(fc[0] - x_max) < 1e-5)[0]
top_left: np.ndarray = top[np.where(np.abs(fc[0][top] < (x_max / 2)))[0]]

# Left and right cell indices
cc: np.ndarray = g_bulk.cell_centers
cells_left: np.ndarray = np.where(np.abs(cc[0] < (x_max / 2)))[0]
cells_right: np.ndarray = np.where(np.abs(cc[0] > (x_max / 2)))[0]

# Cell indices corresponding to the blocking fractures
blocking_idx: list = [
    cc[0] < (0.50 * x_max),
    cc[0] > (0.25 * x_max),
    cc[1] < (0.80 * y_max),
    cc[1] > (0.60 * y_max),
]
mult_cond: bool = np.logical_and(blocking_idx[0], blocking_idx[1])
mult_cond: bool = np.logical_and(blocking_idx[2], mult_cond)
mult_cond: bool = np.logical_and(blocking_idx[3], mult_cond)
blocking_cells: np.ndarray = np.where(mult_cond)[0]

# Boundary conditions
bc_faces: np.ndarray = g_bulk.get_boundary_faces()
bc_type: np.ndarray = np.array(bc_faces.size * ["neu"])
bc_type[np.in1d(bc_faces, top_left)] = "dir"
bc: pp.BoundaryCondition = pp.BoundaryCondition(
    g_bulk, faces=bc_faces, cond=list(bc_type)
)
bc_values: np.ndarray = np.zeros(g_bulk.num_faces)
bc_values[top_left] = 0 + y_max  # 0 (pressure_head) + y_max (elevation_head)

# Hydraulic conductivity
K_SAT: np.ndarray = 0.00922 * np.ones(g_bulk.num_cells)  # conductive bulk cells
if blocking_block:
    K_SAT[mult_cond] = 1e-18  # hydraulic conductivity of blocking cells

# Initialize bulk data
specified_parameters: dict = {
    "second_order_tensor": pp.SecondOrderTensor(K_SAT),  # [cm/s]
    "bc": bc,
    "bc_values": bc_values,
    "source": np.zeros(g_bulk.num_cells),
    "elevation": g_bulk.cell_centers[dim_max - 1],
    "mass_weight": np.ones(g_bulk.num_cells),
    "theta_r": 0.102,  # residual water content [-]
    "theta_s": 0.368,  # saturated water content [-]
    "alpha_vG": 0.0335,  # van Genuchten parameter [1/cm]
    "n_vG": 2.0,  # van Genuchten parameter [-]
    "m_vG": 0.5,  # (1 - 1 / n_vG) van Genuchten parameter [-]
    "time_step": tsc.dt,  # [s]
}
pp.initialize_data(g_bulk, d_bulk, param_key, specified_parameters)

# Parameters for the fracture grids
for g, d in gb:
    if g.dim < dim_max:
        specified_parameters = {
            "aperture": 0.1,
            "datum": np.min(g_frac.face_centers[dim_max - 1]),
            "elevation": g.cell_centers[dim_max - 1],
        }
    pp.initialize_data(g, d, param_key, specified_parameters)

# Parameters for the mortar
for e, d in gb.edges():
    mg = d["mortar_grid"]
    g_sec, _ = gb.nodes_of_edge(e)
    d_sec = gb.node_props(g_sec)
    aperture = d_sec[pp.PARAMETERS][param_key]["aperture"]
    k_perp_val = 0.00922 / (2 * aperture)
    normal_diffusivity = k_perp_val * np.ones(mg.num_cells)
    is_conductive = np.zeros(mg.num_cells, dtype=np.int8)
    data = {
        "normal_diffusivity": normal_diffusivity,
        "is_conductive": is_conductive,
        "elevation": mg.cell_centers[dim_max - 1],
    }
    pp.initialize_data(mg, d, param_key, data)

# %% Set initial states
z_bulk: np.ndarray = g_bulk.cell_centers[dim_max - 1]
z_frac: np.ndarray = g_frac.cell_centers[dim_max - 1]
z_frac_ghost: np.ndarray = g_frac_ghost.cell_centers[dim_max - 1]

for g, d in gb:
    if g.dim == gb.dim_max():
        pp.set_state(
            d, state={node_var: -1000 + d[pp.PARAMETERS][param_key]["elevation"]}
        )
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})
    else:
        pp.set_state(
            d, state={node_var: np.array([0 + d[pp.PARAMETERS][param_key]["datum"]])}
        )
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})

for _, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d, state={edge_var: np.zeros(mg.num_cells)})
    pp.set_iterate(d, iterate={edge_var: d[pp.STATE][edge_var]})

# %% AD variables and manager
grid_list: list = [g for g, _ in gb]
edge_list: list = [e for e, _ in gb.edges()]

# Construct degree of freedom manager and equations manager
dof_manager: pp.DofManager = pp.DofManager(gb)
equation_manager: pp.ad.EquationManager = pp.ad.EquationManager(gb, dof_manager)

# Assign primary variables to their corresponding grids
hb: pp.ad.MergedVariable = equation_manager.merge_variables([(g_bulk, node_var)])
hf: pp.ad.MergedVariable = equation_manager.merge_variables(
    [(g, node_var) for g in [g_frac]]
)
lmbda: pp.ad.MergedVariable = equation_manager.merge_variables(
    [(e, edge_var) for e in edge_list]
)

# Shorthands. Note that the following merged variables all have different id's
hb_m: pp.ad.MergedVariable = hb.previous_iteration()
hb_n: pp.ad.MergedVariable = hb.previous_timestep()
hf_m: pp.ad.MergedVariable = hf.previous_iteration()
hf_n: pp.ad.MergedVariable = hf.previous_timestep()
lmbda_m: pp.ad.MergedVariable = lmbda.previous_iteration()
lmbda_n: pp.ad.MergedVariable = lmbda.previous_timestep()

# Useful secondary variables
psib: pp.ad.Operator = hb - pp.ad.Array(z_bulk)  # pressure head
psib_m: pp.ad.Operator = hb_m - pp.ad.Array(z_bulk)  # pressure head at prev iter
psib_n: pp.ad.Operator = hb_n - pp.ad.Array(z_bulk)  # pressure head at prev time

# %% AD operators and discrete expressions

# Grid operators
div_bulk = pp.ad.Divergence(grids=[g_bulk])
bound_bulk = pp.ad.BoundaryCondition(param_key, grids=[g_bulk])
projections = pp.ad.MortarProjections(gb=gb, grids=grid_list, edges=edge_list)
ghost_projections = pp.ad.MortarProjections(
    gb=gb_ghost, grids=ghost_grid_list, edges=ghost_edge_list
)
subdomain_proj_scalar = pp.ad.SubdomainProjections(grids=grid_list)
bulk_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(g_bulk)
bulk_face_rest: pp.ad.Matrix = subdomain_proj_scalar.face_restriction(g_bulk)
bulk_face_prol: pp.ad.Matrix = subdomain_proj_scalar.face_prolongation(g_bulk)
frac_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(g_frac)
frac_cell_prol: pp.ad.Matrix = subdomain_proj_scalar.cell_prolongation(g_frac)

# Soil water retention curves
swrc = SoilWaterRetentionCurves(gb=gb, param_key=param_key)
theta_ad: pp.ad.Function = pp.ad.Function(swrc.water_content, "water content")
krw_ad: pp.ad.Function = pp.ad.Function(swrc.relative_permeability, "rel perm")
smc_ad: pp.ad.Function = pp.ad.Function(swrc.moisture_capacity, "moisture capacity")

# MPFA discretization
mpfa_bulk: pp.ad.MpfaAd = pp.ad.MpfaAd(param_key, [g_bulk])

# Obtain single phase flow to compute directionality of upwind scheme
flux1p_bulk: pp.ad.Operator = (
    mpfa_bulk.flux * hb_m
    + mpfa_bulk.bound_flux * bound_bulk
    + mpfa_bulk.bound_flux
    * bulk_face_rest
    * projections.mortar_to_primary_int
    * lmbda_m
)

# Upwinding of relative permeabilities
upwind = UpwindFluxBasedAd(g_bulk, d_bulk, param_key)
# TODO: Note that we have to pass the pressure head, not the hydraulic head to the relative
#  permeability function. Also, note that this requires to pass the dirichlet values of the
#  pressure head, not the hydraulic head. So, this has to be taken care of...
zface_bulk: np.ndarray = g_bulk.face_centers[dim_max - 1]
psib_dir: np.ndarray = np.nan * np.ones(g_bulk.num_faces)
psib_dir[bc.is_dir] = bc_values[bc.is_dir] - zface_bulk[bc.is_dir]
krw_faces_ad: pp.ad.Operator = upwind(
    krw_ad(psib_m), krw_ad(pp.ad.Array(psib_dir)), flux1p_bulk
)

# Multiphase Darcy fluxes
flux_bulk: pp.ad.Operator = (
    krw_faces_ad * mpfa_bulk.flux * hb
    + krw_faces_ad * mpfa_bulk.bound_flux * bound_bulk
    + krw_faces_ad
    * mpfa_bulk.bound_flux
    * bulk_face_rest
    * projections.mortar_to_primary_int
    * lmbda
)

# Treatment of source and accumulation terms
# NOTE: The expression containing the active ad variable (psi_bulk)
# must be placed at the left of the (final) expression.
# Otherwise, numpy will broadcast the expression leading to an undesirable output format
dt_ad = ParameterScalar(param_key, "time_step", grid=g_bulk)
source_bulk = pp.ad.ParameterArray(param_key, "source", grids=[g_bulk])
mass_bulk = pp.ad.MassMatrixAd(param_key, [g_bulk])

if linearization == "newton":
    accum_bulk_active = mass_bulk.mass * theta_ad(psib)
    accum_bulk_inactive = mass_bulk.mass * theta_ad(psib_n) * (-1)
elif linearization == "modified_picard":
    accum_bulk_active = mass_bulk.mass * psib * smc_ad(psib_m)
    accum_bulk_inactive = mass_bulk.mass * (
        theta_ad(psib_m) - smc_ad(psib_m) * psib_m - theta_ad(psib_n)
    )
elif linearization == "l_scheme":
    L = 0.0025
    accum_bulk_active = L * mass_bulk.mass * psib
    accum_bulk_inactive = mass_bulk.mass * (
        theta_ad(psib_m) - L * psib_m - theta_ad(psib_n)
    )
else:
    raise NotImplementedError(
        "Linearization scheme not implemented. Use 'newton', "
        "'modified_picard', or 'l_scheme'."
    )

accumulation_bulk = accum_bulk_active + accum_bulk_inactive
conserv_bulk_eq = accumulation_bulk + dt_ad * div_bulk * flux_bulk - dt_ad * source_bulk

# Discretize and evaluate
conserv_bulk_eq.discretize(gb=gb)
conserv_bulk_num = conserv_bulk_eq.evaluate(dof_manager=dof_manager)

# %% Declare equations for the fracture
# kw=kw)
fracvol = FractureVolume(g_frac_ghost, d_frac, param_key)
vol_ad = pp.ad.Function(fracvol.fracture_volume, name="Volume")
vol_cap_ad = pp.ad.Function(fracvol.volume_capacity, name="Volume capacity")
gfh = GhostFractureHydraulicHead(gb=gb, ghost_grid=g_frac_ghost)
ghost_hf_ad = pp.ad.Function(
    gfh.get_ghost_hyd_head, name="Ghost fracture hydraulic head func"
)
ghost_hf = ghost_hf_ad(hf)

linearization = "newton"
if linearization == "newton":
    accum_frac_active = vol_ad(hf)
    accum_frac_inactive = vol_ad(hf_n) * (-1)
elif linearization == "modified_picard":
    accum_frac_active = hf * vol_cap_ad(hf_m)
    accum_frac_inactive = vol_ad(hf_m) - vol_cap_ad(hf_m) * hf_m - vol_ad(hf_n)
elif linearization == "l_scheme":
    L = 0.015
    accum_frac_active = L * hf
    accum_frac_inactive = vol_ad(hf_m) - L * hf_m - vol_ad(hf_n)
else:
    raise NotImplementedError(
        "Linearization scheme not implemented. Use 'newton', "
        "'modified_picard', or 'l_scheme'."
    )

# Conservation equation in the fracture: This is a water volume balance
aperture_ad = ParameterScalar(param_key, "aperture", grid=g_frac)
# vol_frac = vol_ad(h_frac)
# vol_frac_n = vol_ad(hf_n)
sources_from_mortar = frac_cell_rest * projections.mortar_to_secondary_int * lmbda
# conserv_frac_eq = vol_frac - dt_ad * aperture_ad * sources_from_mortar - vol_frac_n
accum_frac = accum_frac_active + accum_frac_inactive
conserv_frac_eq = accum_frac - dt_ad * aperture_ad * sources_from_mortar

# Evaluate and discretize
conserv_frac_eq.discretize(gb=gb)
conserva_frac_num = conserv_frac_eq.evaluate(dof_manager=dof_manager)

# %% Declare equations for the interface
mpfa_global = pp.ad.MpfaAd(param_key, [g_bulk, g_frac])
robin = pp.ad.RobinCouplingAd(param_key, edge_list)

# Projected bulk pressure traces onto the mortar grid
mortar_hb = (
    projections.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_cell
    * hb
    + projections.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_face
    * bulk_face_rest
    * projections.mortar_to_primary_int
    * lmbda
)

mortar_hb_m = (
    projections.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_cell
    * hb_m
    + projections.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_face
    * bulk_face_rest
    * projections.mortar_to_primary_int
    * lmbda_m
)

# Projected fracture pressure (from the ghost grid) onto the mortar grid
ghost_proj = GhostProjection(gb_ghost=gb_ghost, g_fracture=g_frac_ghost)
frac_to_mortar = pp.ad.Function(ghost_proj.secondary_to_mortar, name="sec_to_mortar")
mortar_hf = frac_to_mortar(ghost_hf)

# Array parameter that keeps track of conductive (1) and blocking (0) mortar cells
# Note that if it is blocking, the whole discrete equation is removed for that mortar cell
is_conductive = pp.ad.ParameterArray(param_key, "is_conductive", edges=edge_list)

# Interface flux
mortar_flux = robin.mortar_scaling * (mortar_hb - mortar_hf) * is_conductive
interface_flux_eq = mortar_flux + robin.mortar_discr * lmbda

# interface_flux_eval = pp.ad.Expression(interface_flux_eq, dof_manager)
interface_flux_eq.discretize(gb=gb)
interface_flux_num = interface_flux_eq.evaluate(dof_manager=dof_manager)

# %% Assemble discrete equations, feed into the equation manager, and discretize.
eqs = [conserv_bulk_eq, conserv_frac_eq, interface_flux_eq]
equation_manager.equations += eqs
equation_manager.discretize(gb)

# %% Initialize exporter. Note that we use the ghost grid rather than the physical one
pp.set_state(
    d_bulk_ghost,
    state={
        node_var: d_bulk[pp.STATE][node_var],
        "pressure_head": d_bulk[pp.STATE][node_var] - z_bulk,
    },
)
pp.set_state(
    d_frac_ghost,
    state={
        node_var: d_frac[pp.PARAMETERS][param_key]["datum"]
        * np.ones(g_frac_ghost.num_cells),
        "pressure_head": d_frac[pp.STATE][node_var] - z_frac_ghost,
    },
)
water_table = [d_frac[pp.STATE][node_var][0]]
water_vol = [fracvol.fracture_volume(d_frac[pp.STATE][node_var])[0]]
exporter_ghost = pp.Exporter(gb_ghost, "out", "twodim_simplex_vert")
exporter_ghost.write_vtu([node_var, "pressure_head"], time_step=0)

# %% Time loop
total_iteration_counter: int = 0
iters: list = []
abs_tol: float = 1e-5
is_mortar_conductive: np.ndarray = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
control_faces: np.ndarray = is_mortar_conductive

# Time loop
while tsc.time < tsc.time_final:
    tsc.time += tsc.dt
    iteration_counter: int = 0
    residual_norm: float = 1.0
    rel_res: float = 1.0

    # Solver loop
    while iteration_counter <= tsc.iter_max and not residual_norm < abs_tol:

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
            "time",
            tsc.time,
            "iter",
            iteration_counter,
            "res",
            residual_norm,
            "rel_res",
            rel_res,
            "dt",
            tsc.dt,
        )

        # Prepare next iteration
        iteration_counter += 1
        total_iteration_counter += 1
        # end of iteration loop

    # Recompute solution if we did not achieve convergence
    if residual_norm > abs_tol or np.isnan(residual_norm):
        tsc.next_time_step(recompute_solution=True, iterations=iteration_counter - 1)
        param_update.update_time_step(tsc.dt)
        set_iterate_as_state(gb, node_var, edge_var)
        continue

    # Recompute solution if negative volume is encountered
    if is_water_volume_negative(gb, node_var, [g_frac]):
        tsc.next_time_step(recompute_solution=True, iterations=iteration_counter - 1)
        param_update.update_time_step(tsc.dt)
        print(f"Encountered negative volume. Reducing dt and recomputing solution.")
        set_iterate_as_state(gb, node_var, edge_var)
        continue

    # Recompute solution is capillary barrier is overcome. Note that dt remains the same
    is_mortar_conductive = get_conductive_mortars(
        gb, dof_manager, param_key, mortar_hb, mortar_hf, edge_list
    )
    if control_faces.sum() == 0 and is_mortar_conductive.sum() > 0:
        param_update.update_mortar_conductivity_state(is_mortar_conductive, edge_list)
        print(
            f"The faces {np.where(is_mortar_conductive)[0]} are saturated. "
            f"Solution will be recomputed."
        )
        control_faces = is_mortar_conductive
        set_iterate_as_state(gb, node_var, edge_var)
        tsc.time -= tsc.dt  # correct time since we are going to recompute the solution
        continue
    else:
        is_mortar_conductive: np.ndarray = np.zeros(
            gb.num_mortar_cells(), dtype=np.int8
        )
        param_update.update_mortar_conductivity_state(is_mortar_conductive, edge_list)
        control_faces = is_mortar_conductive

    # Save number of iterations and time step
    iters.append(iteration_counter - 1)
    times.append(tsc.time)
    dts.append(tsc.dt)

    # Succesful convergence
    tsc.next_time_step(recompute_solution=False, iterations=iteration_counter - 1)
    param_update.update_time_step(tsc.dt)
    print(f"Fracture water table height: {d_frac[pp.STATE][node_var][0]}")
    print(
        f"Fracture water volume: {fracvol.fracture_volume(d_frac[pp.STATE][node_var])[0]}"
    )
    water_table.append(d_frac[pp.STATE][node_var][0])
    water_vol.append(fracvol.fracture_volume(d_frac[pp.STATE][node_var]))
    dof_manager.distribute_variable(solution, additive=False)
    print()

    # Export to ParaView
    pp.set_state(
        data=d_bulk_ghost,
        state={
            node_var: d_bulk[pp.STATE][node_var],
            "pressure_head": d_bulk[pp.STATE][node_var] - z_bulk,
        },
    )
    pp.set_state(
        data=d_frac_ghost,
        state={
            node_var: gfh.get_ghost_hyd_head(d_frac[pp.STATE][node_var]),
            "pressure_head": gfh.get_ghost_hyd_head(d_frac[pp.STATE][node_var])
            - z_frac_ghost,
        },
    )
    if tsc.time in tsc.schedule:
        export_counter += 1
        exporter_ghost.write_vtu([node_var, "pressure_head"], time_step=export_counter)

# %% Plotting
plot_hydraulic_head = True
plot_volume = True
plot_dt = True
plot_iters = True

if plot_volume:
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.array(times) / 3600, np.array(water_vol), linewidth=2)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Water volume [cc]")
    plt.show()
    fig.savefig("volume_evolution.pdf")

if plot_hydraulic_head:
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.array(times) / 3600, np.array(water_table), linewidth=2)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Fracture hydraulic head [cm]")
    plt.show()
    fig.savefig("hydraulic_head_evolution.pdf")

if plot_dt:
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.array(times) / 3600, np.array(dts), linewidth=2)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Time step [s]")
    plt.show()
    fig.savefig("time_steps.pdf")

if plot_iters:
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.array(times[1:]) / 3600, np.array(iters), linewidth=2)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Number of iterations")
    plt.show()
    fig.savefig("number_of_iters.pdf")
