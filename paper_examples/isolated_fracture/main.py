import os
import pickle

import mdunsat as mdu
import numpy as np
import porepy as pp
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from grid_factory import GridGenerator
from mdunsat.ad_utils import (bulk_cc_var_to_mortar_grid,
                              get_conductive_mortars, set_iterate_as_state,
                              set_state_as_iterate)
from mdunsat.soil_catalog import soil_catalog

# %% Retrieve grid buckets
gfo = GridGenerator(
    mesh_args={"mesh_size_frac": 2, "mesh_size_bound": 2.5},
    csv_file="network.csv",
    domain={"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
    constraints=[1, 2, 3, 4],
)
gb, ghost_gb = gfo.get_grid_buckets()

grid_list = gfo.get_grid_list(gb)
bulk_list = gfo.get_bulk_list(gb)
frac_list = gfo.get_fracture_list(gb)
edge_list = gfo.get_edge_list(gb)

d_bulk = gb.node_props(bulk_list[0])
d_frac = gb.node_props(frac_list[0])
d_edge = gb.edge_props(edge_list[0])

ghost_grid_list = gfo.get_grid_list(ghost_gb)
ghost_bulk_list = gfo.get_bulk_list(ghost_gb)
ghost_frac_list = gfo.get_fracture_list(ghost_gb)
ghost_edge_list = gfo.get_edge_list(ghost_gb)

# %% Time parameters
schedule = list(np.linspace(0, 4800, 100, dtype=np.int32))
tsc = pp.TimeSteppingControl(
    schedule=schedule,
    dt_init=0.01,
    dt_min_max=(0.01, 48),
    iter_max=30,
    iter_optimal_range=(10, 15),
    iter_lowupp_factor=(1.5, 0.5),
    recomp_factor=0.5,
    recomp_max=12,
    print_info=True,
)

# Initialize list of variables that will be saved at each time step
times = [tsc.time]
dts = [tsc.dt]
water_vol = [0]
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
param_update = mdu.ParameterUpdate(gb, param_key)  # object to update parameters

for g, d in gb:
    if g.dim == gb.dim_max():
        # Store values of bounding box
        x_min: float = gb.bounding_box()[0][0]
        y_min: float = gb.bounding_box()[0][1]
        x_max: float = gb.bounding_box()[1][0]
        y_max: float = gb.bounding_box()[1][1]

        # Face indices of the boundaries
        fc: np.ndarray = g.face_centers
        top: np.ndarray = np.where(np.abs(fc[1] - y_max) < 1e-5)[0]
        bottom: np.ndarray = np.where(np.abs(fc[1]) < 1e-5)[0]
        left: np.ndarray = np.where(np.abs(fc[0]) < 1e-5)[0]
        right: np.ndarray = np.where(np.abs(fc[0] - x_max) < 1e-5)[0]
        top_left: np.ndarray = top[np.where(np.abs(fc[0][top] < (x_max / 2)))[0]]

        # Left and right cell indices
        cc: np.ndarray = g.cell_centers
        cells_left: np.ndarray = np.where(np.abs(cc[0] < (x_max / 2)))[0]
        cells_right: np.ndarray = np.where(np.abs(cc[0] > (x_max / 2)))[0]

        # Cell indices corresponding to the blocking fractures
        blocking_idx: list = [
            cc[0] < (0.50 * x_max),
            cc[0] > (0.30 * x_max),
            cc[1] < (0.80 * y_max),
            cc[1] > (0.60 * y_max),
        ]
        mult_cond: bool = np.logical_and(blocking_idx[0], blocking_idx[1])
        mult_cond: bool = np.logical_and(blocking_idx[2], mult_cond)
        mult_cond: bool = np.logical_and(blocking_idx[3], mult_cond)
        blocking_cells: np.ndarray = np.where(mult_cond)[0]

        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(bc_faces.size * ["neu"])
        bc_type[np.in1d(bc_faces, top_left)] = "dir"
        bc: pp.BoundaryCondition = pp.BoundaryCondition(
            g, faces=bc_faces, cond=list(bc_type)
        )
        bc_values: np.ndarray = np.zeros(g.num_faces)
        bc_values[top_left] = 2.5 + y_max  # 2.5 (psi) + y_max (max(zeta))

        # Hydraulic conductivity
        K_SAT_NEWMEX = soil_catalog["new_mexico"].K_s / 3600  # [cm/s]
        K_SAT_SANDY_CLAY_LOAM = soil_catalog["sandy_clay_loam"].K_s / 3600  # [cm/s]
        K_SAT = K_SAT_NEWMEX * np.ones(g.num_cells)
        K_SAT[mult_cond] = K_SAT_SANDY_CLAY_LOAM

        # VanGenuchten-Mualem Parameters
        THETA_SAT_NEWMEX = soil_catalog["new_mexico"].theta_sat  # [-]
        THETA_SAT = THETA_SAT_NEWMEX * np.ones(g.num_cells)
        THETA_SAT_SANDY_CLAY_LOAM = soil_catalog["sandy_clay_loam"].theta_sat  # [-]
        THETA_SAT[mult_cond] = THETA_SAT_SANDY_CLAY_LOAM

        THETA_RES_NEWMEX = soil_catalog["new_mexico"].theta_r  # [-]
        THETA_RES = THETA_RES_NEWMEX * np.ones(g.num_cells)
        THETA_RES_SANDY_CLAY_LOAM = soil_catalog["sandy_clay_loam"].theta_r  # [-]
        THETA_RES[mult_cond] = THETA_RES_SANDY_CLAY_LOAM

        ALPHA_VG_NEWMEX = soil_catalog["new_mexico"].alpha  # [1/cm]
        ALPHA_VG = ALPHA_VG_NEWMEX * np.ones(g.num_cells)
        ALPHA_VG_SANDY_CLAY_LOAM = soil_catalog["sandy_clay_loam"].alpha  # [1/cm]
        ALPHA_VG[mult_cond] = ALPHA_VG_SANDY_CLAY_LOAM

        N_VG_NEWMEX = soil_catalog["new_mexico"].n  # [-]
        N_VG = N_VG_NEWMEX * np.ones(g.num_cells)
        N_VG_SANDY_CLAY_LOAM = soil_catalog["sandy_clay_loam"].n  # [-]
        N_VG[mult_cond] = N_VG_SANDY_CLAY_LOAM

        # Initialize bulk data
        specified_parameters: dict = {
            "second_order_tensor": pp.SecondOrderTensor(K_SAT),  # [cm/s]
            "bc": bc,
            "bc_values": bc_values,
            "source": np.zeros(g.num_cells),
            "elevation": g.cell_centers[gb.dim_max() - 1],
            "mass_weight": np.ones(g.num_cells),
            "theta_s": THETA_SAT,
            "theta_r": THETA_RES,
            "alpha_vG": ALPHA_VG,
            "n_vG": N_VG,
            "m_vG": 1 - N_VG ** (-1),
            "time_step": tsc.dt,  # [s]
        }
        pp.initialize_data(g, d, param_key, specified_parameters)

    else:
        # Parameters for the fracture grids
        specified_parameters = {
            "aperture": 0.1,  # [cm]
            "datum": np.min(g.face_centers[gb.dim_max() - 1]),
            "elevation": g.cell_centers[gb.dim_max() - 1],
            "sin_alpha": 1.0,
            "width": 1.0,
            "pressure_threshold": 0,
        }
        pp.initialize_data(g, d, param_key, specified_parameters)

# Parameters for the mortar
for e, d in gb.edges():
    mg = d["mortar_grid"]
    ones = np.ones(mg.num_cells)
    zeros = np.zeros(mg.num_cells)
    g_low, g_high = gb.nodes_of_edge(e)
    d_low, d_high = gb.node_props(g_low), gb.node_props(g_high)
    aperture = d_low[pp.PARAMETERS][param_key]["aperture"]
    sat_conductivity = bulk_cc_var_to_mortar_grid(gb, K_SAT)
    sat_normal_diffusivity = (2 * sat_conductivity) / aperture  # [1/s]
    is_conductive = zeros
    data = {
        "sat_normal_diffusivity": sat_normal_diffusivity,
        "normal_diffusivity": sat_normal_diffusivity,
        "is_conductive": is_conductive,
        "elevation": mg.cell_centers[gb.dim_max() - 1],
        "pressure_threshold": 0,
    }
    pp.initialize_data(mg, d, param_key, data)

# %% Set initial states
for g, d in gb:
    if g.dim == gfo.dim:
        pp.set_state(
            d, state={node_var: -500 + d[pp.PARAMETERS][param_key]["elevation"]}
        )
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})
    else:
        pp.set_state(
            d,
            state={node_var: np.array([d[pp.PARAMETERS][param_key]["datum"]])},
        )
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})

for _, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d, state={edge_var: np.zeros(mg.num_cells)})
    pp.set_iterate(d, iterate={edge_var: d[pp.STATE][edge_var]})

# %% Create degree of freedom manager and equation manager
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

# %% Assign primary variables to their corresponding grids
h_bulk = equation_manager.merge_variables([(g, node_var) for g in bulk_list])
h_frac = equation_manager.merge_variables([(g, node_var) for g in frac_list])
lmbda = equation_manager.merge_variables([(e, edge_var) for e in edge_list])

# Shorthands. Note that the following merged variables all have different id's
h_bulk_m = h_bulk.previous_iteration()
h_bulk_n = h_bulk.previous_timestep()
h_frac_m = h_frac.previous_iteration()
h_frac_n = h_frac.previous_timestep()
lmbda_m = lmbda.previous_iteration()
lmbda_n = lmbda.previous_timestep()

# Useful auxiliary variables
zc_bulk_ad = pp.ad.Array(bulk_list[0].cell_centers[gfo.dim - 1])
psib: pp.ad.Operator = h_bulk - zc_bulk_ad  # pressure head (active)
psib_m: pp.ad.Operator = h_bulk_m - zc_bulk_ad  # pressure head at prev iter
psib_n: pp.ad.Operator = h_bulk_n - zc_bulk_ad  # pressure head at prev time

# %% Grid operators and cell restrictions

# Grid operators
div_bulk = pp.ad.Divergence(grids=bulk_list)
bound_bulk = pp.ad.BoundaryCondition(param_key, grids=bulk_list)
projections = pp.ad.MortarProjections(gb=gb, grids=grid_list, edges=edge_list)
ghost_projections = pp.ad.MortarProjections(
    gb=ghost_gb, grids=ghost_grid_list, edges=ghost_edge_list
)
subdomain_proj_scalar = pp.ad.SubdomainProjections(grids=grid_list)
bulk_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(bulk_list)
bulk_face_rest: pp.ad.Matrix = subdomain_proj_scalar.face_restriction(bulk_list)
bulk_face_prol: pp.ad.Matrix = subdomain_proj_scalar.face_prolongation(bulk_list)
frac_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(frac_list)
frac_cell_prol: pp.ad.Matrix = subdomain_proj_scalar.cell_prolongation(frac_list)

# %% Governing equations in the bulk

# Soil water retention curves
vgm = mdu.VanGenuchtenMualem(gb=gb, param_key=param_key, dof_manager=dof_manager)
theta_ad: pp.ad.Function = vgm.water_content(as_ad=True)
krw_ad: pp.ad.Function = vgm.relative_permeability(as_ad=True)
smc_ad: pp.ad.Function = vgm.moisture_capacity(as_ad=True)

# MPFA discretization
mpfa_bulk = pp.ad.MpfaAd(param_key, bulk_list)

# Obtain single phase flow to compute directionality of upwind scheme
flux_single_phase_bulk: pp.ad.Operator = (
    mpfa_bulk.flux * h_bulk_m
    + mpfa_bulk.bound_flux * bound_bulk
    + mpfa_bulk.bound_flux
    * bulk_face_rest
    * projections.mortar_to_primary_int
    * lmbda_m
)

# Upwinding of relative permeabilities
upwind = mdu.FluxBaseUpwindAd(gb=gb, grid_list=bulk_list, param_key=param_key)
zf_bulk_ad = pp.ad.Array(bulk_list[0].face_centers[gb.dim_max() - 1])
psi_bc_ad = bound_bulk - zf_bulk_ad
psi_bc_ad.set_name("boundary pressure head")
krw_faces_ad: pp.ad.Operator = upwind(
    krw_ad(psib_m), krw_ad(psi_bc_ad), flux_single_phase_bulk
)
krw_faces_ad.discretize(gb)
krw_faces_ad.evaluate(dof_manager)

# Multiphase Darcy fluxes
flux_bulk: pp.ad.Operator = (
    krw_faces_ad * mpfa_bulk.flux * h_bulk
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
dt_ad = mdu.ParameterScalar(param_key, "time_step", grids=bulk_list)
source_bulk = pp.ad.ParameterArray(param_key, "source", grids=bulk_list)
mass_bulk = pp.ad.MassMatrixAd(param_key, grids=bulk_list)

linearization = "modified_picard"  # linearization of the bulk equations
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

# %% Governing equations in the fracture

# Get water volume as a function of the hydraulic head, and its first derivative
fv = mdu.FractureVolume(gb=gb, fracture_grids=frac_list, param_key=param_key)
vol_ad: pp.ad.Function = fv.fracture_volume(as_ad=True)
vol_cap_ad: pp.ad.Function = fv.volume_capacity(as_ad=True)
vol = fv.fracture_volume(as_ad=False)

linearization = "newton"  # linearization of the bulk equations
if linearization == "newton":
    accum_frac_active = vol_ad(h_frac)
    accum_frac_inactive = vol_ad(h_frac_n) * (-1)
elif linearization == "modified_picard":
    accum_frac_active = h_frac * vol_cap_ad(h_frac_m)
    accum_frac_inactive = (
        vol_ad(h_frac_m) - vol_cap_ad(h_frac_m) * h_frac_m - vol_ad(h_frac_n)
    )
elif linearization == "l_scheme":
    L = 0.015
    accum_frac_active = L * h_frac
    accum_frac_inactive = vol_ad(h_frac_m) - L * h_frac_m - vol_ad(h_frac_n)
else:
    raise NotImplementedError(
        "Linearization scheme not implemented. Use 'newton', "
        "'modified_picard', or 'l_scheme'."
    )

# Concatenate apertures from relevant grids, and converted into a pp.ad.Matrix
aperture = np.array(
    [
        d[pp.PARAMETERS][param_key]["aperture"]
        for g, d in gb
        if g.dim == gb.dim_max() - 1
    ]
)
aperture_ad = pp.ad.Matrix(sps.spdiags(aperture, 0, aperture.size, aperture.size))
# Retrieve sources from mortar
sources_from_mortar = frac_cell_rest * projections.mortar_to_secondary_int * lmbda
# Accumulation terms
accum_frac = accum_frac_active + accum_frac_inactive
# Declare conservation equation
conserv_frac_eq = accum_frac - 0.5 * dt_ad * sources_from_mortar

# Evaluate and discretize
conserv_frac_eq.discretize(gb=gb)
conserva_frac_num = conserv_frac_eq.evaluate(dof_manager=dof_manager)

# %% Governing equations on the interfaces
robin = pp.ad.RobinCouplingAd(param_key, edge_list)

proj_tr_psi_l = pp.ad.ParameterArray(
    param_keyword=param_key,
    array_keyword="pressure_threshold",
    edges=edge_list,
    name="interface pressure threshold",
)

# Projected bulk pressure traces onto the mortar grid
proj_tr_h_bulk = (
    projections.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_cell
    * h_bulk
    + projections.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_face
    * bulk_face_rest
    * projections.mortar_to_primary_int
    * lmbda
)

proj_tr_h_bulk_m = (
    projections.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_cell
    * h_bulk_m
    + projections.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_face
    * bulk_face_rest
    * projections.mortar_to_primary_int
    * lmbda_m
)

# Get projected ghost fracture hydraulic head onto the adjacent mortar grids
pfh = mdu.GhostHydraulicHead(gb=gb, ghost_gb=ghost_gb)
frac_to_mortar_ad: pp.ad.Function = pfh.proj_fra_hyd_head(as_ad=True)
proj_h_frac = frac_to_mortar_ad(h_frac)
proj_h_frac_m = frac_to_mortar_ad(h_frac_m)

# Compute interface relative permeabilities
elevation_intf = pp.ad.ParameterArray(param_key, "elevation", edges=edge_list)
interface_upwind: pp.ad.Operator = mdu.InterfaceUpwindAd()
proj_tr_psi_bulk_m = proj_tr_h_bulk_m - elevation_intf
proj_psi_frac_m = proj_h_frac_m - elevation_intf
krw_intf = interface_upwind(
    proj_tr_psi_bulk_m,
    krw_ad(proj_tr_psi_bulk_m),
    proj_psi_frac_m,
    krw_ad(proj_psi_frac_m),
)

# Array parameter that keeps track of conductive (1) and blocking (0) mortar cells
# Note that if it is blocking, the whole discrete equation is removed for that mortar cell
is_conductive = pp.ad.ParameterArray(param_key, "is_conductive", edges=edge_list)

# Interface flux
mortar_flux = (
    robin.mortar_discr * krw_intf * (proj_tr_h_bulk - proj_h_frac) * is_conductive
)
interface_flux_eq = mortar_flux + lmbda
interface_flux_eq.discretize(gb=gb)
interface_flux_num = interface_flux_eq.evaluate(dof_manager=dof_manager)

# %% Assemble discrete equations, feed into the equation manager, and discretize.
eqs = {
    "bulk_conservation": conserv_bulk_eq,
    "fracture_conservation": conserv_frac_eq,
    "interface_fluxes": interface_flux_eq,
}
equation_manager.equations.update(eqs)
equation_manager.discretize(gb)

# %% Initialize exporter. Note that we use the ghost grid rather than the physical one
export_counter = 0
exporter = pp.Exporter(gb, "isolated", "out")
exporter.write_vtu([node_var], time_step=export_counter)

# %% Time loop
total_iteration_counter: int = 0
iters: list = []
ABS_TOL: float = 1e-3
is_mortar_conductive: np.ndarray = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
control_faces: np.ndarray = is_mortar_conductive
scheduled_time = tsc.schedule[1]

# Time loop
while tsc.time < tsc.time_final:
    tsc.time += tsc.dt
    itr: int = 0
    res_norm: float = 1e8
    rel_res: float = 1e8
    print(f"Time: {round(tsc.time, 2)}")

    # Solver loop
    while itr <= tsc.iter_max and not res_norm < ABS_TOL:

        # Solve system of equations and distribute variables to pp.ITERATE
        A, b = equation_manager.assemble()
        solution = spla.spsolve(A, b)
        dof_manager.distribute_variable(solution, additive=True, to_iterate=True)

        # Compute 'error' as norm of the residual
        res_norm = np.linalg.norm(b, 2)
        if itr == 0:
            init_res_norm = res_norm
        else:
            inti_res_norm = max(res_norm, init_res_norm)
        rel_res = res_norm / init_res_norm

        # Prepare next iteration
        itr += 1
        total_iteration_counter += 1
    # end of solver loop

    # Recompute solution if we did not achieve convergence
    if res_norm > ABS_TOL or np.isnan(res_norm):
        tsc.next_time_step(recompute_solution=True, iterations=itr - 1)
        param_update.update_time_step(tsc.dt)
        set_iterate_as_state(gb, node_var, edge_var)
        continue
    # end of checking

    # Recompute solution if negative volume is encountered
    if np.any(vol(h_frac.evaluate(dof_manager).val) < 0):
        print(f"Encountered negative volume. Reducing dt and recomputing solution.")
        tsc.next_time_step(recompute_solution=True, iterations=itr - 1)
        param_update.update_time_step(tsc.dt)
        set_iterate_as_state(gb, node_var, edge_var)
        continue
    # end of checking

    # Recompute solution is capillary barrier is overcome. Note that dt remains the same
    is_mortar_conductive = get_conductive_mortars(
        gb,
        dof_manager,
        param_key,
        proj_tr_h_bulk,
        proj_h_frac,
        proj_tr_psi_l,
        edge_list,
    )
    if control_faces.sum() == 0 and is_mortar_conductive.sum() > 0:
        param_update.update_mortar_conductivity_state(is_mortar_conductive, edge_list)
        print("Encountered saturated mortar cells. Recomputing solution")
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
    iters.append(itr - 1)
    times.append(tsc.time)
    dts.append(tsc.dt)

    # Successful convergence
    print(f"Solution converged in {itr - 1} iterations.")
    tsc.next_time_step(recompute_solution=False, iterations=itr - 1)
    if tsc.time + tsc.dt > scheduled_time:
        tsc.dt = scheduled_time - tsc.time
        print("Changing dt to match scheduled time.")
    param_update.update_time_step(tsc.dt)
    if (
        vol(d_frac[pp.STATE][node_var])[0]
        < frac_list[0].cell_volumes.sum() * d_frac[pp.PARAMETERS][param_key]["aperture"]
    ):
        water_vol.append(vol(d_frac[pp.STATE][node_var])[0])
    else:
        water_vol.append(vol(d_frac[pp.STATE][node_var])[0])
    set_state_as_iterate(gb, node_var, edge_var)

    # Export if time is in schedule
    if np.isclose(tsc.time, scheduled_time, atol=1e-3):
        export_counter += 1
        exporter.write_vtu([node_var], time_step=export_counter)
        if not np.isclose(tsc.time, tsc.time_final, atol=1e-3):
            scheduled_time = tsc.schedule[export_counter + 1]

    print()

# %% Dump to pickle

# Create the directory if it does not exist
if not os.path.exists("out"):
    os.makedirs("out")

# Dump into the pickle file
with open("out/water_volume.pickle", "wb") as handle:
    pickle.dump([times, water_vol], handle, protocol=pickle.HIGHEST_PROTOCOL)
