import mdunsat as mdu
import numpy as np
import porepy as pp
import pickle
import scipy.sparse.linalg as spla
import os

from grid_factory import GridGenerator
from mdunsat.ad_utils import (
    get_conductive_mortars,
    get_ghost_hydraulic_head,
    set_state_as_iterate,
    set_iterate_as_state,
    bulk_cc_var_to_mortar_grid,
)
from mdunsat.soil_catalog import soil_catalog


# %% Retrieve grid buckets
gfo = GridGenerator(
    mesh_args={"mesh_size_frac": 2, "mesh_size_bound": 2.5},
    csv_file="network.csv",
    domain={"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
    constraints=[3, 4],
)
gfo.get_grid_buckets()
gb, ghost_gb = gfo.get_grid_buckets()

grid_list = gfo.grid_list(gb)
bulk_list = gfo.bulk_grid_list(gb)
frac_list = gfo.fracture_grid_list(gb)
lfn_list = gfo.local_fracture_network_grid_list(gb)
edge_list = gfo.edge_list(gb)
frac_edge_list = gfo.fracture_edge_list(gb)

g_bulk = gb.grids_of_dimension(2)[0]
g_top = gb.grids_of_dimension(1)[0]
g_left = gb.grids_of_dimension(1)[1]
g_right = gb.grids_of_dimension(1)[2]

d_bulk = gb.node_props(g_bulk)
d_top = gb.node_props(g_top)
d_left = gb.node_props(g_left)
d_right = gb.node_props(g_right)

ghost_grid_list = gfo.grid_list(ghost_gb)
ghost_bulk_list = gfo.bulk_grid_list(ghost_gb)
ghost_frac_list = gfo.fracture_grid_list(ghost_gb)
ghost_lfn_list = gfo.local_fracture_network_grid_list(ghost_gb)
ghost_edge_list = gfo.edge_list(ghost_gb)
ghost_frac_edge_list = gfo.fracture_edge_list(ghost_gb)

# %% Time parameters
schedule = list(np.linspace(0, 1130, 100))
tsc = pp.TimeSteppingControl(
    schedule=schedule,
    dt_init=0.1,
    dt_min_max=(0.001, 1 * pp.HOUR),
    iter_max=30,
    iter_optimal_range=(10, 15),
    iter_lowupp_factor=(1.3, 0.7),
    recomp_factor=0.5,
    recomp_max=30,
    print_info=True,
)
times = [tsc.time]
dts = [tsc.dt]
export_counter: int = 0

# %% Assign model parameters
# Keywords
param_key: str = "flow"
node_var: str = "hydraulic_head"
edge_var: str = "mortar_flux"

# Primary variables
for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {node_var: {"cells": 1}}
for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {edge_var: {"cells": 1}}

# Parameter assignment
param_update = mdu.ParameterUpdate(gb, param_key)  # object to update parameters

for g, d in gb:

    # Set parameters for the bulk
    if g.dim == gb.dim_max():
        # For convenience, store values of bounding box
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
        top_cc_idx = np.logical_and(cc[1] < (1.00 * y_max), cc[1] > (0.80 * y_max))
        mid_cc_idx = np.logical_and(cc[1] < (0.80 * y_max), cc[1] > (0.70 * y_max))
        bot_cc_idx = np.logical_and(cc[1] < (0.70 * y_max), cc[1] > (0.00 * y_max))

        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(bc_faces.size * ["neu"])
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=list(bc_type))
        bc_values = np.zeros(g.num_faces)
        bc_values[top] = -0.05

        # Saturated hydraulic conductivity
        K_SAT = np.ones(g.num_cells)
        K_SAT_TOP_LAYER = soil_catalog["sandy_loam"].K_s / 3600  # [cm/s]
        K_SAT_MID_LAYER = soil_catalog["sandy_clay_loam"].K_s / 3600  # [cm/s]
        K_SAT_BOT_LAYER = soil_catalog["loamy_sand"].K_s / 3600  # [cm/s]
        K_SAT[top_cc_idx] = K_SAT_TOP_LAYER
        K_SAT[mid_cc_idx] = K_SAT_MID_LAYER
        K_SAT[bot_cc_idx] = K_SAT_BOT_LAYER

        # Residual water content
        THETA_R = np.ones(g.num_cells)
        THETA_R_TOP_LAYER = soil_catalog["sandy_loam"].theta_r  # [-]
        THETA_R_MID_LAYER = soil_catalog["sandy_clay_loam"].theta_r  # [-]
        THETA_R_BOT_LAYER = soil_catalog["loamy_sand"].theta_r  # [-]
        THETA_R[top_cc_idx] = THETA_R_TOP_LAYER
        THETA_R[mid_cc_idx] = THETA_R_MID_LAYER
        THETA_R[bot_cc_idx] = THETA_R_BOT_LAYER

        # Saturated water content
        THETA_S = np.ones(g.num_cells)
        THETA_S_TOP_LAYER = soil_catalog["sandy_loam"].theta_sat  # [-]
        THETA_S_MID_LAYER = soil_catalog["sandy_clay_loam"].theta_sat  # [-]
        THETA_S_BOT_LAYER = soil_catalog["loamy_sand"].theta_sat  # [-]
        THETA_S[top_cc_idx] = THETA_S_TOP_LAYER
        THETA_S[mid_cc_idx] = THETA_S_MID_LAYER
        THETA_S[bot_cc_idx] = THETA_S_BOT_LAYER

        # alpha vanGenuchten parameter
        ALPHA_VG = np.ones(g.num_cells)
        ALPHA_VG_TOP_LAYER = soil_catalog["sandy_loam"].alpha  # [1/cm]
        ALPHA_VG_MID_LAYER = soil_catalog["sandy_clay_loam"].alpha  # [1/cm]
        ALPHA_VG_BOT_LAYER = soil_catalog["loamy_sand"].alpha  # [1/cm]
        ALPHA_VG[top_cc_idx] = ALPHA_VG_TOP_LAYER
        ALPHA_VG[mid_cc_idx] = ALPHA_VG_MID_LAYER
        ALPHA_VG[bot_cc_idx] = ALPHA_VG_BOT_LAYER

        # n vanGenuchten parameter
        N_VG = np.ones(g.num_cells)
        N_VG_TOP_LAYER = soil_catalog["sandy_loam"].n  # [-]
        N_VG_MID_LAYER = soil_catalog["sandy_clay_loam"].n  # [-]
        N_VG_BOT_LAYER = soil_catalog["loamy_sand"].n  # [-]
        N_VG[top_cc_idx] = N_VG_TOP_LAYER
        N_VG[mid_cc_idx] = N_VG_MID_LAYER
        N_VG[bot_cc_idx] = N_VG_BOT_LAYER

        # Initialize bulk data
        specified_parameters: dict = {
            "second_order_tensor": pp.SecondOrderTensor(K_SAT),  # [cm/s]
            "bc": bc,
            "bc_values": bc_values,
            "source": np.zeros(g.num_cells),
            "elevation": g.cell_centers[gb.dim_max() - 1],
            "mass_weight": np.ones(g.num_cells),
            "theta_r": THETA_R,
            "theta_s": THETA_S,
            "alpha_vG": ALPHA_VG,
            "n_vG": N_VG,
            "m_vG": 1 - N_VG ** (-1),
            "time_step": tsc.dt,
        }

    # Set parameters for 1D fractures
    elif g.dim == gb.dim_max() - 1:
        # Note that boundary values are not effectively used, but for discretization
        # purposes, we still have to declare the fields
        bc_faces = g.get_boundary_faces()
        bc_type = bc_faces.size * ["neu"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        specified_parameters = {
            "sin_alpha": gfo.fracture_sin_alpha(g),
            "width": 1.0,
            "aperture": 0.1,
            "bc": bc,
            "bc_values": np.zeros(g.num_faces),
            "datum": np.min(g.face_centers[gb.dim_max() - 1]),
            "elevation": g.cell_centers[gb.dim_max() - 1],
            "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells)),
            "pressure_threshold": 0,
        }
        pp.initialize_data(g, d, param_key, specified_parameters)

    # Set dummy parameters for 0D points
    else:
        specified_parameters = {
            "width": 1.0,
            "sin_alpha": 1.0,
            "aperture": 0.02,
            "datum": np.min(g.cell_centers[gb.dim_max() - 1]),
            "elevation": g.cell_centers[gb.dim_max() - 1],
            "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells)),
            "pressure_threshold": 0,
        }

    pp.initialize_data(g, d, param_key, specified_parameters)

# Parameters for the mortar grids
for e, d in gb.edges():
    mg = d["mortar_grid"]
    ones = np.ones(mg.num_cells)
    zeros = np.zeros(mg.num_cells)
    g_sec, _ = gb.nodes_of_edge(e)
    d_sec = gb.node_props(g_sec)
    aperture = d_sec[pp.PARAMETERS][param_key]["aperture"]
    sat_normal_conductivity = bulk_cc_var_to_mortar_grid(gb, K_SAT, [e])
    normal_diffusivity = 2 * sat_normal_conductivity / aperture  # [1/s]
    is_conductive = zeros
    if mg.dim == gb.dim_max() - 1:
        data = {
            "normal_diffusivity": normal_diffusivity,
            "is_conductive": is_conductive,
            "elevation": mg.cell_centers[gb.dim_max() - 1],
            "pressure_threshold": zeros,
        }
    else:  # 0-dimensional are not effectively used, since we used a reduced geometry
        data = {
            "sat_normal_diffusivity": 0,
            "normal_diffusivity": 0,
            "is_conductive": is_conductive,
            "elevation": mg.cell_centers[gb.dim_max() - 1],
            "pressure_threshold": zeros,
        }
    pp.initialize_data(mg, d, param_key, data)

# %% Set initial states
#
for g, d in gb:
    if g.dim == gb.dim_max():
        top_cells = g.cell_centers[1] > 80
        h0 = -500 + d[pp.PARAMETERS][param_key]["elevation"]
        h0[top_cells] = -5 + d[pp.PARAMETERS][param_key]["elevation"][top_cells]
        pp.set_state(d, state={node_var: h0})
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})
    else:
        pp.set_state(d, state={node_var: d[pp.PARAMETERS][param_key]["datum"]})
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})

for _, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d, state={edge_var: np.zeros(mg.num_cells)})
    pp.set_iterate(d, iterate={edge_var: d[pp.STATE][edge_var]})

# %% Create degree of freedom manager and equation manager
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

# %% Assign primary variables to their corresponding grids
h = equation_manager.merge_variables([(g, node_var) for g in grid_list])
lmbda = equation_manager.merge_variables([(e, edge_var) for e in edge_list])

# Shorthands. Note that the following merged variables all have different id's
h_m = h.previous_iteration()
h_n = h.previous_timestep()
lmbda_m = lmbda.previous_iteration()
lmbda_n = lmbda.previous_timestep()

# Useful auxiliary variables
zeta = pp.ad.ParameterArray(
    param_keyword=param_key,
    array_keyword="elevation",
    grids=grid_list,
    name="elevation",
)
psi = h - zeta  # pressure head ad
psi_m = h_m - zeta  # pressure head previous iteration
psi_n = h_n - zeta  # pressure head previous time step

# %% Grid operators and cell restrictions

# Grid operators
div_bulk = pp.ad.Divergence(grids=bulk_list)
bound_bulk = pp.ad.BoundaryCondition(param_key, grids=bulk_list)

proj = pp.ad.MortarProjections(gb=gb, grids=grid_list, edges=edge_list)
ghost_proj = pp.ad.MortarProjections(
    gb=ghost_gb,
    grids=ghost_grid_list,
    edges=ghost_edge_list,
)

subdomain_proj_scalar = pp.ad.SubdomainProjections(grids=grid_list)
bulk_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(bulk_list)
bulk_face_rest: pp.ad.Matrix = subdomain_proj_scalar.face_restriction(bulk_list)
bulk_face_prol: pp.ad.Matrix = subdomain_proj_scalar.face_prolongation(bulk_list)
frac_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(lfn_list)
frac_cell_prol: pp.ad.Matrix = subdomain_proj_scalar.cell_prolongation(lfn_list)

h_bulk = bulk_cell_rest * h  # bulk hydraulic head
h_lfn = frac_cell_rest * h  # hydraulic heads in the local fracture network

# %% Governing equations in the bulk

# Soil water retention curves
vgm = mdu.VanGenuchtenMualem(gb=gb, param_key=param_key, dof_manager=dof_manager)
theta_ad: pp.ad.Function = vgm.water_content(as_ad=True)
krw_ad: pp.ad.Function = vgm.relative_permeability(as_ad=True)
smc_ad: pp.ad.Function = vgm.moisture_capacity(as_ad=True)

# MPFA discretization
mpfa_bulk = pp.ad.MpfaAd(param_key, bulk_list)

# Obtain single phase flow to compute directionality of upwind scheme
flux_single_phase = (
    mpfa_bulk.flux * bulk_cell_rest * h_m
    + mpfa_bulk.bound_flux * bound_bulk
    + mpfa_bulk.bound_flux * bulk_face_rest * proj.mortar_to_primary_int * lmbda_m
)

# Upwinding of relative permeabilities
upwind = mdu.FluxBaseUpwindAd(gb=gb, grid_list=bulk_list, param_key=param_key)

zeta_faces_bulk = pp.ad.Array(bulk_list[0].face_centers[gb.dim_max() - 1])
psi_bound_bulk = bound_bulk - zeta_faces_bulk
psi_bulk_m = bulk_cell_rest * psi_m
krw_faces = upwind(krw_ad(psi_bulk_m), krw_ad(psi_bound_bulk), flux_single_phase)

# Multiphase Darcy fluxes
flux = (
    krw_faces * mpfa_bulk.flux * bulk_cell_rest * h
    + krw_faces * mpfa_bulk.bound_flux * bound_bulk
    + krw_faces
    * mpfa_bulk.bound_flux
    * bulk_face_rest
    * proj.mortar_to_primary_int
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
    accum_bulk_active = mass_bulk.mass * theta_ad(bulk_cell_rest * psi)
    accum_bulk_inactive = mass_bulk.mass * theta_ad(bulk_cell_rest * psi_n) * (-1)
elif linearization == "modified_picard":
    accum_bulk_active = mass_bulk.mass * (
        (bulk_cell_rest * psi) * smc_ad(bulk_cell_rest * psi_m)
    )
    accum_bulk_inactive = mass_bulk.mass * (
        theta_ad(bulk_cell_rest * psi_m)
        - smc_ad(bulk_cell_rest * psi_m) * (bulk_cell_rest * psi_m)
        - theta_ad(bulk_cell_rest * psi_n)
    )
elif linearization == "l_scheme":
    L = 0.001
    accum_bulk_active = L * mass_bulk.mass * (bulk_cell_rest * psi)
    accum_bulk_inactive = mass_bulk.mass * (
        theta_ad(bulk_cell_rest * psi_m)
        - L * (bulk_cell_rest * psi_m)
        - theta_ad(bulk_cell_rest * psi_n)
    )
else:
    raise NotImplementedError(
        "Linearization scheme not implemented. Use 'newton', "
        "'modified_picard', or 'l_scheme'."
    )

accumulation_bulk = accum_bulk_active + accum_bulk_inactive
conserv_bulk_eq = accumulation_bulk + dt_ad * div_bulk * flux - dt_ad * source_bulk
conserv_bulk_eq.discretize(gb=gb)
conserv_bulk_num_new = conserv_bulk_eq.evaluate(dof_manager=dof_manager).val

# %% Governing equations in the fracture

# Get water volume as a function of the hydraulic head, and its first derivative
fv = mdu.FractureVolume(gb=gb, fracture_grids=lfn_list, param_key=param_key)
vol_ad: pp.ad.Function = fv.fracture_volume(as_ad=True)
vol_cap_ad: pp.ad.Function = fv.volume_capacity(as_ad=True)
vol = fv.fracture_volume(as_ad=False)

linearization = "newton"  # linearization of the fracture equations
if linearization == "newton":
    accum_frac_active = vol_ad(frac_cell_rest * h)
    accum_frac_inactive = vol_ad(frac_cell_rest * h_n) * (-1)
elif linearization == "modified_picard":
    accum_frac_active = (frac_cell_rest * h) * vol_cap_ad(frac_cell_rest * h_m)
    accum_frac_inactive = (
        vol_ad(frac_cell_rest * h_m)
        - vol_cap_ad(frac_cell_rest * h_m) * (frac_cell_rest * h_m)
        - vol_ad(frac_cell_rest * h_n)
    )
elif linearization == "l_scheme":
    L = 0.015
    accum_frac_active = L * frac_cell_rest * h
    accum_frac_inactive = (
        vol_ad(frac_cell_rest * h_m)
        - L * frac_cell_rest * h_m
        - vol_ad(frac_cell_rest * h_n)
    )
else:
    raise NotImplementedError(
        "Linearization scheme not implemented. Use 'newton', "
        "'modified_picard', or 'l_scheme'."
    )

# Retrieve sources from mortar
sources_from_mortar = frac_cell_rest * proj.mortar_to_secondary_int * lmbda
# Accumulation terms
accum_frac = accum_frac_active + accum_frac_inactive
# Declare conservation equation
conserv_frac_eq = accum_frac - 0.5 * dt_ad * sources_from_mortar
conserv_frac_eq.discretize(gb=gb)
conserv_frac_num_new = conserv_frac_eq.evaluate(dof_manager=dof_manager).val

# %% Governing equations on the interfaces
mpfa_global = pp.ad.MpfaAd(param_key, grid_list)
robin = pp.ad.RobinCouplingAd(param_key, edge_list)

proj_tr_psi_l = pp.ad.ParameterArray(
    param_keyword=param_key,
    array_keyword="pressure_threshold",
    edges=edge_list,
    name="interface pressure threshold",
)

# Get projected higher-dimensional trace of the hydraulic head onto the mortars
proj_h_high = (
    proj.primary_to_mortar_avg * mpfa_global.bound_pressure_cell * h
    + proj.primary_to_mortar_avg
    * mpfa_global.bound_pressure_face
    * proj.mortar_to_primary_int
    * lmbda
)

proj_h_high_m = (
    proj.primary_to_mortar_avg * mpfa_global.bound_pressure_cell * h_m
    + proj.primary_to_mortar_avg
    * mpfa_global.bound_pressure_face
    * proj.mortar_to_primary_int
    * lmbda_m
)

# Get projected lower-dimensional (ghost) hydraulic head onto the mortars
pfh = mdu.GhostHydraulicHead(gb=gb, ghost_gb=ghost_gb)
frac_to_mortar_ad: pp.ad.Function = pfh.proj_fra_hyd_head(as_ad=True)
proj_h_low = frac_to_mortar_ad(frac_cell_rest * h)

# Array parameter that keeps track of conductive (1) and blocking (0) mortar cells
# Note that if it is blocking, the whole discrete equation is removed for that mortar cell
is_conductive = pp.ad.ParameterArray(param_key, "is_conductive", edges=edge_list)

# Interface flux
mortar_flux = robin.mortar_discr * (proj_h_low - proj_h_high) * is_conductive
interface_flux_eq = mortar_flux + lmbda
interface_flux_eq.discretize(gb=gb)
interface_flux_num_new = interface_flux_eq.evaluate(dof_manager=dof_manager).val

# %% Assemble discrete equations, feed into the equation manager, and discretize.
eqs = {
    "bulk_conservation": conserv_bulk_eq,
    "fracture_conservation": conserv_frac_eq,
    "interface_fluxes": interface_flux_eq,
}
equation_manager.equations.update(eqs)
equation_manager.discretize(gb)

# %% Initialize exporter. Note that we use the ghost grid rather than the physical one
d_bulk = gb.node_props(bulk_list[0])
d_bulk_ghost = ghost_gb.node_props(ghost_bulk_list[0])
z_bulk = bulk_list[0].cell_centers[1]

d_top = gb.node_props(frac_list[0])
d_top_ghost = ghost_gb.node_props(ghost_frac_list[0])
z_top = frac_list[0].cell_centers[1]
z_top_ghost = ghost_frac_list[0].cell_centers[1]

d_left = gb.node_props(frac_list[1])
d_left_ghost = ghost_gb.node_props(ghost_frac_list[1])
z_left = frac_list[1].cell_centers[1]
z_left_ghost = ghost_frac_list[1].cell_centers[1]

d_right = gb.node_props(frac_list[2])
d_right_ghost = ghost_gb.node_props(ghost_frac_list[2])
z_right = frac_list[2].cell_centers[1]
z_right_ghost = ghost_frac_list[2].cell_centers[1]

d_point = gb.node_props(lfn_list[-1])
d_point_ghost = ghost_gb.node_props(ghost_lfn_list[-1])

# # Set state in bulk ghost grid
pp.set_state(
    data=d_bulk_ghost,
    state={
        node_var: d_bulk[pp.STATE][node_var],
        "pressure_head": (
            d_bulk[pp.STATE][node_var] - d_bulk[pp.PARAMETERS][param_key]["elevation"]
        ),
    },
)

# Set state in fracture ghost grids
pp.set_state(
    data=d_top_ghost,
    state={
        node_var: d_top[pp.PARAMETERS][param_key]["datum"]
        * np.ones(ghost_frac_list[0].num_cells),
        "pressure_head": d_top[pp.PARAMETERS][param_key]["datum"]
        * np.ones(ghost_frac_list[0].num_cells)
        - z_top_ghost,
    },
)

pp.set_state(
    data=d_left_ghost,
    state={
        node_var: d_left[pp.PARAMETERS][param_key]["datum"]
        * np.ones(ghost_frac_list[1].num_cells),
        "pressure_head": d_left[pp.PARAMETERS][param_key]["datum"]
        * np.ones(ghost_frac_list[1].num_cells)
        - z_left_ghost,
    },
)

pp.set_state(
    data=d_right_ghost,
    state={
        node_var: d_right[pp.PARAMETERS][param_key]["datum"]
        * np.ones(ghost_frac_list[2].num_cells),
        "pressure_head": d_left[pp.PARAMETERS][param_key]["datum"]
        * np.ones(ghost_frac_list[2].num_cells)
        - z_right_ghost,
    },
)

pp.set_state(data=d_point_ghost, state={node_var: 0, "pressure_head": 0})

exporter_ghost = pp.Exporter(ghost_gb, "junction", "out")
exporter_ghost.write_vtu([node_var, "pressure_head"], time_step=0)

# %% Time loop
total_iteration_counter = 0
iters = []
ABS_TOL = 1e-3
is_mortar_conductive = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
control_faces = is_mortar_conductive
scheduled_time = tsc.schedule[1]

water_top = [0]
water_left = [0]
water_right = [0]

# Time loop
while tsc.time < tsc.time_final:
    tsc.time += tsc.dt
    itr: int = 0
    res_norm: float = 1e8
    rel_res: float = 1e8
    print(f"Time: {np.round(tsc.time, 2)}")

    # Solver loop
    while itr <= tsc.iter_max and not res_norm < ABS_TOL:
        # Solve system of equations and distribute variables to pp.ITERATE
        A, b = equation_manager.assemble()
        solution = spla.spsolve(A, b, use_umfpack=True)
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
    if np.any(vol((frac_cell_rest * h).evaluate(dof_manager).val) < 0):
        tsc.next_time_step(recompute_solution=True, iterations=itr - 1)
        param_update.update_time_step(tsc.dt)
        print(f"Encountered negative volume. Reducing dt and recomputing solution.")
        set_iterate_as_state(gb, node_var, edge_var)
        continue
    # end of checking

    # Recompute solution is capillary barrier is overcome. Note that dt remains the same
    is_mortar_conductive = get_conductive_mortars(
        gb, dof_manager, param_key, proj_h_high, proj_h_low, proj_tr_psi_l, edge_list
    )
    if control_faces.sum() == 0 and is_mortar_conductive.sum() > 0:
        param_update.update_mortar_conductivity_state(is_mortar_conductive, edge_list)
        print("Encountered saturated mortar cells. Solution will be updated.")
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
    # end of checking

    # Enforce hydrostatic equilibration
    vol_t = vol(h_lfn.evaluate(dof_manager).val)[0]
    vol_l = vol(h_lfn.evaluate(dof_manager).val)[1]
    vol_r = vol(h_lfn.evaluate(dof_manager).val)[2]

    a_t = d_top[pp.PARAMETERS][param_key]["aperture"]
    a_l = d_left[pp.PARAMETERS][param_key]["aperture"]
    a_r = d_right[pp.PARAMETERS][param_key]["aperture"]

    vol_tmax = g_top.cell_volumes[0] * a_t
    vol_lmax = g_left.cell_volumes[0] * a_l
    vol_rmax = g_right.cell_volumes[0] * a_r

    sin_t = d_top[pp.PARAMETERS][param_key]["sin_alpha"]
    sin_l = d_left[pp.PARAMETERS][param_key]["sin_alpha"]
    sin_r = d_right[pp.PARAMETERS][param_key]["sin_alpha"]

    zmin_t = d_top[pp.PARAMETERS][param_key]["datum"]
    zmin_l = d_left[pp.PARAMETERS][param_key]["datum"]
    zmin_r = d_right[pp.PARAMETERS][param_key]["datum"]

    # Left and right fractures can receive water
    if (vol_lmax - vol_l) > vol_t / 2 and (vol_rmax - vol_r) > vol_t / 2:
        vol_l = vol_l + vol_t / 2
        vol_r = vol_r + vol_t / 2
        vol_t = 0
        # Correct values of hydraulic head after redistribution
        d_top[pp.STATE][pp.ITERATE][node_var] = (vol_t / a_t) * sin_t + zmin_t
        d_left[pp.STATE][pp.ITERATE][node_var] = (vol_l / a_l) * sin_l + zmin_l
        d_right[pp.STATE][pp.ITERATE][node_var] = (vol_r / a_r) * sin_r + zmin_r

    # Right will be filled by and the rest is poured into left
    elif vol_t / 2 > (vol_rmax - vol_r):
        vol_to_fill_right = vol_rmax - vol_r
        vol_r = vol_rmax
        residual_vol = vol_t - vol_to_fill_right
        if (vol_lmax - vol_l) > residual_vol:
            vol_l = vol_l + residual_vol
            vol_t = 0
        else:
            vol_to_fill_left = vol_lmax - vol_l
            vol_l = vol_lmax
            residual_vol = residual_vol - vol_to_fill_left
            vol_t = residual_vol
            if vol_t > vol_tmax:
                vol_t = vol_tmax
        # Correct values of hydraulic head after redistribution
        if vol_t <= vol_tmax:
            d_top[pp.STATE][pp.ITERATE][node_var] = (vol_t / a_t) * sin_t + zmin_t
        if vol_l <= vol_lmax:
            d_left[pp.STATE][pp.ITERATE][node_var] = (vol_l / a_l) * sin_l + zmin_l
        if vol_r <= vol_rmax:
            d_right[pp.STATE][pp.ITERATE][node_var] = (vol_r / a_r) * sin_r + zmin_r

    # Save number of iterations and time step
    iters.append(itr - 1)
    times.append(tsc.time)
    dts.append(tsc.dt)
    water_left.append(vol_l)
    water_right.append(vol_r)
    water_top.append(vol_t)

    # Successful convergence
    tsc.next_time_step(recompute_solution=False, iterations=itr - 1)
    if tsc.time + tsc.dt > scheduled_time:
        tsc.dt = scheduled_time - tsc.time
        print("Changing dt to match scheduled time.")
    param_update.update_time_step(tsc.dt)
    set_state_as_iterate(gb, node_var, edge_var)
    print()

    # Export to ParaView
    pp.set_state(
        data=d_bulk_ghost,
        state={
            node_var: d_bulk[pp.STATE][node_var],
            "pressure_head": d_bulk[pp.STATE][node_var]
            - d_bulk[pp.PARAMETERS][param_key]["elevation"],
        },
    )
    pp.set_state(
        data=d_top_ghost,
        state={
            node_var: get_ghost_hydraulic_head(
                ghost_frac_list[0], d_top[pp.STATE][node_var]
            ),
            "pressure_head": d_top_ghost[pp.STATE][node_var] - z_top_ghost,
        },
    )
    pp.set_state(
        data=d_left_ghost,
        state={
            node_var: get_ghost_hydraulic_head(
                ghost_frac_list[1], d_left[pp.STATE][node_var]
            ),
            "pressure_head": d_left_ghost[pp.STATE][node_var] - z_left_ghost,
        },
    )
    pp.set_state(
        data=d_right_ghost,
        state={
            node_var: get_ghost_hydraulic_head(
                ghost_frac_list[2], d_right[pp.STATE][node_var]
            ),
            "pressure_head": d_right_ghost[pp.STATE][node_var] - z_right_ghost,
        },
    )

    # Export if time is in schedule
    if tsc.time in tsc.schedule:
        export_counter += 1
        exporter_ghost.write_vtu([node_var, "pressure_head"], time_step=export_counter)
        if not np.isclose(tsc.time, tsc.time_final, atol=1e-3):
            scheduled_time = tsc.schedule[export_counter + 1]

# %% Dump to pickle

# Create the directory if it does not exist
if not os.path.exists("out"):
    os.makedirs("out")

with open("out/water_volume.pickle", "wb") as handle:
    pickle.dump(
        [times, water_left, water_right, water_top],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
