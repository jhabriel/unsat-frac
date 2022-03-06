import mdunsat as mdu
import numpy as np
import porepy as pp
import pickle
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from grid_factory import GridGenerator
from mdunsat.ad_utils import (
    get_conductive_mortars,
    get_ghost_hydraulic_head,
    set_state_as_iterate,
    set_iterate_as_state
)

# %% Retrieve grid buckets
gfo = GridGenerator(
    mesh_args={"mesh_size_frac": 10, "mesh_size_bound": 10},
    csv_file="network.csv",
    domain={"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
    constraints=[2, 3, 4, 5],
)
gb, ghost_gb = gfo.get_grid_buckets()

grid_list = gfo.grid_list(gb)
bulk_list = gfo.bulk_grid_list(gb)
frac_list = gfo.fracture_grid_list(gb)
lfn_list = gfo.local_fracture_network__grid_list(gb)
edge_list = gfo.edge_list(gb)
frac_edge_list = gfo.fracture_edge_list(gb)

ghost_grid_list = gfo.grid_list(ghost_gb)
ghost_bulk_list = gfo.bulk_grid_list(ghost_gb)
ghost_frac_list = gfo.fracture_grid_list(ghost_gb)
ghost_lfn_list = gfo.local_fracture_network__grid_list(ghost_gb)
ghost_edge_list = gfo.edge_list(ghost_gb)
ghost_frac_edge_list = gfo.fracture_edge_list(ghost_gb)

# Uncomment to export grid
# export_mesh = pp.Exporter(ghost_gb, file_name="grid", folder_name="out")
# export_mesh.write_vtu(ghost_gb)

# %% Time parameters
schedule = list(np.linspace(0, 300, 5, dtype=np.int32))
tsc = pp.TimeSteppingControl(
    schedule=schedule,
    dt_init=1.0,
    dt_min_max=(0.01, 0.25 * pp.HOUR),
    iter_max=100,
    iter_optimal_range=(4, 7),
    iter_lowupp_factor=(1.3, 0.7),
    recomp_factor=0.5,
    recomp_max=12,
    print_info=True,
)
times = [tsc.time]
dts = [tsc.dt]
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

gb.pressure_threshold = -22.1  # [cm]
for g, d in gb:

    # Set parameters for the bulk
    if g.dim == gb.dim_max():
        # For convinience, store values of bounding box
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
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=list(bc_type))
        bc_values = np.zeros(g.num_faces)
        bc_values[top_left] = -15 + y_max  # -15 (pressure_head) + y_max (elevation_head)

        # Saturatted hydraulic conductivity
        K_SAT: np.ndarray = 0.00922 * np.ones(g.num_cells)  # conductive bulk cells
        K_SAT[mult_cond] = 5.55E-6  # hydraulic conductivity of blocking cells

        # Initialize bulk data
        specified_parameters: dict = {
            "second_order_tensor": pp.SecondOrderTensor(K_SAT),  # [cm/s]
            "bc": bc,
            "bc_values": bc_values,
            "source": np.zeros(g.num_cells),
            "elevation": g.cell_centers[gb.dim_max() - 1],
            "mass_weight": np.ones(g.num_cells),
            "theta_r": 0.102,  # residual water content [-]
            "theta_s": 0.368,  # saturated water content [-]
            "alpha_vG": 0.0335,  # van Genuchten parameter [1/cm]
            "n_vG": 2.0,  # van Genuchten parameter [-]
            "m_vG": 0.5,  # (1 - 1 / n_vG) van Genuchten parameter [-]
            "time_step": tsc.dt,  # [s]
        }

    # Set parameters for 1D fractures
    elif g.dim == gb.dim_max() - 1:
        # Note that boundary values are not effectively used, but for discretization purposes
        # we still have to declare the fields
        bc_faces = g.get_boundary_faces()
        bc_type = bc_faces.size * ["neu"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        specified_parameters = {
            "aperture": 0.1,
            "bc": bc,
            "bc_values": np.zeros(g.num_faces),
            "datum": np.min(g.face_centers[gb.dim_max() - 1]),
            "elevation": g.cell_centers[gb.dim_max() - 1],
            "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells))
        }
        pp.initialize_data(g, d, param_key, specified_parameters)

    # Set parameters for 0D points
    else:
        specified_parameters = {
            "aperture": 0.02,
            "datum": np.min(g.cell_centers[gb.dim_max() - 1]),
            "elevation": g.cell_centers[gb.dim_max() - 1],
            "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells))
        }

    pp.initialize_data(g, d, param_key, specified_parameters)

# Parameters for the mortar
for e, d in gb.edges():
    mg = d["mortar_grid"]
    ones = np.ones(mg.num_cells)
    zeros = np.zeros(mg.num_cells)
    g_sec, _ = gb.nodes_of_edge(e)
    d_sec = gb.node_props(g_sec)
    time_scale_factor = 1E10
    aperture = d_sec[pp.PARAMETERS][param_key]["aperture"]
    sat_conductivity = 0.00922  # [cm/s]
    sat_normal_diffusivity = (sat_conductivity / (2 * aperture)) * ones  # [1/s]
    is_conductive = zeros
    if mg.dim == gb.dim_max() - 1:
        data = {
            "sat_normal_diffusivity": sat_normal_diffusivity,
            "normal_diffusivity": sat_normal_diffusivity,
            "is_conductive": is_conductive,
            "elevation": mg.cell_centers[gb.dim_max() - 1],
        }
    else:
        data = {
            "sat_normal_diffusivity": sat_normal_diffusivity * time_scale_factor,
            "normal_diffusivity": sat_normal_diffusivity * time_scale_factor,
            "is_conductive": is_conductive,
            "elevation": mg.cell_centers[gb.dim_max() - 1],
        }
    pp.initialize_data(mg, d, param_key, data)

# %% Set initial states
for g, d in gb:
    if g.dim == gb.dim_max():
        pp.set_state(d, state={node_var: -30 + d[pp.PARAMETERS][param_key]["elevation"]})
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})
    else:
        pp.set_state(d, state={node_var: np.array([gb.pressure_threshold + d[pp.PARAMETERS][
            param_key]["datum"]])})
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
# h_bulk = equation_manager.merge_variables([(g, node_var) for g in bulk_list])
# h_frac = equation_manager.merge_variables([(g, node_var) for g in lfn_list])
lmbda = equation_manager.merge_variables([(e, edge_var) for e in edge_list])

# Shorthands. Note that the following merged variables all have different id's
h_m = h.previous_iteration()
h_n = h.previous_timestep()
# h_bulk_m = h_bulk.previous_iteration()
# h_bulk_n = h_bulk.previous_timestep()
# h_frac_m = h_frac.previous_iteration()
# h_frac_n = h_frac.previous_timestep()
lmbda_m = lmbda.previous_iteration()
lmbda_n = lmbda.previous_timestep()

# Useful auxiliary variables
zeta = pp.ad.ParameterArray(
    param_keyword=param_key, array_keyword="elevation", grids=grid_list, name="elevation"
)
psi = h - zeta  # pressure head ad
psi_m = h_m - zeta  # pressure head previous iteration
psi_n = h_n - zeta  # pressure head previous time step

# zc_bulk_ad = pp.ad.Array(bulk_list[0].cell_centers[gb.dim_max() - 1])
# psib: pp.ad.Operator = h_bulk - zc_bulk_ad  # pressure head (active)
# psib_m: pp.ad.Operator = h_bulk_m - zc_bulk_ad  # pressure head at prev iter
# psib_n: pp.ad.Operator = h_bulk_n - zc_bulk_ad  # pressure head at prev time

# %% Grid operators and cell restrictions

# Grid operators
div_bulk = pp.ad.Divergence(grids=bulk_list)
bound_bulk = pp.ad.BoundaryCondition(param_key, grids=bulk_list)

proj = pp.ad.MortarProjections(gb=gb, grids=grid_list, edges=edge_list)
ghost_proj = pp.ad.MortarProjections(gb=ghost_gb, grids=ghost_grid_list, edges=ghost_edge_list)

subdomain_proj_scalar = pp.ad.SubdomainProjections(grids=grid_list)
bulk_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(bulk_list)
bulk_face_rest: pp.ad.Matrix = subdomain_proj_scalar.face_restriction(bulk_list)
bulk_face_prol: pp.ad.Matrix = subdomain_proj_scalar.face_prolongation(bulk_list)
frac_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(lfn_list)
frac_cell_prol: pp.ad.Matrix = subdomain_proj_scalar.cell_prolongation(lfn_list)

# %% Governing equations in the bulk

# Soil water retention curves
vgm = mdu.VanGenuchtenMualem(gb=gb, param_key=param_key)
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

# flux_single_phase_bulk: pp.ad.Operator = (
#         mpfa_bulk.flux * h_bulk_m
#         + mpfa_bulk.bound_flux * bound_bulk
#         + mpfa_bulk.bound_flux
#         * bulk_face_rest
#         * proj.mortar_to_primary_int
#         * lmbda_m
#)

# Upwinding of relative permeabilities
upwind = mdu.FluxBaseUpwindAd(gb=gb, grid_list=bulk_list, param_key=param_key)

zeta_faces_bulk = pp.ad.Array(bulk_list[0].face_centers[gb.dim_max() - 1])
psi_bound_bulk = bound_bulk - zeta_faces_bulk
psi_bulk_m = bulk_cell_rest * psi_m
krw_faces = upwind(krw_ad(psi_bulk_m), krw_ad(psi_bound_bulk), flux_single_phase)

# zf_bulk_ad = pp.ad.Array(bulk_list[0].face_centers[gb.dim_max() - 1])
# psi_bc_ad = bound_bulk - zf_bulk_ad
# krw_faces_ad: pp.ad.Operator = upwind(
#     krw_ad(psib_m), krw_ad(psi_bc_ad), flux_single_phase_bulk
# )

# Multiphase Darcy fluxes
flux = (
    krw_faces * mpfa_bulk.flux * bulk_cell_rest * h
    + krw_faces * mpfa_bulk.bound_flux * bound_bulk
    + krw_faces * mpfa_bulk.bound_flux * bulk_face_rest * proj.mortar_to_primary_int * lmbda
)

# flux_bulk: pp.ad.Operator = (
#         krw_faces_ad * mpfa_bulk.flux * h_bulk
#         + krw_faces_ad * mpfa_bulk.bound_flux * bound_bulk
#         + krw_faces_ad
#         * mpfa_bulk.bound_flux
#         * bulk_face_rest
#         * proj.mortar_to_primary_int
#         * lmbda
# )

# Treatment of source and accumulation terms
# NOTE: The expression containing the active ad variable (psi_bulk)
# must be placed at the left of the (final) expression.
# Otherwise, numpy will broadcast the expression leading to an undesirable output format
dt_ad = mdu.ParameterScalar(param_key, "time_step", grids=bulk_list)
source_bulk = pp.ad.ParameterArray(param_key, "source", grids=bulk_list)
mass_bulk = pp.ad.MassMatrixAd(param_key, grids=bulk_list)

linearization = "modified_picard"  # linearization of the bulk equations
# if linearization == "newton":
#     accum_bulk_active = mass_bulk.mass * theta_ad(psib)
#     accum_bulk_inactive = mass_bulk.mass * theta_ad(psib_n) * (-1)
# elif linearization == "modified_picard":
#     accum_bulk_active = mass_bulk.mass * psib * smc_ad(psib_m)
#     accum_bulk_inactive = mass_bulk.mass * (
#             theta_ad(psib_m) - smc_ad(psib_m) * psib_m - theta_ad(psib_n)
#     )
# elif linearization == "l_scheme":
#     L = 0.0025
#     accum_bulk_active = L * mass_bulk.mass * psib
#     accum_bulk_inactive = mass_bulk.mass * (
#             theta_ad(psib_m) - L * psib_m - theta_ad(psib_n)
#     )
# else:
#     raise NotImplementedError(
#         "Linearization scheme not implemented. Use 'newton', "
#         "'modified_picard', or 'l_scheme'."
#     )
#
# accumulation_bulk = accum_bulk_active + accum_bulk_inactive
# conserv_bulk_eq = accumulation_bulk + dt_ad * div_bulk * flux_bulk - dt_ad * source_bulk
# conserv_bulk_eq.discretize(gb=gb)
# conserv_bulk_num = conserv_bulk_eq.evaluate(dof_manager=dof_manager).val

if linearization == "newton":
    accum_bulk_active = mass_bulk.mass * theta_ad(bulk_cell_rest * psi)
    accum_bulk_inactive = mass_bulk.mass * theta_ad(bulk_cell_rest * psi_n) * (-1)
elif linearization == "modified_picard":
    accum_bulk_active = mass_bulk.mass * (
            (bulk_cell_rest * psi) * smc_ad(bulk_cell_rest * psi_m)
    )
    accum_bulk_inactive = mass_bulk.mass * (
            theta_ad(bulk_cell_rest * psi_m) - smc_ad(bulk_cell_rest * psi_m) *
            (bulk_cell_rest * psi_m) - theta_ad(bulk_cell_rest * psi_n)
    )
elif linearization == "l_scheme":
    L = 0.0025
    accum_bulk_active = L * mass_bulk.mass * (bulk_cell_rest * psi)
    accum_bulk_inactive = mass_bulk.mass * (
            theta_ad(bulk_cell_rest * psi_m)
            - L * (bulk_cell_rest * psi_m) - theta_ad(bulk_cell_rest * psi_n)
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

# print(np.linalg.norm(conserv_bulk_num_new - conserv_bulk_num))
# print()

# %% Governing equations in the fracture

# Get water volume as a function of the hydraulic head, and its first derivative
fv = mdu.FractureVolume(gb=gb, fracture_grids=lfn_list, param_key=param_key)
vol_ad: pp.ad.Function = fv.fracture_volume(as_ad=True)
vol_cap_ad: pp.ad.Function = fv.volume_capacity(as_ad=True)
vol = fv.fracture_volume(as_ad=False)

linearization = "newton"  # linearization of the fracture equations
# if linearization == "newton":
#     accum_frac_active = vol_ad(h_frac)
#     accum_frac_inactive = vol_ad(h_frac_n) * (-1)
# elif linearization == "modified_picard":
#     accum_frac_active = h_frac * vol_cap_ad(h_frac_m)
#     accum_frac_inactive = vol_ad(h_frac_m) - vol_cap_ad(h_frac_m) * h_frac_m - vol_ad(h_frac_n)
# elif linearization == "l_scheme":
#     L = 0.015
#     accum_frac_active = L * h_frac
#     accum_frac_inactive = vol_ad(h_frac_m) - L * h_frac_m - vol_ad(h_frac_n)
# else:
#     raise NotImplementedError(
#         "Linearization scheme not implemented. Use 'newton', "
#         "'modified_picard', or 'l_scheme'."
#     )
#
# # Retrieve sources from mortar
# sources_from_mortar = frac_cell_rest * proj.mortar_to_secondary_int * lmbda
# # Accumulation terms
# accum_frac = accum_frac_active + accum_frac_inactive
# # Declare conservation equation
# conserv_frac_eq = accum_frac - dt_ad * sources_from_mortar
# conserv_frac_eq.discretize(gb=gb)
# conserv_frac_num = conserv_frac_eq.evaluate(dof_manager=dof_manager).val

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
conserv_frac_eq = accum_frac - dt_ad * sources_from_mortar
conserv_frac_eq.discretize(gb=gb)
conserv_frac_num_new = conserv_frac_eq.evaluate(dof_manager=dof_manager).val

# print(np.linalg.norm(conserv_frac_num_new - conserv_frac_num))
# print()

# %% Governing equations on the interfaces
mpfa_global = pp.ad.MpfaAd(param_key, grid_list)
robin = pp.ad.RobinCouplingAd(param_key, edge_list)

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

# # Projected bulk pressure traces onto the mortar grid
# proj_tr_h_bulk = (
#         proj.primary_to_mortar_avg
#         * bulk_face_prol
#         * mpfa_bulk.bound_pressure_cell
#         * h_bulk
#         + proj.primary_to_mortar_avg
#         * bulk_face_prol
#         * mpfa_bulk.bound_pressure_face
#         * bulk_face_rest
#         * proj.mortar_to_primary_int
#         * lmbda
# )
#
# proj_tr_h_bulk_m = (
#         proj.primary_to_mortar_avg
#         * bulk_face_prol
#         * mpfa_bulk.bound_pressure_cell
#         * h_bulk_m
#         + proj.primary_to_mortar_avg
#         * bulk_face_prol
#         * mpfa_bulk.bound_pressure_face
#         * bulk_face_rest
#         * proj.mortar_to_primary_int
#         * lmbda_m
# )

# Get projected ghost fracture hydraulic head onto the adjacent mortar grids
pfh = mdu.GhostHydraulicHead(gb=gb, ghost_gb=ghost_gb)
frac_to_mortar_ad: pp.ad.Function = pfh.proj_fra_hyd_head(as_ad=True)
# proj_h_frac = frac_to_mortar_ad(h_frac)
proj_h_low = frac_to_mortar_ad(frac_cell_rest * h)

# Array parameter that keeps track of conductive (1) and blocking (0) mortar cells
# Note that if it is blocking, the whole discrete equation is removed for that mortar cell
is_conductive = pp.ad.ParameterArray(param_key, "is_conductive", edges=edge_list)

# # Interface flux
# mortar_flux = robin.mortar_discr * (proj_tr_h_bulk - proj_h_frac) * is_conductive
# interface_flux_eq = mortar_flux + lmbda
# interface_flux_eq.discretize(gb=gb)
# interface_flux_num = interface_flux_eq.evaluate(dof_manager=dof_manager).val

# Interface flux
mortar_flux = robin.mortar_discr * (proj_h_low - proj_h_high) * is_conductive
interface_flux_eq = mortar_flux + lmbda
interface_flux_eq.discretize(gb=gb)
interface_flux_num_new = interface_flux_eq.evaluate(dof_manager=dof_manager).val

# %% Assemble discrete equations, feed into the equation manager, and discretize.
eqs = {"bulk_conservation": conserv_bulk_eq,
       "fracture_conservation": conserv_frac_eq,
       "interface_fluxes": interface_flux_eq
       }
equation_manager.equations.update(eqs)
equation_manager.discretize(gb)

# %% Initialize exporter. Note that we use the ghost grid rather than the physical one
# d_bulk = gb.node_props(bulk_list[0])
# d_bulk_ghost = ghost_gb.node_props(ghost_bulk_list[0])
# z_bulk = bulk_list[0].cell_centers[1]
#
# d_frac_bottom = gb.node_props(frac_list[0])
# d_frac_ghost_bottom = ghost_gb.node_props(ghost_frac_list[0])
# z_frac_bottom = frac_list[0].cell_centers[1]
# z_frac_ghost_bottom = ghost_frac_list[0].cell_centers[1]
#
# d_frac_top = gb.node_props(frac_list[1])
# d_frac_ghost_top = ghost_gb.node_props(ghost_frac_list[1])
# z_frac_top = frac_list[1].cell_centers[1]
# z_frac_ghost_top = ghost_frac_list[1].cell_centers[1]
#
# d_point = gb.node_props(lfn_list[-1])
# d_point_ghost = ghost_gb.node_props(ghost_lfn_list[-1])
#
# d_edge_bottom = gb.edge_props(edge_list[0])
# d_edge_ghost_bottom = ghost_gb.edge_props(ghost_edge_list[0])
#
# d_edge_top = gb.edge_props(edge_list[1])
# d_edge_ghost_top = ghost_gb.edge_props(ghost_edge_list[1])
#
# # # Set state in bulk ghost grid
# pp.set_state(
#     data=d_bulk_ghost,
#     state={
#         node_var: d_bulk[pp.STATE][node_var],
#     },
# )
#
# # Set state in fracture ghost grids
# pp.set_state(
#     data=d_frac_ghost_bottom,
#     state={
#         node_var: d_frac_bottom[pp.PARAMETERS][param_key]["datum"]
#         * np.ones(ghost_frac_list[0].num_cells),
#     },
# )
# pp.set_state(
#     data=d_frac_ghost_top,
#     state={
#         node_var: d_frac_top[pp.PARAMETERS][param_key]["datum"]
#         * np.ones(ghost_frac_list[1].num_cells),
#     },
# )
#
# pp.set_state(
#     data=d_point_ghost,
#     state={
#         node_var: 0
#     }
#
# )
#
#
# # Set state in edges
# pp.set_state(
#     data=d_edge_ghost_top,
#     state={
#         edge_var: d_edge_top[pp.STATE][edge_var]
#     }
# )
#
# pp.set_state(
#     data=d_edge_ghost_bottom,
#     state={
#         edge_var: d_edge_bottom[pp.STATE][edge_var]
#     }
# )
#
#
# # # Correct values of pressure head in the fracture if negative
# # for val in d_frac_ghost[pp.STATE]["pressure_head"] <= 0:
# #     d_frac_ghost[pp.STATE]["pressure_head"][val] = 0
#
# water_table_bottom = [d_frac_bottom[pp.STATE][node_var][0] - pressure_threshold]
# water_table_top = [d_frac_top[pp.STATE][node_var][0] - pressure_threshold]
# water_vol_bottom = [vol(d_frac_bottom[pp.STATE][node_var])[0]]
# water_vol_top = [vol(d_frac_top[pp.STATE][node_var])[0]]
#exporter_ghost = pp.Exporter(ghost_gb, "morse_code", "out")
#exporter_ghost.write_vtu([node_var], time_step=0)

# %% Time loop
total_iteration_counter = 0
iters = []
ABS_TOL = 1E-3
is_mortar_conductive = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
control_faces = is_mortar_conductive

# Time loop
while tsc.time < tsc.time_final:
    tsc.time += tsc.dt
    itr: int = 0
    res_norm: float = 1E8
    rel_res: float = 1E8

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
        print("t", tsc.time, "itr", itr, "res", res_norm, "rel_res", rel_res, "dt", tsc.dt)

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
        gb, dof_manager, param_key, proj_h_high, proj_h_low, edge_list
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
    # end of checking

    # Save number of iterations and time step
    iters.append(itr - 1)
    times.append(tsc.time)
    dts.append(tsc.dt)

    # Succesful convergence
    tsc.next_time_step(recompute_solution=False, iterations=itr - 1)
    param_update.update_time_step(tsc.dt)
    # print(f"Fracture water table height: {d_frac[pp.STATE][node_var][0] - pressure_threshold}")
    # water_table.append(d_frac[pp.STATE][node_var][0] - pressure_threshold)
    # water_vol.append(vol(d_frac[pp.STATE][node_var]))
    set_state_as_iterate(gb, node_var, edge_var)
    vols = vol((frac_cell_rest * h).evaluate(dof_manager).val)
    print(f"Top Fracture water volume: {vols[1]}")
    print(f"Point water volume: {vols[2]}")
    print(f"Bottom Fracture water volume: {vols[0]}")
    print()

    # # Export to ParaView
    # pp.set_state(
    #     data=d_bulk_ghost,
    #     state={node_var: d_bulk[pp.STATE][node_var]},
    #)
    # pp.set_state(
    #     data=d_frac_ghost,
    #     state={
    #         node_var: get_ghost_hydraulic_head(ghost_frac_list[0], d_frac[pp.STATE][node_var]),
    #         "pressure_head": get_ghost_hydraulic_head(
    #             ghost_frac_list[0], d_frac[pp.STATE][node_var]) - z_frac_ghost,
    #     },
    # )
    # pp.set_state(
    #     data=d_edge_ghost,
    #     state={
    #         edge_var: d_edge[pp.STATE][edge_var]
    #     }
    # )
    # # Correct values of pressure head in the fracture if negative
    # for val in d_frac_ghost[pp.STATE]["pressure_head"] <= 0:
    #     d_frac_ghost[pp.STATE]["pressure_head"][val] = 0
    # if tsc.time in tsc.schedule:
    #     export_counter += 1
    #     exporter_ghost.write_vtu([node_var], time_step=export_counter)

#%% Export results
# iters.insert(0, 0)  # insert 0 at the beginning of the list
# d = dict()
# d["time"] = np.array(times) / 3600  # time in [hours]
# d["water_volume"] = np.array(water_vol)  # water volume in [cm^3]
# d["water_table"] = np.array(water_table)  # fracture water table in [cm]
# d["time_step"] = np.array(dts)  # time steps in [s]
# d["iterations"] = np.array(iters)  # iterations
#
# file_name = "out.plk"
# open_file = open(file_name, "wb")
# pickle.dump(d, open_file)
# open_file.close()
