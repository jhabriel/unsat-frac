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
    mesh_args={"mesh_size_frac": 4, "mesh_size_bound": 5},
    csv_file="network.csv",
    domain={"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
    constraints=[1, 2, 3, 4],
)
_, gb = gfo.get_grid_buckets()

grid_list = gfo.get_grid_list(gb)
bulk_list = gfo.get_bulk_list(gb)
frac_list = gfo.get_fracture_list(gb)
edge_list = gfo.get_edge_list(gb)

# ghost_grid_list = gfo.get_grid_list(ghost_gb)
# ghost_bulk_list = gfo.get_bulk_list(ghost_gb)
# ghost_frac_list = gfo.get_fracture_list(ghost_gb)
# ghost_edge_list = gfo.get_edge_list(ghost_gb)

# Uncomment to export grid
# export_mesh = pp.Exporter(ghost_gb, file_name="grid", folder_name="out")
# export_mesh.write_vtu(ghost_gb)

# %% Time parameters
schedule = list(np.linspace(0, 48*pp.HOUR, 40, dtype=np.int32))
tsc = pp.TimeSteppingControl(
    schedule=schedule,
    dt_init=100,
    dt_min_max=(10, 1 * pp.HOUR),
    iter_max=13,
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

pressure_threshold = -22.1  # [cm]
soil_hydraulic_conductivity = 0.00922  # [cm/s]
fracture_hydraulic_conductivity = 1E3 * soil_hydraulic_conductivity  # [cm/s]

for g, d in gb:
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
        bc: pp.BoundaryCondition = pp.BoundaryCondition(
            g, faces=bc_faces, cond=list(bc_type)
        )
        bc_values: np.ndarray = np.zeros(g.num_faces)
        bc_values[top_left] = -75 + y_max  # -15 (pressure_head) + y_max (elevation_head)

        # Hydraulic conductivity
        K_SAT: np.ndarray = 0.00922 * np.ones(g.num_cells)  # conductive bulk cells
        K_SAT[mult_cond] = 5.55E-11  # hydraulic conductivity of blocking cells

        # Initialize bulk data
        ones = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        specified_parameters: dict = {
            "second_order_tensor": pp.SecondOrderTensor(K_SAT),  # [cm/s]
            "bc": bc,
            "bc_values": bc_values,
            "source": zeros,
            "zeta": g.cell_centers[gb.dim_max() - 1],
            "zetaf": g.face_centers[gb.dim_max() - 1],
            "mass_weight": ones,
            "theta_r": 0.102,  # residual water content [-]
            "theta_s": 0.368,  # saturated water content [-]
            "alpha_vg": 0.0335,  # van Genuchten parameter [1/cm]
            "n_vg": 2.0,  # van Genuchten parameter [-]
            "m_vg": 0.5,  # (1 - 1 / n_vG) van Genuchten parameter [-]
            "time_step": tsc.dt,  # [s]
            "darcy_flux": np.zeros(g.num_faces),
        }
        pp.initialize_data(g, d, param_key, specified_parameters)

    else:
        ones = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        perm = pp.SecondOrderTensor(fracture_hydraulic_conductivity * ones)
        # No-flow Neumann conditions
        bc = pp.BoundaryCondition(g)
        bc_val = np.zeros(g.num_faces)
        specified_data = {'second_order_tensor': perm,
                          'bc': bc,
                          'bc_values': bc_val,
                          "aperture": 0.1,
                          "datum": 10,
                          "zeta": g.cell_centers[gb.dim_max() - 1],
                          "zetaf": g.face_centers[gb.dim_max() - 1],
                          "source": np.zeros(g.num_cells),
                          "mass_weight": ones,
                          "theta_r": 0.102,  # residual water content [-]
                          "theta_s": 0.368,  # saturated water content [-]
                          "alpha_vg": 0.0335,  # van Genuchten parameter [1/cm]
                          "n_vg": 2.0,  # van Genuchten parameter [-]
                          "m_vg": 0.5,  # (1 - 1 / n_vG) van Genuchten parameter [-]
                          "darcy_flux": np.zeros(g.num_faces),
                          }

        # # Parameters for the fracture grids
        # specified_parameters = {
        #     "aperture": 0.1,
        #     "datum": np.min(g.face_centers[gb.dim_max() - 1]),
        #     "elevation": g.cell_centers[gb.dim_max() - 1],
        # }
        pp.initialize_data(g, d, param_key, specified_data)

# Parameters for the mortar
for e, d in gb.edges():
    mg = d["mortar_grid"]
    ones = np.ones(mg.num_cells)
    zeros = np.zeros(mg.num_cells)
    g_sec, _ = gb.nodes_of_edge(e)
    d_sec = gb.node_props(g_sec)
    aperture = d_sec[pp.PARAMETERS][param_key]["aperture"]
    sat_conductivity = 0.00922  # [cm/s]
    sat_normal_diffusivity = (sat_conductivity / (2 * aperture)) * ones  # [1/s]
    is_conductive = zeros
    data = {
        "sat_normal_diffusivity": sat_normal_diffusivity,
        "normal_diffusivity": sat_normal_diffusivity,
        "is_conductive": is_conductive,
        "elevation": mg.cell_centers[gb.dim_max() - 1],
    }
    pp.initialize_data(mg, d, param_key, data)

# %% Set initial states
for g, d in gb:
    if g.dim == gfo.dim:
        pp.set_state(d, state={node_var: -500 + d[pp.PARAMETERS][param_key]["zeta"]})
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})
    else:
        pp.set_state(d, state={node_var: -900 * np.ones(g.num_cells)})
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})

for _, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d, state={edge_var: np.zeros(mg.num_cells)})
    pp.set_iterate(d, iterate={edge_var: d[pp.STATE][edge_var]})

# %% Create degree of freedom manager and equation manager
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

# %%
div = pp.ad.Divergence(grids=grid_list)
mortar_proj = pp.ad.MortarProjections(gb=gb, grids=grid_list, edges=edge_list)
bound_ad = pp.ad.BoundaryCondition(param_key, grids=grid_list)

# %% Assign primary variables to their corresponding grids
h = equation_manager.merge_variables([(g, node_var) for g in grid_list])
lmbda = equation_manager.merge_variables([(e, edge_var) for e in edge_list])

h_m = h.previous_iteration()
h_n = h.previous_timestep()
lmbda_m = lmbda.previous_iteration()
lmbda_n = lmbda.previous_timestep()

zeta = pp.ad.ParameterArray(param_key, "zeta", grid_list)  # elevation head

psi = h - zeta
psi_m = h_m - zeta
psi_n = h_n - zeta

# %% Equations on subdomains -> Richards

# Soil water retention curves
swrc = mdu.SWRC(gb=gb, grid_list=grid_list, param_key=param_key)
theta_ad: pp.ad.Function = swrc.water_content()
krw_ad: pp.ad.Function = swrc.relative_permeability()
krw_fc_ad: pp.ad.Function = swrc.relative_permeability(cc=False)
smc_ad: pp.ad.Function = swrc.moisture_capacity()

# Single phase flux
mpfa = pp.ad.MpfaAd(param_key, grid_list)
single_phase_flux = (
        mpfa.flux * h
        + mpfa.bound_flux * bound_ad
        + mpfa.bound_flux * mortar_proj.mortar_to_primary_int * lmbda
)

#upwind = mdu.FluxBaseUpwindAd(gb=gb, grid_list=bulk_list, param_key=param_key)
upwind = pp.ad.UpwindAd(keyword=param_key, grids=grid_list)
zetaf = pp.ad.ParameterArray(param_key, "zetaf", grid_list)
krw_face = (
        upwind.upwind * krw_ad(psi_m)
        - upwind.bound_transport_dir * krw_fc_ad(bound_ad - zetaf)
)
flux = (
        krw_face * (mpfa.flux * h)
        + krw_face * (mpfa.bound_flux * bound_ad)
        + krw_face * (mpfa.bound_flux * mortar_proj.mortar_to_primary_int * lmbda)
)

# Treatment of source and accumulation terms
# NOTE: The expression containing the active ad variable (psi_bulk)
# must be placed at the left of the (final) expression.
# Otherwise, numpy will broadcast the expression leading to an undesirable output format
dt_ad = mdu.ParameterScalar(param_key, "time_step", grids=bulk_list)
source = pp.ad.ParameterArray(param_key, "source", grids=grid_list)
mass = pp.ad.MassMatrixAd(param_key, grids=grid_list)
sources_from_mortar = mortar_proj.mortar_to_secondary_int * lmbda


linearization = "newton"  # linearization of the bulk equations
if linearization == "newton":
    accum_active = mass.mass * theta_ad(psi)
    accum_inactive = mass.mass * theta_ad(psi_n) * (-1)
elif linearization == "modified_picard":
    accum_active = mass.mass * psi * smc_ad(psi_m)
    accum_inactive = mass.mass * (
            theta_ad(psi_m) - smc_ad(psi_m) * psi_m - theta_ad(psi_n)
    )
elif linearization == "l_scheme":
    L = 0.0025
    accum_active = L * mass.mass * psi
    accum_inactive = mass.mass * (
            theta_ad(psi_m) - L * psi_m - theta_ad(psi_n)
    )
else:
    raise NotImplementedError(
        "Linearization scheme not implemented. Use 'newton', "
        "'modified_picard', or 'l_scheme'."
    )

accumulation = accum_active + accum_inactive
conserv_eq = accumulation + dt_ad * div * flux + dt_ad * sources_from_mortar - dt_ad * source

# Discretize and evaluate
conserv_eq.discretize(gb=gb)
conserv_num = conserv_eq.evaluate(dof_manager=dof_manager)

# %% Governing equations on the interfaces
robin = pp.ad.RobinCouplingAd(param_key, edge_list)

# Projected bulk pressure traces onto the mortar grid
h_high = (
        mortar_proj.primary_to_mortar_avg * mpfa.bound_pressure_cell * h
        + mortar_proj.primary_to_mortar_avg
        * mpfa.bound_pressure_face
        * mortar_proj.mortar_to_primary_int
        * lmbda
)

h_high_m = (
        mortar_proj.primary_to_mortar_avg * mpfa.bound_pressure_cell * h_m
        + mortar_proj.primary_to_mortar_avg
        * mpfa.bound_pressure_face
        * mortar_proj.mortar_to_primary_int
        * lmbda_m
)

h_low = mortar_proj.secondary_to_mortar_avg * h
h_low_m = mortar_proj.secondary_to_mortar_avg * h_m

# Array parameter that keeps track of conductive (1) and blocking (0) mortar cells
# Note that if it is blocking, the whole discrete equation is removed for that mortar cell
is_conductive = pp.ad.ParameterArray(param_key, "is_conductive", edges=edge_list)

# Interface flux
mortar_flux = robin.mortar_discr * (h_high - h_low) * is_conductive
interface_flux_eq = mortar_flux + lmbda
interface_flux_eq.discretize(gb=gb)
interface_flux_num = interface_flux_eq.evaluate(dof_manager=dof_manager)

# %% Assemble discrete equations, feed into the equation manager, and discretize.
eqs = {"conservation": conserv_eq,
       "interface_fluxes": interface_flux_eq,
       }
equation_manager.equations.update(eqs)
equation_manager.discretize(gb)

# # %% Initialize exporter. Note that we use the ghost grid rather than the physical one
# d_bulk = gb.node_props(bulk_list[0])
# d_bulk_ghost = ghost_gb.node_props(ghost_bulk_list[0])
# z_bulk = bulk_list[0].cell_centers[1]
#
# d_frac = gb.node_props(frac_list[0])
# d_frac_ghost = ghost_gb.node_props(ghost_frac_list[0])
# z_frac = frac_list[0].cell_centers[1]
# z_frac_ghost = ghost_frac_list[0].cell_centers[1]
#
# d_edge = gb.edge_props(edge_list[0])
# d_edge_ghost = ghost_gb.edge_props(ghost_edge_list[0])
#
# # Set state in bulk ghost grid
# pp.set_state(
#     data=d_bulk_ghost,
#     state={
#         node_var: d_bulk[pp.STATE][node_var],
#         "pressure_head": d_bulk[pp.STATE][node_var] - z_bulk,
#     },
# )
# # Pressure head is not exported correctly. Try with a different method. Perhaps, np.max(
# # el_actual, 0)*
# # Set state in fracture ghost grid
# pp.set_state(
#     data=d_frac_ghost,
#     state={
#         node_var: d_frac[pp.PARAMETERS][param_key]["datum"]
#                   * np.ones(ghost_frac_list[0].num_cells),
#         "pressure_head": d_frac[pp.STATE][node_var] - z_frac_ghost,
#     },
# )
# # Set state in edges
# pp.set_state(
#     data=d_edge_ghost,
#     state={
#         edge_var: d_edge[pp.STATE][edge_var]
#     }
# )
#
# # Correct values of pressure head in the fracture if negative
# for val in d_frac_ghost[pp.STATE]["pressure_head"] <= 0:
#     d_frac_ghost[pp.STATE]["pressure_head"][val] = 0
#
# water_table = [d_frac[pp.STATE][node_var][0] - pressure_threshold]
# water_vol = [vol(d_frac[pp.STATE][node_var])[0]]
exporter = pp.Exporter(gb, "full_richards", "out")
exporter.write_vtu([node_var, "pressure_head", edge_var], time_step=0)

# %% Time loop
total_iteration_counter: int = 0
iters: list = []
abs_tol: float = 1e-7
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
        pp.fvutils.compute_darcy_flux(gb,
                                      keyword_store=param_key,
                                      lam_name=edge_var,
                                      p_name=node_var,
                                      from_iterate=True)
        equation_manager.discretize(gb)

        # Solve system of equations and distribute variables to pp.ITERATE
        A, b = equation_manager.assemble()
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
    # if np.any(vol(h_frac.evaluate(dof_manager).val) < 0):
    #     tsc.next_time_step(recompute_solution=True, iterations=iteration_counter - 1)
    #     param_update.update_time_step(tsc.dt)
    #     print(f"Encountered negative volume. Reducing dt and recomputing solution.")
    #     set_iterate_as_state(gb, node_var, edge_var)
    #     continue

    # # Recompute solution is capillary barrier is overcome. Note that dt remains the same
    # is_mortar_conductive = get_conductive_mortars(
    #     gb, dof_manager, param_key, proj_tr_h_bulk, proj_h_frac, edge_list
    # )
    # if control_faces.sum() == 0 and is_mortar_conductive.sum() > 0:
    #     param_update.update_mortar_conductivity_state(is_mortar_conductive, edge_list)
    #     print(
    #         f"The faces {np.where(is_mortar_conductive)[0]} are saturated. "
    #         f"Solution will be recomputed."
    #     )
    #     control_faces = is_mortar_conductive
    #     set_iterate_as_state(gb, node_var, edge_var)
    #     tsc.time -= tsc.dt  # correct time since we are going to recompute the solution
    #     continue
    # else:
    #     is_mortar_conductive: np.ndarray = np.zeros(
    #         gb.num_mortar_cells(), dtype=np.int8
    #     )
    #     param_update.update_mortar_conductivity_state(is_mortar_conductive, edge_list)
    #     control_faces = is_mortar_conductive

    # Save number of iterations and time step
    iters.append(iteration_counter - 1)
    times.append(tsc.time)
    dts.append(tsc.dt)

    # Succesful convergence
    tsc.next_time_step(recompute_solution=False, iterations=iteration_counter - 1)
    param_update.update_time_step(tsc.dt)
    set_state_as_iterate(gb, node_var, edge_var)
    print()

    # Export to ParaView
    if tsc.time in tsc.schedule:
        export_counter += 1
        exporter.write_vtu([node_var, "pressure_head", edge_var], time_step=export_counter)

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
