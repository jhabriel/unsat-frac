# %% Import modules
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sps
import matplotlib.pyplot as plt
from typing import List
from matplotlib.pyplot import spy as sparsity
import mdunsat as mdu

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

# %% Global simulation parameters
blocking_block: bool = False
linearization: str = "newton"

# %% Make grid
if not blocking_block:
    gfo = mdu.GridFactory(
        dim=2,
        mesh_args={"mesh_size_frac": 5, "mesh_size_bound": 5},
        csv_file="morse_code.csv",
        domain={"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
    )
else:
    gfo = mdu.GridFactory(
        dim=2,
        mesh_args={"mesh_size_frac": 5, "mesh_size_bound": 5},
        csv_file="net_with_block.csv",
        domain={"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
    )

gb, ghost_gb = gfo.get_grid_buckets()

bulk_list = gfo.get_bulk_list(gb)
frac_list = gfo.get_fracture_list(gb)
grid_list = gfo.get_grid_list(gb)
edge_list = gfo.get_edge_list(gb)

ghost_frac_list = gfo.get_fracture_list(ghost_gb)
ghost_grid_list = gfo.get_grid_list(ghost_gb)
ghost_edge_list = gfo.get_edge_list(ghost_gb)

# %% Time parameters
schedule = list(np.linspace(0, 4 * pp.HOUR, 60, dtype=np.int32))
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

# %% Parameter assignment

# Parameter keywords
kw: str = "flow"
node_var: str = "hydraulic_head"
edge_var: str = "mortar_flux"

# Declare primary variables
for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {node_var: {"cells": 1}}
for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {edge_var: {"cells": 1}}

# Parameter assignment
param_update = ParameterUpdate(gb, kw)  # object to update parameters

for g, d in gb:
    if g.dim == gfo.dim:  # the bulk
        # Face indices of the boundaries
        fc = g.face_centers
        top = np.where(np.abs(fc[1] - gfo.ymax) < 1e-5)[0]
        bottom = np.where(np.abs(fc[1]) < 1e-5)[0]
        left = np.where(np.abs(fc[0]) < 1e-5)[0]
        right = np.where(np.abs(fc[0] - gfo.xmax) < 1e-5)[0]
        top_left = top[np.where(np.abs(fc[0][top] < (gfo.xmax / 2)))[0]]
        # Left and right cell indices
        cc = g.cell_centers
        cells_left = np.where(np.abs(cc[0] < (gfo.xmax / 2)))[0]
        cells_right = np.where(np.abs(cc[0] > (gfo.xmax / 2)))[0]
        # Cell indices corresponding to the blocking fractures
        blocking_idx = [
            cc[0] < (0.50 * gfo.xmax),
            cc[0] > (0.25 * gfo.xmax),
            cc[1] < (0.80 * gfo.ymax),
            cc[1] > (0.60 * gfo.ymax),
        ]
        mult_cond = np.logical_and(blocking_idx[0], blocking_idx[1])
        mult_cond = np.logical_and(blocking_idx[2], mult_cond)
        mult_cond = np.logical_and(blocking_idx[3], mult_cond)
        blocking_cells = np.where(mult_cond)[0]
        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(bc_faces.size * ["neu"])
        bc_type[np.in1d(bc_faces, top_left)] = "dir"
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=list(bc_type))
        bc_values = np.zeros(g.num_faces)
        bc_values[top_left] = 0 + gfo.ymax  # 0 (pressure_head) + y_max (elevation_head)
        # Hydraulic conductivity
        K_sat = 0.00922 * np.ones(g.num_cells)  # conductive bulk cells
        if blocking_block:
            K_sat[mult_cond] = 1e-18  # hydraulic conductivity of blocking cells
        # Initialize bulk data
        specified_parameters = {
            "second_order_tensor": pp.SecondOrderTensor(K_sat),  # [cm/s]
            "bc": bc,
            "bc_values": bc_values,
            "source": np.zeros(g.num_cells),
            "elevation": g.cell_centers[gfo.dim - 1],
            "mass_weight": np.ones(g.num_cells),
            "theta_r": 0.102,  # residual water content [-]
            "theta_s": 0.368,  # saturated water content [-]
            "alpha_vG": 0.0335,  # van Genuchten parameter [1/cm]
            "n_vG": 2.0,  # van Genuchten parameter [-]
            "m_vG": 0.5,  # (1 - 1 / n_vG) van Genuchten parameter [-]
            "time_step": tsc.dt  # [s]
        }
    elif g.dim == (gfo.dim - 1):  # fractures
        specified_parameters = {
            "aperture": 0.1,  # constant for the moment
            "datum": np.min(g.face_centers[gfo.dim - 1]),
            "elevation": g.cell_centers[gfo.dim - 1],
        }
    else:  # fracture intersections and points
        specified_parameters = {
            "datum": np.min(g.cell_centers[gfo.dim - 1]),
            "elevation": g.cell_centers[gfo.dim - 1],
        }
    pp.initialize_data(g, d, kw, specified_parameters)

# Parameters for the mortar. Note that we are only interested in mortars that have
# dimensionality equal to the fractures
for e, d in gb.edges():
    mg = d["mortar_grid"]
    if mg.dim == (gfo.dim - 1):
        g_secondary, _ = gb.nodes_of_edge(e)
        d_secondary = gb.node_props(g_secondary)
        aperture = d_secondary[pp.PARAMETERS][kw]["aperture"]
        k_perp_val = 0.00922 / (2 * aperture)
        normal_diffusivity = k_perp_val * np.ones(mg.num_cells)
        is_conductive = np.zeros(mg.num_cells, dtype=np.int8)
        data = {
            "normal_diffusivity": normal_diffusivity,
            "is_conductive": is_conductive,
            "elevation": mg.cell_centers[gfo.dim - 1],
        }
    else:
        data = {}
    pp.initialize_data(mg, d, kw, data)

# %% Set initial states
for g, d in gb:
    if g.dim == gfo.dim:
        pp.set_state(d, state={node_var: -1000 + d[pp.PARAMETERS][kw]["elevation"]})
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})
    else:
        pp.set_state(d, state={node_var: np.array([0 + d[pp.PARAMETERS][kw]["datum"]])})
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})

for _, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d, state={edge_var: np.zeros(mg.num_cells)})
    pp.set_iterate(d, iterate={edge_var: d[pp.STATE][edge_var]})

# %% AD variables and manager

# Create degree of freedom manager and equation manager
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

# Assign primary variables to their corresponding grids
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

# %% AD operators and discrete expressions

# Grid operators
div_bulk = pp.ad.Divergence(grids=bulk_list)
bound_bulk = pp.ad.BoundaryCondition(kw, grids=bulk_list)
proj = pp.ad.MortarProjections(gb=gb, grids=grid_list, edges=edge_list)
ghost_proj = pp.ad.MortarProjections(gb=ghost_gb, grids=ghost_grid_list, edges=ghost_edge_list)
subdomain_proj_scalar = pp.ad.SubdomainProjections(grids=grid_list)
bulk_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(bulk_list)
bulk_face_rest: pp.ad.Matrix = subdomain_proj_scalar.face_restriction(bulk_list)
bulk_face_prol: pp.ad.Matrix = subdomain_proj_scalar.face_prolongation(bulk_list)
frac_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(frac_list)
frac_cell_prol: pp.ad.Matrix = subdomain_proj_scalar.cell_prolongation(frac_list)

# Soil water retention curves
vgm = mdu.VanGenuchtenMualem(gb=gb, param_key=kw)
theta_ad: pp.ad.Function = vgm.water_content(as_ad=True)
krw_ad: pp.ad.Function = vgm.relative_permeability(as_ad=True)
smc_ad: pp.ad.Function = vgm.moisture_capacity(as_ad=True)

# MPFA discretization
mpfa_bulk: pp.ad.MpfaAd = pp.ad.MpfaAd(kw, bulk_list)

# Obtain single phase flow to compute directionality of upwind scheme
flux1p_bulk: pp.ad.Operator = (
    mpfa_bulk.flux * h_bulk_m
    + mpfa_bulk.bound_flux * bound_bulk
    + mpfa_bulk.bound_flux
    * bulk_face_rest
    * proj.mortar_to_primary_int
    * lmbda_m
)

# Upwinding of relative permeabilities in the bulk
upwind: mdu.FluxBaseUpwindAd(gb=gb, grid_list=bulk_list, param_key=kw)
zf_bulk_ad = pp.ad.Array(bulk_list[0].face_centers[gfo.dim - 1])
psi_bc_ad = bound_bulk - zf_bulk_ad  # pressure head at the faces (only used the dir bc values)
krw_faces_ad: pp.ad.Operator = upwind(krw_ad(psib_m), krw_ad(psi_bc_ad), flux1p_bulk)

# Multiphase Darcy flux
flux_bulk: pp.ad.Operator = (
        krw_faces_ad * mpfa_bulk.flux * h_bulk
        + krw_faces_ad * mpfa_bulk.bound_flux * bound_bulk
        + krw_faces_ad
        * mpfa_bulk.bound_flux
        * bulk_face_rest
        * proj.mortar_to_primary_int
        * lmbda
)

# Accumulation terms and sources
dt_ad = mdu.ParameterScalar(kw, "time_step", grids=bulk_list)
source_bulk = pp.ad.ParameterArray(kw, "source", grids=bulk_list)
mass_bulk = pp.ad.MassMatrixAd(kw, grids=bulk_list)

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

# Thus far, thus good... Start from here

# %% Declare equations for the fractures
fracvol = mdu.FractureVolume(g_frac_ghost, d_frac, kw)
vol_ad = pp.ad.Function(fracvol.fracture_volume, name="Volume")
vol_cap_ad = pp.ad.Function(fracvol.volume_capacity, name="Volume capacity")
gfh = GhostFractureHydraulicHead(gb=gb, ghost_grid=g_frac_ghost)
ghost_hf_ad = pp.ad.Function(
    gfh.get_ghost_hyd_head, name="Ghost fracture hydraulic head func"
)
ghost_hf = ghost_hf_ad(h_frac)

linearization = "newton"
if linearization == "newton":
    accum_frac_active = vol_ad(h_frac)
    accum_frac_inactive = vol_ad(hf_n) * (-1)
elif linearization == "modified_picard":
    accum_frac_active = h_frac * vol_cap_ad(hf_m)
    accum_frac_inactive = vol_ad(hf_m) - vol_cap_ad(hf_m) * hf_m - vol_ad(hf_n)
elif linearization == "l_scheme":
    L = 0.015
    accum_frac_active = L * h_frac
    accum_frac_inactive = vol_ad(hf_m) - L * hf_m - vol_ad(hf_n)
else:
    raise NotImplementedError(
        "Linearization scheme not implemented. Use 'newton', "
        "'modified_picard', or 'l_scheme'."
    )

# Conservation equation in the fracture: This is a water volume balance
aperture_ad = ParameterScalar(kw, "aperture", grid=g_frac)
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
mpfa_global = pp.ad.MpfaAd(kw, [g_bulk, g_frac])
robin = pp.ad.RobinCouplingAd(kw, edge_list)

# Projected bulk pressure traces onto the mortar grid
mortar_hb = (
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
is_conductive = pp.ad.ParameterArray(kw, "is_conductive", edges=edge_list)

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
        node_var: d_frac[pp.PARAMETERS][kw]["datum"]
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
        gb, dof_manager, kw, mortar_hb, mortar_hf, edge_list
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
