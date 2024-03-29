import mdunsat as mdu
import numpy as np
import porepy as pp
import scipy.sparse.linalg as spla
from exact_solution import ExactSolution
from grid_factory import GridGenerator
from mdunsat.ad_utils import (
    get_conductive_mortars,
    set_iterate_as_state,
    set_state_as_iterate,
)
from mdunsat.analysis_utils import relative_l2_error


def manufactured_model(
    mesh_size: float,
    export_to_paraview=False,
) -> dict[str, float]:
    """
    Model for convergence analysis of the numerical examples section.

    Parameters:
        mesh_size: Mesh size to be used in the analysis.
        export_to_paraview: Whether to export to Paraview.

    Returns:
        Dictionary containing the L2-errors for the hydraulic head in the bulk,
        the integrated Darcy fluxes in the bulk, the integrated normal fluxes on the
        interfaces, the hydraulic head in the fracture, and the volume of water in
        the fracture.

    """

    # %% Retrieve exact solution object
    ex = ExactSolution()

    # %% Retrieve grid buckets
    gfo = GridGenerator(
        mesh_args={"mesh_size_frac": mesh_size, "mesh_size_bound": mesh_size},
        csv_file="network.csv",
        domain={"xmin": 0, "ymin": 0, "xmax": 1.0, "ymax": 1.0},
        constraints=[1, 2],
    )
    gb, ghost_gb = gfo.get_grid_buckets()

    grid_list = gfo.get_grid_list(gb)
    bulk_list = gfo.get_bulk_list(gb)
    frac_list = gfo.get_fracture_list(gb)
    edge_list = gfo.get_edge_list(gb)

    ghost_bulk_list = gfo.get_bulk_list(ghost_gb)
    ghost_frac_list = gfo.get_fracture_list(ghost_gb)
    ghost_edge_list = gfo.get_edge_list(ghost_gb)

    for g, _ in gb:
        if g.dim == 2:
            g_bulk = g
        else:
            g_frac = g

    for e, d in gb.edges():
        g_intf = d["mortar_grid"]

    # %% Time parameters
    tsc = pp.TimeSteppingControl(
        schedule=[0, 0.5],
        dt_init=0.25 / 16,
        dt_min_max=(0.01, 0.1),
        iter_max=200,
        iter_optimal_range=(1, 100),  # dirty trick to use constant time step
        iter_lowupp_factor=(1.3, 0.7),
        recomp_factor=0.5,
        recomp_max=12,
        print_info=True,
    )

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
            # Boundary conditions
            bc_faces = g.get_boundary_faces()
            bc_type = np.array(bc_faces.size * ["dir"])
            bc = pp.BoundaryCondition(g, faces=bc_faces, cond=list(bc_type))
            bc_values = ex.rock_boundary_hydraulic_head(g, time=tsc.time)

            # Initialize bulk data
            pressure_threshold = 0 * np.ones(g.num_cells)
            specified_parameters: dict = {
                "pressure_threshold": pressure_threshold,
                "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells)),
                "bc": bc,
                "bc_values": bc_values,
                "source": ex.rock_source(g, time=tsc.time),
                "elevation": g.cell_centers[gb.dim_max() - 1],
                "mass_weight": np.ones(g.num_cells),
                "time_step": tsc.dt,  # [s]
            }
            pp.initialize_data(g, d, param_key, specified_parameters)

        else:
            # Parameters for the fracture grids
            specified_parameters = {
                "aperture": ex.aperture,
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
        zeros = np.zeros(mg.num_cells)
        exact_normal_flux = (
            ex.interface_darcy_flux(mg, tsc.time_final) / mg.cell_volumes
        )
        zeta_mortar = mg.cell_centers[gb.dim_max() - 1]
        psi_l = np.mean(pressure_threshold)
        exact_hyd_head_jump = ex.c_sat - (0 + zeta_mortar)  # exact jump
        normal_diffusivity = exact_normal_flux / (exact_hyd_head_jump * 1)
        is_conductive = zeros
        data = {
            "normal_diffusivity": normal_diffusivity,
            "is_conductive": is_conductive,
            "elevation": mg.cell_centers[gb.dim_max() - 1],
            "pressure_threshold": 0 * np.ones(mg.num_cells),
        }
        pp.initialize_data(mg, d, param_key, data)

    # %% Set initial states
    for g, d in gb:
        if g.dim == gfo.dim:
            pp.set_state(
                d,
                state={node_var: ex.rock_hydraulic_head(bulk_list[0], time=0)},
            )
            pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})
        else:
            pp.set_state(
                d,
                state={
                    node_var: np.array(
                        [
                            np.mean(pressure_threshold)
                            + d[pp.PARAMETERS][param_key]["datum"]
                        ]
                    )
                },
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
    subdomain_proj_scalar = pp.ad.SubdomainProjections(grids=grid_list)
    bulk_face_rest: pp.ad.Matrix = subdomain_proj_scalar.face_restriction(bulk_list)
    bulk_face_prol: pp.ad.Matrix = subdomain_proj_scalar.face_prolongation(bulk_list)
    frac_cell_rest: pp.ad.Matrix = subdomain_proj_scalar.cell_restriction(frac_list)

    # %% Governing equations in the bulk
    # Soil water retention curves
    def theta(pressure_head):

        if isinstance(pressure_head, pp.ad.Ad_array):
            is_unsat = 1 - pp.ad.heaviside(pressure_head.val, 1)
            is_sat = 1 - is_unsat
            theta_val = (1 - pressure_head) ** (-1) * is_unsat + 1 * is_sat
        else:
            is_unsat = 1 - pp.ad.heaviside(pressure_head, 1)
            is_sat = 1 - is_unsat
            theta_val = (1 - pressure_head) ** (-1) * is_unsat + 1 * is_sat

        return theta_val

    def krw(pressure_head):

        if isinstance(pressure_head, pp.ad.Ad_array):
            is_unsat = 1 - pp.ad.heaviside(pressure_head.val, 1)
            is_sat = 1 - is_unsat
            krw_val = pp.ad.exp(pressure_head) * is_unsat + 1 * is_sat
        else:
            is_unsat = 1 - pp.ad.heaviside(pressure_head, 1)
            is_sat = 1 - is_unsat
            krw_val = np.exp(pressure_head) * is_unsat + 1 * is_sat

        return krw_val

    def smc(pressure_head):

        if isinstance(pressure_head, pp.ad.Ad_array):
            is_unsat = 1 - pp.ad.heaviside(pressure_head.val, 1)
            is_sat = 1 - is_unsat
            smc_val = (1 - pressure_head) ** (-2) * is_unsat + 0 * is_sat
        else:
            is_unsat = 1 - pp.ad.heaviside(pressure_head, 1)
            is_sat = 1 - is_unsat
            smc_val = (1 - pressure_head) ** (-2) * is_unsat + 0 * is_sat

        return smc_val

    theta_ad = pp.ad.Function(theta, name="water content")
    krw_ad = pp.ad.Function(krw, name="relative permeability")
    smc_ad = pp.ad.Function(smc, name="specific moisture capacity")

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
    krw_faces_ad: pp.ad.Operator = upwind(
        krw_ad(psib_m), krw_ad(psi_bc_ad), flux_single_phase_bulk
    )

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
        raise NotImplementedError

    accumulation_bulk = accum_bulk_active + accum_bulk_inactive
    conserv_bulk_eq = (
        accumulation_bulk + dt_ad * div_bulk * flux_bulk - dt_ad * source_bulk
    )

    # Discretize and evaluate
    conserv_bulk_eq.discretize(gb=gb)

    # %% Governing equations in the fracture

    # Get water volume as a function of the hydraulic head, and its first derivative
    fv = mdu.FractureVolume(gb=gb, fracture_grids=frac_list, param_key=param_key)
    vol_ad: pp.ad.Function = fv.fracture_volume(as_ad=True)
    vol_cap_ad: pp.ad.Function = fv.volume_capacity(as_ad=True)
    vol = fv.fracture_volume(as_ad=False)

    linearization = "newton"  # linearization of the fracture equations
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
        raise NotImplementedError

    # Retrieve sources from mortar
    sources_from_mortar = frac_cell_rest * projections.mortar_to_secondary_int * lmbda
    # Accumulation terms
    accum_frac = accum_frac_active + accum_frac_inactive
    # Declare conservation equation
    # NOTE: We need to divide the sources_from_mortar by 2 to get the correct result
    conserv_frac_eq = accum_frac - 0.5 * tsc.dt * sources_from_mortar

    # Evaluate and discretize
    conserv_frac_eq.discretize(gb=gb)

    # %% Governing equations on the interfaces
    robin = pp.ad.RobinCouplingAd(param_key, edge_list)

    # Projected pressure threshold traces onto the mortar grid.
    # Can be parsed right away since this is a non-ad array and always known
    # proj_tr_psi_l: np.ndarray = (
    #         projections.primary_to_mortar_avg.parse(gb)
    #         * bulk_face_prol.parse(gb)
    #         * mpfa_bulk.bound_pressure_cell.parse(gb)
    #         * pressure_threshold
    # )
    proj_tr_psi_l = pp.ad.ParameterArray(
        param_keyword=param_key,
        array_keyword="pressure_threshold",
        edges=edge_list,
        name="interface pressure threshold",
    )

    # Projected bulk pressure traces onto the mortar grid.
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

    proj_tr_psi_bulk_m = (
        projections.primary_to_mortar_avg
        * bulk_face_prol
        * mpfa_bulk.bound_pressure_cell
        * psib_m
        + projections.primary_to_mortar_avg
        * bulk_face_prol
        * mpfa_bulk.bound_pressure_face
        * bulk_face_rest
        * projections.mortar_to_primary_int
        * lmbda_m
    )

    # Get projected ghost fracture hydraulic head onto the adjacent mortar grids
    pfh = mdu.GhostHydraulicHead(
        gb=gb,
        ghost_gb=ghost_gb,
        # mortar_proj_pressure_threshold=proj_tr_psi_l,
    )
    frac_to_mortar_ad: pp.ad.Function = pfh.proj_fra_hyd_head(as_ad=True)
    proj_h_frac = frac_to_mortar_ad(h_frac)

    # Array parameter that keeps track of conductive (1) and blocking (0) mortar cells
    # Note that if it is blocking, the whole discrete equation is removed for that mortar cell
    is_conductive = pp.ad.ParameterArray(param_key, "is_conductive", edges=edge_list)

    # Interface flux
    mortar_flux = (
        krw_ad(proj_tr_psi_bulk_m)
        * robin.mortar_discr
        * (proj_tr_h_bulk - proj_h_frac)
        * is_conductive
    )
    interface_flux_eq = mortar_flux + lmbda
    interface_flux_eq.discretize(gb=gb)

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

    d_frac = gb.node_props(frac_list[0])
    d_frac_ghost = ghost_gb.node_props(ghost_frac_list[0])
    z_frac_ghost = ghost_frac_list[0].cell_centers[1]

    d_edge = gb.edge_props(edge_list[0])
    d_edge_ghost = ghost_gb.edge_props(ghost_edge_list[0])

    # Set state in bulk ghost grid
    pp.set_state(
        data=d_bulk_ghost,
        state={
            node_var: d_bulk[pp.STATE][node_var],
            "pressure_head": d_bulk[pp.STATE][node_var] - z_bulk,
        },
    )
    # Set state in fracture ghost grid
    pp.set_state(
        data=d_frac_ghost,
        state={
            node_var: d_frac[pp.PARAMETERS][param_key]["datum"]
            * np.ones(ghost_frac_list[0].num_cells),
            "pressure_head": d_frac[pp.STATE][node_var] - z_frac_ghost,
        },
    )
    # Set state in edges
    pp.set_state(data=d_edge_ghost, state={edge_var: d_edge[pp.STATE][edge_var]})

    # Correct values of pressure head in the fracture if negative
    for val in d_frac_ghost[pp.STATE]["pressure_head"] <= 0:
        d_frac_ghost[pp.STATE]["pressure_head"][val] = np.mean(pressure_threshold)

    # %% Time loop
    total_iteration_counter: int = 0
    iters: list = []
    abs_tol: float = 1e-10
    is_mortar_conductive: np.ndarray = np.zeros(gb.num_mortar_cells(), dtype=np.int32)
    control_faces: np.ndarray = is_mortar_conductive

    # Time loop
    while tsc.time < tsc.time_final:
        tsc.time += tsc.dt
        iteration_counter: int = 0
        residual_norm: float = 1.0
        print(f"Time: {tsc.time}")

        zeta_mortar = mg.cell_centers[gb.dim_max() - 1]
        exact_hyd_head_jump = ex.c_sat - (0 + zeta_mortar)  # exact jump
        d_edge[pp.PARAMETERS]["flow"]["normal_diffusivity"] = (
            ex.interface_darcy_flux(mg, tsc.time) / mg.cell_volumes
        ) / exact_hyd_head_jump
        d_bulk[pp.PARAMETERS]["flow"]["bc_values"] = ex.rock_boundary_hydraulic_head(
            g_bulk, tsc.time
        )
        d_bulk[pp.PARAMETERS]["flow"]["source"] = ex.rock_source(g_bulk, tsc.time)

        equation_manager.discretize(gb)

        # Solver loop
        while iteration_counter <= tsc.iter_max and not residual_norm < abs_tol:

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

            # Prepare next iteration
            iteration_counter += 1
            total_iteration_counter += 1
            # end of iteration loop

        # Recompute solution if we did not achieve convergence
        if residual_norm > abs_tol or np.isnan(residual_norm):
            tsc.next_time_step(
                recompute_solution=True, iterations=iteration_counter - 1
            )
            param_update.update_time_step(tsc.dt)
            set_iterate_as_state(gb, node_var, edge_var)
            continue

        # Recompute solution if negative volume is encountered
        if np.any(vol(h_frac.evaluate(dof_manager).val) < 0):
            tsc.next_time_step(
                recompute_solution=True, iterations=iteration_counter - 1
            )
            param_update.update_time_step(tsc.dt)
            print(f"Encountered negative volume. Reducing dt and recomputing solution.")
            set_iterate_as_state(gb, node_var, edge_var)
            continue

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
            param_update.update_mortar_conductivity_state(
                is_mortar_conductive, edge_list
            )
            print("Encountered saturated mortar cells. Recomputing solution")
            # print(f"Number of mortar cells: {gb.num_mortar_cells()}")
            # print(f"Number of conductive mortars: {is_mortar_conductive.sum()}")

            control_faces = is_mortar_conductive
            set_iterate_as_state(gb, node_var, edge_var)
            tsc.time -= (
                tsc.dt
            )  # correct time since we are going to recompute the solution
            continue
        else:
            is_mortar_conductive: np.ndarray = np.zeros(
                gb.num_mortar_cells(), dtype=np.int8
            )
            param_update.update_mortar_conductivity_state(
                is_mortar_conductive, edge_list
            )
            control_faces = is_mortar_conductive

        # Save number of iterations and time step
        iters.append(iteration_counter - 1)

        # Successful convergence
        tsc.next_time_step(recompute_solution=False, iterations=iteration_counter - 1)
        param_update.update_time_step(tsc.dt)

        """We need to fix the values of the volume and hydraulic head, since the 
        exact solution is based on the assumption of zero initial volume."""
        vol_frac_mpfa = (
            vol(d_frac[pp.STATE][node_var])[0]
            + vol(d_frac[pp.STATE][pp.ITERATE][node_var])[0]
        )
        h_frac_mpfa = (
            d_frac[pp.STATE][node_var][0] + d_frac[pp.STATE][pp.ITERATE][node_var][0]
        ) - 0.25
        set_state_as_iterate(gb, node_var, edge_var)
        print(f"h1_mpfa: {h_frac_mpfa}")
        print(f"h1_ex: {ex.fracture_hydraulic_head(tsc.time)}")
        print(f"vol_mpfa: {vol_frac_mpfa}")
        print(f"vol_ex: {ex.fracture_volume(tsc.time)}")

        print()

    if export_to_paraview:
        d_bulk_ghost[pp.STATE]["exact_h"] = ex.rock_hydraulic_head(
            g_bulk, tsc.time_final
        )
        d_frac_ghost[pp.STATE]["exact_h"] = np.ones(ghost_frac_list[0].num_cells)
        exporter_ghost = pp.Exporter(ghost_gb, "conv_analysis", "out")
        exporter_ghost.write_vtu([node_var, "exact_h"])

    #%% Compute errors
    h_bulk_mpfa = d_bulk[pp.STATE][pp.ITERATE]["hydraulic_head"]
    h_bulk_exact = ex.rock_hydraulic_head(g_bulk, tsc.time_final)
    error_h_bulk = relative_l2_error(g_bulk, h_bulk_exact, h_bulk_mpfa, True, True)

    q_bulk_mpfa = flux_bulk.evaluate(dof_manager).val
    q_bulk_exact = ex.rock_darcy_flux(g_bulk, tsc.time_final)
    error_q_bulk = relative_l2_error(g_bulk, q_bulk_exact, q_bulk_mpfa, True, False)

    q_intf_mpfa = d_edge[pp.STATE][pp.ITERATE]["mortar_flux"]
    q_intf_exact = ex.interface_darcy_flux(g_intf, tsc.time_final)
    error_q_intf = relative_l2_error(mg, q_intf_exact, q_intf_mpfa, True, True)

    # h_frac_mpfa = h_frac.evaluate(dof_manager).val
    h_frac_exact = ex.fracture_hydraulic_head(tsc.time_final)
    error_h_frac = relative_l2_error(g_frac, h_frac_exact, h_frac_mpfa, True, True)

    # vol_frac_mpfa = vol(h_frac_mpfa)
    vol_frac_exact = ex.fracture_volume(tsc.time_final)
    error_vol_frac = relative_l2_error(
        g_frac, vol_frac_exact, vol_frac_mpfa, True, True
    )

    # print(
    #     f"Summary of errors: \n"
    #     f"Error h_bulk: {error_h_bulk} \n"
    #     f"Error q_bulk: {error_q_bulk} \n"
    #     f"Error q_intf: {error_q_intf} \n"
    #     f"Error h_frac: {error_h_frac} \n"
    #     f"Error vol_frac: {error_vol_frac} \n"
    # )

    errors_dict = {
        "error_h_bulk": error_h_bulk,
        "error_q_bulk": error_q_bulk,
        "error_q_intf": error_q_intf,
        "error_h_frac": error_h_frac,
        "error_vol_frac": error_vol_frac,
    }

    return errors_dict
