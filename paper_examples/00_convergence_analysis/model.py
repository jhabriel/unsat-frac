def model(mesh_size: float) -> dict[str, float]:
    import mdunsat as mdu
    import numpy as np
    import porepy as pp
    import pickle
    import scipy.sparse as sps
    import scipy.sparse.linalg as spla
    import matplotlib.pyplot as plt

    from grid_factory import GridGenerator
    from mdunsat.ad_utils import (
        get_conductive_mortars,
        get_ghost_hydraulic_head,
        set_state_as_iterate,
        set_iterate_as_state,
    )
    from exact_solution import ExactSolution
    from typing import Union

    def relative_l2_error(
            grid: Union[pp.Grid, pp.MortarGrid],
            true_array: np.ndarray,
            approx_array: np.ndarray,
            is_scalar: bool,
            is_cc: bool,
    ) -> float:
        """Compute discrete L2-error.
        Parameters:
            grid: Either a subdomain grid or a mortar grid.
            true_array: Array containing the true values of a given variable.
            approx_array: Array containing the approximate values of a given variable.
            is_scalar: Whether the variable is a scalar quantity. Use ``False`` for
                vector quantities. For example, ``is_scalar=True`` for pressure, whereas
                ``is_scalar=False`` for displacement.
            is_cc: Whether the variable is associated to cell centers. Use ``False``
                for variables associated to face centers. For example, ``is_cc=True``
                for pressures, whereas ``is_scalar=False`` for subdomain fluxes.
        Returns:
            Discrete relative L2-error between the true and approximated arrays.
        Raises:
            ValueError if a mortar grid is given and ``is_cc=False``.
        """
        # Sanity check
        if isinstance(grid, pp.MortarGrid) and not is_cc:
            raise ValueError("Mortar variables can only be cell-centered.")

        # Obtain proper measure
        if is_cc:
            if is_scalar:
                meas = grid.cell_volumes
            else:
                meas = grid.cell_volumes.repeat(grid.dim)
        else:
            assert isinstance(grid, pp.Grid)
            if is_scalar:
                meas = grid.face_areas
            else:
                meas = grid.face_areas.repeat(grid.dim)

        # Compute error
        numerator = np.sqrt(np.sum(meas * np.abs(true_array - approx_array) ** 2))
        denominator = np.sqrt(np.sum(meas * np.abs(true_array) ** 2))

        return numerator / denominator

    # %% Retrieve exact solution object
    ex = ExactSolution()

    # %% Retrieve grid buckets
    gfo = GridGenerator(
        mesh_args={"mesh_size_frac": 0.0125, "mesh_size_bound": 0.0125},
        csv_file="network.csv",
        domain={"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        constraints=[1, 2],
    )
    gb, ghost_gb = gfo.get_grid_buckets()

    grid_list = gfo.get_grid_list(gb)
    bulk_list = gfo.get_bulk_list(gb)
    frac_list = gfo.get_fracture_list(gb)
    edge_list = gfo.get_edge_list(gb)

    ghost_grid_list = gfo.get_grid_list(ghost_gb)
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

    # Uncomment to export grid
    # export_mesh = pp.Exporter(ghost_gb, file_name="grid", folder_name="out")
    # export_mesh.write_vtu(ghost_gb)

    # %% Time parameters
    tsc = pp.TimeSteppingControl(
        schedule=[0, 0.5],
        dt_init=0.5,
        dt_min_max=(0.01, 0.5),
        iter_max=300,
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

    pressure_threshold = -5
    for g, d in gb:
        if g.dim == gb.dim_max():
            # Boundary conditions
            bc_faces = g.get_boundary_faces()
            bc_type = np.array(bc_faces.size * ["dir"])
            bc = pp.BoundaryCondition(g, faces=bc_faces, cond=list(bc_type))
            bc_values = ex.rock_boundary_hydraulic_head(g, time=0.5)

            # Initialize bulk data
            specified_parameters: dict = {
                "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells)),
                "bc": bc,
                "bc_values": bc_values,
                "source": ex.rock_source(g, time=0.5),
                "elevation": g.cell_centers[gb.dim_max() - 1],
                "mass_weight": np.ones(g.num_cells),
                "time_step": tsc.dt,  # [s]
            }
            pp.initialize_data(g, d, param_key, specified_parameters)

        else:
            # Parameters for the fracture grids
            specified_parameters = {
                "aperture": 0.1,
                "datum": np.min(g.face_centers[gb.dim_max() - 1]),
                "elevation": g.cell_centers[gb.dim_max() - 1],
                "sin_alpha": 1.0,
                "width": 1.0,
            }
            pp.initialize_data(g, d, param_key, specified_parameters)

    # Parameters for the mortar
    for e, d in gb.edges():
        mg = d["mortar_grid"]
        ones = np.ones(mg.num_cells)
        zeros = np.zeros(mg.num_cells)
        g_sec, _ = gb.nodes_of_edge(e)
        d_sec = gb.node_props(g_sec)
        aperture = d_sec[pp.PARAMETERS][param_key]["aperture"]
        sat_conductivity = 1e-2  # [cm/s]
        ex_lmbda = ex.interface_darcy_flux(mg, 0.5) / mg.cell_volumes
        kappa = ex_lmbda / (4)
        is_conductive = zeros
        data = {
            "sat_normal_diffusivity": kappa,
            "normal_diffusivity": kappa,
            "is_conductive": is_conductive,
            "elevation": mg.cell_centers[gb.dim_max() - 1],
        }
        pp.initialize_data(mg, d, param_key, data)

    # %% Set initial states
    for g, d in gb:
        if g.dim == gfo.dim:
            pp.set_state(d, state={node_var: -1 + d[pp.PARAMETERS][param_key][
                "elevation"]})
            pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})
        else:
            pp.set_state(
                d,
                state={
                    node_var: np.array(
                        [pressure_threshold + d[pp.PARAMETERS][param_key]["datum"]]
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

    # # Soil water retention curves
    # vgm = mdu.VanGenuchtenMualem(gb=gb, param_key=param_key)
    # theta_ad: pp.ad.Function = vgm.water_content(as_ad=True)
    # krw_ad: pp.ad.Function = vgm.relative_permeability(as_ad=True)
    # smc_ad: pp.ad.Function = vgm.moisture_capacity(as_ad=True)

    # Soil water retention curves
    def theta(pressure_head):
        return (1 - pressure_head) ** (-1)

    def krw(pressure_head):
        return pp.ad.exp(pressure_head)

    def smc(pressure_head):
        return (1 - pressure_head) ** (-2)

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

    linearization = "newton"  # linearization of the bulk equations
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
        raise NotImplementedError

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
    # TODO: sources_from_mortar are the integrated mortar fluxes. Check if we need to scale
    #  this by some factor, before multiplying by the aperture.
    conserv_frac_eq = accum_frac - dt_ad * sources_from_mortar

    # Evaluate and discretize
    conserv_frac_eq.discretize(gb=gb)
    conserva_frac_num = conserv_frac_eq.evaluate(dof_manager=dof_manager)

    # %% Governing equations on the interfaces
    mpfa_global = pp.ad.MpfaAd(param_key, grid_list)
    robin = pp.ad.RobinCouplingAd(param_key, edge_list)

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

    # Array parameter that keeps track of conductive (1) and blocking (0) mortar cells
    # Note that if it is blocking, the whole discrete equation is removed for that mortar cell
    is_conductive = pp.ad.ParameterArray(param_key, "is_conductive", edges=edge_list)

    # Interface flux
    mortar_flux = robin.mortar_discr * (proj_tr_h_bulk - proj_h_frac) * is_conductive
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
    d_bulk = gb.node_props(bulk_list[0])
    d_bulk_ghost = ghost_gb.node_props(ghost_bulk_list[0])
    z_bulk = bulk_list[0].cell_centers[1]

    d_frac = gb.node_props(frac_list[0])
    d_frac_ghost = ghost_gb.node_props(ghost_frac_list[0])
    z_frac = frac_list[0].cell_centers[1]
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
    # Pressure head is not exported correctly. Try with a different method. Perhaps, np.max(
    # el_actual, 0)*
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
        d_frac_ghost[pp.STATE]["pressure_head"][val] = pressure_threshold

    water_table = [d_frac[pp.STATE][node_var][0] - pressure_threshold]
    water_vol = [vol(d_frac[pp.STATE][node_var])[0]]
    exporter_ghost = pp.Exporter(ghost_gb, "single_frac", "out")
    exporter_ghost.write_vtu([node_var, "pressure_head", edge_var], time_step=0)

    # %% Time loop
    total_iteration_counter: int = 0
    iters: list = []
    abs_tol: float = 1e-6
    is_mortar_conductive: np.ndarray = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
    control_faces: np.ndarray = is_mortar_conductive

    # Time loop
    while tsc.time < tsc.time_final:
        tsc.time += tsc.dt
        iteration_counter: int = 0
        residual_norm: float = 1.0
        rel_res: float = 1.0
        print(f"Time: {tsc.time}")

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
            rel_res = residual_norm / initial_residual_norm

            # Uncomment for full info
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
            tsc.next_time_step(recompute_solution=True,
                               iterations=iteration_counter - 1)
            param_update.update_time_step(tsc.dt)
            set_iterate_as_state(gb, node_var, edge_var)
            continue

        # # Recompute solution if negative volume is encountered
        # if np.any(vol(h_frac.evaluate(dof_manager).val) < 0):
        #     tsc.next_time_step(recompute_solution=True, iterations=iteration_counter - 1)
        #     param_update.update_time_step(tsc.dt)
        #     print(f"Encountered negative volume. Reducing dt and recomputing solution.")
        #     set_iterate_as_state(gb, node_var, edge_var)
        #     continue

        # Recompute solution is capillary barrier is overcome. Note that dt remains the same
        is_mortar_conductive = get_conductive_mortars(
            gb, dof_manager, param_key, proj_tr_h_bulk, proj_h_frac, edge_list
        )
        if control_faces.sum() == 0 and is_mortar_conductive.sum() > 0:
            param_update.update_mortar_conductivity_state(is_mortar_conductive,
                                                          edge_list)
            # print(
            #     f"The faces {np.where(is_mortar_conductive)[0]} are saturated. "
            #     f"Solution will be recomputed."
            # )
            print("Encountered saturated mortar cells. Recomputing solution")
            control_faces = is_mortar_conductive
            set_iterate_as_state(gb, node_var, edge_var)
            tsc.time -= tsc.dt  # correct time since we are going to recompute the solution
            continue
        else:
            is_mortar_conductive: np.ndarray = np.zeros(
                gb.num_mortar_cells(), dtype=np.int8
            )
            param_update.update_mortar_conductivity_state(is_mortar_conductive,
                                                          edge_list)
            control_faces = is_mortar_conductive

        # Save number of iterations and time step
        iters.append(iteration_counter - 1)
        times.append(tsc.time)
        dts.append(tsc.dt)

        # Successful convergence
        tsc.next_time_step(recompute_solution=False, iterations=iteration_counter - 1)
        param_update.update_time_step(tsc.dt)
        print(
            f"Fracture water table height: {d_frac[pp.STATE][node_var][0] - pressure_threshold}"
        )
        print(f"Fracture water volume: {vol(d_frac[pp.STATE][node_var])[0]}")
        water_table.append(d_frac[pp.STATE][node_var][0] - pressure_threshold)
        water_vol.append(vol(d_frac[pp.STATE][node_var])[0])
        set_state_as_iterate(gb, node_var, edge_var)
        print()

        # # Export to ParaView
        # pp.set_state(
        #     data=d_bulk_ghost,
        #     state={
        #         node_var: d_bulk[pp.STATE][node_var],
        #         "pressure_head": d_bulk[pp.STATE][node_var] - z_bulk,
        #     },
        # )
        # pp.set_state(
        #     data=d_frac_ghost,
        #     state={
        #         node_var: get_ghost_hydraulic_head(
        #             ghost_frac_list[0], d_frac[pp.STATE][node_var]
        #         ),
        #         "pressure_head": get_ghost_hydraulic_head(
        #             ghost_frac_list[0], d_frac[pp.STATE][node_var]
        #         )
        #         - z_frac_ghost,
        #     },
        # )
        # pp.set_state(data=d_edge_ghost, state={edge_var: d_edge[pp.STATE][edge_var]})
        # # Correct values of pressure head in the fracture if negative
        # for val in d_frac_ghost[pp.STATE]["pressure_head"] <= 0:
        #     d_frac_ghost[pp.STATE]["pressure_head"][val] = 0
        # if tsc.time in tsc.schedule:
        #     export_counter += 1
        #     exporter_ghost.write_vtu(
        #         [node_var, "pressure_head", edge_var], time_step=export_counter
        #     )

    # # %% Plotting
    h_bulk_mpfa = d_bulk[pp.STATE][pp.ITERATE]["hydraulic_head"]
    h_bulk_exact = ex.rock_hydraulic_head(g_bulk, 0.5)

    q_bulk_mpfa = flux_bulk.evaluate(dof_manager).val
    q_bulk_exact = ex.rock_darcy_flux(g_bulk, 0.5)

    q_intf_mpfa = d_edge[pp.STATE][pp.ITERATE]["mortar_flux"]
    q_intf_exact = ex.interface_darcy_flux(g_intf, 0.5)

    h_frac_mpfa = h_frac.evaluate(dof_manager).val + 5 - 0.25
    h_frac_exact = ex.fracture_hydraulic_head(0.5)
    #
    # pp.plot_grid(
    #     grid_list[0], h_bulk_mpfa, linewidth=0, plot_2d=True, title="h_bulk (MPFA)"
    # )
    # pp.plot_grid(
    #     grid_list[0], h_bulk_exact, linewidth=0, plot_2d=True, title="h_bulk (Exact)"
    # )

    # %% Errors
    error_h_bulk = relative_l2_error(g_bulk, h_bulk_exact, h_bulk_mpfa, True, True)
    error_q_bulk = relative_l2_error(g_bulk, q_bulk_exact, q_bulk_mpfa, True, False)
    error_q_intf = relative_l2_error(mg, q_intf_exact, q_intf_mpfa, True, True)
    error_h_frac = np.abs(h_frac_mpfa - h_frac_exact)

    print(
        f"Summary of errors: \n"
        f"Error h_bulk: {error_h_bulk} \n"
        f"Error q_bulk: {error_q_bulk} \n"
        f"Error q_intf: {error_q_intf} \n"
        f"Error h_frac: {error_h_frac[0]}"
    )

    out = {
        "error_h_bulk": error_h_bulk,
        "error_q_bulk": error_q_bulk,
        "error_q_intf": error_q_intf,
        "error_h_frac": error_h_frac,
    }

    return out