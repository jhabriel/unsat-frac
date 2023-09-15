# Import packages
import mdunsat as mdu
import numpy as np
import porepy as pp
import scipy.sparse.linalg as spla

from mdunsat.ad_utils import set_iterate_as_state, set_state_as_iterate


def gradient_segments(
        num_segments: int, increment_factor: float, start: float, end: float
) -> np.ndarray:
    """Return a list of segments with a geometrically increasing gradient.

    Parameters:
        num_segments: Number of segments.
        increment_factor: Factor by which the gradient increases.
        start: Start value.
        end: End value.

    Returns:
        np.ndarray: Array of points dividing the interval into segments.
        Only the second end point is included in the array.

    """
    length = end - start
    increments = np.power(increment_factor, np.arange(num_segments))
    segments = np.cumsum(increments) / np.sum(increments) * length + start
    return segments


#%% ---> Start of the equi-dimensional model
def equidim_model(
        matrix_params: dict,
        fracture_params: dict,
        block_params: dict,
        out_fname: str,
):

    #%% Create grid
    x_left_block = np.concatenate((np.array([0.0]), gradient_segments(30, 0.995, 0, 30)))
    x_block = gradient_segments(20, 0.995, 30, 49.5)
    x_left = np.concatenate((x_left_block, x_block))
    # x_fracture = np.array([49.75, 49.90, 50.0, 50.10, 50.25])
    # x_fracture = np.array([50])
    x_fracture = np.array([])
    x_right = np.concatenate(
        (np.array([50.5]), 50.5 + np.cumsum(np.flip(x_left[1:] - x_left[:-1])))
    )
    x = np.concatenate((x_left, x_fracture))
    x = np.concatenate((x, x_right))
    y = np.linspace(0, 100, np.int(100/2.5 + 1))  # (100/2.5 + 1) will also do the job
    gb = pp.meshing.tensor_grid([], x=x, y=y, z=None)
    g = gb.grids_of_dimension(2)[0]
    # Uncomment to plot grid
    # pp.plot_grid(g, plot_2d=True)


    #%% Time-stepping control
    schedule = list(np.linspace(0, 4800, 100, dtype=np.int32))
    tsc = pp.TimeSteppingControl(
        schedule=schedule,
        dt_init=0.01,
        dt_min_max=(0.01, 100),
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
    export_counter: int = 0

    #%% Assign parameters
    # Keywords
    param_key: str = "flow"
    node_var: str = "hydraulic_head"

    # Declare primary variable
    for _, d in gb:
        d[pp.PRIMARY_VARIABLES] = {node_var: {"cells": 1}}

    # Parameter assigment
    param_update = mdu.ParameterUpdate(gb, param_key)

    for g, d in gb:

        # Store values of bounding box
        y_max: float = gb.bounding_box()[1][1]

        # Face indices of the boundaries
        fc: np.ndarray = g.face_centers
        top: np.ndarray = np.where(np.abs(fc[1] - y_max) < 1e-5)[0]
        top_left: np.ndarray = top[np.where(np.abs(fc[0][top]) < 49.5)[0]]

        # Left and right cell indices
        cc: np.ndarray = g.cell_centers

        # Cell indices corresponding to the block of clay
        blocking_idx: list = [
            cc[0] < 49.5,
            cc[0] > 30.0,
            cc[1] < 80.0,
            cc[1] > 60.0,
        ]
        mult_cond: bool = np.logical_and(blocking_idx[0], blocking_idx[1])
        mult_cond: bool = np.logical_and(blocking_idx[2], mult_cond)
        mult_cond: bool = np.logical_and(blocking_idx[3], mult_cond)

        # Cell indices for the fractures
        fracture_idx: list = [
            cc[0] > 49.5,
            cc[0] < 50.5,
            cc[1] < 90.0,
            cc[1] > 10.0,
        ]
        frac_cond: bool = np.logical_and(fracture_idx[0], fracture_idx[1])
        frac_cond: bool = np.logical_and(fracture_idx[2], frac_cond)
        frac_cond: bool = np.logical_and(fracture_idx[3], frac_cond)
        frac_cells: np.ndarray = np.where(frac_cond)[0]

        # Boundary conditions
        bc_faces = g.get_boundary_faces()
        bc_type = np.array(bc_faces.size * ["neu"])
        bc_type[np.in1d(bc_faces, top_left)] = "dir"
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=list(bc_type))
        bc_values = np.zeros(g.num_faces)
        bc_values[top_left] = 2.5 + y_max  # 2.5 (psi) + y_max (max(zeta))

        # Hydraulic conductivity [cm/s]
        K_SAT = matrix_params["K_sat"] * np.ones(g.num_cells)
        K_SAT[mult_cond] = block_params["K_sat"]
        K_SAT[frac_cells] = fracture_params["K_sat"]

        # Water content at saturation conditions [-]
        THETA_SAT = matrix_params["theta_sat"] * np.ones(g.num_cells)
        THETA_SAT[mult_cond] = block_params["theta_sat"]
        THETA_SAT[frac_cond] = fracture_params["theta_sat"]

        # Water content at residual conditions [-]
        THETA_RES = matrix_params["theta_res"] * np.ones(g.num_cells)
        THETA_RES[mult_cond] = block_params["theta_res"]
        THETA_RES[frac_cond] = fracture_params["theta_res"]

        # van Genuchten alpha parameter [1 / cm]
        ALPHA_VG = matrix_params["alpha_vg"] * np.ones(g.num_cells)
        ALPHA_VG[mult_cond] = block_params["alpha_vg"]
        ALPHA_VG[frac_cond] = fracture_params["alpha_vg"]

        # van Genuchten n parameter [-]
        N_VG = matrix_params["n_vg"] * np.ones(g.num_cells)
        N_VG[mult_cond] = block_params["n_vg"]
        N_VG[frac_cond] = fracture_params["n_vg"]

        # Initialize bulk data
        specified_parameters = {
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


    # %% Set initial states
    for g, d in gb:
        pp.set_state(
            d, state={node_var: -500 + d[pp.PARAMETERS][param_key]["elevation"]}
        )
        pp.set_iterate(d, iterate={node_var: d[pp.STATE][node_var]})

    # %% Create degree of freedom manager and equation manager
    dof_manager = pp.DofManager(gb)
    equation_manager = pp.ad.EquationManager(gb, dof_manager)

    # %% Assign primary variables to their corresponding grids
    h = equation_manager.merge_variables([(g, node_var) for g in [g]])

    # Shorthands. Note that the following merged variables all have different id's
    h_m = h.previous_iteration()
    h_n = h.previous_timestep()

    # Useful auxiliary variables
    zc_ad = pp.ad.Array(g.cell_centers[gb.dim_max() - 1])
    psi: pp.ad.Operator = h - zc_ad  # pressure head (active)
    psi_m: pp.ad.Operator = h_m - zc_ad  # pressure head at prev iter
    psi_n: pp.ad.Operator = h_n - zc_ad  # pressure head at prev time

    # %% Grid operators
    div = pp.ad.Divergence(grids=[g])
    bound = pp.ad.BoundaryCondition(param_key, grids=[g])

    # %% Governing equations

    # Soil water retention curves
    vgm = mdu.VanGenuchtenMualem(gb=gb, param_key=param_key, dof_manager=dof_manager)
    theta_ad: pp.ad.Function = vgm.water_content(as_ad=True)
    krw_ad: pp.ad.Function = vgm.relative_permeability(as_ad=True)
    smc_ad: pp.ad.Function = vgm.moisture_capacity(as_ad=True)

    # MPFA discretization
    mpfa = pp.ad.MpfaAd(param_key, [g])

    # Obtain single-phase flow to compute directionality of upwind scheme
    flux_single_phase = mpfa.flux * h_m + mpfa.bound_flux * bound

    # Upwinding of relative permeabilities
    upwind = mdu.FluxBaseUpwindAd(gb=gb, grid_list=[g], param_key=param_key)
    zf_ad = pp.ad.Array(g.face_centers[gb.dim_max() - 1])
    psi_bc_ad = bound - zf_ad
    psi_bc_ad.set_name("boundary pressure head")
    krw_faces_ad = upwind(krw_ad(psi_m), krw_ad(psi_bc_ad), flux_single_phase)
    krw_faces_ad.discretize(gb)
    krw_faces_ad.evaluate(dof_manager)

    # Multiphase Darcy fluxes
    flux = krw_faces_ad * mpfa.flux * h + krw_faces_ad * mpfa.bound_flux * bound

    # Treatment of source and accumulation terms
    # NOTE: The expression containing the active ad variable (psi_bulk)
    # must be placed at the left of the (final) expression.
    # Otherwise, numpy will broadcast the expression leading to an undesirable output format
    dt_ad = mdu.ParameterScalar(param_key, "time_step", grids=[g])
    source = pp.ad.ParameterArray(param_key, "source", grids=[g])
    mass_matrix = pp.ad.MassMatrixAd(param_key, grids=[g])

    linearization = "modified_picard"
    if linearization == "newton":
        accum_active = mass_matrix.mass * theta_ad(psi)
        accum_inactive = mass_matrix.mass * theta_ad(psi_n) * (-1)
    elif linearization == "modified_picard":
        accum_active = mass_matrix.mass * psi * smc_ad(psi_m)
        accum_inactive = mass_matrix.mass * (
            theta_ad(psi_m) - smc_ad(psi_m) * psi_m - theta_ad(psi_n)
        )
    elif linearization == "l_scheme":
        L = 0.0025
        accum_active = L * mass_matrix.mass * psi
        accum_inactive = mass_matrix.mass * (
            theta_ad(psi_m) - L * psi_m - theta_ad(psi_n)
        )
    else:
        raise NotImplementedError(
            "Linearization scheme not implemented. Use 'newton', "
            "'modified_picard', or 'l_scheme'."
        )

    accumulation = accum_active + accum_inactive
    conservation = accumulation + dt_ad * div * flux - dt_ad * source

    # %% Assemble discrete equations, feed into the equation manager, and discretize.
    eqs = {"conservation": conservation}
    equation_manager.equations.update(eqs)
    equation_manager.discretize(gb)

    # %% Initialize exporter. Note that we use the ghost grid rather than the physical one
    export_counter = 0
    for _, d in gb:
        pp.set_state(d, state={"frac_cells": np.int8(frac_cond)})
        pp.set_iterate(d, iterate={"frac_cells": d[pp.STATE]["frac_cells"]})
    exporter = pp.Exporter(gb, "equidim" + out_fname, out_fname)
    exporter.write_vtu([node_var, "frac_cells"], time_step=export_counter)

    # %% Time loop
    total_iteration_counter: int = 0
    iters: list = []
    ABS_TOL: float = 1e-3
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
            set_iterate_as_state(gb, node_var, "dummy_var")
            continue
        # end of checking

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
        set_state_as_iterate(gb, node_var, "dummy_var")

        # Export if time is in schedule
        if np.isclose(tsc.time, scheduled_time, atol=1e-3):
            export_counter += 1
            exporter.write_vtu([node_var], time_step=export_counter)
            if not np.isclose(tsc.time, tsc.time_final, atol=1e-3):
                scheduled_time = tsc.schedule[export_counter + 1]

        print()