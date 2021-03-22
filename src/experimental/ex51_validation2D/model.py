import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import quadpy as qp
import mdestimates as mde

import mdestimates.estimates_utils as utils


def model(gb, method):
    """
    Runs main model for Validation 5.2 from the paper

    Parameters
    ----------
    gb : PorePy Object
        Grid bucket
    method: String
        Numerical method, e.g.: 'TPFA', 'MPFA', 'RTO', 'MVEM'

    Returns
    -------
    h_max : Float
        Maximum diameter of the grid bucket
    estimate_2d: Float
        Square root of the sum of the square of the local estimates for the 2D domain
    true_error_2d: Float
        Square root of the sum of the square of the true error for the 2D domain
    num_cells_2d: Integer
        Number of 2d cells
    estimate_1d: Float
        Square root of the sum of the square of the local estimates for the12D domain
    true_error_1d: Float
        Square root of the sum of the square of the true error for the 1D domain
    num_cells_1d: Integer
        Number of 1d cells
    estimate_mortar: Float
        Square root of the sum of the square of the local estimates for the interface
    true_error_mortar: Float
        Square root of the sum of the square of the true error for the interface
    num_cells_mortar: Integer
        Number of mortar cells
    global_estimate: Float
        Global estimate as given by the majorant
    true_error: Float
        True error for the whole problem
    I_eff: Float
        Effectivity index, i.e.: global_estimate / true_error

    """

    #%% Method type
    def fv(scheme):
        """
        Checks wheter a numerical method is FV or not

        Parameters
        ----------
        scheme : string
            Numerical method.


        Returns
        -------
        bool
            True if the numerical method is FV, false otherwise.

        """
        if scheme in ["mpfa", "MPFA", "tpfa", "TPFA"]:
            return True
        elif scheme in ["rt0", "RT0", "mvem", "MVEM"]:
            return False
        else:
            raise ("Method unrecognized")

    # Get hold of grids and dictionaries
    g_2d = gb.grids_of_dimension(2)[0]
    g_1d = gb.grids_of_dimension(1)[0]
    h_max = gb.diameter()
    d_2d = gb.node_props(g_2d)
    d_1d = gb.node_props(g_1d)
    d_e = gb.edge_props([g_1d, g_2d])
    mg = d_e["mortar_grid"]

    # Get hold of mesh sizes
    h_1partial2 = 0.5 / g_2d.frac_pairs.shape[1]
    h_1 = 0.5 / g_1d.num_cells
    h_gamma = 0.5 / (mg.num_cells / 2)

    # Mappings
    cell_faces_map, _, _ = sps.find(g_2d.cell_faces)
    cell_nodes_map, _, _ = sps.find(g_2d.cell_nodes())

    # Populate the data dictionaries with pp.STATE
    for g, d in gb:
        pp.set_state(d)

    for e, d in gb.edges():
        mg = d["mortar_grid"]
        pp.set_state(d)

    def get_2d_boundary_indices(g):
        """
        Obtain list of boolean indices for the boundary faces

        Parameters
        ----------
        g : Grid
            2D Grid.

        Raises
        ------
        ValueError
            If the g.dim is different from 2.

        Returns
        -------
        idx : List
            Containing the boolean indices as NumPy boolean arrays.
        """

        # Check dimensionality of the grid
        if g.dim != 2:
            raise ValueError("Dimension should be 2")

        # Obtaining coordinates
        x = g.face_centers

        # Get boundary faces
        bcf = g.get_boundary_faces()

        # Obtaining the boolean indices of the three different regions
        bottom = x[1][bcf] < 0.25
        middle = (x[1][bcf] >= 0.25) & (x[1][bcf] <= 0.75)
        top = x[1][bcf] > 0.75

        # We want only the boundary faces, not all of them
        bottom_bc = bcf[bottom]
        middle_bc = bcf[middle]
        top_bc = bcf[top]

        # Create a list containing the different indices for exporting
        bound_idx_list = [bottom_bc, middle_bc, top_bc]

        return bound_idx_list

    def get_2d_cell_indices(g):
        """
        Get indices of the three regions where the bulk is decomposed

        Parameters
        ----------
        g : Porepy grid
            2D grid

        Raises
        ------
        ValueError
            If the g.dim is different from 2.

        Returns
        -------
        bool_list : List
            List of numpy boolean arrays containing the different regions.
        bool_array : Numpy Array of size g.num_cells
            Array containing the labeling (1 to 3) of the different regions.
        """

        # Check grid dimension
        if g.dim != 2:
            raise ValueError("This function is meant for the 2D bulk")

        # Get cell center coordinates
        x = g.cell_centers

        # Obtaining the boolean indices of the three different regions
        bottom = x[1] < 0.25
        middle = (x[1] >= 0.25) & (x[1] <= 0.75)
        top = x[1] > 0.75

        # Create a list containing the different indices for exporting
        cell_idx_list = [bottom, middle, top]

        # It is useful to assing a label to them, so we con plot them in Paraview
        subregions = 1 * bottom + 2 * middle + 3 * top

        return cell_idx_list, subregions

    # Get the cell center boolean arrays
    cell_idx_list, regions_2d = get_2d_cell_indices(g_2d)

    # Get the boundary faces boolean arrays
    bound_idx_list = get_2d_boundary_indices(g_2d)

    #%% Get analytical expressions
    def get_exact_2d_pressure(g):
        """
        Get exact pressures (symbolic, NumPy, cell-centered) as given by the
        analytical solution of Validation 5.3 from the paper

        Parameters
        ----------
        g : PorePy Grid
            2D grid

        Raises
        ------
        ValueError
            If the grid dimension is different from 2.

        Returns
        -------
        p2d_sym_list : List of length 3
            List of symbolic expressions.
        p2d_numpy_list : List of length 3
            List of Lambda functions as given by p2d_sym_list.
        p2d_cc : NumPy nd-Array of size g_2d.num_cells
            Cell-centered pressures

        """

        # Check grid dimensionality
        if g.dim != 2:
            raise ValueError("Dimension must be 2")

        # Get cell center coordinates and cell center boolean indices
        x2d_cc = g.cell_centers
        cell_idx_list, _ = get_2d_cell_indices(g)

        # Define symbolic symbols
        x, y = sym.symbols("x y")

        # Define the three-dimensional exact form for each subregion. See also the
        # Appendix of the paper.
        p2d_bottom_sym = ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5
        p2d_middle_sym = ((x - 0.5) ** 2) ** (0.5)
        p2d_top_sym = ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5

        # Create a list that contains all symbolic expressions
        p2d_sym_list = [p2d_bottom_sym, p2d_middle_sym, p2d_top_sym]

        # Convert to Numpy expressions
        p2d_numpy_list = []
        for p in p2d_sym_list:
            p2d_numpy_list.append(sym.lambdify((x, y), p, "numpy"))

        # Obtain cell-center pressures
        p2d_cc = np.zeros(g.num_cells)
        for (p, idx) in zip(p2d_numpy_list, cell_idx_list):
            p2d_cc += p(x2d_cc[0], x2d_cc[1]) * idx

        return p2d_sym_list, p2d_numpy_list, p2d_cc

    def get_2d_boundary_values(g, bound_idx_list, p2d_numpy_list):
        """
        Get boundary values (satisfying the exact pressure field) of the 3D domain

        Parameters
        ----------
        g : PorePy grid
            2D grid
        bound_idx_list : List of length 3
            List containing the boolean NumPy nd-arrays (of length g_2d.num_faces)
        p2d_numpy_list : List of length 3
            List containing the exact pressures as NumPy expressions

        Raises
        ------
        ValueError
            If the grid dimension is different from 2.

        Returns
        -------
        bc_values : NumPy nd-array of size g_2d.num_faces
            Boundary values as given by the exact solution.

        """

        # Check if grid is two-dimensional
        if g.dim != 2:
            raise ValueError("Dimension should be 2")

        # Get face-center coordinates
        x = g.face_centers

        # Intialize boundary values array
        bc_values = np.zeros(g.num_faces)

        # Evaluate exact pressure at external boundary faces at each region
        for (p, idx) in zip(p2d_numpy_list, bound_idx_list):
            bc_values[idx] = p(x[0][idx], x[1][idx])

        return bc_values

    def get_exact_2d_pressure_gradient(g, p2d_sym_list):
        """
        Get exact pressure gradient (symbolic, NumPy, and cell-centered) for the 2D domain

        Parameters
        ----------
        g : PorePy grid
            2D grid.
        p2d_sym_list : List of length 3
            Containing the symbolic exact pressures for each subregion.

        Raises
        ------
        ValueError
            If the grid dimension is different from 2.

        Returns
        -------
        gradp2d_sym_list : List of length 3
            Containing the symbolic exact pressure gradient expressions
        gradp2d_numpy_list : List of length 3
            Containing the exact NumPy expressions for the pressure gradient.
        gradp2d_cc : NumPy nd-array of size g_2d.num_cells
            Cell-centered evaluated exact pressure gradient.

        """

        # Check dimensionality of the grid
        if g.dim != 2:
            raise ValueError("Dimension must be 2")

        # Define symbolic symbols
        x, y = sym.symbols("x y")

        # Get cell center coordinates, and cell center boolean indices
        x2d_cc = g.cell_centers
        cell_idx_list, _ = get_2d_cell_indices(g)

        # Obtain gradient of the pressures
        gradp2d_sym_list = []
        for p in p2d_sym_list:
            gradp2d_sym_list.append([sym.diff(p, x), sym.diff(p, y)])

        # Convert to Numpy expressions
        gradp2d_numpy_list = []
        for gradp in gradp2d_sym_list:
            gradp2d_numpy_list.append(
                [
                    sym.lambdify((x, y), gradp[0], "numpy"),
                    sym.lambdify((x, y), gradp[1], "numpy"),
                ]
            )

        # Obtain cell-centered pressure gradients
        gradpx_cc = np.zeros(g.num_cells)
        gradpy_cc = np.zeros(g.num_cells)
        gradpz_cc = np.zeros(g.num_cells)
        for (gradp, idx) in zip(gradp2d_numpy_list, cell_idx_list):
            gradpx_cc += gradp[0](x2d_cc[0], x2d_cc[1]) * idx
            gradpy_cc += gradp[1](x2d_cc[0], x2d_cc[1]) * idx
        gradp2d_cc = np.array([gradpx_cc, gradpy_cc, gradpz_cc])

        return gradp2d_sym_list, gradp2d_numpy_list, gradp2d_cc

    def get_exact_2d_velocity(g, gradp2d_list):
        """
        Get exact velocity (symbolic, NumPy, and cell-centered) for the 2D domain

        Parameters
        ----------
        g : PorePy grid
            2D grid.
        gradp2d_list : List of length 3
            Containing the symbolic exact pressure gradient.

        Raises
        ------
        ValueError
            If the grid dimension is different from 2.

        Returns
        -------
        u2d_sym_list : List of length 3
            Containing the exact symbolic expressions for the velocity.
        u2d_numpy_list : List of length 3
            Containing the exact NumPy expressions for the velocity.
        u2d_cc : NumPy nd-Array of size g_2d.num_cells
            Containing the cell-centered evaluated exact velocity.

        """

        # Check dimensionality of the grid
        if g.dim != 2:
            raise ValueError("Dimension must be 2")

        # Define symbolic symbols
        x, y = sym.symbols("x y")

        # Get cell center coordinates, and cell center boolean indices
        x2d_cc = g.cell_centers
        cell_idx_list, _ = get_2d_cell_indices(g)

        # Obtain velocities
        u2d_sym_list = []
        for gradp in gradp2d_list:
            u2d_sym_list.append([-gradp[0], -gradp[1]])

        # Convert to Numpy expressions
        u2d_numpy_list = []
        for u in u2d_sym_list:
            u2d_numpy_list.append(
                [
                    sym.lambdify((x, y), u[0], "numpy"),
                    sym.lambdify((x, y), u[1], "numpy"),
                ]
            )

        # Obtain cell-centered pressure gradients
        ux_cc = np.zeros(g.num_cells)
        uy_cc = np.zeros(g.num_cells)
        uz_cc = np.zeros(g.num_cells)
        for (u, idx) in zip(u2d_numpy_list, cell_idx_list):
            ux_cc += u[0](x2d_cc[0], x2d_cc[1]) * idx
            uy_cc += u[1](x2d_cc[0], x2d_cc[1]) * idx
        u2d_cc = np.array([ux_cc, uy_cc, uz_cc])

        return u2d_sym_list, u2d_numpy_list, u2d_cc

    def get_exact_2d_source_term(g, u2d_sym_list):
        """
        Get exact source term (satisfying the mass conservation equation) for the
        2D domain.

        Parameters
        ----------
        g : PorePy grid
            2D grid.
        u2d_sym_vel : List of length 3
            Containing the exact symbolic velocities for each subregion.

        Raises
        ------
        ValueError
            If the dimensionality of the grid is different from 2

        Returns
        -------
        f2d_sym_list : List of length 3
            Containing the exact symbolic source term.
        f2d_numpy_list : List of length 3
            Containing the exact NumPy expressions for the source term.
        f2d_cc : NumPy nd-array of size g_2d.num_cells
            Exact cell-centered evaluated source terms.

        """

        # Check grid dimensionality
        if g.dim != 2:
            raise ValueError("Dimension must be 2")

        # Define symbolic symbols
        x, y = sym.symbols("x y")

        # Get cell center coordinates, and cell center boolean indices
        x2d_cc = g.cell_centers
        cell_idx_list, _ = get_2d_cell_indices(g)

        # Obtain source term
        f2d_sym_list = []
        for u in u2d_sym_list:
            f2d_sym_list.append((sym.diff(u[0], x) + sym.diff(u[1], y)))

        # Convert to Numpy expressions
        f2d_numpy_list = []
        for f in f2d_sym_list:
            f2d_numpy_list.append(sym.lambdify((x, y), f, "numpy"))

        # Obtain cell-center source terms
        f2d_cc = np.zeros(g.num_cells)
        for (f, idx) in zip(f2d_numpy_list, cell_idx_list):
            f2d_cc += f(x2d_cc[0], x2d_cc[1]) * idx

        return f2d_sym_list, f2d_numpy_list, f2d_cc

    # Get hold of exact pressures
    p2d_sym_list, p2d_numpy_list, p2d_cc = get_exact_2d_pressure(g_2d)

    # Get hold of exact pressure gradients
    gradp2d_sym_list, gradp2d_numpy_list, gradp2d_cc = get_exact_2d_pressure_gradient(
        g_2d, p2d_sym_list
    )

    # Get hold of exact velocity
    u2d_sym_list, u2d_numpy_list, u2d_cc = get_exact_2d_velocity(g_2d, gradp2d_sym_list)

    # Get hold of exact source term
    f2d_sym_list, f2d_numpy_list, f2d_cc = get_exact_2d_source_term(g_2d, u2d_sym_list)

    # Get hold of external boundary values (#TEST IF WE'RE RETRIEVING THE RIGHT VALUES)
    bc_vals_2d = get_2d_boundary_values(g_2d, bound_idx_list, p2d_numpy_list)

    #%% Obtain integrated source terms
    def integrate_source_2d(g, f2d_numpy_list, cell_idx_list):
        """
        Computes the exact integral of the source term for the 2D domain

        Parameters
        ----------
        g : PorePy grid
            2D grid.
        f2d_numpy_list : List of length 3
            Containing the exact NumPy expressions for the source term.
        cell_idx_list : List of length 3
            Containing the boolean indices for the cells of the 2D domain

        Returns
        -------
        integral : NumPy nd-Array of size g_2d.num_cells
            Integral of the source term

        """

        # Declare integration method and get hold of elements in QuadPy format
        int_method = qp.t2.get_good_scheme(3)  # a scheme of degree 3 should be enough
        elements = utils.get_quadpy_elements(g, g)

        # We now declare the different integrand regions
        integral = np.zeros(g.num_cells)
        for (f, idx) in zip(f2d_numpy_list, cell_idx_list):

            # Declare integrand
            def integrand(x):
                return f(x[0], x[1])

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return integral

    # Obtain the integrated source terms
    integrated_f2d = integrate_source_2d(g_2d, f2d_numpy_list, cell_idx_list)

    #%% Obtain numerical solution
    parameter_keyword = "flow"
    max_dim = gb.dim_max()

    # Set parameters in the subdomains
    for g, d in gb:

        # Get hold of boundary faces and declare bc-type. We assign Dirichlet
        # bc to the bulk, and no-flux for the 2D fracture
        bc_faces = g.get_boundary_faces()
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        specified_parameters = {"bc": bc}

        # Also set the values - specified as vector of size g.num_faces
        bc_vals = np.zeros(g.num_faces)
        if g.dim == max_dim:
            bc_vals = bc_vals_2d
        specified_parameters["bc_values"] = bc_vals

        # (Integrated) source terms are given by the exact solution
        if g.dim == max_dim:
            source_term = integrated_f2d
        else:
            source_term = -2 * g.cell_volumes

        specified_parameters["source"] = source_term

        # Initialize default data
        pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)

    # Next loop over the edges
    for e, d in gb.edges():

        # Set the normal diffusivity
        data = {"normal_diffusivity": 1}

        # Add parameters: We again use keywords to identify sets of parameters.
        mg = d["mortar_grid"]
        pp.initialize_data(mg, d, parameter_keyword, data)

    # Discretize model according to the numerical method
    if method in ["mpfa", "MPFA"]:
        subdomain_discretization = pp.Mpfa(keyword=parameter_keyword)
    elif method in ["tpfa", "TPFA"]:
        subdomain_discretization = pp.Tpfa(keyword=parameter_keyword)
    elif method in ["rt0", "RT0"]:
        subdomain_discretization = pp.RT0(keyword=parameter_keyword)
    elif method in ["mvem", "MVEM"]:
        subdomain_discretization = pp.MVEM(keyword=parameter_keyword)
    else:
        raise ValueError("Method not implemented")

    # Discretize source term according to the method family
    if fv(method):
        source_discretization = pp.ScalarSource(keyword=parameter_keyword)
    else:
        source_discretization = pp.DualScalarSource(keyword=parameter_keyword)

    # Define keywords
    subdomain_variable = "pressure"
    flux_variable = "flux"
    subdomain_operator_keyword = "diffusion"
    edge_discretization = pp.RobinCoupling(
        parameter_keyword, subdomain_discretization, subdomain_discretization
    )
    edge_variable = "interface_flux"
    coupling_operator_keyword = "interface_diffusion"

    # Loop over all subdomains in the GridBucket
    if fv(method):  # FV methods
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 0}}
            d[pp.DISCRETIZATION] = {
                subdomain_variable: {
                    subdomain_operator_keyword: subdomain_discretization,
                    "source": source_discretization,
                }
            }
    else:  # FEM methods
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {
                subdomain_variable: {
                    subdomain_operator_keyword: subdomain_discretization,
                    "source": source_discretization,
                }
            }

    # Next, loop over the edges
    for e, d in gb.edges():
        # Get the grids of the neighboring subdomains
        g1, g2 = gb.nodes_of_edge(e)
        # The interface variable has one degree of freedom per cell in the mortar grid
        d[pp.PRIMARY_VARIABLES] = {edge_variable: {"cells": 1}}

        # The coupling discretization links an edge discretization with variables
        d[pp.COUPLING_DISCRETIZATION] = {
            coupling_operator_keyword: {
                g1: (subdomain_variable, subdomain_operator_keyword),
                g2: (subdomain_variable, subdomain_operator_keyword),
                e: (edge_variable, edge_discretization),
            }
        }

    # Assemble, solve, and distribute variables
    assembler = pp.Assembler(gb)
    assembler.discretize()
    A, b = assembler.assemble_matrix_rhs()
    sol = sps.linalg.spsolve(A, b)
    assembler.distribute_variable(sol)

    # Overwrite d[pp.STATE][subdomain_variable] to be consistent with FEM methods
    for g, d in gb:
        discr = d[pp.DISCRETIZATION][subdomain_variable][subdomain_operator_keyword]
        pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable], d).copy()
        flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d).copy()
        d[pp.STATE][subdomain_variable] = pressure
        d[pp.STATE][flux_variable] = flux

    #%% Obtain error estimates (and transfer them to d[pp.STATE])
    estimates = mde.ErrorEstimate(gb, lam_name=edge_variable)
    estimates.estimate_error()
    estimates.transfer_error_to_state()
    majorant = estimates.get_majorant()
    error_estimate_2d = estimates.get_local_errors(g_2d, d_2d)
    error_estimate_1d = estimates.get_local_errors(g_1d, d_1d)
    error_estimate_mortar = estimates.get_local_errors(mg, d_e)
    estimates.print_summary(scaled=False)

    #%% Evaluate reconstructed quantities
    def get_cc_reconp(estimates, cell_idx_list):
        """
        Get hold of cell-centered evaluated reconstructed pressures

        Parameters
        ----------
        estimates : Error Estimates Object
            Error estimates object after mde.ErrorEstimate() has been applied
        cell_idx_list : List of length 3
            Containing the boolean cell indices of the subregions of the 3D domain

        Returns
        -------
        NumPy nd-Array
            Cell-centered evaluated reconstructed pressure of the 2D domain.
        NumPy nd-Array
            Cell-centered evaluated reconstructed pressure of the 1D domain.

        """

        # Get hold of estimates keyword
        kw_e = estimates.estimates_kw

        for g, d in gb:

            # Get hold of reconstructed pressure
            recon_p = d[kw_e]["recon_p"].copy()
            p = utils.poly2col(recon_p)

            # Obtain cell-centered coordinates
            x = g.cell_centers

            # Evaluate the 3D-reconstructed pressure
            if g.dim == 2:
                rp_cc_2d = np.zeros([g.num_cells, 1])
                for idx in cell_idx_list:
                    rp_cc_2d += (
                        p[0] * x[0].reshape(g.num_cells, 1)
                        + p[1] * x[1].reshape(g.num_cells, 1)
                        + p[2]
                    ) * idx.reshape(g.num_cells, 1)
            # Evaluate the 2D-reconstructed pressure
            else:
                rp_cc_1d = p[0] * x[0].reshape(g.num_cells, 1) + p[1]

        return rp_cc_2d.flatten(), rp_cc_1d.flatten()

    def get_cc_reconvel(estimates, cell_idx_list):
        """
        Get hold of cell-centered evaluated reconstructed velocities

        Parameters
        ----------
        estimates : Error Estimates Object
            Error estimates object after mde.ErrorEstimate() has been applied
        cell_idx_list : List of length 3
            Containing the boolean cell indices of the subregions of the 2D domain

        Returns
        -------
        NumPy nd-Array
            Cell-centered evaluated reconstructed velocity of the 2D domain.
        NumPy nd-Array
            Cell-centered evaluated reconstructed velocity of the 1D domain.

        """

        # Get hold of estimates keyword
        kw_e = estimates.estimates_kw

        for g, d in gb:

            # Get hold of reconstructed pressure
            recon_u = d[kw_e]["recon_u"].copy()
            u = utils.poly2col(recon_u)

            # Obtain cell-centered coordinates
            x = g.cell_centers

            # Evaluate the 2D-reconstructed pressure
            if g.dim == 2:
                ru_cc_2d_x = np.zeros([g.num_cells, 1])
                ru_cc_2d_y = np.zeros([g.num_cells, 1])
                ru_cc_2d_z = np.zeros([g.num_cells, 1])

                for idx in cell_idx_list:
                    ru_cc_2d_x += (
                        u[0] * x[0].reshape(g.num_cells, 1) + u[1]
                    ) * idx.reshape(g.num_cells, 1)
                    ru_cc_2d_y += (
                        u[0] * x[1].reshape(g.num_cells, 1) + u[2]
                    ) * idx.reshape(g.num_cells, 1)

                ru_cc_2d = np.array(
                    [ru_cc_2d_x.flatten(), ru_cc_2d_y.flatten(), ru_cc_2d_z.flatten()]
                )

            # Evaluate the 1D-reconstructed pressure
            else:
                ru_cc_1d_x = (u[0] * x[0].reshape(g.num_cells, 1) + u[1]).flatten()
                ru_cc_1d_y = np.zeros(g.num_cells)
                ru_cc_1d_z = np.zeros(g.num_cells)
                ru_cc_1d = np.array([ru_cc_1d_x, ru_cc_1d_y, ru_cc_1d_z])

        return ru_cc_2d, ru_cc_1d

    # Get hold of cell-centered reconstructed pressure for the 2D and 1D domain
    reconp_cc_2d, reconp_cc_1d = get_cc_reconp(estimates, cell_idx_list)

    # Get hold of cell-centered reconstructed velocity for the 3D and 2D domain
    reconu_cc_2d, reconu_cc_1d = get_cc_reconvel(estimates, cell_idx_list)

    #%% Compute true errors for the pressure, i.e., ||| p - p_h |||
    def compute_pressure_2d_true_error(
        g, d, estimates, gradp2d_numpy_list, cell_idx_list
    ):
        """
        Computes true "pressure" error for the 2D subdomain

        Parameters
        ----------
        g : PorePy Grid
            Two-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 2D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.
        gradp2d_numpy_list : List
            List of NumPy lambda functions for each subregion.
        cell_idx_list : List
            List of Numpy boolean array for each subregion.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 2.

        Returns
        -------
        integral: NumPy array of size g_2d.num_cells
            (Squared) of the true errors for each element of the grid.

        """

        # Check if dimension is 2
        if g.dim != 2:
            raise ValueError("Dimension should be 2")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_p = d[kwe]["recon_p"].copy()
        pr = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(3)
        elements = utils.get_quadpy_elements(g, g)

        # Compute the true error for each subregion
        integral = np.zeros(g.num_cells)

        for (gradp, idx) in zip(gradp2d_numpy_list, cell_idx_list):

            # Declare integrand and add subregion contribution
            def integrand(x):
                gradp_exact_x = gradp[0](x[0], x[1])
                gradp_exact_y = gradp[1](x[0], x[1])

                gradp_recon_x = pr[0] * np.ones_like(x[0])
                gradp_recon_y = pr[1] * np.ones_like(x[1])

                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2

                return int_x + int_y

            integral += method.integrate(integrand, elements) * idx

        return integral

    def compute_pressure_1d_true_error(g, d, estimates):
        """
        Computes true "pressure" error for the 1D subdomain (the fracture)

        Parameters
        ----------
        g : PorePy Grid
            One-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 1D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 1.

        Returns
        -------
        integral: NumPy nd-Array of size g_1d.num_cells
            (Squared) of the true errors for each element of the grid.
        """

        # Check if dimension is 1
        if g.dim != 1:
            raise ValueError("Dimension should be 1")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_p = d[kwe]["recon_p"].copy()
        pr = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.c1.newton_cotes_closed(4)
        g_rot = utils.rotate_embedded_grid(g)
        elements = utils.get_quadpy_elements(g, g_rot)

        # Compute the true error
        def integrand(x):

            gradp_exact_x = np.zeros_like(x[0])

            gradp_recon_x = pr[0] * np.ones_like(x[0])

            int_x = (gradp_exact_x - gradp_recon_x) ** 2

            return int_x

        integral = method.integrate(integrand, elements)

        return integral

    def compute_pressure_mortar_true_error(d_e, estimates):
        """
        Computes the true "pressure" error for the mortar grid

        Parameters
        ----------
        d_e : Dictionary
            Dictionary of the interface
        estimates: Error estimate object
            Error estimate object as obtained with mde.ErrorEstimate()

        Raises
        ------
        ValueError
            If the dimension of the mortar grid is different from 2.

        Returns
        -------
        true_error_mortar: NumPy nd-array of shape (mg.num_cells, 1)
            True error (squared) for each element of the mortar grid.

        """

        # Import functions
        from mdestimates._error_evaluation import (
            _sorted_highdim_edge_grid,
            _sorted_side_grid,
            _sorted_low_grid,
            _merge_grids,
            _get_grid_uniongrid_elements,
            _project_poly_to_merged_grid,
        )

        """
        Computes the diffusive error (squared) for the entire mortar grid. This
        function should be used when there exists a non-matching coupling between
        the grids and the interfaces. If the coupling involves matching grids,
        interface_diffusive_error_1d() should be used. However, in principle, the
        output should be the same.

        Parameters
        ----------
        e : PorePy object
            Edge.
        d_e : Dictionary
            Interface dictionary.

        Returns
        -------
        diffusive_error: NumPy Array of size mg.num_cells
            Diffusive error (squared) incurred on each mortar cell.

        """

        # Get hold of grids and dictionaries
        g_l, g_h = estimates.gb.nodes_of_edge(e)
        mg = d_e["mortar_grid"]

        # Obtain the number of sides of the mortar grid
        num_sides = mg.num_sides()
        if num_sides == 2:
            sides = [-1, 1]
        else:
            sides = [1]

        # Loop over the sides of the mortar grid
        true_error = np.zeros(mg.num_cells)

        for side in sides:

            # Get rotated grids and sorted elements
            high_grid, frac_faces = _sorted_highdim_edge_grid(g_h, g_l, mg, side)
            mortar_grid, mortar_cells = _sorted_side_grid(mg, g_l, side)
            low_grid, low_cells = _sorted_low_grid(g_l)

            # Merge the three grids into one
            merged_grid = _merge_grids(low_grid, mortar_grid, high_grid)

            # Note that the following mappings are local for each merged grid.
            # For example, to retrieve the global fracture faces indices, we should
            # write frac_faces[merged_high_ele], and to retrieve the global mortar
            # cells, we should write mortar_cells[merged_mortar_ele]
            # Retrieve element mapping from sorted grids to merged grid
            merged_high_ele = _get_grid_uniongrid_elements(merged_grid, high_grid)
            merged_mortar_ele = _get_grid_uniongrid_elements(merged_grid, mortar_grid)
            merged_low_ele = _get_grid_uniongrid_elements(merged_grid, low_grid)

            # Get projected pressure jump, normal permeabilities, and normal velocities
            pressure_jump, k_perp, _ = _project_poly_to_merged_grid(
                estimates,
                e,
                d_e,
                [low_cells, mortar_cells, frac_faces],
                [merged_low_ele, merged_mortar_ele, merged_high_ele],
            )

            # Define integration method and obtain quadpy elements
            method = qp.c1.newton_cotes_closed(4)
            qp_ele = utils.get_qp_elements_from_union_grid_1d(merged_grid)

            # Define integrand
            def integrand(x):
                coors = x[np.newaxis, :, :]  # this is needed for 1D grids
                p_jump = utils.eval_P1(pressure_jump, coors)  # eval pressure jump
                return (k_perp ** (-0.5) + k_perp ** (0.5) * p_jump) ** 2

            # Evaluate integral
            diffusive_error_merged = method.integrate(integrand, qp_ele)

            # Sum errors corresponding to a mortar cell
            diffusive_error_side = np.zeros(len(mortar_cells))
            for mortar_element in range(len(mortar_cells)):
                idx = mortar_cells[mortar_element] == mortar_cells[merged_mortar_ele]
                diffusive_error_side[mortar_element] = diffusive_error_merged[idx].sum()

            # Append into the list
            true_error[mortar_cells] = diffusive_error_side

        return true_error

    #%% Compute true errors for the velocity, i.e., ||| u - u_h |||_*
    def compute_velocity_2d_true_error(g, d, estimates, u2d_numpy_list, cell_idx_list):
        """
        Computes true "velocity" error for the 2D subdomain

        Parameters
        ----------
        g : PorePy Grid
            Two-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 2D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.
        u2d_numpy_list : List
            List of NumPy lambda functions for each subregion.
        cell_idx_list : List
            List of Numpy boolean array for each subregion.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 2.

        Returns
        -------
        integral: NumPy array of size g_2d.num_cells
            (Squared) of the true errors for each element of the grid.

        """

        # Check if dimension is 2
        if g.dim != 2:
            raise ValueError("Dimension should be 2")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of numerical velocities and create list of coefficients
        recon_u = d[kwe]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(3)
        elements = utils.get_quadpy_elements(g, g)

        # Compute the true error for each subregion
        integral = np.zeros(g.num_cells)

        for (vel, idx) in zip(u2d_numpy_list, cell_idx_list):

            # Declare integrand and add subregion contribution
            def integrand(x):
                vel_exact_x = vel[0](x[0], x[1])
                vel_exact_y = vel[1](x[0], x[1])

                vel_recon_x = u[0] * x[0] + u[1]
                vel_recon_y = u[0] * x[1] + u[2]

                int_x = (vel_exact_x - vel_recon_x) ** 2
                int_y = (vel_exact_y - vel_recon_y) ** 2

                return int_x + int_y

            integral += method.integrate(integrand, elements) * idx

        return integral

    def compute_velocity_1d_true_error(g, d, estimates):
        """
        Computes true "velocity" error for the 1D subdomain (the fracture)

        Parameters
        ----------
        g : PorePy Grid
            Two-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 2D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 1.

        Returns
        -------
        integral: NumPy array of size g_1d.num_cells
            (Squared) of the true errors for each element of the grid.

        """

        # Check if dimension if 1
        if g.dim != 1:
            raise ValueError("Dimension should be 1")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of approximated velocities and create list of coefficients
        recon_u = d[kwe]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.c1.newton_cotes_closed(4)
        g_rot = utils.rotate_embedded_grid(g)
        elements = utils.get_quadpy_elements(g, g_rot)

        # Compute the true error
        def integrand(x):

            vel_exact_x = np.zeros_like(x[0])

            vel_recon_x = u[0] * x + u[1]

            int_x = (vel_exact_x - vel_recon_x) ** 2

            return int_x

        integral = method.integrate(integrand, elements)

        return integral

    def compute_velocity_mortar_true_error(d_e, estimates):
        """
        Computes the true "velocity" error for the mortar grid

        Parameters
        ----------
        d_e : Dictionary
            Dictionary of the interface
        estimates: Error estimate object
            Error estimate object as obtained with mde.ErrorEstimate()

        Raises
        ------
        ValueError
            If the dimension of the mortar grid is different from 1.

        Returns
        -------
        true_error_mortar: NumPy nd-array of shape (mg.num_cells, 1)
            True error (squared) for each element of the mortar grid.

        """

        # Get mortar grid
        mg = d_e["mortar_grid"]

        # Sanity check
        if mg.dim != 1:
            raise ValueError("Mortar grid must be one-dimensional")

        V = mg.cell_volumes
        lmbda = d_e[pp.STATE][estimates.lam_name].copy()
        mortar_flux = lmbda / V
        lbmda_diff = (1.0 - mortar_flux) ** 2
        true_error_mortar = lbmda_diff * V

        return true_error_mortar

    #%% Obtain true errors

    # Pressure true errors
    true_pressure_error_2d = compute_pressure_2d_true_error(
        g_2d, d_2d, estimates, gradp2d_numpy_list, cell_idx_list
    )
    true_pressure_error_1d = compute_pressure_1d_true_error(g_1d, d_1d, estimates)
    true_pressure_error_mortar = compute_pressure_mortar_true_error(d_e, estimates)

    # Velocity true errors
    true_velocity_error_2d = compute_velocity_2d_true_error(
        g_2d, d_2d, estimates, u2d_numpy_list, cell_idx_list
    )
    true_velocity_error_1d = compute_velocity_1d_true_error(g_1d, d_1d, estimates)
    true_velocity_error_mortar = compute_velocity_mortar_true_error(d_e, estimates)

    #%% Compute effectivity index
    true_pressure_error = np.sqrt(
        true_pressure_error_2d.sum()
        + true_pressure_error_1d.sum()
        + true_pressure_error_mortar.sum()
    )

    true_velocity_error = np.sqrt(
        true_velocity_error_2d.sum()
        + true_velocity_error_1d.sum()
        + true_velocity_error_mortar.sum()
    )

    I_eff_p = majorant / true_pressure_error
    I_eff_u = majorant / true_velocity_error
    I_eff_combined = (3 * majorant) / (true_pressure_error + true_velocity_error)

    #%% Return
    return (
        h_max,
        error_estimate_2d,
        true_pressure_error_2d.sum() ** 0.5,
        true_velocity_error_2d.sum() ** 0.5,
        h_1partial2,
        error_estimate_1d,
        true_pressure_error_1d.sum() ** 0.5,
        true_velocity_error_1d.sum() ** 0.5,
        h_1,
        error_estimate_mortar,
        true_pressure_error_mortar.sum() ** 0.5,
        true_velocity_error_mortar.sum() ** 0.5,
        h_gamma,
        majorant,
        true_pressure_error.sum(),
        true_velocity_error.sum(),
        I_eff_p,
        I_eff_u,
        I_eff_combined,
    )
