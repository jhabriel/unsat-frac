# Importing modules
import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import quadpy as qp
import mdestimates as mde

import mdestimates.estimates_utils as utils

# Main function
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
    estimate_3d: Float
        Square root of the sum of the square of the local estimates for the 3D domain
    true_error_3d: Float
        Square root of the sum of the square of the true error for the 3D domain
    num_cells_3d: Integer
        Number of 3d cells
    estimate_2d: Float
        Square root of the sum of the square of the local estimates for the 2D domain
    true_error_2d: Float
        Square root of the sum of the square of the true error for the 2D domain
    num_cells_2d: Integer
        Number of 2d cells
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
    g_3d = gb.grids_of_dimension(3)[0]
    g_2d = gb.grids_of_dimension(2)[0]
    h_max = gb.diameter()
    d_3d = gb.node_props(g_3d)
    d_2d = gb.node_props(g_2d)
    d_e = gb.edge_props([g_2d, g_3d])
    mg = d_e["mortar_grid"]

    # Mappings
    cell_faces_map, _, _ = sps.find(g_3d.cell_faces)
    cell_nodes_map, _, _ = sps.find(g_3d.cell_nodes())

    # Populate the data dictionaries with pp.STATE
    for g, d in gb:
        pp.set_state(d)

    for e, d in gb.edges():
        mg = d["mortar_grid"]
        pp.set_state(d)

    def get_3d_boundary_indices(g):
        """
        Obtain list of boolean indices for the boundary faces

        Parameters
        ----------
        g : Grid
            3D Grid.

        Raises
        ------
        ValueError
            If the g.dim is different from 3.

        Returns
        -------
        idx : List
            Containing the boolean indices as NumPy boolean arrays.
        """

        # Check dimensionality of the grid
        if g.dim != 3:
            raise ValueError("Dimension should be 3")

        # Obtaining coordinates
        x = g.face_centers

        # Get boundary faces
        bcf = g.get_boundary_faces()

        # Obtaining the boolean indices of the nine different regions
        bottom_front = (x[1][bcf] < 0.25) & (x[2][bcf] < 0.25)
        bottom_middle = (x[1][bcf] < 0.25) & (x[2][bcf] > 0.25) & (x[2][bcf] < 0.75)
        bottom_back = (x[1][bcf] < 0.25) & (x[2][bcf] > 0.75)
        front = (x[1][bcf] > 0.25) & (x[1][bcf] < 0.75) & (x[2][bcf] < 0.25)
        middle = (
            (x[1][bcf] >= 0.25)
            & (x[1][bcf] <= 0.75)
            & (x[2][bcf] >= 0.25)
            & (x[2][bcf] <= 0.75)
        )
        back = (x[1][bcf] > 0.25) & (x[1][bcf] < 0.75) & (x[2][bcf] > 0.75)
        top_front = (x[1][bcf] > 0.75) & (x[2][bcf] < 0.25)
        top_middle = (x[1][bcf] > 0.75) & (x[2][bcf] > 0.25) & (x[2][bcf] < 0.75)
        top_back = (x[1][bcf] > 0.75) & (x[2][bcf] > 0.75)

        # We want only the boundary faces, not all of them
        bottom_front_bc = bcf[bottom_front]
        bottom_middle_bc = bcf[bottom_middle]
        bottom_back_bc = bcf[bottom_back]
        front_bc = bcf[front]
        middle_bc = bcf[middle]
        back_bc = bcf[back]
        top_front_bc = bcf[top_front]
        top_middle_bc = bcf[top_middle]
        top_back_bc = bcf[top_back]

        # Create a list containing the different indices for exporting
        bound_idx_list = [
            bottom_front_bc,
            bottom_middle_bc,
            bottom_back_bc,
            front_bc,
            middle_bc,
            back_bc,
            top_front_bc,
            top_middle_bc,
            top_back_bc,
        ]

        return bound_idx_list

    def get_3d_cell_indices(g):
        """
        Get indices of the nine regions where the bulk is decomposed

        Parameters
        ----------
        g : Porepy grid
            3D grid

        Raises
        ------
        ValueError
            If the g.dim is different from 3.

        Returns
        -------
        bool_list : List
            List of numpy boolean arrays containing the different regions.
        bool_array : Numpy Array of size g.num_cells
            Array containing the labeling (1 to 9) of the different regions.
        """

        # Check grid dimension
        if g.dim != 3:
            raise ValueError("This function is meant for the 3D bulk")

        # Get cell center coordinates
        x = g.cell_centers

        # Obtaining the boolean indices of the nine different regions
        bottom_front = (x[1] < 0.25) & (x[2] < 0.25)
        bottom_middle = (x[1] < 0.25) & (x[2] > 0.25) & (x[2] < 0.75)
        bottom_back = (x[1] < 0.25) & (x[2] > 0.75)
        front = (x[1] > 0.25) & (x[1] < 0.75) & (x[2] < 0.25)
        middle = (x[1] >= 0.25) & (x[1] <= 0.75) & (x[2] >= 0.25) & (x[2] <= 0.75)
        back = (x[1] > 0.25) & (x[1] < 0.75) & (x[2] > 0.75)
        top_front = (x[1] > 0.75) & (x[2] < 0.25)
        top_middle = (x[1] > 0.75) & (x[2] > 0.25) & (x[2] < 0.75)
        top_back = (x[1] > 0.75) & (x[2] > 0.75)

        # Create a list containing the different indices for exporting
        cell_idx_list = [
            bottom_front,
            bottom_middle,
            bottom_back,
            front,
            middle,
            back,
            top_front,
            top_middle,
            top_back,
        ]

        # It is useful to assing a label to them, so we con plot them in Paraview
        subregions = (
            1 * bottom_front
            + 2 * bottom_middle
            + 3 * bottom_back
            + 4 * front
            + 5 * middle
            + 6 * back
            + 7 * top_front
            + 8 * top_middle
            + 9 * top_back
        )

        return cell_idx_list, subregions

    # Get the cell center boolean arrays
    cell_idx_list, regions_3d = get_3d_cell_indices(g_3d)

    # Get the boundary faces boolean arrays
    bound_idx_list = get_3d_boundary_indices(g_3d)

    #%% Get analytical expressions
    def get_exact_3d_pressure(g):
        """
        Get exact pressures (symbolic, NumPy, cell-centered) as given by the
        analytical solution of Validation 5.3 from the paper

        Parameters
        ----------
        g : PorePy Grid
            3D grid

        Raises
        ------
        ValueError
            If the grid dimension is different from 3.

        Returns
        -------
        p3d_sym_list : List of length 9
            List of symbolic expressions.
        p3d_numpy_list : List of length 9
            List of Lambda functions as given by p3d_sym_list.
        p3d_cc : NumPy nd-Array of size g_3d.num_cells
            Cell-centered pressures

        """

        # Check grid dimensionality
        if g.dim != 3:
            raise ValueError("Dimension must be 3")

        # Get cell center coordinates and cell center boolean indices
        x3d_cc = g.cell_centers
        cell_idx_list, _ = get_3d_cell_indices(g)

        # Define symbolic symbols
        x, y, z = sym.symbols("x y z")

        # Define the three-dimensional exact form for each subregion. See also the
        # Appendix of the paper.
        p3d_bottom_front_sym = ((x - 0.5) ** 2 + (y - 0.25) ** 2 + (z - 0.25) ** 2) ** (
            0.5
        )
        p3d_bottom_middle_sym = ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** (0.5)
        p3d_bottom_back_sym = ((x - 0.5) ** 2 + (y - 0.25) ** 2 + (z - 0.75) ** 2) ** (
            0.5
        )
        p3d_front_sym = ((x - 0.5) ** 2 + (z - 0.25) ** 2) ** (0.5)
        p3d_middle_sym = ((x - 0.5) ** 2) ** (0.5)
        p3d_back_sym = ((x - 0.5) ** 2 + (z - 0.75) ** 2) ** (0.5)
        p3d_top_front_sym = ((x - 0.5) ** 2 + (y - 0.75) ** 2 + (z - 0.25) ** 2) ** (
            0.5
        )
        p3d_top_middle_sym = ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** (0.5)
        p3d_top_back_sym = ((x - 0.5) ** 2 + (y - 0.75) ** 2 + (z - 0.75) ** 2) ** (0.5)

        # Create a list that contains all symbolic expressions
        p3d_sym_list = [
            p3d_bottom_front_sym,
            p3d_bottom_middle_sym,
            p3d_bottom_back_sym,
            p3d_front_sym,
            p3d_middle_sym,
            p3d_back_sym,
            p3d_top_front_sym,
            p3d_top_middle_sym,
            p3d_top_back_sym,
        ]

        # Convert to Numpy expressions
        p3d_numpy_list = []
        for p in p3d_sym_list:
            p3d_numpy_list.append(sym.lambdify((x, y, z), p, "numpy"))

        # Obtain cell-center pressures
        x3d_cc = g.cell_centers
        p3d_cc = np.zeros(g.num_cells)
        for (p, idx) in zip(p3d_numpy_list, cell_idx_list):
            p3d_cc += p(x3d_cc[0], x3d_cc[1], x3d_cc[2]) * idx

        return p3d_sym_list, p3d_numpy_list, p3d_cc

    def get_3d_boundary_values(g, bound_idx_list, p3d_numpy_list):
        """
        Get boundary values (satisfying the exact pressure field) of the 3D domain

        Parameters
        ----------
        g : PorePy grid
            3D grid
        bound_idx_list : List of length 9
            List containing the boolean NumPy nd-arrays (of length g_3d.num_faces)
        p3d_numpy_list : List of length 9
            List containing the exact pressures as NumPy expressions

        Raises
        ------
        ValueError
            If the grid dimension is different from 3.

        Returns
        -------
        bc_values : NumPy nd-array of size g_3d.num_faces
            Boundary values as given by the exact solution.

        """

        # Check if grid is three-dimensional
        if g.dim != 3:
            raise ValueError("Dimension should be 3")

        # Get face-center coordinates
        x = g.face_centers

        # Intialize boundary values array
        bc_values = np.zeros(g.num_faces)

        # Evaluate exact pressure at external boundary faces at each region
        for (p, idx) in zip(p3d_numpy_list, bound_idx_list):
            bc_values[idx] = p(x[0][idx], x[1][idx], x[2][idx])

        return bc_values

    def get_exact_3d_pressure_gradient(g, p3d_sym_list):
        """
        Get exact pressure gradient (symbolic, NumPy, and cell-centered) for the 3D domain

        Parameters
        ----------
        g : PorePy grid
            3D grid.
        p3d_sym_list : List of length 9
            Containing the symbolic exact pressures for each subregion.

        Raises
        ------
        ValueError
            If the grid dimension is different from 3.

        Returns
        -------
        gradp3d_sym_list : List of length 9
            Containing the symbolic exact pressure gradient expressions
        gradp3d_numpy_list : List of length 9
            Containing the exact NumPy expressions for the pressure gradient.
        gradp3d_cc : NumPy nd-array of size g_3d.num_cells
            Cell-centered evaluated exact pressure gradient.

        """

        # Check dimensionality of the grid
        if g.dim != 3:
            raise ValueError("Dimension must be 3")

        # Define symbolic symbols
        x, y, z = sym.symbols("x y z")

        # Get cell center coordinates, and cell center boolean indices
        x3d_cc = g.cell_centers
        cell_idx_list, _ = get_3d_cell_indices(g)

        # Obtain gradient of the pressures
        gradp3d_sym_list = []
        for p in p3d_sym_list:
            gradp3d_sym_list.append([sym.diff(p, x), sym.diff(p, y), sym.diff(p, z)])

        # Convert to Numpy expressions
        gradp3d_numpy_list = []
        for gradp in gradp3d_sym_list:
            gradp3d_numpy_list.append(
                [
                    sym.lambdify((x, y, z), gradp[0], "numpy"),
                    sym.lambdify((x, y, z), gradp[1], "numpy"),
                    sym.lambdify((x, y, z), gradp[2], "numpy"),
                ]
            )

        # Obtain cell-centered pressure gradients
        gradpx_cc = np.zeros(g.num_cells)
        gradpy_cc = np.zeros(g.num_cells)
        gradpz_cc = np.zeros(g.num_cells)
        for (gradp, idx) in zip(gradp3d_numpy_list, cell_idx_list):
            gradpx_cc += gradp[0](x3d_cc[0], x3d_cc[1], x3d_cc[2]) * idx
            gradpy_cc += gradp[1](x3d_cc[0], x3d_cc[1], x3d_cc[2]) * idx
            gradpz_cc += gradp[2](x3d_cc[0], x3d_cc[1], x3d_cc[2]) * idx
        gradp3d_cc = np.array([gradpx_cc, gradpy_cc, gradpz_cc])

        return gradp3d_sym_list, gradp3d_numpy_list, gradp3d_cc

    def get_exact_3d_velocity(g, gradp3d_list):
        """
        Get exact velocity (symbolic, NumPy, and cell-centered) for the 3D domain

        Parameters
        ----------
        g : PorePy grid
            3D grid.
        gradp3d_list : List of length 9
            Containing the symbolic exact pressure gradient.

        Raises
        ------
        ValueError
            If the grid dimension is different from 3.

        Returns
        -------
        u3d_sym_list : List of length 9
            Containing the exact symbolic expressions for the velocity.
        u3d_numpy_list : List of length 9
            Containing the exact NumPy expressions for the velocity.
        u3d_cc : NumPy nd-Array of size g_3d.num_cells
            Containing the cell-centered evaluated exact velocity.

        """

        # Check dimensionality of the grid
        if g.dim != 3:
            raise ValueError("Dimension must be 3")

        # Define symbolic symbols
        x, y, z = sym.symbols("x y z")

        # Get cell center coordinates, and cell center boolean indices
        x3d_cc = g.cell_centers
        cell_idx_list, _ = get_3d_cell_indices(g)

        # Obtain velocities
        u3d_sym_list = []
        for gradp in gradp3d_list:
            u3d_sym_list.append([-gradp[0], -gradp[1], -gradp[2]])

        # Convert to Numpy expressions
        u3d_numpy_list = []
        for u in u3d_sym_list:
            u3d_numpy_list.append(
                [
                    sym.lambdify((x, y, z), u[0], "numpy"),
                    sym.lambdify((x, y, z), u[1], "numpy"),
                    sym.lambdify((x, y, z), u[2], "numpy"),
                ]
            )

        # Obtain cell-centered pressure gradients
        ux_cc = np.zeros(g.num_cells)
        uy_cc = np.zeros(g.num_cells)
        uz_cc = np.zeros(g.num_cells)
        for (u, idx) in zip(u3d_numpy_list, cell_idx_list):
            ux_cc += u[0](x3d_cc[0], x3d_cc[1], x3d_cc[2]) * idx
            uy_cc += u[1](x3d_cc[0], x3d_cc[1], x3d_cc[2]) * idx
            uz_cc += u[2](x3d_cc[0], x3d_cc[1], x3d_cc[2]) * idx
        u3d_cc = np.array([ux_cc, uy_cc, uz_cc])

        return u3d_sym_list, u3d_numpy_list, u3d_cc

    def get_exact_3d_source_term(g, u3d_sym_vel):
        """
        Get exact source term (satisfying the mass conservation equation) for the
        3D domain.

        Parameters
        ----------
        g : PorePy grid
            3D grid.
        u3d_sym_vel : List of length 9
            Containing the exact symbolic velocities for each subregion.

        Raises
        ------
        ValueError
            If the dimensionality of the grid is different from 3

        Returns
        -------
        f3d_sym_list : List of length 9
            Containing the exact symbolic source term.
        f3d_numpy_list : List of length 9
            Containing the exact NumPy expressions for the source term.
        f3d_cc : NumPy nd-array of size g_3d.num_cells
            Exact cell-centered evaluated source terms.

        """

        # Check grid dimensionality
        if g.dim != 3:
            raise ValueError("Dimension must be 3")

        # Define symbolic symbols
        x, y, z = sym.symbols("x y z")

        # Get cell center coordinates, and cell center boolean indices
        x3d_cc = g.cell_centers
        cell_idx_list, _ = get_3d_cell_indices(g)

        # Obtain source term
        f3d_sym_list = []
        for u in u3d_sym_list:
            f3d_sym_list.append(
                (sym.diff(u[0], x) + sym.diff(u[1], y) + sym.diff(u[2], z))
            )

        # Convert to Numpy expressions
        f3d_numpy_list = []
        for f in f3d_sym_list:
            f3d_numpy_list.append(sym.lambdify((x, y, z), f, "numpy"))

        # Obtain cell-center source terms
        f3d_cc = np.zeros(g.num_cells)
        for (f, idx) in zip(f3d_numpy_list, cell_idx_list):
            f3d_cc += f(x3d_cc[0], x3d_cc[1], x3d_cc[2]) * idx

        return f3d_sym_list, f3d_numpy_list, f3d_cc

    # Get hold of exact pressures
    p3d_sym_list, p3d_numpy_list, p3d_cc = get_exact_3d_pressure(g_3d)

    # Get hold of exact pressure gradients
    gradp3d_sym_list, gradp3d_numpy_list, gradp3d_cc = get_exact_3d_pressure_gradient(
        g_3d, p3d_sym_list
    )

    # Get hold of exact velocity
    u3d_sym_list, u3d_numpy_list, u3d_cc = get_exact_3d_velocity(g_3d, gradp3d_sym_list)

    # Get hold of exact source term
    f3d_sym_list, f3d_numpy_list, f3d_cc = get_exact_3d_source_term(g_3d, u3d_sym_list)

    # Get hold of external boundary values (#TEST IF WE'RE RETRIEVING THE RIGHT VALUES)
    bc_vals_3d = get_3d_boundary_values(g_3d, bound_idx_list, p3d_numpy_list)

    #%% Obtain integrated source terms
    def integrate_source_3d(g, f3d_numpy_list, cell_idx_list):
        """
        Computes the exact integral of the source term for the 3D domain

        Parameters
        ----------
        g : PorePy grid
            3D grid.
        f3d_numpy_list : List of length 9
            Containing the exact NumPy expressions for the source term.
        cell_idx_list : List of length 9
            Containing the boolean indices for the cells of the 3D domain

        Returns
        -------
        integral : NumPy nd-Array of size g_3d.num_cells
            Integral of the source term

        """

        # Declare integration method and get hold of elements in QuadPy format
        int_method = qp.t3.get_good_scheme(3)  # a scheme of degree 3 should be enough
        elements = utils.get_quadpy_elements(g_3d, g_3d)

        # We now declare the different integrand regions
        integral = np.zeros(g_3d.num_cells)
        for (f, idx) in zip(f3d_numpy_list, cell_idx_list):

            # Declare integrand
            def integrand(x):
                return f(x[0], x[1], x[2])

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return integral

    # Obtain the integrated source terms
    integrated_f3d = integrate_source_3d(g_3d, f3d_numpy_list, cell_idx_list)

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
            bc_vals = bc_vals_3d
        specified_parameters["bc_values"] = bc_vals

        # (Integrated) source terms are given by the exact solution
        if g.dim == max_dim:
            source_term = integrated_f3d
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

    #%% Obain error estimates (and transfer them to d[pp.STATE])
    estimates = mde.ErrorEstimate(gb, lam_name=edge_variable)
    estimates.estimate_error()
    estimates.transfer_error_to_state()
    majorant = estimates.get_majorant()
    error_estimate_3d = estimates.get_local_errors(g_3d, d_3d)
    error_estimate_2d = estimates.get_local_errors(g_2d, d_2d)
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
        cell_idx_list : List of length 9
            Containing the boolean cell indices of the subregions of the 3D domain

        Returns
        -------
        NumPy nd-Array
            Cell-centered evaluated reconstructed pressure of the 3D domain.
        NumPy nd-Array
            Cell-centered evaluated reconstructed pressure of the 2D domain.

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
            if g.dim == 3:
                rp_cc_3d = np.zeros([g.num_cells, 1])
                for idx in cell_idx_list:
                    rp_cc_3d += (
                        p[0] * x[0].reshape(g.num_cells, 1)
                        + p[1] * x[1].reshape(g.num_cells, 1)
                        + p[2] * x[2].reshape(g.num_cells, 1)
                        + p[3]
                    ) * idx.reshape(g.num_cells, 1)
            # Evaluate the 2D-reconstructed pressure
            else:
                rp_cc_2d = (
                    p[0] * x[0].reshape(g.num_cells, 1)
                    + p[1] * x[1].reshape(g.num_cells, 1)
                    + p[2]
                )

        return rp_cc_3d.flatten(), rp_cc_2d.flatten()

    def get_cc_reconvel(estimates, cell_idx_list):
        """
        Get hold of cell-centered evaluated reconstructed velocities

        Parameters
        ----------
        estimates : Error Estimates Object
            Error estimates object after mde.ErrorEstimate() has been applied
        cell_idx_list : List of length 9
            Containing the boolean cell indices of the subregions of the 3D domain

        Returns
        -------
        NumPy nd-Array
            Cell-centered evaluated reconstructed velocity of the 3D domain.
        NumPy nd-Array
            Cell-centered evaluated reconstructed velocity of the 2D domain.

        """

        # Get hold of estimates keyword
        kw_e = estimates.estimates_kw

        for g, d in gb:

            # Get hold of reconstructed pressure
            recon_u = d[kw_e]["recon_u"].copy()
            u = utils.poly2col(recon_u)

            # Obtain cell-centered coordinates
            x = g.cell_centers

            # Evaluate the 3D-reconstructed pressure
            if g.dim == 3:
                ru_cc_3d_x = np.zeros([g.num_cells, 1])
                ru_cc_3d_y = np.zeros([g.num_cells, 1])
                ru_cc_3d_z = np.zeros([g.num_cells, 1])

                for idx in cell_idx_list:
                    ru_cc_3d_x += (
                        u[0] * x[0].reshape(g.num_cells, 1) + u[1]
                    ) * idx.reshape(g.num_cells, 1)
                    ru_cc_3d_y += (
                        u[0] * x[1].reshape(g.num_cells, 1) + u[2]
                    ) * idx.reshape(g.num_cells, 1)
                    ru_cc_3d_z += (
                        u[0] * x[2].reshape(g.num_cells, 1) + u[3]
                    ) * idx.reshape(g.num_cells, 1)

                ru_cc_3d = np.array(
                    [ru_cc_3d_x.flatten(), ru_cc_3d_y.flatten(), ru_cc_3d_z.flatten()]
                )

            # Evaluate the 2D-reconstructed pressure
            else:
                ru_cc_2d_x = (u[0] * x[0].reshape(g.num_cells, 1) + u[1]).flatten()
                ru_cc_2d_y = (u[0] * x[1].reshape(g.num_cells, 1) + u[2]).flatten()
                ru_cc_2d_z = np.zeros(g.num_cells)
                ru_cc_2d = np.array([ru_cc_2d_x, ru_cc_2d_y, ru_cc_2d_z])

        return ru_cc_3d, ru_cc_2d

    # Get hold of cell-centered reconstructed pressure for the 3D and 2D domain
    reconp_cc_3d, reconp_cc_2d = get_cc_reconp(estimates, cell_idx_list)

    # Get hold of cell-centered reconstructed velocity for the 3D and 2D domain
    reconu_cc_3d, reconu_cc_2d = get_cc_reconvel(estimates, cell_idx_list)

    #%% Compute true errors for the pressure, i.e., ||| p - p_h |||
    def compute_pressure_3d_true_error(
        g, d, estimates, gradp3d_numpy_list, cell_idx_list
    ):
        """
        Computes true "pressure" error for the 3D subdomain

        Parameters
        ----------
        g : PorePy Grid
            Three-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 3D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.
        gradp3d_numpy_list : List
            List of NumPy lambda functions for each subregion.
        cell_idx_list : List
            List of Numpy boolean array for each subregion.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 3.

        Returns
        -------
        integral: NumPy array of size g_3d.num_cells
            (Squared) of the true errors for each element of the grid.

        """

        # Check if dimension is 3
        if g.dim != 3:
            raise ValueError("Dimension should be 3")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_p = d[kwe]["recon_p"].copy()
        pr = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.t3.get_good_scheme(3)
        elements = utils.get_quadpy_elements(g, g)

        # Compute the true error for each subregion
        integral = np.zeros(g.num_cells)

        for (gradp, idx) in zip(gradp3d_numpy_list, cell_idx_list):

            # Declare integrand and add subregion contribution
            def integrand(x):
                gradp_exact_x = gradp[0](x[0], x[1], x[2])
                gradp_exact_y = gradp[1](x[0], x[1], x[2])
                gradp_exact_z = gradp[2](x[0], x[1], x[2])

                gradp_recon_x = pr[0] * np.ones_like(x[0])
                gradp_recon_y = pr[1] * np.ones_like(x[1])
                gradp_recon_z = pr[2] * np.ones_like(x[2])

                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2
                int_z = (gradp_exact_z - gradp_recon_z) ** 2

                return int_x + int_y + int_z

            integral += method.integrate(integrand, elements) * idx

        return integral

    def compute_pressure_2d_true_error(g, d, estimates):
        """
        Computes the true "pressure" error for the 2D domain (the fracture)

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
            If dimension of the grid is different from 2.

        Returns
        -------
        NumPy nd-Array of size g_2d.num_cells
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
        g_rot = utils.rotate_embedded_grid(g)
        elements = utils.get_quadpy_elements(g, g_rot)

        # Compute the true error
        def integrand(x):

            gradp_exact_x = np.zeros_like(x[0])
            gradp_exact_y = np.zeros_like(x[1])

            gradp_recon_x = pr[0] * np.ones_like(x[0])
            gradp_recon_y = pr[1] * np.ones_like(x[1])

            int_x = (gradp_exact_x - gradp_recon_x) ** 2
            int_y = (gradp_exact_y - gradp_recon_y) ** 2

            return int_x + int_y

        return method.integrate(integrand, elements)

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
            _get_high_pressure_trace,
            _get_low_pressure,
        )

        def compute_sidegrid_error(estimates, side_tuple):
            """
            This functions projects a mortar quantity to the side grids, and then
            performs the integration on the given side grid.

            Parameters
            ----------
            side_tuple : Tuple
                Containing the sidegrids

            Returns
            -------
            true_error_side : NumPy nd-Array of size (sidegrid.num_cells, 1)
                True error (squared) for each element of the side grid.

            """

            # Get projector and sidegrid object
            projector = side_tuple[0]
            sidegrid = side_tuple[1]

            # Rotate side-grid
            sidegrid_rot = utils.rotate_embedded_grid(sidegrid)

            # Obtain QuadPy elements
            elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)

            # Project relevant quantities to the side grids
            deltap_side = projector * deltap
            k_side = projector * k
            n = projector.shape[0]
            true_jump = -np.ones(n).reshape(n, 1)

            # Declare integrand
            def integrand(x):
                p_jump = utils.eval_P1(deltap_side, x)
                return (k_side ** (0.5) * (true_jump - p_jump)) ** 2

            # Compute integral
            true_error_side = method.integrate(integrand, elements)

            return true_error_side

        # Get hold of mortar grid and check the dimensionality
        mg = d_e["mortar_grid"]
        if mg.dim != 2:
            raise ValueError("Expected two-dimensional grid")

        # Obtain higher- and lower-dimensional grids and dictionaries
        g_l, g_h = gb.nodes_of_edge(e)
        d_h = gb.node_props(g_h)
        d_l = gb.node_props(g_l)

        # Retrieve normal diffusivity
        normal_diff = d_e[pp.PARAMETERS]["flow"]["normal_diffusivity"]
        if isinstance(normal_diff, int) or isinstance(normal_diff, float):
            k = normal_diff * np.ones([mg.num_cells, 1])
        else:
            k = normal_diff.reshape(mg.num_cells, 1)

        # Face-cell map between higher- and lower-dimensional subdomains
        frac_faces = sps.find(mg.primary_to_mortar_avg().T)[0]
        frac_cells = sps.find(mg.secondary_to_mortar_avg().T)[0]

        # Obtain the trace of the higher-dimensional pressure
        tracep_high = _get_high_pressure_trace(estimates, g_l, g_h, d_h, frac_faces)

        # Obtain the lower-dimensional pressure
        p_low = _get_low_pressure(estimates, g_l, d_l, frac_cells)

        # Now, we can work with the pressure difference
        deltap = p_low - tracep_high

        # Declare integration method
        method = qp.t2.get_good_scheme(3)

        # Retrieve side-grids tuples
        sides = mg.project_to_side_grids()

        # Compute the errors for each sidegrid
        mortar_error = []
        for side in sides:
            mortar_error.append(compute_sidegrid_error(estimates, side))

        # Concatenate into one numpy array
        true_error_mortar = np.concatenate(mortar_error).sum()

        return true_error_mortar

    #%% Compute true errors for the velocity, i.e., ||| u - u_h |||_*
    def compute_velocity_3d_true_error(g, d, estimates, u3d_numpy_list, cell_idx_list):
        """
        Computes the true "velocity" error for the 3D subdomain

        Parameters
        ----------
        g : PorePy Grid
            Three-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 3D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.
        u3d_numpy_list : List
            List of NumPy lambda functions for each subregion.
        cell_idx_list : List
            List of Numpy boolean array for each subregion.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 3.

        Returns
        -------
        integral: NumPy array of size g_3d.num_cells
            (Squared) of the true errors for each element of the grid.

        """

        # Check if dimension is 3
        if g.dim != 3:
            raise ValueError("Dimension should be 3")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of approximated velocities and create list of coeffcients
        recon_u = d[kwe]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.t3.get_good_scheme(3)
        elements = utils.get_quadpy_elements(g, g)

        # Compute the true error for each subregion
        integral = np.zeros(g.num_cells)

        for (vel, idx) in zip(u3d_numpy_list, cell_idx_list):

            # Declare integrand and add subregion contribution
            def integrand(x):
                vel_exact_x = vel[0](x[0], x[1], x[2])
                vel_exact_y = vel[1](x[0], x[1], x[2])
                vel_exact_z = vel[2](x[0], x[1], x[2])

                vel_recon_x = u[0] * x[0] + u[1]
                vel_recon_y = u[0] * x[1] + u[2]
                vel_recon_z = u[0] * x[2] + u[3]

                int_x = (vel_exact_x - vel_recon_x) ** 2
                int_y = (vel_exact_y - vel_recon_y) ** 2
                int_z = (vel_exact_z - vel_recon_z) ** 2

                return int_x + int_y + int_z

            integral += method.integrate(integrand, elements) * idx

        return integral

    def compute_velocity_2d_true_error(g, d, estimates):
        """
        Computes the true "velocity" error for the 2D subdomain (the fracture)

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
            If dimension of the grid is different from 2.

        Returns
        -------
        NumPy nd-Array of size g_2d.num_cells
            (Squared) of the true errors for each element of the grid.
        """

        # Check if dimension is 2
        if g.dim != 2:
            raise ValueError("Dimension should be 2")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_u = d[kwe]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(3)
        g_rot = utils.rotate_embedded_grid(g)
        elements = utils.get_quadpy_elements(g, g_rot)

        # Compute the true error
        def integrand(x):

            vel_exact_x = np.zeros_like(x[0])
            vel_exact_y = np.zeros_like(x[1])

            vel_recon_x = u[0] * x[0] + u[1]
            vel_recon_y = u[0] * x[1] + u[2]

            int_x = (vel_exact_x - vel_recon_x) ** 2
            int_y = (vel_exact_y - vel_recon_y) ** 2

            return int_x + int_y

        return method.integrate(integrand, elements)

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
            If the dimension of the mortar grid is different from 2.

        Returns
        -------
        true_error_mortar: NumPy nd-array of shape (mg.num_cells, 1)
            True error (squared) for each element of the mortar grid.

        """

        # Get mortar grid
        mg = d_e["mortar_grid"]

        # Sanity check
        if mg.dim != 2:
            raise ValueError("Mortar grid must be two-dimensional")

        # Obtain difference between exact and approximated mortar fluxes
        V = mg.cell_volumes
        lmbda = d_e[pp.STATE][estimates.lam_name].copy()
        mortar_flux = lmbda / V
        lbmda_diff = (1.0 - mortar_flux) ** 2
        true_error_mortar = lbmda_diff * V

        return true_error_mortar

    #%% Obtain true errors
    true_pressure_error_3d = compute_pressure_3d_true_error(
        g_3d, d_3d, estimates, gradp3d_numpy_list, cell_idx_list
    )
    true_pressure_error_2d = compute_pressure_2d_true_error(g_2d, d_2d, estimates)
    true_pressure_error_mortar = compute_pressure_mortar_true_error(d_e, estimates)

    true_velocity_error_3d = compute_velocity_3d_true_error(
        g_3d, d_3d, estimates, u3d_numpy_list, cell_idx_list
    )
    true_velocity_error_2d = compute_velocity_2d_true_error(g_2d, d_2d, estimates)
    true_velocity_error_mortar = compute_velocity_mortar_true_error(d_e, estimates)

    #%% Compute effectivity index
    true_pressure_error = np.sqrt(
        true_pressure_error_3d.sum()
        + true_pressure_error_2d.sum()
        + true_pressure_error_mortar.sum()
    )

    true_velocity_error = np.sqrt(
        true_velocity_error_3d.sum()
        + true_velocity_error_2d.sum()
        + true_velocity_error_mortar.sum()
    )

    I_eff_pressure = majorant / true_pressure_error
    I_eff_velocity = majorant / true_velocity_error
    I_eff_combined = (3 * majorant) / (true_pressure_error + true_velocity_error)

    #%% Return
    return (
        h_max,
        error_estimate_3d,
        true_pressure_error_3d.sum() ** 0.5,
        true_velocity_error_3d.sum() ** 0.5,
        g_3d.num_cells,
        error_estimate_2d,
        true_pressure_error_2d.sum() ** 0.5,
        true_velocity_error_2d.sum() ** 0.5,
        g_2d.num_cells,
        error_estimate_mortar,
        true_pressure_error_mortar.sum() ** 0.5,
        true_velocity_error_mortar.sum() ** 0.5,
        mg.num_cells,
        majorant,
        true_pressure_error.sum(),
        true_velocity_error.sum(),
        I_eff_pressure,
        I_eff_velocity,
        I_eff_combined,
    )
