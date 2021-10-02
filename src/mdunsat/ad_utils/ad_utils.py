""" 
Collection of utility AD operators and functions 
Author: @jv
"""

#%% Importing modules
import porepy as pp
import numpy as np
import scipy.sparse as sps

from porepy.numerics.ad.operators import Operator, ApplicableOperator
from porepy.numerics.ad.functions import heaviside
from porepy.numerics.ad.forward_mode import Ad_array
from typing import Callable, Optional, Tuple, List, Any, Union


def get_conductive_mortar_cells(
        gb: pp.GridBucket,
        dof_manager: pp.DofManager,
        bulk_pressure_trace: pp.ad.Operator,
        fracture_pressure: pp.ad.Operator
        ) -> np.ndarray:

    pressure_threshold = 0  # TODO: This has to be an input, eventually
    num_mortar_cells = gb.num_mortar_cells()
    is_mortar_cell_conductive = np.zeros(num_mortar_cells, dtype=np.int8)

    # Evaluate ad operators
    trace_p, _ = eval_ad_operator(bulk_pressure_trace, gb, dof_manager, print_expression=False)
    frac_p, _ = eval_ad_operator(fracture_pressure, gb, dof_manager, print_expression=False)

    # TODO: This loop has to be optimized
    # If pressure in the fracture is greater than the pressure threshold,
    # promote the mortar cell to be conductive. This is the first condition
    # that we have to check. Otherwise, check if the pressure trace is
    # greater (or equal) than the pressure threshold, if that is the case
    # also promote that cell to conductive.
    for cell in range(0, trace_p.size):
        if frac_p[cell] > pressure_threshold:
            is_mortar_cell_conductive[cell] = 1
        else:
            if trace_p[cell] >= pressure_threshold:
                is_mortar_cell_conductive[cell] = 1

    return is_mortar_cell_conductive


def set_iterate_to_state(gb, bulk_var, fracture_var, mortar_var):

    for g, d in gb:
        if g.dim == gb.dim_max():
            pp.set_iterate(data=d, iterate={bulk_var: d[pp.STATE][bulk_var]})
        else:
            pp.set_iterate(data=d, iterate={fracture_var: d[pp.STATE][fracture_var]})

    for _, d in gb.edges():
        pp.set_iterate(data=d, iterate={mortar_var: d[pp.STATE][mortar_var]})


def eval_ad_operator(
        ad_expression: pp.ad.Operator,
        grid_bucket: pp.GridBucket,
        dof_manager: pp.DofManager,
        name: str = None,
        print_expression: bool = False) -> tuple:
    """
    Utility function for rapid evaluation of ad expressions.

    NOTE:
    -----
    This function is meant mostly for debugging purposes, use it with caution otherwise.

    PARAMETERS:
    ----------
    ad_expression: pp.ad.Operator
        (Active) ad expression to be evaluated. Note that if atomic ad variables are passed, it will result in an error
    grid_bucket: pp.GridBucket
        Grid bucket containing the mixed-dimensional geometry and necessary discretization matrices in the respective
        data dictionaries
    dof_manager: pp.DofManager
        Degree of freedom manager of the ad problem
    name: (optional) String
        Name of the ad expression passed by the user. Default is None.
    print_expression: (optional) Bool
       Prints the value and the Jacobian of the resulting ad expression in the console. Default is False.

    RAISES:
    ------
    TypeError:
        If ad_expression is not of the type: pp.ad.Operator

    RETURNS:
    --------
    expression_num.val: np.nd_array
        Values of the evaluated expression
    expression_num.jac : sps.spmatrix
        Jacobian of the evaluated expression
    """

    # Sanity check: Make sure ad expression is of the type pp.ad.Operator
    # In particular, we want to avoid evaluating atomic ad variables and numpy-like objects
    if not isinstance(ad_expression, pp.ad.Operator):
        raise TypeError("Ad expression can only be of the type pp.ad.Operator")

    # Evaluate ad expression
    expression_eval = pp.ad.Expression(ad_expression, dof_manager)

    # Discretize ad expression
    expression_eval.discretize(grid_bucket)

    # Parse expression to retrieve the numerical values
    expression_num = expression_eval.to_ad(grid_bucket)

    # Print if necessary: Meant only for small arrays and matrices, a.k.a. debugging.
    if print_expression:
        if name is None:
            print('Evaluation of ad expression: \n')
            print(f'Array with values: \n {expression_num.val} \n')
            print(f'Jacobian with values: \n {expression_num.jac.A} \n')
        else:
            print(f'Evaluation of ad expression: {name} \n')
            print(f'Array with values: \n {expression_num.val} \n')
            print(f'Jacobian with values: \n {expression_num.jac.A} \n')

    return expression_num.val, expression_num.jac


def is_water_volume_negative(
        gb: pp.GridBucket,
        fracture_var: str,
        fracture_list: List[pp.Grid]) -> bool:
    """
    Checks wheter negative water volume in the fractures is encountered

    Parameters
    ----------
    gb : GridBucket
    fracture_var : fracture variable keyword
    fracture_list : List of fracture grids

    Returns
    -------
    bool : True or False

    """

    # Sanity check
    for g in fracture_list:
        if g.dim == gb.dim_max():
            raise ValueError("Function meant only for fracture grids.")

    is_negative = False
    for g in fracture_list:
        d = gb.node_props(g)
        if np.any(d[pp.STATE][pp.ITERATE][fracture_var] < 0):
            is_negative = True

    return is_negative


class ParameterScalar(Operator):
    """Extracts a scalar from the parameter dictionary for a given grid or edge

    Can be used to change scalar parameters (e.g., time_step) during a simulation
    without having to redefine the equations. This class is needed since ParameterArray
    returns an ndarray in the parsing process, which creates dimensionality issues.
    """

    def __init__(
            self,
            param_keyword: str,
            scalar_keyword: str,
            grid: Optional[pp.Grid] = None,
            edge: Optional[Tuple[pp.Grid, pp.Grid]] = None
    ):
        """Construct a wrapper for a scalar parameter for a grid or edge

        Parameters:

            param_keyword (str): Keyword that should be used to access the data dictionary
                to get the relevant parameter dictionary.
            grid (pp.Grid): Grid.
            edge (Tuple of pp.Grid): Tuple of grids defining the edge.

        Example:
            To get the time step for a given grid, initialize with param_keyword='flow',
            and scalar_keyword='time_step'.

        """

        # Check whether a grid or an edge is passed
        if (grid is None) and (edge is None):
            raise ValueError("ParameterScalar needs at least a grid or an edge")

        # Only a grid, or an edge can be passed
        if (grid is not None) and (edge is not None):
            raise ValueError("Too many inputs. Expected one grid OR one edge.")

        self.param_keyword = param_keyword
        self.scalar_keyword = scalar_keyword
        self._g = grid
        self._e = edge
        self._set_tree()

    def __repr__(self) -> str:
        s = (
            f"Will access the scalar parameter with keyword {self.param_keyword}"
            f" and scalar keyword {self.scalar_keyword}."
        )
        return s

    def parse(self, gb: pp.GridBucket) -> float:
        """Convert the Ad expression into a float

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid.

        Returns:
            float: Value of the scalar.

        """
        if self._g is not None:
            data = gb.node_props(self._g)
            val = data[pp.PARAMETERS][self.param_keyword][self.scalar_keyword]
            if isinstance(val, float) or isinstance(val, int):
                return float(val)
            else:
                raise TypeError("ParameterScalar expects scalar to be int or float.")

        if self._e is not None:
            data = gb.edge_props(self._e)
            val = data[pp.PARAMETERS][self.param_keyword][self.scalar_keyword]
            if isinstance(val, float) or isinstance(val, int):
                return float(val)
            else:
                raise TypeError("ParameterScalar expects scalar to be int or float.")


class ParameterUpdate:
    """
    Parent class for updating parameters
    """

    def __init__(self, gb, param_key):
        self._gb = gb
        self._param_key = param_key

    def update_mortar_conductivity_state(
            self,
            is_mortar_conductive: np.ndarray,
            edges_list: List[Tuple[pp.Grid, pp.Grid]]
            ):
        """
        Updates the state of mortar cells for a given set of edges
        """

        for e in edges_list:
            d = self._gb.edge_props(e)
            d[pp.PARAMETERS][self._param_key]["is_conductive"] = is_mortar_conductive

    def update_time_step(self, dt: Union[int, float]):
        """
        Updates the state of the time step in the bulk dictionary
        """
        max_dim = self._gb.dim_max()
        g_bulk = self._gb.grids_of_dimension(max_dim)[0]
        d_bulk = self._gb.node_props(g_bulk)
        d_bulk[pp.PARAMETERS][self._param_key]["time_step"] = dt


class TimeSteppingControl:
    """Parent class for iteration-based time stepping control routine."""

    def __init__(
            self,
            time_init_final: Tuple[float, float],
            dt_init: float,
            dt_min_max: Tuple[float, float],
            iter_max: int,
            iter_optimal_range: Tuple[int, int],
            iter_lowupp_factor: Optional[Tuple[float, float]] = None,
            recomp_factor: Optional[float] = None,
            recomp_max: Optional[int] = None,
    ):
        """Computes the next time step based on the number of non-linear iterations.

        Parameters:
            time_init_final (Tuple of float): Initial and final simulation times.
            dt_init (float): Initial time step.
            dt_min_max (Tuple of float): Minimum and maximum permissible time steps.
            iter_max (int): Maximum number of iterations.
            iter_optimal_range (Tuple of int): Lower and upper optimal iteration range.
            iter_lowupp_factor (Tuple of float, optional): Lower and upper multiplication
                factors. Default is (1.3, 0.7).
            recomp_factor (float). Failed-to-converge recomputation factor. Default is 0.5.
            recomp_max (int). Failed-to-converge maximum recomputation attempts. Default is 10.

        """

        # Sanity checks
        if time_init_final[0] < 0:
            raise ValueError("Initial time cannot be negative")
        elif time_init_final[1] < time_init_final[0]:
            raise ValueError("Final time cannot be smaller than initial time")

        if dt_init <= 0:
            raise ValueError("Initial time step must be positive")
        elif dt_init > time_init_final[1]:
            raise ValueError("Inital time step cannot be larger than final simulation time.")
        elif dt_init < dt_min_max[0]:
            raise ValueError("Intial time step cannot be smaller than minimum time step.")
        elif dt_init > dt_min_max[1]:
            raise ValueError("Intial time step cannot be larger than maximum time step.")

        if dt_min_max[0] > dt_min_max[1]:
            s = "Minimum time step cannot be larger than maximum time step."
            raise ValueError(s)

        if iter_max <= 0:
            raise ValueError("Maximum amount of iterations must be a postive integer")

        if iter_optimal_range[0] > iter_optimal_range[1]:
            s = "Lower optimal iteration range cannot be larger than"
            s += " upper optimal iteration range."
            raise ValueError(s)
        elif iter_optimal_range[1] > iter_max:
            s = "Upper optimal iteration range cannot be larger than"
            s += " maximum amount of iterations."
            raise ValueError(s)

        if iter_lowupp_factor is not None and (iter_lowupp_factor[0] <= 1):
            raise ValueError("Expected lower multiplication factor > 1")
        elif iter_lowupp_factor is not None and (iter_lowupp_factor[1] >= 1):
            raise ValueError("Expected upper multiplication factor < 1")

        if (recomp_factor is not None) and recomp_factor >= 1:
            raise ValueError("Expected recomputation factor < 1")

        if (recomp_max is not None) and recomp_max <= 0:
            raise ValueError("Number of recomputation attempts must be a positive integer")

        # Initial and final time
        self.time_init, self.time_final = time_init_final

        # Initial time step
        self.dt_init = dt_init

        # Minimum and maximum allowable time steps
        self.dt_min, self.dt_max = dt_min_max

        # Maximum amount of iterations
        self.iter_max = iter_max

        # Target iteration range
        self.iter_low, self.iter_upp = iter_optimal_range

        # Lower and upper multiplication factors
        if iter_lowupp_factor is not None:
            self.iter_low_factor, self.iter_upp_factor = iter_lowupp_factor
        else:
            self.iter_low_factor = 1.3
            self.iter_upp_factor = 0.7

        # Recomputation multiplication factor
        if recomp_factor is not None:
            self.recomp_factor = recomp_factor
        else:
            self.recomp_factor = 0.5

        # Number of permissible recomputation attempts
        if recomp_max is not None:
            self.recomp_max = recomp_max
        else:
            self.recomp_max = 10

        # Initially, time = initial time and dt = initial dt
        self.time = self.time_init
        self.dt = self.dt_init
        self.recomp_sol = False
        self._recomp_num = 0

    def __repr__(self) -> str:

        s = "Time-stepping control object with atributes:\n"
        s += f"Initial simulation time = {self.time_init}\n"
        s += f"Final simulation time = {self.time_final}\n"
        s += f"Initial time step = {self.dt_init}\n"
        s += f"Minimum time step = {self.dt_min}\n"
        s += f"Maximum time step = {self.dt_max}\n"
        s += f"Lower optimal iteration range = {self.iter_low}\n"
        s += f"Upper optimal iteration range = {self.iter_upp}\n"
        s += f"Below lower optimal iteration range multiplication factor = {self.iter_low_factor}\n"
        s += f"Above upper optimal iteration range multiplication factor = {self.iter_upp_factor}\n"
        s += f"Failed-to-converge recomputation multiplication factor = {self.recomp_factor}\n"
        s += f"Failed-to-converge maximum recomputation attempts = {self.recomp_max}"

        return s

    def next_time_step(self, iters):
        """
        Determines the next time step based on the previous amount of iterations needed
        to reach convergence. If convergence was not achieved, then the time step is
        reduced by recomp_factor. The time-stepping control routine will recompute the
        solution recomp_max times. Otherwise, an error will be raised and the simulation
        stopped.

        Parameters
        iters (int): Number of non-linear iterations. In time-dependent simulations,
            this tipically represent the number of iterations for a time step.

        Returns
        -------
        dt: float
            Next time step
        """

        # First, check if we are allow to recompute the solution
        if self._recomp_num > self.recomp_max:
            s = f"Solution did not convergece after {self.recomp_max}"
            s += " recomputing attempts."
            raise ValueError(s)

        # If iters == max_iter:
        #   Decrease time step by the recomputing factor
        #   Update time (since solution will be recomputed)
        #   Set to True the recomputation flag
        #   Increase counter that keeps track of how many times the solution was recomputed
        if iters == self.iter_max:
            print("Solution did not convergece. Reducing time step and recomputing solution.")
            self.time -= self.dt  # reduce time
            self.dt = self.dt * self.recomp_factor # reduce time step
            self.recomp_sol = True
            self._recomp_num += 1
            return self.dt
        else:
            self.recomp_sol = False
            self._recomp_num = 0

        # If iters < max_iter. Proceed to determine the next time step using the
        # following criteria.
        # If iters is less than the lower optimal iteration range "iter_low", we can relax
        # the time step, and multiply by a lower multiplication factor greater than 1,
        # i.e., "factor_low". If the number of iterations is greater than the upper optimal
        # iteration range "iter_upp", we have to decrease the time step by multiplying by an upper
        # multiplication factor smaller than 1, i.e., "factor_upp". If neither of these situations
        # occur, then the number iterations lies between the optimal iteration range,
        # and the time step remains unchanged.
        if iters <= self.iter_low:
            self.dt = self.dt * self.iter_low_factor
            print("Relaxing time step.")
        elif iters >= self.iter_upp:
            self.dt = self.dt * self.iter_upp_factor
            print("Restricting time step.")

        # Check if the calculated time step is less than the minimum allowable time step
        if self.dt < self.dt_min:
            self.dt = self.dt_min
            print("Calculated time step is smaller than dt_min. Using dt_min instead.")

        # Check if the calculated time step is greater than the maximum allowable time step
        if self.dt > self.dt_max:
            self.dt = self.dt_max
            print("Calculated time step is greater than dt_max. Using dt_max instead.")

        # Check if we reach the final simulation time with the calculated time step
        if (self.time + self.dt) > self.time_final:
            self.dt = self.time_final - self.time
            print("Adapting time step to reach final simulation time.")

        return self.dt

    # @staticmethod
    # def check_negative_water_volume(self, dt_old, water_volume, factor=0.5):
    #
    #     # In order to avoid negative values of water volume inside the fracture, we
    #     # need to control the time step so that only a permissible amount of water
    #     # may leave the fracture. This is controlled by checking if there are any
    #     # fractures with negative volume, if this is the case, then we multiply
    #     # the previous time step by a factor less than 1, i.e., "factor". Note
    #     # that this sanity check should override the calculation of next time step,
    #     # since negative water volume is by no means permissible.
    #     if np.any(water_volume < 0):
    #         dt = dt_old * factor
    #     else:
    #         dt = dt_old
    #
    #     return dt

#%% Ghost Projections
class GhostProjection:

    def __init__(self, gb_ghost, g_fracture):

        self._gb = gb_ghost
        self._g = g_fracture

        # Get hold of ghost mortar projections
        mortar_projection = pp.ad.MortarProjections(gb=gb_ghost)
        self._secondary_to_mortar_avg = mortar_projection.secondary_to_mortar_avg.parse(gb=gb_ghost)

        # Get hold of ghost subdomain projections
        subdomain_projection = pp.ad.SubdomainProjections(gb=gb_ghost)
        self._cell_prolongation = subdomain_projection.cell_prolongation(self._g).parse(gb=gb_ghost)

    def secondary_to_mortar(self, fracture_pressure):

        proj_fracture_pressure = self._secondary_to_mortar_avg * self._cell_prolongation * fracture_pressure

        return proj_fracture_pressure


#%% Fracture pressure-related classes
class HydrostaticFracturePressure:

    """
    Given a volume of water, it returns the hydrostatic pressure head in the fracture. It is assumed
    that the aperture is given as a field in the grid's data dictionary. If the pressure calculated
    pressure head is negative, the value for that cell is corrected and assumed to be zero.
    """

    def __init__(self, ghost_grid, data, param_key):

        self._g = ghost_grid
        self._d = data
        self._param_key = param_key

        # Get datum in absolute coordinates
        if self._g.dim == 2:  # if fracture grid is 2D, then ambient dimension is 3D
            datum = np.min(self._g.face_centers[2])
        elif self._g.dim == 1:  # if fracture grid is 1D, then ambient dimension is 2D
            datum = np.min(self._g.face_centers[1])
        else:
            raise NotImplementedError(f"Grid cannot be of dimension {self._g.dim}")
        self._datum = datum

        # Get aperture
        aperture = self._d[pp.PARAMETERS][self._param_key]["aperture"]
        if not (isinstance(aperture, int) or isinstance(aperture, float)):
            raise ValueError("Aperture can only be a scalar for now.")
        self._aperture = aperture

        # Get cell centers
        if self._g.dim == 2:
            cell_centers = self._g.cell_centers[2]
        elif self._g.dim == 1:
            cell_centers = self._g.cell_centers[1]
        else:
            raise NotImplementedError(f"Grid cannot be of dimension {self._g.dim}")
        self._cc = cell_centers

    def __repr__(self) -> str:
        return "Hydrostatic water fracture pressure Ad operator"

    def get_pressure_head(self, water_volume):

        # Check size of the input first, water_volume is only meant to have size 1
        if isinstance(water_volume, pp.ad.Ad_array):
            if water_volume.val.size > 1:
                raise ValueError("Only 1 degree of freedom is permitted in the fracture.")
        elif isinstance(water_volume, np.ndarray):
            if water_volume.size > 1:
                raise ValueError("Only 1 degree of freedom is permitted in the fracture.")

        # Create broadcasting matrix. This is needed to obtain the right shape of the Jacobian
        # since PorePy won't do that for us automatically.
        broadcaster = sps.csr_matrix(np.ones_like(self._cc)).reshape((-1, 1))

        # Obtain the height of the air-water interface in absolute coordinates
        # TODO: This might be updated when we move to two-dimensional fractures
        airwater_interface = water_volume * (1/self._aperture) + self._datum

        # Get the value of the pressure head
        pressure_head = broadcaster * airwater_interface - self._cc
        # Perform sanity check on the size of the Jacobian
        if isinstance(pressure_head, pp.ad.Array):
            if not pressure_head.jac.shape[0] == self._cc.shape[0]:
                raise ValueError(f"Expected Jacobian with {self._cc.shape[0]} rows. Got {pressure_head.jac.shape[0]} instead.")

        # Now, we need to correct the values of the pressure head since negative values are not
        # permitted. Again, this might change when we handle different capillary barrier values
        if isinstance(pressure_head, pp.ad.Ad_array):
            pressure_head.val[pressure_head.val < 0] = 0
        elif isinstance(pressure_head, np.ndarray):
            pressure_head[pressure_head < 0] = 0

        return pressure_head



#%% INTERFACE UPSTREAM WEIGHTING
class InterfaceUpwindAd(ApplicableOperator):
    """
    Computes the interface relative permeabilities based on the (projected) 
    pressure jump associated with the bulk and fractur potentials.
    """

    def __init__(self):

        self._set_tree()

    def __repr__(self) -> str:
        return "Interface upwind AD operator"

    #TODO: Add sanity check to check if input matches amount of mortar cells in gb
    #TODO: Write tests
    def apply(self, trace_p_bulk, krw_trace_p_bulk, p_frac, krw_p_frac):
        """
        Apply method for upwinding of interface relative permeabilities.
                
        Parameters
        ----------
        trace_p_bulk : nd-array of size total_num_of_mortar_cells
            Mortar-projected bulk pressure trace
        krw_trace_p_bulk : nd-array of size total_num_of_mortar_cells
            Mortar-projected relative permeabilities of bulk pressure trace.
        p_frac : nd-array of size total_num_of_mortar_cells
            Mortar-projected fracture pressures.
        krw_p_frac : nd-array of size total_num_of_mortar_cells
            Mortar-projected relative permeabilites of fracture presure
            
        Raises
        ------
        TypeError
            If one of the input arguments is an Ad Array

        Returns
        -------
        interface_krw : Sparse Matrix of size total_num_mortar_cells ** 2
            Diagonal matrix with each entry representing the value of 
            the relative permeability associated with the mortar cell
        """

        # Sanity check of input type
        if (isinstance(trace_p_bulk, pp.ad.Ad_array) or
            isinstance(krw_trace_p_bulk, pp.ad.Ad_array) or
            isinstance(p_frac, pp.ad.Ad_array) or
            isinstance(krw_p_frac, pp.ad.Ad_array)):
            raise TypeError("Input cannot be of type Ad array")
        else:
            pressure_jump = trace_p_bulk - p_frac
            hs_10 = heaviside(pressure_jump, zerovalue=0)
            hs_01 = heaviside(-pressure_jump, zerovalue=0)
            vals = hs_10 * krw_trace_p_bulk + hs_01 * krw_p_frac
            n = len(trace_p_bulk)
            interface_krw = sps.spdiags(vals, 0, n, n)

        return interface_krw

#%% BULK FACE AVERAGING SCHEMES

# Arithmetic average of the bulk
class ArithmeticAverageAd(ApplicableOperator):
    """
    Computes the face arithmetic average of a cell-based array
    """

    def __init__(self, g, d, param_key):

        self._set_tree()
        self._g = g  # grid
        self._d = d  # dictionary
        self._param_key = param_key  # keyword

    def __repr__(self) -> str:
        return "Arithmetic average AD operator"

    def apply(self, inner_values, dir_bound_values):
        """
        Apply arithmetich average 

        Parameters
        ----------
        inner_values : np.ndarray of size g.num_cells
            Cell-center values to be averaged
        dir_bound_values : np.ndarray of size g.num_faces
            Containing values of Dirichlet boundary data. Neumann data not used.

        Raises
        ------
        TypeError
            If the input is an AD_array. Only non AD objects are permitted.

        Returns
        -------
        Numpy Array of size g.num_faces
            Arithmetic averaged values at the faces of the grid

        """

        if isinstance(inner_values, Ad_array):
            raise TypeError("Object cannot be of the type Ad_array")
        else:
            # Retrieve usefuld data
            bc = self._d[pp.PARAMETERS][self._param_key]["bc"]
            dir_fcs = bc.is_dir.nonzero()  # dirichlet boundary faces
            int_fcs = self._g.get_internal_faces()  # internal faces

            # Faces neighboring mapping
            fcs_neigh = np.zeros((self._g.num_faces, 2), dtype=int)
            fcs_neigh[:, 0] = self._g.cell_face_as_dense()[0]
            fcs_neigh[:, 1] = self._g.cell_face_as_dense()[1]
            int_fcs_neigh = fcs_neigh[int_fcs]

            # Initialize array
            face_avg = np.ones(self._g.num_faces) # Neumann krw=1.0

            # Values at Dirichlet boundaries
            dir_cells_neigh = fcs_neigh[dir_fcs]
            if dir_cells_neigh.size > 0:
                dir_cells = dir_cells_neigh[(dir_cells_neigh >= 0).nonzero()]
                face_avg[dir_fcs] = 0.5 * (
                    dir_bound_values[dir_fcs] + inner_values[dir_cells]
                )

            # Values at internal faces
            face_avg[int_fcs] = 0.5 * (
                inner_values[int_fcs_neigh[:, 0]]
                + inner_values[int_fcs_neigh[:, 1]]
            )

        return sps.spdiags(face_avg, 0, self._g.num_faces, self._g.num_faces)


# Flux-based upwinding scheme
class UpwindFluxBasedAd(ApplicableOperator):
    """ Flux based upwinding of cell-center arrays """
    # Credits: @jwboth

    def __init__(self, g, d, param_key, hs: Callable = heaviside):

        self._set_tree()
        self._g = g
        self._d = d
        self._param_key = param_key
        self._heaviside = hs

        # Construct projection from cell-valued arrays to face-valued arrays with values to the
        # "left" and "right" of the face, here denoted by '0' and '1', respectively.
        cf_dense = g.cell_face_as_dense()
        cf_inner = [c >= 0 for c in cf_dense]

        row = [np.arange(g.num_faces)[cf_inner[i]] for i in range(0, 2)]
        col = [cf_dense[i][cf_inner[i]] for i in range(0, 2)]
        data = [np.ones_like(row[i]) for i in range(0, 2)]
        self._cf_inner = [
            sps.csr_matrix(
                (data[i], (row[i], col[i])),
                shape=(g.num_faces, g.num_cells),
                dtype=float,
            )
            for i in range(0, 2)
        ]

        # Store which 'left' and 'right' cells of all faces correspond to the Dirichlet
        # boundary.
        cf_is_boundary = np.logical_not(cf_inner)
        self._cf_is_boundary = cf_is_boundary
        self._is_dir = d[pp.PARAMETERS][param_key]["bc"].is_dir.copy()
        self._cf_is_dir = [np.logical_and(cf_is_boundary[i], self._is_dir) for i in range(0, 2)]
        self._is_neu = d[pp.PARAMETERS][param_key]["bc"].is_neu.copy()
        self._cf_is_neu = [np.logical_and(cf_is_boundary[i], self._is_neu) for i in range(0, 2)]


    def __repr__(self) -> str:
        return " Flux-based upwind AD face operator"

    def apply(self, inner_values, dir_bound_values, face_flux):
        """Compute transmissibility via upwinding over faces.

        Idea: 'face value' = 'left cell value' * Heaviside('flux from left')
                           + 'right cell value' * Heaviside('flux from right').
        
        Parameters
        ----------
        inner_values : np.ndarray of size g.num_cells
            Cell-center values to be averaged
        dir_bound_values : np.ndarray of size g.num_faces
            Containing values of Dirichlet boundary data. Neumann data not used.
        face_flux : np.ndarray of size g.num_faces
            Containing fluxes for all faces of the grid. Note that the fluxes
            are only used to deterime the directionality.

        Raises
        ------
        TypeError
            If the input is an AD_array. Only non AD objects are permitted.

        Returns
        -------
        Numpy Array of size g.num_faces
            Arithmetic averaged values at the faces of the grid

        """

        # Rename internal properties
        hs = self._heaviside
        cf_inner = self._cf_inner
        cf_is_boundary = self._cf_is_boundary

        # Use Dirichlet boundary data where suitable.
        # Neglect Neumann boundaries since Neumann boundary data does not play a role.
        if isinstance(inner_values, Ad_array) or isinstance(face_flux, Ad_array):
            raise TypeError("Object cannot be of the type Ad_array")
        else:
            val_f = [cf_inner[i] * inner_values for i in range(0, 2)]
            for i in range(0, 2):
                val_f[i][cf_is_boundary[i]] = dir_bound_values[cf_is_boundary[i]]

        # Evaluate the Heaviside function of the "flux directions".
        hs_f_01 = hs(face_flux)
        hs_f_10 = hs(-face_flux)

        # Determine the face mobility by utilizing the general idea (see above).
        face_upwind = val_f[0] * hs_f_01 + val_f[1] * hs_f_10

        # Deal with Neumann boundary conditions
        face_upwind[self._is_neu] = 1.0

        return sps.spdiags(face_upwind, 0, self._g.num_faces, self._g.num_faces)


#%% SOIL WATER RETENTION CURVES

class vanGenuchten:
    def __init__(self, g, d, param_key):
        self._g = g
        self._d = d
        self._param_key = param_key

        params = self._d[pp.PARAMETERS][self._param_key]
        self.alpha_vG = params["alpha_vG"]
        self.theta_r = params["theta_r"]
        self.theta_s = params["theta_s"]
        self.n_vG = params["n_vG"]
        self.m_vG = params["m_vG"]

    def __repr__(self):
        return "Soil Water Retention Curve: van Genuchtem-Mualem model"

    def is_unsat(self, p):
        """ Determine whether the cell is saturated or not """

        if isinstance(p, pp.ad.Ad_array):
            raise TypeError("Pressure cannot be AD. Expected inactive variable.")
        else:
            # {1, pressure_head < 0
            # {0, otherwise
            return 1 - heaviside(p, 1)

    def water_content(self, p):
        """ Water content as a function of the pressure head"""

        if isinstance(p, pp.ad.Ad_array):
            is_unsat = self.is_unsat(p.val)
            is_sat = 1 - is_unsat
            num = self.theta_s - self.theta_r
            den = (1 + (self.alpha_vG * pp.ad.abs(p)) ** self.n_vG) ** self.m_vG
            theta = ((num * den ** (-1) + self.theta_r) * is_unsat
                     + self.theta_s * is_sat)
        else:
            is_unsat = self.is_unsat(p)
            is_sat = 1 - is_unsat
            num = self.theta_s - self.theta_r
            den = (1 + (self.alpha_vG * np.abs(p)) ** self.n_vG) ** self.m_vG
            theta = (num / den + self.theta_r) * is_unsat + self.theta_s * is_sat

        return theta

    def effective_saturation(self, p):
        """ Effective saturation as a function of the water content """

        num = self.water_content(p) - self.theta_r
        den = self.theta_s - self.theta_r
        s_eff = num * den ** (-1)

        return s_eff

    def relative_permeability(self, p):
        """ Relative permeability as a function of the effective saturation"""

        if isinstance(p, pp.ad.Ad_array):
            raise TypeError("Pressure cannot be AD. Expected previous_iteration()")
        else:
            krw = (self.effective_saturation(p) ** (0.5) *
                   (1 - (1 - self.effective_saturation(p) ** (1/self.m_vG))
                    ** self.m_vG) ** 2 )

        return krw


    def moisture_capacity(self, p):
        """ Specific moisture capacity as a function of the pressure head"""

        if isinstance(p, pp.ad.Ad_array):
            raise TypeError("Pressure cannot be AD. Expected previous_iteration()")
        else:
            is_unsat = self.is_unsat(p)
            is_sat = 1 - is_unsat
            num = (- self.m_vG * self.n_vG *
                   (self.theta_s - self.theta_r) *
                   (self.alpha_vG * np.abs(p)) ** self.n_vG)
            den = (p * ((self.alpha_vG * np.abs(p)) ** self.n_vG + 1)
                   ** (self.m_vG + 1))
            C = (np.divide(num, den, out=np.zeros_like(num), where=den!=0) * self.is_unsat(p)
                + 0 * is_sat)

        return C


