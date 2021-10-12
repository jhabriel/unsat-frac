""" 
Collection of utility AD operators and functions 
Author: @jv
"""

# %% Importing modules
import porepy as pp
import numpy as np
import scipy.sparse as sps

from porepy.numerics.ad.operators import Operator, ApplicableOperator
from porepy.numerics.ad.functions import heaviside

# from porepy.numerics.ad.forward_mode import Ad_array
from typing import Callable, Optional, Tuple, List, Any, Union, NewType, Literal

# Typing abbreviations
Scalar = Union[int, float]
AdArray = pp.ad.Ad_array
NonAd = Union[Scalar, np.ndarray]
Edge = Tuple[pp.Grid, pp.Grid]


def get_conductive_mortars(
    gb: pp.GridBucket,
    dof_manager: pp.DofManager,
    param_key: str,
    proj_tr_hb: pp.ad.Operator,
    proj_hf: pp.ad.Operator,
    edge_list: List[Edge],
) -> np.ndarray:
    """
    Determines the conductivity state of a mortar cell, i.e., conductive (1) or blocking (0).

    Parameters:
        gb (GridBucket): Mixed-dimensional grid bucket
        dof_manager (DofManager): Degree of freedom manager for the coupled problem.
        param_key (str): Parameter keyword for accesing data dictionary, i.e., flow.
        proj_tr_hb (pp.ad.Operator): Projected hydraulic head traces from the primary grid
            onto the mortar grids.
        proj_hf (pp.ad.Operator): Projected hydraulic head from the secondary grid onto the
            mortar grids.
        edge_list (List of Edges): List of edges. It is critical the ordering of the edges
            to be respected.

    Returns:
        is_conductive (np.ndarray of bools): True if is conductive, False if is blocking.

    """
    pressure_threshold = 0  # TODO: This has to be retrieve from some data dictionary
    is_conductive = np.zeros(gb.num_mortar_cells(), dtype=np.int8)
    zeta = pp.ad.ParameterArray(param_key, "elevation", edges=edge_list).parse(gb)

    # Evaluate operators
    proj_tr_hb.discretize(gb=gb)
    tr_hb = proj_tr_hb.evaluate(dof_manager)

    proj_hf.discretize(gb=gb)
    hf = proj_hf.evaluate(dof_manager)

    # TODO: This loop can be optimized... I'm just too lazy to vectorise it <3
    # If pressure in the fracture is greater than the pressure threshold, promote the mortar
    # cell to be conductive. This is the first condition that we have to check. Otherwise,
    # check if the pressure trace is greater (or equal) than the pressure threshold,
    # if that is the case also promote that cell to conductive.
    for mortar_cell in range(0, tr_hb.val.size):
        if hf.val[mortar_cell] > (pressure_threshold + zeta[mortar_cell]):
            is_conductive[mortar_cell] = 1
        else:
            if tr_hb.val[mortar_cell] >= (pressure_threshold + zeta[mortar_cell]):
                is_conductive[mortar_cell] = 1

    return is_conductive


# CONSIDER IT DEPRECATED
def get_conductive_mortar_cells(
    gb: pp.GridBucket,
    dof_manager: pp.DofManager,
    bulk_pressure_trace: pp.ad.Operator,
    fracture_pressure: pp.ad.Operator,
) -> np.ndarray:
    pressure_threshold = 0  # TODO: This has to be an input, eventually
    num_mortar_cells = gb.num_mortar_cells()
    is_mortar_cell_conductive = np.zeros(num_mortar_cells, dtype=np.int8)

    # Evaluate operators
    bulk_pressure_trace.discretize(gb=gb)
    trace_p = bulk_pressure_trace.evaluate(dof_manager=dof_manager)
    fracture_pressure.discretize(gb=gb)
    frac_p = fracture_pressure.evaluate(dof_manager=dof_manager)

    # TODO: This loop has to be optimized
    # If pressure in the fracture is greater than the pressure threshold, promote the mortar
    # cell to be conductive. This is the first condition that we have to check. Otherwise,
    # check if the pressure trace is greater (or equal) than the pressure threshold,
    # if that is the case also promote that cell to conductive.
    for cell in range(0, trace_p.val.size):
        if frac_p.val[cell] > pressure_threshold:
            is_mortar_cell_conductive[cell] = 1
        else:
            if trace_p.val[cell] >= pressure_threshold:
                is_mortar_cell_conductive[cell] = 1

    return is_mortar_cell_conductive


def set_iterate_as_state(gb: pp.GridBucket, node_var: str, edge_var: str):
    """
    (Re)sets the iterate as the state in all nodes and edges of the grid bucket. This
    function is particularly useful for "going back" to the previous state after recomputing
    a solution.

    Parameters:
        gb (Grid Bucket): Mixed-dimensional grid bucket
        node_var (str): Keyword for the node variable, i.e., "hydraulic_head"
        edge_var (str): Keyword for the edge variable, i.e., "mortar_flux"

    """

    # Loop through all nodes of the gb
    for _, d in gb:
        pp.set_iterate(data=d, iterate={node_var: d[pp.STATE][node_var]})

    # Loop through all the edges of the gb
    for _, d in gb.edges():
        pp.set_iterate(data=d, iterate={edge_var: d[pp.STATE][edge_var]})


# CONSIDER IT DEPRECATED
def set_iterate_to_state(gb, bulk_var, fracture_var, mortar_var):
    for g, d in gb:
        if g.dim == gb.dim_max():
            pp.set_iterate(data=d, iterate={bulk_var: d[pp.STATE][bulk_var]})
        else:
            pp.set_iterate(data=d, iterate={fracture_var: d[pp.STATE][fracture_var]})

    for _, d in gb.edges():
        pp.set_iterate(data=d, iterate={mortar_var: d[pp.STATE][mortar_var]})


# CONSIDER IT DEPRECATED
def eval_ad_operator(
    ad_expression: pp.ad.Operator,
    grid_bucket: pp.GridBucket,
    dof_manager: pp.DofManager,
    name: str = None,
    print_expression: bool = False,
) -> tuple:
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
            print("Evaluation of ad expression: \n")
            print(f"Array with values: \n {expression_num.val} \n")
            print(f"Jacobian with values: \n {expression_num.jac.A} \n")
        else:
            print(f"Evaluation of ad expression: {name} \n")
            print(f"Array with values: \n {expression_num.val} \n")
            print(f"Jacobian with values: \n {expression_num.jac.A} \n")

    return expression_num.val, expression_num.jac


# TODO: Check if this is still relevant
def is_water_volume_negative(
    gb: pp.GridBucket, fracture_var: str, fracture_list: List[pp.Grid]
) -> bool:
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
        if np.any(d[pp.STATE][pp.ITERATE][fracture_var] < 10):
            is_negative = True

    return is_negative


class ParameterScalar(Operator):
    """Extracts a scalar from the parameter dictionary for a given grid or edge

    Can be used to change scalar parameters (e.g., time_step) during a simulation
    without having to redefine the equations. This class is necessary since ParameterArray
    returns an ndarray in the parsing process, which creates dimensionality errors due to
    broadcasting issues.
    """

    def __init__(
        self,
        param_keyword: str,
        scalar_keyword: str,
        grid_list: Optional[List[pp.Grid]] = None,
        edge_list: Optional[List[Edge]] = None,
    ):
        """Construct a wrapper for a scalar parameter for a grid list or an edge list

        Parameters:
            param_keyword (str): Keyword that should be used to access the data dictionary
                to get the relevant parameter dictionary.
            grid_list (List of pp.Grid): List containing the grid from which the parameter
                scalar should be retrieved. If len(grid_list) > 1, an error will be raised.
            edge_list (Tuple of pp.Grid): List containing the edge from which the parameter
                scalar should be retrieved. If len(edge_list) > 1, an error will be raised.

        Example:
            dt_ad = ParameterScalar("flow", "time_step", grid_list=[g])
        """

        super().__init__()

        if (grid_list is None) and (edge_list is None):
            raise ValueError(
                "ParameterScalar needs at least a grid list or an edge list."
            )
        if (grid_list is not None) and (edge_list is not None):
            raise ValueError("grid_list and edge_list cannot be passed simultaneously.")
        if grid_list is not None and len(grid_list) > 1:
            raise ValueError("Expected a grid list with only one grid.")
        if edge_list is not None and len(edge_list) > 1:
            raise ValueError("Expected an edge list with only one edge.")

        self.param_keyword = param_keyword
        self.scalar_keyword = scalar_keyword
        self._grid_list = grid_list
        self._edge_list = edge_list

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
        if self._grid_list is not None:
            data = gb.node_props(self._grid_list[0])
            val = data[pp.PARAMETERS][self.param_keyword][self.scalar_keyword]
        else:
            data = gb.edge_props(self._edge_list[0])
            val = data[pp.PARAMETERS][self.param_keyword][self.scalar_keyword]

        # We only allow for int or float
        if isinstance(val, float):
            return val
        elif isinstance(val, int):
            return float(val)
        else:
            raise TypeError(
                f"Expected 'int' or 'float'. Encountered {type(val)} instead."
            )


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
        edges_list: List[Tuple[pp.Grid, pp.Grid]],
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
            raise ValueError(
                "Inital time step cannot be larger than final simulation time."
            )
        elif dt_init < dt_min_max[0]:
            raise ValueError(
                "Intial time step cannot be smaller than minimum time step."
            )
        elif dt_init > dt_min_max[1]:
            raise ValueError(
                "Intial time step cannot be larger than maximum time step."
            )

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
            raise ValueError(
                "Number of recomputation attempts must be a positive integer"
            )

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
            print(
                "Solution did not convergece. Reducing time step and recomputing solution."
            )
            self.time -= self.dt  # reduce time
            self.dt = self.dt * self.recomp_factor  # reduce time step
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


# %% Ghost Projections
class GhostProjection:
    def __init__(self, gb_ghost: pp.GridBucket, g_fracture: pp.Grid):
        self._gb = gb_ghost
        self._gridlist: List[pp.Grid] = [g for g, _ in self._gb]
        self._edgelist: List[Edge] = [e for e, _ in self._gb.edges()]
        self._gfrac = g_fracture

        # Get hold of ghost mortar projections
        proj = pp.ad.MortarProjections(
            gb=self._gb, grids=self._gridlist, edges=self._edgelist
        )
        # We only need secondary -> mortar (average)
        self._secondary_to_mortar_avg = proj.secondary_to_mortar_avg.parse(gb=self._gb)

        # Get hold of ghost subdomain projections
        subdomain_projection = pp.ad.SubdomainProjections(grids=self._gridlist)
        # We need the prolongation from the fracture to the global number of cells
        self._cell_prolongation = subdomain_projection.cell_prolongation(
            grids=[self._gfrac]
        ).parse(gb=self._gb)

    def secondary_to_mortar(self, fracture_pressure: pp.ad.Operator) -> pp.ad.Operator:
        proj_fracture_pressure = (
            self._secondary_to_mortar_avg * self._cell_prolongation * fracture_pressure
        )

        return proj_fracture_pressure


# %% Fracture pressure-related classes


class GhostFractureHydraulicHead:
    """
    Given the "real" fracture hydraulic head, compute the "ghost" fracture hydraulic head.
    """

    # TODO: Extend this to several fractures. The idea will be to accept a list of ghost
    #  grids instead of only one grid.
    def __init__(self, gb: pp.GridBucket, ghost_grid: pp.Grid):
        """Init method for the class."""

        self._g: pp.Grid = ghost_grid
        self._gb: pp.GridBucket = gb
        self._dim_max: int = self._gb.dim_max()

        # Sanity check
        if self._g.dim >= self._dim_max:
            raise ValueError(f"Ghost grid cannot be of dimension {self._g.dim}.")

        # Get cell centers and store them in a list
        self._cc = self._g.cell_centers[self._dim_max - 1]

    def __repr__(self) -> str:
        return "Ghost fracture hydraulic head AD object."

    def get_ghost_hyd_head(
        self, h_frac: Union[AdArray, NonAd]
    ) -> Union[AdArray, NonAd]:
        """
        Computes the hydraulic head for a ghost fracture grid given the hydraulic head with
        dof = 1.

        Parameters:
            h_frac (Ad_array or non-ad): fracture hydraulic head of physcial grid bucket.
            This can be an active Ad_array of val.size = 1 or a non-ad scalar object.
        Returns:
            ghost_h_frac (Ad_array or non-ad): hydraulic head corresponding to the ghost
            fracture grid. If the ghost fracture cell is dry, the hydraulic head is equal to
            the elevation of such cell, otherwise, it takes the value of h_frac.

        Mathematically, the condition is given by
                            { z_cc,    h_frac < z_cc
            ghost_h_frac =  {
                            { h_frac,  otherwise
        """

        if isinstance(h_frac, pp.ad.Ad_array):
            # Create broadcasting matrix. This is needed to obtain the right shape of the
            # Jacobian since PorePy won't do that for us automatically.
            broadcaster = sps.csr_matrix(np.ones_like(self._cc)).reshape((-1, 1))
            # Broadcasted hydraulic head
            ghost_h_frac = broadcaster * h_frac
            # Perform sanity check to the size of the Jacobian
            if not ghost_h_frac.jac.shape[0] == self._cc.shape[0]:
                raise ValueError(
                    f"Expected Jacobian with {self._cc.shape[0]} rows. "
                    f"Got {ghost_h_frac.jac.shape[0]} instead."
                )
            # Now, we need to correct the values of the hydraulic head for the dry parts of
            # the fracture domain. Essentially, in these cells, the pressure head = 0,
            # since there is only air. Therefore h = 0 + z_cc = z_cc for the dry cells.
            dry_cells = ghost_h_frac.val < self._cc
            ghost_h_frac.val[dry_cells] = self._cc[dry_cells]
        else:
            ghost_h_frac = h_frac * np.ones_like(self._cc)
            dry_cells = ghost_h_frac < self._cc
            ghost_h_frac[dry_cells] = self._cc[dry_cells]

        return ghost_h_frac


class FractureVolume:
    def __init__(self, ghost_grid: pp.Grid, data: dict, param_key: str):

        self._g: pp.Grid = ghost_grid
        self._d: dict = data
        self._param_key: str = param_key

        # Get datum in absolute coordinates
        if self._g.dim == 2:  # if fracture grid is 2D, then ambient dimension is 3D
            datum: float = np.min(self._g.face_centers[2])
        elif self._g.dim == 1:  # if fracture grid is 1D, then ambient dimension is 2D
            datum: float = np.min(self._g.face_centers[1])
        else:
            raise NotImplementedError(f"Grid cannot be of dimension {self._g.dim}")
        self._datum: float = datum

        # Get aperture
        aperture: float = self._d[pp.PARAMETERS][self._param_key]["aperture"]
        if not (isinstance(aperture, int) or isinstance(aperture, float)):
            raise ValueError("Aperture can only be a scalar for now.")
        self._aperture: float = aperture

        # Get cell centers
        if self._g.dim == 2:
            cell_centers: np.ndarray = self._g.cell_centers[2]
        elif self._g.dim == 1:
            cell_centers: np.ndarray = self._g.cell_centers[1]
        else:
            raise NotImplementedError(f"Grid cannot be of dimension {self._g.dim}")
        self._cc: np.ndarray = cell_centers

    def __repr__(self) -> str:
        return "Hydrostatic water fracture pressure Ad operator"

    def fracture_volume(
        self, hydraulic_head: Union[AdArray, NonAd]
    ) -> Union[AdArray, NonAd]:

        # Get grid volume
        grid_vol: float = self._g.cell_volumes.sum() * self._aperture

        # In its most basic form, the water volume V_f is related to the hydraulic head h_f
        # via the following piecewise-linear relationship:
        #
        #               { aper * (h_f - datum),    h_f <= grid_vol/aper + datum
        #   V_f(h_f) =  {
        #               { grid_vol          ,      otherwise
        #
        # Note that this relationship is only valid for water in hydrostatic equilibrium
        # BUG: Most likely there is a bug here in the else statement, note that water volume
        #  will not be a function of the hydraulic_head anymore, and in the way is
        #  represented right now, it will not even be an pp.ad.Ad_array object
        if isinstance(hydraulic_head, pp.ad.Ad_array):
            water_volume = self._aperture * (hydraulic_head - self._datum)
            # condition = hydraulic_head.val <= (grid_vol / self._aperture + self._datum)
            # Correct values for dry cells
            water_volume.val[water_volume.val >= grid_vol] = grid_vol
        else:
            water_volume = self._aperture * (hydraulic_head - self._datum)
            # condition = hydraulic_head <= (grid_vol / self._aperture + self._datum)
            # Correct values for dry cells
            water_volume[water_volume >= grid_vol] = grid_vol

        return water_volume

    def volume_capacity(self, hydraulic_head: Union[AdArray, NonAd]) -> NonAd:
        """Computes the derivative of the volume with respect to the hydraulic head

         Parameters:
             hydraulic_head: NonAd object for the moment

         Returns:
            derivative of the volume with respect to the hydraulic head
        """
        # Get grid volume
        grid_vol: float = self._g.cell_volumes.sum() * self._aperture

        if isinstance(hydraulic_head, pp.ad.Ad_array):
            raise ValueError("Hydraulic head cannot be pp.ad.Ad_Array.")
        else:
            is_full = self._aperture * (hydraulic_head + self._datum) >= grid_vol
            if not is_full:
                return self._aperture
            else:
                return 0


class HydrostaticFracturePressure:
    """
    Given a volume of water, it returns the hydrostatic pressure head in the fracture. It is assumed
    that the aperture is given as a field in the grid's data dictionary. If the pressure calculated
    pressure head is negative, the value for that cell is corrected and assumed to be zero.
    """

    def __init__(self, ghost_grid: pp.Grid, data: dict, param_key: str):

        self._g: pp.Grid = ghost_grid
        self._d: dict = data
        self._param_key: str = param_key

        # Get datum in absolute coordinates
        if self._g.dim == 2:  # if fracture grid is 2D, then ambient dimension is 3D
            datum: float = np.min(self._g.face_centers[2])
        elif self._g.dim == 1:  # if fracture grid is 1D, then ambient dimension is 2D
            datum: float = np.min(self._g.face_centers[1])
        else:
            raise NotImplementedError(f"Grid cannot be of dimension {self._g.dim}")
        self._datum: float = datum

        # Get aperture
        aperture: float = self._d[pp.PARAMETERS][self._param_key]["aperture"]
        if not (isinstance(aperture, int) or isinstance(aperture, float)):
            raise ValueError("Aperture can only be a scalar for now.")
        self._aperture: float = aperture

        # Get cell centers
        if self._g.dim == 2:
            cell_centers: np.ndarray = self._g.cell_centers[2]
        elif self._g.dim == 1:
            cell_centers: np.ndarray = self._g.cell_centers[1]
        else:
            raise NotImplementedError(f"Grid cannot be of dimension {self._g.dim}")
        self._cc: np.ndarray = cell_centers

    def __repr__(self) -> str:
        return "Hydrostatic water fracture pressure Ad operator"

    def get_pressure_head(
        self, water_volume: Union[pp.ad.Ad_array, Scalar, np.ndarray]
    ) -> Union[pp.ad.Ad_array, Scalar, np.ndarray]:
        """
        Computes pressure head for a ghost fracture grid given the water volume.

        Parameters
            water_volume (Ad_array or Scalar or np.ndarray of size 1): water volume. This can
            be an active Ad_array of val.size = 1 or a non-ad scalar object.
        Return

        """

        if isinstance(water_volume, pp.ad.Ad_array):
            # Create broadcasting matrix. This is needed to obtain the right shape of the
            # Jacobian since PorePy won't do that for us automatically.
            broadcaster = sps.csr_matrix(np.ones_like(self._cc)).reshape((-1, 1))
            # Obtain the height of the air-water interface in absolute coordinates
            # TODO: This might be updated when we move to two-dimensional fractures
            airwater_interface = water_volume * (1 / self._aperture) + self._datum
            # Get the value of the pressure head
            pressure_head = broadcaster * airwater_interface - self._cc
            # Perform sanity check on the size of the Jacobian
            if not pressure_head.jac.shape[0] == self._cc.shape[0]:
                raise ValueError(
                    f"Expected Jacobian with {self._cc.shape[0]} rows. "
                    f"Got {pressure_head.jac.shape[0]} instead."
                )
            # Now, we need to correct the values of the pressure head since negative values
            # are not permitted. Again, this might change when we handle different capillary
            # barrier values
            pressure_head.val[pressure_head.val < 0] = 0
        else:
            airwater_interface = water_volume / self._aperture + self._datum
            pressure_head = airwater_interface - self._cc
            pressure_head[pressure_head < 0] = 0

        return pressure_head


# %% INTERFACE UPSTREAM WEIGHTING
class InterfaceUpwindAd(ApplicableOperator):
    """
    Computes the interface relative permeabilities based on the (projected)
    pressure jump associated with the bulk and fractur potentials.
    """

    def __init__(self):

        self._set_tree()

    def __repr__(self) -> str:
        return "Interface upwind AD operator"

    # TODO: Add sanity check to check if input matches amount of mortar cells in gb
    # TODO: Write tests
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
        if (
            isinstance(trace_p_bulk, pp.ad.Ad_array)
            or isinstance(krw_trace_p_bulk, pp.ad.Ad_array)
            or isinstance(p_frac, pp.ad.Ad_array)
            or isinstance(krw_p_frac, pp.ad.Ad_array)
        ):
            raise TypeError("Input cannot be of type Ad array")
        else:
            pressure_jump = trace_p_bulk - p_frac
            hs_10 = heaviside(pressure_jump, zerovalue=0)
            hs_01 = heaviside(-pressure_jump, zerovalue=0)
            vals = hs_10 * krw_trace_p_bulk + hs_01 * krw_p_frac
            n = len(trace_p_bulk)
            interface_krw = sps.spdiags(vals, 0, n, n)

        return interface_krw


# %% BULK FACE AVERAGING SCHEMES

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

        if isinstance(inner_values, pp.ad.Ad_array):
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
            face_avg = np.ones(self._g.num_faces)  # Neumann krw=1.0

            # Values at Dirichlet boundaries
            dir_cells_neigh = fcs_neigh[dir_fcs]
            if dir_cells_neigh.size > 0:
                dir_cells = dir_cells_neigh[(dir_cells_neigh >= 0).nonzero()]
                face_avg[dir_fcs] = 0.5 * (
                    dir_bound_values[dir_fcs] + inner_values[dir_cells]
                )

            # Values at internal faces
            face_avg[int_fcs] = 0.5 * (
                inner_values[int_fcs_neigh[:, 0]] + inner_values[int_fcs_neigh[:, 1]]
            )

        return sps.spdiags(face_avg, 0, self._g.num_faces, self._g.num_faces)


# Flux-based upwinding scheme
class UpwindFluxBasedAd(ApplicableOperator):
    """Flux based upwinding of cell-center arrays"""

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
        self._cf_is_dir = [
            np.logical_and(cf_is_boundary[i], self._is_dir) for i in range(0, 2)
        ]
        self._is_neu = d[pp.PARAMETERS][param_key]["bc"].is_neu.copy()
        self._cf_is_neu = [
            np.logical_and(cf_is_boundary[i], self._is_neu) for i in range(0, 2)
        ]

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
        if isinstance(inner_values, pp.ad.Ad_array) or isinstance(
            face_flux, pp.ad.Ad_array
        ):
            raise TypeError("Object cannot be of the type pp.ad.Ad_array.")
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


# %% SOIL WATER RETENTION CURVES
class SoilWaterRetentionCurves:
    """Parent class for soil-water retention curves (SWRC)."""

    def __init__(self, gb: pp.GridBucket, param_key: str):
        """Init method for the class.

        Parameters:
            gb (GridBucket): Mixed-dimensional grid bucket. Note that we assume that the
                relevant parameters, i.e., "alpha_vG", "theta_r", "theta_s", "n_vG", "m_vG",
                "hydrostatic_volume" exist in the grid data dictionaries.
            param_key (str): Keyword to access the data parameter.

        """

        self._gb: pp.GridBucket = gb
        self._kw: str = param_key

        # Note that assume that the SWRC parameters are stored in the dictionary of
        # the ambient grid. In the future, we could assign different SWRC to each grid.
        # However, for this project, this works justs fine.
        g_bulk: pp.Grid = self._gb.grids_of_dimension(self._gb.dim_max())[0]
        d_bulk: dict = self._gb.node_props(g_bulk)
        param_dict: dict = d_bulk[pp.PARAMETERS][self._kw]
        self.alpha_vG: Scalar = param_dict["alpha_vG"]  # alpha parameter
        self.theta_r: Scalar = param_dict["theta_r"]  # residual water content
        self.theta_s: Scalar = param_dict["theta_s"]  # water content at sat contditions
        self.n_vG: Scalar = param_dict["n_vG"]  # n parameter
        self.m_vG: Scalar = param_dict["m_vG"]  # m parameter

    def __repr__(self):
        # TODO: Add a proper __repr__ showing the attributes
        return "Soil-water retention curve (van Genuchten-Mualem) object."

    def water_content(
        self, pressure_head: Union[AdArray, NonAd]
    ) -> Union[AdArray, NonAd]:
        """Water content as function of the pressure head.

        Parameters:
            pressure_head (Ad-array or non-ad object): pressure head
        Returns:
            theta (Ad-array or non-ad object): water content
        """

        if isinstance(pressure_head, pp.ad.Ad_array):
            is_unsat = self._is_unsat(pressure_head.val)
            is_sat = 1 - is_unsat
            num = self.theta_s - self.theta_r
            den = (
                1 + (self.alpha_vG * pp.ad.abs(pressure_head)) ** self.n_vG
            ) ** self.m_vG
            theta = (
                num * den ** (-1) + self.theta_r
            ) * is_unsat + self.theta_s * is_sat
        else:  # typically int, float, or np.ndarray
            is_unsat = self._is_unsat(pressure_head)
            is_sat = 1 - is_unsat
            num = self.theta_s - self.theta_r
            den = (
                1 + (self.alpha_vG * np.abs(pressure_head)) ** self.n_vG
            ) ** self.m_vG
            theta = (num / den + self.theta_r) * is_unsat + self.theta_s * is_sat

        return theta

    def effective_saturation(
        self, pressure_head: Union[AdArray, NonAd]
    ) -> Union[AdArray, NonAd]:
        """Effective saturation as a function of the pressure head.

        Parameters:
            pressure_head (Ad-array or non-ad object): pressure head
        Returns:
            s_eff (Ad-array or non-ad object): effective (normalized) saturation
        """

        num = self.water_content(pressure_head) - self.theta_r
        den = self.theta_s - self.theta_r
        s_eff = num * den ** (-1)

        return s_eff

    def relative_permeability(self, pressure_head: NonAd) -> NonAd:
        """Relative permeability as a function of the pressure head.

        Parameters:
            pressure_head (non-ad object): pressure head
        Returns:
            krw (non-ad object): water relative permeability
        """

        # TODO: Add possibility to pass an pp.ad.Array
        if isinstance(pressure_head, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:  # typically int, float, or np.ndarray
            s_eff = self.effective_saturation(pressure_head)
            krw = s_eff ** 0.5 * (1 - (1 - s_eff ** (1 / self.m_vG)) ** self.m_vG) ** 2

        return krw

    def moisture_capacity(self, pressure_head: NonAd) -> NonAd:
        """Specific moisture capacity as a function of the pressure head

        Parameters:
            pressure_head (non-ad object): pressure head
        Returns:
            moist_capacity (non-ad object): moisture capacitiy, i.e., d(theta)/d(psi).
        """

        # TODO: Add possibility to pass an pp.ad.Array
        if isinstance(pressure_head, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:  # typically int, float, or np.ndarray
            is_unsat = self._is_unsat(pressure_head)
            is_sat = 1 - is_unsat
            num = (
                -self.m_vG
                * self.n_vG
                * (self.theta_s - self.theta_r)
                * (self.alpha_vG * np.abs(pressure_head)) ** self.n_vG
            )
            den = pressure_head * (
                (self.alpha_vG * np.abs(pressure_head)) ** self.n_vG + 1
            ) ** (self.m_vG + 1)
            # Here, we have to be particulary careful with zero division. If zero is
            # encountered in the denominator, we force the moisture capacity to be zero.
            moist_capacity = (
                np.divide(num, den, out=np.zeros_like(num), where=den != 0)
                * self._is_unsat(pressure_head)
                + 0 * is_sat
            )

        return moist_capacity

    # Helpers
    @staticmethod
    def _is_unsat(pressure_head: NonAd) -> NonAd:
        """Determines whether is saturated or not based on the value of the pressure head.

        Parameters
            pressure_head (non-ad): containing the values of the pressure heads.
        Returns
            non-ad: 1 if pressure_head < 0, and 0 otherwise.
        """
        if isinstance(pressure_head, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:
            return 1 - heaviside(pressure_head, 1)


class vanGenuchten:
    """ "Parent class for van Genuchten-Mualem soil-water retention curves"""

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
        """Determine whether the cell is saturated or not"""

        if isinstance(p, pp.ad.Ad_array):
            raise TypeError("Pressure cannot be AD. Expected inactive variable.")
        else:
            # {1, pressure_head < 0
            # {0, otherwise
            return 1 - heaviside(p, 1)

    def water_content(self, p):
        """Water content as a function of the pressure head"""

        if isinstance(p, pp.ad.Ad_array):
            is_unsat = self.is_unsat(p.val)
            is_sat = 1 - is_unsat
            num = self.theta_s - self.theta_r
            den = (1 + (self.alpha_vG * pp.ad.abs(p)) ** self.n_vG) ** self.m_vG
            theta = (
                num * den ** (-1) + self.theta_r
            ) * is_unsat + self.theta_s * is_sat
        else:
            is_unsat = self.is_unsat(p)
            is_sat = 1 - is_unsat
            num = self.theta_s - self.theta_r
            den = (1 + (self.alpha_vG * np.abs(p)) ** self.n_vG) ** self.m_vG
            theta = (num / den + self.theta_r) * is_unsat + self.theta_s * is_sat

        return theta

    def effective_saturation(self, p):
        """Effective saturation as a function of the water content"""

        num = self.water_content(p) - self.theta_r
        den = self.theta_s - self.theta_r
        s_eff = num * den ** (-1)

        return s_eff

    def relative_permeability(self, p):
        """Relative permeability as a function of the effective saturation"""

        if isinstance(p, pp.ad.Ad_array):
            raise TypeError("Pressure cannot be AD. Expected previous_iteration()")
        else:
            krw = (
                self.effective_saturation(p) ** (0.5)
                * (
                    1
                    - (1 - self.effective_saturation(p) ** (1 / self.m_vG)) ** self.m_vG
                )
                ** 2
            )

        return krw

    def moisture_capacity(self, p):
        """Specific moisture capacity as a function of the pressure head"""

        if isinstance(p, pp.ad.Ad_array):
            raise TypeError("Pressure cannot be AD. Expected previous_iteration()")
        else:
            is_unsat = self.is_unsat(p)
            is_sat = 1 - is_unsat
            num = (
                -self.m_vG
                * self.n_vG
                * (self.theta_s - self.theta_r)
                * (self.alpha_vG * np.abs(p)) ** self.n_vG
            )
            den = p * ((self.alpha_vG * np.abs(p)) ** self.n_vG + 1) ** (self.m_vG + 1)
            C = (
                np.divide(num, den, out=np.zeros_like(num), where=den != 0)
                * self.is_unsat(p)
                + 0 * is_sat
            )

        return C
