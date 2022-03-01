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
    pressure_threshold = -22.1  # TODO: This has to be retrieved from some data dictionary
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


def set_state_as_iterate(gb: pp.GridBucket, node_var: str, edge_var: str):
    """
    Sets pp.STATE as pp.ITERATE in all nodes and edges of the grid bucket.

    Parameters:
        gb (Grid Bucket): Mixed-dimensional grid bucket
        node_var (str): Keyword for the node variable, i.e., "hydraulic_head"
        edge_var (str): Keyword for the edge variable, i.e., "mortar_flux"

    """

    # Loop through all nodes of the gb
    for _, d in gb:
        pp.set_state(data=d, state={node_var: d[pp.STATE][pp.ITERATE][node_var]})

    # Loop through all the edges of the gb
    for _, d in gb.edges():
        pp.set_state(data=d, state={edge_var: d[pp.STATE][pp.ITERATE][edge_var]})


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
    tol = 1e-3

    # Sanity check
    for g in fracture_list:
        if g.dim == gb.dim_max():
            raise ValueError("Function meant only for fracture grids.")

    is_negative = False
    for g in fracture_list:
        d = gb.node_props(g)
        if np.any(d[pp.STATE][pp.ITERATE][fracture_var] < (-12.1 - tol)):  # hardcoded
            is_negative = True

    # Instead of -12.1, we have to use: z_min + pressure_threshold

    return is_negative


def get_ghost_hydraulic_head(ghost_g: pp.Grid, h_frac: np.ndarray) -> np.ndarray:
    """
    Computes the hyd head for a ghost fracture grid given the hyd head with one d.o.f.

    Parameters:
        ghost_g (pp.Grid): Fracture ghost grid
        h_frac (np.ndarray): fracture hydraulic head of physcial grid bucket.

    Returns:
        ghost_h_frac (np.ndarray): hydraulic head corresponding to the ghost
            fracture grid. If the ghost fracture cell is dry, the hydraulic head is equal to
            the elevation of such cell, otherwise, it takes the value of h_frac.

    Technical Note:
    ---------------
    Mathematically, the condition is given by
                        { z_cc,    h_frac < z_cc
        ghost_h_frac =  {
                        { h_frac,  otherwise
    """

    cc = ghost_g.cell_centers[1]
    ghost_h_frac = h_frac * np.ones_like(cc)
    dry_cells = ghost_h_frac < cc
    ghost_h_frac[dry_cells] = cc[dry_cells]

    return ghost_h_frac


class ParameterUpdate:
    """
    Parent class for updating parameters
    """

    def __init__(self, gb, param_key):
        self._gb = gb
        self._param_key = param_key

    def update_normal_diffusivity(
        self,
        is_mortar_conductive: np.ndarray,
        edge_list: List[Tuple[pp.Grid, pp.Grid]],
    ):
        """
        Updates the normal diffusivity for a given set of edges
        """

        for e in edge_list:
            d = self._gb.edge_props(e)
            d[pp.PARAMETERS][self._param_key]["normal_diffusivity"] = (
                is_mortar_conductive
                * d[pp.PARAMETERS][self._param_key]["sat_normal_diffusivity"]
            )

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

# %% Ghost Projections
# class GhostProjection:
#     def __init__(self, gb_ghost: pp.GridBucket, g_fracture: pp.Grid):
#         self._gb = gb_ghost
#         self._gridlist: List[pp.Grid] = [g for g, _ in self._gb]
#         self._edgelist: List[Edge] = [e for e, _ in self._gb.edges()]
#         self._gfrac = g_fracture
#
#         # Get hold of ghost mortar projections
#         proj = pp.ad.MortarProjections(
#             gb=self._gb, grids=self._gridlist, edges=self._edgelist
#         )
#         # We only need secondary -> mortar (average)
#         self._secondary_to_mortar_avg = proj.secondary_to_mortar_avg.parse(gb=self._gb)
#
#         # Get hold of ghost subdomain projections
#         subdomain_projection = pp.ad.SubdomainProjections(grids=self._gridlist)
#         # We need the prolongation from the fracture to the global number of cells
#         self._cell_prolongation = subdomain_projection.cell_prolongation(
#             grids=[self._gfrac]
#         ).parse(gb=self._gb)
#
#     def secondary_to_mortar(self, fracture_pressure: pp.ad.Operator) -> pp.ad.Operator:
#         proj_fracture_pressure = (
#             self._secondary_to_mortar_avg * self._cell_prolongation * fracture_pressure
#         )
#
#         return proj_fracture_pressure


# # %% Fracture pressure-related classes
#
# # CONSIDERED IT DEPRECATED
# class GhostFractureHydraulicHead:
#     """
#     Given the "real" fracture hydraulic head, compute the "ghost" fracture hydraulic head.
#     """
#
#     # TODO: Extend this to several fractures. The idea will be to accept a list of ghost
#     #  grids instead of only one grid.
#     def __init__(self, gb: pp.GridBucket, ghost_grid: pp.Grid):
#         """Init method for the class."""
#
#         self._g: pp.Grid = ghost_grid
#         self._gb: pp.GridBucket = gb
#         self._dim_max: int = self._gb.dim_max()
#
#         # Sanity check
#         if self._g.dim >= self._dim_max:
#             raise ValueError(f"Ghost grid cannot be of dimension {self._g.dim}.")
#
#         # Get cell centers and store them in a list
#         self._cc = self._g.cell_centers[self._dim_max - 1]
#
#     def __repr__(self) -> str:
#         return "Ghost fracture hydraulic head AD object."
#
#     def get_ghost_hyd_head(
#         self, h_frac: Union[AdArray, NonAd]
#     ) -> Union[AdArray, NonAd]:
#         """
#         Computes the hydraulic head for a ghost fracture grid given the hydraulic head with
#         dof = 1.
#
#         Parameters:
#             h_frac (Ad_array or non-ad): fracture hydraulic head of physcial grid bucket.
#             This can be an active Ad_array of val.size = 1 or a non-ad scalar object.
#         Returns:
#             ghost_h_frac (Ad_array or non-ad): hydraulic head corresponding to the ghost
#             fracture grid. If the ghost fracture cell is dry, the hydraulic head is equal to
#             the elevation of such cell, otherwise, it takes the value of h_frac.
#
#         Mathematically, the condition is given by
#                             { z_cc,    h_frac < z_cc
#             ghost_h_frac =  {
#                             { h_frac,  otherwise
#         """
#
#         if isinstance(h_frac, pp.ad.Ad_array):
#             # Create broadcasting matrix. This is needed to obtain the right shape of the
#             # Jacobian since PorePy won't do that for us automatically.
#             broadcaster = sps.csr_matrix(np.ones_like(self._cc)).reshape((-1, 1))
#             # Broadcasted hydraulic head
#             ghost_h_frac = broadcaster * h_frac
#             # Perform sanity check to the size of the Jacobian
#             if not ghost_h_frac.jac.shape[0] == self._cc.shape[0]:
#                 raise ValueError(
#                     f"Expected Jacobian with {self._cc.shape[0]} rows. "
#                     f"Got {ghost_h_frac.jac.shape[0]} instead."
#                 )
#             # Now, we need to correct the values of the hydraulic head for the dry parts of
#             # the fracture domain. Essentially, in these cells, the pressure head = 0,
#             # since there is only air. Therefore h = 0 + z_cc = z_cc for the dry cells.
#             dry_cells = ghost_h_frac.val < self._cc
#             ghost_h_frac.val[dry_cells] = self._cc[dry_cells]
#         else:
#             ghost_h_frac = h_frac * np.ones_like(self._cc)
#             dry_cells = ghost_h_frac < self._cc
#             ghost_h_frac[dry_cells] = self._cc[dry_cells]
#
#         return ghost_h_frac
#
#
# # %% INTERFACE UPSTREAM WEIGHTING
# class InterfaceUpwindAd(ApplicableOperator):
#     """
#     Computes the interface relative permeabilities based on the (projected)
#     pressure jump associated with the bulk and fractur potentials.
#     """
#
#     def __init__(self):
#
#         self._set_tree()
#
#     def __repr__(self) -> str:
#         return "Interface upwind AD operator"
#
#     # TODO: Add sanity check to check if input matches amount of mortar cells in gb
#     # TODO: Write tests
#     def apply(self, trace_p_bulk, krw_trace_p_bulk, p_frac, krw_p_frac):
#         """
#         Apply method for upwinding of interface relative permeabilities.
#
#         Parameters
#         ----------
#         trace_p_bulk : nd-array of size total_num_of_mortar_cells
#             Mortar-projected bulk pressure trace
#         krw_trace_p_bulk : nd-array of size total_num_of_mortar_cells
#             Mortar-projected relative permeabilities of bulk pressure trace.
#         p_frac : nd-array of size total_num_of_mortar_cells
#             Mortar-projected fracture pressures.
#         krw_p_frac : nd-array of size total_num_of_mortar_cells
#             Mortar-projected relative permeabilites of fracture presure
#
#         Raises
#         ------
#         TypeError
#             If one of the input arguments is an Ad Array
#
#         Returns
#         -------
#         interface_krw : Sparse Matrix of size total_num_mortar_cells ** 2
#             Diagonal matrix with each entry representing the value of
#             the relative permeability associated with the mortar cell
#         """
#
#         # Sanity check of input type
#         if (
#             isinstance(trace_p_bulk, pp.ad.Ad_array)
#             or isinstance(krw_trace_p_bulk, pp.ad.Ad_array)
#             or isinstance(p_frac, pp.ad.Ad_array)
#             or isinstance(krw_p_frac, pp.ad.Ad_array)
#         ):
#             raise TypeError("Input cannot be of type Ad array")
#         else:
#             pressure_jump = trace_p_bulk - p_frac
#             hs_10 = heaviside(pressure_jump, zerovalue=0)
#             hs_01 = heaviside(-pressure_jump, zerovalue=0)
#             vals = hs_10 * krw_trace_p_bulk + hs_01 * krw_p_frac
#             n = len(trace_p_bulk)
#             interface_krw = sps.spdiags(vals, 0, n, n)
#
#         return interface_krw