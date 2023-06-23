"""
Collection of ad operators of extended functionality relative to the ones available in PorePy.
"""

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

__all__ = ["FluxBaseUpwindAd", "InterfaceUpwindAd", "ParameterScalar"]


class FluxBaseUpwindAd(ApplicableOperator):
    """Flux based upwinding for faces"""

    # Credits: @jwboth

    def __init__(
        self,
        gb: pp.GridBucket,
        grid_list: List[pp.Grid],
        param_key: str,
        hs: Callable = pp.ad.heaviside,
    ):

        super().__init__()
        self._set_tree()
        self._gb = gb
        self._grid_list = grid_list
        self._kw = param_key
        self._hs = hs

        # Sanity check
        if len(grid_list) > 1:
            raise NotImplementedError(
                "FluxBaseUpwindAd implemented only for a single grid"
            )

        self._g = grid_list[0]
        self._d = gb.node_props(self._g)

        # Construct projection from cell-valued arrays to face-valued arrays with values to the
        # "left" and "right" of the face, here denoted by '0' and '1', respectively.
        cf_dense = self._g.cell_face_as_dense()
        cf_inner = [c >= 0 for c in cf_dense]

        row = [np.arange(self._g.num_faces)[cf_inner[i]] for i in range(0, 2)]
        col = [cf_dense[i][cf_inner[i]] for i in range(0, 2)]
        data = [np.ones_like(row[i]) for i in range(0, 2)]
        self._cf_inner = [
            sps.csr_matrix(
                (data[i], (row[i], col[i])),
                shape=(self._g.num_faces, self._g.num_cells),
                dtype=float,
            )
            for i in range(0, 2)
        ]

        # Store which 'left' and 'right' cells of all faces correspond to the Dirichlet
        # boundary.
        cf_is_boundary = np.logical_not(cf_inner)
        self._cf_is_boundary = cf_is_boundary
        self._is_dir = self._d[pp.PARAMETERS][param_key]["bc"].is_dir.copy()
        self._cf_is_dir = [
            np.logical_and(cf_is_boundary[i], self._is_dir) for i in range(0, 2)
        ]
        self._is_neu = self._d[pp.PARAMETERS][param_key]["bc"].is_neu.copy()
        self._cf_is_neu = [
            np.logical_and(cf_is_boundary[i], self._is_neu) for i in range(0, 2)
        ]

    def __repr__(self) -> str:
        return " Flux-based upwind AD face operator"

    def apply(
        self,
        inner_values: np.ndarray,
        dir_bound_values: np.ndarray,
        face_flux: np.ndarray,
    ) -> sps.spdiags:
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
            Upstream values at the faces of the grid

        """

        # Use Dirichlet boundary data where suitable.
        # Neglect Neumann boundaries since Neumann boundary data does not play a role.
        if isinstance(inner_values, pp.ad.Ad_array) or isinstance(
            face_flux, pp.ad.Ad_array
        ):
            raise TypeError("Object cannot be of the type pp.ad.Ad_array.")
        else:
            val_f = [self._cf_inner[i] * inner_values for i in range(0, 2)]
            for i in range(0, 2):
                val_f[i][self._cf_is_boundary[i]] = dir_bound_values[
                    self._cf_is_boundary[i]
                ]

        # Evaluate the Heaviside function of the "flux directions".
        hs_f_01 = self._hs(face_flux)
        hs_f_10 = self._hs(-face_flux)

        # Determine the face mobility by utilizing the general idea (see above).
        face_upwind = val_f[0] * hs_f_01 + val_f[1] * hs_f_10

        # Deal with Neumann boundary conditions
        face_upwind[self._is_neu] = 1.0

        return sps.spdiags(face_upwind, 0, self._g.num_faces, self._g.num_faces)


class InterfaceUpwindAd(ApplicableOperator):
    """
    Computes the interface relative permeabilities based on the (projected)
    pressure jump associated with the bulk and fracture pressure heads.
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
        grids: Optional[List[pp.Grid]] = None,
        edges: Optional[List[Edge]] = None,
    ):
        """Construct a wrapper for a scalar parameter for a grid list or an edge list

        Parameters:
            param_keyword (str): Keyword that should be used to access the data dictionary
                to get the relevant parameter dictionary.
            grids (List of pp.Grid): List containing the grid from which the parameter
                scalar should be retrieved. If len(grid_list) > 1, an error will be raised.
            edges (Tuple of pp.Grid): List containing the edge from which the parameter
                scalar should be retrieved. If len(edge_list) > 1, an error will be raised.

        Example:
            dt_ad = ParameterScalar("flow", "time_step", grid_list=[g])
        """

        super().__init__()

        if (grids is None) and (edges is None):
            raise ValueError(
                "ParameterScalar needs at least a grid list or an edge list."
            )
        if (grids is not None) and (edges is not None):
            raise ValueError("grid_list and edge_list cannot be passed simultaneously.")
        if grids is not None and len(grids) > 1:
            raise ValueError("Expected a grid list with only one grid.")
        if edges is not None and len(edges) > 1:
            raise ValueError("Expected an edge list with only one edge.")

        self.param_keyword = param_keyword
        self.scalar_keyword = scalar_keyword
        self._grid_list = grids
        self._edge_list = edges

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
