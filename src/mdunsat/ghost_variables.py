"""
Classes for handling ghost variables. That is, variables living in the ghost grid bucket.
"""

import porepy as pp
import numpy as np
import scipy.sparse as sps
from porepy.numerics.ad.operators import Operator, ApplicableOperator
from typing import Callable, Optional, Tuple, List, Any, Union, NewType, Literal

# Typing abbreviations
Scalar = Union[int, float]
AdArray = pp.ad.Ad_array
MergedAdVariable = pp.ad.MergedVariable
NonAd = Union[Scalar, np.ndarray]
Edge = Tuple[pp.Grid, pp.Grid]


__all__ = ["GhostHydraulicHead"]


class GhostHydraulicHead:
    """Projects the fracture hydraulic head onto the mortar grids"""

    def __init__(self, gb, ghost_gb, dof_manager):
        # TODO: ADD DOC

        self._gb: pp.GridBucket = gb
        self._ghost_gb: pp.GridBucket = ghost_gb
        self._dof_manager: pp.DofManager = dof_manager
        self._maxdim: int = self._gb.dim_max()

        # Get list of ghost grids and ghost edges
        ghost_grids: List[pp.Grid] = []
        ghost_edges: List[Edge] = []
        for g, _ in self._ghost_gb:
            if g.dim >= self._maxdim - 1:
                ghost_grids.append(g)
        for e, d in self._ghost_gb.edges():
            mg = d["mortar_grid"]
            if mg.dim >= self._maxdim - 1:
                ghost_edges.append(e)
        self._ghost_grids = ghost_grids
        self._ghost_edges = ghost_edges

        # Get mortar projections
        ghost_mortar_proj: pp.ad.MortarProjections = pp.ad.MortarProjections(
            gb=self._ghost_gb, grids=self._ghost_grids, edges=self._ghost_edges
        )
        self._ghost_mortar_proj = ghost_mortar_proj

        # Get subdomain projections
        ghost_subdomain_proj: pp.ad.SubdomainProjections = pp.ad.SubdomainProjections(
            grids=self._ghost_grids
        )
        self._ghost_subdomain_proj = ghost_subdomain_proj

        # Get elevation heads of the fracture grids by concatenating all ndarrays into one
        ghost_frac_grids: List[pp.Grid] = []
        for g, _ in self._ghost_gb:
            if g.dim == self._maxdim - 1:
                ghost_frac_grids.append(g)
        self._cc: np.ndarray = np.concatenate(
            [g.cell_centers[self._maxdim - 1] for g in ghost_frac_grids]
        )
        self._ghost_frac_grids = ghost_frac_grids

    # Public methods
    def proj_fra_hyd_head(self, as_ad=False) -> "_proj_frac_hyd_head":
        """ "Projects the fracture hydraulic head onto the mortar space.

        Parameters:
            as_ad : If true, the function is passed as an pp.ad.Function, otherwise it is
                passed as a regular class method.
        Returns:
            The class method _proj_frac_hyd_head. Either as an pp.ad.Function, or as regular
            function.
        """

        if as_ad:
            return pp.ad.Function(
                self._proj_frac_hyd_head, name="Projected fracture hyd head"
            )
        else:
            return self._proj_frac_hyd_head

    # Private methods
    def _proj_frac_hyd_head(self, h_frac: Union[AdArray, NonAd]) -> [AdArray, NonAd]:

        if isinstance(h_frac, pp.ad.Ad_array):
            # Broadcast the fracture hydraulic head. The resulting ad operator will consist of
            # concatanated broadcasted hydraulic heads. The size of the operator is given by
            # the number of ghost fracture cells on each fracture grid.
            broad_matrix: sps.lil_matrix = self._get_broadcasting_matrix()
            hfrac_broad: pp.ad.Ad_array = broad_matrix * h_frac

            # This chunk might be helpful if we're doing things the non-hacky way
            # We need to correct the values of the hydraulic head for the dry parts of
            # the fracture domain. Essentially, in these cells, the pressure head = 0,
            # since there is only air. Therefore h = 0 + z_cc = z_cc for the dry cells.
            # Note that by multiplying by the wet and dry matrices the Jacobian
            # is 0 for the dry cells. This could indeed be problematic. If this is the case,
            # we will have to somehow manually change the Jacobian and fill with ones the
            # relevant columns
            # wet_mat, dry_mat = self._get_wet_and_dry_cells(hfrac_broad)
            # hfrac: pp.ad.Operator = wet_mat * hfrac_broad + dry_mat * self._cc

            # Now, we need to correct the values of the hydraulic head for the dry parts of
            # the fracture domain. Essentially, in these cells, the pressure head = 0,
            # since there is only air. Therefore h = 0 + z_cc = z_cc for the dry cells.
            # WARNING: Note that we are only changing the "values" of the hydraulic head
            # ad_Array, but its Jacobian remains unchanged. Not sure about the repercusion
            # that this might have. But if we need to do things correctly, we should apply
            # something on the lines of the chunk from above :)
            dry_cells: np.ndarray = hfrac_broad.val < self._cc
            hfrac_broad.val[dry_cells] = self._cc[dry_cells]

            # Now we are ready to project the hydraulic head onto the mortar grids. To this
            # aim, we first need the relevant subdomain projections and mortar projections.
            cell_prolongation = self._ghost_subdomain_proj.cell_prolongation(
                grids=self._ghost_frac_grids
            ).parse(gb=self._ghost_gb)

            secondary_to_mortar_avg = self._ghost_mortar_proj.secondary_to_mortar_avg.parse(
                self._ghost_gb)

            proj_h_frac: pp.ad.Ad_array = (
                secondary_to_mortar_avg * cell_prolongation * hfrac_broad
            )
        else:
            # Check proper doc above
            broad_matrix: sps.csr_matrix = self._get_broadcasting_matrix()
            hfrac_broad: np.ndarray = broad_matrix * h_frac
            dry_cells = hfrac_broad < self._cc
            hfrac_broad[dry_cells] = self._cc[dry_cells]
            cell_prolongation = self._ghost_subdomain_proj.cell_prolongation(
                grids=self._ghost_frac_grids
            ).parse(gb=self._ghost_gb)
            secondary_to_mortar_avg = self._ghost_mortar_proj.secondary_to_mortar_avg.parse(
                self._ghost_gb)
            proj_h_frac = secondary_to_mortar_avg * cell_prolongation * hfrac_broad

        return proj_h_frac

    def _get_broadcasting_matrix(self) -> sps.lil_matrix:
        """
        Returns the broadcasting matrix that acts on the fracture hydraulic head

        Returns:
            broad_mat (pp.ad.Matrix): Broadcasting matrix of size (total_num_of_frac_cells x
                num_fractures).
        """

        # Create a list where each item is the number of cells corresponding to each frac grid
        num_cells: List[int] = [g.num_cells for g in self._ghost_frac_grids]

        # The number of rows of the broadcasting matrix corresponds to the total number of
        # fracture cells. The number of columns corresponds to the total number of
        # fractures, which must also coincide with the size of the active ad variable h_frac.
        rows: int = sum(num_cells)
        cols: int = len(self._ghost_frac_grids)
        broad_matrix = sps.lil_matrix(np.zeros(shape=(rows, cols)))

        # Populate the matrix column-wise with ones.
        for col in range(0, cols):
            broad_matrix[col * num_cells[col]: (col + 1) * num_cells[col], col] = 1

        return broad_matrix

    def _get_wet_and_dry_cells(
        self, broadcasted_h_frac: pp.ad.Ad_array
    ) -> Tuple[sps.dia_matrix, sps.dia_matrix]:
        """
        Determines wheter a fracture cell is wet or dry.

        Parameters:
            broadcasted_h_frac (pp.ad.Operator): broadcasted fracture hydraulic head.

        Returns:
            Tuple of wet and dry diagonal sparse matrices
        """
        # Substract the elevation head. Note that if h - z < 0 it means that the elevation
        # head is greater than the hydraulic head, implying that the pressure head is
        # negative, and consequently not saturated. The wet_cells np.array has elements
        # equal to 1 if the cell is wet, and 0 otherwise
        wet_cells = np.heaviside(broadcasted_h_frac.val - self._cc, 0)
        dry_cells = 1 - wet_cells
        wet_cell_sp = sps.spdiags(wet_cells, 0, wet_cells.size, wet_cells.size)
        dry_cell_sp = sps.spdiags(dry_cells, 0, dry_cells.size, dry_cells.size)

        return wet_cell_sp, dry_cell_sp
