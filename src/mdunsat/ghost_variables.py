from __future__ import annotations

import porepy as pp
import numpy as np
import scipy.sparse as sps
from typing import Callable, Optional, Tuple, List, Any, Union, NewType, Literal

# Typing abbreviations
Scalar = Union[int, float]
AdArray = pp.ad.Ad_array
MergedAdVariable = pp.ad.MergedVariable
NonAd = Union[Scalar, np.ndarray]
Edge = Tuple[pp.Grid, pp.Grid]

__all__ = ["GhostHydraulicHead"]


class GhostHydraulicHead:
    """Handles the discretization and projection of variables living in the ghost grids.

    Args:
        gb (pp.GridBucket): Mixed-dimensional (physical) grid bucket.
        ghost_gb (pp.GridBucket): Mixed-dimensional (ghost) grid bucket.

    Note:
        Fracture grids from gb should have only one degree of freedom, whereas fracture
        grids from the ghost gb should have as many degrees of freedom as their adjacent
        mortar grid.

    Public methods:
        proj_frac_hyd_head: projects the hydraulic head from the ghost secondary grids onto
            the physical mortar grids.
    """

    def __init__(self, gb: pp.GridBucket, ghost_gb: pp.GridBucket):
        self._gb: pp.GridBucket = gb
        self._ghost_gb: pp.GridBucket = ghost_gb

        # Get the maximum dimension
        self._maxdim: int = self._gb.dim_max()

        # Get the list of ghost grids with g.dim >= (self._maxdim - 1).
        ghost_grids: List[pp.Grid] = []
        for g, _ in self._ghost_gb:
            if g.dim >= self._maxdim - 1:
                ghost_grids.append(g)
        self._ghost_grids = ghost_grids

        # Get the list of ghost grids with g.dim == self._maxdim (ghost fracture grids).
        ghost_frac_grids: List[pp.Grid] = []
        for g, _ in self._ghost_gb:
            if g.dim == self._maxdim - 1:
                ghost_frac_grids.append(g)
        self._ghost_frac_grids = ghost_frac_grids

        # Get the list of edges with mg.dim == self._maxdim (ghost edges).
        ghost_edges: List[Edge] = []
        for e, d in self._ghost_gb.edges():
            mg = d["mortar_grid"]
            if mg.dim >= self._maxdim - 1:
                ghost_edges.append(e)
        self._ghost_edges = ghost_edges

        # Get the ghost mortar projections.
        ghost_mortar_proj: pp.ad.MortarProjections = pp.ad.MortarProjections(
            gb=self._ghost_gb, grids=self._ghost_grids, edges=self._ghost_edges
        )
        self._ghost_mortar_proj = ghost_mortar_proj

        # Get the ghost subdomain projections.
        ghost_subdomain_proj: pp.ad.SubdomainProjections = pp.ad.SubdomainProjections(
            grids=self._ghost_grids
        )
        self._ghost_subdomain_proj = ghost_subdomain_proj

        # Get the elevation heads of the fracture grids by concatenating all ndarrays.
        self._cc: np.ndarray = np.concatenate(
            [g.cell_centers[self._maxdim - 1] for g in ghost_frac_grids]
        )

    def __repr__(self) -> str:
        return "Ghost Hydraulic Head Ad Object."

    # Public methods
    def proj_fra_hyd_head(
        self, as_ad=False
    ) -> Union[pp.ad.Function, _proj_frac_hyd_head]:
        """Projects the fracture hydraulic head onto the mortar space.

        Args:
            as_ad (bool): True if function must be passed as an pp.ad.Function. False if
                function must be pass a bound method.

        Returns:
            The method _proj_frac_hyd_head. Either as a pp.ad.Function or as bound method.
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
            # the number of ghost fracture cells in each fracture grid.
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
                self._ghost_gb
            )

            proj_h_frac: pp.ad.Ad_array = (
                secondary_to_mortar_avg * cell_prolongation * hfrac_broad
            )
        else:
            # Check proper doc above
            broad_matrix: sps.lil_matrix = self._get_broadcasting_matrix()
            hfrac_broad: np.ndarray = broad_matrix * h_frac
            dry_cells = hfrac_broad < self._cc
            hfrac_broad[dry_cells] = self._cc[dry_cells]
            cell_prolongation = self._ghost_subdomain_proj.cell_prolongation(
                grids=self._ghost_frac_grids
            ).parse(gb=self._ghost_gb)
            secondary_to_mortar_avg = self._ghost_mortar_proj.secondary_to_mortar_avg.parse(
                self._ghost_gb
            )
            proj_h_frac = secondary_to_mortar_avg * cell_prolongation * hfrac_broad

        return proj_h_frac

    def _get_broadcasting_matrix(self) -> sps.lil_matrix:
        """
        Compute broadcasting matrix that acts on the fracture hydraulic head.

        Returns:
            broad_mat (sps.lil_matrix): Broadcasting matrix of size (total_num_of_frac_cells x
                num_fractures).
        """

        # Create a list where item is number of cells corresponding to each frac grid
        num_cells: List[int] = [g.num_cells for g in self._ghost_frac_grids]

        # The number of rows of the broadcasting matrix corresponds to the total number of
        # fracture cells. The number of columns corresponds to the total number of
        # fractures, which must also coincide with the size of the active ad variable h_frac.
        rows: int = sum(num_cells)
        cols: int = len(self._ghost_frac_grids)
        broad_matrix = sps.lil_matrix(np.zeros(shape=(rows, cols)))

        # Populate the matrix column-wise. The idea is to set to 1 the elements
        # corresponding to the number of ghost cells in each col, and leave the rest of the
        # elements untouched.
        for col in range(0, cols):
            broad_matrix[col * num_cells[col] : (col + 1) * num_cells[col], col] = 1

        return broad_matrix

    def _get_wet_and_dry_cells(
        self, broadcasted_h_frac: pp.ad.Ad_array
    ) -> Tuple[sps.dia_matrix, sps.dia_matrix]:
        """
        Determine wheter a fracture cell is wet or dry.

        Args:
            broadcasted_h_frac (pp.ad.Ad_array): concatanated broadcasted fracture hydraulic
                head.

        Returns:
            (wet_mat, dry_mat) (sps.dia_matrix): wet_mat has diagonal elements equal to 1
                if the cell is wet, 0 otherwise. dry_mat has diagonal elements equal to 1 if
                the cell is dry, 0 otherwise.
        """
        # Substract the elevation head. Note that if h - z < 0 it means that the elevation
        # head is greater than the hydraulic head, implying that the pressure head is
        # negative, and consequently not saturated. The wet_cells np.array has elements
        # equal to 1 if the cell is wet, and 0 otherwise
        wet_cells: np.ndarray = np.heaviside(broadcasted_h_frac.val - self._cc, 0)
        dry_cells: np.ndarray = 1.0 - wet_cells
        wet_cell_sp = sps.spdiags(wet_cells, 0, wet_cells.size, wet_cells.size)
        dry_cell_sp = sps.spdiags(dry_cells, 0, dry_cells.size, dry_cells.size)

        return wet_cell_sp, dry_cell_sp
