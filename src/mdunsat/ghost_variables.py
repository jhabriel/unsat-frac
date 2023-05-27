from __future__ import annotations

import porepy as pp
import numpy as np
import scipy.sparse as sps
from typing import Tuple, List, Union

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

    def __init__(
            self,
            gb: pp.GridBucket,
            ghost_gb: pp.GridBucket,
    ):

        # Physical grid bucket
        self._gb: pp.GridBucket = gb

        # Ghost grid bucket
        self._ghost_gb: pp.GridBucket = ghost_gb

        # List grids
        self._ghost_grid: List[pp.Grid] = [
            g for g, _ in self._ghost_gb
        ]

        # List of lower-dimensional grids
        self._ghost_low_dim_grids: List[pp.Grid] = [
            g for g, _ in self._ghost_gb if g.dim < self._gb.dim_max()
        ]

        # List of  edges
        self._ghost_edges: List[Edge] = [
            e for e, _ in self._ghost_gb.edges()
        ]

        # Get the list of ghost grids with g.dim >= (self._maxdim - 1).
        # ghost_grids: List[pp.Grid] = []
        # for g, _ in self._ghost_gb:
        #     if g.dim >= self._maxdim - 1:
        #         ghost_grids.append(g)
        # self._ghost_grids = ghost_grids

        # Get the list of ghost grids with g.dim == self._maxdim (ghost fracture grids).
        # ghost_frac_grids: List[pp.Grid] = []
        # for g, _ in self._ghost_gb:
        #     if g.dim == self._maxdim - 1:
        #         ghost_frac_grids.append(g)
        # self._ghost_frac_grids = ghost_frac_grids

        # Get the list of edges with mg.dim == self._maxdim (ghost edges).
        # ghost_edges: List[Edge] = []
        # for e, d in self._ghost_gb.edges():
        #     mg = d["mortar_grid"]
        #     if mg.dim >= self._maxdim - 1:
        #         ghost_edges.append(e)
        # self._ghost_edges = ghost_edges

        # Ghost mortar proj
        self._ghost_mortar_proj: pp.ad.MortarProjections = pp.ad.MortarProjections(
            gb=self._ghost_gb, grids=self._ghost_grid, edges=self._ghost_edges
        )

        # Ghost subdomain proj
        self._ghost_subdomain_proj: pp.ad.SubdomainProjections = pp.ad.SubdomainProjections(
            grids=self._ghost_grid
        )

        # Get the elevation heads of the lower-dimensional grids by concatenating all ndarrays
        self._cc: np.ndarray = np.concatenate(
            [g.cell_centers[self._gb.dim_max() - 1] for g in self._ghost_low_dim_grids]
        )

        # # Save the pressure threshold projected onto the mortar grids
        # self._mortar_psi_l = mortar_proj_pressure_threshold

        # Concatenate the representative pressure thresholds for each fracture
        self._psi_l: np.ndarray = np.asarray([
            d[pp.PARAMETERS]["flow"]["pressure_threshold"] for g, d in self._gb if
            g.dim < self._gb.dim_max()
        ])

    def __repr__(self) -> str:
        return "Ghost Hydraulic Head Ad Object."

    # Public methods
    # def project_pressure_threshold_onto_fracture(self) -> np.ndarray:
    #     """Project the pressure threshold from interfaces to fractures.
    #
    #     Returns:
    #         Projected pressure threshold in the fractures.
    #
    #     """
    #
    #     # We need to project from the mortar grid onto the fractures
    #     # This will hopefully handle heterogeneous capillary entry pressures.
    #     # Physically coherent results can only be expected when the soil is the same
    #     # at both sides of the fracture, since there is no known rule to average the
    #     # pressure threshold when there is more than one type of soil. My educated
    #     # guess, however, is that a harmonic mean should give decent results.
    #
    #     # Get cell restriction using all subdomain grids of dimension nd-1
    #     cell_restriction = self._ghost_subdomain_proj.cell_restriction(
    #         grids=self._ghost_low_dim_grids
    #     ).parse(gb=self._ghost_gb)
    #
    #     # Get projection matrix from mortar to fractures. Since we are projecting an
    #     # intensive quantity, we use *_avg
    #     mortar_to_secondary_avg = self._ghost_mortar_proj.mortar_to_secondary_avg.parse(
    #         gb=self._ghost_gb,
    #     )
    #
    #     # Project the mortar pressure thresholds onto the fractures
    #     # The factor 0.5 is included to keep the same value of capillary threshold.
    #     # If we do not include this, we'll get the double.
    #     frac_psi_l: np.ndarray = (
    #             0.5 * cell_restriction * mortar_to_secondary_avg * self._mortar_psi_l
    #     )
    #
    #     return frac_psi_l

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
        """Project the fracture hydraulic head onto the interfaces.

        The method deals with Ad_array and numpy arrays.

        The general idea is to broadcast the value of the hydraulic head and then
        perform a matching projection. However, special attention is demanded to
        distinguish between "dry" and "wet" cells.

        """

        #psi_l: np.ndarray = self.project_pressure_threshold_onto_fracture()


        if isinstance(h_frac, pp.ad.Ad_array):
            # Broadcast the fracture hydraulic head. The resulting ad operator will
            # consist of concatanated broadcasted hydraulic heads. The size of the
            # operator is given by the number of ghost fracture cells in each
            # fracture grid.
            broad_matrix: sps.lil_matrix = self._get_broadcasting_matrix()
            hfrac_broad: pp.ad.Ad_array = broad_matrix * h_frac

            # Now, we broadcast the pressure threshold
            psi_L_broad: np.array = broad_matrix * self._psi_l

            # This chunk might be helpful if we're doing things the non-hacky way We
            # need to correct the values of the hydraulic head for the dry parts of
            # the fracture domain. Essentially, in these cells, the pressure head =
            # pressure_threshold, since there is only air. Therefore, h = h_dry =
            # psi_L + z_cc for the dry cells. Note that by multiplying by the wet and
            # dry matrices the Jacobian is 0 for the dry cells. This could indeed be
            # problematic. If this is the case, we will have to somehow manually
            # change the Jacobian and fill with ones the relevant columns wet_mat,
            # dry_mat = self._get_wet_and_dry_cells(hfrac_broad) hfrac:
            # pp.ad.Operator = wet_mat * hfrac_broad + dry_mat * h_dry

            # WARNING: Note that we are only changing the "values" of the hydraulic
            # head ad_Array, but its Jacobian remains unchanged. Not sure about the
            # repercussion that this might have. But if we need to do things
            # correctly, we should apply something on the lines of the chunk from
            # above :)
            dry_cells = (hfrac_broad.val - self._cc) <= psi_L_broad
            hfrac_broad.val[dry_cells] = self._cc[dry_cells] + psi_L_broad[dry_cells]

            # Now we are ready to project the hydraulic head onto the mortar grids.
            # To this aim, we first need the relevant subdomain proj and mortar proj.
            cell_prolongation = self._ghost_subdomain_proj.cell_prolongation(
                grids=self._ghost_low_dim_grids
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
            psi_L_broad: np.array = broad_matrix * self._psi_l
            dry_cells = (hfrac_broad - self._cc) <= psi_L_broad
            hfrac_broad[dry_cells] = self._cc[dry_cells] + psi_L_broad[dry_cells]
            cell_prolongation = self._ghost_subdomain_proj.cell_prolongation(
                grids=self._ghost_low_dim_grids
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
        num_cells: List[int] = [g.num_cells for g in self._ghost_low_dim_grids]
        cum_sum = list(np.cumsum(num_cells))
        cum_sum.insert(0, 0)

        # The number of rows of the broadcasting matrix corresponds to the total number of
        # fracture cells. The number of columns corresponds to the total number of
        # fractures, which must also coincide with the size of the active ad variable h_frac.
        rows: int = sum(num_cells)
        cols: int = len(self._ghost_low_dim_grids)
        broad_matrix = sps.lil_matrix(np.zeros(shape=(rows, cols)))

        # Populate the matrix column-wise. The idea is to set to 1 the elements
        # corresponding to the number of ghost cells in each col, and leave the rest of the
        # elements untouched.
        # BUG: Broadcasting matrix is not constructed correctly when subdomains have
        #  different number of ghost cells
        for col in range(0, cols):
            base: int = cum_sum[col]
            roof: int = cum_sum[col + 1]
            broad_matrix[base:roof, col] = 1

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
