from typing import List, Literal, Tuple, Union

import numpy as np
import porepy as pp
import scipy.sparse as sps

from mdunsat.ad_utils import bulk_cc_var_to_mortar_grid

# Typing abbreviations
Scalar = Union[int, float]
AdArray = pp.ad.Ad_array
NonAd = Union[Scalar, np.ndarray]
Edge = Tuple[pp.Grid, pp.Grid]

__all__ = ["SWRC", "FractureVolume", "VanGenuchtenMualem", "capillary_threshold"]


def capillary_threshold(hydraulic_conductivity: np.ndarray) -> np.ndarray:
    """Pressure threshold as a function of the hydraulic conductivity as given by [1].

    Parameters:
        hydraulic_conductivity: Hydraulic conductivity of the soil in [m/s].

    Returns:
        Capillary pressure threshold in [m].

    References:
        [1] Tokunaga and Wan, 2001. doi: 10.1029/2001WR000245

    """
    DENSITY = 1014  # kg / m^3
    VISCOSITY = 1e-03  # Pa . s
    GRAVITY = 9.81  # m / s^2
    GAMMA = DENSITY * GRAVITY

    a = -(0.5**0.64) / GAMMA
    b = ((VISCOSITY * hydraulic_conductivity) / GAMMA) ** (-0.32)

    return a * b


class FractureVolume:
    """Constitutive relationship for fracture volume as a function of the hyd head,"""

    def __init__(
        self, gb: pp.GridBucket, fracture_grids: List[pp.Grid], param_key: str
    ):
        """
        Init method for the class.

        It is assumed that the parameter data dictionaries of the given fracture
        grids contain the keys: "aperture", "datum", "sin_alpha", "width",
        and "mean_threshold".

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket.
            fracture_grids (List of pp.Grid): List of fracture grids.
            param_key (str): Parameter keyword for accessing the data parameters.

        """
        self._gb: pp.GridBucket = gb
        self._grids: List[pp.Grid] = fracture_grids
        self._kw: str = param_key
        self._N: int = len(self._grids)  # number of fracture grids

        # Retrieve variables and concatenate
        frac_vol = []
        aperture = []
        datum = []
        sin_alpha = []
        width = []
        pressure_threshold = []
        for g in self._grids:
            d = self._gb.node_props(g)
            frac_vol.append(g.cell_volumes * d[pp.PARAMETERS][self._kw]["aperture"])
            aperture.append(d[pp.PARAMETERS][self._kw]["aperture"])
            datum.append(d[pp.PARAMETERS][self._kw]["datum"])
            sin_alpha.append(d[pp.PARAMETERS][self._kw]["sin_alpha"])
            width.append(d[pp.PARAMETERS][self._kw]["width"])
            pressure_threshold.append(d[pp.PARAMETERS][self._kw]["pressure_threshold"])
        self._fracvol: np.ndarray = np.array(frac_vol)
        self._aperture: np.ndarray = np.array(aperture)
        self._datum: np.ndarray = np.array(datum)
        self._sin_alpha: np.ndarray = np.array(sin_alpha)
        self._width: np.ndarray = np.array(width)
        self._pressure_threshold: np.ndarray = np.array(pressure_threshold)

    def __repr__(self) -> str:
        return "Water volume as a function of hydraulic head."

    # Public Methods
    def fracture_volume(
        self, as_ad: bool = False
    ) -> Union["_fracture_volume", pp.ad.Function]:
        """
        Fracture volume as a function of the hydraulic head.

        Parameters:
            as_ad (bool): If True the function is wrapped as a pp.ad.Function. If False,
            the function is passed a regular function.

        Returns:
            Callable or pp.ad.Function corresponding to the fracture volume

        """
        if as_ad:
            return pp.ad.Function(self._fracture_volume, name="fracture volume")
        else:
            return self._fracture_volume

    def volume_capacity(
        self, as_ad: bool = False
    ) -> Union["_volume_capacity", pp.ad.Function]:
        """
        Volume capacity as a function of the hydraulic head.

        Parameters:
            as_ad (bool): If True the function is wrapped as a pp.ad.Function. If False,
            the function is passed a regular function.

        Returns:
            regular classmethod or pp.ad.Function corresponding to the fracture volume

        """
        if as_ad:
            return pp.ad.Function(self._volume_capacity, name="volume capacity")
        else:
            return self._volume_capacity

    # Private methods
    def _fracture_volume(
        self, hydraulic_head: Union[AdArray, NonAd]
    ) -> Union[AdArray, NonAd]:
        """
        Compute fracture (or intersection) volume as a function of the hydraulic head

        Parameters:
            hydraulic_head (pp.ad.Ad_array or non-ad object): containing the value of the
            hydraulic head.

        Returns:
            (pp.ad.Ad_array or non-ad object): containing the value of the fracture volume.

        """
        # In its most basic form, the water volume V_f is related to the hydraulic head h_f
        # via the following piecewise-linear relationship:
        #
        #               { aper * (h_f - datum),    h_f <= fracture_volume/aper + datum
        #   V_f(h_f) =  {
        #               { fracture_volume,         otherwise
        #
        # Note that this relationship is only valid for water in hydrostatic equilibrium

        if isinstance(hydraulic_head, pp.ad.Ad_array):
            # We need to transform the aperture into a diagonal matrix to be able to
            # perform the multiplication and thus avoid broadcasting errors
            aper = sps.spdiags(self._aperture, 0, self._N, self._N)
            sin_alpha_inv = sps.spdiags(self._sin_alpha ** (-1), 0, self._N, self._N)
            width = sps.spdiags(self._width, 0, self._N, self._N)
            water_volume: pp.ad.Ad_array = (
                aper
                * width
                * sin_alpha_inv
                * (hydraulic_head - self._datum - self._pressure_threshold)
            )

            # Correct values of water volume accordingly
            for idx, _ in enumerate(self._grids):
                # If the water volume > the fracture, then set the
                # volume = fracture_volume
                if water_volume.val[idx] > self._fracvol[idx]:
                    water_volume.val[idx] = self._fracvol[idx]
        else:
            # Here, we don't need to do anything, numpy will take care of correctly
            # broadcasting everything for us
            water_volume: np.ndarray = (
                self._aperture
                * self._width
                * (
                    (hydraulic_head - self._datum - self._pressure_threshold)
                    / self._sin_alpha
                )
            )

            # Correct values of water volume accordingly
            for idx in range(hydraulic_head.size):
                # If the water volume > the fracture, then set the
                # volume = fracture_volume
                if water_volume[idx] > self._fracvol[idx]:
                    water_volume[idx] = self._fracvol[idx]
        return water_volume

    def _volume_capacity(self, hydraulic_head: Union[AdArray, NonAd]) -> NonAd:
        """Computes the derivative of the volume with respect to the hydraulic head

        Parameters:
            hydraulic_head: non-ad object (for the moment) containing the value of the
            hydraulic head.

        Returns:
           non-ad object: derivative of the volume w.r.t the hydraulic head
        """

        if isinstance(hydraulic_head, pp.ad.Ad_array):
            raise ValueError("Hydraulic head cannot be pp.ad.Ad_Array.")
        else:
            water_volume: np.ndarray = self._aperture * (hydraulic_head - self._datum)
            vol_capacity: np.ndarray = self._aperture
            # If the calculated water volume is greater than the fracture volume,
            # set the volume capacity to zero
            for idx, _ in enumerate(self._grids):
                if water_volume[idx] > self._fracvol[idx]:
                    vol_capacity[idx] = 0

        return vol_capacity


class SWRC:

    # TODO: REMOVE CLASS

    def __init__(self, param_key: str, gb: pp.GridBucket, grid_list: List[pp.Grid]):

        self.kw = param_key
        self.grid_list = grid_list
        self.gb = gb

        # Concatanate van Genuchten parameters
        val_alpha_vg_cc = []
        val_theta_s_cc = []
        val_theta_r_cc = []
        val_n_vg_cc = []
        val_m_vg_cc = []

        val_alpha_vg_fc = []
        val_theta_s_fc = []
        val_theta_r_fc = []
        val_n_vg_fc = []
        val_m_vg_fc = []

        for g in self.grid_list:
            d = self.gb.node_props(g)
            # Cell-centered values
            cc_ones = np.ones(g.num_cells)
            val_alpha_vg_cc.append(d[pp.PARAMETERS][self.kw]["alpha_vg"] * cc_ones)
            val_theta_s_cc.append(d[pp.PARAMETERS][self.kw]["theta_s"] * cc_ones)
            val_theta_r_cc.append(d[pp.PARAMETERS][self.kw]["theta_r"] * cc_ones)
            val_n_vg_cc.append(d[pp.PARAMETERS][self.kw]["n_vg"] * cc_ones)
            val_m_vg_cc.append(d[pp.PARAMETERS][self.kw]["m_vg"] * cc_ones)
            # Face-centered values
            fc_ones = np.ones(g.num_faces)
            val_alpha_vg_fc.append(d[pp.PARAMETERS][self.kw]["alpha_vg"] * fc_ones)
            val_theta_s_fc.append(d[pp.PARAMETERS][self.kw]["theta_s"] * fc_ones)
            val_theta_r_fc.append(d[pp.PARAMETERS][self.kw]["theta_r"] * fc_ones)
            val_n_vg_fc.append(d[pp.PARAMETERS][self.kw]["n_vg"] * fc_ones)
            val_m_vg_fc.append(d[pp.PARAMETERS][self.kw]["m_vg"] * fc_ones)

        self._alpha_vg_cc = np.hstack([v for v in val_alpha_vg_cc])
        self._theta_s_cc = np.hstack([v for v in val_theta_s_cc])
        self._theta_r_cc = np.hstack([v for v in val_theta_r_cc])
        self._n_vg_cc = np.hstack([v for v in val_n_vg_cc])
        self._m_vg_cc = np.hstack([v for v in val_m_vg_cc])

        self._alpha_vg_fc = np.hstack([v for v in val_alpha_vg_fc])
        self._theta_s_fc = np.hstack([v for v in val_theta_s_fc])
        self._theta_r_fc = np.hstack([v for v in val_theta_r_fc])
        self._n_vg_fc = np.hstack([v for v in val_n_vg_fc])
        self._m_vg_fc = np.hstack([v for v in val_m_vg_fc])

    # Public methods
    def water_content(
        self,
        as_ad: bool = True,
        cc: bool = True,
    ) -> Union["_water_content_cc", "_water_content_fc", pp.ad.Function]:
        """
        Water content as a function of the pressure head.

        Parameters:
            as_ad (bool): If True the function is wrapped as a pp.ad.Function. If False,
            the function is passed a regular function.

        Returns:
            regular classmethod or pp.ad.Function corresponding to the water content

        """
        if as_ad:
            if cc:
                return pp.ad.Function(self._water_content_cc, name="water content")
            else:
                return pp.ad.Function(self._water_content_fc, name="water content")
        else:
            if cc:
                return self._water_content_cc
            else:
                return self._water_content_fc

    def relative_permeability(
        self, as_ad: bool = True, cc: bool = True
    ) -> Union[
        "_relative_permeability_cc", "_relative_permeability_fc", pp.ad.Function
    ]:
        """
        Relative permeability as a function of the pressure head.

        Parameters:
            as_ad (bool): If True the function is wrapped as a pp.ad.Function. If False,
            the function is passed a regular function.

        Returns:
            regular classmethod or pp.ad.Function corresponding to the relative permeability

        """
        if as_ad:
            if cc:
                return pp.ad.Function(
                    self._relative_permeability_cc, name="relative permeability"
                )
            else:
                return pp.ad.Function(
                    self._relative_permeability_fc, name="relative permeability"
                )
        else:
            if cc:
                return self._relative_permeability_cc
            else:
                return self._relative_permeability_fc

    def moisture_capacity(
        self,
        as_ad: bool = True,
        cc: bool = True,
    ) -> Union["_moisture_capacity_cc", "_moisture_capacity_fc", pp.ad.Function]:
        """
        Specific moisture capacity as a function of the pressure head.

        Parameters:
            as_ad (bool): If True the function is wrapped as a pp.ad.Function. If False,
            the function is passed a regular function.

        Returns:
            regular classmethod or pp.ad.Function corresponding to the moisture capacity

        """
        if as_ad:
            if cc:
                return pp.ad.Function(
                    self._moisture_capacity_cc, name="specific moisture capacity"
                )
            else:
                return pp.ad.Function(
                    self._moisture_capacity_fc, name="specific moisture capacity"
                )
        else:
            if cc:
                return self._moisture_capacity_cc
            else:
                return self._moisture_capacity_fc

    # Private methods
    def _water_content_cc(self, psi):
        if isinstance(psi, pp.ad.Ad_array):
            is_unsat = self._is_unsat(psi.val)
            is_sat = 1 - is_unsat
            numer = self._theta_s_cc - self._theta_r_cc
            denom = (
                1 + (pp.ad.abs(psi) * self._alpha_vg_cc) ** self._n_vg_cc
            ) ** self._m_vg_cc
            theta = (
                denom ** (-1) * numer + self._theta_r_cc
            ) * is_unsat + self._theta_s_cc * is_sat
        else:  # typically int, float, or np.ndarray
            is_unsat = self._is_unsat(psi)
            is_sat = 1 - is_unsat
            numer = self._theta_s_cc - self._theta_r_cc
            denom = (
                1 + (self._alpha_vg_cc * np.abs(psi)) ** self._n_vg_cc
            ) ** self._m_vg_cc
            theta = (
                numer / denom + self._theta_r_cc
            ) * is_unsat + self._theta_s_cc * is_sat

        return theta

    def _water_content_fc(self, psi):
        if isinstance(psi, pp.ad.Ad_array):
            is_unsat = self._is_unsat(psi.val)
            is_sat = 1 - is_unsat
            numer = self._theta_s_fc - self._theta_r_fc
            denom = (
                1 + (pp.ad.abs(psi) * self._alpha_vg_fc) ** self._n_vg_fc
            ) ** self._m_vg_fc
            theta = (
                denom ** (-1) * numer + self._theta_r_fc
            ) * is_unsat + self._theta_s_fc * is_sat
        else:  # typically int, float, or np.ndarray
            is_unsat = self._is_unsat(psi)
            is_sat = 1 - is_unsat
            numer = self._theta_s_fc - self._theta_r_fc
            denom = (
                1 + (self._alpha_vg_fc * np.abs(psi)) ** self._n_vg_fc
            ) ** self._m_vg_fc
            theta = (
                numer / denom + self._theta_r_fc
            ) * is_unsat + self._theta_s_fc * is_sat

        return theta

    def _effective_saturation_cc(
        self, psi: Union[AdArray, NonAd]
    ) -> Union[AdArray, NonAd]:
        """Effective saturation as a function of the pressure head.

        Parameters:
            psi (Ad-array or non-ad object): pressure head
        Returns:
            s_eff (Ad-array or non-ad object): effective (normalized) saturation
        """

        numer = self._water_content_cc(psi) + self._theta_r_cc * (-1)
        denom = self._theta_s_cc + self._theta_r_cc * (-1)
        s_eff = denom ** (-1) * numer

        return s_eff

    def _effective_saturation_fc(
        self, psi: Union[AdArray, NonAd]
    ) -> Union[AdArray, NonAd]:
        """Effective saturation as a function of the pressure head.

        Parameters:
            psi (Ad-array or non-ad object): pressure head
        Returns:
            s_eff (Ad-array or non-ad object): effective (normalized) saturation
        """

        numer = self._water_content_fc(psi) + self._theta_r_fc * (-1)
        denom = self._theta_s_fc + self._theta_r_fc * (-1)
        s_eff = denom ** (-1) * numer

        return s_eff

    def _relative_permeability_cc(self, psi: NonAd) -> NonAd:
        """Relative permeability as a function of the pressure head.

        Parameters:
            psi (non-ad object): pressure head
        Returns:
            krw (non-ad object): water relative permeability
        """

        # TODO: Add possibility to pass an pp.ad.Array
        if isinstance(psi, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:  # typically int, float, or np.ndarray
            s_eff = self._effective_saturation_cc(psi)
            krw = (
                s_eff**0.5
                * (1 - (1 - s_eff ** (1 / self._m_vg_cc)) ** self._m_vg_cc) ** 2
            )

        return krw

    def _relative_permeability_fc(self, psi: NonAd) -> NonAd:
        """Relative permeability as a function of the pressure head.

        Parameters:
            psi (non-ad object): pressure head
        Returns:
            krw (non-ad object): water relative permeability
        """

        # TODO: Add possibility to pass an pp.ad.Array
        if isinstance(psi, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:  # typically int, float, or np.ndarray
            s_eff = self._effective_saturation_fc(psi)
            krw = (
                s_eff**0.5
                * (1 - (1 - s_eff ** (1 / self._m_vg_fc)) ** self._m_vg_fc) ** 2
            )

        return krw

    def _moisture_capacity_cc(self, psi: NonAd) -> NonAd:
        """Specific moisture capacity as a function of the pressure head

        Parameters:
            psi (non-ad object): pressure head
        Returns:
            moist_capacity (non-ad object): moisture capacitiy, i.e., d(theta)/d(psi).
        """

        # TODO: Add possibility to pass an pp.ad.Array
        if isinstance(psi, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:  # typically int, float, or np.ndarray
            is_unsat = self._is_unsat(psi)
            is_sat = 1 - is_unsat
            num = (
                -self._m_vg_cc
                * self._n_vg_cc
                * (self._theta_s_cc - self._theta_r_cc)
                * (self._alpha_vg_cc * np.abs(psi)) ** self._n_vg_cc
            )
            den = psi * ((self._alpha_vg_cc * np.abs(psi)) ** self._n_vg_cc + 1) ** (
                self._m_vg_cc + 1
            )
            # Here, we have to be particularly careful with division by zero. If zero is
            # encountered in the denominator, we force the moisture capacity to be zero.
            moist_capacity = (
                np.divide(num, den, out=np.zeros_like(num), where=den != 0)
                * self._is_unsat(psi)
                + 0 * is_sat
            )

        return moist_capacity

    def _moisture_capacity_fc(self, psi: NonAd) -> NonAd:
        """Specific moisture capacity as a function of the pressure head

        Parameters:
            psi (non-ad object): pressure head
        Returns:
            moist_capacity (non-ad object): moisture capacitiy, i.e., d(theta)/d(psi).
        """

        # TODO: Add possibility to pass an pp.ad.Array
        if isinstance(psi, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:  # typically int, float, or np.ndarray
            is_unsat = self._is_unsat(psi)
            is_sat = 1 - is_unsat
            num = (
                -self._m_vg_fc
                * self._n_vg_fc
                * (self._theta_s_fc - self._theta_r_fc)
                * (self._alpha_vg_fc * np.abs(psi)) ** self._n_vg_fc
            )
            den = psi * ((self._alpha_vg_fc * np.abs(psi)) ** self._n_vg_fc + 1) ** (
                self._m_vg_fc + 1
            )
            # Here, we have to be particularly careful with division by zero. If zero is
            # encountered in the denominator, we force the moisture capacity to be zero.
            moist_capacity = (
                np.divide(num, den, out=np.zeros_like(num), where=den != 0)
                * self._is_unsat(psi)
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
            return 1 - pp.ad.heaviside(pressure_head, 1)


class VanGenuchtenMualem:
    """Parent class for van Genuchten Mualem model."""

    def __init__(
        self,
        gb: pp.GridBucket,
        param_key: str,
        dof_manager: pp.DofManager,
    ):
        """Init method for the class.

        Parameters:
            gb (GridBucket): Mixed-dimensional grid bucket. Note that we assume that the
                relevant parameters, i.e., "alpha_vG", "theta_r", "theta_s", "n_vG", "m_vG",
                "hydrostatic_volume" exist in the grid data dictionaries.
            param_key (str): Keyword to access the data parameter.

        """

        self._gb: pp.GridBucket = gb
        """Grid bucket."""

        self._dof_manager: pp.DofManager = dof_manager
        """Degree of freedom manager."""

        self._kw: str = param_key
        """Keyword for accessing the parameters."""

        self._num_faces = self._gb.grids_of_dimension(self._gb.dim_max())[0].num_faces
        """Get number of faces in the bulk."""

        self._num_cells = self._gb.grids_of_dimension(self._gb.dim_max())[0].num_cells
        """Get number of cells in the bulk."""

        self._num_mortar_cells = self._gb.num_mortar_cells()

        # Note that assume that the SWRC parameters are stored in the dictionary of
        # the ambient grid. In the future, we could assign different SWRC to each grid.
        # However, for this project, this works just fine.
        g_bulk: pp.Grid = self._gb.grids_of_dimension(self._gb.dim_max())[0]
        d_bulk: dict = self._gb.node_props(g_bulk)
        param_dict: dict = d_bulk[pp.PARAMETERS][self._kw]
        self.alpha_vG: np.ndarray = param_dict["alpha_vG"]  # alpha parameter
        self.theta_r: np.ndarray = param_dict["theta_r"]  # residual water content
        self.theta_s: np.ndarray = param_dict[
            "theta_s"
        ]  # water content at sat contditions
        self.n_vG: np.ndarray = param_dict["n_vG"]  # n parameter
        self.m_vG: np.ndarray = param_dict["m_vG"]  # m parameter

        # We need to project the cell-centered values onto the boundaries of the bulk,
        # since these are needed to evaluate the quantities of interest that depend
        # on the boundary pressure heads. The projection matrix is obtained using a
        # mock MPFA discretization with zero-Neumann bc.
        self.alpha_vG_bound: np.ndarray
        self.theta_r_bound: np.ndarray
        self.theta_s_bound: np.ndarray
        self.n_vG_bound: np.ndarray
        self.m_vG_bound: np.ndarray
        self._project_vals_to_boundary()

        # Now, we need to project the face-centered values onto the mortar grids.
        # This is used for the computation of the relative permeabilities on the
        # interfaces. We assume that the gridding of the internal boundaries of the
        # bulk match with the gridding of the interfaces.
        self.alpha_vG_mortar = bulk_cc_var_to_mortar_grid(self._gb, self.alpha_vG)
        self.theta_s_mortar = bulk_cc_var_to_mortar_grid(self._gb, self.theta_s)
        self.theta_r_mortar = bulk_cc_var_to_mortar_grid(self._gb, self.theta_r)
        self.n_vG_mortar = bulk_cc_var_to_mortar_grid(self._gb, self.n_vG)
        self.m_vG_mortar = bulk_cc_var_to_mortar_grid(self._gb, self.m_vG)

    def __repr__(self):
        # TODO: Add a proper __repr__ showing the attributes
        return "Soil-water retention curve (van Genuchten-Mualem) object."

    # Helper methods for __init__
    def _project_vals_to_boundary(self):
        """Get values of vanGenuchten properties at the boundary of the domain."""
        g_bulk = self._gb.grids_of_dimension(self._gb.dim_max())[0]
        mock_mpfa, mock_data = pp.Mpfa(self._kw), {}
        pp.initialize_default_data(g_bulk, mock_data, self._kw)
        mock_mpfa.discretize(g_bulk, mock_data)
        proj = mock_data[pp.DISCRETIZATION_MATRICES][self._kw]["bound_pressure_cell"]
        self.alpha_vG_bound = proj * self.alpha_vG
        self.theta_r_bound = proj * self.theta_r
        self.theta_s_bound = proj * self.theta_s
        self.n_vG_bound = proj * self.n_vG
        self.m_vG_bound = proj * self.m_vG

    # Public methods
    def water_content(
        self,
        as_ad: bool = False,
    ) -> Union["_water_content", pp.ad.Function]:
        """Water content as a function of the pressure head.

        Parameters:
            as_ad: If True the function is wrapped as a pp.ad.Function. If False,
                the function is passed as a regular function.

        Returns:
            Callable or pp.ad.Function corresponding to the water content.

        """
        if as_ad:
            return pp.ad.Function(self._water_content, name="Water content")
        else:
            return self._water_content

    def relative_permeability(
        self,
        as_ad: bool = False,
    ) -> Union["_relative_permeability", pp.ad.Function]:
        """Relative permeability as a function of the pressure head.

        Parameters:
            as_ad: If True the function is wrapped as a pp.ad.Function. If False,
                the function is passed as a regular function.

        Returns:
            Callable or pp.ad.Function corresponding to the relative permeability

        """
        if as_ad:
            return pp.ad.Function(
                self._relative_permeability,
                name="Relative permeability",
            )
        else:
            return (self._relative_permeability,)

    def moisture_capacity(
        self,
        as_ad: bool = False,
    ) -> Union["_moisture_capacity", pp.ad.Function]:
        """Specific moisture capacity as a function of the pressure head.

        Parameters:
            as_ad: If True the function is wrapped as a pp.ad.Function. If False,
                the function is passed as a regular function.

        Returns:
            Callable or pp.ad.Function corresponding to the moisture capacity.

        """
        if as_ad:
            return pp.ad.Function(
                self._moisture_capacity,
                name="Specific moisture capacity",
            )
        else:
            return (self._moisture_capacity,)

    # Private methods
    def _water_content(
        self,
        pressure_head: Union[AdArray, NonAd],
    ) -> Union[AdArray, NonAd]:
        """Water content as a function of the pressure head.

        Parameters:
            pressure_head (Ad-array or non-ad object): Pressure head.

        Returns:
            theta (Ad-array or non-ad object): Water content.

        """

        theta_s, theta_r, alpha_vG, n_vG, m_vG = self._get_vG_properties(pressure_head)

        if isinstance(pressure_head, pp.ad.Ad_array):

            is_unsat = self._is_unsat(pressure_head.val)
            is_sat = 1 - is_unsat
            num = theta_s - theta_r
            den = (1 + (alpha_vG * pp.ad.abs(pressure_head)) ** n_vG) ** m_vG
            theta = (num * den ** (-1) + theta_r) * is_unsat + theta_s * is_sat

        else:  # typically int, float, or np.ndarray

            is_unsat = self._is_unsat(pressure_head)
            is_sat = 1 - is_unsat
            num = theta_s - theta_r
            den = (1 + (alpha_vG * np.abs(pressure_head)) ** n_vG) ** m_vG
            theta = (num / den + theta_r) * is_unsat + theta_s * is_sat

        return theta

    def _effective_saturation(
        self, pressure_head: Union[AdArray, NonAd]
    ) -> Union[AdArray, NonAd]:
        """Effective saturation as a function of the pressure head.

        Parameters:
            pressure_head (Ad-array or non-ad object): Pressure head.

        Returns:
            s_eff (Ad-array or non-ad object): Effective (normalized) saturation.

        """
        theta_s, theta_r, *_ = self._get_vG_properties(pressure_head)
        num = self._water_content(pressure_head) - theta_r
        den = theta_s - theta_r
        s_eff = num * den ** (-1)

        return s_eff

    def _relative_permeability(self, pressure_head: NonAd) -> NonAd:
        """Relative permeability as a function of the pressure head.

        Parameters:
            pressure_head (non-ad object): pressure head.

        Returns:
            krw (non-ad object): water relative permeability

        """
        _, _, _, _, m_vG = self._get_vG_properties(pressure_head)

        if isinstance(pressure_head, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:  # typically int, float, or np.ndarray
            s_eff = self._effective_saturation(pressure_head)
            krw = s_eff**0.5 * (1 - (1 - s_eff ** (1 / m_vG)) ** m_vG) ** 2

        return krw

    def _moisture_capacity(self, pressure_head: NonAd) -> NonAd:
        """Specific moisture capacity as a function of the pressure head.

        Parameters:
            pressure_head (non-ad object): Pressure head.

        Returns:
            moist_capacity (non-ad object): Moisture capacity, i.e., d(theta)/d(psi).

        """
        theta_s, theta_r, alpha_vG, n_vG, m_vG = self._get_vG_properties(pressure_head)

        if isinstance(pressure_head, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:  # typically int, float, or np.ndarray
            is_unsat = self._is_unsat(pressure_head)
            is_sat = 1 - is_unsat
            num = (
                -m_vG
                * n_vG
                * (theta_s - theta_r)
                * (alpha_vG * np.abs(pressure_head)) ** n_vG
            )
            den = pressure_head * ((alpha_vG * np.abs(pressure_head)) ** n_vG + 1) ** (
                m_vG + 1
            )
            # Here, we have to be particularly careful with zero division. If zero is
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
        """Determines whether is saturated based on the value of the pressure head.

        Parameters
            pressure_head (non-ad): containing the values of the pressure heads.

        Returns
            non-ad: 1 if pressure_head < 0, and 0 otherwise.

        """
        if isinstance(pressure_head, pp.ad.Ad_array):
            raise TypeError("Pressure head cannot be AD. Expected non-ad object.")
        else:
            return 1 - pp.ad.heaviside(pressure_head, 1)

    def _get_vG_properties(
        self,
        pressure_head: Union[AdArray, NonAd],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve vanGenuchten parameters based on whether they are defined in the
        cell centers or on the boundaries of the domain.

        Todo:
         This is an extremely poor way to differentiate between subdomain, boundary,
         and interface quantities. We should not use the number of elements in each
         grid, but something less prone to bugs.

        Parameters:
            pressure_head: Pressure head in the bulk.

        Returns:
            Tuple of five numpy arrays, containing theta_s, theta_r, alpha_vG, n_vG,
            and m_vG.

        """
        if isinstance(pressure_head, AdArray):
            array_length = pressure_head.val.size
        else:
            array_length = pressure_head.size

        if array_length == self._num_cells:
            assert array_length == self._num_cells
            theta_s = self.theta_s
            theta_r = self.theta_r
            alpha_vG = self.alpha_vG
            n_vG = self.n_vG
            m_vG = self.m_vG
        elif array_length == self._num_faces:
            assert array_length == self._num_faces
            theta_s = self.theta_s_bound
            theta_r = self.theta_r_bound
            alpha_vG = self.alpha_vG_bound
            n_vG = self.n_vG_bound
            m_vG = self.m_vG_bound
        elif array_length == self._num_mortar_cells:
            assert array_length == self._num_mortar_cells
            theta_s = self.theta_s_mortar
            theta_r = self.theta_r_mortar
            alpha_vG = self.alpha_vG_mortar
            n_vG = self.n_vG_mortar
            m_vG = self.m_vG_mortar
        else:
            raise ValueError(
                "Hydraulic properties only defined at the interior of "
                "the bulk, the boundary of the bulk, and on the "
                "interfaces."
            )

        return theta_s, theta_r, alpha_vG, n_vG, m_vG
