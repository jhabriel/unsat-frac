import porepy as pp
import numpy as np
from typing import Callable, Optional, Tuple, List, Any, Union, NewType

# Typing abbreviations
Scalar = Union[int, float]
AdArray = pp.ad.Ad_array
NonAd = Union[Scalar, np.ndarray]
Edge = Tuple[pp.Grid, pp.Grid]


class FractureVolume:

    # TODO: This has to be refactored such that it returns one expression per fracture grid.
    # possibly, it will take the ghost_fracture_grids and return the function

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

    def fracture_volume(self, hydraulic_head: Union[AdArray, NonAd]) -> Union[AdArray, NonAd]:

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


class VanGenuchtenMualem:
    """Parent class for van Genuchten Mualem model."""

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

    # Public methods
    def water_content(self, as_ad: bool = False) -> Union['_water_content', pp.ad.Function]:
        """
        Water content as a function of the pressure head.

        Parameters:
            as_ad (bool): If True the function is wrapped as a pp.ad.Function. If False,
            the function is passed a regular function.

        Returns:
            regular classmethod or pp.ad.Function corresponding to the water content

        """
        if as_ad:
            return pp.ad.Function(self._water_content, name="water content")
        else:
            return self._water_content

    def relative_permeability(self, as_ad: bool = False) -> Union['_relative_permeability',
                                                                  pp.ad.Function]:
        """
        Relative permeability as a function of the pressure head.

        Parameters:
            as_ad (bool): If True the function is wrapped as a pp.ad.Function. If False,
            the function is passed a regular function.

        Returns:
            regular classmethod or pp.ad.Function corresponding to the relative permeability

        """
        if as_ad:
            return pp.ad.Function(self._relative_permeability, name="relative permeability")
        else:
            return self._relative_permeability

    def moisture_capacity(self, as_ad: bool = False) -> Union['_moisture_capacity',
                                                              pp.ad.Function]:
        """
        Specific moisture capacity as a function of the pressure head.

        Parameters:
            as_ad (bool): If True the function is wrapped as a pp.ad.Function. If False,
            the function is passed a regular function.

        Returns:
            regular classmethod or pp.ad.Function corresponding to the moisture capacity

        """
        if as_ad:
            return pp.ad.Function(self._moisture_capacity, name="specific moisture capacity")
        else:
            return self._moisture_capacity

    # Private methods
    def _water_content(
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

    def _effective_saturation(
            self, pressure_head: Union[AdArray, NonAd]
    ) -> Union[AdArray, NonAd]:
        """Effective saturation as a function of the pressure head.

        Parameters:
            pressure_head (Ad-array or non-ad object): pressure head
        Returns:
            s_eff (Ad-array or non-ad object): effective (normalized) saturation
        """

        num = self._water_content(pressure_head) - self.theta_r
        den = self.theta_s - self.theta_r
        s_eff = num * den ** (-1)

        return s_eff

    def _relative_permeability(self, pressure_head: NonAd) -> NonAd:
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
            s_eff = self._effective_saturation(pressure_head)
            krw = s_eff ** 0.5 * (1 - (1 - s_eff ** (1 / self.m_vG)) ** self.m_vG) ** 2

        return krw

    def _moisture_capacity(self, pressure_head: NonAd) -> NonAd:
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
            return 1 - pp.ad.heaviside(pressure_head, 1)
