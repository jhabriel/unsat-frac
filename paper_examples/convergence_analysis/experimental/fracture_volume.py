import porepy as pp
import numpy as np
import scipy.sparse as sps
from typing import List, Union, Tuple

Scalar = Union[int, float]
AdArray = pp.ad.Ad_array
NonAd = Union[Scalar, np.ndarray]
Edge = Tuple[pp.Grid, pp.Grid]


class FractureVolume:
    """Constitutive relationship for fracture volume as a function of the hydraulic head"""

    def __init__(
        self, gb: pp.GridBucket, fracture_grids: List[pp.Grid], param_key: str
    ):
        """
        Init method for the class.

        It is assumed that the parameter data dictionaries of the given fracture grids
        contain the keys: "aperture" and "datum".

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket.
            fracture_grids (List of pp.Grid): List of fracture grids.
            param_key (str): Parameter keyword for accessing the data parameters.
        """
        self._gb: pp.GridBucket = gb
        self._grids: List[pp.Grid] = fracture_grids
        self._kw: str = param_key
        self._N: int = len(self._grids)  # number of fracture grids

        # Get fracture volume, aperture, and datum
        # TODO: There should be a better way to do this. But for the meantime, it works.
        frac_vol = []
        aperture = []
        datum = []
        sin_alpha = []
        width = []
        for g in self._grids:
            d = self._gb.node_props(g)
            frac_vol.append(g.cell_volumes * d[pp.PARAMETERS][self._kw]["aperture"])
            aperture.append(d[pp.PARAMETERS][self._kw]["aperture"])
            datum.append(d[pp.PARAMETERS][self._kw]["datum"])
            sin_alpha.append(d[pp.PARAMETERS][self._kw]["sin_alpha"])
            width.append(d[pp.PARAMETERS][self._kw]["width"])
        self._fracvol: np.ndarray = np.array(frac_vol)
        self._aperture: np.ndarray = np.array(aperture)
        self._datum: np.ndarray = np.array(datum)
        self._sin_alpha: np.ndarray = np.array(sin_alpha)
        self._width: np.ndarray = np.array(width)
        self._pressure_threshold = -5

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
            regular classmethod or pp.ad.Function corresponding to the fracture volume

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
            # We need to transform the aperture into a diagonal matrix to be able to perform
            # the multiplication and thus avoid broadcasting errors
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
            # for idx, _ in enumerate(self._grids):
            # If the water volume > the fracture or fracture intersection volume, then
            # set the volume = fracture_volume
            # if water_volume.val[idx] > self._fracvol[idx]:
            # pass
            # water_volume.val[idx] = self._fracvol[idx]
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
            # for idx in range(hydraulic_head.size):
            # If the water volume > the fracture or fracture intersection volume, then
            # set the volume = fracture_volume
            # if water_volume[idx] > self._fracvol[idx]:
            # pass
            # water_volume[idx] = self._fracvol[idx]
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
