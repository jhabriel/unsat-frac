""" 
Collection of utility AD operators and functions 
Author: @jv
"""

#%% Importing modules
import porepy as pp
import numpy as np
import scipy.sparse as sps

from porepy.numerics.ad.operators import ApplicableOperator
from porepy.numerics.ad.functions import heaviside
from porepy.numerics.ad.forward_mode import Ad_array
from typing import Callable

#%%# FACE AVERAGING SCHEMES 

# Arithmetic average of the bulk
class ArithmeticAverageAd(ApplicableOperator):
    """
    Computes the face arithmetic average of a cell-based AD array. 
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
        inner_values : AD_array
            Previous iterate of AD_array.
        dir_bound_values : AD operator
            Containing the Dirchlet boundary values.

        Raises
        ------
        ValueError
            If the input is not an AD_array. For example, a numpy array
            is not permitted.

        Returns
        -------
        Numpy Array of size g.num_faces
            Arithmetic averaged values of a given ad_array.

        """

        bc = self._d[pp.PARAMETERS][self._param_key]["bc"]
        neu_fcs = bc.is_neu.nonzero()  # Neumann boundary faces
        dir_fcs = bc.is_dir.nonzero()  # dirichlet boundary faces
        int_fcs = self._g.get_internal_faces()  # internal faces

        # Faces neighboring mapping
        fcs_neigh = np.zeros((self._g.num_faces, 2), dtype=int)
        fcs_neigh[:, 0] = self._g.cell_face_as_dense()[0]
        fcs_neigh[:, 1] = self._g.cell_face_as_dense()[1]

        # Internal faces neighboring mapping
        int_fcs_neigh = fcs_neigh[int_fcs]

        # Initialize array
        face_avg = np.zeros(self._g.num_faces)

        # Values at Neumman boundaries (Not really used)
        face_avg[neu_fcs] = np.nan

        if isinstance(inner_values, Ad_array):
            raise ValueError("Cell-center array cannot be of the type Ad_array")
        else:
            # Values at Dirichlet boundaries
            dir_cells_neigh = fcs_neigh[dir_fcs]
            dir_cells = dir_cells_neigh[(dir_cells_neigh >= 0).nonzero()][0]
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
    # Credits: Jakub Both
    
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
        is_dir = d[pp.PARAMETERS][param_key]["bc"].is_dir.copy()
        self._cf_is_dir = [
            np.logical_and(cf_is_boundary[i], is_dir) for i in range(0, 2)
        ]

    def __repr__(self) -> str:
        return " Flux-based Upwind AD face operator"

    def apply(self, mobility_inner, mobility_bound, face_flux):
        """Compute transmissibility via upwinding over faces. Use monotonicityexpr for
        deciding directionality.

        Idea: 'face value' = 'left cell value' * Heaviside('flux from left')
                           + 'right cell value' * Heaviside('flux from right').
        """

        # TODO only implemented for scalar relative permeabilities so far
        # TODO so far not for periodic bondary conditions.

        # Rename internal properties
        hs = self._heaviside
        cf_inner = self._cf_inner
        cf_is_boundary = self._cf_is_boundary

        # Determine direction-determining cell values to the left(0) and right(1) of each face.
        # Use Dirichlet boundary data where suitable.
        # Neglect Neumann boundaries since face transmissibilities at Neumann boundary data
        # anyhow does not play a role.
        # assert (face_flux, np.ndarray)  # TODO extend to Ad_arrays

        # Do the same for the mobility as for the direction-determining arrays.
        if isinstance(mobility_inner, Ad_array):
            raise ValueError("Cell-center array cannot be of the type Ad_array")
        else:
            mob_f = [cf_inner[i] * mobility_inner for i in range(0, 2)]
            for i in range(0, 2):
                mob_f[i][cf_is_boundary[i]] = mobility_bound[cf_is_boundary[i]]

        # Evaluate the Heaviside function of the "flux directions".
        hs_f_01 = hs(face_flux.val)
        hs_f_10 = hs(-face_flux.val)

        # Determine the face mobility by utilizing the general idea (see above).
        face_mobility = mob_f[0] * hs_f_01 + mob_f[1] * hs_f_10

        return sps.spdiags(face_mobility, 0, self._g.num_faces, self._g.num_faces)


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
        print("Soil Water Retention Curve: van Genuchtem-Mualem model")
        
    def _is_unsat(self, pressure_head):
        """ Determine whether the cell is saturated or not """
        
        if isinstance(pressure_head, pp.ad.Ad_array):
            raise ValueError("Only numerical value of the array should be passed")
        else:
            # {1, pressure_head < 0
            # {0, otherwise
            return 1 - heaviside(pressure_head, 1)  
        
    
    def water_content(self, pressure_head):
        """ Water content as a function of the pressure head"""
        
        if isinstance(pressure_head, pp.ad.Ad_array):
            num = self.theta_s - self.theta_r
            den = (1 + (self.alpha_vG * pp.ad.abs(pressure_head)) ** self.n_vG) ** self.m_vG
            theta = (
                (num * (den ** -1) + self.theta_r) * self._is_unsat(pressure_head.val)
                + self.theta_s  * (1 - self._is_unsat(pressure_head.val)) 
                )
        else:
            num = self.theta_s - self.theta_r
            den = (1 + (self.alpha_vG * np.abs(pressure_head)) ** self.n_vG) ** self.m_vG
            theta = (
                (num * (den ** -1) + self.theta_r) * self._is_unsat(pressure_head)
                + self.theta_s  * (1 - self._is_unsat(pressure_head)) 
                )
        
        return theta
    
    
    def relative_permeability(self, pressure_head):
        """ Relative permeability as a function of the pressure head"""

        if isinstance(pressure_head, pp.ad.Ad_array):
            num = (1 - (self.alpha_vG * pp.ad.abs(pressure_head)) ** (self.n_vG-1)
                   * (1 + (self.alpha_vG * pp.ad.abs(pressure_head)) ** self.n_vG) ** (-self.m_vG)
                   ) ** 2
            den = (1 + (self.alpha_vG * pp.ad.abs(pressure_head) ** self.n_vG) ) ** (0.5 * self.m_vG)
            krw = ((num * (den ** -1)) * self._is_unsat(pressure_head.val)
                + 1 * (1 - self._is_unsat(pressure_head.val)))
        else:
            num = (1 - (self.alpha_vG * np.abs(pressure_head)) ** (self.n_vG-1)
                   * (1 + (self.alpha_vG * np.abs(pressure_head)) ** self.n_vG) ** (-self.m_vG)
                   ) ** 2
            den = (1 + (self.alpha_vG * np.abs(pressure_head) ** self.n_vG) ) ** (0.5 * self.m_vG)
            krw = ((num * (den ** -1)) * self._is_unsat(pressure_head)
                + 1 * (1 - self._is_unsat(pressure_head)))
        
        return krw
        
    
    def moisture_capacity(self, pressure_head):
        """ Specific moisture capacity as a function of the pressure head"""
        
        # NOTE: We have to take care of potential division by zero. For this, 
        # we use np.divide. I'm not sure what implication this has when an
        # Ad_array is passed as argument
        if isinstance(pressure_head, pp.ad.Ad_array):
            num = (- self.m_vG * self.n_vG * 
                   (self.theta_s - self.theta_r) * 
                   (self.alpha_vG * np.abs(pressure_head.val)) ** self.n_vG)
            den = ( pressure_head.val * 
                   ((self.alpha_vG * np.abs(pressure_head.val)) ** self.n_vG + 1) 
                   ** (self.m_vG + 1))
            C = (np.divide(num, den, out=np.zeros_like(num), where=den!=0) * self._is_unsat(pressure_head.val)
                + 0 * (1 - self._is_unsat(pressure_head.val)))
        else:
            num = (- self.m_vG * self.n_vG * 
                   (self.theta_s - self.theta_r) * 
                   (self.alpha_vG * np.abs(pressure_head)) ** self.n_vG)
            den = ( pressure_head * 
                   ((self.alpha_vG * np.abs(pressure_head)) ** self.n_vG + 1) 
                   ** (self.m_vG + 1))
            C = (np.divide(num, den, out=np.zeros_like(num), where=den!=0) * self._is_unsat(pressure_head)
                + 0 * (1 - self._is_unsat(pressure_head)))
                  
        return C