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


#%% INTERFACE UPSTREAM WEIGHTING
class InterfaceUpwindAd(ApplicableOperator):
    """
    Computes the interface relative permeabilities based on the (projected) 
    pressure jump associated with the bulk and fractur potentials.
    """

    def __init__(self):
        
        self._set_tree()
        
    def __repr__(self) -> str:
        return "Interface upwind AD operator"
    
    #TODO: Add sanity check to check if input matches amount of mortar cells in gb
    #TODO: Write tests
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
        if (isinstance(trace_p_bulk, pp.ad.Ad_array) or
            isinstance(krw_trace_p_bulk, pp.ad.Ad_array) or
            isinstance(p_frac, pp.ad.Ad_array) or
            isinstance(krw_p_frac, pp.ad.Ad_array)):
            raise TypeError("Input cannot be of type Ad array")
        else:       
            pressure_jump = trace_p_bulk - p_frac
            hs_10 = heaviside(pressure_jump, zerovalue=0)
            hs_01 = heaviside(-pressure_jump, zerovalue=0)       
            vals = hs_10 * krw_trace_p_bulk + hs_01 * krw_p_frac
            n = len(trace_p_bulk)
            interface_krw = sps.spdiags(vals, 0, n, n) 
            
        return interface_krw

#%% BULK FACE AVERAGING SCHEMES 

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

        if isinstance(inner_values, Ad_array):
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
            face_avg = np.ones(self._g.num_faces) # Neumann krw=1.0
                          
            # Values at Dirichlet boundaries
            dir_cells_neigh = fcs_neigh[dir_fcs]
            if dir_cells_neigh.size > 0:
                dir_cells = dir_cells_neigh[(dir_cells_neigh >= 0).nonzero()]
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
        self._cf_is_dir = [np.logical_and(cf_is_boundary[i], self._is_dir) for i in range(0, 2)]
        self._is_neu = d[pp.PARAMETERS][param_key]["bc"].is_neu.copy()
        self._cf_is_neu = [np.logical_and(cf_is_boundary[i], self._is_neu) for i in range(0, 2)]


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
        if isinstance(inner_values, Ad_array) or isinstance(face_flux, Ad_array):
            raise TypeError("Object cannot be of the type Ad_array")
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
        return "Soil Water Retention Curve: van Genuchtem-Mualem model"
        
    def is_unsat(self, p):
        """ Determine whether the cell is saturated or not """
        
        if isinstance(p, pp.ad.Ad_array):
            raise TypeError("Pressure cannot be AD. Expected inactive variable.")
        else:
            # {1, pressure_head < 0
            # {0, otherwise
            return 1 - heaviside(p, 1)  
        
    
    def water_content(self, p):
        """ Water content as a function of the pressure head"""
                
        if isinstance(p, pp.ad.Ad_array):
            is_unsat = self.is_unsat(p.val)
            is_sat = 1 - is_unsat        
            num = self.theta_s - self.theta_r
            den = (1 + (self.alpha_vG * pp.ad.abs(p)) ** self.n_vG) ** self.m_vG
            theta = ((num * den ** (-1) + self.theta_r) * is_unsat
                     + self.theta_s * is_sat)
        else:
            is_unsat = self.is_unsat(p)
            is_sat = 1 - is_unsat        
            num = self.theta_s - self.theta_r
            den = (1 + (self.alpha_vG * np.abs(p)) ** self.n_vG) ** self.m_vG
            theta = (num / den + self.theta_r) * is_unsat + self.theta_s * is_sat
        
        return theta
    
    def effective_saturation(self, p):
        """ Effective saturation as a function of the water content """
        
        num = self.water_content(p) - self.theta_r
        den = self.theta_s - self.theta_r 
        s_eff = num * den ** (-1)
        
        return s_eff
    
    
    def relative_permeability(self, p):
        """ Relative permeability as a function of the effective saturation"""

        if isinstance(p, pp.ad.Ad_array):
            raise TypeError("Pressure cannot be AD. Expected previous_iteration()")
        else:
            krw = (self.effective_saturation(p) ** (0.5) * 
                   (1 - (1 - self.effective_saturation(p) ** (1/self.m_vG)) 
                    ** self.m_vG) ** 2 )

        return krw

    
    def moisture_capacity(self, p):
        """ Specific moisture capacity as a function of the pressure head"""
           
        if isinstance(p, pp.ad.Ad_array):
            raise TypeError("Pressure cannot be AD. Expected previous_iteration()")
        else:
            is_unsat = self.is_unsat(p)
            is_sat = 1 - is_unsat  
            num = (- self.m_vG * self.n_vG * 
                   (self.theta_s - self.theta_r) * 
                   (self.alpha_vG * np.abs(p)) ** self.n_vG)
            den = (p * ((self.alpha_vG * np.abs(p)) ** self.n_vG + 1) 
                   ** (self.m_vG + 1))
            C = (np.divide(num, den, out=np.zeros_like(num), where=den!=0) * self.is_unsat(p)
                + 0 * is_sat)      
            
        return C
            
            
            