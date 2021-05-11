"""
Caller for convergence test re_bulk_no_gravity.py

@author: jv
"""

import porepy as pp
import numpy as np
import itertools

from model_richards_dirbc_nograv import model as dirbc_nograv
from model_richards_mixbc_nograv import model as mixbc_nograv

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
        
def grid_tri_unstruct(mesh_size):
    
    domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
    network_2d = pp.FractureNetwork2d(None, None, domain)
    mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}
    gb = network_2d.mesh(mesh_args)

    return gb
    
def grid_cart_struct(mesh_size):

    n = int(1/mesh_size)
    gb = pp.meshing.cart_grid([], nx=[n, n], physdims=[1.0, 1.0])

    return gb

def test_richards_bulk_dirbc_nograv():
    """
    Check whether discrete-L2 relative errors for the pressure and the
    fluxes decrease monotonicallly as the mesh grid decreases
    """
    
    # We examine the following combinations on a unit square
    mesh_sizes = [0.2, 0.1, 0.05, 0.025]
    grids = ["cartesian", "triangular"]
    avg_schemes = ["arithmetic", "upwind"]
    solutions = ["parabolic", "trigonometric"]
    
    # Initialize dictionary and fields
    d = AutoVivification()
    for i in itertools.product(solutions, grids, avg_schemes):
        d[i[0]][i[1]][i[2]]["mesh_sizes"] = mesh_sizes
        d[i[0]][i[1]][i[2]]["pressure_error"] = []
        d[i[0]][i[1]][i[2]]["flux_error"] = []
    
    # Prepare grid buckets
    gb_cart = [ ]
    gb_tria = [ ]
    for mesh_size in mesh_sizes:
        gb_cart.append(grid_cart_struct(mesh_size))
        gb_tria.append(grid_tri_unstruct(mesh_size)) 
    
    # Populate fields
    for i in itertools.product(solutions, grids, avg_schemes, mesh_sizes):
        if i[1] == "cartesian":
            gb_idx = mesh_sizes.index(i[3])
            pressure_error, flux_error = dirbc_nograv(i[0], gb_cart[gb_idx], i[2])
        elif i[1] == "triangular":
            gb_idx = mesh_sizes.index(i[3])
            pressure_error, flux_error = dirbc_nograv(i[0], gb_tria[gb_idx], i[2])
    
        d[i[0]][i[1]][i[2]]["pressure_error"].append(pressure_error)
        d[i[0]][i[1]][i[2]]["flux_error"].append(flux_error) 

    # Check if errors decrease monotonically consistenly
    for i in itertools.product(solutions, grids, avg_schemes):

        p_error = np.array(d[i[0]][i[1]][i[2]]["pressure_error"])
        q_error = np.array(d[i[0]][i[1]][i[2]]["flux_error"])
        
        # Type -s flag in pytest to show convergence errors    
        print(f'Solution: {i[0]}, Grid: {i[1]}, Face average: {i[2]}')
        print(f"Pressure error: {p_error}")
        print(f"Flux error: {q_error}")
        print()

        assert np.all(p_error[1:] <= p_error[:-1])
        assert np.all(q_error[1:] <= q_error[:-1])


def test_richards_bulk_mixbc_nograv():
    """
    Check whether discrete-L2 relative errors for the pressure and the
    fluxes decrease monotonicallly as the mesh grid decreases
    """
    
    # We examine the following combinations on a unit square
    mesh_sizes = [0.2, 0.1, 0.05, 0.025]
    grids = ["cartesian", "triangular"]
    avg_schemes = ["arithmetic", "upwind"]
    solutions = ["parabolic", "trigonometric"]
    
    # Initialize dictionary and fields
    d = AutoVivification()
    for i in itertools.product(solutions, grids, avg_schemes):
        d[i[0]][i[1]][i[2]]["mesh_sizes"] = mesh_sizes
        d[i[0]][i[1]][i[2]]["pressure_error"] = []
        d[i[0]][i[1]][i[2]]["flux_error"] = []
    
    # Prepare grid buckets
    gb_cart = [ ]
    gb_tria = [ ]
    for mesh_size in mesh_sizes:
        gb_cart.append(grid_cart_struct(mesh_size))
        gb_tria.append(grid_tri_unstruct(mesh_size)) 
    
    # Populate fields
    for i in itertools.product(solutions, grids, avg_schemes, mesh_sizes):
        if i[1] == "cartesian":
            gb_idx = mesh_sizes.index(i[3])
            pressure_error, flux_error = mixbc_nograv(i[0], gb_cart[gb_idx], i[2])
        elif i[1] == "triangular":
            gb_idx = mesh_sizes.index(i[3])
            pressure_error, flux_error = mixbc_nograv(i[0], gb_tria[gb_idx], i[2])
    
        d[i[0]][i[1]][i[2]]["pressure_error"].append(pressure_error)
        d[i[0]][i[1]][i[2]]["flux_error"].append(flux_error) 

    # Check if errors decrease monotonically consistenly
    for i in itertools.product(solutions, grids, avg_schemes):

        p_error = np.array(d[i[0]][i[1]][i[2]]["pressure_error"])
        q_error = np.array(d[i[0]][i[1]][i[2]]["flux_error"])
        
        # Type -s flag in pytest to show convergence errors    
        print(f'Solution: {i[0]}, Grid: {i[1]}, Face average: {i[2]}')
        print(f"Pressure error: {p_error}")
        print(f"Flux error: {q_error}")
        print()

        assert np.all(p_error[1:] <= p_error[:-1])
        assert np.all(q_error[1:] <= q_error[:-1])
        
