from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import porepy as pp

Edge = Tuple[pp.Grid, pp.Grid]


class GridGenerator:
    """Parent class in charge of creating physical and ghost grid buckets"""

    def __init__(
        self,
        mesh_args: Union[Dict[str, int], Dict[str, float]],
        csv_file: str,
        domain: Optional[Union[Dict[str, int], Dict[str, float]]] = None,
        constraints: Optional[List[int]] = None,
    ):
        """
        Init method for the class

        Parameters:
            mesh_args (dict): containing the mesh parameters. Expected input is: mesh_args =
                {'mesh_size_frac': mesh_frac_size, 'mesh_size_bound': mesh_size_bound,
                'mesh_size_min': mesh_size_min}. Note that 'mesh_size_min' is optional for
                2D problems but mandatory for 3D problems.
            csv_file (str): path to the CSV file containing the fracture network and mesh
                constraints.
            domain (dict): dictionary containing the bounding box of the 2d domain. Expected
                input is domain = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}.
                Default is None. An error is raised if dim = 2 and domain is None.
            constraints (List of integers): Each item in the list corresponds to the row
                number of the CSV file corresponding to the constraint. Default is None.

        """

        # Assign attributes
        self.mesh_args = mesh_args
        self.file = csv_file
        self.domain = domain
        self.constraints = constraints
        self.dim = 2

    def __repr__(self) -> str:
        return "Grid factory object corresponding to the first example."

    def get_grid_buckets(self) -> Tuple[pp.GridBucket, pp.GridBucket]:
        """Construct physical and ghost grid buckets"""

        # First, create the ghost grid bucket
        network = pp.fracture_importer.network_2d_from_csv(
            self.file, domain=self.domain
        )
        if self.constraints is None:
            gb_ghost = network.mesh(self.mesh_args)
        else:
            gb_ghost = network.mesh(self.mesh_args, constraints=self.constraints)

        # Retrieve the 1D fractures

        # Create a list containing all the ghost fracture grids (in this case only 1)
        ghost_frac_list = [g for g, _ in gb_ghost if g.dim == 1]

        # Create a list containing the physical nodes of the fractures. Note that we use the
        # ghost fractures, but this does not matter, since we are only interested in
        # retrieving the coordinates of the fracture tips
        frac_nodes: List[np.ndarray] = []
        for g in ghost_frac_list:
            xmin: float = g.face_centers[0][0]
            xmax: float = g.face_centers[0][-1]
            ymin: float = g.face_centers[1][0]
            ymax: float = g.face_centers[1][-1]
            zmin: float = g.face_centers[2][0]
            zmax: float = g.face_centers[2][-1]
            nodes = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
            frac_nodes.append(nodes)

        # Get the physical length of the fractures. For 1D fractures, this is easy,
        # since the volume matches the length due to the dimensionality reduction. For 2D,
        # however, it would be more complicated.
        frac_length = [g.cell_volumes.sum() for g in ghost_frac_list]

        # Create the physical fractures as 1D grids with only one cell and their length
        mono_fracs = [
            pp.CartGrid(np.array([1]), physdims=length) for length in frac_length
        ]

        # Perturb the nodes of the created fracture grids. The perturbation
        # will be perform such that their nodes match with the physical nodes
        # retrieved earlier from the ghost grid bucket.
        for grid, node in zip(mono_fracs, frac_nodes):
            grid.nodes = node
            grid.compute_geometry()

        # Add zero-dimensional grid
        mono_fracs.append(gb_ghost.grids_of_dimension(0)[0])

        # We can now create our physical grid bucket, i.e., the one that will be used in the
        # simulation. Note that making a copy of the ghost grid bucket as in gb_ghost.copy()
        # does NOT work because this is not a hard copy of the object. Thus, we need to
        # create once again an independent object from scratch.
        if self.constraints is None:
            gb = network.mesh(self.mesh_args)
        else:
            gb = network.mesh(self.mesh_args, constraints=self.constraints)

        # Retrieve the list of fracture grids
        frac_list = [g for g, _ in gb if g.dim <= 1]

        # Now, we have to create a dictionary that keeps track of the mapping between the
        # old and new grids. That is, the g_map used by the replace_grids() method from the
        # pp.GridBucket class.
        # Reminder: key is the old grid, item is the new grid
        g_map: Dict[pp.Grid, pp.Grid] = {}
        for g_old, g_new in zip(frac_list, mono_fracs):
            g_map.update({g_old: g_new})

        # Finally, replace the old with new (mono) grids
        gb.replace_grids(g_map=g_map)

        return gb, gb_ghost

    # Beginning of methods related to grid and edge lists
    @staticmethod
    def grid_list(gb: pp.GridBucket) -> List[pp.Grid]:
        """
        Returns the list of grids

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket

        Returns
            (List of Grids): Containing the grids.
        """

        return [g for g, _ in gb]

    @staticmethod
    def bulk_grid_list(gb: pp.GridBucket) -> List[pp.Grid]:
        """Returns the list of bulk grids.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket

        Returns:
            (List of Grids): Containing the bulk grids.
        """

        return [g for g, _ in gb if g.dim == gb.dim_max()]

    @staticmethod
    def fracture_grid_list(gb: pp.GridBucket) -> List[pp.Grid]:
        """Returns the list of fracture grids.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket

        Returns
            (List of Grids): Containing the fracture grids.
        """

        return [g for g, _ in gb if g.dim == gb.dim_max() - 1]

    @staticmethod
    def local_fracture_network_grid_list(gb: pp.GridBucket) -> List[pp.Grid]:
        """Returns the list of lower-dimensional grids, including intersections

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket

        Returns
            (List of Grids): Containing the grids of lower-dimensional grids
        """

        return [g for g, _ in gb if g.dim < gb.dim_max()]

    @staticmethod
    def edge_list(gb: pp.GridBucket) -> List[Edge]:
        """Returns the list of edges.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket

        Returns:
            (List of Edges): Containing the edges.
        """

        return [e for e, _ in gb.edges()]

    @staticmethod
    def fracture_edge_list(gb: pp.GridBucket) -> List[Edge]:
        """Returns the list of edges between the bulk and fractures (without frac intersect.)

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket

        Returns:
            (List of Edges): Containing the edges.
        """

        return [e for e, d in gb.edges() if d["mortar_grid"].dim == gb.dim_max() - 1]

    @staticmethod
    def fracture_sin_alpha(g: pp.Grid, tol=1e-4) -> float:
        """Returns the sin of the angle of the fracture. Needed for relating h and V.

        Parameters:
            g (pp.Grid): PorePy grid.
            tol (float): tolerance. Default is 1e-4.
        Returns
            sin_alpha (float): grud angle in radians.
        """

        hipo = g.cell_volumes.sum()
        x_component = np.abs(g.face_centers[0][0] - g.face_centers[0][1])
        y_component = np.abs(g.face_centers[1][0] - g.face_centers[1][1])

        if x_component < tol:  # fracture is vertical
            sin_alpha = 1
        elif y_component < tol:  # fracture is horizontal
            sin_alpha = 0
        else:
            sin_alpha = y_component / hipo

        return sin_alpha
