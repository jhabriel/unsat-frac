import porepy as pp
import numpy as np
from typing import Union, List, Tuple, Optional, Literal, Dict

__all__ = ["GridFactory"]

Edge = Tuple[pp.Grid, pp.Grid]


class GridFactory:
    """Parent class in charge of creating physical and ghost grid buckets"""

    def __init__(
        self,
        dim: Literal[2, 3],
        mesh_args: Union[Dict[str, int], Dict[str, float]],
        csv_file: str,
        domain: Optional[Union[Dict[str, int], Dict[str, float]]] = None,
        has_domain: Optional[Literal[True, False]] = None,
        constraints: Optional[List[int]] = None,
    ):
        """
        Init method for the class

        Parameters:
            dim (int): dimensionality of the problem. Can be either 2 or 3.
            mesh_args (dict): containing the mesh parameters. Expected input is: mesh_args =
                {'mesh_size_frac': mesh_frac_size, 'mesh_size_bound': mesh_size_bound,
                'mesh_size_min': mesh_size_min}. Note that 'mesh_size_min' is optional for
                2D problems but mandatory for 3D problems.
            csv_file (str): path to the CSV file containing the fracture network and mesh
                constraints.
            domain (dict): dictionary containing the bounding box of the 2d domain. Expected
                input is domain = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}.
                Default is None. An error is raised if dim = 2 and domain is None.
            has_domain (bool): True if the domain is passed in the first row of the CSV
                file, and False otherwise. Note that has_domain is only used for 3D problems.
                Default is None. An error is raised id dim = 3 and has_domain is None.
            constraints (List of integers): Each item in the list corresponds to the row
                number of the CSV file corresponding to the constraint. Default is None.

        """

        # Check if dim is passed correctly
        if dim not in [2, 3]:
            raise ValueError(f"Expected dim 2 or 3. Got {dim} instead.")

        # Check if mesh_args is passed correctly
        if dim == 2:
            expected_keys = ["mesh_size_frac", "mesh_size_bound"]
            msg = f"Not enough keys in mesh_args. Expected at least: {expected_keys}."
            for key in expected_keys:
                if key not in mesh_args.keys():
                    raise KeyError(msg)
        elif dim == 3:
            expected_keys = ["mesh_size_frac", "mesh_size_bound", "mesh_size_min"]
            msg = f"Not enough keys in mesh_args. Expected: {expected_keys}."
            for key in expected_keys:
                if key not in mesh_args.keys():
                    raise KeyError(msg)

        # Check if domain is passed correctly
        if dim == 2 and domain is None:
            raise ValueError("domain cannot be None for 2D problems.")
        elif dim == 2 and domain is not None:
            expected_keys = ["xmin", "xmax", "ymin", "ymax"]
            msg = f"Not enough keys in domain. Expected: {expected_keys}."
            for key in expected_keys:
                if key not in domain.keys():
                    raise KeyError(msg)

        # Check if has_domain is passed correctly
        if dim == 3 and has_domain is None:
            raise ValueError("has_domain cannot be None for 3D problems.")
        elif dim == 3 and not isinstance(has_domain, bool):
            raise TypeError("has_domain must be either True or False.")

        # Assign attributes
        self.dim = dim
        self.mesh_args = mesh_args
        self.file = csv_file
        self.domain = domain
        self.has_domain = has_domain
        self.constraints = constraints

    def __repr__(self) -> str:
        return "GridFactory object."

    def get_grid_buckets(self) -> Tuple[pp.GridBucket, pp.GridBucket]:
        """Wrapper of 2D and 3D functionalities"""

        if self.dim == 2:
            gb_and_ghostgb = self._get_2d_gb()
        else:
            gb_and_ghostgb = self._get_3d_gb()

        return gb_and_ghostgb

    def _get_2d_gb(self) -> Tuple[pp.GridBucket, pp.GridBucket]:
        """Construct physical and ghost grid buckets"""

        # First, create the ghost grid bucket
        network = pp.fracture_importer.network_2d_from_csv(
            self.file, domain=self.domain
        )
        if self.constraints is not None:
            gb_ghost = network.mesh(self.mesh_args)
        else:
            gb_ghost = network.mesh(self.mesh_args, constraints=self.constraints)

        # Just for convinience...
        self.xmin = gb_ghost.bounding_box()[0][0]
        self.ymin = gb_ghost.bounding_box()[0][1]
        self.xmax = gb_ghost.bounding_box()[1][0]
        self.ymax = gb_ghost.bounding_box()[1][1]

        # Retrieve the 1D fractures

        # Create a list containing all the ghost fracture grids
        ghost_frac_list: List[pp.Grid] = []
        for ghost_g, _ in gb_ghost:
            if ghost_g.dim == 1:
                ghost_frac_list.append(ghost_g)

        # Create a list containing the physical nodes of the fractures. Note that we use the
        # ghost fractures, but this does not matter, since we are only interested in
        # retrieving the coordinates of the fracture tips
        frac_nodes: List[np.ndarray] = []
        for g in ghost_frac_list:
            xmin: float = g.bounding_box()[0][0]
            ymin: float = g.bounding_box()[0][1]
            xmax: float = g.bounding_box()[1][0]
            ymax: float = g.bounding_box()[1][1]
            zmin: float = 0.0
            zmax: float = 0.0
            nodes = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
            frac_nodes.append(nodes)

        # Get the physical length of the fractures. For 1D fractures, this is easy,
        # since the volume matches the length due to the dimensionality reduction. For 2D,
        # however, it would be more complicated.
        frac_length: List[float] = []
        for g in ghost_frac_list:
            frac_length.append(g.cell_volumes.sum())

        # Create the physical fractures as 1D grids with only one cell and their
        # corresponding length
        mono_fracs: List[pp.Grid] = []
        for lenght in frac_length:
            mono_fracs.append(pp.CartGrid(np.array([1]), physdims=lenght))

        # Now, we have to perturb the nodes of the created fracture grids. The perturbation
        # will be perform such that their nodes match with the physical nodes retrieved
        # earlier from the ghost grid bucket.
        for grid, node in zip(mono_fracs, frac_nodes):
            grid.nodes = node
            grid.compute_geometry()

        # We can now create our physical grid bucket, i.e., the one that will be used in the
        # simulation. Note that making a copy of the ghost grid bucket as in gb_ghost.copy()
        # does NOT work because this is not a hard copy of the object. Thus, we need to
        # create one again an independent object from scratch.
        if self.constraints is not None:
            gb = network.mesh(self.mesh_args)
        else:
            gb = network.mesh(self.mesh_args, constraints=self.constraints)

        # Retrive the list of fracture grids
        frac_list: List[pp.Grid] = []
        for g, _ in gb:
            if g.dim == 1:
                frac_list.append(g)

        # Now, we have to create a dictionary that keeps track of the mapping between the
        # old and new grids. That is, the g_map used by the replace_grids() method from the
        # pp.GridBucket class.
        g_map: Dict[pp.Grid, pp.Grid] = {}  # key is old grid, item is new grid
        for g_old, g_new in zip(frac_list, mono_fracs):
            g_map.update({g_old: g_new})

        # Finally, replace the old with new (mono) grids
        gb.replace_grids(g_map=g_map)

        return gb, gb_ghost

    def _get_3d_gb(self) -> Tuple[pp.GridBucket, pp.GridBucket]:
        raise NotImplementedError("Not yet implemented... sorry :/")

    # List getters
    def get_bulk_list(self, gb: pp.GridBucket) -> List[pp.Grid]:
        """Returns the list of bulk grids.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket
        Returns
            bulk_list (List of Grids): Containing the bulk grids.
        """
        bulk_list: List[pp.Grid] = []
        for g, _ in gb:
            if g.dim == self.dim:
                bulk_list.append(g)
        return bulk_list

    def get_fracture_list(self, gb: pp.GridBucket) -> List[pp.Grid]:
        """Returns the list of fracture grids.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket
        Returns
            fracture_list (List of Grids): Containing the fracture grids.
        """
        fracture_list: List[pp.Grid] = []
        for g, _ in gb:
            if g.dim == (self.dim - 1):
                fracture_list.append(g)
        return fracture_list

    def get_grid_list(self, gb: pp.GridBucket) -> List[pp.Grid]:
        """
        Returns the list of grids of dimensionality >= (dim - 1)

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket
        Returns
            grid_list (List of Grids): Containing the grids.
        """
        grid_list: List[pp.Grid] = []
        for g, _ in gb:
            if g.dim >= (self.dim - 1):
                grid_list.append(g)
        return grid_list

    def get_edge_list(self, gb: pp.GridBucket) -> List[Edge]:
        """Returns the list of edges for which the dimensionality of the mortar grid matches
        dimensionality of the fractures.

        Parameters:
            gb (pp.GridBucket): Mixed-dimensional grid bucket
        Returns
            edge_list (List of Edges): Containing the edges.
        """
        edge_list: List[Edge] = []
        for e, d in gb.edges():
            mg: pp.MortarGrid = d["mortar_grid"]
            if mg.dim == (self.dim - 1):
                edge_list.append(e)
        return edge_list
