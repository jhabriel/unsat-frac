import numpy as np
import porepy as pp


def create_mesh(mesh_size: float = 0.2):
    """
    Creates an unstructured 3D mesh for a given target mesh size

    Parameters
    ----------
    mesh_size : float, optional
        Target mesh size. The default is 0.2.
    Returns
    -------
    gb : PorePy Object
        Porepy grid bucket object.
    """
    # Load fracture network: Fracture + Ghost Fractures
    network_3d = pp.fracture_importer.network_3d_from_csv("fracture_network.csv")

    # Create mesh_arg dictionary
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_bound": mesh_size,
        "mesh_size_min": mesh_size / 10,
    }

    # Construct grid bucket
    # ghost_fracs = list(np.arange(1, 25))  # 1 to 24
    # gb = network_3d.mesh(mesh_args, constraints=ghost_fracs)

    gb = network_3d.mesh(mesh_args)

    return gb


# Create GridBucket
gb = create_mesh()
print(gb)

# Export
exporter = pp.Exporter(gb, file_name="frac_3d", folder_name="out")
exporter.write_vtu()
