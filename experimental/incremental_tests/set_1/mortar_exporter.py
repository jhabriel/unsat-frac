import porepy as pp
import numpy as np

#%% Create a grid bucket
fracs = np.array([[0.5, 0.5], [0, 1]])
nx = np.array([2, 1])
phys_dims = np.array([1, 1])
gb = pp.meshing.cart_grid([fracs], nx=nx, physdims=phys_dims)

#%% Set states
node_var = "node_var"
mortar_var = "mortar_var"
for g, d in gb:
    pp.set_state(d, state={node_var: g.dim * np.ones(g.num_cells)})
for e, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d, state={mortar_var: np.zeros(mg.num_cells)})

#%% Export
save = pp.Exporter(gb, file_name="out")
save.write_vtu([node_var, mortar_var])
