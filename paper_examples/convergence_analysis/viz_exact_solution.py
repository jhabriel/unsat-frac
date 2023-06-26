import porepy as pp
import os

from grid_factory import GridGenerator
from exact_solution import ExactSolution

# %% Retrieve exact solution object
ex = ExactSolution()

# %% Retrieve grid bucket
mesh_size = 0.01
gfo = GridGenerator(
    mesh_args={"mesh_size_frac": mesh_size, "mesh_size_bound": mesh_size},
    csv_file="network.csv",
    domain={"xmin": 0, "ymin": 0, "xmax": 1.0, "ymax": 1.0},
    constraints=[1, 2],
)
gb, _ = gfo.get_grid_buckets()
g_bulk = gb.grids_of_dimension(gb.dim_max())[0]
g_frac = gb.grids_of_dimension(gb.dim_max() - 1)[0]

d_bulk = gb.node_props(g_bulk)
d_bulk[pp.STATE] = {}

d_frac = gb.node_props(g_frac)
d_frac[pp.STATE] = {}

# %% Get exact solution
final_time = 0.5
h_bulk_ex = ex.rock_hydraulic_head(g_bulk, final_time)
h_frac_ex = ex.fracture_hydraulic_head(final_time)

d_bulk[pp.STATE]["hydraulic_head"] = h_bulk_ex
d_frac[pp.STATE]["hydraulic_head"] = h_frac_ex

# %%  Uncomment to plot with PorePy
# pp.plot_grid(g_bulk, h_bulk_ex, linewidth=0, plot_2d=True)
# pp.plot_grid(g_bulk, psi_bulk_ex, linewidth=0, plot_2d=True)

# %% Export to ParaView

# Create the directory if it does not exist
if not os.path.exists("out"):
    os.makedirs("out")

# Create ParaView
exporter = pp.Exporter(gb, "conv_analysis", "out")
exporter.write_vtu(["hydraulic_head"])

