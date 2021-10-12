"""
        Water infiltration in a initially dry soil column

Grids: Cartesian (structured) or triangular (unstructured)
Water retention curves: van Genuchten-Mualem
Linearization scheme: Modified Picard Iteration (Celia et. al. 1990)
Face averaging: Upwind or Arithmetic
Exact manufactured solution: Parabolic or Trigonometric
Boundary conditions: Mixed. Dirichlet top and bottom. Neumann left and right.

@author: jhabriel.varela@uib.no
"""

#%% Import modules
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib.pyplot import spy as sparsity

from porepy.numerics.ad.grid_operators import DirBC
from mdunsat.ad_utils.ad_utils import (
    InterfaceUpwindAd,
    ArithmeticAverageAd,
    UpwindFluxBasedAd,
    vanGenuchten,
)


def eval_ad_expression(
    ad_expression, gb, dof_manager, name=None, print_expression=True
):
    expression_eval = pp.ad.Expression(ad_expression, dof_manager)
    expression_eval.discretize(gb)
    expression_num = expression_eval.to_ad(gb)
    if print_expression:
        if isinstance(expression_num, pp.ad.Ad_array):
            if name is None:
                print("Evaluation of ad expression: \n")
                print(f"Array with values: \n {expression_num.val} \n")
                print(f"Jacobian with values: \n {expression_num.jac.A} \n")
            else:
                print(f"Evaluation of ad expression: {name} \n")
                print(f"Array with values: \n {expression_num.val} \n")
                print(f"Jacobian with values: \n {expression_num.jac.A} \n")

    return expression_num


#%% Make grid
vert_frac = np.array([[50, 50], [0, 100]])
nx = np.array([2, 1])
L = np.array([100, 100])
gb = pp.meshing.cart_grid([vert_frac], nx=nx, physdims=L)
exporter = pp.Exporter(gb, "new_mexico", "out")
exporter.write_vtu()

dim_max = gb.dim_max()
g_bulk = gb.grids_of_dimension(dim_max)[0]
g_frac = gb.grids_of_dimension(dim_max - 1)[0]

d_bulk = gb.node_props(g_bulk)
d_frac = gb.node_props(g_frac)
d_edge = gb.edge_props((g_bulk, g_frac))

z_cc = g_bulk.cell_centers[dim_max - 1]
z_fc = g_bulk.face_centers[dim_max - 1]

#%% Physical parameters
K_sat = 0.0922  # [cm/s] Saturated hydraulic conductivity
theta_r = 0.102  # [-] residual water content
theta_s = 0.368  # [-] saturated water content, a.k.a porosity
alpha_vG = 0.0335  # [1/cm] van Genuchten parameter
n_vG = 2  # van Genuchten parameter
m_vG = 1 - 1 / n_vG  # van Genuchten parameter

#%% Time parameters
final_time = 3600 * 4
num_time_steps = 100
time = 0
dt = final_time / num_time_steps

#%% Printing parameters
psi_b_l = []
psi_b_r = []
tr_psi_b_l = []
tr_psi_b_r = []
lambda_l = []
lambda_r = []

#%% Assign parameters
param_key = "flow"
pressure_var = "pressure_head"
mortar_var = "mortar_flux"

for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}

for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {mortar_var: {"cells": 1}}


def assign_data(param_key, time):

    # Assign node parameters
    for g, d in gb:

        # Parameters for the bulk
        if g.dim == gb.dim_max():
            nc = g.num_cells
            fc = g.face_centers
            Lx = g.bounding_box()[1][0]

            perm = pp.SecondOrderTensor(K_sat * np.ones(nc))

            left = np.where(np.abs(fc[0]) < 1e-5)[0]
            right = np.where(np.abs(fc[0] - Lx) < 1e-5)[0]

            bc_faces = g.get_boundary_faces()
            bc_type = np.array(bc_faces.size * ["neu"])
            bc_type[np.in1d(bc_faces, left)] = "dir"
            bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

            bc_values = np.zeros(g.num_faces)
            bc_values[left] = -75.0  # [cm]
            # Add gravity contribution to Dirichlet faces
            # bc_values[bc.is_dir] += z_fc[bc.is_dir]

            source_term = np.zeros(nc)

            specified_parameters = {
                "second_order_tensor": perm,
                "bc": bc,
                "bc_values": bc_values,
                "source": source_term,
                "mass_weight": np.ones(nc),
                "theta_r": theta_r,
                "theta_s": theta_s,
                "alpha_vG": alpha_vG,
                "n_vG": n_vG,
                "m_vG": m_vG,
            }

            pp.initialize_data(g, d, param_key, specified_parameters)

        # Parameters for the fracture
        else:
            perm = pp.SecondOrderTensor(K_sat * np.ones(g.num_cells))
            # No-flow Neumann conditions
            bc = pp.BoundaryCondition(g)
            bc_val = np.zeros(g.num_faces)
            specified_data = {
                "second_order_tensor": perm,
                "bc": bc,
                "bc_values": bc_val,
            }
            d = pp.initialize_data(g, d, param_key, specified_data)

    # Assign interface parameters
    for e, d in gb.edges():

        data = {"normal_diffusivity": K_sat}
        mg = d["mortar_grid"]
        pp.initialize_data(mg, d, param_key, data)


#%% Set initial states
for g, d in gb:
    pp.set_state(d)
    pp.set_iterate(d)
    if g.dim == 2:
        d[pp.STATE][pressure_var] = -1000 * np.ones(g.num_cells)
    else:
        d[pp.STATE][pressure_var] = -100000 * np.ones(g.num_cells)
    d[pp.STATE][pp.ITERATE][pressure_var] = d[pp.STATE][pressure_var].copy()

for e, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][mortar_var] = np.zeros(mg.num_cells)
    d[pp.STATE][pp.ITERATE][mortar_var] = d[pp.STATE][mortar_var].copy()

# Export to printing arrays
psi_b_l.append(d_bulk[pp.STATE][pressure_var][0])
psi_b_r.append(d_bulk[pp.STATE][pressure_var][0])
tr_psi_b_l.append(d_bulk[pp.STATE][pressure_var][0])
tr_psi_b_r.append(d_bulk[pp.STATE][pressure_var][0])
lambda_l.append(d_edge[pp.STATE][mortar_var][1])
lambda_r.append(d_edge[pp.STATE][mortar_var][0])


#%% AD variables and manager
grid_list = [g for g, _ in gb]
edge_list = [e for e, _ in gb.edges()]
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
psi = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])
lmbda = equation_manager.merge_variables([(e, mortar_var) for e in edge_list])

psi_m = psi.previous_iteration()
psi_n = psi.previous_timestep()
lmbda_m = lmbda.previous_iteration()
lmbda_n = lmbda.previous_timestep()

#%% Assign data
assign_data(param_key, time)

#%% Declare equations for the bulk

# Grid operators
div_bulk = pp.ad.Divergence([g_bulk])
mortar_proj = pp.ad.MortarProjections(gb=gb)
bound_bulk = pp.ad.BoundaryCondition(param_key, grids=[g_bulk])  # boundary vals
dirbc_bulk = DirBC(bound_bulk, [g_bulk])  # dirichlet bc vals
subdomain_proj_scalar = pp.ad.SubdomainProjections(gb=gb)
bulk_cell_proj = subdomain_proj_scalar.cell_restriction(g_bulk)
bulk_face_proj = subdomain_proj_scalar.face_restriction(g_bulk)
frac_cell_proj = subdomain_proj_scalar.cell_restriction(g_frac)

# Soil water retention curves
swrc = vanGenuchten(g_bulk, d_bulk, param_key)
theta_ad = pp.ad.Function(swrc.water_content, "water content")
krw_ad = pp.ad.Function(swrc.relative_permeability, "relative permeability")
C_ad = pp.ad.Function(swrc.moisture_capacity, "moisture capacity")

# MPFA discretization
mpfa_bulk = pp.ad.MpfaAd(param_key, [g_bulk])

# Face-averaging of relative permeabilities
avg_method = "upwind"
if avg_method == "arithmetic":
    arithmetic_avg = ArithmeticAverageAd(g_bulk, d_bulk, param_key)
    krw_faces_ad = arithmetic_avg(krw_ad(psi_m), krw_ad(dirbc_bulk))
elif avg_method == "upwind":
    upwind = UpwindFluxBasedAd(g_bulk, d_bulk, param_key)
    flux1p_bulk = (
        mpfa_bulk.flux * bulk_cell_proj * psi_m
        + mpfa_bulk.bound_flux * bound_bulk
        + mpfa_bulk.bound_flux
        * bulk_face_proj
        * mortar_proj.mortar_to_primary_int
        * lmbda_m
    )
    krw_faces_ad = upwind(
        krw_ad(bulk_cell_proj * psi_m), krw_ad(dirbc_bulk), flux1p_bulk
    )
else:
    raise ValueError("Averaging method not implemented")

# Darcy fluxes
flux_bulk = (
    krw_faces_ad * mpfa_bulk.flux * bulk_cell_proj * psi
    + krw_faces_ad * mpfa_bulk.bound_flux * bound_bulk
    + krw_faces_ad
    * mpfa_bulk.bound_flux
    * bulk_face_proj
    * mortar_proj.mortar_to_primary_int
    * lmbda
)

# Source and accumulation terms (Linearization: Modified Picard iteration)
# Note: The expression containing the active ad variable (psi) must
# be placed at the left of the (final) expression. Otherwise, numpy
# will broadcast the expression leading to and undesirable ouput format
source_bulk = pp.ad.ParameterArray(param_key, "source", grids=[g_bulk])
mass_bulk = pp.ad.MassMatrixAd(param_key, [g_bulk])
accum_bulk_active = mass_bulk.mass * bulk_cell_proj * psi * C_ad(bulk_cell_proj * psi_m)
accum_bulk_inactive = mass_bulk.mass * (
    theta_ad(bulk_cell_proj * psi_m)
    - C_ad(bulk_cell_proj * psi_m) * (bulk_cell_proj * psi_m)
    - theta_ad(bulk_cell_proj * psi_n)
)
accumulation_bulk = accum_bulk_active + accum_bulk_inactive
sources_from_mortar = mortar_proj.mortar_to_secondary_int * lmbda

conserv_bulk_eq = accumulation_bulk + dt * div_bulk * flux_bulk - dt * source_bulk
conserv_bulk_eval = pp.ad.Expression(conserv_bulk_eq, dof_manager)
conserv_bulk_eval.discretize(gb)
conserv_bulk_num = conserv_bulk_eval.to_ad(gb)
print(f">> Conservation bulk: \n {conserv_bulk_num.jac.A} \n")

#%% Declare equations for the fracture
conserv_frac_eq = (
    frac_cell_proj * (psi - psi_n)
    + dt * frac_cell_proj * mortar_proj.mortar_to_secondary_int * lmbda
)
conserv_frac_eval = pp.ad.Expression(conserv_frac_eq, dof_manager)
conserv_frac_eval.discretize(gb)

#%% Declare equations for the interface
mpfa_global = pp.ad.MpfaAd(param_key, grid_list)
robin = pp.ad.RobinCouplingAd(param_key, edge_list)

# Projected bulk pressure traces onto the mortar space
tr_psi_bulk = (
    mortar_proj.primary_to_mortar_avg * mpfa_global.bound_pressure_cell * psi
    + mortar_proj.primary_to_mortar_avg
    * mpfa_global.bound_pressure_face
    * mortar_proj.mortar_to_primary_int
    * lmbda
)

tr_psi_bulk_m = (
    mortar_proj.primary_to_mortar_avg * mpfa_global.bound_pressure_cell * psi_m
    + mortar_proj.primary_to_mortar_avg
    * mpfa_global.bound_pressure_face
    * mortar_proj.mortar_to_primary_int
    * lmbda_m
)

# Projected fracture pressure onto the mortar space
psi_frac = mortar_proj.secondary_to_mortar_avg * psi
psi_frac_m = mortar_proj.secondary_to_mortar_avg * psi_m

# Upwinding of relative permeability on the interfaces
upwind_interface = InterfaceUpwindAd()
krw_interface_ad = upwind_interface(
    tr_psi_bulk_m, krw_ad(tr_psi_bulk_m), psi_frac_m, krw_ad(psi_frac_m)
)

# Regularized Heaviside function
psi_threshold = -80
regularization_parameter = 1e-2


def sealed_heaviside(psi_trace):
    x = psi_trace - psi_threshold
    return pp.ad.regularized_heaviside_2(x, eps=regularization_parameter)


H_eps = pp.ad.Function(sealed_heaviside, name="Regularized Heaviside function")

# Interface flux

lambda_single_ph = (
    robin.mortar_scaling * (tr_psi_bulk - psi_frac) + robin.mortar_discr * lmbda
)
lambda_multi_ph = krw_interface_ad * lambda_single_phase
interface_flux = H_eps(tr_psi_bulk) * interface_flux_krw

eval_test = eval_ad_expression(
    H_eps(tr_psi_bulk) * interface_flux_krw, gb, dof_manager, name="test"
)

assert False

# interface_flux_1p = robin.mortar_scaling * (tr_psi_bulk - psi_frac) + robin.mortar_discr * lmbda
# interface_flux_1p_eval = eval_ad_expression(interface_flux_1p, gb, dof_manager)

interface_flux_eq = (
    H_eps(tr_psi_bulk) * robin.mortar_scaling * (tr_psi_bulk - psi_frac)
    + robin.mortar_discr * lmbda
)


# interface_flux_eq = (
#     krw_interface_ad * robin.mortar_scaling * (tr_psi_bulk - psi_frac) * H_eps(tr_psi_bulk_m)
#     + robin.mortar_discr * lmbda
#     )

# interface_flux_eval = pp.ad.Expression(interface_flux_eq, dof_manager)
# interface_flux_eval.discretize(gb)

interface_flux_eval = eval_ad_expression(
    interface_flux_eq, gb, dof_manager, name="lambda"
)
assert False


expression_eval = pp.ad.Expression(interface_flux_eq, dof_manager)
expression_eval.discretize(gb)
expression_num = expression_eval.to_ad(gb)
print("OK")
assert False


#%% Assemble the system of equations
eqs = [
    pp.ad.Expression(conserv_bulk_eq, dof_manager, name="conservation bulk"),
    pp.ad.Expression(conserv_frac_eq, dof_manager, name="conservation fracture"),
    pp.ad.Expression(interface_flux_eq, dof_manager, name="interface flux"),
]
equation_manager.equations += eqs

#%% Initialize exporter
d_bulk[pp.STATE]["S_eff"] = swrc.effective_saturation(d_bulk[pp.STATE][pressure_var])
d_frac[pp.STATE]["S_eff"] = np.zeros(g_frac.num_cells)

exporter = pp.Exporter(gb, "new_mexico", "out")
exporter.write_vtu([pressure_var, "S_eff"], time_step=0)

#%% Time loop
total_iteration_counter = 0
for n in range(1, num_time_steps + 1):
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1
    time += dt

    print("Current time: ", np.round(time, decimals=1))

    while iteration_counter <= 30 and not (rel_res < 1e-6 or residual_norm < 1e-6):

        # Solve for pressure increment and update pressure
        A, b = equation_manager.assemble_matrix_rhs()
        solution = spla.spsolve(A, b)

        # # Distribute variable to local data dictionaries
        dof_manager.distribute_variable(solution, additive=True, to_iterate=True)

        # Compute 'error' as norm of the residual
        residual_norm = np.linalg.norm(b, 2)
        if iteration_counter == 0:
            initial_residual_norm = residual_norm
        else:
            initial_residual_norm = max(residual_norm, initial_residual_norm)
        rel_res = residual_norm / initial_residual_norm
        print(
            "iteration",
            iteration_counter,
            "abs res",
            residual_norm,
            "rel res",
            residual_norm / initial_residual_norm,
        )

        # Prepare next iteration
        iteration_counter += 1
        total_iteration_counter += 1

    print()

    # Update next time step solution. Note that additive should be False here
    dof_manager.distribute_variable(solution, additive=False)

    # Retrieve pressure trace
    trace_psi = eval_ad_expression(
        tr_psi_bulk_m, gb, dof_manager, print_expression=False
    )

    # Export to printing arrays
    psi_b_l.append(d_bulk[pp.STATE][pressure_var][0])
    psi_b_r.append(d_bulk[pp.STATE][pressure_var][1])
    tr_psi_b_l.append(trace_psi[1])
    tr_psi_b_r.append(trace_psi[0])
    lambda_l.append(d_edge[pp.STATE][mortar_var][1])
    lambda_r.append(d_edge[pp.STATE][mortar_var][0])

    # Export to PARAVIEW
    if np.mod(n, 10) == 0:
        exporter.write_vtu([pressure_var, "S_eff"], time_step=n)


#%% Pressure trace evolution
fig, (ax1, ax2) = plt.subplots(1, 2)

time_array = np.concatenate(
    (np.array([0]), np.linspace(dt, final_time, num_time_steps))
)

# SUBPLOT 1 -> Left pressure head

# Plot left boundary pressure -> -75 [cm]
ax1.plot(
    [time_array[0], time_array[-1]],
    [-75, -75],
    color="red",
    linewidth=2,
    linestyle="-",
    label=r"$\Psi_{bc}$",
)

# Plot minimum entry pressure -> -80 [cm]
ax1.plot(
    [time_array[0], time_array[-1]],
    [pressure_threshold, pressure_threshold],
    color="green",
    linewidth=2,
    linestyle="-",
    label=r"$\Psi_{L}$",
)

# Plot left pressure trace
ax1.plot(
    time_array,
    tr_psi_b_l,
    color="orange",
    linewidth=2,
    label=r"$\mathrm{tr} \,\, \Psi_{left}^m$",
)

# Set axes limits
ax1.set_ylim([-150, -70])

# Set labels
ax1.set_xlabel("Time [s]")
ax1.set_ylabel(r"Pressure head [cm]")

# Set legend
ax1.legend(
    loc="lower right", fontsize="small", numpoints=1, frameon=True, handlelength=0.5
)


# SUBPLOT 2 -> Right pressure head

# Plot left boundary pressure -> -75 [cm]
ax2.plot(
    [time_array[0], time_array[-1]],
    [-75, -75],
    color="red",
    linewidth=2,
    linestyle="-",
    label=r"$\Psi_{bc}$",
)

# Plot minimum entry pressure -> -80 [cm]
ax2.plot(
    [time_array[0], time_array[-1]],
    [pressure_threshold, pressure_threshold],
    color="green",
    linewidth=2,
    linestyle="-",
    label=r"$\Psi_{L}$",
)

# Plot left pressure trace
ax2.plot(
    time_array,
    tr_psi_b_r,
    color="orange",
    linewidth=2,
    label=r"$\mathrm{tr} \,\, \Psi_{right}^m$",
)

# Set axes limits
ax2.set_ylim([-1010, -65])

# Set labels
ax1.set_xlabel("Time [s]")
ax1.set_ylabel(r"Pressure head [cm]")

# Set legend
ax2.legend(loc="center", fontsize="small", numpoints=1, frameon=True, handlelength=0.5)

fig.tight_layout()
fig.savefig("pressureTraceEvolution.pdf", transparent=True)


#%% Mortar fluxes evolution
fig, (ax1, ax2) = plt.subplots(1, 2)

time_array = np.concatenate(
    (np.array([0]), np.linspace(dt, final_time, num_time_steps))
)

# SUBPLOT 1 -> Left pressure head

# Plot left pressure trace
ax1.plot(time_array, lambda_l, color="orange", linewidth=2, label=r"$\lambda_{left}$")

# Set axes limits
# ax1.set_ylim([-150, -70])

# Set labels
ax1.set_xlabel("Time [s]")
ax1.set_ylabel(r"Mortar flux")

# Set legend
ax1.legend(
    loc="lower right", fontsize="small", numpoints=1, frameon=True, handlelength=0.5
)


# SUBPLOT 2 -> Right pressure head

# Plot left pressure trace
ax2.plot(time_array, lambda_r, color="orange", linewidth=2, label=r"$\lambda_{right}$")

# Set axes limits
# ax2.set_ylim([-1010, -65])

# Set labels
ax1.set_xlabel("Time [s]")
# ax1.set_ylabel(r'Pressure head [cm]')

# Set legend
ax2.legend(loc="center", fontsize="small", numpoints=1, frameon=True, handlelength=0.5)

fig.tight_layout()
fig.savefig("pressureAndMortarEvolution.pdf", transparent=True)
