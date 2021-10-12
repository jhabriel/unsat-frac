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

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.serif": "Computer Modern Roman",
        "font.sans-serif": "Computer Modern Sans Serif",
        "font.cursive": "Zapf Chancery",
    }
)

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


# %% Make grid
vert_frac = np.array([[50, 50], [0, 100]])
nx = np.array([20, 1])
L = np.array([100, 100])
gb = pp.meshing.cart_grid([vert_frac], nx=nx, physdims=L)
#  exporter = pp.Exporter(gb, "new_mexico", "out")
#  exporter.write_vtu()

dim_max = gb.dim_max()
g_bulk = gb.grids_of_dimension(dim_max)[0]
g_frac = gb.grids_of_dimension(dim_max - 1)[0]

d_bulk = gb.node_props(g_bulk)
d_frac = gb.node_props(g_frac)
d_edge = gb.edge_props((g_bulk, g_frac))

z_cc = g_bulk.cell_centers[dim_max - 1]
z_fc = g_bulk.face_centers[dim_max - 1]

# %% Physical parameters
K_sat = 0.1  # [cm/s] Saturated hydraulic conductivity
theta_r = 0.102  # [-] residual water content
theta_s = 0.368  # [-] saturated water content, a.k.a porosity
alpha_vG = 0.0335  # [1/cm] van Genuchten parameter
n_vG = 2  # van Genuchten parameter
m_vG = 1 - 1 / n_vG  # van Genuchten parameter

# %% Time parameters
final_time = 1000
num_time_steps = 10
time = 0
dt = final_time / num_time_steps

# %% Printing parameters
psi_b_l = []
psi_b_r = []
psi_f = []
tr_psi_b_l = []
tr_psi_b_r = []
lambda_l = []
lambda_r = []
water_vol = []
beta_interface = []

# %% Assign parameters
parameter_key = "flow"
bulk_var = "pressure_head"
fracture_var = "water_volume"
mortar_var = "mortar_flux"

# Declare primary variables
d_bulk[pp.PRIMARY_VARIABLES] = {bulk_var: {"cells": 1}}
d_frac[pp.PRIMARY_VARIABLES] = {fracture_var: {"cells": 1}}
for _, d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {mortar_var: {"cells": 1}}

# Assign data and boundary conditions
def assign_data(param_key, time):
    # Assign node parameters
    for g, d in gb:

        # Parameters for the bulk
        if g.dim == gb.dim_max():
            nc = g.num_cells
            fc = g.face_centers

            perm = pp.SecondOrderTensor(K_sat * np.ones(nc))

            left = np.where(np.abs(fc[0]) < 1e-5)[0]

            bc_faces = g.get_boundary_faces()
            bc_type = np.array(bc_faces.size * ["neu"])
            # bc_type[np.in1d(bc_faces, left)] = "dir"
            bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

            bc_values = np.zeros(g.num_faces)
            bc_values[left] = -0.01  # [cm/s]
            # Uncomment the line from below to add gravity contribution to Dirichlet faces if necessary
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
                "aperture": 0.1 * np.ones(g.num_cells),
                "datum": 0.0,
                "water_volume": np.array([0.0]),
                "is_sat": np.full(g.num_cells, False),
            }
            pp.initialize_data(g, d, param_key, specified_data)

    # Assign interface parameters
    for _, d in gb.edges():
        data = {"normal_diffusivity": K_sat / (2 * 0.1)}
        mg = d["mortar_grid"]
        pp.initialize_data(mg, d, param_key, data)


# %% Set initial states
for g, d in gb:
    pp.set_state(d)
    pp.set_iterate(d)
    if g.dim == gb.dim_max():
        d[pp.STATE][bulk_var] = -1000 * np.ones(g.num_cells)
        d[pp.STATE][bulk_var][0] = -7.5
        d[pp.STATE][pp.ITERATE][bulk_var] = d[pp.STATE][bulk_var].copy()
    else:
        d[pp.STATE][fracture_var] = np.zeros(g.num_cells)
        d[pp.STATE][pp.ITERATE][fracture_var] = d[pp.STATE][fracture_var].copy()

for _, d in gb.edges():
    mg = d["mortar_grid"]
    pp.set_state(d)
    pp.set_iterate(d)
    d[pp.STATE][mortar_var] = 0 * np.ones(mg.num_cells)
    d[pp.STATE][pp.ITERATE][mortar_var] = d[pp.STATE][mortar_var].copy()

# Export to printing arrays
# psi_f.append(d_frac[pp.STATE][fracture_var][0])
# psi_b_l.append(d_bulk[pp.STATE][bulk_var][0])
# psi_b_r.append(d_bulk[pp.STATE][bulk_var][1])
# tr_psi_b_l.append(d_bulk[pp.STATE][bulk_var][0])
# tr_psi_b_r.append(d_bulk[pp.STATE][bulk_var][1])
# lambda_l.append(d_edge[pp.STATE][mortar_var][1])
# lambda_r.append(d_edge[pp.STATE][mortar_var][0])
# water_vol.append(0)
# beta_interface.append(0)

#%% AD variables and manager
# Declare grid lists
bulk_grid_list = [g_bulk]
frac_grid_list = [g_frac]
edge_list = [e for e, _ in gb.edges()]

# Construct degree of freedom manager and equations manager
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

# Assign primary variables to their corresponding grids
psi_bulk = equation_manager.merge_variables([(g, bulk_var) for g in bulk_grid_list])
vol_frac = equation_manager.merge_variables([(g, fracture_var) for g in frac_grid_list])
lmbda = equation_manager.merge_variables([(e, mortar_var) for e in edge_list])

# Shorthands...
psi_bulk_m = psi_bulk.previous_iteration()
psi_bulk_n = psi_bulk.previous_timestep()
vol_frac_m = vol_frac.previous_iteration()
vol_frac_n = vol_frac.previous_timestep()
lmbda_m = lmbda.previous_iteration()
lmbda_n = lmbda.previous_timestep()

# %% Assign data
assign_data(parameter_key, time)

# %% Declare equations for the bulk

# Grid operators
div_bulk = pp.ad.Divergence([g_bulk])
mortar_proj = pp.ad.MortarProjections(gb=gb)
bound_bulk = pp.ad.BoundaryCondition(parameter_key, grids=[g_bulk])  # boundary vals
dirbc_bulk = DirBC(bound_bulk, [g_bulk])  # dirichlet bc vals
subdomain_proj_scalar = pp.ad.SubdomainProjections(gb=gb)
bulk_cell_rest = subdomain_proj_scalar.cell_restriction(g_bulk)
bulk_face_rest = subdomain_proj_scalar.face_restriction(g_bulk)
bulk_face_prol = subdomain_proj_scalar.face_prolongation(g_bulk)
frac_cell_rest = subdomain_proj_scalar.cell_restriction(g_frac)
frac_cell_prol = subdomain_proj_scalar.cell_prolongation(g_frac)

# Soil water retention curves
swrc = vanGenuchten(g_bulk, d_bulk, parameter_key)
theta_ad = pp.ad.Function(swrc.water_content, "water content")
krw_ad = pp.ad.Function(swrc.relative_permeability, "relative permeability")
C_ad = pp.ad.Function(swrc.moisture_capacity, "moisture capacity")

# MPFA discretization
mpfa_bulk = pp.ad.MpfaAd(parameter_key, [g_bulk])

# Face-averaging of relative permeabilities
avg_method = "upwind"
if avg_method == "arithmetic":
    arithmetic_avg = ArithmeticAverageAd(g_bulk, d_bulk, param_key)
    krw_faces_ad = arithmetic_avg(krw_ad(psi_bulk_m), krw_ad(dirbc_bulk))
elif avg_method == "upwind":
    upwind = UpwindFluxBasedAd(g_bulk, d_bulk, parameter_key)
    flux1p_bulk = (
        mpfa_bulk.flux * psi_bulk_m
        + mpfa_bulk.bound_flux * bound_bulk
        + mpfa_bulk.bound_flux
        * bulk_face_rest
        * mortar_proj.mortar_to_primary_int
        * lmbda_m
    )
    krw_faces_ad = upwind(krw_ad(psi_bulk_m), krw_ad(dirbc_bulk), flux1p_bulk)
else:
    raise NotImplementedError("Averaging method not implemented")

# Darcy fluxes
flux_bulk = (
    krw_faces_ad * mpfa_bulk.flux * psi_bulk
    + krw_faces_ad * mpfa_bulk.bound_flux * bound_bulk
    + krw_faces_ad
    * mpfa_bulk.bound_flux
    * bulk_face_rest
    * mortar_proj.mortar_to_primary_int
    * lmbda
)

# linearization = "newton"
linearization = "modified_picard"
# linearization = "l_scheme"

# Treatment of source and accumulation terms
# NOTE: The expression containing the active ad variable (psi_bulk) must be placed at the left of the (final) expression.
# Otherwise, numpy will broadcast the expression leading to an undesirable output format
source_bulk = pp.ad.ParameterArray(parameter_key, "source", grids=[g_bulk])
mass_bulk = pp.ad.MassMatrixAd(parameter_key, [g_bulk])
if linearization == "newton":
    accum_bulk_active = mass_bulk.mass * theta_ad(psi_bulk)
    accum_bulk_inactive = mass_bulk.mass * theta_ad(psi_bulk_n) * (-1)
elif linearization == "modified_picard":
    accum_bulk_active = mass_bulk.mass * psi_bulk * C_ad(psi_bulk_m)
    accum_bulk_inactive = mass_bulk.mass * (
        theta_ad(psi_bulk_m) - C_ad(psi_bulk_m) * psi_bulk_m - theta_ad(psi_bulk_n)
    )
elif linearization == "l_scheme":
    L = 0.0025
    accum_bulk_active = L * mass_bulk.mass * psi_bulk
    accum_bulk_inactive = mass_bulk.mass * (
        theta_ad(psi_bulk_m) - L * psi_bulk_m - theta_ad(psi_bulk_n)
    )
else:
    raise NotImplementedError(
        "Linearization scheme not implemented. Use 'newton', 'modified_picard', or 'l_scheme'."
    )

accumulation_bulk = accum_bulk_active + accum_bulk_inactive
conserv_bulk_eq = accumulation_bulk + dt * div_bulk * flux_bulk - dt * source_bulk

# Discretize and evaluate
conserv_bulk_eval = pp.ad.Expression(conserv_bulk_eq, dof_manager)
conserv_bulk_eval.discretize(gb)
conserv_bulk_num = conserv_bulk_eval.to_ad(gb)
# print(f'>> Conservation bulk: \n {conserv_bulk_num.jac.A} \n')

# %% Declare equations for the fracture


def fracture_pressure_head(water_volume):
    """
    This function computes the pressure head in the fracture given a volume of water.
    """

    aperture = 0.1  # [cm]
    datum = 0  # [cm]
    cell_centers = g_frac.cell_centers[1]

    local_interface_height = water_volume * aperture ** (-1)
    absolute_interface_height = local_interface_height + datum
    pressure_head = absolute_interface_height - cell_centers

    if isinstance(pressure_head, pp.ad.Ad_array):
        if np.all(pressure_head.val) < 0:
            pressure_head.val = -5 * np.ones(g_frac.num_cells)
    else:
        if pressure_head.all() < 0:
            pressure_head = -5 * np.ones(g_frac.num_cells)

    return pressure_head


psi_fracture = pp.ad.Function(fracture_pressure_head, name="Water pressure head")

# Water volume balance in the fracture


def fracture_heaviside(fracture_psi):
    return pp.ad.regularized_heaviside_2(fracture_psi, eps=0.1)


H_frac = pp.ad.Function(
    fracture_heaviside, name="Regularized Heaviside function for the fracture"
)

# Conservation equation in the fracture: This is a water volume balance
aperture = 0.1
sources_from_mortar = frac_cell_rest * mortar_proj.mortar_to_secondary_int * lmbda
conserv_frac_eq = vol_frac - sources_from_mortar * aperture * dt - vol_frac_n

conserv_frac_eval = pp.ad.Expression(conserv_frac_eq, dof_manager)
conserv_frac_eval.discretize(gb)
conserva_frac_num = conserv_frac_eval.to_ad(gb)

# %% Declare equations for the interface
mpfa_global = pp.ad.MpfaAd(parameter_key, [g_bulk, g_frac])  # not really needed?
robin = pp.ad.RobinCouplingAd(parameter_key, edge_list)

# Projected bulk pressure traces onto the mortar space
mortar_trace_psi_bulk = (
    mortar_proj.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_cell
    * psi_bulk
    + mortar_proj.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_face
    * bulk_face_rest
    * mortar_proj.mortar_to_primary_int
    * lmbda
)

mortar_trace_psi_bulk_m = (
    mortar_proj.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_cell
    * psi_bulk_m
    + mortar_proj.primary_to_mortar_avg
    * bulk_face_prol
    * mpfa_bulk.bound_pressure_face
    * bulk_face_rest
    * mortar_proj.mortar_to_primary_int
    * lmbda_m
)

# Projected fracture pressure onto the mortar space
mortar_psi_frac = (
    mortar_proj.secondary_to_mortar_avg * frac_cell_prol * psi_fracture(vol_frac)
)
mortar_psi_frac_m = (
    mortar_proj.secondary_to_mortar_avg * frac_cell_prol * psi_fracture(vol_frac_m)
)

# Upwinding of relative permeability on the interfaces
upwind_interface = InterfaceUpwindAd()
krw_interface_ad = upwind_interface(
    mortar_trace_psi_bulk_m,
    krw_ad(mortar_trace_psi_bulk_m),
    mortar_psi_frac_m,
    krw_ad(mortar_psi_frac_m),
)

# Regularized Heaviside function
entry_pressure = -5
regularization_parameter = 0.1


# def inflow_condition(psi_trace):
#     x = psi_trace - entry_pressure
#     return pp.ad.regularized_heaviside_2(x, eps=regularization_parameter)

# Interface flux
lam_1ph = 0 * robin.mortar_scaling * (mortar_trace_psi_bulk - mortar_psi_frac)
# lam_mph = krw_interface_ad * lam_1ph
# lam = inflow(mortar_trace_psi_bulk) * lam_mph
interface_flux_eq = lam_1ph + robin.mortar_discr * lmbda

interface_flux_eval = pp.ad.Expression(interface_flux_eq, dof_manager)
interface_flux_eval.discretize(gb)
interface_flux_num = interface_flux_eval.to_ad(gb)

# eval_test = eval_ad_expression(interface_flux_eq, gb, dof_manager, name="lambda")

# %% Assemble the system of equations
eqs = [
    pp.ad.Expression(conserv_bulk_eq, dof_manager, name="conservation bulk"),
    pp.ad.Expression(conserv_frac_eq, dof_manager, name="conservation fracture"),
    pp.ad.Expression(interface_flux_eq, dof_manager, name="interface flux"),
]
equation_manager.equations += eqs

# %% Initialize exporter
# d_bulk[pp.STATE]["S_eff"] = swrc.effective_saturation(d_bulk[pp.STATE][pressure_var])
# d_frac[pp.STATE]["S_eff"] = np.zeros(g_frac.num_cells)
#
# exporter = pp.Exporter(gb, "new_mexico", "out")
# exporter.write_vtu([pressure_var, "S_eff"], time_step=0)

# %% Time loop
total_iteration_counter = 0
for n in range(1, num_time_steps + 1):
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1
    time += dt

    print("Current time: ", np.round(time, decimals=1))

    while iteration_counter <= 100 and not (rel_res < 1e-4 or residual_norm < 1e-3):

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

    # Update next time step solution. Note that additive should be False here
    dof_manager.distribute_variable(solution, additive=False)

    print(f"Fracture water volume: {d_frac[pp.STATE][fracture_var][0]}")
    print(
        f"Fracture pressure head: {eval_ad_expression(mortar_psi_frac_m, gb, dof_manager, print_expression=False)[0]}"
    )
    print(f"Bulk (left) pressure head: {d_bulk[pp.STATE][bulk_var][0]}")
    print(f"Bulk (right) pressure head: {d_bulk[pp.STATE][bulk_var][1]}")
    print(
        f"Bulk (left) pressure trace: {eval_ad_expression(mortar_trace_psi_bulk_m, gb, dof_manager, print_expression=False)[1]}"
    )
    print(
        f"Bulk (right) pressure trace: {eval_ad_expression(mortar_trace_psi_bulk_m, gb, dof_manager, print_expression=False)[0]}"
    )
    print(f"Mortar (left): {d_edge[pp.STATE][mortar_var][1]}")
    print(f"Mortar (right): {d_edge[pp.STATE][mortar_var][0]}")
    # eval_ad_expression(lam_mph, gb, dof_manager)
    print()

    # Retrieve pressure trace
    # trace_psi = eval_ad_expression(tr_psi_bulk_m, gb, dof_manager, print_expression=False)
    # print(f"Water volume: {water_volume_n}")
    # print(f"Fracture pressure: {d_frac[pp.STATE][pressure_var][0]}")
    # print(f"Left bulk pressure: {d_bulk[pp.STATE][pressure_var][0]}")
    # print(f"Right bulk pressure: {d_bulk[pp.STATE][pressure_var][1]}")
    # print(f"Left pressure trace: {trace_psi[1]}")
    # print(f"Right pressure trace: {trace_psi[0]}")
    # print(f"Left mortar: {d_edge[pp.STATE][mortar_var][1]}")
    # print(f"Right mortar: {d_edge[pp.STATE][mortar_var][1]}")

    # Export to printing arrays
    # psi_f.append(d_frac[pp.STATE][pressure_var][0])
    # psi_b_l.append(d_bulk[pp.STATE][pressure_var][0])
    # psi_b_r.append(d_bulk[pp.STATE][pressure_var][1])
    # tr_psi_b_l.append(trace_psi[1])
    # tr_psi_b_r.append(trace_psi[0])
    # lambda_l.append(d_edge[pp.STATE][mortar_var][1])
    # lambda_r.append(d_edge[pp.STATE][mortar_var][0])
    # water_vol.append(water_volume_n)
    # beta_interface.append(beta)

    # Export to PARAVIEW
    # if np.mod(n, 10) == 0:
    #     exporter.write_vtu([pressure_var, "S_eff"], time_step=n)


# #%% Cell center pressures evolution
# fig, ax1 = plt.subplots(1, 1, constrained_layout=True)
#
# time_array = np.concatenate((np.array([0]), np.linspace(dt, final_time, num_time_steps)))
#
# # Bulk, left side
# ax1.plot(time_array, psi_b_l, color="red", linewidth=2, label=r'left pressure head')
# ax1.plot(time_array, psi_b_r, color="blue", linewidth=2, label=r'right pressure head')
# ax1.plot(time_array, psi_f, color="green", linewidth=2, label=r'fracture pressure head')
#
# # Set labels
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel(r'Pressure head [cm]')
#
# # Set legend
# ax1.legend()
# plt.show()
#
#
# # %% Pressure trace evolution
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
#
# time_array = np.concatenate((np.array([0]), np.linspace(dt, final_time, num_time_steps)))
#
# # SUBPLOT 1 -> Left pressure head
#
# # Plot left boundary pressure -> -75 [cm]
# ax1.plot(
#     [time_array[0], time_array[-1]],
#     [-75, -75],
#     color="red",
#     linewidth=2,
#     linestyle="-",
#     marker="None",
#     label=r"$\psi_{bc} = -75\;\mathrm{[cm]}$"
# )
#
# # Plot minimum entry pressure -> -80 [cm]
# ax1.plot(
#     [time_array[0], time_array[-1]],
#     [psi_threshold, psi_threshold],
#     color="green",
#     linewidth=2,
#     linestyle="-",
#     label=r"$\psi_{L} = -80\;\mathrm{[cm]}$"
# )
#
# # Plot left pressure trace
# ax1.plot(
#     time_array,
#     tr_psi_b_l,
#     color='orange',
#     linewidth=2,
#     linestyle="-",
#     marker="None",
#     label=r"$\mathrm{tr} \; \psi_{\mathrm{left}}$"
# )
#
# # Set axes limits
# ax1.set_ylim([-1000, -70])
#
# # Set labels
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel(r'Pressure head [cm]')
#
# # Set legend
# ax1.legend(
#     loc="lower right",
#     fontsize="small",
#     numpoints=1,
#     frameon=True,
#     handlelength=0.5
# )
#
# # SUBPLOT 2 -> Right pressure head
#
# # Plot left boundary pressure -> -75 [cm]
# ax2.plot(
#     [time_array[0], time_array[-1]],
#     [-75, -75],
#     color="red",
#     linewidth=2,
#     linestyle="-",
#     marker="None",
#     label=r"$\psi_{\mathrm{bc}}=-75\;\mathrm{[cm]}$"
# )
#
# # Plot minimum entry pressure -> -80 [cm]
# ax2.plot(
#     [time_array[0], time_array[-1]],
#     [psi_threshold, psi_threshold],
#     color="green",
#     linewidth=2,
#     linestyle="-",
#     label=r"$\psi_{L}=-80\;\mathrm{[cm]}$"
# )
#
# # Plot left pressure trace
# ax2.plot(
#     time_array,
#     tr_psi_b_r,
#     color='orange',
#     linewidth=2,
#     label=r"$\mathrm{tr} \,\, \psi_{right}^m$"
# )
#
# # Set axes limits
# ax2.set_ylim([-1010, -65])
#
# # Set labels
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel(r'Pressure head [cm]')
# ax2.set_xlabel('Time [s]')
# ax2.set_ylabel(r'Pressure head [cm]')
#
# # Set legend
# ax2.legend(
#     loc="center",
#     fontsize="small",
#     numpoints=1,
#     frameon=True,
#     handlelength=0.5
# )
#
# plt.show()
# plt.subplots_adjust(wspace=0.2)
# #fig.savefig("pressureTraceEvolution.pdf", transparent=True)
#
# # %% Mortar fluxes evolution
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
#
# time_array = np.concatenate((np.array([0]), np.linspace(dt, final_time, num_time_steps)))
#
# # SUBPLOT 1 -> Left pressure head
#
# # Plot left pressure trace
# ax1.plot(
#     time_array,
#     lambda_l,
#     color='orange',
#     linewidth=2,
#     label=r"$\lambda_{left}$"
# )
#
# # Set axes limits
# # ax1.set_ylim([-150, -70])
#
# # Set labels
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel(r'Mortar flux')
#
# # Set legend
# ax1.legend(
#     loc="upper left",
#     fontsize="small",
#     numpoints=1,
#     frameon=True,
#     handlelength=0.5
# )
#
# # SUBPLOT 2 -> Right pressure head
#
# # Plot left pressure trace
# ax2.plot(
#     time_array,
#     lambda_r,
#     color='orange',
#     linewidth=2,
#     label=r"$\lambda_{right}$"
# )
#
# # Set axes limits
# # ax2.set_ylim([-1010, -65])
#
# # Set labels
# ax1.set_xlabel('Time [s]')
# ax2.set_xlabel('Time [s]')
# # ax1.set_ylabel(r'Pressure head [cm]')
#
# # Set legend
# ax2.legend(
#     loc="center",
#     fontsize="small",
#     numpoints=1,
#     frameon=True,
#     handlelength=0.5
# )
#
# plt.show()
# fig.tight_layout()
# #fig.savefig("pressureAndMortarEvolution.pdf", transparent=True)
#
# # %% Fracture volume and interface evolution
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
#
# time_array = np.concatenate((np.array([0]), np.linspace(dt, final_time, num_time_steps)))
#
# # SUBPLOT 1 -> Left pressure head
#
# # Plot left pressure trace
# ax1.plot(
#     time_array,
#     water_vol,
#     color='orange',
#     linewidth=2,
#     label=r"$V_{f}$"
# )
#
# # Set axes limits
# # ax1.set_ylim([-150, -70])
#
# # Set labels
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel(r'Water volume')
#
# # Set legend
# ax1.legend(
#     loc="upper left",
#     fontsize="small",
#     numpoints=1,
#     frameon=True,
#     handlelength=0.5
# )
#
# # SUBPLOT 2 -> Right pressure head
#
# # Plot left pressure trace
# ax2.plot(
#     time_array,
#     beta_interface,
#     color='blue',
#     linewidth=2,
#     label=r"$\beta_{f}$"
# )
#
# # Set axes limits
# # ax2.set_ylim([-1010, -65])
#
# # Set labels
# ax1.set_xlabel('Time [s]')
# ax2.set_xlabel('Time [s]')
# ax1.set_ylabel('Water volume')
# ax2.set_ylabel('Interface position')
# # ax1.set_ylabel(r'Pressure head [cm]')
#
# # Set legend
# ax2.legend(
#     loc="center",
#     fontsize="small",
#     numpoints=1,
#     frameon=True,
#     handlelength=0.5
# )
#
# plt.show()
# fig.tight_layout()
# #fig.savefig("volumeAndInterface.pdf", transparent=True)
