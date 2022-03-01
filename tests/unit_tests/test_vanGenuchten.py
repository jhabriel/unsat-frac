import porepy as pp
import numpy as np
import sympy as sym
import pytest

from mdunsat.ad_utils import vanGenuchten

#%% Exact expressions
p = sym.symbols("p", real=True)

theta_r = 0.1
theta_s = 0.4
alpha = 0.4
n = 2
m = 0.5

theta_sym = (theta_s - theta_r) / (1 + (alpha * (p ** 0.5) ** 2.0) ** n) ** m + theta_r
S_eff = (theta_sym - theta_r) / (theta_s - theta_r)
krw_sym = S_eff ** 0.5 * (1 - (1 - S_eff ** (1 / m)) ** m) ** 2
C_sym = sym.diff(theta_sym, p)

theta_ex = sym.lambdify(p, theta_sym, "numpy")
krw_ex = sym.lambdify(p, krw_sym, "numpy")
C_ex = sym.lambdify(p, C_sym, "numpy")

#%% Set up code
gb = pp.meshing.cart_grid([], nx=[3, 2])
g = gb.grids_of_dimension(2)[0]
d = gb.node_props(g)

# Assign parameters
perm = pp.SecondOrderTensor(np.ones(g.num_cells))
param_key = "flow"
pressure_var = "pressure_head"
d[pp.PRIMARY_VARIABLES] = {pressure_var: {"cells": 1}}
specified_data = {
    "theta_r": theta_r,
    "theta_s": theta_s,
    "alpha_vG": alpha,
    "n_vG": n,
    "m_vG": m,
}
d = pp.initialize_data(g, d, param_key, specified_data)

# Initiliaze data
cc = g.cell_centers
pp.set_state(d)
pp.set_iterate(d)
psi_val = np.zeros(g.num_cells)
d[pp.STATE][pressure_var] = psi_val
d[pp.STATE][pp.ITERATE][pressure_var] = psi_val

# AD variables and manager
grid_list = [g]
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
psi = equation_manager.merge_variables([(g, pressure_var) for g in grid_list])

# Call van Genuchten class
swrc = vanGenuchten(g, d, param_key)
theta_ad = pp.ad.Function(swrc.water_content, "water content")
krw_ad = pp.ad.Function(swrc.relative_permeability, "relative permeability")
C_ad = pp.ad.Function(swrc.moisture_capacity, "moisture capacity")

#%% Tests

# Tests involving expected types
def test_exception_krw():
    """ Only passes if a TypeError exception is raised"""
    psi_val = np.array([-10000, -500, -10, 0, 1000, 10000])
    d[pp.STATE][pressure_var] = psi_val
    d[pp.STATE][pp.ITERATE][pressure_var] = psi_val

    with pytest.raises(TypeError):
        krw_val = pp.ad.Expression(krw_ad(psi), dof_manager)
        krw_val.to_ad(gb)


def test_exception_C():
    """ Only passes if a TypeError exception is raised"""
    psi_val = np.array([-10000, -500, -10, 0, 1000, 10000])
    d[pp.STATE][pressure_var] = psi_val
    d[pp.STATE][pp.ITERATE][pressure_var] = psi_val

    with pytest.raises(TypeError):
        C_val = pp.ad.Expression(C_ad(psi), dof_manager)
        C_val.to_ad(gb)


# Thresholding tests
def test_theta_threshold():
    """ Test if water content function gives correct unsat/sat values"""
    theta_0 = (
        swrc.water_content(-1) < d[pp.PARAMETERS][param_key]["theta_s"]
        and swrc.water_content(-1) >= d[pp.PARAMETERS][param_key]["theta_r"]
    )
    theta_1 = swrc.water_content(0) == d[pp.PARAMETERS][param_key]["theta_s"]
    theta_2 = swrc.water_content(1) == d[pp.PARAMETERS][param_key]["theta_s"]
    theta_3 = swrc.water_content(1000) == d[pp.PARAMETERS][param_key]["theta_s"]

    assert [theta_0, theta_1, theta_2, theta_3] == [True, True, True, True]


def test_krw_threshold():
    """ Test if relative permeability function gives correct unsat/sat values"""
    krw_0 = swrc.relative_permeability(-1) < 1.0 and swrc.relative_permeability(-1) >= 0
    krw_1 = swrc.relative_permeability(0) == 1.0
    krw_2 = swrc.relative_permeability(1) == 1.0
    krw_3 = swrc.relative_permeability(1000) == 1.0

    assert [krw_0, krw_1, krw_2, krw_3] == [True, True, True, True]


def test_C_threshold():
    """ Test if moisture capacity function gives correct unsat/sat values"""

    C_0 = swrc.moisture_capacity(-1) != 0
    C_1 = swrc.moisture_capacity(1) == 0
    C_2 = swrc.moisture_capacity(1000) == 0

    assert [C_0, C_1, C_2] == [True, True, True]


# Testing values
def test_ad_theta():
    """ Test if ad water content function gives correct values """
    psi_val = np.array([-10000, -500, -10, 0, 1000, 10000])
    d[pp.STATE][pressure_var] = psi_val
    d[pp.STATE][pp.ITERATE][pressure_var] = psi_val

    theta_eval = pp.ad.Expression(theta_ad(psi), dof_manager)
    theta_eval.discretize(gb)
    theta_num = theta_eval.to_ad(gb)
    theta_true = []
    for val in psi_val:
        if val < 0:
            theta_true.append(theta_ex(val))
        else:
            theta_true.append(d[pp.PARAMETERS][param_key]["theta_s"])
    theta_true = np.array(theta_true)

    np.testing.assert_almost_equal(theta_num.val, theta_true)


def test_ad_krw():
    """ Test if ad rel perm gives correct values """
    psi_val = np.array([-10000, -500, -10, 0, 1000, 10000])
    d[pp.STATE][pressure_var] = psi_val
    d[pp.STATE][pp.ITERATE][pressure_var] = psi_val

    krw_eval = pp.ad.Expression(krw_ad(psi.previous_iteration()), dof_manager)
    krw_eval.discretize(gb)
    krw_num = krw_eval.to_ad(gb)
    krw_true = []
    for val in psi_val:
        if val < 0:
            krw_true.append(krw_ex(val))
        else:
            krw_true.append(1.0)
    krw_true = np.array(krw_true)

    np.testing.assert_almost_equal(krw_num, krw_true)


def test_ad_C():
    """ Test if ad moisture capacity gives correct values """
    psi_val = np.array([-10000, -500, -10, 0, 1000, 10000])
    d[pp.STATE][pressure_var] = psi_val
    d[pp.STATE][pp.ITERATE][pressure_var] = psi_val

    C_eval = pp.ad.Expression(C_ad(psi.previous_iteration()), dof_manager)
    C_num = C_eval.to_ad(gb)
    C_true = []
    for val in psi_val:
        if val < 0:
            C_true.append(C_ex(val))
        else:
            C_true.append(0.0)
    C_true = np.array(C_true)

    np.testing.assert_almost_equal(C_num, C_true)


# Compatibility test
def test_theta_jac_equal_C():
    """ Test if theta_ad.jac == C """
    psi_val = np.array([-10000, -500, -10, 0, 1000, 10000])
    d[pp.STATE][pressure_var] = psi_val
    d[pp.STATE][pp.ITERATE][pressure_var] = psi_val

    theta_eval = pp.ad.Expression(theta_ad(psi), dof_manager)
    theta_eval.discretize(gb)
    theta_num = theta_eval.to_ad(gb)
    theta_jac = theta_num.jac.diagonal()

    C_eval = pp.ad.Expression(C_ad(psi.previous_iteration()), dof_manager)
    C_num = C_eval.to_ad(gb)

    np.testing.assert_almost_equal(theta_jac, C_num)
