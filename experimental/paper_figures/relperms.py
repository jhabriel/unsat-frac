#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:46:33 2021

@author: jv
"""
import numpy as np
import matplotlib.pyplot as plt

import pylab

params = {"text.usetex": False, "mathtext.fontset": "stixsans"}
pylab.rcParams.update(params)

# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'cm'


#%% Physical parameters
K_sat = 0.0922  # [cm/s] Saturated hydraulic conductivity
theta_r = 0.102  # [-] residual water content
theta_s = 0.368  # [-] saturated water content, a.k.a porosity
alpha_vG = 0.0335  # [1/cm] van Genuchten parameter
n_vG = 2  # van Genuchten parameter
m_vG = 1 - 1 / n_vG  # van Genuchten parameter

#%% SWRC
def is_unsat(p):
    """Determine whether the cell is saturated or not"""
    # {1, pressure_head < 0
    # {0, otherwise
    return 1 - np.heaviside(p, 1)


def water_content(p):
    """Water content as a function of the pressure head"""
    unsat = is_unsat(p)
    sat = 1 - unsat
    num = theta_s - theta_r
    den = (1 + (alpha_vG * np.abs(p)) ** n_vG) ** m_vG
    theta = (num * den ** (-1) + theta_r) * unsat + theta_s * sat
    return theta


def effective_saturation(p):
    """Effective saturation as a function of the water content"""
    num = water_content(p) - theta_r
    den = theta_s - theta_r
    s_eff = num * den ** (-1)
    return s_eff


def relative_permeability(p):
    """Relative permeability as a function of the effective saturation"""
    krw = (
        effective_saturation(p) ** (0.5)
        * (1 - (1 - effective_saturation(p) ** (1 / m_vG)) ** m_vG) ** 2
    )
    return krw


#%% Soil water retention curves
psi = np.linspace(-1000, 0, 200)
krw = relative_permeability(psi)

fig1, ax1 = plt.subplots(1, 1)

ax1.plot(psi, krw, color="green", linewidth=3)

ax1.set_xlabel(r"$\psi$", fontsize=15)

ax1.set_ylabel(r"$k^{rw}\left(\psi\right)$", fontsize=15)

fig1.tight_layout()

fig1.savefig("krw.pdf", transparent=True)
