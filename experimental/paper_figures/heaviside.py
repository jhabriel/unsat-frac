import porepy as pp
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt


#%% Regularized Heaviside function (Chan&Vese, 2001)
val = np.linspace(-0.3, 0.3, 200)
jac = sps.diags(np.ones_like(val), offsets=0)
xx = pp.ad.Ad_array(val=val, jac=jac)
hs_1 = pp.ad.regularized_heaviside_1(xx, eps=0.2)
hs_2 = pp.ad.regularized_heaviside_1(xx, eps=0.1)
hs_3 = pp.ad.regularized_heaviside_1(xx, eps=0.01)

fig1, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(val, hs_1.val, color="orange", linewidth=2, label=r"$\varepsilon=0.2$")
ax2.plot(
    val, hs_1.jac.diagonal(), linewidth=2, color="orange", label=r"$\varepsilon=0.2$"
)

ax1.plot(val, hs_2.val, color="red", linewidth=2, label=r"$\varepsilon=0.1$")
ax2.plot(val, hs_2.jac.diagonal(), linewidth=2, color="red", label=r"$\varepsilon=0.1$")

ax1.plot(val, hs_3.val, color="green", linewidth=2, label=r"$\varepsilon=0.01$")
ax2.plot(
    val, hs_3.jac.diagonal(), linewidth=2, color="green", label=r"$\varepsilon=0.01$"
)

ax1.set_xlabel(r"$\mathrm{tr} \,\Psi_m - \Psi_L$")
ax2.set_xlabel(r"$\mathrm{tr} \,\Psi_m - \Psi_L$")

ax1.set_ylabel(r"$H_\varepsilon\left(\mathrm{tr} \,\Psi_m - \Psi_L\right)$")
ax2.set_ylabel(r"$\delta_\varepsilon\left(\mathrm{tr} \,\Psi_m - \Psi_L\right)$")

ax1.legend(
    loc="upper left", fontsize="small", numpoints=1, frameon=True, handlelength=0.5
)
ax2.legend(
    loc="upper right", fontsize="small", numpoints=1, frameon=True, handlelength=0.5
)

fig1.tight_layout()

fig1.savefig("heaviside_C&V.pdf", transparent=True)

#%% Regularized Heaviside function (Nordbotten&Both, 2021)
val = np.linspace(-0.3, 0.3, 200)
jac = sps.diags(np.ones_like(val), offsets=0)
xx = pp.ad.Ad_array(val=val, jac=jac)
hs_1 = pp.ad.regularized_heaviside_2(xx, eps=0.2)
hs_2 = pp.ad.regularized_heaviside_2(xx, eps=0.1)
hs_3 = pp.ad.regularized_heaviside_2(xx, eps=0.01)

fig2, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(val, hs_1.val, color="orange", linewidth=2, label=r"$\varepsilon=0.2$")
ax2.plot(
    val, hs_1.jac.diagonal(), linewidth=2, color="orange", label=r"$\varepsilon=0.2$"
)

ax1.plot(val, hs_2.val, color="red", linewidth=2, label=r"$\varepsilon=0.1$")
ax2.plot(val, hs_2.jac.diagonal(), linewidth=2, color="red", label=r"$\varepsilon=0.1$")

ax1.plot(val, hs_3.val, color="green", linewidth=2, label=r"$\varepsilon=0.01$")
ax2.plot(
    val, hs_3.jac.diagonal(), linewidth=2, color="green", label=r"$\varepsilon=0.01$"
)

ax1.set_xlabel(r"$\mathrm{tr} \,\Psi_m - \Psi_L$")
ax2.set_xlabel(r"$\mathrm{tr} \,\Psi_m - \Psi_L$")

ax1.set_ylabel(r"$H_\varepsilon\left(\mathrm{tr} \,\Psi_m - \Psi_L\right)$")
ax2.set_ylabel(r"$\delta_\varepsilon\left(\mathrm{tr} \,\Psi_m - \Psi_L\right)$")

ax1.legend(
    loc="upper left", fontsize="small", numpoints=1, frameon=True, handlelength=0.5
)
ax2.legend(
    loc="upper right", fontsize="small", numpoints=1, frameon=True, handlelength=0.5
)

fig2.tight_layout()

fig2.savefig("heaviside_N&B.pdf", transparent=True)
