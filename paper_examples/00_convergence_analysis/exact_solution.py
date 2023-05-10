from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import porepy as pp


class ExactSolution:
    """Parent class for the exact solution."""

    def __init__(self):
        """Constructor of the class."""

        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Fracture's aperture
        a_f = 0.1
        c_unsat = -1

        # Distance and bubble functions
        distance_fun = [
            ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5,  # bottom
            ((x - 0.5) ** 2) ** 0.5,  # middle
            ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5,  # top
        ]
        bubble_fun = (y - 0.25) ** 2 * (y - 0.75) ** 2

        # Exact hydraulic head in the rock
        psi_rock = [
            t * (distance_fun[0] ** 2.5) + c_unsat,
            t * (distance_fun[1] ** 2.5 + bubble_fun * distance_fun[1]) + c_unsat,
            t * (distance_fun[2] ** 2.5) + c_unsat,
        ]

        # Exact pressure head in the rock
        h_rock = [psi + y for psi in psi_rock]

        # Exact water content in the rock
        theta_rock = [1 / (1 - psi) for psi in psi_rock]

        # Exact relative permeability
        krw_rock = [sym.exp(psi) for psi in psi_rock]

        # Exact Darcy flux in the rock
        q_rock = [
            [-krw * sym.diff(h, x), -krw * sym.diff(h, y)]
            for (krw, h) in zip(krw_rock, h_rock)
        ]

        # Exact divergence of the flux in the rock
        div_q_rock = [sym.diff(q[0], x) + sym.diff(q[1], y) for q in q_rock]

        # Exact accumulation term in the rock
        accum_rock = [sym.diff(theta, t) for theta in theta_rock]

        # Exact source term in the rock
        f_rock = [accum + div_q for (accum, div_q) in zip(accum_rock, div_q_rock)]

        # Exact flux on the interface (mortar fluxes)
        q_intf = t * bubble_fun * sym.exp(c_unsat)

        # Exact accumulation in the fracture
        accum_frac = sym.integrate(2 * q_intf, (y, 0.25, 0.75))

        # Exact water volume in the fracture (here we solve the ODE exactly)
        """Note that accum_frac has the form c t, where c is a constant. To retrieve 
        such constant, we use a dirty trick which consists in evaluating the 
        expression with t = 1. Note that this does NOT imply that the time is 1, 
        but simply that we don't want to modify the constant c."""
        jump_coeff = accum_frac.subs({"t": 1})
        vol_frac = (jump_coeff * t**2) / 2

        # Exact hydraulic head in the fracture
        h_frac = vol_frac * a_f

        # Public attributes
        self.h_rock = h_rock
        self.psi_rock = psi_rock
        self.theta_rock = theta_rock
        self.krw_rock = krw_rock
        self.q_rock = q_rock
        self.f_rock = f_rock

        self.q_intf = q_intf

        self.accum_frac = accum_frac
        self.vol_frac = vol_frac
        self.h_frac = h_frac

        # Private attributes
        self._bubble = bubble_fun

    def rock_hydraulic_head(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact hydraulic head [m] at the cell centers.

        Parameters:
            sd_rock: Rock grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_cells, )`` containing the exact hydraulic
            heads at the cell centers at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of cell indices
        cc = sd_rock.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        h_fun = [sym.lambdify((x, y, t), h, "numpy") for h in self.h_rock]

        # Cell-centered hydraulic heads
        h_cc = np.zeros(sd_rock.num_cells)
        for (h, idx) in zip(h_fun, cell_idx):
            h_cc += h(cc[0], cc[1], time) * idx

        return h_cc

    def rock_pressure_head(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact pressure head [m] at the cell centers.

        Parameters:
            sd_rock: Rock grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_cells, )`` containing the exact pressure
            heads at the cell centers at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of cell indices
        cc = sd_rock.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        psi_fun = [sym.lambdify((x, y, t), psi, "numpy") for psi in self.psi_rock]

        # Cell-centered pressure heads
        psi_cc = np.zeros(sd_rock.num_cells)
        for (psi, idx) in zip(psi_fun, cell_idx):
            psi_cc += psi(cc[0], cc[1], time) * idx

        return psi_cc

    def rock_darcy_flux(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact rock Darcy flux [m^3 * s^-1] at the face centers .

        Parameters:
            sd_rock: Rock grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_faces, )`` containing the exact Darcy
            fluxes at the face centers at the given ``time``.

        Note:
            The returned fluxes are already scaled with ``sd_rock.face_normals``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd_rock.face_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Lambdify bubble function
        bubble_fun = sym.lambdify((y, t), self._bubble, "numpy")

        # Lambdify expression
        q_fun = [
            [
                sym.lambdify((x, y, t), q[0], "numpy"),
                sym.lambdify((x, y, t), q[1], "numpy"),
            ]
            for q in self.q_rock
        ]

        # Face-centered Darcy fluxes
        fn = sd_rock.face_normals
        q_fc = np.zeros(sd_rock.num_faces)
        for (q, idx) in zip(q_fun, face_idx):
            q_fc += (
                q[0](fc[0], fc[1], time) * fn[0] + q[1](fc[0], fc[1], time) * fn[1]
            ) * idx

        # We need to correct the values of the exact Darcy fluxes at the internal
        # boundaries since they evaluate to NaN due to a division by zero (this
        # happens because the distance function evaluates to zero on internal
        # boundaries).

        # For the correction, we exploit the fact that (rho_rock * q_rock) \dot n =
        # (rho_intf * q_intf) holds in a continuous sense. Furthermore, for our
        # problem, rho_rock = rho_intf = 1.0 at x = 0.5 and 0.25 <= y <= 0.75 . Thus,
        # the previous equality can be simplified to q_rock \dot n = q_intf on the
        # internal boundaries.

        # Here, we cannot use the face normals since we'll get wrong signs (not
        # entirely sure why). Instead, we multiply by the face area and the face sign.
        frac_faces = np.where(sd_rock.tags["fracture_faces"])[0]
        q_fc[frac_faces] = (
            bubble_fun(fc[1][frac_faces], time)
            * sd_rock.face_areas[frac_faces]
            * sd_rock.signs_and_cells_of_boundary_faces(frac_faces)[0]
        )

        return q_fc

    def rock_source(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
        """Compute exact integrated rock source.

        Parameters:
            sd_rock: Rock grid.
            time: float

        Returns:
            Array of ``shape=(sd_rock.num_cells, )`` containing the exact integrated
            sources at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of cell indices
        cc = sd_rock.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        f_fun = [sym.lambdify((x, y, t), f, "numpy") for f in self.f_rock]

        # Integrated cell-centered sources
        vol = sd_rock.cell_volumes
        f_cc = np.zeros(sd_rock.num_cells)
        for (f, idx) in zip(f_fun, cell_idx):
            f_cc += f(cc[0], cc[1], time) * vol * idx

        return f_cc

    def interface_darcy_flux(self, intf: pp.MortarGrid, time: float) -> np.ndarray:
        """Compute exact mortar fluxes at the interface.

        Parameters:
            intf: Mortar grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(intf.num_cells, )`` containing the exact mortar fluxes
            at the given ``time``.

        Note:
            The returned mortar fluxes are already scaled with ``intf.cell_volumes``.

        """
        # Symbolic variable
        y, t = sym.symbols("y t")

        # Cell centers and volumes
        cc = intf.cell_centers
        vol = intf.cell_volumes

        # Lambdify expression
        lmbda_fun = sym.lambdify((y, t), self.q_intf, "numpy")

        # Evaluate and integrate
        lmbda_cc = lmbda_fun(cc[1], time) * vol

        return lmbda_cc

    def fracture_volume(self, time: float) -> float:
        """Evaluates exact fracture volume at a given time"""
        t = sym.symbols("t")
        vol_fun = sym.lambdify(t, self.vol_frac, "numpy")
        return float(vol_fun(time))

    def fracture_hydraulic_head(self, time: float) -> float:
        """Evaluates exact fracture volume at a given time"""
        t = sym.symbols("t")
        h_fun = sym.lambdify(t, self.h_frac, "numpy")
        return float(h_fun(time))

    # -----> Others
    def rock_boundary_hydraulic_head(self, sd_rock: pp.Grid, time: float) -> np.ndarray:
        """Exact hydraulic head at the boundary faces.

        Parameters:
            sd_rock: Rock grid.
            time: time in seconds.

        Returns:
            Array of ``shape=(sd_rock.num_faces, )`` with the exact pressure values
            on the exterior boundary faces at the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get list of face indices
        fc = sd_rock.face_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Boundary faces
        bc_faces = sd_rock.get_boundary_faces()

        # Lambdify expression
        h_fun = [sym.lambdify((x, y, t), h, "numpy") for h in self.h_rock]

        # Boundary pressures
        h_bf = np.zeros(sd_rock.num_faces)
        for (h, idx) in zip(h_fun, face_idx):
            h_bf[bc_faces] += h(fc[0], fc[1], time)[bc_faces] * idx[bc_faces]

        return h_bf


#%% Runner
ex = ExactSolution()
