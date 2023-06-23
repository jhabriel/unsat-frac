"""Module containing a light-weighted functional test for the manufactured solution."""

import numpy as np
import pytest
import sys
sys.path.append("..")

from paper_examples.convergence_analysis.model_with_non_zero_psi_l import manufactured_model


@pytest.fixture(scope="module")
def known_errors() -> dict[str, float]:
    """Set known error for the simulation."""
    return {
        'error_h_bulk': 0.0006247275150745988,
        'error_q_bulk': 0.002857856303142884,
        'error_q_intf': 0.0007655866736162515,
        'error_h_frac': 0.0021671576743287793,
        'error_vol_frac': 0.0021671576743287958
    }


@pytest.fixture(scope="module")
def actual_errors() -> dict[str, float]:
    """Run simulation and retrieve the actual errors."""
    return manufactured_model(mesh_size=0.1)


@pytest.mark.parametrize("var", ["h_bulk", "q_bulk", "q_intf", "h_frac", "vol_frac"])
def test_errors(var, known_errors, actual_errors):
    np.testing.assert_allclose(
        actual=actual_errors["error_" + f"{var}"],
        desired=known_errors["error_" + f"{var}"],
        atol=1e-8,
        rtol=1e-4,
    )

