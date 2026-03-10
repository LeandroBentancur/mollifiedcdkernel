"""
tests/test_quadrature.py
Tests for quadrature_S: the spherical quadrature rule integrates
polynomials of the advertised degree exactly.
"""

import numpy as np
import pytest
from quadrature_S import sphere_Quadrature
import harmonic_basis as hb


def integrate_on_sphere(f, numvars, exactness_degree):
    """Integrate f over S^{numvars-1} using the quadrature rule."""
    pts, wts = sphere_Quadrature(numvars, exactness_degree)
    pts = np.asarray(pts, dtype=np.float64)
    wts = np.asarray(wts, dtype=np.float64)
    return float(np.sum(wts * f(pts)))


@pytest.mark.parametrize("numvars", [2, 3, 4])
def test_constant_integrates_to_sphere_area(numvars):
    """Integrating the constant 1 gives the surface area of S^{numvars-1}."""
    result = integrate_on_sphere(lambda X: np.ones(len(X)), numvars, exactness_degree=4)
    expected = hb.sphere_area(numvars)
    assert abs(result - expected) / expected < 1e-10, (
        f"Surface area mismatch for numvars={numvars}: got {result}, expected {expected}"
    )


@pytest.mark.parametrize("numvars", [2, 3, 4])
def test_odd_polynomial_integrates_to_zero(numvars):
    """
    Odd polynomials integrate to zero by symmetry.
    Test: integral of x_1 over S^{numvars-1} = 0.
    """
    result = integrate_on_sphere(
        lambda X: X[:, 0],
        numvars,
        exactness_degree=4
    )
    assert abs(result) < 1e-10, (
        f"Odd polynomial integral should be 0, got {result:.2e} for numvars={numvars}"
    )


@pytest.mark.parametrize("numvars", [2, 3, 4])
def test_quadratic_monomial(numvars):
    """
    Integral of x_1^2 over S^{numvars-1} = |S^{numvars-1}| / numvars
    (by symmetry: sum of integrals of x_i^2 equals integral of |x|^2 = area).
    """
    result = integrate_on_sphere(
        lambda X: X[:, 0] ** 2,
        numvars,
        exactness_degree=4
    )
    expected = hb.sphere_area(numvars) / numvars
    assert abs(result - expected) / expected < 1e-10, (
        f"Quadratic monomial integral mismatch for numvars={numvars}: "
        f"got {result:.6f}, expected {expected:.6f}"
    )
