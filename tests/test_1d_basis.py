"""
tests/test_1d_basis.py
Tests for harmonic_analysis_1d: Gegenbauer basis orthonormality and projection.
"""

import numpy as np
import pytest
import harmonic_analysis_1d as ha1


@pytest.mark.parametrize("numvars,degree", [
    (3, 5),
    (3, 10),
    (4, 5),
])
def test_gegenbauer_basis_orthonormality(numvars, degree):
    """Orthonormal Gegenbauer basis satisfies <phi_i, phi_j> = delta_ij."""
    from scipy.special import roots_gegenbauer
    alpha = (numvars - 2) / 2.0
    n_nodes = max(4 * (degree + 1), 20)
    if abs(alpha) < 1e-14:
        i = np.arange(1, n_nodes + 1)
        nodes   = np.cos((2*i - 1) * np.pi / (2*n_nodes))
        weights = np.full(n_nodes, np.pi / n_nodes)
    else:
        nodes, weights = roots_gegenbauer(n_nodes, alpha)
    basis = ha1.generate_gegenbauer_basis(numvars=numvars, degree=degree)
    phivals = np.vstack([phi(nodes) for phi in basis])
    gram = (phivals * weights[None, :]) @ phivals.T
    err = float(np.max(np.abs(gram - np.eye(degree + 1))))
    assert err < 1e-10, f"Gram error {err:.2e} for numvars={numvars}, degree={degree}"


@pytest.mark.parametrize("numvars,degree", [
    (3, 5),
    (3, 10),
    (4, 5),
])
def test_gegenbauer_basis_evaluator(numvars, degree):
    """Vectorized evaluator produces the same orthonormal rows as the basis list."""
    from scipy.special import roots_gegenbauer
    alpha = (numvars - 2) / 2.0
    n_nodes = max(4 * (degree + 1), 20)
    if abs(alpha) < 1e-14:
        i = np.arange(1, n_nodes + 1)
        nodes   = np.cos((2*i - 1) * np.pi / (2*n_nodes))
        weights = np.full(n_nodes, np.pi / n_nodes)
    else:
        nodes, weights = roots_gegenbauer(n_nodes, alpha)
    evaluator = ha1.generate_gegenbauer_basis_evaluator(numvars, degree)
    phivals = evaluator(nodes)
    assert phivals.shape == (degree + 1, nodes.size)
    gram = (phivals * weights[None, :]) @ phivals.T
    err = float(np.max(np.abs(gram - np.eye(degree + 1))))
    assert err < 1e-10, f"Evaluator Gram error {err:.2e} for numvars={numvars}, degree={degree}"


@pytest.mark.parametrize("numvars,degree", [
    (3, 5),
    (3, 10),
    (4, 5),
])
def test_projection_reconstruction(numvars, degree):
    """Projecting a polynomial onto the basis recovers its coefficients exactly."""
    rng = np.random.default_rng(42)
    basis    = ha1.generate_gegenbauer_basis(numvars=numvars, degree=degree)
    c_true   = rng.normal(size=(degree + 1,))
    def f(t):
        return sum(c * phi(t) for c, phi in zip(c_true, basis))
    a_proj = np.asarray(
        ha1.define_projection(f, numvars=numvars, degree=degree, method="quadrature"),
        dtype=float)
    err = float(np.max(np.abs(a_proj - c_true)))
    assert err < 1e-8, f"Coefficient error {err:.2e} for numvars={numvars}, degree={degree}"