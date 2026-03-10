"""
tests/test_christoffel_core.py
Tests for christoffel.py: moment matrix, lambda vector, and the MCD estimator.
"""

import numpy as np
import pytest
import harmonic_basis as hb
import christoffel as ch
import mollifiers as mol
import densities as dens
from quadrature_S import sphere_Quadrature


# ----------------------------------------------------------------
# Moment matrix
# ----------------------------------------------------------------

def test_moment_matrix_is_identity_for_constant_density():
    """
    For constant density f=1 the moment matrix in an orthonormal basis
    equals the identity, since the basis satisfies ∫ φ_i φ_j dy = δ_ij
    with respect to the surface measure dy.
    """
    numvars, degree = 3, 5
    basis = hb.orthonormal_harmonic_basis_up_to_degree(numvars, degree)

    M = ch.compute_moment_matrix_on_sphere(
        basis_funcs=basis,
        density=1.0,          # constant density = 1
        numvars=numvars,
        method="quadrature",
        quadrature_degree=2 * degree + 1,
    )
    err = np.max(np.abs(M - np.eye(len(basis))))
    assert err < 1e-6, f"Moment matrix deviates from identity by {err:.2e}"


def test_moment_matrix_is_symmetric():
    """Moment matrix must always be symmetric."""
    numvars, degree = 3, 4
    basis = hb.orthonormal_harmonic_basis_up_to_degree(numvars, degree)

    M = ch.compute_moment_matrix_on_sphere(
        basis_funcs=basis,
        density=lambda X: dens.von_mises_fisher_density(X, numvars=numvars, kappa=3.0),
        numvars=numvars,
        method="quadrature",
        quadrature_degree=2 * degree,
    )
    assert np.max(np.abs(M - M.T)) < 1e-12, "Moment matrix is not symmetric"


# ----------------------------------------------------------------
# Lambda vector
# ----------------------------------------------------------------

def test_lambda_vector_length_matches_basis():
    """lambda_vector must have the same length as the harmonic basis."""
    numvars, degree = 3, 5
    mollifier = mol.default_mollifier(numvars=numvars, deg=degree)
    basis = hb.orthonormal_harmonic_basis_up_to_degree(numvars, degree)

    lam = ch.compute_lambda_vector_for_basis(
        numvars=numvars, max_degree=degree, mollifier_1d=mollifier)

    assert len(lam) == len(basis), (
        f"lambda_vector length {len(lam)} != basis length {len(basis)}"
    )


# ----------------------------------------------------------------
# MCD estimator
# ----------------------------------------------------------------

def test_mcd_estimator_returns_positive_values():
    """
    The MCD polynomial (before inversion) must be positive at all
    quadrature points for a reasonable density.
    """
    numvars, degree = 3, 5
    basis     = hb.orthonormal_harmonic_basis_up_to_degree(numvars, degree)
    mollifier = mol.default_mollifier(numvars=numvars, deg=degree)
    density   = lambda X: dens.von_mises_fisher_density(X, numvars=numvars, kappa=3.0)

    pts, wts = sphere_Quadrature(numvars, 2 * degree)
    pts = np.asarray(pts, dtype=np.float64)

    M = ch.compute_moment_matrix_on_sphere(
        basis, density, numvars=numvars,
        method="quadrature", quadrature_degree=2 * degree)

    gamma = ch.mollified_christoffel_evaluator(
        pts, M, basis, numvars, degree, mollifier)

    assert np.all(gamma > 0), (
        f"{np.sum(gamma <= 0)} / {len(gamma)} MCD values are non-positive"
    )


def test_mcd_estimate_integrates_to_one():
    """
    The density estimate 1/MCD, normalized, must integrate to 1
    under the quadrature rule.
    """
    numvars, degree = 3, 5
    basis     = hb.orthonormal_harmonic_basis_up_to_degree(numvars, degree)
    mollifier = mol.default_mollifier(numvars=numvars, deg=degree)
    density   = lambda X: dens.von_mises_fisher_density(X, numvars=numvars, kappa=3.0)

    pts, wts = sphere_Quadrature(numvars, 2 * degree)
    pts = np.asarray(pts, dtype=np.float64)
    wts = np.asarray(wts, dtype=np.float64)

    M = ch.compute_moment_matrix_on_sphere(
        basis, density, numvars=numvars,
        method="quadrature", quadrature_degree=2 * degree)

    gamma    = ch.mollified_christoffel_evaluator(
        pts, M, basis, numvars, degree, mollifier)
    estimate = 1.0 / gamma
    mass     = float(np.sum(wts * estimate / np.sum(wts * estimate)))

    assert abs(mass - 1.0) < 1e-10, f"Normalized estimate integrates to {mass:.6f}, expected 1"
