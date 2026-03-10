"""
tests/test_harmonic_basis.py
Tests for harmonic_basis: orthonormality of the spherical harmonic basis.
"""

import numpy as np
import pytest
import harmonic_basis as hb

@pytest.mark.parametrize("numvars,max_degree", [
    (3, 5),
    (3, 8),
    (4, 4),
])
def test_basis_orthonormality(numvars, max_degree):
    """Gram matrix of the orthonormal basis equals the identity."""
    from quadrature_S import sphere_Quadrature
    basis    = hb.orthonormal_harmonic_basis_up_to_degree(numvars, max_degree)
    pts, wts = sphere_Quadrature(numvars, 2 * max_degree + 1)
    pts = np.asarray(pts, dtype=np.float64)
    wts = np.asarray(wts, dtype=np.float64).flatten()
    V   = np.array([f(pts) for f in basis])
    G   = V @ (wts[:, None] * V.T)
    sup_offdiag = float(np.max(np.abs(G - np.diag(np.diag(G)))))
    sup_diag    = float(np.max(np.abs(np.diag(G) - 1.0)))
    assert sup_offdiag < 1e-6, f"Off-diagonal error {sup_offdiag:.2e}"
    assert sup_diag    < 1e-6, f"Diagonal error {sup_diag:.2e}"


def test_harmonics_dimension():
    """Dimension formula matches known values."""
    # S^1 (circle): dim = 1 for degree 0, 2 for all other degrees
    assert hb.harmonics_dimension(2, 0) == 1
    assert hb.harmonics_dimension(2, 1) == 2
    assert hb.harmonics_dimension(2, 5) == 2

    # S^2 (standard sphere): dim = 2*d + 1
    for d in range(6):
        assert hb.harmonics_dimension(3, d) == 2 * d + 1, (
            f"harmonics_dimension(3, {d}) should be {2*d+1}"
        )


def test_sphere_area():
    """Sphere area formula matches known values."""
    # S^1: circumference = 2*pi
    assert abs(hb.sphere_area(2) - 2 * np.pi) < 1e-12
    # S^2: surface area = 4*pi
    assert abs(hb.sphere_area(3) - 4 * np.pi) < 1e-12
