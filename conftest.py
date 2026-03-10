"""
conftest.py — shared pytest fixtures for the MCD test suite.

Fixtures are built once per session where possible to avoid
rebuilding expensive objects (bases, quadrature rules) in every test.
"""

import numpy as np
import pytest
import harmonic_basis as hb
from quadrature_S import sphere_Quadrature
import sys
import os

# Make the project root importable so pytest can find all modules
sys.path.insert(0, os.path.dirname(__file__))


# ----------------------------------------------------------------
# Small parameters used throughout the fast tests
# ----------------------------------------------------------------
NUMVARS  = 3   # points on S^2 subset R^3
DEGREE   = 5   # low degree: fast, still exercises all code paths


@pytest.fixture(scope="session")
def numvars():
    return NUMVARS


@pytest.fixture(scope="session")
def degree():
    return DEGREE


@pytest.fixture(scope="session")
def harmonic_basis(numvars, degree):
    """Orthonormal spherical harmonic basis up to DEGREE on S^{NUMVARS-1}."""
    return hb.orthonormal_harmonic_basis_up_to_degree(numvars, degree)


@pytest.fixture(scope="session")
def quad_points_weights(numvars, degree):
    """Spherical quadrature rule exact to degree 2*DEGREE."""
    pts, wts = sphere_Quadrature(numvars, 2 * degree)
    pts = np.asarray(pts, dtype=np.float64)
    wts = np.asarray(wts, dtype=np.float64)
    return pts, wts
