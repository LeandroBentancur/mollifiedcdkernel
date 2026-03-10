"""
tests/test_mollifiers.py
Tests for mollifiers: each mollifier should be non-negative on [-1,1]
and the default degree schedule should be consistent.
"""

import numpy as np
import pytest
import mollifiers as mol


T_EVAL = np.linspace(-1.0, 1.0, 500)


@pytest.mark.parametrize("numvars,k", [
    (3, 5),
    (3, 10),
    (4, 5),
])
def test_polynomial_mollifier_nonnegative(numvars, k):
    """Polynomial mollifier must be non-negative on [-1, 1]."""
    g = mol.polynomial_mollifier(numvars=numvars, k=k)
    vals = g(T_EVAL)
    assert np.all(vals >= -1e-12), (
        f"polynomial_mollifier has negative values for numvars={numvars}, k={k}: "
        f"min={vals.min():.2e}"
    )


@pytest.mark.parametrize("numvars,degree", [
    (3, 5),
    (3, 10),
    (4, 5),
])
def test_gegenbauer_mollifier_nonnegative(numvars, degree):
    """Gegenbauer mollifier must be non-negative on [-1, 1]."""
    g = mol.define_gegenbauer_mollifier(numvars=numvars, degree=degree)
    vals = g(T_EVAL)
    assert np.all(vals >= -1e-12), (
        f"gegenbauer_mollifier has negative values for numvars={numvars}, degree={degree}: "
        f"min={vals.min():.2e}"
    )


@pytest.mark.parametrize("deg,expected_k", [
    (5,  int(5  ** (4/3))),
    (10, int(10 ** (4/3))),
    (20, int(20 ** (4/3))),
    (30, int(30 ** (4/3))),
])
def test_default_mollifier_degree_schedule(deg, expected_k):
    """Default mollifier degree schedule matches floor(deg^(4/3))."""
    assert mol.default_mollifier_degree(deg) == expected_k, (
        f"default_mollifier_degree({deg}) = {mol.default_mollifier_degree(deg)}, "
        f"expected {expected_k}"
    )


def test_default_mollifier_returns_callable():
    """default_mollifier returns a callable that accepts arrays."""
    g = mol.default_mollifier(numvars=3, deg=10)
    assert callable(g)
    vals = g(T_EVAL)
    assert vals.shape == T_EVAL.shape
