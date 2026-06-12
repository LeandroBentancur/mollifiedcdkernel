"""
error_decomposition.py — shared error-decomposition pipeline for the
Mollified Christoffel-Darboux density estimator on the sphere.

Both experiment scripts (main_christoffel.py, run_christoffel_error_charts.py)
call decompose_errors, so the estimator pipeline and the L2 convention live in a
single place and cannot drift apart.

The total error splits as
    estimate - f  =  (estimate - infinite_reference) + (infinite_reference - f)
       total_err          proj_err                          approx_err
(see docs/mathematical_context.md, section 6).
"""

from dataclasses import dataclass
import numpy as np

import mollifiers as mol
import christoffel as ch
from quadrature_S import sphere_Quadrature


def l2_error(values, weights):
    """L2(lambda) norm on the sphere: sqrt(sum_k w_k e_k^2 / sum_k w_k).

    The quadrature weights sum to the surface area omega_{n-1}, so dividing by
    their sum converts to the normalized probability measure lambda (see
    docs/implementation_notes.md, Sphere convention 1). The same definition is
    used for all three error curves.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


@dataclass
class ErrorDecomposition:
    """Result of decompose_errors at a single harmonic degree."""
    quad_points: np.ndarray
    quad_weights: np.ndarray
    f_vals: np.ndarray                 # normalized true density at quad points
    christoffel_approx: np.ndarray     # normalized MCD density estimate
    infinite_approx: np.ndarray        # infinite-degree reference
    moment_matrix: np.ndarray
    total_err: np.ndarray
    proj_err: np.ndarray
    approx_err: np.ndarray
    total_L2: float
    proj_L2: float
    approx_L2: float
    total_sup: float
    proj_sup: float
    approx_sup: float


def decompose_errors(basis, f_callable, numvars, deg, mollifier=None, quad=None):
    """Run the MCD estimator at degree deg and split the total error into its
    projection and approximation parts.

    Parameters
    basis      : orthonormal spherical-harmonic basis up to degree deg.
    f_callable : density, f(X) -> values at points X of shape (n_points, numvars).
    numvars    : ambient dimension (points lie on S^{numvars-1}).
    deg        : harmonic degree.
    mollifier  : 1D mollifier g(t); default is the polynomial mollifier with
                 k = floor(deg^(4/3)) (paper default).
    quad       : optional (points, weights); default Gauss product rule exact to
                 degree 2*deg.

    Returns an ErrorDecomposition.
    """
    if mollifier is None:
        mollifier = mol.default_mollifier(numvars=numvars, deg=deg)

    if quad is None:
        quad_points, quad_weights = sphere_Quadrature(numvars, 2 * deg)
    else:
        quad_points, quad_weights = quad
    quad_points = np.asarray(quad_points, dtype=np.float64)
    quad_weights = np.asarray(quad_weights, dtype=np.float64)

    # Normalized true density (with respect to the probability measure lambda)
    f_vals = np.asarray(f_callable(quad_points), dtype=float)
    f_vals /= np.sum(quad_weights * f_vals)

    # MCD polynomial and moment matrix at degree deg, evaluated at the quad nodes
    christoffel_poly, moment_matrix = ch.mcd_polynomial(
        f_callable, numvars, deg, quad_points,
        basis=basis, mollifier=mollifier, moment_quadrature_degree=2 * deg)

    # MCD density estimate, self-normalized to integrate to 1
    christoffel_func = 1.0 / christoffel_poly
    christoffel_approx = christoffel_func / np.sum(quad_weights * christoffel_func)

    # Infinite-degree reference (limit of the estimate as deg -> infinity)
    infinite_approx = ch.compute_infinite_christoffel(
        quad_points, quad_weights, f_vals, mollifier)

    # Error decomposition
    total_err = christoffel_approx - f_vals
    proj_err = christoffel_approx - infinite_approx
    approx_err = infinite_approx - f_vals

    return ErrorDecomposition(
        quad_points=quad_points, quad_weights=quad_weights, f_vals=f_vals,
        christoffel_approx=christoffel_approx, infinite_approx=infinite_approx,
        moment_matrix=moment_matrix,
        total_err=total_err, proj_err=proj_err, approx_err=approx_err,
        total_L2=l2_error(total_err, quad_weights),
        proj_L2=l2_error(proj_err, quad_weights),
        approx_L2=l2_error(approx_err, quad_weights),
        total_sup=float(np.max(np.abs(total_err))),
        proj_sup=float(np.max(np.abs(proj_err))),
        approx_sup=float(np.max(np.abs(approx_err))),
    )
