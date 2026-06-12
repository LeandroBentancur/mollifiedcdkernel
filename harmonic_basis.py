import math
from functools import lru_cache
from typing import Callable, List, Optional
import numpy as np
import scipy.special as sc
import harmonic_analysis_1d as ha1
from quadrature_S import sphere_Quadrature


@lru_cache(maxsize=32)
def funk_hecke_constant_cached(numvars: int) -> float:
    """
    Funk-Hecke multiplicative constant used when lifting a 1D integral against
    the Gegenbauer weight to the sphere S^{numvars-1}.

    It is exactly the surface area of the equatorial sphere,
    |S^{numvars-2}| = sphere_area(numvars - 1). Cached by `numvars`.
    """
    if numvars < 2:
        raise ValueError("numvars must be >= 2")
    return sphere_area(numvars - 1)


@lru_cache(maxsize=32)
def sphere_area(numvars: int) -> float:
    """
    Surface area of the unit sphere S^{numvars-1} embedded in R^{numvars}.

    Returns |S^{n-1}| = 2 * pi^{n/2} / Gamma(n/2).

    Uses the log-gamma formula for numerical stability.
    Cached by `numvars`.
    """
    if numvars < 1:
        raise ValueError("numvars must be >= 1")
    log_area = np.log(2.0) + (numvars / 2.0) * np.log(np.pi) - sc.gammaln(numvars / 2.0)
    return float(np.exp(log_area))


def zonal_func_centered_at_y(
    degree: int,
    X: np.ndarray,
    y: np.ndarray,
    gegenbauer_basis: Optional[List[Callable]] = None,
    C_dim: Optional[float] = None,
) -> np.ndarray:
    """
    Return the zonal function of `degree` centered at `y`, evaluated at nodes `X`.
    The result is normalized to have unit L2 norm on the sphere.

    X has shape (Q, n), each row a point on the sphere. y has shape (n,).
    gegenbauer_basis and C_dim can be passed precomputed to avoid redundant work;
    if omitted they are computed on the fly (C_dim is cached by numvars).

    Returns an array of shape (Q,).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (Q, numvars)")
    Q, n = X.shape
    if y.shape != (n,):
        raise ValueError("y must be a 1D array of length numvars")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    t = X @ y  # inner products, shape (Q,)

    if degree == 0:
        # normalized constant function on S^{n-1}
        return np.ones(Q) / np.sqrt(sphere_area(n))

    # degrees > 0
    if gegenbauer_basis is None:
        gegenbauer_basis = ha1.generate_gegenbauer_basis(n, degree)
    basis_func = gegenbauer_basis[degree]

    if C_dim is None:
        C_dim = funk_hecke_constant_cached(n)

    phi_vals = basis_func(t)  # basis callable accepts vector t
    return phi_vals / np.sqrt(C_dim)


def harmonics_dimension(numvars: int, degree: int) -> int:
    """
    Dimension of homogeneous spherical harmonics of a given degree
    in `numvars` variables (i.e. on S^{numvars-1}).

    This is the standard combinatorial formula:
      dim = C(numvars+degree-1, degree) - C(numvars+degree-3, degree-2)
    with the usual conventions for binomial coefficients.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if numvars < 2:
        raise ValueError("numvars must be at least 2")

    def comb(n: int, k: int) -> int:
        return math.comb(n, k) if 0 <= k <= n else 0

    if degree == 0:
        return 1

    return comb(numvars + degree - 1, degree) - comb(numvars + degree - 3, degree - 2)


def evaluate_zonals_on_centers(
    basis_func: Callable,
    centers: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Vectorized evaluation of the zonal basis function `basis_func` at
    all pairs (center, X_eval). Returns an array of shape (num_centers, Qe)
    where Qe = X_eval.shape[0].
    """
    centers = np.asarray(centers)
    X_eval = np.asarray(X_eval)
    if centers.ndim != 2 or X_eval.ndim != 2:
        raise ValueError("centers and X_eval must both be 2D arrays")
    # Compute inner products: (Qe, num_centers)
    T = X_eval @ centers.T
    # Evaluate basis function on flattened entries -> reshape back
    T_flat = T.ravel()
    phi_flat = basis_func(T_flat)
    phi = phi_flat.reshape(T.shape).T  # now (num_centers, Qe)
    return phi


class _DegreeBlock:
    """Evaluation data for all harmonics of a single degree.

    All harmonics of one degree are linear combinations of the SAME zonal
    functions (same centers and Gegenbauer polynomial), differing only by the
    coefficient rows. Storing them together lets us evaluate the shared zonal
    matrix once per degree instead of once per harmonic.

    For degree 0 the block is the single constant function and `area_sphere`
    is set; for degree > 0, `basis_func`, `centers`, `coeffs`, `C_dim` are set.
    """

    def __init__(self, dim, *, area_sphere=None,
                 basis_func=None, centers=None, coeffs=None, C_dim=None):
        self.dim = dim
        self.area_sphere = area_sphere
        self.basis_func = basis_func
        self.centers = centers
        self.coeffs = coeffs
        self.C_dim = C_dim

    def evaluate(self, X):
        """Evaluate the degree's harmonics at points X -> (dim, Q)."""
        Q = X.shape[0]
        if self.area_sphere is not None:
            return np.ones((1, Q), dtype=float) / np.sqrt(self.area_sphere)
        # Shared zonal matrix, computed ONCE for the whole degree.
        phi = evaluate_zonals_on_centers(self.basis_func, self.centers, X)
        phi = phi / np.sqrt(self.C_dim)                       # (dim, Q)
        out = np.empty((self.dim, Q), dtype=float)
        # Per-harmonic combination, identical arithmetic to the callables.
        for i in range(self.dim):
            out[i, :] = self.coeffs[i] @ phi
        return out


class HarmonicBasis:
    """An orthonormal spherical-harmonic basis, stored by degree.

    All harmonics of one degree are linear combinations of the same zonal
    functions, so they are grouped into `_DegreeBlock` objects. The basis is
    evaluated through `evaluate(X)` (or `evaluate_basis_matrix`), which assembles
    the full (dim, Q) value matrix one degree at a time. `len()` returns the
    basis dimension.
    """

    def __init__(self, blocks):
        self._blocks = list(blocks)
        self.dim = sum(block.dim for block in self._blocks)

    def __len__(self):
        return self.dim

    def evaluate(self, X):
        """Evaluate every basis function at points X. Returns (dim, Q)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Q = X.shape[0]
        out = np.empty((self.dim, Q), dtype=float)
        row = 0
        for block in self._blocks:
            out[row:row + block.dim, :] = block.evaluate(X)
            row += block.dim
        return out


def evaluate_basis_matrix(basis, X):
    """Evaluate a basis on points X, returning a (dim, Q) matrix.

    Uses the fast per-degree path when `basis` is a HarmonicBasis; otherwise
    falls back to looping over a plain iterable of callables.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if isinstance(basis, HarmonicBasis):
        return basis.evaluate(X)
    Q = X.shape[0]
    out = np.empty((len(basis), Q), dtype=float)
    for i, f in enumerate(basis):
        out[i, :] = np.asarray(f(X), dtype=float).ravel()
    return out


def cholesky_with_jitter(A: np.ndarray, max_attempts: int = 6, initial_jitter: float = 0.0) -> np.ndarray:
    """
    Attempt Cholesky decomposition of A, increasing jitter on the diagonal until success
    or until attempts exhausted. Returns lower-triangular L such that A + jitter*I = L L^T.
    Raises RuntimeError if decomposition fails.
    """
    jitter = initial_jitter
    for attempt in range(max_attempts):
        try:
            L = np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))
            return L
        except np.linalg.LinAlgError:
            jitter = jitter if jitter > 0 else 1e-12
            jitter *= 10.0
    raise RuntimeError("Cholesky decomposition failed even after adding jitter")


def orthonormal_harmonic_basis_numerical(
    numvars: int,
    degree: int,
    rng_seed: int = 42,
    verbose: bool = False,
) -> "HarmonicBasis":
    """
    Generate a numerical orthonormal basis of spherical harmonics of exact `degree`
    on S^{numvars-1}. Returns a HarmonicBasis; evaluate it with evaluate_basis_matrix
    (or basis.evaluate(X)), which returns a (dim, Q) value matrix.

    Implementation notes:
    - degree == 0 returns the exact normalized constant function.
    - degree > 0 builds linear combinations of random zonals, orthonormalized
      by a discrete Gram matrix on quadrature nodes.
    - The basis is deterministic for a fixed rng_seed (default 42). All
      experiments in the paper use the default seed.
    """
    dim = harmonics_dimension(numvars, degree)
    if dim == 0:
        return HarmonicBasis([])

    # degree == 0: exact normalized constant function
    if degree == 0:
        block = _DegreeBlock(1, area_sphere=sphere_area(numvars))
        return HarmonicBasis([block])

    # degree > 0: numerical construction
    quad_deg = max(2 * degree, 2)
    quad_points, quad_weights = sphere_Quadrature(numvars, quad_deg)
    quad_points = np.asarray(quad_points)
    quad_weights = np.asarray(quad_weights).flatten()

    if quad_points.ndim != 2 or quad_points.shape[1] != numvars:
        raise RuntimeError("sphere_Quadrature must return nodes with shape (Q, numvars)")

    Q = quad_points.shape[0]
    if quad_weights.shape[0] != Q:
        raise RuntimeError("Number of weights must match number of nodes Q")

    # RNG and basis
    rng = np.random.default_rng(rng_seed)
    gegenbauer_basis = ha1.generate_gegenbauer_basis(numvars, degree)
    basis_func = gegenbauer_basis[degree]
    C_dim = funk_hecke_constant_cached(numvars)

    # Generate random zonal evaluations; collect in list to avoid repeated vstack
    vals_list = []
    centers_list = []
    max_tries = dim * 6
    tries = 0
    while len(vals_list) < dim and tries < max_tries:
        tries += 1
        y = rng.normal(size=numvars)
        norm_y = np.linalg.norm(y)
        if norm_y == 0:
            continue
        y /= norm_y
        vals = zonal_func_centered_at_y(degree, quad_points, y, gegenbauer_basis, C_dim)
        if not np.all(np.isfinite(vals)):
            continue
        vals_list.append(vals)
        centers_list.append(y)

    if len(vals_list) < dim:
        raise RuntimeError(f"Insufficient zonal functions generated (got {len(vals_list)} / {dim})")

    # Build V matrix: shape (dim, Q)
    V = np.vstack(vals_list)  # (dim, Q)
    W = quad_weights
    sqrtW = np.sqrt(W)

    # Build Gram matrix efficiently: (V * sqrtW) @ (V * sqrtW).T
    V_weighted = V * sqrtW[None, :]
    B = V_weighted @ V_weighted.T

    # Cholesky with jitter for stability
    L = cholesky_with_jitter(B)

    # Solve for coefficients: L_inv @ I  (we want L^{-1})
    L_inv = np.linalg.solve(L, np.eye(L.shape[0]))
    coeffs = L_inv  # each row contains coefficients for one orthonormal basis function

    # Group all harmonics of this degree into one block for fast evaluation.
    centers_arr = np.vstack(centers_list)  # shape (dim, numvars)
    block = _DegreeBlock(dim, basis_func=basis_func, centers=centers_arr,
                         coeffs=coeffs, C_dim=C_dim)

    if verbose:
        condB = np.linalg.cond(B)
        print(f"Generated orthonormal basis degree={degree}, dim={dim}, cond(B)={condB:.2e}, tries={tries}")

    return HarmonicBasis([block])


def orthonormal_harmonic_basis_up_to_degree(
    numvars: int,
    max_degree: int,
    rng_seed: int = 42,
    verbose: bool = False,
) -> "HarmonicBasis":
    """
    Generate an orthonormal basis of spherical harmonics of all degrees <= max_degree
    by concatenating the per-degree blocks.
    """
    all_blocks = []
    for d in range(max_degree + 1):
        basis_d = orthonormal_harmonic_basis_numerical(
            numvars=numvars,
            degree=d,
            rng_seed=rng_seed,
            verbose=verbose,
        )
        all_blocks.extend(basis_d._blocks)
    return HarmonicBasis(all_blocks)


def check_basis_orthonormality(
    basis,
    numvars: int,
    max_degree: int,
    tol: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """
    Verify orthonormality of a spherical harmonic basis via quadrature.
    Returns a dict with L1 and sup errors for diagonal and off-diagonal
    entries of the Gram matrix.
    Raises AssertionError if errors exceed tol.
    """
    dim_basis = len(basis)
    quad_deg = 2 * max_degree + 1
    quad_nodes, quad_weights = sphere_Quadrature(numvars, quad_deg)
    quad_nodes   = np.asarray(quad_nodes,   dtype=np.float64)
    quad_weights = np.asarray(quad_weights, dtype=np.float64).flatten()
    quad_size = len(quad_nodes)

    vals = evaluate_basis_matrix(basis, quad_nodes)

    G = vals @ (quad_weights[:, None] * vals.T)
    L1_offdiag  = float(np.sum(np.abs(G - np.diag(np.diag(G)))))
    sup_offdiag = float(np.max(np.abs(G - np.diag(np.diag(G)))))
    L1_diag     = float(np.sum(np.abs(np.diag(G) - 1.0)))
    sup_diag    = float(np.max(np.abs(np.diag(G) - 1.0)))

    if verbose:
        print(f"[check_basis_orthonormality] numvars={numvars}, max_degree={max_degree}, dim={dim_basis}")
        print(f"  L1 off-diagonal={L1_offdiag:.3e}, sup off-diagonal={sup_offdiag:.3e}")
        print(f"  L1 diagonal={L1_diag:.3e},     sup diagonal={sup_diag:.3e}")

    assert L1_offdiag < tol, (
        f"Basis not orthogonal: L1 off-diagonal error = {L1_offdiag:.3e}")
    for i, val in enumerate(np.diag(G)):
        assert abs(val - 1.0) < tol, (
            f"Basis function {i} not normalized: norm^2 = {val:.3e}")

    return {
        "L1_error_offdiag":  L1_offdiag,
        "sup_error_offdiag": sup_offdiag,
        "L1_error_diag":     L1_diag,
        "sup_error_diag":    sup_diag,
    }