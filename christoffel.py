import numpy as np
import harmonic_analysis_1d as ha1
import harmonic_basis as hb
import mollifiers as mol
from quadrature_S import sphere_Quadrature
import time


def compute_moment_matrix_on_sphere(
    basis_funcs,
    density,
    numvars,
    quadrature_degree,
):
    """
    Compute the moment matrix M_{ij} = ∫ φ_i(y) φ_j(y) f(y) dy over the
    sphere S^{numvars-1} using the spherical Gauss product quadrature rule,
    where {φ_i} = basis_funcs is a set of functions on the sphere and f is the
    density.

    density may be a callable f(X) -> values, or a constant (int/float).
    """
    Points, weights = sphere_Quadrature(numvars=numvars, exactness_degree=quadrature_degree)
    Points = np.asarray(Points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_points = Points.shape[0]

    # Evaluate the density at the quadrature nodes
    if callable(density):
        density_vals = np.asarray(density(Points), dtype=np.float64)
    else:
        # constant density
        density_vals = np.full(n_points, float(density), dtype=np.float64)

    # Evaluate basis (per-degree fast path) and assemble M = Φ diag(f·w) Φ^T
    PHI = hb.evaluate_basis_matrix(basis_funcs, Points)
    final_weights = density_vals * weights
    M = np.einsum('ik,jk,k->ij', PHI, PHI, final_weights, optimize=True)
    M = 0.5 * (M + M.T)   # enforce symmetry
    return np.asarray(M, dtype=np.float64)


def compute_lambda_vector_for_basis(
    numvars: int,
    max_degree: int,
    mollifier_1d, 
    n_nodes: int = None,
    method: str = "quadrature",
    tol: float = 1e-12,
    verbose: bool = False,
    return_all: bool = False
):
    """
    Computes the coefficients of h(y)=mollifier(<x,y>) over the orthonormal harmonic basis on S^{n-1}
    """
    if n_nodes is None:
        n_nodes = max_degree*4

    # Project the mollifier onto the orthonormal Gegenbauer basis
    coeffs_1d = np.asarray(
        ha1.define_projection(
            func=mollifier_1d,
            numvars=numvars,
            degree=max_degree,
            method=method,
            n_nodes=n_nodes,
            tol=tol,
            verbose =False,
        ),
        dtype=float
    )

    # Calculate the Funk–Hecke constant
    C_d = hb.funk_hecke_constant_cached(numvars)

    lambda_by_deg = C_d * coeffs_1d

    # Expand the (d+1) vector according to multiplicity of each harmonic space of degree j
    degrees_list = []
    for j in range(max_degree + 1):
        m = hb.harmonics_dimension(numvars, j)
        degrees_list.extend([j] * m)
    lambda_vector = np.array([lambda_by_deg[j] for j in degrees_list], dtype=float)

    # --- Diagnostic block ---
    if verbose:
        print(f"[compute_lambda_vector_for_basis] numvars={numvars}, max_degree={max_degree}, n_nodes={n_nodes}")
        print(f"  C_d = {C_d:.6e}")
        print(f"  coeffs_1d (first 8): {np.round(coeffs_1d[:8], 6)}")
        print(f"  lambda_by_deg (first 8): {np.round(lambda_by_deg[:8], 6)}")
        print(f"  lambda_vector (first 8): {np.round(lambda_vector[:8], 6)} ... last 8: {np.round(lambda_vector[-8:], 6)}")
        lam_min = float(np.min(lambda_vector))
        lam_max = float(np.max(lambda_vector))
        print(f"  λ_min={lam_min:.3e}, λ_max={lam_max:.3e}, ratio={lam_max / lam_min:.3e}")

    if return_all:
        return {
            "lambda_vector": lambda_vector,
            "coeffs_1d": coeffs_1d,
            "lambda_by_deg": lambda_by_deg,
            "C_d": C_d,
        }
    return lambda_vector

def mollified_christoffel_evaluator(
    X,
    moment_matrix,
    harmonic_basis,
    numvars: int,
    degree: int,
    mollifier,
    verbose: bool = False,
):
    """
    Evaluate the mollified Christoffel polynomial at a set of points X.

    Returns: gamma ndarray, shape (n_points,), 
    with gamma[j] = h_x(X[:,j])^T (M^{-1} h_x(X[:,j])) and h_x(y) = mollifier(<x,y>)
    """

    t_start = time.time()

    # Basic checks
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array with shape (n_points, numvars).")
    if X.ndim != 2:
        raise ValueError("X must be 2-D with shape (n_points, numvars).")
    n_points, n_coords = X.shape
    if n_coords != numvars:
        raise ValueError(f"X has {n_coords} coords per point but numvars={numvars} was given.")

    if not isinstance(moment_matrix, np.ndarray) or moment_matrix.ndim != 2:
        raise TypeError("moment_matrix must be a 2-D numpy array.")
    dim_basis = moment_matrix.shape[0]
    if moment_matrix.shape[0] != moment_matrix.shape[1]:
        raise ValueError("moment_matrix must be square (dim_basis x dim_basis).")

    # Evaluate the harmonic basis functions across X (per-degree fast path)
    phi_evaluated = hb.evaluate_basis_matrix(harmonic_basis, X)
    if phi_evaluated.shape[0] != dim_basis:
        raise ValueError(
            f"Basis returned {phi_evaluated.shape[0]} functions; expected {dim_basis} "
            "to match the moment matrix."
        )

    # Calculate the projection coefficients of h_x (y) = mollifier(<x,y>) for x in X
    lambda_vector = compute_lambda_vector_for_basis(
        numvars=numvars, max_degree=degree, mollifier_1d=mollifier)
    lambda_vector = np.asarray(lambda_vector, dtype=float).ravel()
    if lambda_vector.shape[0] != dim_basis:
        raise ValueError("lambda_vector length does not match basis dimension implied by moment_matrix.")

    h_x_coef = phi_evaluated * lambda_vector[:, None]   # broadcasting


    t_after_projection = time.time()
    if verbose:
        print(f"[mollified_christoffel_evaluator] projection computed in {t_after_projection - t_start:.3f}s")

    # Solve M_reg * A = h_x_coef for A (dim_basis x n_points)
    eps_diag = 1e-12
    moment_matrix_reg = moment_matrix.astype(float, copy=False) + eps_diag * np.eye(dim_basis)

    # Try Cholesky
    try:
        L = np.linalg.cholesky(moment_matrix_reg)
        # Solve L Y = h_x_coef
        Y = np.linalg.solve(L, h_x_coef)
        # gamma = h_x_coef.T L^{-1}.T L^{-1} h_x_coef = Y.T Y
        gamma = np.square(Y).sum(axis=0)
    except np.linalg.LinAlgError:
        A = np.linalg.solve(moment_matrix_reg, h_x_coef)
        gamma = np.einsum('ij,ij->j', h_x_coef, A)

    t_after_solve = time.time()
    if verbose:
        print(f"[mollified_christoffel_evaluator] linear solve done in {t_after_solve - t_after_projection:.3f}s")

    n_nonpositive = int(np.sum(gamma <= 0))
    if verbose and n_nonpositive > 0:
        print(f"[mollified_christoffel_evaluator] warning: {n_nonpositive}/{n_points} gamma_j values <= 0")

    t_end = time.time()
    if verbose:
        print(f"[mollified_christoffel_evaluator] total time {t_end - t_start:.3f}s")


    return gamma


def mcd_polynomial(
    density,
    numvars: int,
    degree: int,
    eval_points: np.ndarray,
    basis=None,
    mollifier=None,
    moment_quadrature_degree: int = None,
):
    """
    Evaluate the Mollified Christoffel-Darboux polynomial MCD_d(x, x) at
    eval_points on S^{numvars-1}.

    The moment matrix of the target measure (whose density is `density`) is
    built by quadrature, the mollified evaluation functionals are projected via
    Funk-Hecke, and MCD(x, x) = h_x^T M^{-1} h_x is returned at each x.

    density     : callable f(X) -> values, or a constant, defining the moments.
    numvars     : ambient dimension; points lie on S^{numvars-1}.
    degree      : harmonic degree d.
    eval_points : (m, numvars) points on the sphere.
    basis       : optional orthonormal harmonic basis up to `degree`; built on
                  demand if omitted.
    mollifier   : 1D mollifier g(t); default polynomial mollifier, k = floor(d^(4/3)).
    moment_quadrature_degree : exactness of the moment-matrix quadrature;
                  default 2*degree.

    Returns (poly, moment_matrix): poly has shape (m,), moment_matrix is (N, N).
    """
    if basis is None:
        basis = hb.orthonormal_harmonic_basis_up_to_degree(numvars, degree)
    if mollifier is None:
        mollifier = mol.default_mollifier(numvars=numvars, deg=degree)
    if moment_quadrature_degree is None:
        moment_quadrature_degree = 2 * degree

    eval_points = np.asarray(eval_points, dtype=np.float64)
    moment_matrix = compute_moment_matrix_on_sphere(
        basis, density, numvars, moment_quadrature_degree)
    poly = mollified_christoffel_evaluator(
        eval_points, moment_matrix, basis, numvars, degree, mollifier)
    return poly, moment_matrix


def estimate_density(
    density,
    numvars: int,
    degree: int,
    eval_points: np.ndarray,
    normalize_weights=None,
    **kwargs,
):
    """
    Mollified Christoffel-Darboux density estimate f_hat = 1 / MCD_d(x, x) at
    eval_points on S^{numvars-1}.

    High-level entry point: it builds the moment matrix and the MCD polynomial
    (see mcd_polynomial) and inverts. The pointwise norm ||phi||^2 that appears
    in the theoretical estimate f_hat = ||phi||^2 / MCD cancels under the
    self-normalization below, so it is not formed explicitly.

    If `normalize_weights` is given — quadrature weights at eval_points that cover
    the sphere — the estimate is rescaled to integrate to 1 against them.
    Extra keyword arguments (basis, mollifier, moment_quadrature_degree) are
    forwarded to mcd_polynomial.

    Returns f_hat of shape (m,).
    """
    poly, _ = mcd_polynomial(density, numvars, degree, eval_points, **kwargs)
    f_hat = 1.0 / poly
    if normalize_weights is not None:
        normalize_weights = np.asarray(normalize_weights, dtype=np.float64)
        f_hat = f_hat / np.sum(normalize_weights * f_hat)
    return f_hat


def compute_infinite_christoffel(
    quad_points: np.ndarray,
    quad_weights: np.ndarray,
    density_vals: np.ndarray,
    mollifier,
    block: int = 512,
) -> np.ndarray:
    """
    Compute the infinite-degree Christoffel function (theoretical reference)
    at each quadrature point, normalized to integrate to 1.

    This is the limit of the MCD estimator as degree goes to infinity,
    used to decompose the total error into projection and approximation parts.

    quad_points has shape (N, d), quad_weights and density_vals have shape (N,).
    mollifier is the 1D function g(t) defining the kernel K(x,y) = g(<x,y>)^2.

    Returns an array of shape (N,) normalized to integrate to 1.

    The reference is P_inf(x) = ∫ g(<x,y>)^2 / f(y) dy, whose integrand is an
    N×N matrix. It is computed in row tiles of at most `block` evaluation points
    so peak memory is O(block·N) instead of O(N^2); the per-row reduction is
    unchanged, so the result is bit-for-bit identical to forming the full matrix.
    """
    N = quad_points.shape[0]
    infinite_christoffel_poly = np.empty(N, dtype=float)
    for start in range(0, N, block):
        stop = min(start + block, N)
        # K(x, y) = g(<x,y>)^2 / f(y) for x in this tile, integrated over y
        inner_products = quad_points[start:stop] @ quad_points.T   # (b, N)
        integrand = mollifier(inner_products) ** 2 / density_vals[None, :]
        infinite_christoffel_poly[start:stop] = np.sum(
            integrand * quad_weights[None, :], axis=1)             # (b,)
    infinite_christoffel_func = 1.0 / infinite_christoffel_poly
    norm = np.sum(quad_weights * infinite_christoffel_func)
    return infinite_christoffel_func / norm