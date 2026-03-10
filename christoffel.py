import numpy as np
import sympy as sp
import harmonic_analysis_1d as ha1
import harmonic_basis as hb
from quadrature_S import sphere_Quadrature
from scipy.special import roots_gegenbauer
import time

def sample_uniform_sphere(numvars, n_points, seed=None):
    """
    Generate points uniformly distributed on the sphere S^{numvars-1}.

    Parameters:
    numvars : ambient dimension (points lie in R^{numvars} on S^{numvars-1})
    n_points : number of points to generate.
    seed : optional 

    Returns:
    X : ndarray (n_points, numvars) uniform points on the sphere.
    """

    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_points, numvars))

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms

    return X

def compute_moment_matrix_on_sphere(
    basis_funcs,
    density,
    numvars,
    method="montecarlo", # admits 'montecarlo' and 'quadrature'
    N_MC:int=None,
    quadrature_degree=None,
    seed:int= 42
):
    """
    Compute the moment matrix M_{ij} = ∫ φ_i(y) φ_j(y) f(y) dy
    over the sphere S^{numvars-1}, using Monte-Carlo quadrature,
    where {φ_i} is basis_funcs a set of functions over the sphere.
    """
    
    basis_dim = len(basis_funcs)

    # Build points and weights
    if method == "montecarlo":
        Points = sample_uniform_sphere(numvars = numvars, n_points=N_MC, seed= seed)
        # Surface area of S^{numvars-1}
        area = hb.sphere_area(numvars)
        # Vector of size of Points and uniform weights
        weights = np.full(N_MC, float(area / N_MC), dtype=np.float64)
    elif method == "quadrature":
        if quadrature_degree is None:
            raise ValueError("quadrature_degree must be provided when method='quadrature'.")
        Points, weights = sphere_Quadrature(numvars=numvars, exactness_degree=quadrature_degree)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'montecarlo' or 'quadrature.")
    Points = np.asarray(Points, dtype=np.float64)
    n_points = Points.shape[0]

    # Evaluate the density
    # It takes into account three different types of functions
    if callable(density):
        # numpy function
        density_vals = np.asarray(density(Points), dtype=np.float64)
    elif isinstance(density, (int, float, np.floating)):
        # constant
        density_vals = np.full(n_points, float(density), dtype=np.float64)
    else:
        # Symbolic expression
        Y_syms = sp.symbols([f"y{i+1}" for i in range(numvars)])
        density_np = sp.lambdify(Y_syms, density, "numpy")
        density_vals = np.array([density_np(*pt) for pt in Points], dtype=np.float64)

    # Evaluate basis
    PHI = np.empty((basis_dim, n_points), dtype=np.float64)
    for i, phi in enumerate(basis_funcs):
        PHI[i, :] = np.asarray(phi(Points), dtype=np.float64).ravel()

    # --- moment matrix ---
    final_weights = density_vals * weights
    # M = PHI @ (PHI * weights[None, :]).T
    M = np.einsum('ik,jk,k->ij', PHI, PHI, final_weights, optimize=True)
    # Enforce symmetry 
    M = 0.5 * (M + M.T)
    return np.asarray(M, dtype=np.float64)


def compute_tau_k(numvars: int, mollifier,
                  quadrature_degree: int, verbose: bool = False):
    """
    Compute  τ_k = ∫_{S^{d-1}} g(<x,y>)^2 dy(y)  using 1D Gauss–Gegenbauer quadrature,
    where g is the mollifier provided.

    Parameters
    numvars : ambient dimension (S^{numvars-1})
    quadrature_degree : degree of quadrature
    mollifier : If not None, this function is evaluated instead of g_n.
    """

    # Gegenbauer parameter
    alpha = (numvars - 2) / 2.0

    # Gauss–Gegenbauer nodes and weights
    xk, wk = roots_gegenbauer(quadrature_degree, alpha)

    # Evaluate the integrand g(x)^2 on the quadrature nodes
    vals = np.asarray(mollifier(xk), dtype=float)
    vals_sq = vals * vals

    # 1D quadrature approximation of ∫ g_n(t)^2 w(t) dt
    I = np.sum(wk * vals_sq)

    # Funk–Hecke normalization factor to lift the 1D integral to the sphere
    C_numvars = hb.funk_hecke_constant_cached(numvars)

    tau_k = C_numvars * I

    if verbose:
        print(
            f"[compute_tau_k] numvars={numvars}, quadrature_degree={quadrature_degree}\n"
            f"  Integral = {I:.6e}, C_numvars = {C_numvars:.6e}, tau_k = {tau_k:.6e}"
        )

    return tau_k


def degree_multiplicity_vector(numvars: int, max_degree: int):
    """
    Return the list of harmonic degrees associated with the orthonormal
    spherical harmonic basis up to the given maximum degree.

    For each degree d, the multiplicity is equal to the dimension of the
    space of spherical harmonics of degree d in R^{numvars}. The output
    is a flat list where each degree appears as many times as its
    multiplicity.

    Parameters
    numvars : ambient dimension (points lie on S^{numvars-1}).
    max_degree : maximum harmonic degree to include.

    Returns
    A list of degrees with multiplicities, in ascending order.
    """
    # Basic validation
    if numvars < 2:
        raise ValueError("numvars must be >= 2.")
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0.")

    # Compute multiplicities for degrees 0,...,max_degree
    dims = [hb.harmonics_dimension(numvars, d) for d in range(max_degree + 1)]
    degrees = [d for d, m in enumerate(dims) for _ in range(m)]

    return degrees



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
    epsilon: float = 1e-15,
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

    # Evaluate the harmonic basis functions across X
    phi_evaluated = np.empty((dim_basis, n_points), dtype=float)
    for j, basis_func in enumerate(harmonic_basis):
        vals = basis_func(X)
        vals = np.asarray(vals, dtype=float).ravel()
        if vals.shape[0] != n_points:
            raise ValueError(
                f"Basis function {j} returned array of length {vals.shape[0]}; expected {n_points}."
            )
        phi_evaluated[j, :] = vals
    



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

def compute_infinite_christoffel(
    quad_points: np.ndarray,
    quad_weights: np.ndarray,
    density_vals: np.ndarray,
    mollifier,
) -> np.ndarray:
    """
    Compute the infinite-degree Christoffel function (theoretical reference)
    at each quadrature point, normalized to integrate to 1.

    This is the limit of the MCD estimator as degree goes to infinity,
    used to decompose the total error into projection and approximation parts.

    quad_points has shape (N, d), quad_weights and density_vals have shape (N,).
    mollifier is the 1D function g(t) defining the kernel K(x,y) = g(<x,y>)^2.

    Returns an array of shape (N,) normalized to integrate to 1.
    """
    # K(x, y) = g(<x,y>)^2 / f(y), integrated over y
    inner_products = quad_points @ quad_points.T          # (N, N)
    mollifier_vals = np.array(
        [mollifier(row) for row in inner_products]
    )                                                      # (N, N)
    integrand = mollifier_vals ** 2 / density_vals[None, :]
    infinite_christoffel_poly = np.sum(
        integrand * quad_weights[None, :], axis=1
    )                                                      # (N,)
    infinite_christoffel_func = 1.0 / infinite_christoffel_poly
    norm = np.sum(quad_weights * infinite_christoffel_func)
    return infinite_christoffel_func / norm