import numpy as np
from scipy.integrate import quad
from scipy.special import roots_gegenbauer, gammaln
from typing import List, Callable, Optional

# ============================================================================
#  Gegenbauer basis module
# ============================================================================

def gegenbauer_recurrence(alpha: float, degree: int, t: np.ndarray) -> np.ndarray:
    """
    Evaluate Gegenbauer polynomials C_n^{(alpha)}(t) for n=0..degree.
    Returns an array of shape (degree+1, len(t)).
    """
    t = np.atleast_1d(t)
    vals = np.zeros((degree + 1, t.size), dtype=float)
    vals[0, :] = 1.0

    if degree == 0:
        return vals

    # Chebyshev special case (alpha = 0)
    if alpha == 0.0:
        vals[1, :] = t
        for n in range(1, degree):
            vals[n + 1, :] = 2 * t * vals[n, :] - vals[n - 1, :]
        return vals

    # General Gegenbauer recurrence
    vals[1, :] = 2 * alpha * t
    for n in range(1, degree):
        a = 2.0 * (n + alpha) / (n + 1)
        b = (n + 2 * alpha - 1) / (n + 1)
        vals[n + 1, :] = a * t * vals[n, :] - b * vals[n - 1, :]

    return vals


def gegenbauer_norm(alpha: float, degree: int) -> float:
    """
    Squared L^2 norm of Gegenbauer polynomials. We use logarithmics for numerical stability.
    """
    # For the case alpha=0 we use the Chebyshev polynomials
    if alpha == 0.0:
        return np.pi if degree == 0 else np.pi / 2.0

    # General case
    ln_num = np.log(np.pi) + (1.0 - 2.0 * alpha) * np.log(2.0) + gammaln(degree + 2.0 * alpha)
    ln_den = np.log(degree + alpha) + gammaln(degree + 1.0) + 2.0 * gammaln(alpha)
    return float(np.exp(ln_num - ln_den))


def generate_gegenbauer_basis(numvars: int, degree: int) -> List[Callable[[np.ndarray], np.ndarray]]:
    """
    Return a list [phi_0, ..., phi_degree] of functions phi_k(t) that evaluate
    the orthonormal Gegenbauer basis on [-1,1] for the sphere S^{numvars-1}.
    """
    alpha = (numvars - 2.0) / 2.0

    def make_phi(k: int) -> Callable[[np.ndarray], np.ndarray]:
        def phi(t: np.ndarray) -> np.ndarray:
            t_arr = np.atleast_1d(t)
            vals = gegenbauer_recurrence(alpha, k, t_arr)[k, :]
            norm = np.sqrt(gegenbauer_norm(alpha, k))
            return vals / norm
        return phi

    return [make_phi(k) for k in range(degree + 1)]

def generate_gegenbauer_basis_evaluator(numvars: int, degree: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return an array (degree+1, len(t)) with orthonormal Gegenbauer polynomials evaluated at t.
    """
    alpha = (numvars - 2.0) / 2.0

    norms = np.sqrt([gegenbauer_norm(alpha, k) for k in range(degree + 1)])
    norms = norms[:, None]

    def evaluator(t: np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(t).astype(float)
        vals = gegenbauer_recurrence(alpha, degree, t_arr)
        return vals / norms

    return evaluator




# ============================================================================
#  Projection operator
# ============================================================================

def define_projection(
    func: Callable[[np.ndarray], np.ndarray],
    numvars: int,
    degree: int,
    method: str = "quadrature",
    n_nodes: Optional[int] = None,
    tol: float = 1e-12,
    verbose: bool = False,
) -> List[float]:
    """
    Project `func` to the orthonormal Gegenbauer basis up to `degree` and
    parameter alpha in correspondence with numvars.

    There are two possible methods:
      - 'quadrature': use Gauss quadrature
      - 'quad': adaptive integration

    Returns a list with the coefficients [a_0, ..., a_degree]
    """
    alpha = (numvars - 2.0) / 2.0
    # basis is an orthonormal basis
    basis = generate_gegenbauer_basis(numvars, degree)

    coeffs = np.zeros(degree + 1, dtype=float)

    if method == "quadrature":
        # Define the size of the quadrature
        if n_nodes is None:
            n_nodes = 8 * (degree + 1)

        # Gauss nodes & weights for the appropriate weight
        if alpha == 0.0:
            i = np.arange(1, n_nodes + 1)
            nodes = np.cos((2.0 * i - 1.0) * np.pi / (2.0 * n_nodes))
            weights = np.full(n_nodes, np.pi / n_nodes)
        else:
            nodes, weights = roots_gegenbauer(n_nodes, alpha)

        fvals = np.atleast_1d(func(nodes)).astype(float)
        
        evaluator = generate_gegenbauer_basis_evaluator(numvars, degree)
        basis_vals = evaluator(nodes)  # shape (degree+1, n_nodes)

        coeffs = basis_vals @ (weights * fvals)

    elif method == "quad":
        weight_func = (lambda tval: (1.0 - tval**2) ** (alpha - 0.5)) if alpha != 0.0 else (lambda tval: 1.0)
        # We calculate the inner products
        for k, phi_k in enumerate(basis):
            integrand = lambda tval: func(np.atleast_1d(tval)) * phi_k(tval) * weight_func(tval)
            val, _ = quad(lambda tt: integrand(tt)[0] if np.ndim(integrand(tt)) > 0 else integrand(tt),
                          -1.0, 1.0, epsabs=tol)
            coeffs[k] = val

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'quadrature' or 'quad'.")

    if verbose:
        print(f"Projection finished: {degree + 1} coefficients computed using method='{method}', n_nodes={n_nodes}.")

    return coeffs.tolist()