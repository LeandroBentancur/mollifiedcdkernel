import numpy as np
from scipy.integrate import quad
from typing import Callable, List, Optional
import harmonic_analysis_1d as ha1
from scipy.special import roots_gegenbauer

# =========================================================
# Define the polynomial mollifier
# =========================================================

def polynomial_mollifier(numvars:int, k: int):
    alpha = (numvars - 2.0) / 2.0
    roots, weights = roots_gegenbauer(k, alpha)
    xs = ((1+roots)/2)**k
    norm = np.sqrt(np.sum(weights * (xs**2)))
    def mollifier(t):
        t_arr = ((1+t)/2)**k
        values = t_arr/norm
        return values
    return mollifier

def default_mollifier_degree(deg: int) -> int:
    """
    Return the mollifier truncation parameter k for a given harmonic degree.
    Paper default: k = floor(deg^(4/3)).
    """
    return int(np.floor(deg ** (4 / 3)))


def default_mollifier(numvars: int, deg: int):
    """
    Build and return the default polynomial mollifier for a given
    harmonic degree, using the paper's k = floor(deg^(4/3)) schedule.
    """
    k = default_mollifier_degree(deg)
    return polynomial_mollifier(numvars=numvars, k=k)

# ==========================================
# Define the Gegenbauer mollifier
# ==========================================

def define_gegenbauer_mollifier(numvars: int, degree: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns gegenbauer_mollifier(t) = c * |prod_{j}(t - lambda_j)| where {lambda_j} are the roots
    of the Gegenbauer polynomial of degree 'degree' except the largest one and 'c' is a
    constant such that gegenbauer_mollifier integrates 1.
    """

    # Degree 0 case: the function is constant
    if degree == 0:
        return lambda t: np.ones_like(np.atleast_1d(t), dtype=float)

    alpha = (numvars - 2.0) / 2.0

    # ---- 1) Gets the roots ---------------------------------------------------
    if alpha == 0.0: # Chebyshev case
        k = np.arange(degree + 1)
        roots = np.cos((2 * k + 1) * np.pi / (2 * (degree + 1)))
    else: # Gegenbauer case
        roots, _ = roots_gegenbauer(degree + 1, alpha)

    roots = np.sort(np.asarray(roots, dtype=float))
    roots_filtered = roots[:-1]  # discards the largest root

    # ---- 2) Unnormalized mollifier  ------------------------
    if roots_filtered.size == 0:
        def g_raw(t):
            return np.ones_like(np.atleast_1d(t), dtype=float)
    else:
        def g_raw(t):
            t_arr = np.atleast_1d(t).astype(float)
            return np.abs(np.prod(t_arr[:, None] - roots_filtered[None, :], axis=1))

    # ---- 3) Defines the weight -------------------------------------
    if alpha == 0.0:
        # Chebyshev case
        def weight(t):
            return 1.0
        cut_points = np.sort(np.concatenate(([0.0],
                                             np.arccos(roots_filtered),
                                             [np.pi])))
        interval_iter = zip(cut_points[:-1], cut_points[1:])
        integrate = lambda a, b: quad(lambda th: g_raw(np.cos(th)),
                                      a, b, epsabs=1e-15, limit=400)[0]
    else:
        # Gegenbauer case
        def weight(t):
            return (1.0 - t**2)**(alpha - 0.5)
        cut_points = np.sort(np.concatenate(([-1.0], roots_filtered, [1.0])))
        interval_iter = zip(cut_points[:-1], cut_points[1:])
        integrate = lambda a, b: quad(lambda tt: g_raw(tt) * weight(tt),
                                      a, b, epsabs=1e-15, limit=400)[0]

    # ---- 4) Normalization constant ------------------------------
    I = sum(integrate(a, b) for a, b in interval_iter)

    if not np.isfinite(I) or I <= 0.0:
        raise RuntimeError(f"Normalization integral invalid: {I}")

    c = 1.0 / I

    # ---- 5) Final mollifier ----------------------------------------
    def gegenbauer_mollifier(t):
        t_arr = np.atleast_1d(t)
        vals = c * g_raw(t_arr)
        return float(vals.item()) if np.isscalar(t) else vals

    return gegenbauer_mollifier

def define_projected_gegenbauer_mollifier(
    numvars: int,
    degree: int,
    method: str = "quadrature",
    n_nodes: Optional[int] = None,
    tol: float = 1e-12,
    verbose: bool = False,
) -> List[float]:
    """
    Build the gegenbauer_mollifier and return its projection coefficients on the Gegenbauer basis.
    """
    gegenbauer_mollifier = define_gegenbauer_mollifier(numvars, degree)
    if method == "quadrature" and n_nodes is None:
        n_nodes = 1000 * (degree + 1)

    return ha1.define_projection(
        func=gegenbauer_mollifier,
        numvars=numvars,
        degree=degree,
        method=method,
        n_nodes=n_nodes,
        tol=tol,
        verbose=verbose,
    )

# =========================================================
# Define the mollifier given by the von Mises density
# =========================================================

def make_vonmises_1d(kappa=5.0, alpha=0.5):
    xs = np.linspace(-1.0, 1.0, 1000)
    g = np.exp(kappa * xs)
    w = (1.0 - xs**2)**alpha
    norm = np.trapz(g * w, xs)
    def mollifier(t):
        t_arr = np.asarray(t)
        values = np.exp(kappa * t_arr) / norm
        return values
    return mollifier




