import numpy as np
import math
import scipy.special as sc
from scipy.special import ive, iv


def sphere_area(numvars: int) -> float:
    """Surface area of S^{numvars-1}. Delegates to harmonic_basis to avoid duplication."""
    import harmonic_basis as hb
    return hb.sphere_area(numvars)


# =========================================================
# Define test densities
# =========================================================

def constant_density(X, numvars):
    X = np.atleast_2d(X)
    area = sphere_area(numvars)
    return np.full(X.shape[0], 1.0 / area)

def von_mises_fisher_density(X, numvars, kappa=4.0, mu=None):
    X = np.atleast_2d(X)
    N, d = X.shape
    assert d == numvars, "Dim mismatch"

    # direction
    if mu is None:
        mu = np.zeros(d)
        mu[-1] = 1.0
    mu = mu / np.linalg.norm(mu)

    # normalization constant
    nu = d/2 - 1
    C_d = (kappa**nu) / ((2*np.pi)**(d/2) * iv(nu, kappa))

    # evaluate
    return C_d * np.exp(kappa * (X @ mu))

def mixture_von_mises_sphere(X, numvars, centers=None, kappas=None, weights=None):
    X = np.atleast_2d(X)
    N, d = X.shape
    assert d == numvars, "Dim mismatch"

    if centers is None:
        centers = np.eye(d)
        centers[1] = -centers[1]
    if kappas is None:
        kappas = [5.0] * len(centers)
    if weights is None:
        weights = [1/len(centers)] * len(centers)

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    f_tot = np.zeros(N)
    for w, mu, k in zip(weights, centers, kappas):
        mu = mu / np.linalg.norm(mu)
        nu = d/2 - 1
        C_d = (k**nu) / ((2*np.pi)**(d/2) * ive(nu, k))
        f_tot += w * C_d * np.exp(k * (X @ mu))

    return f_tot