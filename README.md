# Mollified Christoffel-Darboux Kernels — Numerical Experiments

This repository contains the numerical experiments for the paper:

> **Mollified Christoffel-Darboux Kernels and Density Recovery on Varieties**
> Leandro Bentancur, Didier Henrion, Mauricio Velasco
> Preprint, 2025. *(arXiv link to be added)*

---

## What this code does

This code implements the **Mollified Christoffel-Darboux (MCD) density estimator** on the
unit sphere S^{n-1} ⊂ R^n. The key idea is to replace the point-evaluation functional in
the variational characterization of the CD kernel by a smoothed (mollified) version. This
yields an estimator that:

- converges directly to the density f **without requiring knowledge of the equilibrium
  measure** of the domain,
- is computable from moment data via a moment matrix and a single linear solve,
- satisfies an **improved dichotomy property**: the MCD polynomial is uniformly bounded
  on the interior of the support as the degree grows, while growing exponentially off
  the support,
- achieves **explicit convergence rates** depending on the Sobolev regularity of f and
  the choice of mollifier. On the sphere, algebraic mollifiers constructed from zonal
  polynomials yield rates that strictly improve on previously known estimates.

The estimator at a point x ∈ S^{n-1} is:

    MCD_{d,k}(x) = h_x^T M^{-1} h_x

where M is the moment matrix of μ in the orthonormal spherical harmonic basis up to
degree d, and h_x is the vector of mollified evaluation functionals computed via the
Funk-Hecke theorem. The density estimate is 1 / MCD_{d,k}(x), normalized to integrate to 1.

The total estimation error decomposes as:

    ||estimate - f|| ≤ projection error + approximation error

where the projection error comes from finite-dimensional truncation and the approximation
error comes from mollification. The scripts in this repository compute both components
and verify the convergence rates predicted by the theory.

---

## Repository structure

**Core library**

| File | Role |
|------|------|
| `harmonic_analysis_1d.py` | Gegenbauer polynomial basis on [-1,1]: recurrence, norms, L² projection |
| `harmonic_basis.py` | Orthonormal spherical harmonic basis on S^{n-1} via numerical Gram-Schmidt on zonal functions |
| `christoffel.py` | Core estimator: moment matrix, Funk-Hecke λ-vector, MCD polynomial evaluator, infinite Christoffel function |
| `quadrature_S.py` | Spherical quadrature rules via recursive Gauss-Gegenbauer product construction |
| `mollifiers.py` | Mollifier functions: polynomial mollifier, Gegenbauer-root mollifier, von Mises mollifier, default degree schedule |
| `densities.py` | Test densities on the sphere: von Mises-Fisher, mixtures, constant |
| `plotting_christoffel.py` | 3D visualization utilities and density comparison plots |

**Experiment scripts**

| File | Role |
|------|------|
| `plotting_christoffel.py` | Produces Figure 1: true density and three MCD approximations at degrees 10, 20, 30 |
| `run_christoffel_error_charts.py` | Produces Figure 2: log-log convergence rate plot of L2 error vs harmonic degree |
| `main_christoffel.py` | Full experiment runner: L2 and sup errors across degrees and densities |

**Tests**

| File | Role |
|------|------|
| `tests/test_1d_basis.py` | Orthonormality and projection accuracy of the Gegenbauer basis |
| `tests/test_harmonic_basis.py` | Orthonormality of the spherical harmonic basis, dimension formula, sphere area |
| `tests/test_quadrature.py` | Exactness of the spherical quadrature rule |
| `tests/test_mollifiers.py` | Non-negativity and consistency of mollifier functions |
| `tests/test_christoffel_core.py` | Moment matrix, lambda vector, and MCD estimator correctness |

---

## Installation

**Requirements:** Python 3.12

```bash
git clone https://github.com/LeandroBentancur/mollifiedcdkernel.git
cd mollifiedcdkernel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the tests

```bash
pytest tests/ -v
```

All 39 tests should pass. The test suite covers the quadrature rule, the 1D Gegenbauer
basis, the spherical harmonic basis, the mollifier functions, and the core MCD estimator.

---

## Reproducing the paper figures

### Figure 1 — Density recovery on S² at three degrees

Produces a 2×2 panel showing the true density (mixture of von Mises-Fisher with κ=3)
alongside MCD approximations at degrees 10, 20, and 30 using the default polynomial
mollifier with k = floor(d^{4/3}).

```bash
python3 plotting_christoffel.py
```

Output is saved to `plot_christoffel_comparison/`.

### Figure 2 — L2 convergence rate

Produces a log-log plot of the total, projection, and approximation L2 errors
vs harmonic degree, together with the theoretical reference curve d^{-4/3}.

```bash
python3 run_christoffel_error_charts.py --density mixture_von_mises_kappa3 --degrees 5 10 15 20 25 30 35 40
```

Available density choices: `constant`, `von_mises_kappa2`, `von_mises_kappa3`,
`mixture_von_mises_kappa3`, `mixture_von_mises_kappa5`.

Output is saved to `error_charts_polynomial/`.

---

## Quick usage example

```python
import numpy as np
import harmonic_basis as hb
import mollifiers as mol
import densities as dens
import christoffel as ch
from quadrature_S import sphere_Quadrature

numvars = 3   # points live on S² ⊂ R³
degree  = 20

# 1. Build orthonormal spherical harmonic basis up to degree d
basis = hb.orthonormal_harmonic_basis_up_to_degree(numvars, degree)

# 2. Define a target density and the default mollifier (k = floor(d^{4/3}))
density   = lambda X: dens.von_mises_fisher_density(X, numvars=numvars, kappa=3.0)
mollifier = mol.default_mollifier(numvars=numvars, deg=degree)

# 3. Build quadrature rule and compute the moment matrix
quad_pts, quad_wts = sphere_Quadrature(numvars, 2 * degree)
quad_pts = np.asarray(quad_pts, dtype=np.float64)
quad_wts = np.asarray(quad_wts, dtype=np.float64)

M = ch.compute_moment_matrix_on_sphere(
    basis, density, numvars=numvars,
    method="quadrature", quadrature_degree=2 * degree)

# 4. Evaluate the MCD polynomial at the quadrature points
mcd_poly = ch.mollified_christoffel_evaluator(
    quad_pts, M, basis, numvars, degree, mollifier)

# 5. The density estimate is 1 / MCD, normalized to integrate to 1
estimate = 1.0 / mcd_poly
estimate /= np.sum(quad_wts * estimate)
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{BentancurHenrionVelasco2025,
  title  = {Mollified Christoffel-Darboux Kernels and Density Recovery on Varieties},
  author = {Bentancur, Leandro and Henrion, Didier and Velasco, Mauricio},
  year   = {2025},
  note   = {Preprint}
}
```
