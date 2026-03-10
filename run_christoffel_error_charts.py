import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

import harmonic_basis as hb
import mollifiers as mol
import densities as dens
import christoffel as ch
from quadrature_S import sphere_Quadrature


# =========================================================
# Density registry
# All available densities for the sphere S^{n-1}.
# To add a new density: add an entry here, no other changes needed.
# =========================================================

DENSITY_REGISTRY = {
    "constant":                 lambda X: dens.constant_density(X, numvars=X.shape[1]),
    "von_mises_kappa2":         lambda X: dens.von_mises_fisher_density(X, numvars=X.shape[1], kappa=2.0),
    "von_mises_kappa3":         lambda X: dens.von_mises_fisher_density(X, numvars=X.shape[1], kappa=3.0),
    "mixture_von_mises_kappa3": lambda X: dens.mixture_von_mises_sphere(X, numvars=X.shape[1], kappas=[3.0, 3.0, 3.0]),
    "mixture_von_mises_kappa5": lambda X: dens.mixture_von_mises_sphere(X, numvars=X.shape[1], kappas=[5.0, 5.0, 5.0]),
}


# =========================================================
# Argument parser
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute L2 convergence rates of the MCD estimator vs degree "
            "and produce log-log error charts."
        )
    )
    parser.add_argument(
        "--density",
        type=str,
        required=True,
        choices=list(DENSITY_REGISTRY.keys()),
        help="Density to estimate. Choices: " + ", ".join(DENSITY_REGISTRY.keys()),
    )
    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 25, 30, 35, 40],
        help="List of harmonic degrees to run (default: 5 10 15 20 25 30 35 40).",
    )
    return parser.parse_args()


# =========================================================
# Helper: log-log slope
# =========================================================

def compute_slope(degrees, errors):
    logd = np.log(degrees)
    loge = np.log(errors)
    slope, _ = np.polyfit(logd, loge, 1)
    return slope


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    args = parse_args()
    degrees   = args.degrees
    dens_name = args.density
    f_callable = DENSITY_REGISTRY[dens_name]

    numvars = 3

    print("\n=== MCD convergence rate experiment ===\n")
    print(f"Density  : {dens_name}")
    print(f"Degrees  : {degrees}")
    print(f"Mollifier: polynomial, k = floor(deg^(4/3))  [paper default]\n")

    base_dir = Path(__file__).resolve().parent
    out_dir  = base_dir / "error_charts_polynomial"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "total":        [],
        "projection":   [],
        "approximation":[],
    }

    for deg in degrees:

        print("\n" + "=" * 40)
        print(f"Degree {deg}")
        print("=" * 40)

        t0 = time.time()
        basis = hb.orthonormal_harmonic_basis_up_to_degree(numvars, deg)
        t1 = time.time()
        print(f"Basis built in {t1 - t0:.3f}s")

        hb.check_basis_orthonormality(
            basis, numvars, max_degree=deg, tol=1e-6, verbose=False)

        # Mollifier: polynomial at degree floor(deg^(4/3))  [paper default]
        mollifier = mol.default_mollifier(numvars=numvars, deg=deg)

        t2 = time.time()

        # --- Quadrature ---
        quad_points, quad_weights = sphere_Quadrature(numvars, 2 * deg)
        quad_points  = np.asarray(quad_points,  dtype=np.float64)
        quad_weights = np.asarray(quad_weights, dtype=np.float64)

        # --- Normalize density ---
        f_vals = f_callable(quad_points).astype(float)
        f_vals /= np.sum(quad_weights * f_vals)

        # --- Moment matrix ---
        moment_matrix = ch.compute_moment_matrix_on_sphere(
            basis, f_callable, numvars=numvars,
            method="quadrature", quadrature_degree=2*deg)

        # --- MCD estimator ---
        christoffel_poly  = ch.mollified_christoffel_evaluator(
            quad_points, moment_matrix, basis, numvars, deg, mollifier)
        christoffel_func  = 1.0 / christoffel_poly
        christoffel_approx = christoffel_func / np.sum(quad_weights * christoffel_func)

        # --- Infinite Christoffel (theoretical reference) ---
        infinite_approx = ch.compute_infinite_christoffel(
            quad_points, quad_weights, f_vals, mollifier)

        # --- Errors ---
        total_err  = christoffel_approx - f_vals
        proj_err   = christoffel_approx - infinite_approx
        approx_err = infinite_approx    - f_vals

        total_error_L2  = float(np.sqrt(np.sum(quad_weights * total_err**2) / np.sum(quad_weights)))
        proj_error_L2   = float(np.sqrt(np.sum(quad_weights * proj_err**2)) / np.sum(quad_weights))
        approx_error_L2 = float(np.sqrt(np.sum(quad_weights * approx_err**2)) / np.sum(quad_weights))

        t3 = time.time()
        print(f"Done in {t3 - t2:.3f}s")

        results["total"].append(total_error_L2)
        results["projection"].append(proj_error_L2)
        results["approximation"].append(approx_error_L2)

    # =========================================================
    # Print slopes
    # =========================================================

    degrees_np = np.array(degrees)
    total  = np.array(results["total"])
    proj   = np.array(results["projection"])
    approx = np.array(results["approximation"])

    slope_total  = compute_slope(degrees_np, total)
    slope_proj   = compute_slope(degrees_np, proj)
    slope_approx = compute_slope(degrees_np, approx)

    print("\n" + "=" * 40)
    print(f"Density: {dens_name}")
    print("-" * 40)
    print(f"Slope total error        : {slope_total:.4f}")
    print(f"Slope projection error   : {slope_proj:.4f}")
    print(f"Slope approximation error: {slope_approx:.4f}")
    print(f"Expected slope (theory)  : {-4/3:.4f}")
    print("=" * 40)

    # =========================================================
    # Plot
    # =========================================================

    # Reference curve d^{-4/3} aligned to the last point
    C_ref   = total[-1] * degrees_np[-1] ** (4 / 3)
    ref_curve = C_ref / degrees_np ** (4 / 3)

    plt.figure(figsize=(7, 5))

    plt.loglog(degrees_np, total,  "-o", linewidth=2, label="Total error")
    plt.loglog(degrees_np, proj,   "-s", linewidth=2, label="Projection error")
    plt.loglog(degrees_np, approx, "-^", linewidth=2, label="Approximation error")
    plt.loglog(degrees_np, ref_curve,
               color="black", linewidth=2, linestyle="--", label=r"$C\, d^{-4/3}$")

    plt.xlabel("Degree")
    plt.ylabel("L2 error")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    filename = f"error_chart_{dens_name}.png"
    plt.savefig(out_dir / filename, dpi=300)
    plt.close()

    print(f"\nChart saved to: {out_dir / filename}")
