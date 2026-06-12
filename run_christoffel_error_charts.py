import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

import harmonic_basis as hb
import error_decomposition as ed
from densities import DENSITY_REGISTRY


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

        # (Basis orthonormality is covered by the test suite, so it is not
        # re-checked here; doing so would only slow the convergence sweep.)

        # Mollifier defaults to the polynomial mollifier, k = floor(deg^(4/3)).
        result = ed.decompose_errors(basis, f_callable, numvars, deg)

        t3 = time.time()
        print(f"Done in {t3 - t1:.3f}s")

        results["total"].append(result.total_L2)
        results["projection"].append(result.proj_L2)
        results["approximation"].append(result.approx_L2)

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
