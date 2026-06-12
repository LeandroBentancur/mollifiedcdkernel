import argparse
import numpy as np
import harmonic_basis as hb
import error_decomposition as ed
from densities import DENSITY_REGISTRY
from pathlib import Path
import time
import pandas as pd
import plotting_christoffel as plt_ch



# =========================================================
# Argument parser
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Mollified Christoffel-Darboux density estimator on S^2 "
            "and save L2/sup errors to a CSV file."
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
        default=[30, 35, 40],
        help="List of harmonic degrees to run (default: 30 35 40).",
    )
    return parser.parse_args()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    args = parse_args()
    degrees    = args.degrees
    dens_name  = args.density
    f_callable = DENSITY_REGISTRY[dens_name]

    print("=== Mollified Christoffel-Darboux density estimator ===\n")
    print(f"Density  : {dens_name}")
    print(f"Degrees  : {degrees}")
    print(f"Mollifier: polynomial, k = floor(deg^(4/3))  [paper default]\n")

    numvars = 3

    base_dir     = Path(__file__).resolve().parent
    out_dir_path = base_dir / "test_christoffel"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir_path / "christoffel_errors_table.csv"

    results = []

    for deg in degrees:
        t0_basis = time.time()
        print("\n" + "=" * 60)
        print(f"  Degree: {deg}")
        print("=" * 60)

        # --- Build orthonormal spherical harmonic basis ---
        harmonic_basis = hb.orthonormal_harmonic_basis_up_to_degree(numvars, deg)
        len_basis = len(harmonic_basis)
        basis_errors = hb.check_basis_orthonormality(
            harmonic_basis, numvars, max_degree=deg, tol=1e-6, verbose=True)
        basis_L1_error_offdiag = basis_errors['L1_error_offdiag']
        basis_L1_error_diag    = basis_errors['L1_error_diag']
        t1_basis = time.time()
        print(f"Basis built in {t1_basis - t0_basis:.2f}s")

        quadrature_degree = 2 * deg
        moll_name = "polynomial"
        print(f"Mollifier degree = {deg}")

        t0 = time.time()

        # --- Error decomposition (estimator + projection/approximation split) ---
        # Mollifier defaults to the polynomial mollifier, k = floor(deg^(4/3)).
        result = ed.decompose_errors(harmonic_basis, f_callable, numvars, deg)

        quad_size   = len(result.quad_weights)
        cond_number = np.linalg.cond(result.moment_matrix)

        total_error_L2  = result.total_L2
        total_error_sup = result.total_sup
        proj_error_L2   = result.proj_L2
        proj_error_sup  = result.proj_sup
        approx_error_L2 = result.approx_L2
        approx_error_sup = result.approx_sup

        # --- Plot ---
        plt_ch.plot_estimator_vs_true(
            result.quad_points, result.f_vals, result.christoffel_approx,
            density_name=dens_name, mollifier_name=moll_name,
            numvars=numvars, degree=deg,
            plot_mode="quadrature", out_dir_path=out_dir_path,
        )

        t1 = time.time()
        print(f"Done in {t1 - t0:.2f}s")

        results.append([
                    dens_name, deg, moll_name,
                    total_error_L2, total_error_sup,
                    proj_error_L2, proj_error_sup,
                    approx_error_L2, approx_error_sup,
                    len_basis, cond_number,
                    quadrature_degree, quad_size,
                    t1 - t0, t1_basis - t0_basis,
                    basis_L1_error_offdiag, basis_L1_error_diag,
                ])

    # --- Save results ---
    df_new = pd.DataFrame(
        results,
        columns=[
            "Density", "Degree", "Mollifier",
            "total_error_L2", "total_error_sup",
            "projection_error_L2", "projection_error_sup",
            "approximation_error_L2", "approximation_error_sup",
            "len_basis", "cond_moment_matrix",
            "Quadrature Degree", "Quad Size",
            "Time test", "Time Building Basis",
            "basis_L1_error_offdiag", "basis_L1_error_diag",
        ],
    )

    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
