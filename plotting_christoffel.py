import numpy as np
import mollifiers as mol
import densities as dens
import harmonic_basis as hb
import christoffel as ch
from matplotlib import pyplot as plt, cm
from pathlib import Path
import time

def _make_spherical_grid(n_theta, n_phi):
    """Return spherical grid arrays and flattened Cartesian points.

    Returns:
    theta, phi, Xs, Ys, Zs, X_grid (N x 3), grid_shape
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)


    Xs = np.sin(theta) * np.cos(phi)
    Ys = np.sin(theta) * np.sin(phi)
    Zs = np.cos(theta)


    X_grid = np.stack([Xs.ravel(), Ys.ravel(), Zs.ravel()], axis=1)
    grid_shape = phi.shape
    
    return theta, phi, Xs, Ys, Zs, X_grid, grid_shape

def _face_average(values):
    """Average per quad face for correct surface facecolors.

    Input values are expected to be 2D arrays with shape grid_shape.
    """
    return (values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:]) / 4.0

def _evaluate_mollified_christoffel(
    X_grid,
    harmonic_basis,
    moment_matrix,
    numvars,
    degree,
    mollifier_1d
):
    """Evaluate mollified Christoffel and return an L1-normalized array.

    The function returns values normalized by their mean so that the
    returned array has mean 1, matching the previous behaviour.
    """
    values = ch.mollified_christoffel_evaluator(
    X_grid,
    harmonic_basis=harmonic_basis,
    moment_matrix=moment_matrix,
    numvars=numvars,
    degree=degree,
    mollifier=mollifier_1d,
    verbose=False,
    )
    values = 1.0 / values
    values = values.astype(float)
    values /= np.sqrt(np.mean(values**2))
    return values

def plot_estimator_vs_true(
    quad_points: np.ndarray,
    f_true: np.ndarray,
    f_estimated: np.ndarray,
    density_name: str = None,
    mollifier_name: str = None,
    numvars: int = 3,
    degree: int = None,
    plot_mode: str = "quadrature",
    out_dir_path=None,
):
    """
    Plot the true density and the MCD estimator side by side.

    quad_points has shape (N, 3), f_true and f_estimated have shape (N,).
    plot_mode can be 'quadrature' (scatter plot at quadrature nodes) or
    'surface' (smooth mesh interpolated on a spherical grid). Only implemented
    for numvars=3.
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt

    if numvars != 3:
        return  # only implemented for S^2

    name = density_name or "unknown_density"
    moll = mollifier_name or "unknown_mollifier"

    fmin = min(np.min(f_true), np.min(f_estimated))
    fmax = max(np.max(f_true), np.max(f_estimated))
    norm = plt.Normalize(vmin=fmin, vmax=fmax)
    cmap = cm.viridis

    if plot_mode == "quadrature":
        X, Y, Z = quad_points[:, 0], quad_points[:, 1], quad_points[:, 2]

        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(X, Y, Z, c=cmap(norm(f_true)), s=12, marker="o", depthshade=False)
        ax1.set_title("True density")
        ax1.set_box_aspect([1, 1, 1])
        ax1.axis("off")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(X, Y, Z, c=cmap(norm(f_estimated)), s=12, marker="o", depthshade=False)
        ax2.set_title("Recovered density")
        ax2.set_box_aspect([1, 1, 1])
        ax2.axis("off")

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(np.concatenate([f_true, f_estimated]))
        plt.colorbar(mappable, ax=[ax1, ax2], fraction=0.035, pad=0.05)

        deg_str = f"deg={degree}" if degree is not None else ""
        plt.suptitle(f"{name} | n={numvars}, {deg_str}, mollifier={moll}")
        plt.tight_layout()

        if out_dir_path is not None:
            fname = f"quad_points_surface_{name}_mollifier-{moll}_n{numvars}_deg{degree}.png"
            plt.savefig(out_dir_path / fname, dpi=200)

        plt.close()

    elif plot_mode == "surface":
        n_theta, n_phi = 100, 200
        theta = np.linspace(0, np.pi, n_theta)
        phi_arr = np.linspace(0, 2 * np.pi, n_phi)
        theta, phi_arr = np.meshgrid(theta, phi_arr)
        Xs = np.sin(theta) * np.cos(phi_arr)
        Ys = np.sin(theta) * np.sin(phi_arr)
        Zs = np.cos(theta)

        f_true_2d     = f_true.reshape(phi_arr.shape)
        f_est_2d      = f_estimated.reshape(phi_arr.shape)
        face_true     = _face_average(f_true_2d)
        face_est      = _face_average(f_est_2d)

        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(Xs, Ys, Zs, facecolors=cmap(norm(face_true)),
                         rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax1.set_title("True density")
        ax1.axis("off")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot_surface(Xs, Ys, Zs, facecolors=cmap(norm(face_est)),
                         rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax2.set_title("Recovered density")
        ax2.axis("off")

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(np.concatenate([f_true.ravel(), f_estimated.ravel()]))
        plt.colorbar(mappable, ax=[ax1, ax2], fraction=0.035, pad=0.05)

        deg_str = f"deg={degree}" if degree is not None else ""
        plt.suptitle(f"{name} | n={numvars}, {deg_str}, mollifier={moll}")
        plt.tight_layout()

        if out_dir_path is not None:
            fname = f"surface_{name}_mollifier-{moll}_n{numvars}_deg{degree}.png"
            plt.savefig(out_dir_path / fname, dpi=200)

        plt.close()
    else:
        raise ValueError("plot_mode must be 'quadrature' or 'surface'.")

def plot_mollified_christoffel_comparison_on_sphere_3(
    degrees,
    mollifier_1d_list,
    basis_list,
    density,
    n_theta=100,
    n_phi=200,
    density_name=None,
    mollifier_family=None,
    out_dir_path=None,
):
    """Plot the true density on S^2 alongside three MCD estimators in a 2x2 panel.

    degrees is a sequence of 3 ints. mollifier_1d_list and basis_list are
    sequences of 3 callables and basis function lists respectively, one per degree.
    density is a callable mapping (N, 3) points on S^2 to R.
    """

    # -----------------------------
    # Sanity checks
    # -----------------------------
    if not callable(density):
        raise TypeError("density must be a callable mapping S^2 to R.")

    if len(degrees) != 3 or len(mollifier_1d_list) != 3 or len(basis_list) != 3:
        raise ValueError("This function expects exactly 3 estimators.")

    for d in degrees:
        if d <= 0:
            raise ValueError("Degrees must be positive integers.")

    # -----------------------------
    # Spherical grid and true density
    # -----------------------------
    theta, phi, Xs, Ys, Zs, X_grid, grid_shape = _make_spherical_grid(n_theta, n_phi)

    f_true_flat = density(X_grid).astype(float)
    f_true_flat /= np.sqrt(np.mean(f_true_flat**2))

    # -----------------------------
    # Compute moment matrices and evaluate estimators
    # -----------------------------
    numvars = 3

    moment_matrix_1 = ch.compute_moment_matrix_on_sphere(
        basis_funcs=basis_list[0], density=density, numvars=numvars,
        method="quadrature", quadrature_degree=2 * degrees[0])
    moment_matrix_2 = ch.compute_moment_matrix_on_sphere(
        basis_funcs=basis_list[1], density=density, numvars=numvars,
        method="quadrature", quadrature_degree=2 * degrees[1])
    moment_matrix_3 = ch.compute_moment_matrix_on_sphere(
        basis_funcs=basis_list[2], density=density, numvars=numvars,
        method="quadrature", quadrature_degree=2 * degrees[2])

    f_approx_1_flat = _evaluate_mollified_christoffel(
        X_grid, harmonic_basis=basis_list[0], moment_matrix=moment_matrix_1, numvars=numvars, degree=degrees[0], mollifier_1d=mollifier_1d_list[0]
    )

    f_approx_2_flat = _evaluate_mollified_christoffel(
        X_grid, harmonic_basis=basis_list[1], moment_matrix=moment_matrix_2, numvars=numvars, degree=degrees[1], mollifier_1d=mollifier_1d_list[1]
    )

    f_approx_3_flat = _evaluate_mollified_christoffel(
        X_grid, harmonic_basis=basis_list[2], moment_matrix=moment_matrix_3, numvars=numvars, degree=degrees[2], mollifier_1d=mollifier_1d_list[2]
    )

    # -----------------------------
    # Reshape to grid
    # -----------------------------
    f_true = f_true_flat.reshape(grid_shape)
    f_approx_1 = f_approx_1_flat.reshape(grid_shape)
    f_approx_2 = f_approx_2_flat.reshape(grid_shape)
    f_approx_3 = f_approx_3_flat.reshape(grid_shape)

    # -----------------------------
    # Common color normalization across all panels
    # -----------------------------
    fmin = min(f_true.min(), f_approx_1.min(), f_approx_2.min(), f_approx_3.min())
    fmax = max(f_true.max(), f_approx_1.max(), f_approx_2.max(), f_approx_3.max())

    norm = plt.Normalize(vmin=fmin, vmax=fmax)
    cmap = cm.viridis

    facecolors = [
        cmap(norm(_face_average(f_true))),
        cmap(norm(_face_average(f_approx_1))),
        cmap(norm(_face_average(f_approx_2))),
        cmap(norm(_face_average(f_approx_3))),
    ]

    # -----------------------------
    # Plot: 2x2 square with colorbar on the right (spanning both rows)
    # -----------------------------
    fig = plt.figure(figsize=(10, 9))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.0, 1.0, 0.05],  # thinner colorbar
        wspace=0.01,
        hspace=-0.1,
    )

    ax_tl = fig.add_subplot(gs[0, 0], projection="3d")
    ax_tr = fig.add_subplot(gs[0, 1], projection="3d")
    ax_bl = fig.add_subplot(gs[1, 0], projection="3d")
    ax_br = fig.add_subplot(gs[1, 1], projection="3d")
    ax_cbar = fig.add_subplot(gs[:, 2])

    # FORCE DRAW so positions are finalized
    fig.canvas.draw()

    # Shrink colorbar height (Solution 1)
    bbox = ax_cbar.get_position()
    new_h = 0.40 * bbox.height
    new_y = bbox.y0 + 0.5 * bbox.height - 0.5 * new_h
    ax_cbar.set_position([bbox.x0, new_y, bbox.width, new_h])




    axes = [ax_tl, ax_tr, ax_bl, ax_br]
    panels = [f_true, f_approx_1, f_approx_2, f_approx_3]

    titles = [
        "True density",
        f"Estimator (degree {degrees[0]})",
        f"Estimator (degree {degrees[1]})",
        f"Estimator (degree {degrees[2]})",
    ]

    # Plot spheres
    for ax, values, colors, title in zip(axes, panels, facecolors, titles):
        ax.plot_surface(
            Xs, Ys, Zs,
            facecolors=colors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        ax.view_init(elev=30, azim=-75)

        # MAKE SPHERES FILL THE AXES
        ax.set_box_aspect([0.8, 0.8, 0.8])
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([-0.8, 0.8])
        ax.set_zlim([-0.8, 0.8])

        # Bold titles
        ax.set_title(title, fontsize=14, fontweight="bold", y=1)
        ax.axis("off")

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.concatenate([f_true_flat, f_approx_1_flat, f_approx_2_flat, f_approx_3_flat]))
    fig.colorbar(mappable, cax=ax_cbar)

    # REMOVE GLOBAL WHITE SPACE
    fig.subplots_adjust(left=0.0, right=0.93, bottom=0.0, top=0.95)

    # Saving
    name = density_name or "unknown_density"
    moll = mollifier_family or "unknown_mollifier"

    if out_dir_path is not None:
        filename = f"surface_{name}_mollifier-{moll}_deg_{degrees[0]}_{degrees[1]}_{degrees[2]}.png"
        fig.savefig(out_dir_path / filename, dpi=300, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    '''
    This module was built for plotting a density in S^2 together
    with two mollified Christoffel density estimators of different degrees
    '''
    degree_list = [(10, 20, 30)]
    
    for degrees in degree_list:
        print(f"=== Plot Christoffel density recovery for degrees {degrees[0]}, {degrees[1]} and {degrees[2]} ===\n")

        numvars = 3
        alpha = 0.5  # weight exponent for (1 - t^2)^alpha

        base_dir = Path(__file__).resolve().parent
        out_dir_path = base_dir / "plot_christoffel_comparison"
        out_dir_path.mkdir(parents=True, exist_ok=True)


        # -------------------------------------------------
        # Orthonormal harmonic basis
        # -------------------------------------------------
        print("------------   Preparing the orthonormal basis   ------------")
        basis_list = []
        for deg in degrees:
            time_0 = time.time()
            harmonic_basis = hb.orthonormal_harmonic_basis_up_to_degree(numvars, deg)
            basis_list.append(harmonic_basis)
            time_1 = time.time()
            time_basis = time_1- time_0
            print(f"Time for preparing the degree {deg} basis: {time_basis:.2f}s\n")

        # -------------------------------------------------
        # Selecting density and mollifier
        # -------------------------------------------------
        density = lambda X: dens.mixture_von_mises_sphere(X, numvars=X.shape[1], kappas=[3.0, 3.0, 3.0])
        density_name = "Mixture of von Mises-Fisher with kappa 3"

        mollifier_family = "polynomial_default"
        mollifier_list = [mol.default_mollifier(numvars=numvars, deg=deg) for deg in degrees]

        time_2 = time.time()
        plot_mollified_christoffel_comparison_on_sphere_3(
            degrees= degrees,
            mollifier_1d_list= mollifier_list,
            basis_list= basis_list,
            density = density,
            n_theta=100,
            n_phi=200,
            density_name= density_name,
            mollifier_family= mollifier_family,
            out_dir_path= out_dir_path,
        )
        time_3 = time.time()
        time_plotting = time_3- time_2
        print(f"Time for preparing the plotting: {time_plotting:.2f}s\n")


