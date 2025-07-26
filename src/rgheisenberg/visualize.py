"""visualize.py
Utility functions to inspect Legendre–Fourier coefficient (LFC) vectors
interactively.  No heavy numerics live here – only lightweight plotting
wrappers around NumPy, SciPy and Matplotlib.

The public API purposely mirrors the rest of the package:

* `plot_coefficients(lfc, ax=None, **scatter_kw)` – scatter plot of the
  coefficients versus their harmonic index ℓ.
* `plot_series(lfc, theta=None, ax=None, **plot_kw)` – reconstruct the
  angular Boltzmann weight \(W(\theta)\) over the range
  \([-2\pi, 2\pi]\) and draw it as a line plot.

Both helpers return the `matplotlib.axes.Axes` object so that callers can
further customise the figure (titles, log–scales, etc.).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.special import sph_harm

__all__ = [
    "plot_coefficients",
    "plot_series",
]


# -----------------------------------------------------------------------------
# Scatter view of the coefficient vector
# -----------------------------------------------------------------------------

def plot_coefficients(
    lfc: ArrayLike | list[ArrayLike],
    *,
    ax: plt.Axes | None = None,
    show: bool = True,
    labels: list[str] | None = None,
    **scatter_kw,
) -> plt.Axes:
    """Scatter‐plot the LFC entries *lfc* against their harmonic index ℓ.

    Parameters
    ----------
    lfc
        One–dimensional array of Legendre–Fourier coefficients, a 2D array (pool), or a list of such arrays.
    ax
        Existing matplotlib axes to draw on.  If *None* (default) the current
        axes from :pyfunc:`matplotlib.pyplot.gca` is used.
    show
        If *True* (default) call :pyfunc:`matplotlib.pyplot.show` before
        returning when *ax* was *None*.
    labels
        Optional list of labels for each LFC when plotting multiple.
    **scatter_kw
        Additional keyword arguments forwarded verbatim to
        :pymeth:`matplotlib.axes.Axes.scatter` (colour, marker style, …).
    """
    # Handle list of LFCs
    if isinstance(lfc, (list, tuple)) and not isinstance(lfc, np.ndarray):
        n_lfcs = len(lfc)
        if n_lfcs == 0:
            raise ValueError("lfc list is empty")
        if labels is not None and len(labels) != n_lfcs:
            raise ValueError("labels must match number of lfc arrays")
        if ax is None:
            ax = plt.gca()
            created_ax = True
        else:
            created_ax = False
        handles = []
        legend_labels = []
        for i, lfc_i in enumerate(lfc):
            arr = np.asarray(lfc_i, dtype=float)
            if arr.ndim == 1:
                x = np.arange(arr.size)
                y = arr
                y_err = None
            elif arr.ndim == 2:
                x = np.arange(arr.shape[1])
                y = arr.mean(axis=0)
                y_err = arr.std(axis=0)
            else:
                raise ValueError("Each lfc must be 1- or 2-D array")
            label = labels[i] if labels is not None else f"LFC {i+1}"
            if y_err is None:
                h = ax.scatter(x, y, label=label, **scatter_kw)
            else:
                h = ax.errorbar(x, y, yerr=y_err, fmt="o", capsize=3, label=label, **scatter_kw)
            handles.append(h)
            legend_labels.append(label)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel("coefficient value")
        ax.set_title("Legendre–Fourier coefficients")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.legend(frameon=False)
        if created_ax and show:
            plt.show()
        return ax

    # Single LFC or pool
    arr = np.asarray(lfc, dtype=float)

    # Determine if we have a single vector (1-D) or a pool (2-D)
    if arr.ndim == 1:
        x = np.arange(arr.size)
        y = arr
        y_err = None
    elif arr.ndim == 2:
        x = np.arange(arr.shape[1])
        y = arr.mean(axis=0)
        y_err = arr.std(axis=0)
    else:
        raise ValueError("lfc must be 1- or 2-D array")

    if ax is None:
        ax = plt.gca()
        created_ax = True
    else:
        created_ax = False

    if y_err is None:
        ax.scatter(x, y, **scatter_kw)
    else:
        # Error bars: ±1σ by default
        ax.errorbar(x, y, yerr=y_err, fmt="o", capsize=3, **scatter_kw)

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("coefficient value")
    ax.set_title("Legendre–Fourier coefficients" + (" (mean ± σ)" if y_err is not None else ""))
    ax.axhline(0.0, color="k", lw=0.5)

    if y_err is None:
        ax.legend(["coefficients"], frameon=False)
    else:
        ax.legend(["mean ± σ"], frameon=False)

    if created_ax and show:
        plt.show()
    return ax

# -----------------------------------------------------------------------------
# Reconstructed angular series
# -----------------------------------------------------------------------------

def _legendre_series(theta: np.ndarray, lfc: np.ndarray) -> np.ndarray:
    """Helper: evaluate sum_l L_l P_l(cos θ) for *theta* array."""
    # np.polynomial.legendre.legval expects the variable *x* = cos θ
    x = np.cos(theta)
    return np.polynomial.legendre.legval(x, lfc)


def plot_series(
    lfc: ArrayLike,
    *,
    theta: ArrayLike | None = None,
    ax: plt.Axes | None = None,
    n_pts: int = 1024,
    show: bool = True,
    **plot_kw,
) -> plt.Axes:
    """Plot the reconstructed Boltzmann weight *W(θ)* from *lfc*.

    The function plotted is

    $$ W(\theta) = \sum_{\ell=0}^{\ell_{\max}} L_{\ell} P_{\ell}(\cos\theta). $$

    Parameters
    ----------
    lfc
        LFC vector.
    theta
        Iterable of angular samples in *radians* at which to evaluate the
        series.  If *None* (default) a uniform grid over
        \([-2\pi, 2\pi]\) with *n_pts* points is used.
    ax
        Existing axes to draw on; created if *None*.
    n_pts
        Number of sample points when *theta* is *None*.
    show
        If *True* (default) call :pyfunc:`matplotlib.pyplot.show` when *ax* is
        created internally.
    **plot_kw
        Forwarded to :pyfunc:`matplotlib.axes.Axes.plot`.
    """
    coeffs = np.asarray(lfc, dtype=float)
    if theta is None:
        theta = np.linspace(-2.0 * np.pi, 2.0 * np.pi, n_pts)
    else:
        theta = np.asarray(theta, dtype=float)

    W = _legendre_series(theta, coeffs)

    if ax is None:
        ax = plt.gca()
        created_ax = True
    else:
        created_ax = False

    ax.plot(theta, W, **plot_kw)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$W(\theta)$")
    ax.set_title("Reconstructed series from LFCs")

    if created_ax and show:
        plt.show()
    return ax


def reconstruct_from_legendre(lfc, theta, phi, theta_p=np.pi/2, phi_p=0):
    """
    Reconstruct f(theta, phi) from Legendre coefficients using the spherical harmonics addition theorem.
    The second direction (theta', phi') is fixed (default: pi/2, 0).
    Note: scipy.special.sph_harm uses sph_harm(m, l, phi, theta) (phi first!).
    """
    lfc = np.asarray(lfc, dtype=float)
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    # Prepare meshgrid if needed
    if theta.ndim == 1 and phi.ndim == 1:
        theta, phi = np.meshgrid(theta, phi, indexing='ij')
    result = np.zeros_like(theta, dtype=complex)
    lmax = lfc.size - 1
    for l in range(lmax + 1):
        factor = 4 * np.pi / (2 * l + 1)
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            Y_lm_p = sph_harm(m, l, phi_p, theta_p)
            result += lfc[l] * factor * Y_lm * np.conj(Y_lm_p)
    return result.real


def plot_legendre_surface(
    lfc,
    *,
    theta=None,
    phi=None,
    theta_p=np.pi/2,
    phi_p=0,
    ax=None,
    n_theta=100,
    n_phi=100,
    show=True,
    **surf_kw,
) -> plt.Axes:
    """
    Plot a 3D surface reconstructed from Legendre coefficients using the spherical harmonics addition theorem.
    The second direction (theta', phi') is fixed (default: pi/2, 0).
    Note: scipy.special.sph_harm uses sph_harm(m, l, phi, theta) (phi first!).
    """
    if theta is None:
        theta = np.linspace(0, np.pi, n_theta)
    if phi is None:
        phi = np.linspace(-np.pi, np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    u = reconstruct_from_legendre(lfc, theta_grid, phi_grid, theta_p, phi_p)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        created_ax = True
    else:
        created_ax = False

    surf = ax.plot_surface(theta_grid, phi_grid, u, **surf_kw)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_zlabel(r"$u$")
    ax.set_title("Reconstructed surface from Legendre coefficients")

    if created_ax and show:
        plt.show()
    return ax 