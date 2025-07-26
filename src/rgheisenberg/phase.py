"""phase.py
High-level helpers for computing phase diagrams with :class:`RGController`.
The legacy repository contained many stand-alone scripts that each scanned a
single parameter (``p``, ``q``, ``Δ``, …) while bracket-searching the critical
coupling ``J``.  This module unifies those patterns behind a small set of
extensible utilities:

1. A generic binary-search routine :func:`find_critical_J` that, given a
   *predicate* operating on the final RG pool, returns the transition value of
   *J* at fixed external parameters.
2. Thin wrappers that replicate the old 2-D phase-diagram scans such as
   ``p-vs-Tc`` or ``Δ-vs-Tc`` by repeatedly calling the above.

Numerical heavylifting remains in :pymod:`rgheisenberg.kernels`; this layer is
pure Python and therefore easy to prototype and unit-test.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Tuple

import numpy as np

from .controller import RGController

__all__ = [
    "default_order_predicate",
    "default_nematic_predicate",
    "find_critical_J",
    "scan_phase_boundary",
    "find_critical_g",
    "scan_vertical_boundary",
]

# ---------------------------------------------------------------------------
# Default predicates
# ---------------------------------------------------------------------------


def default_order_predicate(pool: np.ndarray, thresh: float = 0.99) -> bool:
    """Return *True* if the system is *ordered* (magnetised) according to LFC₀.

    The criterion mirrors historical scripts: if the absolute value of the
    zeroth coefficient of *any* site deviates from unity we count the pool as
    ordered.  Users can pass their own predicate with alternate definitions.
    """

    return bool(np.any(np.abs(pool[:, 0]) < thresh))


def default_nematic_predicate(pool: np.ndarray, thresh: float = 15.37) -> bool:  # noqa: D401,E501
    """Heuristic nematic ordering criterion from legacy scripts."""

    avg_lfc = np.mean(np.abs(pool), axis=0)
    return float(np.sum(avg_lfc)) < thresh


# ---------------------------------------------------------------------------
# Core search routine
# ---------------------------------------------------------------------------


def _run_rg(
    ctrl: RGController,
    steps: int = 25,
    *,
    J: float,
    p: float,
    q: float,
    g: float = 0.0,
    annealed: bool = True,
) -> np.ndarray:
    """Initialise *ctrl* at coupling *J* and run *steps* RG iterations."""

    ctrl.initialize_pool(J=J, p=p, q=q)

    remaining = steps
    if annealed:
        ctrl.step_vacancy(J=J, g=g)
        remaining -= 1

    for _ in range(max(remaining, 0)):
        ctrl.step()

    return ctrl.get_pool()


def find_critical_J(
    *,
    ctrl_factory: Callable[[bool], RGController],
    predicate: Callable[[np.ndarray], bool] = default_order_predicate,
    J_lo: float,
    J_hi: float,
    p: float,
    q: float,
    g: float = 0.0,
    steps: int = 25,
    max_iter: int = 25,
    tol: float = 1e-3,
    annealed: bool = False,
    bond_move_first: bool = False,
) -> float:
    """Binary-search the critical *J* where *predicate* flips.

    Parameters
    ----------
    ctrl_factory
        Callable returning a *fresh* :class:`RGController` for each probe.
        Should accept a boolean parameter for bond_move_first.
    predicate
        Function evaluated on the final RG pool; should return *True* on the
        *ordered* side of the transition.
    J_lo, J_hi
        Initial bracketing interval.  The predicate *must* differ at the two
        ends (otherwise the search aborts).
    p, q, g
        Physical parameters forwarded to :py:meth:`RGController.initialize_pool`
        and :py:meth:`~RGController.step_vacancy`.
    steps
        Number of RG iterations per probe.
    tol
        Relative tolerance on the interval width.
    """

    pred_lo = predicate(_run_rg(ctrl_factory(bond_move_first), steps, J=J_lo, p=p, q=q, g=g, annealed=annealed))
    pred_hi = predicate(_run_rg(ctrl_factory(bond_move_first), steps, J=J_hi, p=p, q=q, g=g, annealed=annealed))

    if pred_lo == pred_hi:
        raise ValueError("Predicate does not change over the initial bracket.")

    for _ in range(max_iter):
        J_mid = 0.5 * (J_lo + J_hi)
        pred_mid = predicate(
            _run_rg(ctrl_factory(bond_move_first), steps, J=J_mid, p=p, q=q, g=g, annealed=annealed)
        )
        if pred_mid == pred_lo:
            J_lo = J_mid
            pred_lo = pred_mid
        else:
            J_hi = J_mid
            pred_hi = pred_mid
        if abs(J_hi - J_lo) / J_mid < tol:
            break

    return 0.5 * (J_lo + J_hi)


# ---------------------------------------------------------------------------
# Vertical scans: critical g at fixed J
# ---------------------------------------------------------------------------


def find_critical_g(
    *,
    ctrl_factory: Callable[[bool], RGController],
    predicate: Callable[[np.ndarray], bool] = default_order_predicate,
    g_lo: float,
    g_hi: float,
    J: float,
    p: float,
    q: float,
    steps: int = 25,
    max_iter: int = 25,
    tol: float = 1e-3,
    annealed: bool = False,
    bond_move_first: bool = False,
) -> float:
    """Binary-search the critical *g* (Δ/J) keeping *J* fixed."""

    pred_lo = predicate(_run_rg(ctrl_factory(bond_move_first), steps, J=J, p=p, q=q, g=g_lo, annealed=annealed))
    pred_hi = predicate(_run_rg(ctrl_factory(bond_move_first), steps, J=J, p=p, q=q, g=g_hi, annealed=annealed))

    if pred_lo == pred_hi:
        raise ValueError("Predicate does not change over the initial bracket.")

    for _ in range(max_iter):
        g_mid = 0.5 * (g_lo + g_hi)
        pred_mid = predicate(
            _run_rg(ctrl_factory(bond_move_first), steps, J=J, p=p, q=q, g=g_mid, annealed=annealed)
        )
        if pred_mid == pred_lo:
            g_lo = g_mid
            pred_lo = pred_mid
        else:
            g_hi = g_mid
            pred_hi = pred_mid
        if abs(g_hi - g_lo) / g_mid < tol:
            break

    return 0.5 * (g_lo + g_hi)


def scan_vertical_boundary(
    J_values: Sequence[float],
    *,
    ctrl_factory: Callable[[bool], RGController],
    fixed_params: dict,
    predicate: Callable[[np.ndarray], bool] = default_order_predicate,
    steps: int = 25,
    max_iter: int = 25,
    tol: float = 1e-3,
    annealed: bool = False,
    bond_move_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """For every *J* in ``J_values`` find critical *g* (vertical line)."""

    g_crit = np.empty_like(np.asarray(J_values, dtype=float))

    for i, J in enumerate(J_values):
        params = dict(fixed_params)
        g_crit[i] = find_critical_g(
            ctrl_factory=ctrl_factory,
            predicate=predicate,
            steps=steps,
            max_iter=max_iter,
            tol=tol,
            J=J,
            annealed=annealed,
            bond_move_first=bond_move_first,
            **params,
        )

    return np.asarray(J_values, dtype=float), g_crit


# ---------------------------------------------------------------------------
# Convenience wrappers for 2-D diagrams
# ---------------------------------------------------------------------------


def scan_phase_boundary(
    varied: Sequence[float],
    *,
    ctrl_factory: Callable[[bool], RGController],
    fixed_params: dict,  # J bounds + other physics params
    predicate: Callable[[np.ndarray], bool] = default_order_predicate,
    steps: int = 25,
    max_iter: int = 25,
    tol: float = 1e-3,
    annealed: bool = False,
    bond_move_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a critical line over *varied* parameter values.

    ``fixed_params`` must provide at least ``J_lo`` and ``J_hi`` to bracket the
    transition at *every* point along the scan, plus any of ``p``, ``q``, ``g``.
    Additional keys are ignored.
    """

    J_crit = np.empty_like(np.asarray(varied, dtype=float))

    for i, x in enumerate(varied):
        params = dict(fixed_params)  # shallow copy
        # Identify which physical coordinate is being scanned
        if "p" in params and np.isnan(params["p"]):
            params["p"] = x
        elif "q" in params and np.isnan(params["q"]):
            params["q"] = x
        elif "g" in params and np.isnan(params["g"]):
            params["g"] = x
        else:
            raise ValueError("Could not determine which parameter is varied.")

        J_crit[i] = find_critical_J(
            ctrl_factory=ctrl_factory,
            predicate=predicate,
            steps=steps,
            max_iter=max_iter,
            tol=tol,
            annealed=annealed,
            bond_move_first=bond_move_first,
            **params,
        )

    return np.asarray(varied, dtype=float), J_crit 