"""analysis.py
Thin analysis helpers that build on :class:`rgheisenberg.controller.RGController`.
No heavy numerics should live here; implementations will delegate computation
back to the controller.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List
import warnings

from .controller import RGController


def normalize_pool(ctrl: RGController) -> np.ndarray:
    """Return the current normalized pool from *ctrl*.

    The controller keeps its pool row-normalised, so this function is a thin
    proxy that simply returns a copy; included for backward compatibility with
    previous analysis scripts.
    """
    return ctrl.get_pool()


def pool_trajectory(ctrl: RGController, steps: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return list of pool snapshots and log‐norms after *steps* RG iterations."""

    pools: List[np.ndarray] = [ctrl.get_pool()]
    for _ in range(steps):
        ctrl.step()
        pools.append(ctrl.get_pool())
    return pools, np.asarray(ctrl._log_norms)


def phase_diagram(ctrl: RGController, steps: int):  # noqa: D401
    """DEPRECATED – use :pyfunc:`pool_trajectory` instead."""
    warnings.warn(
        "analysis.phase_diagram() is deprecated; use pool_trajectory() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return pool_trajectory(ctrl, steps) 