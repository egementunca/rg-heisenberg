"""analysis.py
Thin analysis helpers that build on :class:`rgheisenberg.controller.RGController`.
No heavy numerics should live here; implementations will delegate computation
back to the controller.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List

from .controller import RGController


def normalize_pool(ctrl: RGController) -> np.ndarray:
    """Return the current normalized pool from *ctrl*.

    The controller keeps its pool row-normalised, so this function is a thin
    proxy that simply returns a copy; included for backward compatibility with
    previous analysis scripts.
    """
    return ctrl.get_pool()


def phase_diagram(ctrl: RGController, steps: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Run *steps* RG iterations and collect the pool trajectory.

    No plotting is performed here—only minimal data collection suitable for
    downstream visualisation routines.
    """
    pools: List[np.ndarray] = [ctrl.get_pool()]
    for _ in range(steps):
        ctrl.step()
        pools.append(ctrl.get_pool())
    return pools, np.asarray(ctrl._log_norms) 