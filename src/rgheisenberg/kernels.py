"""kernels.py
Numba-accelerated mathematical kernels for the classical Heisenberg model.

Only the heavy triple-loop sections are wrapped with ``numba.njit`` to keep
Python-level call overhead minimal while still allowing easy unit testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numba as nb
import numpy as np
from scipy.special import spherical_jn

__all__ = [
    "lfc_initialize",
    "bond_move",
    "bond_move_log",
    "decimate3",
    "decimateVacancy",
]

# ---------------------------------------------------------------------------
# Clebsch–Gordan coefficients
# ---------------------------------------------------------------------------
# These coefficients are stored externally to keep the repository size small.
# We attempt to locate them relative to this file; users should place
# ``cleb.npy`` in one of the searched paths or provide their own.
_CLEB_PATHS: Final[list[Path]] = [
    Path(__file__).with_name("cleb.npy"),  # same directory
    Path(__file__).parent / "../data/cleb.npy",  # legacy location
]

_cg: np.ndarray
for _p in _CLEB_PATHS:
    try:
        _cg = np.load(_p)
        break
    except FileNotFoundError:
        continue
else:
    # --- regenerate Gaunt tensor -------------------------------------------
    from sympy.physics.wigner import wigner_3j
    LMAX = 50                       # keep 50 – same as old data file
    _cg = np.zeros((LMAX + 1, LMAX + 1, LMAX + 1), dtype=float)

    for l1 in range(LMAX + 1):
        for l2 in range(LMAX + 1):
            for l in range(abs(l1 - l2), min(l1 + l2, LMAX) + 1):
                w = float(wigner_3j(l1, l2, l, 0, 0, 0))
                # Gaunt formula:  G = 2 * (3-j)^2
                _cg[l1, l2, l] = 2.0 * w * w

    np.save(Path(__file__).with_name("cleb.npy"), _cg)

clebsch_gordan: Final[np.ndarray] = _cg

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def lfc_initialize(J: float, l_prec: int) -> np.ndarray:
    """Return Legendre–Fourier coefficients for coupling *J* (pure Python)."""
    x = np.arange(l_prec)
    return np.real((2 * x + 1) * (1j ** x) * spherical_jn(x, -1j * J))


# ---------------------------------------------------------------------------
# Bond–move kernels
# ---------------------------------------------------------------------------


@nb.njit(fastmath=True)
def _bond_move_core(lfc1: np.ndarray, lfc2: np.ndarray, cg: np.ndarray):  # noqa: D401
    l_prec = lfc1.shape[0]
    out = np.zeros(l_prec)
    cg_shape = cg.shape
    for l in nb.prange(l_prec):
        val = 0.0
        for l1 in range(l_prec):
            for l2 in range(l_prec):
                if (
                    l1 < cg_shape[0]
                    and l2 < cg_shape[1]
                    and l < cg_shape[2]
                ):
                    coeff = cg[l1, l2, l]
                else:
                    coeff = 0.0  # out-of-range → no coupling contribution
                if coeff != 0.0:
                    val += lfc1[l1] * lfc2[l2] * coeff
        out[l] = val
    return out


def bond_move(lfc1: np.ndarray, lfc2: np.ndarray):  # noqa: D401
    """Bond-move kernel returning *normalized* LFCs and ``log(norm)``."""
    raw = _bond_move_core(lfc1, lfc2, clebsch_gordan)
    y = np.amax(np.abs(raw))
    return raw / y, float(np.log(y))


# Pure log-space version – useful when magnitudes span many orders.
@nb.njit(fastmath=True)
def bond_move_log(lfc1: np.ndarray, lfc2: np.ndarray):  # noqa: D401
    l_prec = lfc1.shape[0]
    log_out = np.full(l_prec, -np.inf)

    log_lfc1 = np.log(np.abs(lfc1) + 1e-300)
    log_lfc2 = np.log(np.abs(lfc2) + 1e-300)
    log_cg = np.log(np.abs(clebsch_gordan) + 1e-300)

    for l in nb.prange(l_prec):
        acc = -np.inf
        for l1 in range(l_prec):
            for l2 in range(l_prec):
                term = log_lfc1[l1] + log_lfc2[l2] + log_cg[l1, l2, l]
                acc = np.logaddexp(acc, term)
        log_out[l] = acc

    max_log_val = np.amax(log_out)
    log_out -= max_log_val
    return np.exp(log_out), max_log_val


# ---------------------------------------------------------------------------
# Decimation kernels
# ---------------------------------------------------------------------------


@nb.njit(fastmath=True)
def decimate3(lfc1: np.ndarray, lfc2: np.ndarray, lfc3: np.ndarray):  # noqa: D401
    l_prec = lfc1.shape[0]
    odd = 2 * np.arange(l_prec) + 1
    dec = (lfc1 * lfc2 * lfc3) / (odd * odd)
    x = np.amax(np.abs(dec))
    return dec / x, float(np.log(x))


@nb.njit()
def decimateVacancy(
    lfc1: np.ndarray, lfc2: np.ndarray, lfc3: np.ndarray, delta: float
):  # noqa: D401
    l_prec = lfc1.shape[0]
    odd = 2 * np.arange(l_prec) + 1

    dec = (lfc1 * lfc2 * lfc3) / (odd * odd)
    combined = np.exp(-4.0 * delta) * dec

    base = np.zeros(l_prec)
    base[0] = 1.0

    aux2, aux3 = np.zeros(l_prec), np.zeros(l_prec)
    aux2[0] = np.exp(-2.0 * delta) * lfc2[0]
    aux3[0] = np.exp(-2.0 * delta) * lfc3[0]

    lfc = combined + base + aux2 + aux3
    x = np.amax(np.abs(lfc))
    return lfc / x, float(np.log(x)) 