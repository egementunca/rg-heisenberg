"""controller.py
High-level orchestration of Renormalization-Group (RG) flows for the classical
Heisenberg model. The controller keeps track of a *pool* of Legendre–Fourier
coefficients (LFCs), applies decimation and bond-move kernels, and exposes
thermodynamic quantities such as free energy, internal energy, and specific
heat.

Only controller logic lives here—heavy numerics are delegated to
:pyfile:`rgheisenberg.kernels`.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import kernels

__all__ = ["RGController"]


class RGController:
    """Manage RG pool state and provide thermodynamic quantities."""

    # ------------------------------------------------------------------
    # Constructor & helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        l_prec: int,
        pool_size: int,
        dim: int,
        n_bm: int,
        seed: int = 17,
        dtype: str | np.dtype = "float64",
        norm: str = "max",  # 'max' (L∞) or 'l2'
    ) -> None:
        self.l_prec = int(l_prec)
        self.pool_size = int(pool_size)
        self.dim = int(dim)
        self.n_bm = int(n_bm)
        self.dtype = np.dtype(dtype)
        if norm not in {"max", "l2"}:
            raise ValueError("norm must be 'max' or 'l2'")
        self._norm_type = norm
        self._seed = int(seed)

        # Dedicated RNG – reproducibility & thread-safety
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Pool and bookkeeping
        self._pool: np.ndarray | None = None  # (pool_size, l_prec)
        self._log_norms: List[float] = []  # store one value per generation for analysis compat
        self._cur_gen_sum: float = 0.0
        # Aggregate log-norm subtotal per RG generation (G_n)
        self._gen_log_sums: List[float] = []
        self._steps: int = 0  # number of RG iterations performed

        # Vacancy tracking (Δ parameter of the bond-dilution model)
        self._delta: float | None = None  # Updated each generation when vacancies are enabled

        # Scratch attribute used by finite-difference helpers
        self._current_J: float | None = None

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _record_ln(self, ln: float) -> None:
        """Accumulate *ln* into the current generation subtotal."""
        self._cur_gen_sum += ln

    def _normalize_vec(self, vec: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return normalised *vec* and log(norm) according to *self._norm_type*."""
        if self._norm_type == "max":
            norm_val = np.amax(np.abs(vec))
        else:  # 'l2'
            norm_val = np.linalg.norm(vec)

        if norm_val == 0.0:
            return vec.copy(), 0.0
        return vec / norm_val, float(np.log(norm_val))

    def _decimate(self, l1: np.ndarray, l2: np.ndarray, l3: np.ndarray) -> np.ndarray:
        """Return decimated LFCs, accounting for vacancies if enabled."""
        if self._delta is None:
            res, ln = kernels.decimate3(l1, l2, l3)
        else:
            res, ln = kernels.decimateVacancy(l1, l2, l3, float(self._delta))
            # accumulate lnΛ for Δ update
            self._ln_lambda_sum += ln
            self._ln_lambda_cnt += 1

        # Re-normalise if using L2
        if self._norm_type == "l2":
            raw = res * np.exp(ln)
            norm_val = np.linalg.norm(raw)
            if norm_val != 0.0:
                res = raw / norm_val
                ln = np.log(norm_val)

        self._record_ln(float(ln))
        return res

    def _bond_move(self, l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
        # Vacancy bonds (identity) do not contribute to bond moves.
        if self._delta is None:
            res, ln = kernels.bond_move(l1, l2)
            # Re-normalise if needed
            if self._norm_type == "l2":
                raw = res * np.exp(ln)
                norm_val = np.linalg.norm(raw)
                if norm_val != 0.0:
                    res = raw / norm_val
                    ln = np.log(norm_val)

            self._record_ln(float(ln))
            return res

        # Simple rule: if either operand is identity coupling (vacancy), pass through the other.
        # Identity vector has l=0 component 1 and others 0 (within tolerance).
        def _is_identity(v: np.ndarray) -> bool:
            return np.isclose(v[0], 1.0) and np.allclose(v[1:], 0.0)

        if _is_identity(l1):
            out = l2.copy()
            ln = 0.0
        elif _is_identity(l2):
            out = l1.copy()
            ln = 0.0
        else:
            out, ln = kernels.bond_move(l1, l2)
            if self._norm_type == "l2":
                raw = out * np.exp(ln)
                norm_val = np.linalg.norm(raw)
                if norm_val != 0.0:
                    out = raw / norm_val
                    ln = np.log(norm_val)

        # Accumulate lnΛ statistics for Δ update if active
        if self._delta is not None:
            self._ln_lambda_sum += ln
            self._ln_lambda_cnt += 1

        self._record_ln(float(ln))
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # 0. Pool initialisation -------------------------------------------------
    def initialize_pool(self, J: float, p: float, q: float) -> None:
        """Create the initial LFC pool.

        Parameters
        ----------
        J
            Exchange coupling (positive for ferromagnetic).
        p
            Probability of an antiferromagnetic (+/-J) bond.
        q
            Fraction of vacancies (broken bond → identity coupling).
        """
        ferro = kernels.lfc_initialize(J, self.l_prec).astype(self.dtype)
        anti = kernels.lfc_initialize(-J, self.l_prec).astype(self.dtype)
        vacancy = np.zeros(self.l_prec, dtype=self.dtype)
        vacancy[0] = 1.0  # identity coupling

        pool = np.empty((self.pool_size, self.l_prec), dtype=self.dtype)

        # Determine counts
        v_count = int(round(q * self.pool_size))
        remaining = self.pool_size - v_count
        ferro_count = int(round((1.0 - p) * remaining))

        pool[:v_count] = vacancy
        pool[v_count : v_count + ferro_count] = ferro
        pool[v_count + ferro_count :] = anti

        # Shuffle so that RNG draws see mixed distribution
        self._rng.shuffle(pool, axis=0)

        # Normalise each row and record log-norms
        # Keep track of physical parameters for finite-difference helpers
        self._init_p = float(p)
        self._init_q = float(q)
        self._init_g: float | None = None  # populated by step_vacancy

        self._log_norms.clear()
        self._gen_log_sums.clear()
        self._delta = None  # reset vacancy parameter
        norm_pool = np.empty_like(pool)
        for i in range(self.pool_size):
            norm_pool[i], ln = self._normalize_vec(pool[i])
            self._record_ln(ln)

        self._pool = norm_pool
        self._steps = 0

        # Record initial generation (n=0) subtotal G_0
        self._gen_log_sums.append(self._cur_gen_sum)
        self._log_norms.append(self._cur_gen_sum)
        self._cur_gen_sum = 0.0

    # 1. First RG step (vacancy model) -------------------------------------
    def step_vacancy(self, J: float, g: float) -> None:
        if self._pool is None:
            raise RuntimeError("Pool not initialised. Call initialize_pool() first.")

        # Initial Δ parameter according to legacy definition
        delta = g * J
        self._delta = float(delta)
        new_pool = np.empty_like(self._pool)

        # Track lnΛ statistics for Δ update
        ln_sum = 0.0

        for i in range(self.pool_size):
            ids = self._rng.integers(0, self.pool_size, size=3)
            l1, l2, l3 = self._pool[ids[0]], self._pool[ids[1]], self._pool[ids[2]]
            l_dec, ln = kernels.decimateVacancy(l1, l2, l3, float(delta))
            if self._norm_type == "l2":
                raw = l_dec * np.exp(ln)
                norm_val = np.linalg.norm(raw)
                if norm_val != 0.0:
                    l_dec = raw / norm_val
                    ln = np.log(norm_val)

            new_pool[i] = l_dec
            self._record_ln(ln)
            ln_sum += ln

        self._pool = new_pool

        gen_ln_avg = ln_sum / self.pool_size

        # Accumulate generation subtotal (G_n)
        self._gen_log_sums.append(self._cur_gen_sum)
        self._log_norms.append(self._cur_gen_sum)
        self._cur_gen_sum = 0.0

        # Update Δ for use in subsequent generations (Δ' = 2Δ - lnΛ)
        self._delta = 2.0 * self._delta - gen_ln_avg

        self._init_g = float(g)
        self._steps += 1

    # 2. Subsequent RG steps -------------------------------------------------
    def step(self) -> None:
        if self._pool is None:
            raise RuntimeError("Pool not initialised.")

        # Prepare lnΛ accumulators for Δ update when vacancies are active
        self._ln_lambda_sum = 0.0
        self._ln_lambda_cnt = 0

        # Decimation phase
        dec_pool = np.empty_like(self._pool)
        for i in range(self.pool_size):
            ids = self._rng.integers(0, self.pool_size, size=3)
            dec_pool[i] = self._decimate(
                self._pool[ids[0]], self._pool[ids[1]], self._pool[ids[2]]
            )

        # Bond-move phases (n_bm − 1 iterations following decimation)
        current = dec_pool
        for _ in range(max(0, self.n_bm - 1)):
            nxt = np.empty_like(current)
            for j in range(self.pool_size):
                idx1, idx2 = self._rng.integers(0, self.pool_size, size=2)
                nxt[j] = self._bond_move(current[idx1], current[idx2])
            current = nxt

        self._pool = current

        # Record subtotal for this generation
        self._gen_log_sums.append(self._cur_gen_sum)
        self._log_norms.append(self._cur_gen_sum)
        self._cur_gen_sum = 0.0

        # Update Δ if vacancies are active and decimations were performed
        if self._delta is not None and self._ln_lambda_cnt > 0:
            ln_avg = self._ln_lambda_sum / self._ln_lambda_cnt
            self._delta = 2.0 * self._delta - ln_avg
        self._steps += 1

    # 3. Thermodynamic quantities -----------------------------------------
    def free_energy(self) -> float:
        """Return pooled free-energy density (arbitrary units)."""
        # Series f = Σ_n G_n / b^{d n}
        if not self._gen_log_sums:
            return 0.0

        b = max(self.n_bm, 1)
        scale = float(b) ** self.dim

        accum = 0.0
        factor = 1.0  # b^{d*0} for n = 0
        for G_n in self._gen_log_sums:
            accum += G_n / factor
            factor *= scale  # prepare b^{d*(n+1)}

        return float(accum)

    # Finite-difference helpers
    def _eval_f(self, J: float) -> float:
        """Estimate free-energy density at coupling *J* by rerunning RG.

        A fresh controller is built with the original parameters and the same
        RNG seed to minimise statistical noise.  The RG trajectory is
        replayed for the same number of iterations as the parent instance and
        the resulting free energy returned.
        """

        replica = RGController(
            l_prec=self.l_prec,
            pool_size=self.pool_size,
            dim=self.dim,
            n_bm=self.n_bm,
            seed=self._seed,
            dtype=self.dtype,
            norm=self._norm_type,
        )

        replica.initialize_pool(J=J, p=self._init_p, q=self._init_q)

        # Sync RNG state for variance reduction
        try:
            replica._rng.bit_generator.state = self._rng.bit_generator.state.copy()
        except AttributeError:
            replica._rng.bit_generator.state = self._rng.bit_generator.state

        if self._init_g is not None:
            replica.step_vacancy(J=J, g=self._init_g)
            remaining = max(self._steps - 1, 0)
        else:
            remaining = self._steps

        for _ in range(remaining):
            replica.step()

        return float(replica.free_energy())

    def internal_energy(self, J: float, dJ: float = 1e-4) -> float:
        """Central finite difference of *f* with respect to *J*."""
        return (self._eval_f(J + dJ) - self._eval_f(J - dJ)) / (2.0 * dJ)

    def specific_heat(self, J: float, dJ: float = 1e-4) -> float:
        """Second finite difference of *f* with respect to *J*."""
        f_plus = self._eval_f(J + dJ)
        f_0 = self._eval_f(J)
        f_minus = self._eval_f(J - dJ)
        return (f_plus - 2.0 * f_0 + f_minus) / (dJ ** 2)

    # 4. Introspection ------------------------------------------------------
    def get_pool(self) -> np.ndarray:
        """Return a *copy* of the current pool (row-normalised)."""
        if self._pool is None:
            raise RuntimeError("Pool not initialised.")
        return self._pool.copy() 