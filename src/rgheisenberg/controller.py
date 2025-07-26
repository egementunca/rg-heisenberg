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

__all__ = ["RGController", "RGControllerB2"]

# ---------------------------------------------------------------------------
# Numerical safeguards
# ---------------------------------------------------------------------------

# Machine epsilon for numerical differentiation
import math
_MACHINE_EPS: float = 2.220446049250313e-16  # np.finfo(float).eps

# Lattice rescaling factor (b); hard‐coded
_B_FACTOR: int = 3


class RGController:
    """Manage RG pool state and provide thermodynamic quantities.
    
    Each RG step consists of decimation and bond move operations. By default,
    decimation is performed first, followed by bond moves. The order can be
    reversed using the `bond_move_first` parameter, but this feature is only
    available for non-annealed systems (when q=0, no vacancies).
    """

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
        bond_move_first: bool = False,  # If True, do bond move before decimation
    ) -> None:
        self.l_prec = int(l_prec)
        """Number of Legendre–Fourier coefficients per bond."""
        self.pool_size = int(pool_size)
        """Number of bonds in the pool."""
        self.dim = int(dim)
        """Lattice dimension."""
        self.n_bm = int(n_bm)
        """Number of bond moves per generation."""
        self.dtype = np.dtype(dtype)
        """Data type for the pool."""
        self._seed = int(seed)
        """Random seed."""
        self._bond_move_first = bool(bond_move_first)
        """Swap the MK order (bond move -> decimation). Only valid for non-vacancy systems."""
        self._rng: np.random.Generator = np.random.default_rng(seed)
        """Dedicated RNG – reproducibility & thread-safety"""        
        self._pool: np.ndarray | None = None
        """Pool of Legendre–Fourier coefficients."""
        self._log_norms: List[float] = []
        """Store one value per generation for analysis compat"""
        self._cur_gen_sum: float = 0.0
        """Aggregate log-norm subtotal per RG generation (G_n)"""
        self._gen_log_sums: List[float] = []
        """Number of RG iterations performed"""
        self._steps: int = 0

        """Vacancy tracking (per-bond Δ parameters for the bond-dilution model)"""
        """When vacancies are inactive this attribute stays None; once step_vacancy() is called it becomes a 1-D float array parallel to *self._pool*."""
        self._delta_pool: np.ndarray | None = None
        """shape: (pool_size,)"""

        """Scratch attribute used by finite-difference helpers"""
        self._current_J: float | None = None

        """Identity coupling vector (cached)"""
        self._identity_vec: np.ndarray | None = None
        
        """Initialization parameters for finite-difference helpers"""
        self._init_p: float | None = None
        self._init_q: float | None = None
        self._init_J: float | None = None
        self._init_g: float | None = None

        """RNG state history – list of bit_generator.state snapshots *before* each RG generation (index 0 corresponds to the state immediately after initialise_pool()).  This enables common-random-numbers when replaying the trajectory at slightly perturbed couplings."""
        self._rng_history: list[dict] = []

        # Note: bond_move_first validation is deferred until step_vacancy() is called
        # since we don't know at construction time whether vacancies will be used

    
    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _record_ln(self, ln: float) -> None:
        """Accumulate *ln* into the current generation subtotal."""
        self._cur_gen_sum += ln

    def _normalize_vec(self, vec: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return normalised *vec* and log(max norm)."""
        norm_val = np.amax(np.abs(vec))
        if norm_val == 0.0:
            return vec.copy(), 0.0
        return vec / norm_val, float(np.log(norm_val))

    def _decimate(self, l1: np.ndarray, l2: np.ndarray, l3: np.ndarray, delta: float = 0.0) -> np.ndarray:
        """Return decimated LFCs, accounting for vacancies if enabled.
        
        Parameters
        ----------
        l1, l2, l3 : np.ndarray
            Input LFC vectors for decimation
        delta : float, default=0.0
            Vacancy parameter (only used when vacancies are active)
        """
        if self._delta_pool is None:
            # decimate3 returns RAW coefficients and ln‖·‖∞
            raw, ln0 = kernels.decimate3(l1, l2, l3)
            norm_val = np.exp(ln0)
            res = raw / norm_val if norm_val else raw
            ln = ln0  # Record the original log norm of the raw result
        else:
            res, ln = kernels.decimateVacancy(l1, l2, l3, delta)

        self._record_ln(float(ln))
        return res

    def _bond_move(self, l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
        # Vacancy bonds (identity) do not contribute to bond moves.
        if self._delta_pool is None:
            res, ln = kernels.bond_move(l1, l2)
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

        # Note: Δ statistics are computed directly in step() method when needed

        self._record_ln(float(ln))
        return out

    # ------------------------------------------------------------------
    # Vacancy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vacancy_prob(delta: float) -> float:
        """Return probability that a bond with *Δ* becomes a vacancy."""
        if delta >= 200.0:
            return 0.0
        if delta <= -200.0:
            return 1.0
        x = np.exp(-delta)
        # probFunc from legacy code: e^{-2Δ}/(1 + 2 e^{-Δ} + e^{-2Δ})
        return (x**2) / (1.0 + 2.0 * x + x**2)

    def _maybe_vacate(self, lfc: np.ndarray, delta: float) -> np.ndarray:
        """Return *lfc* or the identity vector depending on vacancy prob."""
        if self._rng.random() < self._vacancy_prob(delta):
            if self._identity_vec is None:
                self._identity_vec = np.zeros(self.l_prec, dtype=self.dtype)
                self._identity_vec[0] = 1.0
            return self._identity_vec
        return lfc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # 0. Pool initialisation -------------------------------------------------
    def initialize_pool(self, J: float, p: float, q: float, normalize_initial: bool = True) -> None:
        """Create the initial LFC pool.

        Parameters
        ----------
        J
            Exchange coupling (positive for ferromagnetic).
        p
            Probability of an antiferromagnetic (+/-J) bond.
        q
            Fraction of vacancies (broken bond → identity coupling).
        normalize_initial
            If True (default), normalize the initial LFCs. If False, keep raw LFCs but still record ln(norm) for free energy bookkeeping.
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

        self._init_p = float(p)
        self._init_q = float(q)
        self._init_J = float(J)  # Store J for pool_size=1 case
        self._init_g: float | None = None  # populated by step_vacancy

        self._log_norms.clear()
        self._gen_log_sums.clear()
        self._delta_pool = None  # reset vacancy parameter
        norm_pool = np.empty_like(pool)
        
        # Record only ONE initial log norm (like notebook)
        initial_log_norm = 0.0
        for i in range(self.pool_size):
            if normalize_initial:
                norm_pool[i], ln = self._normalize_vec(pool[i])
            else:
                ln = self._normalize_vec(pool[i])[1]
                norm_pool[i] = pool[i].copy()
            # Use log norm from last pool entry (matches notebook single LFC approach)
            initial_log_norm = ln
        
        # Record only ONE log norm for initial generation
        self._record_ln(initial_log_norm)

        self._pool = norm_pool
        self._steps = 0

        # Record RNG state so that generation 0 can be replayed verbatim.
        self._rng_history = [self._rng.bit_generator.state.copy()]

        # Record initial generation (n=0) subtotal G_0
        self._gen_log_sums.append(self._cur_gen_sum)
        self._log_norms.append(self._cur_gen_sum)
        self._cur_gen_sum = 0.0

        # Keep track of physical parameters for finite-difference helpers
        self._init_p = float(p)
        self._init_q = float(q)
        self._init_g: float | None = None  # populated by step_vacancy

        self._delta_pool = None  # reset vacancy parameter

    # 1. First RG step (vacancy model) -------------------------------------
    def step_vacancy(self, J: float, g: float) -> None:
        """First generation for the vacancy model (global Δ = g·J)."""

        # Validate bond_move_first compatibility with vacancy systems
        if self._bond_move_first:
            raise ValueError("bond_move_first=True is not supported with vacancy systems. Use bond_move_first=False.")

        # Snapshot RNG state *before* any random draws for this generation.
        self._rng_history.append(self._rng.bit_generator.state.copy())

        if self._pool is None:
            raise RuntimeError("Pool not initialised. Call initialize_pool() first.")

        delta0 = float(g * J)  # legacy global Δ definition

        new_pool = np.empty_like(self._pool)
        new_delta = np.empty(self.pool_size, dtype=float)

        ln_sum = 0.0  # accumulate lnΛ for generation subtotal

        for i in range(self.pool_size):
            ids = self._rng.integers(0, self.pool_size, size=3)
            l1, l2, l3 = (
                self._pool[ids[0]],
                self._pool[ids[1]],
                self._pool[ids[2]],
            )

            res, ln = kernels.decimateVacancy(l1, l2, l3, delta0)

            new_pool[i] = res
            new_delta[i] = 2.0 * delta0 - ln  # Δ' formula for first generation

            self._record_ln(float(ln))
            ln_sum += ln

        # Commit new state
        self._pool = new_pool
        self._delta_pool = new_delta

        # Generation bookkeeping
        self._gen_log_sums.append(self._cur_gen_sum)
        self._log_norms.append(self._cur_gen_sum)
        self._cur_gen_sum = 0.0

        self._init_g = float(g)
        self._steps += 1

    # 2. Subsequent RG steps -------------------------------------------------
    def step(self, n: int = 1) -> None:
        """Perform *n* subsequent RG generations (default 1).

        Calling :py:meth:`step` with an integer argument greater than one
        conveniently executes that many RG iterations back-to-back, saving
        the caller from writing an explicit loop.  For ``n == 1`` the
        behaviour is identical to earlier versions of the library.
        """

        if n < 1:
            raise ValueError("n must be a positive integer")
        
        # Validate bond_move_first compatibility - only check when vacancies are active
        if self._delta_pool is not None and self._bond_move_first:
            raise ValueError("bond_move_first=True is not supported with vacancy systems. Use bond_move_first=False.")

        # Fast-path for multiple iterations – delegate to the single-step
        # implementation below.
        if n > 1:
            for _ in range(n):
                # Recursive call with the default n=1 executes exactly one
                # generation each time.
                self.step()
            return

        # Snapshot RNG state for common-random-numbers replay.
        self._rng_history.append(self._rng.bit_generator.state.copy())

        # -----------------------------------------------------------------
        # Single-generation implementation (legacy code path)
        # -----------------------------------------------------------------

        if self._pool is None:
            raise RuntimeError("Pool not initialised.")

        if self._delta_pool is None:
            # ---------------- Non-annealed path: no vacancies -----------------
            # Order of decimation and bond move can be controlled here

            if self._bond_move_first and self.n_bm > 1:
                # Bond move phase first (no vacancies, only if n_bm > 1)
                base_pool = self._pool
                current = self._pool
                for _ in range(max(0, self.n_bm - 1)):
                    nxt = np.empty_like(current)
                    for j in range(self.pool_size):
                        idx1 = self._rng.integers(0, self.pool_size)   # from base_pool
                        idx2 = self._rng.integers(0, self.pool_size)   # from evolving pool
                        nxt[j] = self._bond_move(base_pool[idx1], current[idx2])
                    current = nxt
                input_for_decimation = current
            else:
                # No bond move phase if n_bm == 1
                input_for_decimation = self._pool

            # Decimation phase (no vacancies)
            dec_pool = np.empty_like(input_for_decimation)
            for i in range(self.pool_size):
                idxs = self._rng.integers(0, self.pool_size, size=3)
                dec_pool[i] = self._decimate(
                    input_for_decimation[idxs[0]], input_for_decimation[idxs[1]], input_for_decimation[idxs[2]], 0.0
                )
            self._pool = dec_pool

            if not self._bond_move_first and self.n_bm > 1:
                # Decimation first, then bond moves (only if n_bm > 1)
                base_pool = self._pool
                current = self._pool
                for _ in range(max(0, self.n_bm - 1)):
                    nxt = np.empty_like(current)
                    for j in range(self.pool_size):
                        idx1 = self._rng.integers(0, self.pool_size)   # from base_pool
                        idx2 = self._rng.integers(0, self.pool_size)   # from evolving pool
                        nxt[j] = self._bond_move(base_pool[idx1], current[idx2])
                    current = nxt
                self._pool = current

            # Record subtotal and finish
            self._gen_log_sums.append(self._cur_gen_sum)
            self._log_norms.append(self._cur_gen_sum)
            self._cur_gen_sum = 0.0

            self._steps += 1
            return

        # Decimation ----------------------------------------------------
        new_pool = np.empty_like(self._pool)
        new_delta = np.empty_like(self._delta_pool)

        for i in range(self.pool_size):
            idxs = self._rng.integers(0, self.pool_size, size=3)
            l1, d1 = self._pool[idxs[0]], self._delta_pool[idxs[0]]
            l2, d2 = self._pool[idxs[1]], self._delta_pool[idxs[1]]
            l3, d3 = self._pool[idxs[2]], self._delta_pool[idxs[2]]

            l1v = self._maybe_vacate(l1, d1)
            l2v = self._maybe_vacate(l2, d2)
            l3v = self._maybe_vacate(l3, d3)

            res, ln = kernels.decimate3(l1v, l2v, l3v)

            new_pool[i] = res
            new_delta[i] = d1 + d3 - ln  # Δ' for decimation

            self._record_ln(float(ln))

        # Bond-move phases --------------------------------------------
        base_pool = new_pool            # stays constant (decimation output)
        base_delta = new_delta

        current_pool = new_pool         # evolves with each BM iteration
        current_delta = new_delta

        for _ in range(max(0, self.n_bm - 1)):
            next_pool = np.empty_like(current_pool)
            next_delta = np.empty_like(current_delta)

            for j in range(self.pool_size):
                # Operand 1: always drawn from *base_pool* (original dec outcome)
                idx1 = self._rng.integers(0, self.pool_size)
                l1, d1 = base_pool[idx1], base_delta[idx1]

                # Operand 2: drawn from the evolving pool of previous BM step
                idx2 = self._rng.integers(0, self.pool_size)
                l2, d2 = current_pool[idx2], current_delta[idx2]

                # Strict annealed schedule – NO *second* vacancy sampling
                # inside the bond-move loop.  Operands have already been
                # converted to vacancies (identity couplings) during the
                # decimation stage if required.
                res, ln = kernels.bond_move(l1, l2)

                # Track log‐norms for thermodynamic bookkeeping
                self._record_ln(float(ln))

                next_pool[j] = res
                next_delta[j] = d1 + d2 - ln  # Δ' for bond move

            current_pool, current_delta = next_pool, next_delta

        # Commit generation -------------------------------------------
        self._pool = current_pool
        self._delta_pool = current_delta

        self._gen_log_sums.append(self._cur_gen_sum)
        self._log_norms.append(self._cur_gen_sum)
        self._cur_gen_sum = 0.0

        self._steps += 1

    # 3. Thermodynamic quantities -----------------------------------------
    def free_energy(self) -> float:
        """Return pooled free-energy density via series Σ G_n / (b^d)^n."""

        if not self._gen_log_sums:
            return 0.0

        scale = float(_B_FACTOR) ** self.dim  # (b^d)

        accum = 0.0
        factor = 1.0  # (b^d)^0 for n = 0
        for G_n in self._gen_log_sums:
            accum += G_n / factor
            factor *= scale  # update to (b^d)^{n+1}

        return float(accum)

    # Finite-difference helpers
    def _eval_f(
        self,
        J: float,
        *,
        seed: int | None = None,
        inherit_state: bool = True,
    ) -> float:
        """Estimate free-energy density at coupling *J* by rerunning RG.

        A fresh controller is built with the original parameters and the same
        RNG seed to minimise statistical noise.  The RG trajectory is
        replayed for the same number of iterations as the parent instance and
        the resulting free energy returned.
        
        For consistent specific heat calculations, always use inherit_state=True
        and consistent seeding to minimize finite difference noise.
        """

        replica = RGController(
            l_prec=self.l_prec,
            pool_size=self.pool_size,
            dim=self.dim,
            n_bm=self.n_bm,
            seed=self._seed if seed is None else int(seed),
            dtype=self.dtype,

            bond_move_first=self._bond_move_first,
        )

        # Always use stored initialization parameters for consistency
        p = getattr(self, '_init_p', 0.0) or 0.0
        q = getattr(self, '_init_q', 0.0) or 0.0
        replica.initialize_pool(J=J, p=p, q=q)

        # Always use inherit_state=True for finite differences to minimize noise
        if inherit_state:
            history = self._rng_history  # local alias for brevity

            # Generation 0 – vacancy or ordinary step depending on parent run
            if self._init_g is not None:
                if len(history) < 2:
                    raise RuntimeError("RNG history is incomplete – cannot replay")
                try:
                    replica._rng.bit_generator.state = history[1].copy()
                except AttributeError:
                    replica._rng.bit_generator.state = history[1]
                replica.step_vacancy(J=J, g=self._init_g)
                start_idx = 2  # next generation index in history
            else:
                start_idx = 1  # first regular RG generation

            # Subsequent generations (if any)
            for idx in range(start_idx, len(history)):
                try:
                    replica._rng.bit_generator.state = history[idx].copy()
                except AttributeError:
                    replica._rng.bit_generator.state = history[idx]
                replica.step()
        else:
            # Fully independent replica – optionally still aligned at n=0
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

    def internal_energy(self, J: float, dJ: float | None = None) -> float:
        """Central finite difference of *f* with respect to *J*.

        If *dJ* is not provided an adaptive value is chosen based on the
        coupling strength to balance accuracy and numerical stability.
        """

        beta = float(J)

        if dJ is None:
            # Use same safe step size as specific_heat to avoid instability
            dJ = 1e-6

        # Use same seed for all evaluations to maintain common random numbers
        f_plus = self._eval_f(beta + dJ, inherit_state=True)
        f_minus = self._eval_f(beta - dJ, inherit_state=True)
        
        return (f_plus - f_minus) / (2.0 * dJ)

    def specific_heat(
        self,
        J: float,
        dJ: float | None = None,
        *,
        method: str = "simple",
        validate: bool = True,
    ) -> float:
        """Estimate the specific heat C = β² d²f/dβ² using finite differences.

        Uses simple central difference by default which is robust and accurate
        for most cases. For the classical Heisenberg model with proper norm choice,
        this should give reliable results.

        Parameters
        ----------
        J : float
            Coupling strength (identified with β in thermodynamics)
        dJ : float, optional
            Step size for finite difference. If None, uses adaptive step size
            optimized for the free energy scale (~ O(1)).
        method : str, default="simple"  
            Method for calculating specific heat:
            - "simple": Simple central difference (recommended)
            - "richardson": Richardson extrapolation (for noisy cases)
            - "robust": Enhanced method with validation (recommended for testing)
        validate : bool, default=True
            Whether to validate results and provide warnings for unphysical values.
        """

        beta = float(J)

        if dJ is None:
            # CRITICAL: Avoid the numerical instability "death zone" (1e-4 to 1e-2)
            # Use very small steps that work reliably
            dJ = 1e-6  # Safe small step size that avoids instability

        if method == "robust":
            return self._specific_heat_robust(J, dJ, validate)
        elif method == "richardson":
            return self._specific_heat_richardson(J, dJ)
        elif method == "simple":
            return self._specific_heat_simple(J, dJ)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _specific_heat_simple(self, J: float, dJ: float) -> float:
        """Simple central difference specific heat: C = β² (f(β+h) - 2f(β) + f(β-h))/h²
        
        This is the recommended method for most cases as it's robust and doesn't
        suffer from the complexities of Richardson extrapolation.
        """
        beta = float(J)
        
        # Use same seed for all evaluations to maintain common random numbers
        f_plus = self._eval_f(beta + dJ, inherit_state=True)
        f_zero = self._eval_f(beta, inherit_state=True)  
        f_minus = self._eval_f(beta - dJ, inherit_state=True)
        
        # Central difference for second derivative
        second_deriv = (f_plus - 2.0 * f_zero + f_minus) / (dJ ** 2)
        
        # Convert to specific heat: C = β² f''(β)
        return beta ** 2 * second_deriv

    def _specific_heat_robust(self, J: float, dJ: float, validate: bool = True) -> float:
        """Robust specific heat calculation with multiple step sizes and validation.
        
        This method uses multiple step sizes to check consistency and provides
        warnings for potentially problematic results.
        """
        beta = float(J)
        
        # Use step sizes optimized for the free energy scale
        step_sizes = [dJ, dJ*1.5, dJ*0.7]  # Closer spacing for better consistency
        results = []
        
        for i, h in enumerate(step_sizes):
            try:
                # Use same seed for all evaluations to maintain common random numbers
                f_plus = self._eval_f(beta + h, inherit_state=True)
                f_zero = self._eval_f(beta, inherit_state=True)  
                f_minus = self._eval_f(beta - h, inherit_state=True)
                
                # Central difference for second derivative
                second_deriv = (f_plus - 2.0 * f_zero + f_minus) / (h ** 2)
                C = beta ** 2 * second_deriv
                results.append(C)
                
                if validate:
                    print(f"  Step size {h:.3f}: C = {C:.6f}, f+ = {f_plus:.6f}, f0 = {f_zero:.6f}, f- = {f_minus:.6f}")
                
            except Exception as e:
                if validate:
                    print(f"  Warning: Failed calculation with step size {h:.3f}: {e}")
                continue
        
        if not results:
            raise RuntimeError("All step sizes failed in robust specific heat calculation")
        
        # Use the first result (base step size) as primary
        C_final = results[0]
        
        if validate:
            if len(results) > 1:
                std_dev = np.std(results)
                mean_val = np.mean(results)
                print(f"  Results across step sizes: {[f'{r:.6f}' for r in results]}")
                print(f"  Standard deviation: {std_dev:.6f}")
                print(f"  Using result: {C_final:.6f}")
                
                if len(results) > 1 and std_dev > abs(mean_val) * 0.2:
                    print(f"  WARNING: High variability across step sizes (σ/|μ| = {std_dev/abs(mean_val):.3f})")
            
            if C_final < 0:
                print(f"  WARNING: Negative specific heat ({C_final:.6f}) - may indicate numerical issues or step size problems")
            
            if abs(C_final) > 100:
                print(f"  WARNING: Very large specific heat ({C_final:.6f}) - may indicate step size issues")
        
        return float(C_final)

    def _specific_heat_richardson(self, J: float, dJ: float) -> float:
        """Richardson extrapolation specific heat for complex/noisy cases.
        
        Uses Richardson extrapolation with two different step sizes to
        reduce discretization error. More computationally expensive but
        can be useful for very noisy systems.
        """
        beta = float(J)

        def _second_derivative_stencil(h: float, seed_offset: int = 0) -> float:
            """Calculate second derivative stencil with given step size."""
            
            f_plus = self._eval_f(beta + h, inherit_state=True)
            f_zero = self._eval_f(beta, inherit_state=True)
            f_minus = self._eval_f(beta - h, inherit_state=True)
            
            return (f_plus - 2.0 * f_zero + f_minus) / (h ** 2)

        # Calculate with two different step sizes
        h1 = dJ
        h2 = dJ / 2.0
        
        # Ensure h2 doesn't get too small
        h2 = max(h2, 1e-4)
        
        # Richardson extrapolation: more accurate estimate
        # f''(β) ≈ (4*S(h/2) - S(h)) / 3
        second_deriv_h1 = _second_derivative_stencil(h1)
        second_deriv_h2 = _second_derivative_stencil(h2)
        
        second_deriv = (4.0 * second_deriv_h2 - second_deriv_h1) / 3.0
        
        # Convert to specific heat: C = β² f''(β)
        return beta ** 2 * second_deriv

    # 4. Introspection ------------------------------------------------------
    def get_pool(self) -> np.ndarray:
        """Return a *copy* of the current pool (row-normalised)."""
        if self._pool is None:
            raise RuntimeError("Pool not initialised.")
        return self._pool.copy()

# ---------------------------------------------------------------------------
# B=2 Scaling Controller
# ---------------------------------------------------------------------------

class RGControllerB2(RGController):
    """RG Controller with b=2 scaling factor for decimation by 2.
    
    This controller implements b=2 scaling instead of the default b=3, 
    using 2-bond decimation instead of 3-bond decimation. Suitable for
    checking specific scaling behavior for pure systems (p=0) with pool size 1.
    """
    
    def __init__(
        self,
        l_prec: int,
        pool_size: int,
        dim: int,
        n_bm: int,
        seed: int = 17,
        dtype: str | np.dtype = "float64",
        bond_move_first: bool = False,
    ) -> None:
        super().__init__(
            l_prec=l_prec,
            pool_size=pool_size,
            dim=dim,
            n_bm=n_bm,
            seed=seed,
            dtype=dtype,
            bond_move_first=bond_move_first,
        )
        # Override the b factor for this controller
        self._b_factor = 2
    
    def _apply_kernel_raw(self, kernel_func, *args) -> tuple[np.ndarray, float]:
        """Apply kernel and return raw result + log norm without recording.
        
        This matches the notebook approach where we only record the FINAL 
        log norm per generation, not intermediate ones.
        
        Parameters
        ----------
        kernel_func : callable
            The kernel function to apply (e.g., kernels.decimate2, kernels.bond_move)
        *args : tuple
            Arguments to pass to the kernel function
            
        Returns
        -------
        tuple[np.ndarray, float]
            Raw kernel output and its log norm
        """
        if self._delta_pool is not None:
            raise NotImplementedError("Vacancy operations not yet implemented for b=2")
        
        # Call the kernel function - returns (raw_result, ln_norm)
        return kernel_func(*args)
    
    def _decimate_b2(self, l1: np.ndarray, l2: np.ndarray) -> tuple[np.ndarray, float]:
        """Decimation operation returning raw result and log norm."""
        return self._apply_kernel_raw(kernels.decimate2, l1, l2)
    
    def _bond_move_b2(self, l1: np.ndarray, l2: np.ndarray) -> tuple[np.ndarray, float]:
        """Bond move operation returning raw result and log norm."""
        return self._apply_kernel_raw(kernels.bond_move, l1, l2)
    
    def step(self, n: int = 1) -> None:
        """Perform *n* RG generations using b=2 scaling with user's scheme."""
        if n < 1:
            raise ValueError("n must be a positive integer")
        
        # Validate bond_move_first compatibility - only check when vacancies are active
        if self._delta_pool is not None and self._bond_move_first:
            raise ValueError("bond_move_first=True is not supported with vacancy systems. Use bond_move_first=False.")
            
        if self._delta_pool is not None:
            raise NotImplementedError("Vacancy steps not yet implemented for b=2")
        
        # Fast-path for multiple iterations
        if n > 1:
            for _ in range(n):
                self.step()
            return
        
        # Snapshot RNG state for replay
        self._rng_history.append(self._rng.bit_generator.state.copy())
        
        if self._pool is None:
            raise RuntimeError("Pool not initialised.")
        
        # For pool_size=1: exact notebook behavior
        # For pool_size>1: standard pool RG logic
        
        if self.pool_size == 1:
            # EXACT notebook logic for single LFC
            lfc_in = self._pool[0].copy()
            
            if self._bond_move_first and self.n_bm > 1:
                # Bond move first (like getGvalsBondMoveFirst)
                _lfc = lfc_in
                for j in range(self.n_bm - 1):
                    _lfc, g = self._bond_move_b2(_lfc, lfc_in)
                _lfc, g = self._decimate_b2(_lfc, _lfc)
            else:
                # Decimation first (like getGvalsBondMove)
                lfc_out, g = self._decimate_b2(lfc_in, lfc_in)
                lfc_bond_in = lfc_out
                for j in range(max(0, self.n_bm - 1)):
                    lfc_out, g = self._bond_move_b2(lfc_out, lfc_bond_in)
                _lfc = lfc_out
            
            # Normalize final result (like notebook)
            norm_out = np.amax(np.abs(_lfc))
            if norm_out != 0.0:
                _lfc = _lfc / norm_out
            
            self._pool[0] = _lfc
            final_log_norm = g
            
        else:
            # Standard pool RG logic for pool_size > 1
            final_log_norm = 0.0
            
            if self._bond_move_first and self.n_bm > 1:
                # Bond move first, then decimation
                base_pool = self._pool.copy()
                current_pool = self._pool.copy()
                
                for _ in range(max(0, self.n_bm - 1)):
                    next_pool = np.empty_like(current_pool)
                    for j in range(self.pool_size):
                        idx1 = self._rng.integers(0, self.pool_size)   # from base_pool
                        idx2 = self._rng.integers(0, self.pool_size)   # from evolving pool
                        lfc_result, ln_norm = self._bond_move_b2(base_pool[idx1], current_pool[idx2])
                        # Normalize result but DON'T record log norm yet
                        norm_val = np.exp(ln_norm)
                        next_pool[j] = lfc_result / norm_val if norm_val != 0.0 else lfc_result
                    current_pool = next_pool
                input_for_decimation = current_pool
            else:
                # No bond move phase if n_bm == 1
                input_for_decimation = self._pool.copy()

            # Decimation phase using b=2 (only 2 vectors)
            dec_pool = np.empty_like(input_for_decimation)
            for i in range(self.pool_size):
                idxs = self._rng.integers(0, self.pool_size, size=2)
                lfc_result, ln_norm = self._decimate_b2(
                    input_for_decimation[idxs[0]], 
                    input_for_decimation[idxs[1]]
                )
                dec_pool[i] = lfc_result  # Don't normalize yet
                final_log_norm = ln_norm  # last pool entry wins
            
            # Normalize the entire pool after all operations
            for i in range(self.pool_size):
                norm_val = np.amax(np.abs(dec_pool[i]))
                if norm_val > 0:
                    dec_pool[i] = dec_pool[i] / norm_val
            self._pool = dec_pool

            if not self._bond_move_first and self.n_bm > 1:
                # Decimation first, then bond moves (only if n_bm > 1)
                base_pool = self._pool.copy()
                current_pool = self._pool.copy()
                
                for _ in range(max(0, self.n_bm - 1)):
                    next_pool = np.empty_like(current_pool)
                    for j in range(self.pool_size):
                        idx1 = self._rng.integers(0, self.pool_size)   # from base_pool
                        idx2 = self._rng.integers(0, self.pool_size)   # from evolving pool
                        lfc_result, ln_norm = self._bond_move_b2(base_pool[idx1], current_pool[idx2])
                        next_pool[j] = lfc_result  # Don't normalize yet
                        final_log_norm = ln_norm  # last operation wins
                    
                    # Normalize the entire pool after all operations
                    for j in range(self.pool_size):
                        norm_val = np.amax(np.abs(next_pool[j]))
                        if norm_val > 0:
                            next_pool[j] = next_pool[j] / norm_val
                    current_pool = next_pool
                self._pool = current_pool

        # Record ONLY the final log norm (matches notebook exactly)
        self._record_ln(final_log_norm)
        
        # Finish generation
        self._gen_log_sums.append(self._cur_gen_sum)
        self._log_norms.append(self._cur_gen_sum)
        self._cur_gen_sum = 0.0
        self._steps += 1
    
    def _eval_f(
        self,
        J: float,
        *,
        seed: int | None = None,
        inherit_state: bool = True,
    ) -> float:
        """Estimate free-energy density at coupling *J* by rerunning RG with b=2 scaling.
        
        For consistent specific heat calculations, this method ensures the same
        behavior regardless of pool size to avoid inconsistencies in derivatives.
        """
        
        # Create RGControllerB2 replica instead of RGController  
        replica = RGControllerB2(
            l_prec=self.l_prec,
            pool_size=self.pool_size,
            dim=self.dim,
            n_bm=self.n_bm,
            seed=self._seed if seed is None else int(seed),
            dtype=self.dtype,
            bond_move_first=self._bond_move_first,
        )

        # Use the stored initialization parameters consistently
        p = getattr(self, '_init_p', 0.0) or 0.0
        q = getattr(self, '_init_q', 0.0) or 0.0
        g = getattr(self, '_init_g', None)
        replica.initialize_pool(J=J, p=p, q=q)

        # Always use inherit_state=True for finite differences to minimize noise
        if inherit_state:
            history = self._rng_history  # local alias for brevity

            # Generation 0 – vacancy or ordinary step depending on parent run
            if g is not None:
                if len(history) < 2:
                    raise RuntimeError("RNG history is incomplete – cannot replay")
                try:
                    replica._rng.bit_generator.state = history[1].copy()
                except AttributeError:
                    replica._rng.bit_generator.state = history[1]
                replica.step_vacancy(J=J, g=g)
                start_idx = 2  # next generation index in history
            else:
                start_idx = 1  # first regular RG generation

            # Subsequent generations (if any)
            for idx in range(start_idx, len(history)):
                try:
                    replica._rng.bit_generator.state = history[idx].copy()
                except AttributeError:
                    replica._rng.bit_generator.state = history[idx]
                replica.step()
        else:
            # Fully independent replica – optionally still aligned at n=0
            try:
                replica._rng.bit_generator.state = self._rng.bit_generator.state.copy()
            except AttributeError:
                replica._rng.bit_generator.state = self._rng.bit_generator.state

            if g is not None:
                replica.step_vacancy(J=J, g=g)
                remaining = max(self._steps - 1, 0)
            else:
                remaining = self._steps

            for _ in range(remaining):
                replica.step()

        return float(replica.free_energy())    
    def free_energy(self) -> float:
        """Return pooled free-energy density using b=2 scaling: Σ G_n / (2^d)^n.
        
        Uses consistent scaling regardless of pool size to ensure smooth derivatives
        for specific heat calculations.
        """

        if not self._gen_log_sums:
            return 0.0

        # Use standard b=2 scaling consistently
        scale = float(self._b_factor) ** self.dim  # (2^d)
        
        accum = 0.0
        factor = 1.0  # (2^d)^0 for n = 0
        for G_n in self._gen_log_sums:
            accum += G_n / factor
            factor *= scale  # update to (2^d)^{n+1}
        
        return float(accum)
