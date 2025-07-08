# RG Heisenberg

Renormalization-Group flows for the classical Heisenberg model implemented in Python.

This package provides:

-   **Numerical kernels** (`rgheisenberg.kernels`) – Numba-accelerated bond-move and decimation operations.
-   **Controller** (`rgheisenberg.controller.RGController`) – high-level orchestration of RG steps, pool initialisation and thermodynamic observables.
-   **Analysis helpers** (`rgheisenberg.analysis` & `rgheisenberg.phase`) – thin wrappers for producing phase diagrams and tracking order parameters.

## Installation

```bash
python -m pip install git+https://github.com/yourname/rg-heisenberg.git
```

(Or clone and install locally: `pip install -e .`)

Requires Python 3.9+ and the following scientific packages: NumPy, SciPy, and Numba.

## Quickstart

```python
import numpy as np
from rgheisenberg import RGController, find_critical_J

# Build a controller factory (useful when scanning)
ctrl_factory = lambda: RGController(l_prec=8, pool_size=256, dim=3, n_bm=2)

# Find critical coupling at p = q = 0 (pure ferromagnet)
Jc = find_critical_J(
    ctrl_factory=ctrl_factory,
    J_lo=0.1,
    J_hi=2.0,
    p=0.0,
    q=0.0,
)
print(f"Critical J ≈ {Jc:.3f}")
```

## Analytical formulation

This section collects the exact equations implemented in the code so that you can reproduce every number analytically or port the algorithm to a different language.

### Legendre–Fourier (LFC) representation

For each nearest-neighbour bond we expand the Boltzmann weight

\[ W(\theta) = \exp\bigl[J\,\mathbf S_i\!\cdot\!\mathbf S_j\bigr] \]

in Legendre polynomials,

\[ W(\theta) = \sum*{\ell=0}^{\ell*\text{max}} L*\ell P*\ell(\cos\theta), \quad L*\ell = (2\ell+1)\,i^{\,\ell}\, j*\ell(-iJ), \]

with the spherical Bessel function \(j\_\ell\). The RG “pool” therefore stores vectors

\[ \mathbf L = \bigl(L*0,\dots,L*{\ell\_\text{max}}\bigr). \]

### Star–triangle (decimation) transformation

Given three bonds `lfc1`, `lfc2`, `lfc3` arranged in a triangle we replace them by one effective bond

\[ L^{\text{dec}}_\ell = \frac{L^{(1)}_\ell L^{(2)}_\ell L^{(3)}_\ell}{(2\ell+1)^2}. \]

With vacancies (bond-dilution) the transformation becomes

\[
\begin{aligned}
L^{\prime}_\ell &= \mathrm e^{-4\Delta}\,\frac{L^{(1)}_\ell L^{(2)}_\ell L^{(3)}_\ell}{(2\ell+1)^2}, & \ell>0, \\
L^{\prime}\_0 &= \mathrm e^{-4\Delta} L^{(1)}\_0 L^{(2)}\_0 L^{(3)}\_0 + 1 + \mathrm e^{-2\Delta}\bigl(L^{(2)}\_0 + L^{(3)}\_0\bigr).
\end{aligned}
\]

### Bond-move transformation

Moving a bond across a site combines two LFCs through the reduced Gaunt tensor

\[ L^{\prime}_\ell = \sum_{\ell*1,\ell_2} L^{(1)}*{\ell*1}\,L^{(2)}*{\ell*2}\,G*{\ell*1\ell_2\ell}, \qquad G*{\ell_1\ell_2\ell}=2\Bigl(\begin{smallmatrix} \ell_1 & \ell_2 & \ell \\ 0 & 0 & 0 \end{smallmatrix}\Bigr)^2. \]

### Normalisation and scale factor

After **every** decimation or bond-move we divide by the max-norm
\(\Lambda=\max*{\ell}|L^{\prime}*\ell|\) and store \(\ln\Lambda\). Optionally, the controller rescales the vector once more to unit Euclidean norm when it is constructed with `norm="l2"`.

### RG generation

A generation consists of a star–triangle decimation followed by \(n*{\text{bm}}-1\) bond moves.
The lattice spacing grows by the block-size \(b=n*{\text{bm}}\).

### Vacancy parameter flow

The vacancy parameter starts at

\[ \Delta_0 = g J. \]

After each generation we compute the mean scale factor
\(\langle \ln\Lambda \rangle\) over **all** decimations and bond moves performed in that generation and update

\[ \boxed{\;\Delta\_{n+1} = 2\,\Delta_n\; - \; \langle \ln\Lambda \rangle_n\;} \]

(the factor 2 reflects the halving of each bond’s length when coarse-graining a cubic lattice).

### Free energy and thermodynamic observables

We accumulate the generation subtotal

\[ G*n = \sum*{\text{ops in }n}\! \ln\Lambda, \]

and obtain the dimensionless free-energy density

\[ f = \sum\_{n=0}^{\infty} \frac{G_n}{b^{d n}}, \]

where \(d\) is the spatial dimension. Numerical derivatives give

-   internal energy \(u = \partial f/\partial J\),
-   specific heat \(c = \partial^2 f/\partial J^2\).

## Data file: `cleb.npy`

The Clebsch–Gordan tensor (`cleb.npy`) must be placed inside the package directory (`rgheisenberg/`) **or** in `data/cleb.npy` relative to the same location. This binary file is not included in the repository to keep the size small.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
