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

Below we list the exact algebra implemented in the code so that every number
can be reproduced on paper or in another language.

### Legendre–Fourier (LFC) representation

For a nearest-neighbour bond the Boltzmann weight is

$$
W(\theta)=\exp\bigl[J\,\mathbf S_i\!\cdot\!\mathbf S_j\bigr].
$$

Expanding in Legendre polynomials gives

$$
W(\theta)=\sum_{\ell=0}^{\ell_\text{max}} L_{\ell} P_{\ell}(\cos\theta),
\qquad
L_{\ell}=(2\ell+1)\,i^{\,\ell}\,j_{\ell}(-iJ),
$$

where $j_{\ell}$ is the spherical Bessel function. Each bond is therefore
represented by the vector

$$
\mathbf L = (L_0,\dots,L_{\ell_\text{max}}).
$$

### Star–triangle (decimation) transformation

Given three bonds $\mathbf L^{(1)},\mathbf L^{(2)},\mathbf L^{(3)}$ arranged
in a triangle, they are replaced by a single effective bond with coefficients

$$
L^{\text{dec}}_{\ell}=\frac{L^{(1)}_{\ell}\,L^{(2)}_{\ell}\,L^{(3)}_{\ell}}{(2\ell+1)^2}.
$$

With vacancies the rule becomes

$$
\begin{aligned}
L'_{\ell} &= e^{-4\Delta}
            \frac{L^{(1)}_{\ell}\,L^{(2)}_{\ell}\,L^{(3)}_{\ell}}{(2\ell+1)^2},
            && \ell>0,\\[4pt]
L'_0 &= e^{-4\Delta} L^{(1)}_0 L^{(2)}_0 L^{(3)}_0
       + 1
       + e^{-2\Delta}\bigl(L^{(2)}_0 + L^{(3)}_0\bigr).
\end{aligned}
$$

### Bond-move transformation

Moving a bond across a site combines two LFCs through the (reduced) Gaunt
tensor:

$$
L'_{\ell}=\sum_{\ell_1,\ell_2} L^{(1)}_{\ell_1}\,L^{(2)}_{\ell_2}\,G_{\ell_1\ell_2\ell},
\qquad
G_{\ell_1\ell_2\ell}=2\!
\begin{pmatrix}
\ell_1 & \ell_2 & \ell\\
0       & 0      & 0
\end{pmatrix}^2.
$$

### Normalisation and scale factor

After **every** decimation or bond move we divide by

$$
\Lambda=\max_{\ell}|L'_{\ell}|
$$

and store $\ln\Lambda$. When the controller is created with
`norm="l2"` the vector is subsequently rescaled to unit Euclidean norm.

### RG generation

One RG generation consists of a star–triangle decimation followed by
$n_{\text{bm}}-1$ bond moves. The lattice spacing therefore grows by the
block-size $b=n_{\text{bm}}$.

### Vacancy parameter flow

The vacancy parameter is initialised as

$$
\Delta_0 = g\,J.
$$

After generation $n$ we average the recorded scale factors and update

$$
\boxed{\;\Delta_{n+1}=2\,\Delta_n-\langle\ln\Lambda\rangle_n\;}
$$

(the factor 2 reflects the doubling of the lattice spacing in three
dimensions).

### Free energy and thermodynamic observables

We accumulate the generation subtotal

$$
G_n = \sum_{\text{ops in }n}\!\ln\Lambda
$$

and form the dimensionless free-energy density

$$
f=\sum_{n=0}^{\infty}\frac{G_n}{b^{d n}},
$$

where $d$ is the spatial dimension. Numerical derivatives of $f$ give the
internal energy $u=\partial f/\partial J$ and specific heat
$c=\partial^2 f/\partial J^2$.

## Data file: `cleb.npy`

The Clebsch–Gordan tensor (`cleb.npy`) must be placed inside the package directory (`rgheisenberg/`) **or** in `data/cleb.npy` relative to the same location. This binary file is not included in the repository to keep the size small.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
