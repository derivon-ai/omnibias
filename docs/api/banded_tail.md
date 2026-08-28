# Banded tail-inverse bound & one-sided radii closure

`omnibias.core.verified.radii_spectral`'s existence-proof pipeline needs a
tail-inverse-norm estimate for the linear part of `F(a) = ell*a + Q(a,a) - f`,
but its only such estimate (`mu = sup_{|k|>N} |ell(k)|^{-1}`) requires `ell` to
be a **diagonal** Fourier multiplier. `omnibias.core.verified.banded` supplies
the missing tail-inverse bound for a **banded** (non-diagonal) linear
operator — one whose matrix couples each tail index to a bounded set of
nearby indices, e.g. a self-similar scaling generator `x . grad`. It
certifies `||M^{-1}||` for the operator norm induced by the (unweighted) sup
norm `ell^infty` — the same "max absolute row sum" convention already used by
`omnibias.core.verified.linalg.inf_norm_matrix` — under the row-wise
diagonal-dominance hypothesis `|M_ii| >= diag_lower` and
`sum_{j!=i} |M_ij| <= off_diagonal_row_sum_upper < diag_lower`, uniformly over
every tail row. `radii_spectral.BandedLinearPart` plus
`tail_inverse_bound_from_banded` is the Phase-2 wiring: a manufactured-solution
existence proof can use nearest-neighbour couplings instead of a diagonal
symbol. The original diagonal `laplacian_symbol` path stays byte-identical.

```python
from omnibias.core.verified.banded import (
    banded_tail_inverse_bound,
    finite_band_row_sum_bound,
    geometric_band_row_sum_bound,
)

# A nearest-neighbour banded tail: |M_ii| >= 5, and the two off-diagonal
# neighbours are each bounded by 1.5 in magnitude, so s = 3.0 < 5.0.
s = finite_band_row_sum_bound([1.5, 1.5])
bound = banded_tail_inverse_bound(5.0, s, bandwidth=2)
assert abs(bound.hi - 1.0 / (5.0 - s)) < 1e-9  # 1/(d_min - s), outward rounded

# A decaying-envelope band (|M_{i,i+d}| <= 0.4 * 0.5**|d|) is bounded the
# same way, without ever assuming finite support.
s_env = geometric_band_row_sum_bound(coeff=0.4, ratio=0.5)
env_bound = banded_tail_inverse_bound(3.0, s_env)
assert env_bound.hi > 0.0

# Diagonal dominance failing is a ValueError, never a silently-fabricated bound.
try:
    banded_tail_inverse_bound(2.0, 2.5)
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError")
```

## One-sided radii-polynomial closure

`omnibias.core.verified.radii_series` is the minimal one-sided counterpart of
`radii_spectral`: the same `(Y0, Z0, Z1, Z2)` shape, but posed in the
one-sided `ell^1_nu` Banach algebra
`omnibias.core.verified.sequence_space.ValidatedSeries` for
`F(a) = ell*a + a*a - f = 0` with a **diagonal** scalar symbol `ell(n)`
(audited: no such pipeline existed anywhere in `verified/` before this — see
the module docstring for the full audit trail). `series_radii_certificate`
builds a *split* approximate inverse (a numerically-inverted finite Jacobian
block plus the exact diagonal tail inverse from `banded_tail_inverse_bound`'s
`s=0` case), assembles the radii-polynomial bounds, and — when a contracting
radius exists — returns a sealed
`omnibias.core.verified.kantorovich.RadiiCertificate` proving a **true**,
unique zero near the supplied approximation.

```python
from omnibias.core.verified.radii_series import (
    SeriesProblem,
    constant_symbol,
    constant_tail_inverse_bound,
    series_radii_certificate,
)

# Manufacture an exact solution a* for F(a) = 4*a + a*a - f = 0 by choosing
# the forcing f = 4*a* + a**a* directly, then certify existence near a*.
symbol = constant_symbol(4.0)
mu = constant_tail_inverse_bound(4.0)
a_star = [0.1, 0.05, 0.02, 0.01, 0.005]

from omnibias.core.verified.sequence_space import ValidatedSeries

# Pad to work_trunc + 1 = 2*trunc + 1 = 9 coefficients so the product a* * a*
# (degree <= 8) is computed exactly, with no truncation overflow.
ab = ValidatedSeries.from_coeffs([*a_star, *([0.0] * 4)], nu=1.05, tail=0.0)
forcing = list((ab.scale(4.0) + ab * ab).coeffs)

problem = SeriesProblem(
    trunc=4, nu=1.05, linear_symbol=symbol, tail_inverse_bound=mu, forcing=forcing
)
result = series_radii_certificate(problem, a_star)
assert result.proved
assert result.radius is not None and result.radius > 0.0
```

Both restrictions — a diagonal linear part, and the nonlinearity fixed to the
algebra's own square `a*a` — remain for the one-sided `radii_series` path. The
Fourier (`radii_spectral`) side now accepts a `BandedLinearPart`; a banded
one-sided linear part is still future work.

```python
from omnibias.core.verified.radii_spectral import (
    constant_coefficient_band,
    laplacian_symbol,
    tail_inverse_bound_from_banded,
)

diag = laplacian_symbol(4.0, 1.0)
band = constant_coefficient_band(diag, {(1,): 0.05, (-1,): 0.05})
assert band.bandwidth == 1
mu = tail_inverse_bound_from_banded(29.0, {(1,): 0.05, (-1,): 0.05}, 1.05)
assert mu > 0.0
```

## API

::: omnibias.core.verified.banded
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.core.verified.radii_series
    options:
      show_root_heading: false
      heading_level: 3
