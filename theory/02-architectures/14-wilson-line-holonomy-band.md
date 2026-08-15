# 02-14 The Wilson-line holonomy band

## 1. Thesis and status

The slab between two parallel hyperplanes is exactly the region a parallel
transport crosses, so the `band` and `integral` roles lift from scalars to
**path-ordered holonomies**: the gauge generalization of the finite-gap
operators, closed form in the abelian and transverse-constant cases and an
explicitly finite truncation otherwise.

- **Status**: gated (G1–G5 CI; abelian + transverse-constant closed form; no YM / mass gap / continuum claim)
- **Depends on**: 01-05, 01-10
- **Blocks**: 07-04

## 2. Where it lands

`packages/omnibias-geometry/src/omnibias/geometry/gauge/band/` with torch and
jax twins. The gauge submodule already owns transport and Wilson loops; this is
one more operator there.

## 3. Prior art in omnibias

- `packages/omnibias-geometry/src/omnibias/geometry/gauge/{torch,jax}/ops/nonintegrable.py`
  — `parallel_transport(state, conn, curve, *, t0, t1, substeps, generators)`,
  `parallel_transport_from_arrays`, `wilson_loop(...)`. Path-ordered transport
  along a curve, already implemented in both backends.
- `packages/omnibias-geometry/src/omnibias/geometry/gauge/_core/kernels.py` —
  `wilson_line_exponents(xp, A_path, tangents, generators, coupling, dt)`,
  which assembles the per-segment anti-Hermitian exponents
  `X_i = -i g (A_mu^a xdot^mu dt) T^a`.
- `packages/omnibias-geometry/src/omnibias/geometry/gauge/lattice/` —
  `plaquette_trace`, `average_plaquette`, `wilson_loop_trace`,
  `wilson_loops_ensemble`.
- `packages/omnibias-geometry/src/omnibias/geometry/gauge/transfer/gap.py` —
  `certified_transfer_matrix_gap(transfer, *, lattice_spacing, deflate)`,
  returning a `TransferGapResult` with a certified lower bound on
  `m a = -ln(|lambda_1| / lambda_0)` for a **fixed** matrix, with
  `continuum_claim = False` already fixed in its honesty text.
- `docs/operator-surface.md` — the `integral` role: `S(z + b_hi) - S(z + b_lo)`
  with `S' = sigma`.

**Confirmed gap.** Transport exists along a *given curve*. Nothing connects it
to the band geometry, and there is no operator whose domain is "the slab between
two parallel planes".

## 4. Mathematics

### The scalar band, restated

The `integral` role computes

```
B(z) = S(z + b_hi) - S(z + b_lo),      S' = sigma
```

that is, the integral of `sigma` across the slab between the two planes. Spec
01-10 already names this correctly: it is a **nonlocal** functional along a
path, not a jet coordinate at a point. That is precisely the structure a
parallel transport has.

### The gauge lift

Given a connection `A_mu` valued in a Lie algebra with generators `T^a`, the
parallel transport across the slab along the normal direction `w` is

```
U(z_lo -> z_hi) = P exp( -i g integral_{z_lo}^{z_hi} A_mu(x(s)) xdot^mu ds )
```

with `P` the path ordering. Define the **holonomy band operator** as this `U`,
with the two planes supplying the endpoints. Three regimes:

**Abelian.** For a `U(1)` connection the path ordering is trivial and

```
U = exp( -i g integral_{z_lo}^{z_hi} A . dx )
```

so the exponent is a scalar line integral. If `A` restricted to the normal
direction is expressible in the activation dictionary, the integral is the
closed-form antiderivative window and **`U` is closed form**: the scalar `band`
role, exponentiated.

**Transverse-constant non-abelian.** If `A_mu` is constant along the transport
path (constant within the slab), the path ordering is again trivial because all
the exponents commute with themselves, and

```
U = exp( -i g A_mu xdot^mu (z_hi - z_lo) )
```

a single matrix exponential of a known argument. **Closed form**, with the
matrix exponential computable in closed form for `SU(2)` (Rodrigues formula) and
by eigendecomposition in general.

**General non-abelian.** Path ordering matters and there is no closed form. The
honest construction is an **explicitly finite truncation**: either the standard
product of `N` segment exponentials (which is what `parallel_transport` already
does with `substeps`) or a Magnus expansion truncated at a stated order,

```
Omega_1 = integral X(s) ds
Omega_2 = (1/2) integral ds1 integral^{s1} ds2 [X(s1), X(s2)]
U ~ exp( Omega_1 + Omega_2 + ... )
```

with a convergence condition on `integral ||X||`. The Magnus route is worth
having because its terms are iterated integrals of the connection, and iterated
integrals of dictionary functions are again closed form — so a **finite-order
Magnus holonomy is closed form even in the non-abelian case**, with a stated
truncation error.

That is the actual contribution: not a new physics claim, but the observation
that the band geometry plus the closed-form antiderivative gives closed-form
Magnus terms, so the truncation error is a computed quantity rather than a step
count.

### Gauge covariance

Under a gauge transformation `g(x)`, the holonomy transforms as

```
U -> g(x_hi) U g(x_lo)^{-1}
```

so `tr U` around a closed loop is gauge invariant but the open band holonomy is
not. Any feature built from it must either close the loop or contract with
appropriately transforming objects. This is a correctness requirement, not a
detail: **a network feature built from an open Wilson line is gauge dependent
and therefore physically meaningless on its own.**

### The band as a lattice link

On a lattice, the link variable `U_mu(x)` *is* the holonomy across one lattice
spacing, that is across a slab of width `a`. So the band operator is the
continuum object that lattice links discretize, and a network parameterizing
band holonomies is parameterizing a gauge configuration. That connects directly
to the existing lattice observables (`plaquette_trace`, `wilson_loop_trace`) and
to the transfer-matrix gap machinery.

## 5. Worked example

**Abelian band, closed form.** Take `U(1)` with a connection whose normal
component is `A(z) = A_0 sigma'(z)` — a localized flux concentrated near the
plane `z = 0`, with `sigma = tanh`.

The exponent across a slab `[z_lo, z_hi]` is

```
-i g A_0 integral_{z_lo}^{z_hi} sigma'(z) dz = -i g A_0 [ tanh(z) ]_{z_lo}^{z_hi}
```

For `z_lo = -1`, `z_hi = 1`:

```
tanh(1) - tanh(-1) = 2 * 0.7615942 = 1.5231884
U = exp( -i g A_0 * 1.5231884 )
```

With `g A_0 = 1`: `U = exp(-1.5231884 i) = cos(1.5231884) - i sin(1.5231884)
= 0.0475899 - 0.9988670 i`. Closed form, one `tanh` evaluation at each edge, and
the total flux through the slab is exactly the antiderivative difference.

Widening the slab to `[-inf, inf]` gives total flux `2 g A_0`, and the holonomy
saturates: `U = exp(-2i)`. The band operator therefore interpolates between "no
flux enclosed" and "all flux enclosed" as the gap opens, which is the physical
content of the finite-gap role.

**Non-abelian, transverse-constant.** Take `SU(2)` with constant
`A = (A^1, A^2, A^3) = (0.3, 0, 0.4)` along the path and slab width `L = 1`,
coupling `g = 1`. The exponent is `-i (A^a T^a) L` with `T^a = sigma^a / 2`, so
with `|A| = 0.5`,

```
U = exp( -i (0.5) (n^a sigma^a / 2) )  with n = (0.6, 0, 0.8)
  = cos(0.25) I - i sin(0.25) ( n^a sigma^a )
  = 0.9689124 I - 0.2474040 i ( 0.6 sigma^1 + 0.8 sigma^3 )
```

by the Rodrigues formula for `SU(2)`. Closed form, no path ordering needed,
because the exponents commute along the path.

**Where it stops being closed form.** Let `A` rotate within the slab, say
`A(z) = 0.5 ( cos(k z), 0, sin(k z) )`. Now `[X(s_1), X(s_2)] != 0` and the
first Magnus correction is nonzero:

```
Omega_2 = (1/2) integral_0^L ds1 integral_0^{s1} ds2 [ X(s1), X(s2) ]
```

The commutator of two `SU(2)` elements is again `SU(2)`, and the double integral
of products of `cos(k z)` and `sin(k z)` is elementary, so `Omega_2` is closed
form. The truncation error after `Omega_2` is `O((|A| L)^3)`, which for
`|A| L = 0.5` is about `0.125` times a numerical constant — large enough that
the third term matters, and small enough that the expansion converges. **The
error is a computed number, which is the whole point.**

## 6. Proposed API

Does not exist yet.

```python
# omnibias/geometry/gauge/band/_core.py
@dataclass(frozen=True)
class HolonomyBand:
    normal: tuple[float, ...]
    lo: float                       # b_lo
    hi: float                       # b_hi
    algebra: LieAlgebra
    coupling: float

class BandRegime(StrEnum):
    ABELIAN = "abelian"                       # closed form
    TRANSVERSE_CONSTANT = "transverse_constant"  # closed form
    MAGNUS = "magnus"                          # finite truncation, error bound
    PRODUCT = "product"                        # substep product, existing path

def classify_regime(conn: GaugeConnectionSpec, band: HolonomyBand) -> BandRegime:
    """Detects which closed-form case applies; never silently downgrades."""

def magnus_terms(conn, band, *, order: int) -> tuple[Array, ...]:
    """Iterated integrals of the connection; closed form for dictionary
    connections."""
def magnus_truncation_bound(conn, band, *, order: int) -> Interval: ...
```

```python
# omnibias/geometry/gauge/band/torch.py  (and jax twin)
def band_holonomy(
    state: FieldState, conn: GaugeConnectionSpec, band: HolonomyBand, *,
    regime: BandRegime | None = None, magnus_order: int = 2,
    substeps: int = 32, generators: Tensor | None = None,
) -> Tensor:
    """Reduces to `parallel_transport` in the PRODUCT regime; must agree with it
    to the stated truncation bound in every other regime."""

def band_wilson_loop(state, conn, bands: Sequence[HolonomyBand]) -> Tensor:
    """Closes a loop from band segments so the result is gauge invariant."""
```

`classify_regime` returning an explicit enum rather than a boolean is
deliberate: the difference between closed form and truncation must be visible in
the return value.

## 7. Practical use cases

1. **Gauge-covariant network features.** A layer whose feature is a band
   holonomy transforms correctly by construction, which is the right inductive
   bias for lattice gauge data.
2. **Continuum-to-lattice bridge.** A band of width `a` is a lattice link, so a
   band-parameterized network can be read as a gauge configuration and fed to
   the existing observables.
3. **Flux quantization diagnostics.** In the abelian case the band holonomy is
   the enclosed flux, exactly, which makes topological quantities readable.
4. **Aharonov-Bohm-type problems.** Where the physics is precisely a holonomy
   around a region, the band operator is the natural primitive.
5. **Input to finite transfer-gap work** (spec 07-04), where sharper
   configurations feed `certified_transfer_matrix_gap`.

## 8. Acceptance gates

Baseline: the existing `parallel_transport` with a large `substeps`.

- **G1 regime detection.** `classify_regime` correctly identifies abelian and
  transverse-constant cases on a curated suite and never returns a closed-form
  regime for a genuinely path-ordered connection.
- **G2 closed-form exactness.** In the two closed-form regimes,
  `band_holonomy` matches `parallel_transport` with `substeps = 4096` to
  `<= 1e-12`, at a fraction of the cost.
- **G3 Magnus bound soundness.** `magnus_truncation_bound` upper-bounds the
  measured deviation from the high-`substeps` reference on a dense parameter
  grid **and** a random sample, with zero violations, and the measured error
  decays at the predicted order as `|A| L` shrinks.
- **G4 gauge covariance.** Under a random gauge transformation, the open band
  holonomy transforms as `g(x_hi) U g(x_lo)^{-1}` to `<= 4 ulp`, and
  `band_wilson_loop` is invariant to `<= 4 ulp`. A test asserts that an open
  holonomy used as a feature is flagged, so gauge-dependent features cannot be
  built by accident.
- **G5 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/holonomy_band.py`: regime classification, closed-form exactness
  and speedup, Magnus convergence and bound tightness, gauge-covariance checks.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/holonomy/`.

## 10. Honesty and scope

- The band comes from the founding structure (two parallel hyperplanes with a
  **finite** gap), not from a collapse: the gap is held open, which is the
  opposite of the founding `delta -> 0` bias collapse. No temperature collapse
  appears either.
- **Closed form only in the two stated regimes.** The general case is an
  explicitly finite truncation with a computed error, and `classify_regime`
  makes that visible in the type. Never describe the general non-abelian band
  holonomy as closed form.
- **Open Wilson lines are gauge dependent.** Any physical quantity must come
  from a closed loop or a properly contracted object. G4 enforces this.
- This spec adds an operator. It makes **no claim about Yang-Mills**, about a
  mass gap, or about the continuum limit. Where its output feeds
  `certified_transfer_matrix_gap`, that function's own honesty text applies
  verbatim: a fixed matrix, at one spacing, in finite dimension, with
  `continuum_claim = False`.
- Certificate tier: sound enclosure for the Magnus truncation bound.

## 11. Open questions and risks

- **Magnus convergence.** The expansion converges only for
  `integral ||X|| ds` below a threshold (a known bound of order `pi` for the
  standard estimate). Outside it the truncation is meaningless, and the API must
  refuse rather than return a number.
- **Matrix exponentials in a training loop** are expensive and their gradients
  are delicate. For `SU(2)` the Rodrigues formula is cheap and exact; for larger
  groups, measure before promising.
- **Detection reliability.** `classify_regime` must be conservative: a
  false positive for a closed-form regime silently produces wrong physics.
  Prefer `PRODUCT` when unsure.
- **Falsifier.** If the closed-form regimes never occur in realistic
  configurations, the operator reduces to a wrapper around the existing
  transport and should be presented as one.

## 12. Implementation checklist

- [ ] `packages/omnibias-geometry/src/omnibias/geometry/gauge/band/_core.py`
- [ ] torch and jax twins reusing `wilson_line_exponents` and
      `parallel_transport_from_arrays`; no second transport implementation
- [ ] `classify_regime` with a conservative default and a false-positive test
- [ ] Closed-form exactness tests versus high-`substeps` transport
- [ ] Magnus bound soundness test plus a refusal test outside the convergence
      radius
- [ ] Gauge-covariance and loop-invariance tests
- [ ] `benchmarks/holonomy_band.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
