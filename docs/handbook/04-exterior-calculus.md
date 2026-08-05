# Chapter 4 — Exterior calculus & the de Rham–Hodge complex

> Module: `omnibias.symbolic` (source: `omnibias.symbolic.exterior_discovery`).
> [API reference](../api/symbolic.md) has the precise signatures.

## What this chapter gives you

This is the antisymmetric-tensor face of the field surface. A **differential
`k`-form** is a thing you integrate over `k`-dimensional surfaces; in coordinates
it is a sum \(\omega = \sum_I \omega_I\,dx^I\) over increasing multi-indices `I`.
Here each coefficient \(\omega_I\) is a `FieldJet`, so it carries its exact
closed-form partials, and the operators below are exact.

The star of the show is the **exterior derivative `d`** — a single, metric-free
operator that *is* the gradient on a 0-form, the curl on a 1-form, and the
divergence on a 2-form. Its defining property,

\[
  d\circ d = 0,
\]

is exactly the classical pair `curl(grad f) = 0` and `div(curl F) = 0`. Because
mixed partials are exact (symmetry of second derivatives is exact, not
approximate), `d∘d = 0` holds here to **machine precision at any order**. The same
identity is the homogeneous Maxwell law `dF = 0` for `F = dA`.

```
 0-form f ──d──▶ 1-form df (grad) ──d──▶ 2-form (curl) ──d──▶ 3-form (div) ──d──▶ 0
            ⋆ Hodge star ↕            δ codifferential ↕         Δ = dδ + δd
```

The Hodge star `⋆`, codifferential `δ`, and Hodge Laplacian `Δ` are implemented
for the **flat Euclidean** metric with standard orientation (the curved
Laplace–Beltrami on functions is [Chapter 3](03-differential-geometry.md)).

---

## Building forms

### `DifferentialForm`

A frozen dataclass: `degree`, `dim`, `components` (a dict from each increasing
`k`-index to a `FieldJet`), and `var_names`. Degree-0 forms use the single key
`()`. Properties `.order`, `.n`, `.X`; methods `.component(I)`, `.value(I)`,
`.max_abs()` (sup-norm of the component values). All components share sample
points and jet order so the form differentiates as a whole.

### `scalar_form`

<!-- docs-test: signature -->
```python
scalar_form(jet) -> DifferentialForm   # degree 0
```
Wrap a scalar field's `FieldJet` as a 0-form.

### `one_form`

<!-- docs-test: signature -->
```python
one_form(jets) -> DifferentialForm     # degree 1
```
Build a 1-form \(\sum_i \omega_i\,dx^i\) from one component `FieldJet` per axis
(needs exactly `dim` of them).

### `differential_form`

<!-- docs-test: signature -->
```python
differential_form(degree, components, *, dim=None, var_names=None) -> DifferentialForm
```
Assemble a general `k`-form from a dict of components, normalising index keys to
increasing tuples. Use for 2-forms and higher.

```python
import numpy as np
from omnibias.symbolic import (
    fit_neural_field_nd, extract_field_jet, scalar_form, one_form,
)
rng = np.random.default_rng(0)
X = rng.uniform(-0.6, 0.6, size=(48, 3))
names = ("x", "y", "z")
fjet = extract_field_jet(fit_neural_field_nd(X, rng.normal(size=48), hidden=32, seed=0, var_names=names), X, max_order=3)
gjets = [extract_field_jet(fit_neural_field_nd(X, rng.normal(size=48), hidden=24, seed=10+i, var_names=names), X, max_order=3) for i in range(3)]

f0 = scalar_form(fjet)            # a 0-form
omega = one_form(gjets)           # a 1-form
print(f0.degree, omega.degree)    # 0 1
```

---

## The operators

### `exterior_derivative`

<!-- docs-test: signature -->
```python
exterior_derivative(form) -> DifferentialForm    # degree +1, order -1
```
The metric-free `d`:
\((d\omega)_{j_0\dots j_k} = \sum_p (-1)^p\,\partial_{j_p}\,\omega_{j_0\dots\hat{j_p}\dots j_k}\).
On a 0-form it is the gradient 1-form; on a 1-form (3-D) the curl 2-form; on a
2-form (3-D) the divergence 3-form. Raises on a top-degree form or order `< 1`.

```python
from omnibias.symbolic import exterior_derivative
df = exterior_derivative(f0)      # the gradient 1-form of f
print(df.degree, df.order)        # 1, fjet.order - 1
```

### `hodge_star`

<!-- docs-test: signature -->
```python
hodge_star(form) -> DifferentialForm     # degree m-k, order preserved
```
The flat-Euclidean Hodge star `⋆`: \(\star(dx^I) = \varepsilon(I, I^c)\,dx^{I^c}\)
with `I^c` the complementary increasing index. `⋆1` is the volume form, and
\(\star\star = (-1)^{k(m-k)}\). No differentiation.

```python
import numpy as np
from omnibias.symbolic import hodge_star
back = hodge_star(hodge_star(omega))   # ** = (-1)^{1*2} = +1 here (k=1, m=3)
print(np.allclose(back.value((0,)), omega.value((0,))))   # True
```

### `codifferential`

<!-- docs-test: signature -->
```python
codifferential(form) -> DifferentialForm    # degree -1, order -1
```
The flat-Euclidean codifferential `δ`, the metric adjoint of `d`:
\((\delta\omega)_{I'} = -\sum_a \partial_a\,\omega_{aI'}\). On a 1-form it is
*minus* the divergence; `δ∘δ = 0`; and \(\delta = (-1)^{m(k+1)+1}\star d\star\).
Raises on a 0-form.

```python
from omnibias.symbolic import codifferential
div_like = codifferential(omega)   # -∇·F, a 0-form
print(div_like.degree)             # 0
```

### `hodge_laplacian`

<!-- docs-test: signature -->
```python
hodge_laplacian(form) -> DifferentialForm    # degree preserved, order -2
```
The Hodge–de Rham Laplacian \(\Delta = d\delta + \delta d\). On the flat Euclidean
metric it acts component-wise as *minus* the ordinary Laplacian (the
zero-curvature Weitzenböck identity), so on a 0-form
\(\Delta f = -\sum_a\partial^2_a f\). Needs order `≥ 2`.

```python
import numpy as np
from omnibias.symbolic import hodge_laplacian, field_laplacian
hf = hodge_laplacian(f0).value(())
print(np.allclose(hf, -field_laplacian(fjet)))    # True
```

### `wedge`

<!-- docs-test: signature -->
```python
wedge(a, b) -> DifferentialForm    # degree p+q (values only)
```
The pointwise exterior product `a ∧ b`:
\((a\wedge b)_K = \sum_{I+J=K}\varepsilon(I,J)\,a_I b_J\). The result is an
order-0 form (values only) — enough for graded commutativity
\(a\wedge b = (-1)^{pq} b\wedge a\) and \(\alpha\wedge\alpha = 0\) for odd-degree
`α`.

```python
import numpy as np
from omnibias.symbolic import wedge
ab = wedge(df, df)                 # df ∧ df = 0 (odd degree wedged with itself)
print(ab.max_abs() < 1e-12)        # True
```

---

## Correspondence & physics

### `gradient_form` and `curl_form`

<!-- docs-test: signature -->
```python
gradient_form(scalar_jet) -> DifferentialForm    # d of a 0-form = grad
curl_form(vector_jets) -> DifferentialForm        # d of a 1-form = curl
```
Convenience wrappers naming the classical operators. The component values of
`gradient_form(f)` equal `field_gradient(f)`; those of `curl_form(F)` (3-D) equal
`field_curl(F)`.

### `electromagnetic_field_2form`

<!-- docs-test: signature -->
```python
electromagnetic_field_2form(potential) -> DifferentialForm
```
The field strength `F = dA` from a 1-form potential `A`. The homogeneous Maxwell
equations `dF = 0` are then exactly the `d∘d = 0` identity.

```python
from omnibias.symbolic import electromagnetic_field_2form, closedness_residual
F = electromagnetic_field_2form(omega)        # A := omega (a 1-form potential)
print(closedness_residual(F) < 1e-10)         # True  -> dF = 0 (homogeneous Maxwell)
```

### `closedness_residual` and `coclosedness_residual`

<!-- docs-test: signature -->
```python
closedness_residual(form) -> float      # sup-norm of dω   (0 ⇔ closed)
coclosedness_residual(form) -> float    # sup-norm of δω   (0 ⇔ co-closed)
```
Scalar certificates. A gradient field is closed (`dω = 0`); a divergence-free
field is co-closed. Use them as machine-precision identity checks in tests.

```python
from omnibias.symbolic import gradient_form, closedness_residual
print(closedness_residual(gradient_form(fjet)) < 1e-10)   # True: curl(grad f) = 0
```

### `evaluate_exterior_calculus`

<!-- docs-test: signature -->
```python
evaluate_exterior_calculus(*, seed=0) -> dict[str, float]
```
Fits a random 3-D field and reports the sup-norm residuals of the headline
identities — `d d f = 0` (curl grad), `d d ω = 0` (div curl), `δ δ = 0`, the
Hodge-star roundtrip `⋆⋆ = (-1)^{k(m-k)}`, and `Δ f = −lap f` — all exact to
machine precision. The one-call certification of the whole exterior surface.

```python
from omnibias.symbolic import evaluate_exterior_calculus
report = evaluate_exterior_calculus(seed=0)
print({k: f"{v:.1e}" for k, v in report.items()})
# every residual ~1e-12 or smaller
```

---

**Next:** [Chapter 5 — Information theory](05-information-theory.md) switches from
geometry to probability: entropy, divergences, and mutual information, in
differentiable, NumPy, and certified-interval flavours.
