# CCF-on-the-line calibration

The Córdoba–Córdoba–Fontelos (CCF) equation
\(\theta_t + (H\theta)\,\theta_x = 0\) is the canonical 1D nonlocal-transport
analogue of the 3D Euler vorticity equation. Its **self-similar blow-up
profile** \(\Theta\) solves the stationary equation

\[
\mathcal{E}(\Theta,\lambda)
  = (1+\lambda)\,y\,\Theta'(y) - \lambda\,\Theta(y) + s\,(H\Theta)(y)\,\Theta'(y) = 0
\]

(transport form; the flux form adds \(+\,s\,\Theta\,(H\Theta)'\)). The existence
of a smooth \(\Theta\) at an admissible \(\lambda\) forces a finite-time gradient
singularity, with amplitude \(\sim (1-t)^{\lambda}\) and gradient
\(\sim (1-t)^{-1}\).

`certified_ccf_selfsimilar_blowup_attempt` represents \(\Theta\) in the
**verified even Poisson basis** \(p_a(x)=a/(x^2+a^2)\), whose whole-line Hilbert
transform \(H[p_a]=q_a(x)=x/(x^2+a^2)\) is *exact and quadrature-free*
([`omnibias.core.verified.line`](../api/core.md)). It then evaluates the residual
and its Jacobian/Hessian as outward-rounded intervals on a verified collocation
grid and attempts a **Newton–Kantorovich radii-polynomial closure**. This page
is the honest calibration: *which inequality closed or failed, and by how much.*

!!! warning "Scope — this is a research phase that can end `BLOCKED`"
    A closure certifies a true zero of the **finite collocation map** near the
    candidate — a profile that makes the residual vanish at the collocation
    nodes, plus a two-sided enclosure of \(\lambda\). The whole-line residual
    sup-norm is now **certified** (`residual_certified_sup`), and the continuum
    profile linearization carries a **certified operator-norm bound + Neumann
    invertibility test** (`continuum_operator`). What stays open is driving the
    certified residual and the Neumann \(\rho\) below their closure thresholds.
    This is a 1D-model result: `honesty.unproven_claim`, `three_d_claim`, and
    `whole_line_certified` are all `False`. It is **not** 3D Navier–Stokes/Euler
    and **not** a global-regularity claim.

## The radii polynomial

Because the CCF collocation map is **quadratic** in the unknowns
\((c_2,\dots,c_n,\lambda)\) (the first coefficient \(c_1\) is the fixed
amplitude normalization) with a *constant* Hessian, the Newton–Kantorovich
radii polynomial is exact:

\[
g(r) = Z_2\,r^2 - (1-Z_1)\,r + Y_0,
\qquad
\begin{aligned}
Y_0 &= \lVert B\,\mathcal{E}(\bar x)\rVert_\infty,\\
Z_1 &= \lVert I - B\,D\mathcal{E}(\bar x)\rVert_\infty,\\
Z_2 &= \lVert B\rVert_\infty\,\sup\lvert D^2\mathcal{E}\rvert,
\end{aligned}
\]

where \(B\approx D\mathcal{E}(\bar x)^{-1}\) (the
[`neumann_inverse_norm_bound`](../api/core.md) substrate certifies
\(\lVert B\rVert\) and \(Z_1\)). The closure
\(g(r)\le 0\) admits a solution — and a true profile within \(r_-\) of the
candidate, unique up to \(r_+\) — **iff**

\[
Z_1 < 1
\qquad\text{and}\qquad
\Delta := (1-Z_1)^2 - 4\,Z_2\,Y_0 \;\ge\; 0 .
\]

\(\Delta\) is evaluated as a rigorous *lower* bound (outward-rounded upper bounds
for \(Y_0, Z_1, Z_2\)), so a reported closure is theorem-grade.

## A candidate that does not close (`BLOCKED`)

```python
from omnibias.pinn.certified import certified_ccf_selfsimilar_blowup_attempt

cert = certified_ccf_selfsimilar_blowup_attempt(
    coeffs=[1.0, 0.4, -0.2], scales=[0.6, 1.3, 2.1], lam=0.5,
)
assert cert["closure_certified"] is False
print(cert["closure_report"]["failed_inequality"])
```

The report names the unmet inequality and the size of the gap:

| quantity | value | meaning |
|---|---|---|
| \(Y_0\) | `1.984` | normalized residual — far from zero |
| \(Z_1\) | `7.9e-14` | linear defect (the inverse is fine) |
| \(Z_2\) | `34.51` | curvature \(\lVert B\rVert\,\sup\lvert D^2\mathcal{E}\rvert\) |
| \(\Delta\) | **`-272.9`** | discriminant **< 0 → no closure** |

The verdict is `BLOCKED`, and `failed_inequality` reads:
*"radii-polynomial discriminant (1-Z1)^2 - 4 Z2 Y0 = -272.885 < 0 (short by
272.885); residual Y0 = 1.98402 too large relative to inverse norm and curvature
Z2 = 34.5115."* The gap is dominated by \(Y_0\): the generic profile is simply
not close to a self-similar solution.

## Find, then certify (`PROVED`, collocation-level)

`refine_ccf_selfsimilar_profile` runs an ordinary float Newton iteration (the
*find* step — not a proof) on the normalized collocation system; the certificate
then attempts the rigorous closure.

```python
from omnibias.pinn.certified import (
    certified_ccf_selfsimilar_blowup_attempt,
    refine_ccf_selfsimilar_profile,
)

refined = refine_ccf_selfsimilar_profile(
    coeffs=[1.0, -0.5, 0.3], scales=[0.6, 1.3, 2.1], lam=0.6,
)
# refined["residual_max_abs"] ~ 2e-16; refined["lam"] ~ -1.283174

cert = certified_ccf_selfsimilar_blowup_attempt(
    coeffs=refined["coeffs"], scales=refined["scales"],
    lam=refined["lam"], nodes=refined["nodes"],
)
assert cert["closure_certified"] is True
print(cert["lambda_enclosure"])          # two-sided certified lambda
print(cert["profile_enclosure_radius"])  # r_minus
```

Now the radii polynomial closes:

| quantity | value | meaning |
|---|---|---|
| \(Y_0\) | `5.3e-14` | residual at the refined candidate is ~machine zero |
| \(Z_1\) | `1.4e-13` | linear defect |
| \(Z_2\) | `75.69` | curvature |
| \(\Delta\) | **`1.000`** | discriminant **≥ 0 → closure** |
| \(r_-\) | `5.3e-14` | existence radius (enclosure of \(\lVert x^*-\bar x\rVert\)) |
| \(r_+\) | `1.3e-2` | uniqueness radius |

with the certified two-sided enclosure
\(\lambda^\ast \in [-1.2831742073,\,-1.2831742073]\) (radius \(\sim 5\times
10^{-14}\)). The verdict through the
[prove/disprove machine](proof-machine.md) is `PROVED`:

```python
from omnibias.pinn.certified import build_default_machine
from omnibias.core.proof import Conjecture

v = build_default_machine().evaluate(
    Conjecture("CCF self-similar", "ccf_selfsimilar_blowup",
               {"coeffs": [1.0, -0.5, 0.3], "scales": [0.6, 1.3, 2.1],
                "lam": 0.6, "refine": True})
)
assert v.status == "PROVED"
assert v.replay_ok is True   # the numpy line-Hilbert twin agrees
```

## How far is the remaining gap?

This is the literal *"how far is the remaining 95%"* question. A collocation
closure makes the residual vanish at the \(n\) nodes — but **not** on the whole
line. The same certificate quantifies that gap honestly, from
`cert["closure_report"]`:

| quantity | value | what it measures |
|---|---|---|
| `residual_sampled_sup` | `1.149` | \(\sup\lvert\mathcal{E}\rvert\) **sampled** between nodes on \([0,y_{\text{trunc}}]\) — a lower-bound estimate |
| `residual_certified_core_sup` | **`1.225`** | **rigorous** \(\sup_{\lvert y\rvert\le y_{\text{trunc}}}\lvert\mathcal{E}\rvert\) via per-cell Taylor models |
| `residual_certified_sup` | **`1.225`** | certified whole-line \(\sup_{y\in\mathbb R}\lvert\mathcal{E}\rvert=\max(\text{core},\text{far-field})\) |
| `far_field_residual_bound` | `0.269` | certified \(\lvert\mathcal{E}(y)\rvert\le\) this for \(\lvert y\rvert\ge y_{\text{trunc}}\) |
| `far_field_trunc` | `5.76` | the truncation radius \(y_{\text{trunc}}\) |
| `hilbert_far_field_tail_on_core` | `0.071` | far-field contribution to \(H[\Theta]\) on the core (via `hilbert_tail_bound`) |

The headline: the 3-node collocation profile interpolates a self-similar zero at
the nodes, yet its genuine whole-line residual is \(O(1)\) between them. Note the
**certified** sup `1.225` is *strictly larger* than the **sampled** `1.149` — a
200-point grid simply misses the true peak by \(\sim 7\%\), which is exactly why
a sampled sup is not a proof.

### The between-node sup is now certified (Taylor models)

The certificate discharges the between-node obligation with the
[`TaylorModel`](../api/core.md) substrate. On each cell
\([y_k,y_{k+1}]\) it builds the residual

\[
\mathcal{E}(y) = (1+\lambda)\,y\,\Theta'(y) - \lambda\,\Theta(y)
              + s\,(H\Theta)(y)\,\Theta'(y)
\]

as a **degree-6 Taylor model in the relative variable** \(y-\text{center}\),
assembling each \(p_a, p_a', q_a, q_a'\) from a rigorous
`TaylorModel.reciprocal` of \(y^2+a^2>0\) (a geometric \(1/(1+g)\) series with a
certified analytic tail). Unlike a point sample, `model.bound()` rigorously
encloses \(\mathcal{E}\) over the *whole* cell, so the max of its magnitude over
all cells is a certified \(\sup\). Because \(\mathcal{E}\) is even, partitioning
\([0,y_{\text{trunc}}]\) covers \([-y_{\text{trunc}},y_{\text{trunc}}]\); cells
whose relative variation is too large for the reciprocal series are bisected
automatically. Combined with the far-field tail this yields a certified
*whole-line* residual sup-norm — replacing the merely sampled `residual_sampled_sup`.

### The continuum linearized operator is now bounded

The next piece of the continuum gap — the Fréchet-derivative bound — is also
certified, in `cert["continuum_operator"]`. The profile linearization of the
transport residual is

\[
D\mathcal{E}[\Theta]\,h
  = (1+\lambda)\,y\,h' - \lambda\,h + s\,(H\Theta)\,h' + s\,(Hh)\,\Theta'.
\]

Split it as \(D\mathcal{E}=T+R\) with the **scaling part**
\(T=(1+\lambda)\,y\partial_y-\lambda\) and the nonlocal remainder \(R\). The
dilation group \((U_t f)(y)=e^{t/2}f(e^t y)\) is unitary on \(L^2(\mathbb R)\),
so its generator \(J=\tfrac12+y\partial_y\) is skew-adjoint and
\(T=-(\tfrac12+\tfrac32\lambda)\,I+(1+\lambda)\,J\) is **normal**. Hence the
resolvent norm is *exact* (using \(\langle y h',h\rangle=-\tfrac12\lVert h\rVert^2\)):

\[
\lVert T^{-1}\rVert_{L^2}=\frac{1}{\lvert \tfrac12+\tfrac32\lambda\rvert}=:\kappa
\qquad(\lambda\neq-\tfrac13).
\]

Because \(H\Theta/y=\sum_i c_i/(y^2+a_i^2)\) is **bounded**, \(RT^{-1}\) is bounded
on \(L^2\), giving a certified Neumann ratio

\[
\rho:=\lVert RT^{-1}\rVert_{L^2}
  \le \lVert H\Theta/y\rVert_\infty\,\frac{1+\lvert\lambda\rvert\kappa}{\lvert1+\lambda\rvert}
    + \lVert\Theta'\rVert_\infty\,\kappa,
\]

and **if \(\rho<1\) then \(D\mathcal{E}\) is invertible on \(L^2\)** with
\(\lVert D\mathcal{E}^{-1}\rVert\le\kappa/(1-\rho)\) — a genuine *whole-line*
operator certificate, not a finite section. The sup-norms are certified by the
same per-cell Taylor-model machinery as the residual.

For the refined 3-node profile (`cert["continuum_operator"]`):

| quantity | value | meaning |
|---|---|---|
| `scaling_inverse_norm_bound` \(\kappa\) | `0.702` | **exact** \(1/\lvert\tfrac12+\tfrac32\lambda\rvert\) |
| `htheta_over_y_sup` | `2.091` | certified \(\lVert H\Theta/y\rVert_\infty\) |
| `theta_prime_sup` | `1.502` | certified \(\lVert\Theta'\rVert_\infty\) |
| `forward_operator_bound` | `4.593` | certified \(\lVert D\mathcal{E}\rVert_{X\to L^2}\) |
| `neumann_rho` \(\rho\) | **`15.09`** | \(\rho\ge1\Rightarrow\) **invertibility not yet certified** |

So the continuum *Fréchet derivative* is bounded (`forward_operator_bound`
finite — obligation discharged), but the *inverse* test does not yet close
(\(\rho=15.1\gg1\)) for this large-amplitude profile. A small-amplitude profile
does close, exercising the full path:

```python
from omnibias.pinn.certified import certified_ccf_linearized_operator_bound

op = certified_ccf_linearized_operator_bound(
    coeffs=[0.03, -0.02, 0.01], scales=[0.6, 1.3, 2.1], lam=0.6,
)
assert op["rho_closes"] is True               # rho ~ 0.10 < 1
assert op["continuum_invertible_certified"] is True
print(op["inverse_norm_bound"])               # ||DE^{-1}||_{L^2} <= ~0.80
```

The flux form is honestly out of scope here: its extra \(s\,\Theta\,(Hh)'\) term
needs an \(H^1\) space, so `continuum_operator["supported"]` is `False` with its
own obligation. What remains for a theorem is to drive **both** the certified
residual and the Neumann \(\rho\) below their thresholds (more basis terms/nodes,
and the nonlocal compactness). The remaining obligations are recorded verbatim:

```python
print(cert["linearized_operator"]["open_obligations"])
# ['continuum_linearized_neumann_rho_below_one',
#  'shrink_certified_residual_sup_below_function_space_closure_threshold']
```

## Independent replay

Every certificate has a numpy-only twin in
[`omnibias.symbolic.ccf`](../api/symbolic.md) that imports nothing from
`omnibias.pinn.certified`, recomputes \(Y_0, Z_1, Z_2, \Delta\) from a *separate* closed-form
line-Hilbert code path, and confirms the closed/blocked verdict:

```python
from omnibias.symbolic.ccf import (
    verify_ccf_selfsimilar_blowup_attempt,
    verify_ccf_linearized_operator_bound,
)

report = verify_ccf_selfsimilar_blowup_attempt(cert)
assert report["replay_match"] is True

# the continuum operator block has its own independent twin: it recomputes kappa
# and rho, and *densely samples* the coefficient functions to confirm the
# certified sup-norms genuinely dominate (a forged, understated sup is caught).
op_replay = verify_ccf_linearized_operator_bound(cert["continuum_operator"])
assert op_replay["replay_match"] is True
assert op_replay["sup_dominates_samples"] is True
```

The exactness of \(H[p_a]=q_a\) — on which the whole closure rests — is
anti-faking-tested against an independent high-precision principal-value
quadrature (`mpmath`) at the collocation nodes.

## Where this sits

- Certificate + closure: `omnibias.pinn.certified.certified_ccf_selfsimilar_blowup_attempt`.
- Continuum linearized-operator bound: `omnibias.pinn.certified.certified_ccf_linearized_operator_bound` (exact dilation-generator resolvent + Neumann test).
- Find step: `refine_ccf_selfsimilar_profile`; closure algebra: `radii_polynomial_closure`.
- Verified substrate: [`omnibias.core.verified.line`](../api/core.md) (exact even-profile Hilbert), [`TaylorModel`](../api/core.md) (between-node + coefficient sups), and `omnibias.core.verified.linalg` (Neumann inverse bound).
- Front door + verdict semantics: [prove/disprove machine](proof-machine.md).
- The transport/flux residual itself: [CCF singularities](ccf-singularity.md).
