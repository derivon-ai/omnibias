# omnibias-verify

Certified neural-network verification on top of the rigorous derivative tower
(`omnibias.core.verified`). Two layers:

- **Exact `_core`** (pure Python, no backend imports): a backend-neutral
  `Network` description and rigorous forward propagation. Interval bound
  propagation (IBP) is the sound baseline; the Taylor-model engine propagates a
  multivariate polynomial enclosure (`TaylorModelMV`) through the net, composing
  smooth activations (`tanh` / `sigmoid` / exact `GELU`) through the closed-form
  derivative tower and relaxing the nonsmooth ones (ReLU triangle, group
  `max`-pool), with branch-and-bound over the input box for arbitrary tightness.
- **Backend frontends** (`torch` / `jax`): ingest a trained `nn.Sequential` or a
  JAX `(W, b)` parameter stack into the neutral `Network` so verification never
  touches the framework again. Equal weights produce **bit-identical** networks.

!!! note "Soundness, not completeness"
    Every enclosure provably contains the true output set, so a "certified"
    verdict is a proof. The verifier is **not complete**: it returns an
    inconclusive verdict rather than over-claim when branch-and-bound runs out of
    budget. This is verification of a fixed, trained network — not training or
    attack search.

## Why Taylor models beat IBP

IBP collapses every layer to a box, so it forgets that the *same* input drives
many neurons; the errors compound. A Taylor model keeps the polynomial *shape* in
the input variables all the way through, so correlated terms cancel
(`x - x = 0` exactly). The activation expansion only converges while the
per-neuron input range stays inside the radius of convergence (`pi/2` for `tanh`),
so past that the engine gracefully falls back to the interval enclosure; the
final bound is always intersected with an IBP pass and is therefore **never
looser than IBP**. Branch-and-bound splits the box to bring each neuron back
inside the radius where the polynomial wins.

```python
from omnibias.core.verified.interval import Interval
from omnibias.verify import Network, affine_layer, TanhLayer
from omnibias.verify import taylor_output_bounds, certify_robustness, lipschitz_bound

net = Network([
    affine_layer([[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0]),
    TanhLayer(),
    affine_layer([[1.0, 1.0]], [0.0]),
])
box = [Interval(-0.4, 0.4), Interval(-0.4, 0.4)]

# Tighter-than-IBP output enclosure.
(out,) = taylor_output_bounds(net, box, order=3)

# A Lipschitz bound (max abs row sum of the interval Jacobian).
L = lipschitz_bound(net, box, norm="inf")

# Robustness of an L-infinity ball around a point.
cert = certify_robustness(net, x0=[0.1, -0.2], eps=0.05, true_label=0)
```

## Certificates

Four sound guarantees, all computed with outward-rounded arithmetic:

- **robustness** — the class margin stays positive over an `L^inf` ball, proved
  with the branch-and-bound read-out (`certify_robustness`);
- **Lipschitz** — an induced-operator-norm bound from a rigorous interval
  enclosure of the Jacobian (`lipschitz_bound`, `interval_jacobian`);
- **monotonicity** — the certified sign of a single partial derivative over the
  box (`monotonicity`);
- **reachable set** — an axis-aligned enclosure of the output set
  (`reachable_box`).

## Certified global optimization

Gradient descent (and L-BFGS, Adam, …) find a *local* minimum and can never
certify it is the global one: for a non-convex objective the true minimum may sit
in an unexplored basin. `certified_minimize` answers the global question with a
**proof**. It runs an interval branch-and-bound (Moore–Skelboe) that maintains a
rigorous enclosure `[f_lower, f_upper]` of the global minimum over a box, so

```
f_lower <= min_{x in box} f(x) <= f_upper
```

holds **unconditionally** — regardless of the budget — and the search stops once
the certified gap `f_upper - f_lower` drops below `tol`. The objective is written
as an *interval extension* (`Interval` algebra plus the `*_iv` transcendentals from
`omnibias.core.verified`); the frontier is a priority queue keyed by each sub-box's
certified lower bound, so a box that already cannot beat the incumbent is pruned
soundly.

Two omnibias-specific accelerators use *exact* interval-derivative enclosures (from
the closed-form `omnibias.core.verified.jet_mv`, or supplied by hand):

- **monotonicity test** — if `df/dx_i` has a constant sign over a box, the minimum
  lies on a face, so the axis is collapsed to the min-achieving endpoint;
- **mean-value (centered) form** — `f(box) ⊆ f(c) + Σ_i g_i(box)(x_i − c_i)`
  intersected with the natural extension, which cancels the first-order dependency
  overestimation and gives a much tighter lower bound.

`certify_strict_local_min` complements it: via the interval `LDLᵀ` inertia
(`omnibias.core.verified.eig_operator.is_positive_definite`) it certifies the
Hessian is positive definite over a box, upgrading "best found" to a *strict*
minimizer.

For an **omnibias network** you don't write the interval extension at all.
`certified_network_minimize` takes a trained `JetMLP`-like model (any object
exposing `_layer_specs()` — the omnibias torch/jax PINNs do) or a raw
`(W, b, name)` layer list, and the closed-form verified jet
(`omnibias.core.verified.jet_mv`) supplies the value enclosure (jet row 0), the
exact interval **gradient** (`jet_gradient`), and — for
`certify_network_strict_local_min` — the Hessian (`jet_hessian`). It is the same
`σ⁽ⁿ⁾` derivative tower that drives the differentiable and PDE registers, now
producing a *global* certificate:

```python
from omnibias.verify import certified_network_minimize

# u(x, y) = -exp(-x^2/2) - exp(-y^2/2): a gaussian-bump well, min -2 at the origin.
layers = [
    ([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], "gaussian"),
    ([[-1.0, -1.0]], [0.0], None),
]
r = certified_network_minimize(layers, [(-2.0, 2.0), (-2.0, 2.0)], tol=1e-3)
assert r.f_lower <= -2.0 <= r.f_upper      # proved enclosure of the global minimum
# or pass a trained network directly:  certified_network_minimize(net, box)
```

```python
from omnibias.verify import certified_minimize

# Six-hump camel: an interval extension (one Interval per axis -> Interval).
# `Interval` arithmetic accepts plain floats, so the algebra reads like the formula.
def camel(b):
    x, y = b
    return (4.0 - 2.1 * x**2 + x**4 / 3.0) * x**2 + x * y + (-4.0 + 4.0 * y**2) * y**2

def camel_grad(b):
    x, y = b
    return [8.0 * x - 8.4 * x**3 + 2.0 * x**5 + y,
            x - 8.0 * y + 16.0 * y**3]

r = certified_minimize(camel, [(-3.0, 3.0), (-2.0, 2.0)], tol=1e-4, grad=camel_grad)
assert r.converged                           # certified gap <= tol
assert r.f_lower <= -1.031628 <= r.f_upper   # proved enclosure of the global minimum
```

!!! note "Sound at any dimension, but low-dimensional in practice"
    The enclosure is sound in every dimension, but branch-and-bound cost grows
    exponentially in the *worst* case. Empirically the driver is **enclosure
    looseness (dependency overestimation)**, not the raw dimension: objectives
    whose interval extension is exact at the minimizer (e.g. a sum of squares)
    certify in a single box independent of `d`, whereas a loose bound (repeated
    variables, coupling) forces refinement. This is for low-dimensional global
    problems and certified read-outs — not million-parameter training.

### What "certified" buys you

Certification is a different guarantee from what a stochastic global optimiser
provides, not a claim of finding *lower* minima. On a suite of standard non-convex
2-D test functions (six-hump camel, Himmelblau, Branin, Rastrigin, Ackley, Beale,
Goldstein–Price, Booth, Matyas, three-hump camel, Lévi N.13, McCormick, Easom,
Styblinski–Tang):

- **omnibias returns a proof.** It closes the certified gap to tolerance on all of
  them (typically sub-second with the exact interval-derivative accelerators). The
  hardest is Goldstein–Price, whose product-of-quadratic-forms structure has a very
  loose *natural* extension — an independent branch-and-bound on the mature
  codac/Ibex interval kernel bottoms out at a lower bound of **−404** on the same
  box. The exact interval **gradient** (mean-value form) already closes it, and the
  **second-order + interval-Newton** accelerators cut the box count several-fold
  further; only the natural-extension-only path (no derivatives) stalls.
- **Uncertified solvers usually find the global — silently.** SciPy
  `differential_evolution` and `dual_annealing` locate the global optimum on almost
  every function, but return no enclosure and no proof; the deterministic `shgo`
  misses the global on six-hump camel; and random-restart local BFGS is trapped in
  a non-global basin on the multimodal functions (up to 100 % of restarts on
  Rastrigin / Lévi / Ackley). None of them can *certify* that no better point
  exists in the box.

So the branch-and-bound is the tool you reach for when you need the *guarantee*
`f_lower ≤ min f ≤ f_upper` (verified optimization, safety envelopes, certified
read-outs), and a population/annealing method is the tool you reach for when you
only need a good point fast. As an **independent soundness witness**, a
branch-and-bound built on the codac/Ibex interval kernel brackets the same minima on
all 14 functions and its certified intervals overlap omnibias's — while omnibias's
exact-derivative accelerators make its enclosures tighter at equal effort. The niche
is that the *same* closed-form derivative tower supplies the enclosures, gradients,
and Hessians with no external solver.

### Certified output bounds for a trained network

Because `certified_network_minimize` takes the network itself, you can put a
rigorous envelope around a *trained* model's output over an input region — certify
the minimum directly, and the maximum by minimising the negated readout:

```python
import torch
from omnibias.torch.architectures import JetMLP
from omnibias.verify import certified_network_minimize
from omnibias.verify._core.jet_ingest import verified_layers

# Any model exposing `_layer_specs()` works; here a small trained-shaped JetMLP.
torch.manual_seed(0)
trained = JetMLP(in_dim=2, hidden=8, out_dim=1, depth=2, base="tanh")

layers = verified_layers(trained)                    # (W, b, name) tuples
w, b, name = layers[-1]                              # negate the affine readout
neg = [*layers[:-1], ([[-v for v in r] for r in w],
                      None if b is None else [-v for v in b], name)]

box = [(0.0, 1.0), (0.0, 0.2)]
lo = certified_network_minimize(trained, box, tol=1e-2).f_lower  # net(x) >= lo on box
hi = -certified_network_minimize(neg, box, tol=1e-2).f_lower     # net(x) <= hi on box
# [lo, hi] provably encloses the network's output range over the whole region.
```

Unlike a finite grid sweep or a stochastic search, `[lo, hi]` is a *proof*: the
network's output cannot leave it anywhere in the region.

### Second-order bound + interval-Newton contractor

When you also supply an interval **Hessian** (`hess=`), `certified_minimize` adds two
accelerators that use exact curvature — both preserve the unconditional guarantee:

- the **second-order (Taylor) lower bound** `f(c) + g(c)·(x−c) + ½(x−c)ᵀH(box)(x−c)`,
  which converges *quadratically* as boxes shrink and dominates on refined boxes where
  the first-order mean-value form plateaus (it is intersected with the natural and
  mean-value forms, so it can only tighten);
- the **interval-Newton / Krawczyk contractor**, which contracts a strictly interior,
  full-dimensional sub-box to the part that can still hold a stationary point and
  discards it outright when `K(X) ∩ X = ∅`. It is gated by `_interior_full_dim`, so a
  minimizer sitting on the *domain boundary* (where `∇f ≠ 0`) is provably never pruned
  — set `use_newton=False` to keep only the second-order bound.

```python
from omnibias.core.verified.interval import Interval

# Exact interval Hessian of the camel objective above (the constant cross term
# needs an explicit degenerate interval).
def camel_hess(b):
    x, y = b
    one = Interval.point(1.0)
    return [[8.0 - 25.2 * x**2 + 10.0 * x**4, one],
            [one, -8.0 + 48.0 * y**2]]

box = [(-3.0, 3.0), (-2.0, 2.0)]
r = certified_minimize(camel, box, tol=1e-4, grad=camel_grad, hess=camel_hess)
# same certified enclosure, ~3× fewer boxes on camel (more on stiffer objectives)
```

For a network, pass `second_order=True` (the jet is lifted to order 2 and the exact
`jet_hessian` supplies `hess` automatically).

### Warm-started certified minimize

The certified enclosure is unconditional, but branch-and-bound closes its gap faster
from a strong incumbent. `certified_minimize(..., seeds=[pt, …])` — and the same
`seeds=` threaded through `certified_network_minimize` — initialises `f_upper` with
any concrete feasible point(s): each seed is clamped into the box, then evaluated as a
point-box. Because a feasible value can only *lower* `f_upper` and never touches the
sound lower bound, **a seed can never widen the certified enclosure** — it only prunes
sub-boxes sooner, so `boxes_explored` never exceeds the un-seeded run. Correctness is
sound by construction, independent of the seed.

The natural seed source is the network's own **closed-form jet gradient**: a few steps
of projected gradient descent using `jet_gradient` (the exact input-space gradient, no
autodiff graph) locate a good point cheaply. `omnibias.verify.torch.warm_start` and its
bit-identical `omnibias.verify.jax.warm_start` twin expose `descent_seeds(net, box, …)`
and a `warm_started_network_minimize(net, box, …)` convenience — *search* with the fast
differentiable jet, *prove* with the verified one:

```python
from omnibias.verify.torch import warm_started_network_minimize

r = warm_started_network_minimize(trained, [(-1.5, 1.5), (-1.5, 1.5)], tol=1e-2)
assert r.f_lower <= r.f_upper      # identical sound enclosure, reached in fewer boxes
```

### Certified critical points and basin flatness

`certified_critical_points(grad, hess, box)` is the rigorous, tractable form of "solve
`∇f = 0`": a Krawczyk-accelerated branch-and-bound that *encloses every* root, certifies
existence & uniqueness where `K(X) ⊆ int(X)`, and classifies each as `min` / `max` /
`saddle`. On Himmelblau over `[−5,5]²` it returns all nine stationary points (four minima,
four saddles, one maximum), each certified unique. `certified_flatness(hess, box)` returns
a rigorous enclosure of the smallest and largest Hessian eigenvalue — a *certified*
basin-sharpness measure (the exact-curvature analogue of the flat-minima heuristic).

!!! warning "Flatness ranks generalization, not globality"
    A rigorous flatness (basin-width) enclosure characterises *local* curvature; a wide
    basin can still be a **non-global** local minimum. Use it to rank candidate minima or
    as an exact-curvature regulariser — for a global-optimality *proof* use
    `certified_minimize`.

`certified_network_critical_points` and `certified_network_flatness` are the network-fed
wrappers (they source `∇` and `H` from the closed-form order-2 jet).

### Train-then-certify: a sealed network certificate

`certify_trained_network(net, box, ...)` is the one-call bridge from a *trained* model to
a tamper-evident proof. It runs `certified_network_minimize` (and, on request,
`certified_network_flatness` / `certify_network_strict_local_min`), then **seals** the
minimum enclosure into a v1 certificate (`omnibias.core.proof.certificate`) whose `meta`
records the ingested-weight digest (via `verified_layer_bundle`), the input box, the argmin,
the certified gap, and the honest `converged` flag. The returned `NetworkCertificate` exposes
`.verified` (recompute the digest — any post-hoc edit to a bound is detected) and `.converged`
(did the gap reach `tol`):

```python
from omnibias.verify import certify_trained_network

box = [(-1.5, 1.5), (-1.5, 1.5)]
nc = certify_trained_network(trained, box, tol=1e-2, flatness=True)   # trained JetMLP-like net
assert nc.verified                       # sealed digest matches the body (untampered)
assert nc.result.f_lower <= nc.result.f_upper   # unconditional min enclosure over the box
# nc.certificate is canonical JSON; nc.flatness carries the certified Hessian-eigenvalue band
```

See the [train-then-certify cookbook](../cookbook/train-then-certify.md) and the runnable
`examples/train_then_certify.py`. Honest scope is inherited from interval branch-and-bound:
small nets, low-dimensional input boxes, the verified-tower activations
(`tanh` / `sigmoid` / `gaussian` / `silu` / `gelu` / `softplus`) — a certified
read-out, not a continuum / global-regularity-grade statement.

### Proof-carrying training: a strict-local-minimum certificate

The certificates above bound a trained network's *read-out over an input box*; the closed-form
verified jet differentiates the output w.r.t. the **inputs** and treats the weights as constants.
Proof-carrying *training* asks the dual, parameter-space question: given a trained `θ*` and fixed
data, is `θ*` a genuine, locally-unique, strict local minimum of the training loss
`L(θ) = (1/N) Σ_i ‖net_θ(x_i) − y_i‖²` — not merely a point where the optimiser happened to stop?

`certify_trained_min` answers it with a proof. Over a parameter ball `B(θ*, radius)` it (1) proves a
**locally-unique stationary point** via the Krawczyk operator on `∇_θ L`, (2) proves the Hessian is
**positive definite** on the whole box via the interval `LDLᵀ` inertia
(`omnibias.core.verified.eig_operator.is_positive_definite`, bracketed by a positive-definite shift
so `eig_min.lo > 0` is a *sharp* certified lower bound), and (3) **seals** a v1 certificate whose
interval payload is that `eig_min` enclosure — so the finite `eig_min > 0` sign obligation is
re-checkable by the Lean kernel (`lean=True`; it sets `theorem_prover_verified` only on a genuine
`lake build` pass and degrades gracefully with no toolchain).

When the Hessian box is certified positive definite, the certificate additionally seals its full
interval `LDLᵀ` **pivot vector** as a `pd_certificate` (`cert.positive_definite`). This is the payload
handed to the kernel: the Lean bridge emits the *inertia-vector* obligation
`allPivotsPos [⟨lo,hi⟩, …] = true`, discharged by the sorry-free
`Omnibias.matrix_positive_definite_certified` lemma (`formal/omnibias-verified-kernel/Omnibias/LDLT.lean`).
So `theorem_prover_verified` now reflects **kernel-verified matrix positive-definiteness** — every
pivot of the interval factorisation is proven strictly positive (zero negative inertia) — rather than
only the single scalar `eig_min > 0` shadow. (The factorisation `S = L D Lᵀ` itself, and Sylvester's
congruence step, remain trusted Python inputs — the same trust base as before, only finer-grained.)

The missing ingredient — a rigorous *parameter-space* gradient / Hessian enclosure — is supplied by
`ParamSpaceLoss`, an interval forward-mode second-order jet (hyper-dual numbers over `Interval`)
that propagates through the exact activation towers; it exposes the `grad` / `hessian` / `value`
callbacks the proof stack already consumes.

```python
import math
from omnibias.verify import MLPArchitecture, certify_trained_min, flat_params_from_layers

# a tiny tanh unit  net(x) = v·tanh(w·x + b) + c,  fit exactly to a realizable teacher
arch = MLPArchitecture(dims=(1, 1, 1), activation="tanh")
theta = flat_params_from_layers([([[-1.3]], [0.25]), ([[1.2]], [0.1])])  # [w, b, v, c]

def net(t, x):
    w, b, v, c = t
    return v * math.tanh(w * x + b) + c

grid = [-2.0 + 4.0 * i / 11.0 for i in range(12)]
data = [((x,), (net(theta, x),)) for x in grid]

cert = certify_trained_min(arch, data, theta, radius=1e-3)
assert cert.certified                    # locally-unique stationary point AND positive-definite Hessian
assert cert.verified                     # sealed digest matches the body (untampered)
assert cert.flatness.eig_min.lo > 0.0    # strict: the smallest Hessian eigenvalue is proven positive
assert cert.positive_definite            # the full LDLᵀ pivot vector is sealed as a pd_certificate

# the kernel re-checks the whole inertia vector (allPivotsPos), not just the scalar eig_min:
from omnibias.core.proof import check_certificate, generate_obligation
assert "allPivotsPos" in generate_obligation(cert.pd_certificate)   # matrix-PD obligation, not the scalar shadow
result = check_certificate(cert.pd_certificate)                     # runs `lake build` when a Lean toolchain is present
assert result.verified or not result.available                      # kernel-verified, or gracefully skipped
```

#### Reaching past a single unit — the L2-regularised objective

The certifiable network size is set by *conditioning*, not by arithmetic tightness. An
over-parametrised `tanh` network fit to realisable data sits in a near-flat valley (smallest true
Hessian eigenvalue `~1e-7`): it is not a *strict* minimum of the bare loss at all, so no enclosure
can honestly certify it. The interval Hessian enclosure is already *linear* in the box radius, so
this is a property of the problem — not a looseness to engineer away. (An affine / zonotope
enclosure engine was implemented and measured for exactly this purpose; on this shallow,
cancellation-free Hessian sweep it has no wrapping to fight and only *widened* the bound ~1.4×, so
it was dropped.)

Passing `l2 > 0` instead certifies the **L2-regularised** objective `J(θ) = L(θ) + l2·‖θ‖²` (MSE
plus weight decay), whose Hessian `Hess(L) + 2·l2·I` lifts every eigenvalue by `2·l2`. Because the
regulariser is a pure quadratic it is folded into the gradient (`+2·l2·θ`) and Hessian (`+2·l2·I`)
*analytically and exactly* — no added width. A network genuinely trained with weight decay `l2`
therefore has a strict, certifiable regularised minimum even where the bare loss is flat:

<!-- docs-test: skip reason="train_with_weight_decay stands in for the reader's own optimiser" -->
```python
# a 2-hidden-unit tanh net (P = 7): the two units are redundant, so the *bare* loss min is
# near-flat (not strict, not certifiable). The minimiser of J = L + 0.2·‖θ‖² is, however, a
# certified strict local min of the regularised objective.
arch2 = MLPArchitecture(dims=(1, 2, 1), activation="tanh")
theta_reg = train_with_weight_decay(arch2, data, l2=0.2)  # argmin J (any optimiser)

cert = certify_trained_min(arch2, data, theta_reg, radius=1e-3, l2=0.2)
assert cert.certified                       # eig_min.lo ≈ +0.11 > 0 over B(θ*, 1e-3)
assert cert.certificate["meta"]["l2"] == 0.2
```

Supply the `θ*` that actually minimises `J` (i.e. was trained with the *same* `l2`); the certificate
is about the regularised objective, so a bare-loss minimiser is (correctly) refused at `l2 > 0`.

!!! note "A rigorous *local* proof of a strict minimum — not global training"
    The certificate is a ball around `θ*` for a small `tanh` / `sigmoid` MLP with fixed data,
    matching the documented interval scope. It proves `θ*` is a locally-unique strict local minimum
    (of `L`, or of `J = L + l2·‖θ‖²` when `l2 > 0`); it is **not** a global-optimality or global-regularity-grade
    claim. The certifiable radius shrinks with the smallest Hessian eigenvalue — a near-flat
    direction only certifies on a very small ball (honestly reported as `certified = False` past it),
    and weight decay `l2 > 0` is what lifts that floor by `2·l2`.

### Proof-carrying optimization: a certified subspace trust-region step

`certify_trained_min` certifies where an optimiser *stopped*; `certify_subspace_step` certifies each
step it *takes*. It fuses the differentiable subspace-tensor optimiser
(`omnibias.torch.optim.JetSubspaceTensor`) with the rigorous and formal registers into a single
primitive neither has alone — a **proof-carrying optimization trajectory**.

Restrict the loss to a small `k`-dimensional trust region `ψ(a) = L(θ₀ + Q·a)`, `‖a‖ ≤ r`.
`enclose_subspace_model` builds the order-3 multivariate Taylor model of `ψ` over the box: its
polynomial part is the *exact* reduced model `m(a) = ψ(0) + c·a + ½·aᵀHa + ⅙·T[a,a,a]` (the same
`(c, H, T)` the torch `taylor_subspace_model` returns) and its scalar `.remainder` `R` rigorously
bounds the model-vs-truth error `ψ(a) − m(a)` for **every** `a` in the box. The activation towers are
composed onto the Taylor model in closed form (Makino–Berz: the exact `σ⁽ⁿ⁾` from the verified
`sigma_tower_interval` plus a Lagrange remainder for the truncated series), so every degree `> 3` term
lands soundly in `R`.

For a step `a*` (the Cauchy point, or one ingested from torch `solve_subspace_trust_region`) the true
decrease is rigorously enclosed — `ψ(0) ∈ m(0)+R` and `ψ(a*) ∈ m(a*)+R` share the uniform remainder,
so `Δ = ψ(0) − ψ(a*) ∈ [pred − w, pred + w]` with `pred = m(0) − m(a*)` and `w = R.width`. The step
is **certified to strictly decrease the true loss** iff `pred − w > 0`. Because `R` scales like
`O(r⁴)` while `pred` scales like `O(r)`, a small enough radius always certifies — the same "small
box" mechanism as the strict-min certificate.

```python
import math
from omnibias.verify import (
    MLPArchitecture, ParamSpaceLoss, certify_subspace_step, certify_trajectory, krylov_basis,
)

arch = MLPArchitecture(dims=(1, 1, 1), activation="tanh")

def net(t, x):                                   # net(x) = v·tanh(w·x + b) + c
    w, b, v, c = t
    return v * math.tanh(w * x + b) + c

teacher = [-1.3, 0.25, 1.2, 0.1]
data = [((x,), (net(teacher, x),)) for x in [-1.0 + 0.25 * i for i in range(9)]]
theta0 = [0.5, -0.3, 0.8, 0.2]                   # current, non-stationary parameters

basis = krylov_basis(ParamSpaceLoss(arch, data), theta0, k=2)   # orthonormal span{g, Hg}
cert = certify_subspace_step(arch, data, theta0, basis, radius=0.05)

assert cert.certified                            # pred − width(R) > 0: a proven true-loss decrease
assert cert.verified                             # both sealed digests match their bodies
assert cert.decrease_enclosure.lo > 0.0          # rigorous lower bound on the real loss drop

# A whole proof-carrying descent: rebuild the basis, certify, and apply only certified steps
# (shrinking the trust radius on refusal). Every returned certificate satisfies .certified.
certs = certify_trajectory(arch, data, theta0, radius=0.05, k=2, steps=5)
assert all(c.certified for c in certs)
```

The margin is sealed as a one-signed interval certificate whose `> 0` obligation the Lean kernel can
re-check (`lean=True`), alongside a companion certificate recording the model-vs-truth remainder `R`.

!!! note "A rigorous *per-step* descent guarantee — not global optimisation"
    Each certificate proves one step strictly decreases the *true* loss over a small trust region for
    a small `tanh` / `sigmoid` MLP with fixed data; the Taylor-model remainder (not autodiff) is the
    enclosure of record. It is a local guarantee, not a convergence or global-optimality claim. The
    torch bridge is optional — feed a `JetSubspaceTensor` basis and a `solve_subspace_trust_region`
    step straight into `certify_subspace_step` to certify the differentiable optimiser's own move.

### Proof-carrying training, globally: a global-minimum certificate

`certify_trained_min` proves a *local* strict minimum (a ball around one `θ*`); `certify_trained_global_min`
asks the harder, global question over a whole **parameter search box**: what is a rigorous enclosure of
`min_θ J(θ)`, and is a located point the certified global minimiser to a tolerance? It is the
parameter-space twin of `certify_trained_network` (which minimises a read-out over an *input* box). The
certified global optimiser (`certified_minimize`) already consumes exactly the interval `value` / `grad` /
`hessian` enclosures `ParamSpaceLoss` exposes, so the monotonicity test, mean-value / second-order lower
bounds, and the interval-Newton (Krawczyk) contractor all apply unchanged, and the sealed payload is the
rigorous global-minimum enclosure `[f_lower, f_upper]`.

```python
from omnibias.verify import MLPArchitecture, certify_trained_global_min

arch = MLPArchitecture(dims=(1, 1), activation="tanh")     # affine readout, P = 2 (convex MSE)
data = [((-1.0,), (-0.5,)), ((0.0,), (0.0,)), ((1.0,), (0.5,))]   # realizable: y = 0.5·x

cert = certify_trained_global_min(arch, data, [(-1.0, 1.0)] * arch.n_params, tol=1e-3)
assert cert.converged                       # certified global gap f_upper − f_lower ≤ tol
assert cert.result.f_lower <= 1e-6          # realizable ⇒ the global training loss is 0
assert cert.verified                        # sealed digest matches the body (untampered)

# A non-realizable box gives a certified *positive* global minimum — a kernel-checkable claim
# that the architecture cannot fit the data exactly (Lean fires only when f_lower > 0):
bad = certify_trained_global_min(arch, [((-1.0,), (1.0,)), ((1.0,), (-1.0,))],
                                 [(0.5, 1.5)] * arch.n_params, tol=1e-3,
                                 strict_local_min=True, lean=True)
assert bad.result.f_lower > 0.0 and bad.strict_local_min      # global AND strict at the argmin
```

!!! note "A rigorous *global* proof for tiny nets — not million-parameter training"
    Interval branch-and-bound is *sound for any dimension* but exponential in the parameter count in
    the worst case, and each explored box triggers an `O(P²)` hyper-dual gradient/Hessian sweep — so this
    is for **tiny** networks (`1-1-1`, `1-2-1`) over a bounded weight box, exactly like the input-space
    network certificate. Within a finite `max_boxes` budget the enclosure is always sound
    (`f_lower ≤ min J ≤ f_upper`); `converged` reports honestly whether the certified gap reached `tol`.

### Verified parameter-jet: arbitrary-order rigorous derivatives

`mlp_jet_mv` differentiates a read-out w.r.t. its *inputs*; `param_jet` is its **parameter-space**
counterpart of *arbitrary order* `N`. It builds the order-`N` `TaylorModelMV` of the objective
`J(θ₀ + δ)` over a box in the parameter directions — the **full** parameter space (`δ ∈ [−r, r]^P`) or a
low-dimensional **subspace** (`δ = Q·a`, `a ∈ [−r, r]^k`) — with exact polynomial coefficients and a
rigorous model-vs-truth remainder. One pass yields every mixed partial up to total order `N`: read off
the symmetric `m`-th derivative tensor with `.grad()` (`m=1`), `.hessian()` (`m=2`), or `.tensor(m)`.

The concrete win: the full-`P` order-2 jet returns the **entire interval Hessian in a single Taylor-model
pass**, versus the `O(P²)` hyper-dual sweeps of `ParamSpaceLoss` — an alternate Hessian provider for the
strict-local-min, global, and (kernel-verified) positive-definiteness certificates. `subspace_step` is now
the order-3 subspace special case of this one primitive.

```python
from omnibias.verify import MLPArchitecture, param_jet

arch = MLPArchitecture(dims=(1, 1, 1), activation="tanh")   # P = 4
theta0 = [0.7, -0.3, 1.1, 0.2]
data = [((-1.0,), (0.4,)), ((0.0,), (0.1,)), ((0.8,), (-0.3,))]

jet = param_jet(arch, data, theta0, order=3, radius=5e-2)   # full-P order-3 Taylor model of J
g = jet.grad()                       # gradient   (tuple of P interval enclosures)
H = jet.hessian()                    # Hessian    — one TM pass, not P² hyper-dual sweeps
T = jet.tensor(3)                    # third-derivative tensor (nested P×P×P)
assert jet.bound().lo <= jet.value().hi        # J over the box ⊇ J(θ₀); remainder is jet.remainder
```

!!! note "A rigorous *local* Taylor model — coefficient count is `C(P+N, P)`"
    This is for **tiny** networks, **low** order, or a **small** subspace (`tanh` / `sigmoid`, fixed
    data); it is a rigorous local model about `θ₀` with a sound remainder over the box, not a global or
    open-problem claim.

## Proof-carrying PDE path

`verified_layers` extracts `(W, b, activation)` tuples from a `JetMLP`-style model
for the certified multivariate jet. `verified_layer_bundle` adds reproducibility
metadata such as activation names, dtype strings, parameter counts, domain
metadata, provenance, and a layer digest. `certify_pinn_aposteriori` then runs the
core PDE certificate path and returns a sealed a-posteriori error certificate with
adaptive residual diagnostics.

See the [proof-carrying PDE cookbook](../cookbook/proof-carrying-pde.md) and the
`examples/proof_carrying_pde/` smoke demo.

### Verified stochastic layer: rigorous Fokker-Planck / Ito residuals

The `omnibias.score` package composes the SDE operators (`score`, `ito_generator`,
`fokker_planck`) from the closed-form field primitives; `certify_fokker_planck_residual`
and `certify_ito_generator_residual` give them a **rigorous register**, the SDE sibling of
the PDE path above. For an Ito diffusion `dX = b(X) dt + sigma(X) dW` (`a = sigma sigma^T`)
they enclose the operator residual over a spatial box from the certified order-2 jet of the
density / test-function net:

- the **generator** `L f = b . grad f + 1/2 a_ij d_i d_j f` (optionally `L f + c f - g`), and
- the **Fokker-Planck adjoint** `L* p = -(div(b) p + b . grad p) + 1/2 a_ij d_i d_j p`
  (spatially-constant `a`, the common constant-noise case).

The drift `b`, its divergence `div b` and the diffusion `a` are the caller's analytic
inputs (exactly as in `omnibias.score`); the returned enclosure holds for *every* point of
the box, so `.residual_sup` is a certified sup-norm bound sealed into a tamper-evident
certificate. On the Ornstein-Uhlenbeck stationary density — reproduced *exactly* by a
one-unit `gaussian` MLP — the sealed residual encloses the true `L* p = 0` and tightens
under subdivision. Honest scope: a rigorous **local** (finite-box) enclosure of a given
network's operator residual, not a global or open-problem claim.

```python
import math
from omnibias.verify import certify_fokker_planck_residual
from omnibias.core.verified.interval import Interval

theta, sigma_sq = 0.8, 0.5
var = sigma_sq / (2.0 * theta)                      # OU stationary variance
w = 1.0 / math.sqrt(var)
density = [([[w]], [0.0], "gaussian"), ([[1.0]], [0.0], None)]   # p(x) = exp(-x^2 / (2 var))

cert = certify_fokker_planck_residual(
    density, [(-1.5, 1.5)],
    drift=[lambda box: Interval.point(-theta) * box[0]],          # b(x) = -theta x
    diffusion=[[sigma_sq]], drift_divergence=-theta, splits=64,
)
assert cert.residual.lo <= 0.0 <= cert.residual.hi   # sound: encloses L* p_inf = 0
assert cert.verified                                 # sealed, digest intact
```

### Certified finite-difference gradient checking

`certified_gradient_check(autodiff_grad, axis_oracles, point)` is the rigorous form of
the ubiquitous ML "grad check": instead of comparing an autodiff gradient to a
*fixed-epsilon* finite difference (which silently trades truncation error against
floating-point cancellation), it encloses each true partial derivative in an `Interval`
via the difference package's certified remainder engine (`certified_fd_error_general`) /
`TaylorModelMV`, then returns a per-coordinate interval enclosing
`autodiff_grad_i − ∂f/∂x_i`. A correct gradient is **accepted** (every band contains 0);
a scaled-wrong gradient is **rejected**, and the sealed certificate records the signed
per-axis discrepancy. `mlp_axis_oracles` / `network_axis_oracles` build the axis oracles
from the closed-form activation tower, and `verify.torch.ingest` / `verify.jax.ingest`
feed real networks. The baseline fixed-epsilon one-sided check misses the cancellation the
certified band catches.

## Public API

::: omnibias.verify
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

## Interval bound propagation (baseline)

::: omnibias.verify._core.propagate
    options:
      show_root_heading: false
      heading_level: 3

## Taylor-model propagation

::: omnibias.verify._core.taylor
    options:
      show_root_heading: false
      heading_level: 3

## Branch-and-bound

::: omnibias.verify._core.bab
    options:
      show_root_heading: false
      heading_level: 3

## Certified global optimization

::: omnibias.verify._core.global_opt
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.verify._core.net_global_opt
    options:
      show_root_heading: false
      heading_level: 3

## Proof-carrying training (parameter-space strict local minimum)

::: omnibias.verify._core.param_loss
    options:
      show_root_heading: false
      heading_level: 3

## Proof-carrying training (parameter-space global minimum)

::: omnibias.verify._core.param_global
    options:
      show_root_heading: false
      heading_level: 3

## Verified parameter-jet (arbitrary-order parameter-space Taylor model)

::: omnibias.verify._core.param_jet
    options:
      show_root_heading: false
      heading_level: 3

## Proof-carrying optimization (certified subspace trust-region step)

::: omnibias.verify._core.subspace_step
    options:
      show_root_heading: false
      heading_level: 3

## Verified stochastic layer (Fokker-Planck / Ito operator residuals)

::: omnibias.verify._core.stochastic
    options:
      show_root_heading: false
      heading_level: 3

## Certified finite-difference gradient checking

::: omnibias.verify._core.gradient_check
    options:
      show_root_heading: false
      heading_level: 3

## Interval-Newton and certified critical points

::: omnibias.verify._core.newton
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.verify._core.stationary
    options:
      show_root_heading: false
      heading_level: 3

## Activation enclosures

::: omnibias.verify._core.activations
    options:
      show_root_heading: false
      heading_level: 3

## Property certificates

::: omnibias.verify._core.certificates
    options:
      show_root_heading: false
      heading_level: 3

## Certified PDE stencil truncation error

`certified_stencil_truncation` / `certified_laplacian_truncation` bound the
truncation error of a finite-difference stencil on any `TaylorModel`-enclosable
function — not just the activation dictionary — by reusing the difference package's
general remainder engine (`certified_fd_error_general`). The returned certificate
carries the certified consistency order `O(hᵖ)`, which
`measured_consistency_order` cross-checks against the empirical order from a step
halving, closing the "activation-dictionary coupling" gap in the remainder engine.

::: omnibias.verify._core.pde_stencil
    options:
      show_root_heading: false
      heading_level: 3

## Certified domain integral (multivariate)

The rigorous-register companion to the differentiable `omnibias-measure` package:
a *sound* enclosure of `int_Omega f dx` over a box `Omega`, so the returned
`DomainIntegralCertificate` satisfies `lo <= int_Omega f dx <= hi`
**unconditionally**. `certified_domain_integral` subdivides the box
(uniform or adaptive branch-and-bound on the widest axis) and, per cell, integrates
either a `TaylorModelMV` of the integrand (exact polynomial part + outward-rounded
remainder, via `TaylorModelMV.definite_integral`) or an interval fallback. For a
trained network, `network_integrand_model` builds the Taylor model of a readout (or
a power of it) and `certified_network_integral` is the one-call wrapper -- e.g.
`power=2` gives a certified `L^2` mass. The enclosure contains a dense grid **and** a
random sample of true values (the mandatory soundness test).

::: omnibias.verify._core.quadrature
    options:
      show_root_heading: false
      heading_level: 3

## Certified L^p / Sobolev network norms

`certified_lp_norm` encloses `(int_Omega |net(x)|^p dx)^{1/p}` and
`certified_sobolev_norm` the `H^k` norm of an ingested `Network` -- the certified
twin of the differentiable `omnibias.fields.{torch,jax}.sobolev_norm`. Even `p` takes a Taylor
path (`network_integrand_model` with `power=p`); odd `p` and ReLU take an interval /
layer-cake path over per-cell output ranges with a rigorous `_interval_pth_root`.
The Sobolev norm sums the `L^2` mass (via `certified_domain_integral`) and the
squared partials from the rigorous `interval_jacobian` (sound for the ReLU
subgradient). `certified_layer_cake_integral` is the general certified integral of a
non-negative transform (`abs` / `relu`) of a readout, for non-smooth integrands where
derivative-remainder rules go vacuous. All seal as v1 certificates with a `scope`
field and carry a dense-grid + random-sample soundness check.

::: omnibias.verify._core.norms
    options:
      show_root_heading: false
      heading_level: 3

## Frontends

::: omnibias.verify.torch.ingest
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.verify.jax.ingest
    options:
      show_root_heading: false
      heading_level: 3

## Warm-start (closed-form jet gradient descent)

::: omnibias.verify.torch.warm_start
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.verify.jax.warm_start
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).
