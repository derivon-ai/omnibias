# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Proof-carrying training: certify a trained parameter vector is a strict local minimum.

The rest of :mod:`omnibias.verify` certifies a network read-out over an *input* box -- the
closed-form verified jet (:func:`omnibias.core.verified.jet_mv.mlp_jet_mv`) differentiates the
output with respect to the inputs and treats the weights as constants. Proof-carrying *training*
asks the dual question about the **parameters**: given a trained ``theta*`` and fixed data, is
``theta*`` a genuine, locally-unique, strict local minimum of the training objective -- not merely a
point where the optimiser happened to stop?

The missing ingredient is a rigorous **parameter-space** enclosure of ``grad_theta`` and
``Hess_theta`` of the loss. The parameter map ``theta |-> net_theta(x)`` is *not* a standard MLP
(the read-out ``W_2 tanh(W_1 x + b_1)`` is bilinear in the two weight matrices), so ``mlp_jet_mv``
does not apply. Instead this module runs a small **interval forward-mode second-order jet**
(hyper-dual numbers over :class:`~omnibias.core.verified.interval.Interval`): seeding two parameter
axes with the nilpotents ``eps1, eps2`` and propagating through the exact activation towers
(``tanh`` / ``sigmoid`` via the Riccati identities) yields, in one sweep, a sound enclosure of the
value, both first partials, and the mixed second partial over the whole parameter box. Sweeping the
``P(P+1)/2`` axis pairs assembles the full interval gradient and Hessian -- exactly the ``GradFn`` /
``HessianFn`` callbacks the existing proof stack consumes.

Those enclosures feed the backend-agnostic machinery unchanged: :func:`krawczyk_image` /
:func:`krawczyk_unique` prove a *locally-unique stationary point*, and the interval ``LDL^T`` inertia
(:func:`omnibias.core.verified.eig_operator.is_positive_definite`, bracketed by a positive-definite
shift) proves the Hessian is *positive definite* on the box (``eig_min.lo > 0``, i.e. a **strict**
local min). The result is sealed as a tamper-evident :class:`~omnibias.core.proof.certificate.Cert`;
its full interval ``LDL^T`` **pivot vector** is additionally sealed as a ``positive_definite``
certificate, so the Lean kernel re-checks the whole ``allPivotsPos`` inertia obligation (matrix
positive-definiteness) rather than only the scalar ``eig_min > 0`` shadow.

**Conditioning & L2 regularisation.** The size of network this scales to is set by the *conditioning*
of the loss, not by the arithmetic. The interval Hessian enclosure is already *linear* in the box
radius (a shallow, non-compounding computation -- there is no wrapping to fight, and an affine /
zonotope engine was measured only to widen it, so it was rejected). The real ceiling is that an
over-parametrised ``tanh`` network fit to realisable data sits in a near-flat valley (smallest true
Hessian eigenvalue ``~1e-7``): it is not a *strict* local minimum at all, so no enclosure can
honestly certify one. Passing ``l2 > 0`` instead certifies the **L2-regularised** objective
``J(theta) = L(theta) + l2 * ||theta||^2`` (mean-squared error ``L`` plus weight decay), whose
Hessian ``Hess L + 2 l2 I`` lifts every eigenvalue by ``2 l2``. A network genuinely trained with
weight decay ``l2`` therefore has a strict, certifiable regularised minimum even where the bare loss
is flat -- which is what lets the certificate reach beyond a single unit.

Honest scope: small networks, fixed data, ``tanh`` / ``sigmoid`` activations. The certificate is
*local* -- a ball around ``theta*`` -- and is a rigorous proof of a strict local minimum, **not** a
global-optimality or open-problem claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.proof.certificate import (
    Cert,
    interval_certificate,
    positive_definite_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.verified.eig_operator import interval_ldlt_pivots, is_positive_definite
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.transcend import sigmoid_iv, tanh_iv
from omnibias.verify._core.global_opt import Box
from omnibias.verify._core.newton import krawczyk_image, krawczyk_unique
from omnibias.verify._core.stationary import FlatnessResult

_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)


@dataclass(frozen=True)
class HyperDual:
    r"""A second-order forward-mode jet ``f = val + d1 eps1 + d2 eps2 + d12 eps1 eps2``.

    The two nilpotents satisfy ``eps1^2 = eps2^2 = 0``, so seeding parameter axis ``a`` with
    ``eps1`` and axis ``b`` with ``eps2`` makes ``d1 = df/dtheta_a``, ``d2 = df/dtheta_b`` and
    ``d12 = d^2 f / dtheta_a dtheta_b`` (and, with both nilpotents on the *same* axis, ``d12`` is the
    second derivative there). Every component is an :class:`Interval`, so a propagation over a
    parameter box yields a sound enclosure of each derivative over that box.
    """

    val: Interval
    d1: Interval
    d2: Interval
    d12: Interval

    @classmethod
    def constant(cls, value: IntervalLike) -> HyperDual:
        """A jet with zero derivatives (a constant / non-seeded quantity)."""
        return cls(Interval.from_value(value), _ZERO, _ZERO, _ZERO)

    def __add__(self, other: HyperDual) -> HyperDual:
        return HyperDual(self.val + other.val, self.d1 + other.d1, self.d2 + other.d2, self.d12 + other.d12)

    def __sub__(self, other: HyperDual) -> HyperDual:
        return HyperDual(self.val - other.val, self.d1 - other.d1, self.d2 - other.d2, self.d12 - other.d12)

    def __mul__(self, other: HyperDual) -> HyperDual:
        # product rule to second order: (eps1 eps2) coeff picks up the cross term d1*d2 + d2*d1
        return HyperDual(
            self.val * other.val,
            self.d1 * other.val + self.val * other.d1,
            self.d2 * other.val + self.val * other.d2,
            self.d12 * other.val + self.d1 * other.d2 + self.d2 * other.d1 + self.val * other.d12,
        )


def _chain(u: HyperDual, f: Interval, fp: Interval, fpp: Interval) -> HyperDual:
    r"""Apply a scalar function to ``u`` given its value ``f`` and derivatives ``f'``, ``f''`` at ``u.val``.

    ``g(u) = f + f'(u - u.val) + 1/2 f''(u - u.val)^2`` expanded to second order gives the four
    hyper-dual components below (``(d1 eps1 + d2 eps2)^2 = 2 d1 d2 eps1 eps2``).
    """
    return HyperDual(f, fp * u.d1, fp * u.d2, fp * u.d12 + fpp * u.d1 * u.d2)


def _tanh(u: HyperDual) -> HyperDual:
    t = tanh_iv(u.val)
    tp = _ONE - t * t  # tanh' = 1 - tanh^2
    tpp = Interval.point(-2.0) * t * tp  # tanh'' = -2 tanh (1 - tanh^2)
    return _chain(u, t, tp, tpp)


def _sigmoid(u: HyperDual) -> HyperDual:
    s = sigmoid_iv(u.val)
    sp = s * (_ONE - s)  # sigma' = sigma (1 - sigma)
    spp = sp * (_ONE - Interval.point(2.0) * s)  # sigma'' = sigma' (1 - 2 sigma)
    return _chain(u, s, sp, spp)


_ACTIVATIONS = {"tanh": _tanh, "sigmoid": _sigmoid}


@dataclass(frozen=True)
class MLPArchitecture:
    r"""A fully-connected MLP shape for the parameter-space loss.

    ``dims = (d_in, h_1, ..., h_L, d_out)``; every hidden layer applies ``activation`` and the
    read-out layer is affine. The flat parameter layout is, per layer in order, the weight matrix
    ``W`` (shape ``(out, in)``) row-major followed by the bias ``b`` (length ``out``) -- matching
    :meth:`omnibias.torch...Linear`-style ``[weight, bias]`` extraction.
    """

    dims: tuple[int, ...]
    activation: str = "tanh"

    def __post_init__(self) -> None:
        if len(self.dims) < 2:
            raise ValueError(f"dims needs at least (in, out), got {self.dims}")
        if any(d < 1 for d in self.dims):
            raise ValueError(f"all dims must be >= 1, got {self.dims}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {self.activation!r}")

    @property
    def layer_shapes(self) -> list[tuple[int, int]]:
        """``[(out, in), ...]`` for each affine layer."""
        return [(self.dims[i + 1], self.dims[i]) for i in range(len(self.dims) - 1)]

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters (weights + biases)."""
        return sum(out * inp + out for out, inp in self.layer_shapes)

    @property
    def out_dim(self) -> int:
        return self.dims[-1]


def flat_params_from_layers(layers: Sequence[tuple[Sequence[Sequence[float]], Sequence[float]]]) -> list[float]:
    r"""Flatten ``[(W, b), ...]`` (per-layer weight matrix + bias) into the parameter vector.

    ``W`` is ``(out, in)`` row-major, ``b`` is length ``out``; layers concatenated in order. This is
    the inverse layout of :meth:`MLPArchitecture` unpacking and matches the order a torch/jax
    ``Linear`` stack exposes its ``[weight, bias]`` parameters.
    """
    flat: list[float] = []
    for weight, bias in layers:
        for row in weight:
            flat.extend(float(w) for w in row)
        flat.extend(float(b) for b in bias)
    return flat


def _unpack(arch: MLPArchitecture, theta: Sequence[HyperDual]) -> list[tuple[list[list[HyperDual]], list[HyperDual]]]:
    out: list[tuple[list[list[HyperDual]], list[HyperDual]]] = []
    off = 0
    for n_out, n_in in arch.layer_shapes:
        weight = [[theta[off + r * n_in + c] for c in range(n_in)] for r in range(n_out)]
        off += n_out * n_in
        bias = [theta[off + r] for r in range(n_out)]
        off += n_out
        out.append((weight, bias))
    return out


def _forward(arch: MLPArchitecture, theta: Sequence[HyperDual], x: Sequence[float]) -> list[HyperDual]:
    act = _ACTIVATIONS[arch.activation]
    layers = _unpack(arch, theta)
    a: list[HyperDual] = [HyperDual.constant(Interval.point(float(xi))) for xi in x]
    last = len(layers) - 1
    for depth, (weight, bias) in enumerate(layers):
        z: list[HyperDual] = []
        for o in range(len(weight)):
            acc = bias[o]
            row = weight[o]
            for j in range(len(row)):
                acc = acc + row[j] * a[j]
            z.append(acc)
        a = z if depth == last else [act(zo) for zo in z]
    return a


def _loss(
    arch: MLPArchitecture,
    theta: Sequence[HyperDual],
    data: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> HyperDual:
    n = len(data)
    if n == 0:
        raise ValueError("data must be non-empty")
    acc = HyperDual.constant(_ZERO)
    for x, y in data:
        out = _forward(arch, theta, x)
        for o, yo in enumerate(y):
            diff = out[o] - HyperDual.constant(Interval.point(float(yo)))
            acc = acc + diff * diff
    return acc * HyperDual.constant(Interval.from_rational(Fraction(1, n)))


class ParamSpaceLoss:
    r"""Interval enclosures of the parameter-space training objective and its derivatives.

    Wraps a fixed :class:`MLPArchitecture` and dataset ``[(x_i, y_i), ...]`` and exposes the three
    callbacks the certified-optimisation stack consumes -- :meth:`value` (an ``ObjectiveFn``),
    :meth:`grad` (a ``GradFn``) and :meth:`hessian` (a ``HessianFn``) -- each mapping a *parameter*
    box to a sound enclosure over that box.

    ``l2`` adds a weight-decay term to the objective: with ``l2 > 0`` the enclosed quantity is the
    **regularised** loss ``J(theta) = L(theta) + l2 * ||theta||^2`` (over *all* parameters). Because
    the regulariser is a pure quadratic, it is folded into the gradient (``+ 2 l2 theta``) and
    Hessian (``+ 2 l2 I``) *analytically and exactly* -- it lifts every Hessian eigenvalue by
    ``2 l2`` without widening the enclosure at all.

    A one-slot cache keyed by the box bounds lets a back-to-back ``grad`` + ``hessian`` on the same
    box share a single hyper-dual sweep.
    """

    def __init__(
        self,
        arch: MLPArchitecture,
        data: Sequence[tuple[Sequence[float], Sequence[float]]],
        *,
        l2: float = 0.0,
    ) -> None:
        if l2 < 0.0:
            raise ValueError(f"l2 (weight decay) must be >= 0, got {l2}")
        self.arch = arch
        self.l2 = float(l2)
        self.data = [(tuple(float(v) for v in x), tuple(float(v) for v in y)) for x, y in data]
        for x, y in self.data:
            if len(x) != arch.dims[0]:
                raise ValueError(f"input dim {len(x)} != arch input dim {arch.dims[0]}")
            if len(y) != arch.out_dim:
                raise ValueError(f"target dim {len(y)} != arch output dim {arch.out_dim}")
        self._cache_key: tuple[tuple[float, float], ...] | None = None
        self._cache: tuple[list[Interval], list[list[Interval]]] | None = None

    def _box(self, box: Sequence[IntervalLike]) -> list[Interval]:
        return [Interval.from_value(b) for b in box]

    def value(self, box: Sequence[IntervalLike]) -> Interval:
        """Enclosure of ``J(theta) = L(theta) + l2 ||theta||^2`` over the parameter box."""
        ivs = self._box(box)
        theta = [HyperDual.constant(iv) for iv in ivs]
        val = _loss(self.arch, theta, self.data).val
        if self.l2 != 0.0:
            reg = _ZERO
            for iv in ivs:
                reg = reg + iv * iv
            val = val + Interval.point(self.l2) * reg
        return val

    def _grad_hessian(self, box: Sequence[IntervalLike]) -> tuple[list[Interval], list[list[Interval]]]:
        ivs = self._box(box)
        key = tuple((iv.lo, iv.hi) for iv in ivs)
        if self._cache_key == key and self._cache is not None:
            return self._cache
        p = self.arch.n_params
        if len(ivs) != p:
            raise ValueError(f"box has {len(ivs)} axes but the model has {p} parameters")
        grad: list[Interval] = [_ZERO] * p
        hess: list[list[Interval]] = [[_ZERO] * p for _ in range(p)]
        for a in range(p):
            for b in range(a, p):
                theta = [
                    HyperDual(
                        ivs[k],
                        _ONE if k == a else _ZERO,
                        _ONE if k == b else _ZERO,
                        _ZERO,
                    )
                    for k in range(p)
                ]
                loss = _loss(self.arch, theta, self.data)
                hess[a][b] = loss.d12
                hess[b][a] = loss.d12
                if a == b:
                    grad[a] = loss.d1
        if self.l2 != 0.0:
            # J = L + l2 ||theta||^2: exact quadratic regulariser, grad += 2 l2 theta, Hess += 2 l2 I.
            two_l2 = Interval.point(2.0 * self.l2)
            for a in range(p):
                grad[a] = grad[a] + two_l2 * ivs[a]
                hess[a][a] = hess[a][a] + two_l2
        self._cache_key, self._cache = key, (grad, hess)
        return grad, hess

    def grad(self, box: Sequence[IntervalLike]) -> list[Interval]:
        """Enclosure of ``grad_theta J`` over the parameter box (a ``GradFn``)."""
        return self._grad_hessian(box)[0]

    def hessian(self, box: Sequence[IntervalLike]) -> list[list[Interval]]:
        """Enclosure of ``Hess_theta J`` over the parameter box (a ``HessianFn``)."""
        return self._grad_hessian(box)[1]


@dataclass(frozen=True)
class TrainingCertificate:
    r"""A sealed, tamper-evident proof that ``theta*`` is a strict local minimum of the objective.

    Bundles the certified Krawczyk uniqueness flag (a locally-unique stationary point), the
    :class:`~omnibias.verify.FlatnessResult` (rigorous Hessian-eigenvalue enclosure, whose
    ``eig_min.lo > 0`` is the strict-min certificate), the sealed v1
    :class:`~omnibias.core.proof.certificate.Cert` (payload = the ``eig_min`` interval, so the
    scalar ``eig_min > 0`` obligation is Lean-checkable), and the optional
    :class:`~omnibias.core.proof.lean_check.LeanCheckResult`.

    When the Hessian box is certified positive definite, ``pd_certificate`` additionally seals the
    full interval ``LDL^T`` **pivot vector**; this is the payload handed to the Lean kernel (its
    ``allPivotsPos`` inertia obligation is the whole positive-definiteness statement, not the single
    ``eig_min`` scalar shadow), so :attr:`theorem_prover_verified` reflects kernel-verified matrix
    positive-definiteness.
    """

    theta_star: tuple[float, ...]
    box: tuple[tuple[float, float], ...]
    unique_stationary: bool
    strict_local_min: bool
    flatness: FlatnessResult
    loss_enclosure: Interval
    certificate: Cert
    lean: LeanCheckResult | None = None
    pd_certificate: Cert | None = None

    @property
    def verified(self) -> bool:
        """``True`` iff every sealed certificate's digest matches its body (untampered)."""
        if not verify_certificate_digest(self.certificate):
            return False
        return self.pd_certificate is None or verify_certificate_digest(self.pd_certificate)

    @property
    def certified(self) -> bool:
        """``True`` iff ``theta*`` is proven a locally-unique **strict** local minimum."""
        return self.unique_stationary and self.strict_local_min

    @property
    def positive_definite(self) -> bool:
        """``True`` iff the Hessian box was certified PD and its pivot vector was sealed."""
        return self.pd_certificate is not None

    @property
    def theorem_prover_verified(self) -> bool:
        """``True`` only when the Lean kernel genuinely re-checked the sealed obligation.

        When a :attr:`pd_certificate` is present the re-checked obligation is the full
        ``allPivotsPos`` inertia vector (kernel-verified matrix positive-definiteness); otherwise
        it is the scalar ``eig_min > 0`` fact.
        """
        return self.lean is not None and self.lean.verified


def _shifted(h: list[list[Interval]], s: float) -> list[list[Interval]]:
    shift = Interval.point(s)
    return [[h[i][j] - shift if i == j else h[i][j] for j in range(len(h))] for i in range(len(h))]


def _gershgorin(h: list[list[Interval]]) -> tuple[float, float]:
    r"""A sound outer bracket ``[lam_lo, lam_hi]`` for *all* Hessian eigenvalues over the box.

    Gershgorin's disc theorem, evaluated with interval magnitudes: every eigenvalue of every point
    matrix in ``h`` lies in ``[min_i(h_ii.lo - R_i), max_i(h_ii.hi + R_i)]`` with row radius
    ``R_i = sum_{j != i} |h_ij|`` (outward ``mag``). Used only to seed the shift bisection.
    """
    n = len(h)
    lo = float("inf")
    hi = float("-inf")
    for i in range(n):
        radius = 0.0
        for j in range(n):
            if j != i:
                radius += h[i][j].mag
        lo = min(lo, h[i][i].lo - radius)
        hi = max(hi, h[i][i].hi + radius)
    return lo, hi


def _eig_bracket(h: list[list[Interval]], *, iters: int = 100) -> FlatnessResult:
    r"""Sharp interval enclosure of the extreme Hessian eigenvalues via PD-shift bisection.

    ``H - s I`` is certified positive definite (interval ``LDL^T`` inertia, monotone in ``s``) iff
    ``s`` is a rigorous *lower* bound on the smallest eigenvalue of every point matrix in the box.
    Bisecting for the largest such ``s`` yields a sharp certified ``eig_min.lo`` -- far tighter than
    a generic generalized-eigenvalue enclosure when the curvature is small. The upper end
    ``eig_min.hi`` uses ``lambda_min <= min_i H_ii`` (Rayleigh at a coordinate axis); ``eig_max`` is
    bracketed by ``max_i H_ii.lo <= lambda_max <= `` Gershgorin.
    """
    n = len(h)
    g_lo, g_hi = _gershgorin(h)
    diag_min_hi = min(h[i][i].hi for i in range(n))
    diag_max_lo = max(h[i][i].lo for i in range(n))
    # Bisect s in [lo, hi]: PD(H - s I) holds for s below lambda_min, fails above it.
    lo = g_lo - 1.0  # H - lo I is strongly diagonally dominant PD here
    hi = max(diag_min_hi, lo)
    if not is_positive_definite(_shifted(h, lo)):
        lo = min(lo, -abs(g_lo) - abs(g_hi) - 1.0)  # push further below the spectrum
    for _ in range(iters):
        if hi - lo <= 1e-15 * max(1.0, abs(lo), abs(hi)):
            break
        mid = 0.5 * (lo + hi)
        if is_positive_definite(_shifted(h, mid)):
            lo = mid
        else:
            hi = mid
    eig_min = Interval(lo, max(diag_min_hi, lo))
    eig_max = Interval(min(diag_max_lo, g_hi), g_hi)
    return FlatnessResult(eig_min=eig_min, eig_max=eig_max)


def certify_trained_min(
    arch: MLPArchitecture,
    data: Sequence[tuple[Sequence[float], Sequence[float]]],
    theta_star: Sequence[float],
    *,
    radius: float = 1e-3,
    l2: float = 0.0,
    lean: bool = False,
    provenance: dict[str, Any] | None = None,
) -> TrainingCertificate:
    r"""Certify a trained ``theta*`` is a locally-unique strict local minimum of the objective.

    The certified objective is ``J(theta) = L(theta) + l2 * ||theta||^2`` (mean-squared error plus
    optional weight decay; ``l2 = 0`` is the bare loss). Over the parameter ball ``B(theta*, radius)``
    (an axis-aligned box) this:

    1. proves a **locally-unique stationary point** via the Krawczyk operator on ``grad_theta J``
       (``K(X) subset int(X)``; no explicit Lipschitz constant needed);
    2. proves the Hessian is **positive definite** on the whole box via the interval ``LDL^T``
       inertia (:func:`omnibias.core.verified.eig_operator.is_positive_definite`), bracketed by a
       positive-definite shift so ``eig_min.lo > 0`` is a *sharp* certified lower bound on the
       smallest eigenvalue -- so any stationary point in the box is a *strict* local minimum;
    3. **seals** a v1 certificate whose interval payload is the ``eig_min`` enclosure (so the finite
       ``eig_min > 0`` sign obligation is Lean-checkable), recording ``theta*``, the box, ``l2``, the
       Krawczyk uniqueness flag, and the objective enclosure in ``meta``;
    4. optionally runs the Lean kernel on that obligation (``lean=True``); the
       :attr:`~TrainingCertificate.theorem_prover_verified` flag is set only on a genuine kernel
       pass and degrades gracefully to ``False`` when no toolchain is present.

    ``l2 > 0`` lifts every Hessian eigenvalue by ``2 l2``, which is what lets the certificate reach
    beyond a single unit: an over-parametrised ``tanh`` fit to realisable data is near-flat
    (``eig_min ~ 0``) and *not* a strict minimum of the bare loss, but a network trained with weight
    decay ``l2`` has a genuinely strict regularised minimum. Supply the ``theta*`` that actually
    minimises ``J`` (i.e. was trained with the same ``l2``).

    Honest scope: small networks, fixed data, ``tanh`` / ``sigmoid`` activations. This is a rigorous
    *local* proof (a ball around ``theta*``), not a global-optimality or open-problem claim.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if l2 < 0.0:
        raise ValueError(f"l2 (weight decay) must be >= 0, got {l2}")
    theta = [float(t) for t in theta_star]
    if len(theta) != arch.n_params:
        raise ValueError(f"theta_star has {len(theta)} entries but the model has {arch.n_params} parameters")

    problem = ParamSpaceLoss(arch, data, l2=l2)
    box: Box = tuple(Interval(t - radius, t + radius) for t in theta)

    image = krawczyk_image(problem.grad, problem.hessian, box)
    unique = image is not None and krawczyk_unique(image, box)

    hbox = [[Interval.from_value(hij) for hij in row] for row in problem.hessian(box)]
    flat = _eig_bracket(hbox)
    strict = flat.certified_positive_definite  # eig_min.lo > 0 (sharp PD-shift bound)
    loss_enclosure = problem.value(box)

    # Full inertia vector: the interval LDL^T pivots of the Hessian box. When every pivot's lower
    # endpoint is positive the box is certified positive definite -- the whole PD statement, which
    # the Lean kernel re-checks as `allPivotsPos` rather than the single `eig_min > 0` shadow.
    pivots = interval_ldlt_pivots(hbox)
    pd_matrix = pivots is not None and all(p.lo > 0.0 for p in pivots)

    honesty = {"unproven_claim": False, "strict_local_min": bool(strict and unique)}
    objective = "L(theta)" if l2 == 0.0 else f"L(theta) + {float(l2)!r} * ||theta||^2"
    claim = (
        f"the trained parameters theta* are a locally-unique strict local minimum of {objective}: "
        "the objective Hessian's smallest eigenvalue over B(theta*, radius) is enclosed by the interval"
    )
    meta: dict[str, Any] = {
        "kind": "strict_local_min",
        "theta_star": list(theta),
        "radius": float(radius),
        "l2": float(l2),
        "box": [[iv.lo, iv.hi] for iv in box],
        "dims": list(arch.dims),
        "activation": arch.activation,
        "n_params": arch.n_params,
        "n_data": len(problem.data),
        "unique_stationary": bool(unique),
        "strict_local_min": bool(strict),
        "positive_definite": bool(pd_matrix),
        "eig_min": [flat.eig_min.lo, flat.eig_min.hi],
        "eig_max": [flat.eig_max.lo, flat.eig_max.hi],
        "loss_enclosure": [loss_enclosure.lo, loss_enclosure.hi],
        "provenance": dict(provenance) if provenance is not None else {},
    }
    cert = interval_certificate(claim, flat.eig_min, honesty=honesty, meta=meta)

    pd_cert: Cert | None = None
    if pd_matrix and pivots is not None:
        pd_claim = (
            f"the objective Hessian of {objective} over B(theta*, radius) is positive definite: "
            "every interval LDL^T pivot of the symmetric Hessian box is strictly positive "
            "(zero negative inertia)"
        )
        pd_cert = positive_definite_certificate(
            pd_claim, pivots, matrix=hbox, honesty=honesty, meta=meta
        )

    # Only Lean-check a genuinely positive, finite obligation; a non-strict certificate carries no
    # positive obligation. Prefer the kernel-verified matrix-PD pivot vector when it was sealed,
    # falling back to the scalar `eig_min > 0` interval otherwise.
    lean_target = pd_cert if pd_cert is not None else cert
    lean_res = check_certificate(lean_target) if (lean and strict) else None

    return TrainingCertificate(
        theta_star=tuple(theta),
        box=tuple((iv.lo, iv.hi) for iv in box),
        unique_stationary=bool(unique),
        strict_local_min=bool(strict),
        flatness=flat,
        loss_enclosure=loss_enclosure,
        certificate=cert,
        lean=lean_res,
        pd_certificate=pd_cert,
    )


__all__ = [
    "HyperDual",
    "MLPArchitecture",
    "ParamSpaceLoss",
    "TrainingCertificate",
    "certify_trained_min",
    "flat_params_from_layers",
]
