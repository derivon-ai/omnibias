# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exterior calculus / de Rham--Hodge complex on the closed-form neural jet.

This is the antisymmetric-tensor face of the field surface. A
:class:`DifferentialForm` stores, per increasing multi-index, a *component*
:class:`~omnibias.symbolic.field_discovery.FieldJet`, so every component carries
its exact closed-form partials. On top of that:

* :func:`exterior_derivative` -- the metric-free ``d``, the single operator that
  unifies grad / curl / div (``d`` on a 0-/1-/2-form respectively);
* :func:`hodge_star`, :func:`codifferential`, :func:`hodge_laplacian` -- the
  flat-Euclidean ``*``, ``delta = +/- * d *`` and the Hodge--de Rham Laplacian
  ``Delta = d delta + delta d``;
* :func:`wedge` -- the (pointwise) exterior product.

The headline is the **fundamental identity** ``d . d = 0``: applied to a 0-form it
*is* ``curl(grad f) = 0``; to a 1-form in 3-D it *is* ``div(curl F) = 0``. Here it
holds to machine precision for *any* order, because the mixed partials are exact
(symmetry of second derivatives is exact, not approximate). The same identity is
the homogeneous Maxwell law ``dF = 0`` for ``F = dA`` (see
:func:`electromagnetic_field_2form`).

The Hodge star / codifferential / Laplacian are implemented for the **flat
Euclidean** metric with the standard orientation; the curved Laplace--Beltrami on
functions lives in :mod:`omnibias.symbolic.geometry_discovery`. On flat space the
Hodge Laplacian acts component-wise as *minus* the ordinary Laplacian (zero
curvature Weitzenbock), which these tests pin down exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from omnibias.core.multi_index import multi_indices
from omnibias.symbolic.field_discovery import (
    FieldJet,
    field_derivative_jet,
)

FormIndex = tuple[int, ...]


# --------------------------------------------------------------------------- #
# Index / permutation utilities
# --------------------------------------------------------------------------- #
def _increasing_indices(dim: int, degree: int) -> list[FormIndex]:
    """All strictly-increasing ``degree``-index basis labels (``[()]`` for degree 0)."""
    return [tuple(c) for c in combinations(range(dim), degree)]


def _permutation_sign(seq: tuple[int, ...]) -> int:
    """Sign of the permutation that sorts ``seq`` (distinct ints) into increasing order."""
    sign = 1
    s = list(seq)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[i] > s[j]:
                sign = -sign
    return sign


def _complement(index: FormIndex, dim: int) -> FormIndex:
    return tuple(sorted(set(range(dim)) - set(index)))


# --------------------------------------------------------------------------- #
# FieldJet arithmetic (component algebra)
# --------------------------------------------------------------------------- #
def _zero_jet(X: np.ndarray, order: int, var_names: tuple[str, ...]) -> FieldJet:
    dim = X.shape[1]
    partials = {a: np.zeros(X.shape[0]) for a in multi_indices(dim, order)}
    return FieldJet(X=X, order=order, partials=partials, var_names=var_names)


def _scale_jet(jet: FieldJet, scalar: float) -> FieldJet:
    return FieldJet(
        X=jet.X,
        order=jet.order,
        partials={a: scalar * v for a, v in jet.partials.items()},
        var_names=jet.var_names,
    )


def _combine_jets(
    terms: list[tuple[float, FieldJet]],
    X: np.ndarray,
    order: int,
    var_names: tuple[str, ...],
) -> FieldJet:
    """Linear combination ``sum c_i jet_i`` of equal-order jets (empty -> zero)."""
    dim = X.shape[1]
    acc = {a: np.zeros(X.shape[0]) for a in multi_indices(dim, order)}
    for c, jet in terms:
        if jet.order != order:
            raise ValueError(f"cannot combine order {jet.order} jet at order {order}")
        for a in acc:
            acc[a] = acc[a] + c * jet.partials[a]
    return FieldJet(X=X, order=order, partials=acc, var_names=var_names)


def _value_jet(X: np.ndarray, values: np.ndarray, var_names: tuple[str, ...]) -> FieldJet:
    """An order-0 jet holding only the values (used for the pointwise wedge)."""
    dim = X.shape[1]
    return FieldJet(
        X=X,
        order=0,
        partials={(0,) * dim: np.asarray(values, dtype=float)},
        var_names=var_names,
    )


# --------------------------------------------------------------------------- #
# Differential form
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DifferentialForm:
    """A differential ``k``-form: one component :class:`FieldJet` per ``dx^I``.

    ``components[I]`` is the coefficient of the basis form ``dx^{i_1} ^ ... ^
    dx^{i_k}`` for the strictly-increasing index ``I``; degree-0 forms use the
    single key ``()``. Every component shares the sample points and jet order, so
    the form can be differentiated (via :func:`exterior_derivative`) as a whole.
    """

    degree: int
    dim: int
    components: dict[FormIndex, FieldJet]
    var_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.degree < 0 or self.degree > self.dim:
            raise ValueError(f"degree must be in [0, {self.dim}], got {self.degree}")
        expected = set(_increasing_indices(self.dim, self.degree))
        if set(self.components) != expected:
            raise ValueError("components must be exactly the increasing k-indices")
        if len(self.var_names) != self.dim:
            raise ValueError("var_names must match dim")
        orders = {c.order for c in self.components.values()}
        if len(orders) != 1:
            raise ValueError(f"all components must share a jet order, got {orders}")
        ns = {c.n for c in self.components.values()}
        if len(ns) != 1:
            raise ValueError("all components must share sample count")

    @property
    def order(self) -> int:
        return int(next(iter(self.components.values())).order)

    @property
    def n(self) -> int:
        return int(next(iter(self.components.values())).n)

    @property
    def X(self) -> np.ndarray:
        return np.asarray(next(iter(self.components.values())).X)

    def component(self, index: FormIndex) -> FieldJet:
        return self.components[tuple(index)]

    def value(self, index: FormIndex) -> np.ndarray:
        return np.asarray(self.components[tuple(index)].value())

    def max_abs(self) -> float:
        """Largest absolute component value -- the sup-norm of the form's values."""
        return max(float(np.max(np.abs(c.value()))) for c in self.components.values())


def scalar_form(jet: FieldJet) -> DifferentialForm:
    """A 0-form (scalar field) from its :class:`FieldJet`."""
    return DifferentialForm(
        degree=0, dim=jet.dim, components={(): jet}, var_names=jet.var_names
    )


def one_form(jets: list[FieldJet] | tuple[FieldJet, ...]) -> DifferentialForm:
    """A 1-form ``sum omega_i dx^i`` from one component :class:`FieldJet` per axis."""
    if len(jets) == 0:
        raise ValueError("one_form needs at least one component")
    dim = jets[0].dim
    if len(jets) != dim:
        raise ValueError(f"one_form needs {dim} components, got {len(jets)}")
    comps: dict[FormIndex, FieldJet] = {(i,): jets[i] for i in range(dim)}
    return DifferentialForm(degree=1, dim=dim, components=comps, var_names=jets[0].var_names)


def differential_form(
    degree: int,
    components: dict[FormIndex, FieldJet],
    *,
    dim: int | None = None,
    var_names: tuple[str, ...] | None = None,
) -> DifferentialForm:
    """Assemble a general form, normalising index keys to increasing tuples."""
    norm = {tuple(k): v for k, v in components.items()}
    sample = next(iter(norm.values()))
    d = dim if dim is not None else sample.dim
    names = var_names if var_names is not None else sample.var_names
    return DifferentialForm(degree=degree, dim=d, components=norm, var_names=names)


# --------------------------------------------------------------------------- #
# Exterior derivative d
# --------------------------------------------------------------------------- #
def exterior_derivative(form: DifferentialForm) -> DifferentialForm:
    r"""The exterior derivative ``d``: a ``k``-form -> ``(k+1)``-form (order drops by 1).

    ``(d omega)_{j_0...j_k} = sum_p (-1)^p partial_{j_p} omega_{j_0...^j_p...j_k}``.
    Metric-free and exact: on a 0-form it is the gradient 1-form, on a 1-form (3-D)
    the curl 2-form, on a 2-form (3-D) the divergence 3-form.
    """
    m, k = form.dim, form.degree
    if k >= m:
        raise ValueError(f"exterior derivative of a top-degree {m}-form is zero")
    if form.order < 1:
        raise ValueError("exterior derivative needs component order >= 1")
    out_order = form.order - 1
    comps: dict[FormIndex, FieldJet] = {}
    for j in _increasing_indices(m, k + 1):
        terms: list[tuple[float, FieldJet]] = []
        for p, axis in enumerate(j):
            sub = j[:p] + j[p + 1 :]
            sign = 1.0 if p % 2 == 0 else -1.0
            terms.append((sign, field_derivative_jet(form.components[sub], axis)))
        comps[j] = _combine_jets(terms, form.X, out_order, form.var_names)
    return DifferentialForm(degree=k + 1, dim=m, components=comps, var_names=form.var_names)


# --------------------------------------------------------------------------- #
# Hodge star / codifferential / Hodge Laplacian (flat Euclidean metric)
# --------------------------------------------------------------------------- #
def hodge_star(form: DifferentialForm) -> DifferentialForm:
    r"""Flat-Euclidean Hodge star ``*``: a ``k``-form -> ``(m-k)``-form (pointwise).

    ``*(dx^I) = eps(I, I^c) dx^{I^c}`` with ``I^c`` the complementary increasing
    index. Order-preserving (no differentiation). ``*1`` is the volume form and
    ``** = (-1)^{k(m-k)}`` on the Euclidean metric.
    """
    m, k = form.dim, form.degree
    comps: dict[FormIndex, FieldJet] = {}
    for index in _increasing_indices(m, k):
        comp_c = _complement(index, m)
        sign = _permutation_sign(tuple(index) + comp_c)
        comps[comp_c] = _scale_jet(form.components[index], float(sign))
    return DifferentialForm(degree=m - k, dim=m, components=comps, var_names=form.var_names)


def codifferential(form: DifferentialForm) -> DifferentialForm:
    r"""Flat-Euclidean codifferential ``delta``: a ``k``-form -> ``(k-1)``-form.

    ``(delta omega)_{I'} = -sum_a partial_a omega_{a I'}`` (antisymmetrised), the
    metric adjoint of ``d``. On a 1-form this is *minus* the divergence; ``delta``
    is nilpotent (``delta . delta = 0``) and equals ``(-1)^{m(k+1)+1} * d *``.
    """
    m, k = form.dim, form.degree
    if k == 0:
        raise ValueError("codifferential of a 0-form is identically zero")
    if form.order < 1:
        raise ValueError("codifferential needs component order >= 1")
    out_order = form.order - 1
    comps: dict[FormIndex, FieldJet] = {}
    for sub in _increasing_indices(m, k - 1):
        terms: list[tuple[float, FieldJet]] = []
        for axis in range(m):
            if axis in sub:
                continue
            full = tuple(sorted((axis,) + sub))
            sign = _permutation_sign((axis,) + sub)
            terms.append((-float(sign), field_derivative_jet(form.components[full], axis)))
        comps[sub] = _combine_jets(terms, form.X, out_order, form.var_names)
    return DifferentialForm(degree=k - 1, dim=m, components=comps, var_names=form.var_names)


def hodge_laplacian(form: DifferentialForm) -> DifferentialForm:
    r"""Hodge--de Rham Laplacian ``Delta = d delta + delta d`` (order drops by 2).

    On the flat Euclidean metric this acts component-wise as *minus* the ordinary
    Laplacian (``(Delta omega)_I = -sum_a partial^2_a omega_I``), the zero-curvature
    Weitzenbock identity. On a 0-form, ``Delta f = delta d f = -sum_a partial^2_a f``.
    """
    m, k = form.dim, form.degree
    if form.order < 2:
        raise ValueError("Hodge Laplacian needs component order >= 2")
    out_order = form.order - 2
    pieces: list[DifferentialForm] = []
    if k < m:
        pieces.append(codifferential(exterior_derivative(form)))  # delta d
    if k >= 1:
        pieces.append(exterior_derivative(codifferential(form)))  # d delta
    comps: dict[FormIndex, FieldJet] = {}
    for index in _increasing_indices(m, k):
        terms = [(1.0, piece.components[index]) for piece in pieces]
        comps[index] = _combine_jets(terms, form.X, out_order, form.var_names)
    return DifferentialForm(degree=k, dim=m, components=comps, var_names=form.var_names)


# --------------------------------------------------------------------------- #
# Wedge product (pointwise) + closedness certificates
# --------------------------------------------------------------------------- #
def wedge(a: DifferentialForm, b: DifferentialForm) -> DifferentialForm:
    r"""Pointwise exterior product ``a ^ b`` (degree ``p + q``; values only).

    ``(a ^ b)_K = sum_{I + J = K} eps(I, J) a_I b_J``. The result is an order-0
    form (its component *values* only), enough for the graded-commutativity
    identity ``a ^ b = (-1)^{pq} b ^ a`` and ``alpha ^ alpha = 0`` for odd-degree
    ``alpha``.
    """
    if a.dim != b.dim:
        raise ValueError("wedge needs forms of equal dimension")
    m, p, q = a.dim, a.degree, b.degree
    if p + q > m:
        raise ValueError(f"wedge degree {p + q} exceeds dim {m}")
    comps: dict[FormIndex, FieldJet] = {}
    for whole in _increasing_indices(m, p + q):
        total = np.zeros(a.n)
        for i_part in combinations(whole, p):
            j_part = tuple(x for x in whole if x not in i_part)
            sign = _permutation_sign(tuple(i_part) + j_part)
            total = total + sign * a.value(i_part) * b.value(j_part)
        comps[whole] = _value_jet(a.X, total, a.var_names)
    return DifferentialForm(degree=p + q, dim=m, components=comps, var_names=a.var_names)


def closedness_residual(form: DifferentialForm) -> float:
    """Sup-norm of ``d omega`` -- zero iff the form is closed (``d omega = 0``)."""
    if form.degree == form.dim:
        return 0.0
    return exterior_derivative(form).max_abs()


def coclosedness_residual(form: DifferentialForm) -> float:
    """Sup-norm of ``delta omega`` -- zero iff the form is co-closed."""
    if form.degree == 0:
        return 0.0
    return codifferential(form).max_abs()


# --------------------------------------------------------------------------- #
# grad / curl / div correspondence + Maxwell
# --------------------------------------------------------------------------- #
def gradient_form(scalar_jet: FieldJet) -> DifferentialForm:
    r"""The gradient 1-form ``df`` of a scalar field (``d`` of a 0-form)."""
    return exterior_derivative(scalar_form(scalar_jet))


def curl_form(vector_jets: list[FieldJet] | tuple[FieldJet, ...]) -> DifferentialForm:
    r"""The curl 2-form ``d omega`` of a vector field's 1-form (``d`` of a 1-form)."""
    return exterior_derivative(one_form(vector_jets))


def electromagnetic_field_2form(potential: DifferentialForm) -> DifferentialForm:
    r"""The electromagnetic field strength ``F = dA`` from the potential 1-form ``A``.

    The homogeneous Maxwell equations ``dF = 0`` are then the ``d . d = 0`` identity
    (:func:`closedness_residual` of ``F`` is zero to machine precision).
    """
    if potential.degree != 1:
        raise ValueError("the electromagnetic potential A must be a 1-form")
    return exterior_derivative(potential)


# --------------------------------------------------------------------------- #
# Identity certification + neural end-to-end
# --------------------------------------------------------------------------- #
def evaluate_exterior_calculus(*, seed: int = 0) -> dict[str, float]:
    """Certify the de Rham--Hodge identities on a random neural field (residuals).

    Fits a smooth 3-D :class:`NeuralFieldND`, builds the gradient 1-form, and
    reports the sup-norm residuals of: ``d d f = 0`` (curl grad), ``d d omega = 0``
    (div curl), ``delta delta = 0``, the Hodge-star roundtrip ``** = (-1)^{k(m-k)}``,
    and ``Delta f = -lap f``. All are exact (machine precision) by construction.
    """
    from omnibias.symbolic.field_discovery import (
        extract_field_jet,
        field_laplacian,
        fit_neural_field_nd,
    )

    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.6, 0.6, size=(64, 3))
    names = ("x", "y", "z")
    f = fit_neural_field_nd(X, rng.normal(size=64), hidden=32, seed=seed, var_names=names)
    fjet = extract_field_jet(f, X, max_order=3)
    g = [fit_neural_field_nd(X, rng.normal(size=64), hidden=24, seed=seed + 10 + i,
                             var_names=names) for i in range(3)]
    gjets = [extract_field_jet(gi, X, max_order=3) for gi in g]

    f0 = scalar_form(fjet)
    omega = one_form(gjets)
    dd_f = closedness_residual(gradient_form(fjet))
    dd_omega = exterior_derivative(exterior_derivative(omega)).max_abs()
    delta2 = codifferential(codifferential(exterior_derivative(omega))).max_abs()

    # Hodge-star roundtrip on the 1-form: ** = (-1)^{k(m-k)} = +1 here (k=1,m=3).
    star_roundtrip = max(
        float(np.max(np.abs(hodge_star(hodge_star(omega)).value(i) - omega.value(i))))
        for i in [(0,), (1,), (2,)]
    )
    # Hodge Laplacian of the 0-form equals minus the ordinary Laplacian.
    hodge_vs_lap = float(
        np.max(np.abs(hodge_laplacian(f0).value(()) + field_laplacian(fjet)))
    )
    return {
        "dd_scalar_curl_grad": dd_f,
        "dd_oneform_div_curl": dd_omega,
        "delta_squared": delta2,
        "star_roundtrip": star_roundtrip,
        "hodge_laplacian_vs_laplacian": hodge_vs_lap,
    }
