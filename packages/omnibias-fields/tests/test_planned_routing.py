# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""A deep field that offers ``planned_chain`` is routed through the planner; one that does not, or opts out, is not."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.planned import PlannedChain, planned_routing, planned_routing_enabled
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState

DIMS = [3, 7, 7, 2]  # (x, y, t) -> (u, v)
AXES = ("x", "y", "t")


def _layers(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k, (a, b) in enumerate(zip(DIMS[:-1], DIMS[1:], strict=True)):
        out.append(
            (
                rng.standard_normal((b, a)) * 0.6 / np.sqrt(a),
                0.2 * rng.standard_normal(b),
                "tanh" if k < len(DIMS) - 2 else None,
            )
        )
    return out


class _Backend:
    """Array namespace plus the dense autodiff reference the stub field's own methods use."""

    def __init__(self, name):
        self.name = name
        if name == "torch":
            import torch

            self.xp = torch
            from omnibias.fields.torch import _ops_dispatch as ops

            self.asarray = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
            self.to_np = lambda a: a.detach().numpy()
        else:
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            self.xp = jnp
            from omnibias.fields.jax import _ops_dispatch as ops

            self.asarray = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)
            self.to_np = np.asarray
        self.ops = ops

    def forward(self, layers, x):
        a = x
        for W, b, spec in layers:
            a = a @ W.T + b
            if spec == "tanh":
                a = self.xp.tanh(a)
        return a

    def hessian(self, f, z):
        if self.name == "torch":
            return self.xp.func.jacrev(self.xp.func.jacrev(f))(z)
        import jax

        return jax.hessian(f)(z)

    def grad(self, f, z):
        if self.name == "torch":
            return self.xp.func.jacrev(f)(z)
        import jax

        return jax.grad(f)(z)

    def vmap(self, f):
        if self.name == "torch":
            return self.xp.func.vmap(f)
        import jax

        return jax.vmap(f)


class DeepField:
    """A ``jet_mlp``-marked field with autodiff methods, optionally offering the dense chain."""

    _omnibias_dispatch = "jet_mlp"

    def __init__(self, backend, layers, *, offer_chain=True):
        self.b = backend
        self.layers = [(backend.asarray(W), backend.asarray(bias), s) for W, bias, s in layers]
        self.offer_chain = offer_chain
        self.coordinate_spec = CoordinateSpec(axes=AXES, time_axis="t")
        self.components = ComponentSpec(names=("u", "v"))
        self.legacy_calls = 0

    def evaluate(self, coords):
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=self.b.ops,
            sigma_cache=SigmaCache(z=coords),
        )

    # -- the opt-in hook ---------------------------------------------------------------------
    def planned_chain(self, state, names):
        if not self.offer_chain:
            return None
        idx = np.asarray([self.components.names.index(n) for n in names])
        *hidden, (W, bias, spec) = self.layers
        layers = [*hidden, (W[idx], bias[idx], spec)]
        spatial = [state.coordinate_spec.axis_index(a) for a in state.coordinate_spec.spatial_axes]
        return PlannedChain(layers, coordinates=spatial)

    # -- the field's own (legacy) methods: dense autodiff over the spatial axes ----------------
    def _spatial(self, state):
        return [state.coordinate_spec.axis_index(a) for a in state.coordinate_spec.spatial_axes]

    def _lap(self, f, axes):
        def g(z):
            H = self.b.hessian(f, z)
            return sum(H[a, a] for a in axes)

        return g

    def _power(self, state, name, power):
        c = self.components.names.index(name)
        axes = self._spatial(state)

        def f(z):
            return self.b.forward(self.layers, z)[c]

        for _ in range(power):
            f = self._lap(f, axes)
        return self.b.vmap(f)(state.coords)

    def laplacian(self, state, name):
        self.legacy_calls += 1
        return self._power(state, name, 1)

    def biharmonic(self, state, name):
        self.legacy_calls += 1
        return self._power(state, name, 2)

    def polylaplacian(self, state, name, *, k):
        self.legacy_calls += 1
        return self._power(state, name, k)

    def derivative(self, state, name, *, axis, order):
        self.legacy_calls += 1
        assert order == 1
        c = self.components.names.index(name)
        a = state.coordinate_spec.axis_index(axis)
        return self.b.vmap(
            lambda z: self.b.grad(lambda w: self.b.forward(self.layers, w)[c], z)[a]
        )(state.coords)


@pytest.fixture(params=["torch", "jax"])
def backend(request):
    pytest.importorskip(request.param)
    return _Backend(request.param)


def _points(backend, n=5, seed=3):
    return backend.asarray(np.random.default_rng(seed).standard_normal((n, 3)))


def test_routed_operators_match_the_fields_own_methods_and_skip_them(backend):
    field = DeepField(backend, _layers())
    state = field.evaluate(_points(backend))
    ops = backend.ops
    with planned_routing(False):
        want = {
            "lap": ops.laplacian(state, "u"),
            "bilap": ops.biharmonic(state, "u"),
            "poly3": ops.polylaplacian(state, "v", k=3),
        }
    field.legacy_calls = 0
    got = {
        "lap": ops.laplacian(state, "u"),
        "bilap": ops.biharmonic(state, "u"),
        "poly3": ops.polylaplacian(state, "v", k=3),
    }
    assert field.legacy_calls == 0, "the planner was not used"
    for key in ("lap", "bilap", "poly3"):
        np.testing.assert_allclose(
            backend.to_np(got[key]), backend.to_np(want[key]), rtol=1e-9, atol=1e-10
        )
        assert tuple(got[key].shape) == (5,)


def test_routed_divergence_over_the_spatial_axes_matches_the_fields_own_derivatives(backend):
    field = DeepField(backend, _layers(seed=2))
    state = field.evaluate(_points(backend, seed=5))
    ops = backend.ops
    with planned_routing(False):
        want = backend.to_np(ops.divergence(state, ("u", "v")))
    field.legacy_calls = 0
    got = backend.to_np(ops.divergence(state, ("u", "v")))
    assert field.legacy_calls == 0
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-10)


class _NoHook(DeepField):
    planned_chain = None  # a field class that never defined the hook


def test_a_field_without_the_hook_or_one_that_declines_keeps_its_own_methods(backend):
    for field in (DeepField(backend, _layers(), offer_chain=False), _NoHook(backend, _layers())):
        state = field.evaluate(_points(backend))
        backend.ops.laplacian(state, "u")
        backend.ops.biharmonic(state, "u")
        assert field.legacy_calls == 2


def test_routing_switch_is_scoped_and_hook_type_is_checked(backend):
    assert planned_routing_enabled()
    with planned_routing(False):
        assert not planned_routing_enabled()
        with planned_routing(True):
            assert planned_routing_enabled()
        assert not planned_routing_enabled()
    assert planned_routing_enabled()

    class Bad(DeepField):
        def planned_chain(self, state, names):
            return [("not", "a", "chain")]

    state = Bad(backend, _layers()).evaluate(_points(backend))
    with pytest.raises(TypeError, match="PlannedChain"):
        backend.ops.laplacian(state, "u")


def test_gradients_reach_the_layer_tensors_of_the_routed_operator():
    torch = pytest.importorskip("torch")
    be = _Backend("torch")
    field = DeepField(be, _layers(seed=4))
    for W, b, _ in field.layers:
        W.requires_grad_(True)
        b.requires_grad_(True)
    state = field.evaluate(_points(be))
    out = be.ops.laplacian(state, "u")
    params = [t for W, b, _ in field.layers for t in (W, b)]
    grads = torch.autograd.grad(out.sum(), params, allow_unused=True)
    assert grads[0] is not None and float(grads[0].abs().sum()) > 0
    with planned_routing(False):
        ref = torch.autograd.grad(be.ops.laplacian(state, "u").sum(), params, allow_unused=True)
    for g, r in zip(grads, ref, strict=True):
        if r is None:
            assert g is None or float(g.abs().sum()) == 0.0
        else:
            torch.testing.assert_close(g, r, rtol=1e-8, atol=1e-10)


class OneOutputField(DeepField):
    """A single-component deep field that hands the planner its parameters as they are (stable tensors)."""

    def __init__(self, backend, seed=0, grad=False):
        rng = np.random.default_rng(seed)
        dims = [3, 7, 7, 1]
        layers = [
            (
                rng.standard_normal((b, a)) * 0.6 / np.sqrt(a),
                0.2 * rng.standard_normal(b),
                "tanh" if k < len(dims) - 2 else None,
            )
            for k, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True))
        ]
        super().__init__(backend, layers)
        self.components = ComponentSpec(names=("u",))
        if grad:
            for W, b, _ in self.layers:
                W.requires_grad_(True)
                b.requires_grad_(True)

    def planned_chain(self, state, names):
        spatial = [state.coordinate_spec.axis_index(a) for a in state.coordinate_spec.spatial_axes]
        return PlannedChain(self.layers, coordinates=spatial)


def _torch_routing(monkeypatch, size=8):
    pytest.importorskip("torch")
    from omnibias.fields.torch.ops import _planned

    built = []
    real = _planned._build

    def counting(*a, **k):
        built.append(1)
        return real(*a, **k)

    monkeypatch.setattr(_planned, "_build", counting)
    monkeypatch.setattr(_planned, "OPERATOR_CACHE_SIZE", size)
    _planned._OPERATORS.clear()
    return _planned, built


def test_torch_routing_reuses_the_compiled_operator_for_stable_parameters(monkeypatch):
    planned, built = _torch_routing(monkeypatch)
    be = _Backend("torch")
    field = OneOutputField(be)
    for seed in (3, 4, 5):
        state = field.evaluate(_points(be, seed=seed))
        be.ops.laplacian(state, "u")
    assert len(built) == 1
    be.ops.biharmonic(field.evaluate(_points(be)), "u")  # another operator: its own program
    assert len(built) == 2
    # new parameter tensors of the same structure never read the operator built for the old ones
    other = OneOutputField(be, seed=9)
    got = be.ops.laplacian(other.evaluate(_points(be)), "u")
    assert len(built) == 3
    with planned_routing(False):
        want = be.ops.laplacian(other.evaluate(_points(be)), "u")
    np.testing.assert_allclose(be.to_np(got), be.to_np(want), rtol=1e-9, atol=1e-10)


def test_torch_routing_cached_operator_matches_fresh_operators_through_training_updates(monkeypatch):
    torch = pytest.importorskip("torch")
    be = _Backend("torch")

    def run(size):
        planned, _ = _torch_routing(monkeypatch, size)
        field = OneOutputField(be, seed=2, grad=True)
        params = [t for W, b, _ in field.layers for t in (W, b)]
        out = []
        for step in range(3):
            state = field.evaluate(_points(be, seed=10 + step))
            lap = be.ops.laplacian(state, "u")
            bil = be.ops.biharmonic(state, "u")
            loss = (lap**2).sum() + (bil**2).sum()
            grads = torch.autograd.grad(loss, params)
            out.append(([be.to_np(lap), be.to_np(bil)], [g.detach().numpy().copy() for g in grads]))
            with torch.no_grad():  # an optimizer step: in place, the same tensors
                for p, g in zip(params, grads, strict=True):
                    p.sub_(1e-3 * g)
        return out

    cached, fresh = run(8), run(0)
    for (vc, gc), (vf, gf) in zip(cached, fresh, strict=True):
        for a, b in zip(vc + gc, vf + gf, strict=True):
            np.testing.assert_array_equal(a, b)


def test_torch_routing_bypasses_the_cache_for_derived_tensors_and_bounds_it(monkeypatch):
    planned, built = _torch_routing(monkeypatch, size=2)
    be = _Backend("torch")
    derived = DeepField(be, _layers())  # row selection: new tensors on every call
    for _ in range(3):
        be.ops.laplacian(derived.evaluate(_points(be)), "u")
    assert len(built) == 3 and len(planned._OPERATORS) <= 1  # rebuilt every call; one entry per structure
    fields = [OneOutputField(be, seed=s) for s in range(4)]
    for f in fields:
        be.ops.laplacian(f.evaluate(_points(be)), "u")
    assert len(planned._OPERATORS) <= 2


def test_vector_laplacians_are_one_planner_call_and_match_the_components(backend, monkeypatch):
    field = DeepField(backend, _layers(seed=6))
    state = field.evaluate(_points(backend, seed=7))
    ops = backend.ops
    module = __import__(f"omnibias.fields.{backend.name}.ops.high_order", fromlist=["_routed"])
    calls = []
    real = module._routed
    monkeypatch.setattr(module, "_routed", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    with planned_routing(False):
        want = {
            "lap": ops.vector_laplacian(state, ("u", "v")),
            "bil": ops.vector_biharmonic(state, ("u", "v")),
            "poly": ops.vector_polylaplacian(state, ("u", "v"), k=3),
        }
    calls.clear()
    field.legacy_calls = 0
    got = {
        "lap": ops.vector_laplacian(state, ("u", "v")),
        "bil": ops.vector_biharmonic(state, ("u", "v")),
        "poly": ops.vector_polylaplacian(state, ("u", "v"), k=3),
    }
    assert field.legacy_calls == 0 and calls == [("u", "v")] * 3
    for key, value in got.items():
        assert tuple(value.shape) == (5, 2)
        np.testing.assert_allclose(backend.to_np(value), backend.to_np(want[key]), rtol=1e-8, atol=1e-9)
    one = ops.vector_laplacian(state, ("u",))  # a single component keeps the per-component path and the shape
    assert tuple(one.shape) == (5, 1)


def test_generic_hessian_fallback_requests_each_pair_once_and_mirrors_it(backend, monkeypatch):
    """Mixed partials commute: the fallback asks for the upper triangle and mirrors it."""
    module = __import__(f"omnibias.fields.{backend.name}.ops.high_order", fromlist=["hessian"])
    n, batch = 3, 4
    raw = np.random.default_rng(11).standard_normal((batch, n, n))
    truth = raw + np.swapaxes(raw, -1, -2)
    calls = {"second": [], "mixed": []}

    def second(state, name, *, axis, order):
        calls["second"].append((axis, order))
        return backend.asarray(truth[:, axis, axis])

    def mixed(state, name, axes, orders):
        calls["mixed"].append((tuple(axes), tuple(orders)))
        i, j = axes
        return backend.asarray(truth[:, i, j])

    monkeypatch.setattr(module, "derivative", second)
    monkeypatch.setattr(module, "mixed_partial", mixed)
    field = DeepField(backend, _layers())
    field._omnibias_dispatch = "generic"  # none of the closed-form dispatches
    state = field.evaluate(_points(backend, n=batch))
    full = module.hessian(state, "u")
    assert sorted(calls["second"]) == [(0, 2), (1, 2), (2, 2)]
    assert sorted(calls["mixed"]) == [((0, 1), (1, 1)), ((0, 2), (1, 1)), ((1, 2), (1, 1))]
    np.testing.assert_array_equal(backend.to_np(full), truth)
    sub = module.hessian(state, "u", axes=(0, 2))
    np.testing.assert_array_equal(backend.to_np(sub), truth[:, [0, 2], :][:, :, [0, 2]])


def test_jax_program_cache_is_bounded_and_serves_every_weight_value(monkeypatch):
    pytest.importorskip("jax")
    from omnibias.fields.jax.ops import _planned

    monkeypatch.setattr(_planned, "PROGRAM_CACHE_SIZE", 2)
    _planned._PROGRAMS.clear()
    be = _Backend("jax")
    seen = []
    for seed in (1, 2, 3):  # same structure, other weights: one program
        field = DeepField(be, _layers(seed=seed))
        got = be.ops.laplacian(field.evaluate(_points(be)), "u")
        with planned_routing(False):
            want = be.ops.laplacian(field.evaluate(_points(be)), "u")
        np.testing.assert_allclose(be.to_np(got), be.to_np(want), rtol=1e-9, atol=1e-10)
        seen.append(len(_planned._PROGRAMS))
    assert seen == [1, 1, 1]
    field = DeepField(be, _layers())
    for points in (4, 5, 6):  # three more structures: at most two programs are kept
        be.ops.laplacian(field.evaluate(_points(be, n=points)), "u")
        assert len(_planned._PROGRAMS) <= 2


def test_jax_diffusion_is_read_back_once_per_array(monkeypatch):
    pytest.importorskip("jax")
    from omnibias.fields.jax.ops import _planned

    _Backend("jax")  # enables x64
    import jax.numpy as jnp

    box = [jnp.asarray(np.diag([1.0, 2.0, 3.0]))]
    metric = box[0]
    reads = []
    real = np.array

    def counting(a, *args, **kwargs):
        if a is box[0]:
            reads.append(1)
        return real(a, *args, **kwargs)

    monkeypatch.setattr(_planned.np, "array", counting)
    first = _planned._host_diffusion(metric)
    second = _planned._host_diffusion(metric)
    assert len(reads) == 1 and second[0] is first[0] and second[1] == first[1]
    other = _planned._host_diffusion(jnp.asarray(np.diag([1.0, 2.0, 4.0])))
    assert other[1] != first[1]
    monkeypatch.undo()
    ident = id(metric)
    box.clear()
    del metric, first, second, counting
    import gc

    gc.collect()
    assert ident not in _planned._DIFFUSIONS  # the entry goes with its array


def test_quadrature_conversions_are_remembered_per_rule_dtype_and_device():
    from omnibias.fields._core.quadrature import QuadratureSpec, _converted

    spec = QuadratureSpec(
        rule="test", nodes=np.linspace(-1.0, 1.0, 5).reshape(-1, 1), weights=np.full(5, 0.4)
    )
    made = []

    def make(tag):
        def build():
            made.append(tag)
            return object()

        return build

    a = _converted(spec, ("weights", "f64"), make("a"))
    assert _converted(spec, ("weights", "f64"), make("a2")) is a
    assert _converted(spec, ("weights", "f32"), make("b")) is not a
    refused = _converted(spec, ("nodes", "f64"), make("c"), keep=lambda v: False)
    assert _converted(spec, ("nodes", "f64"), make("c2")) is not refused  # a refused value is not remembered
    assert made == ["a", "b", "c", "c2"]
    other = QuadratureSpec(rule="test", nodes=spec.nodes.copy(), weights=spec.weights.copy())
    assert _converted(other, ("weights", "f64"), make("d")) is not a  # per rule object
