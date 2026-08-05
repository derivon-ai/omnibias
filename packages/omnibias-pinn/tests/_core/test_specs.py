# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Schema invariants for the pure-Python ``_core`` module."""

from __future__ import annotations

import pytest
from omnibias.pinn import (
    ComponentSpec,
    CoordinateSpec,
    EquationSpec,
    IncompressibilityPolicy,
    ResidualPolicy,
)
from omnibias.pinn._core import registry

# -------------------- CoordinateSpec --------------------


def test_coordinate_spec_basic():
    spec = CoordinateSpec(("x", "y", "t"))
    assert spec.axes == ("x", "y", "t")
    assert spec.ndim == 3
    assert spec.spatial_axes == ("x", "y")
    assert spec.n_spatial == 2
    assert spec.time_axis == "t"
    assert spec.is_periodic("x") is False
    assert spec.is_time("t") is True
    assert spec.is_spatial("x") is True


def test_coordinate_spec_periodic():
    spec = CoordinateSpec(
        ("x", "y", "z", "t"), periodicity=(True, True, True, False),
    )
    assert spec.is_periodic("x")
    assert not spec.is_periodic("t")
    spec2 = CoordinateSpec(("x", "y"), periodicity=True)
    assert all(spec2.periodicity)


def test_coordinate_spec_axis_index():
    spec = CoordinateSpec(("x", "y", "t"))
    assert spec.axis_index("x") == 0
    assert spec.axis_index("t") == 2
    assert spec.axis_index(0) == 0
    assert spec.axis_index(-1) == 2
    with pytest.raises(KeyError):
        spec.axis_index("q")
    with pytest.raises(IndexError):
        spec.axis_index(99)
    with pytest.raises(TypeError):
        spec.axis_index(1.5)


def test_coordinate_spec_no_time():
    spec = CoordinateSpec(("x", "y"), time_axis=None)
    assert spec.time_axis is None
    assert spec.spatial_axes == ("x", "y")


def test_coordinate_spec_explicit_time_axis():
    spec = CoordinateSpec(("x", "tau"), time_axis="tau")
    assert spec.time_axis == "tau"
    assert spec.spatial_axes == ("x",)


def test_coordinate_spec_unknown_time_axis_raises():
    with pytest.raises(ValueError):
        CoordinateSpec(("x", "y"), time_axis="t")


def test_coordinate_spec_duplicates_raise():
    with pytest.raises(ValueError):
        CoordinateSpec(("x", "x"))


def test_coordinate_spec_domain_validation():
    spec = CoordinateSpec(("x", "y"), domain=[(0.0, 1.0), (-1.0, 1.0)])
    assert spec.domain == ((0.0, 1.0), (-1.0, 1.0))
    with pytest.raises(ValueError):
        CoordinateSpec(("x", "y"), domain=[(0.0, 1.0)])
    with pytest.raises(ValueError):
        CoordinateSpec(("x",), domain=[(1.0, 0.0)])


def test_coordinate_spec_repr_and_asdict():
    spec = CoordinateSpec(
        ("x", "t"), periodicity=(True, False), domain=[(0.0, 1.0), (0.0, 2.0)],
    )
    r = repr(spec)
    assert "axes=('x', 't')" in r
    d = spec.asdict()
    assert d["axes"] == ["x", "t"]
    assert d["periodicity"] == [True, False]
    assert d["domain"] == [[0.0, 1.0], [0.0, 2.0]]


def test_coordinate_spec_equality_and_hash():
    a = CoordinateSpec(("x", "y", "t"))
    b = CoordinateSpec(("x", "y", "t"))
    assert a == b
    assert hash(a) == hash(b)
    c = CoordinateSpec(("x", "y", "z"), time_axis=None)
    assert a != c


# -------------------- ComponentSpec --------------------


def test_component_spec_basic():
    spec = ComponentSpec(("u", "v", "w", "p"))
    assert spec.names == ("u", "v", "w", "p")
    assert spec.n_components == 4
    assert spec.index("v") == 1
    assert "u" in spec
    assert "q" not in spec


def test_component_spec_groups():
    spec = ComponentSpec(
        ("u", "v", "w", "p"),
        groups={"velocity": ("u", "v", "w")},
    )
    assert spec.is_group("velocity")
    assert spec.group_members("velocity") == ("u", "v", "w")
    assert "velocity" in spec
    assert spec.all_known_names() == ("u", "v", "w", "p", "velocity")


def test_component_spec_group_collisions():
    with pytest.raises(ValueError):
        ComponentSpec(("u",), groups={"u": ("u",)})
    with pytest.raises(ValueError):
        ComponentSpec(("u", "v"), groups={"velocity": ("q",)})
    with pytest.raises(ValueError):
        ComponentSpec(("u", "v"), groups={"velocity": ()})
    with pytest.raises(ValueError):
        ComponentSpec(("u", "v"), groups={"velocity": ("u", "u")})


def test_component_spec_duplicate_names():
    with pytest.raises(ValueError):
        ComponentSpec(("u", "u"))


def test_component_spec_empty():
    with pytest.raises(ValueError):
        ComponentSpec(())


def test_component_spec_iter_and_len():
    spec = ComponentSpec(("a", "b"))
    assert list(spec) == ["a", "b"]
    assert len(spec) == 2


def test_component_spec_equality_and_hash():
    a = ComponentSpec(("u", "v"), groups={"vel": ("u", "v")})
    b = ComponentSpec(("u", "v"), groups={"vel": ("u", "v")})
    assert a == b
    assert hash(a) == hash(b)
    c = ComponentSpec(("u", "v"), groups={"vel": ("v", "u")})
    assert a != c


def test_component_spec_repr_and_asdict():
    spec = ComponentSpec(("u", "v"), groups={"vel": ("u", "v")})
    r = repr(spec)
    assert "names=('u', 'v')" in r
    d = spec.asdict()
    assert d["names"] == ["u", "v"]
    assert d["groups"] == {"vel": ["u", "v"]}


# -------------------- EquationSpec --------------------


def test_equation_spec_basic():
    spec = EquationSpec(
        "navier_stokes",
        required_components=("u", "v", "w", "p"),
        required_groups=("velocity",),
        meta={"form": "primitive_3d"},
    )
    assert spec.name == "navier_stokes"
    assert spec.required_components == ("u", "v", "w", "p")
    assert spec.required_groups == ("velocity",)
    assert spec.requires_time is True
    assert dict(spec.meta) == {"form": "primitive_3d"}


def test_equation_spec_validation_against_state():
    # Build a fake state-like object with components/coordinate_spec attrs.
    class FakeState:
        def __init__(self, components, coordinate_spec):
            self.components = components
            self.coordinate_spec = coordinate_spec

    spec = EquationSpec(
        "ns",
        required_components=("u", "v", "p"),
        required_groups=("velocity",),
    )
    good = FakeState(
        ComponentSpec(("u", "v", "p"), groups={"velocity": ("u", "v")}),
        CoordinateSpec(("x", "y", "t")),
    )
    spec.validate_state(good)  # no raise

    missing_comp = FakeState(
        ComponentSpec(("u", "v"), groups={"velocity": ("u", "v")}),
        CoordinateSpec(("x", "y", "t")),
    )
    with pytest.raises(ValueError):
        spec.validate_state(missing_comp)

    missing_group = FakeState(
        ComponentSpec(("u", "v", "p")),
        CoordinateSpec(("x", "y", "t")),
    )
    with pytest.raises(ValueError):
        spec.validate_state(missing_group)

    no_time = FakeState(
        ComponentSpec(("u", "v", "p"), groups={"velocity": ("u", "v")}),
        CoordinateSpec(("x", "y"), time_axis=None),
    )
    with pytest.raises(ValueError):
        spec.validate_state(no_time)


def test_residual_policy_defaults():
    p = ResidualPolicy()
    assert p.reduction == "mean"
    assert p.sobolev_p == 1.0


def test_incompressibility_policy_defaults():
    p = IncompressibilityPolicy()
    assert p.mode == "soft"
    assert p.weight == 1.0


# -------------------- registry --------------------


def test_registry_round_trip():
    registry.clear()
    spec = EquationSpec("toy", required_components=("u",))
    registry.register_spec(spec)
    assert registry.get_equation_spec("toy") == spec
    assert "toy" in registry.list_equation_specs()
    assert registry.has_equation_spec("toy")
    with pytest.raises(KeyError):
        registry.get_equation_spec("missing")
    registry.clear()


def test_registry_factory_round_trip():
    registry.clear()
    spec = EquationSpec("toy", required_components=("u",))
    registry.register_spec(spec)
    registry.register_factory("toy", "torch", lambda **kw: ("torch_factory", kw))
    registry.register_factory("toy", "jax", lambda **kw: ("jax_factory", kw))
    fac_t = registry.get_equation_factory("toy", "torch")
    assert fac_t(viscosity=1e-3)[0] == "torch_factory"
    fac_j = registry.get_equation_factory("toy", "jax")
    assert fac_j(viscosity=1e-3)[0] == "jax_factory"
    with pytest.raises(ValueError):
        registry.register_factory("toy", "numpy", lambda: None)
    with pytest.raises(ValueError):
        registry.register_factory("toy", "torch", lambda: None)
    registry.clear()


def test_navier_stokes_registry_wiring_with_backend_factories():
    from omnibias.pinn.jax.equations import NavierStokes as JaxNavierStokes
    from omnibias.pinn.torch.equations import NavierStokes as TorchNavierStokes

    registry.clear()
    spec = EquationSpec(
        "navier_stokes",
        required_components=("u", "v", "w", "p"),
        required_groups=("velocity",),
        meta={"form": "primitive_3d"},
    )
    registry.register_spec(spec)
    registry.register_factory("navier_stokes", "torch", TorchNavierStokes)
    registry.register_factory("navier_stokes", "jax", JaxNavierStokes)

    assert registry.get_equation_spec("navier_stokes") == spec
    torch_factory = registry.get_equation_factory("navier_stokes", "torch")
    jax_factory = registry.get_equation_factory("navier_stokes", "jax")
    assert isinstance(torch_factory(viscosity=0.1), TorchNavierStokes)
    assert isinstance(jax_factory(viscosity=0.1), JaxNavierStokes)
    registry.clear()
