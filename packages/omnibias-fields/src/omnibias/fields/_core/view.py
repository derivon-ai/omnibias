# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Option 1 attribute DSL: :class:`ComponentView` and :class:`VectorView`.

These are the user-facing views that turn ``state.u.dt`` into a single
delegate to ``state.ops.derivative(state, "u", axis=coordinate_spec.time_axis)``.
The DSL is
backend-agnostic -- both torch and jax ops modules expose the same
``derivative``, ``gradient``, ``laplacian``, ... entry points, so the same
view classes serve both.

Design contract
---------------

Every property on :class:`ComponentView` or :class:`VectorView` is exactly
one line: it forwards to the backend ops module. Math lives in the kernel,
not in the view. This is what makes the bit-parity tests cheap: the
``test_view_delegation.py`` matrix asserts ``state.u.<attr> == ops.<fn>(
state, ...)`` for every (component, op) pair, and adding a new op grows
the matrix by one row.

To extend this surface for a third-party op, register the op via the
``ops_registry`` and the view's ``__getattr__`` will pick it up
automatically.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.components import ComponentSpec
    from omnibias.fields._core.state import FieldState

T = TypeVar("T")


def did_you_mean(name: str, components: ComponentSpec) -> str:
    """Build a friendly :class:`AttributeError` message for typos."""
    available_components = components.names
    available_groups = tuple(g for g, _ in components.groups)
    candidates = available_components + available_groups
    suggestions = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    base = (
        f"FieldState has no component or group {name!r}. "
        f"Available components: {available_components!r}"
    )
    if available_groups:
        base += f" or groups: {available_groups!r}"
    if suggestions:
        base += f". Did you mean {suggestions[0]!r}?"
    else:
        base += "."
    return base


class ComponentView(Generic[T]):
    """Single-component view: ``state.u``.

    Each attribute forwards to a one-line kernel call. The view is frozen
    by construction (we use ``__slots__``) so it can be cached freely; in
    practice it is just a (state, name) pair.
    """

    __slots__ = ("_state", "_name")

    def __init__(self, state: FieldState, name: str) -> None:
        if not state.components.is_component(name):
            raise KeyError(
                f"{name!r} is not a scalar component of "
                f"{state.components.names!r}"
            )
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_name", name)

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover
        raise AttributeError("ComponentView is frozen")

    @property
    def state(self) -> FieldState:
        return self._state

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> T:
        return self._state.ops.value(self._state, self._name)

    def _spatial_axis(self, index: int, attr: str) -> str:
        axes = self._state.coordinate_spec.spatial_axes
        if index >= len(axes):
            raise AttributeError(
                f"{attr} requires spatial axis index {index}, but "
                f"coordinate_spec.spatial_axes={axes!r}"
            )
        return axes[index]

    @property
    def dt(self) -> T:
        time_axis = self._state.coordinate_spec.time_axis
        if time_axis is None:
            raise AttributeError(
                "dt requires a time axis on the coordinate spec "
                f"(got {self._state.coordinate_spec!r})"
            )
        return self._state.ops.derivative(
            self._state, self._name, axis=time_axis
        )

    @property
    def dx(self) -> T:
        return self._state.ops.derivative(
            self._state, self._name, axis=self._spatial_axis(0, "dx")
        )

    @property
    def dy(self) -> T:
        return self._state.ops.derivative(
            self._state, self._name, axis=self._spatial_axis(1, "dy")
        )

    @property
    def dz(self) -> T:
        return self._state.ops.derivative(
            self._state, self._name, axis=self._spatial_axis(2, "dz")
        )

    @property
    def grad(self) -> T:
        """Spatial gradient ``grad u`` of shape ``(B, n_spatial)``.

        **Spatial only**: the time axis (if any) is excluded, consistent with
        :attr:`lap` and :attr:`hess_spatial` -- but *not* with :attr:`hess`.
        """
        return self._state.ops.gradient(self._state, self._name)

    @property
    def lap(self) -> T:
        """Spatial Laplacian ``sum_i d^2u/dx_i^2`` over the spatial axes only
        (the time axis is excluded)."""
        return self._state.ops.laplacian(self._state, self._name)

    @property
    def hess(self) -> T:
        """**Full** Hessian over *all* coordinate axes, shape ``(B, D, D)``.

        .. warning::
            Unlike :attr:`grad` and :attr:`lap` (which are spatial-only), this
            includes the **time axis** when the field has one, so on a
            spacetime field ``hess`` is ``(B, D, D)`` with the
            ``d^2/dt^2`` / mixed ``d^2/dx dt`` entries present. Use
            :attr:`hess_spatial` for the spatial-only ``(B, n_spatial,
            n_spatial)`` Hessian that matches :attr:`grad` / :attr:`lap`.
        """
        return self._state.ops.hessian(self._state, self._name)

    @property
    def hess_spatial(self) -> T:
        """Spatial-only Hessian, shape ``(B, n_spatial, n_spatial)``.

        Excludes the time axis; this is the spatial counterpart to the
        full-spacetime :attr:`hess`, and is axis-consistent with :attr:`grad`
        and :attr:`lap`.
        """
        return self._state.ops.spatial_hessian(self._state, self._name)

    @property
    def biharm(self) -> T:
        return self._state.ops.biharmonic(self._state, self._name)

    @property
    def grad_sq(self) -> T:
        """``|grad u|^2`` (spatial)."""
        return self._state.ops.grad_squared_norm(self._state, self._name)

    def dalembertian(self, *, c: float = 1.0, signature: str = "mostly_plus") -> T:
        """d'Alembert / wave operator ``box u`` (requires a time axis)."""
        return self._state.ops.dalembertian(
            self._state, self._name, c=c, signature=signature,
        )

    def diffusive_flux(self, *, diffusivity: float | str = 1.0) -> T:
        """Fickian flux ``F = -D grad u`` (shape ``(B, n_spatial)``)."""
        return self._state.ops.diffusive_flux(
            self._state, self._name, diffusivity=diffusivity,
        )

    def variable_diffusion(self, *, diffusivity: float | str = 1.0) -> T:
        """``div(D grad u)`` with constant or scalar-field ``D``."""
        return self._state.ops.variable_coefficient_diffusion(
            self._state, self._name, diffusivity=diffusivity,
        )

    def fickian_flux(self, *, diffusivity: float | str = 1.0) -> T:
        """Fick's first law ``J = -D grad c`` (chemistry alias of ``diffusive_flux``)."""
        return self._state.ops.fickian_flux(
            self._state, self._name, diffusivity=diffusivity,
        )

    def darcy_flux(self, *, permeability: float | str = 1.0, viscosity: float = 1.0) -> T:
        """Darcy seepage velocity ``q = -(k/mu) grad p`` (this component is ``p``)."""
        return self._state.ops.darcy_flux(
            self._state, self._name, permeability=permeability, viscosity=viscosity,
        )

    def reaction_diffusion(
        self,
        *,
        diffusivity: float | str = 1.0,
        reaction: Any = None,
        source: str | float | None = None,
    ) -> T:
        """Reaction-diffusion residual ``d_t c - div(D grad c) - R(c) - s`` (needs a time axis)."""
        return self._state.ops.reaction_diffusion_residual(
            self._state, scalar=self._name, diffusivity=diffusivity,
            reaction=reaction, source=source,
        )

    def poisson_residual(
        self, *, source: str | float | None = None, permittivity: float | str = 1.0,
    ) -> T:
        """Poisson residual ``div(eps grad phi) + rho`` (this component is ``phi``)."""
        return self._state.ops.poisson_residual(
            self._state, self._name, source=source, permittivity=permittivity,
        )

    def d(self, axis: int | str, order: int = 1) -> T:
        """Generic single-axis derivative escape hatch."""
        return self._state.ops.derivative(
            self._state, self._name, axis=axis, order=order,
        )

    def dn(
        self,
        axes: Sequence[int | str],
        orders: Sequence[int],
    ) -> T:
        """Mixed partial: ``d^|orders| u / dx_{axes[0]}^{orders[0]} ...``."""
        return self._state.ops.mixed_partial(
            self._state, self._name, tuple(axes), tuple(orders),
        )

    def polylap(self, k: int) -> T:
        """``Delta^k u`` via the polylaplacian kernel."""
        return self._state.ops.polylaplacian(self._state, self._name, k=k)

    def p_lap(self, p: float) -> T:
        """``div(|grad u|^{p-2} grad u)`` via the p-Laplacian op."""
        return self._state.ops.p_laplacian(self._state, self._name, p=p)

    def directional(self, direction: T) -> T:
        """``d_direction u = grad u . direction``."""
        return self._state.ops.directional_derivative(
            self._state, self._name, direction=direction,
        )

    def grad_of_d(self, axis: int | str) -> T:
        """``grad(d u / dx_axis)`` over all coordinate axes."""
        return self._state.ops.gradient_of_derivative(
            self._state, self._name, axis=axis,
        )

    def integrate(self, *, rule: Any) -> T:
        """Definite integral over the domain implied by ``rule`` (quadrature)."""
        return self._state.ops.integrate(self._state, self._name, rule=rule)

    def line_integral(self, curve: Any, *, rule: Any) -> T:
        """Gradient-theorem line integral ``int_C grad u . dr`` along ``curve``.

        Equals ``u(curve(t1)) - u(curve(t0))`` by the multivariate FTC; ``state``
        must be pre-evaluated at ``curve(quadrature_nodes(rule))``.
        """
        return self._state.ops.line_integral(self._state, self._name, curve, rule=rule)

    def l2_norm(self, *, rule: Any) -> T:
        """``||u||_{L2}`` over the quadrature domain."""
        return self._state.ops.l2_norm(self._state, self._name, rule=rule)

    def sobolev_norm(
        self, *, rule: Any, k: int = 1, weights: tuple[float, ...] | None = None,
    ) -> T:
        """``||u||_{H^k}`` over the quadrature domain (``k`` in ``{0, 1, 2}``)."""
        return self._state.ops.sobolev_norm(
            self._state, self._name, rule=rule, k=k, weights=weights,
        )

    # Allow registry-based extensions: ``state.u.symmetric_laplacian`` works
    # if a third party registered "symmetric_laplacian" via ops_registry.
    def __getattr__(self, attr: str) -> Any:
        # Lookup in ops registry.
        from omnibias.fields._core.ops_registry import lookup
        op = lookup(attr)
        if op is None:
            raise AttributeError(
                f"ComponentView has no attribute {attr!r} and no op "
                f"registered under that name. Available registered ops: "
                f"{tuple(self._state.ops.list_ops()) if hasattr(self._state.ops, 'list_ops') else ()}"
            )
        # Bind: registered ops are callables of (state, name, **kwargs);
        # accessed as a property they should evaluate immediately, so we
        # call op(state, name) and return the result.
        return op(self._state, self._name)

    def __repr__(self) -> str:
        return (
            f"ComponentView({self._name!r}, "
            f"field={type(self._state.field).__name__})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ComponentView)
            and other._state is self._state
            and other._name == self._name
        )

    def __hash__(self) -> int:
        return hash((id(self._state), self._name))


class VectorView(Generic[T]):
    """Group view: ``state.velocity``.

    Behaves as a ``Sequence[ComponentView]`` (iteration, index, length),
    plus the vector-level operator surface (curl, div, advect, ...). All
    of these forward to one-line kernel calls.
    """

    __slots__ = ("_state", "_names")

    def __init__(self, state: FieldState, names: Sequence[str]) -> None:
        normalised = tuple(names)
        if len(normalised) == 0:
            raise ValueError("VectorView requires at least one component name")
        for n in normalised:
            if not state.components.is_component(n):
                raise KeyError(
                    f"{n!r} not in components {state.components.names!r}"
                )
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_names", normalised)

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover
        raise AttributeError("VectorView is frozen")

    @property
    def state(self) -> FieldState:
        return self._state

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self) -> Iterator[ComponentView]:
        return iter(ComponentView(self._state, n) for n in self._names)

    def __getitem__(self, key: int | str | slice) -> Any:
        if isinstance(key, int):
            return ComponentView(self._state, self._names[key])
        if isinstance(key, str):
            if key not in self._names:
                raise KeyError(
                    f"{key!r} not in vector view {self._names!r}"
                )
            return ComponentView(self._state, key)
        if isinstance(key, slice):
            return tuple(
                ComponentView(self._state, n) for n in self._names[key]
            )
        raise TypeError(
            f"VectorView index must be int, str, or slice; got {type(key).__name__}"
        )

    @property
    def value(self) -> T:
        return self._state.ops.stack_components(self._state, self._names)

    @property
    def jac(self) -> T:
        return self._state.ops.jacobian(self._state, self._names)

    @property
    def spatial_jac(self) -> T:
        return self._state.ops.spatial_jacobian(self._state, self._names)

    @property
    def div(self) -> T:
        return self._state.ops.divergence(self._state, self._names)

    @property
    def curl(self) -> T:
        return self._state.ops.curl(self._state, self._names)

    @property
    def vort(self) -> T:
        """Alias for :attr:`curl` (vorticity = curl of velocity)."""
        return self.curl

    @property
    def lap(self) -> T:
        """Vector Laplacian: ``(Delta u_1, ..., Delta u_C)``."""
        return self._state.ops.vector_laplacian(self._state, self._names)

    @property
    def biharm(self) -> T:
        """Vector biharmonic: ``(Delta^2 u_1, ..., Delta^2 u_C)``."""
        return self._state.ops.vector_biharmonic(self._state, self._names)

    @property
    def hess(self) -> T:
        """Stacked component Hessians."""
        return self._state.ops.vector_hessian(self._state, self._names)

    @property
    def strain_rate(self) -> T:
        return self._state.ops.strain_rate(self._state, self._names)

    @property
    def rate_of_rotation(self) -> T:
        """Antisymmetric spin tensor ``W = 0.5(J - J^T)`` (complements ``strain_rate``)."""
        return self._state.ops.rate_of_rotation_tensor(self._state, self._names)

    @property
    def grad_div(self) -> T:
        """``grad(div u)``."""
        return self._state.ops.gradient_of_divergence(self._state, self._names)

    @property
    def curl_curl(self) -> T:
        """``curl(curl u)`` (2D / 3D)."""
        return self._state.ops.curl_of_curl(self._state, self._names)

    @property
    def deformation_gradient(self) -> T:
        return self._state.ops.deformation_gradient(self._state, self._names)

    @property
    def dt(self) -> T:
        time_axis = self._state.coordinate_spec.time_axis
        if time_axis is None:
            raise AttributeError(
                "dt requires a time axis on the coordinate spec "
                f"(got {self._state.coordinate_spec!r})"
            )
        return self._state.ops.vector_derivative(
            self._state, self._names, axis=time_axis,
        )

    def advect(self) -> T:
        """Self-advection: ``(u . nabla) u``. The standard NS term."""
        return self._state.ops.advection(self._state, velocity=self._names)

    def advect_by(self, other: VectorView) -> T:
        """``(other . nabla) self``. Use for cross-advection of e.g. scalars."""
        if not isinstance(other, VectorView):
            raise TypeError(
                f"advect_by argument must be a VectorView, got {type(other).__name__}"
            )
        if other.state is not self._state:
            raise ValueError(
                "VectorView.advect_by: both views must share the same FieldState"
            )
        return self._state.ops.advection(
            self._state, velocity=other._names, target=self._names,
        )

    def material_derivative(self) -> T:
        """``D/Dt = d/dt + (u . nabla)``; standard NS material derivative."""
        return self._state.ops.material_derivative(
            self._state, velocity=self._names,
        )

    def skew_advect(self, scalar: str | None = None) -> T:
        """Skew-symmetric (energy/enstrophy-conserving) advection."""
        return self._state.ops.skew_symmetric_advection(
            self._state, velocity=self._names, scalar=scalar,
        )

    def newtonian_stress(self, *, viscosity: float = 1.0, pressure: str | None = None) -> T:
        """Incompressible-Newtonian Cauchy stress ``-p I + 2 mu eps``."""
        return self._state.ops.newtonian_stress(
            self._state, self._names, viscosity=viscosity, pressure=pressure,
        )

    def elastic_stress(self, *, lam: float = 1.0, mu: float = 1.0) -> T:
        """Isotropic linear-elastic (Hooke) stress ``lam tr(eps) I + 2 mu eps``."""
        return self._state.ops.linear_elastic_stress(
            self._state, self._names, lam=lam, mu=mu,
        )

    def viscous_dissipation(self, *, viscosity: float = 1.0) -> T:
        """Viscous dissipation rate ``2 mu eps:eps >= 0``."""
        return self._state.ops.viscous_dissipation(
            self._state, self._names, viscosity=viscosity,
        )

    def stokes_residual(
        self, *, pressure: str, viscosity: float = 1.0, body_force=None,  # type: ignore[no-untyped-def]
    ) -> T:
        """Stokes momentum residual ``mu Delta u - grad p + f``."""
        return self._state.ops.stokes_residual(
            self._state, velocity=self._names, pressure=pressure,
            viscosity=viscosity, body_force=body_force,
        )

    def navier_cauchy_residual(
        self, *, lam: float = 1.0, mu: float = 1.0, body_force=None,  # type: ignore[no-untyped-def]
    ) -> T:
        """Linear-elastostatics residual ``(lam+mu) grad(div u) + mu Delta u + f``."""
        return self._state.ops.navier_cauchy_residual(
            self._state, displacement=self._names, lam=lam, mu=mu, body_force=body_force,
        )

    def polylap(self, k: int) -> T:
        """Vector polylaplacian: ``(Delta^k u_1, ..., Delta^k u_C)``."""
        return self._state.ops.vector_polylaplacian(self._state, self._names, k=k)

    def dalembertian(self, *, c: float = 1.0, signature: str = "mostly_plus") -> T:
        """Componentwise wave operator ``box u_i`` (e.g. ``box A`` in Lorenz gauge)."""
        return self._state.ops.vector_dalembertian(
            self._state, self._names, c=c, signature=signature,
        )

    def __repr__(self) -> str:
        return (
            f"VectorView({list(self._names)!r}, "
            f"field={type(self._state.field).__name__})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, VectorView)
            and other._state is self._state
            and other._names == self._names
        )

    def __hash__(self) -> int:
        return hash((id(self._state), self._names))


__all__ = ["ComponentView", "VectorView", "did_you_mean"]
