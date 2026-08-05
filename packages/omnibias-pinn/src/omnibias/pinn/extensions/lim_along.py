# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``lim_along`` -- expose the jet ``lim`` operator through the ops registry.

The closed-form jet limit (``mlp_jet`` + ``lhopital_ratio``) is a *model-level*
operator: it needs the layer stack of the network, not an evaluated
:class:`~omnibias.fields.FieldState`. The built-in field ops (``grad``,
``laplacian``, ...) instead act on the evaluated state. This module bridges the
two via the documented ``ops_registry`` extension point so that, *opt-in*, a
limit can be read as an attribute on a component view::

    from omnibias.pinn.extensions import register_lim_along
    from omnibias.pinn.jax.losses import asymptotic_ratio

    register_lim_along()                       # add the op (opt-in, global)
    state = field(coords)
    state.extra["lim_along"] = {               # per-state closures
        "u": lambda: asymptotic_ratio(layers, x0, v, rate=1),
    }
    state.u.lim_along                          # -> differentiable limit of u

The registered op is intentionally a thin adaptor: it looks up a user-supplied
zero-argument closure in ``state.extra['lim_along'][component]`` and calls it.
The closure is what actually computes the differentiable limit (typically built
from :func:`omnibias.jax.losses.asymptotic_ratio` or its torch twin), which keeps
this module backend-neutral and keeps the honest computation at the model level.

Registration is *opt-in* so the fixed v0.1 field-ops surface is untouched unless
a user explicitly calls :func:`register_lim_along`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from omnibias.fields import ops_registry

if TYPE_CHECKING:  # pragma: no cover -- typing-only
    from omnibias.fields import FieldState

#: Key under which per-state ``lim_along`` closures live in ``FieldState.extra``.
LIM_ALONG_KEY = "lim_along"


def _lim_along_op(state: FieldState[Any], name: str) -> Any:
    """Registry adaptor: evaluate the ``lim_along`` closure for ``name``.

    Reads ``state.extra[LIM_ALONG_KEY]`` -- a ``{component: callable}`` mapping --
    and calls the closure registered for the requested component. Raises a clear
    error when no closure is available rather than returning a silent ``None``.
    """
    # ``FieldState.__getattr__`` routes attribute access to component views, so
    # mypy cannot see the ``extra`` slot; cast to the real per-state cache type.
    extra = cast("dict[str, Any]", state.extra)
    closures = extra.get(LIM_ALONG_KEY)
    if not isinstance(closures, dict):
        raise KeyError(
            "lim_along requires state.extra['lim_along'] to be a "
            "{component: callable} mapping; set it before accessing "
            f"state.<component>.{LIM_ALONG_KEY}"
        )
    if name not in closures:
        available = tuple(sorted(closures))
        raise KeyError(
            f"no lim_along closure registered for component {name!r}; "
            f"available: {available}"
        )
    closure = closures[name]
    if not callable(closure):
        raise TypeError(
            f"state.extra['lim_along'][{name!r}] must be callable, got "
            f"{type(closure).__name__}"
        )
    return closure()


def register_lim_along(*, name: str = LIM_ALONG_KEY, overwrite: bool = False) -> str:
    """Register the ``lim_along`` op on the global :mod:`ops_registry` (opt-in).

    Parameters
    ----------
    name
        Registry name (and the per-state ``extra`` key) to bind. Defaults to
        ``"lim_along"``.
    overwrite
        If ``True``, silently replace an existing registration under ``name``;
        otherwise a re-registration raises (matching the registry's default).

    Returns
    -------
    str
        The registered ``name`` (convenient for cleanup in tests).
    """
    if overwrite and ops_registry.lookup(name) is not None:
        ops_registry.unregister(name)
    ops_registry.register(name)(_lim_along_op)
    return name


def unregister_lim_along(*, name: str = LIM_ALONG_KEY) -> None:
    """Remove the ``lim_along`` op from the registry (no-op if absent)."""
    ops_registry.unregister(name)


__all__ = [
    "LIM_ALONG_KEY",
    "register_lim_along",
    "unregister_lim_along",
]
