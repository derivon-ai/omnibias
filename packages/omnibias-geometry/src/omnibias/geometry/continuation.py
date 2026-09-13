# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numerical continuation of user-supplied finite residual families.

Derivatives are supplied explicitly (including tower-derived derivatives). No
finite differences are hidden here. Results describe the supplied finite map;
validated consumers live in ``omnibias.dynamics.continuation``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Map = Callable[[Array], Array]


@dataclass(frozen=True)
class ImplicitFamily:
    """F(y)=0, y=(state, parameter); jacobian is (n,n+1).

    ``second`` and ``third`` contain unnormalized derivatives on all y axes.
    The final parameter is a continuation coordinate, not a temperature.
    """

    value: Map
    jacobian: Map
    second: Map | None = None
    third: Map | None = None
    name: str = "finite residual family"


@dataclass(frozen=True)
class Branch:
    points: Array
    tangents: Array
    residuals: Array
    steps: Array
    status: str


@dataclass(frozen=True)
class Bifurcation:
    kind: str
    point: Array
    augmented_point: Array
    residual: float
    transversality: float
    coefficient: float
    nondegenerate: bool
    frequency: float = 0.0
    certified: bool = False


def _vector(x: object) -> Array:
    out = np.asarray(x, dtype=float)
    if out.ndim != 1 or not np.all(np.isfinite(out)):
        raise ValueError("expected a finite vector")
    return out


def _newton(
    value: Map, jacobian: Map, start: Array, tol: float, iterations: int
) -> tuple[Array, bool]:
    x = start.copy()
    for _ in range(iterations):
        f = np.asarray(value(x), dtype=float)
        if not np.all(np.isfinite(f)):
            return x, False
        norm = float(np.linalg.norm(f, np.inf))
        if norm <= tol:
            return x, True
        try:
            delta = np.linalg.solve(jacobian(x), -f)
        except np.linalg.LinAlgError:
            return x, False
        for k in range(12):
            trial = x + (0.5**k) * delta
            if np.linalg.norm(value(trial), np.inf) < norm:
                x = trial
                break
        else:
            return x, False
    return x, bool(np.linalg.norm(value(x), np.inf) <= tol)


def branch_tangent(family: ImplicitFamily, point: Array, orientation: Array | None = None) -> Array:
    """Unit numerical null vector on a regular one-dimensional stratum."""
    y = _vector(point)
    j = np.asarray(family.jacobian(y), dtype=float)
    if j.shape != (y.size - 1, y.size) or not np.all(np.isfinite(j)):
        raise ValueError("jacobian must have shape (n,n+1) and finite entries")
    _, s, vh = np.linalg.svd(j, full_matrices=True)
    if s[-1] <= 1e-12 * max(1.0, float(s[0])):
        raise ValueError("branch Jacobian has lost row rank")
    t = vh[-1].copy()
    if orientation is not None and float(t @ orientation) < 0:
        t = -t
    return np.asarray(t, dtype=float)


def continue_branch(
    family: ImplicitFamily,
    start: Array,
    *,
    direction: Array | None = None,
    step: float = 0.05,
    n_steps: int = 50,
    min_step: float = 1e-6,
    max_step: float | None = None,
    tol: float = 1e-10,
    max_corrector: int = 15,
) -> Branch:
    """Pseudo-arclength predictor/corrector with bounded step reduction.

    The initial point must already solve the supplied residual to ``tol``.
    A fold in the last coordinate is traversed without switching parameters.
    """
    y = _vector(start).copy()
    if y.size < 2 or step <= 0 or min_step <= 0 or min_step > step or n_steps < 0 or tol <= 0:
        raise ValueError("invalid continuation dimensions, steps, or tolerance")
    cap = step if max_step is None else float(max_step)
    if cap < step or not np.isfinite(cap):
        raise ValueError("max_step must be finite and at least step")
    if np.linalg.norm(family.value(y), np.inf) > tol:
        raise ValueError("start must solve the residual to tolerance")
    orientation = _vector(direction) if direction is not None else np.eye(y.size)[-1]
    t = branch_tangent(family, y, orientation)
    points, tangents, residuals, steps = (
        [y.copy()],
        [t.copy()],
        [float(np.linalg.norm(family.value(y), np.inf))],
        [0.0],
    )
    ds, status = step, "complete"
    for _ in range(n_steps):
        accepted = False
        while ds >= min_step:
            predicted = y + ds * t

            def f(z: Array, tangent: Array = t, prediction: Array = predicted) -> Array:
                return np.concatenate((family.value(z), [float(tangent @ (z - prediction))]))

            def j(z: Array, tangent: Array = t) -> Array:
                return np.vstack((family.jacobian(z), tangent))

            corrected, accepted = _newton(f, j, predicted, tol, max_corrector)
            if accepted:
                break
            ds *= 0.5
        if not accepted:
            status = "corrector_failed"
            break
        try:
            new_t = branch_tangent(family, corrected, t)
        except ValueError:
            status = "rank_event"
            break
        y, t = corrected, new_t
        points.append(y.copy())
        tangents.append(t.copy())
        residuals.append(float(np.linalg.norm(family.value(y), np.inf)))
        steps.append(ds)
        ds = min(cap, ds * 1.2)
    return Branch(
        np.asarray(points), np.asarray(tangents), np.asarray(residuals), np.asarray(steps), status
    )


def fold_system(family: ImplicitFamily, dimension: int) -> tuple[Map, Map]:
    """Augmented equations (F, D_state F v, (v.v-1)/2)."""
    if family.second is None:
        raise ValueError("fold localization requires second derivatives")
    second = family.second
    n = dimension

    def value(z: Array) -> Array:
        y, v = z[: n + 1], z[n + 1 :]
        return np.concatenate((family.value(y), family.jacobian(y)[:, :n] @ v, [(v @ v - 1) / 2]))

    def jacobian(z: Array) -> Array:
        y, v = z[: n + 1], z[n + 1 :]
        j = family.jacobian(y)
        out = np.zeros((2 * n + 1, 2 * n + 1))
        out[:n, : n + 1] = j
        out[n : 2 * n, : n + 1] = np.einsum("ijk,j->ik", second(y)[:, :n, :], v)
        out[n : 2 * n, n + 1 :] = j[:, :n]
        out[-1, n + 1 :] = v
        return out

    return value, jacobian


def locate_fold(family: ImplicitFamily, guess: Array, *, tol: float = 1e-10) -> Bifurcation:
    y = _vector(guess)
    n = y.size - 1
    _, _, vh = np.linalg.svd(family.jacobian(y)[:, :n])
    f, j = fold_system(family, n)
    z, ok = _newton(f, j, np.concatenate((y, vh[-1])), tol, 30)
    y, v = z[: n + 1], z[n + 1 :]
    u, s, _ = np.linalg.svd(family.jacobian(y)[:, :n])
    left = u[:, -1]
    cross = float(left @ family.jacobian(y)[:, -1])
    assert family.second is not None
    quadratic = float(left @ np.einsum("ijk,j,k->i", family.second(y)[:, :n, :n], v, v) / 2)
    simple = n == 1 or s[-2] > tol
    nondegenerate = bool(ok and simple and abs(cross) > tol and abs(quadratic) > tol)
    return Bifurcation(
        "fold", y, z, float(np.linalg.norm(f(z), np.inf)), cross, quadratic, nondegenerate
    )


def hopf_system(family: ImplicitFamily, reference: NDArray[np.complex128]) -> tuple[Map, Map]:
    """Augmented equilibrium/eigenpair system with unit norm and fixed phase."""
    if family.second is None:
        raise ValueError("Hopf localization requires second derivatives")
    second = family.second
    n = reference.size

    def value(z: Array) -> Array:
        y, a, b, w = z[: n + 1], z[n + 1 : 2 * n + 1], z[2 * n + 1 : 3 * n + 1], z[-1]
        mat = family.jacobian(y)[:, :n]
        return np.concatenate(
            (
                family.value(y),
                mat @ a + w * b,
                mat @ b - w * a,
                [(a @ a + b @ b - 1) / 2, reference.real @ b - reference.imag @ a],
            )
        )

    def jacobian(z: Array) -> Array:
        y, a, b, w = z[: n + 1], z[n + 1 : 2 * n + 1], z[2 * n + 1 : 3 * n + 1], z[-1]
        jac = family.jacobian(y)
        mat = jac[:, :n]
        h = second(y)[:, :n, :]
        out = np.zeros((3 * n + 2, 3 * n + 2))
        out[:n, : n + 1] = jac
        out[n : 2 * n, : n + 1] = np.einsum("ijk,j->ik", h, a)
        out[2 * n : 3 * n, : n + 1] = np.einsum("ijk,j->ik", h, b)
        out[n : 2 * n, n + 1 : 2 * n + 1] = mat
        out[n : 2 * n, 2 * n + 1 : 3 * n + 1] = w * np.eye(n)
        out[2 * n : 3 * n, n + 1 : 2 * n + 1] = -w * np.eye(n)
        out[2 * n : 3 * n, 2 * n + 1 : 3 * n + 1] = mat
        out[n : 2 * n, -1] = b
        out[2 * n : 3 * n, -1] = -a
        out[-2, n + 1 : 2 * n + 1] = a
        out[-2, 2 * n + 1 : 3 * n + 1] = b
        out[-1, n + 1 : 2 * n + 1] = -reference.imag
        out[-1, 2 * n + 1 : 3 * n + 1] = reference.real
        return out

    return value, jacobian


def locate_hopf(family: ImplicitFamily, guess: Array, *, tol: float = 1e-9) -> Bifurcation:
    """Nondegenerate Hopf diagnostic using Kuznetsov's first Lyapunov coefficient.

    The tensors are derivatives B=D²F,C=D³F (no factorial normalization).
    Formula: Elements of Applied Bifurcation Theory, equation (5.62).
    """
    if family.second is None or family.third is None:
        raise ValueError("Hopf normal form requires second and third derivatives")
    y = _vector(guess)
    n = y.size - 1
    vals, vecs = np.linalg.eig(family.jacobian(y)[:, :n])
    candidates = [i for i, v in enumerate(vals) if v.imag > tol]
    if not candidates:
        raise ValueError("no positive-frequency eigenpair near guess")
    k = min(candidates, key=lambda i: abs(vals[i].real))
    q = vecs[:, k].astype(complex)
    f, j = hopf_system(family, q)
    z, ok = _newton(f, j, np.concatenate((y, q.real, q.imag, [vals[k].imag])), tol, 30)
    y = z[: n + 1]
    q = z[n + 1 : 2 * n + 1] + 1j * z[2 * n + 1 : 3 * n + 1]
    w = float(z[-1])
    mat = family.jacobian(y)[:, :n]
    eig = np.linalg.eigvals(mat)
    h, c = family.second(y)[:, :n, :n], family.third(y)[:, :n, :n, :n]

    def b(u: NDArray[np.complex128], v: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.einsum("ijk,j,k->i", h, u, v)  # type: ignore[no-any-return]

    ev, pv = np.linalg.eig(mat.T.astype(complex))
    p = pv[:, int(np.argmin(abs(ev + 1j * w)))]
    pair = np.vdot(p, q)
    if abs(pair) <= tol or w <= tol:
        raise ValueError("degenerate eigenpair normalization")
    p = p / np.conj(pair)
    try:
        term = np.einsum("ijkl,j,k,l->i", c, q, q, q.conj()) - 2 * b(
            q, np.asarray(np.linalg.solve(mat, b(q, q.conj())), dtype=np.complex128)
        )
        term += b(
            q.conj(),
            np.asarray(np.linalg.solve(2j * w * np.eye(n) - mat, b(q, q)), dtype=np.complex128),
        )
        coefficient = float(np.real(np.vdot(p, term)) / (2 * w))
        dy = np.concatenate((-np.linalg.solve(mat, family.jacobian(y)[:, -1]), [1.0]))
        dmat = np.einsum("ijk,k->ij", family.second(y)[:, :n, :], dy)
        cross = float(np.real(np.vdot(p, dmat @ q)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("resonance prevents nondegenerate Hopf normal form") from exc
    critical = sum(abs(v.real) <= 10 * tol for v in eig)
    nondegenerate = bool(ok and critical == 2 and abs(cross) > tol and abs(coefficient) > tol)
    return Bifurcation(
        "hopf", y, z, float(np.linalg.norm(f(z), np.inf)), cross, coefficient, nondegenerate, w
    )


@dataclass(frozen=True)
class BoundaryPath:
    """Numerical geodesic toward a thin observation-manifold direction."""

    points: Array
    velocities: Array
    smallest_singular_values: Array
    status: str


def follow_model_boundary(
    jacobian: Map,
    directional_second: Callable[[Array, Array], Array],
    start: Array,
    *,
    direction: Array | None = None,
    step: float = 0.02,
    n_steps: int = 50,
    rtol: float = 1e-10,
    admissible: Callable[[Array], bool] | None = None,
    max_parameter_norm: float = 1e6,
) -> BoundaryPath:
    """Trace a regular observation-manifold geodesic with RK4.

    On a full-column-rank chart J, acceleration is -J^+ D²F[v,v]. Whiten
    J and D²F consistently for a weighted observation metric. Initial speed
    in observation space is one; the weakest initial direction is the default.
    A rank/domain/budget stop proposes a boundary; it does not prove a limit.
    """
    x = _vector(start).copy()
    if step <= 0 or not np.isfinite(step) or n_steps < 0 or rtol <= 0 or not np.isfinite(rtol):
        raise ValueError("positive finite step/rank tolerance and nonnegative step count required")
    if max_parameter_norm <= 0 or not np.isfinite(max_parameter_norm):
        raise ValueError("positive finite parameter bound required")

    def data(y: Array) -> tuple[Array, Array, Array]:
        if not np.all(np.isfinite(y)) or np.linalg.norm(y) > max_parameter_norm:
            raise ValueError("parameter_bound")
        if admissible is not None and not admissible(y):
            raise ValueError("domain_boundary")
        mat = np.asarray(jacobian(y), dtype=float)
        if (
            mat.ndim != 2
            or mat.shape[1] != x.size
            or mat.shape[0] < x.size
            or not np.all(np.isfinite(mat))
        ):
            raise ValueError("full-column-rank observation chart required")
        _, singular, right = np.linalg.svd(mat, full_matrices=False)
        if singular[-1] <= rtol * max(1.0, float(singular[0])):
            raise ValueError("numerical_rank_boundary")
        return mat, singular, right

    mat, singular, right = data(x)
    velocity = np.asarray(right[-1] if direction is None else _vector(direction), dtype=float)
    if velocity.shape != x.shape or np.linalg.norm(mat @ velocity) == 0:
        raise ValueError("nonzero compatible initial direction required")
    velocity = velocity / np.linalg.norm(mat @ velocity)
    points, velocities, floors = [x.copy()], [velocity.copy()], [float(singular[-1])]
    status = "complete"

    def rhs(y: Array, v: Array) -> tuple[Array, Array]:
        j, _, _ = data(y)
        second = np.asarray(directional_second(y, v), dtype=float)
        if second.shape != (j.shape[0],) or not np.all(np.isfinite(second)):
            raise ValueError("invalid directional second derivative")
        acceleration = -np.linalg.lstsq(j, second, rcond=None)[0]
        return v, np.asarray(acceleration, dtype=float)

    for _ in range(n_steps):
        try:
            k1, l1 = rhs(x, velocity)
            k2, l2 = rhs(x + step * k1 / 2, velocity + step * l1 / 2)
            k3, l3 = rhs(x + step * k2 / 2, velocity + step * l2 / 2)
            k4, l4 = rhs(x + step * k3, velocity + step * l3)
            new_x = x + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            new_v = velocity + step * (l1 + 2 * l2 + 2 * l3 + l4) / 6
            _, singular, _ = data(new_x)
        except ValueError as exc:
            status = str(exc)
            break
        x, velocity = new_x, new_v
        points.append(x.copy())
        velocities.append(velocity.copy())
        floors.append(float(singular[-1]))
    return BoundaryPath(np.asarray(points), np.asarray(velocities), np.asarray(floors), status)


__all__ = [
    "Bifurcation",
    "BoundaryPath",
    "Branch",
    "ImplicitFamily",
    "branch_tangent",
    "continue_branch",
    "fold_system",
    "follow_model_boundary",
    "hopf_system",
    "locate_fold",
    "locate_hopf",
]
