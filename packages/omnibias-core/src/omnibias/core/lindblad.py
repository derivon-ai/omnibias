# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Open-system GKSL / Lindblad dynamics (theory 09-32).

A time-independent Gorini–Kossakowski–Sudarshan–Lindblad generator is a
linear semigroup. With column-stacking ``vec``::

    d(rho)/dt = L[rho] = -i [H, rho]
        + sum_k gamma_k (A_k rho A_k^dag - 1/2 {A_k^dag A_k, rho})
    rho(t) = exp(t L) rho(0)
    d^n rho / dt^n = L^n rho(t)

Arbitrary-order time derivatives therefore come from **one** propagator
evaluation plus matrix powers of ``L``. That is the *linear-semigroup*
analogue of the sigma tower -- one transcendental evaluation regardless
of order -- and must be labelled as such, not as the activation tower.
Only the qubit Bloch family and pure dephasing are closed form; general
finite ``d`` is a numerical propagator (see :func:`honesty_payload`).

Two collapse senses are named in this module and must not be conflated:

* The founding bias collapse (spread ``delta -> 0``, ``K`` biases
  coalesce into a smooth ``sigma^(K-1)``) never appears here.
* The ``beta -> inf`` zero-temperature limit of a thermal qubit
  population is the founding **temperature collapse** (feasibility
  sense: parameter ``beta``, surviving object ``indicator``), already
  minted by :mod:`omnibias.core.occupancy` / the founding spec. This
  module names that limit as an honest reference and does not request
  a new slot for it. The *new* registry slot is ``relaxation``
  (parameter ``relaxation_rate``, surviving object ``steady_state``) in
  :mod:`omnibias.core.collapse.relaxation`.

Do not conflate either sense with the other, and do not read a
``beta -> inf`` step as a Born–Markov derivation, a non-Markovian
claim, or a measurement-problem resolution -- see
:func:`honesty_payload`.

The GKSL form is a caller input. There is no derivation from a
system-bath Hamiltonian here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.core.occupancy import FermiModel, occupancy

ComplexMatrix = NDArray[np.complex128]


def honesty_payload() -> dict[str, bool]:
    """Permanently-false claims this module must never assert."""
    return {
        "founding_bias_collapse": False,
        "temperature_collapse": False,
        "requests_new_collapse_registry_slot": True,
        "wave_function_collapse_claim": False,
        "measurement_problem_resolved": False,
        "single_outcome_claim": False,
        "born_rule_derived": False,
        "markovian_model_declared_not_derived": True,
        "non_markovian_claim": False,
        "general_closed_form_claim": False,
        "thermodynamic_limit_taken": False,
        "continuum_limit_taken": False,
        "quantum_advantage_claim": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "float_residual_is_proof": False,
    }


def _as_square(matrix: object, *, name: str) -> ComplexMatrix:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
        raise ValueError(f"{name} must be a non-empty square matrix, got shape {arr.shape}")
    return arr


def _is_hermitian(matrix: ComplexMatrix, *, atol: float = 1e-12) -> bool:
    return bool(np.allclose(matrix, matrix.conj().T, atol=atol, rtol=0.0))


@dataclass(frozen=True)
class LindbladModel:
    """A finite-dimensional time-independent GKSL generator.

    ``hamiltonian`` must be square and Hermitian. Each jump operator is
    the same dimension; every rate must be finite and nonnegative.
    """

    hamiltonian: tuple[tuple[complex, ...], ...]
    jumps: tuple[tuple[tuple[complex, ...], ...], ...]
    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        h = _as_square(self.hamiltonian, name="hamiltonian")
        if not _is_hermitian(h):
            raise ValueError("hamiltonian must be Hermitian")
        object.__setattr__(
            self,
            "hamiltonian",
            tuple(tuple(complex(x) for x in row) for row in h),
        )
        dim = h.shape[0]
        jumps = tuple(self.jumps)
        rates = tuple(float(r) for r in self.rates)
        if len(jumps) != len(rates):
            raise ValueError("jumps and rates must have the same length")
        boxed: list[tuple[tuple[complex, ...], ...]] = []
        for k, jump in enumerate(jumps):
            arr = _as_square(jump, name=f"jumps[{k}]")
            if arr.shape[0] != dim:
                raise ValueError(
                    f"jumps[{k}] has shape {arr.shape}, expected ({dim}, {dim})"
                )
            boxed.append(tuple(tuple(complex(x) for x in row) for row in arr))
            rate = rates[k]
            if not math.isfinite(rate) or rate < 0.0:
                raise ValueError(f"rates[{k}] must be finite and >= 0, got {rate!r}")
        object.__setattr__(self, "jumps", tuple(boxed))
        object.__setattr__(self, "rates", tuple(rates))

    @property
    def dim(self) -> int:
        return len(self.hamiltonian)

    def hamiltonian_array(self) -> ComplexMatrix:
        return np.asarray(self.hamiltonian, dtype=np.complex128)

    def jump_arrays(self) -> tuple[ComplexMatrix, ...]:
        return tuple(np.asarray(jump, dtype=np.complex128) for jump in self.jumps)


def apply_lindblad(model: LindbladModel, rho: object) -> ComplexMatrix:
    """Evaluate ``L[rho]`` by the GKSL formula (not the superoperator)."""
    rho_arr = _as_square(rho, name="rho")
    if rho_arr.shape[0] != model.dim:
        raise ValueError(f"rho dim {rho_arr.shape[0]} != model dim {model.dim}")
    ham = model.hamiltonian_array()
    commutator = ham @ rho_arr - rho_arr @ ham
    out = -1j * commutator
    for jump, rate in zip(model.jump_arrays(), model.rates, strict=True):
        if rate == 0.0:
            continue
        dag = jump.conj().T
        lindblad = jump @ rho_arr @ dag - 0.5 * (dag @ jump @ rho_arr + rho_arr @ dag @ jump)
        out = out + rate * lindblad
    return out


def liouvillian(model: LindbladModel) -> ComplexMatrix:
    """Complex superoperator ``L`` of shape ``(d**2, d**2)``, column-stacked.

    ``vec(A X B) = (B.T kron A) vec(X)``.
    """
    dim = model.dim
    eye = np.eye(dim, dtype=np.complex128)
    ham = model.hamiltonian_array()
    gen = -1j * (np.kron(eye, ham) - np.kron(ham.T, eye))
    for jump, rate in zip(model.jump_arrays(), model.rates, strict=True):
        if rate == 0.0:
            continue
        dag = jump.conj().T
        left = dag @ jump
        dissipator = (
            np.kron(np.conj(jump), jump)
            - 0.5 * (np.kron(eye, left) + np.kron(left.T, eye))
        )
        gen = gen + rate * dissipator
    return np.asarray(gen, dtype=np.complex128)


def _expm(matrix: ComplexMatrix) -> ComplexMatrix:
    """Scaling-and-squaring Taylor ``exp(M)`` (no scipy dependency)."""
    n = matrix.shape[0]
    inf_norm = float(np.linalg.norm(matrix, ord=np.inf))
    squarings = 0 if inf_norm <= 0.5 else max(0, int(math.ceil(math.log2(inf_norm))) + 1)
    scaled = matrix / (2.0**squarings)
    term = np.eye(n, dtype=np.complex128)
    acc = term.copy()
    for k in range(1, 24):
        term = term @ scaled / k
        acc = acc + term
        if float(np.linalg.norm(term, ord=np.inf)) < 1e-18:
            break
    for _ in range(squarings):
        acc = acc @ acc
    return np.asarray(acc, dtype=np.complex128)


def propagator(model: LindbladModel, time: float) -> ComplexMatrix:
    """``exp(t L)`` as a complex ``(d**2, d**2)`` matrix."""
    t = float(time)
    if not math.isfinite(t) or t < 0.0:
        raise ValueError(f"time must be finite and nonnegative, got {time!r}")
    if t == 0.0:
        dim2 = model.dim * model.dim
        return np.eye(dim2, dtype=np.complex128)
    return _expm(t * liouvillian(model))


def _vec(rho: ComplexMatrix) -> ComplexMatrix:
    return rho.reshape(-1, order="F")


def _unvec(vector: ComplexMatrix, dim: int) -> ComplexMatrix:
    return vector.reshape((dim, dim), order="F")


def density_matrix(model: LindbladModel, rho0: object, time: float) -> ComplexMatrix:
    """``rho(t) = exp(t L) rho(0)``."""
    rho = _as_square(rho0, name="rho0")
    if rho.shape[0] != model.dim:
        raise ValueError(f"rho0 dim {rho.shape[0]} != model dim {model.dim}")
    evolved = propagator(model, time) @ _vec(rho)
    return _unvec(evolved, model.dim)


def time_derivative_tower(
    model: LindbladModel, rho0: object, time: float, *, order: int
) -> tuple[ComplexMatrix, ...]:
    """``(rho, L rho, ..., L^order rho)`` at time ``t``, one propagator call."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    rho = _as_square(rho0, name="rho0")
    if rho.shape[0] != model.dim:
        raise ValueError(f"rho0 dim {rho.shape[0]} != model dim {model.dim}")
    gen = liouvillian(model)
    vec = propagator(model, time) @ _vec(rho)
    rows: list[ComplexMatrix] = []
    for _ in range(order + 1):
        rows.append(_unvec(vec, model.dim))
        vec = gen @ vec
    return tuple(rows)


def steady_state(model: LindbladModel) -> ComplexMatrix:
    """Unique trace-1 kernel vector of ``L``, if the augmented system is full rank.

    Raises ``ValueError`` when the kernel is a manifold (pure dephasing)
    rather than a unique fixed point.
    """
    gen = liouvillian(model)
    dim = model.dim
    dim2 = dim * dim
    augmented = gen.copy()
    augmented[-1, :] = 0.0
    for i in range(dim):
        augmented[-1, i + i * dim] = 1.0
    rhs = np.zeros(dim2, dtype=np.complex128)
    rhs[-1] = 1.0
    rank = int(np.linalg.matrix_rank(augmented, tol=1e-10))
    if rank < dim2:
        raise ValueError("steady state is not unique")
    vec = np.asarray(np.linalg.solve(augmented, rhs), dtype=np.complex128)
    return _unvec(vec, dim)


def dissipative_gap(model: LindbladModel) -> float:
    """Second-largest real part of ``eig(L)``. A proposer, never a verdict.

    For a uniquely relaxing generator this is strictly negative. A
    nonnegative value means the float spectral abscissa does not certify
    contraction (unitary, or a kernel of dimension greater than one).
    """
    values = np.linalg.eigvals(liouvillian(model))
    reals = np.sort(values.real)[::-1]
    if reals.size < 2:
        return float(reals[0])
    return float(reals[1])


def thermal_steady_population(*, omega: float, beta: float) -> float:
    """Excited-state Fermi occupancy ``sigma(-beta * omega)``.

    Delegates to :func:`omnibias.core.occupancy.occupancy` with
    ``FermiModel(beta=beta, mu=0.0)``. The ``beta -> inf`` limit of this
    value is the founding temperature collapse (feasibility sense) and
    is not re-requested as a registry slot.
    """
    w = float(omega)
    b = float(beta)
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError(f"omega must be finite and > 0, got {omega!r}")
    return occupancy(FermiModel(beta=b, mu=0.0), w)


def qubit_pure_dephasing(*, rate: float) -> LindbladModel:
    """Two-level pure dephasing: ``A = sigma_z``, ``gamma = rate / 2``.

    Off-diagonals then decay as ``exp(-rate t)``, matching einselection
    with ``Gamma = rate``.
    """
    r = float(rate)
    if not math.isfinite(r) or r < 0.0:
        raise ValueError(f"rate must be finite and >= 0, got {rate!r}")
    return LindbladModel(
        hamiltonian=((0.0, 0.0), (0.0, 0.0)),
        jumps=(((1.0 + 0.0j, 0.0j), (0.0j, -1.0 + 0.0j)),),
        rates=(r / 2.0,),
    )


def qubit_thermal(
    *,
    omega: float,
    beta: float,
    gamma_down: float,
    gamma_phi: float = 0.0,
) -> LindbladModel:
    """Two-level thermal amplitude damping with optional dephasing.

    ``H = diag(0, omega)``, ``A_down = |0><1|`` at ``gamma_down``,
    ``A_up = |1><0|`` at ``gamma_down * exp(-beta * omega)`` (detailed
    balance), optional ``A_phi = sigma_z`` at ``gamma_phi``.
    """
    w = float(omega)
    b = float(beta)
    gd = float(gamma_down)
    gp = float(gamma_phi)
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError(f"omega must be finite and > 0, got {omega!r}")
    if not math.isfinite(gd) or gd < 0.0:
        raise ValueError(f"gamma_down must be finite and >= 0, got {gamma_down!r}")
    if not math.isfinite(gp) or gp < 0.0:
        raise ValueError(f"gamma_phi must be finite and >= 0, got {gamma_phi!r}")
    gamma_up = gd * math.exp(-b * w)
    jumps: list[tuple[tuple[complex, ...], ...]] = [
        ((0.0j, 1.0 + 0.0j), (0.0j, 0.0j)),
        ((0.0j, 0.0j), (1.0 + 0.0j, 0.0j)),
    ]
    rates: list[float] = [gd, gamma_up]
    if gp > 0.0:
        jumps.append(((1.0 + 0.0j, 0.0j), (0.0j, -1.0 + 0.0j)))
        rates.append(gp)
    return LindbladModel(
        hamiltonian=((0.0, 0.0), (0.0, complex(w))),
        jumps=tuple(jumps),
        rates=tuple(rates),
    )


def qubit_bloch_solution(
    *,
    omega: float,
    gamma_down: float,
    gamma_up: float,
    gamma_phi: float,
    rho0: object,
    time: float,
) -> ComplexMatrix:
    """Closed-form two-level T1/T2/thermal trajectory.

    ``T1^{-1} = gamma_down + gamma_up``,
    ``T2^{-1} = 1/(2 T1) + 2 gamma_phi``,
    ``p_e(t) = p_ss + (p_e(0) - p_ss) exp(-t/T1)`` with
    ``p_ss = gamma_up / (gamma_up + gamma_down)`` (or the initial
    population when both rates vanish),
    ``rho_01(t) = rho_01(0) exp(i omega t) exp(-t/T2)``.
    """
    rho = _as_square(rho0, name="rho0")
    if rho.shape != (2, 2):
        raise ValueError(f"qubit_bloch_solution requires a 2x2 rho0, got {rho.shape}")
    t = float(time)
    if not math.isfinite(t) or t < 0.0:
        raise ValueError(f"time must be finite and nonnegative, got {time!r}")
    w = float(omega)
    gd = float(gamma_down)
    gu = float(gamma_up)
    gp = float(gamma_phi)
    rate_sum = gd + gu
    p_e0 = float(rho[1, 1].real)
    if rate_sum > 0.0:
        p_ss = gu / rate_sum
        t1 = 1.0 / rate_sum
        p_e = p_ss + (p_e0 - p_ss) * math.exp(-t / t1)
        t2_inv = 0.5 / t1 + 2.0 * gp
    else:
        p_e = p_e0
        t2_inv = 2.0 * gp
    p_g = 1.0 - p_e
    decay = math.exp(-t * t2_inv)
    phase = cmath_exp_i(w * t)
    c01 = rho[0, 1] * phase * decay
    return np.array([[p_g, c01], [np.conj(c01), p_e]], dtype=np.complex128)


def cmath_exp_i(theta: float) -> complex:
    """``exp(i theta)`` via ``cos`` / ``sin`` (no complex ``exp`` branch)."""
    return complex(math.cos(theta), math.sin(theta))


__all__ = [
    "LindbladModel",
    "apply_lindblad",
    "density_matrix",
    "dissipative_gap",
    "honesty_payload",
    "liouvillian",
    "propagator",
    "qubit_bloch_solution",
    "qubit_pure_dephasing",
    "qubit_thermal",
    "steady_state",
    "thermal_steady_population",
    "time_derivative_tower",
]
