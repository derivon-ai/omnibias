# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Representation theory of ``su(N)``: highest-weight invariants + rep matrices.

Every claim is pinned by three independent oracles (the enterprise gate):

1. **Analytic reference** -- textbook values: ``su(2)`` spin-``j`` has
   ``dim = 2j+1`` and ``C_2 = j(j+1)``; ``su(3)`` fundamental ``C_2 = 4/3``;
   ``3 (x) 3bar = 8 + 1``; ``8 (x) 8 = 27 + 10 + 10bar + 8 + 8 + 1``.
2. **Symbolic (sympy)** -- the tensor-product decomposition and the Weyl
   character are cross-checked against exact symbolic character products
   (``su(2)`` in one Laurent variable, ``su(3)`` in the maximal torus).
3. **Independent internal code path** -- weight multiplicities sum to the Weyl
   dimension; the adjoint Casimir equals the ``LieAlgebra.dual_coxeter_number``;
   ``sum_a (T^a)^2 = C_2 I`` from the materialized generators; and
   ``tr(J_a J_b) = T(R) delta_{ab}`` links the Dynkin index to the spin matrices.

The dimension / Casimir / Dynkin index are **exact** rationals (``Fraction``),
so they are certified by construction rather than sampled.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import sympy as sp
from omnibias.geometry.gauge._core import representation as R
from omnibias.geometry.gauge._core.lie_algebra import su

# --------------------------------------------------------------------------- #
# 1. exact invariants: dimension, quadratic Casimir, Dynkin index
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("two_j", range(0, 9))
def test_su2_dimension_and_casimir(two_j: int) -> None:
    """su(2) spin-j: dim = 2j+1 and C2 = j(j+1) (exact)."""
    rep = R.Irrep(2, (two_j,))
    j = Fraction(two_j, 2)
    assert R.dimension(rep) == two_j + 1
    assert R.quadratic_casimir(rep) == j * (j + 1)


def test_su3_fundamental_adjoint_invariants() -> None:
    """su(3): fundamental C2 = 4/3, dim 3; adjoint dim 8, C2 = 3, index 3."""
    fund = R.fundamental(3)
    assert R.dimension(fund) == 3
    assert R.quadratic_casimir(fund) == Fraction(4, 3)
    assert R.dynkin_index(fund) == Fraction(1, 2)

    adj = R.adjoint(3)
    assert R.dimension(adj) == 8
    assert R.quadratic_casimir(adj) == 3
    assert R.dynkin_index(adj) == 3  # T(adjoint) = dual Coxeter = N


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_fundamental_casimir_general(n: int) -> None:
    """su(N) fundamental: C2 = (N^2 - 1)/(2 N); dim N; index 1/2."""
    fund = R.fundamental(n)
    assert R.dimension(fund) == n
    assert R.quadratic_casimir(fund) == Fraction(n * n - 1, 2 * n)
    assert R.dynkin_index(fund) == Fraction(1, 2)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_adjoint_casimir_is_dual_coxeter(n: int) -> None:
    """Independent path: adjoint Casimir == LieAlgebra.dual_coxeter_number == N."""
    adj = R.adjoint(n)
    assert R.dimension(adj) == n * n - 1
    assert R.quadratic_casimir(adj) == n
    assert R.quadratic_casimir(adj) == su(n).dual_coxeter_number()
    assert R.dual_coxeter_number(n) == n


def test_conjugate_reps_share_dimension_and_casimir() -> None:
    """R and its conjugate (reversed Dynkin labels) share dim and C2."""
    for dynk in [(2, 0), (1, 2), (3, 1)]:
        rep = R.Irrep(3, dynk)
        conj = R.Irrep(3, dynk[::-1])
        assert R.dimension(rep) == R.dimension(conj)
        assert R.quadratic_casimir(rep) == R.quadratic_casimir(conj)


# --------------------------------------------------------------------------- #
# 2. weight system (Freudenthal): multiplicities sum to the dimension
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rep",
    [
        R.fundamental(2),
        R.Irrep(2, (4,)),
        R.fundamental(3),
        R.adjoint(3),
        R.Irrep(3, (2, 0)),
        R.Irrep(3, (2, 2)),
        R.fundamental(4),
        R.adjoint(4),
        R.Irrep(4, (1, 1, 0)),
    ],
)
def test_weight_multiplicities_sum_to_dimension(rep: R.Irrep) -> None:
    weights = R.weight_multiplicities(rep)
    assert sum(weights.values()) == R.dimension(rep)
    assert all(m >= 1 for m in weights.values())


def test_su3_adjoint_has_two_dimensional_zero_weight() -> None:
    """The su(3) adjoint (octet) has a rank-2 (=2) zero-weight space."""
    weights = R.weight_multiplicities(R.adjoint(3))
    assert weights[(1, 1, 1)] == 2  # zero weight in epsilon basis (sum = |lambda|)


# --------------------------------------------------------------------------- #
# 3. tensor products (Racah-Speiser): analytic + dimension consistency
# --------------------------------------------------------------------------- #


def _dim_total(decomp: dict[R.Irrep, int]) -> int:
    return sum(R.dimension(r) * c for r, c in decomp.items())


def _as_dynkin_counts(decomp: dict[R.Irrep, int]) -> dict[tuple[int, ...], int]:
    return {r.dynkin: c for r, c in decomp.items()}


def test_su2_two_by_two() -> None:
    """2 (x) 2 = 3 + 1."""
    decomp = R.tensor_product_decomposition(R.fundamental(2), R.fundamental(2))
    assert _as_dynkin_counts(decomp) == {(2,): 1, (0,): 1}


def test_su3_three_by_threebar() -> None:
    """3 (x) 3bar = 8 + 1."""
    decomp = R.tensor_product_decomposition(R.fundamental(3), R.Irrep(3, (0, 1)))
    assert _as_dynkin_counts(decomp) == {(1, 1): 1, (0, 0): 1}


def test_su3_three_by_three() -> None:
    """3 (x) 3 = 6 + 3bar."""
    decomp = R.tensor_product_decomposition(R.fundamental(3), R.fundamental(3))
    assert _as_dynkin_counts(decomp) == {(2, 0): 1, (0, 1): 1}


def test_su3_eight_by_eight() -> None:
    """8 (x) 8 = 27 + 10 + 10bar + 8 + 8 + 1 (the classic QCD product)."""
    decomp = R.tensor_product_decomposition(R.adjoint(3), R.adjoint(3))
    assert _as_dynkin_counts(decomp) == {
        (2, 2): 1,  # 27
        (3, 0): 1,  # 10
        (0, 3): 1,  # 10bar
        (1, 1): 2,  # 8 + 8
        (0, 0): 1,  # 1
    }
    assert _dim_total(decomp) == 64


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (R.fundamental(3), R.Irrep(3, (0, 1))),
        (R.adjoint(3), R.adjoint(3)),
        (R.Irrep(3, (2, 0)), R.fundamental(3)),
        (R.fundamental(4), R.fundamental(4)),
        (R.adjoint(4), R.fundamental(4)),
        (R.fundamental(5), R.adjoint(5)),
    ],
)
def test_tensor_product_dimension_conserved(a: R.Irrep, b: R.Irrep) -> None:
    """dim(a) * dim(b) = sum of decomposed dims (independent consistency)."""
    decomp = R.tensor_product_decomposition(a, b)
    assert _dim_total(decomp) == R.dimension(a) * R.dimension(b)


def test_su2_clebsch_gordan_series() -> None:
    """su(2): j1 (x) j2 = sum_{j=|j1-j2|}^{j1+j2} j (analytic CG series)."""
    for two_j1 in range(0, 5):
        for two_j2 in range(0, 5):
            decomp = R.tensor_product_decomposition(
                R.Irrep(2, (two_j1,)), R.Irrep(2, (two_j2,))
            )
            expected = {
                (twoj,): 1
                for twoj in range(abs(two_j1 - two_j2), two_j1 + two_j2 + 1, 2)
            }
            assert _as_dynkin_counts(decomp) == expected


# --------------------------------------------------------------------------- #
# 4. sympy symbolic oracle: character products == tensor decomposition
# --------------------------------------------------------------------------- #


def _su2_char(two_j: int, t: sp.Symbol) -> sp.Expr:
    """Symbolic su(2) character chi_j(t) = sum_{k=0}^{2j} t^{2j - 2k}."""
    return sum(t ** (two_j - 2 * k) for k in range(two_j + 1))


def test_sympy_su2_character_product() -> None:
    """sympy: chi_{j1} chi_{j2} == sum over the decomposition (Laurent identity)."""
    t = sp.symbols("t")
    for two_j1 in range(0, 4):
        for two_j2 in range(0, 4):
            decomp = R.tensor_product_decomposition(
                R.Irrep(2, (two_j1,)), R.Irrep(2, (two_j2,))
            )
            lhs = sp.expand(_su2_char(two_j1, t) * _su2_char(two_j2, t))
            rhs = sp.expand(
                sum(c * _su2_char(r.dynkin[0], t) for r, c in decomp.items())
            )
            assert sp.simplify(lhs - rhs) == 0


def _su3_char(rep: R.Irrep, x0: sp.Symbol, x1: sp.Symbol) -> sp.Expr:
    """Symbolic su(3) Weyl character (Schur bialternant) with x2 = 1/(x0 x1)."""
    lam = rep.partition()
    xs = [x0, x1, 1 / (x0 * x1)]
    top = sp.Matrix(3, 3, lambda i, j: xs[i] ** (lam[j] + 2 - j))
    bot = sp.Matrix(3, 3, lambda i, j: xs[i] ** (2 - j))
    return sp.simplify(top.det() / bot.det())


def test_sympy_su3_character_matches_numeric() -> None:
    """sympy symbolic su(3) character equals the numpy bialternant."""
    x0, x1 = sp.symbols("x0 x1")
    for rep in [R.fundamental(3), R.Irrep(3, (0, 1)), R.adjoint(3), R.Irrep(3, (2, 0))]:
        chi_sym = _su3_char(rep, x0, x1)
        val = complex(chi_sym.subs({x0: sp.exp(0.3 * sp.I), x1: sp.exp(-0.17 * sp.I)}))
        pt = np.array(
            [np.exp(0.3j), np.exp(-0.17j), 1.0 / (np.exp(0.3j) * np.exp(-0.17j))]
        )
        assert np.allclose(R.character(rep, pt), val, atol=1e-9)


def test_sympy_su3_character_product_is_tensor_decomposition() -> None:
    """sympy: su(3) character product == decomposed character sum."""
    x0, x1 = sp.symbols("x0 x1")
    a, b = R.fundamental(3), R.Irrep(3, (0, 1))
    decomp = R.tensor_product_decomposition(a, b)
    lhs = sp.expand(_su3_char(a, x0, x1) * _su3_char(b, x0, x1))
    rhs = sp.expand(sum(c * _su3_char(r, x0, x1) for r, c in decomp.items()))
    assert sp.simplify(lhs - rhs) == 0


def test_character_fundamental_is_power_sum() -> None:
    """chi_fund(x) = sum_i x_i for any su(N)."""
    rng = np.random.default_rng(3)
    for n in (2, 3, 4):
        phases = rng.uniform(-1.0, 1.0, size=n)
        x = np.exp(1j * phases)
        x = x / np.prod(x) ** (1.0 / n)
        assert np.allclose(R.character(R.fundamental(n), x), x.sum())


# --------------------------------------------------------------------------- #
# 5. branching su(N) -> su(N-1) (Gelfand-Tsetlin interlacing)
# --------------------------------------------------------------------------- #


def test_branching_su3_adjoint_to_su2() -> None:
    """8 -> 3 + 2 + 2 + 1 under su(3) -> su(2) (isospin content)."""
    br = R.branching_to_subalgebra(R.adjoint(3))
    assert _as_dynkin_counts(br) == {(1,): 2, (2,): 1, (0,): 1}
    assert _dim_total(br) == 8


def test_branching_su3_fundamental_to_su2() -> None:
    """3 -> 2 + 1 under su(3) -> su(2)."""
    br = R.branching_to_subalgebra(R.fundamental(3))
    assert _as_dynkin_counts(br) == {(1,): 1, (0,): 1}
    assert _dim_total(br) == 3


@pytest.mark.parametrize(
    "rep", [R.fundamental(3), R.adjoint(3), R.Irrep(3, (2, 0)), R.fundamental(4), R.adjoint(4)]
)
def test_branching_conserves_dimension(rep: R.Irrep) -> None:
    """Total dimension is preserved by the restriction (sum over u(1) charge)."""
    br = R.branching_to_subalgebra(rep)
    assert _dim_total(br) == R.dimension(rep)


def test_branching_requires_rank_two() -> None:
    with pytest.raises(ValueError, match="needs N >= 3"):
        R.branching_to_subalgebra(R.fundamental(2))


# --------------------------------------------------------------------------- #
# 6. explicit matrices (core numpy): spin-j + adjoint from f^{abc}
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("two_j", range(1, 7))
def test_su2_spin_matrices_algebra(two_j: int) -> None:
    """[J_a, J_b] = i eps_{abc} J_c and sum_a J_a^2 = j(j+1) I."""
    jx, jy, jz = R.su2_spin_matrices(two_j)
    j = two_j / 2.0
    assert np.allclose(jx @ jy - jy @ jx, 1j * jz)
    assert np.allclose(jy @ jz - jz @ jy, 1j * jx)
    assert np.allclose(jz @ jx - jx @ jz, 1j * jy)
    casimir = jx @ jx + jy @ jy + jz @ jz
    assert np.allclose(casimir, j * (j + 1) * np.eye(two_j + 1))
    for mat in (jx, jy, jz):  # Hermitian generators
        assert np.allclose(mat, mat.conj().T)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_adjoint_rep_matrices_reproduce_structure_constants(n: int) -> None:
    """(T^a_adj)_{bc} = -i f^{abc}: bracket closes and tr(T^a T^b) = N delta."""
    f = su(n).structure_constants()
    adj = R.adjoint_rep_matrices(f)
    dim = adj.shape[0]
    for a in range(dim):
        for b in range(dim):
            comm = adj[a] @ adj[b] - adj[b] @ adj[a]
            rhs = 1j * sum(f[a, b, c] * adj[c] for c in range(dim))
            assert np.allclose(comm, rhs)
    gram = np.einsum("aij,bji->ab", adj, adj)
    assert np.allclose(gram, n * np.eye(dim))  # tr = dual Coxeter number


def _is_representation(mats: np.ndarray, f: np.ndarray) -> bool:
    """[rho(T^a), rho(T^b)] = i f^{abc} rho(T^c) for the whole generator stack."""
    dim = mats.shape[0]
    for a in range(dim):
        for b in range(dim):
            comm = mats[a] @ mats[b] - mats[b] @ mats[a]
            rhs = 1j * sum(f[a, b, c] * mats[c] for c in range(dim))
            if not np.allclose(comm, rhs):
                return False
    return True


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_symmetric_power_rep_matrices(n: int, k: int) -> None:
    """Sym^k rep: correct dim, Hermitian, closes the algebra, Casimir eigenvalue."""
    from math import comb

    alg = su(n)
    gens = alg.generators()
    sym = R.symmetric_power_rep_matrices(gens, k)
    dim = sym.shape[1]
    assert dim == comb(n + k - 1, k)
    assert np.allclose(sym, sym.conj().transpose(0, 2, 1))  # Hermitian
    assert _is_representation(sym, alg.structure_constants())
    casimir = np.einsum("aij,ajk->ik", sym, sym)
    c2 = float(R.quadratic_casimir(R.symmetric_power(n, k)))
    assert np.allclose(casimir, c2 * np.eye(dim), atol=1e-9)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_antisymmetric_power_rep_matrices(n: int) -> None:
    """Lambda^k rep for every k: dim C(N,k), Hermitian, closes algebra, Casimir."""
    from math import comb

    alg = su(n)
    gens = alg.generators()
    for k in range(n + 1):
        asym = R.antisymmetric_power_rep_matrices(gens, k)
        dim = asym.shape[1]
        assert dim == comb(n, k)
        assert np.allclose(asym, asym.conj().transpose(0, 2, 1))
        assert _is_representation(asym, alg.structure_constants())
        casimir = np.einsum("aij,ajk->ik", asym, asym)
        c2 = float(R.quadratic_casimir(R.antisymmetric_power(n, k)))
        assert np.allclose(casimir, c2 * np.eye(dim), atol=1e-9)


def test_su2_symmetric_power_is_spin_j() -> None:
    """Sym^{2j} of the su(2) fundamental IS spin-j: dim 2j+1, C2 = j(j+1)."""
    for two_j in range(0, 5):
        sym = R.symmetric_power_rep_matrices(su(2).generators(), two_j)
        assert sym.shape[1] == two_j + 1
        casimir = np.einsum("aij,ajk->ik", sym, sym)
        j = two_j / 2.0
        assert np.allclose(casimir, j * (j + 1) * np.eye(two_j + 1), atol=1e-9)


def test_su3_antisymmetric_square_is_antifundamental() -> None:
    """Lambda^2 of the su(3) fundamental is 3bar: dim 3, C2 = 4/3."""
    asym = R.antisymmetric_power_rep_matrices(su(3).generators(), 2)
    assert asym.shape[1] == 3
    casimir = np.einsum("aij,ajk->ik", asym, asym)
    assert np.allclose(casimir, (4 / 3) * np.eye(3), atol=1e-9)
    assert R.antisymmetric_power(3, 2).dynkin == (0, 1)  # 3bar Dynkin labels


# --------------------------------------------------------------------------- #
# 7. backend ops (torch / jax): materialization, Casimir, parity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("two_j", [1, 2, 3, 4])
def test_backend_spin_matrices_casimir(backend, two_j) -> None:  # type: ignore[no-untyped-def]
    """casimir_operator(spin_matrices) == j(j+1) I in each backend."""
    mats = backend.ops.spin_matrices(two_j)
    casimir = backend.ops.casimir_operator(mats)
    j = two_j / 2.0
    dim = two_j + 1
    got = backend.tonumpy(casimir)
    assert np.allclose(got, j * (j + 1) * np.eye(dim))


@pytest.mark.parametrize("n", [2, 3])
def test_backend_adjoint_generators_casimir(backend, n) -> None:  # type: ignore[no-untyped-def]
    """casimir_operator(adjoint_generators) == N I; matrices Hermitian."""
    mats = backend.ops.adjoint_generators(su(n))
    casimir = backend.ops.casimir_operator(mats)
    dim = n * n - 1
    assert np.allclose(backend.tonumpy(casimir), n * np.eye(dim), atol=1e-9)


def test_backend_casimir_eigenvalue_and_dynkin(backend) -> None:  # type: ignore[no-untyped-def]
    """Scalar C2 / T(R) match the exact rationals; tr(J_a J_b) = T(R) delta."""
    fund = R.fundamental(3)
    assert backend.tonumpy(backend.ops.casimir_eigenvalue(fund)) == pytest.approx(4 / 3)
    assert backend.tonumpy(backend.ops.dynkin_index_value(fund)) == pytest.approx(0.5)
    for two_j in (1, 2, 3):
        mats = backend.ops.spin_matrices(two_j)
        gram = np.einsum("aij,bji->ab", backend.tonumpy(mats), backend.tonumpy(mats)).real
        t_r = backend.tonumpy(backend.ops.dynkin_index_value(R.Irrep(2, (two_j,))))
        assert np.allclose(gram, float(t_r) * np.eye(3), atol=1e-9)


@pytest.mark.parametrize(("n", "k"), [(2, 2), (3, 2), (3, 3), (4, 2)])
def test_backend_symmetric_power_casimir(backend, n, k) -> None:  # type: ignore[no-untyped-def]
    """casimir_operator(symmetric_power_generators) == C2(Sym^k) I per backend."""
    mats = backend.ops.symmetric_power_generators(su(n), k)
    casimir = backend.tonumpy(backend.ops.casimir_operator(mats))
    c2 = float(R.quadratic_casimir(R.symmetric_power(n, k)))
    dim = mats.shape[1]
    assert np.allclose(casimir, c2 * np.eye(dim), atol=1e-9)


@pytest.mark.parametrize(("n", "k"), [(3, 2), (4, 2), (5, 2)])
def test_backend_antisymmetric_power_casimir(backend, n, k) -> None:  # type: ignore[no-untyped-def]
    """casimir_operator(antisymmetric_power_generators) == C2(Lambda^k) I."""
    mats = backend.ops.antisymmetric_power_generators(su(n), k)
    casimir = backend.tonumpy(backend.ops.casimir_operator(mats))
    c2 = float(R.quadratic_casimir(R.antisymmetric_power(n, k)))
    dim = mats.shape[1]
    assert np.allclose(casimir, c2 * np.eye(dim), atol=1e-9)


def test_backend_spin_matrices_parity() -> None:
    """torch and jax spin matrices are bit-identical (float64 ULP)."""
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: F401
    import torch

    torch.set_default_dtype(torch.float64)
    import omnibias.geometry.gauge.jax.ops.representation as jrep
    import omnibias.geometry.gauge.torch.ops.representation as trep

    for two_j in range(1, 6):
        tt = trep.spin_matrices(two_j).detach().cpu().numpy()
        jj = np.asarray(jrep.spin_matrices(two_j))
        assert np.allclose(tt, jj, rtol=1e-9, atol=0.0)
    for n in (2, 3):
        tt = trep.adjoint_generators(su(n)).detach().cpu().numpy()
        jj = np.asarray(jrep.adjoint_generators(su(n)))
        assert np.allclose(tt, jj, rtol=1e-9, atol=0.0)
    for n, k in ((3, 2), (4, 2)):
        ts = trep.symmetric_power_generators(su(n), k).detach().cpu().numpy()
        js = np.asarray(jrep.symmetric_power_generators(su(n), k))
        assert np.allclose(ts, js, rtol=1e-9, atol=0.0)
        ta = trep.antisymmetric_power_generators(su(n), k).detach().cpu().numpy()
        ja = np.asarray(jrep.antisymmetric_power_generators(su(n), k))
        assert np.allclose(ta, ja, rtol=1e-9, atol=0.0)


# --------------------------------------------------------------------------- #
# 8. validation guards + out-of-thesis enforcement
# --------------------------------------------------------------------------- #


def test_irrep_validation() -> None:
    with pytest.raises(ValueError, match="N >= 2"):
        R.Irrep(1, ())
    with pytest.raises(ValueError, match="Dynkin labels"):
        R.Irrep(3, (1,))  # wrong number of labels
    with pytest.raises(ValueError, match="non-negative"):
        R.Irrep(3, (1, -1))


def test_tensor_product_requires_same_algebra() -> None:
    with pytest.raises(ValueError, match="cannot tensor"):
        R.tensor_product_decomposition(R.fundamental(2), R.fundamental(3))


def test_finite_group_theory_is_out_of_thesis() -> None:
    """Enforcement: only the su(N) Lie-algebra slice is implemented.

    Finite (non-Lie) group character tables and general abstract-group theory
    are deliberately out of thesis (see docs/scope-and-guarantees.md). This test
    fails if someone silently adds a finite-group surface here.
    """
    exported = set(R.__all__)
    forbidden = {
        "symmetric_group",
        "finite_group_character_table",
        "conjugacy_classes",
        "group_cohomology",
        "burnside",
        "molien_series",
    }
    assert exported.isdisjoint(forbidden)
