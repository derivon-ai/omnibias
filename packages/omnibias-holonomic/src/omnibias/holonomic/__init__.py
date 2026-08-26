# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias-holonomic: a D-finite computer-algebra engine with Lean-certified identities.

Exact, dependency-free machinery for holonomic (D-finite / P-recursive) sequences and
functions:

* **Ore algebra** (:class:`OreAlgebra`, :class:`OrePolynomial`) -- the skew polynomial
  rings ``R[S; sigma]`` (shift) and ``R[D; delta]`` (differential) with non-commutative
  multiplication, a genuine right-Euclidean domain (:func:`ore_divmod`, :func:`gcrd`,
  :func:`lclm`, :func:`symmetric_product`), and the :class:`DFinite` / :class:`PRecursive`
  objects they annihilate -- closed under sum / Hadamard product **symbolically for all
  ``n``** (via ``lclm`` / ``symmetric_product``), with the verified-ansatz path as a
  labelled fallback;
* **Gosper's algorithm** (:func:`gosper_sum`, :func:`gosper_definite_sum`) -- unconditional,
  closed-form indefinite / definite hypergeometric summation with an exact certificate;
* **true Zeilberger + WZ** (:func:`zeilberger`, :class:`ZeilbergerCertificate`,
  :func:`wz_pair`, :func:`wz_certificate`) -- exact creative telescoping (a telescoper ``L``
  plus a rational cofactor ``R(n, k)``) solved by an exact null space, needing no guesser
  and handling degenerate sums natively; the guessed-then-range-verified
  :func:`creative_telescoping` is kept as the fast path;
* **Petkovsek's Hyper** (:func:`hyper`) -- all hypergeometric-term solutions of a shift
  recurrence, on the scoped rational-root / linear :mod:`~omnibias.holonomic._core.factor`
  substrate (:func:`rational_roots`, :func:`square_free`);
* **q-holonomic** (:func:`q_shift_algebra`, :func:`q_gosper`, :func:`q_zeilberger`) --
  the q-analogues on :mod:`omnibias.qcalculus` primitives, with exact q-rational
  certificates and the ``q -> 1`` distinct-limit framing;
* **transforms & closures** (:func:`dfinite_to_precursive`, :func:`precursive_to_dfinite`,
  :func:`dfinite_derivative`, :func:`dfinite_integral`, :func:`dfinite_compose_poly`) --
  the exact ODE <-> coefficient-recurrence bridge and the D-finite closure operations;
* **guessing** (:func:`guess_recurrence`, :func:`guess_dfinite`, :func:`guess_algebraic`) --
  minimal P-recursive / differential / algebraic annihilators, guessed by exact null space
  and verified on held-out terms;
* **asymptotics** (:func:`precursive_asymptotics`, :func:`certified_asymptotic`) -- the
  Poincare-Perron leading rate / exponent (numerical), bridged to
  :func:`omnibias.difference.transfer_theorem` for a certified coefficient where the
  singularity is known;
* **Lean-certified identities** (:func:`prove_hypergeometric_identity`,
  :func:`prove_identity_zeilberger`) -- classic binomial identities discharged, per
  coefficient, as ``rational_identity`` obligations the omnibias Lean kernel checks; the
  Zeilberger path emits ``P(n, k) == 0`` obligations that hold for **all** ``n``, so
  ``theorem_prover_verified`` is earned **only** on a genuine ``lake`` pass and never
  forged.

Honesty: operator algebra, Gosper / Zeilberger sums, transforms, and the certificate
payloads are exact / ``closed-form``; the *guessing* step (which annihilator fits) is
heuristic and labelled, the verification and the Lean obligations are rigorous;
:mod:`~omnibias.holonomic._core.factor` is scoped to the rational-root / linear regime and
:mod:`~omnibias.holonomic._core.asymptotics` returns a ``numerical`` leading term. It builds
on :mod:`omnibias.difference` (probe harness + transfer theorem), :mod:`omnibias.qcalculus`
(q-primitives), :mod:`omnibias.symbolic` (recurrence guessing), and the
:mod:`omnibias.core.proof` Lean loop.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.holonomic._core import (
    AsymptoticEstimate,
    DFinite,
    GosperResult,
    HolonomicProof,
    OreAlgebra,
    OreDivision,
    OrePolynomial,
    Poly,
    PolyN,
    PRecursive,
    ProperTerm,
    QGosperResult,
    QRecurrence,
    Summand,
    Telescoper,
    ZeilbergerCertificate,
    binomial_nk,
    certified_asymptotic,
    creative_telescoping,
    dfinite_add,
    dfinite_cauchy,
    dfinite_compose_poly,
    dfinite_derivative,
    dfinite_hadamard,
    dfinite_integral,
    dfinite_to_precursive,
    diff_algebra,
    dispersion_set,
    empirical_rate,
    gcrd,
    geometric_k,
    gosper_definite_sum,
    gosper_normal_form,
    gosper_sum,
    guess_algebraic,
    guess_dfinite,
    guess_recurrence,
    hyper,
    jacobian_det,
    lclm,
    ore_divmod,
    peval,
    pgcd,
    precursive_asymptotics,
    precursive_to_dfinite,
    prove_hypergeometric_identity,
    prove_identity_zeilberger,
    q_from_p,
    q_gosper,
    q_gosper_definite_sum,
    q_gosper_normal_form,
    q_shift_algebra,
    q_zeilberger,
    rational_roots,
    recurrence_to_operator,
    roots_with_multiplicity,
    shift_algebra,
    square_free,
    summand_sum,
    sylvester_resultant,
    symmetric_product,
    term_ratio_annihilates,
    to_poly,
    wz_certificate,
    wz_pair,
    zeilberger,
)
from omnibias.holonomic._core.export import AnnihilatorExport, export_annihilator
from omnibias.holonomic._core.layer import HolonomicLayerSpec, fit_holonomic_layer, holonomic_jet
from omnibias.holonomic.jacobian_n2 import (
    JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED,
    JacobianN2DegreeFamily,
    JacobianN2HomogeneousFamily,
    escalate_n2_result,
    jacobian_n2_box_statement,
    jacobian_n2_homog_statement,
    jacobian_n2_honesty,
    reject_jacobian_proof_claim,
    seal_jacobian_honesty,
)
from omnibias.holonomic.jacobian_n2_case_a import (
    CASE_A_B31_KIND,
    case_a_b31_statement,
    replay_case_a_b31,
    seal_case_a_b31,
)
from omnibias.holonomic.jacobian_n2_inverse import gabber_n2_test
from omnibias.holonomic.jacobian_n2_normalize import (
    classify_leftover_chart,
    o1_normalize,
    replay_leftover_certificate,
    seal_leftover_certificate,
)
from omnibias.holonomic.keller import (
    alpoge_map,
    fiber_report,
    gallagher_map,
    search_tangent_sweep,
    verify_alpoge_map,
    verify_gallagher_map,
)
from omnibias.holonomic.proofmachine import build_holonomic_machine
from omnibias.holonomic.rank_syzygy import (
    HOLONOMIC_SYZYGY,
    certify_holonomic_syzygy,
    integerize_matrix,
)

try:
    __version__ = _pkg_version("omnibias-holonomic")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "exempt: D-finite Ore algebra"

__all__ = [
    "AnnihilatorExport",
    "AsymptoticEstimate",
    "CASE_A_B31_KIND",
    "DFinite",
    "GosperResult",
    "HOLONOMIC_SYZYGY",
    "HolonomicLayerSpec",
    "HolonomicProof",
    "JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED",
    "JacobianN2DegreeFamily",
    "JacobianN2HomogeneousFamily",
    "OreAlgebra",
    "OreDivision",
    "OrePolynomial",
    "PRecursive",
    "Poly",
    "PolyN",
    "ProperTerm",
    "QGosperResult",
    "QRecurrence",
    "Summand",
    "Telescoper",
    "ZeilbergerCertificate",
    "__lineage__",
    "__version__",
    "alpoge_map",
    "binomial_nk",
    "build_holonomic_machine",
    "case_a_b31_statement",
    "certified_asymptotic",
    "certify_holonomic_syzygy",
    "classify_leftover_chart",
    "creative_telescoping",
    "dfinite_add",
    "dfinite_cauchy",
    "dfinite_compose_poly",
    "dfinite_derivative",
    "dfinite_hadamard",
    "dfinite_integral",
    "dfinite_to_precursive",
    "diff_algebra",
    "dispersion_set",
    "empirical_rate",
    "escalate_n2_result",
    "export_annihilator",
    "fiber_report",
    "fit_holonomic_layer",
    "gabber_n2_test",
    "gallagher_map",
    "gcrd",
    "geometric_k",
    "gosper_definite_sum",
    "gosper_normal_form",
    "gosper_sum",
    "guess_algebraic",
    "guess_dfinite",
    "guess_recurrence",
    "holonomic_jet",
    "hyper",
    "integerize_matrix",
    "jacobian_det",
    "jacobian_n2_box_statement",
    "jacobian_n2_homog_statement",
    "jacobian_n2_honesty",
    "lclm",
    "o1_normalize",
    "ore_divmod",
    "peval",
    "pgcd",
    "precursive_asymptotics",
    "precursive_to_dfinite",
    "prove_hypergeometric_identity",
    "prove_identity_zeilberger",
    "q_from_p",
    "q_gosper",
    "q_gosper_definite_sum",
    "q_gosper_normal_form",
    "q_shift_algebra",
    "q_zeilberger",
    "rational_roots",
    "recurrence_to_operator",
    "reject_jacobian_proof_claim",
    "replay_case_a_b31",
    "replay_leftover_certificate",
    "roots_with_multiplicity",
    "seal_case_a_b31",
    "seal_jacobian_honesty",
    "seal_leftover_certificate",
    "search_tangent_sweep",
    "shift_algebra",
    "square_free",
    "summand_sum",
    "sylvester_resultant",
    "symmetric_product",
    "term_ratio_annihilates",
    "to_poly",
    "verify_alpoge_map",
    "verify_gallagher_map",
    "wz_certificate",
    "wz_pair",
    "zeilberger",
]
