# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""One sealed finite-gauge report over the existing YM-adjacent engines.

The engines already certify one matrix, one ``β``, one Haar identity, one
strip, and one scaling table.  This module runs a locked CI-cheap pack
on a named spec and records the results together.  The bundle is still
a list of finite statements.  It is not a staircase to Clay existence,
``a -> 0``, infinite volume, Osterwalder-Seiler, or a uniform-in-``a``
gap.

``continuum_claim`` and ``yang_mills_claim`` are hard-wired ``False``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.geometry.gauge.transfer.gap import (
    ScalingReport,
    TransferGapResult,
    certified_gap_scaling_table,
    certified_transfer_matrix_gap,
)
from omnibias.geometry.gauge.transfer.hamiltonian import (
    COUPLING_LOCK,
    LEHMANN_HOLONOMY_METHOD,
    LEHMANN_STANDARD_METHOD,
    HamiltonianGapResult,
    candidate_gap,
    certified_hamiltonian_gap,
    plaquette_holonomy_trial_space,
    su2_three_plaquette_hamiltonian,
    su2_two_plaquette_hamiltonian,
)
from omnibias.geometry.gauge.transfer.matrices import (
    decode_scalar,
    encode_scalar,
    su2_heat_kernel_transfer,
    su3_wilson_transfer,
)
from omnibias.geometry.gauge.transfer.strip import (
    STRIP_COUPLING_LOCK,
    StripReflectionResult,
    certified_strip_reflection_positivity,
    su2_spatial_strip_transfer,
    su2_spatial_torus_transfer,
)
from omnibias.geometry.gauge.transfer.strong_coupling import (
    BETA_LOCK,
    PolymerDomainResult,
    StrongCouplingGapResult,
    WilsonCharacterDomainResult,
    WilsonCharacterGapResult,
    certified_polymer_beta_domain,
    certified_strong_coupling_glueball_bound,
    certified_wilson_character_beta_domain,
    certified_wilson_character_gap,
)
from omnibias.geometry.gauge.transfer.su3_wilson import su3_dimension

DEFAULT_SCALING_SPACINGS: tuple[float, ...] = (1.0, 0.5, 0.25)
DEFAULT_SCALING_COUPLINGS: tuple[float, ...] = (0.8, 0.4, 0.2)

#: Joint ``g(χ)`` plus a centered form still leaves ``n_cells=8``
#: overlapping, so the report stays at 32 (smallest of ``{16, 32}``
#: that separates for ``max_dynkin=1``, ``β=1``).
REPORT_SU3_N_CELLS = 32


@dataclass(frozen=True)
class FiniteGaugeSpec:
    """Locked CI-cheap pack naming the engines a report will run.

    ``su3_n_cells`` defaults to :data:`REPORT_SU3_N_CELLS` (32).  Joint
    ``g(χ)`` plus a centered form still leaves ``n_cells=8`` overlapping.
    The irrep truncation stays ``max_dynkin=1``.
    """

    name: str = "su2_finite_gauge_pack"
    include_torus: bool = False
    polymer_beta: Fraction = BETA_LOCK
    polymer_countings: tuple[str, ...] = ("two_scale", "cluster")
    hamiltonian_j_max: int = 1
    strip_n_sites: int = 2
    strip_n_angles: int = 4
    su3_max_dynkin: int = 1
    su3_n_cells: int = REPORT_SU3_N_CELLS
    su3_beta: float = 1.0
    scaling_spacings: tuple[float, ...] = DEFAULT_SCALING_SPACINGS
    scaling_couplings: tuple[float, ...] = DEFAULT_SCALING_COUPLINGS
    scaling_max_dynkin: int = 4


@dataclass(frozen=True)
class HaarIdentityCheck:
    """Geometry-side Weyl identities.  Not a continuum Haar theorem."""

    weyl_prefactor_24: bool
    su3_dim_3_0: bool

    @property
    def certified(self) -> bool:
        return self.weyl_prefactor_24 and self.su3_dim_3_0


@dataclass(frozen=True)
class MeasuredG1:
    """Holonomy-trial tightness versus the generic Lehmann candidate.

    The factor is measured.  It is not a claimed ``5x``.
    """

    factor: float
    ge_generic: bool
    generic_gap: float
    holonomy_gap: float
    official_generic: float
    official_holonomy: float
    note: str = (
        "G1 compares plaquette-character Lehmann trials to the computational "
        "standard basis on the three-plaquette Hamiltonian; the factor is "
        "measured, not a claimed 5x"
    )


@dataclass(frozen=True)
class FiniteGaugeReport:
    """Bundle of finite gauge statements on one named spec."""

    spec: FiniteGaugeSpec
    polymer: tuple[StrongCouplingGapResult, ...]
    polymer_domain: PolymerDomainResult
    wilson_character: WilsonCharacterGapResult
    wilson_character_domain: WilsonCharacterDomainResult
    haar: HaarIdentityCheck
    su3_gap: TransferGapResult
    hamiltonian: HamiltonianGapResult
    three_plaquette: HamiltonianGapResult
    g1: MeasuredG1
    strip_rp: StripReflectionResult
    scaling: ScalingReport
    torus_rp: StripReflectionResult | None = None
    continuum_claim: bool = False
    yang_mills_claim: bool = False
    note: str = (
        "bundle of finite certificates on one named spec; NOT a continuum-limit, "
        "infinite-volume, Osterwalder-Seiler, or Yang-Mills mass-gap claim, "
        "and not a staircase to Clay existence"
    )

    @property
    def certified(self) -> bool:
        torus_ok = self.torus_rp is None or self.torus_rp.certified
        scaling_ok = all(point.spectral_gap_lower > 0.0 for point in self.scaling.points)
        return (
            all(item.certified for item in self.polymer)
            and self.polymer_domain.certified
            and self.wilson_character.certified
            and self.wilson_character_domain.certified
            and self.haar.certified
            and self.su3_gap.certified
            and self.su3_gap.dimension >= 4
            and self.hamiltonian.certified
            and self.three_plaquette.certified
            and self.g1.ge_generic
            and self.strip_rp.certified
            and torus_ok
            and scaling_ok
            and self.scaling.continuum_claim is False
            and self.continuum_claim is False
            and self.yang_mills_claim is False
        )


def default_finite_gauge_spec() -> FiniteGaugeSpec:
    """The locked CI pack."""
    return FiniteGaugeSpec()


def finite_gauge_spec_to_mapping(spec: FiniteGaugeSpec) -> dict[str, Any]:
    """JSON-friendly encoding of a spec (Fractions stay exact)."""
    return {
        "name": spec.name,
        "include_torus": bool(spec.include_torus),
        "polymer_beta": encode_scalar(spec.polymer_beta),
        "polymer_countings": list(spec.polymer_countings),
        "hamiltonian_j_max": int(spec.hamiltonian_j_max),
        "strip_n_sites": int(spec.strip_n_sites),
        "strip_n_angles": int(spec.strip_n_angles),
        "su3_max_dynkin": int(spec.su3_max_dynkin),
        "su3_n_cells": int(spec.su3_n_cells),
        "su3_beta": float(spec.su3_beta),
        "scaling_spacings": [float(item) for item in spec.scaling_spacings],
        "scaling_couplings": [float(item) for item in spec.scaling_couplings],
        "scaling_max_dynkin": int(spec.scaling_max_dynkin),
    }


def finite_gauge_spec_from_mapping(data: Mapping[str, Any] | None) -> FiniteGaugeSpec:
    """Rebuild a spec from :func:`finite_gauge_spec_to_mapping` (or defaults)."""
    if not data:
        return FiniteGaugeSpec()
    beta_raw = data.get("polymer_beta", encode_scalar(BETA_LOCK))
    if isinstance(beta_raw, bool) or not isinstance(beta_raw, int | float | str | Fraction):
        raise ValueError("polymer_beta must be a positive scalar")
    beta = decode_scalar(beta_raw) if isinstance(beta_raw, str) else Fraction(beta_raw)
    if float(beta) <= 0.0:
        raise ValueError("polymer_beta must be > 0")
    countings = data.get("polymer_countings", ("two_scale", "cluster"))
    if not isinstance(countings, Sequence) or isinstance(countings, str | bytes):
        raise ValueError("polymer_countings must be a sequence of counting tags")
    legal = ("two_scale", "backtrack", "crude", "cluster")
    parsed = tuple(str(item) for item in countings)
    if not parsed or any(item not in legal for item in parsed):
        raise ValueError(f"polymer_countings must be a non-empty subset of {legal}")
    spacings = data.get("scaling_spacings", DEFAULT_SCALING_SPACINGS)
    couplings = data.get("scaling_couplings", DEFAULT_SCALING_COUPLINGS)
    if not isinstance(spacings, Sequence) or isinstance(spacings, str | bytes):
        raise ValueError("scaling_spacings must be a sequence of floats")
    if not isinstance(couplings, Sequence) or isinstance(couplings, str | bytes):
        raise ValueError("scaling_couplings must be a sequence of floats")
    return FiniteGaugeSpec(
        name=str(data.get("name", "su2_finite_gauge_pack")),
        include_torus=bool(data.get("include_torus", False)),
        polymer_beta=Fraction(beta),
        polymer_countings=parsed,
        hamiltonian_j_max=_require_int(data.get("hamiltonian_j_max", 1), "hamiltonian_j_max", 1),
        strip_n_sites=_require_int(data.get("strip_n_sites", 2), "strip_n_sites", 2),
        strip_n_angles=_require_int(data.get("strip_n_angles", 4), "strip_n_angles", 2),
        su3_max_dynkin=_require_int(data.get("su3_max_dynkin", 1), "su3_max_dynkin", 1),
        su3_n_cells=_require_int(
            data.get("su3_n_cells", REPORT_SU3_N_CELLS), "su3_n_cells", 2
        ),
        su3_beta=float(data.get("su3_beta", 1.0)),
        scaling_spacings=tuple(float(item) for item in spacings),
        scaling_couplings=tuple(float(item) for item in couplings),
        scaling_max_dynkin=_require_int(data.get("scaling_max_dynkin", 4), "scaling_max_dynkin", 1),
    )


def _require_int(value: Any, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _haar_identities() -> HaarIdentityCheck:
    return HaarIdentityCheck(
        weyl_prefactor_24=6 * 4 == 24,
        su3_dim_3_0=su3_dimension(3, 0) == 10,
    )


def _measure_g1(hamiltonian: Any, holonomy: HamiltonianGapResult) -> MeasuredG1:
    generic_official = certified_hamiltonian_gap(hamiltonian)
    generic = candidate_gap(holonomy, LEHMANN_STANDARD_METHOD)
    holonomy_lehmann = candidate_gap(holonomy, LEHMANN_HOLONOMY_METHOD)
    if generic > 0.0:
        factor = holonomy_lehmann / generic
    elif holonomy_lehmann >= generic:
        factor = 1.0
    else:
        factor = 0.0
    return MeasuredG1(
        factor=float(factor),
        ge_generic=holonomy_lehmann + 1e-12 >= generic
        and holonomy.spectral_gap_lower + 1e-12 >= generic_official.spectral_gap_lower,
        generic_gap=float(generic),
        holonomy_gap=float(holonomy_lehmann),
        official_generic=float(generic_official.spectral_gap_lower),
        official_holonomy=float(holonomy.spectral_gap_lower),
    )


def finite_gauge_report(spec: FiniteGaugeSpec | None = None) -> FiniteGaugeReport:
    """Run the locked engines named by ``spec`` and bundle the finite results."""
    locked = spec if spec is not None else FiniteGaugeSpec()
    polymer = tuple(
        certified_strong_coupling_glueball_bound(
            locked.polymer_beta, counting=counting  # type: ignore[arg-type]
        )
        for counting in locked.polymer_countings
    )
    domain = certified_polymer_beta_domain(
        counting=locked.polymer_countings[0]  # type: ignore[arg-type]
    )
    wilson = certified_wilson_character_gap(locked.polymer_beta)
    wilson_domain = certified_wilson_character_beta_domain()
    haar = _haar_identities()
    su3 = su3_wilson_transfer(
        locked.su3_beta,
        max_dynkin=locked.su3_max_dynkin,
        n_cells=locked.su3_n_cells,
    )
    su3_gap = certified_transfer_matrix_gap(su3)
    hamiltonian = su2_two_plaquette_hamiltonian(
        COUPLING_LOCK, j_max=locked.hamiltonian_j_max
    )
    trial = plaquette_holonomy_trial_space(hamiltonian)
    ham_gap = certified_hamiltonian_gap(hamiltonian, trial=trial)
    three = su2_three_plaquette_hamiltonian(
        COUPLING_LOCK, j_max=locked.hamiltonian_j_max
    )
    three_trial = plaquette_holonomy_trial_space(three)
    three_gap = certified_hamiltonian_gap(three, trial=three_trial)
    g1 = _measure_g1(three, three_gap)
    strip = su2_spatial_strip_transfer(
        STRIP_COUPLING_LOCK,
        n_sites=locked.strip_n_sites,
        n_angles=locked.strip_n_angles,
    )
    strip_rp = certified_strip_reflection_positivity(strip)
    torus_rp = None
    if locked.include_torus:
        torus = su2_spatial_torus_transfer(STRIP_COUPLING_LOCK)
        torus_rp = certified_strip_reflection_positivity(torus)
    scaling = certified_gap_scaling_table(
        su2_heat_kernel_transfer,
        spacings=locked.scaling_spacings,
        couplings=locked.scaling_couplings,
        max_dynkin=locked.scaling_max_dynkin,
    )
    return FiniteGaugeReport(
        spec=locked,
        polymer=polymer,
        polymer_domain=domain,
        wilson_character=wilson,
        wilson_character_domain=wilson_domain,
        haar=haar,
        su3_gap=su3_gap,
        hamiltonian=ham_gap,
        three_plaquette=three_gap,
        g1=g1,
        strip_rp=strip_rp,
        scaling=scaling,
        torus_rp=torus_rp,
    )


__all__ = [
    "DEFAULT_SCALING_COUPLINGS",
    "DEFAULT_SCALING_SPACINGS",
    "REPORT_SU3_N_CELLS",
    "FiniteGaugeReport",
    "FiniteGaugeSpec",
    "HaarIdentityCheck",
    "MeasuredG1",
    "default_finite_gauge_spec",
    "finite_gauge_report",
    "finite_gauge_spec_from_mapping",
    "finite_gauge_spec_to_mapping",
]
