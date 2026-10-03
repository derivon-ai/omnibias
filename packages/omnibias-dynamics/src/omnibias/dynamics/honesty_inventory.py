# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Catalog of Hilbert-16 honesty flags and why each stays false.

Flat ``False`` booleans in certificate payloads are ambiguous: some mean
"not implemented yet", others mean "structurally outside this tool class".
This module records the distinction and attaches ``_honesty_reasons`` to
payloads so downstream readers do not treat the two as identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HonestyReason = Literal["unimplemented", "structural_limit", "gate_open", "derived_false"]

__all__ = [
    "HONESTY_INVENTORY",
    "HonestyFlagSpec",
    "HonestyReason",
    "build_honesty",
    "inventory_report",
]


@dataclass(frozen=True)
class HonestyFlagSpec:
    reason: HonestyReason
    detail: str


HONESTY_INVENTORY: dict[str, HonestyFlagSpec] = {
    "full_hilbert16_solved": HonestyFlagSpec(
        "derived_false",
        "Re-derived from H16Obligation entries; never stamped by hand.",
    ),
    "hilbert16_part_a_solved": HonestyFlagSpec(
        "derived_false",
        "Part-A 22-oval target has no witness; derived from ledger entries.",
    ),
    "hilbert16_part_b_quadratic_solved": HonestyFlagSpec(
        "derived_false",
        "G1-G6 and DRR cases remain open; derived from ledger entries.",
    ),
    "g1_passed": HonestyFlagSpec(
        "gate_open",
        "Uniform confluent passage gate; route-specific negatives do not discharge it.",
    ),
    "g4_passed": HonestyFlagSpec(
        "gate_open",
        "Bounded-format / effective-zero gate; not earned by finite replay alone.",
    ),
    "physical_return_membership_proved": HonestyFlagSpec(
        "unimplemented",
        "Needs corner-window closure plus uniform remainder consumption.",
    ),
    "collar_return_membership_proved": HonestyFlagSpec(
        "unimplemented",
        "Earned only by certify_collar_membership on a positive collar.",
    ),
    "corner_window_external": HonestyFlagSpec(
        "unimplemented",
        "Corner limit x->0 not covered until a collar sequence closes.",
    ),
    "uniform_remainder_proved": HonestyFlagSpec(
        "unimplemented",
        "Remainder bound must be derived and consumed as a premise.",
    ),
    "graphic_finite_cyclicity_proved": HonestyFlagSpec(
        "structural_limit",
        "Uniform graphic cyclicity is not a finite rational obligation.",
    ),
    "drr_case_closed": HonestyFlagSpec(
        "gate_open",
        "Each DRR case is a separate dated obligation.",
    ),
    "bautin_ideal_stabilization_proved": HonestyFlagSpec(
        "unimplemented",
        "Finite-order stabilization can be verified; no all-orders recurrence or termination theorem is supplied.",
    ),
    "finite_order_stabilization_verified": HonestyFlagSpec(
        "unimplemented",
        "True only when every computed higher focal value has an exact ideal-membership witness.",
    ),
    "finite_jet_all_orders_barrier": HonestyFlagSpec(
        "structural_limit",
        "A finite focal prefix has formal continuations both inside and outside the current ideal; a field-derived all-orders recurrence is required.",
    ),
    "all_orders_focal_recurrence_proved": HonestyFlagSpec(
        "unimplemented",
        "Needs a recurrence or finite-termination theorem controlling every uncomputed focal value.",
    ),
    "actual_singular_return_map_derived": HonestyFlagSpec(
        "unimplemented",
        "The focal prefix is derived at one monodromic origin, not from the physical return around an arbitrary singular graphic.",
    ),
    "g2_passed": HonestyFlagSpec(
        "gate_open",
        "Finite focal computations do not provide an all-orders identity calculus for arbitrary singular return maps.",
    ),
    "songling_exact_source_transcribed": HonestyFlagSpec(
        "unimplemented",
        "True when the published rational Songling coefficients are represented exactly.",
    ),
    "published_four_cycle_theorem_replayed": HonestyFlagSpec(
        "unimplemented",
        "Needs a local replay of all four published return-map enclosures; bibliographic metadata is not a certificate.",
    ),
    "four_hyperbolic_returns_certified": HonestyFlagSpec(
        "unimplemented",
        "Needs four disjoint first-return root boxes whose derivative enclosures exclude one.",
    ),
    "h2_lower_bound_independently_certified": HonestyFlagSpec(
        "unimplemented",
        "The published H(2)>=4 theorem is known, but this flag requires an independent omnibias replay.",
    ),
    "bautin_stabilization_verified_to_order": HonestyFlagSpec(
        "unimplemented",
        "Set when V_k lies in (V_1,V_2,V_3) for every computed k>N.",
    ),
    "resonant_monomials_derived_not_declared": HonestyFlagSpec(
        "unimplemented",
        "True only when a normal-form certificate is sealed.",
    ),
    "new_closing_map": HonestyFlagSpec(
        "unimplemented",
        "True when the eps^3 sep^2 outgoing section bounds the kill-sequence W-ratio.",
    ),
    "weighted_section_obstruction": HonestyFlagSpec(
        "structural_limit",
        "Exact scalar and chart-O obstructions show that intrinsic eta-transversality does not provide ordinary C2 overlap matching or a uniform chart-O margin.",
    ),
    "frozen_section_scale_no_go": HonestyFlagSpec(
        "structural_limit",
        "True only for a single positive scale whose event and frozen-section W-ratio factors must be bounded separately.",
    ),
    "all_quasihomogeneous_atlases_excluded": HonestyFlagSpec(
        "structural_limit",
        "A moving sep^2 section, tracked cancellation, or multistage chart lies outside the single-scale frozen-section theorem.",
    ),
    "direct_ln_format_barrier": HonestyFlagSpec(
        "structural_limit",
        "The direct tau/log-W chain has fixed length and coefficients but its analytic domain and chain sup norm grow on the corrected kill sequence.",
    ),
    "fixed_time_complex_flow_enclosed": HonestyFlagSpec(
        "unimplemented",
        "Earned only by the outward-rounded complex Taylor-flow enclosure on the declared fixed-time cell.",
    ),
    "complex_first_hit_holomorphic": HonestyFlagSpec(
        "unimplemented",
        "Needs a complex implicit-event theorem with a uniform nonzero event derivative; fixed-time complex flow alone is insufficient.",
    ),
    "complex_normal_event_branch_certified": HonestyFlagSpec(
        "unimplemented",
        "Earned only on a declared complex cell by parametric complex interval Newton plus an independently replayed real first-hit.",
    ),
    "complex_physical_return_family_certified": HonestyFlagSpec(
        "unimplemented",
        "The local cubic normal-form event branch is not the full physical quadratic singular return family or its complete atlas.",
    ),
    "complex_separation_event_cover_certified": HonestyFlagSpec(
        "unimplemented",
        "Earned only for the declared regular V-event of the cubic comparison field across the full sep cell.",
    ),
    "complex_physical_e_out_cover_certified": HonestyFlagSpec(
        "unimplemented",
        "Earned only for the physical outgoing E_out section of the cubic comparison field on a finite complex sep cover with matched adjacent branches and independent real first-hit replay.",
    ),
    "physical_overlap_matching_proved": HonestyFlagSpec(
        "unimplemented",
        "The physical outgoing E_out cover still lacks the incoming E_sigma branch, full D-C/chart-O entry-exit composition, and the actual quadratic return map.",
    ),
    "actual_return_ln_membership_proved": HonestyFlagSpec(
        "unimplemented",
        "Needs an exact differential-polynomial chain for the physical first-hit family, not coordinate functions or a variational upper bound.",
    ),
    "normalized_ln_return_member": HonestyFlagSpec(
        "unimplemented",
        "An exact positive zero-preserving normalization may evade the direct format barrier, but none is constructed for the physical return family.",
    ),
    "g3_passed": HonestyFlagSpec(
        "gate_open",
        "The actual jointly parameterized return and admission family lacks a uniformly bounded LN, Pfaffian-over-LN, or quasianalytic format.",
    ),
    "picard_fuchs_instance_certified": HonestyFlagSpec(
        "unimplemented",
        "True only for an exact-Q syzygy of one declared Hamiltonian period; it is not physical DRR return membership.",
    ),
    "conditional_abelian_return_transfer": HonestyFlagSpec(
        "unimplemented",
        "Exact epsilon-threshold arithmetic is conditional on a supplied physical expansion, root cover, and uniform remainder bounds.",
    ),
    "drr_graphic_abelian_reduction": HonestyFlagSpec(
        "unimplemented",
        "No open DRR nilpotent saddle-node graphic is reduced here to the declared regular Hamiltonian oval family.",
    ),
    "drr_return_remainder_derived": HonestyFlagSpec(
        "unimplemented",
        "Needs a source-derived uniform second-order remainder for the physical DRR displacement.",
    ),
    "drr_endpoint_capture": HonestyFlagSpec(
        "unimplemented",
        "Regular Abelian contours exclude the singular center, separatrix, and graphic endpoint neighborhoods.",
    ),
    "drr_abelian_transfer_certified": HonestyFlagSpec(
        "gate_open",
        "Requires graphic reduction, physical return membership, a derived remainder, and singular endpoint capture.",
    ),
    "chart_dependent_gamma_survives_kill_sequence": HonestyFlagSpec(
        "unimplemented",
        "True when a chart-dependent gamma(s/r) closes every atlas cell on the kill sequence.",
    ),
    "scale_dichotomy_c2_remainder": HonestyFlagSpec(
        "structural_limit",
        "Physical C2 remainder on the coalescing atlas is not finite algebra.",
    ),
    "hk_theorem_24_used": HonestyFlagSpec(
        "structural_limit",
        "HK Theorem 2.4 hypotheses fail on the named kill sequences.",
    ),
    "fold_leading_c2_remainder": HonestyFlagSpec(
        "structural_limit",
        "Exact fold I-map C2 is not a physical remainder versus B_eps.",
    ),
    "inner_z_compact_bound": HonestyFlagSpec(
        "unimplemented",
        "True on the lambda=0 slow-line Cauchy majorant; fold compact is fold_z_compact_bound.",
    ),
    "fold_z_compact_bound": HonestyFlagSpec(
        "unimplemented",
        "True on a declared real fold compact of (L, lambda1) with Picard-included k; not physical C2.",
    ),
    "physical_c2_remainder": HonestyFlagSpec(
        "unimplemented",
        "Frozen-Z C2 identities are not a uniform-in-eps remainder, Z_x, or sep>0.",
    ),
    "z_x_bound": HonestyFlagSpec(
        "unimplemented",
        "True when matching-chart Z_x=Z_v eps/ell identities plus |Z_x|<1/100 on the fold I-map compact r in [1.4, 1.6], eps in [0, 0.02]; not sep>0 or G1.",
    ),
    "z_v_bound": HonestyFlagSpec(
        "unimplemented",
        "True when holomorphic Z_v identities plus an Interval enclosure |Z_v|<1/4 on the cancelled-N kill compact exclude 0; not fold Z_x, sep>0, or G1.",
    ),
    "z_slow_v_bound": HonestyFlagSpec(
        "unimplemented",
        "True when slow-line Z_V=-Z_v/ell identities plus |Z_V|<1/4 on the kill compact and holomorphic |Z_v|<1/4 on the fold wall r in [1.4, 1.6] exclude 0; not fold I-map Z_x, sep>0, or G1.",
    ),
    "outgoing_x_corridor_bounded": HonestyFlagSpec(
        "unimplemented",
        "True when the cleared two-root I-map from r1(1+theta) to a compact x_* has a vanishing r1 log r1 majorant; not height-section first-hit.",
    ),
    "post_corridor_margin": HonestyFlagSpec(
        "unimplemented",
        "True when T_*=Theta(eps^2) and (h/h_e)^(C eps)->1 after the x-corridor, independently of r1; not height-section first-hit.",
    ),
    "height_envelope_alpha0": HonestyFlagSpec(
        "unimplemented",
        "True when the C=0 T-h comparison and uniform |q| ratio are sealed; not actual-field T-h=O(eps) or height-section first-hit.",
    ),
    "q_ratio_c2": HonestyFlagSpec(
        "unimplemented",
        "True when the leading |q| ratio is <2 for every x on lambda1=-2; not k=1+O(eps), zeta remainder, or height-section first-hit.",
    ),
    "k_zeta_compact": HonestyFlagSpec(
        "unimplemented",
        "True when the C=0 normal k is 1+O(nu) with exact O(nu^2) remainder and the cubic prefactor is sealed; not a Z bound, T-h along the orbit, or height-section first-hit.",
    ),
    "kill_z_compact_bound": HonestyFlagSpec(
        "unimplemented",
        "True on a declared lambda1=-2 compact L in [0, 1] including L=0 with Picard-included k; rectangular and not small enough for C=2+delta or height-section first-hit.",
    ),
    "cancelled_n_usable_z": HonestyFlagSpec(
        "unimplemented",
        "True when cancelled-N holomorphic Z is enclosed on a declared real kill compact with 2 eps |V| |Z| < 1; slow-line only, not T-h along the orbit, C!=0 |g_h|, or height-section first-hit.",
    ),
    "height_mix_gh": HonestyFlagSpec(
        "unimplemented",
        "True when C!=0 ell/V mixing identities and |ell_h|=|C| nu^2 hold on a declared compact with ell>0; not T-h along the orbit or height-section first-hit.",
    ),
    "orbit_th_c0": HonestyFlagSpec(
        "unimplemented",
        "True when the actual-versus-comparison T_h gap splits as (k-1)+(q-C eps(T+eps^2))/h and equals k-1 at flux touching; not an integrated T-h orbit or height-section first-hit.",
    ),
    "th_integral_majorant": HonestyFlagSpec(
        "unimplemented",
        "True when the comparison-bootstrap integral of (T-h)_h is < 9 eps on a declared compact after T<=K(eps^2+h); not a Lohner-validated (V,h) orbit or height-section first-hit.",
    ),
    "vh_orbit_certified": HonestyFlagSpec(
        "unimplemented",
        "True when a QR-Lohner prefix of the cubic (V,h) field stays below 9 eps in T-h and certify_stopped_event hits V=-1/4; not the physical E_sigma section or height-section first-hit.",
    ),
    "e_out_first_hit": HonestyFlagSpec(
        "unimplemented",
        "True when certify_stopped_event hits matching-chart E_out = V+rho+nu rho h+C nu^2 rho h^2, the image of x=rho/nu under V=-eps x, on L in {9/25, 1/16, 0}; GRAZING E_sigma is excluded including at L=0; not uniform eps->0 or G1.",
    ),
    "e_out_eps_pack": HonestyFlagSpec(
        "unimplemented",
        "True when E_out first-hit is certified on the finite shrinking pack n in {16, 20, 25} at L=0 inside T=n^2/8; not a uniform-in-eps theorem, GRAZING E_sigma, or G1.",
    ),
    "e_out_speed_bound": HonestyFlagSpec(
        "unimplemented",
        "True when F_V = eps phi, phi(-eps) = eps^2(1+4 eps)>0, g<0, and T <= (rho-eps)/(3 eps^3) on the kill-line cubic; O(1/eps^3) comparison majorant, not Lohner for every eps, GRAZING E_sigma, or G1.",
    ),
    "e_sigma_speed_bound": HonestyFlagSpec(
        "unimplemented",
        "True when incoming reverse cubic has F(0)=-4 eps^3(1+eps), phi(0)<0 on eps in (0,1/16], and T <= 1/(4 eps^3(1+eps)); O(1/eps^3) comparison majorant, not certified E_sigma first-hit or G1.",
    ),
    "incoming_vwall_hit": HonestyFlagSpec(
        "unimplemented",
        "True when certify_stopped_event hits V=1/4 on the reverse cubic from V=0, h=4 eps^3, on L in {9/25, 1/16, 0}; Lohner wrapping still refuses E_sigma from V=0; not GRAZING first-hit or G1.",
    ),
    "e_sigma_first_hit": HonestyFlagSpec(
        "unimplemented",
        "True when certify_stopped_event hits E_sigma from the declared incoming point (V,h)=(3/4,1/4) on L in {9/25, 1/16, 0}; the GRAZING start V=0 still excludes E_sigma on this compact Lohner horizon; not the GRAZING band from V=0 or G1.",
    ),
    "e_sigma_from0_hit": HonestyFlagSpec(
        "unimplemented",
        "True when the V-parametrized comparison tube from GRAZING V=0 isolates a unique increasing E_sigma zero on L in {9/25, 1/16, 0} at eps=1/16; Lohner wrapping still refuses certify_stopped_event from V=0; not uniform eps or G1.",
    ),
    "e_sigma_unif_hit": HonestyFlagSpec(
        "unimplemented",
        "True when eight Interval slabs cover eps in [0, 1/8] with E_sigma>0 and dE/dV>0 on the cancelled height majorant; a single slab wraps; not a Lohner event for every eps or G1.",
    ),
    "e_sigma_wall_hit": HonestyFlagSpec(
        "unimplemented",
        "True when certify_stopped_event hits E_sigma from (V,h)=(1/4,1/40) inside the certified GRAZING-from-V=0 V=1/4 return box on L in {9/25, 1/16, 0}; not enclosure continuation of the whole h-box, not a single Lohner run from V=0, or G1.",
    ),
    "e_sigma_box_hit": HonestyFlagSpec(
        "unimplemented",
        "True when twelve h-slabs covering [1/50, 4/125] at V=1/4 certify E_sigma at L=0, a declared sub-box of every L-pack wall box; a single slab is unresolved; not the whole wall h-interval, not a single Lohner run from V=0, or G1.",
    ),
    "e_sigma_span_hit": HonestyFlagSpec(
        "unimplemented",
        "True when twenty-one h-slabs covering [19/1000, 1/25] at V=1/4 certify E_sigma at L=0, a declared span containing the whole L=0 wall box; a single slab is unresolved; not the L in {9/25, 1/16} walls, not a single Lohner run from V=0, or G1.",
    ),
    "e_sigma_pack_hit": HonestyFlagSpec(
        "unimplemented",
        "True when eighteen h-slabs covering [17/1000, 7/200] at V=1/4 certify E_sigma on L in {9/25, 1/16}, a declared span containing both remaining L-pack wall boxes; a single slab is unresolved; not a single Lohner run from V=0, or G1.",
    ),
    "e_sigma_eps_pack": HonestyFlagSpec(
        "unimplemented",
        "True when aligned (V,h)=(1/4,1/40) certifies E_sigma at eps=1/n for n in {16, 20, 25} and each GRAZING-from-V=0 V=1/4 box contains that point; a short horizon is unresolved; not a wall-span cover at every n, not a single Lohner run from V=0, or G1.",
    ),
    "e_sigma_oneshot_hit": HonestyFlagSpec(
        "unimplemented",
        "True when a single certify_stopped_event from GRAZING V=0, h=4 eps^3 hits E_sigma on L in {9/25, 1/16, 0} at eps=1/16; a short horizon is unresolved; not uniform in eps, not Z_x, or G1.",
    ),
    "e_sigma_oneshot_eps": HonestyFlagSpec(
        "unimplemented",
        "True when a single certify_stopped_event from GRAZING V=0 hits E_sigma at eps=1/n for n in {16, 20, 25} on L=0; a short horizon is unresolved; not uniform in eps, not Z_x, or G1.",
    ),
    "e_sigma_eps_span_hit": HonestyFlagSpec(
        "unimplemented",
        "True when three eps-slabs covering [1/25, 1/16] from aligned (1/4,1/40) certify E_sigma at L=0, a declared compact containing {1/16, 1/20, 1/25}; a single slab is unresolved; GRAZING V=0 on the last slab is unresolved; not every eps, not Z_x, or G1.",
    ),
    "e_sigma_eps_lo_hit": HonestyFlagSpec(
        "unimplemented",
        "True when six eps-slabs covering [1/64, 1/16] from aligned (1/4,1/40) certify E_sigma at L=0, a declared compact containing the previous [1/25, 1/16] span and {1/16, 1/20, 1/25, 1/32, 1/64}; a single slab is unresolved; GRAZING V=0 on the last slab is unresolved; not every eps, not Z_x, or G1.",
    ),
    "outgoing_first_hit": HonestyFlagSpec(
        "unimplemented",
        "First-hit of the large physical height section on L=1/n; k=1+O(eps) is not this flag.",
    ),
    "orbit_continuation_on_kill": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-B height inflation has Picard-included |Delta x|<1/3 on lambda1=-2, sep in (0, 1], eps in [0, 1/16]; not Stage A/C, C2 remainder, first-hit, or G1.",
    ),
    "stage_a_wall": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-A wall identities hold, Interval a>1/4, and leading Psi_pre factor B_-(a)/(B_-(0) sep^2)<1/4 on sep in [0, 1] at theta=1/8; not dx_e/dkappa, chi, Stage C, first-hit, or G1.",
    ),
    "chi_b_bound": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line chi_b identities hold and Interval sep*S_pre<3 with chi_b<9 on sep in [1/2^16, 1]; not dx_e/dkappa, Stage C, first-hit, or G1.",
    ),
    "dx_e_leading": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line dx_e leading identities hold, Interval prefactor <1/2, and threshold net exponent >1/8 after the y0 log remainder; not the uniform-in-chi bound, Stage C, first-hit, or G1.",
    ),
    "dx_e_unif": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line uniform-in-chi dx_e identities hold, Interval C<2, and extra>1/16 on the chi_b compact; not Stage C, first-hit, dx_e off the kill line, or G1.",
    ),
    "stage_c_amin": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C a_min identities hold, Interval end_lo>1/4, and 1/x<8 on the Stage-B end box; not an outgoing orbit, first-hit, C2, or G1.",
    ),
    "stage_c_exit": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C exit identities hold, Interval T_e/eps^2 in (1/16, 1), and T_e > h_e on the Stage-B end box; not an outgoing orbit, first-hit, C2, or G1.",
    ),
    "stage_c_th": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C T_h identities hold and Interval T_h>1/2 on the Stage-B end box at y_1=1; not an outgoing orbit, first-hit, C2, or G1.",
    ),
    "stage_c_gap": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C start-gap identities hold and Interval (T_e-h_1)/eps^2>1/32 at y_1=1; not an outgoing orbit, first-hit, C2, or G1.",
    ),
    "stage_c_env": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C C=0 envelope identities hold and Interval (1/32)(eps^2+h)<=T(h)<=eps^2+h; not a C!=0 orbit, first-hit, C2, or G1.",
    ),
    "stage_c_if": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C C=2 integrating-factor identities hold, Interval exponent<=3, and (h/h_1)^{C eps}<32; not T(h)<=C(eps^2+h) after the remaining integral, first-hit, C2, or G1.",
    ),
    "stage_c_int": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C C=2 T(h)-integral identities hold and Interval T(h)<=64(eps^2+h); not first-hit, C2, or G1.",
    ),
    "stage_c_lo": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C C=2 lower-envelope identities hold and Interval T(h)>=(1/32)(eps^2+h); not first-hit, C2, or G1.",
    ),
    "stage_c_k": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C C=2 tight-ratio identities hold and Interval T(h)<=6(eps^2+h); not T-h=O(eps), first-hit, C2, or G1.",
    ),
    "stage_c_boot": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C C=2 T-h bootstrap identities hold and Interval T-h<1; not O(eps), first-hit, C2, or G1.",
    ),
    "stage_c_rect": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C continuation-rectangle identities hold and Interval left wall<3, right wall>0; not first-hit, C2, or G1.",
    ),
    "stage_c_hit": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C comparison first-hit identities hold and Interval time to h=1 is <1024; not Lohner, signed-label section, chart O, C2, or G1.",
    ),
    "stage_c_sec": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C E_out comparison first-hit identities hold and Interval E_out>0 at start, dE/dh<0, h_hit<1/8; not Lohner, chart O, C2, or G1.",
    ),
    "stage_c_oneshot": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C Lohner first-hit of matching-chart x=4 from (x,y)=(1/4,1) holds on sep in {0, 3/5, 1} at eps=1/16; not every eps, chart O, C2, or G1.",
    ),
    "stage_c_oneshot_eps": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C shrinking-eps Lohner pack identities hold and unique transverse first-hit of x=n/4 at n in {16, 20, 25}; not every eps, chart O, C2, or G1.",
    ),
    "stage_c_eps_span": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line Stage-C parametric-eps Lohner cover identities hold and unique transverse first-hit of 4 eps x=1 on four 1/800 slabs covering [23/400, 1/16]; not every eps, chart O, C2, or G1.",
    ),
    "stage_c_origin": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line matching-chart Lohner pack identities hold and unique transverse first-hit of x=4 from (x,y)=(1/4,1) on sep in {3/2, 7/4, 2} (r1 in {1/4, 1/8, 0}); not every r1, complete first-hit on chart O, C2, or G1.",
    ),
    "stage_c_origin_span": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line parametric-sep matching-chart Lohner cover identities hold and unique transverse first-hit of x=4 on eight 1/16 slabs covering [3/2, 2]; not every r1, complete first-hit on chart O, C2, or G1.",
    ),
    "stage_c_origin_iface": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line nearer-interface matching-chart Lohner cover identities hold and unique transverse first-hit of x=4 from (x,y)=(1/8,1) on eight 1/32 slabs covering [7/4, 2]; not every r1, complete first-hit on chart O, C2, or G1.",
    ),
    "stage_c_origin_near": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line matching-chart Lohner cover identities hold and unique transverse first-hit of x=4 from (x,y)=(1/16,1) on eight 1/64 slabs covering [15/8, 2]; not every r1, complete first-hit on chart O, C2, or G1.",
    ),
    "stage_c_origin_x32": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line matching-chart Lohner cover identities hold and unique transverse first-hit of x=4 from (x,y)=(1/32,1) on eight 1/128 slabs covering [31/16, 2]; not every r1, complete first-hit on chart O, C2, or G1.",
    ),
    "stage_c_compare": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line comparison identities hold and the phase-wise speed bound reaches x=8 from (x,y)=(1/4,1) for every r1 in [0,1] and every eps in [1/32, 1/16]; not every eps, Lohner, complete first-hit on chart O, C2, or G1.",
    ),
    "stage_c_uniform": HonestyFlagSpec(
        "unimplemented",
        "True when kill-line neck identities hold and dy/dx keeps x' >= eps/2 from (x,y)=(1/4,1) for every r1 in [0,1] and every eps in (0, 1/16], so x=(1/4)/eps is hit; not Lohner, eps>1/16, the shrinking interface, C2, or G1.",
    ),
    "stage_c_interface": HonestyFlagSpec(
        "unimplemented",
        "True when entrance gap 1/4 and the neck bound from x=1/2 keep x' >= eps/5 for every start in (0, 1/2], every r1 in [0,1], and every eps in (0, 1/16]; an interface landing in (0, 1/2] is included; not Lohner, the height-section flag, or G1.",
    ),
    "sep_spre": HonestyFlagSpec(
        "unimplemented",
        "True when dyadic-plus-tail Interval sep*S_pre < 11/5 and chi_b <= 8 on every sep in (0, 1], and the dx_e log remainder keeps the net exponent > 1/8; not dx_e off the kill line, uniform-in-chi, Stage C, first-hit, or G1.",
    ),
    "dx_e_off": HonestyFlagSpec(
        "unimplemented",
        "True when h(sep/r1)<11/5 on u in (0, 2] and, for lambda1 in [-4, -2] and sep in (0, 1], the net exponent stays >1/8 with chi_b<=21/5 and C<2; not lambda1<-4, not lambda1 in (-2, 0), Stage C, first-hit, or G1.",
    ),
    "dx_e_ray": HonestyFlagSpec(
        "unimplemented",
        "True when, for every rstar>=1 and every sep in (0, 1], the X<=2*rstar tau-coefficient keeps the net exponent >1/8 with chi_b<=21/5 and C<2; not lambda1 in (-2, 0), Stage C, first-hit, or G1.",
    ),
    "dx_e_near": HonestyFlagSpec(
        "unimplemented",
        "True when, for every lambda1 in [-3/2, -2) and every sep in (0, 1], h(u)<11/5 on u in (0, 4] and the net exponent stays >1/8 with chi_b<=26/5 and C<2; not lambda1 in (-3/2, 0), Stage C, first-hit, or G1.",
    ),
    "dx_e_open": HonestyFlagSpec(
        "unimplemented",
        "True when, for every lambda1 in (-3/2, 0) and every sep in (0, min(1, (8/5) rstar)), the eps cap keeps the net exponent >1/8 and C<(1/5)/a; not the fixed eps=1/16 slab, Stage C, first-hit, or G1.",
    ),
}


def build_honesty(**overrides: bool) -> dict[str, object]:
    """Return flat flags plus ``_honesty_reasons`` for every false entry."""
    flags: dict[str, bool] = dict(overrides)
    reasons: dict[str, dict[str, str]] = {}
    for name, value in flags.items():
        if value:
            continue
        spec = HONESTY_INVENTORY.get(name)
        if spec is None:
            reasons[name] = {
                "reason": "unimplemented",
                "detail": "flag not catalogued in HONESTY_INVENTORY",
            }
        else:
            reasons[name] = {"reason": spec.reason, "detail": spec.detail}
    return {**flags, "_honesty_reasons": reasons}


def inventory_report() -> dict[str, dict[str, str]]:
    """Full catalog keyed by flag name."""
    return {
        name: {"reason": spec.reason, "detail": spec.detail}
        for name, spec in HONESTY_INVENTORY.items()
    }
