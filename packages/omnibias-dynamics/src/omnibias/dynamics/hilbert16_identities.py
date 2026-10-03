# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact finite identities for the Hilbert XVI coalescence atlas.

A hit certifies the named rational identity. It does not prove a C2 remainder,
G1, or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.proof.catalog import CatalogEntry, register_catalog
from omnibias.core.proof.discovery import (
    Candidate,
    ExactCheck,
    FiniteFamily,
    Statement,
    run_discovery,
)
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.abelian_return_transfer import (
    residual_derivative_threshold,
    residual_normalized_displacement,
    residual_value_threshold,
)
from omnibias.dynamics.cancelled_n import (
    residual_delta_slow,
    residual_ell_V_nu,
    residual_n_cancelled,
    residual_t_L,
    residual_t_lambda,
    residual_z0_holomorphic,
)
from omnibias.dynamics.canonical_zeta import (
    residual_beta_linear_jet,
    residual_c0_jet,
    residual_field_matches_closed,
    residual_k_implicit,
    residual_k_lambda0,
    residual_limiting_cubic,
    residual_sqrt_inverse,
    residual_v0_quadratic,
    residual_V_factor,
    residual_zeta_at_zero,
    residual_zeta_plus_one_canceled,
)
from omnibias.dynamics.chi_b import (
    residual_c_decay,
    residual_chi_declared,
    residual_limit_nine,
)
from omnibias.dynamics.dx_e_leading import (
    residual_net_floor,
    residual_prefactor,
    residual_slope_half,
)
from omnibias.dynamics.dx_e_near import (
    residual_near_chi,
    residual_near_u,
    residual_near_wall,
)
from omnibias.dynamics.dx_e_off import (
    residual_off_chi,
    residual_off_u,
    residual_off_wall,
)
from omnibias.dynamics.dx_e_open import (
    residual_open_chi,
    residual_open_gap,
    residual_open_wall,
)
from omnibias.dynamics.dx_e_ray import (
    residual_ray_chi,
    residual_ray_coeff,
    residual_ray_x,
)
from omnibias.dynamics.dx_e_unif import (
    residual_c_weaker,
    residual_extra_half,
    residual_lift_nine,
)
from omnibias.dynamics.e_out_eps import (
    residual_h0_matching,
    residual_horizon_n2,
    residual_v0_matching,
    residual_vdot_kill_init,
)
from omnibias.dynamics.e_out_section import (
    residual_e_out_group,
    residual_e_out_lead,
    residual_e_sigma_nu0,
    residual_section_embed,
)
from omnibias.dynamics.e_out_speed import (
    residual_F_expand,
    residual_F_V_phi,
    residual_phi_right,
    residual_T_cubic,
)
from omnibias.dynamics.e_sigma_box import (
    residual_cover_contains,
    residual_cover_hi,
    residual_cover_lo,
    residual_cover_width,
)
from omnibias.dynamics.e_sigma_eps import (
    residual_sigma_h0_16,
    residual_sigma_h0_20,
    residual_sigma_horizon,
    residual_sigma_pack_sum,
)
from omnibias.dynamics.e_sigma_eps_lo import (
    residual_elo_contains_32,
    residual_elo_lo,
    residual_elo_pack_L,
    residual_elo_width,
)
from omnibias.dynamics.e_sigma_eps_span import (
    residual_espan_contains,
    residual_espan_lo,
    residual_espan_pack_L,
    residual_espan_width,
)
from omnibias.dynamics.e_sigma_from0 import (
    residual_de_sigma_lo,
    residual_f_cubic_factor,
    residual_log_taylor_num,
    residual_ratio_split,
    residual_vdot_rev_split,
)
from omnibias.dynamics.e_sigma_hit import (
    residual_e_sigma_in_group,
    residual_e_sigma_in_lead,
    residual_restart_h,
    residual_restart_v,
)
from omnibias.dynamics.e_sigma_in import (
    residual_hdot_rev,
    residual_v0_grazing,
    residual_vdot_rev_init,
    residual_vin_wall,
)
from omnibias.dynamics.e_sigma_oneshot import (
    residual_oneshot_h0,
    residual_oneshot_pack_L,
    residual_oneshot_short,
    residual_oneshot_T,
)
from omnibias.dynamics.e_sigma_oneshot_eps import (
    residual_oeps_h0_25,
    residual_oeps_pack_sum,
    residual_oeps_T20,
    residual_oeps_T25,
)
from omnibias.dynamics.e_sigma_pack import (
    residual_pack_align,
    residual_pack_hi,
    residual_pack_lo,
    residual_pack_width,
)
from omnibias.dynamics.e_sigma_span import (
    residual_span_align_hi,
    residual_span_align_lo,
    residual_span_hi,
    residual_span_width,
)
from omnibias.dynamics.e_sigma_speed import (
    residual_F_zero,
    residual_phi_zero,
    residual_phi_zero_factor,
    residual_T_in,
)
from omnibias.dynamics.e_sigma_unif import (
    residual_de_eps0,
    residual_E_eps0,
    residual_I_cancel,
    residual_I_limit,
)
from omnibias.dynamics.e_sigma_wall import (
    residual_align_h,
    residual_align_v,
    residual_e_align_nu0,
    residual_e_align_start,
)
from omnibias.dynamics.entry_exit_leading import (
    residual_double_root_entry_exit,
    residual_height_inflation_dx_dy,
    residual_partial_fraction_numerators,
    residual_product_alpha_one,
)
from omnibias.dynamics.fold_leading import (
    residual_chi_log_second,
    residual_fold_d2x,
    residual_fold_dx,
    residual_fold_log_d1,
    residual_fold_log_d2,
    residual_fold_reciprocal,
    residual_fold_reciprocal_log_c2,
    residual_interface_relative,
    residual_lifted_d2x,
    residual_lifted_dx,
    residual_lifted_log_c2,
    residual_lifted_log_d1,
    residual_relative_remainder,
    residual_z_relative_on_lift,
    residual_zeta_rho,
)
from omnibias.dynamics.fold_z_x import (
    residual_fold_x_interior,
    residual_zx_chain,
    residual_zx_declared,
)
from omnibias.dynamics.fold_zeta import (
    residual_fold_bminus,
    residual_fold_disc,
    residual_fold_L,
    residual_fold_lambda,
)
from omnibias.dynamics.height_envelope import (
    residual_amgm_qbound,
    residual_envelope_exit,
    residual_q_envelope,
    residual_th_alpha0,
)
from omnibias.dynamics.height_mix import (
    residual_ell_mix,
    residual_g_lead,
    residual_V_h_c,
    residual_V_mix,
    residual_V_v_ell,
)
from omnibias.dynamics.k_zeta_remainder import (
    residual_cubic_vs_tworoot,
    residual_k_box_gap,
    residual_k_lead_embed,
    residual_k_normal,
    residual_k_normal_vs_lead,
    residual_k_turn,
    residual_relative_prefactor,
)
from omnibias.dynamics.kill_zeta import (
    residual_kill_disc_gap,
    residual_kill_lambda,
    residual_kill_product,
    residual_kill_sum,
    residual_kill_tworoot,
)
from omnibias.dynamics.ln_format_barrier import (
    residual_kill_log_w,
    residual_kill_norm,
    residual_kill_tau,
)
from omnibias.dynamics.orbit_th import (
    residual_th_split,
    residual_th_touching,
)
from omnibias.dynamics.outgoing_corridor import (
    residual_interface_embed,
    residual_J_numerator,
    residual_root_sum,
)
from omnibias.dynamics.physical_c2 import (
    residual_frozen_c2_gap,
    residual_frozen_log_d1_gap,
    residual_lift_cubic_mismatch,
)
from omnibias.dynamics.post_corridor import (
    residual_h_exit,
    residual_q_leading,
    residual_T_kinetic,
    residual_V_embed,
)
from omnibias.dynamics.q_ratio_c2 import (
    residual_kill_square,
    residual_ratio_disc,
    residual_ratio_gap,
    residual_ratio_outer,
)
from omnibias.dynamics.quasihomogeneous_dichotomy import (
    residual_event_log_expansion,
    residual_moving_section_log_expansion,
    residual_outgoing_log_expansion,
)
from omnibias.dynamics.scale_dichotomy import (
    residual_blowup_height,
    residual_blowup_height_ratio,
    residual_event_leading,
    residual_joint_sep_r1_sum,
    residual_second_kappa_difference,
)
from omnibias.dynamics.sep_spre import (
    residual_spre_chi,
    residual_spre_net,
    residual_spre_tail,
)
from omnibias.dynamics.shrinking_root_leading import (
    residual_c_limit,
    residual_r1_limit_dx,
    residual_rescaled_b,
    residual_two_root_dx,
    residual_xi_limit,
)
from omnibias.dynamics.stage_a import (
    residual_a_min,
    residual_B_slope,
    residual_B_wall,
)
from omnibias.dynamics.stage_b import (
    residual_alpha_pos,
    residual_dx_declared,
    residual_r1_half,
)
from omnibias.dynamics.stage_c import (
    residual_amin_floor,
    residual_amin_written,
    residual_inv_guess,
)
from omnibias.dynamics.stage_c_boot import (
    residual_c_log,
    residual_lin_fifteen,
    residual_one_k,
)
from omnibias.dynamics.stage_c_compare import (
    residual_cmp_emax,
    residual_cmp_phases,
    residual_cmp_xfar,
)
from omnibias.dynamics.stage_c_env import (
    residual_env_add,
    residual_half_gap,
    residual_te_unit,
)
from omnibias.dynamics.stage_c_eps_span import (
    residual_cspan_join,
    residual_cspan_n800,
    residual_cspan_slabs,
)
from omnibias.dynamics.stage_c_exit import (
    residual_eps_y0,
    residual_gap_room,
    residual_te_half,
)
from omnibias.dynamics.stage_c_gap import (
    residual_h1_cube,
    residual_start_gap,
    residual_wall_gap,
)
from omnibias.dynamics.stage_c_hit import (
    residual_hmax_gap,
    residual_inv_floor,
    residual_time_pre,
)
from omnibias.dynamics.stage_c_if import (
    residual_c_triple,
    residual_sqrt_edge,
    residual_twice_six,
)
from omnibias.dynamics.stage_c_int import (
    residual_ceps_eight,
    residual_inv_seven,
    residual_one_minus,
)
from omnibias.dynamics.stage_c_interface import (
    residual_iface_edge,
    residual_iface_phases,
    residual_iface_time,
)
from omnibias.dynamics.stage_c_k import (
    residual_k_slope,
    residual_six_eps,
    residual_twice_three,
)
from omnibias.dynamics.stage_c_lo import (
    residual_half_minus,
    residual_one_eps,
    residual_wrap_room,
)
from omnibias.dynamics.stage_c_oneshot import (
    residual_hit_T,
    residual_short_T,
    residual_x_section,
)
from omnibias.dynamics.stage_c_oneshot_eps import (
    residual_ceps_n20_sec,
    residual_ceps_n25_sec,
    residual_ceps_T25,
)
from omnibias.dynamics.stage_c_origin import (
    residual_r1_o_eighth,
    residual_r1_o_quarter,
    residual_r1_origin,
)
from omnibias.dynamics.stage_c_origin_iface import (
    residual_oiface_join,
    residual_oiface_n32,
    residual_oiface_slabs,
)
from omnibias.dynamics.stage_c_origin_near import (
    residual_onear_join,
    residual_onear_n64,
    residual_onear_slabs,
)
from omnibias.dynamics.stage_c_origin_span import (
    residual_ospan_join,
    residual_ospan_n16,
    residual_ospan_slabs,
)
from omnibias.dynamics.stage_c_origin_x32 import (
    residual_ox32_join,
    residual_ox32_n128,
    residual_ox32_slabs,
)
from omnibias.dynamics.stage_c_rect import (
    residual_amin_eps,
    residual_hmax_two,
    residual_twice_two,
)
from omnibias.dynamics.stage_c_sec import (
    residual_corr_sum,
    residual_rho_start,
    residual_vh_floor,
)
from omnibias.dynamics.stage_c_th import (
    residual_mid_product,
    residual_th_half_room,
    residual_th_three_four,
)
from omnibias.dynamics.stage_c_uniform import (
    residual_unif_neck,
    residual_unif_phases,
    residual_unif_time,
)
from omnibias.dynamics.th_integral import (
    residual_ftc_linear,
    residual_log_sqrt_prefactor,
    residual_majorant_split,
    residual_sqrt_ratio_sq,
    residual_th_dh,
)
from omnibias.dynamics.vh_orbit import (
    field_f,
    field_g,
    residual_g_jet_vh,
    residual_hdot_vh,
    residual_q_plus_f,
    residual_Vdot_split,
)
from omnibias.dynamics.weighted_section import (
    residual_origin_weighted_speed,
    residual_weighted_hit_d2q,
    residual_weighted_hit_dq,
)
from omnibias.dynamics.z_slow_v import (
    residual_ell_min,
    residual_V_v_slow,
    residual_ZV_chain,
)
from omnibias.dynamics.z_v_bound import (
    residual_q1_v,
    residual_wall_ratio_v,
    residual_zv_quotient,
)
from omnibias.dynamics.z_x_gap import (
    residual_unfrozen_log_d1_gap,
    residual_unfrozen_recovers_frozen,
    residual_zx_extra,
)

IDENTITY_NAMES: tuple[str, ...] = (
    "double_root",
    "w_recovers_height_power",
    "outgoing_as_w_ratio",
    "r1_product",
    "sigma_kappa",
    "blowup_height_scale",
    "blowup_height_ratio",
    "event_leading_kappa",
    "joint_sep_r1_sum",
    "hk_second_kappa_vanishes",
    "weighted_hit_dq",
    "weighted_hit_d2q",
    "origin_weighted_speed",
    "quasi_event_log",
    "quasi_frozen_log",
    "quasi_moving_log",
    "ln_kill_tau",
    "ln_kill_log_w",
    "ln_kill_norm",
    "abelian_transfer_value",
    "abelian_transfer_derivative",
    "abelian_transfer_normalized",
    "partial_fraction_numerators",
    "double_root_entry_exit",
    "product_alpha_one",
    "height_inflation_dx_dy",
    "r1_half",
    "dx_declared",
    "alpha_pos",
    "B_wall",
    "B_slope",
    "a_min_kill",
    "limit_nine",
    "c_decay",
    "chi_declared",
    "dx_prefactor",
    "slope_half",
    "net_floor",
    "extra_half",
    "c_weaker",
    "lift_nine",
    "amin_written",
    "inv_guess",
    "amin_floor",
    "te_half",
    "eps_y0",
    "gap_room",
    "mid_product",
    "th_three_four",
    "th_half_room",
    "start_gap",
    "h1_cube",
    "wall_gap",
    "te_unit",
    "half_gap",
    "env_add",
    "c_triple",
    "twice_six",
    "sqrt_edge",
    "ceps_eight",
    "one_minus",
    "inv_seven",
    "half_minus",
    "one_eps",
    "wrap_room",
    "six_eps",
    "twice_three",
    "k_slope",
    "one_k",
    "lin_fifteen",
    "c_log",
    "hmax_two",
    "twice_two",
    "amin_eps",
    "inv_floor",
    "time_pre",
    "hmax_gap",
    "rho_start",
    "vh_floor",
    "corr_sum",
    "x_section",
    "hit_T",
    "short_T",
    "ceps_n20_sec",
    "ceps_n25_sec",
    "ceps_T25",
    "cspan_join",
    "cspan_slabs",
    "cspan_n800",
    "r1_origin",
    "r1_o_quarter",
    "r1_o_eighth",
    "ospan_join",
    "ospan_slabs",
    "ospan_n16",
    "oiface_join",
    "oiface_slabs",
    "oiface_n32",
    "onear_join",
    "onear_slabs",
    "onear_n64",
    "ox32_join",
    "ox32_slabs",
    "ox32_n128",
    "cmp_emax",
    "cmp_xfar",
    "cmp_phases",
    "unif_neck",
    "unif_phases",
    "unif_time",
    "iface_edge",
    "iface_phases",
    "iface_time",
    "spre_tail",
    "spre_chi",
    "spre_net",
    "off_wall",
    "off_u",
    "off_chi",
    "ray_x",
    "ray_coeff",
    "ray_chi",
    "near_wall",
    "near_u",
    "near_chi",
    "open_wall",
    "open_chi",
    "open_gap",
    "fold_dx",
    "fold_d2x",
    "fold_log_d1",
    "fold_log_d2",
    "fold_reciprocal",
    "fold_reciprocal_log_c2",
    "relative_remainder",
    "interface_relative",
    "zeta_rho",
    "lifted_dx",
    "lifted_d2x",
    "lifted_log_d1",
    "lifted_log_c2",
    "z_relative_on_lift",
    "chi_log_second",
    "two_root_dx",
    "r1_limit_dx",
    "c_limit",
    "rescaled_b",
    "xi_limit",
    "v0_quadratic",
    "sqrt_inverse",
    "V_factor",
    "zeta_at_zero",
    "limiting_cubic",
    "k_lambda0",
    "beta_linear_jet",
    "zeta_plus_one_canceled",
    "k_implicit",
    "field_matches_closed",
    "c0_jet",
    "fold_lambda",
    "fold_L",
    "fold_disc",
    "fold_bminus",
    "lift_cubic_mismatch",
    "frozen_log_d1_gap",
    "frozen_c2_gap",
    "unfrozen_log_d1_gap",
    "unfrozen_recovers_frozen",
    "zx_extra",
    "zx_chain",
    "zx_declared",
    "fold_x_interior",
    "q1_v",
    "zv_quotient",
    "wall_ratio_v",
    "V_v_slow",
    "ZV_chain",
    "ell_min",
    "J_numerator",
    "root_sum",
    "interface_embed",
    "V_embed",
    "T_kinetic",
    "h_exit",
    "q_leading",
    "amgm_qbound",
    "envelope_exit",
    "th_alpha0",
    "q_envelope",
    "ratio_gap",
    "ratio_disc",
    "kill_square",
    "ratio_outer",
    "k_turn",
    "k_normal",
    "k_lead_embed",
    "k_normal_vs_lead",
    "k_box_gap",
    "cubic_vs_tworoot",
    "relative_prefactor",
    "kill_lambda",
    "kill_product",
    "kill_sum",
    "kill_disc_gap",
    "kill_tworoot",
    "delta_slow",
    "ell_V_nu",
    "t_lambda",
    "t_L",
    "n_cancelled",
    "z0_holomorphic",
    "ell_mix",
    "V_mix",
    "V_v_ell",
    "V_h_c",
    "g_lead",
    "th_split",
    "th_touching",
    "th_dh",
    "majorant_split",
    "ftc_linear",
    "sqrt_ratio_sq",
    "log_sqrt_prefactor",
    "q_plus_f",
    "hdot_vh",
    "g_jet_vh",
    "Vdot_split",
    "e_sigma_nu0",
    "e_out_lead",
    "section_embed",
    "e_out_group",
    "v0_matching",
    "h0_matching",
    "horizon_n2",
    "vdot_kill_init",
    "F_expand",
    "F_V_phi",
    "phi_right",
    "T_cubic",
    "F_zero",
    "phi_zero",
    "phi_zero_factor",
    "T_in",
    "v0_grazing",
    "hdot_rev",
    "vdot_rev_init",
    "vin_wall",
    "restart_v",
    "restart_h",
    "e_sigma_in_lead",
    "e_sigma_in_group",
    "f_cubic_factor",
    "vdot_rev_split",
    "ratio_split",
    "log_taylor_num",
    "de_sigma_lo",
    "I_cancel",
    "I_limit",
    "E_eps0",
    "de_eps0",
    "align_v",
    "align_h",
    "e_align_nu0",
    "e_align_start",
    "cover_lo",
    "cover_hi",
    "cover_width",
    "cover_contains",
    "span_hi",
    "span_width",
    "span_align_lo",
    "span_align_hi",
    "pack_lo",
    "pack_hi",
    "pack_width",
    "pack_align",
    "sigma_pack_sum",
    "sigma_h0_16",
    "sigma_h0_20",
    "sigma_horizon",
    "oneshot_h0",
    "oneshot_T",
    "oneshot_short",
    "oneshot_pack_L",
    "oeps_pack_sum",
    "oeps_T20",
    "oeps_T25",
    "oeps_h0_25",
    "espan_lo",
    "espan_width",
    "espan_contains",
    "espan_pack_L",
    "elo_lo",
    "elo_width",
    "elo_contains_32",
    "elo_pack_L",
)

HILBERT16_IDENTITY_KIND = "hilbert16_coalescence_identities"


def _honesty(*, ok: bool) -> dict[str, bool]:
    return {
        "finite_algebra_replayed": ok,
        "parent_status_open": True,
        "g1_passed": False,
        "g4_passed": False,
        "full_graphic_cyclicity_proved": False,
        "full_hilbert16_solved": False,
        "saddle_node_c2_remainder": False,
        "two_blowup_c2_remainder": False,
        "scale_dichotomy_c2_remainder": False,
        "fold_leading_c2_remainder": False,
    }


def residual_double_root(rstar: Fraction, x: Fraction) -> Fraction:
    return rstar**2 + (-2 * rstar) * x + x**2 - (x - rstar) ** 2


def residual_w_height(eps: Fraction, w: Fraction, height_power: Fraction) -> Fraction:
    return eps * w - height_power


def residual_outgoing_w_ratio(
    eps: Fraction, w_max: Fraction, w_e: Fraction, height_max_power: Fraction, height_e_power: Fraction
) -> Fraction:
    return height_max_power / height_e_power - w_max / w_e


def residual_r1_product(lam1: Fraction, sep: Fraction, L: Fraction) -> Fraction:
    return (-lam1 - sep) / 2 * (-lam1 + sep) - 2 * L


def residual_sigma_kappa(sigma: Fraction, sep: Fraction, chi: Fraction, r1: Fraction) -> Fraction:
    return sigma * (chi * r1 / sep) - (sigma / sep) * chi * r1


def _sample(name: str) -> Fraction:
    rstar, x = Fraction(3, 2), Fraction(-1, 5)
    if name == "double_root":
        return residual_double_root(rstar, x)
    if name == "w_recovers_height_power":
        eps, w = Fraction(1, 4), Fraction(3, 2)
        return residual_w_height(eps, w, eps * w)
    if name == "outgoing_as_w_ratio":
        eps, w_max, w_e = Fraction(1, 3), Fraction(4), Fraction(2)
        return residual_outgoing_w_ratio(eps, w_max, w_e, eps * w_max, eps * w_e)
    if name == "r1_product":
        lam1, sep = Fraction(-4), Fraction(2)
        L = (lam1**2 - sep**2) / 4
        return residual_r1_product(lam1, sep, L)
    if name == "sigma_kappa":
        return residual_sigma_kappa(Fraction(1, 3), Fraction(1, 7), Fraction(2), Fraction(5))
    if name == "blowup_height_scale":
        eps, sigma, eta = Fraction(1, 2), Fraction(1, 3), Fraction(2)
        return residual_blowup_height(eps, sigma, eta, eps**3 * sigma**2 * eta)
    if name == "blowup_height_ratio":
        eps, eta = Fraction(1, 2), Fraction(2)
        sigma1, sigma2 = Fraction(1, 3), Fraction(1, 5)
        return residual_blowup_height_ratio(
            eps**3 * sigma1**2 * eta, eps**3 * sigma2**2 * eta, sigma1, sigma2
        )
    if name == "event_leading_kappa":
        return residual_event_leading(
            Fraction(13, 6), Fraction(1, 2), Fraction(3), Fraction(1), Fraction(1, 3), Fraction(2)
        )
    if name == "joint_sep_r1_sum":
        return residual_joint_sep_r1_sum(Fraction(-4), Fraction(2), Fraction(1))
    if name == "weighted_hit_dq":
        rate, q = Fraction(3, 2), Fraction(1, 16)
        return residual_weighted_hit_dq(rate, q, -1 / (rate * q))
    if name == "weighted_hit_d2q":
        rate, q = Fraction(3, 2), Fraction(1, 16)
        return residual_weighted_hit_d2q(rate, q, 1 / (rate * q**2))
    if name == "origin_weighted_speed":
        q, theta, eta0 = Fraction(1, 16), Fraction(1, 8), Fraction(2)
        return residual_origin_weighted_speed(q, theta, eta0, q * (1 + theta) * eta0)
    if name == "quasi_event_log":
        a, b, n, log_n = Fraction(2, 3), Fraction(5, 4), Fraction(7), Fraction(3, 2)
        event_log = (1 - b) * n**2 - a * log_n
        return residual_event_log_expansion(event_log, a, b, n**2, log_n)
    if name == "quasi_frozen_log":
        a, b, n, log_n = Fraction(2, 3), Fraction(5, 4), Fraction(7), Fraction(3, 2)
        frozen_log = 2 * b * n + (2 * a + 3) * log_n / n
        return residual_outgoing_log_expansion(frozen_log, a, b, 0, n, log_n)
    if name == "quasi_moving_log":
        a, b, n, log_n = Fraction(2, 3), Fraction(5, 4), Fraction(7), Fraction(3, 2)
        moving_log = (2 * b - 2) * n + 2 * a * log_n / n
        return residual_moving_section_log_expansion(
            moving_log, a, b, Fraction(2), 3, n, log_n
        )
    if name == "ln_kill_tau":
        n = Fraction(7)
        return residual_kill_tau(n, n)
    if name == "ln_kill_log_w":
        n = Fraction(7)
        return residual_kill_log_w(2 * n, n)
    if name == "ln_kill_norm":
        n = Fraction(7)
        return residual_kill_norm(3 * n, n)
    if name == "abelian_transfer_value":
        margin, bound = Fraction(1, 20), Fraction(2)
        return residual_value_threshold(margin, bound, margin / bound)
    if name == "abelian_transfer_derivative":
        margin, bound = Fraction(1, 4), Fraction(3)
        return residual_derivative_threshold(margin, bound, margin / bound)
    if name == "abelian_transfer_normalized":
        epsilon = Fraction(1, 40)
        abelian, remainder = Fraction(2, 5), Fraction(1, 7)
        displacement = epsilon * abelian + epsilon**2 * remainder
        return residual_normalized_displacement(
            displacement,
            epsilon,
            abelian,
            remainder,
        )
    if name == "hk_second_kappa_vanishes":
        return residual_second_kappa_difference(Fraction(3, 5), Fraction(1, 7))
    if name == "partial_fraction_numerators":
        return residual_partial_fraction_numerators(Fraction(2), Fraction(1), Fraction(4))
    if name == "double_root_entry_exit":
        return residual_double_root_entry_exit(Fraction(5, 2), Fraction(3, 2))
    if name == "product_alpha_one":
        return residual_product_alpha_one(Fraction(2), Fraction(3), Fraction(1, 2), Fraction(5))
    if name == "height_inflation_dx_dy":
        return residual_height_inflation_dx_dy(
            Fraction(1, 3), Fraction(7, 2), Fraction(5), Fraction(4)
        )
    if name == "r1_half":
        return residual_r1_half(Fraction(3, 5))
    if name == "dx_declared":
        return residual_dx_declared()
    if name == "alpha_pos":
        return residual_alpha_pos()
    if name == "B_wall":
        return residual_B_wall(Fraction(3, 5))
    if name == "B_slope":
        return residual_B_slope(Fraction(3, 5))
    if name == "a_min_kill":
        return residual_a_min()
    if name == "limit_nine":
        return residual_limit_nine()
    if name == "c_decay":
        return residual_c_decay()
    if name == "chi_declared":
        return residual_chi_declared()
    if name == "dx_prefactor":
        return residual_prefactor()
    if name == "slope_half":
        return residual_slope_half()
    if name == "net_floor":
        return residual_net_floor()
    if name == "extra_half":
        return residual_extra_half()
    if name == "c_weaker":
        return residual_c_weaker()
    if name == "lift_nine":
        return residual_lift_nine()
    if name == "amin_written":
        return residual_amin_written()
    if name == "inv_guess":
        return residual_inv_guess()
    if name == "amin_floor":
        return residual_amin_floor()
    if name == "te_half":
        return residual_te_half()
    if name == "eps_y0":
        return residual_eps_y0()
    if name == "gap_room":
        return residual_gap_room()
    if name == "mid_product":
        return residual_mid_product()
    if name == "th_three_four":
        return residual_th_three_four()
    if name == "th_half_room":
        return residual_th_half_room()
    if name == "start_gap":
        return residual_start_gap()
    if name == "h1_cube":
        return residual_h1_cube()
    if name == "wall_gap":
        return residual_wall_gap()
    if name == "te_unit":
        return residual_te_unit()
    if name == "half_gap":
        return residual_half_gap()
    if name == "env_add":
        return residual_env_add()
    if name == "c_triple":
        return residual_c_triple()
    if name == "twice_six":
        return residual_twice_six()
    if name == "sqrt_edge":
        return residual_sqrt_edge()
    if name == "ceps_eight":
        return residual_ceps_eight()
    if name == "one_minus":
        return residual_one_minus()
    if name == "inv_seven":
        return residual_inv_seven()
    if name == "half_minus":
        return residual_half_minus()
    if name == "one_eps":
        return residual_one_eps()
    if name == "wrap_room":
        return residual_wrap_room()
    if name == "six_eps":
        return residual_six_eps()
    if name == "twice_three":
        return residual_twice_three()
    if name == "k_slope":
        return residual_k_slope()
    if name == "one_k":
        return residual_one_k()
    if name == "lin_fifteen":
        return residual_lin_fifteen()
    if name == "c_log":
        return residual_c_log()
    if name == "hmax_two":
        return residual_hmax_two()
    if name == "twice_two":
        return residual_twice_two()
    if name == "amin_eps":
        return residual_amin_eps()
    if name == "inv_floor":
        return residual_inv_floor()
    if name == "time_pre":
        return residual_time_pre()
    if name == "hmax_gap":
        return residual_hmax_gap()
    if name == "rho_start":
        return residual_rho_start()
    if name == "vh_floor":
        return residual_vh_floor()
    if name == "corr_sum":
        return residual_corr_sum()
    if name == "x_section":
        return residual_x_section()
    if name == "hit_T":
        return residual_hit_T()
    if name == "short_T":
        return residual_short_T()
    if name == "ceps_n20_sec":
        return residual_ceps_n20_sec()
    if name == "ceps_n25_sec":
        return residual_ceps_n25_sec()
    if name == "ceps_T25":
        return residual_ceps_T25()
    if name == "cspan_join":
        return residual_cspan_join()
    if name == "cspan_slabs":
        return residual_cspan_slabs()
    if name == "cspan_n800":
        return residual_cspan_n800()
    if name == "r1_origin":
        return residual_r1_origin()
    if name == "r1_o_quarter":
        return residual_r1_o_quarter()
    if name == "r1_o_eighth":
        return residual_r1_o_eighth()
    if name == "ospan_join":
        return residual_ospan_join()
    if name == "ospan_slabs":
        return residual_ospan_slabs()
    if name == "ospan_n16":
        return residual_ospan_n16()
    if name == "oiface_join":
        return residual_oiface_join()
    if name == "oiface_slabs":
        return residual_oiface_slabs()
    if name == "oiface_n32":
        return residual_oiface_n32()
    if name == "onear_join":
        return residual_onear_join()
    if name == "onear_slabs":
        return residual_onear_slabs()
    if name == "onear_n64":
        return residual_onear_n64()
    if name == "ox32_join":
        return residual_ox32_join()
    if name == "ox32_slabs":
        return residual_ox32_slabs()
    if name == "ox32_n128":
        return residual_ox32_n128()
    if name == "cmp_emax":
        return residual_cmp_emax()
    if name == "cmp_xfar":
        return residual_cmp_xfar()
    if name == "cmp_phases":
        return residual_cmp_phases()
    if name == "unif_neck":
        return residual_unif_neck()
    if name == "unif_phases":
        return residual_unif_phases()
    if name == "unif_time":
        return residual_unif_time()
    if name == "iface_edge":
        return residual_iface_edge()
    if name == "iface_phases":
        return residual_iface_phases()
    if name == "iface_time":
        return residual_iface_time()
    if name == "spre_tail":
        return residual_spre_tail()
    if name == "spre_chi":
        return residual_spre_chi()
    if name == "spre_net":
        return residual_spre_net()
    if name == "off_wall":
        return residual_off_wall()
    if name == "off_u":
        return residual_off_u()
    if name == "off_chi":
        return residual_off_chi()
    if name == "ray_x":
        return residual_ray_x()
    if name == "ray_coeff":
        return residual_ray_coeff()
    if name == "ray_chi":
        return residual_ray_chi()
    if name == "near_wall":
        return residual_near_wall()
    if name == "near_u":
        return residual_near_u()
    if name == "near_chi":
        return residual_near_chi()
    if name == "open_wall":
        return residual_open_wall()
    if name == "open_chi":
        return residual_open_chi()
    if name == "open_gap":
        return residual_open_gap()
    if name == "fold_dx":
        return residual_fold_dx(Fraction(1), Fraction(3))
    if name == "fold_d2x":
        return residual_fold_d2x(Fraction(1), Fraction(3))
    if name == "fold_log_d1":
        return residual_fold_log_d1(Fraction(1), Fraction(3))
    if name == "fold_log_d2":
        return residual_fold_log_d2(Fraction(1), Fraction(3))
    if name == "fold_reciprocal":
        return residual_fold_reciprocal(Fraction(3), Fraction(3))
    if name == "fold_reciprocal_log_c2":
        return residual_fold_reciprocal_log_c2(Fraction(3), Fraction(3))
    if name == "relative_remainder":
        return residual_relative_remainder(
            Fraction(2), Fraction(1, 4), Fraction(3), Fraction(19, 4)
        )
    if name == "interface_relative":
        return residual_interface_relative(Fraction(2), Fraction(1), Fraction(3), Fraction(4))
    if name == "zeta_rho":
        return residual_zeta_rho(
            Fraction(2), Fraction(9, 4), Fraction(-3), Fraction(1, 5), Fraction(1, 3), Fraction(7, 2)
        )
    if name == "lifted_dx":
        return residual_lifted_dx(Fraction(2), Fraction(3), Fraction(1, 4))
    if name == "lifted_d2x":
        return residual_lifted_d2x(Fraction(2), Fraction(3), Fraction(1, 4))
    if name == "lifted_log_d1":
        return residual_lifted_log_d1(Fraction(2), Fraction(3), Fraction(1, 4))
    if name == "lifted_log_c2":
        return residual_lifted_log_c2(Fraction(2), Fraction(3), Fraction(1, 4))
    if name == "z_relative_on_lift":
        return residual_z_relative_on_lift(Fraction(1, 5), Fraction(1, 3), Fraction(7, 2))
    if name == "chi_log_second":
        return residual_chi_log_second(
            Fraction(2), Fraction(1, 5), Fraction(3, 2), Fraction(4), Fraction(1, 3)
        )
    if name == "two_root_dx":
        return residual_two_root_dx(Fraction(2), Fraction(1, 5), Fraction(4))
    if name == "r1_limit_dx":
        return residual_r1_limit_dx(Fraction(2), Fraction(1, 5), Fraction(4))
    if name == "c_limit":
        return residual_c_limit(Fraction(1, 5), Fraction(4))
    if name == "rescaled_b":
        return residual_rescaled_b(Fraction(2), Fraction(1, 5), Fraction(4))
    if name == "xi_limit":
        return residual_xi_limit(Fraction(2), Fraction(1, 5), Fraction(4))
    if name == "v0_quadratic":
        return residual_v0_quadratic(Fraction(5, 16), Fraction(4, 5))
    if name == "sqrt_inverse":
        nu, v = Fraction(5, 16), Fraction(1, 2)
        return residual_sqrt_inverse(nu, v, Fraction(1) - v - nu * v**2)
    if name == "V_factor":
        nu, v, v0 = Fraction(5, 16), Fraction(1, 2), Fraction(4, 5)
        return residual_V_factor(nu, v, v0, Fraction(1) - v - nu * v**2)
    if name == "zeta_at_zero":
        return residual_zeta_at_zero(Fraction(5, 16), Fraction(4, 5))
    if name == "limiting_cubic":
        return residual_limiting_cubic(Fraction(2))
    if name == "k_lambda0":
        return residual_k_lambda0(Fraction(5, 16), Fraction(4, 5))
    if name == "beta_linear_jet":
        return residual_beta_linear_jet(Fraction(5, 16), Fraction(4, 5))
    if name == "zeta_plus_one_canceled":
        return residual_zeta_plus_one_canceled(
            Fraction(5, 16), Fraction(1, 2), Fraction(4, 5)
        )
    if name == "k_implicit":
        return residual_k_implicit(
            Fraction(5, 16), Fraction(4, 5), Fraction(2), Fraction(0), Fraction(288, 125)
        )
    if name == "field_matches_closed":
        return residual_field_matches_closed(
            Fraction(5, 16), Fraction(1, 2), Fraction(4, 5)
        )
    if name == "c0_jet":
        return residual_c0_jet(
            Fraction(5, 16), Fraction(4, 5), Fraction(2), Fraction(1, 7), Fraction(288, 125)
        )
    if name == "fold_lambda":
        return residual_fold_lambda(Fraction(3, 2), Fraction(-3))
    if name == "fold_L":
        return residual_fold_L(Fraction(3, 2), Fraction(9, 4))
    if name == "fold_disc":
        return residual_fold_disc(Fraction(9, 4), Fraction(-3))
    if name == "fold_bminus":
        return residual_fold_bminus(Fraction(2), Fraction(3, 2))
    if name == "lift_cubic_mismatch":
        return residual_lift_cubic_mismatch(
            Fraction(2), Fraction(3, 2), Fraction(1, 3), Fraction(1, 5)
        )
    if name == "frozen_log_d1_gap":
        return residual_frozen_log_d1_gap(
            Fraction(2), Fraction(3, 2), Fraction(1, 4), Fraction(1, 5), Fraction(7, 2)
        )
    if name == "frozen_c2_gap":
        return residual_frozen_c2_gap(
            Fraction(2), Fraction(3, 2), Fraction(1, 4), Fraction(1, 5), Fraction(7, 2)
        )
    if name == "unfrozen_log_d1_gap":
        return residual_unfrozen_log_d1_gap(
            Fraction(2),
            Fraction(3, 2),
            Fraction(1, 4),
            Fraction(1, 5),
            Fraction(7, 2),
            Fraction(5, 2),
        )
    if name == "unfrozen_recovers_frozen":
        return residual_unfrozen_recovers_frozen(
            Fraction(2), Fraction(3, 2), Fraction(1, 4), Fraction(1, 5), Fraction(7, 2)
        )
    if name == "zx_extra":
        return residual_zx_extra(
            Fraction(2),
            Fraction(3, 2),
            Fraction(1, 4),
            Fraction(1, 5),
            Fraction(7, 2),
            Fraction(5, 2),
        )
    if name == "zx_chain":
        return residual_zx_chain(Fraction(5, 16), Fraction(1, 2), Fraction(4, 5))
    if name == "zx_declared":
        return residual_zx_declared()
    if name == "fold_x_interior":
        return residual_fold_x_interior()
    if name == "q1_v":
        return residual_q1_v(Fraction(5, 16), Fraction(1, 2), Fraction(4, 5))
    if name == "zv_quotient":
        return residual_zv_quotient(Fraction(5, 16), Fraction(1, 2), Fraction(4, 5))
    if name == "wall_ratio_v":
        return residual_wall_ratio_v(Fraction(5, 16), Fraction(1, 2), Fraction(4, 5))
    if name == "V_v_slow":
        return residual_V_v_slow(Fraction(5, 16), Fraction(1, 2))
    if name == "ZV_chain":
        return residual_ZV_chain(Fraction(5, 16), Fraction(1, 2), Fraction(4, 5))
    if name == "ell_min":
        return residual_ell_min()
    if name == "J_numerator":
        return residual_J_numerator(Fraction(1), Fraction(1, 5), Fraction(4))
    if name == "root_sum":
        return residual_root_sum(Fraction(1, 5), Fraction(4), Fraction(-21, 5))
    if name == "interface_embed":
        r1, theta, eps = Fraction(1, 5), Fraction(1, 2), Fraction(1, 7)
        x_lo = r1 * (1 + theta)
        return residual_interface_embed(-eps * x_lo, eps, r1, theta, x_lo)
    if name == "V_embed":
        xstar, eps = Fraction(1), Fraction(1, 7)
        return residual_V_embed(-eps * xstar, eps, xstar)
    if name == "T_kinetic":
        xstar, eps = Fraction(1), Fraction(1, 7)
        return residual_T_kinetic((eps * xstar) ** 2 / 2, eps, xstar)
    if name == "h_exit":
        eps, y0 = Fraction(1, 7), Fraction(8)
        return residual_h_exit(eps**3 * y0, eps, y0)
    if name == "q_leading":
        r1, r2 = Fraction(1, 5), Fraction(4)
        return residual_q_leading(r1 * r2, -(r1 + r2), Fraction(1), r1, r2)
    if name == "amgm_qbound":
        eps = Fraction(1, 16)
        return residual_amgm_qbound(eps, 2 * eps)
    if name == "envelope_exit":
        xstar, eps, y0 = Fraction(1), Fraction(1, 16), Fraction(6)
        t_exit = (eps * xstar) ** 2 / 2
        h_exit = eps**3 * y0
        return residual_envelope_exit(t_exit, eps, h_exit, xstar, y0)
    if name == "th_alpha0":
        xstar, eps, y0, hmax = Fraction(1), Fraction(1, 16), Fraction(6), Fraction(1)
        t_exit = (eps * xstar) ** 2 / 2
        h_exit = eps**3 * y0
        return residual_th_alpha0(t_exit + hmax - h_exit, hmax, t_exit, h_exit)
    if name == "q_envelope":
        r1, r2, xstar, eps = Fraction(1, 5), Fraction(4), Fraction(1), Fraction(1, 16)
        t_exit = (eps * xstar) ** 2 / 2
        q_abs = eps**3 * (xstar - r1) * (r2 - xstar)
        return residual_q_envelope(q_abs, eps, t_exit, xstar, r1, r2)
    if name == "ratio_gap":
        r1, r2 = Fraction(1, 5), Fraction(9, 5)
        return residual_ratio_gap(Fraction(1), r1, r2, Fraction(-2), r1 * r2)
    if name == "ratio_disc":
        r1, r2 = Fraction(1, 5), Fraction(9, 5)
        return residual_ratio_disc(r1, r2, Fraction(-2), r1 * r2)
    if name == "kill_square":
        return residual_kill_square(Fraction(1), Fraction(9, 25))
    if name == "ratio_outer":
        r1, r2 = Fraction(1, 5), Fraction(9, 5)
        return residual_ratio_outer(Fraction(3), r1, r2, r1 * r2)
    if name == "k_turn":
        nu, v, a, c = Fraction(1, 16), Fraction(1, 2), Fraction(1), Fraction(0)
        kay = 1 - a * nu * v + c * nu**2 * v * v
        return residual_k_turn(kay, a, nu, v, c)
    if name == "k_normal":
        return residual_k_normal(Fraction(1), Fraction(1, 16), Fraction(1, 2))
    if name == "k_lead_embed":
        return residual_k_lead_embed(Fraction(1, 16), Fraction(1, 2))
    if name == "k_normal_vs_lead":
        return residual_k_normal_vs_lead(Fraction(1, 16), Fraction(1, 2))
    if name == "k_box_gap":
        return residual_k_box_gap(Fraction(2))
    if name == "cubic_vs_tworoot":
        r1, r2 = Fraction(1, 5), Fraction(9, 5)
        return residual_cubic_vs_tworoot(
            r1 * r2, -(r1 + r2), Fraction(1, 16), Fraction(1), r1, r2
        )
    if name == "relative_prefactor":
        nu = Fraction(1, 16)
        return residual_relative_prefactor(nu, nu)
    if name == "kill_lambda":
        return residual_kill_lambda(Fraction(-2))
    if name == "kill_product":
        return residual_kill_product(Fraction(9, 25), Fraction(1, 5))
    if name == "kill_sum":
        return residual_kill_sum(Fraction(1, 5), Fraction(9, 5))
    if name == "kill_disc_gap":
        return residual_kill_disc_gap(Fraction(-2), Fraction(9, 25))
    if name == "kill_tworoot":
        r1, r2 = Fraction(1, 5), Fraction(9, 5)
        return residual_kill_tworoot(Fraction(1), r1, r2, r1 * r2)
    if name == "delta_slow":
        return residual_delta_slow(
            Fraction(5, 16),
            Fraction(1, 2),
            Fraction(4, 5),
            Fraction(2),
            Fraction(0),
            Fraction(288, 125),
        )
    if name == "ell_V_nu":
        return residual_ell_V_nu(Fraction(5, 16), Fraction(1, 2), Fraction(4, 5))
    if name == "t_lambda":
        return residual_t_lambda(
            Fraction(5, 16),
            Fraction(1, 2),
            Fraction(4, 5),
            Fraction(2),
            Fraction(288, 125),
        )
    if name == "t_L":
        return residual_t_L(
            Fraction(5, 16),
            Fraction(1, 2),
            Fraction(4, 5),
            Fraction(2),
            Fraction(0),
        )
    if name == "n_cancelled":
        return residual_n_cancelled(
            Fraction(5, 16),
            Fraction(1, 2),
            Fraction(4, 5),
            Fraction(2),
            Fraction(0),
            Fraction(288, 125),
        )
    if name == "z0_holomorphic":
        return residual_z0_holomorphic(
            Fraction(5, 16), Fraction(1, 2), Fraction(4, 5)
        )
    if name == "ell_mix":
        return residual_ell_mix(
            Fraction(1, 16), Fraction(1, 2), Fraction(2), Fraction(1, 4)
        )
    if name == "V_mix":
        return residual_V_mix(
            Fraction(1, 16), Fraction(1, 2), Fraction(2), Fraction(1, 4)
        )
    if name == "V_v_ell":
        return residual_V_v_ell(
            Fraction(1, 16), Fraction(1, 2), Fraction(2), Fraction(1, 4)
        )
    if name == "V_h_c":
        return residual_V_h_c(Fraction(1, 16), Fraction(1, 2), Fraction(2))
    if name == "g_lead":
        return residual_g_lead(Fraction(1, 16), Fraction(1, 3), Fraction(1, 4))
    if name == "th_split":
        return residual_th_split(
            Fraction(3, 16),
            Fraction(1, 4),
            Fraction(17, 16),
            Fraction(2),
            Fraction(1, 16),
            Fraction(1, 8),
        )
    if name == "th_touching":
        return residual_th_touching(
            Fraction(2),
            Fraction(1, 16),
            Fraction(1, 8),
            Fraction(1, 4),
            Fraction(17, 16),
        )
    if name == "th_dh":
        return residual_th_dh(Fraction(3, 16), Fraction(1, 4), Fraction(17, 16))
    if name == "majorant_split":
        return residual_majorant_split(
            Fraction(2), Fraction(2), Fraction(1, 16), Fraction(1, 4)
        )
    if name == "ftc_linear":
        return residual_ftc_linear(
            Fraction(1, 8), Fraction(1, 16), Fraction(1, 4), Fraction(1, 4)
        )
    if name == "sqrt_ratio_sq":
        return residual_sqrt_ratio_sq(
            Fraction(1, 16), Fraction(32), Fraction(4), Fraction(1)
        )
    if name == "log_sqrt_prefactor":
        return residual_log_sqrt_prefactor(
            Fraction(1, 16), Fraction(32), Fraction(4), Fraction(1)
        )
    if name == "q_plus_f":
        return residual_q_plus_f(
            Fraction(9, 25), Fraction(-2), Fraction(1, 16), Fraction(1, 16)
        )
    if name == "hdot_vh":
        v_coord = -Fraction(1, 16)
        height = Fraction(4) / Fraction(4096)
        return residual_hdot_vh(-v_coord * height, v_coord, height)
    if name == "g_jet_vh":
        nu = Fraction(1, 16)
        v_coord = -nu
        return residual_g_jet_vh(field_g(nu, v_coord), nu, v_coord)
    if name == "Vdot_split":
        L, lam1, eps = Fraction(9, 25), Fraction(-2), Fraction(1, 16)
        v_coord = -eps
        height = Fraction(4) * eps**3
        eff = field_f(L, lam1, eps, v_coord)
        gee = field_g(eps, v_coord)
        return residual_Vdot_split(eff + height * gee, eff, height, gee)
    if name == "e_sigma_nu0":
        v_coord = -Fraction(1, 16)
        height = Fraction(4) * Fraction(1, 16) ** 3
        return residual_e_sigma_nu0(
            v_coord, height, Fraction(1), Fraction(1, 4), Fraction(2)
        )
    if name == "e_out_lead":
        return residual_e_out_lead(-Fraction(1, 16), Fraction(1, 4), Fraction(2))
    if name == "section_embed":
        return residual_section_embed(Fraction(1, 16), Fraction(1, 4))
    if name == "e_out_group":
        v_coord = -Fraction(1, 16)
        height = Fraction(4) * Fraction(1, 16) ** 3
        return residual_e_out_group(
            v_coord, height, Fraction(1, 4), Fraction(1, 16), Fraction(2)
        )
    if name == "v0_matching":
        eps = Fraction(1, 16)
        return residual_v0_matching(-eps, eps)
    if name == "h0_matching":
        eps = Fraction(1, 16)
        return residual_h0_matching(4 * eps**3, eps)
    if name == "horizon_n2":
        return residual_horizon_n2(Fraction(16), Fraction(32))
    if name == "vdot_kill_init":
        eps = Fraction(1, 16)
        v0 = -eps
        h0 = 4 * eps**3
        vdot = field_f(Fraction(0), Fraction(-2), eps, v0) + h0 * field_g(eps, v0)
        return residual_vdot_kill_init(vdot, eps)
    if name == "F_expand":
        return residual_F_expand(Fraction(-1, 8), Fraction(1, 16))
    if name == "F_V_phi":
        return residual_F_V_phi(Fraction(-1, 8), Fraction(1, 16))
    if name == "phi_right":
        return residual_phi_right(Fraction(1, 16))
    if name == "T_cubic":
        return residual_T_cubic(Fraction(1, 16), Fraction(1, 4), Fraction(256))
    if name == "F_zero":
        return residual_F_zero(Fraction(1, 16))
    if name == "phi_zero":
        return residual_phi_zero(Fraction(1, 16))
    if name == "phi_zero_factor":
        return residual_phi_zero_factor(Fraction(1, 16))
    if name == "T_in":
        return residual_T_in(Fraction(1, 16), Fraction(16384, 17))
    if name == "v0_grazing":
        return residual_v0_grazing(Fraction(0))
    if name == "hdot_rev":
        return residual_hdot_rev(Fraction(1, 8), Fraction(1, 8), Fraction(1))
    if name == "vdot_rev_init":
        eps = Fraction(1, 16)
        v0 = Fraction(0)
        h0 = 4 * eps**3
        vdot_rev = -(field_f(Fraction(0), Fraction(-2), eps, v0) + h0 * field_g(eps, v0))
        return residual_vdot_rev_init(vdot_rev, eps)
    if name == "vin_wall":
        return residual_vin_wall(Fraction(1, 4), Fraction(1, 4))
    if name == "restart_v":
        return residual_restart_v(Fraction(3, 4))
    if name == "restart_h":
        return residual_restart_h(Fraction(1, 4))
    if name == "e_sigma_in_lead":
        return residual_e_sigma_in_lead(
            Fraction(3, 4), Fraction(1, 4), Fraction(-1), Fraction(1, 4), Fraction(2)
        )
    if name == "e_sigma_in_group":
        return residual_e_sigma_in_group(
            Fraction(3, 4),
            Fraction(1, 4),
            Fraction(-1),
            Fraction(1, 4),
            Fraction(1, 16),
            Fraction(2),
        )
    if name == "f_cubic_factor":
        return residual_f_cubic_factor(Fraction(1, 2), Fraction(1, 16))
    if name == "vdot_rev_split":
        return residual_vdot_rev_split(Fraction(1, 2), Fraction(1, 8), Fraction(1, 16))
    if name == "ratio_split":
        return residual_ratio_split(Fraction(1, 2), Fraction(1, 16))
    if name == "log_taylor_num":
        return residual_log_taylor_num(Fraction(6, 79))
    if name == "de_sigma_lo":
        return residual_de_sigma_lo(
            Fraction(55, 79), Fraction(79, 80), Fraction(1, 4), Fraction(6, 5)
        )
    if name == "I_cancel":
        return residual_I_cancel(Fraction(6, 5), Fraction(1, 16))
    if name == "I_limit":
        return residual_I_limit(Fraction(6, 5))
    if name == "E_eps0":
        vstar = Fraction(6, 5)
        return residual_E_eps0(
            vstar,
            vstar * vstar / 2,
            Fraction(-1),
            Fraction(1, 4),
            Fraction(2),
        )
    if name == "de_eps0":
        return residual_de_eps0(Fraction(1, 4), Fraction(6, 5), Fraction(7, 10))
    if name == "align_v":
        return residual_align_v(Fraction(1, 4))
    if name == "align_h":
        return residual_align_h(Fraction(1, 40))
    if name == "e_align_nu0":
        return residual_e_align_nu0(
            Fraction(1, 4), Fraction(1, 40), Fraction(-1), Fraction(1, 4)
        )
    if name == "e_align_start":
        return residual_e_align_start(
            Fraction(1, 4),
            Fraction(1, 40),
            Fraction(-1),
            Fraction(1, 4),
            Fraction(1, 16),
            Fraction(2),
        )
    if name == "cover_lo":
        return residual_cover_lo(Fraction(1, 50))
    if name == "cover_hi":
        return residual_cover_hi(Fraction(4, 125))
    if name == "cover_width":
        return residual_cover_width(Fraction(1, 50), Fraction(4, 125))
    if name == "cover_contains":
        return residual_cover_contains(
            Fraction(1, 40), Fraction(1, 50), Fraction(4, 125)
        )
    if name == "span_hi":
        return residual_span_hi(Fraction(40, 1000))
    if name == "span_width":
        return residual_span_width(Fraction(19, 1000), Fraction(1, 25))
    if name == "span_align_lo":
        return residual_span_align_lo(Fraction(1, 40), Fraction(19, 1000))
    if name == "span_align_hi":
        return residual_span_align_hi(Fraction(1, 25), Fraction(1, 40))
    if name == "pack_lo":
        return residual_pack_lo(Fraction(17, 1000))
    if name == "pack_hi":
        return residual_pack_hi(Fraction(35, 1000))
    if name == "pack_width":
        return residual_pack_width(Fraction(17, 1000), Fraction(7, 200))
    if name == "pack_align":
        return residual_pack_align(Fraction(1, 40), Fraction(17, 1000))
    if name == "sigma_pack_sum":
        return residual_sigma_pack_sum(Fraction(1, 16), Fraction(1, 20), Fraction(1, 25))
    if name == "sigma_h0_16":
        return residual_sigma_h0_16(Fraction(1, 1024))
    if name == "sigma_h0_20":
        return residual_sigma_h0_20(Fraction(1, 2000))
    if name == "sigma_horizon":
        return residual_sigma_horizon(Fraction(80), Fraction(1, 8))
    if name == "oneshot_h0":
        return residual_oneshot_h0(Fraction(1, 1024))
    if name == "oneshot_T":
        return residual_oneshot_T(Fraction(280), Fraction(1, 4))
    if name == "oneshot_short":
        return residual_oneshot_short(Fraction(200), Fraction(1, 4))
    if name == "oneshot_pack_L":
        return residual_oneshot_pack_L(Fraction(9, 25), Fraction(1, 16))
    if name == "oeps_pack_sum":
        return residual_oeps_pack_sum(Fraction(1, 16), Fraction(1, 20), Fraction(1, 25))
    if name == "oeps_T20":
        return residual_oeps_T20(Fraction(400), Fraction(1, 4))
    if name == "oeps_T25":
        return residual_oeps_T25(Fraction(1000), Fraction(1, 4))
    if name == "oeps_h0_25":
        return residual_oeps_h0_25(Fraction(4, 15625))
    if name == "espan_lo":
        return residual_espan_lo(Fraction(1, 25))
    if name == "espan_width":
        return residual_espan_width(Fraction(1, 25), Fraction(1, 16))
    if name == "espan_contains":
        return residual_espan_contains(Fraction(1, 20), Fraction(1, 25), Fraction(1, 16))
    if name == "espan_pack_L":
        return residual_espan_pack_L(Fraction(9, 25), Fraction(1, 16))
    if name == "elo_lo":
        return residual_elo_lo(Fraction(1, 64))
    if name == "elo_width":
        return residual_elo_width(Fraction(1, 64), Fraction(1, 16))
    if name == "elo_contains_32":
        return residual_elo_contains_32(Fraction(1, 32), Fraction(1, 64), Fraction(1, 16))
    if name == "elo_pack_L":
        return residual_elo_pack_L(Fraction(9, 25), Fraction(1, 16))
    raise KeyError(name)


def identity_verdict(name: str) -> Any:
    residual = _sample(name)
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False)


@dataclass
class Hilbert16IdentityFamily:
    """Finite box of named exact identities. Parent remains open."""

    name: str = HILBERT16_IDENTITY_KIND
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="hilbert16_coalescence_identities",
            obligation=(
                "exact double-root, W-height, outgoing W-ratio, shrinking-root, "
                "sigma-kappa, blow-up height, event-leading, joint-axis, "
                "slow-line partial-fraction, tracked-product, Stage-B "
                "height-inflation, Stage-A wall, chi_b threshold, dx_e leading, "
                "uniform-in-chi dx_e, Stage-C a_min, Stage-C exit, Stage-C T_h, Stage-C start-gap, Stage-C C=0 envelope, Stage-C C=2 integrating factor, Stage-C C=2 T(h) majorant, Stage-C C=2 lower T(h) envelope, Stage-C C=2 tight T(h) ratio, Stage-C C=2 T-h bootstrap, Stage-C continuation rectangle, Stage-C comparison first-hit of h=1, Stage-C comparison first-hit of E_out, Stage-C Lohner first-hit of matching-chart x=4, shrinking-eps Stage-C Lohner pack, parametric-eps Stage-C Lohner cover, chart-O Stage-C Lohner pack, parametric-sep chart-O Lohner cover, nearer-interface chart-O Lohner cover, x=1/16 chart-O Lohner cover, x=1/32 chart-O Lohner cover, uniform-r1 comparison first-hit on an eps slab, uniform comparison first-hit for every eps in (0, 1/16], interface comparison first-hit from every start in (0, 1/2], kill-line sep*S_pre on every sep in (0, 1], dx_e on lambda1 in [-4, -2], dx_e for every lambda1 <= -2, dx_e on lambda1 in [-3/2, -2), dx_e for lambda1 in (-3/2, 0), fold I-map, "
                "inner/outer remainder, shrinking-root I-map, canonical "
                "zeta, fold-compact disc, frozen-Z C2, unfrozen-Z "
                "Z_x first-log-derivative, matching-chart fold Z_x, holomorphic Z_v, slow-line "
                "Z_V chain, outgoing-corridor, "
                "and post-corridor V/T/h/q, height-envelope, C=2 "
                "|q|-ratio, k=1+O(eps)/cubic-prefactor, kill-compact "
                "Z, cancelled-N holomorphic Z, C!=0 height-mixing, "
                "C=0 T_h-gap, comparison-bootstrap T-h-integral, "
                "cubic (V,h)-orbit, matching-chart E_out, "
                "shrinking-eps E_out-pack, kill-line comparison "
                "speed, incoming GRAZING comparison-speed, incoming "
                "V=1/4 first-hit, declared-point E_sigma first-hit, "
                "and comparison GRAZING E_sigma-from-0, uniform "
                "cancelled-height E_sigma, orbit-aligned wall "
                "E_sigma, wall-box h-interval E_sigma-cover, "
                "L=0 whole-wall h-span E_sigma, L-pack wall-span "
                "E_sigma, shrinking-eps aligned E_sigma, "
                "one-shot Lohner E_sigma-from-0, shrinking-eps "
                "one-shot E_sigma, compact aligned parametric-eps "
                "E_sigma, and lower aligned parametric-eps E_sigma "
                "identities over Q"
            ),
            parent="Hilbert XVI",
            parent_status="open",
            existential=True,
        )
    )

    def cardinality(self) -> int:
        return len(IDENTITY_NAMES)

    def origin(self) -> str:
        return IDENTITY_NAMES[0]

    def neighbors(self, candidate: Candidate) -> Sequence[str]:
        if candidate not in IDENTITY_NAMES:
            return ()
        index = IDENTITY_NAMES.index(candidate)  # type: ignore[arg-type]
        out: list[str] = []
        if index > 0:
            out.append(IDENTITY_NAMES[index - 1])
        if index + 1 < len(IDENTITY_NAMES):
            out.append(IDENTITY_NAMES[index + 1])
        return out

    def score(self, candidate: Candidate) -> int:
        return 0 if candidate in IDENTITY_NAMES else -1

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if candidate not in IDENTITY_NAMES:
            return None
        residual = _sample(str(candidate))
        ok = residual == 0
        verdict = identity_verdict(str(candidate))
        return ExactCheck(
            ok=ok,
            payload={
                "identity": candidate,
                "residual": [residual.numerator, residual.denominator],
                "verdict": verdict.status,
                "honesty": _honesty(ok=ok),
            },
        )


def replay_hilbert16_identities(*, budget: int = 290) -> Any:
    family: FiniteFamily = Hilbert16IdentityFamily()
    return run_discovery(family.statement, family, "score_guided", budget=budget, collect=True)


register_catalog(
    CatalogEntry(
        kind=HILBERT16_IDENTITY_KIND,
        obligation="exact coalescence identities over Q",
        parent="Hilbert XVI",
        parent_status="open",
        package="omnibias-dynamics",
        mode="exact_replay",
        complete=True,
        existential=True,
    ),
    replay_hilbert16_identities,
)
