# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Finite Hilbert XVI chart-cell ledger and one Maletto combinatorial type.

A complete list of labels is not G1, G4, or Hilbert XVI. Verdicts apply to
exact identities only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

CellStatus = str


@dataclass(frozen=True)
class ChartCell:
    name: str
    regime: str
    first_hit: str
    remainder_order: int
    overlaps: tuple[str, ...]
    identity_verdict: str
    g1_item_closed: bool


def _honesty() -> dict[str, bool]:
    return {
        "g1_passed": False,
        "g4_passed": False,
        "full_graphic_cyclicity_proved": False,
        "full_hilbert16_solved": False,
    }


def chart_cells() -> tuple[ChartCell, ...]:
    """Named cells of the current atlas, including recorded kill sequences."""

    return (
        ChartCell("N", "Delta >= chi_N > 0", "selected tube", 2, ("F",), "PROVED", False),
        ChartCell("F", "sep >= sep0, L >= Lmin", "selected tube", 2, ("N", "D", "stage_b", "stage_a", "chi_b", "dx_e_leading", "dx_e_unif", "stage_c", "stage_c_exit", "stage_c_th", "stage_c_gap", "stage_c_env", "stage_c_if", "stage_c_int", "stage_c_lo", "stage_c_k", "stage_c_boot", "stage_c_rect", "stage_c_hit", "stage_c_sec", "stage_c_oneshot", "stage_c_oneshot_eps", "stage_c_eps_span", "stage_c_origin", "stage_c_origin_span", "stage_c_origin_iface", "stage_c_origin_near", "stage_c_origin_x32", "stage_c_compare", "stage_c_uniform", "stage_c_interface", "sep_spre", "dx_e_off", "dx_e_ray", "dx_e_near", "dx_e_open"), "PROVED", False),
        ChartCell(
            "D",
            "epsilon |log sep| <= 1, L >= Lmin",
            "selected tube",
            1,
            ("F", "WF", "weighted_section"),
            "PROVED",
            False,
        ),
        ChartCell(
            "C",
            "sep = 0, L >= Lmin",
            "leading I-map; remainder versus field open",
            0,
            ("WF", "fold_imap", "canonical_zeta", "fold_zeta", "weighted_section"),
            "PROVED",
            False,
        ),
        ChartCell(
            "O",
            "L -> 0, lambda1 <= -lmin",
            "incoming; rescaled I-map; outgoing collision",
            0,
            ("kill_shrinking_root", "weighted_section", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell("WF", "W-fold, sigma = sqrt(epsilon)", "selected tube", 0, ("D", "WS", "LI"), "PROVED", False),
        ChartCell("WS", "W-separation, chi = O(1)", "selected tube", 0, ("WF", "D", "WL"), "PROVED", False),
        ChartCell(
            "LI",
            "tau = epsilon log(1/sep)",
            "selected tube",
            0,
            ("WF", "WS", "D"),
            "PROVED",
            False,
        ),
        ChartCell(
            "WL",
            "Lambda = epsilon log(1/W)",
            "selected tube",
            0,
            ("WS", "WF"),
            "PROVED",
            False,
        ),
        ChartCell(
            "cellA",
            "LN/exp candidate, tau = epsilon log(1/sep)",
            "incoming only; physical LN membership open",
            0,
            ("cellAB", "WS", "WL"),
            "BLOCKED",
            False,
        ),
        ChartCell(
            "cellB",
            "LN/exp candidate, shrinking r1",
            "incoming only; proposed wall eventually negative",
            0,
            ("cellAB", "O"),
            "DISPROVED",
            False,
        ),
        ChartCell(
            "cellAB",
            "coordinate overlap of cellA and cellB",
            "coordinate identity only",
            0,
            ("cellA", "cellB"),
            "BLOCKED",
            False,
        ),
        ChartCell(
            "kill_super_small_sep",
            "sep = exp(-1/epsilon^2), lambda1 = -3, L = (9-sep^2)/4",
            "admitted; tracked first-derivative product; continuation remainder open",
            0,
            ("product_bound",),
            "PROVED",
            False,
        ),
        ChartCell(
            "product_bound",
            "sep^2 * (h_1 / (eps^3 mu sep^2))^(C eps) = sep^(2-2 C eps) * (h_1/(eps^3 mu))^(C eps)",
            "algebraic product; kill-line Stage-B height inflation sealed",
            0,
            ("kill_super_small_sep", "WS", "F", "stage_b", "stage_a", "chi_b", "dx_e_leading", "dx_e_unif", "stage_c", "stage_c_exit", "stage_c_th", "stage_c_gap", "stage_c_env", "stage_c_if", "stage_c_int", "stage_c_lo", "stage_c_k", "stage_c_boot", "stage_c_rect", "stage_c_hit", "stage_c_sec", "stage_c_oneshot", "stage_c_oneshot_eps", "stage_c_eps_span", "stage_c_origin", "stage_c_origin_span", "stage_c_origin_iface", "stage_c_origin_near", "stage_c_origin_x32", "stage_c_compare", "stage_c_uniform", "stage_c_interface", "sep_spre", "dx_e_off", "dx_e_ray", "dx_e_near", "dx_e_open"),
            "PROVED",
            False,
        ),
        ChartCell(
            "weighted_section",
            "intrinsic eta-section h=eps^3 sep^2 eta0",
            "transverse for sep>0; q-derivatives singular at D-C and speed vanishes at O",
            0,
            ("D", "C", "O", "kill_super_small_sep"),
            "DISPROVED",
            False,
        ),
        ChartCell(
            "quasihomogeneous_dichotomy",
            "single scale sigma=eps^a sep^b against frozen h=eps^N",
            "all monomial weights excluded; moving sections and multistage atlases not excluded",
            0,
            ("weighted_section", "kill_super_small_sep", "WF", "WS"),
            "PROVED",
            False,
        ),
        ChartCell(
            "ln_format_barrier",
            "direct tau/log-W LN chain on finite kill-sequence truncations",
            "chain length and coefficients fixed; outer radius and sup norm diverge",
            0,
            ("cellA", "kill_super_small_sep", "quasihomogeneous_dichotomy"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_b",
            "kill-line Stage-B dx/dy = eps/x; Picard |Delta x|<1/3 on sep in (0, 1], eps in [0, 1/16]",
            "height inflation enclosed; Stage A wall is stage_a; Stage C a_min is stage_c; not C2",
            0,
            ("product_bound", "F", "kill_shrinking_root", "stage_a", "stage_c"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_a",
            "kill-line Stage-A walls a = r1 - theta sep, B_-(a) = theta(1+theta) sep^2 at theta=1/8",
            "wall identities plus Interval a>1/4 and Psi_pre factor <1/4; chi_b sealed; dx_e leading sealed; uniform-in-chi dx_e sealed; not Stage C",
            0,
            ("stage_b", "F", "product_bound", "chi_b", "dx_e_leading", "dx_e_unif"),
            "PROVED",
            False,
        ),
        ChartCell(
            "chi_b",
            "kill-line chi_b = 4/r1; two-slab sep*S_pre < 3 on sep in [1/2^16, 1]",
            "chi threshold enclosed; dx_e leading sealed; uniform-in-chi dx_e sealed; not Stage C",
            0,
            ("stage_a", "F", "product_bound", "dx_e_leading", "dx_e_unif", "sep_spre", "dx_e_off", "dx_e_ray", "dx_e_near", "dx_e_open"),
            "PROVED",
            False,
        ),
        ChartCell(
            "dx_e_leading",
            "kill-line dx_e prefactor <1/2 and threshold net exponent >1/8 after y0 log remainder",
            "leading factors enclosed; uniform-in-chi dx_e is dx_e_unif; not Stage C",
            0,
            ("stage_a", "chi_b", "F", "dx_e_unif", "sep_spre", "dx_e_off", "dx_e_ray", "dx_e_near", "dx_e_open"),
            "PROVED",
            False,
        ),
        ChartCell(
            "dx_e_unif",
            "kill-line dx_e <= C sep^2 exp(-(3/32) chi) with C<2 on the chi_b compact",
            "uniform-in-chi majorant enclosed; Stage C a_min is stage_c; not an outgoing orbit",
            0,
            ("dx_e_leading", "chi_b", "stage_a", "F", "stage_c"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c",
            "kill-line Stage-C a_min=1/4 after Stage B; integrating factor 1/x<8",
            "geometric floor enclosed; Stage-C exit energy is stage_c_exit; not first-hit",
            0,
            ("stage_b", "F", "dx_e_unif", "stage_c_exit"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_exit",
            "kill-line Stage-C T_e/eps^2 in (1/16, 1) and T_e > h_e on the Stage-B end box",
            "exit energy enclosed; Stage-C T_h is stage_c_th; not first-hit",
            0,
            ("stage_c", "F", "stage_b", "stage_c_th"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_th",
            "kill-line Stage-C leading T_h>1/2 on the Stage-B end box at y_1=1",
            "leading T_h floor enclosed; Stage-C start gap is stage_c_gap; not first-hit",
            0,
            ("stage_c_exit", "F", "stage_c", "stage_c_gap"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_gap",
            "kill-line Stage-C (T_e-h_1)/eps^2>1/32 at y_1=1 on the Stage-B end box",
            "start gap enclosed at actual Stage C height; C=0 T envelope is stage_c_env; not first-hit",
            0,
            ("stage_c_th", "F", "stage_c_exit", "stage_c_env"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_env",
            "kill-line Stage-C C=0 sandwich (1/32)(eps^2+h)<=T(h)<=eps^2+h",
            "C=0 two-sided T envelope enclosed; C=2 integrating factor is stage_c_if; not first-hit",
            0,
            ("stage_c_gap", "F", "stage_c_th", "stage_c_if"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_if",
            "kill-line Stage-C C=2 integrating factor (h/h_1)^{C eps}<32; exponent <= 3",
            "C=2 factor enclosed; C=2 T(h) majorant is stage_c_int; not first-hit",
            0,
            ("stage_c_env", "F", "stage_c_gap", "stage_c_int"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_int",
            "kill-line Stage-C C=2 T(h)<=64(eps^2+h); slope 16/7 at the compact edge",
            "C=2 T(h) majorant enclosed; C=2 lower envelope is stage_c_lo; not first-hit",
            0,
            ("stage_c_if", "F", "stage_c_exit", "stage_c_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_lo",
            "kill-line Stage-C C=2 T(h)>=(1/32)(eps^2+h); start remainder > 1/16",
            "C=2 lower T(h) envelope enclosed; C=2 tight ratio is stage_c_k; not first-hit",
            0,
            ("stage_c_int", "F", "stage_c_th", "stage_c_exit", "stage_c_k"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_k",
            "kill-line Stage-C C=2 T(h)<=6(eps^2+h); edge factor < 3",
            "C=2 tight T(h) ratio enclosed; C=2 T-h bootstrap is stage_c_boot; not first-hit",
            0,
            ("stage_c_lo", "F", "stage_c_int", "stage_c_boot"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_boot",
            "kill-line Stage-C C=2 T-h<1 at K=6; linear 15 eps, log 42 eps^3 ln(1/eps)",
            "C=2 T-h bootstrap enclosed; continuation rectangle is stage_c_rect; not first-hit",
            0,
            ("stage_c_k", "F", "stage_c_gap", "stage_c_rect"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_rect",
            "kill-line Stage-C V in [-2, -1/64] on h in [h_1, 1]; left wall sqrt(4)=2",
            "continuation rectangle enclosed; comparison first-hit of h=1 is stage_c_hit; not signed-label section",
            0,
            ("stage_c_boot", "F", "stage_c", "stage_c_hit"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_hit",
            "kill-line Stage-C comparison first-hit of h=1; time <= 192 ln(16)",
            "comparison first-hit of h=1 enclosed; E_out from Stage-C start is stage_c_sec; not Lohner or chart O",
            0,
            ("stage_c_rect", "F", "stage_c", "stage_c_sec"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_sec",
            "kill-line Stage-C comparison first-hit of E_out; start gap 1/6, dE/dh room 59/256",
            "comparison first-hit of E_out from Stage-C start; Lohner from Stage-C start is stage_c_oneshot; not chart O",
            0,
            ("stage_c_hit", "F", "stage_c_th", "stage_c_exit", "stage_c_oneshot"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_oneshot",
            "kill-line Stage-C Lohner first-hit of matching-chart x=4 from (x,y)=(1/4,1); sep in {0, 3/5, 1}",
            "unique transverse first-hit at eps=1/16; shrinking-eps pack is stage_c_oneshot_eps; not every eps or chart O",
            0,
            ("stage_c_sec", "F", "stage_c_th", "stage_c_exit", "stage_c_oneshot_eps"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_oneshot_eps",
            "kill-line Stage-C shrinking-eps Lohner pack n in {16, 20, 25}; x=n/4 from (x,y)=(1/4,1)",
            "unique transverse first-hit at three squares; parametric-eps cover is stage_c_eps_span; not every eps or chart O",
            0,
            ("stage_c_oneshot", "F", "stage_c_th", "stage_c_exit", "stage_c_eps_span"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_eps_span",
            "kill-line Stage-C parametric-eps Lohner cover of [23/400, 1/16]; four 1/800 slabs",
            "unique transverse first-hit on a declared compact containing 1/16; not every eps or chart O",
            0,
            ("stage_c_oneshot_eps", "F", "stage_c_th", "stage_c_exit", "stage_c_origin"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_origin",
            "kill-line matching-chart Lohner pack toward chart O; sep in {3/2, 7/4, 2}",
            "unique transverse first-hit at r1 in {1/4, 1/8, 0} from compact x=1/4; not every r1 or complete first-hit on chart O",
            0,
            ("stage_c_eps_span", "F", "stage_c_th", "stage_c_exit", "stage_c_origin_span"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_origin_span",
            "kill-line parametric-sep matching-chart Lohner cover of [3/2, 2]; eight 1/16 slabs",
            "unique transverse first-hit on a declared sep compact containing sep=2; not every r1 or complete first-hit on chart O",
            0,
            ("stage_c_origin", "F", "stage_c_th", "stage_c_exit", "stage_c_origin_iface"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_origin_iface",
            "kill-line nearer-interface matching-chart Lohner cover of [7/4, 2] from x=1/8; eight 1/32 slabs",
            "unique transverse first-hit on a declared sep compact containing sep=2 from x=1/8; not every r1 or complete first-hit on chart O",
            0,
            ("stage_c_origin_span", "F", "stage_c_th", "stage_c_exit", "stage_c_origin_near"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_origin_near",
            "kill-line nearer-interface matching-chart Lohner cover of [15/8, 2] from x=1/16; eight 1/64 slabs",
            "unique transverse first-hit on a declared sep compact containing sep=2 from x=1/16; not every r1 or complete first-hit on chart O",
            0,
            ("stage_c_origin_iface", "F", "stage_c_th", "stage_c_exit", "stage_c_origin_x32"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_origin_x32",
            "kill-line nearer-interface matching-chart Lohner cover of [31/16, 2] from x=1/32; eight 1/128 slabs",
            "unique transverse first-hit on a declared sep compact containing sep=2 from x=1/32; not every r1 or complete first-hit on chart O",
            0,
            ("stage_c_origin_near", "F", "stage_c_th", "stage_c_exit", "stage_c_compare"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_compare",
            "kill-line comparison first-hit for every r1 in [0, 1] and eps in [1/32, 1/16], out to x=8",
            "phase-wise Interval speed bound from (x,y)=(1/4,1); freezing y at 1 stalls; not every eps or complete first-hit on chart O",
            0,
            ("stage_c_origin_x32", "F", "stage_c_sec", "stage_c_oneshot", "stage_c_uniform"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_uniform",
            "kill-line comparison first-hit for every r1 in [0, 1] and every eps in (0, 1/16]",
            "dy/dx bound keeps dx/dsigma >= eps/2 from (x,y)=(1/4,1) out to x=(1/4)/eps; holding y at 1 stalls; not Lohner or the shrinking interface",
            0,
            ("stage_c_compare", "F", "stage_c_sec", "stage_c_interface"),
            "PROVED",
            False,
        ),
        ChartCell(
            "stage_c_interface",
            "kill-line comparison first-hit from every start in (0, 1/2], every r1 in [0, 1], every eps in (0, 1/16]",
            "entrance gap 1/4 and neck bound from x=1/2 keep dx/dsigma >= eps/5; an interface in (0, 1/2] is included; holding y at 1 stalls",
            0,
            ("stage_c_uniform", "F", "outgoing_corridor"),
            "PROVED",
            False,
        ),
        ChartCell(
            "sep_spre",
            "kill-line sep*S_pre < 11/5 and chi_b <= 8 for every sep in (0, 1]",
            "dyadic slabs plus a sep->0 tail; the dx_e log remainder keeps the net exponent > 1/8; feeding S=4 stalls; not dx_e off the kill line",
            0,
            ("chi_b", "dx_e_leading", "stage_a", "F"),
            "PROVED",
            False,
        ),
        ChartCell(
            "dx_e_off",
            "dx_e leading factors for every lambda1 in [-4, -2] and every sep in (0, 1]",
            "h(sep/r1)<11/5; net exponent >1/8; chi_b<=21/5; C<2; the cap 1 stalls; not lambda1<-4 or lambda1 in (-2, 0)",
            0,
            ("sep_spre", "dx_e_leading", "dx_e_unif", "stage_a", "F"),
            "PROVED",
            False,
        ),
        ChartCell(
            "dx_e_ray",
            "dx_e leading factors for every lambda1 <= -2 and every sep in (0, 1]",
            "X<=2*rstar keeps the net exponent >1/8, chi_b<=21/5, and C<2; dropping the rstar surplus stalls; not lambda1 in (-2, 0)",
            0,
            ("dx_e_off", "dx_e_leading", "stage_a", "F"),
            "PROVED",
            False,
        ),
        ChartCell(
            "dx_e_near",
            "dx_e leading factors for every lambda1 in [-3/2, -2) and every sep in (0, 1]",
            "a>=1/8 keeps h(u)<11/5 on u<=4, the net exponent >1/8, chi_b<=26/5, and C<2; dropping the rstar surplus stalls; not lambda1 in (-3/2, 0)",
            0,
            ("dx_e_ray", "dx_e_off", "stage_a", "F"),
            "PROVED",
            False,
        ),
        ChartCell(
            "dx_e_open",
            "dx_e leading factors for every lambda1 in (-3/2, 0)",
            "eps cap keeps the net exponent >1/8 and C<(1/5)/a on sep<(8/5) rstar; eps=1/16 at sep=1/4096 stalls; not Stage C or first-hit",
            0,
            ("dx_e_near", "dx_e_ray", "stage_a", "F"),
            "PROVED",
            False,
        ),
        ChartCell(
            "fold_imap",
            "sep = 0 implicit I(r-delta)=kappa",
            "exact dx/dkappa and leading C2 of log D'; remainder versus field open",
            0,
            ("C", "canonical_zeta", "fold_zeta", "physical_c2", "z_x_gap", "z_v_bound", "z_slow_v", "fold_z_x"),
            "PROVED",
            False,
        ),
        ChartCell(
            "kill_shrinking_root",
            "L = 1/n, lambda1 = -2, r1 -> 0; r1-theta*sep < 0",
            "incoming; two-root I-map; outgoing collision",
            0,
            ("O", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "canonical_zeta",
            "r=-1 slow-line zeta, lambda0=lambda1=0; Cauchy majorant for Z",
            "algebraic embedding; fold compact is fold_zeta",
            0,
            ("fold_imap", "C", "fold_zeta"),
            "PROVED",
            False,
        ),
        ChartCell(
            "fold_zeta",
            "sep=0 fold wall L=r^2, lambda1=-2 r; r in [1.4, 1.6]",
            "Picard k plus Cauchy majorant for Z; not physical C2",
            0,
            ("fold_imap", "C", "canonical_zeta", "physical_c2", "z_x_gap", "z_v_bound", "z_slow_v", "fold_z_x"),
            "PROVED",
            False,
        ),
        ChartCell(
            "physical_c2",
            "frozen-Z C2 remainder versus lifted fold",
            "exact gap identities; unfrozen Z_x identities sealed; holomorphic Z_v, slow-line Z_V, and fold I-map Z_x sealed; sep>0 open",
            0,
            ("fold_imap", "fold_zeta", "z_x_gap", "z_v_bound", "z_slow_v", "fold_z_x"),
            "PROVED",
            False,
        ),
        ChartCell(
            "z_x_gap",
            "unfrozen-Z first-log-derivative gap including Z_x",
            "exact gap identities; holomorphic Z_v, slow-line Z_V, and fold I-map Z_x sealed; sep>0 open",
            0,
            ("fold_imap", "fold_zeta", "physical_c2", "z_v_bound", "z_slow_v", "fold_z_x"),
            "PROVED",
            False,
        ),
        ChartCell(
            "z_v_bound",
            "holomorphic Z_v identities plus |Z_v|<1/4 on the cancelled-N kill compact",
            "Interval box excludes 0; slow-line Z_V is z_slow_v; fold I-map Z_x is fold_z_x; sep>0 open",
            0,
            ("fold_imap", "fold_zeta", "physical_c2", "z_x_gap", "cancelled_n", "z_slow_v", "fold_z_x"),
            "PROVED",
            False,
        ),
        ChartCell(
            "z_slow_v",
            "slow-line Z_V=-Z_v/ell plus |Z_V|<1/4 on the kill compact and holomorphic |Z_v|<1/4 on the fold wall",
            "Interval boxes exclude 0; fold I-map Z_x is fold_z_x; sep>0 open",
            0,
            ("fold_imap", "fold_zeta", "physical_c2", "z_x_gap", "z_v_bound", "cancelled_n", "fold_z_x"),
            "PROVED",
            False,
        ),
        ChartCell(
            "fold_z_x",
            "matching-chart Z_x=Z_v eps/ell plus |Z_x|<1/100 on the fold I-map compact r in [1.4, 1.6]",
            "Interval |Z_x|<1/100; not sep>0; first-hit open",
            0,
            ("fold_imap", "fold_zeta", "physical_c2", "z_x_gap", "z_v_bound", "z_slow_v"),
            "PROVED",
            False,
        ),
        ChartCell(
            "outgoing_corridor",
            "r1 -> 0 two-root I-map from r1(1+theta) to compact x_*",
            "bounded slow time; height-section first-hit open; a_min fails",
            0,
            ("O", "kill_shrinking_root", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "post_corridor",
            "after x-corridor: T_*=Theta(eps^2), (h/h_e)^(C eps)->1",
            "restored first-root hypotheses; height-section first-hit open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "height_envelope",
            "C=0 T-h conservation; uniform |q| ratio as r1->0",
            "alpha-0 envelope; actual-field first-hit open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "q_ratio_c2",
            "lambda1=-2 leading |q| ratio <2 for every x",
            "V-only C=2 bound; k=1+O(eps) and event open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "k_zeta_remainder",
            "C=0 normal k=1+O(nu); cubic correction eps^4 x^3/3",
            "exact O(nu^2) remainder; Z bound and event open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "kill_zeta",
            "lambda1=-2, L in [0,1] including L=0; Cauchy majorant for Z",
            "finite rectangular bound; not small enough for C=2+delta",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "cancelled_n",
            "cancelled-N holomorphic Z; 2 eps |V| |Z| < 1 on the slow-line kill compact",
            "usable C=2+delta prefactor on h=0; holomorphic Z_v, slow-line Z_V, and fold I-map Z_x sealed; T-h integral and event open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo", "z_v_bound", "z_slow_v", "fold_z_x"),
            "PROVED",
            False,
        ),
        ChartCell(
            "height_mix",
            "C!=0 ell/V mixing; |g_h|=O(nu^2) on a declared compact",
            "coordinate mixing sealed; T-h integral and event open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "orbit_th",
            "C=0 actual-versus-comparison T_h gap; equals k-1 at flux touching",
            "pointwise gap; comparison-bootstrap integral and event open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "th_integral",
            "comparison-bootstrap integral of T-h; majorant < 9 eps after T<=K(eps^2+h)",
            "comparison majorant; cubic Lohner orbit sealed; matching-chart E_out sealed; GRAZING E_sigma open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "vh_orbit",
            "cubic (V,h) QR-Lohner prefix; certified first-hit of V=-1/4",
            "declared V-wall; matching-chart E_out sealed; GRAZING E_sigma open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_out_section",
            "matching-chart E_out first-hit of x=rho/nu under V=-eps x",
            "certified E_out on L in {9/25, 1/16, 0}; shrinking-eps pack sealed; GRAZING E_sigma open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_out_eps",
            "shrinking-eps E_out pack n in {16, 20, 25} at L=0 inside T=n^2/8",
            "finite pack sealed; comparison speed sealed; GRAZING E_sigma open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_out_speed",
            "kill-line comparison speed: F increasing, -Vdot >= 3 eps^3, T <= (rho-eps)/(3 eps^3)",
            "O(1/eps^3) comparison majorant; incoming GRAZING comparison sealed; GRAZING first-hit open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_speed",
            "incoming GRAZING comparison: F decreasing, Vdot_rev >= 4 eps^3(1+eps), T <= 1/(4 eps^3(1+eps))",
            "O(1/eps^3) comparison majorant; incoming V=1/4 wall sealed; E_sigma first-hit open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_in",
            "incoming GRAZING-chart V=1/4 first-hit on the reverse cubic",
            "certified V=1/4 on L in {9/25, 1/16, 0}; comparison E_sigma from V=0 sealed; Lohner from V=0 open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_hit",
            "declared-point GRAZING E_sigma first-hit from (V,h)=(3/4,1/4)",
            "certified E_sigma on L in {9/25, 1/16, 0}; comparison from V=0 sealed; Lohner from V=0 open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_from0",
            "comparison GRAZING E_sigma unique zero from V=0 on the reverse cubic",
            "sign-change + dE/dV>0 on L in {9/25, 1/16, 0}; uniform comparison sealed; Lohner from V=0 open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_unif",
            "uniform comparison GRAZING E_sigma on eps in [0, 1/8] at V*=6/5",
            "eight Interval slabs keep E>0; wall-box h-interval E_sigma cover sealed; Lohner from V=0 open",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_wall",
            "orbit-aligned GRAZING E_sigma from (V,h)=(1/4,1/40) inside the V=1/4 wall box",
            "certified E_sigma on L in {9/25, 1/16, 0}; point in GRAZING-from-V=0 box; wall-box cover sealed; not Lohner from V=0",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_box",
            "wall-box h-interval GRAZING E_sigma cover of [1/50, 4/125] at V=1/4",
            "twelve slabs certify at L=0; L-pack wall-span sealed; not Lohner from V=0",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_span",
            "L=0 whole-wall h-span GRAZING E_sigma cover of [19/1000, 1/25] at V=1/4",
            "twenty-one slabs certify at L=0; L-pack wall-span sealed; not Lohner from V=0",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_pack",
            "L in {9/25, 1/16} wall-span GRAZING E_sigma cover of [17/1000, 7/200] at V=1/4",
            "eighteen slabs certify on both remaining L; one-shot from V=0 sealed; not uniform eps",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_eps",
            "shrinking-eps aligned GRAZING E_sigma pack n in {16, 20, 25} at L=0",
            "aligned (1/4,1/40) certifies at three n; shrinking-eps one-shot pack sealed; not uniform eps",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_oneshot",
            "one-shot Lohner GRAZING E_sigma from V=0 at eps=1/16",
            "single certify_stopped_event hits E_sigma on L in {9/25, 1/16, 0}; shrinking-eps one-shot pack sealed; not uniform eps",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot_eps", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_oneshot_eps",
            "shrinking-eps one-shot Lohner GRAZING E_sigma pack n in {16, 20, 25} at L=0",
            "single certify_stopped_event from V=0 at three n; compact aligned eps-span sealed; not every eps",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_eps_span", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_eps_span",
            "compact aligned parametric-eps GRAZING E_sigma cover of [1/25, 1/16] at L=0",
            "three slabs certify from aligned (1/4,1/40); L-pack last slab; lower [1/64, 1/16] cover sealed; not every eps",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_lo"),
            "PROVED",
            False,
        ),
        ChartCell(
            "e_sigma_eps_lo",
            "lower aligned parametric-eps GRAZING E_sigma cover of [1/64, 1/16] at L=0",
            "six slabs certify from aligned (1/4,1/40); L-pack last slab; not every eps; not Lohner from V=0",
            0,
            ("O", "kill_shrinking_root", "outgoing_corridor", "post_corridor", "height_envelope", "q_ratio_c2", "k_zeta_remainder", "kill_zeta", "cancelled_n", "height_mix", "orbit_th", "th_integral", "vh_orbit", "e_out_section", "e_out_eps", "e_out_speed", "e_sigma_speed", "e_sigma_in", "e_sigma_hit", "e_sigma_from0", "e_sigma_unif", "e_sigma_wall", "e_sigma_box", "e_sigma_span", "e_sigma_pack", "e_sigma_eps", "e_sigma_oneshot", "e_sigma_oneshot_eps", "e_sigma_eps_span"),
            "PROVED",
            False,
        ),
    )


def g1_from_cells(cells: Sequence[ChartCell] | None = None) -> bool:
    rows = tuple(cells) if cells is not None else chart_cells()
    return all(cell.g1_item_closed for cell in rows)


def ledger_payload() -> dict[str, object]:
    rows = chart_cells()
    return {
        "cells": [
            {
                "name": cell.name,
                "regime": cell.regime,
                "first_hit": cell.first_hit,
                "remainder_order": cell.remainder_order,
                "overlaps": list(cell.overlaps),
                "identity_verdict": cell.identity_verdict,
                "g1_item_closed": cell.g1_item_closed,
            }
            for cell in rows
        ],
        "g1_passed": g1_from_cells(rows),
        "honesty": _honesty(),
    }


def is_dyck_bits(word: Sequence[int]) -> bool:
    height = 0
    for bit in word:
        if bit not in (0, 1):
            return False
        height += 1 if bit == 1 else -1
        if height < 0:
            return False
    return height == 0


def bezout_edge_bound(counts: Sequence[int], degree: int) -> bool:
    return all(type(n) is int and 0 <= n <= degree for n in counts)


def replay_maletto_type(
    counts: Sequence[int],
    words: Sequence[Sequence[int]],
    trees: Sequence[tuple[int, Sequence[int]]],
    *,
    degree: int,
) -> dict[str, object]:
    """Replay one combinatorial curve type. Not a classification or G5 pass."""

    dyck = all(is_dyck_bits(word) for word in words)
    trees_ok = all(is_dyck_bits(shape) for _, shape in trees)
    bezout = bezout_edge_bound(counts, degree)
    ok = dyck and trees_ok and bezout and len(counts) == 6 and len(words) == 4
    return {
        "combinatorial_type_ok": ok,
        "dyck_words": dyck,
        "floating_trees": trees_ok,
        "bezout_edge_bound": bezout,
        "algebraic_smoothness": "BLOCKED",
        "honesty": _honesty(),
        "scope": "One (n, W, T) type. Not 119 cubics, not the octic target, not G1.",
    }


# Published §1.1 example of Maletto, arXiv:2606.21449v1: the quartic
# f = x^4 + x^2 y^2 + 2 x y^3 - y^4 - 2 x^3 z + x y^2 z - y^3 z
#     - 3 x^2 z^2 - 2 x y z^2 + 2 y^2 z^2 + y z^3 + z^4
# together with the NWT triple (n, W, T).
MALETTO_QUARTIC_EXAMPLE: Mapping[str, object] = {
    "degree": 4,
    "source": "Maletto arXiv:2606.21449v1 §1.1",
    "counts": (1, 2, 1, 1, 0, 1),
    "words": ((1, 1, 0, 0), (1, 0), (1, 0, 1, 0), (1, 0)),
    "trees": ((10, (1, 0)),),
    "terms": (
        ((4, 0, 0), 1),
        ((2, 2, 0), 1),
        ((1, 3, 0), 2),
        ((0, 4, 0), -1),
        ((3, 0, 1), -2),
        ((1, 2, 1), 1),
        ((0, 3, 1), -1),
        ((2, 0, 2), -3),
        ((1, 1, 2), -2),
        ((0, 2, 2), 2),
        ((0, 1, 3), 1),
        ((0, 0, 4), 1),
    ),
}


def chart_dependent_g1_reaudit(chart: str = "I_2^1/I_4^1") -> dict[str, object]:
    """Test chart-dependent ``gamma(s/r)`` on the kill sequence (route-specific)."""
    from omnibias.dynamics.g1_passage import (
        chi_tracking_gamma,
        frozen_exponent_kill_sequence_report,
        frozen_majorant_log_ratio,
    )
    from omnibias.dynamics.scale_dichotomy import KILL_EPS, kill_sep

    eps = KILL_EPS
    sep = kill_sep(eps)
    kappa = 1.0 / sep
    gamma_chart = chi_tracking_gamma(sep, 1.0)
    chart_log = frozen_majorant_log_ratio(sep, 1.0, kappa, 1.0, gamma_chart)
    frozen = frozen_exponent_kill_sequence_report(eps=eps)
    shrinking = rematch_shrinking_root(Fraction(1, 1000), Fraction(-2))
    return {
        "chart": chart,
        "gamma_chart_dependent": gamma_chart,
        "chart_log_ratio_at_kill": chart_log,
        "chart_survives_kill_sequence": chart_log <= 0.0,
        "frozen_exponent_blocks_uniform_majorant": frozen.frozen_exceeds_one,
        "shrinking_root_rematch": shrinking,
        "all_cells_g1_closed": g1_from_cells(),
        "parent_g1_passed": False,
        "route_specific": True,
        "detail": (
            "chart-dependent gamma=s/r cancels the kappa term at a point; "
            "does not close the atlas without a uniform majorant"
        ),
    }


def rematch_shrinking_root(L: Fraction, lambda1: Fraction, *, Lmin: Fraction = Fraction(1, 2)) -> dict[str, object]:
    """Coefficient membership of the shrinking-root sequence in existing charts."""

    lambda0 = -L
    first_root = L >= Lmin and lambda1 < 0
    height_nonneg = lambda1 >= 0
    exponential = L >= Lmin
    grazing_coeff = lambda0 <= 0 and abs(lambda1) <= 2 and L <= 2
    return {
        "L": [L.numerator, L.denominator],
        "lambda1": [lambda1.numerator, lambda1.denominator],
        "lambda0": [lambda0.numerator, lambda0.denominator],
        "incoming_first_hit_retained": True,
        "first_root_Lmin": first_root,
        "height_nonnegative_lambda1": height_nonneg,
        "exponential_Lmin": exponential,
        "grazing_coefficient_box": grazing_coeff,
        "selected_small_label_first_root": first_root,
        "sr2_rematch": False,
        "honesty": _honesty(),
    }
