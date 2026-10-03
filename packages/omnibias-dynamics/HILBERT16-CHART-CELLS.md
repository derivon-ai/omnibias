# Finite chart-cell ledger

This note records the named cells of the current coalescence atlas. It is
combinatorial bookkeeping. A complete list of labels is not G1, not G4,
and not Hilbert XVI. Verdict collapse applies only to exact identities.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## Cells

| Cell | Regime | First-hit | Remainder order | G1 item |
| --- | --- | --- | --- | --- |
| N | `Delta >= chi_N > 0` | selected tube | 2 | open |
| F | `sep >= sep0`, `L >= Lmin` | selected tube | 2 | open |
| D | `epsilon \|log sep\| <= 1` | selected tube | 1 | open |
| C | `sep = 0` | leading I-map; remainder versus field open | 0 | open |
| O | `L -> 0` | incoming; rescaled I-map; outgoing collision | 0 | open |
| WF | W-fold, `sigma = sqrt(epsilon)` | selected tube | 0 | open |
| WS | W-separation, `chi = O(1)` | selected tube | 0 | open |
| LI | `tau = epsilon log(1/sep)` | selected tube | 0 | open |
| WL | `Lambda = epsilon log(1/W)` | selected tube | 0 | open |
| cellA | LN/exp candidate with `tau` and `W` coordinates | incoming only; physical LN membership open | 0 | open |
| cellB | LN/exp candidate with base-dependent shrinking wall | proposed `r1-theta*sep` eventually negative | 0 | open |
| cellAB | coordinate overlap of cellA/cellB | coordinate identity only | 0 | open |
| kill_super_small_sep | `sep = exp(-1/epsilon^2)` | admitted; tracked first-derivative product | 0 | open (C2 / continuation remainder) |
| product_bound | `sep^2 * (h_1/(eps^3 mu sep^2))^(C eps)` | algebraic identity; kill-line Stage-B sealed | 0 | open (not G1) |
| weighted_section | intrinsic `eta`-section `h=eps^3 sep^2 eta0` | transverse for fixed `sep>0`; ordinary `q=sep^2` derivatives singular at D-C and speed vanishes at O | 0 | disproved as a G1 route |
| quasihomogeneous_dichotomy | one scale `sigma=eps^a sep^b` against frozen `h=eps^N` | every rational monomial weight excluded; moving sections and multistage atlases not excluded | 0 | proved frozen-section no-go only |
| ln_format_barrier | direct `tau`/log-W chain on finite kill-sequence truncations | chain length and coefficients fixed; outer radius and sup norm grow linearly | 0 | proved direct-format no-go only |
| stage_b | kill-line Stage-B `dx/dy=eps/x`; Picard `|Delta x|<1/3` | height inflation enclosed; Stage A wall is stage_a; Stage C `a_min` is stage_c | 0 | open |
| stage_a | kill-line Stage-A walls `a=r1-theta sep`; `B_-(a)=theta(1+theta)sep^2` | wall identities plus Interval `a>1/4` and `Psi_pre` factor `<1/4`; `chi_b` sealed; `dx_e` leading sealed; uniform-in-`chi` `dx_e` sealed | 0 | open |
| chi_b | kill-line `chi_b=4/r1`; two-slab `sep*S_pre<3` on `sep in [1/2^16, 1]` | `chi` threshold enclosed; `dx_e` leading sealed; uniform-in-`chi` `dx_e` sealed | 0 | open |
| dx_e_leading | kill-line `dx_e` prefactor `<1/2`; threshold net exponent `>1/8` | leading factors enclosed; uniform-in-`chi` `dx_e` is `dx_e_unif`; not Stage C | 0 | open |
| dx_e_unif | kill-line `dx_e <= C sep^2 exp(-(3/32) chi)` with `C<2` | uniform-in-`chi` majorant enclosed; Stage C `a_min` is `stage_c` | 0 | open |
| stage_c | kill-line Stage-C `a_min=1/4` after Stage B; `1/x<8` | geometric floor enclosed; Stage-C exit energy is `stage_c_exit`; not first-hit | 0 | open |
| stage_c_exit | kill-line Stage-C `T_e/eps^2 in (1/16, 1)` and `T_e > h_e` | exit energy enclosed; Stage-C `T_h` is `stage_c_th`; not first-hit | 0 | open |
| stage_c_th | kill-line Stage-C leading `T_h>1/2` at `y_1=1` | leading `T_h` floor enclosed; Stage-C start gap is `stage_c_gap`; not first-hit | 0 | open |
| stage_c_gap | kill-line Stage-C `(T_e-h_1)/eps^2>1/32` at `y_1=1` | start gap enclosed at actual Stage C height; C=0 T envelope is `stage_c_env`; not first-hit | 0 | open |
| stage_c_env | kill-line Stage-C C=0 sandwich `(1/32)(eps^2+h)<=T(h)<=eps^2+h` | C=0 two-sided T envelope enclosed; C=2 integrating factor is `stage_c_if`; not first-hit | 0 | open |
| stage_c_if | kill-line Stage-C C=2 integrating factor `(h/h_1)^{C eps}<32`; exponent `<=3` | C=2 factor enclosed; C=2 T(h) majorant is `stage_c_int`; not first-hit | 0 | open |
| stage_c_int | kill-line Stage-C C=2 `T(h)<=64(eps^2+h)`; slope `16/7` | C=2 T(h) majorant enclosed; C=2 lower envelope is `stage_c_lo`; not first-hit | 0 | open |
| stage_c_lo | kill-line Stage-C C=2 `T(h)>=(1/32)(eps^2+h)`; start remainder `> 1/16` | C=2 lower T(h) envelope enclosed; C=2 tight ratio is `stage_c_k`; not first-hit | 0 | open |
| stage_c_k | kill-line Stage-C C=2 `T(h)<=6(eps^2+h)`; edge factor `< 3` | C=2 tight T(h) ratio enclosed; C=2 T-h bootstrap is `stage_c_boot`; not first-hit | 0 | open |
| stage_c_boot | kill-line Stage-C C=2 `T-h<1` at `K=6`; linear `15 eps` | C=2 T-h bootstrap enclosed; continuation rectangle is `stage_c_rect`; not first-hit | 0 | open |
| stage_c_rect | kill-line Stage-C `V in [-2, -1/64]` on `h in [h_1, 1]` | continuation rectangle enclosed; comparison first-hit of `h=1` is `stage_c_hit`; not signed-label section | 0 | open |
| stage_c_hit | kill-line Stage-C comparison first-hit of `h=1`; time `<= 192 ln(16)` | comparison first-hit of `h=1` enclosed; `E_out` from Stage-C start is `stage_c_sec`; not Lohner or chart O | 0 | open |
| stage_c_sec | kill-line Stage-C comparison first-hit of `E_out`; start gap `1/6` | comparison first-hit of `E_out` from Stage-C start; Lohner from Stage-C start is `stage_c_oneshot`; not chart O | 0 | open |
| stage_c_oneshot | kill-line Stage-C Lohner first-hit of matching-chart `x=4` from `(x,y)=(1/4,1)` | unique transverse first-hit at `eps=1/16`; shrinking-eps pack is `stage_c_oneshot_eps`; not every `eps` or chart O | 0 | open |
| stage_c_oneshot_eps | kill-line Stage-C shrinking-eps Lohner pack `n in {16, 20, 25}` | unique transverse first-hit at three squares; parametric-eps cover is `stage_c_eps_span`; not every `eps` or chart O | 0 | open |
| stage_c_eps_span | kill-line Stage-C parametric-eps Lohner cover of `[23/400, 1/16]` | unique transverse first-hit on four `1/800` slabs; origin pack is `stage_c_origin`; not every `eps` or chart O | 0 | open |
| stage_c_origin | kill-line matching-chart Lohner pack toward chart O; `sep in {3/2, 7/4, 2}` | unique transverse first-hit at `r1 in {1/4, 1/8, 0}` from compact `x=1/4`; parametric-sep cover is `stage_c_origin_span`; not every `r1` or complete first-hit on chart O | 0 | open |
| stage_c_origin_span | kill-line parametric-sep matching-chart Lohner cover of `[3/2, 2]` | unique transverse first-hit on eight `1/16` slabs; nearer-interface cover is `stage_c_origin_iface`; not every `r1` or complete first-hit on chart O | 0 | open |
| stage_c_origin_iface | kill-line nearer-interface matching-chart Lohner cover of `[7/4, 2]` from `x=1/8` | unique transverse first-hit on eight `1/32` slabs; `x=1/16` cover is `stage_c_origin_near`; not every `r1` or complete first-hit on chart O | 0 | open |
| stage_c_origin_near | kill-line nearer-interface matching-chart Lohner cover of `[15/8, 2]` from `x=1/16` | unique transverse first-hit on eight `1/64` slabs; `x=1/32` cover is `stage_c_origin_x32`; not every `r1` or complete first-hit on chart O | 0 | open |
| stage_c_origin_x32 | kill-line nearer-interface matching-chart Lohner cover of `[31/16, 2]` from `x=1/32` | unique transverse first-hit on eight `1/128` slabs; uniform-in-`r1` comparison is `stage_c_compare`; not every `r1` or complete first-hit on chart O | 0 | open |
| stage_c_compare | kill-line comparison first-hit for every `r1` in `[0, 1]` and `eps` in `[1/32, 1/16]` | phase-wise speed bound from `(x,y)=(1/4,1)` reaches `x=8`; uniform comparison is `stage_c_uniform`; not every `eps` or complete first-hit on chart O | 0 | open |
| stage_c_uniform | kill-line comparison first-hit for every `r1` in `[0, 1]` and every `eps` in `(0, 1/16]` | `dy/dx` keeps `dx/dσ >= eps/2` out to `x=(1/4)/eps`; every start in `(0, 1/2]` is `stage_c_interface`; not Lohner or the shrinking interface past `1/2` | 0 | open |
| stage_c_interface | kill-line comparison first-hit from every start in `(0, 1/2]` | entrance gap `1/4` and neck bound keep `dx/dσ >= eps/5`; an interface in `(0, 1/2]` is included; not the height-section flag | 0 | open |
| sep_spre | kill-line `sep * S_pre < 11/5` for every `sep` in `(0, 1]` | dyadic slabs plus a `sep -> 0` tail; `chi_b <= 8`; the `dx_e` log remainder keeps the net exponent `> 1/8`; not `dx_e` off the kill line | 0 | open |
| dx_e_off | `dx_e` factors for every `lambda1` in `[-4, -2]` and every `sep` in `(0, 1]` | `h(sep/r1) < 11/5`; net exponent `> 1/8`; `chi_b <= 21/5`; `C < 2`; not `lambda1 < -4` or `lambda1` in `(-2, 0)` | 0 | open |
| dx_e_ray | `dx_e` factors for every `lambda1 <= -2` and every `sep` in `(0, 1]` | `X <= 2 rstar` keeps the net exponent `> 1/8`; `chi_b <= 21/5`; `C < 2`; not `lambda1` in `(-2, 0)` | 0 | open |
| dx_e_near | `dx_e` factors for every `lambda1` in `[-3/2, -2)` and every `sep` in `(0, 1]` | `a >= 1/8`; `h(u) < 11/5` on `u <= 4`; net exponent `> 1/8`; `chi_b <= 26/5`; `C < 2`; not `lambda1` in `(-3/2, 0)` | 0 | open |
| dx_e_open | `dx_e` factors for every `lambda1` in `(-3/2, 0)` | `eps` cap keeps the net exponent `> 1/8`; `C < (1/5)/a` on `sep < (8/5) rstar` | 0 | open |
| fold_imap | `sep = 0` implicit `I(r-delta)=kappa` | exact `dx/dkappa` and leading C2 of `log D'` | 0 | open (remainder versus field) |
| kill_shrinking_root | `L = 1/n`, `lambda1 = -2` | incoming; two-root I-map; outgoing collision | 0 | open |
| canonical_zeta | `r=-1`, `lambda=0` slow-line `zeta` | algebraic embedding; Cauchy majorant; fold compact is fold_zeta | 0 | open |
| fold_zeta | fold wall `L=r^2`, `lambda1=-2 r`, `r in [1.4, 1.6]` | Picard `k` plus Cauchy majorant for `Z`; not physical C2 | 0 | open |
| physical_c2 | frozen-Z C2 versus lifted fold | exact gap identities; unfrozen `Z_x` identities sealed; fold I-map `Z_x` sealed; `sep>0` open | 0 | open |
| z_x_gap | unfrozen-Z first-log-derivative gap including `Z_x` | exact gap identities; fold I-map `Z_x` sealed; `sep>0` open | 0 | open |
| z_v_bound | holomorphic `Z_v` identities plus `|Z_v|<1/4` on the cancelled-N kill compact | Interval box excludes 0; slow-line `Z_V` is z_slow_v; fold I-map `Z_x` is fold_z_x; `sep>0` open | 0 | open |
| z_slow_v | slow-line `Z_V=-Z_v/ell` plus `|Z_V|<1/4` on the kill compact and holomorphic `|Z_v|<1/4` on the fold wall | Interval boxes exclude 0; fold I-map `Z_x` is fold_z_x; `sep>0` open | 0 | open |
| fold_z_x | matching-chart `Z_x=Z_v eps/ell` plus `|Z_x|<1/100` on the fold I-map compact | Interval `|Z_x|<1/100`; not `sep>0`; first-hit open | 0 | open |
| outgoing_corridor | `r1 -> 0` I-map from `r1(1+theta)` to compact `x_*` | bounded slow time; height-section first-hit open | 0 | open |
| post_corridor | after x-corridor: `T_*=Theta(eps^2)`, `(h/h_e)^{C eps}->1` | restored first-root hypotheses; height-section first-hit open | 0 | open |
| height_envelope | `C=0` `T-h` conservation; uniform `|q|` ratio as `r1->0` | alpha-0 envelope; actual-field first-hit open | 0 | open |
| q_ratio_c2 | `lambda1=-2` leading `|q|` ratio `<2` for every `x` | V-only `C=2` bound; `k=1+O(eps)` and event open | 0 | open |
| k_zeta_remainder | `C=0` normal `k=1+O(nu)`; cubic `eps^4 x^3/3` | exact `O(nu^2)` remainder; `Z` bound and event open | 0 | open |
| kill_zeta | `lambda1=-2`, `L in [0,1]` including `L=0`; Cauchy majorant for `Z` | finite rectangular bound; not small enough for `C=2+delta` | 0 | open |
| cancelled_n | cancelled-N holomorphic `Z`; `2 eps |V| |Z| < 1` on the slow-line kill compact | usable `C=2+delta` prefactor on `h=0`; holomorphic `Z_v` bound sealed; event open | 0 | open |
| height_mix | `C!=0` `ell`/`V` mixing; `|g_h|=O(nu^2)` | coordinate mixing sealed; `T-h` along orbit and event open | 0 | open |
| orbit_th | `C=0` actual-versus-comparison `T_h` gap; equals `k-1` at flux touching | pointwise gap; comparison-bootstrap integral and event open | 0 | open |
| th_integral | comparison-bootstrap `T-h` integral; majorant `< 9 eps` after `T<=K(eps^2+h)` | comparison majorant; cubic Lohner orbit sealed; matching-chart `E_out` sealed; GRAZING `E_sigma` open | 0 | open |
| vh_orbit | cubic `(V,h)` QR-Lohner prefix; certified first-hit of `V=-1/4` | declared `V`-wall; matching-chart `E_out` sealed; GRAZING `E_sigma` open | 0 | open |
| e_out_section | matching-chart `E_out` first-hit of `x=rho/nu` under `V=-eps x` | certified `E_out` on `L in {9/25, 1/16, 0}`; shrinking-eps pack sealed; GRAZING `E_sigma` open | 0 | open |
| e_out_eps | shrinking-eps `E_out` pack `n in {16, 20, 25}` at `L=0` inside `T=n^2/8` | finite pack sealed; comparison speed sealed; GRAZING `E_sigma` open | 0 | open |
| e_out_speed | kill-line comparison speed: `F` increasing, `-Vdot >= 3 eps^3`, `T <= (rho-eps)/(3 eps^3)` | `O(1/eps^3)` comparison majorant; incoming GRAZING comparison sealed; GRAZING first-hit open | 0 | open |
| e_sigma_speed | incoming GRAZING comparison: `F` decreasing, `Vdot_rev >= 4 eps^3(1+eps)`, `T <= 1/(4 eps^3(1+eps))` | `O(1/eps^3)` comparison majorant; incoming `V=1/4` wall sealed; `E_sigma` first-hit open | 0 | open |
| e_sigma_in | incoming GRAZING-chart `V=1/4` first-hit on the reverse cubic | certified `V=1/4` on `L in {9/25, 1/16, 0}`; comparison `E_sigma` from `V=0` sealed; Lohner from `V=0` open | 0 | open |
| e_sigma_hit | declared-point GRAZING `E_sigma` first-hit from `(V,h)=(3/4,1/4)` | certified `E_sigma` on `L in {9/25, 1/16, 0}`; comparison from `V=0` sealed; Lohner from `V=0` open | 0 | open |
| e_sigma_from0 | comparison GRAZING `E_sigma` unique zero from `V=0` on the reverse cubic | sign-change + `dE/dV>0` on `L in {9/25, 1/16, 0}`; uniform comparison sealed; Lohner from `V=0` open | 0 | open |
| e_sigma_unif | uniform comparison GRAZING `E_sigma` on `eps in [0, 1/8]` at `V*=6/5` | eight Interval slabs keep `E>0`; wall-box `h`-interval `E_sigma` cover sealed; Lohner from `V=0` open | 0 | open |
| e_sigma_wall | orbit-aligned GRAZING `E_sigma` from `(V,h)=(1/4,1/40)` inside the `V=1/4` wall box | certified `E_sigma` on `L in {9/25, 1/16, 0}`; wall-box cover sealed; not Lohner from `V=0` | 0 | open |
| e_sigma_box | wall-box `h`-interval GRAZING `E_sigma` cover of `[1/50, 4/125]` at `V=1/4` | twelve slabs certify at `L=0`; L-pack wall-span sealed; not Lohner from `V=0` | 0 | open |
| e_sigma_span | L=0 whole-wall `h`-span GRAZING `E_sigma` cover of `[19/1000, 1/25]` at `V=1/4` | twenty-one slabs certify at `L=0`; L-pack wall-span sealed; not Lohner from `V=0` | 0 | open |
| e_sigma_pack | `L in {9/25, 1/16}` wall-span GRAZING `E_sigma` cover of `[17/1000, 7/200]` at `V=1/4` | eighteen slabs certify on both remaining `L`; shrinking-eps aligned pack sealed; not Lohner from `V=0` | 0 | open |
| e_sigma_eps | shrinking-eps aligned GRAZING `E_sigma` pack `n in {16, 20, 25}` at `L=0` | aligned `(1/4,1/40)` certifies at three `n`; one-shot from `V=0` sealed; not uniform `eps` | 0 | open |
| e_sigma_oneshot | one-shot Lohner GRAZING `E_sigma` from `V=0` at `eps=1/16` | single `certify_stopped_event` hits `E_sigma` on `L in {9/25, 1/16, 0}`; shrinking-eps one-shot pack sealed; not uniform `eps` | 0 | open |
| e_sigma_oneshot_eps | shrinking-eps one-shot Lohner GRAZING `E_sigma` pack `n in {16, 20, 25}` at `L=0` | single `certify_stopped_event` from `V=0` at three `n`; compact aligned `eps`-span sealed; not every `eps` | 0 | open |
| e_sigma_eps_span | compact aligned parametric-eps GRAZING `E_sigma` cover of `[1/25, 1/16]` at `L=0` | three slabs certify from aligned `(1/4,1/40)`; L-pack last slab; lower `[1/64, 1/16]` cover sealed; not every `eps` | 0 | open |
| e_sigma_eps_lo | lower aligned parametric-eps GRAZING `E_sigma` cover of `[1/64, 1/16]` at `L=0` | six slabs certify from aligned `(1/4,1/40)`; L-pack last slab; not every `eps`; not Lohner from `V=0` | 0 | open |

Identity verdicts on the double-root, W-ratio, shrinking-root, blow-up
height, leading event, and joint-axis algebra are `PROVED` as finite
rational residuals. Physical C2 remainders stay `BLOCKED`. Charts LI and
WL record the log-intermediate attempt of
[the next-atlas note](HILBERT16-NEXT-ATLAS.md); they do not close the
scale dichotomy. The super-small sequence remains admitted on the
selected incoming connector.  The subsequent
[LN/exp cell test](HILBERT16-LN-PASSAGE.md) proves the two elementary
logarithmic-derivative lemmas but finds that corrected kill A still has an
unbounded matching `W`-ratio and that the proposed cellB radius becomes
negative. `g1_passed` is true only if every G1 item
is written-closed. It is false on this list.

The parallel G5 slice replays one Maletto `(n, W, T)` quartic. Dyck and
Bézout checks are exact; complex smoothness is attached by the existing
plane-curve witness in the two-blow-up benchmark. A complete real scheme
stays `BLOCKED`. That replay cannot pass G1 or the octic target.

## Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_two_blowup.py \
  packages/omnibias-dynamics/tests/test_next_atlas.py -q
```
