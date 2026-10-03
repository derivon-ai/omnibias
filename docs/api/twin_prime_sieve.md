# Twin-prime sieve research ledger

`omnibias.holonomic.twin_prime` provides exact finite checks for two research
tracks. Neither track currently establishes an infinite prime-gap theorem.

## Parity-sensitive asymptotic-sieve track

For

\[
a_m=\mu^2(m)\Lambda(m-2),
\qquad
A_d(x)=g(d)A(x)+r_d(x),
\]

the local factors are

\[
G_p=1-\frac{1}{p(p-1)},\qquad
g(p)=\frac{p-1}{p^2-p-1}\quad(p>2).
\]

`asymptotic_sieve_local_factor` checks over the rationals that

\[
G_p\frac{1-g(p)}{1-1/p}
=1-\frac{1}{(p-1)^2}
\]

for every declared odd prime; the factor at \(p=2\) is \(2\).
`certify_asymptotic_sieve_local_product` proves the corresponding finite
Euler-product prefix. It deliberately does not infer an infinite product.
`local_factor_obligation_certificates` additionally emits one
`rational_identity` obligation per local factor and one for the prefix product;
these are accepted by the finite Lean-obligation generator. In the recorded
validation run, all 26 obligations through \(p=97\) passed genuine Lean kernel
builds. This formal result covers only the finite prefix.

### Exact FI combinatorial replay

Friedlander--Iwaniec equation (3.1) starts from the Vaughan-type identity

\[
f(n>z)=F(n;y)-F(n;y,z)+f_1(n;y,z)+f_2(n;y,z)+f_3(n;y,z),
\qquad F=f*1.
\]

The terminal domain \(b>y,\ c>z\) is assigned disjointly using the exact
endpoints

\[
\begin{aligned}
f_1 &: b>sy,\ c>sz,\\
f_2 &: y<b\le sy,\ c>z,\\
f_3 &: b>sy,\ z<c\le sz.
\end{aligned}
\]

`fi_vaughan_coefficient_case` expands both sides as integer coefficient
vectors on the formal values \(f(c)\). Equality of those vectors establishes
the identity for every arithmetic function on the declared divisor lattice,
not merely for sampled function values. `certify_fi_combinatorial_replay`
also checks unique terminal-pair assignment and the exact dyadic partition

\[
(y,2^J y]=\bigsqcup_{j=0}^{J-1}(2^j y,2^{j+1}y].
\]

The smoke certificate replays every \(n\le128\) at \(y=z=4,\ s=8\). Its three
rational obligations all passed genuine Lean kernel builds. This earns finite
gate A1a, the exact combinatorial core. It does not earn A1: logarithmic smoothing, the
upper-bound sieve insertion, the analytic estimates for \(T,S_1,S_2,S_3\),
and uniform asymptotic passage remain external.

The next structural subchecks are also explicit:

- `select_fi_dyadic_s` uses rational squared comparisons to select the unique
  power of two in \((a,2a]\), avoiding floating square roots;
- `verify_dyadic_membership` checks pointwise open-left/closed-right ownership,
  not only equality of total widths;
- `certify_rho_insertion` checks
  \(\rho_n\Lambda(n>Z)=\Lambda(n>Z)\) on declared finite squarefree support,
  including \(\lambda_1=1\), \(|\lambda_\nu|\le1\), \(\nu\le\Delta<Z\), and
  finite nonnegativity;
- `canonical_fi_term_ledger` assigns all 13 analytic leaves exactly once.
  \(S_2\) routes to \(B'[\gamma=1]\), \(S_3^-\) routes to
  \(B'[\gamma(c,t)]\), the declared remainder leaves route to \(R'\), and
  \(S_1\) does not route to \(B'\).

The term-ledger validator rejects missing or duplicate leaves, wrong estimate
families, shift-averaged substitutes for fixed shift \(2\), and any input whose
\(q=1\) specialization already implies the target correlation. These finite
subchecks are gate A1b0. Full A1b remains blocked on exact smoothing and range
bookkeeping; a provenance declaration detects stated circularity but cannot
replace independent mathematical review.

`RationalCell`, `KnownEstimate`, and `certify_mobius_cell_reduction` describe
a rational atlas in the \((\log r/\log x,\log s/\log x)\) plane. The checker
proves that cells discharged by named Type I/II/III or remainder estimates,
together with the cells still requiring Möbius cancellation, exactly partition
the classical region. It can certify that the remaining region is strictly
smaller.

The explicit Friedlander--Iwaniec support also needs
`LogAffineBound` and `LogPolyhedralCell`, because it depends on

\[
u=\frac{\log R}{\log x},\quad
v=\frac{\log N}{\log x},\quad
w=\frac{\log C}{\log x},\quad
\ell=\frac{\log\log x}{\log x}
\]

and contains oblique faces such as \(u+v\le1\).
`certify_fi_terminal_atlas` records both the moving terminal shell

\[
1-131\ell<u+v\le1
\]

and an exact rational fallback with \(63/64<u+v\le1\). For
\[
\mathcal P_0=\{u\ge0,\ 5/16<v<1/2,\ 0\le w\le1/4,\ u+v\le1\},
\]
the checker obtains
\[
\operatorname{vol}(\mathcal P_0)=\frac{57}{2048},\qquad
\operatorname{vol}(\mathcal M_{1/64})=\frac3{4096},
\]
so the fallback shell is exactly \(1/38\) of the original log-volume.
Three `rational_identity` obligations seal the partition, ratio, and exponent
arithmetic. All three passed genuine Lean kernel builds in the recorded
validation run.

`fi_rational_terminal_family` exposes the stronger no-go observation. For any
rational \(0<\varepsilon<1/2\), the shell
\(1-\varepsilon<u+v\le1\) has unresolved fraction

\[
\frac{\operatorname{vol}(\mathcal M_\varepsilon)}
{\operatorname{vol}(\mathcal P_0)}
=\frac{32}{19}\varepsilon.
\]

Hence this coordinate volume can be made arbitrarily small while the full
fixed-shift parity obstruction survives. Minimizing atlas volume is therefore
not a valid proxy for progress on the analytic estimate.

The moving cutoff is chosen because
\[
\sum_{k\le x/(\log x)^{131}}\Lambda(k-2)\tau(k)^7
\ll x(\log x)^{-3};
\]
the minimal exponent \(131=127+1+3\) uses the established seventh
divisor-moment degree \(127\), one logarithm for \(\Lambda\), and the three
logarithms required by FI's \(B'\). Thus, conditional on Mirsky's asymptotic and that standard
moment theorem, the far-product part of FI's weighted \(B'\) norm is
elementary. This does **not** estimate the terminal shell.

The geometric certificate and exponent arithmetic are finite. The following
analytic obligations remain false in the honesty payload:

- replay of the complete Friedlander–Iwaniec reduction;
- a remainder estimate beyond level \(x^{2/3}\);
- the fixed-shift Möbius bilinear estimate on the terminal shell;
- uniform passage from finite scales to all sufficiently large \(x\).

No known estimate removes a positive-width cell from that near-hyperbola
shell for the fixed shift \(2\). Shift-averaged cancellation cannot discharge
this fixed-shift obligation, and a specialization that already asserts a
twin-prime correlation is rejected as circular.

### Fixed-shift signed-determinant route

The terminal search exposes one exact transform worth retaining. On coprime
squarefree \(r,s\), put \(s=dt\). Then

\[
\gamma_C(s)\mu(rs)
=
\sum_{\substack{dt=s,\ d\le C\\r,d,t\ {\rm pairwise\ coprime}}}
\mu(r)\mu(t).
\]

Combining this with the oriented identity

\[
\Lambda(n)=\sum_{qk=n}\mu(q)\log k
\]

turns each term into a signed fixed-difference determinant

\[
rdt-qk=2.
\]

`certify_fixed_shift_determinant_replay` verifies both identities
coefficientwise in the free basis \(\{\log p\}\) on a declared finite divisor
lattice. It records that the \(q=1\) contribution is the smooth
\(\log(rdt-2)\), the \(k=1\) term is zero, shift averaging is absent, and
Möbius signs have not been erased by a termwise triangle inequality. The
power-of-two von Mangoldt support is identified as a separate sparse case.

The strongest directly compatible current input found in the audit is
[Wright, Corollary 2.2](https://arxiv.org/abs/2604.25177), a 2026 preprint on
fixed-residue unbalanced convolutions. `wright_fixed_shift_range` checks only
its rational exponent conditions; coefficient hypotheses, the fixed-shift
size condition, main terms, and application to a determinant box remain
external. For complementary boxes, the arbitrary-coefficient kernel in
[Dong--Robles--Zeindler, Theorem 1.6](https://arxiv.org/abs/2601.00292)
suggests a possible completion route but is not itself a determinant theorem.

For the proposed positive-width cell

\[
\frac{63}{64}<u+v\le1,\qquad
\frac{39}{100}\le v\le\frac25,\qquad
0\le z=\frac{\log D}{\log x}\le\frac14,
\]

`certify_fixed_shift_kernel_cell` proves the exact lower kernel credit

\[
\eta\ge\frac{41}{640}.
\]

Targeting a final \(x^{-1/1000}\) saving leaves at most
\(1009/16000\) for every completion, norm, and dyadic-recombination loss.
The Heath--Brown \(J=7\) small-factor comparison has exact exponent room
\(101/630-1/7=11/630\). These are finite geometry and bookkeeping facts,
not an analytic saving.

The remaining candidate lemma must uniformly control every smooth dyadic
fixed-\(2\) determinant box for arbitrary dual coefficients
\(|\varepsilon_r|\le1\), including signed main terms and zero frequencies.
No such loss-budgeted completion is currently proved. This is gate A2d and
is strictly stronger than the finite A1b1 determinant replay.

### Local-preconditioner no-go

The proposed finite local reweighting

\[
\rho_\theta(m)=\prod_{\substack{p\le P_0\\p\mid m}}\theta_p
\]

does not create parity-sensitive information. For odd \(p\le P_0\), writing
\(K_p=p(p-2)+(p-1)\theta_p\), its modified factors satisfy

\[
G_{\theta,p}=\frac{K_p}{p(p-1)},\qquad
g_\theta(p)=\frac{(p-1)\theta_p}{K_p},\qquad
G_{\theta,p}H_{\theta,p}=\frac{p(p-2)}{(p-1)^2}.
\]

Thus the combined singular factor is independent of \(\theta_p\). Moreover,
the target sum is unchanged at every upper prime larger than \(P_0\); only
cutoff-local and sparse prime-power corrections remain. Algebraically,
\(\rho_\theta\) is a finite linear combination of hard presieves, and for
\(0\le\theta_p\le1\) it is a convex mixture of them.

Consequently, no preconditioner optimizer is shipped. It should be reconsidered
only after A1 supplies the exact FI norm and a uniform proof—not finite-scale
tuning—shows that the reweighting reduces that norm or the required Möbius
region.

## Bounded-gap matrix track

`certify_matrix_close_pair` attaches an exact positive-semidefinite matrix
\(Q_i\) to each shift \(h_i\). It verifies

\[
\sum_{i\in E}Q_i\preceq I
\]

for every maximal set \(E\) whose shifts are pairwise farther apart than the
target gap. `certify_matrix_variational_witness` then checks an exact rational
face-Gram score.

This finite implication is useful for testing non-scalar close-pair scoring,
but no source-valid crossing is known for the current bounded-gap sieve. The
next conventional optimization target is the complete \(k=39\) certificate
needed for \(H_1\le182\); improving only the existing \(k=40\) margin would not
improve the gap bound.

`h1_186_baseline` records the named OpenAI *Improved Short Gaps Between
Primes* finite certificate: \(k=40\), 77 basis coefficients, grid size
98,304, 97 loss components, 149 raw source forms, and reported final margin

\[
\frac{230382667}{10^{13}}>\frac1{50000}.
\]

`build_prime_gap_input_manifest` independently replays the pure rational input
generation from the evaluator pinned at commit
`61340d0b74163003b32756bb16e91d9209a5e330`. At \(k=40\) it regenerates the
reported 29/43 source ladders, their 28/39 retained prefixes, six source
groups, 97 components, 52 outer and 45 inner tasks, 149 raw forms, face
dimension 39, and convolution length 98,264. This is an exact source-manifest
replay. The separate numerical receipt below now covers the full cap and source
evaluation.

`h1_182_target` removes the terminal shift \(186\), leaving an admissible
39-tuple of diameter \(182\). `certify_bounded_gap_quadratic_witness`
checks a rationalized generalized-Rayleigh candidate exactly:

\[
\rho_*\,c^\mathsf TAc
-(1+1/50000)c^\mathsf TBc>0,
\qquad \rho_*=\frac{2624989}{10^7}.
\]

The checker requires \(c^\mathsf TBc>0\) and a positive-semidefinite
denominator form. Its `PROVED` status covers only the declared rational
matrices and coefficient vector. The source-valid derivation of \(A,B\),
outward cap and source-loss enclosures, support inequalities, distribution
inputs, and the analytic DHL implication remain explicit external premises.
The complete direct \(k=39\) receipt below misses \(1/50000\), so no crossing
is known.

At \(k=39\), the parameterized manifest gives face dimension 38 and
convolution length 98,265. It regenerates the same 97-task/149-form source
schedule and exact physical upper shell maxima. Base-shell cell indices shift
by one; event-clipped upper indices are then recomputed from their
dimension-independent rational endpoints. `prime_gap_engine_layout` also derives the mask and moment
dimensions, normalization multiplier and power, and the hidden radial midpoint
offset. The latter is exactly \(39/2\), not the \(k=40\) value 20.

`scripts/parameterize_primegaps186.py` authenticates the pinned evaluator by
SHA-256, rewrites every audited dimension dependency exactly once, refuses
source drift, and syntax-checks the generated \(k=39\) evaluator. This includes
both face-moment calls, masks, normalization, midpoint offset, driver checks,
and receipt text. Generation alone does not evaluate the Arb/FLINT cap and
source forms; a complete runtime receipt is required.
A trusted \(k=39\) run first requires a complete \(k=40\) replay with the same
authenticated source law and signed-convolution strategy.

Generated evaluators also support `--resume-log` and `--checkpoint`. Resume
loading accepts only complete JSON events whose task dictionaries exactly match
the regenerated inventory, rejects conflicting caps or duplicate rows, and
durably flushes each new component. Thus an interrupted multi-hour run can
reuse verified cap and source records without treating partial output as a
certificate; only all 97 components permit final assembly.

The standard `python-flint==0.9.0` wheel fails the upstream signed-FFT
regression. Because the corrected FLINT 3.6 patch is not published, the
generator replaces the evaluator's only signed polynomial convolution by the
exact identity \(P Q=P_+Q-P_-Q\), where all coefficients of
\(P_+,P_-,Q\) are nonnegative intervals. It retains fail-fast regressions for
nonnegative fixed-integer convolution and positive/negative Arb split
convolution. `certify_signed_convolution_split` independently replays this
identity in exact rational interval arithmetic, including sign-crossing
coefficients. This is a transparent algorithmic workaround, not the unpublished
baseline build. The complete \(k=40\) split run has now evaluated all 97
components and 149 raw forms and produced

\[
\frac{69162467338708766984467}
     {2960664736250000000000000000}
\approx2.33604523\cdot10^{-5}>\frac1{50000}.
\]

This reproduces the published margin floor and licenses the same evaluator path
for \(k=39\); it is not a bitwise comparison with the unavailable corrected
FLINT build.

The same authenticated evaluator, checkpointed until all 97 source components
were present, now has a direct \(k=39\) receipt. `certify_prime_gap_numerical_receipt`
accepts it: 97 components, 149 raw forms, and the margin recomputed with
`Fraction`,

\[
-\frac{5040770402079525193269995987}{860722528790000000000000000000}
\approx -5.85644065\cdot 10^{-3}
< \frac1{50000}.
\]

The status is `DISPROVED`. `finite_k39_crossing_found`,
`source_valid_k39_crossing_proved`, and `h1_182_claim` stay false. The
premises in `H1_182_EXTERNAL_PREMISES`, including analytic transfer to
DHL\([39,2]\), are not discharged. The sealed summary is
[`twin_prime_k39_numerical_receipt_sealed.json`](../benchmarks/twin_prime_k39_numerical_receipt_sealed.json);
its source-receipt digest is
`14de5c36bae9be4a1f73570fe7e28680da47718b33c530477d604a451f8e01c1`.

`certify_prime_gap_numerical_receipt` is the independent consumer for a
completed run. It reconstructs all 97 task parameters from the rational
manifest, rejects missing, duplicated, or altered tasks and wrong raw-form
counts, and recomputes cap rounding, every raw-relative and component-relative
ceiling, the source loss, quotient, margin, pass flag, and dimension-dependent
normalization using `Fraction`. At \(k=40\) it separately
reports whether the published margin floor was reproduced.
`scripts/verify_prime_gap_receipt.py RECEIPT --dimension 40` seals that replay
with a canonical SHA-256 binding to the complete source receipt. The public
sealed summary is
[`twin_prime_k40_numerical_receipt_sealed.json`](../benchmarks/twin_prime_k40_numerical_receipt_sealed.json);
its source-receipt digest is
`4b51346d5d268313ee92f6c7ed18ac7913b987d7236800f5b8cb636ef271d198`.

### Quadratic extraction contract

Before searching new weights, `prime_gap_quadratic_kernel_spec` fixes the
signature-major, radial-degree-minor order of all 77 coefficients and the 3,003
entries in a symmetric upper triangle. It also binds the pinned source digest,
\(k,N\), the 160/224/192-bit arithmetic policy, signed-convolution strategy,
and all 97 task keys. An extracted receipt must additionally bind its generated
source digest. `ExactSymmetricIntervalMatrix.contract`
uses the sign of each rational \(c_ic_j\) to select the sound endpoint and
applies the off-diagonal factor two exactly.

The ideal cap forms \(I,J_0,J_+,J_{\rm tail}\) and all 149 unrounded source
forms are homogeneous quadratics in these coefficients. The current reported
endpoints and decimal ceilings are not one global quadratic.
`certify_prime_gap_rounding_reserve` therefore proves a separate inventory-wide
reserve for both rounding stages:

\[
\delta=\sum_{\rm outer}
\left(\frac{y+y^{-1}}{10^{18}}+\frac1{10^{12}}\right)
+\sum_{\rm inner}
\left(\frac{d}{10^{18}}+\frac1{10^{12}}\right)
\approx 9.70094937136989\cdot10^{-11}.
\]

Thus an interval-Gram artifact can screen rational candidates with its
quadratic source bound plus \(\delta I^+\), followed by one complete direct
receipt. The descriptor and rounding contracts are proved. The five \(k=39\)
matrices have now been extracted from the pinned evaluator and accepted by
`certify_prime_gap_quadratic_receipt`: exact dyadic endpoints, both source
digests, the 160/224/192 policy, descriptor order, and the 97-task inventory.
`quadratic_kernels_extracted` is true on that certificate because the matrices
exist. Their values are not independently recomputed, a positive candidate
screen is not a crossing, and `certify_prime_gap_rounding_reserve` still
reports `quadratic_kernels_extracted` false. The sealed summary is
[`twin_prime_k39_quadratic_receipt_sealed.json`](../benchmarks/twin_prime_k39_quadratic_receipt_sealed.json);
its source-receipt digest is
`c7f881847d7582b46d79a4849140a06ff94883db16ba95b5c92ec0cd3a34b4ca`.

`certify_prime_gap_quadratic_receipt` is the fail-closed artifact consumer. It
requires all five 77-by-77 upper triangles
\((I,J_0,J_+,J_{\rm tail},S)\), exact dyadic endpoints, the authenticated source
and generated-source digests, arithmetic policy, descriptor order, and complete
task inventory. `scripts/verify_prime_gap_quadratic_receipt.py` seals that
structural replay. `certify_prime_gap_quadratic_candidate` then contracts one
rational vector, applies the signed \(J_{\rm tail}\) coefficient, cap
floor/ceiling, and the proved source reserve. A positive result remains a
matrix-screening result until the matrix values are independently validated and
the candidate passes a complete direct 97-component receipt.

## Reproducibility and status

The smoke artifact
[`twin_prime_sieve_smoke.json`](../benchmarks/twin_prime_sieve_smoke.json)
contains:

- exact local-factor prefix replay;
- coefficientwise FI (3.1), terminal-pair assignment, and dyadic replay;
- coefficientwise fixed-\(2\) signed-determinant replay plus exact
  theorem-range and kernel-cell bookkeeping;
- the exact 3-D FI terminal-shell atlas and its Lean-ready rational obligations;
- finite Möbius-bilinear diagnostics for the shifted-prime sequence;
- a parity-model negative control;
- explicit false flags for every analytic and infinite-parent claim.

The larger diagnostic is written outside the repository by
`python benchmarks/twin_prime_sieve.py --full`. It uses scales through
\(2^{20}\) for exploration and a threshold frozen in advance at the unseen
\(2^{22}\) scale. Its floating logarithmic sums remain exploratory
measurements, not sound enclosures.

The companion
[`twin_prime_bounded_gap_smoke.json`](../benchmarks/twin_prime_bounded_gap_smoke.json)
checks the named \(k=40\) baseline metadata, exact admissibility of the
diameter-182 target, and the required \(39/38\)-dimensional parameterization.
It replays the sealed \(k=40\) numerical summary. The diameter-182 quadratic
forms are assembled; the finite crossing, the source-valid \(k=39\) certificate,
and analytic instantiation stay absent.

At the fixed top-band cell \(C=4\), the unseen-scale normalized shifted-prime
mass was \(0.0194102\), below the pre-registered \(0.025\) threshold. The
parity-model control was \(0.109110\), above its \(0.08\) floor. This passes a
finite falsification test for the diagnostic; it is not the uniform Möbius
bilinear saving required by the asymptotic sieve.

Current gates:

- A0, exact local-factor algebra: earned and Lean-replayed for the finite
  prefix through \(p=97\);
- A1a, exact finite combinatorial replay: earned through \(n=128\);
- A1b0, exact structural subchecks: earned for dyadic selection, finite
  \(\rho\)-insertion, and the 13-leaf routing ledger;
- A1b1, exact fixed-\(2\) determinant transform: earned on finite divisor
  lattices without shift averaging or termwise absolute values;
- A1b, complete structural replay including smoothing and ranges: blocked;
- A1, exact classical asymptotic-sieve replay: blocked;
- A2g, exact terminal-shell geometry: earned; the rational fallback leaves
  exactly \(1/38\) of the classical log-volume;
- A2k, exact candidate-kernel cell credit: earned with floor \(41/640\);
- A2d, a loss-budgeted fixed-\(2\) determinant completion lemma: blocked;
- A2, source-complete strict analytic reduction: blocked until A1 and the
  named asymptotic inputs are independently replayed;
- A3, a new uniform saving on one remaining cell: blocked;
- A4, every analytic cell and asymptotic passage discharged: blocked.

### Advancement audit

No full Twin Prime result, improved published bounded-gap constant, or new
Möbius-cancellation estimate is established here. The \(1/38\) figure is exact
volume in the declared normalized-log coordinates; it is **not** a claim that
\(37/38\) of the analytic difficulty has been removed. The terminal shell
contains the unresolved fixed-shift parity correlation.

The concrete gains are a replayable obligation boundary, exact finite local
and atlas arithmetic, an exact signed fixed-\(2\) determinant transform with a
positive-width kernel-credit ledger, and a no-go result for finite local
preconditioning: that reweighting preserves the combined singular factor and
supplies no new parity information. The terminal-product truncation is an elementary
consequence of established divisor-moment bounds. No novelty claim is made
for that analytic observation without independent literature review.

## Public API

::: omnibias.holonomic.twin_prime
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source
