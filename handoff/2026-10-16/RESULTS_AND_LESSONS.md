# Results and lessons carried to the A100 machine

All ratios are baseline elapsed time / Omnibias elapsed time. Above one favors
Omnibias. Never multiply ratios from different workloads, versions or cost
regimes. Numerical agreement, conditional bounds and measured speed are separate.

## Established scoped results

| Experiment | Result | Meaning and boundary |
|---|---|---|
| Historical ridge Delta^4, D=H=16, B=32 | 5,035.68x vs nested folx forward | Not a speedup over structure-aware Taylor; no parameter backward/training in this ratio |
| Same original folx experiment, first Delta | 0.803x | Folx was already faster in the original October 3 record; prior "better everywhere" wording was wrong, not a newly measured regression |
| Optimized closed ridge, CPU1 B4096, Delta | Forward1.436x, value+gradient1.275x | Strong Taylor receives the same ridge contraction |
| Same, Delta^2 (spatial order4) | Forward1.678x, value+gradient1.511x | VG lower95=1.506 |
| Same, Delta^3 (spatial order6) | Forward2.684x, value+gradient2.193x | VG lower95=2.162; practical lower-order result |
| Same, Delta^4 (spatial order8) | Forward4.306x, value+gradient3.202x | VG lower95=3.146; separate from the 5036x experiment |
| Original-kernel backward optimization itself | B4096 Delta^4 old/newVG1.314x | Forward HLO unchanged; shorter differentiated polynomial circuit is the real change |
| T1200 closed ridge Delta^4, B4096 | Forward1.657x, VG1.248x | Both steady lower bounds>=1; neither2x; no A100 result yet |
| Earlier separable D8 specialization, CPU1 B4096 | VG5.787x (lower5.475) | Cold0.156x; setup+100-step accounting0.577x; not all-regime promotion |
| Later shared closed ridge CPU1 B32 | VG1.186x, cold1.477x, training1.733x | Passed that earlier experiment's all-regime gate, not2x significant; not promotion of latest hardened sources |
| Later shared closed ridge CPU1 B4096 | VG3.133x (lower3.063) | Taylor controls could not qualify the first update; no comparable training speed ratio |
| Later automatic planner, CPU1/CPU8/GPU | MedianVG0.982/0.994/0.985x | Near ties; cold medians0.062/0.081/0.056x; setup is a major problem |
| Frozen-readout hostQR vs Taylor+sameQR, 1000queries | CPU8 median1.136x; GPU1.016x | Only1/3 CPU configs and0/2 GPU configs pass lower-bound gate; not robust2x |
| Generic functions | CPU VG gate5/12; GPU5/12 | Both forward+VG only2/12CPU,3/12GPU; no generic2x result |
| Earlier PDE162 | All reach1e-3;117 reach1e-6 | Spectral controls all reach1e-6; compact heat9/18, compact Burgers0/36 at1e-6; no fair PDE speed ratio |
| Original frozen private package | CPU68 and asserted T1200 GPU68 pass | Subsequent hardening/runtime proposals remain to be numerically validated |

The complete tables and raw evidence are in the exported research directories:

- `original_kernel_audit_20261011/REPORT.txt`, `audit/comparison_table.json`.
- `generic_api_audit_20261012/REPORT.txt`, `comparison.csv`, `audit/summary_all.json`.
- `operator_specialization_20261011/REPORT.txt` (both policies, both batches, both CPU sizes).
- `reuse_execution_20261012/BENCHMARK_REPORT.md`, `comparison_v2.csv`,
  `reuse_comparison.csv`, `audit_v2.json`, `reuse_audit.json`, `pde_audit_v2.json`.
- `ridge_taylor_comparison_20261011/comparison_report.txt`, `accuracy_audit.json`.

## Accuracy is a genuine opportunity, not a blanket guarantee

Omnibias polynomial and Omnibias stable recurrence are different providers.
The stable recurrence is itself a Taylor-series method. It initializes tanh's
slope with 4*sigmoid(2z)*sigmoid(-2z), avoiding subtraction after saturation.
The tested native JAX0.10.2 Taylor rule uses sigmoid*(1-sigmoid) internally.

At derivative16,z3, observed relative errors were polynomial1.805e-7,
stable6.782e-16, JAX2.588e-13: about382x smaller error for stable vs JAX.
At derivative8,z20, stable retains the approximately-2.175e-15 derivative while
the tested polynomial/JAX rules return zero. This tiny absolute error may still
meet an absolute tolerance. At derivative16,z0.1, polynomial1.26e-17 beats
stable2.29e-15 and JAX7.95e-16. No method dominates the whole small grid.
Order32 remains experimental. References are independent high precision, not
vendor guarantees or uniform-domain certificates.

Generic CPU B32 rational/order8 VG was0.696x with equal recorded worst aggregate
gradient discrepancy4.657e-10 for both methods; it did not buy a large accuracy
advantage. Nested/order8 CPU was0.929x with observed worst gradient discrepancy
2.274e-13 vs2.728e-12 (12x smaller). These finite-set maxima are not universal
error ratios. B4096 aggregate gradients were cross-method checked rather than
independently high-precision certified in that earlier generic experiment.

Never attach stable-provider errors to polynomial-provider speed measurements.
Report values/objectives/full gradients and time at the same declared accuracy
for the SAME executable and domain.

## Mathematical insights worth retaining

- Bias collapse is a normalized difference/derivative limit. A hyperplane is
  generally a level-set geometry, not the derivative function itself.
- For ridge u=sum_h c_h phi(w_h.x+b_h), Delta^k u=sum_h c_h||w_h||^(2k)phi^(2k).
  It avoids ambient derivative tensors, but still pays projection/readout work.
- Operator and adjoint demands should be compiled together. Preserve derivatives
  of operator symbols and live geometry, not just activation derivatives.
- For z=AX+b and G=AA^T, Abar=zbar^T X+(Gbar+Gbar^T)A. Do not detach G or omit
  the second term. Accumulate prepared adjoints across chunks before one pullback.
- Terminal separable identities can avoid enormous Bell tables. Generic layouts
  still have combinatorial growth when no structural reduction is proved.
- Fixed linear-residual designs allow QR compression and readout elimination.
  Taylor features deserve the same QR control. Hidden-feature training needs
  proper variable projection/envelope qualification; nonlinear-in-readout
  Burgers residuals cannot use the same linear elimination silently.
- Reuse has structure, frozen-data, optimizer-step and evaluation lifetimes.
  Never cache live projections across parameter updates without proven validity.
- Low-order factored tanh candidates with t=tanh(z),s=sech^2(z):
  phi2=-2ts; phi4=8ts(3s-1); phi6=-16ts(2-30s+45s^2).
  Exact integer-polynomial identities checked; not new mathematics or an earned
  floating-point/runtime improvement. Actual gradient circuits must qualify.
- Constant accumulator count is not O(1) total space. Arrays, derivative state,
  parameters, outputs, reverse state and compiler workspaces still scale.

## Highest-value directions, in priority order

1. Practical orders2–6: operator contractions, shared residual/backward graphs,
   short qualified activation providers, cold/setup amortization and A100 tests.
2. Reduce whole-program work before writing native kernels. Both implementations
   already compile with XLA; replacing Python notation alone need not help.
3. Close real numerical/gradient/domain/resource validation of the existing
   reviewed prototypes. Avoid another endless source-only review cycle.
4. Reuse setup/certificates/executables safely. Measure exact break-even counts;
   expected reuse defaults to one, not optimistic infinite training reuse.
5. Variable projection/time-to-target for applicable trainable-feature linear
   residuals; compare equal Taylor+QR/envelope controls and joint training.
6. Native fused kernels only for measured remaining hotspots. FFT/composition
   is a high-order experiment, not the first optimization for orders2–6.

Useful antecedents: JAX Taylor AD, derivative polynomials, apolarity, Herbie,
Daisy, FPTaylor, FFTW planning, QR/SVD, variable projection, spectral methods,
folx forward Laplacian and known fast composition. The integrated engineering
advantage/novelty must be demonstrated; no historical priority, patentability,
general PDE-to-LP or sustainable moat is established merely by these identities.
