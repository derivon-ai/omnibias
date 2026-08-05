# References

External literature for the models the omnibias certified stacks validate
against. omnibias produces certificates for *finite, discrete* objects and is
structurally gated to never assert a result about the open continuum problems
these papers state -- see [Scope & guarantees](scope-and-guarantees.md).

## Fluid dynamics

- Fefferman, C. L. *Existence and Smoothness of the Navier-Stokes Equation.*
  The authoritative statement of the open global-regularity problem.
- Lions, J.-L. *Quelques methodes de resolution des problemes aux limites non
  lineaires.* Dunod, 1969. Global regularity for hyperdissipative Navier-Stokes
  (`alpha >= 5/4`) -- the proven regime the fractional track targets.
- Tao, T. "Global regularity for a logarithmically supercritical
  hyperdissipative Navier-Stokes equation." *Analysis & PDE* **2** (2009) 361-366.
- Constantin, P., Majda, A. J., Tabak, E. "Formation of strong fronts in the
  2-D quasi-geostrophic thermal active scalar." *Nonlinearity* **7** (1994) 1495-1533.

## Gauge theory

- Streater, R. F., Wightman, A. S. *PCT, Spin and Statistics, and All That.*
  W. A. Benjamin, 1964.
- Osterwalder, K., Schrader, R. "Axioms for Euclidean Green's functions."
  *Communications in Mathematical Physics* **31** (1973) 83-112; **42** (1975) 281-305.
- Wilson, K. G. "Confinement of quarks." *Physical Review D* **10** (1974) 2445.

## Rigorous numerics

- Moore, R. E., Kearfott, R. B., Cloud, M. J. *Introduction to Interval
  Analysis.* SIAM, 2009.
- Tucker, W. *Validated Numerics: A Short Introduction to Rigorous
  Computations.* Princeton University Press, 2011.
- Lohner, R. J. "Computation of guaranteed enclosures for the solutions of
  ordinary initial and boundary value problems." In *Computational Ordinary
  Differential Equations*, Clarendon Press, 1992.
- Neumaier, A., Shcherbina, O. "Safe bounds in linear and mixed-integer linear
  programming." *Mathematical Programming* **99** (2004) 283-296.

## How omnibias uses these

The `omnibias.pinn.certified.*`, `omnibias.geometry.gauge`, and `formal/` stacks
generate, schema-validate, independently replay, and machine-check certificates
for *finite* reduced models: a residual on a fixed grid, an enclosure over a
fixed box, a rollout over a fixed horizon. They never discharge the infinite
analytic obligations of the continuum problems, which remain open. See
[Scope & guarantees](scope-and-guarantees.md) and the
[Navier-Stokes tracks](cookbook/navier-stokes-tracks.md).
