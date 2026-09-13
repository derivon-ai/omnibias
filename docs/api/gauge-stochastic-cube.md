# Original-link SU(2) cube controls

The cube module computes exact rational geometry, original-link derivatives,
and a complete positive character-profile control. Its conditional budgets
are finite arithmetic consequences of written heat-kernel estimates. They
retain their external analytic premises and do not claim a continuum gap.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.stochastic import (
    conditional_cube_feedback_budget,
    conditional_two_face_seam_budget,
    cube_disk_geometry,
    linear_character_cube_control,
    replay_cube_bridge_certificate,
    three_face_strict_feedback_obstruction,
)

one = conditional_cube_feedback_budget()
two = conditional_two_face_seam_budget()
assert one["arithmetic"]["constant_upper"] == "39273/1280"
assert two["arithmetic"]["constant_upper"] == "12179/512"
assert len(cube_disk_geometry(2)["new_edges"]) == 5
assert replay_cube_bridge_certificate(two["certificate"])
control = linear_character_cube_control(Q(-1), tilt=Q(1, 3))
assert Q(control["arithmetic"]["total_fisher"]) >= 0
blocked = three_face_strict_feedback_obstruction(constant=100, theta=Q(3, 4))
assert Q(blocked["arithmetic"]["violation_margin"]) > 0
assert blocked["arithmetic"]["mass_gap_disproved"] is False
```

For one old bottom face, four vertices and eight links are new. Five faces
are added. The normalized heat amplitude is
\(\phi_t^2=\prod_{p=1}^5K_t(U_p)/K_{5t}(B)\). With the original-link metric,
its integrated Fisher energy is \(5F-4a-G\), where \(F\) is the one-face
bridge score square, \(a=\Delta K_{5t}/K_{5t}\), and
\(G=|\nabla\log K_{5t}|^2\). Four old-link derivatives are included.

For two adjacent old faces, the boundary has six links; five links and two
vertices are new. There are four added faces and the Fisher identity becomes
\(4(F-a)+\tfrac32(a-G)\). The internal shared old edge does not occur in
the new faces or boundary word. The uniform small-time theorem in
[the heat-kernel proof](gauge-stochastic-small-time.md) gives the respective
conditional arithmetic budgets

| Old faces | Constant | Feedback coefficient | Domain |
|---|---:|---:|---|
| One | \(39273/1280<31\) | \(2541/4900<3/4\) | \(0<\kappa\le1/64\) |
| Two adjacent | \(12179/512<24\) | \(4235/6272<3/4\) | same |

The operator interpretation is
\(\mathcal J^*H_{new}\mathcal J\le H_{old}+C+(2\theta/\kappa)\sum A_{old}\),
with \(A(U)=2-\operatorname{Tr}U\). Its Haar integration and quadratic-form
proof is maintained in the consuming ensemble-laws project's
`docs/constructive/cube-bridge-proof.md`. Finite certificate replay verifies
arithmetic, not that analytic proof. The corresponding Mathlib project
checks conditional implications and Hilbert identities.

Three corner-adjacent old faces have a sharp obstruction to this same strict
scalar bound. At boundary \(-I\), both the old and new three-face disks have
minimum total action three. Any smooth normalized real fiber extension
therefore has added energy at least \(6/\kappa\). No fixed finite constant
and \(\theta<1\) can bound this by \(C+6\theta/\kappa\) at every small coupling.
The obstruction API produces an exact violating coupling. It does not
refute a mass gap or a comparison between optimized ground energies.

The independent character control uses
\(f(U)=1+c\operatorname{Tr}U\), \(c=r/(1+r^2)\), \(|r|<1\). Its complete
Haar integrals are rational at rational boundary trace and \(r\), since
\(\sqrt{1-4c^2}=(1-r^2)/(1+r^2)\). It is a compact probability law, distinct
from the heat trial and the actual vacuum. `linear_character_cube_point`
returns the density and all 36 original-link log-amplitude derivatives at
twelve exact rational unit quaternions. This is one configuration, not an
ensemble source.

All certificate replayers reconstruct the canonical certificate, including
claim and metadata. The claim ladder is unchanged: no automatic Lean,
continuum or Yang–Mills parent promotion occurs.
