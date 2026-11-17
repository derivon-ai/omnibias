# Deep bias collapse, the chain rule, and PDE discovery

The forward chain sigmoid → tanh → sin represents
\(F(x)=\sin(\tanh(\sigma(x)))\). Its derivatives can be evaluated by a
forward graph transformation, including derivatives of those results with
respect to the original model parameters. Omnibias's existing
[jet compiler](jet-compiler.md) implements this transformation for supported
operations. The important requirement is to propagate the **whole derivative
state** through each layer.

Replacing each activation by its first derivative produces a different graph.
At zero biases and unit weights, the true derivative is

\[
F'(x)=\cos(\tanh(\sigma(x)))
       \bigl(1-\tanh^2(\sigma(x))\bigr)\sigma(x)(1-\sigma(x)).
\]

Independently collapsing and normalizing each of the three local activation
pairs instead produces

\[
G(x)=\cos\left(1-\tanh^2(\sigma'(x))\right).
\]

The difference is structural: the second layer must receive the original
first-layer value along with its derivatives. Evaluating the second activation
at the first layer's derivative changes the represented function. Without the
normalization, each activation difference tends to zero.

## A first-principles construction

For derivatives through order \(N\), introduce a formal variable \(\epsilon\)
with \(\epsilon^{N+1}=0\). Store the input as \(x+\epsilon\). Each graph node
carries coefficients

\[
J_g=\sum_{k=0}^N\frac{g^{(k)}(x)}{k!}\epsilon^k.
\]

Addition acts coefficientwise, multiplication is truncated convolution, and an
activation acts by

\[
\phi(J_g)=\sum_{k=0}^N
\frac{\phi^{(k)}(g(x))}{k!}(J_g-g(x))^k
\pmod{\epsilon^{N+1}}.
\]

Induction over the computational graph proves that every node carries its
correct Taylor jet. Powers above \(N\) cannot contribute to a lower coefficient.
This is an identity of derivative coefficients under the required finite
smoothness assumptions, not a finite-spacing numerical approximation. Floating
point evaluation of the coefficients still has rounding and conditioning limits.

For example, composition yields

\[
(f\circ g)''=f''(g)(g')^2+f'(g)g'',
\]

and at higher orders the same multiplication rule generates the Bell-polynomial
terms. Unit weights are not necessary; ordinary graph multiplication propagates
their factors too. This is established Taylor-mode automatic differentiation,
not a new chain rule. See [Griewank, Utke and Walther, *Evaluating Higher Derivative
Tensors by Forward Propagation of Univariate Taylor Series*](https://doi.org/10.1090/S0025-5718-00-01120-0).

Whole-graph bias differences give another derivation:

\[
\frac1{h^N}\sum_{j=0}^N(-1)^{N-j}\binom Nj
F\bigl(x+(j-N/2)h\bigr)\longrightarrow F^{(N)}(x).
\]

Every shifted input traverses the full original graph. In a first affine layer
\(Wx+b\), an input shift \(hv\) corresponds to shifting that layer's bias by
\(hWv\); it does not require changing all subsequent layer biases. Independently
moving internal biases generally computes a different parameter-direction
derivative. The jet implementation obtains the limiting coefficients directly,
without subtracting almost-equal whole-network outputs.

## Executable deep graph example

```python
import torch
from omnibias.torch.experimental import (
    JetSpec, compile_jet, directional_seed, coefficients_to_derivatives,
)

class DeepChain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.biases = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def forward(self, x):
        return torch.sin(torch.tanh(torch.sigmoid(x + self.biases[0])
                                   + self.biases[1]) + self.biases[2])

model = DeepChain()
x = torch.tensor([[-0.4], [0.3], [0.8]], dtype=torch.float64)
spec = JetSpec.directional(4)
program = compile_jet(model, (x,), spec=spec)
tower = coefficients_to_derivatives(
    program(directional_seed(x, torch.ones_like(x), spec)), spec
)
s = torch.sigmoid(x)
expected_first = torch.cos(torch.tanh(s)) * (1-torch.tanh(s)**2) * s * (1-s)
torch.testing.assert_close(tower[1], expected_first, atol=1e-13, rtol=1e-12)
assert tower.shape == (5, 3, 1)  # raw orders zero through four

# The fourth spatial derivative still depends on the original trainable biases.
tower[4].square().mean().backward()
assert model.biases.grad is not None
assert torch.isfinite(model.biases.grad).all()
```

The dedicated regression compares this chain through order eight against
70-digit independent derivatives, checks parameter gradients, and shows both
whole-graph finite-difference convergence and the failure of local derivative
replacement. These checks concern this graph and supported compiler operations,
not arbitrary Python programs or all nonsmooth functions.

One directional jet has \(N+1\) coefficients. A complete order-\(N\) mixed jet
in \(d\) dimensions has \(\binom{d+N}{N}\) coefficients. Selected operators and
structural contractions can avoid many of them; the chain rule alone does not
remove this growth for every multivariate query.

## What this gives PDE regression

A derivative graph supplies candidate features such as \(u_t\), \(u_{xx}\),
\(u\,u_x\) and \(u_{xxxx}\) for a model \(u\). Fitting a relation between
these features is a separate identification problem. Small sampled residuals
do not prove a symbolic PDE identity, and a single trajectory can satisfy
several different candidate equations.

[Rudy et al., *Data-driven discovery of partial differential equations*](https://arxiv.org/abs/1609.06401)
develop candidate-library discovery and use multiple trajectories to distinguish
possible laws. The profile algebra introduced alongside the ridge compiler
provides a narrower route to exact identities: substitute a structured profile
representation into each candidate operator, then solve for vanishing
coefficients. It certifies identities within the supplied representation and
operator library; it does not infer the unique physical law from arbitrary data.
See [profile algebra and PDE identification](profile-pde.md) for executable examples.

## What this gives optimization

Derivatives, Hessian-vector products and local Taylor models support optimization
steps. They do not by themselves prove global optimality. Even the unshifted
chain above illustrates why: every factor of \(F'\) is strictly positive on
\(\mathbb R\), since \(0<\tanh(\sigma(x))<\tanh(1)<\pi/2\).
Therefore \(F\) is strictly increasing. On a compact interval \([a,b]\), its
minimum is \(F(a)\); on all of \(\mathbb R\), its infimum is zero as
\(x\to-\infty\), with no finite minimizer.

For a general nonconvex graph, proving a global minimum needs further structure
or global bounds. Certified interval branch-and-bound on a declared compact
domain is implemented in [global minimum enclosures](global-optimization.md),
including an immutable interval-network adapter. Graph differentiation supplies
useful local information; the global bounds provide the optimality argument.
