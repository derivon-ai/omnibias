# A limitation study: differentiable factoring does not break RSA

!!! danger "This is a negative result, not an attack"
    This page documents *why* a differentiable soft-gate relaxation of integer
    factoring **fails** to help. There is no attack recipe, no key-size claim, and
    nothing here threatens RSA or any deployed cryptography. The value is the
    boundary it draws around the method: a clear, reproducible demonstration that
    "make it differentiable" does not turn an NP-hard search into an easy one.

A natural hope: encode \(N = p\cdot q\) as Boolean constraints, relax the bits to
\([0,1]\) with the omnibias `beta -> inf` trick, follow the gradient to a
satisfying assignment, and read off \(p\) and \(q\) -- "massively dropping the
search space." This cookbook builds exactly that and shows the hope is false.

## The construction

The unknowns are the bits of \(p\) and \(q\); the single equation is
\(\mathrm{int}(p)\cdot\mathrm{int}(q) = N\). Its multilinear extension is the soft
residual the annealed solver descends (the
[`examples/boolean_limits`](https://github.com/derivon-ai/omnibias/tree/main/examples/boolean_limits)
experiment):

<!-- docs-test: slow -->
```python
from examples.boolean_limits.experiment import solve_one, success_vs_bitlength

solve_one(6, width=2)            # FactorAttempt(n_value=6, factors=(2, 3), verified=True)
success_vs_bitlength((2, 3, 4))  # success rate per factor bit-width -> it collapses
```

```
python -m examples.boolean_limits.run_demo
# Soft-gate factoring success vs factor bit-width (the accuracy cliff):
#   width=2  n= 4 vars  success=100.0%  ########################################
#   width=3  n= 6 vars  success= ~..%   ########
#   width=4  n= 8 vars  success=  ~.%
```

Every reported "success" is an **exact verification** of the Boolean system
(propose-and-verify); the solver never reports an unchecked guess. The success
rate falls off a cliff as the bit-width grows.

## Why the search space does not drop

**1. Parameter count \(\ge\) preimage entropy.** The relaxation introduces one
continuous latent per unknown bit. A semiprime with \(b\)-bit factors has \(\sim b\)
bits of preimage entropy, and the solver carries \(\sim b\) latents. Relaxation is
a *change of variables*, not a reduction -- the free parameter \(c\) of the
[reproductive solution](boolean-equations.md) *is* the search space, re-encoded.
Worse, even *building* the constraint truth table is \(2^n\): the exponential is
still there.

**2. XOR / carry gradient flatness.** Multiplication's bit structure is dominated
by parity (XOR) and carry chains. A parity \(x_1\oplus\dots\oplus x_k\) has a flat
multilinear extension at the cube center -- every first-order partial vanishes, so
first-order gradient information about any single bit is \(O(2^{-k})\). Carries
compound this: the loss landscape is a high-dimensional field of near-degenerate
plateaus and exponentially many spurious minima. Gradient descent has no usable
slope to follow toward the unique satisfying corner.

**3. The \(\beta\) Dirac-limit trade-off.** Sharpening the surrogate
(\(\beta\to\infty\)) is what makes the forward hard, but the surrogate gradient
\(\beta\,\sigma'(\beta z)\) concentrates into a \(2\delta\) Dirac spike at the
decision boundary (the same homotopy as
[better-than-STE binary training](better-than-ste-binary.md)). Small \(\beta\):
smooth but the relaxed optimum sits in the cube interior, far from any integer.
Large \(\beta\): the gradient is zero almost everywhere. There is no schedule that
removes the underlying combinatorial hardness -- it only trades smoothness for
sharpness.

**No-reduction sketch.** If a poly-time, poly-parameter differentiable relaxation
reliably recovered factors, composing it with the (poly-time) verification step
would put `FACTORING` in `BPP`. No such collapse is known or expected, and the
cliff above is the empirical shadow of that barrier.

## Where the differentiable view *does* help

The honest, useful role of this machinery is the **linear** substructure and
heuristic front-ending, never the hard core:

- the **GF(2) fast-path** (`gf2_solve`) solves XOR-linear subsystems exactly in
  polynomial time -- genuine search-space collapse, but only for the linear part;
- the spectrum/influence tools surface structure (low-degree components, biased
  bits) that a classical solver can exploit;
- as a **proposer** in a propose-and-verify loop, the relaxation can suggest
  candidates that an exact verifier (or SAT/SMT solver) then checks.

In other words: `omnibias-boolean` is a differentiable front-end and exact
spectral/equation core that *feeds* classical verifiers. It is not, and does not
try to be, a sound-and-complete reasoner or a cryptanalytic tool.
