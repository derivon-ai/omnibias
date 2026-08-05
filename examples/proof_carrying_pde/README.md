# Proof-carrying PDE demo

This example is the smallest proof-carrying scientific neural-operator loop in
the public tree:

1. describe a network as certified-jet layers,
2. enclose its PDE residual over a domain,
3. combine that residual with an explicit stability estimate,
4. seal the resulting a-posteriori error certificate, and
5. ask the proof machine for a replay/schema/honesty-gated verdict.

Run from the repository root:

```bash
python -m examples.proof_carrying_pde.run_demo
```

The example proves a model statement for a manufactured harmonic affine field. It
does not claim any continuum theorem.
