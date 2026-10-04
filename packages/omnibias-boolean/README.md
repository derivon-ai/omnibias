# omnibias-boolean

**From truth values to exact algebra.** Representations can change while every Boolean result stays fixed.

![From truth values to exact algebra.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-boolean/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-boolean/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-boolean/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-boolean/docs/visuals/scene.py)

Truth tables or Boolean polynomials enter; exact ANF, Walsh coefficients, Boolean derivatives or equation solutions leave. Use this package for finite algebraic structure; use binary for tensor quantization.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://omnibias.ai/api/boolean/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-boolean/src/omnibias/boolean) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-boolean/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Exact Boolean algebra needs neither a small bias spacing nor a temperature schedule. Optional differentiable gates connect it to temperature collapse: finite β gives a smooth relaxation and hardening requires a stated tie rule. Bias-collapse kernels can differentiate those smooth gates, but do not turn approximate gate optimization into an exact Boolean proof.

## Run this README

The examples use `omnibias-boolean` on Python >=3.10. Their installed-wheel
profile is [wheel-tests.toml](https://github.com/derivon-ai/omnibias/blob/codex/pinn-substrate-split/packages/omnibias-boolean/wheel-tests.toml); it selects runtime features, not
an editable workspace. Build the coordinated wheelhouse using the
[release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md), then run
from that main checkout:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-boolean"
```

After this opt-in prerelease is published, the equivalent index command is:

```bash
python -m pip install --pre "omnibias-boolean==0.1.0a2"
```

Existing published consumers may need the historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

Boolean functions have algebraic structure that a generic real-valued tensor does not expose. Boolean lets you move between exact finite representations, inspect interactions with discrete derivatives, and solve bounded algebraic systems. Optional tensor realizations connect those structures to learnable soft gates.

## What you can build

- Truth tables and algebraic normal forms over GF(2).
- Walsh transforms, influences and exact Boolean derivatives.
- Finite equation solvers, with optional Torch/JAX gate and spectrum operations.

Choose exact mode for truth-table identities, bounded logic tests and reference oracles. Choose a tensor realization when the purpose is optimization through a smooth model. Keeping the two separate lets you compare a learned gate with the exact finite behavior it is intended to approximate.

## A working example

```python
from omnibias.boolean import truth_table_from_callable, anf_from_truth_table
from omnibias.boolean import truth_table_from_anf

xor = truth_table_from_callable(lambda a, b: a ^ b, 2)
polynomial = anf_from_truth_table(xor)
assert truth_table_from_anf(polynomial) == xor
print(polynomial)
```

## Choose the right contract

Truth-table size is exponential in the number of variables. Exact enumeration is a bounded reference tool, not a polynomial-time solver for arbitrary large Boolean systems. A soft gate gradient and a Boolean derivative are different operators with different meanings.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/boolean.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-boolean/tests -q
```

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-boolean/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
