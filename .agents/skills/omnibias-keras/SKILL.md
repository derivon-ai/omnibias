---
name: omnibias-keras
description: Maintain Keras 3 activation operators, blocks and trainable layers across selectable backends.
---

# Maintaining omnibias-keras

Owned implementation: [source](../../../packages/omnibias-keras/src/omnibias/keras);
public surface: [API](../../../docs/api/keras.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

This package expresses its tensor work through `keras.ops`. Keep backend-specific
implementation details behind the existing abstractions; importing a selected
backend directly into an otherwise portable layer can silently make other Keras
backends unusable. Activation dispatch and operator blocks have different public
interfaces, so inspect `unit.py`, `blocks/` and the relevant activation before
changing their call signatures.

Create weights through the layer lifecycle and preserve configuration serialization,
shape inference and dtype policies. Identity initialization and growable layers
have contracts beyond forward values: changing the architecture must preserve the
intended function and optimizer-visible variables. Consult `test_identity_nesting.py`
and `test_growable.py` for those transformations.

Select `KERAS_BACKEND` before the interpreter imports Keras. Run each affected
backend in a fresh process, since switching the environment after import does not
switch an already loaded backend. The activation and shared-tower tests catch
backend-specific numerical differences; layer changes additionally need a training
step and serialization or construction check. Keras supports activation-level
operators here; field-level Torch/JAX interfaces belong to other distributions.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-keras/tests/test_shared_tower.py packages/omnibias-keras/tests/test_blocks.py packages/omnibias-keras/tests/test_growable.py -q
```
