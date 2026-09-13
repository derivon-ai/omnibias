# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The optional shared tower preserves every established fastpath row."""

import numpy as np
import pytest
from keras import ops
from omnibias.keras.activations import get_activation


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "softplus"])
def test_registered_shared_tower(name: str) -> None:
    spec = get_activation(name)
    assert spec.tower is not None and spec.fastpath is not None
    value = ops.convert_to_tensor([-0.2, 0.0, 0.5])
    tower = spec.tower(value, 6)
    for order in range(7):
        np.testing.assert_array_equal(ops.convert_to_numpy(tower[order]),
                                      ops.convert_to_numpy(spec.fastpath(value, order)))
    with pytest.raises(ValueError, match="nonnegative"):
        spec.tower(value, -1)
