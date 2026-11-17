# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent value/bias-gradient accuracy of both operator activation providers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _common import provenance, write_json
from jet_compiler import _configure_process, accuracy
from operator_compiler import snapshots, source_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="operator_accuracy.json")
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        raise ValueError("output must be a filename within the artifact directory")
    _configure_process(1)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import mpmath as mp
    import torch
    from omnibias.torch.experimental import OperatorSpec, compile_operator

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    points = [-40., -20., -4., -2., -.3, 0., .3, 2., 4., 20., 40.]
    x = torch.tensor(points, dtype=torch.float64).reshape(-1, 1)
    rows = []
    for name in ("tanh", "sigmoid"):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1, dtype=torch.float64),
                                   torch.nn.Tanh() if name == "tanh" else torch.nn.Sigmoid())
        with torch.no_grad():
            model[0].weight.fill_(1)
            model[0].bias.zero_()
        for order in (1, 2, 4, 8, 16, 32):
            for method in ("recurrence", "polynomial"):
                program = compile_operator(model, (x,), operator=OperatorSpec.mixed((order,)),
                                           batch_ndim=1, activation_method=method)
                values = program(x).reshape(-1)
                with mp.workdps(100):
                    function = mp.tanh if name == "tanh" else lambda z: 1/(1+mp.exp(-z))
                    for i, point in enumerate(points):
                        # tanh and sigmoid-1/2 are odd; their positive even
                        # derivatives vanish exactly at zero. Do not turn
                        # numerical differentiation noise into relative error.
                        wanted = "0" if point == 0 and order % 2 == 0 else mp.nstr(mp.diff(function, mp.mpf(point), order), 100)
                        slope = "0" if point == 0 and (order+1) % 2 == 0 else mp.nstr(mp.diff(function, mp.mpf(point), order+1), 100)
                        gradient = torch.autograd.grad(values[i], model[0].bias, retain_graph=True)[0].item()
                        rows.append({"activation": name, "order": order, "method": method, "input": point,
                                     "value": accuracy(float(values[i].detach()), wanted),
                                     "bias_gradient": accuracy(gradient, slope)})
    path = write_json(args.output, {
        "schema": "omnibias/operator-accuracy/v1", "rows": rows,
        "scope": "float64 CPU, raw derivatives and native reverse-mode bias gradients; independent100-digit references; no timing claims",
        "source_provenance": snapshots(source_paths()+["benchmarks/operator_accuracy.py"]),
        "environment": provenance(schema="omnibias/operator-accuracy/v1", config={})})
    worst = {method: max(row["value"]["relative_error"] or 0 for row in rows if row["method"] == method)
             for method in ("recurrence", "polynomial")}
    print(json.dumps({"path": str(path), "maximum_sampled_value_relative_error": worst}))


if __name__ == "__main__":
    main()
