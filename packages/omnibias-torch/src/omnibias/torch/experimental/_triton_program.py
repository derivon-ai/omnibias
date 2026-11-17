# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Straight-line Triton programs for the fused stream kernels.

Triton has no per-channel lists and no in-kernel autodiff, so the kernels are *generated*: the elementwise update of one
layer is traced once into an SSA graph (:class:`Graph`), its reverse-mode adjoint is derived mechanically from the same
graph (so the backward kernel cannot drift from the forward program, for any order or activation), and both are emitted as
Triton source with one named register tile per stream channel. The generated module is written to a cache directory and
imported (Triton compiles from source files), then compiled per configuration by Triton's own cache.

Constants must be exactly representable in float32: Triton materializes Python floats as float32 constants even next to a
float64 tile, so ``0.1`` would silently lose digits while ``3.0`` and ``-2.0`` (every Faa di Bruno and activation-polynomial
coefficient) are exact. The graph refuses an inexact constant.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from omnibias.core.polynomials import sigmoid_polynomial_coeffs, tanh_polynomial_coeffs

__all__ = ["Cfg", "Graph", "V", "build_layer", "compile_kernels", "kernel_sources"]


class Graph:
    """Append-only SSA graph of elementwise operations on equally shaped (or broadcastable) tiles."""

    def __init__(self) -> None:
        self.ops: list[tuple[str, int, int, Any]] = []

    def node(self, op: str, a: int = -1, b: int = -1, k: Any = None) -> V:
        if op in ("mulc", "addc"):
            kf = float(k)
            if (
                float(np.float32(kf)) != kf
            ):  # NumPy 2 compares float32 with a Python float in float32; widen first
                raise ValueError(f"constant {kf!r} is not exactly representable in float32")
        self.ops.append((op, a, b, k))
        return V(self, len(self.ops) - 1)

    def input(self, name: str) -> V:
        return self.node("in", k=name)


class V:
    """A value in a :class:`Graph` with the arithmetic needed to write the layer updates once, generically."""

    __slots__ = ("g", "i")

    def __init__(self, g: Graph, i: int) -> None:
        self.g, self.i = g, i

    def __add__(self, o: Any) -> V:
        return (
            self.g.node("add", self.i, o.i)
            if isinstance(o, V)
            else self.g.node("addc", self.i, k=o)
        )

    __radd__ = __add__

    def __sub__(self, o: Any) -> V:
        return (
            self.g.node("sub", self.i, o.i)
            if isinstance(o, V)
            else self.g.node("addc", self.i, k=-float(o))
        )

    def __rsub__(self, o: Any) -> V:
        return self.g.node("addc", self.g.node("neg", self.i).i, k=o)

    def __mul__(self, o: Any) -> V:
        return (
            self.g.node("mul", self.i, o.i)
            if isinstance(o, V)
            else self.g.node("mulc", self.i, k=o)
        )

    __rmul__ = __mul__

    def __neg__(self) -> V:
        return self.g.node("neg", self.i)

    def tanh(self) -> V:
        return self.g.node("tanh", self.i)

    def sig(self) -> V:
        return self.g.node("sig", self.i)

    def sin(self) -> V:
        return self.g.node("sin", self.i)

    def cos(self) -> V:
        return self.g.node("cos", self.i)

    def softplus(self) -> V:
        return self.g.node("sp", self.i)


def _horner(coeffs: tuple[float, ...], x: V) -> V:
    """``sum_k coeffs[k] x^k`` (degree >= 1) by Horner's rule on graph values."""
    deg = len(coeffs) - 1
    acc = x * float(coeffs[deg]) + float(coeffs[deg - 1])
    for k in range(deg - 2, -1, -1):
        acc = acc * x + float(coeffs[k])
    return acc


def tower(name: str, z: V, n: int) -> list[V]:
    """``[sigma(z), ..., sigma^(n)(z)]`` from one base node, mirroring the JAX kernel tower."""
    if name == "tanh":
        t = z.tanh()
        return [t] + [_horner(tuple(tanh_polynomial_coeffs(m)), t) for m in range(1, n + 1)]
    if name in ("sigmoid", "softplus"):
        s = z.sig()
        deriv = [_horner(tuple(sigmoid_polynomial_coeffs(m)), s) for m in range(1, n + 1)]
        if name == "sigmoid":
            return [s] + deriv
        sp = z.softplus()
        return [sp] if n == 0 else [sp, s] + deriv[: n - 1]
    if name in ("sin", "cos"):
        s, c = z.sin(), z.cos()
        cycle = [s, c, -s, -c] if name == "sin" else [c, -s, -c, s]
        return [cycle[m % 4] for m in range(n + 1)]
    raise ValueError(f"activation {name!r} has no fused kernel tower")


def faa_di_bruno(s: list[V], u: list[list[V]]) -> list[list[V]]:
    """Physical derivatives ``(sigma o u)^(p)`` per direction; ``u[p-1][j]`` is the p-th coefficient of direction j."""
    n, K = len(u), len(u[0])
    out: list[list[V]] = [[s[1] * u[0][j] for j in range(K)]]
    if n >= 2:
        out.append([s[1] * u[1][j] + s[2] * (u[0][j] * u[0][j]) for j in range(K)])
    if n >= 3:
        out.append(
            [
                s[1] * u[2][j]
                + (s[2] * (u[0][j] * u[1][j])) * 3.0
                + s[3] * (u[0][j] * (u[0][j] * u[0][j]))
                for j in range(K)
            ]
        )
    if n >= 4:
        out.append(
            [
                s[1] * u[3][j]
                + s[2] * ((u[1][j] * u[1][j]) * 3.0 + (u[0][j] * u[2][j]) * 4.0)
                + (s[3] * ((u[0][j] * u[0][j]) * u[1][j])) * 6.0
                + s[4] * ((u[0][j] * u[0][j]) * (u[0][j] * u[0][j]))
                for j in range(K)
            ]
        )
    return out


class Cfg(NamedTuple):
    kind: str  # "lap" | "jet"
    order: int
    nk: int
    acts: tuple[str, ...]
    P: int
    Cp: int
    Kp: int
    BB: int
    dtype: str  # "float64" | "float32"
    highest: bool

    @property
    def S(self) -> int:
        return self.nk + 2 if self.kind == "lap" else 1 + self.order * self.nk

    def key(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()[:16]


class LayerProgram(NamedTuple):
    graph: Graph
    z: list[V]
    w: list[V]
    out: list[V]


def build_layer(cfg: Cfg, act: str) -> LayerProgram:
    """The elementwise update of one layer as a graph: pre-activation channels (and weight rows) -> output channels."""
    g = Graph()
    Z = [g.input(f"Z{c}") for c in range(cfg.S)]
    nk = cfg.nk
    if cfg.kind == "lap":
        W = [g.input(f"W{c}") for c in range(nk)]
        s0, s1, s2 = tower(act, Z[0], 2)
        q2: V | None = None
        for c in range(nk):
            t = (Z[1 + c] * Z[1 + c]) * W[c]
            q2 = t if q2 is None else q2 + t
        assert q2 is not None
        out = [s0] + [s1 * Z[1 + c] for c in range(nk)] + [s2 * q2 + s1 * Z[nk + 1]]
        return LayerProgram(g, Z, W, out)
    n = cfg.order
    s = tower(act, Z[0], n)
    u = [[Z[1 + p * nk + j] for j in range(nk)] for p in range(n)]
    new = faa_di_bruno(s, u)
    return LayerProgram(g, Z, [], [s[0]] + [a for p in range(n) for a in new[p]])


# ------------------------------------------------------------------------------------------------ emission
def _sigmoid_lines(dst: str, a: str, tmp: str) -> list[str]:
    return [
        f"{tmp} = tl.exp(-tl.abs({a}))",
        f"{dst} = tl.where({a} >= 0, 1.0 / (1.0 + {tmp}), {tmp} / (1.0 + {tmp}))",
    ]


def emit_forward(g: Graph, prefix: str, bind: dict[int, str]) -> tuple[list[str], dict[int, str]]:
    """Source lines of the graph; ``bind`` names the input nodes. Returns the lines and every node's variable name."""
    name: dict[int, str] = {}
    lines: list[str] = []
    for i, (op, a, b, k) in enumerate(g.ops):
        v = f"{prefix}{i}"
        if op == "in":
            name[i] = bind[i]
            continue
        name[i] = v
        A = name.get(a, "")
        B = name.get(b, "")
        if op == "add":
            lines.append(f"{v} = {A} + {B}")
        elif op == "sub":
            lines.append(f"{v} = {A} - {B}")
        elif op == "mul":
            lines.append(f"{v} = {A} * {B}")
        elif op == "mulc":
            lines.append(f"{v} = {A} * {float(k)!r}")
        elif op == "addc":
            lines.append(f"{v} = {A} + {float(k)!r}")
        elif op == "neg":
            lines.append(f"{v} = -{A}")
        elif op == "tanh":
            lines += [
                f"{v}_e = tl.exp(-2.0 * tl.abs({A}))",
                f"{v}_r = (1.0 - {v}_e) / (1.0 + {v}_e)",
                f"{v} = tl.where({A} >= 0, {v}_r, -{v}_r)",
            ]
        elif op == "sig":
            lines += _sigmoid_lines(v, A, f"{v}_e")
        elif op == "sin":
            lines.append(f"{v} = tl.sin({A})")
        elif op == "cos":
            lines.append(f"{v} = tl.cos({A})")
        elif op == "sp":
            lines += [
                f"{v}_e = tl.exp(-tl.abs({A}))",
                f"{v} = tl.maximum({A}, 0.0) + tl.log(1.0 + {v}_e)",
            ]
        else:  # pragma: no cover
            raise AssertionError(op)
    return lines, name


def emit_reverse(
    g: Graph, prefix: str, name: dict[int, str], seeds: dict[int, str], wanted: list[int]
) -> tuple[list[str], dict[int, str]]:
    """Adjoint lines given forward variable ``name`` and output adjoint variable ``seeds``; returns the lines and the
    adjoint variable of each ``wanted`` node (a node that influences no output gets an explicit zero)."""
    adj: dict[int, str] = dict(seeds)
    lines: list[str] = []
    counter = [0]

    def push(target: int, expr: str) -> None:
        if target in adj:
            counter[0] += 1
            new = f"{prefix}a{target}_{counter[0]}"
            lines.append(f"{new} = {adj[target]} + ({expr})")
            adj[target] = new
        else:
            new = f"{prefix}a{target}"
            lines.append(f"{new} = {expr}")
            adj[target] = new

    for i in range(len(g.ops) - 1, -1, -1):
        op, a, b, k = g.ops[i]
        if op == "in" or i not in adj:
            continue
        d = adj[i]
        if op == "add":
            push(a, d)
            push(b, d)
        elif op == "sub":
            push(a, d)
            push(b, f"-{d}")
        elif op == "mul":
            push(a, f"{d} * {name[b]}")
            push(b, f"{d} * {name[a]}")
        elif op == "mulc":
            push(a, f"{d} * {float(k)!r}")
        elif op == "addc":
            push(a, d)
        elif op == "neg":
            push(a, f"-{d}")
        elif op == "tanh":
            push(a, f"{d} * (1.0 - {name[i]} * {name[i]})")
        elif op == "sig":
            push(a, f"{d} * {name[i]} * (1.0 - {name[i]})")
        elif op == "sin":
            push(a, f"{d} * tl.cos({name[a]})")
        elif op == "cos":
            push(a, f"-{d} * tl.sin({name[a]})")
        elif op == "sp":
            tmp = f"{prefix}s{i}"
            lines += _sigmoid_lines(tmp, name[a], f"{tmp}_e")
            push(a, f"{d} * {tmp}")
    out: dict[int, str] = {}
    for w in wanted:
        if w in adj:
            out[w] = adj[w]
        else:
            out[w] = f"{prefix}z{w}"
            lines.append(f"{out[w]} = tl.zeros({name[w]}.shape, {name[w]}.dtype)")
    return lines, out


# ------------------------------------------------------------------------------------------ kernel sources
def _dot(a: str, b: str, cfg: Cfg) -> str:
    if cfg.dtype == "float64":
        return f"tl.dot({a}, {b}, out_dtype=tl.float64)"
    prec = '"ieee"' if cfg.highest else '"tf32"'
    return f"tl.dot({a}, {b}, input_precision={prec})"


def _arglist(cfg: Cfg, backward: bool) -> list[str]:
    n_hid = len(cfg.acts) - 1
    args = (
        ["u1_ptr", "M_ptr", "w_ptr"]
        + [f"W{i}_ptr" for i in range(1, n_hid + 1)]
        + [f"b{i}_ptr" for i in range(1, n_hid + 1)]
    )
    args += ["Wo_ptr", "bo_ptr"]
    if backward:
        args += [f"ct{c}_ptr" for c in range(cfg.S)]
        args += (
            ["ubar_ptr", "mbar_ptr"]
            + [f"dW{i}_ptr" for i in range(1, n_hid + 1)]
            + [f"db{i}_ptr" for i in range(1, n_hid + 1)]
        )
        args += ["dWo_ptr", "dbo_ptr"]
    else:
        args += [f"out{c}_ptr" for c in range(cfg.S)]
    return args


def _forward_body(
    cfg: Cfg,
) -> tuple[list[str], dict[tuple[int, int], str], dict[int, tuple[LayerProgram, dict[int, str]]]]:
    """Lines of the forward pass; ``x[(l, c)]`` names output channel ``c`` of layer ``l``."""
    S, nk = cfg.S, cfg.nk
    L: list[str] = [
        "pid = tl.program_id(0)",
        "cols = tl.arange(0, P)",
        "ccols = tl.arange(0, CP)",
        "orow = cols[:, None]",
        "icol = cols[None, :]",
        "rows = pid * BB + tl.arange(0, BB)",
        "tile = rows[:, None] * P + cols[None, :]",
        "z0_0 = tl.load(u1_ptr + tile)",
    ]
    for c in range(nk):
        L.append(f"z0_{1 + c} = tl.zeros((BB, P), DT) + tl.load(M_ptr + {c} * P + cols)[None, :]")
    for c in range(1 + nk, S):
        L.append(f"z0_{c} = tl.zeros((BB, P), DT)")
    if cfg.kind == "lap":
        for c in range(nk):
            L.append(f"wr_{c} = tl.load(w_ptr + {c} * P + cols)[None, :]")
    x: dict[tuple[int, int], str] = {}
    programs: dict[int, tuple[LayerProgram, dict[int, str]]] = {}
    for layer, act in enumerate(cfg.acts):
        if layer > 0:
            L.append(f"W{layer} = tl.load(W{layer}_ptr + orow * P + icol)")
            L.append(f"b{layer} = tl.load(b{layer}_ptr + cols)[None, :]")
            for c in range(S):
                L.append(f"z{layer}_{c} = {_dot(f'x{layer - 1}_{c}', f'tl.trans(W{layer})', cfg)}")
            L.append(f"z{layer}_0 = z{layer}_0 + b{layer}")
        prog = build_layer(cfg, act)
        bind = {v.i: f"z{layer}_{c}" for c, v in enumerate(prog.z)}
        bind.update({v.i: f"wr_{c}" for c, v in enumerate(prog.w)})
        lines, name = emit_forward(prog.graph, f"n{layer}_", bind)
        L += lines
        for c, v in enumerate(prog.out):
            x[(layer, c)] = f"x{layer}_{c}"
            L.append(f"x{layer}_{c} = {name[v.i]}")
        programs[layer] = (prog, name)
    L += [
        "Wo = tl.load(Wo_ptr + ccols[:, None] * P + icol)",
        "bo = tl.load(bo_ptr + ccols)[None, :]",
    ]
    return L, x, programs


def kernel_sources(cfg: Cfg) -> str:
    """Python source of the forward and backward kernels of ``cfg`` (module text, importable)."""
    S, n, nk = cfg.S, len(cfg.acts), cfg.nk
    fwd, x, programs = _forward_body(cfg)
    last = n - 1
    header = ["import triton", "import triton.language as tl", ""]
    consts = (
        "P: tl.constexpr, CP: tl.constexpr, KP: tl.constexpr, BB: tl.constexpr, DT: tl.constexpr"
    )
    # ---- forward
    f = header + ["@triton.jit", f"def fused_forward({', '.join(_arglist(cfg, False))}, {consts}):"]
    body = list(fwd)
    for c in range(S):
        body.append(
            f"o{c} = {_dot(f'x{last}_{c}', 'tl.trans(Wo)', cfg)}" + (" + bo" if c == 0 else "")
        )
        body.append(f"tl.store(out{c}_ptr + rows[:, None] * CP + ccols[None, :], o{c})")
    f += ["    " + ln for ln in body]
    # ---- backward
    b = header + ["@triton.jit", f"def fused_backward({', '.join(_arglist(cfg, True))}, {consts}):"]
    body = list(fwd)
    for c in range(S):
        body.append(f"ct{c} = tl.load(ct{c}_ptr + rows[:, None] * CP + ccols[None, :])")
        body.append(f"xb{last}_{c} = {_dot(f'ct{c}', 'Wo', cfg)}")
    body.append(
        "dWo = " + " + ".join(_dot(f"tl.trans(ct{c})", f"x{last}_{c}", cfg) for c in range(S))
    )
    body.append("tl.store(dWo_ptr + pid * CP * P + ccols[:, None] * P + icol, dWo)")
    body.append("tl.store(dbo_ptr + pid * CP + ccols, tl.sum(ct0, axis=0))")
    for layer in range(last, -1, -1):
        prog, name = programs[layer]
        seeds = {v.i: f"xb{layer}_{c}" for c, v in enumerate(prog.out)}
        wanted = [v.i for v in prog.z]
        lines, adj = emit_reverse(prog.graph, f"r{layer}_", name, seeds, wanted)
        body += lines
        for c, v in enumerate(prog.z):
            body.append(f"zb{layer}_{c} = {adj[v.i]}")
        if layer == 0:
            body.append("tl.store(ubar_ptr + tile, zb0_0)")
            for c in range(nk):
                body.append(
                    f"tl.store(mbar_ptr + pid * KP * P + {c} * P + cols, tl.sum(zb0_{1 + c}, axis=0))"
                )
        else:
            dw = " + ".join(
                _dot(f"tl.trans(zb{layer}_{c})", f"x{layer - 1}_{c}", cfg) for c in range(S)
            )
            body.append(f"dW{layer} = {dw}")
            body.append(f"tl.store(dW{layer}_ptr + pid * P * P + orow * P + icol, dW{layer})")
            body.append(f"tl.store(db{layer}_ptr + pid * P + cols, tl.sum(zb{layer}_0, axis=0))")
            for c in range(S):
                body.append(f"xb{layer - 1}_{c} = {_dot(f'zb{layer}_{c}', f'W{layer}', cfg)}")
    b += ["    " + ln for ln in body]
    return "\n".join(f) + "\n\n\n" + "\n".join(b[len(header) :]) + "\n"


_MODULES: dict[Cfg, Any] = {}  # keyed by the (hashable) configuration: no hashing of its repr on every launch


def _cache_dir() -> Path:
    base = os.environ.get("OMNIBIAS_CACHE_DIR")
    root = (
        Path(base)
        if base
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "omnibias"
    )
    path = root / "triton_programs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def compile_kernels(cfg: Cfg) -> Any:
    """Import (writing it first) the generated module of ``cfg``: ``.fused_forward`` and ``.fused_backward``."""
    mod = _MODULES.get(cfg)
    if mod is not None:
        return mod
    key = cfg.key()
    src = kernel_sources(cfg)
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = _cache_dir() / f"prog_{key}_{digest}.py"
    if not path.exists():
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(src)
        os.replace(tmp, path)
    name = f"omnibias_triton_{key}_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULES[cfg] = mod
    return mod
