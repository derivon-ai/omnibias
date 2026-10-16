"""Fixed Fourier models, exact heat, Cole--Hopf and dealiased ETDRK4.

All methods are classical scientific controls. These are architecture/solver
experiments, never exact acceleration claims for the frozen tanh networks.
"""
import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import mpmath as mp
import jax
import jax.numpy as jnp
ROOT = Path(__file__).resolve().parent

def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + '\n')

def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reference(kind, nu, x, t, precision=110):
    if kind == 'heat':
        u = np.exp(-nu * t) * np.sin(x) + 0.5 * np.exp(-4 * nu * t) * np.sin(2 * x)
        ux = np.exp(-nu * t) * np.cos(x) + np.exp(-4 * nu * t) * np.cos(2 * x)
        uxx = -np.exp(-nu * t) * np.sin(x) - 2 * np.exp(-4 * nu * t) * np.sin(2 * x)
        return np.stack((u, ux, uxx, nu * uxx), axis=-1)
    result = []
    with mp.workdps(precision):
        viscosity = mp.mpf(float(nu))
        c = 1 / (2 * viscosity)
        coefficients = [mp.exp(-c) * mp.besseli(n, c) for n in range(257)]
        for xi, ti in zip(x, t):
            xx = mp.mpf(float(xi))
            tt = mp.mpf(float(ti))
            p = [coefficients[0], mp.mpf(0), mp.mpf(0), mp.mpf(0)]
            co = mp.cos(xx)
            si = mp.sin(xx)
            cn = mp.mpf(1)
            sn = mp.mpf(0)
            for n in range(1, 257):
                cn, sn = (cn * co - sn * si, sn * co + cn * si)
                a = 2 * coefficients[n] * mp.exp(-viscosity * n * n * tt)
                p[0] += a * cn
                p[1] -= n * a * sn
                p[2] -= n * n * a * cn
                p[3] += n ** 3 * a * sn
            if p[0] <= 0:
                raise ArithmeticError('Cole--Hopf positivity not resolved')
            ratio = p[1] / p[0]
            u = -2 * viscosity * ratio
            ux = -2 * viscosity * (p[2] / p[0] - ratio ** 2)
            uxx = -2 * viscosity * (p[3] / p[0] - 3 * p[2] * p[1] / p[0] ** 2 + 2 * ratio ** 3)
            ut = -2 * viscosity ** 2 * (p[3] * p[0] - p[1] * p[2]) / p[0] ** 2
            result.append(tuple((float(v) for v in (u, ux, uxx, ut))))
    return np.asarray(result)

def design(x, t, M, P, T):
    tau = 2 * t / T - 1
    C = np.polynomial.chebyshev.chebvander(tau, P)
    DC = np.stack([np.polynomial.chebyshev.chebval(tau, np.polynomial.chebyshev.chebder(np.eye(P + 1)[p])) * 2 / T for p in range(P + 1)], axis=1)
    modes = np.arange(1, M + 1)
    phase = x[:, None] * modes
    base = np.concatenate((np.cos(phase), np.sin(phase)), axis=1)
    dx = np.concatenate((-np.sin(phase) * modes, np.cos(phase) * modes), axis=1)
    dxx = -base * np.tile(modes * modes, 2)
    cross = lambda A, B: (A[:, :, None] * B[:, None, :]).reshape(len(x), -1)
    F = cross(base, C)
    FX = cross(dx, C)
    FXX = cross(dxx, C)
    FT = cross(base, DC)
    return (t[:, None] * F, t[:, None] * FX, t[:, None] * FXX, F + t[:, None] * FT)

def initial(kind, x):
    if kind == 'heat':
        return (np.sin(x) + 0.5 * np.sin(2 * x), np.cos(x) + np.cos(2 * x), -np.sin(x) - 2 * np.sin(2 * x))
    return (np.sin(x), np.cos(x), -np.sin(x))

def evaluate_model(q, matrices, kind, x):
    u0, ux0, uxx0 = initial(kind, x)
    A, AX, AXX, AT = matrices
    return np.stack((u0 + A @ q, ux0 + AX @ q, uxx0 + AXX @ q, AT @ q), axis=-1)

def residual(q, A, AX, AXX, AT, u0, ux0, uxx0, nu, kind):
    u = u0 + A @ q
    ux = ux0 + AX @ q
    uxx = uxx0 + AXX @ q
    ut = AT @ q
    return ut - nu * uxx + (u * ux if kind == 'burgers' else 0.0)

def model_solve(kind, nu, M, P, method):
    T = 0.5 if kind == 'heat' else 0.3
    x, t = np.meshgrid(np.arange(64) * 2 * np.pi / 64, np.linspace(0, T, 11), indexing='ij')
    x = x.ravel()
    t = t.ravel()
    matrices = design(x, t, M, P, T)
    u0, ux0, uxx0 = initial(kind, x)
    A, AX, AXX, AT = matrices
    start = time.perf_counter()
    q = np.zeros(A.shape[1])
    history = []
    if kind == 'heat':
        q = np.linalg.lstsq(AT - nu * AXX, nu * uxx0, rcond=1e-12)[0]
        history = [float(np.linalg.norm(AT @ q - nu * (uxx0 + AXX @ q)))]
    else:
        args = tuple((jnp.asarray(v) for v in (*matrices, u0, ux0, uxx0)))
        fn = lambda q: residual(q, *args, nu, kind)
        if method == 'analytic_jacobian':

            def jfun(q):
                a, ax, axx, at, z, zx, zxx = args
                u = z + a @ q
                ux = zx + ax @ q
                return at + ux[:, None] * a + u[:, None] * ax - nu * axx
        else:
            jfun = jax.jacfwd(fn)
        f = jax.jit(fn)
        jac = jax.jit(jfun)
        damping = 1e-08
        for step in range(30):
            r = np.asarray(jax.block_until_ready(f(jnp.asarray(q))))
            J = np.asarray(jax.block_until_ready(jac(jnp.asarray(q))))
            history.append(float(np.linalg.norm(r)))
            if np.linalg.norm(r) / np.sqrt(len(r)) < 1e-10:
                break
            direction = np.linalg.lstsq(np.vstack((J, np.sqrt(damping) * np.eye(len(q)))), np.r_[-r, np.zeros(len(q))], rcond=None)[0]
            accepted = False
            for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = q + factor * direction
                rr = np.asarray(f(jnp.asarray(candidate)))
                if np.linalg.norm(rr) < np.linalg.norm(r):
                    q = candidate
                    accepted = True
                    damping = max(1e-12, damping * 0.3)
                    break
            if not accepted:
                damping *= 10
    return (q, (time.perf_counter() - start) * 1000, history)

def spectral(kind, nu, N, steps):
    T = 0.5 if kind == 'heat' else 0.3
    x = np.arange(N) * 2 * np.pi / N
    k = np.fft.fftfreq(N, d=1 / N)
    u0 = initial(kind, x)[0]
    v = jnp.asarray(np.fft.fft(u0))
    k = jnp.asarray(k)
    mask = jnp.abs(k) < N / 3
    if kind == 'heat':
        snapshots = jnp.exp(-nu * jnp.linspace(0, T, steps + 1)[:, None] * k[None, :] ** 2) * v
        return np.asarray(snapshots)
    dt = T / steps
    L = -nu * k * k
    E = jnp.exp(dt * L)
    E2 = jnp.exp(dt * L / 2)
    roots = jnp.exp(1j * jnp.pi * (jnp.arange(1, 33) - 0.5) / 32)
    LR = dt * L[:, None] + roots[None, :]
    mean = lambda a: jnp.mean(jnp.real(a), axis=1)
    Q = dt * mean((jnp.exp(LR / 2) - 1) / LR)
    f1 = dt * mean((-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR * LR)) / LR ** 3)
    f2 = dt * mean((2 + LR + jnp.exp(LR) * (-2 + LR)) / LR ** 3)
    f3 = dt * mean((-4 - 3 * LR - LR * LR + jnp.exp(LR) * (4 - LR)) / LR ** 3)

    def nonlinear(v):
        u = jnp.real(jnp.fft.ifft(v * mask))
        return -0.5j * k * jnp.fft.fft(u * u) * mask

    def step(v, _):
        Nv = nonlinear(v)
        a = E2 * v + Q * Nv
        Na = nonlinear(a)
        b = E2 * v + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlinear(c)
        next_ = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc
        return (next_, next_)
    _, out = jax.jit(lambda v: jax.lax.scan(step, v, None, length=steps))(v)
    return np.vstack((np.asarray(v)[None, :], np.asarray(out)))

def evaluate_spectral(snaps, kind, nu, x, t, T):
    N = snaps.shape[1]
    k = np.fft.fftfreq(N, d=1 / N)
    steps = len(snaps) - 1
    dt = T / steps
    phase = np.exp(1j * x[:, None] * k[None, :])
    mask = np.abs(k) < N / 3

    def rhs(v):
        if kind == 'heat':
            return -nu * k * k * v
        u = np.real(np.fft.ifft(v * mask))
        return -nu * k * k * v + -0.5j * k * np.fft.fft(u * u) * mask
    if kind == 'heat':
        coef = snaps[0][None, :] * np.exp(-nu * t[:, None] * k * k)
        utcoef = -nu * k * k * coef
    else:
        i = np.minimum((t / dt).astype(int), steps - 1)
        s = (t - i * dt) / dt
        v0 = snaps[i]
        v1 = snaps[i + 1]
        f0 = np.asarray([rhs(v) for v in v0])
        f1 = np.asarray([rhs(v) for v in v1])
        s = s[:, None]
        coef = (2 * s ** 3 - 3 * s * s + 1) * v0 + (s ** 3 - 2 * s * s + s) * dt * f0 + (-2 * s ** 3 + 3 * s * s) * v1 + (s ** 3 - s * s) * dt * f1
        utcoef = (6 * s * s - 6 * s) / dt * v0 + (3 * s * s - 4 * s + 1) * f0 + (-6 * s * s + 6 * s) / dt * v1 + (3 * s * s - 2 * s) * f1
    columns = [np.real(np.sum(phase * coef * (1j * k) ** n, axis=1)) / N for n in (0, 1, 2)]
    columns.append(np.real(np.sum(phase * utcoef, axis=1)) / N)
    return np.stack(columns, axis=-1)

def quality(kind, nu, actual, ref):
    u, ux, uxx, ut = actual.T
    r = ut - nu * uxx + (u * ux if kind == 'burgers' else 0.0)
    rel = float(np.linalg.norm(u - ref[:, 0]) / np.linalg.norm(ref[:, 0]))
    normal = max(float(np.linalg.norm(ref[:, 3])), float(np.linalg.norm(nu * ref[:, 2])), 1.0)
    normalized = float(np.linalg.norm(r) / normal)
    return {'solution_relative_L2': rel, 'normalized_residual': normalized, 'targets': {str(t): bool(rel <= t and normalized <= t) for t in (0.001, 1e-06)}, 'finite': bool(np.all(np.isfinite(actual)))}
from functools import lru_cache

@lru_cache(maxsize=16)
def _model_plan(kind, nu, M, P, method):
    T = 0.5 if kind == 'heat' else 0.3
    x, t = np.meshgrid(np.arange(64) * 2 * np.pi / 64, np.linspace(0, T, 11), indexing='ij')
    x, t = (x.ravel(), t.ravel())
    matrices = design(x, t, M, P, T)
    u0, ux0, uxx0 = initial(kind, x)
    args = tuple((jnp.asarray(v) for v in (*matrices, u0, ux0, uxx0)))
    fn = lambda q: residual(q, *args, nu, kind)
    if method == 'analytic_jacobian':

        def jac(q):
            a, ax, axx, at, z, zx, zxx = args
            return at + (zx + ax @ q)[:, None] * a + (z + a @ q)[:, None] * ax - nu * axx
    else:
        jac = jax.jacfwd(fn)
    f = jax.jit(fn)
    j = jax.jit(jac)
    return (matrices, u0, ux0, uxx0, f, j)

def model_solve(kind, nu, M, P, method):
    start = time.perf_counter()
    matrices, u0, ux0, uxx0, f, jac = _model_plan(kind, nu, M, P, method)
    a, ax, axx, at = matrices
    q = np.zeros(a.shape[1])
    history = []
    if kind == 'heat':
        q = np.linalg.lstsq(at - nu * axx, nu * uxx0, rcond=1e-12)[0]
        history = [float(np.linalg.norm(at @ q - nu * (uxx0 + axx @ q)))]
    else:
        damping = 1e-08
        for step in range(30):
            r = np.asarray(jax.block_until_ready(f(jnp.asarray(q))))
            J = np.asarray(jax.block_until_ready(jac(jnp.asarray(q))))
            history.append(float(np.linalg.norm(r)))
            if np.linalg.norm(r) / np.sqrt(len(r)) < 1e-10:
                break
            direction = np.linalg.lstsq(np.vstack((J, np.sqrt(damping) * np.eye(len(q)))), np.r_[-r, np.zeros(len(q))], rcond=None)[0]
            accepted = False
            for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = q + factor * direction
                rr = np.asarray(f(jnp.asarray(candidate)))
                if np.linalg.norm(rr) < np.linalg.norm(r):
                    q = candidate
                    accepted = True
                    damping = max(1e-12, damping * 0.3)
                    break
            if not accepted:
                damping *= 10
    return (q, (time.perf_counter() - start) * 1000, history)

@lru_cache(maxsize=16)
def _spectral_plan(kind, nu, N, steps):
    T = 0.5 if kind == 'heat' else 0.3
    x = np.arange(N) * 2 * np.pi / N
    k = jnp.asarray(np.fft.fftfreq(N, d=1 / N))
    mask = jnp.abs(k) < N / 3
    if kind == 'heat':
        evolution = jnp.exp(-nu * jnp.linspace(0, T, steps + 1)[:, None] * k[None, :] ** 2)
        return jax.jit(lambda v: evolution * v)
    dt = T / steps
    L = -nu * k * k
    E = jnp.exp(dt * L)
    E2 = jnp.exp(dt * L / 2)
    roots = jnp.exp(1j * jnp.pi * (jnp.arange(1, 33) - 0.5) / 32)
    LR = dt * L[:, None] + roots[None, :]
    mean = lambda a: jnp.mean(jnp.real(a), axis=1)
    Q = dt * mean((jnp.exp(LR / 2) - 1) / LR)
    f1 = dt * mean((-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR * LR)) / LR ** 3)
    f2 = dt * mean((2 + LR + jnp.exp(LR) * (-2 + LR)) / LR ** 3)
    f3 = dt * mean((-4 - 3 * LR - LR * LR + jnp.exp(LR) * (4 - LR)) / LR ** 3)

    def nonlinear(v):
        u = jnp.real(jnp.fft.ifft(v * mask))
        return -0.5j * k * jnp.fft.fft(u * u) * mask

    def step(v, _):
        Nv = nonlinear(v)
        a = E2 * v + Q * Nv
        Na = nonlinear(a)
        b = E2 * v + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlinear(c)
        result = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc
        return (result, result)
    return jax.jit(lambda v: jax.lax.scan(step, v, None, length=steps)[1])

def spectral(kind, nu, N, steps):
    x = np.arange(N) * 2 * np.pi / N
    v = jnp.asarray(np.fft.fft(initial(kind, x)[0]))
    result = np.asarray(jax.block_until_ready(_spectral_plan(kind, nu, N, steps)(v)))
    return result if kind == 'heat' else np.vstack((np.asarray(v)[None, :], result))
