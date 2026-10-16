"""Research radix-2 FFT arithmetic emitted entirely as real binary64 operations.

Unlike a native FFT primitive, each butterfly and each saved twiddle has an
explicit error obligation visible to bounds.interpret. This module does not
claim that the resulting algorithm always meets the caller's tolerance or
outperforms native FFT/schoolbook convolution. Coefficients are normalized
Taylor coefficients. No production derivative-order expansion is implied.
"""
from functools import lru_cache
import math
import numpy as np
import jax.numpy as jnp
from .bounds import Bound, ConstantBindings

@lru_cache(maxsize=32)
def _twiddles(size, inverse=False):
    import mpmath as mp
    if size < 2 or size & size - 1:
        raise ValueError('radix-2 twiddle size required')
    with mp.workdps(100):
        sign = 1 if inverse else -1
        real = [mp.cos(2 * mp.pi * k / size) for k in range(size // 2)]
        imag = [sign * mp.sin(2 * mp.pi * k / size) for k in range(size // 2)]
        arrays = []
        enclosures = []
        for values in (real, imag):
            saved = np.array([float(v) for v in values], dtype=np.float64)
            lower = np.nextafter(saved, -np.inf)
            upper = np.nextafter(saved, np.inf)
            for k in range(size // 2):
                if k == 0:
                    saved[k] = 1.0 if len(arrays) == 0 else 0.0
                    lower[k] = upper[k] = saved[k]
                elif size % 4 == 0 and k == size // 4:
                    saved[k] = 0.0 if len(arrays) == 0 else float(sign)
                    lower[k] = upper[k] = saved[k]
            from omnibias.core.verified.transcend import _bracket_iv
            context = mp.ctx_iv.MPIntervalContext()
            context.dps = 100
            for k in range(size // 2):
                if k == 0 or (size % 4 == 0 and k == size // 4):
                    continue
                angle = 2 * context.pi * k / size
                enclosure = context.cos(angle) if len(arrays) == 0 else sign * context.sin(angle)
                lower[k], upper[k] = _bracket_iv(mp, enclosure)
            error = np.nextafter(np.maximum(saved - lower, upper - saved), np.inf)
            error = np.where(lower == upper, 0.0, error)
            saved.setflags(write=False)
            arrays.append(saved)
            enclosures.append(Bound(lower, upper, error))
        return (tuple(arrays), tuple(enclosures))

def twiddle_bindings(max_size):
    """Bounds for every forward/inverse stage up to a power-of-two size."""
    if max_size < 1 or max_size & max_size - 1:
        raise ValueError('positive power-of-two transform size required')
    bindings = ConstantBindings()
    size = 2
    while size <= max_size:
        for inverse in (False, True):
            arrays, bounds = _twiddles(size, inverse)
            for saved, bound in zip(arrays, bounds):
                bindings.bind_object(saved, bound).add_legacy(saved, bound)
        size *= 2
    return bindings

def fft_pair(real, imag=None, *, inverse=False):
    """Complex FFT represented as a pair of real arrays; last-axis transform."""
    n = real.shape[-1]
    if n < 1 or n & n - 1:
        raise ValueError('positive power-of-two FFT length required')
    if real.dtype != jnp.float64 or (imag is not None and imag.dtype != jnp.float64):
        raise ValueError('binary64 FFT inputs required')
    if imag is None:
        imag = jnp.zeros_like(real)
    if imag.shape != real.shape:
        raise ValueError('matching real/imaginary shapes required')
    bits = n.bit_length() - 1
    prefix = real.shape[:-1]
    rank = len(prefix)
    perm = tuple(range(rank)) + tuple(range(rank + bits - 1, rank - 1, -1))

    def reverse(v):
        return jnp.transpose(v.reshape(prefix + (2,) * bits), perm).reshape(v.shape)
    rr, ii = (reverse(real), reverse(imag))
    stage = 2
    while stage <= n:
        half = stage // 2
        view = prefix + (n // stage, stage)
        r, i = (rr.reshape(view), ii.reshape(view))
        ur, ui = (r[..., :half], i[..., :half])
        lr, li = (r[..., half:], i[..., half:])
        (c, s), _ = _twiddles(stage, inverse)
        c, s = (jnp.asarray(c), jnp.asarray(s))
        vr = c * lr - s * li
        vi = s * lr + c * li
        rr = jnp.concatenate((ur + vr, ur - vr), axis=-1).reshape(real.shape)
        ii = jnp.concatenate((ui + vi, ui - vi), axis=-1).reshape(real.shape)
        stage *= 2
    if inverse:
        rr, ii = (rr / n, ii / n)
    return (rr, ii)

def transform_size(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError('nonempty coefficient axis required')
    return 1 << (2 * n - 2).bit_length()

def multiply_radix2(a, b):
    """Linear convolution padded before truncation to the input length."""
    if a.shape != b.shape or not a.ndim:
        raise ValueError('matching nonempty coefficient shapes required')
    n = a.shape[-1]
    padded = transform_size(n)
    padding = [(0, 0)] * (a.ndim - 1) + [(0, padded - n)]
    ar, ai = fft_pair(jnp.pad(a, padding))
    br, bi = fft_pair(jnp.pad(b, padding))
    result, _ = fft_pair(ar * br - ai * bi, ar * bi + ai * br, inverse=True)
    return result[..., :n]

def reciprocal_radix2(a):
    """Newton doubling; qualification must prove a[0] bounded away from zero."""
    if not a.ndim or a.shape[-1] < 1:
        raise ValueError('nonempty coefficient axis required')
    result = 1 / a[..., :1]
    while result.shape[-1] < a.shape[-1]:
        size = min(2 * result.shape[-1], a.shape[-1])
        result = jnp.pad(result, [(0, 0)] * (a.ndim - 1) + [(0, size - result.shape[-1])])
        product = multiply_radix2(a[..., :size], result)
        correction = jnp.concatenate((2 - product[..., :1], -product[..., 1:]), axis=-1)
        result = multiply_radix2(result, correction)
    return result

def qualification_bindings(coefficient_count):
    return twiddle_bindings(transform_size(coefficient_count))
