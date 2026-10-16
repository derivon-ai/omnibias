"""Opt-in finite Fourier models and normalized power-series algorithms."""
from dataclasses import dataclass
import math
import jax.numpy as jnp
import numpy as np

def multiply_schoolbook(a, b):
    if a.shape != b.shape:
        raise ValueError('matching coefficient shapes required')
    return jnp.stack([sum((a[..., j] * b[..., n - j] for j in range(n + 1))) for n in range(a.shape[-1])], axis=-1)

def multiply_fft(a, b):
    if a.shape != b.shape:
        raise ValueError('matching coefficient shapes required')
    n = a.shape[-1]
    padded = 1 << (2 * n - 2).bit_length()
    return jnp.fft.irfft(jnp.fft.rfft(a, n=padded) * jnp.fft.rfft(b, n=padded), n=padded)[..., :n]

def reciprocal_schoolbook(a):
    rows = [1 / a[..., 0]]
    for n in range(1, a.shape[-1]):
        rows.append(-sum((a[..., j] * rows[n - j] for j in range(1, n + 1))) / a[..., 0])
    return jnp.stack(rows, axis=-1)

def reciprocal_newton(a):
    result = 1 / a[..., :1]
    while result.shape[-1] < a.shape[-1]:
        size = min(2 * result.shape[-1], a.shape[-1])
        result = jnp.pad(result, [(0, 0)] * (a.ndim - 1) + [(0, size - result.shape[-1])])
        correction = -multiply_fft(a[..., :size], result)
        correction = correction.at[..., 0].add(2)
        result = multiply_fft(result, correction)
    return result

@dataclass(frozen=True)
class Fourier:
    size: int
    length: float = 2 * math.pi

    def __post_init__(self):
        if self.size < 3 or self.length <= 0:
            raise ValueError('positive period and at least three grid points required')

    def multiplier(self, order):
        if not isinstance(order, int) or order < 0:
            raise ValueError('nonnegative integer order required')
        frequencies = np.fft.fftfreq(self.size, d=self.length / self.size) * 2 * np.pi
        multiplier = (1j * frequencies) ** order
        if self.size % 2 == 0 and order % 2:
            multiplier[self.size // 2] = 0
        return jnp.asarray(multiplier)

    def bundle(self, values, orders):
        if values.shape[-1] != self.size:
            raise ValueError('declared Fourier grid size required')
        spectrum = jnp.fft.fft(values, axis=-1)
        return tuple((jnp.fft.ifft(spectrum * self.multiplier(n), axis=-1).real for n in orders))

    def direct(self, values, points, order=0):
        spectrum = jnp.fft.fft(values) / self.size
        k = jnp.asarray(np.fft.fftfreq(self.size, d=self.length / self.size) * 2 * np.pi)
        return jnp.real(jnp.exp(1j * points[:, None] * k) @ (spectrum * (1j * k) ** order))

    def product(self, a, b):
        k = np.fft.fftfreq(self.size) * self.size
        mask = jnp.asarray(np.abs(k) < self.size / 3)
        af, bf = (jnp.fft.fft(a) * mask, jnp.fft.fft(b) * mask)
        product = jnp.fft.ifft(af).real * jnp.fft.ifft(bf).real
        return jnp.fft.ifft(jnp.fft.fft(product) * mask).real
