# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Latent-state law discovery from a *single observed coordinate*.

Real measurements are usually **partially observed**: you see one scalar channel
``x(t)`` of a higher-dimensional dynamical system.  Takens' delay-embedding
theorem says the delay vector
``X(t) = (x(t), x(t-\tau), ..., x(t-(m-1)\tau))`` generically reconstructs a state
diffeomorphic to the true (hidden) attractor.  This module turns that into a
discovery pipeline:

1. :func:`takens_embedding` -- build the delay-coordinate matrix from one series.
2. an **autoencoder** compresses the delay vectors to a ``latent_dim`` latent
   trajectory ``z(t)``: :class:`LinearAutoencoder` (PCA, exact for data on a
   low-dimensional linear subspace -- the right tool for a linear latent ODE) or
   the small nonlinear :class:`MLPAutoencoder`.
3. :func:`discover_latent_ode` estimates ``dz/dt`` and feeds the latent
   trajectory to :class:`~omnibias.symbolic.field_discovery.FieldLawDiscoverer`
   (one scalar field per latent component, the others injected as exact columns)
   to recover the governing law ``dz_i/dt = f_i(z)``.

.. note::
   The reconstruction is only defined **up to a diffeomorphism** of the latent
   coordinates.  For a *linear* latent ODE the change of coordinates is linear, so
   the recovered system matrix is *similar* to the true one and its **eigenvalues
   (frequencies / growth rates) are coordinate-invariant** -- that is the honest,
   checkable claim.  Nothing here imports a backend; it is pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.symbolic.field_discovery import (
    FieldLawDiscoverer,
    FieldLawResult,
    analytic_field_jet,
)

__all__ = [
    "LatentODEResult",
    "LinearAutoencoder",
    "MLPAutoencoder",
    "discover_latent_ode",
    "finite_difference_derivative",
    "takens_embedding",
]


def takens_embedding(series: np.ndarray, *, dim: int, delay: int = 1) -> np.ndarray:
    r"""Delay-coordinate embedding of a scalar time series.

    Row ``i`` is ``(s[t], s[t-delay], ..., s[t-(dim-1)*delay])`` with
    ``t = (dim-1)*delay + i`` (most-recent coordinate first), so the returned
    matrix has shape ``(len(series) - (dim-1)*delay, dim)`` and row ``i`` is
    aligned to absolute time index ``(dim-1)*delay + i``.
    """
    s = np.asarray(series, dtype=float).reshape(-1)
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    span = (dim - 1) * delay
    n_rows = s.shape[0] - span
    if n_rows <= 0:
        raise ValueError("series too short for the requested embedding")
    cols = [s[span - k * delay : span - k * delay + n_rows] for k in range(dim)]
    return np.stack(cols, axis=1)


def finite_difference_derivative(
    values: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Central finite-difference time derivative along axis 0.

    Returns ``(interior_index, derivative)`` where ``derivative`` corresponds to
    rows ``1 .. n-2`` of ``values`` (endpoints dropped) and ``interior_index`` are
    the original row indices, so callers can realign the value array.
    """
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] < 3:
        raise ValueError("need at least 3 samples for a central difference")
    deriv = (arr[2:] - arr[:-2]) / (2.0 * dt)
    idx = np.arange(1, arr.shape[0] - 1)
    return idx, deriv


@dataclass(frozen=True)
class LinearAutoencoder:
    """PCA autoencoder: an affine encoder/decoder onto the top principal subspace.

    Exact (zero reconstruction error) when the data lies on a ``latent_dim``
    linear subspace -- the situation produced by delay-embedding a *linear* latent
    ODE -- which is why it is the default extractor here.
    """

    latent_dim: int
    mean_: np.ndarray
    components_: np.ndarray  # (latent_dim, n_features), orthonormal rows

    @staticmethod
    def fit(data: np.ndarray, latent_dim: int) -> LinearAutoencoder:
        x = np.asarray(data, dtype=float)
        if x.ndim != 2:
            raise ValueError("data must be 2-D (n_samples, n_features)")
        if not 1 <= latent_dim <= x.shape[1]:
            raise ValueError("latent_dim must be in [1, n_features]")
        mean = x.mean(axis=0)
        _, _, vh = np.linalg.svd(x - mean, full_matrices=False)
        comps = vh[:latent_dim]
        return LinearAutoencoder(latent_dim=latent_dim, mean_=mean, components_=comps)

    def encode(self, data: np.ndarray) -> np.ndarray:
        out: np.ndarray = (np.asarray(data, dtype=float) - self.mean_) @ self.components_.T
        return out

    def decode(self, latent: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.asarray(latent, dtype=float) @ self.components_ + self.mean_
        return out

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(data))

    def reconstruction_rmse(self, data: np.ndarray) -> float:
        x = np.asarray(data, dtype=float)
        return float(np.sqrt(np.mean((x - self.reconstruct(x)) ** 2)))


@dataclass
class MLPAutoencoder:
    """Small nonlinear autoencoder: ``tanh`` encoder + linear decoder bottleneck.

    A genuine (shallow) neural autoencoder for nonlinear attractors, trained by
    full-batch gradient descent with manual backprop (numpy only). For linear
    latent dynamics prefer :class:`LinearAutoencoder`; this one is provided for the
    nonlinear / exploratory regime.
    """

    latent_dim: int
    mean_: np.ndarray
    scale_: np.ndarray
    w_enc: np.ndarray
    b_enc: np.ndarray
    w_dec: np.ndarray
    b_dec: np.ndarray
    losses: tuple[float, ...]

    @staticmethod
    def fit(
        data: np.ndarray,
        latent_dim: int,
        *,
        epochs: int = 4000,
        lr: float = 0.05,
        seed: int = 0,
    ) -> MLPAutoencoder:
        x = np.asarray(data, dtype=float)
        if x.ndim != 2:
            raise ValueError("data must be 2-D (n_samples, n_features)")
        n_features = x.shape[1]
        if not 1 <= latent_dim <= n_features:
            raise ValueError("latent_dim must be in [1, n_features]")
        mean = x.mean(axis=0)
        scale = np.where(x.std(axis=0) < 1e-12, 1.0, x.std(axis=0))
        xn = (x - mean) / scale

        rng = np.random.default_rng(seed)
        w_enc = rng.standard_normal((n_features, latent_dim)) * 0.1
        b_enc = np.zeros(latent_dim)
        w_dec = rng.standard_normal((latent_dim, n_features)) * 0.1
        b_dec = np.zeros(n_features)

        n = xn.shape[0]
        losses: list[float] = []
        for _ in range(epochs):
            z = np.tanh(xn @ w_enc + b_enc)
            recon = z @ w_dec + b_dec
            err = recon - xn
            losses.append(float(np.mean(err**2)))
            d_recon = 2.0 * err / n
            d_w_dec = z.T @ d_recon
            d_b_dec = d_recon.sum(axis=0)
            d_z = d_recon @ w_dec.T
            d_pre = d_z * (1.0 - z**2)
            d_w_enc = xn.T @ d_pre
            d_b_enc = d_pre.sum(axis=0)
            w_dec -= lr * d_w_dec
            b_dec -= lr * d_b_dec
            w_enc -= lr * d_w_enc
            b_enc -= lr * d_b_enc

        return MLPAutoencoder(
            latent_dim=latent_dim,
            mean_=mean,
            scale_=scale,
            w_enc=w_enc,
            b_enc=b_enc,
            w_dec=w_dec,
            b_dec=b_dec,
            losses=tuple(losses),
        )

    def encode(self, data: np.ndarray) -> np.ndarray:
        xn = (np.asarray(data, dtype=float) - self.mean_) / self.scale_
        out: np.ndarray = np.tanh(xn @ self.w_enc + self.b_enc)
        return out

    def decode(self, latent: np.ndarray) -> np.ndarray:
        xn = np.asarray(latent, dtype=float) @ self.w_dec + self.b_dec
        out: np.ndarray = xn * self.scale_ + self.mean_
        return out

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(data))

    def reconstruction_rmse(self, data: np.ndarray) -> float:
        x = np.asarray(data, dtype=float)
        return float(np.sqrt(np.mean((x - self.reconstruct(x)) ** 2)))


@dataclass(frozen=True)
class LatentODEResult:
    """Outcome of latent-ODE discovery from one observed coordinate."""

    latent_trajectory: np.ndarray
    reconstruction_rmse: float
    component_formulas: tuple[str, ...]
    component_results: tuple[FieldLawResult, ...]
    linear_system_matrix: np.ndarray
    eigenvalues: np.ndarray
    note: str

    @property
    def latent_dim(self) -> int:
        return int(self.linear_system_matrix.shape[0])


def _fit_autoencoder(
    embedded: np.ndarray, latent_dim: int, kind: str, **kwargs: Any
) -> LinearAutoencoder | MLPAutoencoder:
    if kind == "linear":
        return LinearAutoencoder.fit(embedded, latent_dim)
    if kind == "mlp":
        return MLPAutoencoder.fit(embedded, latent_dim, **kwargs)
    raise ValueError(f"unknown autoencoder kind {kind!r}; use 'linear' or 'mlp'")


def _component_law(
    t_interior: np.ndarray,
    z_interior: np.ndarray,
    dz_interior: np.ndarray,
    i: int,
    *,
    discoverer: FieldLawDiscoverer,
    split_seed: int,
) -> FieldLawResult:
    """Recover ``dz_i/dt = f_i(z)`` via FieldLawDiscoverer (others injected)."""
    m, d = z_interior.shape
    lookup = {round(float(t_interior[k]), 9): z_interior[k] for k in range(m)}

    def extra_columns_fn(jet: Any) -> dict[str, np.ndarray]:
        ts = jet.X[:, 0]
        cols: dict[str, np.ndarray] = {}
        for j in range(d):
            if j == i:
                continue
            cols[f"z{j}"] = np.array([lookup[round(float(tt), 9)][j] for tt in ts])
        return cols

    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(m)
    n_tr = int(0.6 * m)
    n_va = int(0.2 * m)
    splits = (idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :])

    def make_jet(sel: np.ndarray) -> Any:
        x = t_interior[sel][:, None]
        partials = {(0,): z_interior[sel, i], (1,): dz_interior[sel, i]}
        return analytic_field_jet(x, partials, order=1, var_names=("t",))

    train, val, test = (make_jet(s) for s in splits)
    return discoverer.discover(
        train, val, test, lhs_index=(1,), lhs=f"z{i}", extra_columns_fn=extra_columns_fn
    )


def discover_latent_ode(
    series: np.ndarray,
    *,
    dt: float,
    latent_dim: int = 2,
    embedding_dim: int = 4,
    delay: int = 1,
    autoencoder: str = "linear",
    max_degree: int = 2,
    split_seed: int = 0,
    discoverer: FieldLawDiscoverer | None = None,
    **autoencoder_kwargs: Any,
) -> LatentODEResult:
    r"""Recover a latent ODE ``dz/dt = f(z)`` from a single observed coordinate.

    Pipeline: :func:`takens_embedding` -> autoencoder compression to ``latent_dim``
    -> central-difference ``dz/dt`` -> per-component
    :class:`~omnibias.symbolic.field_discovery.FieldLawDiscoverer`.  The
    ``linear_system_matrix`` collects the degree-1 coefficients and its
    ``eigenvalues`` are the coordinate-invariant frequencies / growth rates.
    """
    embedded = takens_embedding(series, dim=embedding_dim, delay=delay)
    ae = _fit_autoencoder(embedded, latent_dim, autoencoder, **autoencoder_kwargs)
    latent = ae.encode(embedded)
    recon_rmse = ae.reconstruction_rmse(embedded)

    t_full = np.arange(latent.shape[0], dtype=float) * dt
    interior, dz = finite_difference_derivative(latent, dt)
    t_interior = t_full[interior]
    z_interior = latent[interior]

    disco = discoverer if discoverer is not None else FieldLawDiscoverer(max_degree=max_degree, time_axis=0)

    results: list[FieldLawResult] = []
    formulas: list[str] = []
    system = np.zeros((latent_dim, latent_dim), dtype=float)
    for i in range(latent_dim):
        res = _component_law(
            t_interior, z_interior, dz, i, discoverer=disco, split_seed=split_seed
        )
        results.append(res)
        formulas.append(res.formula())
        coef = dict(zip(res.equation.term_names, res.equation.coefficients, strict=True))
        for j in range(latent_dim):
            system[i, j] = float(coef.get(f"z{j}", 0.0))

    eigenvalues = np.linalg.eigvals(system)
    note = (
        "Latent coordinates are defined up to a diffeomorphism; for a linear latent "
        "ODE the recovered system matrix is similar to the truth, so its eigenvalues "
        "(frequencies / growth rates) are the coordinate-invariant, checkable claim."
    )
    return LatentODEResult(
        latent_trajectory=latent,
        reconstruction_rmse=recon_rmse,
        component_formulas=tuple(formulas),
        component_results=tuple(results),
        linear_system_matrix=system,
        eigenvalues=eigenvalues,
        note=note,
    )
