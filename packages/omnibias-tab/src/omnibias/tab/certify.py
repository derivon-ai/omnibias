# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Sound certificates for a trained soft-tree ensemble.

:func:`certify_tab` bundles, for a :class:`~omnibias.tab._core.params.TabParams` and an
input hyper-box, the guarantees a tabular model can *earn* (not just assert):

* **output bounds** -- a rigorous enclosure of every output over the box;
* **Lipschitz** -- an upper bound on the model's sensitivity (any-``n`` continuity);
* **per-feature monotonicity** -- the certified sign of ``dF / dx_f`` over the box, usable
  as a *sound* monotone constraint (GBMs offer this as a soft prior; here it is proved);
* a certified **train-soft / deploy-hard rounding gap** as ``beta -> inf``
  (:func:`certify_tab_gap`) -- how much hardening the soft splits can move the score.

Two engines, both sound (a looser bound only widens the certified gap):

* the **interval** engine (:mod:`omnibias.tab._core.verified`) works for any depth using
  only ``omnibias-core``'s outward-rounded substrate;
* for the **additive** (``depth == 1``) tier -- a genuine ``Linear -> Sigmoid -> Linear``
  network -- the tighter / sealed **verify** engine (``omnibias-verify`` Taylor +
  branch-and-bound) is used when available, and falls back to the interval engine
  otherwise.

Honesty (per the discrete consumers' yes-if framing): the certificates are genuine sound
enclosures, never exact-optimality claims, and the rounding gap is never asserted zero.

Terminology: the ``beta -> inf`` gate hardening is the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from omnibias.tab._core.params import TabParams
from omnibias.tab._core.verified import (
    interval_jacobian,
    interval_output_bounds,
    lipschitz_from_jacobian,
    normalize_box,
    rounding_gap,
)


@dataclass(frozen=True)
class RoundingGapCertificate:
    r"""A certified bound on ``|F_soft - F_hard|`` as ``beta -> inf`` over a sample set."""

    beta: float
    max_gap: float
    mean_gap: float
    measured_max: float
    per_output_max: tuple[float, ...]

    @property
    def is_sound(self) -> bool:
        """The certified bound must dominate the actually-measured gap (self-check)."""
        return self.max_gap >= self.measured_max - 1e-9


@dataclass(frozen=True)
class TabCertificate:
    r"""The certificate bundle returned by :func:`certify_tab`."""

    beta: float
    method: str
    output_bounds: tuple[tuple[float, float], ...]
    lipschitz: float
    output_index: int
    monotonicity: dict[int, str] = field(default_factory=dict)
    monotone_ok: bool | None = None
    rounding: RoundingGapCertificate | None = None

    @property
    def certified(self) -> bool:
        """All *requested* monotonicity constraints held (``True`` if none were requested)."""
        return True if self.monotone_ok is None else self.monotone_ok

    @property
    def rounding_gap(self) -> float | None:
        return None if self.rounding is None else self.rounding.max_gap


def _try_build_verify_network(params: TabParams, beta: float) -> object | None:
    if not params.config.is_additive:
        return None
    try:
        import torch  # noqa: F401
        from omnibias.tab.torch.model import SoftTreeEnsemble
        from omnibias.verify.torch import network_from_sequential
    except ImportError:
        return None
    model = SoftTreeEnsemble(params.config, params)
    seq = model.to_additive_sequential(beta)
    net: object = network_from_sequential(seq)
    return net


def _want_verdict(sign: int) -> str:
    return "increasing" if sign > 0 else "decreasing"


def certify_tab(
    params: TabParams,
    feature_box: object,
    *,
    monotone_features: dict[int, int] | None = None,
    X: np.ndarray | None = None,
    beta: float | None = None,
    output_index: int = 0,
    use_verify: bool = True,
    norm: str = "l2",
) -> TabCertificate:
    r"""Certify a soft-tree model over ``feature_box`` (a ``(2, d)`` lo/hi array or box).

    Parameters
    ----------
    params:
        The trained ensemble (``model.to_params()``).
    feature_box:
        The input hyper-box to certify over (e.g. ``np.stack([X.min(0), X.max(0)])``).
    monotone_features:
        Optional ``{feature_index: +1 | -1}`` constraints (``+1`` = require increasing);
        ``monotone_ok`` reports whether all held. If ``None``, every feature's verdict is
        still recorded but no pass/fail is asserted.
    X:
        Optional samples for the certified soft->hard rounding gap (:func:`certify_tab_gap`).
    beta:
        Gate sharpness to certify at (defaults to the config's ``beta_final`` -- the
        deploy-time value).
    output_index:
        Which output the monotonicity verdicts refer to (default ``0``).
    use_verify:
        Use the tighter / sealed ``omnibias-verify`` engine for the additive tier when
        available (falls back to the interval engine otherwise).
    norm:
        Lipschitz norm (``"l2"`` default or ``"inf"``).
    """
    b = float(params.config.beta_final if beta is None else beta)
    d = params.n_features
    box_iv = normalize_box(feature_box, d)
    feats = list(monotone_features.keys()) if monotone_features is not None else list(range(d))

    net = _try_build_verify_network(params, b) if use_verify else None
    monotonicity: dict[int, str] = {}

    if net is not None:
        from omnibias.verify import lipschitz_bound, reachable_box
        from omnibias.verify import monotonicity as vmono

        method = "verify"
        obounds = reachable_box(net, box_iv)
        output_bounds = tuple((float(iv.lo), float(iv.hi)) for iv in obounds)
        lipschitz = float(lipschitz_bound(net, box_iv, norm=norm))
        for f in feats:
            monotonicity[f] = vmono(net, box_iv, output_index, f).verdict
    else:
        method = "interval"
        obounds = interval_output_bounds(params, feature_box, b)
        output_bounds = tuple((float(iv.lo), float(iv.hi)) for iv in obounds)
        jac = interval_jacobian(params, feature_box, b)
        lipschitz = float(max(lipschitz_from_jacobian(jac, norm=norm)))
        for f in feats:
            dv = jac[output_index][f]
            monotonicity[f] = (
                "increasing" if dv.lo >= 0.0 else "decreasing" if dv.hi <= 0.0 else "unknown"
            )

    monotone_ok: bool | None = None
    if monotone_features is not None:
        monotone_ok = all(
            monotonicity.get(f) == _want_verdict(sign) for f, sign in monotone_features.items()
        )

    rounding = certify_tab_gap(params, X, beta=b) if X is not None else None

    return TabCertificate(
        beta=b,
        method=method,
        output_bounds=output_bounds,
        lipschitz=lipschitz,
        output_index=output_index,
        monotonicity=monotonicity,
        monotone_ok=monotone_ok,
        rounding=rounding,
    )


@dataclass(frozen=True)
class ComposedCertificate:
    r"""Local-box certificate of ``head(encoder(x))``.

    ``method`` is ``ibp+tab`` / ``verify_fused`` / ``ibp_fused`` / ``ibp+arrangement``
    when the encoder ingests as ``Linear`` + ``ReLU`` / ``Tanh`` / ``Sigmoid`` /
    ``GELU`` (a ``Sequential`` or a flattenable ``ModuleList`` wrapper);
    ``tab+tab`` / ``ibp+tab+tab`` / ``arrangement+tab`` when the encoder is (or
    ends with) a SoftTree / Arrangement whose output is enclosed by interval
    bounds. Otherwise ``method="sampled_latent"``: the head is certified on the
    axis-aligned hull of ``encoder`` evaluated on a grid **and** a random sample
    of ``feature_box``. That hull is **not** a sound enclosure of ``E(box)``.
    """

    beta: float
    method: str
    output_bounds: tuple[tuple[float, float], ...]
    latent_bounds: tuple[tuple[float, float], ...]
    tab: TabCertificate | None = None


def _box_array(ivs: tuple[object, ...] | list[object]) -> np.ndarray:
    lo = np.array([float(iv.lo) for iv in ivs], dtype=np.float64)
    hi = np.array([float(iv.hi) for iv in ivs], dtype=np.float64)
    return np.stack([lo, hi])


def _arrangement_output_intervals(
    W: np.ndarray,
    t: np.ndarray,
    cell_logits: np.ndarray,
    z_box: np.ndarray,
    beta: float,
) -> tuple[tuple[float, float], ...]:
    from omnibias.core.verified import Interval
    from omnibias.partition._core.verified import interval_weight_bounds
    from omnibias.tab.arrangement import arrangement_params

    cell = np.asarray(cell_logits, dtype=np.float64)
    if cell.ndim == 1:
        cell = cell.reshape(-1, 1)
    part = arrangement_params(W, t, beta_final=float(beta))
    w_iv = interval_weight_bounds(part, z_box, float(beta))
    bounds: list[tuple[float, float]] = []
    for k in range(int(cell.shape[-1])):
        acc = Interval.point(0.0)
        for ell, weight in enumerate(w_iv):
            acc = acc + weight * float(cell[ell, k])
        bounds.append((float(acc.lo), float(acc.hi)))
    return tuple(bounds)


_INGEST_NAMES = frozenset({"Linear", "ReLU", "Tanh", "Sigmoid", "GELU"})


def _flatten_ingest_layers(module: object) -> list[object] | None:
    """Linear / activation list for IBP, or ``None`` if a layer is unsupported."""
    from torch import nn

    def from_iterable(mods: list[object]) -> list[object] | None:
        out: list[object] = []
        for sub in mods:
            name = type(sub).__name__
            if name in _INGEST_NAMES:
                out.append(sub)
            elif isinstance(sub, (nn.Sequential, nn.ModuleList)):
                inner = from_iterable(list(sub))
                if inner is None:
                    return None
                out.extend(inner)
            else:
                return None
        return out if out else None

    if isinstance(module, nn.Sequential):
        return from_iterable(list(module))
    if isinstance(module, nn.ModuleList):
        return from_iterable(list(module))
    if isinstance(module, nn.Module):
        kids = list(module.children())
        if len(kids) == 1 and isinstance(kids[0], (nn.Sequential, nn.ModuleList)):
            return from_iterable(list(kids[0]))
    return None


def _first_linear_in_features(layers: list[object]) -> int | None:
    for sub in layers:
        if type(sub).__name__ == "Linear" and hasattr(sub, "in_features"):
            return int(sub.in_features)
    return None


def _sampled_feature_points(
    box: np.ndarray, *, per_axis: int = 4, n_rand: int = 80, seed: int = 0
) -> np.ndarray:
    import itertools

    d = int(box.shape[1])
    lo = np.asarray(box[0], dtype=np.float64)
    hi = np.asarray(box[1], dtype=np.float64)
    axes = [np.linspace(lo[f], hi[f], per_axis) for f in range(d)]
    grid = np.array(list(itertools.product(*axes)))
    rng = np.random.default_rng(seed)
    rand = rng.uniform(lo, hi, size=(n_rand, d))
    parts = [grid, rand]
    if d <= 10:
        corners = np.array(list(itertools.product(*zip(lo.tolist(), hi.tolist(), strict=True))))
        parts.append(corners)
    return np.vstack(parts)


def _eval_module(module: object, X: np.ndarray, *, dtype: object, device: object) -> np.ndarray:
    import torch

    xt = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=dtype, device=device)
    with torch.no_grad():
        z = module(xt)  # type: ignore[operator]
    arr = z.detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.asarray(arr, dtype=np.float64)


def _unwrap_head(head: object) -> object:
    try:
        from omnibias.tab.torch.plugin import TabHead
    except ImportError:
        return head
    if isinstance(head, TabHead):
        return head.module
    return head


def _lohi_to_arr(bounds: tuple[tuple[float, float], ...]) -> np.ndarray:
    lo = np.array([a for a, _b in bounds], dtype=np.float64)
    hi = np.array([b for _a, b in bounds], dtype=np.float64)
    return np.stack([lo, hi])


def _tab_encoder_kind(module: object) -> tuple[str | None, object | None]:
    from omnibias.tab.torch.arrangement import ArrangementClassifier
    from omnibias.tab.torch.model import SoftTreeEnsemble

    inner = _unwrap_head(module)
    if isinstance(inner, SoftTreeEnsemble):
        return "tab", inner
    if isinstance(inner, ArrangementClassifier):
        return "arrangement", inner
    return None, None


def _split_tab_encoder(
    encoder: object,
) -> tuple[list[object] | None, object | None, str | None]:
    """Prefix ingest layers + a trailing tab module, or ``(None, None, None)``."""
    from torch import nn

    kind, mod = _tab_encoder_kind(encoder)
    if kind is not None:
        return None, mod, kind
    if not isinstance(encoder, nn.Sequential):
        return None, None, None
    kids = list(encoder)
    if not kids:
        return None, None, None
    kind, mod = _tab_encoder_kind(kids[-1])
    if kind is None:
        return None, None, None
    prefix = kids[:-1]
    if not prefix:
        return None, mod, kind
    layers = _flatten_ingest_layers(nn.Sequential(*prefix))
    if layers is None:
        return None, None, None
    return layers, mod, kind


def _tab_encoder_box(
    module: object, kind: str, box: object, *, beta: float
) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    if kind == "tab":
        ivs = interval_output_bounds(module.to_params(), box, float(beta))
        bounds = tuple((float(iv.lo), float(iv.hi)) for iv in ivs)
        return _box_array(ivs), bounds
    state = module.numpy_state()
    raw = np.asarray(box, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 2:
        d = int(module.n_features)
        raw = _box_array(normalize_box(box, d))
    bounds = _arrangement_output_intervals(
        state["W"], state["t"], state["cell_logits"], raw, float(beta)
    )
    return _lohi_to_arr(bounds), bounds


def certify_composed(
    encoder: object,
    head: object,
    feature_box: object,
    *,
    beta: float | None = None,
    X: np.ndarray | None = None,
    use_verify: bool = True,
) -> ComposedCertificate:
    r"""Certify ``head(encoder(x))`` over ``feature_box``.

    Tries IBP ingest of ``Linear`` + supported activations (a ``Sequential`` or a
    flattenable ``ModuleList`` wrapper). A trailing SoftTree / Arrangement encoder
    is enclosed with interval output bounds (optionally after IBP of a Linear
    prefix). On success, certifies the tab head on the latent box (depth-1
    SoftTree may fuse a pure Linear ingest). Otherwise evaluates the encoder on a
    grid and a random sample, takes the axis-aligned hull of ``z``, and certifies
    the head on that hull with ``method="sampled_latent"`` — **not** a sound
    enclosure of ``E(box)``.
    """
    import torch
    from omnibias.tab._core.verified import normalize_box
    from omnibias.tab.torch.arrangement import ArrangementClassifier
    from omnibias.tab.torch.model import SoftTreeEnsemble
    from omnibias.verify import interval_propagate
    from omnibias.verify.torch import network_from_sequential
    from torch import nn

    if not isinstance(encoder, nn.Module):
        raise TypeError(
            f"encoder must be a torch.nn.Module, got {type(encoder).__name__}"
        )
    head = _unwrap_head(head)

    prefix_layers, tab_enc, tab_kind = _split_tab_encoder(encoder)
    ingest_layers = _flatten_ingest_layers(encoder)
    d_lin = _first_linear_in_features(ingest_layers) if ingest_layers is not None else None
    x_iv: list[object] | None = None
    layers: list[object] | None = ingest_layers
    enc_tag = "sampled_latent"

    if tab_enc is not None:
        mid_box: object = feature_box
        tag_parts: list[str] = []
        if prefix_layers is not None:
            d_pre = _first_linear_in_features(prefix_layers)
            if d_pre is None:
                prefix_layers = None
            else:
                net_p = network_from_sequential(prefix_layers)
                pre_iv = normalize_box(feature_box, d_pre)
                mid = interval_propagate(net_p, pre_iv).output
                mid_box = _box_array(mid)
                tag_parts.append("ibp")
        enc_beta = (
            float(tab_enc.config.beta_final)
            if tab_kind == "tab"
            else float(tab_enc.beta)
        )
        z_arr, latent_bounds = _tab_encoder_box(
            tab_enc, str(tab_kind), mid_box, beta=enc_beta
        )
        tag_parts.append(str(tab_kind))
        enc_tag = "+".join(tag_parts)
        layers = None
        x_iv = None
    elif ingest_layers is not None and d_lin is not None:
        net = network_from_sequential(ingest_layers)
        x_iv = normalize_box(feature_box, d_lin)
        latent = interval_propagate(net, x_iv).output
        z_arr = _box_array(latent)
        latent_bounds = tuple((float(iv.lo), float(iv.hi)) for iv in latent)
        enc_tag = "ibp"
        layers = ingest_layers
    else:
        raw = np.asarray(feature_box, dtype=np.float64)
        d_in = int(raw.shape[-1])
        box_np = _box_array(normalize_box(feature_box, d_in))
        pts = _sampled_feature_points(box_np)
        try:
            ref = next(encoder.parameters())
            dtype, device = ref.dtype, ref.device
        except StopIteration:
            dtype, device = torch.float64, torch.device("cpu")
        z = _eval_module(encoder, pts, dtype=dtype, device=device)
        z_arr = np.stack([z.min(axis=0), z.max(axis=0)])
        latent_bounds = tuple(
            (float(z_arr[0, i]), float(z_arr[1, i])) for i in range(int(z_arr.shape[1]))
        )

    if isinstance(head, SoftTreeEnsemble):
        b = float(head.config.beta_final if beta is None else beta)
        fused_bounds: tuple[tuple[float, float], ...] | None = None
        method = "sampled_latent" if enc_tag == "sampled_latent" else f"{enc_tag}+tab"
        can_fuse = (
            enc_tag == "ibp"
            and head.config.depth == 1
            and layers is not None
            and x_iv is not None
        )
        if can_fuse:
            try:
                seq_head = head.to_additive_sequential(b)
                fused = nn.Sequential(*layers, *list(seq_head.children()))
                net_f = network_from_sequential(fused)
                if use_verify:
                    from omnibias.verify import reachable_box

                    ob = reachable_box(net_f, x_iv)
                    fused_bounds = tuple((float(iv.lo), float(iv.hi)) for iv in ob)
                    method = "verify_fused"
                else:
                    ob = interval_propagate(net_f, x_iv).output
                    fused_bounds = tuple((float(iv.lo), float(iv.hi)) for iv in ob)
                    method = "ibp_fused"
            except TypeError:
                fused_bounds = None
        tab_cert = certify_tab(
            head.to_params(), z_arr, X=X, beta=b, use_verify=use_verify
        )
        output_bounds = fused_bounds if fused_bounds is not None else tab_cert.output_bounds
        return ComposedCertificate(
            beta=b,
            method=method,
            output_bounds=output_bounds,
            latent_bounds=latent_bounds,
            tab=tab_cert,
        )

    if isinstance(head, ArrangementClassifier):
        b = float(head.beta if beta is None else beta)
        state = head.numpy_state()
        output_bounds = _arrangement_output_intervals(
            state["W"], state["t"], state["cell_logits"], z_arr, b
        )
        if X is not None:
            from omnibias.tab.arrangement import certify_arrangement_gap

            z_np = encoder(
                torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=torch.float64)
            )
            _ = certify_arrangement_gap(
                state["W"], state["t"], z_np.detach().cpu().numpy(), beta=b
            )
        method = (
            "sampled_latent" if enc_tag == "sampled_latent" else f"{enc_tag}+arrangement"
        )
        return ComposedCertificate(
            beta=b,
            method=method,
            output_bounds=output_bounds,
            latent_bounds=latent_bounds,
            tab=None,
        )

    raise TypeError(
        f"head must be SoftTreeEnsemble or ArrangementClassifier, got {type(head).__name__}"
    )


def certify_tab_gap(
    params: TabParams,
    X: np.ndarray,
    *,
    beta: float | None = None,
) -> RoundingGapCertificate:
    r"""Certified train-soft / deploy-hard rounding gap on the samples ``X``.

    Returns a sound per-sample, per-output bound on ``|F_soft - F_hard|`` (the score change
    from hardening the soft splits at ``beta -> inf``), aggregated to its max / mean over
    ``X``, alongside the actually-measured gap for the ``is_sound`` self-check.
    """
    b = float(params.config.beta_final if beta is None else beta)
    bound, measured = rounding_gap(params, X, b)
    return RoundingGapCertificate(
        beta=b,
        max_gap=float(bound.max()),
        mean_gap=float(bound.mean()),
        measured_max=float(measured.max()),
        per_output_max=tuple(float(v) for v in bound.max(axis=0)),
    )


__all__ = [
    "ComposedCertificate",
    "RoundingGapCertificate",
    "TabCertificate",
    "certify_composed",
    "certify_tab",
    "certify_tab_gap",
]
