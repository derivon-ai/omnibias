# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Joint operator-predictor learning for tabular scientific regression.

The model is a first practical step toward the "Omnibias as scientific
AutoML/foundation tool" direction: build a named bank of differentiable
operators, learn sparse gates over that bank, and train the predictor head in
the same optimization loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from omnibias.torch.activations.registry import ActivationSpec, get_activation

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class OperatorMetadata:
    """Description of one candidate operator in the joint bank."""

    name: str
    family: str
    complexity: float
    inputs: tuple[int, ...]
    interpretability: float = 1.0


@dataclass(frozen=True)
class FittedJointOperatorRegressor:
    """A fitted joint operator model plus normalization state."""

    model: JointOperatorRegressor
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    history: dict[str, list[float]]

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        # Place the input on the model's device so predict works after the
        # model has been moved to a GPU (CPU-safe no-op otherwise). The
        # ``torch.empty(0)`` default keeps a parameterless model on CPU.
        device = next(self.model.parameters(), torch.empty(0)).device
        xs = (np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_scale
        with torch.no_grad():
            xt = torch.as_tensor(xs, dtype=torch.float32, device=device)
            pred_z: np.ndarray = self.model(xt).cpu().numpy()
        return pred_z * self.y_scale + self.y_mean

    def selected_operators(self, *, threshold: float = 0.2, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.model.selected_operators(threshold=threshold, top_k=top_k)


class JointOperatorRegressor(nn.Module):
    """Differentiable operator bank with trainable selection gates.

    The model intentionally starts simple: named deterministic operators
    provide interpretability, optional trainable OMBU derivative channels add
    latent operator capacity, and a sparse linear head keeps selected operators
    exportable.
    """

    def __init__(
        self,
        in_features: int,
        *,
        include_raw: bool = True,
        include_unary: bool = True,
        include_pairwise: bool = True,
        include_nested: bool = True,
        max_pairwise: int = 128,
        ombu_channels: int = 0,
        ombu_base: str | ActivationSpec[Tensor] = "tanh",
        gate_temperature: float = 1.0,
        stochastic_gates: bool = True,
        initial_gate_logit: float = -0.25,
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if max_pairwise < 0:
            raise ValueError(f"max_pairwise must be >= 0, got {max_pairwise}")
        if ombu_channels < 0:
            raise ValueError(f"ombu_channels must be >= 0, got {ombu_channels}")
        if gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be > 0, got {gate_temperature}")

        self.in_features = in_features
        self.include_raw = include_raw
        self.include_unary = include_unary
        self.include_pairwise = include_pairwise
        self.include_nested = include_nested
        self.max_pairwise = max_pairwise
        self.ombu_channels = ombu_channels
        self.gate_temperature = gate_temperature
        self.stochastic_gates = stochastic_gates

        self.operator_metadata = _build_operator_metadata(
            in_features,
            include_raw=include_raw,
            include_unary=include_unary,
            include_pairwise=include_pairwise,
            include_nested=include_nested,
            max_pairwise=max_pairwise,
            ombu_channels=ombu_channels,
        )
        if not self.operator_metadata:
            raise ValueError("operator bank is empty")

        self.ombu_projection: nn.Linear | None
        self.ombu_spec: ActivationSpec[Tensor] | None
        if ombu_channels:
            self.ombu_projection = nn.Linear(in_features, ombu_channels)
            self.ombu_spec = ombu_base if isinstance(ombu_base, ActivationSpec) else get_activation(ombu_base)
            if self.ombu_spec.fastpath is None:
                raise TypeError("ombu_base must provide a derivative fastpath for joint derivative operators")
            try:
                self.ombu_spec.fastpath(torch.zeros(1), 2)
            except NotImplementedError as exc:
                raise TypeError("ombu_base must support derivative orders 1 and 2") from exc
        else:
            self.ombu_projection = None
            self.ombu_spec = None

        n_ops = len(self.operator_metadata)
        self.gate_logits = nn.Parameter(torch.full((n_ops,), float(initial_gate_logit)))
        self.readout = nn.Linear(n_ops, 1)
        complexity = torch.tensor([item.complexity for item in self.operator_metadata], dtype=torch.float32)
        self.operator_complexity: Tensor
        self.selection_readout_weight: Tensor
        self.register_buffer("operator_complexity", complexity)
        self.register_buffer("selection_readout_weight", torch.zeros(n_ops), persistent=False)

    @property
    def n_operators(self) -> int:
        return len(self.operator_metadata)

    def gate_probabilities(self) -> Tensor:
        return torch.sigmoid(self.gate_logits / self.gate_temperature)

    def gates(self) -> Tensor:
        if self.training and self.stochastic_gates:
            u = torch.rand_like(self.gate_logits).clamp_(1e-6, 1.0 - 1e-6)
            logistic_noise = torch.log(u) - torch.log1p(-u)
            return torch.sigmoid((self.gate_logits + logistic_noise) / self.gate_temperature)
        return self.gate_probabilities()

    def operator_bank(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"x must have shape (batch, {self.in_features}), got {tuple(x.shape)}")

        cols: list[Tensor] = []
        if self.include_raw:
            cols.extend([x[:, j] for j in range(self.in_features)])

        if self.include_unary:
            for j in range(self.in_features):
                col = x[:, j]
                cols.extend(
                    [
                        col.square(),
                        torch.tanh(col),
                        torch.sin(col),
                        torch.cos(col),
                        torch.log1p(col.abs()),
                        1.0 / (1.0 + col.abs()),
                        torch.exp(col.clamp(-3.0, 3.0)),
                    ]
                )

        if self.include_pairwise and self.max_pairwise:
            for idx, (j, k) in enumerate(combinations(range(self.in_features), 2)):
                if idx >= self.max_pairwise:
                    break
                cols.append(x[:, j] * x[:, k])

        if self.include_nested:
            for j in range(self.in_features):
                col = x[:, j]
                square = col.square()
                cols.extend([torch.tanh(square), torch.sin(square), torch.log1p(square)])

        if self.ombu_projection is not None and self.ombu_spec is not None:
            z = self.ombu_projection(x)
            fp = self.ombu_spec.fastpath
            assert fp is not None
            cols.extend([self.ombu_spec.forward(z[:, h]) for h in range(self.ombu_channels)])
            cols.extend([fp(z[:, h], 1) for h in range(self.ombu_channels)])
            cols.extend([fp(z[:, h], 2) for h in range(self.ombu_channels)])

        return torch.stack(cols, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        bank = self.operator_bank(x)
        gated = bank * self.gates()
        out: Tensor = self.readout(gated).squeeze(-1)
        return out

    def complexity_loss(self) -> Tensor:
        probs = self.gate_probabilities()
        return (probs * self.operator_complexity).mean()

    def active_operator_count(self, threshold: float = 0.2) -> int:
        return int((self.gate_probabilities().detach() >= threshold).sum().item())

    def selected_operators(self, *, threshold: float = 0.2, top_k: int | None = None) -> list[dict[str, Any]]:
        with torch.no_grad():
            probs = self.gate_probabilities().detach().cpu()
            weights = self.readout.weight.detach().squeeze(0).cpu()
            selection_weights = self.selection_readout_weight.detach().cpu()
            if torch.count_nonzero(selection_weights) == 0:
                selection_weights = weights
            contribution = (probs * weights.abs()).numpy()
            selection_score = (
                probs
                * selection_weights.abs()
                * torch.tensor([item.interpretability for item in self.operator_metadata])
                / torch.sqrt(self.operator_complexity.detach().cpu())
            ).numpy()
        rows = []
        for idx, meta in enumerate(self.operator_metadata):
            if probs[idx].item() >= threshold:
                rows.append(
                    _selected_payload(
                        idx,
                        meta,
                        probs[idx].item(),
                        weights[idx].item(),
                        contribution[idx],
                        selection_score[idx],
                        selection_weights[idx].item(),
                    )
                )
        if top_k is not None:
            ranked = np.argsort(-selection_score)[:top_k]
            by_idx = {row["index"]: row for row in rows}
            for idx in ranked:
                if int(idx) not in by_idx:
                    meta = self.operator_metadata[int(idx)]
                    rows.append(
                        _selected_payload(
                            int(idx),
                            meta,
                            probs[int(idx)].item(),
                            weights[int(idx)].item(),
                            contribution[int(idx)],
                            selection_score[int(idx)],
                            selection_weights[int(idx)].item(),
                        )
                    )
            rows = sorted(rows, key=lambda row: row["selection_score"], reverse=True)[:top_k]
        else:
            rows = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
        return rows


def fit_joint_operator_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    seed: int = 0,
    epochs: int = 400,
    lr: float = 3e-2,
    batch_size: int | None = None,
    patience: int = 60,
    sparsity_weight: float = 1e-3,
    weight_decay: float = 1e-5,
    train_sample_weight: np.ndarray | None = None,
    val_sample_weight: np.ndarray | None = None,
    asymmetric_weight: float = 0.0,
    validation_asymmetric_weight: float | None = None,
    asymmetric_under_scale: float = 13.0,
    asymmetric_over_scale: float = 10.0,
    standardize_x: bool = False,
    polish_readout: bool = True,
    polish_ridge: float = 1e-6,
    validation_selection_metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    validation_selection_complexity_weight: float = 0.0,
    model_kwargs: dict[str, Any] | None = None,
) -> FittedJointOperatorRegressor:
    """Fit a :class:`JointOperatorRegressor` with train/validation early stopping."""

    if x_val is None:
        x_val = x_train
    if y_val is None:
        y_val = y_train
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    if x_train.ndim != 2:
        raise ValueError("x_train must be a 2D array")
    if x_val.ndim != 2 or x_val.shape[1] != x_train.shape[1]:
        raise ValueError("x_val must be a 2D array with the same feature count as x_train")
    if asymmetric_weight < 0.0:
        raise ValueError(f"asymmetric_weight must be non-negative, got {asymmetric_weight}")
    if validation_asymmetric_weight is None:
        validation_asymmetric_weight = asymmetric_weight
    if validation_asymmetric_weight < 0.0:
        raise ValueError(f"validation_asymmetric_weight must be non-negative, got {validation_asymmetric_weight}")
    if validation_selection_complexity_weight < 0.0:
        raise ValueError(
            "validation_selection_complexity_weight must be non-negative, "
            f"got {validation_selection_complexity_weight}"
        )
    train_weight = _prepare_sample_weight(train_sample_weight, x_train.shape[0], "train_sample_weight")
    val_weight = _prepare_sample_weight(val_sample_weight, x_val.shape[0], "val_sample_weight")

    if standardize_x:
        x_mean = x_train.mean(axis=0)
        x_scale = x_train.std(axis=0)
        x_scale = np.where(x_scale < 1e-6, 1.0, x_scale).astype(np.float32)
    else:
        x_mean = np.zeros(x_train.shape[1], dtype=np.float32)
        x_scale = np.ones(x_train.shape[1], dtype=np.float32)
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std())
    if y_scale < 1e-6:
        y_scale = 1.0

    xs_train = (x_train - x_mean) / x_scale
    xs_val = (x_val - x_mean) / x_scale
    ys_train = (y_train - y_mean) / y_scale
    ys_val = (y_val - y_mean) / y_scale

    torch.manual_seed(seed)
    kwargs = model_kwargs or {}
    model = JointOperatorRegressor(x_train.shape[1], **kwargs)
    # The numpy I/O contract is float32 end-to-end, but the model's parameters
    # are created at the global default dtype. Pin the model to float32 so a
    # process-wide torch.set_default_dtype(torch.float64) cannot desync the
    # float64 params from the float32 inputs (which otherwise crashes the
    # ridge polish and any nn.Linear inside the operator bank).
    model.to(torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    xtr = torch.as_tensor(xs_train, dtype=torch.float32)
    ytr = torch.as_tensor(ys_train, dtype=torch.float32)
    xv = torch.as_tensor(xs_val, dtype=torch.float32)
    yv = torch.as_tensor(ys_val, dtype=torch.float32)
    wtr = torch.as_tensor(train_weight, dtype=torch.float32)
    wv = torch.as_tensor(val_weight, dtype=torch.float32)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_rmse_z": [],
        "val_asymmetric": [],
        "complexity": [],
        "val_selection_score": [],
    }
    best_score = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    n = xtr.shape[0]
    full_batch = batch_size is None or batch_size >= n

    for _ in range(epochs):
        model.train()
        if full_batch:
            batch_indices = [torch.arange(n)]
        else:
            assert batch_size is not None
            perm = torch.randperm(n)
            batch_indices = [perm[start : start + batch_size] for start in range(0, n, batch_size)]
        epoch_losses = []
        for idx in batch_indices:
            opt.zero_grad()
            pred = model(xtr[idx])
            mse = _weighted_mse(pred, ytr[idx], wtr[idx])
            asymmetric = _weighted_asymmetric_rul_loss(
                pred,
                ytr[idx],
                wtr[idx],
                y_scale=y_scale,
                under_scale=asymmetric_under_scale,
                over_scale=asymmetric_over_scale,
            )
            complexity = model.complexity_loss()
            loss = mse + asymmetric_weight * asymmetric + sparsity_weight * complexity
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(xv)
            val_mse = _weighted_mse(val_pred, yv, wv)
            val_asymmetric = _weighted_asymmetric_rul_loss(
                val_pred,
                yv,
                wv,
                y_scale=y_scale,
                under_scale=asymmetric_under_scale,
                over_scale=asymmetric_over_scale,
            )
            val_rmse = float(torch.sqrt(val_mse).cpu())
            val_asymmetric_value = float(val_asymmetric.cpu())
            complexity_value = float(model.complexity_loss().cpu())
            val_pred_cycles = (val_pred.cpu().numpy() * y_scale + y_mean).astype(np.float64)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_rmse_z"].append(val_rmse)
        history["val_asymmetric"].append(val_asymmetric_value)
        history["complexity"].append(complexity_value)

        if validation_selection_metric is None:
            score = (
                val_rmse
                + validation_asymmetric_weight * val_asymmetric_value
                + 0.05 * sparsity_weight * complexity_value
            )
        else:
            score = float(validation_selection_metric(y_val.astype(np.float64), val_pred_cycles))
            if not np.isfinite(score):
                raise ValueError(f"validation_selection_metric returned a non-finite score: {score}")
            score += validation_selection_complexity_weight * complexity_value
        history["val_selection_score"].append(float(score))
        if score < best_score - 1e-6:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        model.selection_readout_weight.copy_(model.readout.weight.detach().squeeze(0))
    if polish_readout:
        _polish_linear_readout(
            model,
            torch.cat([xtr, xv], dim=0),
            torch.cat([ytr, yv], dim=0),
            sample_weight=torch.cat([wtr, wv], dim=0),
            ridge=polish_ridge,
        )
    model.eval()
    return FittedJointOperatorRegressor(
        model=model,
        x_mean=x_mean.astype(np.float32),
        x_scale=x_scale.astype(np.float32),
        y_mean=y_mean,
        y_scale=y_scale,
        history=history,
    )


def _weighted_mse(pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    return torch.sum(weight * (pred - target).square()) / torch.clamp(weight.sum(), min=1e-12)


def _weighted_asymmetric_rul_loss(
    pred: Tensor,
    target: Tensor,
    weight: Tensor,
    *,
    y_scale: float,
    under_scale: float,
    over_scale: float,
) -> Tensor:
    diff_cycles = torch.clamp((pred - target) * float(y_scale), min=-80.0, max=80.0)
    penalties = torch.where(
        diff_cycles < 0.0,
        torch.expm1(-diff_cycles / float(under_scale)),
        torch.expm1(diff_cycles / float(over_scale)),
    )
    return torch.sum(weight * penalties) / torch.clamp(weight.sum(), min=1e-12)


def _prepare_sample_weight(weight: np.ndarray | None, n_rows: int, name: str) -> np.ndarray:
    if weight is None:
        return np.ones(n_rows, dtype=np.float32)
    out = np.asarray(weight, dtype=np.float32)
    if out.shape != (n_rows,):
        raise ValueError(f"{name} must have shape ({n_rows},), got {out.shape}")
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    mean = float(out.mean())
    if mean <= 1e-12:
        raise ValueError(f"{name} must have positive mean")
    return out / mean


def _polish_linear_readout(
    model: JointOperatorRegressor,
    x: Tensor,
    y: Tensor,
    *,
    sample_weight: Tensor,
    ridge: float,
) -> None:
    model.eval()
    with torch.no_grad():
        design = model.operator_bank(x) * model.gate_probabilities()
        # Solve the normal equations in a single dtype: the operator bank,
        # targets, and weights can arrive at different dtypes when the global
        # default has been changed, and torch.linalg.solve rejects mixed inputs.
        work_dtype = design.dtype
        sqrt_w = torch.sqrt(sample_weight.to(work_dtype))
        ones = torch.ones((design.shape[0], 1), dtype=work_dtype, device=design.device)
        design_aug = torch.cat([design, ones], dim=1)
        weighted_design = design_aug * sqrt_w.unsqueeze(1)
        weighted_y = y.to(work_dtype) * sqrt_w
        reg = ridge * torch.eye(design_aug.shape[1], dtype=work_dtype, device=design.device)
        reg[-1, -1] = 0.0
        coef = torch.linalg.solve(weighted_design.T @ weighted_design + reg, weighted_design.T @ weighted_y)
        model.readout.weight.copy_(coef[:-1].reshape(1, -1))
        model.readout.bias.copy_(coef[-1:])


def _build_operator_metadata(
    in_features: int,
    *,
    include_raw: bool,
    include_unary: bool,
    include_pairwise: bool,
    include_nested: bool,
    max_pairwise: int,
    ombu_channels: int,
) -> list[OperatorMetadata]:
    items: list[OperatorMetadata] = []
    if include_raw:
        items.extend(OperatorMetadata(f"x{j + 1}", "raw", 1.0, (j,)) for j in range(in_features))
    if include_unary:
        for j in range(in_features):
            idx = j + 1
            items.extend(
                [
                    OperatorMetadata(f"x{idx}^2", "square", 1.5, (j,)),
                    OperatorMetadata(f"tanh(x{idx})", "tanh", 1.7, (j,)),
                    OperatorMetadata(f"sin(x{idx})", "sin", 2.0, (j,)),
                    OperatorMetadata(f"cos(x{idx})", "cos", 2.0, (j,)),
                    OperatorMetadata(f"log_abs(x{idx})", "log_abs", 1.8, (j,)),
                    OperatorMetadata(
                        f"inv_one_plus_abs(x{idx})", "inv_one_plus_abs", 1.8, (j,)
                    ),
                    OperatorMetadata(f"exp_clipped(x{idx})", "exp_clipped", 2.2, (j,)),
                ]
            )
    if include_pairwise and max_pairwise:
        for pair_idx, (j, k) in enumerate(combinations(range(in_features), 2)):
            if pair_idx >= max_pairwise:
                break
            items.append(OperatorMetadata(f"x{j + 1}*x{k + 1}", "product", 2.2, (j, k)))
    if include_nested:
        for j in range(in_features):
            idx = j + 1
            items.extend(
                [
                    OperatorMetadata(f"tanh(x{idx}^2)", "nested_tanh_square", 2.8, (j,)),
                    OperatorMetadata(f"sin(x{idx}^2)", "nested_sin_square", 3.0, (j,)),
                    OperatorMetadata(f"log_abs(x{idx}^2)", "nested_log_square", 2.8, (j,)),
                ]
            )
    for h in range(ombu_channels):
        items.extend(
            [
                OperatorMetadata(f"ombu_value_{h}", "ombu_value", 3.0, tuple(range(in_features)), 0.4),
                OperatorMetadata(f"ombu_grad_{h}", "ombu_grad", 3.5, tuple(range(in_features)), 0.4),
                OperatorMetadata(f"ombu_curvature_{h}", "ombu_curvature", 4.0, tuple(range(in_features)), 0.4),
            ]
        )
    return items


def _selected_payload(
    idx: int,
    meta: OperatorMetadata,
    probability: float,
    weight: float,
    contribution: float,
    selection_score: float,
    selection_weight: float,
) -> dict[str, Any]:
    return {
        "index": idx,
        "name": meta.name,
        "family": meta.family,
        "inputs": meta.inputs,
        "complexity": meta.complexity,
        "interpretability": meta.interpretability,
        "gate_probability": float(probability),
        "readout_weight": float(weight),
        "selection_readout_weight": float(selection_weight),
        "contribution_score": float(contribution),
        "selection_score": float(selection_score),
        "importance": float(selection_score),
    }


__all__ = [
    "FittedJointOperatorRegressor",
    "JointOperatorRegressor",
    "OperatorMetadata",
    "fit_joint_operator_regressor",
]
