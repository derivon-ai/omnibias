# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""A single, fair training loop shared by every arm.

Architecture, initialisation, optimiser, learning-rate and data order are fixed by
``seed``; the loop's only per-arm degree of freedom is the quantizer it rebinds on
:class:`~examples.binary_vs_ste.models.QuantCtx` each step. ``beta`` is supplied as
a fixed float, an annealed float (:class:`~omnibias.binary.BetaAnnealScheduler`),
or a trained scalar parameter, exactly as the arm dictates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from omnibias.binary import BetaAnnealScheduler
from torch import Tensor, nn
from torch.utils.data import Dataset

from examples.binary_vs_ste.arms import Arm
from examples.binary_vs_ste.data import DatasetSpec, make_loader
from examples.binary_vs_ste.models import QuantCtx, build_model


@dataclass
class RunResult:
    """Outcome of one ``(arm, dataset, seed)`` training run."""

    arm: str
    dataset: str
    seed: int
    epochs: int
    test_acc: float
    best_acc: float
    init_train_loss: float
    final_train_loss: float
    final_beta: float
    history: list[dict[str, float]] = field(default_factory=list)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, *, batch_size: int, device: str) -> float:
    """Top-1 accuracy of ``model`` on ``dataset`` (binary forward is exact)."""
    model.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


def _resolve_beta(
    arm: Arm,
    step: int,
    scheduler: BetaAnnealScheduler | None,
    beta_param: Tensor | None,
) -> float | Tensor:
    if arm.beta_mode == "learnable":
        assert beta_param is not None
        return beta_param.clamp_min(1e-3)
    if arm.beta_mode == "anneal":
        assert scheduler is not None
        return scheduler.value(step)
    return arm.beta


def train_arm(
    arm: Arm,
    train_ds: Dataset,
    test_ds: Dataset,
    spec: DatasetSpec,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    schedule: str = "exp",
    seed: int = 0,
    num_workers: int = 0,
    xnor: bool = False,
    lr_schedule: str = "constant",
    log: bool = False,
) -> RunResult:
    """Train one arm end-to-end and return its :class:`RunResult`.

    ``xnor`` enables the XNOR-Net per-filter weight scale (an equal-opportunity
    accuracy lift); ``lr_schedule`` is ``"constant"`` or ``"cosine"`` (cosine LR
    decay reduces final-epoch noise from sharp surrogates).
    """
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {lr_schedule!r}")
    torch.manual_seed(seed)

    ctx = QuantCtx(fn=arm.make_quantizer(arm.beta), xnor=xnor)
    model = build_model(spec, ctx).to(device)

    beta_param: Tensor | None = None
    param_groups: list[dict[str, object]] = [{"params": list(model.parameters())}]
    if arm.beta_mode == "learnable":
        beta_param = nn.Parameter(torch.tensor(float(arm.beta), device=device))
        param_groups.append({"params": [beta_param], "lr": lr * 0.1})
    optimizer = torch.optim.Adam(param_groups, lr=lr)
    lr_sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if lr_schedule == "cosine"
        else None
    )

    train_loader = make_loader(
        train_ds, batch_size=batch_size, shuffle=True, seed=seed, num_workers=num_workers
    )
    total_steps = max(epochs * len(train_loader), 1)
    scheduler = (
        BetaAnnealScheduler(arm.beta, arm.beta_end, total_steps, schedule)
        if arm.beta_mode == "anneal"
        else None
    )
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_acc = 0.0
    step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        beta_seen = arm.beta
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            beta = _resolve_beta(arm, step, scheduler, beta_param)
            beta_seen = float(beta.detach()) if isinstance(beta, Tensor) else float(beta)
            ctx.fn = arm.make_quantizer(beta)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * int(y.numel())
            seen += int(y.numel())
            step += 1
        train_loss = running / max(seen, 1)
        if lr_sched is not None:
            lr_sched.step()

        if arm.beta_mode == "learnable" and beta_param is not None:
            ctx.fn = arm.make_quantizer(beta_param.clamp_min(1e-3))
        test_acc = evaluate(model, test_ds, batch_size=batch_size, device=device)
        best_acc = max(best_acc, test_acc)
        history.append(
            {"epoch": float(epoch), "train_loss": train_loss, "test_acc": test_acc, "beta": beta_seen}
        )
        if log:
            print(
                f"[{arm.name:>14s} {spec.name:>13s} seed={seed}] "
                f"epoch {epoch + 1:>2d}/{epochs}  loss={train_loss:.4f}  "
                f"acc={test_acc:.4f}  beta={beta_seen:.3g}"
            )

    final_beta = (
        float(beta_param.detach()) if beta_param is not None else history[-1]["beta"]
    )
    return RunResult(
        arm=arm.name,
        dataset=spec.name,
        seed=seed,
        epochs=epochs,
        test_acc=history[-1]["test_acc"],
        best_acc=best_acc,
        init_train_loss=history[0]["train_loss"],
        final_train_loss=history[-1]["train_loss"],
        final_beta=final_beta,
        history=history,
    )


__all__ = ["RunResult", "evaluate", "train_arm"]
