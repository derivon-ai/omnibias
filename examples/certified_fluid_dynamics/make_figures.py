# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deterministic figures for the fractional / hyperdissipative Navier-Stokes track.

Run from the repo root::

    python examples/certified_fluid_dynamics/make_figures.py

Writes PNGs into ``docs/img/``:

- ``fractional_decay_scaling.png`` -- the fractional-shear energy-decay rate
  ``nu m^{2 alpha}`` vs wavenumber (analytic lines + machine-exact measured
  markers); the slope on log-log axes is exactly ``2 alpha``.
- ``fractional_alpha_recovery.png`` -- the learnable-alpha PINN recovering the
  dissipation order jointly with the field (convergence curves + recovered-vs-
  true parity).
- ``fractional_criticality_ladder.png`` -- the 3D regularity regime map:
  supercritical / open (``alpha < 5/4``, incl. the classical ``alpha = 1``
  case) vs critical & subcritical / proven-external (``alpha >= 5/4``, Lions).
- ``tao_divergence_threshold.png`` -- Tao's log-supercritical divergence
  condition ``int dr/(r g^4)``: diverges (regularity proven, external) iff
  ``4 beta <= 1``, borderline ``beta_c = 1/4``.
- ``log_supercritical_beta_recovery.png`` -- the learnable-beta PINN recovering
  ``beta`` straddling the ``0.25`` edge, and with it Tao's regularity side.
- ``fractional_abc3d_convergence.png`` -- the GPU 3D fractional-NS PINN training
  curve + validation rel-L2 (read from the cluster ``metrics.json`` if present).

All code-computed panels use fixed seeds and a single CPU thread, so they are
reproducible. Nothing here claims an NS regularity result; the ``alpha >= 5/4``
and Tao regimes are **external** theorems, only cited.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.certified_fluid_dynamics.fractional_ns_theory import (  # noqa: E402
    CRITICAL_ALPHA_3D,
    classify_regime,
    exact_decaying_shear,
)
from examples.certified_fluid_dynamics.run_fractional_pinn import (  # noqa: E402
    train_recover,
)
from examples.certified_fluid_dynamics.run_log_supercritical_pinn import (  # noqa: E402
    train_recover as train_recover_beta,
)

OUT = Path(os.environ.get("OMNIBIAS_FIG_OUT", str(Path(_REPO_ROOT) / "docs" / "img")))
OUT.mkdir(parents=True, exist_ok=True)
FORMATS = [e.strip() for e in os.environ.get("OMNIBIAS_FIG_FORMATS", "png").split(",") if e.strip()]
PRIMARY = "#3b6fb6"
ACCENT = "#d1495b"
GOOD = "#2a9d8f"
GRID = "#d9dde3"
GOLD = "#e09f3e"
PURPLE = "#6d597a"
PALETTE = [PRIMARY, GOOD, GOLD, ACCENT, PURPLE]


def _save(fig, stem: str) -> None:
    """Save ``fig`` under every configured format (png by default; png,pdf for the paper)."""
    for ext in FORMATS:
        fig.savefig(OUT / f"{stem}.{ext}")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def fig_decay_scaling() -> None:
    """Fractional-shear decay rate ``nu m^{2 alpha}`` vs wavenumber (log-log)."""
    nu = 0.05
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5]
    ms = np.arange(1, 9)
    n, t = 16, 0.5

    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for color, alpha in zip(PALETTE, alphas, strict=False):
        analytic = nu * ms.astype(float) ** (2.0 * alpha)
        # Machine-exact measured rate from the exact solution's energy ratio.
        measured = []
        for m in ms:
            u0, _, _, _ = exact_decaying_shear(n, 0.0, m=int(m), nu=nu, alpha=alpha)
            ut, _, _, _ = exact_decaying_shear(n, t, m=int(m), nu=nu, alpha=alpha)
            measured.append(-np.log(np.sum(ut * ut) / np.sum(u0 * u0)) / (2.0 * t))
        ax.loglog(ms, analytic, "-", color=color, lw=2.2,
                  label=rf"$\alpha={alpha:g}$  (slope $2\alpha={2 * alpha:g}$)")
        ax.loglog(ms, measured, "o", color=color, ms=6, mfc="white", mew=1.6)

    ax.set_xlabel("Fourier wavenumber  $m$")
    ax.set_ylabel(r"energy-decay rate  $\nu\, m^{2\alpha}$")
    ax.set_title("Fractional dissipation: the order sets the decay exponent")
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.98, 0.04, "lines = analytic,  markers = measured (machine-exact)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555")
    _save(fig, "fractional_decay_scaling")
    plt.close(fig)


def fig_alpha_recovery() -> None:
    """Learnable-alpha PINN: recover the fractional order jointly with the field."""
    torch.set_num_threads(1)
    alpha_true = [0.75, 1.0, 1.25, 1.5]
    runs = []
    for a in alpha_true:
        args = Namespace(
            alpha_true=a, alpha_init=0.6, modes=4, hidden=32, depth=3, ny=32,
            n_snapshots=6, collocation=128, nu=0.05, T=1.0, steps=600, lr=3e-3,
            alpha_lr_mult=20.0, phys_weight=1.0, log_every=10, seed=2026,
        )
        runs.append(train_recover(args))

    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    for color, a, run in zip(PALETTE, alpha_true, runs, strict=False):
        steps = [h["step"] for h in run["history"]]
        vals = [h["alpha"] for h in run["history"]]
        ax1.plot(steps, vals, "-", color=color, lw=2.2, label=rf"$\alpha_\mathrm{{true}}={a:g}$")
        ax1.axhline(a, color=color, ls=":", lw=1.2, alpha=0.7)
    ax1.set_xlabel("training step")
    ax1.set_ylabel(r"learnable order  $\alpha$")
    ax1.set_title("Joint field + order recovery converges")
    ax1.legend(loc="center right", fontsize=9)

    rec = [run["alpha_recovered"] for run in runs]
    lo, hi = 0.6, 1.65
    ax2.plot([lo, hi], [lo, hi], "--", color="#999", lw=1.4, label="exact recovery")
    ax2.scatter(alpha_true, rec, s=90, color=PRIMARY, zorder=3, edgecolor="white", linewidth=1.4)
    for a, r in zip(alpha_true, rec, strict=False):
        ax2.annotate(f"|err|={abs(r - a):.1e}", (a, r), textcoords="offset points",
                     xytext=(8, -12), fontsize=8, color="#555")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel(r"true order  $\alpha_\mathrm{true}$")
    ax2.set_ylabel(r"recovered order  $\hat\alpha$")
    ax2.set_title("Recovered vs true dissipation order")
    ax2.legend(loc="upper left", fontsize=9)

    _save(fig, "fractional_alpha_recovery")
    plt.close(fig)


def fig_criticality_ladder() -> None:
    """3D hyperdissipative regularity regime map (honest proven-external vs open)."""
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    lo, hi = 0.5, 1.75

    # Shaded regimes: open (supercritical) vs proven-external (critical/subcritical).
    ax.axvspan(lo, CRITICAL_ALPHA_3D, color=ACCENT, alpha=0.12)
    ax.axvspan(CRITICAL_ALPHA_3D, hi, color=GOOD, alpha=0.14)
    ax.axvline(CRITICAL_ALPHA_3D, color="#333", lw=1.6)

    marks = [
        (0.5, r"$\alpha=\frac{1}{2}$" + "\nsupercritical\n(open)", ACCENT),
        (1.0, r"$\alpha=1$" + "\nclassical NS\n(OPEN)", ACCENT),
        (CRITICAL_ALPHA_3D, r"$\alpha=\frac{5}{4}$" + "\ncritical\n(proven, ext.)", GOOD),
        (1.5, r"$\alpha=\frac{3}{2}$" + "\nsubcritical\n(proven, ext.)", GOOD),
    ]
    for a, label, color in marks:
        row = classify_regime(a)
        assert row["unproven_claim"] is False  # honesty guard baked into the figure
        ax.plot([a], [0.0], "o", ms=13, color=color, mec="white", mew=1.6, zorder=4)
        va = "bottom" if abs(a - 1.0) < 1e-9 else "top"
        dy = 0.16 if va == "bottom" else -0.16
        ax.annotate(label, (a, 0.0), textcoords="offset points", xytext=(0, dy * 100),
                    ha="center", va=va, fontsize=8.5, color="#333")

    ax.text((lo + CRITICAL_ALPHA_3D) / 2, 0.62, "OPEN\n(supercritical)", ha="center",
            fontsize=10, color=ACCENT, fontweight="bold")
    ax.text((CRITICAL_ALPHA_3D + hi) / 2, 0.62, "global regularity\nproven — external\n(Lions 1969)",
            ha="center", fontsize=10, color=GOOD, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([])
    ax.set_xlabel(r"fractional dissipation order  $\alpha$  in  $(-\Delta)^{\alpha}$")
    ax.set_title(r"3D hyperdissipative NS: the criticality ladder ($\alpha_c=5/4$)")
    _save(fig, "fractional_criticality_ladder")
    plt.close(fig)


def _tao_cumulative(beta: float, r: np.ndarray) -> np.ndarray:
    integrand = 1.0 / (r * np.log(np.e + r**2) ** (4.0 * beta))
    dr = np.diff(r)
    mid = 0.5 * (integrand[1:] + integrand[:-1])
    return np.concatenate([[0.0], np.cumsum(mid * dr)])


def fig_tao_threshold() -> None:
    """Tao's divergence condition: int dr/(r g^4) diverges iff 4 beta <= 1 (beta_c=1/4)."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    r = np.logspace(0.0, 12.0, 8000)
    betas = [0.15, 0.20, 0.25, 0.35, 0.50]
    for color, beta in zip(PALETTE, betas, strict=False):
        cum = _tao_cumulative(beta, r)
        applies = 4.0 * beta <= 1.0 + 1e-12
        style = "-" if applies else "--"
        tag = "diverges (reg.)" if applies else "converges (open)"
        ax1.semilogx(r, cum, style, color=color, lw=2.2, label=rf"$\beta={beta:g}$  {tag}")
    ax1.set_xlabel(r"cutoff  $R$")
    ax1.set_ylabel(r"$\int_1^{R} dr\,/\,(r\, g(r)^4)$")
    ax1.set_title("Log-supercritical dissipation:\nintegral diverges iff proven regular")
    ax1.legend(loc="upper left", fontsize=8.5)

    # Regime map vs beta: proven-external (4 beta <= 1) vs open.
    ax2.axvspan(0.05, 0.25, color=GOOD, alpha=0.14)
    ax2.axvspan(0.25, 0.7, color=ACCENT, alpha=0.12)
    ax2.axvline(0.25, color="#333", lw=1.6)
    bb = np.linspace(0.05, 0.7, 200)
    ax2.plot(bb, 4.0 * bb, color=PRIMARY, lw=2.2, label=r"$4\beta$")
    ax2.axhline(1.0, color="#333", ls=":", lw=1.2)
    ax2.text(0.15, 3.2, "proven regular\n(external, Tao 2009)", ha="center", color=GOOD,
             fontsize=10, fontweight="bold")
    ax2.text(0.48, 0.5, "open", ha="center", color=ACCENT, fontsize=11, fontweight="bold")
    ax2.set_xlim(0.05, 0.7)
    ax2.set_ylim(0.0, 3.6)
    ax2.set_xlabel(r"log exponent  $\beta$")
    ax2.set_ylabel(r"$4\beta$  (regularity iff $4\beta\leq 1$)")
    ax2.set_title(r"Tao threshold  $\beta_c=1/4$")
    ax2.legend(loc="lower right", fontsize=9)
    _save(fig, "tao_divergence_threshold")
    plt.close(fig)


def fig_beta_recovery() -> None:
    """Learnable-beta PINN recovering beta straddling Tao's 0.25 edge."""
    torch.set_num_threads(1)
    beta_true = [0.15, 0.20, 0.25, 0.35, 0.50]
    runs = []
    for b in beta_true:
        args = Namespace(
            beta_true=b, beta_init=0.4, modes=6, hidden=32, depth=3, ny=32,
            n_snapshots=8, collocation=128, nu=0.05, T=1.0, steps=1200, lr=3e-3,
            beta_lr_mult=20.0, phys_weight=1.0, log_every=20, seed=2026,
        )
        runs.append(train_recover_beta(args))

    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for color, b, run in zip(PALETTE, beta_true, runs, strict=False):
        steps = [h["step"] for h in run["history"]]
        vals = [h["beta"] for h in run["history"]]
        ax1.plot(steps, vals, "-", color=color, lw=2.0, label=rf"$\beta_\mathrm{{true}}={b:g}$")
        ax1.axhline(b, color=color, ls=":", lw=1.0, alpha=0.7)
    ax1.axhline(0.25, color="#333", lw=1.4, label=r"Tao edge $\beta_c=1/4$")
    ax1.set_xlabel("training step")
    ax1.set_ylabel(r"learnable log-exponent  $\beta$")
    ax1.set_title("Learnable-$\\beta$ recovery at the log-supercritical edge")
    ax1.legend(loc="upper right", fontsize=8)

    lo, hi = 0.08, 0.58
    ax2.axvspan(lo, 0.25, color=GOOD, alpha=0.12)
    ax2.axvspan(0.25, hi, color=ACCENT, alpha=0.10)
    ax2.plot([lo, hi], [lo, hi], "--", color="#999", lw=1.4, label="exact recovery")
    ax2.axvline(0.25, color="#333", lw=1.2)
    ax2.axhline(0.25, color="#333", lw=1.2)
    rec = [run["beta_recovered"] for run in runs]
    ax2.scatter(beta_true, rec, s=90, color=PRIMARY, zorder=3, edgecolor="white", linewidth=1.4)
    for b, r in zip(beta_true, rec, strict=False):
        ax2.annotate(f"|err|={abs(r - b):.1e}", (b, r), textcoords="offset points",
                     xytext=(7, -12), fontsize=7.5, color="#555")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel(r"true  $\beta_\mathrm{true}$")
    ax2.set_ylabel(r"recovered  $\hat\beta$")
    ax2.set_title("Recovered vs true (green = proven, red = open)")
    ax2.legend(loc="upper left", fontsize=8.5)
    _save(fig, "log_supercritical_beta_recovery")
    plt.close(fig)


def _load_frac3d_history() -> dict | None:
    candidates = [
        os.environ.get("OMNIBIAS_FRAC3D_METRICS", ""),
        os.path.join(_REPO_ROOT, "examples", "certified_fluid_dynamics", "data", "fractional_abc3d_metrics.json"),
        os.path.expanduser("artifacts/omnibias_runs/fractional_abc3d/metrics.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    return None


def fig_frac3d_convergence() -> None:
    """GPU 3D fractional-NS PINN: training curve + validation (read from metrics.json)."""
    data = _load_frac3d_history()
    if data is None:
        print("fig_frac3d_convergence: no metrics.json found (run the GPU job first); skipping")
        return
    history = data["history"]
    metrics = data["metrics"]
    steps = [h["step"] for h in history]

    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for key, color, label in [
        ("pde", PRIMARY, "fractional-NS residual"),
        ("ic", GOOD, "initial condition"),
        ("gauge", GOLD, "Coulomb gauge"),
    ]:
        vals = [max(h.get(key, float("nan")), 1e-16) for h in history]
        ax1.semilogy(steps, vals, "-", color=color, lw=2.0, label=label)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss (log scale)")
    a_val = data["config"].get("alpha", 1.0)
    k_val = data["config"].get("shell_wavenumber", 2)
    ax1.set_title(rf"3D fractional NS PINN  ($\alpha={a_val:g}$, shell $K={k_val}$)")
    ax1.legend(loc="upper right", fontsize=9)

    rel = metrics.get("rel_l2_velocity_mean", float("nan"))
    rel_max = metrics.get("rel_l2_velocity_max", float("nan"))
    div = metrics.get("max_abs_divergence", float("nan"))
    labels = ["rel-L2\n(mean)", "rel-L2\n(max)"]
    vals = [rel, rel_max]
    ax2.bar(labels, vals, color=[PRIMARY, GOOD], width=0.55, zorder=3)
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:.2%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.axhline(0.01, color=ACCENT, ls="--", lw=1.4, label="1% target")
    ax2.set_ylabel("relative $L^2$ velocity error vs exact")
    ax2.set_ylim(0, max(0.02, rel_max * 1.4 if rel_max == rel_max else 0.02))
    gpu = data.get("gpu", "GPU")
    ax2.set_title(f"Validation vs exact Beltrami shell\n(max|div u|={div:.1e}, {gpu})")
    ax2.legend(loc="upper right", fontsize=9)
    _save(fig, "fractional_abc3d_convergence")
    plt.close(fig)


def main() -> None:
    fig_decay_scaling()
    fig_alpha_recovery()
    fig_criticality_ladder()
    fig_tao_threshold()
    fig_beta_recovery()
    fig_frac3d_convergence()
    names = sorted(p.name for p in OUT.glob("fractional_*.png"))
    names += sorted(p.name for p in OUT.glob("tao_*.png"))
    names += sorted(p.name for p in OUT.glob("log_supercritical_*.png"))
    print("wrote:", ", ".join(sorted(set(names))))


if __name__ == "__main__":
    main()
