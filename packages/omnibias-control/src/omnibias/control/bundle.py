# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The proof-carrying controller bundle (theory 10-02/10-03, Phase 3).

A single container for the four pieces a deployed policy from this program
should ship with:

1. the policy parameters (opaque to this module -- a flattened parameter
   vector, backend-agnostic);
2. a :class:`omnibias.control.certified.gradient_bias.GradientBiasReport`
   (sound, *conditional* bound on the training-time policy-gradient bias);
3. a :class:`omnibias.control.horizon.CertifiedHorizonResult` (sound bound
   on the truncation window actually used to train it);
4. an *existing* :class:`omnibias.control.problem.RecoverableCertificate`
   (the model-relative CBF-QP feasibility proof, unchanged from before this
   program).

:func:`bundle_status` is a plain conjunction: the bundle is only as strong
as its weakest included certificate, and any missing piece (``None``)
downgrades the overall verdict to ``partial`` rather than silently
counting as passed. Nothing here claims **certified safe control** --
``omnibias.control.__init__`` reserves that exact language; this module
call it what it is, a **model-relative safety certificate** plus two
training-time enclosures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from omnibias.control.certified.gradient_bias import GradientBiasReport
from omnibias.control.horizon import CertifiedHorizonResult
from omnibias.control.problem import RecoverableCertificate

BundleVerdict = Literal["certified", "partial", "uncertified"]

DISCLAIMER = (
    "proof-carrying bundle: conjunction of a conditional gradient-bias enclosure, "
    "a horizon contraction certificate, and a model-relative recoverable-set "
    "certificate; missing pieces downgrade the verdict, never upgrade it; "
    "not a claim of certified safe control"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "certified_safe_control_claimed": False,
        "unconditional_claimed": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
        "skips_chain_rule": False,
    }


@dataclass(frozen=True)
class ControllerBundle:
    """The four-part proof-carrying bundle."""

    theta: object
    gradient_bias: GradientBiasReport | None
    horizon: CertifiedHorizonResult | None
    recoverable: RecoverableCertificate | None
    notes: tuple[str, ...] = ()

    @property
    def verdict(self) -> BundleVerdict:
        return bundle_status(self)


def bundle_status(bundle: ControllerBundle) -> BundleVerdict:
    r"""Conjunctive verdict over whichever certificates are present.

    * ``"certified"`` -- every included certificate is present and its own
      ``certified`` flag is ``True``.
    * ``"partial"`` -- at least one certificate is present and ``True``,
      but at least one slot is either missing or ``False``.
    * ``"uncertified"`` -- no certificate is present, or every present one
      is ``False``.
    """
    slots = [bundle.gradient_bias, bundle.horizon, bundle.recoverable]
    present = [s for s in slots if s is not None]
    if not present:
        return "uncertified"
    flags = [bool(s.certified) for s in present]
    if all(flags) and len(present) == len(slots):
        return "certified"
    if any(flags):
        return "partial"
    return "uncertified"


def build_bundle(
    theta: object,
    *,
    gradient_bias: GradientBiasReport | None = None,
    horizon: CertifiedHorizonResult | None = None,
    recoverable: RecoverableCertificate | None = None,
    notes: tuple[str, ...] = (),
) -> ControllerBundle:
    """Assemble a :class:`ControllerBundle`; any subset of certificates may be omitted."""
    return ControllerBundle(
        theta=theta,
        gradient_bias=gradient_bias,
        horizon=horizon,
        recoverable=recoverable,
        notes=notes,
    )


def summary(bundle: ControllerBundle) -> str:
    """A short, honest, human-readable line -- never claims more than the verdict."""
    verdict = bundle.verdict
    parts = []
    if bundle.gradient_bias is not None:
        parts.append(f"gradient-bias<={bundle.gradient_bias.bound:.4g}")
    if bundle.horizon is not None:
        parts.append(f"horizon={bundle.horizon.horizon}")
    if bundle.recoverable is not None:
        parts.append(f"recoverable={bundle.recoverable.certified}")
    body = ", ".join(parts) if parts else "no certificates attached"
    return f"ControllerBundle[{verdict}]: {body} (model-relative; not certified safe control)"


__all__ = [
    "BundleVerdict",
    "ControllerBundle",
    "DISCLAIMER",
    "build_bundle",
    "bundle_status",
    "honesty_payload",
    "summary",
]
