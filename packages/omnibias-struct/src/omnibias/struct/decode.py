# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Public certified-decoding surface -- prove the Viterbi winner survives an ``eps``-ball.

Re-exports the rigorous decoding certifier from :mod:`omnibias.struct._core.decode`:
:func:`certify_decoding` (worst-case winner-vs-runner-up margin over an emission
``L_inf`` ball, enumeration-free), its DAG / DTW generalization
:func:`certify_decoding_dag` (the same margin over an edge ``L_inf`` ball, so it covers the
alignment / DTW lattices), the :class:`DecodingCertificate` / :class:`DAGDecodingCertificate`
they return, and the sealing + optional-Lean bridges. Decoding is backend-agnostic (a sign
fact about the ``beta -> inf`` max-plus semiring), so there is a single numpy implementation
rather than torch / jax twins; pass a backend tensor's ``.detach().cpu().numpy()`` (torch) or
``np.asarray(...)`` (jax) as the emissions.
"""

from __future__ import annotations

from omnibias.struct._core.decode import (
    DAGDecodingCertificate,
    DAGDecodingProofVerdict,
    DecodingCertificate,
    DecodingProofVerdict,
    certify_decoding,
    certify_decoding_dag,
    check_dag_decoding_certificate,
    check_decoding_certificate,
    seal_dag_decoding_certificate,
    seal_decoding_certificate,
)

__all__ = [
    "DAGDecodingCertificate",
    "DAGDecodingProofVerdict",
    "DecodingCertificate",
    "DecodingProofVerdict",
    "certify_decoding",
    "certify_decoding_dag",
    "check_dag_decoding_certificate",
    "check_decoding_certificate",
    "seal_dag_decoding_certificate",
    "seal_decoding_certificate",
]
