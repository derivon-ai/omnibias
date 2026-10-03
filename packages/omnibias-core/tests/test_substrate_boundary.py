# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The numerical substrate works without application or tensor-framework imports."""

from __future__ import annotations

import subprocess
import sys


def test_core_coefficients_and_certificates_are_application_independent() -> None:
    script = '''
import importlib.abc
import sys

class SubstrateOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        framework = fullname.split(".")[0] in {"torch", "jax", "keras", "tensorflow"}
        application = fullname.startswith("omnibias.") and not (
            fullname == "omnibias.core" or fullname.startswith("omnibias.core.")
        )
        if framework or application:
            raise AssertionError("unexpected dependency: " + fullname)
        return None

sys.meta_path.insert(0, SubstrateOnly())
from omnibias.core import sigmoid_polynomial_coeffs
from omnibias.core.proof import Conjecture, ProofMachine
from omnibias.core.proof.certificate import interval_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation
from omnibias.core.verified.interval import Interval

assert sigmoid_polynomial_coeffs(2) == (0, 1, -3, 2)
certificate = interval_certificate("positive margin", Interval(1.0, 2.0))
assert verify_certificate_digest(certificate)
assert "enclosed_quantity_pos" in generate_obligation(certificate)
assert ProofMachine().evaluate(Conjecture("unknown", "unregistered")).status == "BLOCKED"
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
