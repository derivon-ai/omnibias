# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private exact-arithmetic construction; no backend integration or runtime claim.

The supplied beta values are frozen binary64 constants, interpreted as exact
dyadic real numbers. This constructs the results of the existing fixed-bit
round-to-nearest multiplication circuit, together with their exact errors from
beta**count. It does not declare these constants, a training graph, or its
gradients numerically qualified. No native power or transcendental is used.
"""

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction

SCHEMA = "omnibias-private-adam-fixed-bit-schedule/v1"
OPTIMIZER_SOURCE_SHA256 = (
    "3524a6436fb93a18bd6e4e2466708dc1a857f4a337482995cb8cbcf9444972e1"
)
MAX_ENTRIES = 256
MAX_COUNTER = (1 << 31) - 1
MAX_IDEAL_BITS = 16_000_000


class Refusal(ValueError):
    """The bounded construction cannot establish its own stated contract."""


def _require(condition, message):
    if not condition:
        raise Refusal(message)


def _integer(value, lower, upper, name):
    _require(type(value) is int and lower <= value <= upper, name + " outside bounds")
    return value


def _binary64():
    _require(
        sys.float_info.radix == 2
        and sys.float_info.mant_dig == 53
        and sys.float_info.max_exp == 1024
        and sys.float_info.min_exp == -1021,
        "binary64 host serialization required",
    )


def _beta(value):
    _binary64()
    _require(
        type(value) is float
        and math.isfinite(value)
        and 0 <= value < 1
        and math.copysign(1.0, value) > 0,
        "frozen finite float beta in [0,1), excluding negative zero, required",
    )
    return Fraction.from_float(value)


def _power_two(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _nearest_even_integer(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    return quotient + int(twice > denominator or (twice == denominator and quotient % 2))


def round_nonnegative_binary64(value):
    """Integer-only RN-even for exact fractions in [0,1], gradual underflow.

    Return an exact dyadic Fraction, avoiding a floating error bound calculation.
    The restricted range covers every intermediate of the selected power circuit.
    """
    _require(type(value) is Fraction and 0 <= value <= 1, "fraction in [0,1] required")
    if not value:
        return Fraction(0)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _power_two(exponent):
        exponent -= 1
    quantum = _power_two(max(exponent - 52, -1074))
    scaled = value / quantum
    return _nearest_even_integer(scaled.numerator, scaled.denominator) * quantum


def fixed_bit_power(beta, exponent, bits):
    """Reproduce R2 _power_nonnegative's operation order under explicit RN rules.

    Inactive select operands and the final unused square are still constructed.
    All intermediates stay in [0,1], so no overflow exception is hidden by select.
    """
    base = _beta(beta)
    _integer(bits, 1, 31, "bits")
    _integer(exponent, 0, (1 << bits) - 1, "exponent")
    remaining, accumulator = exponent, Fraction(1)
    for _ in range(bits):
        product = round_nonnegative_binary64(accumulator * base)
        if remaining & 1:
            accumulator = product
        remaining //= 2
        base = round_nonnegative_binary64(base * base)
    _require(remaining == 0, "counter truncation prohibited")
    return accumulator


@dataclass(frozen=True)
class PowerEntry:
    exponent: int
    beta_hex: str
    bits: int
    stored_hex: str
    ideal: Fraction
    absolute_error: Fraction
    origin: str
    slot: str

    @property
    def stored_exact(self):
        return Fraction.from_float(float.fromhex(self.stored_hex))


@dataclass(frozen=True)
class BiasSchedule:
    start_accepted_steps: int
    maximum_accepted_steps: int
    bits: int
    beta1_hex: str
    beta2_hex: str
    origin: str
    ideal_storage_bit_upper_bound: int
    entries: tuple

    def at_accepted_step(self, counter):
        """Choose the next proposal by accepted count, never by attempt number.

        A real device caller must guard the current integer counter BEFORE any
        lookup/evaluation. JAX's silently clamping gather is not such a guard.
        Failed proposals keep the counter and therefore select the same entry.
        """
        _integer(counter, self.start_accepted_steps, self.maximum_accepted_steps - 1, "counter")
        index = counter - self.start_accepted_steps
        _require(index < len(self.entries), "prepared counter range exhausted")
        result = self.entries[index]
        _require(all(entry.exponent == counter + 1 for entry in result), "schedule index drift")
        return result


def prepare_schedule(
    *,
    beta1,
    beta2,
    start_accepted_steps,
    count,
    maximum_accepted_steps=MAX_COUNTER,
    ideal_storage_bit_limit=1_000_000,
):
    """Return immutable, bounded exact constants for two frozen bias powers.

    This is construction only. Input/parameter domains, first-gradient error,
    constant provenance binding, actual device serialization/lookup, transactional
    training and performance qualification are independent remaining obligations.
    """
    betas = (_beta(beta1), _beta(beta2))
    _integer(maximum_accepted_steps, 1, MAX_COUNTER, "maximum accepted steps")
    _integer(start_accepted_steps, 0, maximum_accepted_steps - 1, "start accepted steps")
    _integer(count, 1, MAX_ENTRIES, "count")
    _integer(ideal_storage_bit_limit, 1, MAX_IDEAL_BITS, "ideal storage bit limit")
    _require(start_accepted_steps + count <= maximum_accepted_steps, "counter headroom exhausted")
    bits = maximum_accepted_steps.bit_length()
    exponent_sum = count * (2 * start_accepted_steps + count + 1) // 2
    storage = exponent_sum * sum(
        max(1, beta.numerator.bit_length()) + beta.denominator.bit_length() for beta in betas
    )
    _require(storage <= ideal_storage_bit_limit, "exact ideal-power storage budget exhausted")
    descriptor = {
        "schema": SCHEMA,
        "optimizer_source_sha256": OPTIMIZER_SOURCE_SHA256,
        "beta1_hex": beta1.hex(),
        "beta2_hex": beta2.hex(),
        "start_accepted_steps": start_accepted_steps,
        "count": count,
        "maximum_accepted_steps": maximum_accepted_steps,
        "bits": bits,
    }
    origin = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    entries = []
    for exponent in range(start_accepted_steps + 1, start_accepted_steps + count + 1):
        row = []
        for channel, (beta, ideal_beta) in enumerate(zip((beta1, beta2), betas, strict=True), 1):
            stored = fixed_bit_power(beta, exponent, bits)
            ideal = ideal_beta**exponent
            stored_float = float(stored)  # exact: stored is already representable binary64
            _require(Fraction.from_float(stored_float) == stored, "inexact serialization")
            row.append(
                PowerEntry(
                    exponent,
                    beta.hex(),
                    bits,
                    stored_float.hex(),
                    ideal,
                    abs(stored - ideal),
                    origin,
                    f"{origin}/beta{channel}/power/{exponent}",
                )
            )
        entries.append(tuple(row))
    return BiasSchedule(
        start_accepted_steps,
        maximum_accepted_steps,
        bits,
        beta1.hex(),
        beta2.hex(),
        origin,
        storage,
        tuple(entries),
    )
