# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Static sibling's private conditional sign facts and exact sqrt arithmetic.

No numerical dependency is imported here. This is an unintegrated source
    prototype, not a certificate issuer or a claim that an Adam step is admissible.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import struct
import weakref


CONTRACT = "private-nonnegative-arithmetic/owned-carry-v2"
U = Fraction(1, 2**53)
ETA = Fraction(1, 2**1074)
ASSUMPTIONS = (
    "finite binary64 primitive results; round-to-nearest and gradual underflow",
    "no compiler transformation violating finite multiplication/addition sign",
    "native sqrt on a finite nonnegative input is nonnegative and has absolute "
    "rounding error at most U*sqrt(input)+ETA",
    "integer_pow uses finite parity-preserving multiplication semantics",
    "input/constant Bound ideal intervals and absolute errors are valid premises",
)


def _fraction(value):
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, int) and not isinstance(value, bool):
        result = Fraction(value)
    elif isinstance(value, float) and math.isfinite(value):
        result = Fraction.from_float(value)
    else:
        raise ArithmeticError("finite exact scalar required")
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 8192:
        raise ArithmeticError("exact scalar bit budget exhausted")
    return result


def sqrt_rational_interval(value, *, precision=80):
    """Dyadic lower/upper bounds, proved by integer squaring only."""
    value = _fraction(value)
    if value < 0:
        raise ArithmeticError("sqrt of a negative rational")
    if (
        not isinstance(precision, int)
        or isinstance(precision, bool)
        or not 32 <= precision <= 512
    ):
        raise ValueError("precision outside prototype bounds")
    if not value:
        return Fraction(0), Fraction(0)
    tiny_bits = max(
        0, (value.denominator.bit_length() - value.numerator.bit_length() + 1) // 2
    )
    bits = precision + tiny_bits
    if bits > 4608:
        raise ArithmeticError("sqrt dyadic budget exhausted")
    scale = 1 << (2 * bits)
    scaled_numerator = value.numerator * scale
    quotient, remainder = divmod(scaled_numerator, value.denominator)
    lower_integer = math.isqrt(quotient)
    exact = not remainder and lower_integer * lower_integer == quotient
    upper_integer = lower_integer if exact else lower_integer + 1
    denominator = 1 << bits
    return Fraction(lower_integer, denominator), Fraction(upper_integer, denominator)


def sqrt_error_bound(lo, hi, error):
    """Conditional theorem for BOTH ideal x>=0 and actual X>=0.

    The caller of this private theorem must already possess both sign facts.
    This function deliberately takes no caller-supplied 'nonnegative' flag and
    cannot establish those premises. The interpreter below is their sole owner.
    Original input lo/hi/error are returned verbatim as exact rationals.
    """
    lo, hi, error = map(_fraction, (lo, hi, error))
    if lo > hi or hi < 0 or error < 0:
        raise ArithmeticError("invalid nonnegative sqrt premise")
    ideal_lower = max(lo, Fraction(0))
    ideal_upper = hi
    root_lo, _ = sqrt_rational_interval(ideal_lower)
    _, root_hi = sqrt_rational_interval(ideal_upper)
    _, holder = sqrt_rational_interval(error)
    propagated = holder
    stable = None
    if error == 0:
        propagated = Fraction(0)
    elif ideal_lower > error:
        denominator = (
            sqrt_rational_interval(ideal_lower)[0]
            + sqrt_rational_interval(ideal_lower - error)[0]
        )
        if denominator > 0:
            stable = error / denominator
            propagated = min(propagated, stable)
    # sqrt(actual) <= sqrt(ideal) + propagated <= root_hi + propagated.
    native_rounding = U * (root_hi + propagated) + ETA
    return {
        "original": (lo, hi, error),
        "ideal_root_lower": root_lo,
        "ideal_root_upper": root_hi,
        "propagated_error": propagated,
        "holder_error": holder,
        "stable_error": stable,
        "native_sqrt_rounding_error": native_rounding,
        "total_error": propagated + native_rounding,
    }


def outward_float(value, *, upper):
    value = _fraction(value)
    try:
        result = float(value)
    except OverflowError as exc:
        raise ArithmeticError("binary64 endpoint overflow") from exc
    if not math.isfinite(result):
        raise ArithmeticError("binary64 endpoint overflow")
    represented = Fraction.from_float(result)
    if (upper and represented < value) or (not upper and represented > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    if not math.isfinite(result):
        raise ArithmeticError("outward endpoint overflow")
    return result


@dataclass(frozen=True)
class _Fact:
    issuer: object
    owner: object
    stamp: tuple
    nonnegative: bool
    anchor: int
    shape: tuple
    witness: int


class _FactOutputs(list):
    def __init__(self, values, facts):
        super().__init__(values)
        self.facts = tuple(facts)


class _RangeLedger:
    """One interpreter invocation; no external fact injection or carry export.

    A lineage anchor means, separately for ideal and actual values, that the
    anchor >=0 implies this value >=0 AND the anchor <=0 implies this value <=0.
    In particular anchor==0 implies value==0 numerically (signed zero allowed).
    Anchors identify an entire tensor's *same coordinate map*, never its bytes.
    """

    def __init__(self, limits):
        self.max_entries, self.max_bytes, self.max_scalar_visits = limits
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in limits):
            raise ValueError("positive fact budgets required")
        self._issuer = object()
        self._next_anchor = 0
        self._next_frame = 0
        self._visits = 0
        self._bytes = 0
        self._records = []
        self._sqrt_uses = 0

    def _visit(self, amount):
        if amount < 0 or self._visits + amount > self.max_scalar_visits:
            raise ArithmeticError("fact scalar-visit budget exhausted")
        self._visits += amount

    @staticmethod
    def _representatives(array):
        # A stride-zero axis is an actual coordinate-preserving storage fact,
        # not a test that unrelated numerical values happen to be equal.
        index = tuple(slice(0, 1) if size > 1 and stride == 0 else slice(None)
                      for size, stride in zip(array.shape, array.strides, strict=True))
        return array[index] if any(size > 1 and stride == 0
                                   for size, stride in zip(array.shape, array.strides, strict=True)) else array

    @staticmethod
    def _joint_representatives(arrays):
        shape = tuple(arrays[0].shape)
        if any(tuple(a.shape) != shape for a in arrays):
            raise ArithmeticError("joint fact arrays must share the same coordinates")
        axes = tuple(size > 1 and all(a.strides[i] == 0 for a in arrays)
                     for i, size in enumerate(shape))
        if not any(axes):
            return arrays
        index = tuple(slice(0, 1) if zero else slice(None) for zero in axes)
        return tuple(a[index] for a in arrays)

    def _stamp(self, bound):
        arrays = (bound.lo, bound.hi, bound.error)
        shape = tuple(bound.lo.shape)
        if any(tuple(a.shape) != shape or str(a.dtype) != "float64" for a in arrays):
            raise ArithmeticError("fact requires matching binary64 Bound arrays")
        representatives = tuple(self._representatives(a) for a in arrays)
        self._visit(sum(int(a.size) for a in representatives))
        digest = hashlib.sha256()
        metadata = []
        for a, representative in zip(arrays, representatives, strict=True):
            metadata.append((id(a), tuple(a.shape), tuple(a.strides), str(a.dtype)))
            for item in representative.flat:
                value = float(item)
                if not math.isfinite(value):
                    raise ArithmeticError("nonfinite fact array")
                digest.update(struct.pack("<d", value))
        return (id(bound), shape, tuple(metadata), digest.hexdigest())

    def _valid(self, fact, bound):
        if (
            not isinstance(fact, _Fact)
            or fact.issuer is not self._issuer
            or fact.owner() is not bound
            or fact.stamp != self._stamp(bound)
        ):
            raise ArithmeticError("stale, replaced or foreign originating Bound")
        return fact

    def _enclosure_nonnegative(self, bound):
        arrays = self._joint_representatives((bound.lo, bound.hi, bound.error))
        self._visit(sum(int(a.size) for a in arrays))
        for lo, hi, error in zip(*(a.flat for a in arrays), strict=True):
            lo, hi, error = map(_fraction, (float(lo), float(hi), float(error)))
            if lo > hi or error < 0:
                raise ArithmeticError("invalid Bound premise")
            if lo < 0 or lo - error < 0:
                return False
        return True

    def _positive(self, bound):
        arrays = self._joint_representatives((bound.lo, bound.error))
        self._visit(sum(int(a.size) for a in arrays))
        return all(
            _fraction(float(lo)) > _fraction(float(error))
            for lo, error in zip(*(a.flat for a in arrays), strict=True)
        )

    def _issue(
        self, bound, reason, parents=(), *, nonnegative=False, anchor=None, location=()
    ):
        if len(self._records) >= self.max_entries:
            raise ArithmeticError("fact witness-entry budget exhausted")
        stamp = self._stamp(bound)
        if anchor is None:
            self._next_anchor += 1
            anchor = self._next_anchor
        row = {
            "id": len(self._records) + 1,
            "reason": reason,
            "parents": list(parents),
            "frame_location": list(location),
            "bound_sha256": stamp[-1],
            "shape": list(stamp[1]),
            "ideal_and_actual_nonnegative": bool(nonnegative),
            "zero_preserving_lineage_anchor": anchor,
        }
        size = len(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()) + 1
        if self._bytes + size > self.max_bytes:
            raise ArithmeticError("fact witness-metadata budget exhausted")
        self._records.append(row)
        self._bytes += size
        return _Fact(
            self._issuer,
            weakref.ref(bound),
            stamp,
            bool(nonnegative),
            anchor,
            stamp[1],
            row["id"],
        )

    def seed(self, bound, *, location=()):
        return self._issue(
            bound,
            "complete_ideal_and_actual_input_enclosure",
            nonnegative=self._enclosure_nonnegative(bound),
            location=location,
        )

    def frame(self, jaxpr, args, argument_facts, local, path):
        self._next_frame += 1
        frame = self._next_frame
        if len(args) != len(jaxpr.invars):
            raise ArithmeticError("fact argument arity mismatch")
        if argument_facts is not None and len(argument_facts) != len(args):
            raise ArithmeticError("fact argument witness arity mismatch")
        facts = {}
        for i, var in enumerate(jaxpr.constvars):
            facts[var] = self.seed(local[var], location=(*path, frame, "constant", i))
        for i, (var, bound) in enumerate(zip(jaxpr.invars, args, strict=True)):
            if argument_facts is None:
                facts[var] = self.seed(bound, location=(*path, frame, "input", i))
            else:
                facts[var] = self._valid(argument_facts[i], bound)
        return facts, (*path, frame)

    def read_facts(self, variables, values, local_facts, core, location):
        return [
            self.seed(value, location=(*location, "literal", i))
            if isinstance(var, core.Literal)
            else self._valid(local_facts[var], value)
            for i, (var, value) in enumerate(zip(variables, values, strict=True))
        ]

    def outputs(self, variables, values, local_facts, core, location):
        return _FactOutputs(
            values, self.read_facts(variables, values, local_facts, core, location)
        )

    def indexed(self, source, fact, result, location):
        fact = self._valid(fact, source)
        # Indexing preserves global NN, but never a foreign coordinate lineage.
        return self._issue(
            result,
            "index_global_nonnegative_only",
            (fact.witness,),
            nonnegative=fact.nonnegative,
            location=location,
        )

    def stacked(self, rows, row_facts, results, location):
        facts = []
        for j, result in enumerate(results):
            parents = [
                self._valid(fs[j], row[j])
                for row, fs in zip(rows, row_facts, strict=True)
            ]
            facts.append(
                self._issue(
                    result,
                    "scan_stack_all_rows",
                    tuple(f.witness for f in parents),
                    nonnegative=all(f.nonnegative for f in parents),
                    location=location,
                )
            )
        return facts

    def derive(self, eq, values, facts, out, location):
        facts = [self._valid(f, v) for f, v in zip(facts, values, strict=True)]
        if isinstance(out, _FactOutputs):
            return [self._valid(f, v) for f, v in zip(out.facts, out, strict=True)]
        name, params = eq.primitive.name, eq.params
        all_nn = all(f.nonnegative for f in facts)
        nn = False
        reason = "complete_output_enclosure_only"
        anchor = None
        float64 = bool(eq.outvars) and str(eq.outvars[0].aval.dtype) == "float64"
        if float64 and name in {"add", "add_any", "mul"} and all_nn:
            nn, reason = True, "finite_nonnegative_" + name
        if name == "mul" and float64 and len(facts) == 2:
            a, b = facts
            shape = tuple(out[0].lo.shape)
            if a.anchor == b.anchor and a.shape == b.shape == shape:
                nn, reason = True, "same_source_same_coordinate_zero_preserving_product"
            elif a.nonnegative and b.shape == shape:
                anchor, reason = b.anchor, "nonnegative_left_scale_zero_preserving"
            elif b.nonnegative and a.shape == shape:
                anchor, reason = a.anchor, "nonnegative_right_scale_zero_preserving"
        elif (
            float64 and name == "div" and len(facts) == 2 and self._positive(values[1])
        ):
            nn = facts[0].nonnegative
            if facts[0].shape == tuple(out[0].lo.shape):
                anchor = facts[0].anchor
            reason = "actual_and_ideal_positive_denominator"
        elif float64 and name in {"abs", "sqrt"}:
            nn, reason = True, "finite_native_nonnegative_" + name
        elif float64 and name == "integer_pow" and isinstance(params.get("y"), int):
            exponent = params["y"]
            if exponent >= 0 and (not exponent % 2 or facts[0].nonnegative):
                nn, reason = True, "finite_integer_power_parity"
            if (
                exponent > 0
                and exponent % 2
                and facts[0].shape == tuple(out[0].lo.shape)
            ):
                anchor = facts[0].anchor
        elif name in {"copy", "stop_gradient", "optimization_barrier"}:
            return [
                self._issue(
                    v,
                    "identity_same_coordinate",
                    (f.witness,),
                    nonnegative=f.nonnegative,
                    anchor=f.anchor,
                    location=location,
                )
                for v, f in zip(out, facts, strict=True)
            ]
        elif name in {
            "reshape",
            "transpose",
            "broadcast_in_dim",
            "slice",
            "squeeze",
            "dynamic_slice",
            "unstack",
            "split",
            "rev",
        }:
            nn, reason = facts[0].nonnegative, "view_global_nonnegative_only"
        elif name in {"stack", "concatenate", "pad", "dynamic_update_slice"}:
            selected = facts[:2] if name == "dynamic_update_slice" else facts
            nn, reason = (
                all(f.nonnegative for f in selected),
                "assembled_global_nonnegative_only",
            )
        elif name == "select_n":
            nn, reason = (
                all(f.nonnegative for f in facts[1:]),
                "all_selected_candidates_nonnegative",
            )
        elif name in {"min", "max", "reduce_min", "reduce_max"}:
            nn, reason = all_nn, "finite_nonnegative_" + name
        result = []
        parents = tuple(f.witness for f in facts)
        for value in out:
            known = nn or self._enclosure_nonnegative(value)
            result.append(
                self._issue(
                    value,
                    reason,
                    parents,
                    nonnegative=known,
                    anchor=anchor,
                    location=location,
                )
            )
        return result

    def nonnegative_comparison(self, name, values, facts):
        """Only weak-order comparison against an exact REAL+actual zero."""
        facts = [self._valid(f, v) for f, v in zip(facts, values, strict=True)]
        if len(values) != 2 or name not in {"ge", "le", "lt", "gt"}:
            return None
        def zero(value):
            arrays = self._joint_representatives((value.lo, value.hi, value.error))
            self._visit(sum(int(a.size) for a in arrays))
            return all(float(a) == float(b) == float(e) == 0.0
                       for a, b, e in zip(*(a.flat for a in arrays), strict=True))
        if facts[0].nonnegative and zero(values[1]):
            return True if name == "ge" else False if name == "lt" else None
        if facts[1].nonnegative and zero(values[0]):
            return True if name == "le" else False if name == "gt" else None
        return None

    def sqrt(self, value, fact, bound_type, np, fallback, unary, *, output_dtype):
        if str(output_dtype) != "float64":
            raise ArithmeticError("native sqrt prototype requires emitted binary64")
        fact = self._valid(fact, value)
        if not fact.nonnegative:
            # Preserve the exact old refusal; this prototype never guesses sign.
            return unary(value, fallback)
        self._visit(int(value.lo.size) * 3)
        arrays = [np.empty(value.lo.shape, dtype="float64") for _ in range(3)]
        for i, (lo, hi, error) in enumerate(
            zip(value.lo.flat, value.hi.flat, value.error.flat, strict=True)
        ):
            theorem = sqrt_error_bound(float(lo), float(hi), float(error))
            arrays[0].flat[i] = outward_float(theorem["ideal_root_lower"], upper=False)
            arrays[1].flat[i] = outward_float(theorem["ideal_root_upper"], upper=True)
            arrays[2].flat[i] = outward_float(theorem["total_error"], upper=True)
        self._sqrt_uses += 1
        return bound_type(*arrays)

    def report(self, output_facts):
        return {
            "contract": CONTRACT,
            "assumptions": list(ASSUMPTIONS),
            "facts": [
                dict(
                    row,
                    parents=list(row["parents"]),
                    frame_location=list(row["frame_location"]),
                    shape=list(row["shape"]),
                )
                for row in self._records
            ],
            "output_witness_ids": [f.witness for f in output_facts],
            "output_ideal_and_actual_nonnegative": [
                f.nonnegative for f in output_facts
            ],
            "frames": self._next_frame,
            "sqrt_sign_witness_uses": self._sqrt_uses,
            "scalar_visits": self._visits,
            "serialized_witness_bytes": self._bytes,
            "cross_block_witness_export": "only the private registry-backed LIVE export hook; report is not authority",
            "stride_zero_payload_visits": "known storage-coordinate representatives; full original owner/shape/stride metadata retained",
            "numerical_qualification": False,
            "mathematical_oracle_validation": False,
            "performance_or_public_acceptance": False,
        }


def _fact_budget(max_state_bytes):
    """Private caps without changing the original interpret signature/defaults.

    Reserve two serialized-ledger slots from an explicit old array-storage cap.
    None retains the old absence of a total array cap; fact caps still apply.
    Python object overhead, hashing/source temporaries and RSS are not included.
    """
    if max_state_bytes is not None and (
        type(max_state_bytes) is not int or max_state_bytes < 0
    ):
        raise ValueError("nonnegative integer or None state-storage limit required")
    fact_bytes = (
        8 * 1024**2
        if max_state_bytes is None
        else min(8 * 1024**2, max_state_bytes // 4)
    )
    if fact_bytes <= 0:
        raise ArithmeticError(
            "fact metadata storage budget exhausted before allocation"
        )
    reserve = 0 if max_state_bytes is None else 2 * fact_bytes
    arrays = None if max_state_bytes is None else max_state_bytes - reserve
    return (20000, fact_bytes, 2_000_000), arrays, reserve


def _validate_primitive_kind(eq):
    """Refuse dtype models whose original interval transfer is unproved.

    This does not add any primitive to the original interpreter. Its exact
    checked integer binary/control/view/index operations remain available.
    Integer abs/neg/pow/contraction/additive reduction can wrap while a real
    interval looks positive; those need their own checked integer transfer.
    """
    wrapping = {"abs", "neg", "integer_pow", "reduce_sum", "dot_general", "scatter-add"}
    for var in eq.outvars:
        dtype = getattr(var.aval, "dtype", None)
        kind = getattr(dtype, "kind", None)
        if kind is None:
            name = str(dtype)
            kind = (
                "f"
                if name.startswith("float")
                else "i"
                if name.startswith("int")
                else "u"
                if name.startswith("uint")
                else "b"
                if name == "bool"
                else "?"
            )
        if kind == "f" and str(dtype) != "float64":
            raise NotImplementedError(
                "primitive requires an explicit non-binary64 model"
            )
        if kind not in "fiub":
            raise NotImplementedError(
                "primitive requires an explicit real/integer dtype model"
            )
        if kind in "iu" and eq.primitive.name in wrapping:
            raise NotImplementedError(
                "unproved wrapping integer primitive " + eq.primitive.name
            )
