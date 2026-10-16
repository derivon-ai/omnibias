"""Exact rational operator-span reduction with bounded symbolic expansion.

Current numerical parameter rank/zeros never enter this reduction. Dependencies
are identities of constant-coefficient differential operators. Huge Laplacians
remain formal compact atoms, allowing duplicate reuse without eager expansion.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import jax.numpy as jnp
import numpy as np
from .bundles import build, coefficient_bounds
from .shared_models import SharedModel, build_shared

@dataclass(frozen=True)
class SpanLimits:
    max_terms: int = 10000
    max_work: int = 1000000
    max_coefficient_bits: int = 4096

    def __post_init__(self):
        if any((type(x) is not int or x < 1 for x in vars(self).values())):
            raise ValueError('positive integer span limits required')

@dataclass(frozen=True)
class OperatorSpan:
    basis: tuple
    reconstruction: tuple
    report: dict

    def matrix(self):
        matrix = np.array([[float(c) for c in row] for row in self.reconstruction], dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ArithmeticError('operator reconstruction overflows binary64')
        return matrix

    def constant_bounds(self):
        """Literal conversion bound to pass to actual emitted-graph qualification."""
        from .bounds import Bound
        matrix = self.matrix()
        lo, hi, error = (matrix.copy(), matrix.copy(), np.zeros_like(matrix))
        for index in np.ndindex(matrix.shape):
            exact = self.reconstruction[index[0]][index[1]]
            represented = Fraction(float(matrix[index]))
            if represented > exact:
                lo[index] = np.nextafter(matrix[index], -np.inf)
            if represented < exact:
                hi[index] = np.nextafter(matrix[index], np.inf)
            difference = abs(represented - exact)
            error[index] = np.nextafter(float(difference), np.inf) if difference else 0.0
        key = hashlib.sha256(str((matrix.shape, matrix.dtype.str)).encode() + matrix.tobytes()).hexdigest()
        return {key: Bound(lo, hi, error)}

    def reconstruct(self, outputs):
        matrix = jnp.asarray(self.matrix())
        result = matrix @ jnp.stack(outputs)
        return tuple((result[i] for i in range(len(self.reconstruction))))

    def transpose(self, cotangents):
        result = jnp.asarray(self.matrix()).T @ jnp.stack(cotangents)
        return tuple((result[i] for i in range(len(self.basis))))

def operator_span(operators, dimension, *, limits=SpanLimits()):
    if not operators:
        raise ValueError('nonempty operator sequence required')
    rows = []
    compact = 0
    work = 0
    for op in operators:
        op.check_dimension(dimension)
        if op.mode == 'directional':
            raise NotImplementedError('runtime direction identity is not fixed symbolic metadata')
        if op.mode == 'laplacian':
            import math
            needed = math.comb(dimension + op.power - 1, op.power)
        else:
            needed = len(op.terms) if op.mode == 'polynomial' else 1
        try:
            if needed > limits.max_terms or needed * dimension + work > limits.max_work:
                raise ValueError('local symbolic expansion budget')
            row = {('mixed', alpha): Fraction(coefficient) for alpha, coefficient in op.resolve_terms(dimension)}
        except ValueError:
            if op.mode != 'laplacian':
                raise
            row = {('laplacian', op.power): Fraction(1)}
            compact += 1
        rows.append(row)
        work += len(row) * dimension
        if work > limits.max_work:
            raise ValueError('span planning work exhausted')
    echelon, basis, reconstructions = ([], [], [])
    for op, source in zip(operators, rows):
        remainder, coefficients = (dict(source), {})
        for pivot, vector, representation in echelon:
            work += 1
            if work > limits.max_work:
                raise ValueError('span pivot inspection work exhausted')
            factor = remainder.get(pivot, Fraction(0))
            if not factor:
                continue
            for key, value in vector.items():
                remainder[key] = remainder.get(key, Fraction(0)) - factor * value
                if not remainder[key]:
                    del remainder[key]
            for key, value in representation.items():
                coefficients[key] = coefficients.get(key, Fraction(0)) + factor * value
            work += len(vector) + len(representation)
            if work > limits.max_work:
                raise ValueError('span elimination work exhausted')
        if remainder:
            index = len(basis)
            basis.append(op)
            pivot = min(remainder, key=repr)
            divisor = remainder[pivot]
            vector = {key: value / divisor for key, value in remainder.items()}
            representation = {key: -value / divisor for key, value in coefficients.items()}
            representation[index] = Fraction(1) / divisor
            echelon.append((pivot, vector, representation))
            coefficients = {index: Fraction(1)}
        numbers = [*coefficients.values(), *(v for _, a, b in echelon for v in (*a.values(), *b.values()))]
        work += len(numbers)
        if work > limits.max_work:
            raise ValueError('span coefficient inspection work exhausted')
        if any((max(v.numerator.bit_length(), v.denominator.bit_length()) > limits.max_coefficient_bits for v in numbers)):
            raise ValueError('span coefficient bit budget exhausted')
        reconstructions.append(coefficients)
    work += len(reconstructions) * len(basis)
    if work > limits.max_work:
        raise ValueError('span reconstruction layout budget exhausted')
    matrix = tuple((tuple((row.get(i, Fraction(0)) for i in range(len(basis)))) for row in reconstructions))
    return OperatorSpan(tuple(basis), matrix, {'requested': len(operators), 'basis': len(basis), 'compact_atoms': compact, 'planning_work': work, 'exact_rational_identity': True, 'numerical_qualification': 'reconstruction constants and arithmetic must be bounded'})

def build_reduced(model, operators, provider='parity', *, limits=SpanLimits(), **options):
    dimension = model.model.dimension if isinstance(model, SharedModel) else model.dimension
    span = operator_span(operators, dimension, limits=limits)
    kernel = (build_shared if isinstance(model, SharedModel) else build)(model, span.basis, provider, **options)

    def function(params, points):
        return span.reconstruct(kernel(params, points))
    function.demands = {**kernel.demands, 'operator_span': span.report}
    function.constant_bounds = coefficient_bounds(span.basis, dimension).extend(span.constant_bounds())
    return (function, span)

def build_reduced_streaming(model, operators, objective, provider='parity', *, limits=SpanLimits(), **options):
    """Basis evaluation and reconstruction inside each streamed chunk.

    The original row objective type is retained, including the single-pass sum
    specialization. Only a chunk of original outputs is reconstructed at once;
    full_outputs remains available with its explicit batch-sized output contract.
    """
    from .streaming import build_streaming
    dimension = model.model.dimension if isinstance(model, SharedModel) else model.dimension
    span = operator_span(operators, dimension, limits=limits)
    program = build_streaming(model, span.basis, objective, provider=provider, **options)
    evaluate_basis = program.evaluate
    program.evaluate = lambda params, points, prepared: span.reconstruct(evaluate_basis(params, points, prepared))
    program.report = {**program.report, 'operator_span': span.report}
    program.constant_bounds = coefficient_bounds(span.basis, dimension).extend(span.constant_bounds())
    return (program, span)
