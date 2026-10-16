"""A posteriori unique least-squares solve certificates from small statistics.

QR/SVD computes the candidate. Gram statistics are used only to check it.
The ideal problem is 1/2 ||F c-y||² for the supplied real binary64 F,y.
A numerical rank decision is not an exact-rank certificate.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from .bounds import Bound, binary, dot, reduction, up, down, immutable_bound

def _matmul(a, b):
    a = a if isinstance(a, Bound) else Bound.point(a)
    b = b if isinstance(b, Bound) else Bound.point(b)
    return dot(a, b, (((1,), (0,)), ((), ())))

def _norm_upper(value):
    value = np.asarray(value, dtype=float).ravel()
    squared = binary(Bound.point(value), Bound.point(value), 'mul')
    summed = reduction(squared, (0,))
    return float(up(np.sqrt(max(0.0, float(summed.hi)))))

def _digest(*arrays):
    result = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        result.update(str((a.shape, a.dtype.str)).encode())
        result.update(a.tobytes())
    return result.hexdigest()

@dataclass(frozen=True)
class NormalStatistics:
    gram: Bound
    linear: Bound
    target_square: Bound
    rows: int
    columns: int
    source_digest: str
    host_bytes: int

def prepare_statistics(design, target):
    """Own interval statistics of exact-real FᵀF, Fᵀy and yᵀy, using O(H²)."""
    if np.iscomplexobj(design) or np.iscomplexobj(target):
        raise ValueError('real binary64 least-squares inputs required')
    f, y = (np.asarray(design, dtype=np.float64), np.asarray(target, dtype=np.float64))
    if f.ndim != 2 or min(f.shape) < 1 or y.shape != (f.shape[0],):
        raise ValueError('nonempty design F(M,H) and target y(M) required')
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(y)):
        raise ValueError('finite least-squares inputs required')
    h = _matmul(f.T, f)
    b = _matmul(f.T, y[:, None])
    yy = _matmul(y[None, :], y[:, None])
    gram = immutable_bound(Bound(h.lo, h.hi, 0.0))
    linear = immutable_bound(Bound(b.lo[:, 0], b.hi[:, 0], 0.0))
    square = immutable_bound(Bound(yy.lo[0, 0], yy.hi[0, 0], 0.0))
    size = sum((v.nbytes for bound in (gram, linear, square) for v in (bound.lo, bound.hi, bound.error)))
    return NormalStatistics(gram, linear, square, f.shape[0], f.shape[1], _digest(f, y), int(size))

def _tolerance(tolerances, name, default):
    if tolerances is None:
        return default
    return float(tolerances[name] if isinstance(tolerances, dict) else getattr(tolerances, name))

def certify_solution(statistics, solution, *, numerical_rank=None, tolerances=None):
    """Return a conditional coefficient/optimality/stationarity certificate.

    For an inverse proposal B and exact Gram H, ||I-BH||F<1 proves
    H invertible. H is a Gram matrix, hence positive definite in that case,
    and lambda_min(H)>=(1-rho)/||B||F. With g=Hc-b this yields
      ||c-c*||2 <= ||g||2/lambda_min,
      J(c)-J(c*) <= ||g||2²/(2 lambda_min).
    No normal equation is solved to obtain c. Unknown rank or failed bounds
    return inconclusive and the caller must not advertise an accurate solve.
    """
    report = {'status': 'inconclusive_accuracy', 'qualified': False, 'scope': 'unique exact-real least squares for supplied binary64 design/target', 'assumptions': 'binary64 RN gradual underflow, finite arithmetic, compatible dot/reduction transforms, correctly rounded sqrt', 'statistics_source_digest': statistics.source_digest, 'host_statistics_bytes': statistics.host_bytes, 'normal_statistics_usage': 'verification only; candidate computed through QR/SVD', 'numerical_rank': None if numerical_rank is None else int(numerical_rank), 'rank_is_numerical_diagnostic': True}
    if statistics.rows < statistics.columns:
        return {**report, 'status': 'inconclusive_rank', 'reason': 'underdetermined problem has no certified unique coefficient vector'}
    if numerical_rank is not None and int(numerical_rank) < statistics.columns:
        return {**report, 'status': 'inconclusive_rank', 'reason': 'solver did not retain all columns; numerical rank is not a structural rank proof'}
    if np.iscomplexobj(solution):
        raise ValueError('real solution required')
    c = np.asarray(solution, dtype=float)
    if c.shape != (statistics.columns,) or not np.all(np.isfinite(c)):
        return {**report, 'reason': 'candidate shape/nonfinite coefficient refusal'}
    atol = _tolerance(tolerances, 'forward_atol', 1e-09)
    rtol = _tolerance(tolerances, 'forward_rtol', 2e-10)
    gatol = _tolerance(tolerances, 'gradient_atol', 1e-08)
    grtol = _tolerance(tolerances, 'gradient_rtol', 2e-09)
    if not all((np.isfinite(v) and v >= 0 for v in (atol, rtol, gatol, grtol))) or atol == 0 or gatol == 0:
        raise ValueError('finite nonnegative tolerances with positive absolute parts required')
    h, b = (statistics.gram, statistics.linear)
    midpoint = h.lo * 0.5 + h.hi * 0.5
    try:
        proposal = np.linalg.inv(midpoint)
        if not np.all(np.isfinite(proposal)):
            raise ArithmeticError('nonfinite inverse proposal')
        product = _matmul(proposal, h)
        identity = Bound.point(np.eye(statistics.columns))
        residual = binary(identity, product, 'sub')
        rho = _norm_upper(np.maximum(np.abs(residual.lo), np.abs(residual.hi)))
        bnorm = _norm_upper(np.abs(proposal))
        report.update(inverse_residual_frobenius_upper=rho, inverse_proposal_frobenius_upper=bnorm)
        if not rho < 1 or not bnorm > 0:
            return {**report, 'status': 'inconclusive_rank', 'reason': 'verified inverse residual does not establish positive Gram eigenvalue'}
        lam = float(down(float(down(1 - rho)) / bnorm))
        if not lam > 0:
            return {**report, 'status': 'inconclusive_rank', 'reason': 'positive Gram eigenvalue lower bound underflowed'}
        hc = _matmul(h, c[:, None])
        gradient = binary(Bound(hc.lo[:, 0], hc.hi[:, 0], 0.0), b, 'sub')
        absolute = np.maximum(np.abs(gradient.lo), np.abs(gradient.hi))
        gnorm = _norm_upper(absolute)
        ginf = float(np.max(absolute))
        delta = float(up(gnorm / lam))
        square_norm = float(up(gnorm * gnorm))
        excess = float(up(0.5 * float(up(square_norm / lam))))
        if not all((np.isfinite(v) for v in (gnorm, ginf, delta, excess))):
            raise ArithmeticError('solve certificate bound overflow')
        quadratic = _matmul(c[None, :], Bound(hc.lo, hc.hi, 0.0))
        linear = _matmul(c[None, :], Bound(b.lo[:, None], b.hi[:, None], 0.0))
        q = binary(binary(quadratic, binary(Bound.point(2.0), linear, 'mul'), 'sub'), statistics.target_square, 'add')
        q = binary(Bound.point(0.5), q, 'mul')
        objective_lower = max(0.0, float(q.lo[0, 0]))
        objective_upper = max(0.0, float(q.hi[0, 0]))
        optimal_lower = max(0.0, float(down(objective_lower - excess)))
        coefficient_lower = np.maximum(down(np.abs(c) - delta), 0.0)
        coefficient_tolerance = down(atol + down(rtol * coefficient_lower))
        objective_tolerance = float(down(atol + float(down(rtol * optimal_lower))))
        coefficient_ok = bool(np.all(delta <= coefficient_tolerance))
        objective_ok = bool(excess <= objective_tolerance)
        gradient_ok = bool(ginf <= gatol)
        qualified = coefficient_ok and objective_ok and gradient_ok
        report.update(qualified=qualified, status='qualified' if qualified else 'inconclusive_accuracy', gram_eigenvalue_lower=lam, coefficient_l2_error_bound=delta, coefficient_minimum_tolerance=float(np.min(coefficient_tolerance)), objective_excess_bound=excess, objective_tolerance=objective_tolerance, objective_at_candidate_interval=[objective_lower, objective_upper], objective_optimum_lower=optimal_lower, stationarity_l2_upper=gnorm, stationarity_infinity_upper=ginf, stationarity_tolerance=gatol, gates={'coefficients': coefficient_ok, 'objective_excess': objective_ok, 'stationarity': gradient_ok})
        if not qualified:
            report['reason'] = 'candidate does not establish every unchanged coefficient/optimality/stationarity tolerance'
        return report
    except (ArithmeticError, np.linalg.LinAlgError, ValueError) as error:
        return {**report, 'status': 'inconclusive_rank' if isinstance(error, np.linalg.LinAlgError) else 'inconclusive_accuracy', 'reason': str(error)}
