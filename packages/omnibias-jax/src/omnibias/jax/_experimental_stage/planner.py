"""Bounded, conditional numerical qualification and amortized hardware planning."""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time
import jax
import jax.numpy as jnp
import numpy as np
from .bounds import Bound, interpret
from .bundles import build, joint, objective
from .cache_manager import CACHES, Artifact, digest_array, digest_record, environment_snapshot, environment_info, refresh_environment, configure_persistent_cache

@dataclass(frozen=True)
class Tolerances:
    forward_atol: float = 1e-09
    forward_rtol: float = 2e-10
    gradient_atol: float = 1e-08
    gradient_rtol: float = 2e-09

    def __post_init__(self):
        if any((not np.isfinite(v) or v < 0 for v in asdict(self).values())):
            raise ValueError('finite nonnegative tolerances required')
        if self.forward_atol == 0 or self.gradient_atol == 0:
            raise ValueError('positive absolute tolerances required')

@dataclass(frozen=True)
class Limits:
    candidates: int = 256
    shortlist: int = 4
    cells: int = 4096
    equations: int = 100000
    state_bytes: int = 3 * 1024 ** 3
    cache_entries: int = 16

    def __post_init__(self):
        if not (1 <= self.candidates <= 256 and 1 <= self.shortlist <= 4 and (1 <= self.cells <= 4096) and (self.equations > 0) and (self.state_bytes > 0) and (1 <= self.cache_entries <= 16)):
            raise ValueError('invalid bounded planning limits')

@dataclass(frozen=True)
class Domain:
    parameters: tuple
    inputs: Bound

@dataclass(frozen=True)
class Reuse:
    expected_uses: int = 1
    derivative_scope: str = 'outputs, objective, first parameter gradients'
    lifetime: str = 'complete differentiated step'

    def __post_init__(self):
        if not isinstance(self.expected_uses, int) or self.expected_uses < 1:
            raise ValueError('positive integer expected reuse required')
        if self.derivative_scope != 'outputs, objective, first parameter gradients':
            raise NotImplementedError('unsupported derivative scope')
CACHE = CACHES.plans

def clear_cache():
    CACHES.clear()

def cache_info():
    return {**CACHES.info(), **environment_info()}

def _array_hash(value):
    return digest_array(value)

def identity(model, operators, args, domain, scalar_objective, limits, tolerances, reuse, policy, calibration_args=None, variant=None, selection_variant=None):
    graph = jax.make_jaxpr(scalar_objective)(args[0], args[1], tuple((jnp.zeros(args[1].shape[0], dtype=jnp.float64) for _ in operators)))

    def constants(closed):
        values = [_array_hash(v) for v in getattr(closed, 'consts', ())]
        for equation in getattr(closed, 'jaxpr', closed).eqns:
            for name in ('jaxpr', 'call_jaxpr'):
                sub = equation.params.get(name)
                if sub is not None:
                    values.extend(constants(sub))
        return values
    environment = environment_snapshot()
    common = {'model': asdict(model), 'operators': [op.to_config() for op in operators], 'shapes': [list(a.shape) for a in jax.tree.leaves(args)], 'dtypes': [str(a.dtype) for a in jax.tree.leaves(args)], 'objective_graph': re.sub('0x[0-9a-fA-F]+', 'ADDRESS', str(graph)), 'objective_constants': constants(graph), 'environment': environment['digest'], 'derivative_scope': reuse.derivative_scope, 'variant': variant}
    certificate = {'common': digest_record(common), 'domains': [[_array_hash(v) for v in (b.lo, b.hi, b.error)] for b in (*domain.parameters, domain.inputs)], 'tolerances': asdict(tolerances), 'limits': asdict(limits)}
    calibration = args if calibration_args is None else calibration_args
    selection = {'certificate': digest_record(certificate), 'policy': policy, 'reuse': asdict(reuse), 'limits': asdict(limits), 'selection_variant': selection_variant, 'calibration': [_array_hash(v) for v in jax.tree.leaves(calibration)]}
    record = {'common': common, 'certificate': certificate, 'selection': selection, 'environment_digest': environment['digest'], 'executable_common_key': digest_record(common), 'certificate_common_key': digest_record(certificate), 'selection_key': digest_record(selection)}
    return (record['selection_key'], record)

def _domain_contains(outer, inner):
    boxes_a = (*outer.parameters, outer.inputs)
    boxes_b = (*inner.parameters, inner.inputs)
    return len(boxes_a) == len(boxes_b) and all((np.all(np.asarray(a.lo) <= np.asarray(b.lo)) and np.all(np.asarray(a.hi) >= np.asarray(b.hi)) and (not np.any(a.error)) and (not np.any(b.error)) for a, b in zip(boxes_a, boxes_b)))

def _certificate_reuse(executable_key, domain, tolerances):
    for artifact in CACHES.certificates.values():
        item = artifact.value
        if item['executable_key'] != executable_key or not item['report'].get('admissible'):
            continue
        old = item['tolerances']
        if all((getattr(tolerances, name) >= getattr(old, name) for name in asdict(old))) and _domain_contains(item['domain'], domain):
            return (item['report'], item['key'])
    return (None, None)

def _calibration_arguments(args, domain, seed=831001):
    rng = np.random.default_rng(seed)
    parameters = tuple((jnp.asarray(b.lo + (b.hi - b.lo) * rng.uniform(0.2, 0.8, b.lo.shape), dtype=jnp.float64) for b in domain.parameters))
    lo = np.broadcast_to(domain.inputs.lo, args[1].shape)
    hi = np.broadcast_to(domain.inputs.hi, args[1].shape)
    points = jnp.asarray(lo + (hi - lo) * rng.uniform(0.2, 0.8, lo.shape), dtype=jnp.float64)
    return (parameters, points)

def qualify(function, args, domain, scalar_objective=objective, tolerances=Tolerances(), limits=Limits(), constant_bounds=None, emitted_value_and_grad=None):
    if constant_bounds is None:
        constant_bounds = getattr(function, 'constant_bounds', None)
    objective_graph = jax.make_jaxpr(scalar_objective)(args[0], args[1], function(*args))

    def inspect_value(value):
        if hasattr(value, 'jaxpr'):
            inspect(value.jaxpr)
        elif hasattr(value, 'eqns'):
            inspect(value)
        elif isinstance(value, dict):
            for item in value.values():
                inspect_value(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                inspect_value(item)

    def inspect(graph):
        for equation in graph.eqns:
            if equation.primitive.name in ('stop_gradient', 'custom_jvp_call', 'custom_vjp_call'):
                raise NotImplementedError('objective derivative rule outside proved contract')
            inspect_value(equation.params)
    inspect(objective_graph.jaxpr)
    forward = jax.make_jaxpr(function)(*args)
    gradient = jax.make_jaxpr(joint(function, scalar_objective) if emitted_value_and_grad is None else emitted_value_and_grad)(*args)
    initial = [*domain.parameters, Bound(np.broadcast_to(domain.inputs.lo, args[1].shape), np.broadcast_to(domain.inputs.hi, args[1].shape), 0.0)]
    weight_function = getattr(scalar_objective, '_weight_contract', None)
    if weight_function is None:
        owner = getattr(scalar_objective, '__self__', None)
        descriptor = getattr(owner, 'objective', owner)
        weight_function = getattr(descriptor, 'weight_fn', None)
    weight_graph = jax.make_jaxpr(jax.vmap(weight_function, in_axes=(None, 0)))(*args) if weight_function is not None else None
    if weight_graph is not None:
        inspect(weight_graph.jaxpr)
    pending = [initial]
    leaves = 1
    records = []
    analyses = 0
    output_count = len(function(*args))
    auxiliary_count = output_count if emitted_value_and_grad is None else 0
    while pending:
        boxes = pending.pop()
        analyses += 1
        weight_metadata = None
        remaining = limits.equations
        if weight_graph is not None:
            weights, weight_metadata = interpret(weight_graph, boxes, remaining, max_state_bytes=limits.state_bytes)
            lo, hi = weights[0].enlarged()
            if np.any(lo < 0):
                return {'admissible': False, 'reason': 'nonnegative row weights not established', 'weight_contract': weights[0].summary(tolerances.forward_atol, tolerances.forward_rtol)}
            remaining -= weight_metadata['equations']
        values, fmeta = interpret(forward, boxes, remaining, constant_bounds, max_state_bytes=limits.state_bytes)
        grads, gmeta = interpret(gradient, boxes, limits.equations, constant_bounds, max_state_bytes=limits.state_bytes)
        summary = lambda v: v.summary(tolerances.forward_atol, tolerances.forward_rtol)
        row = {'outputs': [summary(v) for v in values], 'objective': summary(grads[0]), 'auxiliary_outputs': [summary(v) for v in grads[1:1 + auxiliary_count]], 'gradients': [v.summary(tolerances.gradient_atol, tolerances.gradient_rtol) for v in grads[1 + auxiliary_count:]], 'forward_graph': fmeta, 'gradient_graph': gmeta}
        if weight_metadata is not None:
            row['nonnegative_weight_contract'] = weight_metadata
        ok = all((v['admissible'] for v in row['outputs'])) and row['objective']['admissible'] and all((v['admissible'] for v in row['auxiliary_outputs'])) and all((v['admissible'] for v in row['gradients']))
        if ok:
            records.append(row)
            continue
        if leaves >= limits.cells:
            return {'admissible': False, 'reason': 'accuracy/subdivision budget inconclusive', 'last_cell': row, 'leaf_cells': leaves, 'analyzed_cells': analyses}
        widths = [b.hi - b.lo for b in boxes]
        which = int(np.argmax([np.max(w) for w in widths]))
        index = np.unravel_index(np.argmax(widths[which]), widths[which].shape)
        b = boxes[which]
        middle = float(b.lo[index] * 0.5 + b.hi[index] * 0.5)
        if not b.lo[index] < middle < b.hi[index]:
            return {'admissible': False, 'reason': 'no representable split'}
        left, right = (list(boxes), list(boxes))
        hi = b.hi.copy()
        lo = b.lo.copy()
        hi[index] = middle
        lo[index] = middle
        left[which] = Bound(b.lo, hi, 0.0)
        right[which] = Bound(lo, b.hi, 0.0)
        pending.extend((left, right))
        leaves += 1
    return {'admissible': True, 'cells': records, 'leaf_cells': leaves, 'analyzed_cells': analyses, 'contract': 'conditional emitted outputs, objective and first-gradient bounds', 'assumptions': 'see plan identity; native evaluations and compatible compiler arithmetic assumed'}

def timed(function, args, count=11):
    for _ in range(3):
        jax.block_until_ready(function(*args))
    samples = []
    for _ in range(count):
        start = time.perf_counter_ns()
        jax.block_until_ready(function(*args))
        samples.append(time.perf_counter_ns() - start)
    return {'samples_ns': samples, 'median_ns': float(np.median(samples))}

def freeze_callable(function, args):
    """Freeze traced constants while keeping a transformable JAX expression.

    Compiled executables cannot be nested in cond/scan transformations. A closed
    jaxpr with owned constant snapshots can, and cannot retrace a mutated caller
    objective under a certificate established for different constants.
    """
    from jax._src import core
    original = jax.make_jaxpr(function)(*args)
    from .bounds import ConstantBindings
    old_bindings = getattr(function, 'constant_bounds', None)
    if old_bindings is not None and hasattr(old_bindings, 'slots') and any((path for path, _ in old_bindings.slots)):
        raise NotImplementedError('nested slot-only bindings must be converted to object provenance before freezing')
    bindings = ConstantBindings()
    if old_bindings is not None:
        if isinstance(old_bindings, ConstantBindings):
            bindings.legacy.update(old_bindings.legacy)
        else:
            bindings.extend(old_bindings)

    def own(value, path=()):
        if isinstance(value, core.ClosedJaxpr):
            consts = []
            for index, v in enumerate(value.consts):
                snapshot = jax.device_put(np.array(v, copy=True))
                if isinstance(old_bindings, ConstantBindings):
                    bound = old_bindings.resolve(v, path, index)
                    if bound is not None:
                        bindings.bind_object(snapshot, bound).add_legacy(snapshot, bound)
                consts.append(snapshot)
            return core.ClosedJaxpr(own(value.jaxpr, path), tuple(consts))
        if isinstance(value, core.Jaxpr):
            equations = [equation.replace(params=own(equation.params, path + (str(index),))) for index, equation in enumerate(value.eqns)]
            return value.replace(eqns=equations)
        if isinstance(value, dict):
            return {key: own(item, path + (str(key),)) for key, item in value.items()}
        if isinstance(value, tuple):
            items = tuple((own(item, path + (str(index),)) for index, item in enumerate(value)))
            return type(value)(*items) if hasattr(value, '_fields') else items
        if isinstance(value, list):
            return [own(item, path + (str(index),)) for index, item in enumerate(value)]
        return value
    closed = own(original)
    constants = closed.consts
    output_tree = jax.tree.structure(jax.eval_shape(function, *args))
    input_tree = jax.tree.structure(args)

    def frozen(*runtime_args):
        if jax.tree.structure(runtime_args) != input_tree:
            raise ValueError('frozen emitted program argument structure changed')
        outputs = core.eval_jaxpr(closed.jaxpr, constants, *jax.tree.leaves(runtime_args))
        return jax.tree.unflatten(output_tree, outputs)
    frozen.closed_jaxpr = core.ClosedJaxpr(closed.jaxpr, constants)
    frozen.constant_bounds = bindings
    weight_function = getattr(function, '_weight_contract', None)
    if weight_function is None:
        owner = getattr(function, '__self__', None)
        descriptor = getattr(owner, 'objective', owner)
        weight_function = getattr(descriptor, 'weight_fn', None)
    if weight_function is not None and len(args) >= 2:
        frozen._weight_contract = freeze_callable(weight_function, (args[0], args[1][0]))
    return frozen

@jax.jit
def runtime_guard(params, points, lower, upper, input_lower, input_upper):
    valid = jnp.all(jnp.isfinite(points) & (points >= input_lower) & (points <= input_upper))
    for value, lo, hi in zip(params, lower, upper):
        valid = valid & jnp.all(jnp.isfinite(value) & (value >= lo) & (value <= hi))
    return valid

class Program:

    def __init__(self, function, compiled, joint_compiled, domain, report, objective_compiled=None, scalar_objective=objective, objective_function=None):
        self.function, self.kernel, self.gradient_kernel = (function, compiled, joint_compiled)
        self.domain, self.report = (domain, report)
        self.scalar_objective = scalar_objective
        self.objective_gradient_kernel = objective_compiled
        self.objective_gradient_function = objective_function or _objective_gradient(function, scalar_objective)
        self.runtime_boxes = (tuple((jnp.asarray(b.lo) for b in domain.parameters)), tuple((jnp.asarray(b.hi) for b in domain.parameters)), jnp.asarray(domain.inputs.lo), jnp.asarray(domain.inputs.hi))
        self.environment_digest = report.get('environment_digest')

    def guard(self, params, points):
        return runtime_guard(params, points, *self.runtime_boxes)

    def structural_check(self, params, points):
        if tuple(points.shape) != tuple(self.report['input_shape']) or points.dtype != jnp.float64:
            raise ValueError('qualified input shape and binary64 required')
        if len(params) != len(self.domain.parameters) or any((p.shape != b.lo.shape or p.dtype != jnp.float64 for p, b in zip(params, self.domain.parameters))):
            raise ValueError('qualified parameter shapes and binary64 required')
        if self.environment_digest and self.environment_digest != environment_snapshot()['digest']:
            raise ValueError('environment refreshed; rebuild this plan')

    def check(self, params, points):
        self.structural_check(params, points)
        if not bool(self.guard(params, points)):
            raise ValueError('runtime arguments leave the qualified domain')

    def __call__(self, params, points):
        self.check(params, points)
        return self.kernel(params, points)

    def value_and_grad(self, params, points):
        self.check(params, points)
        return self.gradient_kernel(params, points)

    def objective_value_and_grad(self, params, points):
        self.check(params, points)
        return self.objective_gradient_kernel(params, points)

    def device_objective_value_and_grad(self, params, points):
        """Returns device status without a host synchronization; invalid arguments do not evaluate."""
        self.structural_check(params, points)

        def accepted(operands):
            p, x = operands
            value, gradient = self.objective_gradient_function(p, x)
            finite = jnp.isfinite(value)
            for g in jax.tree.leaves(gradient):
                finite = finite & jnp.all(jnp.isfinite(g))
            return (jnp.where(finite, 0, 2).astype(jnp.int32), value, gradient)

        def refused(operands):
            p, _ = operands
            return (jnp.asarray(1, jnp.int32), jnp.asarray(0.0, jnp.float64), jax.tree.map(jnp.zeros_like, p))
        return jax.lax.cond(self.guard(params, points), accepted, refused, (params, points))

def _build_function(model, operators, provider, chunk, remat):
    if hasattr(model, 'role_bindings'):
        from .shared_models import build_shared
        fn = build_shared(model, operators, provider, chunk_size=chunk, rematerialize=remat)
        from .bundles import coefficient_bounds
        fn.constant_bounds = coefficient_bounds(operators, model.model.dimension)
        return fn
    return build(model, operators, provider, chunk_size=chunk, rematerialize=remat)

def _objective_gradient(function, scalar_objective):
    return jax.value_and_grad(lambda p, x: scalar_objective(p, x, function(p, x)))

def _memory_bytes(executable):
    memory = executable.memory_analysis()
    if memory is None:
        raise ArithmeticError('executable memory unavailable')
    return int(memory.argument_size_in_bytes + memory.output_size_in_bytes + memory.temp_size_in_bytes - memory.alias_size_in_bytes)

def _emitted_graph_identity(graph):
    constants = []

    def collect(value):
        if hasattr(value, 'jaxpr'):
            constants.extend((_array_hash(v) for v in value.consts))
            collect(value.jaxpr)
        elif hasattr(value, 'eqns'):
            for equation in value.eqns:
                collect(equation.params)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
    collect(graph)
    return {'graph': re.sub('0x[0-9a-fA-F]+', 'ADDRESS', str(graph)), 'constants': constants}

def compile_operator_bundle(model, operators, params, points, *, domain, scalar_objective=objective, reuse=Reuse(), limits=Limits(), tolerances=Tolerances(), policy='qualified_dispatch', calibration_args=None, streaming_objective=None, chunk_size=256, measured_samples=11, operator_reduction=True, setup_strategy='auto'):
    if not jax.config.x64_enabled or any((a.dtype != jnp.float64 for a in jax.tree.leaves((params, points)))):
        raise ValueError('binary64 runtime arguments/configuration required')
    if policy not in ('specialization_only', 'qualified_dispatch'):
        raise ValueError('unknown selection policy')
    if len(domain.parameters) != len(params) or any((np.any(b.error) for b in (*domain.parameters, domain.inputs))):
        raise ValueError('real zero-initial-error boxes required')
    if not 1 <= measured_samples <= 51:
        raise ValueError('bounded calibration sample count required')
    if setup_strategy not in ('auto', 'full'):
        raise ValueError('setup_strategy must be auto or full')
    planning_mode = ('single_use' if reuse.expected_uses == 1 else 'short_reuse' if reuse.expected_uses <= 10 else 'full') if setup_strategy == 'auto' else 'full'
    if streaming_objective is not None:
        scalar_objective = streaming_objective.full_objective
    start = time.perf_counter()
    args = (params, points)
    objective_descriptor = type(streaming_objective).__module__ + '.' + type(streaming_objective).__qualname__ if streaming_objective is not None else 'full_scalar_callable'
    scalar_objective = freeze_callable(scalar_objective, (params, points, tuple((jnp.zeros(points.shape[0], dtype=jnp.float64) for _ in operators))))
    calibration = _calibration_arguments(args, domain) if calibration_args is None else calibration_args
    for values in (args, calibration):
        if jax.tree.structure(values) != jax.tree.structure(args) or any((a.shape != b.shape or a.dtype != b.dtype for a, b in zip(jax.tree.leaves(values), jax.tree.leaves(args)))):
            raise ValueError('calibration and execution shapes/dtypes must agree')
        for p, b in zip(values[0], domain.parameters):
            if p.shape != b.lo.shape or np.any(np.asarray(p) < b.lo) or np.any(np.asarray(p) > b.hi):
                raise ValueError('parameter outside declared box at preparation')
        if np.any(np.asarray(values[1]) < domain.inputs.lo) or np.any(np.asarray(values[1]) > domain.inputs.hi):
            raise ValueError('input outside declared box at preparation')
    variant = {'streaming': streaming_objective is not None, 'chunk_size': chunk_size if streaming_objective is not None else None, 'operator_reduction': operator_reduction, 'objective_descriptor': objective_descriptor}
    key, ident = identity(model, operators, args, domain, scalar_objective, limits, tolerances, reuse, policy, calibration_args=calibration, variant=variant, selection_variant={'measured_samples': measured_samples, 'setup_strategy': setup_strategy, 'planning_mode': planning_mode})
    cached = CACHES.get_plan(key)
    if cached is not None:
        cached.report['last_lookup_s'] = time.perf_counter() - start
        cached.report['last_lookup_kind'] = 'complete_plan_hit'
        return (cached, cached.report)
    provider_choices = ('parity', 'jax_analytic') if planning_mode == 'single_use' else ('parity', 'sech', 'jax_native', 'jax_analytic') if planning_mode == 'short_reuse' else ('polynomial', 'parity', 'sech', 'recurrence', 'jax_native', 'jax_analytic')
    candidates = [(provider, None, remat, False) for provider in provider_choices for remat in ((False, True) if planning_mode == 'full' else (False,))]
    if points.shape[0] > 256 and streaming_objective is None:
        candidates.extend(((p, 256, False, False) for p in ('parity', 'sech', 'jax_analytic')))
    span_screen = {}
    if operator_reduction:
        try:
            from .reduction import operator_span
            dimension = model.model.dimension if hasattr(model, 'role_bindings') else model.dimension
            span = operator_span(operators, dimension)
            span_screen = span.report
            if len(span.basis) < len(operators):
                candidates += [(p, c, r, True) for p, c, r, _ in candidates]
        except (ArithmeticError, ValueError, NotImplementedError) as error:
            span_screen = {'status': 'inconclusive', 'reason': str(error)}
    complete_candidate_count = len(candidates)
    candidates = candidates[:limits.candidates]
    records, qualified = ([], [])
    pending_certificates = {}
    baseline_names = {'jax_native', 'jax_analytic'}
    for index, (provider, chunk, remat, reduced) in enumerate(candidates):
        row = {'index': index, 'provider': provider, 'chunk': chunk, 'rematerialize': remat, 'baseline': provider in baseline_names, 'operator_reduced': reduced, 'status': 'inconclusive'}
        t = time.perf_counter()
        executable_key = digest_record({'common': ident['executable_common_key'], 'provider': provider, 'chunk': chunk, 'remat': remat, 'operator_reduced': reduced})
        certificate_key = digest_record({'executable': executable_key, 'certificate': ident['certificate_common_key']})
        try:
            fn = _build_function(model, operators, provider, chunk, remat)
            if reduced:
                from .reduction import build_reduced
                fn, _ = build_reduced(model, operators, provider, chunk_size=chunk, rematerialize=remat)
            objective_vg = _objective_gradient(fn, scalar_objective)
            if streaming_objective is not None:
                from .streaming import build_streaming
                if reduced:
                    from .reduction import build_reduced_streaming
                    stream, _ = build_reduced_streaming(model, operators, streaming_objective, provider, chunk_size=chunk_size, rematerialize=remat)
                else:
                    stream = build_streaming(model, operators, streaming_objective, chunk_size=chunk_size, provider=provider, rematerialize=remat)
                objective_vg = stream.value_and_grad
                row['streaming'] = stream.report
            objective_vg = freeze_callable(objective_vg, args)
            row['demands'] = fn.demands
            graph = jax.make_jaxpr(objective_vg)(*args)
            executable_key = digest_record({'base': executable_key, 'forward': _emitted_graph_identity(jax.make_jaxpr(fn)(*args)), 'full_joint': _emitted_graph_identity(jax.make_jaxpr(joint(fn, scalar_objective))(*args)), 'objective_joint': _emitted_graph_identity(graph)})
            certificate_key = digest_record({'executable': executable_key, 'certificate': ident['certificate_common_key']})
            row['equations'] = len(graph.jaxpr.eqns)
            if row['equations'] > limits.equations:
                raise ArithmeticError('graph budget exhausted')
            largest = max((np.prod(v.aval.shape) * 8 for e in graph.jaxpr.eqns for v in e.outvars if hasattr(v.aval, 'shape')), default=0)
            row['largest_array_bytes'] = int(largest)
            if largest > limits.state_bytes:
                raise ArithmeticError('array state budget exhausted')
            exact_certificate = CACHES.certificates.get(certificate_key)
            reused_report, reused_key = (exact_certificate.value['report'], certificate_key) if exact_certificate else _certificate_reuse(executable_key, domain, tolerances)
            if reused_report is not None:
                row['qualification'] = reused_report
                row['certificate_reused_from'] = reused_key
            else:
                first = qualify(fn, args, domain, scalar_objective, tolerances, limits)
                second = qualify(fn, args, domain, scalar_objective, tolerances, limits, emitted_value_and_grad=objective_vg) if first['admissible'] else first
                row['qualification'] = {'admissible': first['admissible'] and second['admissible'], 'complete_outputs_and_gradient': first, 'objective_only_and_gradient': second, 'assumptions': 'conditional binary64 and native primitive hypotheses'}
            row['admissible'] = row['qualification']['admissible']
            if row['admissible']:
                snapshot_domain = Domain(tuple((Bound(b.lo.copy(), b.hi.copy(), b.error.copy()) for b in domain.parameters)), Bound(domain.inputs.lo.copy(), domain.inputs.hi.copy(), domain.inputs.error.copy()))
                certificate = {'report': row['qualification'], 'domain': snapshot_domain, 'tolerances': tolerances, 'executable_key': executable_key, 'key': certificate_key}
                certificate_bytes = sum((b.lo.nbytes + b.hi.nbytes + b.error.nbytes for b in (*snapshot_domain.parameters, snapshot_domain.inputs)))
                certificate_bytes += len(json.dumps(row['qualification'], default=str))
                if CACHES.info()['reported_artifact_bytes'] + sum((a.reported_bytes for a in pending_certificates.values())) + certificate_bytes > limits.state_bytes:
                    raise ArithmeticError('aggregate certificate cache budget exhausted')
                pending_certificates[certificate_key] = Artifact(certificate, certificate_bytes)
                qualified.append((row['equations'], row['largest_array_bytes'], index, fn, objective_vg, executable_key, certificate_key))
                row['status'] = 'conditionally_admissible'
        except (ArithmeticError, ValueError, NotImplementedError, TypeError) as error:
            row.update(admissible=False, reason=str(error))
        row['qualification_s'] = time.perf_counter() - t
        records.append(row)
    qualified.sort(key=lambda v: v[:3])
    finalists = [v for v in qualified if not records[v[2]]['baseline']][:limits.shortlist]
    finalists += [v for v in qualified if records[v[2]]['baseline']]
    measured = []
    pending_executables = {}
    for _, _, index, fn, objective_vg, executable_key, certificate_key in finalists:
        row = records[index]
        t = time.perf_counter()
        try:
            hit = CACHES.executables.get(executable_key)
            if hit is None:
                f = jax.jit(fn).lower(*args).compile()
                vg = jax.jit(joint(fn, scalar_objective)).lower(*args).compile()
                ovg = jax.jit(objective_vg).lower(*args).compile()
                memory = _memory_bytes(f) + _memory_bytes(vg) + _memory_bytes(ovg)
                if memory > limits.state_bytes:
                    raise ArithmeticError('combined executable state budget exhausted')
                if CACHES.info()['reported_artifact_bytes'] + sum((a.reported_bytes for a in pending_executables.values())) + memory > limits.state_bytes:
                    raise ArithmeticError('aggregate executable cache budget exhausted')
                pending_executables[executable_key] = Artifact((f, vg, ovg, objective_vg), memory)
            else:
                f, vg, ovg, objective_vg = hit.value
                memory = hit.reported_bytes
                row['executable_cache_hit'] = True
            if memory > limits.state_bytes:
                raise ArithmeticError('cached executable state budget exhausted')
            row['compile_s'] = time.perf_counter() - t
            row['executable_bytes'] = memory
            row['forward'] = timed(f, calibration, count=measured_samples)
            row['joint'] = timed(vg, calibration, count=measured_samples)
            row['objective_joint'] = timed(ovg, calibration, count=measured_samples)
            wrapper = Program(fn, f, vg, domain, {'input_shape': list(points.shape)}, ovg, scalar_objective, objective_vg)
            row['guard'] = timed(wrapper.guard, calibration, count=measured_samples)
            row['guarded_objective_joint'] = timed(wrapper.objective_value_and_grad, calibration, count=measured_samples)
            measured.append((row['guarded_objective_joint']['median_ns'], memory, index, fn, f, vg, ovg, executable_key, certificate_key, objective_vg))
        except (ArithmeticError, ValueError, NotImplementedError, TypeError) as error:
            row.update(status='inconclusive_compile', reason=str(error))
    search_s = time.perf_counter() - start
    for entry in measured:
        row = records[entry[2]]
        row['estimated_total_s'] = search_s + reuse.expected_uses * entry[0] * 1e-09
    measured.sort(key=lambda v: (records[v[2]]['estimated_total_s'], v[1], v[2]))
    baselines = [v for v in measured if records[v[2]]['baseline']]
    specials = [v for v in measured if not records[v[2]]['baseline']]
    base, special = (baselines[0] if baselines else None, specials[0] if specials else None)
    selected = special if special and (base is None or special[:3] <= base[:3]) else base if policy == 'qualified_dispatch' else None
    report = {'status': 'selected' if selected else 'inconclusive', 'identity': ident, 'key': key, 'environment_digest': ident['environment_digest'], 'candidates': records, 'input_shape': list(points.shape), 'policy': policy, 'candidate_truncation': len(candidates) < complete_candidate_count, 'specialization_shortlist': limits.shortlist, 'expected_uses': reuse.expected_uses, 'operator_span_screen': span_screen, 'planning_s': search_s, 'calibration_id': ident['selection']['calibration'], 'planning_mode': planning_mode, 'setup_strategy': setup_strategy, 'optimality_scope': 'minimum calibrated total estimate among measured admissible finalists only', 'calibration_source': 'separate supplied workload' if calibration_args is not None else 'separate deterministic domain samples seed831001', 'setup_accounting': 'all candidate search, qualification, compilation and calibration charged to cold use', 'promotion': 'none: calibration is not matched evidence'}
    if base and special:
        b = records[base[2]]
        saving = (base[0] - special[0]) * 1e-09
        baseline_setup = b['qualification_s'] + b['compile_s']
        report['break_even_uses'] = max(0.0, search_s - baseline_setup) / saving if saving > 0 else None
        report['break_even_scope'] = 'includes common planner search; standalone baseline setup estimated from measured row'
    if selected is None:
        CACHES.prune()
        return (None, report)
    _, memory, index, fn, f, vg, ovg, executable_key, certificate_key, objective_vg = selected
    report.update(selected=records[index]['provider'], selected_index=index, executable_key=executable_key, certificate_key=certificate_key, executable_bytes=memory, estimated_total_s=records[index]['estimated_total_s'])
    program = Program(fn, f, vg, domain, report, ovg, scalar_objective, objective_vg)
    if executable_key in pending_executables:
        CACHES.executables[executable_key] = pending_executables[executable_key]
    CACHES.certificates[certificate_key] = pending_certificates[certificate_key]
    selection_bytes = len(json.dumps(report, default=str))
    CACHES.selections[key] = Artifact(report, selection_bytes)
    CACHES.put_plan(key, program, executable_keys=(executable_key,), certificate_keys=(certificate_key,), selection_key=key, limit=limits.cache_entries, memory_limit=limits.state_bytes)
    return (program, report)
