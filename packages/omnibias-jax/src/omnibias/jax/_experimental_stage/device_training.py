"""Device-side transactional Adam steps; host authority handles qualification.

Runtime box thresholds may change without recompiling a fixed-shape step. No
evaluation is performed for an invalid state/input, or after a block stops.
"""
from typing import NamedTuple
import time
import jax
import jax.numpy as jnp
import numpy as np
OK, DOMAIN_EXIT, NONFINITE, INVALID_INPUT = range(4)

class Boxes(NamedTuple):
    parameter_lower: object
    parameter_upper: object
    input_lower: object
    input_upper: object

class TrainingState(NamedTuple):
    params: object
    first_moment: object
    second_moment: object
    accepted_steps: object

class StepResult(NamedTuple):
    accepted: TrainingState
    proposed: TrainingState
    delta: object
    loss: object
    status: object
    evaluated: object

class BlockResult(NamedTuple):
    accepted: TrainingState
    proposed: TrainingState
    delta: object
    status: object
    losses: object
    evaluated: object

def initial_state(params):
    zero = jax.tree.map(jnp.zeros_like, params)
    return TrainingState(params, zero, zero, jnp.asarray(0, dtype=jnp.int32))

def copy_state_for_donation(state, *, synchronize=True):
    """Create exclusively owned device buffers for a subsequent donated call.

    Every leaf is copied separately, outside JIT, with ``may_alias=False``.
    This handles aliased first/second moments from initial_state, repeated
    parameter leaves, and accepted/proposed output aliases without reading any
    device values on the host. The original state remains usable. The returned
    state must be relinquished after donation, whether the step accepts or
    rolls back. This boundary is intentionally charged on every safe call.

    A single local CPU/GPU device is the supported execution scope. This is
    ownership management, not a numerical transformation or certificate.
    """
    if not isinstance(state, TrainingState):
        raise TypeError('TrainingState required for state donation')
    leaves = jax.tree.leaves(state)
    if not leaves or any((not isinstance(value, jax.Array) for value in leaves)):
        raise TypeError('donation ownership requires existing device arrays; no implicit host transfers')
    if any((value.is_deleted() for value in leaves)):
        raise ValueError('cannot copy a previously donated/deleted state')
    devices = set().union(*(value.devices() for value in leaves))
    if len(devices) != 1 or any((not value.is_fully_addressable for value in leaves)):
        raise ValueError('state donation supports one fully addressable device')
    device = next(iter(devices))
    started = time.perf_counter()
    owned = jax.tree.map(lambda value: jax.device_put(value, device=device, may_alias=False), state)
    dispatch_s = time.perf_counter() - started
    if synchronize:
        jax.block_until_ready(owned)
    elapsed_s = time.perf_counter() - started
    return (owned, {'ownership': 'independent device copy per state leaf before donation', 'copy_dispatch_s': dispatch_s, 'copy_elapsed_s': elapsed_s, 'copy_synchronized': bool(synchronize), 'copied_bytes': sum((value.nbytes for value in leaves)), 'copied_leaves': len(leaves), 'device': str(device), 'host_parameter_transfers': 0, 'parameter_hashes': 0, 'input_state_preserved': True})

def jit_donated_training(step_or_block):
    """Compile the same guarded graph with state-buffer donation enabled.

    Pass only an exclusively owned state from copy_state_for_donation. Raw
    initial_state may contain aliased buffers and must not be donated directly.
    Donation is a compiler opportunity, not guaranteed memory/runtime savings.
    """
    return jax.jit(step_or_block, donate_argnums=(0,))

def call_donated_training(kernel, state, points, boxes, *, synchronize=True):
    """Preserve caller state while safely running a donated step or block.

    Returns (ordinary StepResult/BlockResult, ownership/timing report). Copying
    occurs on every call so callers may retain prior accepted/proposed states,
    including rollback records. Total time includes that ownership overhead.
    With synchronize=False times describe host dispatch only.
    """
    started = time.perf_counter()
    owned, report = copy_state_for_donation(state, synchronize=synchronize)
    execution_start = time.perf_counter()
    result = kernel(owned, points, boxes)
    if synchronize:
        jax.block_until_ready(result)
    report.update(execution_elapsed_s=time.perf_counter() - execution_start, total_elapsed_s=time.perf_counter() - started, execution_synchronized=bool(synchronize), donation='requested for state argument; actual aliasing is compiler dependent')
    return (result, report)

def _all(tree):
    leaves = jax.tree.leaves(tree)
    return jnp.all(jnp.stack(leaves)) if leaves else jnp.asarray(True)

def finite(tree):
    return _all(jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), tree))

def inside(params, lower, upper):
    return _all(jax.tree.map(lambda p, lo, hi: jnp.all(jnp.isfinite(p) & (p >= lo) & (p <= hi)), params, lower, upper))

def make_adam_step(value_and_grad, *, learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08):
    if not (np.isfinite(learning_rate) and learning_rate > 0 and (0 <= beta1 < 1) and (0 <= beta2 < 1) and np.isfinite(epsilon) and (epsilon > 0)):
        raise ValueError('finite positive Adam scales and beta values in [0,1) required')

    def step(state, points, boxes):
        zero = jax.tree.map(jnp.zeros_like, state.params)
        scalar_zero = jnp.asarray(0.0, dtype=points.dtype)
        pvalid = inside(state.params, boxes.parameter_lower, boxes.parameter_upper)
        xvalid = jnp.all(jnp.isfinite(points) & (points >= boxes.input_lower) & (points <= boxes.input_upper))
        valid = pvalid & xvalid & finite((state.first_moment, state.second_moment))

        def evaluate(_):
            value, gradient = value_and_grad(state.params, points)
            step_number = state.accepted_steps + 1
            moment = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, state.first_moment, gradient)
            variance = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g * g, state.second_moment, gradient)
            delta = jax.tree.map(lambda m, v: -learning_rate * (m / (1 - beta1 ** step_number)) / (jnp.sqrt(v / (1 - beta2 ** step_number)) + epsilon), moment, variance)
            params = jax.tree.map(jnp.add, state.params, delta)
            proposed = TrainingState(params, moment, variance, step_number)
            is_finite = finite((value, gradient, proposed, delta))
            in_domain = inside(params, boxes.parameter_lower, boxes.parameter_upper)
            status = jnp.where(is_finite, jnp.where(in_domain, OK, DOMAIN_EXIT), NONFINITE).astype(jnp.int32)
            accepted = jax.lax.cond(status == OK, lambda _: proposed, lambda _: state, None)
            return StepResult(accepted, proposed, delta, value, status, jnp.asarray(True))

        def reject(_):
            status = jnp.where(pvalid & xvalid, NONFINITE, INVALID_INPUT).astype(jnp.int32)
            return StepResult(state, state, zero, scalar_zero, status, jnp.asarray(False))
        return jax.lax.cond(valid, evaluate, reject, None)
    return step

def make_training_block(step, steps):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError('positive fixed step-block length required')

    def block(state, batches, boxes):
        if batches.shape[0] != steps:
            raise ValueError('one batch per step required')
        zero = jax.tree.map(jnp.zeros_like, state.params)
        initial = StepResult(state, state, zero, jnp.asarray(0.0, batches.dtype), jnp.asarray(OK, jnp.int32), jnp.asarray(False))

        def body(previous, batch):
            active = previous.status == OK
            result = jax.lax.cond(active, lambda _: step(previous.accepted, batch, boxes), lambda _: previous, None)
            evaluated = active & result.evaluated
            loss = jnp.where(evaluated, result.loss, jnp.asarray(0.0, batches.dtype))
            return (result, (loss, evaluated))
        final, (losses, evaluated) = jax.lax.scan(body, initial, batches)
        return BlockResult(final.accepted, final.proposed, final.delta, final.status, losses, evaluated)
    return block

def make_adam_apply_packet(**optimizer_options):
    """Apply an already inspected value/gradient without evaluating QR again.

    The caller binds the packet to this exact state and input before calling.
    This performs the usual finite/domain transactional checks. It neither
    certifies supplied numbers nor supplies an exact-optimizer theorem.
    """

    def apply(state, points, boxes, value, gradient):
        if jax.tree.structure(gradient) != jax.tree.structure(state.params):
            raise ValueError('complete packet gradient structure required')
        if value.shape != () or any((g.shape != p.shape or g.dtype != p.dtype for g, p in zip(jax.tree.leaves(gradient), jax.tree.leaves(state.params), strict=True))):
            raise ValueError('packet value and complete gradient metadata mismatch')
        step = make_adam_step(lambda p, x: (value, gradient), **optimizer_options)
        return step(state, points, boxes)
    return apply

def recenter_boxes(boxes, proposed_params, delta):
    """Host proposal only: returned boxes do not authorize resuming evaluation."""

    def radius(lo, hi, d):
        return np.maximum((np.asarray(hi) - np.asarray(lo)) * 0.5, 16 * np.abs(np.asarray(d)))
    radii = jax.tree.map(radius, boxes.parameter_lower, boxes.parameter_upper, delta)
    lower = jax.tree.map(lambda p, r: np.nextafter(np.asarray(p) - r, -np.inf), proposed_params, radii)
    upper = jax.tree.map(lambda p, r: np.nextafter(np.asarray(p) + r, np.inf), proposed_params, radii)
    return Boxes(lower, upper, boxes.input_lower, boxes.input_upper)

def resume_after_qualification(result, boxes, *, admissible):
    """Explicit host authorization boundary; only a domain-exit proposal resumes."""
    if not admissible:
        raise ValueError('requalification did not establish admissibility')
    if int(result.status) != DOMAIN_EXIT:
        raise ValueError('only a finite domain-exit proposal can resume')
    if not bool(inside(result.proposed.params, boxes.parameter_lower, boxes.parameter_upper)):
        raise ValueError('proposal outside replacement certificate')
    return result.proposed
