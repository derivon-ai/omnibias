"""Explicit role bindings for genuinely shared runtime parameter leaves."""
from dataclasses import dataclass
from .bundles import Model, build

@dataclass(frozen=True)
class SharedModel:
    model: Model
    role_bindings: tuple

    def __post_init__(self):
        count = {'ridge': 4, 'separable': 2, 'bottleneck': 6}[self.model.family]
        if len(self.role_bindings) != count or any((type(i) is not int or i < 0 for i in self.role_bindings)):
            raise ValueError('one nonnegative free-parameter index per model role required')
        if set(self.role_bindings) != set(range(max(self.role_bindings) + 1)):
            raise ValueError('free parameter indices must be contiguous and used')

    @property
    def parameter_count(self):
        return max(self.role_bindings) + 1

    def frozen_readout_eligible(self):
        if self.model.family == 'separable':
            return False
        readout_role = 2 if self.model.family == 'ridge' else 4
        readout = self.role_bindings[readout_role]
        return all((index != readout for role, index in enumerate(self.role_bindings) if role != readout_role))

def build_shared(descriptor, operators, provider='parity', **options):
    kernel = build(descriptor.model, operators, provider, **options)

    def function(params, points):
        if len(params) != descriptor.parameter_count:
            raise ValueError('declared free parameter leaves required')
        roles = tuple((params[i] for i in descriptor.role_bindings))
        return kernel(roles, points)
    function.demands = {**kernel.demands, 'role_bindings': list(descriptor.role_bindings), 'parameter_count': descriptor.parameter_count, 'frozen_readout_eligible': descriptor.frozen_readout_eligible()}
    return function
