# omnibias-partition

Soft partitions of unity.

`omnibias.partition` routes coordinates into regions using smooth split gates.

- `PartitionConfig`, `PartitionParams`, `init_params`: construct a partition.
- `partition_weights`: soft region memberships.
- `hard_assignment`, `hardened_rules`: discrete region choices and rules.
- `RegionModels`: combine compatible region models.
- `certify_partition_gap`: a scoped soft-to-hard gap certificate.

PyTorch and JAX adapters live in `omnibias.partition.torch` and `.jax`.
Increasing inverse temperature sharpens gates. Away from ties, the limit gives
a hard partition; at a split boundary the soft gate remains one half. This
limit is distinct from the small-spacing derivative construction.

Install this distribution with `pip install omnibias-partition`; select its
backend extras when needed. See [guarantees](../guarantees.md).
