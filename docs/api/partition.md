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

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a1** · Python **>=3.10** · **3 - Alpha** · AGPL-3.0-or-later **or commercial**

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-partition/src/omnibias/partition). Modules below are relative to `omnibias.partition`; underscored modules are internal.

`arrangement`, `arrangement.jax`, `arrangement.torch`, `certify`, `jax`, `jax.weights`, `keras`, `keras.weights`, `registry`, `torch`, `torch.weights`.

Exports from `omnibias.partition`:

`Arrangement`, `CellGapCertificate`, `PartitionConfig`, `PartitionGapCertificate`, `PartitionParams`, `RegionModels`, `certify_cell_gap`, `certify_partition_gap`, `combine_outputs`, `gate_activations`, `hard_assignment`, `hard_weights`, `hardened_rules`, `init_params`, `max_cells`, `partition_weights`, `region_code_matrix`, `region_rule`, `soft_membership`.

</details>

<!-- END GENERATED API INVENTORY -->
