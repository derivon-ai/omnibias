# Finite discovery engine

The prove/disprove machine still adjudicates. The discovery engine is the
**proposer** layer over a named finite family: statement → catalog → family →
proposer → exact checker → box-scoped characterization.

A hit certifies the **finite obligation** in that family. It is a witness
in the box you named, not a proof of the parent. A miss on an incomplete
family is `BLOCKED` (`search_incomplete`). An exhaustive miss on a complete
existential family is also `BLOCKED` — do not promote it to “the parent is
true.” A universal statement that finds a counterexample is `DISPROVED`.
A universal statement that exhausts a complete family with no
counterexample is `PROVED` of **that finite universal**, still not the
parent.

Uniqueness is span/box-scoped. First-hit `PROVED` does not imply
`unique_in_family`. Use `collect=True` to recover a solution set.

The CI default proposer is a score-guided discrete walk. `coordinate_newton`
is a finite-difference step on integer coordinates, not `CubicNewton`.
`onehot_anneal` is a 1-flip anneal on the discrete box; it does not import
`omnibias-qubo`. Optional `AnnealDescentProposer` lives in
`omnibias-discrete` and is never a core `get_proposer` name.

```python
from omnibias.core.proof import IntegerIntervalFamily, run_discovery

family = IntegerIntervalFamily(lo=-3, hi=3, target_square=4)
result = run_discovery(family.statement, family, "score_guided", budget=8)
assert result.status == "PROVED"
assert result.candidate in (-2, 2)
assert result.check.payload["honesty"]["jacobian_conjecture_proof_claim"] is False
assert result.characterization.unique_in_family is False
```

A zero budget never proposes, so an incomplete search cannot forge a parent
or an empty-family characterization:

```python
miss = run_discovery(family.statement, family, "score_guided", budget=0)
assert miss.status == "BLOCKED"
assert miss.evaluated == 0
assert miss.search_incomplete is True
```

Collect both squares in the complete interval:

```python
both = run_discovery(family.statement, family, "score_guided", budget=16, collect=True)
assert both.status == "PROVED"
assert set(both.solutions) == {-2, 2}
assert both.characterization.exhausted is True
assert both.characterization.unique_in_family is False
```

A complete box with no witness stays `BLOCKED` for an existential
statement. The same box proves a finite universal:

```python
from omnibias.core.proof import Statement

empty = IntegerIntervalFamily(lo=-1, hi=1, target_square=4)
exist = run_discovery(empty.statement, empty, "score_guided", budget=8)
assert exist.status == "BLOCKED"
assert exist.detail == "no witness in enumerated family"

empty.statement = Statement(
    name="no_square_in_box",
    obligation="every integer x in the interval has x^2 not equal to the target",
    parent="toy",
    parent_status="already_true",
    existential=False,
)
univ = run_discovery(empty.statement, empty, "score_guided", budget=8)
assert univ.status == "PROVED"
assert univ.detail == "no counterexample in complete family"
```

Register a factory on the catalog. `list_catalog` only shows packages that
have already been imported. `discover(kind)` raises if the owning package
was not imported:

```python
from omnibias.core.proof import CatalogEntry, discover, list_catalog, register_catalog

register_catalog(
    CatalogEntry(
        kind="integer_square_docs",
        obligation="some integer x with x^2 equal to the target",
        parent="toy",
        parent_status="already_true",
        package="omnibias.core.proof",
        mode="exact_search",
        complete=True,
    ),
    lambda **kwargs: run_discovery(
        IntegerIntervalFamily(lo=-3, hi=3, target_square=4).statement,
        IntegerIntervalFamily(lo=-3, hi=3, target_square=4),
        "score_guided",
        budget=8,
    ),
)
assert "integer_square_docs" in {entry.kind for entry in list_catalog()}
found = discover("integer_square_docs")
assert found.status == "PROVED"
```

Catalog modes:

- `exact_search` — `run_discovery` with an exact checker
- `exact_replay` — an existing `verify_*` (never a planted fake search)
- `enclosure` — CAP / SOS / gauge / certified PINN payloads
- `empirical` — residual / SINDy loops; never an `ExactCheck` forged from a
  float residual

Exact families already on the loop include Keller deg-2/3, DGG H* / DAG-le6,
a blind `K_5` colouring search, recurrence / tanh-identity / planted-heat
spans, and prefix-verified holonomic guesses. Docs may say the engine found
a witness **in this family**. They must not say omnibias refuted Keller,
Goemans, or the Jacobian conjecture.

## Condition language

The inverted loop searches **conditions**, not colourings or maps. A
`ConditionHypothesis` is a typed point of a finite `GrammarSpec`. Typed
constructors grow the grammar. The accept gate is exact `Q`
(`residual_identically_zero`). Soft SINDy / PINN residuals stay
`empirical` until they snap. An incomplete grammar miss is
`search_incomplete`. It is never “no condition exists.”

```python
from fractions import Fraction

from omnibias.core.proof import (
    ConditionHypothesis,
    ConditionToken,
    GrammarSpec,
    IntegerIntervalFamily,
    KindMetaFamily,
    apply_constructor,
    emit_condition,
    residual_identically_zero,
    run_discovery,
)

seed = ConditionHypothesis(
    sort="jet_monomial",
    tokens=(
        ConditionToken("jet_monomial", "y"),
        ConditionToken("jet_monomial", "yp"),
    ),
)
named = emit_condition(
    seed, parent="Riccati identities", parent_status="already_true"
)
assert named.name == "condition_jet_monomial"

grammar = GrammarSpec(
    sorts=("jet_monomial",),
    tokens_by_sort={"jet_monomial": ("y", "yp")},
    constructors=("compose_jets",),
    max_growth_depth=1,
    complete=False,
)
grown = apply_constructor(seed, "compose_jets", grammar, left="y", right="yp")
assert grown is not None
assert "y*yp" in grown.token_names()
assert apply_constructor(grown, "compose_jets", grammar, left="y", right="y") is None

assert residual_identically_zero([[1, 1]], [1, -1], [0]) is True
assert residual_identically_zero([[1, 1]], [1, -1], [Fraction(1, 10)]) is False

empty = IntegerIntervalFamily(lo=-1, hi=1, target_square=4)
complete_meta = KindMetaFamily(
    sorts=("jet_monomial",),
    families={"jet_monomial": empty},
    grammar_complete=True,
)
empty_grammar = run_discovery(
    complete_meta.statement, complete_meta, "score_guided", budget=8
)
assert empty_grammar.status == "BLOCKED"
assert empty_grammar.detail == "no witness in enumerated grammar"
assert empty_grammar.search_incomplete is False

open_meta = KindMetaFamily(
    sorts=("jet_monomial", "ore"),
    families={"jet_monomial": empty},
    grammar_complete=True,
)
assert open_meta.complete is False
open_miss = run_discovery(open_meta.statement, open_meta, "score_guided", budget=4)
assert open_miss.status == "BLOCKED"
assert open_miss.search_incomplete is True
```

Package sorts (`jet_monomial`, `ore`, `pde_operator`, `sos_template`,
`forbidden_minor`, `conservation`, `fractional_order`, `piecewise_hybrid`,
`dfinite`, `sos_onset`, `edge_colouring`, `extremal_template`,
`residual_sign`) register on import. Docs may say the engine found a
condition **in this grammar**. They must not say no unnamed condition
exists, and they must not promote a miss to a parent theorem.

## Shared observation and class selection

`select_class` is the front door when the input is one dataset, not a
human-picked family. Binders that do not apply return `None` and force
an incomplete grammar. Ranking is a total order on *certified* hits:
exact snap / finite predicate, then enclosure, then empirical. Soft
RMSE never outranks an exact identity. The driver never marks the
grammar complete.

```python
from omnibias.core.proof import Observation, bind_sorts, rank_class_hits, select_class
from omnibias.symbolic.conditions import observation_tanh

obs = observation_tanh()
assert obs.features()[1] == 1
bound = bind_sorts(obs, ("jet_monomial", "pde_operator"))
assert "jet_monomial" in bound
assert "pde_operator" not in bound

selection = select_class(obs, sorts=("jet_monomial", "pde_operator"))
assert selection.best is not None
assert selection.best.sort == "jet_monomial"
assert selection.best.tier == "exact"
assert selection.grammar_complete is False
assert selection.honesty["no_condition_exists_claim"] is False
assert selection.honesty["unnamed_condition_complete_claim"] is False
```

A miss on an incomplete observation is `search_incomplete`. It is never
“no condition exists.”

```python
empty = Observation(tag="unknown")
miss = select_class(empty, sorts=("jet_monomial",))
assert miss.best is None
assert miss.search_incomplete is True
assert miss.honesty["no_condition_exists_claim"] is False
```

`ClassMemory` / `FrequencyGate` only propose a sort order. They never
write `ExactCheck`. Among two exact hits, fewer tokens win:

```python
from omnibias.core.proof import ClassHit, ExactCheck, rank_class_hits

fat = ClassHit(
    sort="ore",
    check=ExactCheck(ok=True, payload={"tier": "exact"}),
    tier="exact",
    tokens=5,
    degree=0,
)
thin = ClassHit(
    sort="jet_monomial",
    check=ExactCheck(ok=True, payload={"tier": "exact"}),
    tier="exact",
    tokens=2,
    degree=4,
)
assert rank_class_hits([fat, thin])[0].sort == "jet_monomial"
```

Pack a table without a tag and grow a product on miss. The stack
loader imports owning packages; core never does.

```python
from omnibias.symbolic.classloop import discover_observation, load_discovery_stack
from omnibias.symbolic.ingest import pack_jets

assert "jet_monomial" in load_discovery_stack()
square = pack_jets(
    {"y": (1, 4, 9, 16), "yp": (2, 4, 6, 8), "ypp": (2, 2, 2, 2)},
    sample_x=(1, 2, 3, 4),
)
assert square.tag == ""
payload = discover_observation(square, sorts=("jet_monomial", "fractional_order"))
assert payload["no_condition_exists_claim"] is False
assert payload["grammar_complete"] is False
assert payload["best"] in ("jet_monomial", "fractional_order")
```
