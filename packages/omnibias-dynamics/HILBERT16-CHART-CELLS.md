# Finite chart-cell ledger

This note records the named cells of the current coalescence atlas. It is
combinatorial bookkeeping. A complete list of labels is not G1, not G4,
and not Hilbert XVI. Verdict collapse applies only to exact identities.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## Cells

| Cell | Regime | First-hit | Remainder order | G1 item |
| --- | --- | --- | --- | --- |
| N | `Delta >= chi_N > 0` | selected tube | 2 | open |
| F | `sep >= sep0`, `L >= Lmin` | selected tube | 2 | open |
| D | `epsilon \|log sep\| <= 1` | selected tube | 1 | open |
| C | `sep = 0` | incomplete | 0 | open |
| O | `L -> 0` | incoming only | 0 | open |
| WF | W-fold, `sigma = sqrt(epsilon)` | selected tube | 0 | open |
| WS | W-separation, `chi = O(1)` | selected tube | 0 | open |
| LI | `tau = epsilon log(1/sep)` | selected tube | 0 | open |
| WL | `Lambda = epsilon log(1/W)` | selected tube | 0 | open |
| kill_super_small_sep | `sep = exp(-1/epsilon^2)` | admitted incoming and chi tube | 0 | open |
| kill_shrinking_root | `L = 1/n`, `lambda1 = -2` | incoming only | 0 | open |

Identity verdicts on the double-root, W-ratio, shrinking-root, blow-up
height, leading event, and joint-axis algebra are `PROVED` as finite
rational residuals. Physical C2 remainders stay `BLOCKED`. Charts LI and
WL record the log-intermediate attempt of
[the next-atlas note](HILBERT16-NEXT-ATLAS.md); they do not close the
scale dichotomy. The super-small sequence remains admitted on the
selected incoming connector. `g1_passed` is true only if every G1 item
is written-closed. It is false on this list.

The parallel G5 slice replays one Maletto `(n, W, T)` quartic. Dyck and
Bézout checks are exact; complex smoothness is attached by the existing
plane-curve witness in the two-blow-up benchmark. A complete real scheme
stays `BLOCKED`. That replay cannot pass G1 or the octic target.

## Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_two_blowup.py \
  packages/omnibias-dynamics/tests/test_next_atlas.py -q
```
