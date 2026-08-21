# Certified scan localization

A single `sech^2` peak at `tau = -0.3` is certified unique in
`[-0.4, -0.2]`. The physical location is `x = -tau` when `w = 1`.

```python
from omnibias.core.verified.interval import Interval
from omnibias.verify.localization import (
    Inconclusive,
    ScanResponse,
    certify_peak,
    seal,
)

resp = ScanResponse.sech2_peak(-0.3, alpha=5.0)
got = certify_peak(resp, box=Interval(-0.4, -0.2))
assert not isinstance(got, Inconclusive)
assert got.offset_enclosure.contains(-0.3)
assert got.physical_enclosure.contains(0.3)
assert got.scope == "local_box"
assert got.unique_in == Interval(-0.4, -0.2)
sealed = seal(got)
assert sealed["payload"]["formal_tier"] == "sound_enclosure"
```

Two well-separated peaks in one box are `Inconclusive`, not a
false uniqueness claim.

```python
from omnibias.core.verified.interval import Interval
from omnibias.verify.localization import Inconclusive, ScanResponse, certify_peak

two = ScanResponse.two_peaks(-0.3, 0.3, alpha=6.0)
got = certify_peak(two, box=Interval(-0.6, 0.6))
assert isinstance(got, Inconclusive)
```
