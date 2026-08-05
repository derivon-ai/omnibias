# Spectral shape analysis

<!-- docs-test: file-skip reason="excerpts from notebook 21, which supplies cot_laplacian and the triangulated charts" -->

The spectrum of the Laplace-Beltrami operator is an **intrinsic** shape
fingerprint ("Shape-DNA"): invariant to rotation, translation, and isometric
bending. The continuous operator the discrete LBO approximates is exactly
`omnibias.geometry`'s `laplace_beltrami`. Full walkthrough:
[`21_spectral_shape_analysis.ipynb`](https://github.com/derivon-ai/omnibias/blob/main/notebooks/21_spectral_shape_analysis.ipynb).

## Surfaces from charts + a discrete LBO

Triangulate a parametric chart `phi(u, v)` and assemble the standard cotangent
stiffness `L` and lumped mass `M` (vertex areas — the discrete \(\sqrt{|g|}\)).
The first eigenvalues approach the analytic sphere spectrum \(l(l+1) = 0, 2, 6,
12, \dots\) with multiplicities \(2l+1\):

```python
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

L, mass = cot_laplacian(V, F)            # V, F from the chart triangulation
vals, vecs = eigsh(sp.csr_matrix(L), k=16, M=sp.diags(mass), sigma=1e-8, which="LM")
# sphere -> vals ~ [0, 2, 2, 2, 6, 6, 6, 6, 6, ...]
```

## Intrinsic classification

Because the spectrum is pose-invariant, a randomly rotated shape is classified
correctly by nearest Shape-DNA distance (drop the trivial zero eigenvalue):

```python
dist = {name: np.linalg.norm(query_spectrum[1:] - ref[name][1:]) for name in classes}
pred = min(dist, key=dist.get)
```

## Spectral segmentation via the Heat-Kernel Signature

The HKS \(k_t(x) = \sum_i e^{-\lambda_i t}\,\phi_i(x)^2\) is a multi-scale,
intrinsic per-vertex descriptor; clustering vertices by HKS partitions the surface
into intrinsically-similar regions:

```python
lam, phi2 = vals[1:], vecs[:, 1:] ** 2
hks = phi2 @ np.exp(-np.outer(lam, ts))          # (n_vertices, n_scales)
_, labels = kmeans2(standardize(hks), 4, minit="++")
```

## Takeaway

Shape-DNA, HKS, intrinsic classification and segmentation all derive from the LBO
spectrum — the discrete face of omnibias's exact `laplace_beltrami`. With the
pullback metric the same pipeline runs directly off a (possibly learned) chart.
