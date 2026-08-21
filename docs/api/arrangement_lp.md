# Arrangement LP with learned facets (03-02)

An inequality LP is one cell of a hyperplane arrangement. This is a
learned-constraint front end on the existing interior-point solver and
the Neumaier-Shcherbina bound -- **not** a new LP algorithm. Status is
**gated**, not shipped. G1–G5 are CI-gated.

Soft membership `prod_i σ(β (b_i - a_i · x))` is **temperature
collapse** (`beta -> inf`, feasibility). It is **not** the founding
bias collapse (`delta -> 0` to `sigma^(K-1)`). Do not conflate the
two. Vertex enumeration is exponential and is a verification tool
below `VERTEX_ENUM_MAX_D = 4` / `VERTEX_ENUM_MAX_N = 16` only. A
float dual without the Neumaier-Shcherbina correction is not a proof;
this module does not offer that path.

Home: `omnibias.convex.arrangement`. No new package.

## API

::: omnibias.convex.arrangement
    options:
      show_root_heading: false
      heading_level: 3
