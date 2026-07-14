# Design: Query-Shape-Preserving Forward Evaluation

> Status: Implementation-ready proposal
> Scope: design and planning only; no implementation is included here
> Created: 2026-07-14
> Companion plan: `docs/design/PLAN_query_shape_preserving_batch_evaluation.md`

## 1. Executive summary

Every forward batch-evaluation surface should preserve the logical shape carried
by its query container:

| Interpolant | Lifetime | Query | Result |
|---|---|---|---|
| 1-D | persistent | `q::AbstractArray{<:Real}` | dense `Array` with `size(q)` |
| 1-D | one-shot | `q::AbstractArray{<:Real}` | dense `Array` with `size(q)` |
| N-D | persistent | AoS `q::AbstractArray` of N-coordinate points | dense `Array` with `size(q)` |
| N-D | one-shot | AoS `q::AbstractArray` of N-coordinate points | dense `Array` with `size(q)` |
| N-D | either | shaped SoA `q::NTuple{N,AbstractArray}` | dense `Array` with shared axis-array size |

In-place forms accept an `AbstractArray` output with exactly the requested
`size` and fill it directly in logical column-major order. Existing vector
queries remain vectors. `GriddedQuery` remains the distinct Cartesian-product
query and keeps its separable fast paths.

The implementation must not flatten via `vec` or `reshape` in a hot path. Those
operations are zero-copy, but local feasibility probes measured small retained
wrapper allocations. The batch kernels already operate by logical query number;
they can write shaped outputs directly with `output[k]`.

## 2. Terminology and query encodings

### 2.1 Shape

This feature preserves `size`, not the concrete query container type and not
custom axes.

```julia
q = reshape(range(0.1, 0.9, 12), 3, 4)
size(itp(q)) == (3, 4)
itp(q) isa Matrix
```

Allocating calls return a standard dense `Array{Tout}`. They must not use
`similar(q, Tout)`: a query container may be read-only, lazy, GPU-backed, or
specialized for coordinate-point elements and may not be an appropriate output
container for interpolated values.

### 2.2 1-D coordinate array

For a 1-D interpolant, every element of the query array is one scalar coordinate.

```julia
q1 = reshape([0.1, 0.2, 0.3, 0.4], 2, 2)
r1 = itp1(q1)                    # 2×2
@assert r1 == map(itp1, q1)
```

### 2.3 N-D AoS: array of points

For an N-D interpolant, each element of an AoS query array is one N-coordinate
point accepted by the existing `_as_ntuple` protocol: normally an `NTuple`, an
`SVector`, or another indexable N-component point.

```julia
q2 = reshape([(0.1, 0.2), (0.3, 0.4),
              (0.5, 0.6), (0.7, 0.8)], 2, 2)
r2 = itp2(q2)                    # 2×2
@assert r2 == map(itp2, q2)
```

A plain `Matrix{Float64}` is not reinterpreted as “points in rows” or “points in
columns”. Its elements are scalar, so it is not an N-D AoS query for N > 1. This
feature does not introduce a new points-as-columns encoding.

### 2.4 N-D shaped SoA: tuple of coordinate arrays

Coordinate arrays with the same size form a pairwise shaped SoA batch.

```julia
qx = [0.1 0.2; 0.3 0.4]
qy = [0.5 0.6; 0.7 0.8]
r = itp2((qx, qy))               # 2×2, pairwise
@assert r[i, j] == itp2((qx[i, j], qy[i, j]))
```

All SoA coordinate arrays must have identical `size`, not merely identical
`length`. The existing tuple-of-vectors API is the one-dimensional special case
and keeps returning a `Vector`.

### 2.5 `GriddedQuery`: Cartesian product, unchanged

`GriddedQuery((xq, yq))` means every combination of independent axis targets.
It is intentionally different from shaped SoA:

```julia
itp2((qx, qy))                    # pairwise, requires size(qx) == size(qy)
itp2(GriddedQuery(xaxis, yaxis))  # Cartesian product, size = (length(xaxis), length(yaxis))
```

Its explicit `_query_size` method and dedicated separable dispatch stay in place
even though the new generic `AbstractArray` rule would report the same size.

## 3. Public API contract

### 3.1 Allocating calls

The output type is determined exactly as today. Only its dimensions change.

```julia
Tout = _promote_eltype(...)       # existing family-specific rule
out = Array{Tout}(undef, _query_size(queries))
```

Rules:

- scalar query -> scalar result, unchanged;
- vector query -> `Vector`, unchanged;
- shaped array query -> dense `Array` of that size;
- shaped SoA -> dense `Array` of the shared coordinate-array size;
- custom non-array query container -> existing flat default unless it overrides
  `_query_size`;
- `GriddedQuery` -> its current N-D output and fast path.

### 3.2 In-place calls

For shaped query methods:

```julia
size(output) == _query_size(queries) || throw(DimensionMismatch(...))
```

Equal length with a different shape is rejected. The existing vector methods may
retain their current error type only when `_query_size(queries) == (length(queries),)`.
Any public method whose query protocol reports a shaped result must perform the
exact-size check even when `output` itself is a `Vector`. In particular, a
four-element output vector is not a valid sink for a `2×2` query.

### 3.3 Evaluation order

Results align in logical column-major order:

```julia
out[k] == evaluate(query_point_k)
```

This matches the existing `_query_length` / `_query_extract(q, k)` protocol and
`GriddedQuery`'s documented linear order.

### 3.4 Error ordering

Existing safety contracts remain load-bearing:

- `_query_validate` and dimensionality checks happen before evaluation;
- NoExtrap validates the complete batch before the first output write;
- same-size validation happens before domain validation and output mutation;
- Fill/Clamp/Wrap semantics and derivative carrier behavior are unchanged.

## 4. Coverage matrix

### 4.1 Persistent 1-D

Covered through `AbstractInterpolant1D` and the family traits/loops:

- `ConstantInterpolant`
- `LinearInterpolant`
- `QuadraticInterpolant`
- `CubicInterpolant`
- `CubicHermiteInterpolant1D`
- `PchipInterpolant1D`
- `CardinalInterpolant1D`
- `AkimaInterpolant1D`

`DerivativeView` out-of-place forwarding already admits arrays; add both the
1-D `output::AbstractArray, q::AbstractArray{<:Real}` forward and the N-D
parent-specialized `output::AbstractArray, queries` forward so derivative views
follow the parent contract without colliding with the series scalar-query form.
The existing persistent N=1 SoA spelling `(q,)` is also covered: if `q` is a
matrix, the result is a matrix rather than a flattened vector.

### 4.2 One-shot 1-D

Dedicated allocating and in-place functions:

- `constant_interp` / `constant_interp!`
- `linear_interp` / `linear_interp!`
- `quadratic_interp` / `quadratic_interp!`
- `cubic_interp` / `cubic_interp!` (raw-grid and cache entry points)
- `hermite_interp` / `hermite_interp!`
- `pchip_interp` / `pchip_interp!`
- `cardinal_interp` / `cardinal_interp!`
- `akima_interp` / `akima_interp!`
- unified `interp` / `interp!` for the seven routable method objects

The tuple-grid N=1 forwarders must accept a shaped real array and route to the
genuine 1-D method, just as their current vector forwarders do.

### 4.3 Persistent N-D

Covered through `AbstractInterpolantND`'s shared batch protocol:

- `ConstantInterpolantND`
- `LinearInterpolantND`
- `QuadraticInterpolantND`
- `CubicInterpolantND`
- `CubicHermiteInterpolantND`
- `HeteroInterpolantND`, including PCHIP/Cardinal/Akima/mixed axes

### 4.4 One-shot N-D

Both the unified and dedicated names are covered:

- unified `interp` / `interp!`
- `constant_interp` / `constant_interp!`
- `linear_interp` / `linear_interp!`
- `quadratic_interp` / `quadratic_interp!`
- `cubic_interp` / `cubic_interp!`
- `hermite_interp` / `hermite_interp!` with `HermitePartials`
- PCHIP/Cardinal/Akima N-D forwarders in `local_hermite_nd_forward.jl`

## 5. Core protocol design

### 5.1 Query shape

Extend `src/core/query_protocol.jl` conceptually as follows. Names may be adjusted
to fit repository style, but semantics are fixed.

```julia
@inline _query_size(q) = (_query_length(q),)
@inline _query_size(q::AbstractArray) = size(q)

@inline _query_length(q::Tuple{AbstractArray, Vararg{AbstractArray}}) = length(first(q))
@inline _query_size(q::Tuple{AbstractArray, Vararg{AbstractArray}}) = size(first(q))
@inline _query_eltype(q::Tuple{AbstractArray, Vararg{AbstractArray}}) =
    promote_type(map(eltype, q)...)

@inline _query_extract(q::Tuple{Vararg{AbstractArray, N}}, k) where {N} =
    ntuple(d -> @inbounds(q[d][k]), Val(N))
```

Use a non-empty tuple signature for methods that call `first`. Keep the
`GriddedQuery` specializations.

### 5.2 Shaped SoA validation

Replace the SoA length-only validation for the shaped protocol with exact-size
validation:

```julia
function _query_validate(q::Tuple{AbstractArray, Vararg{AbstractArray}})
    expected = size(first(q))
    for d in 2:length(q)
        size(q[d]) == expected || _throw_query_axis_size_mismatch(...)
    end
    nothing
end
```

For vectors this checks the same `(n,)` size and is behaviorally equivalent to
the current length check. `_query_check_ndims` should recognize tuple-of-arrays as
SoA and compare tuple length to interpolation dimension N without extracting the
first point.

### 5.3 Output helpers

Add one source of truth, preferably beside the query protocol:

```julia
@inline _alloc_query_output(::Type{T}, q) where {T} =
    Array{T}(undef, _query_size(q))

@inline function _check_query_output_size(output, q)
    expected = _query_size(q)
    size(output) == expected ||
        _throw_query_output_size_mismatch(expected, size(output))
    nothing
end
```

Do not use `similar(q, T)` and do not flatten output.

### 5.4 Direct shaped output sink

Change batch-loop output annotations from `AbstractVector` to `AbstractArray`
where a shaped call can reach them. The loop remains:

```julia
@inbounds for k in 1:_query_length(queries)
    query_k = _extract_query_point(queries, k, Val(N))
    output[k] = evaluate(query_k)
end
```

Concrete `Vector` calls still specialize on `Vector`; widening an internal sink
annotation does not imply dynamic dispatch. Public vector overloads should remain
in place so current dispatch and documentation are stable.

## 6. 1-D implementation design

### 6.1 Persistent public methods

In `src/core/interpolant_protocol.jl`, add less-specific shaped overloads beside
the existing vector methods:

```julia
(itp::AbstractInterpolant1D)(q::AbstractArray{Tq}; kwargs...) where {Tq<:Real}
(itp::AbstractInterpolant1D)(out::AbstractArray,
                             q::AbstractArray{Tq}; kwargs...) where {Tq<:Real}
```

The allocating method uses `_promote_eltype(itp, Tq)` and
`_alloc_query_output`. The in-place method checks exact size. Both reuse the
existing extrap/search resolution and `_itp_vector_loop!` trait.

The current `AbstractVector` overloads are more specific and must remain
unchanged.

Also widen the existing one-axis tuple forwarders from `Tuple{AbstractVector}`
to the shaped equivalent `Tuple{AbstractArray}` for both allocating and in-place
calls. They unwrap `(q,)` and reuse the native 1-D shaped method; they must not
enter the generic N-D point protocol.

### 6.2 Array-aware domain checks

Add `AbstractArray{<:Real}` peers for the vector batch methods in
`src/core/utils.jl`:

- `_is_all_inbounds`
- `_check_domain(..., ::NoExtrap)`
- unit-step `_CachedRange` exclusive-last promotion
- non-NoExtrap pass-through
- Clamp/Fill/Wrap all-in-bounds promotion
- `_throw_batch_oob`

Keep vector methods more specific. Preserve empty-query behavior and
throw-before-write behavior.

### 6.3 Array-aware AutoSearch

In `src/core/search.jl`, add array peers without replacing the vector methods:

- `_is_likely_monotone(::AbstractArray{<:Real})`
- `_resolve_search_policy(::AutoSearch, ::AbstractArray)`
- `_resolve_search_policy(::AutoSearch, ::AbstractArray{<:Real}, ::Nothing)`

Monotonicity is measured in logical column-major order. An explicit policy and
hint retain current semantics.

### 6.4 Family loop sinks

The following internal loops must accept same-size `AbstractArray` query/output
pairs while retaining their vector specializations:

- `_constant_vector_loop!`
- `_linear_vector_loop!` and `_linear_vector_loop_inner!`
- `_linear_interp_loop!` and `_linear_interp_loop_inner!` (one-shot-specific)
- `_quadratic_vector_loop!` and inner loop
- `_cubic_vector_loop!` and inner loop
- both `_hermite_vector_loop!` variants and inner loops

Avoid higher-order closures in the hot loop unless allocation/codegen tests prove
them harmless. Small duplicated family loops are preferable to a boxed generic
closure.

### 6.5 Dedicated one-shot allocators

Each allocating one-shot method keeps its existing `Tout` formula and changes
only buffer construction from `Vector{Tout}(undef, length(q))` to
`_alloc_query_output(Tout, q)` in its new shaped overload.

Each in-place shaped overload checks exact size, performs the same coefficient/
axis preparation once, and calls the array-aware family loop. It must not build a
persistent interpolant as an adapter: constructors may copy data or allocate
long-lived coefficients and would violate one-shot allocation behavior.

### 6.6 Unified 1-D and N=1 tuple-grid forwards

Widen/add shaped query overloads in `src/hetero/interp_1d.jl`; continue routing
through `_interp1d_route` so unified and dedicated APIs remain allocation-equal.

Add shaped N=1 forwards in:

- `linear/nd/linear_nd_interpolant.jl`
- `constant/nd/constant_nd_interpolant.jl`
- `quadratic/nd/quadratic_nd_interpolant.jl`
- `cubic/nd/cubic_nd_interpolant.jl`
- `hetero/local_hermite_nd_forward.jl`

For a one-axis tuple grid, `q::AbstractArray{<:Real}` goes directly to the native
1-D method. A single-axis SoA wrapper `(q,)` may also forward after unwrapping, so
generic tensor code sees the same shape.

## 7. N-D implementation design

### 7.1 Persistent shared protocol

In `src/core/interpolant_protocol.jl`:

- allocating batch calls use `_alloc_query_output` instead of `Vector`;
- add an `AbstractArray` in-place entry with exact-size validation;
- change the existing `output::AbstractVector, queries` entry to validate
  `_query_size(queries)`, not only `_query_length(queries)`. Otherwise
  `itp(vector_output, matrix_queries)` silently keeps the old flattened path;
- prefer `output::AbstractArray, queries` as the generic shaped entry and keep
  `output::AbstractVector, queries` as its more-specific vector entry, with both
  routed through one shape-checking helper. If an implementation instead
  constrains the generic method's query argument and creates crossed signatures,
  add the narrow `AbstractVector × AbstractArray` intersection explicitly;
- let `_nd_batch_pointwise!` and `_interp_nd_batch!` write directly to
  `AbstractArray` outputs;
- keep existing scalar-vector point dispatch unchanged.

In `src/gridded/gridded_dispatch.jl`, replace the fallback
`_nd_batch_pointwise!(vec(out), ...)` with direct shaped output. Separable methods
and their output checks remain unchanged.

### 7.2 Shaped SoA fast checks

The generic query protocol is sufficient for correctness. For performance parity
with vector SoA, add tuple-of-real-array peers where current code specializes on
tuple-of-real-vectors:

- `_validate_nd_domain` in `query_protocol.jl`
- `_resolve_search_nd` and `_check_mono_nd` in `nd_utils.jl`
- the relevant AutoSearch tuple classification in `search.jl`

These methods apply `minimum`/`maximum` and prefix monotonicity per coordinate
array in its logical order. Existing vector methods stay more specific.

### 7.3 Unified one-shot

`src/hetero/hetero_oneshot.jl` already allocates with `_query_size`. Required
changes:

- validate `size(output) == _query_size(queries)` in the generic public in-place
  path;
- remove `flat = ... vec(output)`;
- let `_interp_nd_oneshot_batch_dispatch!` receive the shaped output directly;
- keep the pre-flatten separable `GriddedQuery` hook exactly where it is.

### 7.4 Dedicated one-shot families

Update allocators and output sinks in:

- `linear/nd/linear_nd_oneshot.jl`
- `constant/nd/constant_nd_oneshot.jl`
- `quadratic/nd/quadratic_nd_oneshot.jl`
- `cubic/nd/cubic_nd_oneshot.jl`
- `hermite/nd/hermite_nd_oneshot.jl`
- `hetero/local_hermite_nd_forward.jl`

Each family must preserve its current output eltype rule, coefficient strategy,
pool usage, BC preparation, query validation, and domain pre-scan. Only output
shape/sink typing changes. Existing public `output::AbstractVector, queries`
methods in these files must also adopt exact `_query_size` validation, so they
cannot accept a shaped query merely because lengths match.

## 8. Dispatch and ambiguity audit

| Intersection | Required winner |
|---|---|
| 1-D scalar `Real` vs shaped array | scalar for `Real`, shaped for `AbstractArray{<:Real}` |
| 1-D vector vs shaped array | existing `AbstractVector` method |
| N-D `AbstractVector{<:Real}` | existing single-point vector-to-tuple method |
| N-D AoS `AbstractVector{<:Tuple/SVector}` | existing vector batch behavior |
| N-D AoS matrix | shaped generic batch |
| N-D vector output + shaped query | exact-size rejection, never length-only fallback |
| shaped SoA tuple | tuple-of-arrays protocol, not generic Tuple scalar logic |
| 1-axis `GriddedQuery` | explicit GriddedQuery disambiguators |
| N>1 `GriddedQuery` | explicit separable/fallback GriddedQuery methods |
| anchored query vectors | existing anchor-specific methods |
| `DerivativeView(out, q_array)` vs series scalar output | shaped query method; scalar `q::Real` remains series path |
| N=1 tuple-grid one-shot vs generic N-D one-shot | explicit N=1 collapse forward |

Every implementation phase must run `Aqua.test_ambiguities`. Do not solve an
ambiguity by deleting a current specific vector/Gridded method; add the narrow
tie-breaker at the abstract protocol level when possible.

## 9. Allocation and performance design

### 9.1 Hard allocation targets

- Persistent in-place dense/view shaped queries: 0 B after warmup.
- One-shot in-place shaped queries: no allocations beyond the corresponding
  vector method after warmup; pool-backed coefficient methods should remain 0 B
  under the repository's existing pool test discipline.
- Allocating shaped queries: exactly the returned dense array plus allocations
  already present in the corresponding vector call.
- Existing vector allocation counts: identical.

### 9.2 Why no `vec`

Feasibility probe on Julia 1.12.6, 10,000 Float64 queries:

| Path | Warm allocation | Indicative time |
|---|---:|---:|
| 1-D direct matrix traversal | 0 B | ~94.9 µs |
| 1-D `vec` input/output adapter | 64 B | ~102.7 µs |
| existing 1-D vector | 0 B | ~96.2 µs |
| N-D direct matrix-of-tuples | 0 B | ~109 µs |
| existing N-D vector-of-tuples | 0 B | ~111 µs |

These are architecture probes, not release benchmarks.

### 9.3 Release performance gate

Add representative persistent and one-shot benchmarks to
`benchmark/ci_benchmark.jl`:

- old Vector AoS and SoA lanes (regression sentinels);
- 1-D dense Matrix and noncontiguous view;
- N-D AoS Matrix and shaped SoA matrices;
- Range and Vector grids;
- sorted and random logical query order;
- Linear plus one coefficient-heavy family (Cubic or PCHIP).

Use the repository's confirmed regression rule: greater than 10% against the
same-machine baseline, followed by its re-verification pass, blocks completion.

## 10. Test design

Create `test/test_query_shape_preservation.jl` and add test items incrementally.

### 10.1 Protocol tests

- `_query_size` for vector, matrix, 3-D array, shaped SoA, custom flat query, and
  `GriddedQuery`.
- `_query_length == prod(_query_size)`.
- `_query_extract` column-major order for AoS and shaped SoA.
- SoA mismatched sizes throw before evaluation.
- empty shaped arrays and zero-sized dimensions.

### 10.2 Four-quadrant correctness

For every quadrant, test allocating and in-place forms:

1. 1-D persistent
2. 1-D one-shot
3. N-D persistent
4. N-D one-shot

Use `map`/comprehension scalar evaluation as the reference. Cover value,
derivative, NoExtrap, Clamp, Fill, Wrap/periodic, mixed precision, and views.

### 10.3 Family coverage

- 1-D: Constant, Linear, Quadratic, Cubic, Hermite, PCHIP, Cardinal, Akima.
- N-D: Constant, Linear, Quadratic, Cubic, user Hermite partials, local-Hermite
  forwarders, and heterogeneous method tuples.
- Unified/dedicated parity for both dimensions.
- N=1 tuple-grid parity with native 1-D APIs.

### 10.4 Dispatch guards

- N-D `Vector{Float64}` of length N is still one scalar point.
- N-D `Vector{NTuple{N,T}}` is still a vector batch.
- Matrix of points returns a matrix.
- Vector output paired with a same-length Matrix query throws before mutation.
- shaped SoA returns shared shape.
- `GriddedQuery` still takes the separable path where supported.
- `DerivativeView` forwards shaped allocating and in-place calls.
- Aqua reports zero ambiguities.

### 10.5 Allocation tests

Put setup, warmup, and `@allocated` inside a function barrier, matching existing
tests. Pin:

- 0 B persistent in-place for 1-D Matrix, N-D AoS Matrix, and N-D shaped SoA;
- allocation parity of unified vs dedicated one-shot;
- unchanged existing vector lanes;
- no `vec`/reshape wrapper allocation on shaped paths.

## 11. Documentation and compatibility

Update:

- 1-D batch API docs and examples;
- `docs/src/nd/overview.md` AoS/SoA/GriddedQuery distinction;
- `docs/src/guides/performance_tips.md` shaped in-place allocation guidance;
- public docstrings that currently say “Returns a Vector”;
- release notes with the intentional result-shape change for array-of-points
  inputs and the migration `vec(itp(q))` for callers that require flattening.

Vector and scalar behavior is backward-compatible. The intentional compatibility
change is limited to batch query containers that already carry more than one
dimension but previously returned a flattened vector.

## 12. Explicit non-goals

- Series interpolants. Their natural layout needs a separate decision about the
  series axis plus query axes.
- Adjoint operator construction/application.
- Pre-anchored query arrays and anchor-buffer shape.
- Vector-calculus APIs whose outputs add component/derivative axes.
- Automatic differentiation rules for newly shaped array queries. Existing
  scalar/vector AD behavior must not regress; array-query AD can be designed
  separately.
- Preserving query container type, custom axes, or GPU execution.
- Interpreting numeric N-D matrices as points-in-columns/rows.

## 13. Handoff order

An implementing agent should work strictly in this order:

1. Add RED protocol + N-D persistent tests.
2. Implement query-size/SoA protocol and direct N-D output sinks.
3. Complete N-D one-shot dedicated/unified surfaces.
4. Add array batch domain/search support and 1-D persistent methods.
5. Complete polynomial and Hermite-family 1-D one-shot surfaces plus N=1
   forwards.
6. Run ambiguity/allocation/performance gates and update docs.

Do not mix the phases: each phase has an independently testable public result and
rollback boundary in the companion plan.
