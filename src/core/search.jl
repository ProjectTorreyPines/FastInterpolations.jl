# ========================================
# Search Policy Types and Implementations
# ========================================
#
# This module provides a typed policy system for interval search dispatch.
# The key insight is that algorithm selection is encoded in type parameters,
# enabling compile-time dispatch and zero-overhead for the default path.
#
# Naming Convention:
#   AbstractSearchPolicy (user-facing) → Searcher (internal executor)
#   Searcher → search_interval → _search_direct (O(1) for Range)
#                              → _search_binary (O(log n) for Vector)
#
# Include order dependency: grid_spacing.jl (for ScalarSpacing) → search.jl

# ========================================
# 1. User-Facing Search Policy Types (Exported)
# ========================================

"""
    AbstractSearchPolicy

Abstract supertype for user-facing search policy selection.
Concrete subtypes encode the search algorithm choice.

See also: [`BinarySearch`](@ref), [`LinearBinarySearch`](@ref)
"""
abstract type AbstractSearchPolicy end

"""
    BinarySearch <: AbstractSearchPolicy

Binary search algorithm. O(log n) per query for vectors, O(1) for ranges.
Stateless and thread-safe. This is the default search policy.

# Hint Behavior
When a `hint` argument is provided with `BinarySearch()`, the search automatically
upgrades to `LinearBinarySearch()` (default window) to utilize the hint for locality. Without a hint,
pure binary search is used.

# Example
```julia
val = linear_interp(x, y, 0.5; search=BinarySearch())  # explicit binary search
val = linear_interp(x, y, 0.5)                    # default: AutoSearch()

# With hint: auto-upgrades to LinearBinarySearch() (default window)
hint = Ref(1)
val = itp(0.5; search=BinarySearch(), hint=hint)  # uses LinearBinarySearch() internally
```

See also: [`LinearBinarySearch`](@ref)
"""
struct BinarySearch <: AbstractSearchPolicy end

"""
    LinearSearch <: AbstractSearchPolicy

Pure linear search for **strictly monotonic query sequences**.
Maximum speed with no binary fallback and minimal bounds checking.

# Safety Contract (User Responsibility)
**Preconditions that MUST be satisfied:**
1. Queries must be monotonically ordered (all increasing OR all decreasing)
2. All queries must satisfy `x[1] <= xi <= x[end]` (within interpolation domain)

**If violated**: Undefined behavior (infinite loop or out-of-bounds access)

# Performance Characteristics
- **Best case**: O(1) per query (consecutive intervals)
- **Amortized**: O(1) for properly monotonic sequences
- **Worst case**: O(n) if sequence is not monotonic

# When to Use
- ODE integration with strictly monotonic time stepping
- Streaming data evaluation with guaranteed ordering
- Performance-critical loops where ALL inputs are controlled

# When NOT to Use
- Random access patterns (use [`BinarySearch`](@ref))
- Queries that might exceed domain bounds (use [`LinearBinarySearch`](@ref))
- Untrusted input data (use [`LinearBinarySearch`](@ref))

# Example
```julia
# ODE-style monotonic evaluation
hint = Ref(1)
for t in t_values  # strictly increasing
    y = itp(t; search=LinearSearch(), hint=hint)
end
```

See also: [`LinearBinarySearch`](@ref), [`BinarySearch`](@ref)
"""
struct LinearSearch <: AbstractSearchPolicy end

"""
    LinearBinarySearch{MAX} <: AbstractSearchPolicy

Bounded linear search within a window of `MAX` positions from hint, then binary fallback.
Optimal for **sorted/monotonic query sequences** where consecutive queries tend to
fall in adjacent or nearby intervals.

# Type Parameter
- `MAX::Int`: Size of the linear search window before falling back to binary search.
  This is a compile-time constant encoded in the type parameter.

# Performance Characteristics
- **Best case**: O(1) when query is within `MAX` positions of the hint
- **Worst case**: O(log n) binary search fallback
- **Ideal for**: Sorted queries, time-series data, streaming evaluations

# Construction
Use the factory function (recommended) to construct with a curated set of values:
```julia
LinearBinarySearch()                   # default MAX=8
LinearBinarySearch(linear_window=0)    # hint check only, no walk
LinearBinarySearch(linear_window=4)    # custom MAX=4
```

Or construct the parametric type directly (advanced):
```julia
LinearBinarySearch{8}()            # explicit type parameter
```

# Example
```julia
sorted_queries = sort(rand(1000))
vals = linear_interp(x, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```

See also: [`BinarySearch`](@ref)
"""
struct LinearBinarySearch{MAX} <: AbstractSearchPolicy end

"""
    LinearBinarySearch(linear_window::Integer)
    LinearBinarySearch(; linear_window::Integer=8)

Factory constructor for `LinearBinarySearch{MAX}` with a **curated set of `linear_window` values**.

# Why Restricted Values?
Julia compiles a specialized method for each unique type parameter `MAX`. Allowing
arbitrary integers (1–1000) would cause **type parameter explosion**: hundreds of
specialized methods, increased compile time, and code cache bloat.

By restricting to powers of 2, we limit specialization to just 7 variants while
covering the practical range of use cases.

# Arguments
- `linear_window::Integer=8`: Size of the linear search window before binary fallback.
  **Allowed values**: `0, 1, 2, 4, 8, 16, 32, 64, 128`

# Throws
- `ArgumentError`: If `linear_window` is not one of the allowed values.

# Example
```julia
policy = LinearBinarySearch()                  # LinearBinarySearch{8}()  (default)
policy = LinearBinarySearch(linear_window=2)   # LinearBinarySearch{2}()
policy = LinearBinarySearch(linear_window=16)  # LinearBinarySearch{16}()
policy = LinearBinarySearch(linear_window=3)   # ERROR: ArgumentError
```

# Choosing `linear_window`
- **Zero (0)**: Hint check only, no walk — minimal random-query overhead.
- **Small values (1–2)**: Minimal overhead for mixed query patterns
- **Medium values (4)**: Good for narrow jitter patterns (step size < 2 intervals)
- **Default (8)**: Best balance — +2.5ns random overhead, covers jitter up to ~6 intervals
- **Large values (16–128)**: For wide jitter, highly localized, or very large datasets
"""
function LinearBinarySearch(linear_window::Integer)
    linear_window == 0  && return LinearBinarySearch{0}()
    linear_window == 1  && return LinearBinarySearch{1}()
    linear_window == 2  && return LinearBinarySearch{2}()
    linear_window == 4  && return LinearBinarySearch{4}()
    linear_window == 8  && return LinearBinarySearch{8}()
    linear_window == 16 && return LinearBinarySearch{16}()
    linear_window == 32 && return LinearBinarySearch{32}()
    linear_window == 64 && return LinearBinarySearch{64}()
    linear_window == 128 && return LinearBinarySearch{128}()
    throw(ArgumentError("`linear_window` must be one of (0, 1, 2, 4, 8, 16, 32, 64, 128), got $linear_window"))
end
LinearBinarySearch(; linear_window::Integer = 8) = LinearBinarySearch(linear_window)

"""
    AutoSearch <: AbstractSearchPolicy

Adaptive search policy that resolves at call time based on query type:
- **Scalar** queries (`Real`, `Tuple{Vararg{Real}}`) → `BinarySearch()` — no hint locality to exploit
- **Vector** queries (`AbstractVector`, `Tuple{Vararg{AbstractVector}}`) → `LinearBinarySearch()` — hint continuity benefits sorted sequences
- **Broadcast** (`itp.(xs)`) → `BinarySearch()` per element — fresh searcher each call

This is the default search policy for all interpolants. For known query patterns,
specify `BinarySearch()` or `LinearBinarySearch()` explicitly for optimal performance.

# Example
```julia
itp = linear_interp(x, y)              # stores AutoSearch()
itp(0.5)                               # → BinarySearch() internally (scalar)
itp([0.1, 0.5, 0.9])                   # → LinearBinarySearch() internally (vector)
itp(0.5; search=LinearBinarySearch())        # override: use LinearBinarySearch explicitly
```

See also: [`BinarySearch`](@ref), [`LinearBinarySearch`](@ref)
"""
struct AutoSearch <: AbstractSearchPolicy end

"""
    DirectSearch <: AbstractSearchPolicy

Internal policy for Range grids where interval lookup is always O(1) direct computation.
Short-circuits the adaptive resolution pipeline to avoid Union type propagation from
`_resolve_search_policy(::AutoSearch, ::Vector, ::Nothing)` → `Union{BinarySearch, LinearBinarySearch{N}}`.

Not exported. Created automatically by the 4-arg `_resolve_search` when the grid is `AbstractRange`.
"""
struct DirectSearch <: AbstractSearchPolicy end

# ----------------------------------------
# Prefix Monotonicity Check (for adaptive AutoSearch)
# ----------------------------------------

"""
    _is_likely_monotone(xq::AbstractVector{<:Real}, ::Val{K}=Val(8)) -> Bool

Check if the first K elements of `xq` are monotonically ordered (ascending or descending).
Used by adaptive AutoSearch to choose between LinearBinarySearch and BinarySearch for vector queries.
Returns `false` for short vectors (length < K) since LinearBinarySearch offers negligible benefit.

False positive rate: 1/K! ≈ 2.5e-5 for K=8 (random data appearing sorted in first K elements).
"""
@inline function _is_likely_monotone(xq::AbstractVector{<:Real}, ::Val{K} = Val(8)) where {K}
    n = length(xq)
    n < K && return false
    # Single-pass direction tracking: state ∈ {-1, 0, 1}.
    # state=0 (undecided) adopts the first non-zero diff's sign.
    # Any diff that opposes state → not monotone.
    # Equal consecutive values (diff=0) are always compatible.
    state = 0
    @inbounds for i in 2:K
        diff = xq[i] - xq[i - 1]
        s = (diff > 0) - (diff < 0)
        if state == 0
            state = s
        else
            diff * state < 0 && return false
        end
    end
    return true
end

# ----------------------------------------
# AutoSearch Resolution (query-type adaptive)
# ----------------------------------------
# Resolves AutoSearch to a concrete policy based on query type.
# For non-AutoSearch policies, passes through unchanged (user override honored).
# Must be called BEFORE _to_searcher in all eval paths.

# 1D scalar: no hint locality → pure binary search
@inline _resolve_search_policy(::AutoSearch, ::Real) = BinarySearch()

# GridIdx: search short-circuits anyway; BinarySearch as placeholder policy
@inline _resolve_search_policy(::AutoSearch, ::GridIdx) = BinarySearch()

# 1D vector: sorted locality → linear window with binary fallback
@inline _resolve_search_policy(::AutoSearch, ::AbstractVector) = LinearBinarySearch()

# ND SoA batch: tuple of vectors → LinearBinarySearch per axis
# NOTE: must precede the bare ::Tuple fallback — Tuple{Vararg{AbstractVector}} <: Tuple,
# so Julia's specificity rules handle ordering correctly, but explicit ordering avoids confusion.
@inline _resolve_search_policy(::AutoSearch, ::Tuple{Vararg{AbstractVector}}) = LinearBinarySearch()

# ND scalar: tuple of reals → BinarySearch per axis
@inline _resolve_search_policy(::AutoSearch, ::Tuple) = BinarySearch()

# Passthrough: explicit policies are honored as-is
@inline _resolve_search_policy(p::AbstractSearchPolicy, _) = p

# Tuple of policies: resolve each element (for ND per-axis storage)
@inline _resolve_search_policy(ps::Tuple{Vararg{AbstractSearchPolicy}}, query) =
    map(p -> _resolve_search_policy(p, query), ps)

# 3-arg form: adaptive vector resolution with hint awareness.
# When hint=nothing + AutoSearch + Real vector, checks prefix monotonicity
# to choose BinarySearch (random) vs LinearBinarySearch (sorted).
# When hint IS provided, the caller already has state tied to a specific search
# strategy, so we skip adaptive resolution and defer to the 2-arg form —
# AutoSearch+vector already resolves to LinearBinarySearch there, which is correct
# because hinted callers expect walk-based locality.
# Explicit policy (non-Auto) + any hint → policy unchanged.
@inline _resolve_search_policy(search, xq, hint) = _resolve_search_policy(search, xq)

# AutoSearch + hint present → caller wants locality, resolve to LB regardless of query type.
@inline _resolve_search_policy(::AutoSearch, xq, ::Base.RefValue{Int}) = LinearBinarySearch()

# AutoSearch + no hint + vector → adaptive per prefix monotonicity check.
@inline function _resolve_search_policy(::AutoSearch, xq::AbstractVector{<:Real}, ::Nothing)
    return _is_likely_monotone(xq) ? LinearBinarySearch() : BinarySearch()
end

# ----------------------------------------
# 4-arg form: grid-aware resolution (Range short-circuit)
# ----------------------------------------
# Range grids always use O(1) _search_direct — no search resolution needed.
# Returning DirectSearch() avoids Union{BinarySearch, LinearBinarySearch{N}} from adaptive
# resolution, eliminating LLVM union-splitting in hot loops.

@inline _resolve_search_policy(::AbstractRange, xq, ::AbstractSearchPolicy, hint) = DirectSearch()
@inline _resolve_search_policy(::AbstractVector, xq, search::AbstractSearchPolicy, hint) = _resolve_search_policy(search, xq, hint)

# ----------------------------------------
# Hint Types (Internal)
# ----------------------------------------

"""
    AbstractHint

Abstract supertype for hint/state management types.
"""
abstract type AbstractHint end

"""
    NoHint <: AbstractHint

No state maintained. Used with `BinarySearch` for stateless search.
"""
struct NoHint <: AbstractHint end

"""
    RefHint <: AbstractHint

Loop-local hint using `Base.RefValue{Int}`.
Thread-safe when each loop creates its own RefHint instance.

# Fields
- `idx::Base.RefValue{Int}`: Mutable reference to last found interval index
"""
struct RefHint <: AbstractHint
    idx::Base.RefValue{Int}
    # Inner constructor for external Ref injection (persistent hint)
    RefHint(ref::Base.RefValue{Int}) = new(ref)
end
RefHint() = RefHint(Ref(1))
RefHint(idx::Int) = RefHint(Ref(idx))

# Future: ThreadHint for per-thread slots
# struct ThreadHint{H} <: AbstractHint
#     hint::H
# end

# ----------------------------------------
# Internal Searcher Type (Policy + State)
# ----------------------------------------

"""
    Searcher{P<:AbstractSearchPolicy, H<:AbstractHint, BC<:AbstractBC}

Internal searcher type combining search policy with hint state and (optional) BC context.
Type parameters enable compile-time dispatch with zero runtime overhead.

# Type Parameters
- `P`: Search policy type (`BinarySearch`, `LinearBinarySearch{N}`)
- `H`: Hint type (`NoHint`, `RefHint`)
- `BC`: Boundary condition type (`NoBC` default — zero-size singleton, compiler-elided;
  `PeriodicBC{:exclusive, ...}` enables seam-cell wrap in `search_interval` for the
  oneshot exclusive periodic path with zero data copy)

# Fields
- `hint::H`: The hint instance (NoHint singleton or RefHint with mutable Ref)
- `bc::BC`: BC context (`NoBC()` for non-periodic paths; `PeriodicBC(...)` for oneshot
  exclusive periodic where query-time seam wrap is required)

# Note
Users should not construct Searcher directly. Use the policy types (`BinarySearch()`,
`LinearBinarySearch()`) with the `search` keyword argument instead. The `bc` field is
populated by `_resolve_search(grid, q, search, hint, bc)` at interpolant construction
or oneshot entry.
"""
struct Searcher{P <: AbstractSearchPolicy, H <: AbstractHint, BC <: AbstractBC}
    hint::H
    bc::BC
end

# Outer constructors: default BC to NoBC so existing 2-type-param `Searcher{P, H}(hint)`
# call sites continue to work unchanged. The BC type param is only specified explicitly
# when a periodic BC context is being threaded (oneshot exclusive periodic path).
@inline Searcher{P, H}(hint::H) where {P <: AbstractSearchPolicy, H <: AbstractHint} =
    Searcher{P, H, NoBC}(hint, NoBC())
@inline Searcher{P, H}(hint::H, bc::BC) where {P <: AbstractSearchPolicy, H <: AbstractHint, BC <: AbstractBC} =
    Searcher{P, H, BC}(hint, bc)

"""
    DEFAULT_SEARCHER

Default internal searcher: stateless binary search with no hint and no BC.
Compiles to identical code as direct `_search_interval` call.
"""
const DEFAULT_SEARCHER = Searcher{BinarySearch, NoHint, NoBC}(NoHint(), NoBC())

# ----------------------------------------
# Policy → Searcher Conversion (Thread-Safe)
# ----------------------------------------

"""
    _to_searcher(policy::AbstractSearchPolicy, hint=nothing, bc=NoBC()) -> Searcher

Convert user-facing policy + optional external hint + optional BC context into an
internal Searcher. Creates a new RefHint for stateful policies, ensuring thread safety.

# Arguments
- `hint=nothing` (default): stateless search, Searcher has `NoHint` type param.
- `hint::Base.RefValue{Int}`: persistent hint write-back (ODE/streaming).
- `bc=NoBC()` (default): non-periodic search, Searcher has `NoBC` type param (zero size,
  compiler-elided — bit-identical codegen to pre-BC-refactor path).
- `bc::PeriodicBC{...}`: threads BC into Searcher so that `search_interval` dispatches
  to the exclusive-periodic seam-wrap path at query time (oneshot entry points only).

Via Julia's default-arg expansion, each policy has 2 source methods (`::Nothing` vs
`Base.RefValue{Int}`) that Julia auto-expands to cover all {1, 2, 3}-arity combinations.
"""
@inline _to_searcher(::BinarySearch, ::Nothing = nothing, bc::AbstractBC = NoBC()) =
    Searcher{BinarySearch, NoHint, typeof(bc)}(NoHint(), bc)
@inline _to_searcher(::BinarySearch, hint::Base.RefValue{Int}, bc::AbstractBC = NoBC()) =
    Searcher{BinarySearch, RefHint, typeof(bc)}(RefHint(hint), bc)

@inline _to_searcher(::LinearSearch, ::Nothing = nothing, bc::AbstractBC = NoBC()) =
    Searcher{LinearSearch, RefHint, typeof(bc)}(RefHint(), bc)
@inline _to_searcher(::LinearSearch, hint::Base.RefValue{Int}, bc::AbstractBC = NoBC()) =
    Searcher{LinearSearch, RefHint, typeof(bc)}(RefHint(hint), bc)

@inline _to_searcher(::LinearBinarySearch{MAX}, ::Nothing = nothing, bc::AbstractBC = NoBC()) where {MAX} =
    Searcher{LinearBinarySearch{MAX}, RefHint, typeof(bc)}(RefHint(), bc)
@inline _to_searcher(::LinearBinarySearch{MAX}, hint::Base.RefValue{Int}, bc::AbstractBC = NoBC()) where {MAX} =
    Searcher{LinearBinarySearch{MAX}, RefHint, typeof(bc)}(RefHint(hint), bc)

# AutoSearch fallbacks: _resolve_search_policy should be called first, but if any
# code path misses resolution, fall back to BinarySearch (safe stateless default).
# Hinted form auto-upgrades to default LinearBinarySearch to exploit locality.
@inline _to_searcher(::AutoSearch, ::Nothing = nothing, bc::AbstractBC = NoBC()) =
    Searcher{BinarySearch, NoHint, typeof(bc)}(NoHint(), bc)
@inline _to_searcher(::AutoSearch, hint::Base.RefValue{Int}, bc::AbstractBC = NoBC()) =
    _to_searcher(LinearBinarySearch(), hint, bc)

# DirectSearch: Range grids only. Carries DirectSearch through to Searcher
# so search_interval dispatches on policy type alone (no grid-type branching).
@inline _to_searcher(::DirectSearch, ::Nothing = nothing, bc::AbstractBC = NoBC()) =
    Searcher{DirectSearch, NoHint, typeof(bc)}(NoHint(), bc)
@inline _to_searcher(::DirectSearch, hint::Base.RefValue{Int}, bc::AbstractBC = NoBC()) =
    Searcher{DirectSearch, RefHint, typeof(bc)}(RefHint(hint), bc)

# ----------------------------------------
# Canonical resolver APIs
# ----------------------------------------
# Resolution pipeline:
#   _resolve_search_policy     — AutoSearch/tuple → concrete policy (BinarySearch, LinearBinarySearch, DirectSearch)
#   _resolve_search            — policy + grid + hint [+ bc] → concrete Searcher (THE entry point)
#   _resolve_searcher_for_grid — pre-built Searcher → grid-adapted Searcher (Range → DirectSearch)
#
# _to_searcher is the policy→Searcher converter used internally.

# _resolve_searcher_for_grid: adapt pre-built Searcher to grid type.
# Converts any Searcher to DirectSearch variant when grid is AbstractRange.
# Preserves the BC type param across adaptation (critical for oneshot periodic path
# where bc carries the period for seam wrap).
# P<:AbstractSearchPolicy bound required to avoid method ambiguity with the catchall.
@inline _resolve_searcher_for_grid(::AbstractRange, s::Searcher{P, NoHint, BC}) where {P <: AbstractSearchPolicy, BC <: AbstractBC} =
    Searcher{DirectSearch, NoHint, BC}(NoHint(), s.bc)
@inline _resolve_searcher_for_grid(::AbstractRange, s::Searcher{P, RefHint, BC}) where {P <: AbstractSearchPolicy, BC <: AbstractBC} =
    Searcher{DirectSearch, RefHint, BC}(s.hint, s.bc)
@inline _resolve_searcher_for_grid(_, s::Searcher) = s

# _resolve_search: single entry point for all eval paths.
# Composes policy resolution + searcher creation + (optional) BC threading.
# `bc=NoBC()` (default) is used by the 99% non-periodic path — bit-identical codegen to
# pre-BC-refactor. `bc::PeriodicBC(...)` is supplied only by oneshot periodic entries.
#
# For `PeriodicBC{:exclusive, Nothing}` the period is resolved from the grid here so
# the stored Searcher always carries a concrete period. Without this, the seam branch
# in `search_interval` (`x[1] + s.bc.period`) reduces to `Float + Nothing` → MethodError.
# Persistent interpolants materialize the period at construction; oneshot entries never
# did before this change.
@inline _resolve_search(grid, q, search, hint, bc::AbstractBC = NoBC()) =
    _to_searcher(_resolve_search_policy(grid, q, search, hint), hint, _resolve_bc_period(grid, bc))

# Resolve an unresolved `:exclusive` period against the grid (e.g. inferred from a Range
# step × length); no-op for every other BC so non-periodic dispatch is unchanged.
@inline _resolve_bc_period(_, bc::AbstractBC) = bc
@inline _resolve_bc_period(grid, bc::PeriodicBC{:exclusive, Nothing}) =
    _with_resolved_period(bc, _resolve_exclusive_period(grid, bc))

# ========================================
# 2. Base Implementations
# ========================================
#
# These are the core implementations, moved from utils.jl.
# Function naming:
#   - _search_direct: O(1) direct calculation for uniform grids (AbstractRange)
#   - _search_binary: O(log n) binary search for non-uniform grids (AbstractVector)
#   - search_interval: public dispatcher

# ----------------------------------------
# Interval index semantics (IMPORTANT)
#
# All interval-search functions return `(idx, xL, xR)` where:
# - `idx ∈ 1:(length(x)-1)`
# - `xL` and `xR` are the left/right interval boundaries.
#
# Exact grid points:
# - If `xi == x[i]` for some `i < length(x)`, then `idx == i` (right interval).
# - If `xi == x[end]`, then `idx == length(x)-1`.
# ----------------------------------------

# ----------------------------------------
# Core implementations (accept any Real, convert to grid type internally)
# ----------------------------------------

"""
    _search_direct(x::AbstractRange{T}, xq::Real) where {T<:AbstractFloat}

O(1) direct index calculation for uniform grids (AbstractRange).
Uses `unsafe_trunc` for ~40% faster index calculation.

Unlike `_search_binary`, this function computes the interval index directly
via arithmetic rather than iterative search, exploiting uniform grid spacing.
"""
@inline function _search_direct(x::AbstractRange{T}, xq::Real) where {T}
    n = length(x)
    x_min = first(x)
    dx = Base.step(x)
    # _extract_primal: index calculation is inherently integer (trunc discards fractional
    # part), so only the primal value matters. Computing the full Dual muladd and then
    # truncating would give the same idx but waste partial arithmetic.
    idx = clamp(unsafe_trunc(Int, _extract_primal((xq - x_min) / dx) + 1), 1, n - 1)
    # xL/xR keep full T (may be Dual) for kernel partial propagation.
    xL = muladd(idx - 1, dx, x_min)
    xR = xL + dx
    return idx, xL, xR
end

"""
    _search_direct(x::_CachedRange{T}, xq::Real)

`_CachedRange` specialization: all fields are plain `T` — no TwicePrecision arithmetic.
Uses precomputed `inv_h` (multiply instead of divide) for the index calculation.
"""
@inline function _search_direct(x::_CachedRange{T}, xq::Real) where {T}
    # Primal-based index: see _search_direct(::AbstractRange, ...) comment.
    idx = clamp(unsafe_trunc(Int, _extract_primal(muladd(xq - x.lo, x.inv_h, 1))), 1, x.len - 1)
    xL = muladd(idx - 1, x.h, x.lo)
    xR = xL + x.h
    return idx, xL, xR
end

"""
    _search_binary(x::AbstractVector{T}, xq::Real) where {T<:Real}

O(log n) binary search for non-uniform grids (AbstractVector).

Uses branchless `for` loop with precomputed iteration count via `leading_zeros`
for predictable loop exit on modern CPUs. The inner comparison uses `ifelse` to
compile to ARM64 `csel` / x86 `cmov` — fully branchless binary search body.
"""
@inline function _search_binary(x::AbstractVector{T}, xq::Real) where {T <: Real}
    n = length(x)
    @inbounds begin
        if xq <= x[1]
            idx = 1
        elseif xq >= x[end]
            idx = n - 1
        else
            lo, hi = 1, n
            # Precompute exact iteration count: ceil(log2(hi - lo)) via CLZ
            # n >= 2 guaranteed (grid has ≥2 points), so hi - lo - 1 >= 0
            # Use %UInt64 (bit reinterpret) to avoid InexactError cold path in ASM
            iters = 64 - leading_zeros((hi - lo - 1) % UInt64)
            for _ in 1:iters
                mid = (lo + hi) >> 1
                cond = x[mid] <= xq
                lo = ifelse(cond, mid, lo)
                hi = ifelse(cond, hi, mid)
            end
            idx = lo
        end
    end
    @inbounds xL, xR = x[idx], x[idx + 1]
    return idx, xL, xR
end

"""
    _search_direct(x::_CachedRange{T}, ::ScalarSpacing{T}, xq::Real)

_CachedRange already has inv_h built in — delegate to 2-arg version.
"""
@inline function _search_direct(
        x::_CachedRange{T}, ::ScalarSpacing{T}, xq::Real
    ) where {T}
    return _search_direct(x, xq)
end

"""
    _search_direct(x::AbstractRange{T}, spacing::ScalarSpacing{T}, xq::Real)

Spacing-aware O(1) direct calculation for ScalarSpacing.
Uses pre-computed `inv_h` for multiplication instead of division.
"""
@inline function _search_direct(
        x::AbstractRange{T}, spacing::ScalarSpacing{T}, xq::Real
    ) where {T}
    n = length(x)
    x_min = first(x)
    idx = clamp(unsafe_trunc(Int, _extract_primal(muladd(xq - x_min, spacing.inv_h, 1))), 1, n - 1)
    xL = muladd(idx - 1, spacing.h, x_min)
    xR = xL + spacing.h
    return idx, xL, xR
end

"""
    _search_binary(x::AbstractVector{T}, ::AbstractGridSpacing{T}, xq::Real)

VectorSpacing delegates to non-spacing version.
"""
@inline function _search_binary(
        x::AbstractVector{T}, ::AbstractGridSpacing{T}, xq::Real
    ) where {T <: Real}
    return _search_binary(x, xq)
end


# ========================================
# 3. Hinted Search Implementations
# ========================================
#
# Core implementations: accept any Real, convert to grid type internally.

# ----------------------------------------
# Core hinted implementations
# ----------------------------------------

"""
    _search_linear!(x, xq, hint_ref) -> (idx, xL, xR)

Pure linear search: walks from hint until interval found.
No bounds checking (except initial clamp), no binary fallback.

# Safety Contract
- Caller guarantees `x[1] <= xq <= x[end]` (within domain)
- Caller guarantees monotonic query progression
- Hint is clamped once at start, then trusted
"""
@inline function _search_linear!(
        x::AbstractVector{T},
        xq::Real,
        hint_ref::Base.RefValue{Int},
    ) where {T <: Real}
    ix = hint_ref[]
    n = length(x)
    @inbounds begin
        # Clamp once at start (handles bad initial hint)
        ix = clamp(ix, 1, n - 1)

        # Direct hit - most common case for monotonic queries
        if x[ix] <= xq < x[ix + 1]
            return ix, x[ix], x[ix + 1]
        end

        # LinearSearch walk - NO bounds check, NO fallback
        if xq < x[ix]
            while x[ix] > xq
                ix -= 1
            end
        else  # xq >= x[ix + 1]
            while x[ix + 1] <= xq
                ix += 1
            end
        end
        hint_ref[] = ix
        return ix, x[ix], x[ix + 1]
    end
end

# --- Walk helpers: @generated manual unroll (avoids LLVM loop peeling bloat) ---
#
# @generated produces flat, loop-free code at compile time. This prevents LLVM's
# loop peeling/splitting that inflates `for _ in 1:N` from 59 to 160 instructions for N=2.
#
# For MAX ≤ 16: fully unrolled flat code (optimal for common window sizes).
# For MAX > 16: while-loop (avoids excessive code size; these are rare explicit opt-in values).
#
# Bounds checks (ix <= 1 / ix >= n-1) are kept per-step rather than factored out to a
# call-site guard. Benchmarking showed that a `ix > MAX ? unchecked() : checked()` ternary
# at the call site prevents LLVM from inlining both paths, costing more than the bounds
# checks save. The per-step checks are nearly free: branch predictor is 99%+ accurate
# when hint is not near grid boundary (the common case).

const _WALK_UNROLL_THRESHOLD = 16

"""
Walk left up to MAX steps. Returns `(ix, found)`.

- MAX ≤ $_WALK_UNROLL_THRESHOLD: flat unrolled code (no loops, forward branches only)
- MAX > $_WALK_UNROLL_THRESHOLD: while-loop (bounded code size for large windows)
"""
@generated function _walk_left(x::AbstractVector, xq, ix::Int, ::Val{MAX}) where {MAX}
    MAX == 0 && return :(return (ix, false))
    return if MAX <= _WALK_UNROLL_THRESHOLD
        # Flat unroll: prevents LLVM loop peeling bloat
        stmts = Expr[]
        for _ in 1:MAX
            push!(
                stmts, quote
                    ix <= 1 && return (ix, false)
                    ix -= 1
                    @inbounds x[ix] <= xq && return (ix, true)
                end
            )
        end
        quote
            $(stmts...)
            return (ix, false)
        end
    else
        # While-loop for large windows (avoids excessive code size)
        quote
            lo = max(1, ix - $MAX)
            @inbounds while ix > lo
                ix -= 1
                x[ix] <= xq && return (ix, true)
            end
            return (ix, false)
        end
    end
end

"""
Walk right up to MAX steps. Returns `(ix, found)`.
Same @generated strategy as `_walk_left`.
"""
@generated function _walk_right(x::AbstractVector, xq, ix::Int, n::Int, ::Val{MAX}) where {MAX}
    MAX == 0 && return :(return (ix, false))
    return if MAX <= _WALK_UNROLL_THRESHOLD
        stmts = Expr[]
        for _ in 1:MAX
            push!(
                stmts, quote
                    ix >= n - 1 && return (ix, false)
                    ix += 1
                    @inbounds xq < x[ix + 1] && return (ix, true)
                end
            )
        end
        quote
            $(stmts...)
            return (ix, false)
        end
    else
        quote
            hi = min(n - 1, ix + $MAX)
            @inbounds while ix < hi
                ix += 1
                xq < x[ix + 1] && return (ix, true)
            end
            return (ix, false)
        end
    end
end

"""
    _search_linear_binary!(x, xq, hint_ref, ::Val{MAX}) -> (idx, xL, xR)

Bounded linear search within MAX-sized window, then binary fallback.
Optimal for monotonic query sequences.

# Optimizations over naive implementation:
- Hint clamped once at start: guards against user-provided out-of-range hints (e.g. Ref(0),
  stale hints from a different grid). Internal hints (initialized to 1, updated to valid idx)
  are already valid, so the clamp is a no-op on the hot path.
- No hint write on direct hit: `ix` is unchanged, skip redundant `hint_ref[] = ix`
- Single comparison per linear step: direction already determines one bound
- @generated walk helpers produce flat, loop-free code to avoid LLVM loop peeling bloat
"""
@inline function _search_linear_binary!(
        x::AbstractVector{T},
        xq::Real,
        hint_ref::Base.RefValue{Int},
        ::Val{MAX},
    ) where {T <: Real, MAX}
    ix = hint_ref[]
    n = length(x)
    ix = clamp(ix, 1, n - 1)  # guard against user-provided bad hints (e.g. Ref(0), stale)
    # Precondition: n >= 2 (enforced by all interpolant constructors)
    @inbounds begin
        # Direct hit — most common for sorted/monotonic queries
        xL = x[ix]
        xR = x[ix + 1]
        xL <= xq < xR && return ix, xL, xR  # no hint write (ix unchanged)

        if xq < xL
            # Walk left: xq < x[ix] guaranteed ⟹ after ix-=1,
            # x[ix+1] = old x[ix] > xq — right bound already satisfied.
            # Only need: x[ix] <= xq  (single comparison per step)
            ix, found = _walk_left(x, xq, ix, Val(MAX))
            found && (hint_ref[] = ix; return ix, x[ix], x[ix + 1])
        else  # xq >= xR
            # Walk right: xq >= x[ix+1] guaranteed ⟹ after ix+=1,
            # x[ix] = old x[ix+1] <= xq — left bound already satisfied.
            # Only need: xq < x[ix+1]  (single comparison per step)
            ix, found = _walk_right(x, xq, ix, n, Val(MAX))
            found && (hint_ref[] = ix; return ix, x[ix], x[ix + 1])
        end
    end
    # BinarySearch fallback — full range (narrowing saves < 1 iteration, not worth extra branches)
    idx, xL, xR = _search_binary(x, xq)
    hint_ref[] = idx
    return idx, xL, xR
end

"""
    _search_direct!(x::AbstractRange, xq, hint_ref) -> (idx, xL, xR)

Mutating variant of `_search_direct`: O(1) arithmetic + hint update.
The hint is not used for computation (Range arithmetic is already O(1)),
but updated for correct state tracking in heterogeneous ND grids.
"""
@inline function _search_direct!(
        x::AbstractRange{T}, xq::Real, hint_ref::Base.RefValue{Int}
    ) where {T}
    idx, xL, xR = _search_direct(x, xq)
    hint_ref[] = idx
    return idx, xL, xR
end

"""
    _search_direct!(x::AbstractRange, spacing, xq, hint_ref) -> (idx, xL, xR)

Spacing-aware mutating variant for ND paths: O(1) arithmetic + hint update.
"""
@inline function _search_direct!(
        x::AbstractRange{T}, spacing::ScalarSpacing{T}, xq::Real, hint_ref::Base.RefValue{Int}
    ) where {T}
    idx, xL, xR = _search_direct(x, spacing, xq)
    hint_ref[] = idx
    return idx, xL, xR
end

# ========================================
# 4. Main Dispatcher (search_interval)
# ========================================
#
# Two-layer dispatch:
#   Layer 1 (search_interval): Separates GridIdx from Real queries.
#            Uses unparameterized Searcher so GridIdx is always more specific than Real —
#            no ambiguity with concrete-policy methods in Layer 2.
#   Layer 2 (_search_interval_real): Policy-specific dispatch for Real queries only.
#            GridIdx never reaches this layer.
#
# Naming Convention:
#   - Tg: Grid element type (from x::AbstractVector{Tg})
#   - Tq: Query type (can be Float, Int, Dual, etc.)
#   - xq: Query point (x query)

# --- Layer 1: GridIdx short-circuit (zero search cost) ---
# GridIdx carries a pre-resolved index: skip search entirely, just return the interval.
# Clamp to valid cell range: grid of N points has N-1 cells (valid idx: 1 to N-1).
# GridIdx(N) at the right boundary maps to cell N-1.
# Dispatch on hint type only — policy is irrelevant for GridIdx.

@inline function _search_grididx(x::AbstractVector, xq::GridIdx)
    n = length(x)
    @boundscheck (n >= 2 && 1 <= xq.idx <= n) || throw(BoundsError(x, xq.idx))
    idx = min(xq.idx, n - 1)
    return idx, @inbounds(x[idx]), @inbounds(x[idx + 1])
end

@inline function _search_grididx!(hint::RefHint, x::AbstractVector, xq::GridIdx)
    n = length(x)
    @boundscheck (n >= 2 && 1 <= xq.idx <= n) || throw(BoundsError(x, xq.idx))
    idx = min(xq.idx, n - 1)
    hint.idx[] = idx
    return idx, @inbounds(x[idx]), @inbounds(x[idx + 1])
end

@inline _search_grididx_dispatch(::NoHint, x::AbstractVector, xq::GridIdx) = _search_grididx(x, xq)
@inline _search_grididx_dispatch(h::RefHint, x::AbstractVector, xq::GridIdx) = _search_grididx!(h, x, xq)

# Layer 1: search_interval entry points (GridIdx vs Real, with BC-aware dispatch for Real).
#
# Return shape: 4-tuple `(idx_L, idx_R, xL, xR)` uniformly across all dispatches.
# - NoBC (default, non-periodic): `idx_R = idx_L + 1`, `xR = x[idx_R]` — pure packaging.
# - PeriodicBC{:inclusive}: same as NoBC (matched endpoints, no seam wrap needed at query).
# - PeriodicBC{:exclusive}: when `xq >= x[end]`, return seam-cell tuple `(n, 1, x[n], x[1]+period)`
#   — the ONE optimized dispatch that wraps `idx_R=1` with period-shifted `xR` without any
#   data copy. Internal `_search_interval_real` / `_search_binary` / `_search_direct` are
#   not touched; the seam branch is a single compare at the dispatcher level.
#
# GridIdx path is BC-agnostic (user-supplied explicit index semantics — no wrap).

# --- GridIdx dispatchers: BC-agnostic, pack 4-tuple ---
@inline function search_interval(s::Searcher, x::AbstractVector, xq::GridIdx)
    idx, xL, xR = _search_grididx_dispatch(s.hint, x, xq)
    return idx, idx + 1, xL, xR
end
@inline function search_interval(s::Searcher, x::AbstractVector, ::AbstractGridSpacing, xq::GridIdx)
    idx, xL, xR = _search_grididx_dispatch(s.hint, x, xq)
    return idx, idx + 1, xL, xR
end

# --- Real + any BC that isn't exclusive: pure 3→4 tuple packaging, no seam wrap.
# Covers NoBC, PeriodicBC{:inclusive}, CubicFit, QuadraticFit, and any future BC
# that doesn't require seam handling. The more-specific `<:PeriodicBC{:exclusive}`
# methods below take precedence via Julia method specificity. ---
@inline function search_interval(s::Searcher{P, H, <:AbstractBC}, x::AbstractVector, xq::Real) where {P, H}
    idx, xL, xR = _search_interval_real(s, x, xq)
    return idx, idx + 1, xL, xR
end
@inline function search_interval(s::Searcher{P, H, <:AbstractBC}, x::AbstractVector, spacing::AbstractGridSpacing, xq::Real) where {P, H}
    idx, xL, xR = _search_interval_real(s, x, spacing, xq)
    return idx, idx + 1, xL, xR
end

# No-op for NoHint, write-back for RefHint. Called on the seam fast path so
# subsequent LinearSearch/LinearBinarySearch queries resume walking from the
# seam cell instead of re-walking from the last interior hint position. For
# NoHint this compiles to nothing — dispatch-only.
@inline _writeback_seam_hint(::NoHint, _::Int) = nothing
@inline _writeback_seam_hint(h::RefHint, idx::Int) = (h.idx[] = idx; nothing)

# --- Real + PeriodicBC{:exclusive}: seam wrap at x[n] → x[1]+period ---
@inline function search_interval(s::Searcher{P, H, <:PeriodicBC{:exclusive}}, x::AbstractVector, xq::Real) where {P, H}
    n = length(x)
    @inbounds if xq >= x[n]
        _writeback_seam_hint(s.hint, n)
        return n, 1, x[n], x[1] + s.bc.period
    end
    idx, xL, xR = _search_interval_real(s, x, xq)
    return idx, idx + 1, xL, xR
end
@inline function search_interval(s::Searcher{P, H, <:PeriodicBC{:exclusive}}, x::AbstractVector, spacing::AbstractGridSpacing, xq::Real) where {P, H}
    n = length(x)
    @inbounds if xq >= x[n]
        _writeback_seam_hint(s.hint, n)
        return n, 1, x[n], x[1] + s.bc.period
    end
    idx, xL, xR = _search_interval_real(s, x, spacing, xq)
    return idx, idx + 1, xL, xR
end

# --- Layer 2: Policy-specific Real dispatch (BC widened via `where {BC}`; body unchanged) ---

# BinarySearch + NoHint (zero-overhead)
@inline _search_interval_real(::Searcher{BinarySearch, NoHint, BC}, x::AbstractVector, xq::Real) where {BC} =
    _search_binary(x, xq)
@inline _search_interval_real(::Searcher{BinarySearch, NoHint, BC}, x::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}, xq::Real) where {Tg, BC} =
    _search_binary(x, spacing, xq)

# BinarySearch + RefHint (pure binary + hint write-back, zero search overhead)
@inline function _search_interval_real(p::Searcher{BinarySearch, RefHint, BC}, x::AbstractVector, xq::Real) where {BC}
    idx, xL, xR = _search_binary(x, xq)
    p.hint.idx[] = idx
    return idx, xL, xR
end
@inline function _search_interval_real(p::Searcher{BinarySearch, RefHint, BC}, x::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}, xq::Real) where {Tg, BC}
    idx, xL, xR = _search_binary(x, spacing, xq)
    p.hint.idx[] = idx
    return idx, xL, xR
end

# LinearSearch + RefHint
@inline _search_interval_real(p::Searcher{LinearSearch, RefHint, BC}, x::AbstractVector, xq::Real) where {BC} =
    _search_linear!(x, xq, p.hint.idx)
@inline _search_interval_real(p::Searcher{LinearSearch, RefHint, BC}, x::AbstractVector, ::AbstractGridSpacing, xq::Real) where {BC} =
    _search_linear!(x, xq, p.hint.idx)

# LinearBinarySearch{MAX} + RefHint
@inline _search_interval_real(p::Searcher{LinearBinarySearch{MAX}, RefHint, BC}, x::AbstractVector, xq::Real) where {MAX, BC} =
    _search_linear_binary!(x, xq, p.hint.idx, Val(MAX))
@inline _search_interval_real(p::Searcher{LinearBinarySearch{MAX}, RefHint, BC}, x::AbstractVector, ::AbstractGridSpacing, xq::Real) where {MAX, BC} =
    _search_linear_binary!(x, xq, p.hint.idx, Val(MAX))

# DirectSearch + NoHint (Range grids, zero-overhead)
@inline _search_interval_real(::Searcher{DirectSearch, NoHint, BC}, x::AbstractRange, xq::Real) where {BC} =
    _search_direct(x, xq)
@inline _search_interval_real(::Searcher{DirectSearch, NoHint, BC}, x::AbstractRange{Tg}, spacing::ScalarSpacing{Tg}, xq::Real) where {Tg, BC} =
    _search_direct(x, spacing, xq)

# DirectSearch + RefHint (Range grids with persistent hint)
@inline _search_interval_real(p::Searcher{DirectSearch, RefHint, BC}, x::AbstractRange, xq::Real) where {BC} =
    _search_direct!(x, xq, p.hint.idx)
@inline _search_interval_real(p::Searcher{DirectSearch, RefHint, BC}, x::AbstractRange{Tg}, spacing::ScalarSpacing{Tg}, xq::Real) where {Tg, BC} =
    _search_direct!(x, spacing, xq, p.hint.idx)

# ========================================
# 5. Internal Aliases (for module-internal use)
# ========================================
# For module-internal use without explicit policy.
# Pure delegation to _search_binary/_search_direct which have generic wrappers.

@inline function _search_interval(x::AbstractVector, xq::Real)
    idx, xL, xR = _search_binary(x, xq)
    return idx, idx + 1, xL, xR
end
@inline function _search_interval(x::AbstractRange, xq::Real)
    idx, xL, xR = _search_direct(x, xq)
    return idx, idx + 1, xL, xR
end
@inline function _search_interval(x::AbstractVector, spacing::AbstractGridSpacing, xq::Real)
    idx, xL, xR = _search_binary(x, spacing, xq)
    return idx, idx + 1, xL, xR
end
@inline function _search_interval(x::AbstractRange, spacing::ScalarSpacing, xq::Real)
    idx, xL, xR = _search_direct(x, spacing, xq)
    return idx, idx + 1, xL, xR
end
