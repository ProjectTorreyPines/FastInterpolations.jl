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

See also: [`Binary`](@ref), [`HintedBinary`](@ref), [`LinearBinary`](@ref)
"""
abstract type AbstractSearchPolicy end

"""
    Binary <: AbstractSearchPolicy

Binary search algorithm. O(log n) per query for vectors, O(1) for ranges.
Stateless and thread-safe. This is the default search policy.

# Hint Behavior
When a `hint` argument is provided with `Binary()`, the search automatically
upgrades to `HintedBinary` behavior to utilize the hint. Without a hint,
pure binary search is used.

# Example
```julia
val = linear_interp(x, y, 0.5; search=Binary())  # pure binary search
val = linear_interp(x, y, 0.5)                    # same (default)

# With hint: auto-upgrades to HintedBinary behavior
hint = Ref(1)
val = itp(0.5; search=Binary(), hint=hint)  # uses hinted binary internally
```

See also: [`HintedBinary`](@ref), [`LinearBinary`](@ref)
"""
struct Binary <: AbstractSearchPolicy end

"""
    HintedBinary <: AbstractSearchPolicy

Hinted binary search: O(1) if query hits cached interval, O(log n) fallback.
Useful when queries repeatedly access the same interval region.

# Example
```julia
itp = linear_interp(x, y)
vals = itp(query_points; search=HintedBinary())
```
"""
struct HintedBinary <: AbstractSearchPolicy end

"""
    Linear <: AbstractSearchPolicy

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
- Random access patterns (use [`Binary`](@ref))
- Queries that might exceed domain bounds (use [`LinearBinary`](@ref))
- Untrusted input data (use [`LinearBinary`](@ref))

# Example
```julia
# ODE-style monotonic evaluation
hint = Ref(1)
for t in t_values  # strictly increasing
    y = itp(t; search=Linear(), hint=hint)
end
```

See also: [`LinearBinary`](@ref), [`Binary`](@ref), [`HintedBinary`](@ref)
"""
struct Linear <: AbstractSearchPolicy end

"""
    LinearBinary{MAX} <: AbstractSearchPolicy

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
LinearBinary()               # default MAX=8
LinearBinary(linear_window=4)    # custom MAX=4
```

Or construct the parametric type directly (advanced):
```julia
LinearBinary{8}()            # explicit type parameter
```

# Example
```julia
sorted_queries = sort(rand(1000))
vals = linear_interp(x, y, sorted_queries; search=LinearBinary(linear_window=8))
```

See also: [`Binary`](@ref), [`HintedBinary`](@ref)
"""
struct LinearBinary{MAX} <: AbstractSearchPolicy end

"""
    LinearBinary(linear_window::Integer)
    LinearBinary(; linear_window::Integer=8)

Factory constructor for `LinearBinary{MAX}` with a **curated set of `linear_window` values**.

# Why Restricted Values?
Julia compiles a specialized method for each unique type parameter `MAX`. Allowing
arbitrary integers (1–1000) would cause **type parameter explosion**: hundreds of
specialized methods, increased compile time, and code cache bloat.

By restricting to powers of 2, we limit specialization to just 7 variants while
covering the practical range of use cases.

# Arguments
- `linear_window::Integer=8`: Size of the linear search window before binary fallback.
  **Allowed values**: `1, 2, 4, 8, 16, 32, 64, 128` (powers of 2 from 2⁰ to 2⁷)

# Throws
- `ArgumentError`: If `linear_window` is not one of the allowed values.

# Example
```julia
policy = LinearBinary()              # LinearBinary{8}()  (default)
policy = LinearBinary(linear_window=4)   # LinearBinary{4}()
policy = LinearBinary(linear_window=16)  # LinearBinary{16}()
policy = LinearBinary(linear_window=3)   # ERROR: ArgumentError
```

# Choosing `linear_window`
- **Small values (1–4)**: Lower overhead, but more frequent binary fallbacks
- **Medium values (8–16)**: Good balance for typical sorted query patterns
- **Large values (32–128)**: For highly localized queries or very large datasets
"""
function LinearBinary(linear_window::Integer)
    linear_window == 1  && return LinearBinary{1}()
    linear_window == 2  && return LinearBinary{2}()
    linear_window == 4  && return LinearBinary{4}()
    linear_window == 8  && return LinearBinary{8}()
    linear_window == 16 && return LinearBinary{16}()
    linear_window == 32 && return LinearBinary{32}()
    linear_window == 64 && return LinearBinary{64}()
    linear_window == 128 && return LinearBinary{128}()
    throw(ArgumentError("`linear_window` must be one of (1, 2, 4, 8, 16, 32, 64, 128), got $linear_window"))
end
LinearBinary(; linear_window::Integer=8) = LinearBinary(linear_window)

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

No state maintained. Used with `Binary` for stateless search.
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
    Searcher{P<:AbstractSearchPolicy, H<:AbstractHint}

Internal searcher type combining search policy with hint state.
Type parameters enable compile-time dispatch with zero runtime overhead.

# Type Parameters
- `P`: Search policy type (`Binary`, `HintedBinary`, `LinearBinary{N}`)
- `H`: Hint type (`NoHint`, `RefHint`)

# Fields
- `hint::H`: The hint instance (NoHint singleton or RefHint with mutable Ref)

# Note
Users should not construct Searcher directly. Use the policy types (`Binary()`,
`HintedBinary()`, `LinearBinary()`) with the `search` keyword argument instead.
"""
struct Searcher{P<:AbstractSearchPolicy,H<:AbstractHint}
    hint::H
end

"""
    DEFAULT_SEARCHER

Default internal searcher: stateless binary search with no hint.
Compiles to identical code as direct `_search_interval` call.
"""
const DEFAULT_SEARCHER = Searcher{Binary,NoHint}(NoHint())

# ----------------------------------------
# Policy → Searcher Conversion (Thread-Safe)
# ----------------------------------------

"""
    _to_searcher(policy::AbstractSearchPolicy) -> Searcher

Convert user-facing policy to internal Searcher with fresh hint state.
Creates a new RefHint for stateful policies, ensuring thread safety.
"""
@inline _to_searcher(::Binary) = Searcher{Binary,NoHint}(NoHint())
@inline _to_searcher(::HintedBinary) = Searcher{HintedBinary,RefHint}(RefHint())
@inline _to_searcher(::Linear) = Searcher{Linear,RefHint}(RefHint())
@inline _to_searcher(::LinearBinary{MAX}) where {MAX} = Searcher{LinearBinary{MAX},RefHint}(RefHint())

# ----------------------------------------
# 2-arg overloads: Policy + External Hint
# ----------------------------------------
# Enables persistent hint across scalar calls (ODE/streaming pattern).
# When hint=nothing, behaves identically to 1-arg version.
# When hint=Ref{Int}, stateful policies use the external Ref for persistence.

@inline _to_searcher(::Binary, ::Nothing) = Searcher{Binary,NoHint}(NoHint())
@inline _to_searcher(::Binary, hint::Base.RefValue{Int}) = Searcher{HintedBinary,RefHint}(RefHint(hint))  # auto-upgrade to hinted
@inline _to_searcher(::HintedBinary, ::Nothing) = Searcher{HintedBinary,RefHint}(RefHint())
@inline _to_searcher(::HintedBinary, hint::Base.RefValue{Int}) = Searcher{HintedBinary,RefHint}(RefHint(hint))
@inline _to_searcher(::Linear, ::Nothing) = Searcher{Linear,RefHint}(RefHint())
@inline _to_searcher(::Linear, hint::Base.RefValue{Int}) = Searcher{Linear,RefHint}(RefHint(hint))
@inline _to_searcher(::LinearBinary{MAX}, ::Nothing) where {MAX} = Searcher{LinearBinary{MAX},RefHint}(RefHint())
@inline _to_searcher(::LinearBinary{MAX}, hint::Base.RefValue{Int}) where {MAX} = Searcher{LinearBinary{MAX},RefHint}(RefHint(hint))

# ----------------------------------------
# Searcher passthrough (advanced usage)
# ----------------------------------------
# Allows direct Searcher injection for zero _to_searcher overhead in tight loops.
# Users can pre-construct a Searcher and reuse it across calls.

@inline _to_searcher(s::Searcher) = s
@inline _to_searcher(s::Searcher, ::Nothing) = s
@inline _to_searcher(s::Searcher, ::Base.RefValue{Int}) = s  # already configured, hint ignored

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
# Core implementations (type-matched, optimized)
# ----------------------------------------

"""
    _search_direct(x::AbstractRange{T}, xq::T) where {T<:AbstractFloat}

O(1) direct index calculation for uniform grids (AbstractRange).
Uses `unsafe_trunc` for ~40% faster index calculation.

Unlike `_search_binary`, this function computes the interval index directly
via arithmetic rather than iterative search, exploiting uniform grid spacing.
"""
@inline function _search_direct(x::AbstractRange{T}, xq::T) where {T<:AbstractFloat}
    n = length(x)
    x_min = first(x)
    dx = Base.step(x)
    idx = clamp(unsafe_trunc(Int, (xq - x_min) / dx + 1), 1, n - 1)
    xL = muladd(idx - 1, dx, x_min)
    xR = xL + dx
    return idx, xL, xR
end

"""
    _search_binary(x::AbstractVector{T}, xq::T) where {T<:Real}

O(log n) binary search for non-uniform grids (AbstractVector).
"""
@inline function _search_binary(x::AbstractVector{T}, xq::T) where {T<:Real}
    n = length(x)
    @inbounds begin
        if xq <= x[1]
            idx = 1
        elseif xq >= x[end]
            idx = n - 1
        else
            lo, hi = 1, n
            while hi - lo > 1
                mid = (lo + hi) >> 1
                if x[mid] <= xq
                    lo = mid
                else
                    hi = mid
                end
            end
            idx = lo
        end
    end
    @inbounds xL, xR = x[idx], x[idx + 1]
    return idx, xL, xR
end

"""
    _search_direct(x::AbstractRange{T}, spacing::ScalarSpacing{T}, xq::T)

Spacing-aware O(1) direct calculation for ScalarSpacing.
Uses pre-computed `inv_h` for multiplication instead of division.
"""
@inline function _search_direct(
    x::AbstractRange{T}, spacing::ScalarSpacing{T}, xq::T
) where {T<:AbstractFloat}
    n = length(x)
    x_min = first(x)
    idx = clamp(unsafe_trunc(Int, (xq - x_min) * spacing.inv_h + 1), 1, n - 1)
    xL = muladd(idx - 1, spacing.h, x_min)
    xR = xL + spacing.h
    return idx, xL, xR
end

"""
    _search_binary(x::AbstractVector{T}, ::AbstractGridSpacing{T}, xq::T)

VectorSpacing delegates to non-spacing version.
"""
@inline function _search_binary(
    x::AbstractVector{T}, ::AbstractGridSpacing{T}, xq::T
) where {T<:Real}
    return _search_binary(x, xq)
end

# ----------------------------------------
# Generic wrappers (type-mismatched → convert → optimized)
# ----------------------------------------
#
# These handle cases where query type (Tq) differs from grid type (Tg).
# They extract primal value (for AD support), convert to grid type,
# then dispatch to the optimized type-matched versions above.
#
# Julia's multiple dispatch automatically selects:
#   - Tq == Tg → optimized version directly (no conversion overhead)
#   - Tq != Tg → generic wrapper → convert → optimized version

"""
    _search_direct(x::AbstractRange{Tg}, xq::Tq) where {Tg<:AbstractFloat, Tq<:Real}

Generic wrapper: converts query to grid type, then calls optimized version.
"""
@inline function _search_direct(x::AbstractRange{Tg}, xq::Tq) where {Tg<:AbstractFloat, Tq<:Real}
    return _search_direct(x, _to_grid_type(xq, Tg))
end

"""
    _search_binary(x::AbstractVector{Tg}, xq::Tq) where {Tg<:Real, Tq<:Real}

Generic wrapper: converts query to grid type, then calls optimized version.
"""
@inline function _search_binary(x::AbstractVector{Tg}, xq::Tq) where {Tg<:Real, Tq<:Real}
    return _search_binary(x, _to_grid_type(xq, Tg))
end

"""
    _search_direct(x::AbstractRange{Tg}, spacing::ScalarSpacing{Tg}, xq::Tq)

Generic wrapper with spacing: converts query to grid type, then calls optimized version.
"""
@inline function _search_direct(x::AbstractRange{Tg}, spacing::ScalarSpacing{Tg}, xq::Tq) where {Tg<:AbstractFloat, Tq<:Real}
    return _search_direct(x, spacing, _to_grid_type(xq, Tg))
end

"""
    _search_binary(x::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}, xq::Tq)

Generic wrapper with spacing: converts query to grid type, then calls optimized version.
"""
@inline function _search_binary(x::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}, xq::Tq) where {Tg<:Real, Tq<:Real}
    return _search_binary(x, spacing, _to_grid_type(xq, Tg))
end

# ========================================
# 3. Hinted Search Implementations
# ========================================
#
# Core implementations (type-matched) followed by generic wrappers.

# ----------------------------------------
# Core hinted implementations (type-matched)
# ----------------------------------------

"""
    _search_hinted_binary!(x, xq, hint_ref) -> (idx, xL, xR)

Cached binary search: O(1) if hint valid, O(log n) fallback.
Updates `hint_ref` with the found interval index.
"""
@inline function _search_hinted_binary!(
    x::AbstractVector{T}, xq::T, hint_ref::Base.RefValue{Int}
) where {T<:Real}
    ix = hint_ref[]
    n = length(x)
    @inbounds if 1 <= ix <= n - 1 && x[ix] <= xq < x[ix + 1]
        return ix, x[ix], x[ix + 1]
    end
    idx, xL, xR = _search_binary(x, xq)
    hint_ref[] = idx
    return idx, xL, xR
end

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
    xq::T,
    hint_ref::Base.RefValue{Int},
) where {T<:Real}
    ix = hint_ref[]
    n = length(x)
    @inbounds begin
        # Clamp once at start (handles bad initial hint)
        ix = clamp(ix, 1, n - 1)

        # Direct hit - most common case for monotonic queries
        if x[ix] <= xq < x[ix + 1]
            return ix, x[ix], x[ix + 1]
        end

        # Linear walk - NO bounds check, NO fallback
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

"""
    _search_linear_binary!(x, xq, hint_ref, ::Val{MAX}) -> (idx, xL, xR)

Bounded linear search within MAX-sized window, then binary fallback.
Optimal for monotonic query sequences.
"""
@inline function _search_linear_binary!(
    x::AbstractVector{T},
    xq::T,
    hint_ref::Base.RefValue{Int},
    ::Val{MAX},
) where {T<:Real,MAX}
    ix = hint_ref[]
    n = length(x)
    @inbounds begin
        ix = clamp(ix, 1, n - 1)
        if x[ix] <= xq < x[ix + 1]
            hint_ref[] = ix
            return ix, x[ix], x[ix + 1]
        end
        if xq < x[ix]
            for _ in 1:MAX
                ix -= 1
                ix < 1 && break
                if x[ix] <= xq < x[ix + 1]
                    hint_ref[] = ix
                    return ix, x[ix], x[ix + 1]
                end
            end
        else
            for _ in 1:MAX
                ix += 1
                ix > n - 1 && break
                if x[ix] <= xq < x[ix + 1]
                    hint_ref[] = ix
                    return ix, x[ix], x[ix + 1]
                end
            end
        end
    end
    idx, xL, xR = _search_binary(x, xq)
    hint_ref[] = idx
    return idx, xL, xR
end

"""
    _search_hinted_direct!(x::AbstractRange, xq, hint_ref) -> (idx, xL, xR)

O(1) direct search for uniform grids that also updates `hint_ref`.
Same computation as `_search_direct` — the hint check adds no benefit for Range
(direct arithmetic is already as fast as a hint-hit muladd), so we just update the
hint for correct state tracking in heterogeneous grids.
"""
@inline function _search_hinted_direct!(
    x::AbstractRange{T}, xq::T, hint_ref::Base.RefValue{Int}
) where {T<:AbstractFloat}
    idx, xL, xR = _search_direct(x, xq)
    hint_ref[] = idx
    return idx, xL, xR
end

"""
    _search_hinted_direct!(x::AbstractRange, spacing, xq, hint_ref) -> (idx, xL, xR)

Spacing-aware variant for ND paths. Same as `_search_direct` + hint update.
"""
@inline function _search_hinted_direct!(
    x::AbstractRange{T}, spacing::ScalarSpacing{T}, xq::T, hint_ref::Base.RefValue{Int}
) where {T<:AbstractFloat}
    idx, xL, xR = _search_direct(x, spacing, xq)
    hint_ref[] = idx
    return idx, xL, xR
end

# ----------------------------------------
# Generic wrappers for hinted search (type-mismatched)
# ----------------------------------------

"""Generic wrapper for hinted direct search."""
@inline function _search_hinted_direct!(x::AbstractRange{Tg}, xq::Tq, hint_ref::Base.RefValue{Int}) where {Tg<:AbstractFloat, Tq<:Real}
    return _search_hinted_direct!(x, _to_grid_type(xq, Tg), hint_ref)
end

"""Generic wrapper for hinted direct search with spacing."""
@inline function _search_hinted_direct!(x::AbstractRange{Tg}, spacing::ScalarSpacing{Tg}, xq::Tq, hint_ref::Base.RefValue{Int}) where {Tg<:AbstractFloat, Tq<:Real}
    return _search_hinted_direct!(x, spacing, _to_grid_type(xq, Tg), hint_ref)
end

"""Generic wrapper for hinted binary search."""
@inline function _search_hinted_binary!(x::AbstractVector{Tg}, xq::Tq, hint_ref::Base.RefValue{Int}) where {Tg<:Real, Tq<:Real}
    return _search_hinted_binary!(x, _to_grid_type(xq, Tg), hint_ref)
end

"""Generic wrapper for linear search."""
@inline function _search_linear!(x::AbstractVector{Tg}, xq::Tq, hint_ref::Base.RefValue{Int}) where {Tg<:Real, Tq<:Real}
    return _search_linear!(x, _to_grid_type(xq, Tg), hint_ref)
end

"""Generic wrapper for linear-binary search."""
@inline function _search_linear_binary!(x::AbstractVector{Tg}, xq::Tq, hint_ref::Base.RefValue{Int}, v::Val{MAX}) where {Tg<:Real, Tq<:Real, MAX}
    return _search_linear_binary!(x, _to_grid_type(xq, Tg), hint_ref, v)
end

# ========================================
# 4. Main Dispatcher (search_interval)
# ========================================
#
# Design: Thin dispatchers delegate to internal functions.
# Type conversion happens in _search_* generic wrappers, not here.
# This eliminates code duplication and centralizes conversion logic.
#
# Naming Convention:
#   - Tg: Grid element type (from x::AbstractVector{Tg})
#   - Tq: Query type (can be Float, Int, Dual, etc.)
#   - xq: Query point (x query)

# --- Default: Binary + NoHint (zero-overhead) ---

@inline search_interval(::Searcher{Binary,NoHint}, x::AbstractVector, xq::Real) =
    _search_binary(x, xq)

@inline search_interval(::Searcher{Binary,NoHint}, x::AbstractRange, xq::Real) =
    _search_direct(x, xq)

@inline search_interval(::Searcher{Binary,NoHint}, x::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}, xq::Real) where {Tg} =
    _search_binary(x, spacing, xq)

@inline search_interval(::Searcher{Binary,NoHint}, x::AbstractRange{Tg}, spacing::ScalarSpacing{Tg}, xq::Real) where {Tg} =
    _search_direct(x, spacing, xq)

# --- HintedBinary + RefHint ---

@inline function search_interval(p::Searcher{HintedBinary,RefHint}, x::AbstractVector, xq::Real)
    return _search_hinted_binary!(x, xq, p.hint.idx)
end

# Range: check hint first, fallback to O(1) direct
@inline search_interval(p::Searcher{HintedBinary,RefHint}, x::AbstractRange, xq::Real) =
    _search_hinted_direct!(x, xq, p.hint.idx)

# --- Linear + RefHint ---

@inline function search_interval(p::Searcher{Linear,RefHint}, x::AbstractVector, xq::Real)
    return _search_linear!(x, xq, p.hint.idx)
end

# Range: check hint first, fallback to O(1) direct
@inline search_interval(p::Searcher{Linear,RefHint}, x::AbstractRange, xq::Real) =
    _search_hinted_direct!(x, xq, p.hint.idx)

# --- LinearBinary{MAX} + RefHint ---

@inline function search_interval(p::Searcher{LinearBinary{MAX},RefHint}, x::AbstractVector, xq::Real) where {MAX}
    return _search_linear_binary!(x, xq, p.hint.idx, Val(MAX))
end

@inline search_interval(p::Searcher{LinearBinary{MAX},RefHint}, x::AbstractRange, xq::Real) where {MAX} =
    _search_hinted_direct!(x, xq, p.hint.idx)

# --- Spacing-aware overloads ---
# For uniform grids (AbstractRange + ScalarSpacing): always O(1) direct
# For non-uniform grids (AbstractVector + VectorSpacing): delegate to standard search

# HintedBinary + spacing
@inline search_interval(p::Searcher{HintedBinary,RefHint}, x::AbstractVector, ::AbstractGridSpacing, xq::Real) =
    _search_hinted_binary!(x, xq, p.hint.idx)

@inline search_interval(p::Searcher{HintedBinary,RefHint}, x::AbstractRange, spacing::ScalarSpacing, xq::Real) =
    _search_hinted_direct!(x, spacing, xq, p.hint.idx)

# Linear + spacing
@inline search_interval(p::Searcher{Linear,RefHint}, x::AbstractVector, ::AbstractGridSpacing, xq::Real) =
    _search_linear!(x, xq, p.hint.idx)

@inline search_interval(p::Searcher{Linear,RefHint}, x::AbstractRange, spacing::ScalarSpacing, xq::Real) =
    _search_hinted_direct!(x, spacing, xq, p.hint.idx)

# LinearBinary + spacing
@inline search_interval(p::Searcher{LinearBinary{MAX},RefHint}, x::AbstractVector, ::AbstractGridSpacing, xq::Real) where {MAX} =
    _search_linear_binary!(x, xq, p.hint.idx, Val(MAX))

@inline search_interval(p::Searcher{LinearBinary{MAX},RefHint}, x::AbstractRange, spacing::ScalarSpacing, xq::Real) where {MAX} =
    _search_hinted_direct!(x, spacing, xq, p.hint.idx)

# ========================================
# 5. Internal Aliases (for module-internal use)
# ========================================
# For module-internal use without explicit policy.
# Pure delegation to _search_binary/_search_direct which have generic wrappers.

@inline _search_interval(x::AbstractVector, xq::Real) = _search_binary(x, xq)
@inline _search_interval(x::AbstractRange, xq::Real) = _search_direct(x, xq)
@inline _search_interval(x::AbstractVector, spacing::AbstractGridSpacing, xq::Real) = _search_binary(x, spacing, xq)
@inline _search_interval(x::AbstractRange, spacing::ScalarSpacing, xq::Real) = _search_direct(x, spacing, xq)
