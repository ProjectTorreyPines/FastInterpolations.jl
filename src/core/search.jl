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

Bounded linear search: up to `MAX` linear steps from hint, then binary fallback.
Optimal for **sorted/monotonic query sequences** where consecutive queries tend to
fall in adjacent or nearby intervals.

# Type Parameter
- `MAX::Int`: Maximum linear search steps before falling back to binary search.
  This is a compile-time constant encoded in the type parameter.

# Performance Characteristics
- **Best case**: O(1) when query is within `MAX` steps of the hint
- **Worst case**: O(log n) binary search fallback
- **Ideal for**: Sorted queries, time-series data, streaming evaluations

# Construction
Use the factory function (recommended) to construct with a curated set of values:
```julia
LinearBinary()               # default MAX=8
LinearBinary(max_steps=4)    # custom MAX=4
```

Or construct the parametric type directly (advanced):
```julia
LinearBinary{8}()            # explicit type parameter
```

# Example
```julia
sorted_queries = sort(rand(1000))
vals = linear_interp(x, y, sorted_queries; search=LinearBinary(max_steps=8))
```

See also: [`Binary`](@ref), [`HintedBinary`](@ref)
"""
struct LinearBinary{MAX} <: AbstractSearchPolicy end

"""
    LinearBinary(max_steps::Integer)
    LinearBinary(; max_steps::Integer=8)

Factory constructor for `LinearBinary{MAX}` with a **curated set of `max_steps` values**.

# Why Restricted Values?
Julia compiles a specialized method for each unique type parameter `MAX`. Allowing
arbitrary integers (1–1000) would cause **type parameter explosion**: hundreds of
specialized methods, increased compile time, and code cache bloat.

By restricting to powers of 2, we limit specialization to just 7 variants while
covering the practical range of use cases.

# Arguments
- `max_steps::Integer=8`: Maximum linear search steps before binary fallback.
  **Allowed values**: `1, 2, 4, 8, 16, 32, 64, 128` (powers of 2 from 2⁰ to 2⁷)

# Throws
- `ArgumentError`: If `max_steps` is not one of the allowed values.

# Example
```julia
policy = LinearBinary()              # LinearBinary{8}()  (default)
policy = LinearBinary(max_steps=4)   # LinearBinary{4}()
policy = LinearBinary(max_steps=16)  # LinearBinary{16}()
policy = LinearBinary(max_steps=3)   # ERROR: ArgumentError
```

# Choosing `max_steps`
- **Small values (1–4)**: Lower overhead, but more frequent binary fallbacks
- **Medium values (8–16)**: Good balance for typical sorted query patterns
- **Large values (32–128)**: For highly localized queries or very large datasets
"""
function LinearBinary(max_steps::Integer)
    max_steps == 1  && return LinearBinary{2}()
    max_steps == 2  && return LinearBinary{2}()
    max_steps == 4  && return LinearBinary{4}()
    max_steps == 8  && return LinearBinary{8}()
    max_steps == 16 && return LinearBinary{16}()
    max_steps == 32 && return LinearBinary{32}()
    max_steps == 64 && return LinearBinary{64}()
    max_steps == 128 && return LinearBinary{128}()
    throw(ArgumentError("`max_steps` must be one of (1, 2, 4, 8, 16, 32, 64, 128), got $max_steps"))
end
LinearBinary(; max_steps::Integer=8) = LinearBinary(max_steps)

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

"""
    _search_direct(x::AbstractRange{T}, xi::T) where {T<:AbstractFloat}

O(1) direct index calculation for uniform grids (AbstractRange).
Uses `unsafe_trunc` for ~40% faster index calculation.

Unlike `_search_binary`, this function computes the interval index directly
via arithmetic rather than iterative search, exploiting uniform grid spacing.
"""
@inline function _search_direct(x::AbstractRange{T}, xi::T) where {T<:AbstractFloat}
    n = length(x)
    x_min = first(x)
    dx = Base.step(x)
    idx = clamp(unsafe_trunc(Int, (xi - x_min) / dx + 1), 1, n - 1)
    xL = muladd(idx - 1, dx, x_min)
    xR = xL + dx
    return idx, xL, xR
end

"""
    _search_binary(x::AbstractVector{T}, xi::T) where {T<:AbstractFloat}

O(log n) binary search for non-uniform grids (AbstractVector).
"""
@inline function _search_binary(x::AbstractVector{T}, xi::T) where {T<:AbstractFloat}
    n = length(x)
    @inbounds begin
        if xi <= x[1]
            idx = 1
        elseif xi >= x[end]
            idx = n - 1
        else
            lo, hi = 1, n
            while hi - lo > 1
                mid = (lo + hi) >> 1
                if x[mid] <= xi
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
    _search_direct(x::AbstractRange{T}, spacing::ScalarSpacing{T}, xi::T)

Spacing-aware O(1) direct calculation for ScalarSpacing.
Uses pre-computed `inv_h` for multiplication instead of division.
"""
@inline function _search_direct(
    x::AbstractRange{T}, spacing::ScalarSpacing{T}, xi::T
) where {T<:AbstractFloat}
    n = length(x)
    x_min = first(x)
    idx = clamp(unsafe_trunc(Int, (xi - x_min) * spacing.inv_h + 1), 1, n - 1)
    xL = muladd(idx - 1, spacing.h, x_min)
    xR = xL + spacing.h
    return idx, xL, xR
end

"""
    _search_binary(x::AbstractVector{T}, ::AbstractGridSpacing{T}, xi::T)

VectorSpacing delegates to non-spacing version.
"""
@inline function _search_binary(
    x::AbstractVector{T}, ::AbstractGridSpacing{T}, xi::T
) where {T<:AbstractFloat}
    return _search_binary(x, xi)
end

# ========================================
# 3. Hinted Search Implementations
# ========================================

"""
    _search_hinted_binary!(x, xi, hint_ref) -> (idx, xL, xR)

Cached binary search: O(1) if hint valid, O(log n) fallback.
Updates `hint_ref` with the found interval index.
"""
@inline function _search_hinted_binary!(
    x::AbstractVector{T}, xi::T, hint_ref::Base.RefValue{Int}
) where {T<:AbstractFloat}
    ix = hint_ref[]
    n = length(x)
    @inbounds if 1 <= ix <= n - 1 && x[ix] <= xi < x[ix + 1]
        return ix, x[ix], x[ix + 1]
    end
    idx, xL, xR = _search_binary(x, xi)
    hint_ref[] = idx
    return idx, xL, xR
end

"""
    _search_linear!(x, xi, hint_ref) -> (idx, xL, xR)

Pure linear search: walks from hint until interval found.
No bounds checking (except initial clamp), no binary fallback.

# Safety Contract
- Caller guarantees `x[1] <= xi <= x[end]` (within domain)
- Caller guarantees monotonic query progression
- Hint is clamped once at start, then trusted
"""
@inline function _search_linear!(
    x::AbstractVector{T},
    xi::T,
    hint_ref::Base.RefValue{Int},
) where {T<:AbstractFloat}
    ix = hint_ref[]
    n = length(x)
    @inbounds begin
        # Clamp once at start (handles bad initial hint)
        ix = clamp(ix, 1, n - 1)

        # Direct hit - most common case for monotonic queries
        if x[ix] <= xi < x[ix + 1]
            return ix, x[ix], x[ix + 1]
        end

        # Linear walk - NO bounds check, NO fallback
        if xi < x[ix]
            while x[ix] > xi
                ix -= 1
            end
        else  # xi >= x[ix + 1]
            while x[ix + 1] <= xi
                ix += 1
            end
        end
        hint_ref[] = ix
        return ix, x[ix], x[ix + 1]
    end
end

"""
    _search_linear_binary!(x, xi, hint_ref, ::Val{MAX}) -> (idx, xL, xR)

Bounded linear search: up to MAX steps, then binary fallback.
Optimal for monotonic query sequences.
"""
@inline function _search_linear_binary!(
    x::AbstractVector{T},
    xi::T,
    hint_ref::Base.RefValue{Int},
    ::Val{MAX},
) where {T<:AbstractFloat,MAX}
    ix = hint_ref[]
    n = length(x)
    @inbounds begin
        ix = clamp(ix, 1, n - 1)
        if x[ix] <= xi < x[ix + 1]
            hint_ref[] = ix
            return ix, x[ix], x[ix + 1]
        end
        if xi < x[ix]
            for _ in 1:MAX
                ix -= 1
                ix < 1 && break
                if x[ix] <= xi < x[ix + 1]
                    hint_ref[] = ix
                    return ix, x[ix], x[ix + 1]
                end
            end
        else
            for _ in 1:MAX
                ix += 1
                ix > n - 1 && break
                if x[ix] <= xi < x[ix + 1]
                    hint_ref[] = ix
                    return ix, x[ix], x[ix + 1]
                end
            end
        end
    end
    idx, xL, xR = _search_binary(x, xi)
    hint_ref[] = idx
    return idx, xL, xR
end

# ========================================
# 4. Main Dispatcher (search_interval)
# ========================================

# --- Default: Binary + NoHint (zero-overhead) ---

@inline search_interval(::Searcher{Binary,NoHint}, x::AbstractVector{T}, xi::T) where {T} =
    _search_binary(x, xi)

@inline search_interval(::Searcher{Binary,NoHint}, x::AbstractRange{T}, xi::T) where {T} =
    _search_direct(x, xi)

@inline search_interval(::Searcher{Binary,NoHint}, x, spacing, xi) =
    _search_interval(x, spacing, xi)

# --- HintedBinary + RefHint ---

@inline function search_interval(
    p::Searcher{HintedBinary,RefHint}, x::AbstractVector{T}, xi::T
) where {T}
    return _search_hinted_binary!(x, xi, p.hint.idx)
end

# Range always uses O(1) direct - hint ignored
@inline search_interval(::Searcher{HintedBinary,RefHint}, x::AbstractRange{T}, xi::T) where {T} =
    _search_direct(x, xi)

# --- Linear + RefHint ---

@inline function search_interval(
    p::Searcher{Linear,RefHint}, x::AbstractVector{T}, xi::T
) where {T}
    return _search_linear!(x, xi, p.hint.idx)
end

# Range always uses O(1) direct - hint ignored
@inline search_interval(::Searcher{Linear,RefHint}, x::AbstractRange{T}, xi::T) where {T} =
    _search_direct(x, xi)

# --- LinearBinary{MAX} + RefHint ---

@inline function search_interval(
    p::Searcher{LinearBinary{MAX},RefHint}, x::AbstractVector{T}, xi::T
) where {MAX,T}
    return _search_linear_binary!(x, xi, p.hint.idx, Val(MAX))
end

@inline search_interval(
    ::Searcher{LinearBinary{MAX},RefHint},
    x::AbstractRange{T},
    xi::T,
) where {MAX,T} =
    _search_direct(x, xi)

# --- Spacing-aware overloads ---
# For uniform grids (AbstractRange + ScalarSpacing): always O(1) direct
# For non-uniform grids (AbstractVector + VectorSpacing): delegate to standard search

# HintedBinary + spacing
@inline function search_interval(
    p::Searcher{HintedBinary,RefHint}, x::AbstractVector{T}, ::AbstractGridSpacing{T}, xi::T
) where {T}
    return _search_hinted_binary!(x, xi, p.hint.idx)
end

@inline search_interval(
    ::Searcher{HintedBinary,RefHint}, x::AbstractRange{T}, spacing::ScalarSpacing{T}, xi::T
) where {T} =
    _search_direct(x, spacing, xi)

# Linear + spacing
@inline function search_interval(
    p::Searcher{Linear,RefHint}, x::AbstractVector{T}, ::AbstractGridSpacing{T}, xi::T
) where {T}
    return _search_linear!(x, xi, p.hint.idx)
end

@inline search_interval(
    ::Searcher{Linear,RefHint}, x::AbstractRange{T}, spacing::ScalarSpacing{T}, xi::T
) where {T} =
    _search_direct(x, spacing, xi)

# LinearBinary + spacing
@inline function search_interval(
    p::Searcher{LinearBinary{MAX},RefHint}, x::AbstractVector{T}, ::AbstractGridSpacing{T}, xi::T
) where {MAX,T}
    return _search_linear_binary!(x, xi, p.hint.idx, Val(MAX))
end

@inline search_interval(
    ::Searcher{LinearBinary{MAX},RefHint}, x::AbstractRange{T}, spacing::ScalarSpacing{T}, xi::T
) where {MAX,T} =
    _search_direct(x, spacing, xi)

# ========================================
# 5. Internal Aliases (for module-internal use)
# ========================================
# For module-internal use without explicit policy.
# These dispatch to the appropriate implementation based on grid type.

@inline _search_interval(x::AbstractVector, xi) = _search_binary(x, xi)
@inline _search_interval(x::AbstractRange, xi) = _search_direct(x, xi)
@inline _search_interval(x::AbstractVector, spacing, xi) = _search_binary(x, spacing, xi)
@inline _search_interval(x::AbstractRange, spacing, xi) = _search_direct(x, spacing, xi)

