# ========================================
# Search Policy Types and Implementations
# ========================================
#
# This module provides a typed policy system for interval search dispatch.
# The key insight is that algorithm selection is encoded in type parameters,
# enabling compile-time dispatch and zero-overhead for the default Binary path.
#
# Naming Convention: SearchPolicy → search_interval → _search_binary
#
# Include order dependency: grid_spacing.jl (for ScalarSpacing) → search.jl

# ========================================
# 1. Search Policy Types
# ========================================

"""
    AbstractSearchAlg

Abstract supertype for search algorithm types.
Concrete subtypes encode search strategy in the type system.
"""
abstract type AbstractSearchAlg end

"""
    BinaryAlg <: AbstractSearchAlg

Default binary search algorithm. O(log n) for vectors, O(1) for ranges.
Zero-overhead: compiles to identical code as direct `_find_interval` call.
"""
struct BinaryAlg <: AbstractSearchAlg end

"""
    HintedBinaryAlg <: AbstractSearchAlg

Hinted binary search: O(1) if hint is valid, O(log n) fallback.
Useful when queries repeatedly hit the same interval.
"""
struct HintedBinaryAlg <: AbstractSearchAlg end

"""
    LinearBoundedAlg{MAX} <: AbstractSearchAlg

Bounded linear search: up to MAX linear steps, then binary fallback.
Optimal for monotonic-ish query sequences where consecutive queries
are likely to be in adjacent intervals.

# Type Parameter
- `MAX`: Maximum linear search steps before falling back to binary (compile-time constant)
"""
struct LinearBoundedAlg{MAX} <: AbstractSearchAlg end

# ----------------------------------------
# Hint Types
# ----------------------------------------

"""
    AbstractHint

Abstract supertype for hint/state management types.
"""
abstract type AbstractHint end

"""
    NoHint <: AbstractHint

No state maintained. Used with `BinaryAlg` for stateless search.
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
end
RefHint() = RefHint(Ref(1))
RefHint(idx::Int) = RefHint(Ref(idx))

# Future: ThreadHint for per-thread slots
# struct ThreadHint{H} <: AbstractHint
#     hint::H
# end

# ----------------------------------------
# Combined Policy Type
# ----------------------------------------

"""
    SearchPolicy{Alg<:AbstractSearchAlg, H<:AbstractHint}

Combined search policy type encoding both algorithm and hint strategy.
Type parameters enable compile-time dispatch with zero runtime overhead.

# Type Parameters
- `Alg`: Search algorithm type (`BinaryAlg`, `HintedBinaryAlg`, `LinearBoundedAlg{N}`)
- `H`: Hint type (`NoHint`, `RefHint`)

# Fields
- `hint::H`: The hint instance (NoHint singleton or RefHint with mutable Ref)

# Example
```julia
# Default: stateless binary search
policy = SearchPolicy{BinaryAlg,NoHint}(NoHint())

# Hinted binary with loop-local state
policy = SearchPolicy{HintedBinaryAlg,RefHint}(RefHint(Ref(1)))

# Bounded linear with max 8 steps
policy = SearchPolicy{LinearBoundedAlg{8},RefHint}(RefHint(Ref(1)))
```
"""
struct SearchPolicy{Alg<:AbstractSearchAlg,H<:AbstractHint}
    hint::H
end

"""
    DEFAULT_SEARCH_POLICY

Default search policy: stateless binary search with no hint.
Compiles to identical code as direct `_find_interval` call.
"""
const DEFAULT_SEARCH_POLICY = SearchPolicy{BinaryAlg,NoHint}(NoHint())

# ========================================
# 2. Base Implementations (Binary Search)
# ========================================
#
# These are the core implementations, moved from utils.jl.
# Function naming: _search_binary (internal), search_interval (dispatcher)

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
    _search_binary(x::AbstractRange{T}, xi::T) where {T<:AbstractFloat}

O(1) direct calculation for uniform grids (AbstractRange).
Uses `unsafe_trunc` for ~40% faster index calculation.
"""
@inline function _search_binary(x::AbstractRange{T}, xi::T) where {T<:AbstractFloat}
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
    _search_binary(x::AbstractRange{T}, spacing::ScalarSpacing{T}, xi::T)

Spacing-aware O(1) for ScalarSpacing (multiplication instead of division).
"""
@inline function _search_binary(
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
    _search_linear_bounded!(x, xi, hint_ref, ::Val{MAX}) -> (idx, xL, xR)

Bounded linear search: up to MAX steps, then binary fallback.
Optimal for monotonic query sequences.
"""
@inline function _search_linear_bounded!(
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

# --- Default: BinaryAlg + NoHint (zero-overhead) ---

@inline search_interval(::SearchPolicy{BinaryAlg,NoHint}, x::AbstractVector{T}, xi::T) where {T} =
    _search_binary(x, xi)

@inline search_interval(::SearchPolicy{BinaryAlg,NoHint}, x::AbstractRange{T}, xi::T) where {T} =
    _search_binary(x, xi)

@inline search_interval(::SearchPolicy{BinaryAlg,NoHint}, x, spacing, xi) =
    _search_binary(x, spacing, xi)

# --- HintedBinaryAlg + RefHint ---

@inline function search_interval(
    p::SearchPolicy{HintedBinaryAlg,RefHint}, x::AbstractVector{T}, xi::T
) where {T}
    return _search_hinted_binary!(x, xi, p.hint.idx)
end

# Range always uses O(1) - hint ignored
@inline search_interval(::SearchPolicy{HintedBinaryAlg,RefHint}, x::AbstractRange{T}, xi::T) where {T} =
    _search_binary(x, xi)

# --- LinearBoundedAlg{MAX} + RefHint ---

@inline function search_interval(
    p::SearchPolicy{LinearBoundedAlg{MAX},RefHint}, x::AbstractVector{T}, xi::T
) where {MAX,T}
    return _search_linear_bounded!(x, xi, p.hint.idx, Val(MAX))
end

@inline search_interval(
    ::SearchPolicy{LinearBoundedAlg{MAX},RefHint},
    x::AbstractRange{T},
    xi::T,
) where {MAX,T} =
    _search_binary(x, xi)

# ========================================
# 5. Internal Aliases (for module-internal use)
# ========================================
# For module-internal use without explicit policy.
# search_interval(DEFAULT_SEARCH_POLICY, x, xi) → _search_interval(x, xi)

@inline _search_interval(x, xi) = _search_binary(x, xi)
@inline _search_interval(x, spacing, xi) = _search_binary(x, spacing, xi)

# ========================================
# 6. Deprecated Aliases (backward compat)
# ========================================
# Transitional alias for existing _find_interval callers.
# Will be removed in v0.4.0

"""
    _find_interval(x, xi) -> (idx, xL, xR)

!!! warning "Deprecated"
    `_find_interval` is deprecated and will be removed in v0.4.0.
    Use `_search_interval(x, xi)` instead.

Find interpolation interval. See [`_search_interval`](@ref) for details.
"""
@inline function _find_interval(x, xi)
    Base.depwarn(
        "`_find_interval(x, xi)` is deprecated, use `_search_interval(x, xi)` instead",
        :_find_interval
    )
    return _search_interval(x, xi)
end

"""
    _find_interval(x, spacing, xi) -> (idx, xL, xR)

!!! warning "Deprecated"
    `_find_interval` is deprecated and will be removed in v0.4.0.
    Use `_search_interval(x, spacing, xi)` instead.
"""
@inline function _find_interval(x, spacing, xi)
    Base.depwarn(
        "`_find_interval(x, spacing, xi)` is deprecated, use `_search_interval(x, spacing, xi)` instead",
        :_find_interval
    )
    return _search_interval(x, spacing, xi)
end
