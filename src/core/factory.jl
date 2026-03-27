# ========================================
# User-Facing Factory Functions
# ========================================
#
# Convenience constructors that map user-friendly inputs (Int, Symbol) to
# concrete singleton types. Called at construction/API boundaries only — never
# on hot paths. All return concrete types for zero-allocation, type-stable dispatch.
#
# Design:
#   DerivOp(1)        →  DerivOp{1}()      (Int → parametric singleton)
#   Search(:binary)   →  BinarySearch()     (Symbol → concrete type)
#   Search(BinarySearch())  →  BinarySearch()  (passthrough, generic code friendly)

# ────────────────────────────────────────
# DerivOp Factory (moved from eval_ops.jl)
# ────────────────────────────────────────

# 1D constructor: DerivOp(1) → DerivOp{1}()
@inline DerivOp(n::Int) = _make_derivop(n)

# ND constructor: DerivOp(1, 0) → (DerivOp{1}(), DerivOp{0}())
@inline DerivOp(n1::Int, n2::Int, rest::Int...) = map(_make_derivop, (n1, n2, rest...))

# Internal: Int → concrete DerivOp singleton (union-split friendly for 0-3)
@inline function _make_derivop(n::Int)
    n == 0 && return DerivOp{0}()
    n == 1 && return DerivOp{1}()
    n == 2 && return DerivOp{2}()
    n == 3 && return DerivOp{3}()
    n >= 4 && return DerivOp{n}()
    _derivop_order_error(n)
end

@noinline _derivop_order_error(n::Int) =
    throw(ArgumentError("derivative order must be non-negative, got $n"))

# ────────────────────────────────────────
# Search Factory
# ────────────────────────────────────────

"""
    Search(policy::Symbol; kwargs...)
    Search(s1::Symbol, s2::Symbol, rest::Symbol...)
    Search(policy::AbstractSearchPolicy)

Create a search policy for interval lookup.

Returns a concrete [`AbstractSearchPolicy`](@ref) subtype. The multi-arg form
creates a Tuple for ND per-axis configuration (same pattern as `DerivOp(1, 0)`).

# Options
- `:auto` → [`AutoSearch`](@ref): adaptive (default) — scalar→binary, vector→linear-binary
- `:binary` → [`BinarySearch`](@ref): O(log n), stateless, thread-safe
- `:linear` → [`LinearSearch`](@ref): O(1) amortized walk (requires monotonic queries)
- `:linear_binary` → [`LinearBinarySearch`](@ref): bounded linear walk with binary fallback

# Keywords (single-arg `:linear_binary` only)
- `linear_window::Integer=8`: maximum linear walk steps before binary fallback.
  Must be 0 or a power of 2 from (1, 2, 4, 8, 16, 32, 64, 128).

# Examples
```julia
# 1D
itp = cubic_interp(x, y; search=Search(:binary))
itp = cubic_interp(x, y; search=Search(:linear_binary; linear_window=4))

# ND per-axis (multi-arg form)
itp = cubic_interp((x, y, z), data; search=Search(:binary, :linear_binary, :binary))

# Passthrough — accepts both symbols and policies
Search(:binary)         # → BinarySearch()
Search(BinarySearch())  # → BinarySearch()
```
"""
function Search(sym::Symbol; kwargs...)
    if sym === :linear_binary
        return LinearBinarySearch(; kwargs...)
    end
    isempty(kwargs) || _search_invalid_kwargs_error(sym, kwargs)
    sym === :auto   && return AutoSearch()
    sym === :binary && return BinarySearch()
    sym === :linear && return LinearSearch()
    _search_unknown_error(sym)
end

# ND variadic: Search(:binary, :linear_binary, :binary) → tuple of policies
Search(s1::Symbol, s2::Symbol, rest::Symbol...) = map(Search, (s1, s2, rest...))

Search(p::AbstractSearchPolicy) = p

@noinline function _search_unknown_error(sym::Symbol)
    throw(
        ArgumentError(
            "unknown search policy :$sym; valid options are :auto, :binary, :linear, :linear_binary"
        )
    )
end

@noinline function _search_invalid_kwargs_error(sym::Symbol, kwargs)
    throw(
        ArgumentError(
            "keyword arguments are only supported for :linear_binary, " *
                "got Search(:$sym; $(join(keys(kwargs), ", ")))"
        )
    )
end

# ────────────────────────────────────────
# Extrap Factory
# ────────────────────────────────────────

"""
    Extrap(mode::Symbol; fill_value=nothing)
    Extrap(s1::Symbol, s2::Symbol, rest::Symbol...)
    Extrap(mode::AbstractExtrap)

Create an extrapolation mode for out-of-domain queries.

Returns a concrete [`AbstractExtrap`](@ref) subtype. The multi-arg form
creates a Tuple for ND per-axis configuration (same pattern as `DerivOp(1, 0)`).

# Options
- `:none` → [`NoExtrap`](@ref): throw `DomainError` for out-of-domain queries
- `:clamp` → [`ClampExtrap`](@ref): clamp to nearest boundary value
- `:fill` → [`FillExtrap`](@ref): fill with a constant value (requires `fill_value` keyword)
- `:extend` → [`ExtendExtrap`](@ref): extend interpolation polynomial beyond domain
- `:wrap` → [`WrapExtrap`](@ref): wrap queries into domain (periodic)
- `:constant` → *(deprecated)* alias for `:clamp`

# Examples
```julia
# 1D
itp = cubic_interp(x, y; extrap=Extrap(:clamp))
itp = cubic_interp(x, y; extrap=Extrap(:fill; fill_value=NaN))

# ND per-axis (multi-arg form)
itp = cubic_interp((x, y, z), data; extrap=Extrap(:extend, :none, :wrap))
itp = cubic_interp((x, y), data; extrap=Extrap(:clamp, :fill; fill_value=0.0))

# Passthrough
Extrap(:none)        # → NoExtrap()
Extrap(NoExtrap())   # → NoExtrap()
```
"""
function Extrap(sym::Symbol; fill_value = nothing)
    if fill_value !== nothing && sym !== :fill
        @warn "fill_value keyword is ignored because :$sym extrapolation does not use it; only :fill does"
    end
    sym === :none     && return NoExtrap()
    sym === :clamp  && return ClampExtrap()
    sym === :fill     && return fill_value === nothing ? _extrap_fill_missing_value_error() : FillExtrap(fill_value)
    sym === :extend   && return ExtendExtrap()
    sym === :wrap     && return WrapExtrap()
    if sym === :constant
        Base.depwarn(
            "Extrap(:constant) is deprecated, use Extrap(:clamp) instead.",
            :Extrap
        )
        return ClampExtrap()
    end
    _extrap_unknown_error(sym)
end

# ND variadic: Extrap(:extend, :none, :wrap) → tuple of extrap modes
# Forward `fill_value` kwarg only to :fill axes; warn once if no axis uses :fill
function Extrap(s1::Symbol, s2::Symbol, rest::Symbol...; fill_value = nothing)
    syms = (s1, s2, rest...)
    has_fill_axis = any(==(:fill), syms)
    if fill_value !== nothing && !has_fill_axis
        @warn "fill_value keyword is ignored because no axis uses :fill extrapolation"
    end
    # Only forward fill_value to :fill axes; use nothing for others to suppress per-axis warnings
    fv = has_fill_axis ? fill_value : nothing
    return map(s -> Extrap(s; fill_value = (s === :fill ? fv : nothing)), syms)
end

Extrap(e::AbstractExtrap) = e

@noinline function _extrap_unknown_error(sym::Symbol)
    throw(
        ArgumentError(
            "unknown extrapolation mode :$sym; valid options are :none, :clamp, :fill, :constant (deprecated), :extend, :wrap"
        )
    )
end

@noinline function _extrap_fill_missing_value_error()
    throw(
        ArgumentError(
            "Extrap(:fill) requires a `fill_value` keyword argument, e.g. Extrap(:fill; fill_value=NaN)"
        )
    )
end

# ────────────────────────────────────────
# Side Factory
# ────────────────────────────────────────

"""
    Side(mode::Symbol)
    Side(s1::Symbol, s2::Symbol, rest::Symbol...)
    Side(mode::AbstractSide)

Create a side selection mode for constant interpolation.

Returns a concrete [`AbstractSide`](@ref) subtype. The multi-arg form
creates a Tuple for ND per-axis configuration (same pattern as `DerivOp(1, 0)`).

# Options
- `:nearest` → [`NearestSide`](@ref): nearest neighbor with left tie-breaking at midpoint
- `:left` → [`LeftSide`](@ref): always use left (floor) value
- `:right` → [`RightSide`](@ref): use right (ceiling) value, except at grid points

# Examples
```julia
# 1D
itp = constant_interp(x, y; side=Side(:left))

# ND per-axis (multi-arg form)
itp = constant_interp((x, y), data; side=Side(:left, :nearest))

# Passthrough
Side(:nearest)        # → NearestSide()
Side(NearestSide())   # → NearestSide()
```
"""
function Side(sym::Symbol)
    sym === :nearest && return NearestSide()
    sym === :left    && return LeftSide()
    sym === :right   && return RightSide()
    _side_unknown_error(sym)
end

# ND variadic: Side(:left, :nearest, :right) → tuple of side modes
Side(s1::Symbol, s2::Symbol, rest::Symbol...) = map(Side, (s1, s2, rest...))

Side(s::AbstractSide) = s

@noinline function _side_unknown_error(sym::Symbol)
    throw(
        ArgumentError(
            "unknown side selection :$sym; valid options are :nearest, :left, :right"
        )
    )
end
