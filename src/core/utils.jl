# Internal utility functions for FastInterpolations.jl

# ========================================
# Interval Search (IN search.jl)
# ========================================
# Interval search functions are defined in src/core/search.jl:
#   - _search_direct: O(1) direct calculation for uniform grids (AbstractRange)
#   - _search_binary: O(log n) binary search for non-uniform grids (AbstractVector)
#   - _search_interval: dispatcher that routes to the appropriate implementation

# ========================================
# Promotable Value Type (for convenience dispatch)
# ========================================

"""Standard Julia numeric types that should be auto-promoted in convenience wrappers."""
const _PromotableValue = Union{Integer, AbstractFloat, Complex}

# ========================================
# Type Conversion Helpers
# ========================================

"""
    _to_float(x::AbstractRange, ::Type{FT}) where {FT<:AbstractFloat}

Convert a Range to a float type while preserving Range structure for O(1) index lookup.
Using `FT.(x)` would convert Range to Vector, losing the O(1) optimization.
"""
# General conversion: rebuild as StepRangeLen to preserve O(1) indexing
_to_float(x::AbstractRange, ::Type{FT}) where {FT<:AbstractFloat} =
    range(FT(first(x)), FT(last(x)), length(x))

# Fast-path: already Float Range (StepRangeLen, LinRange, etc.) - return as-is
_to_float(x::AbstractRange{FT}, ::Type{FT}) where {FT<:AbstractFloat} = x

"""
    _to_float(x::AbstractVector{FT}, ::Type{FT}) where {FT<:AbstractFloat}

Identity conversion - return as-is when element type already matches target type.
This enables zero-allocation for Real→Float wrappers when types already match.
"""
_to_float(x::AbstractVector{FT}, ::Type{FT}) where {FT<:AbstractFloat} = x

"""
    _to_float(x::AbstractVector, ::Type{FT}) where {FT<:AbstractFloat}

Convert a Vector to a float type (element-wise broadcast).
Emits a one-time warning since this allocates a new vector.
"""
function _to_float(x::AbstractVector, ::Type{FT}) where {FT<:AbstractFloat}
    @warn "Non-float vector input detected - allocating type conversion. " *
          "For zero-allocation, pre-convert your data: `x_float = $FT.(x)`" maxlog=1
    return FT.(x)
end

# ========================================
# Value Type Helpers (for Complex support)
# ========================================

"""
    _real_eltype(::Type{T}) where {T<:Real} -> Type

Extract the real base type from an element type.
For Real types, returns the type itself.
For Complex{T}, returns T.

This is TYPE-BASED (works with eltype(y) in wrappers).
"""
@inline _real_eltype(::Type{T}) where {T<:Real} = T
@inline _real_eltype(::Type{Complex{T}}) where {T<:Real} = T
# Duck-typing fallback: custom types return themselves (no float base extraction)
@inline _real_eltype(::Type{T}) where {T} = T

"""
    _value_type(::Type{Ty}, ::Type{Tg}) -> Type

Determine the output value type from y element type and grid type.
- Real y → Tg (promotes to grid float type)
- Complex y → Complex{Tg}
"""
@inline _value_type(::Type{T}, ::Type{Tg}) where {T<:Real, Tg<:AbstractFloat} = Tg
@inline _value_type(::Type{Complex{T}}, ::Type{Tg}) where {T<:Real, Tg<:AbstractFloat} = Complex{Tg}
# Duck-typing fallback: custom types preserved as-is (no promotion to grid type)
@inline _value_type(::Type{T}, ::Type{Tg}) where {T, Tg<:AbstractFloat} = T

"""
    _series_output_type(::Type{Tv}, ::Type{Tq}) -> Type

Compute output element type for series evaluation.

For standard numerics, uses `promote_type(Tv, Tq)` to widen correctly
(e.g., Float64 + Dual → Dual for AD support).

For custom Tv without `promote_rule`, `promote_type` falls back to an
abstract typejoin (e.g., Number), which makes the output vector untyped.
In that case, falls back to Tv since the kernel always returns Tv.
"""
@inline function _series_output_type(::Type{Tv}, ::Type{Tq}) where {Tv, Tq}
    Tout = promote_type(Tv, Tq)
    return isconcretetype(Tout) ? Tout : Tv
end

"""
    _promote_value_type(y, ::Type{Tg}) -> (Tv, y_converted)

Promote y-values to appropriate type based on grid type Tg.

# Rules
1. If eltype(y) === Tg → no conversion (identity)
2. If eltype(y) <: Real → convert to Tg
3. If eltype(y) <: Complex → convert to Complex{Tg}
4. Otherwise → convert to promote_type(eltype(y), Tg)

# Returns
Tuple of (Tv::Type, y_converted::AbstractVector{Tv})
"""
@inline function _promote_value_type(y::AbstractVector{Tv_raw}, ::Type{Tg}) where {Tv_raw, Tg<:AbstractFloat}
    if Tv_raw === Tg
        # Already matching float type - no conversion
        return Tg, y
    elseif Tv_raw <: Real
        # Real (including Int, Float32) → promote to Tg
        return Tg, Tg.(y)
    elseif Tv_raw <: Complex
        # Complex{anything} → Complex{Tg}
        Tv = Complex{Tg}
        # Fast-path: already Complex{Tg} (e.g., ComplexF64 with Tg=Float64)
        if Tv_raw === Tv
            return Tv, y
        end
        return Tv, Tv.(y)
    else
        # Other Number types → promote
        Tv = promote_type(Tv_raw, Tg)
        return Tv, Tv.(y)
    end
end

# ========================================
# Unified Input Promotion API
# ========================================

"""
    _promote_itp_inputs(x, y) -> (x_typed, y_typed)

Promote grid (x) and values (y) to compatible types for interpolation.

# Behavior
- Grid (x) is always converted to AbstractFloat via `_to_float`
- Values (y) handling depends on element type:
  - Standard numerics (`<: _PromotableValue`): promoted to match grid float type
  - Custom/duck types: preserved as-is (zero-copy)

# Standard Path (Real, AbstractFloat, Complex)
- Computes target grid type: `Tg = float(promote_type(TX, _real_eltype(TY)))`
- Converts x via `_to_float` (Range structure preserved)
- Promotes y via `_promote_value_type` (handles Real/Complex widening)

# Duck-Typing Path (custom number types)
- Grid type: `Tg = float(TX)` (no y influence)
- y returned unchanged — custom types preserved for generic kernel arithmetic

# Zero-Overhead Guarantee
- `@inline` + compile-time `TY <: _PromotableValue` check → dead branch eliminated
- Returns inputs unchanged when types already match (zero allocation)

# Examples
```julia
x = [0.0, 1.0, 2.0]; y_int = [1, 2, 3]
x_p, y_p = _promote_itp_inputs(x, y_int)    # y_p is Float64[] (promoted)

# Custom types preserved
x_p, y_p = _promote_itp_inputs(x, custom_y)  # y_p stays custom type
```
"""
@inline function _promote_itp_inputs(
    x::AbstractVector{TX},
    y::AbstractVector{TY}
) where {TX<:Real, TY}
    if TY <: _PromotableValue
        # Standard numerics: widen grid to accommodate y precision, promote y
        Tg = float(promote_type(TX, _real_eltype(TY)))
        x_typed = _to_float(x, Tg)
        _, y_typed = _promote_value_type(y, Tg)
        return x_typed, y_typed
    else
        # Custom/duck types: only convert grid to float, preserve y as-is
        Tg = float(TX)
        x_typed = _to_float(x, Tg)
        return x_typed, y
    end
end

"""
    _promote_itp_inputs(x, y, xq::AbstractVector) -> (x_typed, y_typed, xq_typed)

Promote grid (x), values (y), and vector query (xq) to compatible Float types.

# Arguments
- `x`: Grid coordinates (any Real type)
- `y`: Values at grid points (Real or Complex)
- `xq`: Query points (AbstractVector or AbstractRange)

# Returns
- `x_typed`: Grid converted to AbstractFloat
- `y_typed`: Values converted to compatible type
- `xq_typed`: Query converted to grid type (Range structure preserved via `_to_float`)

# Fast-paths
- When types already match, returns inputs unchanged (zero allocation)
- Range inputs remain Range (not converted to Vector)

# Note
For scalar queries, use the 2-arg version and pass the query directly
to preserve ForwardDiff.Dual types for automatic differentiation.
"""
@inline function _promote_itp_inputs(
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    xq::AbstractVector{TQ}
) where {TX<:Real, TY, TQ<:Real}
    x_typed, y_typed = _promote_itp_inputs(x, y)
    Tg = eltype(x_typed)
    xq_typed = _to_float(xq, Tg)
    return x_typed, y_typed, xq_typed
end

# ========================================
# AD Support Helpers
# ========================================

"""
    _to_grid_type(xq, ::Type{Tg}) -> Tg

Convert query point to grid type for index search.
Extracts primal value from AD types (via `_extract_primal`), then converts to Tg.

# Zero-overhead paths (compile-time dispatch)
- `xq::Tg` → returns as-is (identity method selected)
- `xq` after primal extraction equals Tg → `Tg(Tg_value)` optimized away

# Conversion paths
- `xq::Float32` on `Float64` grid → converts to Float64
- `xq::Int` → directly to Tg (no intermediate Float64)
- `xq::Dual{...}` → extracts primal via `_extract_primal` → converts to Tg

# Usage in Search
```julia
# Before (2 lines):
xq_primal = _extract_primal(xq)
xq_conv = Tg(xq_primal)

# After (1 line):
xq_conv = _to_grid_type(xq, Tg)
```
"""
@inline _to_grid_type(xq::Tg, ::Type{Tg}) where {Tg<:Real} = xq  # identity: already correct type
@inline _to_grid_type(xq::Real, ::Type{Tg}) where {Tg<:Real} = Tg(_extract_primal(xq))  # convert via primal extraction

"""
    _extract_primal(xq) -> AbstractFloat

Extract the primal (real) value from a query point for index search.

For regular floats, returns as-is.
For ForwardDiff.Dual (when loaded), returns the primal value.

# Usage in Search
This allows AD types to be used for interpolation queries:
- Use `_extract_primal(xq)` ONLY for index search (comparisons)
- Use original `xq` for arithmetic (preserves AD derivatives)

# AD Extension
ForwardDiff support is added via:
```julia
@inline _extract_primal(xq::ForwardDiff.Dual) = ForwardDiff.value(xq)
```
"""
@inline _extract_primal(xq::T) where {T<:AbstractFloat} = xq
@inline _extract_primal(xq::Real) = xq  # return as-is, let _to_grid_type handle conversion

"""
    _promote_for_anchor(xq::Tq, ::Type{Tg}) -> promoted_xq

Promote query point for anchor construction.

# Behavior
- ForwardDiff.Dual: preserved as-is (for AD support, see extension)
- AbstractFloat: uses promote_type(Tq, Tg) to preserve precision
  - Float64 on Float32 grid → Float64 (preserves query precision)
  - Float32 on Float64 grid → Float64 (uses grid precision)
- Other Real (Int, Rational): converted to grid type Tg

This is needed for cubic anchors which store precomputed weight tuples.
Unlike quadratic (which stores only dL), cubic weights involve complex
floating-point arithmetic that can't be represented as Int/Rational.

# Example
```julia
_promote_for_anchor(0, Float64)      # → 0.0 (Float64)
_promote_for_anchor(1//2, Float64)   # → 0.5 (Float64)
_promote_for_anchor(0.5, Float32)    # → 0.5 (Float64, preserves precision)
_promote_for_anchor(0.5f0, Float32)  # → 0.5f0 (Float32)
_promote_for_anchor(dual, Float64)   # → dual (preserved Dual type)
```
"""
# For AbstractFloat queries: preserve precision using wider type (lossless promotion)
@inline _promote_for_anchor(xq::Tq, ::Type{Tg}) where {Tq<:AbstractFloat, Tg<:AbstractFloat} = convert(promote_type(Tq, Tg), xq)
# For other Real (Int, Rational): convert to grid type (no precision loss for integers)
@inline _promote_for_anchor(xq::Tq, ::Type{Tg}) where {Tq<:Real, Tg<:AbstractFloat} = Tg(xq)


# ========================================
# Domain Validation Helpers
# ========================================

"Scalar domain check for NoExtrap: throws DomainError if out of domain."
@inline function _check_domain(x::AbstractVector, xi::Real, ::NoExtrap)
    x_min, x_max = first(x), last(x)
    (xi < x_min || xi > x_max) && throw(DomainError(xi, "query point outside interpolation domain [$x_min, $x_max]"))
    return nothing
end

"No-op scalar domain check for non-NoExtrap modes."
@inline _check_domain(::AbstractVector, ::Real, ::AbstractExtrap) = nothing

"Vector domain check for NoExtrap: throws DomainError if any point out of domain."
@inline function _check_domain(x::AbstractVector, xi::AbstractVector{<:Real}, ::NoExtrap)
    x_min, x_max = first(x), last(x)
    xq_min, xq_max = minimum(xi), maximum(xi)
    (xq_min < x_min || xq_max > x_max) && throw(DomainError(
        xq_min < x_min ? xq_min : xq_max,
        "query point outside interpolation domain [$x_min, $x_max]"
    ))
    return nothing
end

"No-op vector domain check for non-NoExtrap modes."
@inline _check_domain(::AbstractVector, ::AbstractVector{<:Real}, ::AbstractExtrap) = nothing


