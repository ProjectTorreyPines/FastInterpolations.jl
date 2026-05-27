# Internal utility functions for FastInterpolations.jl

# ── @noinline throw helpers (keep cold error paths out of hot code) ──

@noinline _throw_length_mismatch(na::Int, nb::Int, a::String = "x", b::String = "y") =
    throw(ArgumentError("$a and $b must have same length, got $na and $nb"))

@noinline _throw_grid_too_small(n::Int) =
    throw(ArgumentError("x must have at least 2 elements, got $n"))

# ========================================
# Interval Search (IN search.jl)
# ========================================
# Interval search functions are defined in src/core/search.jl:
#   - _search_direct: O(1) direct calculation for uniform grids (AbstractRange)
#   - _search_binary: O(log n) binary search for non-uniform grids (AbstractVector)
#   - _search_interval: dispatcher that routes to the appropriate implementation

# ========================================
# Type Conversion Helpers
# ========================================

# _to_float for Range types (→ _CachedRange) is defined in cached_range.jl.
# _to_float_adding_endpoint is also defined in cached_range.jl.

"""
    _to_float(x::AbstractVector{T}, ::Type{T}) where {T}

Identity conversion — return as-is when element type already matches target type.
Handles both standard Float types and duck types (e.g. `Vector{Dual{Float64}}`
with target `Dual{Float64}`). Zero-allocation in all cases.
"""
_to_float(x::AbstractVector{T}, ::Type{T}) where {T} = x

"""
    _to_float(x::AbstractVector, ::Type{T}) where {T}

Convert a Vector to target type (element-wise broadcast).
For standard numerics (Int→Float64, Float32→Float64), emits a one-time warning
since this allocates a new vector. For duck types (e.g. `Dual{Int}→Dual{Float64}`),
the same broadcast applies — `T.(x)` dispatches to ForwardDiff's `convert`.
"""
function _to_float(x::AbstractVector, ::Type{T}) where {T}
    _warn_type_conversion(T)
    return T.(x)
end

@noinline function _warn_type_conversion(::Type{T}) where {T}
    @warn "Non-matching vector element type detected — allocating type conversion. " *
        "For zero-allocation, pre-convert your data: `x_typed = $T.(x)`" maxlog = 1
    return nothing
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
@inline _real_eltype(::Type{T}) where {T <: Real} = T
@inline _real_eltype(::Type{Complex{T}}) where {T <: Real} = T
# Duck-typing fallback: custom types return themselves (no float base extraction)
@inline _real_eltype(::Type{T}) where {T} = T

"""
    _promote_grid_float(::Type{Tg}, ::Type{Tv}) -> Type{<:AbstractFloat}

Compute the grid float type, optionally widened by value precision.

- **Promotable values** (`_PromotableValue`): grid widens to accommodate value precision.
  Example: `Float32` grid + `Float64` values → `Float64` grid.
  This prevents per-element conversion overhead in hot evaluation paths.
- **Duck types** (Dual, Measurement, etc.): grid uses only its own type.
  Value type must NOT contaminate the grid (e.g., grid coordinates should never
  carry derivative partials from `ForwardDiff.Dual`).

# Examples
```julia
_promote_grid_float(Int, Float64)    # → Float64 (standard widening)
_promote_grid_float(Float32, Int)    # → Float32 (Int doesn't widen Float32)
_promote_grid_float(Float32, Float64)# → Float64 (value precision wins)
_promote_grid_float(Float64, Dual)   # → Float64 (duck: grid ignores Dual)
_promote_grid_float(Int, Dual)       # → Float64 (duck: float(Int) only)
```
"""
@inline function _promote_grid_float(::Type{Tg}, ::Type{Tv}) where {Tg, Tv}
    if Tv <: _PromotableValue
        return float(promote_type(Tg, _real_eltype(Tv)))
    else
        return float(Tg)
    end
end

"""
    _value_type(::Type{Ty}, ::Type{Tg}) -> Type

Determine the output value type from y element type and grid type.
- Standard numerics (Integer, AbstractFloat, Rational, Complex) → Tg or Complex{Tg}
- Duck types (Dual, Measurement, etc.) → preserved as-is
"""
@inline _value_type(::Type{T}, ::Type{Tg}) where {T <: _PromotableValue, Tg <: AbstractFloat} = Tg
@inline _value_type(::Type{Complex{T}}, ::Type{Tg}) where {T <: Real, Tg <: AbstractFloat} = Complex{Tg}
# Duck-typing fallback for Tv: custom value types preserved as-is
@inline _value_type(::Type{T}, ::Type{Tg}) where {T, Tg <: AbstractFloat} = T
# Duck-typing fallback for Tg: when grid is duck-typed (Dual, Measurement, etc.),
# values are not promoted to grid type (no grid-parameter partials in y).
@inline _value_type(::Type{T}, ::Type{Tg}) where {T, Tg} = T

# Inference probe for `_output_eltype` duck fallback. Standard kernels
# (Linear/Cubic/Quadratic/Hermite) produce `Tv + α·Tv` shapes; Constant's
# `Tv * one(Tq)` lives in the same promotion space.
@inline _kernel_shape_op(yv, q) = yv * q + yv

"""
    _output_eltype(::Type{Tv}, types...) -> Type

Generic output-eltype probe via the universal arithmetic kernel shape
`y*q + y` (`_kernel_shape_op`). Currently used by:

- Internal coefficient eltype (Cubic `Tz`, Quadratic `Tc`).
- Adjoint allocators (`adjoint_protocol.jl`).
- Hetero ND legacy paths and a few series callsites.

Concrete `promote_type` gets Int→Float upgrade (arithmetic kernels divide
— Int chains widen naturally); duck carriers (e.g. `SVector × Dual`) fall
through to `Base.promote_op` on `_kernel_shape_op`, with final fallback
to `Tv` if the op is undefined.

For method-aware output-buffer sizing (Linear/Cubic/Quadratic/Constant/
Hermite), prefer the kernel-op overload below — it predicts the method's
exact kernel return type via `Base.promote_op`.
"""
@inline function _output_eltype(::Type{Tv}, types::Type...) where {Tv}
    Tr = promote_type(Tv, types...)
    if isconcretetype(Tr)
        return (Tr <: _PromotableValue && !(Tr <: AbstractFloat)) ? float(Tr) : Tr
    end
    Tq = length(types) == 0 ? Tv : promote_type(types...)
    Top = Base.promote_op(_kernel_shape_op, Tv, Tq)
    (Top === Union{} || Top === Any) && return Tv
    return Top
end

"""
    _output_eltype(kernel_op, ::Type{Tv}, types...) -> Type

Method-aware output element type via `Base.promote_op` on the method's own
kernel shape. Lets Julia inference predict the kernel's exact return type
— no hand-coded Float upgrade, no `_PromotableValue` enumeration. Use this
overload from a method that declares its kernel shape (e.g., Constant's
`_constant_kernel_shape(xL, yv, xq) = yv * one(xq - xL)`).
"""
@inline function _output_eltype(kernel_op::F, ::Type{Tv}, types::Type...) where {F, Tv}
    Top = Base.promote_op(kernel_op, Tv, types...)
    (Top === Union{} || Top === Any) && return Tv
    return Top
end

# Shared kernel shape for arithmetic methods (Linear/Cubic/Quadratic/Hermite):
# `y + y * (dL/h)` captures the division-by-`h` that drives the Int→Float
# widening — Julia inference predicts the exact kernel return type. Args are
# ordered `(Tg, Tv, Tq)` so callers use `_output_eltype(shape, Tg, Tv, Tq)`,
# matching the codebase's standard type-parameter order.
@inline _arithmetic_kernel_shape(h, yv, dL) = yv + yv * (dL / h)

"""
    _promote_query_eltype(::Type{Tv}, q::Tuple) -> Type

Compute the promoted output element type by folding `promote_type` over `Tv`
and the element types of the tuple `q`. Recursive on `Base.tail` for compile-time
type specialization — each step sees concrete types and collapses to a constant
through Julia's normal inference (no @generated body needed, which would suffer
from world-age issues when promotion rules for `q`'s types are defined in an
extension module loaded after FastInterpolations).

Used by the OnTheFly ND `_collapse_dims` entry points where the pool buffer
type must include the query eltype (for ForwardDiff.Dual compatibility) but
the computation must remain zero-cost for plain-Float64 queries.
"""
@inline _promote_query_eltype(::Type{Tv}, ::Tuple{}) where {Tv} = Tv
@inline function _promote_query_eltype(::Type{Tv}, q::Tuple) where {Tv}
    return _promote_query_eltype(promote_type(Tv, typeof(first(q))), Base.tail(q))
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
@inline function _promote_value_type(y::AbstractVector{Tv_raw}, ::Type{Tg}) where {Tv_raw, Tg <: AbstractFloat}
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
- Promotes y via `_promote_value_type` (handles numeric widening)

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
    ) where {TX, TY}
    Tg = _promote_grid_float(TX, TY)
    x_typed = _to_float(x, Tg)
    # Value promotion: only when BOTH the grid target AND the value type are
    # standard numerics. When Tg is a duck type (e.g. Dual), promoting y to Tg
    # would inject derivative partials into values that carry none — semantically
    # wrong. In that case y passes through unchanged, same as the Tv duck path.
    if TY <: _PromotableValue && Tg <: AbstractFloat
        _, y_typed = _promote_value_type(y, Tg)
        return x_typed, y_typed
    else
        return x_typed, y
    end
end

"""
    _promote_itp_inputs(x, y, xq::AbstractVector) -> (x_typed, y_typed, xq_typed)

Promote grid (x), values (y), and vector query (xq) to compatible Float types.

# Arguments
- `x`: Grid coordinates (any Real type)
- `y`: Values at grid points
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
    ) where {TX, TY, TQ <: Real}
    x_typed, y_typed = _promote_itp_inputs(x, y)
    xq_typed = _promote_query_typed(xq, eltype(x_typed))
    return x_typed, y_typed, xq_typed
end

# ========================================
# Query & Adjoint Promotion Helpers
"""
    _store_grid(x, ::Type{Tg}) -> stored grid

Single-allocation grid storage for interpolant constructors.
- `AbstractVector`: `_convert_copy(x, Tg)` — promote + copy in one step (no double alloc)
- `AbstractRange`: `_to_float(x, Tg)` — `_CachedRange` (stack alloc, preserves O(1) search)

Note: this is the lightweight normalization — Vector inputs stay as plain
`Vector{Tg}`, no spacing cache. Method-family constructors that want
per-cell h/inv_h caching wrap explicitly via `_CachedVector(_store_grid(x, Tg))`
during the build step (so the cache miss path pays the wrap cost just once,
without inflating cache-hit lookups).
"""
@inline _store_grid(x::AbstractVector, ::Type{Tg}) where {Tg} = _convert_copy(x, Tg)
@inline _store_grid(x::AbstractRange, ::Type{Tg}) where {Tg} = _to_float(x, Tg)

"""
    _convert_copy(v::AbstractVector, ::Type{T}) -> Vector{T}

Copy with optional type conversion in a single allocation.
Same-type: equivalent to `copy(v)`. Different-type: equivalent to `Vector{T}(v)`.

Used in interpolant inner constructors to merge promotion + immutability copy.
"""
@inline _convert_copy(v::AbstractVector{T}, ::Type{T}) where {T} = copy(v)
@inline _convert_copy(v::AbstractVector, ::Type{T}) where {T} = Vector{T}(v)

# ========================================

"""
    _promote_query_typed(xq::AbstractVector, ::Type{Tg}) -> AbstractVector

Widen query vector to `promote_type(Tg, Tq)` — never narrows query precision.
Duck-typed queries (`Dual`, `Measurement`, …) pass through unchanged.
"""
@inline function _promote_query_typed(xq::AbstractVector{Tq}, ::Type{Tg}) where {Tq <: Real, Tg}
    if Tq <: _PromotableValue
        return _to_float(xq, promote_type(Tg, Tq))
    else
        return xq
    end
end

"""
    _promote_adjoint_inputs(x, xq) -> (x_promoted, xq_promoted, Tg)

Promote grid and query vectors for adjoint construction.

Shared pattern across all 1D adjoint builders: cubic, linear, quadratic,
constant, pchip, cardinal, akima.
"""
@inline function _promote_adjoint_inputs(
        x::AbstractVector,
        xq::AbstractVector
    )
    Tg = _promote_grid_float(eltype(x), eltype(xq))
    x_p = _to_float(x, Tg)
    xq_p = _promote_query_typed(xq, Tg)
    return x_p, xq_p, Tg
end


# ========================================
# AD Support Helpers
# ========================================


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
@inline _extract_primal(x) = x  # identity fallback; ForwardDiff ext specializes for Dual
# GridIdx <: Real: _extract_primal(g::GridIdx) returns g (identity fallback).

"""
    _effective_autocache(autocache, Tg) -> Bool

Disable autocache for non-standard grid types (e.g. ForwardDiff.Dual).
Enabled for `_PromotableValue` types (AbstractFloat, Integer, Rational) which
have stable grid identity (cache hit rate > 0). Dual grids are ephemeral
(created fresh each AD call), so autocache is disabled for them.
Resolves at specialization time — zero runtime cost on the Float hot path.
"""
@inline _effective_autocache(ac::Bool, ::Type{Tg}) where {Tg} = ac & (Tg <: _PromotableValue)
# Arithmetic then auto-promotes GridIdx → g.val via promote_rule.

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
# For AbstractFloat queries on AbstractFloat grids: preserve precision using wider type
@inline _promote_for_anchor(xq::Tq, ::Type{Tg}) where {Tq <: AbstractFloat, Tg <: AbstractFloat} = convert(promote_type(Tq, Tg), xq)
# For other Real queries (Int, Rational) on AbstractFloat grids: convert to grid type
@inline _promote_for_anchor(xq::Tq, ::Type{Tg}) where {Tq <: Real, Tg <: AbstractFloat} = Tg(xq)
# For duck grids (e.g. Dual): keep xq as-is. Kernel arithmetic auto-promotes via Julia's
# type system (Float * Dual → Dual). Converting xq to Dual would inject zero partials.
@inline _promote_for_anchor(xq, ::Type{Tg}) where {Tg} = xq


# ========================================
# Domain Validation Helpers
# ========================================

@noinline _throw_domain_error(xi, x_min, x_max) =
    throw(DomainError(xi, "query point outside interpolation domain [$x_min, $x_max]"))

"Scalar domain check for NoExtrap: throws DomainError if out of domain."
@inline function _check_domain(x::AbstractVector, xi::Real, ::NoExtrap)
    x_min, x_max = _extract_primal(first(x)), _extract_primal(last(x))
    (xi < x_min || xi > x_max) && _throw_domain_error(xi, x_min, x_max)
    return nothing
end

# _CachedRange: use domain_lo/domain_hi (wider bracket on x86_64 fast path).
@inline function _check_domain(x::_CachedRange, xi::Real, ::NoExtrap)
    lo, hi = _extract_primal(x.domain_lo), _extract_primal(x.domain_hi)
    (xi < lo || xi > hi) && _throw_domain_error(xi, _extract_primal(x.lo), _extract_primal(x.hi))
    return nothing
end

"No-op scalar domain check for non-NoExtrap modes (including InBounds)."
@inline _check_domain(::AbstractVector, ::Real, ::AbstractExtrap) = nothing

"GridIdx is in-domain by construction (bounds-checked at resolution time)."
@inline _check_domain(::AbstractVector, ::GridIdx, ::AbstractExtrap) = nothing
# Disambiguation: GridIdx <: Real creates ambiguity with _CachedRange × NoExtrap methods.
# GridIdx always wins (in-domain by construction).
@inline _check_domain(::_CachedRange, ::GridIdx, ::NoExtrap) = nothing
@inline _check_domain(::AbstractVector, ::GridIdx, ::NoExtrap) = nothing

# ----------------------------------------
# Vector domain checks: validate batch, return InBounds() for per-element elision.
# The @boundscheck wraps only the validation logic; `return InBounds()` always
# executes to maintain type stability. With --check-bounds=no, the O(n) min/max
# scan is skipped but the extrap conversion still happens.
# ----------------------------------------

"""
Vector domain check for NoExtrap: validate batch, return `InBounds()`.

Delegates the batch in-domain test to `_is_all_inbounds`, which dispatches
on axis type (using `domain_lo/hi` for `_CachedRange`'s wider TwicePrecision
bracket on x86_64) and is partial-sign-safe under ForwardDiff. Throw
message uses `first(x)/last(x)` — exact endpoints, not the widened bracket.
"""
@inline function _check_domain(x::AbstractVector, xi::AbstractVector{<:Real}, ::NoExtrap)
    @boundscheck _is_all_inbounds(x, xi) || _throw_batch_oob(x, xi)
    return InBounds()
end

@noinline function _throw_batch_oob(x::AbstractVector, xi::AbstractVector{<:Real})
    qmin, qmax = minimum(xi), maximum(xi)
    x_min = _extract_primal(first(x))
    x_max = _extract_primal(last(x))
    _throw_domain_error(qmin < x_min ? qmin : qmax, x_min, x_max)
end

"No-op vector domain check for non-NoExtrap modes: pass-through extrap."
@inline _check_domain(::AbstractVector, ::AbstractVector{<:Real}, extrap::AbstractExtrap) = extrap

# Closed-domain batch fast path: every OOB policy (`ClampExtrap`, `FillExtrap`,
# `WrapExtrap`) treats `[first(x), last(x)]` as the in-domain interval, so they
# share one batch promotion to `InBounds()`.
@inline function _check_domain(
        x::AbstractVector, xi::AbstractVector{<:Real},
        e::Union{ClampExtrap, FillExtrap, WrapExtrap}
    )
    return _is_all_inbounds(x, xi) ? InBounds() : e
end

"""
True iff every element of `queries` lies in the closed domain
`[first(x), last(x)]`. Enables batch-level fast paths that elide per-query
domain handling (e.g. `_wrap_to_domain` for PeriodicBC, which only needs
to apply when a query is strictly outside `[first, last]`) when no
element is OOB.

Uses two `&&`-chained reductions rather than `extrema`:
- pre-1.13 `extrema` carries a (min, max) tuple dep across the loop that
  blocks LLVM auto-vectorization (~30× slower on Vector{Float64} N=1000)
- `&&` short-circuits when `minimum` is already OOB, skipping the
  `maximum` scan entirely — strictly ≤ `extrema`'s work in all cases

1.13 fixes the SIMD issue, but the short-circuit advantage remains for
the OOB slow-path, so this form stays preferred even post-1.10-LTS.
"""
# `_extract_primal` is required on `first(x)` / `last(x)` here because
# ForwardDiff's `Real <= Dual` comparison includes partial-sign tie-breaking
# at equal primals — so a Float query exactly at the boundary against a
# Dual grid endpoint can flip in/out of bounds based on the partial sign
# alone (see `test/ext/test_linear_dual_grid.jl` "Domain boundary:
# primal-based NoExtrap check (partial-independent)"). Inline calls keep
# the `&&` short-circuit intact.
@inline function _is_all_inbounds(x::AbstractVector, queries::AbstractVector{<:Real})
    isempty(queries) && return true
    return minimum(queries) >= _extract_primal(first(x)) &&
        maximum(queries) <= _extract_primal(last(x))
end

# `_CachedRange`: `domain_lo`/`domain_hi` (≈1 ULP wider than `lo`/`hi` on
# x86_64 TwicePrecision normalization) for safe bounds. Fields are typed
# `T <: AbstractFloat` per the struct, so no `_extract_primal` is needed.
@inline function _is_all_inbounds(x::_CachedRange, queries::AbstractVector{<:Real})
    isempty(queries) && return true
    return minimum(queries) >= x.domain_lo && maximum(queries) <= x.domain_hi
end

# ========================================
# Extrapolation value helpers (shared by all interpolation methods)
# ========================================
# _eval_extrapolation: return the appropriate value for an OOB query with ClampExtrap/FillExtrap.
# Dispatches on op (EvalValue vs derivatives) and extrap type (Clamp vs Fill).
#
# _promote_extrap_val:  promotes an extrap result value to match kernel return type.
# _promote_extrap_zero: carrier-aware zero for the OOB deriv-zero path under
#                       flat extrapolation (Clamp/Fill, any method family).
#                       The leading `0 * val` preserves NaN/Inf at the boundary
#                       sample (IEEE: `0 * NaN = NaN`); `zero(xq) * zero(val)`
#                       carries the query carrier (Dual, etc.) into the result.
# Named _promote_extrap_val (not _promote_extrap) to avoid collision with the struct
# promoter in eval_ops.jl which promotes FillExtrap fill_value at construction time.
@inline _promote_extrap_val(val::Number, xq::Number) = val + zero(xq) * zero(val)
# AbstractArray Tv (e.g. `SVector` y) — broadcast the carrier-propagating
# pattern so scalar OOB matches in-domain kernel's `y * one(dL)` shape and
# agrees with batch path's trait-sized buffer.
@inline _promote_extrap_val(val::AbstractArray, xq::Number) = val .+ zero(xq) .* zero(eltype(val))
@inline _promote_extrap_val(val, xq) = val
@inline _promote_extrap_zero(val::Number, xq::Number) = 0 * val + zero(xq) * zero(val)
@inline _promote_extrap_zero(val::AbstractArray, xq::Number) = 0 .* val .+ zero(xq) .* zero(eltype(val))
@inline _promote_extrap_zero(val, xq) = 0 * val

# _extrap_oob_data: per-extrap "what data sits in the OOB cell".
#   ClampExtrap → `y_bnd`         (boundary y is extended into the OOB region).
#   FillExtrap  → `e.fill_value`  (fill_value is the OOB cell's data).
# `@inline` + singleton dispatch — LLVM specializes per concrete extrap type
# and dead-branch-eliminates; zero overhead on the OOB cold path.
@inline _extrap_oob_data(::ClampExtrap, y_bnd) = y_bnd
@inline _extrap_oob_data(e::FillExtrap, _) = e.fill_value

# OOB evaluation under flat extrapolation (Clamp / Fill): the OOB cell's
# data is fetched via `_extrap_oob_data` (ClampExtrap → `y_bnd`, FillExtrap
# → `fill_value`), then promoted by the op-specific kernel:
#   `EvalValue` → `data + carrier`         (value path)
#   `DerivOp`   → `0 * data + carrier`     (deriv path — 0 × OOB cell data)
# This `data → promote` split makes the deriv path's `_extrap_oob_data`
# call read naturally: we're not asking "what's the derivative" but "what's
# the cell data" — the `0 *` happens inside `_promote_extrap_zero`.
@inline _eval_extrapolation(::EvalValue, y_bnd, ext::Union{ClampExtrap, FillExtrap}, xq) =
    _promote_extrap_val(_extrap_oob_data(ext, y_bnd), xq)
@inline _eval_extrapolation(::DerivOp,   y_bnd, ext::Union{ClampExtrap, FillExtrap}, xq) =
    _promote_extrap_zero(_extrap_oob_data(ext, y_bnd), xq)
