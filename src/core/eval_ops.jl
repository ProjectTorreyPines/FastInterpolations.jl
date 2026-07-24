# ========================================
# Evaluation Operation Types
# ========================================
# Shared types for compile-time dispatch of value/derivative evaluation.
# Must be included first - used by all interpolation files.

"""
    AbstractEvalOp

Abstract type for evaluation operations (value, derivatives).
Used for compile-time dispatch to select appropriate kernel.

Subtypes:
- `DerivOp{0}` (alias `EvalValue`): Evaluate function value f(x)
- `DerivOp{1}` (alias `EvalDeriv1`): Evaluate first derivative f'(x)
- `DerivOp{2}` (alias `EvalDeriv2`): Evaluate second derivative f''(x)
- `DerivOp{3}` (alias `EvalDeriv3`): Evaluate third derivative f'''(x)
- `DerivOp{N}` for N ≥ 4: Returns zero (beyond polynomial degree of supported interpolants)
"""
abstract type AbstractEvalOp end

"""
    DerivOp{N} <: AbstractEvalOp

Parametric singleton for derivative order dispatch.
`N` is the derivative order (0 = value, 1 = first derivative, etc.).
Any non-negative integer is valid; orders beyond the interpolant's polynomial
degree (e.g., `DerivOp{4}` for cubic) return zero automatically.

# Construction
- `DerivOp{0}()`, `DerivOp{1}()` — direct parametric construction
- `DerivOp(n::Int)` — convenience: `DerivOp(1)` → `DerivOp{1}()` (any n ≥ 0)
- `DerivOp(n1, n2, ...)` — ND: `DerivOp(1, 0)` → `(DerivOp{1}(), DerivOp{0}())`

# Backward-compatible aliases
- `EvalValue  = DerivOp{0}`
- `EvalDeriv1 = DerivOp{1}`
- `EvalDeriv2 = DerivOp{2}`
- `EvalDeriv3 = DerivOp{3}`

# Examples
```julia
# 1D scalar evaluation
itp(x; deriv=DerivOp(1))        # first derivative
itp(x; deriv=EvalDeriv2())      # second derivative (using alias)

# ND mixed partials
itp(q; deriv=DerivOp(1, 0))     # ∂f/∂x
itp(q; deriv=DerivOp(0, 2))     # ∂²f/∂y²
```
"""
struct DerivOp{N} <: AbstractEvalOp end

# Backward-compatible const aliases (zero changes needed in kernel code)
const EvalValue = DerivOp{0}
const EvalDeriv1 = DerivOp{1}
const EvalDeriv2 = DerivOp{2}
const EvalDeriv3 = DerivOp{3}

# Factory constructors DerivOp(n) and DerivOp(n1, n2, ...) are in factory.jl

"""
    deriv_order(::DerivOp{N}) -> Int
    deriv_order(::Type{DerivOp{N}}) -> Int

Extract derivative order from a `DerivOp` instance or type.
The type-level method is used inside `@generated` functions where only types are available.
"""
@inline deriv_order(::DerivOp{N}) where {N} = N
@inline deriv_order(::Type{DerivOp{N}}) where {N} = N

Base.show(io::IO, ::DerivOp{N}) where {N} = print(io, "DerivOp{", N, "}()")

# ========================================
# Grid Index Wrapper (NoInterp query element)
# ========================================

"""
    GridIdx(k::Integer)

Grid-index query coordinate. Wraps an integer index for direct grid-point lookup.

After resolution (internal), carries both the index and the grid coordinate value.
Search functions short-circuit when they see a `GridIdx` — zero search cost.

`GridIdx <: Real`: it flows through `Tuple{Vararg{Number, N}}` dispatch transparently.
Before resolution, `val = NaN` — a poison sentinel that propagates visibly if
`_resolve_grididx` is ever skipped. After resolution, `val` holds the grid coordinate
and arithmetic auto-promotes via `promote_rule` (stripping the `GridIdx` wrapper).

# Examples
```julia
# One-shot: slice axis 2 at index 5, interpolate axis 1
interp((x, y), data, (0.5, GridIdx(5)); method=(CubicInterp(), NoInterp()))

# Interpolant: query-time slicing
itp = interp((x, y), data; method=(CubicInterp(), NoInterp()))
itp((0.5, GridIdx(5)))

# GridIdx works on ANY axis (not just NoInterp):
itp_hetero = interp((x, y), data; method=(CubicInterp(), LinearInterp()))
itp_hetero((0.5, GridIdx(10)))   # search short-circuited on axis 2
```
"""
struct GridIdx{T <: Real} <: Real
    idx::Int
    val::T
    function GridIdx(i::Integer)
        i >= 1 || throw(ArgumentError("GridIdx index must be ≥ 1, got $i"))
        return new{Float64}(Int(i), NaN64)
    end
    function GridIdx{T}(i::Int, v::T) where {T <: Real}
        return new{T}(i, v)
    end
end

Base.show(io::IO, g::GridIdx) = print(io, "GridIdx(", g.idx, ")")

# GridIdx <: Real: arithmetic works transparently via promotion.
# promote(GridIdx{T}, S) → promote_type(T, S), stripping the GridIdx wrapper.
# Zero overhead — LLVM compiles to identical code as manual g.val extraction.
Base.promote_rule(::Type{GridIdx{T}}, ::Type{S}) where {T, S <: Real} = promote_type(T, S)
Base.convert(::Type{T}, g::GridIdx) where {T <: Number} = convert(T, g.val)
Base.float(g::GridIdx) = float(g.val)
(::Type{T})(g::GridIdx) where {T <: AbstractFloat} = T(g.val)

"""
    _resolve_grididx(q, grid) -> resolved coordinate

Resolve a bare `GridIdx(k)` to `GridIdx{T}(k, grid[k])`.
`Real` values pass through unchanged.
"""
# Coordinate passthrough: unbounded (duck grids); `GridIdx` stays more specific.
@inline _resolve_grididx(q, ::AbstractVector) = q
@inline function _resolve_grididx(g::GridIdx, grid::AbstractVector{Tg}) where {Tg}
    @boundscheck (1 <= g.idx <= length(grid) || _throw_grididx_oob_resolve(g.idx, length(grid)))
    return @inbounds GridIdx{Tg}(g.idx, grid[g.idx])
end

@noinline _throw_grididx_oob_resolve(idx, n) =
    throw(ArgumentError("GridIdx index $idx out of range 1:$n"))

# ========================================
# Typed Extrapolation Mode Tags
# ========================================
# Promotable Value Type
# ========================================
# Defined here (loaded early) so that eval_ops.jl and all subsequent files
# (utils.jl, nd_utils.jl, etc.) can reference it.

"""
Standard Julia numeric types that should be auto-promoted in convenience wrappers.

This is the **eager-promotion whitelist** (which carrier types `_promote_itp_inputs`
floats to the grid field). It is **not** a wrap-safety predicate: carriers outside it
(`FixedPoint`/`N0f8`, `Gray{N0f8}`, AD `Dual`) are intentionally kept un-promoted, so
arithmetic correctness for them must be handled per-site by `_fielddiff`/`_fieldsum`
— do not gate divided-difference correctness on membership here.
"""
const _PromotableValue = Union{Integer, AbstractFloat, Rational, Complex}


# ========================================
#
# Compile-time type tags for extrapolation mode selection.
# Zero-cost type dispatch at the API boundary.
#
# 1D: Structs store E<:AbstractExtrap, dispatch on concrete subtypes
# ND: Structs store NTuple{N,AbstractExtrap}, resolved via _resolve_extrap_nd

"""
    AbstractExtrap

Abstract type for typed extrapolation mode specification.
Use concrete subtypes at the API boundary for compile-time dispatch.

# Concrete subtypes
- [`NoExtrap`](@ref): Throw `DomainError` for out-of-domain queries
- [`ClampExtrap`](@ref): Clamp to nearest boundary value
- [`FillExtrap`](@ref): Return a user-specified constant for out-of-domain queries
- [`ExtendExtrap`](@ref): Extend interpolation polynomial beyond domain
- [`WrapExtrap`](@ref): Wrap queries into domain (periodic)
- [`InBounds`](@ref): Caller guarantees in-domain; skip domain checks (advanced/internal)

See also: `ConstExtrap()` (deprecated factory, forwards to `ClampExtrap()`).
"""
abstract type AbstractExtrap end

"""
    NoExtrap <: AbstractExtrap

No extrapolation — throws `DomainError` for out-of-domain queries.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=NoExtrap())
```
"""
struct NoExtrap <: AbstractExtrap end

"""
    ClampExtrap <: AbstractExtrap

Constant extrapolation — clamps to nearest boundary value for out-of-domain queries.

Returns `y[1]` for queries below domain, `y[end]` for queries above domain.
In ND, derivatives along OOB axes are zero; orthogonal derivatives are computed
at the clamped boundary point.

# Example
```julia
itp = cubic_interp(x, y; extrap=ClampExtrap())  # clamp to boundary values
itp = cubic_interp(x, y; extrap=ClampExtrap())    # equivalent
```
"""
struct ClampExtrap <: AbstractExtrap end

"""
    FillExtrap{T} <: AbstractExtrap

Fill extrapolation — returns a user-specified constant value for out-of-domain queries.
All derivatives are zero when out-of-domain (constant function).

Standard numerics auto-promote to float; duck types (SVector, etc.) are stored as-is.
Follows the same pattern as `Deriv1{Tv}`.

# Examples
```julia
itp = cubic_interp(x, y; extrap=FillExtrap(NaN))      # NaN outside domain
itp = cubic_interp(x, y; extrap=FillExtrap(0.0))       # zero outside domain
itp = cubic_interp(x, y; extrap=FillExtrap(; fill_value=0))  # kwarg form
```
"""
struct FillExtrap{T} <: AbstractExtrap
    fill_value::T
end
# Outer constructors: standard numerics auto-promote to float
# Custom types fall through to Julia's auto-generated FillExtrap(v) = FillExtrap{typeof(v)}(v)
FillExtrap(v::Real) = FillExtrap{typeof(float(v))}(float(v))
FillExtrap(v::Complex{T}) where {T <: AbstractFloat} = FillExtrap{Complex{T}}(v)
# Kwarg convenience
FillExtrap(; fill_value) = FillExtrap(fill_value)

# Internal union for dispatch where ClampExtrap and FillExtrap share a code path
# (e.g., 1D OOB check + return constant, _handle_axis_extrap coordinate clamping).
const _ClampOrFill = Union{ClampExtrap, FillExtrap}

# ConstExtrap backward-compatible factory (deprecated)
"""
    ConstExtrap()

Deprecated factory function. Use `ClampExtrap()` directly instead.

# Examples
```julia
ConstExtrap()  # → ClampExtrap() (with deprecation warning)
```
"""
function ConstExtrap()
    Base.depwarn("`ConstExtrap()` is deprecated, use `ClampExtrap()` instead.", :ConstExtrap)
    return ClampExtrap()
end

"""
    _promote_extrap(e::AbstractExtrap, ::Type{Tv}) -> AbstractExtrap

Promote a `FillExtrap` fill value to match the interpolant's value type `Tv`.
Mirrors the `_PromotableValue` two-tier pattern used for grid/value promotion:

- `_PromotableValue` fill values (Integer, AbstractFloat, Rational, Complex): auto-converted
  via `convert(Tv, e.fill_value)` at construction time — always safe for standard numerics.
- Duck-type fill values (SVector, Dual, etc.): passed through unchanged. The caller is
  responsible for providing a fill value whose type is already compatible with `Tv`.
- All non-FillExtrap types (ClampExtrap, NoExtrap, etc.): passed through unchanged.
"""
@inline _promote_extrap(e::FillExtrap{<:_PromotableValue}, ::Type{Tv}) where {Tv} =
    FillExtrap{Tv}(convert(Tv, e.fill_value))
@inline _promote_extrap(e::FillExtrap, ::Type) = e
@inline _promote_extrap(e::AbstractExtrap, ::Type) = e

"""
    ExtendExtrap <: AbstractExtrap

Extension extrapolation — extends the interpolation polynomial beyond the domain.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=ExtendExtrap())
```
"""
struct ExtendExtrap <: AbstractExtrap end

"""
    WrapExtrap <: AbstractExtrap

Wrap extrapolation — wraps queries into the closed domain `[first(x), last(x)]`
using modular arithmetic. For periodic data.

Closed-domain convention: `xq == last(x)` is an in-domain boundary query
(returns the right-corner value, e.g. `y[end]` for non-periodic data); only
strictly-OOB queries take the `mod()` path. Matches `ClampExtrap`/`FillExtrap`'s
closed convention. `:inclusive` PeriodicBC: forward **value** is invariant
(validated `y[1] ≈ y[end]`), but **adjoint** sensitivity at `xq == last(x)`
now scatters to slot `n` instead of slot `1` — delta-equivalent under the
`:inclusive` cycle constraint, but observably different if downstream code
does not enforce `y[1] == y[end]` on `f_bar`. `:exclusive` PeriodicBC is
fully invariant (forward + adjoint) via the seam-aware
`_ExclusivePeriodicAxis.search_interval` returning `idx_R = 1` at `xq >= inner[n]`.

Tag struct with no fields: the wrap domain is read directly from the axis at
query time via `first(x)` / `last(x)`. After the surface-API axis resolution
(`_resolve_axis` / `_cache_axis` in `periodic_axis.jl`), every supported axis
type — plain `Vector`, `AbstractRange`, `_CachedRange`, `_ExclusivePeriodicAxis` —
exposes `first/last` that already corresponds to the canonical wrap domain
(including `:exclusive` periodic, where `_ExclusivePeriodicAxis` reports
`last(g) = inner[1] + period`).

# Example
```julia
# Canonical usage — domain auto-configured from axis:
itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive, period=2π))

# Standalone WrapExtrap on any sorted axis:
itp = linear_interp(x, y; extrap=WrapExtrap())
```
"""
struct WrapExtrap <: AbstractExtrap end

# Backward-compat: previous API was `WrapExtrap(x)` materializing the wrap
# domain `[first(x), last(x)]` into the struct's `_x_min`/`_x_max` fields.
# After the tag-struct refactor (axis IS the source of truth for the wrap
# domain), the axis-passing form is redundant — the kernel reads `(first(x),
# last(x))` directly via `_wrap_to_domain(xq, x)`. This shim accepts and
# discards the axis so existing call sites and tests keep compiling.
@inline WrapExtrap(::AbstractVector) = WrapExtrap()

"""
    InBounds <: AbstractExtrap
    InBounds(; first = :inclusive, last = :inclusive)

Caller guarantees all queries are within the interpolation domain.
Skips domain validation for maximum performance.

Used internally by vector loops after batch `_check_domain` validation
(NoExtrap → InBounds conversion), and available to advanced users who
have pre-validated their query points.

The endpoint keywords (each `:inclusive` or `:exclusive`) narrow the promised
interval at the type level (`InBounds{First, Last}`):

| extrap                                           | caller promises              |
|:-------------------------------------------------|:-----------------------------|
| `InBounds()`                                     | `first(x) ≤ xq ≤ last(x)`    |
| `InBounds(last = :exclusive)`                    | `first(x) ≤ xq < last(x)`    |
| `InBounds(first = :exclusive)`                   | `first(x) < xq ≤ last(x)`    |
| `InBounds(first = :exclusive, last = :exclusive)`| `first(x) < xq < last(x)`    |

Like `Base.@inbounds`, violating the promise is undefined for the optimized
paths: the result may be a wrong cell, an out-of-bounds read, or a
`BoundsError`. A stricter promise can only be exploited, never required —
paths without a dedicated lean arm safely treat it as the closed contract.
Currently `last = :exclusive` selects a no-top-cap direct search on unit-step
range axes; `first` is accepted for API completeness and future use.

# Example
```julia
# Skip domain check when you know queries are in-domain
linear_interp(x, y, xq; extrap=InBounds())

# Queries generated in [first(x), last(x)) — never touch the right endpoint
linear_interp(x, y, xq; extrap=InBounds(last = :exclusive))

# On a built interpolant, `InBounds` is the only per-call `extrap` override: it
# opts a query into the fast path without rebuilding (any other mode errors —
# the stored extrapolation contract is not swappable per call).
itp = linear_interp(x, y; extrap=ClampExtrap())
itp(xq; extrap=InBounds())   # opt into the in-domain fast path
```
"""
struct InBounds{First, Last} <: AbstractExtrap
    function InBounds{First, Last}() where {First, Last}
        First isa Symbol || error("InBounds type parameter First must be a Symbol")
        Last isa Symbol || error("InBounds type parameter Last must be a Symbol")
        First in (:inclusive, :exclusive) ||
            error("InBounds type parameter First must be :inclusive or :exclusive")
        Last in (:inclusive, :exclusive) ||
            error("InBounds type parameter Last must be :inclusive or :exclusive")
        return new{First, Last}()
    end
end

# Keyword constructor with validation — mirrors `PeriodicBC(; endpoint=...)`. The
# zero-arg `InBounds()` call constant-folds to the closed singleton, so the ~15
# internal promotion sites (`_check_domain` returns, Clamp/Fill in-domain
# delegations) stay zero-cost.
function InBounds(; first::Symbol = :inclusive, last::Symbol = :inclusive)
    first in (:inclusive, :exclusive) ||
        throw(ArgumentError("first must be :inclusive or :exclusive, got :$first"))
    last in (:inclusive, :exclusive) ||
        throw(ArgumentError("last must be :inclusive or :exclusive, got :$last"))
    return InBounds{first, last}()
end

# Kwarg-form show (`GridIdx`/`DerivOp` precedent): round-trips as `InBounds()` /
# `InBounds(last = :exclusive)` instead of raw type parameters.
function Base.show(io::IO, ::InBounds{First, Last}) where {First, Last}
    parts = String[]
    First === :inclusive || push!(parts, "first = :$First")
    Last === :inclusive || push!(parts, "last = :$Last")
    return print(io, "InBounds(", join(parts, ", "), ")")
end

# ========================================
# Typed Side Selection Tags (Constant Interpolation)
# ========================================
#
# Compile-time type tags for side mode selection.
# Zero-cost type dispatch — no macros needed with proper types.
#
# 1D: Structs store SD<:AbstractSide, dispatch on concrete subtypes
# ND: Structs store Tuple{Vararg{AbstractSide, N}}, resolved via _resolve_side_nd

"""
    AbstractSide

Abstract type for side selection mode in constant interpolation.
Determines which neighbor value to use at non-grid-point locations.

# Concrete subtypes
- [`NearestSide`](@ref): Nearest neighbor with left tie-breaking at midpoint
- [`LeftSide`](@ref): Always use left (floor) value
- [`RightSide`](@ref): Use right (ceiling) value, except at grid points

!!! info "Type dispatch mechanism"
    Interpolant structs store `side` as a type parameter `SD<:AbstractSide`,
    so dispatch is fully monomorphized at compile time — zero overhead.
    For oneshot paths where `side` appears in a Union, Julia union-splits
    automatically (3 concrete subtypes < 4 limit).
"""
abstract type AbstractSide end

"""
    NearestSide <: AbstractSide

Nearest-neighbor side selection with left tie-breaking at midpoint.
Returns left value if distance <= h/2, right value otherwise.

# Example
```julia
itp = constant_interp(x, y; side=NearestSide())
```
"""
struct NearestSide <: AbstractSide end

"""
    LeftSide <: AbstractSide

Left-continuous (floor) side selection. Always returns the left boundary value.

# Example
```julia
itp = constant_interp(x, y; side=LeftSide())
```
"""
struct LeftSide <: AbstractSide end

"""
    RightSide <: AbstractSide

Right-continuous (ceiling) side selection.
Returns right value except at grid points (where it returns left value).

# Example
```julia
itp = constant_interp(x, y; side=RightSide())
```
"""
struct RightSide <: AbstractSide end
