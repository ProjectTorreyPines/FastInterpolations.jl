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
"""
abstract type AbstractEvalOp end

"""
    DerivOp{N} <: AbstractEvalOp

Parametric singleton for derivative order dispatch.
`N` is the derivative order (0 = value, 1 = first derivative, etc.).

# Construction
- `DerivOp{0}()`, `DerivOp{1}()` — direct parametric construction
- `DerivOp(n::Int)` — convenience: `DerivOp(1)` → `DerivOp{1}()`
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
const EvalValue  = DerivOp{0}
const EvalDeriv1 = DerivOp{1}
const EvalDeriv2 = DerivOp{2}
const EvalDeriv3 = DerivOp{3}

# Factory constructors DerivOp(n) and DerivOp(n1, n2, ...) are in factory.jl

"""
    deriv_order(::DerivOp{N}) -> Int

Extract derivative order from a `DerivOp` instance.
"""
@inline deriv_order(::DerivOp{N}) where {N} = N

Base.show(io::IO, ::DerivOp{N}) where {N} = print(io, "DerivOp{", N, "}()")

# ========================================
# Typed Extrapolation Mode Tags
# ========================================
# Promotable Value Type
# ========================================
# Defined here (loaded early) so that eval_ops.jl and all subsequent files
# (utils.jl, nd_utils.jl, etc.) can reference it.

"""Standard Julia numeric types that should be auto-promoted in convenience wrappers."""
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

!!! warning "Union-splitting invariant"
    Keep exactly 5 concrete subtypes. Julia's compiler union-splits up to 4 types;
    ND heterogeneous tuples with all 5 types in one tuple may see dynamic dispatch.
    In practice, interpolants store concrete type parameters so this rarely matters.

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
FillExtrap(v::Complex{T}) where {T<:AbstractFloat} = FillExtrap{Complex{T}}(v)
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

Wrap extrapolation — wraps queries into the domain using modular arithmetic.
For periodic data.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=WrapExtrap())
```
"""
struct WrapExtrap <: AbstractExtrap end

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
