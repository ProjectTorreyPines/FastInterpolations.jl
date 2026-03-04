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
- [`ConstExtrap`](@ref): Clamp to nearest boundary value, or return a user-specified fill value
- [`ExtendExtrap`](@ref): Extend interpolation polynomial beyond domain
- [`WrapExtrap`](@ref): Wrap queries into domain (periodic)

!!! warning "Union-splitting invariant"
    Keep exactly 4 concrete subtypes. Julia's compiler union-splits up to 4 types
    on hot paths; adding a 5th subtype would cause dynamic dispatch and allocation
    in all oneshot/eval code paths.
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
    ConstExtrap{T} <: AbstractExtrap

Constant extrapolation — returns a fixed value for out-of-domain queries.

- `ConstExtrap()`: clamp to nearest boundary value (`y[1]` / `y[end]`)
- `ConstExtrap(value)`: return `value` for all out-of-domain queries

The type parameter `T` is the value type (`Nothing` for boundary clamp, concrete type for fill value).
Follows the same pattern as `Deriv1{Tv}` — standard numerics auto-promote to float,
duck types (SVector, etc.) are stored as-is.

# Examples
```julia
itp = cubic_interp(x, y; extrap=ConstExtrap())        # clamp to boundary
itp = cubic_interp(x, y; extrap=ConstExtrap(NaN))      # NaN outside domain
itp = cubic_interp(x, y; extrap=ConstExtrap(0.0))      # zero outside domain
itp = cubic_interp(x, y; extrap=ConstExtrap(; value=NaN))  # kwarg form
```
"""
struct ConstExtrap{T} <: AbstractExtrap
    value::T
end
# Outer constructors: standard numerics auto-promote to float
# Custom types fall through to Julia's auto-generated ConstExtrap(v) = ConstExtrap{typeof(v)}(v)
ConstExtrap(v::Real) = ConstExtrap{typeof(float(v))}(float(v))
ConstExtrap(v::Complex{T}) where {T<:AbstractFloat} = ConstExtrap{Complex{T}}(v)
# Kwarg convenience (also handles no-arg: ConstExtrap() → value=nothing → ConstExtrap{Nothing})
ConstExtrap(; value=nothing) = ConstExtrap(value)
# Conversion constructor (for _promote_extrap, mirrors Deriv1{Tv}(bc::Deriv1))
ConstExtrap{T}(e::ConstExtrap) where {T} = ConstExtrap{T}(convert(T, e.value))

"""
    _promote_extrap(e::AbstractExtrap, ::Type{Tv}) -> AbstractExtrap

Promote a `ConstExtrap` fill value to match the interpolant's value type `Tv`.
Mirrors `_promote_pointbc` — converts fill value via `convert(Tv, e.value)` at
construction time so eval returns the correct type with zero overhead.

Non-ConstExtrap types and `ConstExtrap{Nothing}` (boundary clamp) pass through unchanged.
Raises `InexactError`/`MethodError` at construction time if `Tv` cannot represent `e.value`.
"""
@inline _promote_extrap(e::ConstExtrap{Nothing}, ::Type) = e
@inline _promote_extrap(e::ConstExtrap, ::Type{Tv}) where {Tv} = ConstExtrap{Tv}(convert(Tv, e.value))
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
