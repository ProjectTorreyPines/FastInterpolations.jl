# Internal utility functions for FastInterpolations.jl

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

"""
    _value_type(::Type{Ty}, ::Type{Tg}) -> Type

Determine the output value type from y element type and grid type.
- Real y → Tg (promotes to grid float type)
- Complex y → Complex{Tg}
"""
@inline _value_type(::Type{T}, ::Type{Tg}) where {T<:Real, Tg<:AbstractFloat} = Tg
@inline _value_type(::Type{Complex{T}}, ::Type{Tg}) where {T<:Real, Tg<:AbstractFloat} = Complex{Tg}


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

Promote grid (x) and values (y) to compatible Float types for interpolation.

# Behavior
- Computes target grid type: `Tg = float(promote_type(TX, _real_eltype(TY)))`
- Converts x via `_to_float` (no-copy if already matching type, Range preserved)
- Converts y via `_promote_value_type` (handles Real/Complex, no-copy if matching)

# Zero-Overhead Guarantee
- `@inline` enables compiler branch elimination
- Returns inputs unchanged when types already match (zero allocation)

# Examples
```julia
x = [0.0, 1.0, 2.0]           # Float64 grid
y_int = [1, 2, 3]             # Int64 values
y_cplx = Complex{Int}[1+2im]  # Complex{Int} values

x_p, y_p = _promote_itp_inputs(x, y_int)   # y_p is Float64[]
x_p, y_p = _promote_itp_inputs(x, y_cplx)  # y_p is ComplexF64[]

# Integer grid also supported
x_int = [0, 1, 2, 3]
x_p, y_p = _promote_itp_inputs(x_int, y_cplx)  # x_p is Float64[], y_p is ComplexF64[]
```
"""
@inline function _promote_itp_inputs(
    x::AbstractVector{TX},
    y::AbstractVector{TY}
) where {TX<:Real, TY}
    Tg = float(promote_type(TX, _real_eltype(TY)))
    x_typed = _to_float(x, Tg)
    _, y_typed = _promote_value_type(y, Tg)
    return x_typed, y_typed
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

"""
    _check_domain(x, xi, ::Val{:none})

Check if scalar query point is within domain for `:none` extrapolation mode.
Throws `DomainError` if `xi` is outside `[first(x), last(x)]`.

Type-relaxed: accepts any Real query type (Int, Float32, Float64, Dual, etc.)
to avoid unnecessary type conversions at call sites.

Uses `@boundscheck` so it's skipped in `@inbounds` blocks for vector paths
that do a single upfront check via the vector dispatch.
"""
@inline function _check_domain(x::AbstractVector, xi::Real, ::Val{:none})
    x_min, x_max = first(x), last(x)
    (xi < x_min || xi > x_max) && throw(DomainError(xi, "query point outside interpolation domain [$x_min, $x_max]"))
    return nothing
end

"""
    _check_domain(x, xi, ::Val)

No-op domain check for extrapolation modes other than `:none`.
"""
@inline _check_domain(::AbstractVector, ::Real, ::Val) = nothing

"""
    _check_domain(x, xi::AbstractVector, ::Val{:none})

Vector-level domain check using minimum/maximum (faster than extrema due to SIMD).
Called once before vector loop, then scalar `_check_domain` is skipped via `@inbounds`.

Type-relaxed: accepts any Real element type to avoid unnecessary conversions.
"""
@inline function _check_domain(x::AbstractVector, xi::AbstractVector{<:Real}, ::Val{:none})
    x_min, x_max = first(x), last(x)
    # NOTE: Using minimum/maximum for potential SIMD optimization over extrema
    # extrema can be ~30x slower than minimum/maximum
    xq_min, xq_max = minimum(xi), maximum(xi)
    (xq_min < x_min || xq_max > x_max) && throw(DomainError(
        xq_min < x_min ? xq_min : xq_max,
        "query point outside interpolation domain [$x_min, $x_max]"
    ))
    return nothing
end

"""
    _check_domain(x, xi::AbstractVector, ::Val)

No-op vector domain check for extrapolation modes other than `:none`.
"""
@inline _check_domain(::AbstractVector, ::AbstractVector{<:Real}, ::Val) = nothing

# ========================================
# Validation Utilities
# ========================================
#
# Centralized validation for keyword arguments.
# @inline ensures zero overhead - compiler inlines the check.

"""
    _validate_extrap(extrap::Symbol) -> Nothing

Validate extrapolation mode symbol. Throws `ArgumentError` if invalid.

Valid options: `:none`, `:constant`, `:extension`, `:wrap`
"""
@inline function _validate_extrap(extrap::Symbol)
    extrap in (:none, :constant, :extension, :wrap) && return nothing
    throw(ArgumentError("`extrap` must be :none, :constant, :extension, or :wrap, got :$extrap"))
end

# ========================================
# Dispatch Macros (Zero-Allocation Branching)
# ========================================
#
# These macros expand to manual if-elseif blocks that avoid union-splitting issues.
# Each branch calls with a concrete Val(:literal), ensuring zero-allocation dispatch.

"""
    @_dispatch_extrap sym => varname body

Dispatch on runtime extrapolation symbol, executing body with concrete Val type.

# Arguments
- `sym => varname`: Pair of symbol variable and binding name for Val type
- `body`: Expression to execute with `varname` bound to concrete Val

# Example
```julia
@_dispatch_extrap extrap => ev begin
    _cubic_interp_impl!(output, cache, y, x_query, ev)
end
```

Expands to:
```julia
let _mode = extrap
    if _mode === :none
        ev = Val(:none)
        _cubic_interp_impl!(output, cache, y, x_query, ev)
    elseif _mode === :constant
        ...
    end
end
```
"""
macro _dispatch_extrap(pair, body)
    # Parse pair: extrap => ev becomes Expr(:call, :(=>), :extrap, :ev)
    pair.head === :call && pair.args[1] === :(=>) ||
        error("@_dispatch_extrap expects `sym => varname`, got: $pair")
    sym = pair.args[2]
    varname = pair.args[3]
    evs = esc(varname)
    quote
        let _mode = $(esc(sym))
            if _mode === :none
                $evs = Val(:none)
                $(esc(body))
            elseif _mode === :constant
                $evs = Val(:constant)
                $(esc(body))
            elseif _mode === :extension
                $evs = Val(:extension)
                $(esc(body))
            elseif _mode === :wrap
                $evs = Val(:wrap)
                $(esc(body))
            else
                throw(ArgumentError("`extrap` must be :none, :constant, :extension, or :wrap, got :$_mode"))
            end
        end
    end
end

"""
    @_dispatch_deriv deriv => op body

Dispatch on runtime deriv integer, executing body with concrete AbstractEvalOp type.

Converts `deriv::Int` (0, 1, 2) to compile-time constant `EvalValue()`, `EvalDeriv1()`,
or `EvalDeriv2()`. This creates a function barrier ensuring type stability downstream.

# Arguments
- `deriv => op`: Pair of deriv expression and symbol to bind the concrete EvalOp type
- `body`: Expression to execute with `op` bound to concrete type

# Example
```julia
@_dispatch_deriv deriv => op begin
    _cubic_interp_impl(x, y, xi, op; extrap=extrap)
end
```

Expands to:
```julia
let _deriv = deriv
    if _deriv == 0
        let op = EvalValue()
            _cubic_interp_impl(x, y, xi, op; extrap=extrap)
        end
    elseif _deriv == 1
        let op = EvalDeriv1()
            ...
        end
    ...
    end
end
```
"""
macro _dispatch_deriv(pair, body)
    # Parse pair: deriv => op becomes Expr(:call, :(=>), :deriv, :op)
    pair.head === :call && pair.args[1] === :(=>) ||
        error("@_dispatch_deriv expects `deriv => op`, got: $pair")
    deriv_expr = pair.args[2]
    op_sym = pair.args[3]
    deriv_var = gensym(:deriv)
    quote
        local $(deriv_var) = $(esc(deriv_expr))
        if $(deriv_var) == 0
            let $(esc(op_sym)) = EvalValue()
                $(esc(body))
            end
        elseif $(deriv_var) == 1
            let $(esc(op_sym)) = EvalDeriv1()
                $(esc(body))
            end
        elseif $(deriv_var) == 2
            let $(esc(op_sym)) = EvalDeriv2()
                $(esc(body))
            end
        elseif $(deriv_var) == 3
            let $(esc(op_sym)) = EvalDeriv3()
                $(esc(body))
            end
        else
            throw(ArgumentError("deriv must be 0, 1, 2, or 3; got $($(deriv_var))"))
        end
    end
end

"""
    @_dispatch_side sym => varname body

Dispatch on runtime side symbol, executing body with concrete Val type.

Converts `side::Symbol` (`:nearest`, `:left`, `:right`) to compile-time constant
`Val(:nearest)`, `Val(:left)`, or `Val(:right)`. This creates a function barrier
ensuring type stability and enabling union-splitting optimization.

# Arguments
- `sym => varname`: Pair of symbol variable and binding name for Val type
- `body`: Expression to execute with `varname` bound to concrete Val

# Example
```julia
@_dispatch_side side => sv begin
    _constant_kernel(op, y_left, y_right, h, dL, sv)
end
```

Expands to:
```julia
let _side = side
    if _side === :nearest
        sv = Val(:nearest)
        _constant_kernel(op, y_left, y_right, h, dL, sv)
    elseif _side === :left
        ...
    end
end
```
"""
macro _dispatch_side(pair, body)
    # Parse pair: side => sv becomes Expr(:call, :(=>), :side, :sv)
    pair.head === :call && pair.args[1] === :(=>) ||
        error("@_dispatch_side expects `sym => varname`, got: $pair")
    sym = pair.args[2]
    varname = pair.args[3]
    svs = esc(varname)
    quote
        let _side = $(esc(sym))
            if _side === :nearest
                $svs = Val(:nearest)
                $(esc(body))
            elseif _side === :left
                $svs = Val(:left)
                $(esc(body))
            elseif _side === :right
                $svs = Val(:right)
                $(esc(body))
            else
                throw(ArgumentError("`side` must be :nearest, :left, or :right, got :$_side"))
            end
        end
    end
end

# ========================================
# ND Extrap Dispatch Helpers
# ========================================

"""
    _is_uniform_extrap_no_periodic(extraps, bcs) -> Bool

Check if all extraps are the same symbol and no axes have periodic BCs.
Used by `@_dispatch_extrap_nd` for the zero-alloc fast path.
"""
@inline function _is_uniform_extrap_no_periodic(
    extraps::NTuple{N, Symbol}, bcs::NTuple{N, AbstractBC}
) where {N}
    for d in 1:N
        _is_periodic_bc(bcs[d]) && return false
    end
    for d in 2:N
        extraps[d] !== extraps[1] && return false
    end
    return true
end

"""
    _is_all_periodic(bcs) -> Bool

Check if all axes have periodic BCs.
"""
@inline function _is_all_periodic(bcs::NTuple{N, AbstractBC}) where {N}
    for d in 1:N
        _is_periodic_bc(bcs[d]) || return false
    end
    return true
end

"""
    _resolve_mixed_extrap_vals(extraps, bcs) -> NTuple{N, Val}

Zero-allocation per-axis extrap resolution for mixed periodic/non-periodic BCs.
Uses `@generated` to unroll the loop at compile time, avoiding closure captures.
"""
@generated function _resolve_mixed_extrap_vals(
    extraps::NTuple{N, Symbol}, bcs::NTuple{N, AbstractBC}
) where {N}
    exprs = [:(FastInterpolations._is_periodic_bc(bcs[$d]) ? Val(:wrap) : FastInterpolations._symbol_to_extrap_val(extraps[$d])) for d in 1:N]
    :(($(exprs...),))
end

"""
    @_dispatch_extrap_nd extraps bcs => ev body

Dispatch extrap symbols to concrete `Val` tuples for type-stable ND evaluation.
Creates if/else branches, each with a concrete `ev` binding (similar to `@_dispatch_deriv`).

This is the ND counterpart of `@_dispatch_extrap(sym => varname, body)` (1D).
The 1D version dispatches a single Symbol; this version dispatches `NTuple{N,Symbol}` + BCs.
`N` is derived automatically from `length(extraps)` (compile-time constant for NTuple).

**Fast paths** (zero-alloc):
- Uniform extrap + no periodic BCs: dispatches to one of 4 branches
  (:none, :constant, :extension, :wrap) → `ntuple(_ -> Val(:sym), vn)`
- All periodic BCs (uniform extrap): all axes → `Val(:wrap)`

**Mixed path** (zero-alloc via `@generated`):
- Per-axis resolution via `_resolve_mixed_extrap_vals` (compile-time unrolled)

# Example
```julia
@_dispatch_extrap_nd extraps bcs => extraps_val begin
    return _cubic_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops)
end
```
"""
macro _dispatch_extrap_nd(extraps_expr, pair, body)
    # Parse pair: bcs => ev_sym
    pair.head === :call && pair.args[1] === :(=>) ||
        error("@_dispatch_extrap_nd expects `bcs => binding`, got: $pair")
    bcs_expr = pair.args[2]
    ev_sym = pair.args[3]

    extraps_var = gensym(:extraps)
    bcs_var = gensym(:bcs)
    valn_var = gensym(:valn)

    quote
        local $(extraps_var) = $(esc(extraps_expr))
        local $(bcs_var) = $(esc(bcs_expr))
        local $(valn_var) = Val(length($(extraps_var)))

        if _is_uniform_extrap_no_periodic($(extraps_var), $(bcs_var))
            if $(extraps_var)[1] === :none
                let $(esc(ev_sym)) = ntuple(_ -> Val(:none), $(valn_var))
                    $(esc(body))
                end
            elseif $(extraps_var)[1] === :constant
                let $(esc(ev_sym)) = ntuple(_ -> Val(:constant), $(valn_var))
                    $(esc(body))
                end
            elseif $(extraps_var)[1] === :extension
                let $(esc(ev_sym)) = ntuple(_ -> Val(:extension), $(valn_var))
                    $(esc(body))
                end
            else
                let $(esc(ev_sym)) = ntuple(_ -> Val(:wrap), $(valn_var))
                    $(esc(body))
                end
            end
        elseif _is_all_periodic($(bcs_var))
            let $(esc(ev_sym)) = ntuple(_ -> Val(:wrap), $(valn_var))
                $(esc(body))
            end
        else
            # Mixed periodic/non-periodic: per-axis resolution via @generated (zero-alloc)
            let $(esc(ev_sym)) = _resolve_mixed_extrap_vals($(extraps_var), $(bcs_var))
                $(esc(body))
            end
        end
    end
end
