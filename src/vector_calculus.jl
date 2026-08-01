# ========================================
# Vector Calculus Operations for ND Interpolants
# ========================================
#
# Fast analytical gradient, hessian, and laplacian using the `deriv` keyword.
# These functions are ~9x faster than ForwardDiff equivalents.
#
# Supports: Any AbstractInterpolantND subtype that implements _locate_cell/_eval_at_cell.
# Currently: CubicInterpolantND, QuadraticInterpolantND, LinearInterpolantND, ConstantInterpolantND
#
# Performance optimization: "locate once, evaluate many"
# All functions perform interval search ONCE per query point, then evaluate the
# kernel multiple times with different derivative ops. This eliminates redundant
# O(log n) binary searches for vector-grid interpolants.
#
# This file is included last in the module to ensure all interpolant types are defined.

# ========================================
# Shared-element-type guards (uniform-eltype outputs)
# ========================================
# Every component of `gradient` (Tuple) and `hessian` (Matrix with the
# components' promoted eltype — abstract for mixed-unit axes, each entry still
# concretely typed) exists, so both return values. Only two cases guard:
# `laplacian` must ADD `∂²f/∂xᵢ²` across axes — dimensionally undefined when
# the axes' units differ — and the in-place forms refuse a store whose eltype
# genuinely cannot hold the components (`T <: eltype(store)`; `Vector{Any}` or
# an abstract-`Quantity` store is accepted). The decision is purely type-level,
# so the checks constant-fold away on Real and same-unit grids.

# ∂²f/∂xᵢ∂xⱼ eltype: the `_deriv1_op` witness folded once per axis (never `h^-2`,
# which is type-unstable for units). `i == j` is the same expression twice.
@inline _deriv2_pair_eltype(::Type{Tv}, ::Type{Gi}, ::Type{Gj}) where {Tv, Gi, Gj} =
    _promote_eltype(_deriv1_op, eltype(Gj), _promote_eltype(_deriv1_op, eltype(Gi), Tv))

# Unrolled at compile time: `@generated` keeps the N² / N type folds out of the
# runtime frame entirely (a `map`/`ntuple` over types can leave Type objects on
# the heap under coverage).
@generated function _nd_hessian_eltype(::Type{Tv}, grids::Tuple{Vararg{Any, N}}) where {Tv, N}
    G = grids.parameters
    terms = [:(_deriv2_pair_eltype(Tv, $(G[i]), $(G[j]))) for i in 1:N for j in 1:N]
    return :(promote_type($(terms...)))
end

@generated function _nd_laplacian_eltype(::Type{Tv}, grids::Tuple{Vararg{Any, N}}) where {Tv, N}
    G = grids.parameters
    terms = [:(_deriv2_pair_eltype(Tv, $(G[i]), $(G[i]))) for i in 1:N]
    return :(promote_type($(terms...)))
end

@generated function _nd_gradient_eltype(::Type{Tv}, grids::Tuple{Vararg{Any, N}}) where {Tv, N}
    G = grids.parameters
    terms = [:(_promote_eltype(_deriv1_op, eltype($(G[i])), Tv)) for i in 1:N]
    return :(promote_type($(terms...)))
end

# Guard for uniform-eltype outputs: the per-axis components must share ONE
# concrete element type. Pure type algebra (isconcretetype ∘ promote) — mixed
# units are just the typical trigger, not a Unitful special case.
@noinline function _throw_nd_component_eltype(what::String, alternative::String, T)
    throw(
        ArgumentError(
            "$what on this ND grid is not supported: the per-axis components " *
                "have no shared concrete element type — they promote to the " *
                "abstract `$T` (typical for mixed-unit axes). $alternative"
        )
    )
end

@inline function _check_nd_laplacian_eltype(::Type{Tv}, grids::Tuple) where {Tv}
    T = _nd_laplacian_eltype(Tv, grids)
    isconcretetype(T) || _throw_nd_component_eltype(
        "laplacian",
        "Its terms `∂²f/∂xᵢ²` would have to be ADDED across axes, which is " *
            "dimensionally undefined here. Query them individually, e.g. " *
            "`itp(query...; deriv = DerivOp(2, 0))`.",
        T,
    )
    return nothing
end

# In-place forms: `T` is the promotion of every component, so `T <: TS` means the
# caller's store holds all of them — accept it even when `T` itself is abstract.
# A concrete `T` (Real, same-unit) short-circuits before the subtype test, so a
# `Float32` store still takes `Float64` components exactly as it always did.
@inline function _check_nd_gradient_store_eltype(::Type{Tv}, grids::Tuple, ::Type{TS}) where {Tv, TS}
    T = _nd_gradient_eltype(Tv, grids)
    (isconcretetype(T) || T <: TS) || _throw_nd_component_eltype(
        "gradient!",
        "Use the allocating `gradient` (a Tuple — each component keeps its own " *
            "type) or pass a store that can hold them (e.g. `Vector{Any}`).",
        T,
    )
    return nothing
end

@inline function _check_nd_hessian_store_eltype(::Type{Tv}, grids::Tuple, ::Type{TS}) where {Tv, TS}
    T = _nd_hessian_eltype(Tv, grids)
    (isconcretetype(T) || T <: TS) || _throw_nd_component_eltype(
        "hessian!",
        "Query the components individually via `deriv = DerivOp(2, 0)` or pass " *
            "a store that can hold them (e.g. `Matrix{Any}`).",
        T,
    )
    return nothing
end

# ========================================
# GRADIENT
# ========================================

# --- Internal helpers (locate-once, @generated) ---
# Factored out so HeteroInterpolantND can call them directly
# when NoInterp is absent, bypassing the override without `invoke`.

@generated function _gradient_generic(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}},
        hint,
    ) where {Tg, Tv, N}
    deriv_calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
                :(_eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]
    # Gradient component i is ∂f/∂xᵢ — the FILL value carries the zero (a NaN
    # fill poisons OOB derivatives, matching the 1D/ND eval rule; interior data
    # never leaks into OOB), scaled by `inv(gridᵢ unit)` into `value/gridᵢ`.
    zero_tuple = [:(0 * zref * _deriv_oneunit(oneunit(eltype(itp.grids[$i])), DerivOp(1))) for i in 1:N]

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _first_fill_value(itp.extraps)
            return tuple($(zero_tuple...))
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        return tuple($(deriv_calls...))
    end
end

"""
    gradient(itp::AbstractInterpolantND, query)

Compute the gradient (vector of partial derivatives) at `query`.

Returns an `NTuple{N}` of partial derivatives `(∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)`.

# Performance
~9x faster than `ForwardDiff.gradient` by using analytical derivatives.
Uses locate-once optimization: interval search performed only once per query point.

# Examples
```julia
itp = cubic_interp((x, y), data)
gradient(itp, (0.5, 0.5))    # → (∂f/∂x, ∂f/∂y)
gradient(itp, [0.5, 0.5])    # Vector input also supported
gradient(itp, 0.5, 0.5)      # splatted scalars also supported
```

See also: [`gradient!`](@ref), [`value_gradient`](@ref), [`hessian`](@ref), [`laplacian`](@ref)
"""
@inline function gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _gradient_generic(itp, query, hint)
end

# Splat convenience: gradient(itp, x, y) → gradient(itp, (x, y)). Scalar-only;
# `Vararg{Number,N}` can't match a batch container, so it never intercepts one.
@inline function gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        q::Vararg{Number, N};
        kw...,
    ) where {Tg, Tv, N}
    return gradient(itp, q; kw...)
end

# Vector API (ForwardDiff patterns). `{<:Number}` mirrors the `Vararg{Number,N}` sibling
# — the scalar-vs-container gate — so an AoS batch (`Vector{<:Tuple}`) is never misread as
# one point (`Quantity <: Number` keeps unit coords in). Same bound on every vector form.
@inline function gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Number};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    length(query) == N || throw(
        DimensionMismatch(
            "expected $N-element vector, got $(length(query))-element vector"
        )
    )
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return collect(gradient(itp, query_tuple; hint = hint))
end

@generated function _gradient_generic!(
        G::AbstractVector,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}},
        hint,
    ) where {Tg, Tv, N}
    stmts = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
                :(G[$i] = _eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        @boundscheck length(G) >= $N || throw(
            DimensionMismatch(
                "gradient output vector must have at least $($N) elements, got $(length(G))"
            )
        )
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _first_fill_value(itp.extraps)
            @inbounds for i in 1:$N
                G[i] = 0 * zref * _deriv_oneunit(oneunit(eltype(itp.grids[i])), DerivOp(1))
            end
            return G
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        @inbounds begin
            $(stmts...)
        end
        return G
    end
end

"""
    gradient!(G, itp::AbstractInterpolantND, query)

Compute the gradient in-place, writing partial derivatives into `G`.

Zero-allocation version of [`gradient`](@ref) for use in optimization loops.

# Examples
```julia
itp = cubic_interp((x, y), data)
G = zeros(2)
gradient!(G, itp, (0.5, 0.5))    # G .= (∂f/∂x, ∂f/∂y)
gradient!(G, itp, [0.5, 0.5])    # G .= (∂f/∂x, ∂f/∂y)
gradient!(G, itp, 0.5, 0.5)      # splatted scalars also supported

# Optim.jl compatible:
grad!(G, x) = gradient!(G, itp, x)
result = optimize(f, grad!, x0, LBFGS())
```

See also: [`gradient`](@ref), [`value_gradient`](@ref), [`hessian!`](@ref)
"""
@inline function gradient!(
        G::AbstractVector,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    _check_nd_gradient_store_eltype(Tv, itp.grids, eltype(G))
    return _gradient_generic!(G, itp, query, hint)
end

# Splat convenience: gradient!(G, itp, x, y) → gradient!(G, itp, (x, y)).
@inline function gradient!(
        G::AbstractVector,
        itp::AbstractInterpolantND{Tg, Tv, N},
        q::Vararg{Number, N};
        kw...,
    ) where {Tg, Tv, N}
    return gradient!(G, itp, q; kw...)
end

# Vector query API
@inline function gradient!(
        G::AbstractVector,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Number};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    length(query) == N || throw(
        DimensionMismatch(
            "expected $N-element query vector, got $(length(query))-element vector"
        )
    )
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return gradient!(G, itp, query_tuple; hint = hint)
end

# ========================================
# VALUE + GRADIENT (locate once)
# ========================================

@generated function _value_gradient_generic(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}},
        hint,
    ) where {Tg, Tv, N}
    value_ops = ntuple(_ -> EvalValue(), N)
    value_call = :(_eval_at_cell(itp, cell, $value_ops))

    deriv_calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
                :(_eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]
    # Gradient component i is ∂f/∂xᵢ — scale the value-space zero by `inv(gridᵢ unit)`
    # so a unit-grid FillExtrap OOB returns `value/gridᵢ` (identity on Real grids).
    zero_tuple = [:(0 * zref * _deriv_oneunit(oneunit(eltype(itp.grids[$i])), DerivOp(1))) for i in 1:N]

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _first_fill_value(itp.extraps)
            return (zref, tuple($(zero_tuple...)))
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        val = $value_call
        grad = tuple($(deriv_calls...))
        return (val, grad)
    end
end

"""
    value_gradient(itp::AbstractInterpolantND, query)

Compute the value and gradient simultaneously at `query`.

Returns `(value, gradient)` where `gradient` is an `NTuple{N}` of partial derivatives.

# Performance
Faster than calling `itp(query)` and `gradient(itp, query)` separately:
interval search is performed only **once** per query point.

# Examples
```julia
itp = cubic_interp((x, y), data)
val, grad = value_gradient(itp, (0.5, 0.5))   # → (f, (∂f/∂x, ∂f/∂y))
val, grad = value_gradient(itp, [0.5, 0.5])    # Vector input also supported
val, grad = value_gradient(itp, 0.5, 0.5)      # splatted scalars also supported

# Optim.jl fg! pattern:
function fg!(F, G, x)
    if G !== nothing && F !== nothing
        val, grad = value_gradient(itp, Tuple(x))
        G .= grad
        return val
    elseif G !== nothing
        gradient!(G, itp, Tuple(x))
        return nothing
    else
        return itp(Tuple(x))
    end
end
result = optimize(Optim.only_fg!(fg!), x0, LBFGS())
```

See also: [`gradient`](@ref), [`gradient!`](@ref)
"""
@inline function value_gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _value_gradient_generic(itp, query, hint)
end

# Splat convenience: value_gradient(itp, x, y) → value_gradient(itp, (x, y)).
@inline function value_gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        q::Vararg{Number, N};
        kw...,
    ) where {Tg, Tv, N}
    return value_gradient(itp, q; kw...)
end

# Vector API
@inline function value_gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Number};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    length(query) == N || throw(
        DimensionMismatch(
            "expected $N-element vector, got $(length(query))-element vector"
        )
    )
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    val, grad_tuple = value_gradient(itp, query_tuple; hint = hint)
    return (val, collect(grad_tuple))
end

# ========================================
# HESSIAN
# ========================================

@generated function _hessian_generic(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}},
        hint,
    ) where {Tg, Tv, N}
    stmts = Expr[]

    # Diagonal: ∂²f/∂xᵢ²
    for i in 1:N
        ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
        push!(stmts, :(H[$i, $i] = _eval_at_cell(itp, cell, $ops)))
    end

    # Off-diagonal (exploit symmetry): ∂²f/∂xᵢ∂xⱼ
    for i in 1:N, j in (i + 1):N
        ops = ntuple(k -> (k == i || k == j) ? DerivOp{1}() : DerivOp{0}(), N)
        push!(
            stmts, quote
                val = _eval_at_cell(itp, cell, $ops)
                H[$i, $j] = val
                H[$j, $i] = val
            end
        )
    end

    # OOB (FillExtrap) zeros per element: the FILL value carries the zero (a
    # NaN fill poisons OOB derivatives — 1D/ND eval rule), scaled into each
    # entry's own `[value]/[gridᵢ·gridⱼ]` space. Also serves abstract-eltype
    # stores, where `zero(eltype(H))` has no method.
    oob_stmts = [
        :(
                H[$i, $j] = 0 * zref *
                _deriv_oneunit(oneunit(eltype(itp.grids[$i])), DerivOp(1)) *
                _deriv_oneunit(oneunit(eltype(itp.grids[$j])), DerivOp(1))
            )
            for i in 1:N for j in 1:N
    ]

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        # Container eltype = the promotion of every component (`_nd_hessian_eltype`,
        # the same witness the store guard consults): concrete for Real/same-unit
        # grids, the tightest abstract `Quantity` supertype for mixed-unit axes —
        # each entry still carries its own concrete units.
        Tq = _nd_hessian_eltype($Tv, itp.grids)
        H = Matrix{Tq}(undef, $N, $N)
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _first_fill_value(itp.extraps)
            $(oob_stmts...)
            return H
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

"""
    hessian(itp::AbstractInterpolantND, query)

Compute the Hessian matrix (matrix of second partial derivatives) at `query`.

Returns an `N×N` matrix where `H[i,j] = ∂²f/∂xᵢ∂xⱼ`. On mixed-unit grids the
entries carry per-element units (`value/gridᵢ·gridⱼ`), so the matrix eltype is
their tightest common supertype (an abstract `Quantity`) — element access is
unit-correct, while whole-matrix linear algebra (`det`, `eigen`, `\\`) does not
apply to a dimensionally heterogeneous matrix.

# Performance
~9x faster than `ForwardDiff.hessian` by using analytical derivatives.
Exploits symmetry: computes only `N(N+1)/2` unique elements.
Uses locate-once optimization: interval search performed only once per query point.

# Examples
```julia
itp = cubic_interp((x, y), data)
H = hessian(itp, (0.5, 0.5))
H = hessian(itp, 0.5, 0.5)    # splatted scalars also supported
# H = [∂²f/∂x²    ∂²f/∂x∂y]
#     [∂²f/∂x∂y   ∂²f/∂y² ]
```

See also: [`gradient`](@ref), [`hessian!`](@ref), [`laplacian`](@ref)
"""
@inline function hessian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _hessian_generic(itp, query, hint)
end

# Splat convenience: hessian(itp, x, y) → hessian(itp, (x, y)).
@inline function hessian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        q::Vararg{Number, N};
        kw...,
    ) where {Tg, Tv, N}
    return hessian(itp, q; kw...)
end

# Vector API
function hessian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Number};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    length(query) == N || throw(
        DimensionMismatch(
            "expected $N-element vector, got $(length(query))-element vector"
        )
    )
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return hessian(itp, query_tuple; hint = hint)
end

@generated function _hessian_generic!(
        H::AbstractMatrix,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}},
        hint,
    ) where {Tg, Tv, N}
    stmts = Expr[]

    # Diagonal: ∂²f/∂xᵢ²
    for i in 1:N
        ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
        push!(stmts, :(H[$i, $i] = _eval_at_cell(itp, cell, $ops)))
    end

    # Off-diagonal (exploit symmetry): ∂²f/∂xᵢ∂xⱼ
    for i in 1:N, j in (i + 1):N
        ops = ntuple(k -> (k == i || k == j) ? DerivOp{1}() : DerivOp{0}(), N)
        push!(
            stmts, quote
                val = _eval_at_cell(itp, cell, $ops)
                H[$i, $j] = val
                H[$j, $i] = val
            end
        )
    end

    # Same per-element OOB zeros as the allocating form — `zero(eltype(H))` has
    # no method for abstract-eltype stores (`Matrix{Any}`), which are accepted.
    oob_stmts = [
        :(
                H[$i, $j] = 0 * zref *
                _deriv_oneunit(oneunit(eltype(itp.grids[$i])), DerivOp(1)) *
                _deriv_oneunit(oneunit(eltype(itp.grids[$j])), DerivOp(1))
            )
            for i in 1:N for j in 1:N
    ]

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        @boundscheck size(H) == ($N, $N) || throw(
            DimensionMismatch(
                "Hessian output matrix must be $($N)×$($N), got $(size(H))"
            )
        )
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _first_fill_value(itp.extraps)
            $(oob_stmts...)
            return H
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

"""
    hessian!(H, itp::AbstractInterpolantND, query)

Compute the Hessian matrix in-place, writing second partial derivatives into `H`.

Zero-allocation version of [`hessian`](@ref) for use in optimization loops.
Exploits symmetry: computes only `N(N+1)/2` unique elements.

# Examples
```julia
itp = cubic_interp((x, y), data)
H = zeros(2, 2)
hessian!(H, itp, (0.5, 0.5))
hessian!(H, itp, 0.5, 0.5)    # splatted scalars also supported

# Optim.jl compatible:
hess!(H, x) = hessian!(H, itp, x)
result = optimize(f, grad!, hess!, x0, NewtonTrustRegion())
```

See also: [`hessian`](@ref), [`gradient!`](@ref)
"""
@inline function hessian!(
        H::AbstractMatrix,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    _check_nd_hessian_store_eltype(Tv, itp.grids, eltype(H))
    return _hessian_generic!(H, itp, query, hint)
end

# Splat convenience: hessian!(H, itp, x, y) → hessian!(H, itp, (x, y)).
@inline function hessian!(
        H::AbstractMatrix,
        itp::AbstractInterpolantND{Tg, Tv, N},
        q::Vararg{Number, N};
        kw...,
    ) where {Tg, Tv, N}
    return hessian!(H, itp, q; kw...)
end

# Vector query API
@inline function hessian!(
        H::AbstractMatrix,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Number};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    length(query) == N || throw(
        DimensionMismatch(
            "expected $N-element query vector, got $(length(query))-element vector"
        )
    )
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return hessian!(H, itp, query_tuple; hint = hint)
end

# ========================================
# LAPLACIAN
# ========================================

@generated function _laplacian_generic(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}},
        hint,
    ) where {Tg, Tv, N}
    deriv_calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
                :(_eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            return 0 * _first_fill_value(itp.extraps) * _deriv_oneunit(oneunit(eltype(itp.grids[1])), DerivOp(2))
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        return +($(deriv_calls...))
    end
end

"""
    laplacian(itp::AbstractInterpolantND, query)

Compute the Laplacian (sum of second partial derivatives) at `query`.

Returns a scalar: `∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ... + ∂²f/∂xₙ²`

# Performance
Faster than computing full Hessian when only the trace is needed.
Uses locate-once optimization: interval search performed only once per query point.

# Examples
```julia
itp = cubic_interp((x, y), data)
∇²f = laplacian(itp, (0.5, 0.5))  # ∂²f/∂x² + ∂²f/∂y²
∇²f = laplacian(itp, 0.5, 0.5)    # splatted scalars also supported

# Equivalent to (but faster than):
# tr(hessian(itp, (0.5, 0.5)))
```

# Applications
- Heat equation: ∂T/∂t = α∇²T
- Wave equation: ∂²u/∂t² = c²∇²u
- Poisson equation: ∇²φ = ρ

See also: [`gradient`](@ref), [`hessian`](@ref)
"""
@inline function laplacian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    _check_nd_laplacian_eltype(Tv, itp.grids)
    return _laplacian_generic(itp, query, hint)
end

# Splat convenience: laplacian(itp, x, y) → laplacian(itp, (x, y)).
@inline function laplacian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        q::Vararg{Number, N};
        kw...,
    ) where {Tg, Tv, N}
    return laplacian(itp, q; kw...)
end

# Vector API
@inline function laplacian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Number};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    length(query) == N || throw(
        DimensionMismatch(
            "expected $N-element vector, got $(length(query))-element vector"
        )
    )
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return laplacian(itp, query_tuple; hint = hint)
end
