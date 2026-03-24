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
# GRADIENT
# ========================================

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
```

See also: [`gradient!`](@ref), [`value_gradient`](@ref), [`hessian`](@ref), [`laplacian`](@ref)
"""
# --- Internal helpers (locate-once, @generated) ---
# Factored out so TensorProductInterpolantND can call them directly
# when NoInterp is absent, bypassing the override without `invoke`.

@generated function _gradient_generic(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        hint,
    ) where {Tg, Tv, N}
    deriv_calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
                :(_eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]
    zero_tuple = [:(0 * zref) for _ in 1:N]

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        search = _resolve_search_nd(itp.searches, Val($N), query_r)
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _zero_ref(itp)
            return tuple($(zero_tuple...))
        end
        cell = _locate_cell(itp, query_r, search, hint)
        return tuple($(deriv_calls...))
    end
end

@inline function gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _gradient_generic(itp, query, hint)
end

# Vector API for compatibility with ForwardDiff patterns
@inline function gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Real};
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

# Optim.jl compatible:
grad!(G, x) = gradient!(G, itp, x)
result = optimize(f, grad!, x0, LBFGS())
```

See also: [`gradient`](@ref), [`value_gradient`](@ref), [`hessian!`](@ref)
"""
@generated function _gradient_generic!(
        G::AbstractVector,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
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
        search = _resolve_search_nd(itp.searches, Val($N), query_r)
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _zero_ref(itp)
            @inbounds for i in 1:$N
                G[i] = 0 * zref
            end
            return G
        end
        cell = _locate_cell(itp, query_r, search, hint)
        @inbounds begin
            $(stmts...)
        end
        return G
    end
end

@inline function gradient!(
        G::AbstractVector,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _gradient_generic!(G, itp, query, hint)
end

# Vector query API
@inline function gradient!(
        G::AbstractVector,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Real};
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
@generated function _value_gradient_generic(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
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
    zero_tuple = [:(0 * zref) for _ in 1:N]

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        search = _resolve_search_nd(itp.searches, Val($N), query_r)
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _zero_ref(itp)
            fill_val = _first_fill_value(itp.extraps)
            return (fill_val, tuple($(zero_tuple...)))
        end
        cell = _locate_cell(itp, query_r, search, hint)
        val = $value_call
        grad = tuple($(deriv_calls...))
        return (val, grad)
    end
end

@inline function value_gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _value_gradient_generic(itp, query, hint)
end

# Vector API
@inline function value_gradient(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Real};
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

"""
    hessian(itp::AbstractInterpolantND, query)

Compute the Hessian matrix (matrix of second partial derivatives) at `query`.

Returns an `N×N` matrix where `H[i,j] = ∂²f/∂xᵢ∂xⱼ`.

# Performance
~9x faster than `ForwardDiff.hessian` by using analytical derivatives.
Exploits symmetry: computes only `N(N+1)/2` unique elements.
Uses locate-once optimization: interval search performed only once per query point.

# Examples
```julia
itp = cubic_interp((x, y), data)
H = hessian(itp, (0.5, 0.5))
# H = [∂²f/∂x²    ∂²f/∂x∂y]
#     [∂²f/∂x∂y   ∂²f/∂y² ]
```

See also: [`gradient`](@ref), [`hessian!`](@ref), [`laplacian`](@ref)
"""
@generated function _hessian_generic(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
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

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        Tq = promote_type(eltype(map(float, query_r)), $Tg, $Tv)
        H = Matrix{Tq}(undef, $N, $N)
        search = _resolve_search_nd(itp.searches, Val($N), query_r)
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            fill!(H, zero(Tq))
            return H
        end
        cell = _locate_cell(itp, query_r, search, hint)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

@inline function hessian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _hessian_generic(itp, query, hint)
end

# Vector API
function hessian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Real};
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

# Optim.jl compatible:
hess!(H, x) = hessian!(H, itp, x)
result = optimize(f, grad!, hess!, x0, NewtonTrustRegion())
```

See also: [`hessian`](@ref), [`gradient!`](@ref)
"""
@generated function _hessian_generic!(
        H::AbstractMatrix,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
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

    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        @boundscheck size(H) == ($N, $N) || throw(
            DimensionMismatch(
                "Hessian output matrix must be $($N)×$($N), got $(size(H))"
            )
        )
        search = _resolve_search_nd(itp.searches, Val($N), query_r)
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            fill!(H, zero(eltype(H)))
            return H
        end
        cell = _locate_cell(itp, query_r, search, hint)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

@inline function hessian!(
        H::AbstractMatrix,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _hessian_generic!(H, itp, query, hint)
end

# Vector query API
@inline function hessian!(
        H::AbstractMatrix,
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Real};
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

# Equivalent to (but faster than):
# tr(hessian(itp, (0.5, 0.5)))
```

# Applications
- Heat equation: ∂T/∂t = α∇²T
- Wave equation: ∂²u/∂t² = c²∇²u
- Poisson equation: ∇²φ = ρ

See also: [`gradient`](@ref), [`hessian`](@ref)
"""
@generated function _laplacian_generic(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
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
        search = _resolve_search_nd(itp.searches, Val($N), query_r)
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            return 0 * _zero_ref(itp)
        end
        cell = _locate_cell(itp, query_r, search, hint)
        return +($(deriv_calls...))
    end
end

@inline function laplacian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _laplacian_generic(itp, query, hint)
end

# Vector API
@inline function laplacian(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Real};
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
