# ========================================
# Vector Calculus Operations for ND Interpolants
# ========================================
#
# Fast analytical gradient, hessian, and laplacian using the `deriv` keyword.
# These functions are ~9x faster than ForwardDiff equivalents.
#
# Supports: Any AbstractInterpolantND subtype that implements the `deriv` keyword API.
# Currently: CubicInterpolantND, CubicInterpolant2D
#
# Required interface for subtypes:
#   itp(query; deriv=Val((d1, d2, ...)))  # Mixed partial derivative evaluation
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

| Method | Time (2D) | Notes |
|--------|-----------|-------|
| `gradient(itp, x)` | **~50ns** | Analytical, fastest |
| `ForwardDiff.gradient` | ~414ns | Dual number propagation |

# Examples
```julia
itp = cubic_interp((x, y), data)
gradient(itp, (0.5, 0.5))    # → (∂f/∂x, ∂f/∂y)
gradient(itp, [0.5, 0.5])    # Vector input also supported
```

See also: [`hessian`](@ref), [`laplacian`](@ref)
"""
@generated function gradient(
    itp::AbstractInterpolantND{Tg, Tv, N},
    query::NTuple{N, <:Real}
) where {Tg, Tv, N}
    # Generate: tuple(itp(query; deriv=Val((1,0,...))), itp(query; deriv=Val((0,1,...))), ...)
    deriv_calls = [
        :(itp(query; deriv=Val($(ntuple(j -> j == i ? 1 : 0, N)))))
        for i in 1:N
    ]
    return :(tuple($(deriv_calls...)))
end

# Vector API for compatibility with ForwardDiff patterns
@inline function gradient(
    itp::AbstractInterpolantND{Tg, Tv, N},
    query::AbstractVector{<:Real}
) where {Tg, Tv, N}
    length(query) == N || throw(DimensionMismatch(
        "expected $N-element vector, got $(length(query))-element vector"
    ))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return collect(gradient(itp, query_tuple))
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

| Method | Time (2D) | Notes |
|--------|-----------|-------|
| `hessian(itp, x)` | **~100ns** | Analytical, fastest |
| `ForwardDiff.hessian` | ~1.5μs | Dual number propagation |

# Examples
```julia
itp = cubic_interp((x, y), data)
H = hessian(itp, (0.5, 0.5))
# H = [∂²f/∂x²    ∂²f/∂x∂y]
#     [∂²f/∂x∂y   ∂²f/∂y² ]
```

See also: [`gradient`](@ref), [`laplacian`](@ref)
"""
@generated function hessian(
    itp::AbstractInterpolantND{Tg, Tv, N},
    query::NTuple{N, <:Real}
) where {Tg, Tv, N}
    # Build assignment expressions for the N×N matrix
    stmts = Expr[]

    # Diagonal: ∂²f/∂xᵢ²
    for i in 1:N
        deriv_spec = ntuple(j -> j == i ? 2 : 0, N)
        push!(stmts, :(H[$i, $i] = itp(query; deriv=Val($deriv_spec))))
    end

    # Off-diagonal (exploit symmetry): ∂²f/∂xᵢ∂xⱼ
    for i in 1:N, j in (i+1):N
        deriv_spec = ntuple(k -> (k == i || k == j) ? 1 : 0, N)
        push!(stmts, quote
            val = itp(query; deriv=Val($deriv_spec))
            H[$i, $j] = val
            H[$j, $i] = val
        end)
    end

    return quote
        Tq = promote_type(eltype(query), $Tg, $Tv)
        H = Matrix{Tq}(undef, $N, $N)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

# Vector API
function hessian(
    itp::AbstractInterpolantND{Tg, Tv, N},
    query::AbstractVector{<:Real}
) where {Tg, Tv, N}
    length(query) == N || throw(DimensionMismatch(
        "expected $N-element vector, got $(length(query))-element vector"
    ))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return hessian(itp, query_tuple)
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
@generated function laplacian(
    itp::AbstractInterpolantND{Tg, Tv, N},
    query::NTuple{N, <:Real}
) where {Tg, Tv, N}
    # Sum of diagonal elements: ∂²f/∂x₁² + ∂²f/∂x₂² + ...
    deriv_calls = [
        :(itp(query; deriv=Val($(ntuple(j -> j == i ? 2 : 0, N)))))
        for i in 1:N
    ]
    return :(+($(deriv_calls...)))
end

# Vector API
@inline function laplacian(
    itp::AbstractInterpolantND{Tg, Tv, N},
    query::AbstractVector{<:Real}
) where {Tg, Tv, N}
    length(query) == N || throw(DimensionMismatch(
        "expected $N-element vector, got $(length(query))-element vector"
    ))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return laplacian(itp, query_tuple)
end
