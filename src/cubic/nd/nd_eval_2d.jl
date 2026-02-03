# ========================================
# Bicubic Evaluation
# ========================================
#
# Callable interface and evaluation functions for 2D bicubic interpolation.
# Uses tensor-product Hermite polynomial evaluation for O(1) queries.

# ========================================
# CALLABLE INTERFACE
# ========================================

"""
    (itp::BicubicInterpolant)(xq, yq; deriv=(0,0), search=...)

Evaluate 2D bicubic interpolant at query point (xq, yq).

# Arguments
- `xq`: X-coordinate of query point
- `yq`: Y-coordinate of query point

# Keywords
- `deriv::Tuple{Int,Int}=(0,0)`: Derivative orders `(dx, dy)` where `dx, dy ∈ {0,1,2,3}`
- `search`: Search policy tuple for each dimension (defaults to interpolant's policies)

# Returns
- Interpolated value (or derivative) at (xq, yq)

# Examples
```julia
itp = cubic_interp((x, y), data)
itp(0.5, 0.3)              # Value at (0.5, 0.3)
itp(0.5, 0.3; deriv=(1,0)) # ∂f/∂x at (0.5, 0.3)
itp(0.5, 0.3; deriv=(0,1)) # ∂f/∂y at (0.5, 0.3)
itp(0.5, 0.3; deriv=(1,1)) # ∂²f/∂x∂y at (0.5, 0.3)
```
"""
@inline function (itp::BicubicInterpolant{Tg, Tv})(
    xq::Real, yq::Real;
    deriv::Tuple{Int,Int}=(0, 0),
    search=(itp.search_x, itp.search_y)
) where {Tg, Tv}
    dx, dy = deriv
    @_dispatch_deriv dx => opx begin
        @_dispatch_deriv dy => opy begin
            return _eval_bicubic(itp, Tg(xq), Tg(yq), opx, opy, search)
        end
    end
end

# Tuple input: (xq, yq)
@inline function (itp::BicubicInterpolant{Tg, Tv})(
    q::Tuple{Real, Real};
    deriv::Tuple{Int,Int}=(0, 0),
    search=(itp.search_x, itp.search_y)
) where {Tg, Tv}
    return itp(q[1], q[2]; deriv=deriv, search=search)
end

# Vector input: evaluate at multiple points
function (itp::BicubicInterpolant{Tg, Tv})(
    xqs::AbstractVector{<:Real},
    yqs::AbstractVector{<:Real};
    deriv::Tuple{Int,Int}=(0, 0),
    search=(itp.search_x, itp.search_y)
) where {Tg, Tv}
    length(xqs) == length(yqs) || throw(DimensionMismatch(
        "query vectors must have same length: got $(length(xqs)) and $(length(yqs))"
    ))

    dx, dy = deriv
    # Determine output type
    Tout = promote_type(Tv, Tg)
    results = Vector{Tout}(undef, length(xqs))

    @_dispatch_deriv dx => opx begin
        @_dispatch_deriv dy => opy begin
            @inbounds for k in eachindex(xqs, yqs)
                results[k] = _eval_bicubic(itp, Tg(xqs[k]), Tg(yqs[k]), opx, opy, search)
            end
        end
    end
    return results
end

# Tuple of vectors input: ((xqs,), (yqs,))
function (itp::BicubicInterpolant{Tg, Tv})(
    q::Tuple{AbstractVector{<:Real}, AbstractVector{<:Real}};
    deriv::Tuple{Int,Int}=(0, 0),
    search=(itp.search_x, itp.search_y)
) where {Tg, Tv}
    return itp(q[1], q[2]; deriv=deriv, search=search)
end

# ========================================
# CORE EVALUATION
# ========================================

"""
    _eval_bicubic(itp, xq, yq, opx, opy, search)

Core evaluation function for bicubic interpolation.
"""
@inline function _eval_bicubic(
    itp::BicubicInterpolant{Tg, Tv},
    xq::Tg, yq::Tg,
    opx::AbstractEvalOp, opy::AbstractEvalOp,
    search_kw
) where {Tg, Tv}
    # Handle extrapolation
    xq_eval = _handle_bicubic_extrap(xq, itp.x, itp.extrap_x)
    yq_eval = _handle_bicubic_extrap(yq, itp.y, itp.extrap_y)

    # Resolve search policies
    search_tuple = _resolve_search_nd(search_kw, Val(2))
    searcher_x = _to_searcher(search_tuple[1])
    searcher_y = _to_searcher(search_tuple[2])

    # Find intervals
    ix, xL, xR = search_interval(searcher_x, itp.x, itp.spacing_x, xq_eval)
    iy, yL, yR = search_interval(searcher_y, itp.y, itp.spacing_y, yq_eval)

    # Get spacing info
    hx = _get_h(itp.spacing_x, ix)
    inv_hx = _get_inv_h(itp.spacing_x, ix)
    hy = _get_h(itp.spacing_y, iy)
    inv_hy = _get_inv_h(itp.spacing_y, iy)

    # Compute distances
    dLx = xq_eval - xL
    dLy = yq_eval - yL

    # Evaluate using tensor-product collapse
    return _eval_bicubic_cell(itp.nodal_derivs.partials, ix, iy, hx, inv_hx, hy, inv_hy, dLx, dLy, opx, opy)
end

# ========================================
# EXTRAPOLATION HANDLING
# ========================================

@inline function _handle_bicubic_extrap(q::Tg, axis::AbstractVector{Tg}, ::Val{:none}) where {Tg}
    @boundscheck _check_domain(axis, q, Val(:none))
    return q
end

@inline function _handle_bicubic_extrap(q::Tg, axis::AbstractVector{Tg}, ::Val{:constant}) where {Tg}
    return clamp(q, first(axis), last(axis))
end

@inline function _handle_bicubic_extrap(q::Tg, axis::AbstractVector{Tg}, ::Val{:wrap}) where {Tg}
    return _wrap_to_domain(q, first(axis), last(axis))
end

# ========================================
# TENSOR-PRODUCT CELL EVALUATION
# ========================================

"""
    _eval_bicubic_cell(partials, ix, iy, hx, inv_hx, hy, inv_hy, dLx, dLy, opx, opy)

Evaluate bicubic interpolation within a single cell using tensor-product collapse.

# Algorithm
1. Collapse along x at y=iy and y=iy+1 using f and ∂f/∂x
2. Collapse along x using ∂f/∂y and ∂²f/∂x∂y
3. Final collapse along y

This uses the Hermite basis functions for each 1D collapse.
"""
@inline function _eval_bicubic_cell(
    partials::Array{Tv, 3},
    ix::Int, iy::Int,
    hx::Tg, inv_hx::Tg,
    hy::Tg, inv_hy::Tg,
    dLx::Tq, dLy::Tq,
    opx::OPX, opy::OPY,
) where {Tv, Tg, Tq, OPX<:AbstractEvalOp, OPY<:AbstractEvalOp}
    @inbounds begin
        # Step 1: Collapse along x at y=iy and y=iy+1 using f and ∂f/∂x
        # g0 = P_x(xq) at y=iy
        g0 = _hermite_kernel_1d(opx,
            partials[1, ix, iy], partials[1, ix + 1, iy],
            partials[2, ix, iy], partials[2, ix + 1, iy],
            hx, inv_hx, dLx)

        # g1 = P_x(xq) at y=iy+1
        g1 = _hermite_kernel_1d(opx,
            partials[1, ix, iy + 1], partials[1, ix + 1, iy + 1],
            partials[2, ix, iy + 1], partials[2, ix + 1, iy + 1],
            hx, inv_hx, dLx)

        # Step 2: Collapse along x using ∂f/∂y and ∂²f/∂x∂y
        # gy0 = (∂P/∂y)_x(xq) at y=iy
        gy0 = _hermite_kernel_1d(opx,
            partials[3, ix, iy], partials[3, ix + 1, iy],
            partials[4, ix, iy], partials[4, ix + 1, iy],
            hx, inv_hx, dLx)

        # gy1 = (∂P/∂y)_x(xq) at y=iy+1
        gy1 = _hermite_kernel_1d(opx,
            partials[3, ix, iy + 1], partials[3, ix + 1, iy + 1],
            partials[4, ix, iy + 1], partials[4, ix + 1, iy + 1],
            hx, inv_hx, dLx)

        # Step 3: Final collapse along y
        # P(xq, yq) = Hermite interpolation of (g0, g1) with derivatives (gy0, gy1)
        return _hermite_kernel_1d(opy, g0, g1, gy0, gy1, hy, inv_hy, dLy)
    end
end
