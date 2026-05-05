# ========================================
# PHS Evaluation Engine
# ========================================
#
# Core evaluation functions for PHSInterpolantND.
# Three layers:
#   1. _phs_find_base_node     — nearest grid node to query point
#   2. _phs_eval_stencil       — evaluate one local PHS interpolant at a query point
#   3. _phs_eval_blended       — weighted blend of multiple local interpolants (C² output)
#
# All mutable workspace (rhs, coeffs) is acquired from thread-local
# AdaptiveArrayPools (via @with_pool) so the hot path is zero-allocation.

# ======================================================
# Layer 1: Base-node lookup
# ======================================================

"""
    _phs_find_base_node(itp::PHSInterpolantND{Tg,Tv,N,K}, query::NTuple{N,<:Real})
        -> NTuple{N,Int}

Find the grid node nearest to `query` in each dimension.
O(1) per axis for uniform (ScalarSpacing) grids; O(log n) for non-uniform.
"""
@inline function _phs_find_base_node(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        query::NTuple{N, <:Real},
    ) where {Tg, Tv, N, K}
    return ntuple(N) do d
        grid = itp.grids[d]
        n = length(grid)
        sp = itp.spacings[d]
        qd = Tg(query[d])
        if sp isa ScalarSpacing
            idx = round(Int, (qd - grid[1]) * sp.inv_h + 1)
        else
            # Binary search for nearest
            lo, hi = 1, n
            while hi - lo > 1
                mid = (lo + hi) >> 1
                grid[mid] <= qd ? (lo = mid) : (hi = mid)
            end
            # lo and hi are the two bracketing nodes; pick nearest
            idx = abs(qd - grid[lo]) <= abs(qd - grid[hi]) ? lo : hi
        end
        clamp(idx, 1, n)
    end
end

"""
    _phs_base_coords(itp, base_idx) -> NTuple{N,Tg}

Physical coordinates of the base node at `base_idx`.
"""
@inline function _phs_base_coords(
        itp::PHSInterpolantND{Tg, Tv, N},
        base_idx::NTuple{N, Int},
    ) where {Tg, Tv, N}
    return ntuple(d -> itp.grids[d][base_idx[d]], N)
end

# ======================================================
# Layer 2: Single-stencil evaluation
# ======================================================

"""
    _phs_build_rhs!(rhs, data, base_idx, offsets, grid_sizes)

Fill `rhs[1:N_stencil]` with the data values at the stencil nodes
(clamped to [1,n] per axis).  `rhs[N_stencil+1:end]` remain zero
(the polynomial consistency constraints).
"""
@inline function _phs_build_rhs!(
        rhs::AbstractVector,
        data::AbstractArray{Tv, N},
        base_idx::NTuple{N, Int},
        offsets::Vector{<:NTuple{N, Int}},
        grid_sizes::NTuple{N, Int},
    ) where {Tv, N}
    @inbounds for (i, off) in enumerate(offsets)
        cidx = ntuple(d -> clamp(base_idx[d] + off[d], 1, grid_sizes[d]), N)
        rhs[i] = data[CartesianIndex(cidx)]
    end
    return rhs
end

# ---- Evaluation dispatch on DerivOp ----

# Helper: difference vector (query - stencil_node_i) in physical space
@inline function _phs_diff(query::NTuple{N, <:Real}, base_coords::NTuple{N, Tg},
        off::NTuple{N, Int}, hs_local::NTuple{N, Tg}) where {N, Tg}
    return ntuple(d -> Tg(query[d]) - (base_coords[d] + Tg(off[d]) * hs_local[d]), N)
end

"""
    _phs_eval_coeffs_value(coeffs, offsets, hs_local, query, base_coords, ::Val{K}, N) -> scalar

Evaluate the PHS interpolant value at `query` given precomputed coefficients.
  f = Σᵢ wᵢ φ(rᵢ) + v₀ + Σⱼ vⱼ (xⱼ - xbase_j)
"""
@inline function _phs_eval_coeffs_value(
        coeffs::AbstractVector{Tv},
        offsets::Vector{<:NTuple{N, Int}},
        hs_local::NTuple{N, Tg},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
    ) where {Tv, Tg, N, K}
    ns = length(offsets)
    y = zero(Tv)

    # RBF sum
    @inbounds @simd for i in 1:ns
        xh = _phs_diff(query, base_coords, offsets[i], hs_local)
        r = sqrt(sum(x -> x * x, xh))
        y += coeffs[i] * _phs_phi(r, Val{K}())
    end

    # Polynomial augmentation: all monomials up to degree (K-1)÷2
    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    y += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    return y
end

"""
    _phs_eval_coeffs_deriv1(coeffs, offsets, hs_local, query, base_coords, ::Val{K}, axis) -> scalar

Evaluate ∂f/∂xξ (Eq. 25).
  fξ = Σᵢ wᵢ φ'(rᵢ) (xξ - xiξ)/rᵢ + vξ
"""
@inline function _phs_eval_coeffs_deriv1(
        coeffs::AbstractVector{Tv},
        offsets::Vector{<:NTuple{N, Int}},
        hs_local::NTuple{N, Tg},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        axis::Int,
    ) where {Tv, Tg, N, K}
    ns = length(offsets)
    y = zero(Tv)

    # r_inv = 1/r when r > eps, 0 otherwise — eliminates the skip-branch and
    # allows @simd to vectorize; phi_prime(r=0) for K≥3 is already 0.
    @inbounds @simd for i in 1:ns
        xh = _phs_diff(query, base_coords, offsets[i], hs_local)
        r = sqrt(sum(x -> x * x, xh))
        r_inv = ifelse(r < eps(Tg), zero(Tg), one(Tg) / r)
        y += coeffs[i] * _phs_phi_prime(r, Val{K}()) * xh[axis] * r_inv
    end
    # Polynomial derivative: ∂/∂x_axis of all augmentation monomials
    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    y += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, axis)
    return y
end

"""
    _phs_eval_coeffs_deriv2(coeffs, offsets, hs_local, query, base_coords, ::Val{K}, ax1, ax2) -> scalar

Evaluate ∂²f/∂xξ∂xζ (Eq. 26).
"""
@inline function _phs_eval_coeffs_deriv2(
        coeffs::AbstractVector{Tv},
        offsets::Vector{<:NTuple{N, Int}},
        hs_local::NTuple{N, Tg},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ax1::Int,
        ax2::Int,
    ) where {Tv, Tg, N, K}
    ns = length(offsets)
    y = zero(Tv)

    # Lift ax1==ax2 branch outside the loop and use r_inv/r2_inv instead of /r, /r2
    # so the loop is branch-free and @simd can vectorize.
    eps2 = eps(Tg)^2
    if ax1 == ax2
        @inbounds @simd for i in 1:ns
            xh = _phs_diff(query, base_coords, offsets[i], hs_local)
            r2 = sum(x -> x * x, xh)
            r = sqrt(r2)
            r_inv  = ifelse(r2 < eps2, zero(Tg), one(Tg) / r)
            r2_inv = r_inv * r_inv
            fp  = _phs_phi_prime(r, Val{K}())
            fpp = _phs_phi_dprime(r, Val{K}())
            xh_ax = xh[ax1]
            # Diagonal: fpp*(xξ-xiξ)²/r² + fp*(1/r - (xξ-xiξ)²/r³)
            y += coeffs[i] * (fpp * xh_ax * xh_ax * r2_inv + fp * (r_inv - xh_ax * xh_ax * r2_inv * r_inv))
        end
    else
        @inbounds @simd for i in 1:ns
            xh = _phs_diff(query, base_coords, offsets[i], hs_local)
            r2 = sum(x -> x * x, xh)
            r = sqrt(r2)
            r_inv  = ifelse(r2 < eps2, zero(Tg), one(Tg) / r)
            r2_inv = r_inv * r_inv
            fp  = _phs_phi_prime(r, Val{K}())
            fpp = _phs_phi_dprime(r, Val{K}())
            # Off-diagonal: fpp*(xξ-xiξ)*(xζ-xiζ)/r² - fp*(xξ-xiξ)*(xζ-xiζ)/r³
            y += coeffs[i] * (fpp - fp * r_inv) * xh[ax1] * xh[ax2] * r2_inv
        end
    end
    # Polynomial second derivatives (non-zero for poly_deg ≥ 2)
    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    y += _phs_eval_poly_deriv2(Δx, poly_exps, coeffs, ns, ax1, ax2)
    return y
end

"""
    _phs_eval_coeffs_value_and_deriv1(coeffs, offsets, hs_local, query, base_coords, ::Val{K}, axis)
        -> (value, deriv1)

Fused single-pass evaluation of both the interpolant value and its first derivative
along `axis`.  Avoids traversing the stencil twice when both are needed (gradient blending).
"""
@inline function _phs_eval_coeffs_value_and_deriv1(
        coeffs::AbstractVector{Tv},
        offsets::Vector{<:NTuple{N, Int}},
        hs_local::NTuple{N, Tg},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        axis::Int,
    ) where {Tv, Tg, N, K}
    ns = length(offsets)
    yv = zero(Tv)
    yd = zero(Tv)

    @inbounds @simd for i in 1:ns
        xh = _phs_diff(query, base_coords, offsets[i], hs_local)
        r2 = sum(x -> x * x, xh)
        r  = sqrt(r2)
        ci = coeffs[i]
        r_inv = ifelse(r < eps(Tg), zero(Tg), one(Tg) / r)
        yv += ci * _phs_phi(r, Val{K}())
        yd += ci * _phs_phi_prime(r, Val{K}()) * xh[axis] * r_inv
    end

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    yv += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    yd += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, axis)
    return yv, yd
end

"""
    _phs_eval_coeffs_value_and_deriv1_and_deriv2(coeffs, offsets, hs_local, query, base_coords,
        ::Val{K}, ax1, ax2) -> (value, deriv1_ax1, deriv2)

Fused single-pass evaluation of value, ∂f/∂x_{ax1}, and ∂²f/∂x_{ax1}∂x_{ax2}.
Used for Hessian blending (diagonal and mixed second derivatives).
For diagonal (ax1==ax2): returns (f, f_ξ, f_ξξ).
For mixed (ax1≠ax2): returns (f, f_ξ, f_ξζ) where first-deriv is w.r.t. ax1.
"""
@inline function _phs_eval_coeffs_value_and_deriv1_and_deriv2(
        coeffs::AbstractVector{Tv},
        offsets::Vector{<:NTuple{N, Int}},
        hs_local::NTuple{N, Tg},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ax1::Int,
        ax2::Int,
    ) where {Tv, Tg, N, K}
    ns = length(offsets)
    yv  = zero(Tv)
    yd1 = zero(Tv)
    yd2 = zero(Tv)

    # Lift is_diag outside the loop and use r_inv/r2_inv for branch-free @simd loops.
    eps_tg = eps(Tg)
    if ax1 == ax2
        @inbounds @simd for i in 1:ns
            xh = _phs_diff(query, base_coords, offsets[i], hs_local)
            r2 = sum(x -> x * x, xh)
            r  = sqrt(r2)
            ci = coeffs[i]
            r_inv  = ifelse(r < eps_tg, zero(Tg), one(Tg) / r)
            r2_inv = r_inv * r_inv
            fp  = _phs_phi_prime(r, Val{K}())
            fpp = _phs_phi_dprime(r, Val{K}())
            yv  += ci * _phs_phi(r, Val{K}())
            yd1 += ci * fp * xh[ax1] * r_inv
            xh_ax = xh[ax1]
            yd2 += ci * (fpp * xh_ax * xh_ax * r2_inv + fp * (r_inv - xh_ax * xh_ax * r2_inv * r_inv))
        end
    else
        @inbounds @simd for i in 1:ns
            xh = _phs_diff(query, base_coords, offsets[i], hs_local)
            r2 = sum(x -> x * x, xh)
            r  = sqrt(r2)
            ci = coeffs[i]
            r_inv  = ifelse(r < eps_tg, zero(Tg), one(Tg) / r)
            r2_inv = r_inv * r_inv
            fp  = _phs_phi_prime(r, Val{K}())
            fpp = _phs_phi_dprime(r, Val{K}())
            yv  += ci * _phs_phi(r, Val{K}())
            yd1 += ci * fp * xh[ax1] * r_inv
            yd2 += ci * (fpp - fp * r_inv) * xh[ax1] * xh[ax2] * r2_inv
        end
    end

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    yv  += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    yd1 += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, ax1)
    yd2 += _phs_eval_poly_deriv2(Δx, poly_exps, coeffs, ns, ax1, ax2)
    return yv, yd1, yd2
end

"""
    _phs_eval_coeffs_value_and_two_deriv1(coeffs, offsets, hs_local, query, base_coords,
        ::Val{K}, ax1, ax2) -> (value, deriv1_ax1, deriv1_ax2)

Fused single-pass evaluation of value, ∂f/∂x_{ax1}, and ∂f/∂x_{ax2}.
Used for mixed-Hessian blending to get both first-derivative components in one loop.
"""
@inline function _phs_eval_coeffs_value_and_two_deriv1(
        coeffs::AbstractVector{Tv},
        offsets::Vector{<:NTuple{N, Int}},
        hs_local::NTuple{N, Tg},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ax1::Int,
        ax2::Int,
    ) where {Tv, Tg, N, K}
    ns = length(offsets)
    yv  = zero(Tv)
    yd1 = zero(Tv)
    yd2 = zero(Tv)

    @inbounds @simd for i in 1:ns
        xh = _phs_diff(query, base_coords, offsets[i], hs_local)
        r2 = sum(x -> x * x, xh)
        r  = sqrt(r2)
        ci = coeffs[i]
        fp_r_inv = _phs_phi_prime(r, Val{K}()) * ifelse(r < eps(Tg), zero(Tg), one(Tg) / r)
        yv  += ci * _phs_phi(r, Val{K}())
        yd1 += ci * fp_r_inv * xh[ax1]
        yd2 += ci * fp_r_inv * xh[ax2]
    end

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    yv  += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    yd1 += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, ax1)
    yd2 += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, ax2)
    return yv, yd1, yd2
end

"""
    _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(coeffs, offsets, hs_local, query, base_coords,
        ::Val{K}, ax1, ax2) -> (value, deriv1_ax1, deriv1_ax2, deriv2_ax1_ax2)

Fused single-pass evaluation of value, ∂f/∂x_{ax1}, ∂f/∂x_{ax2}, and ∂²f/∂x_{ax1}∂x_{ax2}
for the off-diagonal (ax1 ≠ ax2) mixed Hessian case.  Replaces the previous two-pass approach
of calling `_phs_eval_coeffs_value_and_two_deriv1` followed by `_phs_eval_coeffs_deriv2`.
"""
@inline function _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(
        coeffs::AbstractVector{Tv},
        offsets::Vector{<:NTuple{N, Int}},
        hs_local::NTuple{N, Tg},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ax1::Int,
        ax2::Int,
    ) where {Tv, Tg, N, K}
    ns = length(offsets)
    yv   = zero(Tv)
    yd1  = zero(Tv)
    yd2  = zero(Tv)
    yd12 = zero(Tv)

    eps2 = eps(Tg)^2
    @inbounds @simd for i in 1:ns
        xh = _phs_diff(query, base_coords, offsets[i], hs_local)
        r2 = sum(x -> x * x, xh)
        r  = sqrt(r2)
        ci = coeffs[i]
        r_inv  = ifelse(r2 < eps2, zero(Tg), one(Tg) / r)
        r2_inv = r_inv * r_inv
        fp  = _phs_phi_prime(r, Val{K}())
        fpp = _phs_phi_dprime(r, Val{K}())
        yv   += ci * _phs_phi(r, Val{K}())
        yd1  += ci * fp * xh[ax1] * r_inv
        yd2  += ci * fp * xh[ax2] * r_inv
        yd12 += ci * (fpp - fp * r_inv) * xh[ax1] * xh[ax2] * r2_inv
    end

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    yv   += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    yd1  += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, ax1)
    yd2  += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, ax2)
    yd12 += _phs_eval_poly_deriv2(Δx, poly_exps, coeffs, ns, ax1, ax2)
    return yv, yd1, yd2, yd12
end

"""
    _phs_solve_stencil!(itp, base_idx, rhs_buf, coeff_buf) -> (offsets, coeff, hs_local)

Perform the linear solve for the PHS stencil at `base_idx`:
selects offsets/Φ⁻¹, builds the RHS, and computes `coeff = Φ⁻¹ * rhs` (BLAS gemv).
Returns the stencil offsets, coefficient vector (aliasing `coeff_buf`), and grid spacings.
"""
@inline function _phs_solve_stencil!(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        base_idx::NTuple{N, Int},
        rhs_buf::AbstractVector,
        coeff_buf::AbstractVector,
    ) where {Tg, Tv, N, K}
    hs_local   = itp.hs
    grid_sizes = ntuple(d -> length(itp.grids[d]), N)
    shift = _phs_compute_shift(base_idx, itp.stencil_lo, itp.stencil_hi, grid_sizes)
    offsets, phi_inv = if all(iszero, shift) || !haskey(itp.shift_cache, shift)
        itp.stencil_offsets, itp.phi_inv
    else
        itp.shift_cache[shift]
    end
    M  = size(phi_inv, 1)
    actual_rhs   = length(rhs_buf)   == M ? rhs_buf   : similar(rhs_buf, M)
    actual_coeff = length(coeff_buf) == M ? coeff_buf : similar(coeff_buf, M)
    ns = length(offsets)
    @inbounds for i in ns + 1:M; actual_rhs[i] = zero(Tg); end  # zero polynomial tail only
    _phs_build_rhs!(actual_rhs, itp.data, base_idx, offsets, grid_sizes)
    LinearAlgebra.mul!(actual_coeff, phi_inv, actual_rhs)
    return offsets, actual_coeff, hs_local
end

"""
    _phs_eval_from_coeffs(coeffs, offsets, hs_local, query, base_coords, ::Val{K}, ops) -> scalar

Evaluate the local PHS interpolant given precomputed coefficients.
Dispatches to the appropriate `_phs_eval_coeffs_*` function based on `ops`.
"""
@inline function _phs_eval_from_coeffs(
        coeffs::AbstractVector,
        offsets::Vector{<:NTuple{N, Int}},
        hs_local::NTuple{N},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N},
        ::Val{K},
        ops::NTuple{N, AbstractEvalOp},
    ) where {N, K}
    n_deriv = ntuple(d -> deriv_order(ops[d]), N)
    total_order = sum(n_deriv)
    if total_order == 0
        return _phs_eval_coeffs_value(coeffs, offsets, hs_local, query, base_coords, Val{K}())
    elseif total_order == 1
        axis = let a = 0; for d in 1:N; n_deriv[d] == 1 && (a = d; break); end; a; end
        return _phs_eval_coeffs_deriv1(coeffs, offsets, hs_local, query, base_coords, Val{K}(), axis)
    elseif total_order == 2
        ax1, ax2 = 0, 0
        for d in 1:N
            if n_deriv[d] > 0
                if ax1 == 0; ax1 = d
                else;        ax2 = d; break
                end
            end
        end
        ax2 == 0 && (ax2 = ax1)
        return _phs_eval_coeffs_deriv2(coeffs, offsets, hs_local, query, base_coords, Val{K}(), ax1, ax2)
    end
    return zero(eltype(coeffs))
end

"""
    _phs_eval_stencil(itp, base_idx, query, ops, rhs_buf, coeff_buf)
        -> scalar

Evaluate the local PHS interpolant based at `base_idx` at point `query`.
`ops` is an `NTuple{N, DerivOp}` encoding which derivative to compute.

Uses pre-allocated buffers `rhs_buf` and `coeff_buf` (from AdaptiveArrayPools).
"""
@inline function _phs_eval_stencil(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        base_idx::NTuple{N, Int},
        query::NTuple{N, <:Real},
        ops::NTuple{N, AbstractEvalOp},
        rhs_buf::AbstractVector,
        coeff_buf::AbstractVector,
    ) where {Tg, Tv, N, K}
    offsets, coeff, hs_local = _phs_solve_stencil!(itp, base_idx, rhs_buf, coeff_buf)
    base_coords = _phs_base_coords(itp, base_idx)
    return _phs_eval_from_coeffs(coeff, offsets, hs_local, query, base_coords, Val{K}(), ops)
end

# ======================================================
# Layer 3: Blended evaluation
# ======================================================

"""
    _phs_eval_blended(itp, query, ops)
        -> scalar (or zero for out-of-domain)

Evaluate the C²-continuous blended PHS interpolant at `query`.

Algorithm:
  F = N/W  where  N = Σ wᵢ fᵢ,  W = Σ wᵢ
  Derivatives via exact quotient rule:
    F'   = (N' - F·W') / W
    F''  = (N'' - 2·N'·W'/W - F·W'' + 2·F·(W')²/W) / W
"""
@with_pool pool function _phs_eval_blended(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        query::NTuple{N, <:Real},
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, K}
    blend_a   = itp.blend_a
    blend_a3  = itp.blend_a3
    base_idx0 = _phs_find_base_node(itp, query)
    r_idx     = itp.blend_r_idx
    grid_sizes = ntuple(d -> length(itp.grids[d]), N)

    total_deriv = sum(deriv_order(ops[d]) for d in 1:N)

    # Thread-local scratch space — M = max(phi_inv matrix size) across all stencils
    M         = size(itp.phi_inv, 1)
    rhs_buf   = acquire!(pool, Tg, M)
    coeff_buf = acquire!(pool, Tg, M)

    lo_idx = ntuple(d -> max(1, base_idx0[d] - r_idx[d]), N)
    hi_idx = ntuple(d -> min(grid_sizes[d], base_idx0[d] + r_idx[d]), N)
    ranges = ntuple(d -> lo_idx[d]:hi_idx[d], N)

    ops_val = ntuple(_ -> EvalValue(), Val(N))

    # ----------------------------------------------------------------
    # Branch 1: value  F = N/W
    # ----------------------------------------------------------------
    if total_deriv == 0
        sum_w  = zero(Tg)
        sum_wy = zero(Tv)
        for nb_ci in CartesianIndices(ranges)
            nb_idx    = Tuple(nb_ci)
            nb_coords = _phs_base_coords(itp, nb_idx)
            d2 = zero(Tg)
            @inbounds for dim in 1:N; Δ = Tg(query[dim]) - nb_coords[dim]; d2 += Δ * Δ; end
            w = _phs_blend_weight(sqrt(d2), blend_a, blend_a3)
            w < eps(Tg) && continue
            f = Tv(_phs_eval_stencil(itp, nb_idx, query, ops_val, rhs_buf, coeff_buf))
            sum_w  += w
            sum_wy += w * f
        end
        sum_w < eps(Tg) && return zero(Tv)
        return sum_wy / sum_w

    # ----------------------------------------------------------------
    # Branch 2: first derivative
    #   ∂F/∂xξ = (∂N/∂xξ - F·∂W/∂xξ) / W
    #   ∂N/∂xξ = Σ(w'ᵢ·dirξ·fᵢ + wᵢ·∂fᵢ/∂xξ)
    #   ∂W/∂xξ = Σ  w'ᵢ·dirξ
    # ----------------------------------------------------------------
    elseif total_deriv == 1
        grad_ax = let a = 0; for d in 1:N; deriv_order(ops[d]) == 1 && (a = d; break); end; a; end
        sum_w   = zero(Tg)
        sum_wy  = zero(Tv)
        sum_N1  = zero(Tv)   # ∂N/∂xξ
        sum_W1  = zero(Tg)   # ∂W/∂xξ
        for nb_ci in CartesianIndices(ranges)
            nb_idx    = Tuple(nb_ci)
            nb_coords = _phs_base_coords(itp, nb_idx)
            d2 = zero(Tg)
            @inbounds for dim in 1:N; Δ = Tg(query[dim]) - nb_coords[dim]; d2 += Δ * Δ; end
            d_dist = sqrt(d2)
            w, wp = _phs_blend_weight_and_prime(d_dist, blend_a, blend_a3)
            w < eps(Tg) && continue
            offsets_nb, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)
            if d_dist > eps(Tg)
                f, df = _phs_eval_coeffs_value_and_deriv1(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), grad_ax)
                f  = Tv(f)
                df = Tv(df)
                dir = (Tg(query[grad_ax]) - nb_coords[grad_ax]) / d_dist
                sum_w  += w
                sum_wy += w * f
                sum_N1 += wp * dir * f + w * df
                sum_W1 += wp * dir
            else
                f = Tv(_phs_eval_from_coeffs(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ops_val))
                sum_w  += w
                sum_wy += w * f
            end
        end
        sum_w < eps(Tg) && return zero(Tv)
        F = sum_wy / sum_w
        return Tv((sum_N1 - F * sum_W1) / sum_w)

    # ----------------------------------------------------------------
    # Branch 3: second (or mixed) derivative — full quotient rule
    #   F'' = (N'' - 2·N'·W'/W - F·W'' + 2·F·(W')²/W) / W   [diagonal]
    #   or the mixed analogue for ax1 ≠ ax2
    # ----------------------------------------------------------------
    elseif total_deriv == 2
        n_deriv_arr = ntuple(d -> deriv_order(ops[d]), Val(N))
        ax1, ax2 = 0, 0
        for d in 1:N
            if n_deriv_arr[d] > 0
                if ax1 == 0; ax1 = d
                else;        ax2 = d; break
                end
            end
        end
        ax2 == 0 && (ax2 = ax1)
        is_diag = (ax1 == ax2)

        sum_w   = zero(Tg); sum_wy  = zero(Tv)
        sum_N2  = zero(Tv); sum_W2  = zero(Tg)  # ∂²N, ∂²W  w.r.t. the requested axes
        sum_N1  = zero(Tv); sum_W1  = zero(Tg)  # ∂N/∂x_ax1, ∂W/∂x_ax1
        sum_N1b = zero(Tv); sum_W1b = zero(Tg)  # ∂N/∂x_ax2, ∂W/∂x_ax2 (mixed only)

        for nb_ci in CartesianIndices(ranges)
            nb_idx    = Tuple(nb_ci)
            nb_coords = _phs_base_coords(itp, nb_idx)
            d2 = zero(Tg)
            @inbounds for dim in 1:N; Δ = Tg(query[dim]) - nb_coords[dim]; d2 += Δ * Δ; end
            d_dist = sqrt(d2)
            w, wp, wpp = _phs_blend_weight_and_derivs(d_dist, blend_a, blend_a3)
            w < eps(Tg) && continue

            offsets_nb, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)

            if d_dist > eps(Tg)
                da1  = (Tg(query[ax1]) - nb_coords[ax1]) / d_dist
                wxi1 = wp * da1

                if is_diag
                    f, f1, f2 = _phs_eval_coeffs_value_and_deriv1_and_deriv2(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ax1, ax1)
                    f = Tv(f); f1 = Tv(f1); f2 = Tv(f2)
                    sum_w += w; sum_wy += w * f
                    sum_N1 += wxi1 * f + w * f1
                    sum_W1 += wxi1
                    wxixi = wpp * da1 * da1 + wp * (1 - da1 * da1) / d_dist
                    sum_N2 += wxixi * f + 2 * wxi1 * f1 + w * f2
                    sum_W2 += wxixi
                else
                    da2  = (Tg(query[ax2]) - nb_coords[ax2]) / d_dist
                    wxi2 = wp * da2
                    f, f1, f1b, f2 = _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ax1, ax2)
                    f = Tv(f); f1 = Tv(f1); f1b = Tv(f1b); f2 = Tv(f2)
                    sum_w += w; sum_wy += w * f
                    sum_N1 += wxi1 * f + w * f1
                    sum_W1 += wxi1
                    sum_N1b += wxi2 * f + w * f1b
                    sum_W1b += wxi2
                    wxixi = wpp * da1 * da2 - wp * da1 * da2 / d_dist
                    sum_N2 += wxixi * f + wxi1 * f1b + wxi2 * f1 + w * f2
                    sum_W2 += wxixi
                end
            else
                # d≈0: blend-weight derivatives ≈ 0, only stencil contribution survives
                f  = Tv(_phs_eval_from_coeffs(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ops_val))
                f2 = Tv(_phs_eval_from_coeffs(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ops))
                sum_w += w; sum_wy += w * f
                sum_N2 += w * f2
            end
        end

        sum_w < eps(Tg) && return zero(Tv)
        W = sum_w
        F = sum_wy / W

        if is_diag
            # F'' = N''/W - 2·N'·W'/W² - F·W''/W + 2·F·(W'/W)²
            d2F = sum_N2 / W - 2 * sum_N1 * sum_W1 / W^2 - F * sum_W2 / W + 2 * F * (sum_W1 / W)^2
        else
            # ∂²F/∂x_ξ∂x_ζ = N''_ξζ/W - (N'_ξ·W'_ζ + N'_ζ·W'_ξ)/W² - F·W''_ξζ/W + 2F·W'_ξ·W'_ζ/W²
            d2F = sum_N2 / W - (sum_N1 * sum_W1b + sum_N1b * sum_W1) / W^2 -
                  F * sum_W2 / W + 2 * F * sum_W1 * sum_W1b / W^2
        end
        return Tv(d2F)
    end

    return zero(Tv)
end

# ======================================================
# Smoothing transform wrapper — exp-space blending
# ======================================================
#
# The Fortran reference blends g_i(x) = exp(f_i(x)) across base nodes (where
# f_i is the PHS interpolant of log(ρ/ρ₀) at base node i), then multiplies by
# ρ₀.  This is different from blending f_i and applying exp afterward.
#
#   ρ̃(x) = ρ₀(x) · G(x)       G = blend(exp(f_i))          (value, Eq. 21)
#   ∂ρ̃/∂xξ = ∂ρ₀/∂xξ · G + ρ₀ · ∂G/∂xξ                    (gradient, Leibniz)
#   ∂²ρ̃/∂xξ∂xζ = ∂²ρ₀/∂xξ∂xζ·G + ∂ρ₀/∂xξ·∂G/∂xζ
#                + ∂ρ₀/∂xζ·∂G/∂xξ + ρ₀·∂²G/∂xξ∂xζ          (Hessian, Leibniz)
#
# Within the blend loop the chain rules for g_i = exp(f_i) are:
#   dg_ξ     = g · f_ξ
#   d2g_ξξ   = g · (f_ξξ  + f_ξ²)
#   d2g_ξζ   = g · (f_ξζ  + f_ξ · f_ζ)   (ξ ≠ ζ)

"""
    _phs_eval_blended_G(itp, query, ops, rhs_buf, coeff_buf)

Evaluate the weighted blend of exp(f_i) (and its requested derivative) at `query`.
Identical accumulation structure to `_phs_eval_blended`, but replaces f_i with
g_i = exp(f_i) and propagates derivatives via the chain rule.
"""
@with_pool pool function _phs_eval_blended_G(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        query::NTuple{N, <:Real},
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, K}
    blend_a    = itp.blend_a
    blend_a3   = itp.blend_a3
    base_idx0  = _phs_find_base_node(itp, query)
    r_idx      = itp.blend_r_idx
    grid_sizes = ntuple(d -> length(itp.grids[d]), N)

    total_deriv = sum(deriv_order(ops[d]) for d in 1:N)

    M         = size(itp.phi_inv, 1)
    rhs_buf   = acquire!(pool, Tg, M)
    coeff_buf = acquire!(pool, Tg, M)

    lo_idx = ntuple(d -> max(1, base_idx0[d] - r_idx[d]), N)
    hi_idx = ntuple(d -> min(grid_sizes[d], base_idx0[d] + r_idx[d]), N)
    ranges = ntuple(d -> lo_idx[d]:hi_idx[d], N)

    ops_val = ntuple(_ -> EvalValue(), Val(N))

    # ----------------------------------------------------------------
    # Branch 1: G = Σ(wᵢ · gᵢ) / Σwᵢ  where  gᵢ = exp(fᵢ)
    # ----------------------------------------------------------------
    if total_deriv == 0
        sum_w  = zero(Tg)
        sum_wg = zero(Tv)
        for nb_ci in CartesianIndices(ranges)
            nb_idx    = Tuple(nb_ci)
            nb_coords = _phs_base_coords(itp, nb_idx)
            d2 = zero(Tg)
            @inbounds for dim in 1:N; Δ = Tg(query[dim]) - nb_coords[dim]; d2 += Δ * Δ; end
            w = _phs_blend_weight(sqrt(d2), blend_a, blend_a3)
            w < eps(Tg) && continue
            f = Tv(_phs_eval_stencil(itp, nb_idx, query, ops_val, rhs_buf, coeff_buf))
            sum_w  += w
            sum_wg += w * exp(f)
        end
        sum_w < eps(Tg) && return zero(Tv)
        return sum_wg / sum_w

    # ----------------------------------------------------------------
    # Branch 2: ∂G/∂xξ via quotient rule with gᵢ = exp(fᵢ)
    #   N = Σ(wᵢ · gᵢ),  N_ξ = Σ(w'ᵢ · dirξ · gᵢ  +  wᵢ · gᵢ · fᵢ_ξ)
    # ----------------------------------------------------------------
    elseif total_deriv == 1
        grad_ax = let a = 0; for d in 1:N; deriv_order(ops[d]) == 1 && (a = d; break); end; a; end
        sum_w   = zero(Tg)
        sum_wg  = zero(Tv)
        sum_N1  = zero(Tv)
        sum_W1  = zero(Tg)
        for nb_ci in CartesianIndices(ranges)
            nb_idx    = Tuple(nb_ci)
            nb_coords = _phs_base_coords(itp, nb_idx)
            d2 = zero(Tg)
            @inbounds for dim in 1:N; Δ = Tg(query[dim]) - nb_coords[dim]; d2 += Δ * Δ; end
            d_dist = sqrt(d2)
            w, wp = _phs_blend_weight_and_prime(d_dist, blend_a, blend_a3)
            w < eps(Tg) && continue
            offsets_nb, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)
            if d_dist > eps(Tg)
                f, f_ξ = _phs_eval_coeffs_value_and_deriv1(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), grad_ax)
                f = Tv(f); f_ξ = Tv(f_ξ)
                g  = exp(f)
                dir = (Tg(query[grad_ax]) - nb_coords[grad_ax]) / d_dist
                sum_w  += w
                sum_wg += w * g
                sum_N1 += wp * dir * g + w * g * f_ξ
                sum_W1 += wp * dir
            else
                f = Tv(_phs_eval_from_coeffs(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ops_val))
                g = exp(f)
                sum_w  += w
                sum_wg += w * g
            end
        end
        sum_w < eps(Tg) && return zero(Tv)
        G = sum_wg / sum_w
        return Tv((sum_N1 - G * sum_W1) / sum_w)

    # ----------------------------------------------------------------
    # Branch 3: ∂²G/∂xξ∂xζ via quotient rule with gᵢ = exp(fᵢ)
    #   diagonal (ξ=ζ): d2gᵢ = gᵢ · (fᵢ_ξξ + fᵢ_ξ²)
    #   mixed   (ξ≠ζ): d2gᵢ = gᵢ · (fᵢ_ξζ + fᵢ_ξ · fᵢ_ζ)
    # ----------------------------------------------------------------
    elseif total_deriv == 2
        n_deriv_arr = ntuple(d -> deriv_order(ops[d]), Val(N))
        ax1, ax2 = 0, 0
        for d in 1:N
            if n_deriv_arr[d] > 0
                if ax1 == 0; ax1 = d
                else;        ax2 = d; break
                end
            end
        end
        ax2 == 0 && (ax2 = ax1)
        is_diag = (ax1 == ax2)

        ops_d1_1 = ntuple(d -> d == ax1 ? EvalDeriv1() : EvalValue(), Val(N))  # kept for d≈0 branch

        sum_w   = zero(Tg); sum_wg  = zero(Tv)
        sum_N2  = zero(Tv); sum_W2  = zero(Tg)
        sum_N1  = zero(Tv); sum_W1  = zero(Tg)
        sum_N1b = zero(Tv); sum_W1b = zero(Tg)

        for nb_ci in CartesianIndices(ranges)
            nb_idx    = Tuple(nb_ci)
            nb_coords = _phs_base_coords(itp, nb_idx)
            d2 = zero(Tg)
            @inbounds for dim in 1:N; Δ = Tg(query[dim]) - nb_coords[dim]; d2 += Δ * Δ; end
            d_dist = sqrt(d2)
            w, wp, wpp = _phs_blend_weight_and_derivs(d_dist, blend_a, blend_a3)
            w < eps(Tg) && continue

            offsets_nb, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)

            if d_dist > eps(Tg)
                da1  = (Tg(query[ax1]) - nb_coords[ax1]) / d_dist
                wxi1 = wp * da1

                if is_diag
                    f, f_d1, f_d2 = _phs_eval_coeffs_value_and_deriv1_and_deriv2(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ax1, ax1)
                    f = Tv(f); f_d1 = Tv(f_d1); f_d2 = Tv(f_d2)
                    g    = exp(f)
                    dg1  = g * f_d1
                    d2g  = g * (f_d2 + f_d1 * f_d1)
                    wxixi = wpp * da1 * da1 + wp * (1 - da1 * da1) / d_dist
                    sum_w += w; sum_wg += w * g
                    sum_N1 += wxi1 * g + w * dg1
                    sum_W1 += wxi1
                    sum_N2 += wxixi * g + 2 * wxi1 * dg1 + w * d2g
                    sum_W2 += wxixi
                else
                    da2   = (Tg(query[ax2]) - nb_coords[ax2]) / d_dist
                    wxi2  = wp * da2
                    f, f_d1, f_d1b, f_d2 = _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ax1, ax2)
                    f = Tv(f); f_d1 = Tv(f_d1); f_d1b = Tv(f_d1b); f_d2 = Tv(f_d2)
                    g    = exp(f)
                    dg1  = g * f_d1
                    dg2  = g * f_d1b
                    d2g  = g * (f_d2 + f_d1 * f_d1b)
                    wxixi = wpp * da1 * da2 - wp * da1 * da2 / d_dist
                    sum_w += w; sum_wg += w * g
                    sum_N1 += wxi1 * g + w * dg1
                    sum_W1 += wxi1
                    sum_N1b += wxi2 * g + w * dg2
                    sum_W1b += wxi2
                    sum_N2 += wxixi * g + wxi1 * dg2 + wxi2 * dg1 + w * d2g
                    sum_W2 += wxixi
                end
            else
                # d≈0: weight derivatives ≈ 0
                f    = Tv(_phs_eval_from_coeffs(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ops_val))
                g    = exp(f)
                f_d2 = Tv(_phs_eval_from_coeffs(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ops))
                # For the d≈0 diagonal case, f_d1 contributes to d2g; since it's rare, compute separately
                f_d1_sq = is_diag ? Tv(_phs_eval_from_coeffs(coeff_nb, offsets_nb, hs_nb, query, nb_coords, Val{K}(), ops_d1_1))^2 : zero(Tv)
                d2g = g * (f_d2 + f_d1_sq)
                sum_w += w; sum_wg += w * g
                sum_N2 += w * d2g
            end
        end

        sum_w < eps(Tg) && return zero(Tv)
        W = sum_w
        G = sum_wg / W

        if is_diag
            d2G = sum_N2 / W - 2 * sum_N1 * sum_W1 / W^2 - G * sum_W2 / W + 2 * G * (sum_W1 / W)^2
        else
            d2G = sum_N2 / W - (sum_N1 * sum_W1b + sum_N1b * sum_W1) / W^2 -
                  G * sum_W2 / W + 2 * G * sum_W1 * sum_W1b / W^2
        end
        return Tv(d2G)
    end

    return zero(Tv)
end

"""
    _phs_eval_with_transform(itp, query, ops)

Evaluate the PHS interpolant with the log-density smoothing transform.
Blends g_i = exp(f_i) across base nodes (matching the reference Fortran
implementation), then applies the Leibniz rule to recover ρ̃ = ρ₀ · G
and its derivatives.

  value:    ρ̃ = ρ₀ · G
  gradient: ∂ρ̃/∂xξ = ∂ρ₀/∂xξ · G + ρ₀ · ∂G/∂xξ
  Hessian:  ∂²ρ̃/∂xξ∂xζ = ∂²ρ₀/∂xξ∂xζ · G + ∂ρ₀/∂xξ · ∂G/∂xζ
                          + ∂ρ₀/∂xζ · ∂G/∂xξ + ρ₀ · ∂²G/∂xξ∂xζ

# Fused-evaluation protocol for reference callables

When the reference callable `ref` supports the fused protocol, the gradient
and Hessian paths use a single call rather than 2–4 separate calls.  The
protocol consists of two overloadable methods:

  _phs_ref_eval_value_and_grad(ref, query, ax, ops_val, ops_grad)
      → (rho0::Real, rho0_ξ::Real)

  _phs_ref_eval_all(ref, query, ax1, ax2, ops_val, ops_d1, ops_d2, ops)
      → (rho0::Real, rho0_ξ::Real, rho0_ζ::Real, rho0_ξζ::Real)

Default implementations (below) make the same 2 / 4 separate calls as
before.  Custom reference types (e.g. `PromolecularRef`) can override with
a single atom-loop pass.
"""

# ── Default (separate-call) implementations ──────────────────────────────────

@inline function _phs_ref_eval_value_and_grad(ref, query, ax, ::Any, ops_grad)
    return ref(query), ref(query; deriv = ops_grad)
end

@inline function _phs_ref_eval_all(ref, query, ax1, ax2, ::Any, ops_d1, ops_d2, ops)
    rho0    = ref(query)
    rho0_ξ  = ref(query; deriv = ops_d1)
    rho0_ζ  = ax1 == ax2 ? rho0_ξ : ref(query; deriv = ops_d2)
    rho0_ξζ = ref(query; deriv = ops)
    return rho0, rho0_ξ, rho0_ζ, rho0_ξζ
end

# ── Body ─────────────────────────────────────────────────────────────────────

function _phs_eval_with_transform(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        query::NTuple{N, <:Real},
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, K}
    total_deriv = sum(deriv_order(ops[d]) for d in 1:N)
    ref = itp.transform.reference

    ops_val = ntuple(_ -> EvalValue(), Val(N))
    G    = Tg(_phs_eval_blended_G(itp, query, ops_val))

    total_deriv == 0 && return Tv(Tg(ref(query)) * G)

    if total_deriv == 1
        ax = let a = 0; for d in 1:N; deriv_order(ops[d]) == 1 && (a = d; break); end; a; end
        ops_grad = ntuple(d -> d == ax ? DerivOp{1}() : EvalValue(), N)
        G_ξ      = Tg(_phs_eval_blended_G(itp, query, ops_grad))
        rho0, rho0_ξ = _phs_ref_eval_value_and_grad(ref, query, ax, ops_val, ops_grad)
        return Tv(Tg(rho0_ξ) * G + Tg(rho0) * G_ξ)
    end

    if total_deriv == 2
        ax1, ax2 = 0, 0
        for d in 1:N
            if deriv_order(ops[d]) > 0
                if ax1 == 0; ax1 = d
                else;        ax2 = d; break
                end
            end
        end
        ax2 == 0 && (ax2 = ax1)

        ops_d1 = ntuple(d -> d == ax1 ? DerivOp{1}() : EvalValue(), N)
        ops_d2 = ntuple(d -> d == ax2 ? DerivOp{1}() : EvalValue(), N)

        G_ξ    = Tg(_phs_eval_blended_G(itp, query, ops_d1))
        G_ζ    = Tg(_phs_eval_blended_G(itp, query, ops_d2))
        G_ξζ   = Tg(_phs_eval_blended_G(itp, query, ops))

        rho0, rho0_ξ, rho0_ζ, rho0_ξζ =
            _phs_ref_eval_all(ref, query, ax1, ax2, ops_val, ops_d1, ops_d2, ops)

        # Leibniz rule: ρ̃_ξζ = ρ₀_ξζ·G + ρ₀_ξ·G_ζ + ρ₀_ζ·G_ξ + ρ₀·G_ξζ
        return Tv(Tg(rho0_ξζ) * G + Tg(rho0_ξ) * G_ζ + Tg(rho0_ζ) * G_ξ + Tg(rho0) * G_ξζ)
    end

    return zero(Tv)
end

# ======================================================
# Top-level dispatch: with or without transform
# ======================================================
# Julia does not allow partial type-parameter specification in dispatch signatures
# (e.g. Foo{A,B,Nothing} when Foo has 9 params). Instead we dispatch via a
# two-argument helper that specialises on the transform field type, which the
# compiler will constant-fold since T is a type parameter of PHSInterpolantND.

@inline _phs_eval_dispatch(itp, ::Nothing, query, ops) = _phs_eval_blended(itp, query, ops)
@inline _phs_eval_dispatch(itp, ::Any,     query, ops) = _phs_eval_with_transform(itp, query, ops)

@inline function _phs_eval(
        itp::PHSInterpolantND,
        query::NTuple{N, <:Real},
        ops::NTuple{N, AbstractEvalOp},
    ) where {N}
    return _phs_eval_dispatch(itp, itp.transform, query, ops)
end
