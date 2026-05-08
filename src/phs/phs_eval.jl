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
# Stencil Coefficient Cache (thread-local)
# ======================================================
#
# Task-local cache of precomputed stencil coefficient vectors.
# Key:   NTuple{N,Int}  — blend-neighbour base-node grid index
# Value: Vector{Tg}     — coeff = Φ⁻¹ · rhs (length M = stencil_size^N + n_poly)
#
# Coefficients depend only on `itp.data` (fixed after construction) and the
# stencil geometry (also fixed), so they are safe to cache indefinitely per task.
# Thread safety is implicit: each Julia Task gets an independent Dict via
# task_local_storage — no locks needed.
#
# A per-interpolant sub-Dict (keyed by objectid) ensures that multiple
# PHSInterpolantND instances co-exist in the same task without interference.
# The cache is bounded to _PHS_COEFF_CACHE_MAX entries per interpolant to
# prevent unbounded memory use when evaluating over large 3D grids.

const _PHS_COEFF_CACHE_TKEY = :_phs_stencil_coeff_cache
const _PHS_COEFF_CACHE_MAX  = 5_000   # ≈ 20 MB for 516-coeff Float64 stencils

@inline function _phs_get_coeff_cache(
        itp::PHSInterpolantND{Tg, Tv, N, K},
    ) where {Tg, Tv, N, K}
    tid = Threads.threadid()
    # Fallback if somehow more threads were spawned than existed at creation time
    if tid > length(itp.coeff_caches)
        return Dict{NTuple{N, Int}, Vector{Tg}}()
    end
    return itp.coeff_caches[tid]::Dict{NTuple{N, Int}, Vector{Tg}}
end

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

# (Note: _phs_diff has been moved to phs_kernels.jl as a compile-time generated/unrolled function)

"""
    _phs_eval_coeffs_value(coeffs, phys_offsets, query, base_coords, ::Val{K}) -> scalar

Evaluate the PHS interpolant value at `query` given precomputed coefficients.
  f = Σᵢ wᵢ φ(rᵢ) + v₀ + Σⱼ vⱼ (xⱼ - xbase_j)
"""
@inline function _phs_eval_coeffs_value(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
    ) where {Tv, Tg, N, K}
    ns = length(phys_offsets)
    y = zero(Tv)

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))

    # RBF sum
    @fastmath @inbounds @simd for i in 1:ns
        xh = _phs_diff_Δ(Δx, phys_offsets[i])
        r = sqrt(_phs_sum_sq(xh))
        y += coeffs[i] * _phs_phi(r, Val{K}())
    end

    # Polynomial augmentation: all monomials up to degree (K-1)÷2
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    y += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    return y
end

"""
    _phs_eval_coeffs_deriv1(coeffs, phys_offsets, query, base_coords, ::Val{K}, axis) -> scalar

Evaluate ∂f/∂xξ (Eq. 25).
  fξ = Σᵢ wᵢ φ'(rᵢ) (xξ - xiξ)/rᵢ + vξ
"""
@inline function _phs_eval_coeffs_deriv1(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        axis::Int,
    ) where {Tv, Tg, N, K}
    return _phs_eval_coeffs_deriv1(coeffs, phys_offsets, query, base_coords, Val{K}(), Val(axis))
end

@inline function _phs_eval_coeffs_deriv1(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ::Val{axis},
    ) where {Tv, Tg, N, K, axis}
    ns = length(phys_offsets)
    y = zero(Tv)

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))

    if K == 3
        @fastmath @inbounds @simd for i in 1:ns
            xh = _phs_diff_Δ(Δx, phys_offsets[i])
            r = sqrt(_phs_sum_sq(xh))
            y += (3 * coeffs[i] * r) * xh[axis]
        end
    else
        eps_tg = eps(Tg)
        @fastmath @inbounds @simd for i in 1:ns
            xh = _phs_diff_Δ(Δx, phys_offsets[i])
            r = sqrt(_phs_sum_sq(xh))
            r_inv = ifelse(r < eps_tg, zero(Tg), one(Tg) / r)
            ci = coeffs[i]
            ci_fp_r_inv = ci * _phs_phi_prime(r, Val{K}()) * r_inv
            y += ci_fp_r_inv * xh[axis]
        end
    end
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    y += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, Val(axis))
    return y
end

"""
    _phs_eval_coeffs_deriv2(coeffs, phys_offsets, query, base_coords, ::Val{K}, ax1, ax2) -> scalar

Evaluate ∂²f/∂xξ∂xζ (Eq. 26).
"""
@inline function _phs_eval_coeffs_deriv2(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ax1::Int,
        ax2::Int,
    ) where {Tv, Tg, N, K}
    return _phs_eval_coeffs_deriv2(coeffs, phys_offsets, query, base_coords, Val{K}(), Val(ax1), Val(ax2))
end

@inline function _phs_eval_coeffs_deriv2(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ::Val{ax1},
        ::Val{ax2},
    ) where {Tv, Tg, N, K, ax1, ax2}
    ns = length(phys_offsets)
    y = zero(Tv)

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))

    if K == 3
        eps2 = eps(Tg)^2
        if ax1 == ax2
            @fastmath @inbounds @simd for i in 1:ns
                xh = _phs_diff_Δ(Δx, phys_offsets[i])
                r2 = _phs_sum_sq(xh)
                r = sqrt(r2)
                r2_inv = ifelse(r2 < eps2, zero(Tg), one(Tg) / r2)
                ci_3r = 3 * coeffs[i] * r
                factor = xh[ax1] * xh[ax1] * r2_inv
                y += ci_3r * (one(Tg) + factor)
            end
        else
            @fastmath @inbounds @simd for i in 1:ns
                xh = _phs_diff_Δ(Δx, phys_offsets[i])
                r2 = _phs_sum_sq(xh)
                r = sqrt(r2)
                r2_inv = ifelse(r2 < eps2, zero(Tg), one(Tg) / r2)
                ci_3r = 3 * coeffs[i] * r
                factor = xh[ax1] * xh[ax2] * r2_inv
                y += ci_3r * factor
            end
        end
    else
        eps2 = eps(Tg)^2
        if ax1 == ax2
            @fastmath @inbounds @simd for i in 1:ns
                xh = _phs_diff_Δ(Δx, phys_offsets[i])
                r2 = _phs_sum_sq(xh)
                r = sqrt(r2)
                r_inv  = ifelse(r2 < eps2, zero(Tg), one(Tg) / r)
                r2_inv = r_inv * r_inv
                fp  = _phs_phi_prime(r, Val{K}())
                fpp = _phs_phi_dprime(r, Val{K}())
                ci = coeffs[i]
                ci_fp_r_inv = ci * fp * r_inv
                factor = xh[ax1] * xh[ax1] * r2_inv
                y += ci * fpp * factor + ci_fp_r_inv * (one(Tg) - factor)
            end
        else
            @fastmath @inbounds @simd for i in 1:ns
                xh = _phs_diff_Δ(Δx, phys_offsets[i])
                r2 = _phs_sum_sq(xh)
                r = sqrt(r2)
                r_inv  = ifelse(r2 < eps2, zero(Tg), one(Tg) / r)
                r2_inv = r_inv * r_inv
                fp  = _phs_phi_prime(r, Val{K}())
                fpp = _phs_phi_dprime(r, Val{K}())
                ci = coeffs[i]
                ci_fp_r_inv = ci * fp * r_inv
                factor = xh[ax1] * xh[ax2] * r2_inv
                y += (ci * fpp - ci_fp_r_inv) * factor
            end
        end
    end
    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    y += _phs_eval_poly_deriv2(Δx, poly_exps, coeffs, ns, Val(ax1), Val(ax2))
    return y
end

"""
    _phs_eval_coeffs_value_and_deriv1(coeffs, phys_offsets, query, base_coords, ::Val{K}, axis)
        -> (value, deriv1)

Fused single-pass evaluation of both the interpolant value and its first derivative
along `axis`.  Avoids traversing the stencil twice when both are needed (gradient blending).
"""
@inline function _phs_eval_coeffs_value_and_deriv1(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        axis::Int,
    ) where {Tv, Tg, N, K}
    return _phs_eval_coeffs_value_and_deriv1(coeffs, phys_offsets, query, base_coords, Val{K}(), Val(axis))
end

@inline function _phs_eval_coeffs_value_and_deriv1(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ::Val{axis},
    ) where {Tv, Tg, N, K, axis}
    ns = length(phys_offsets)
    yv = zero(Tv)
    yd = zero(Tv)

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))

    if K == 3
        @fastmath @inbounds @simd for i in 1:ns
            xh = _phs_diff_Δ(Δx, phys_offsets[i])
            r2 = _phs_sum_sq(xh)
            r  = sqrt(r2)
            ci = coeffs[i]
            ci_r = ci * r
            ci_3r = 3 * ci_r
            yv += ci_r * r2
            yd += ci_3r * xh[axis]
        end
    else
        eps_tg = eps(Tg)
        @fastmath @inbounds @simd for i in 1:ns
            xh = _phs_diff_Δ(Δx, phys_offsets[i])
            r2 = _phs_sum_sq(xh)
            r  = sqrt(r2)
            ci = coeffs[i]
            r_inv = ifelse(r < eps_tg, zero(Tg), one(Tg) / r)
            ci_fp_r_inv = ci * _phs_phi_prime(r, Val{K}()) * r_inv
            yv += ci * _phs_phi(r, Val{K}())
            yd += ci_fp_r_inv * xh[axis]
        end
    end

    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    yv += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    yd += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, Val(axis))
    return yv, yd
end

"""
    _phs_eval_coeffs_value_and_deriv1_and_deriv2(coeffs, phys_offsets, query, base_coords,
        ::Val{K}, ax1, ax2) -> (value, deriv1_ax1, deriv2)

Fused single-pass evaluation of value, ∂f/∂x_{ax1}, and ∂²f/∂x_{ax1}∂x_{ax2}.
Used for Hessian blending (diagonal and mixed second derivatives).
For diagonal (ax1==ax2): returns (f, f_ξ, f_ξξ).
For mixed (ax1≠ax2): returns (f, f_ξ, f_ξζ) where first-deriv is w.r.t. ax1.
"""
@inline function _phs_eval_coeffs_value_and_deriv1_and_deriv2(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ax1::Int,
        ax2::Int,
    ) where {Tv, Tg, N, K}
    return _phs_eval_coeffs_value_and_deriv1_and_deriv2(coeffs, phys_offsets, query, base_coords, Val{K}(), Val(ax1), Val(ax2))
end

@inline function _phs_eval_coeffs_value_and_deriv1_and_deriv2(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ::Val{ax1},
        ::Val{ax2},
    ) where {Tv, Tg, N, K, ax1, ax2}
    ns = length(phys_offsets)
    yv  = zero(Tv)
    yd1 = zero(Tv)
    yd2 = zero(Tv)

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))

    if K == 3
        eps2 = eps(Tg)^2
        if ax1 == ax2
            @fastmath @inbounds @simd for i in 1:ns
                xh = _phs_diff_Δ(Δx, phys_offsets[i])
                r2 = _phs_sum_sq(xh)
                r  = sqrt(r2)
                r2_inv = ifelse(r2 < eps2, zero(Tg), one(Tg) / r2)
                ci = coeffs[i]
                ci_r = ci * r
                ci_3r = 3 * ci_r
                yv  += ci_r * r2
                yd1 += ci_3r * xh[ax1]
                factor = xh[ax1] * xh[ax1] * r2_inv
                yd2 += ci_3r * (one(Tg) + factor)
            end
        else
            @fastmath @inbounds @simd for i in 1:ns
                xh = _phs_diff_Δ(Δx, phys_offsets[i])
                r2 = _phs_sum_sq(xh)
                r  = sqrt(r2)
                r2_inv = ifelse(r2 < eps2, zero(Tg), one(Tg) / r2)
                ci = coeffs[i]
                ci_r = ci * r
                ci_3r = 3 * ci_r
                yv  += ci_r * r2
                yd1 += ci_3r * xh[ax1]
                factor = xh[ax1] * xh[ax2] * r2_inv
                yd2 += ci_3r * factor
            end
        end
    else
        eps_tg = eps(Tg)
        if ax1 == ax2
            @inbounds @simd for i in 1:ns
                xh = _phs_diff_Δ(Δx, phys_offsets[i])
                r2 = _phs_sum_sq(xh)
                r  = sqrt(r2)
                ci = coeffs[i]
                r_inv  = ifelse(r < eps_tg, zero(Tg), one(Tg) / r)
                r2_inv = r_inv * r_inv
                fp  = _phs_phi_prime(r, Val{K}())
                fpp = _phs_phi_dprime(r, Val{K}())
                ci_fp_r_inv = ci * fp * r_inv
                yv  += ci * _phs_phi(r, Val{K}())
                yd1 += ci_fp_r_inv * xh[ax1]
                factor = xh[ax1] * xh[ax1] * r2_inv
                yd2 += ci * fpp * factor + ci_fp_r_inv * (one(Tg) - factor)
            end
        else
            @inbounds @simd for i in 1:ns
                xh = _phs_diff_Δ(Δx, phys_offsets[i])
                r2 = _phs_sum_sq(xh)
                r  = sqrt(r2)
                ci = coeffs[i]
                r_inv  = ifelse(r < eps_tg, zero(Tg), one(Tg) / r)
                r2_inv = r_inv * r_inv
                fp  = _phs_phi_prime(r, Val{K}())
                fpp = _phs_phi_dprime(r, Val{K}())
                ci_fp_r_inv = ci * fp * r_inv
                yv  += ci * _phs_phi(r, Val{K}())
                yd1 += ci_fp_r_inv * xh[ax1]
                factor = xh[ax1] * xh[ax2] * r2_inv
                yd2 += (ci * fpp - ci_fp_r_inv) * factor
            end
        end
    end

    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    yv  += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    yd1 += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, Val(ax1))
    yd2 += _phs_eval_poly_deriv2(Δx, poly_exps, coeffs, ns, Val(ax1), Val(ax2))
    return yv, yd1, yd2
end

"""
    _phs_eval_coeffs_value_and_two_deriv1(coeffs, phys_offsets, query, base_coords,
        ::Val{K}, ax1, ax2) -> (value, deriv1_ax1, deriv1_ax2)

Fused single-pass evaluation of value, ∂f/∂x_{ax1}, and ∂f/∂x_{ax2}.
Used for mixed-Hessian blending to get both first-derivative components in one loop.
"""
@inline function _phs_eval_coeffs_value_and_two_deriv1(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ax1::Int,
        ax2::Int,
    ) where {Tv, Tg, N, K}
    return _phs_eval_coeffs_value_and_two_deriv1(coeffs, phys_offsets, query, base_coords, Val{K}(), Val(ax1), Val(ax2))
end

@inline function _phs_eval_coeffs_value_and_two_deriv1(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ::Val{ax1},
        ::Val{ax2},
    ) where {Tv, Tg, N, K, ax1, ax2}
    ns = length(phys_offsets)
    yv  = zero(Tv)
    yd1 = zero(Tv)
    yd2 = zero(Tv)

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))

    if K == 3
        @fastmath @inbounds @simd for i in 1:ns
            xh = _phs_diff_Δ(Δx, phys_offsets[i])
            r2 = _phs_sum_sq(xh)
            r  = sqrt(r2)
            ci = coeffs[i]
            ci_r = ci * r
            ci_3r = 3 * ci_r
            yv  += ci_r * r2
            yd1 += ci_3r * xh[ax1]
            yd2 += ci_3r * xh[ax2]
        end
    else
        eps_tg = eps(Tg)
        @fastmath @inbounds @simd for i in 1:ns
            xh = _phs_diff_Δ(Δx, phys_offsets[i])
            r2 = _phs_sum_sq(xh)
            r  = sqrt(r2)
            ci = coeffs[i]
            fp_r_inv = _phs_phi_prime(r, Val{K}()) * ifelse(r < eps_tg, zero(Tg), one(Tg) / r)
            ci_fp_r_inv = ci * fp_r_inv
            yv  += ci * _phs_phi(r, Val{K}())
            yd1 += ci_fp_r_inv * xh[ax1]
            yd2 += ci_fp_r_inv * xh[ax2]
        end
    end

    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    yv  += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    yd1 += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, Val(ax1))
    yd2 += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, Val(ax2))
    return yv, yd1, yd2
end

"""
    _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(coeffs, phys_offsets, query, base_coords,
        ::Val{K}, ax1, ax2) -> (value, deriv1_ax1, deriv1_ax2, deriv2_ax1_ax2)

Fused single-pass evaluation of value, ∂f/∂x_{ax1}, ∂f/∂x_{ax2}, and ∂²f/∂x_{ax1}∂x_{ax2}
for the off-diagonal (ax1 ≠ ax2) mixed Hessian case.  Replaces the previous two-pass approach
of calling `_phs_eval_coeffs_value_and_two_deriv1` followed by `_phs_eval_coeffs_deriv2`.
"""
@inline function _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ax1::Int,
        ax2::Int,
    ) where {Tv, Tg, N, K}
    return _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(coeffs, phys_offsets, query, base_coords, Val{K}(), Val(ax1), Val(ax2))
end

@inline function _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(
        coeffs::AbstractVector{Tv},
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ::Val{ax1},
        ::Val{ax2},
    ) where {Tv, Tg, N, K, ax1, ax2}
    ns = length(phys_offsets)
    yv   = zero(Tv)
    yd1  = zero(Tv)
    yd2  = zero(Tv)
    yd12 = zero(Tv)

    Δx = ntuple(d -> Tg(query[d]) - base_coords[d], Val(N))

    if K == 3
        eps2 = eps(Tg)^2
        @fastmath @inbounds @simd for i in 1:ns
            xh = _phs_diff_Δ(Δx, phys_offsets[i])
            r2 = _phs_sum_sq(xh)
            r  = sqrt(r2)
            r2_inv = ifelse(r2 < eps2, zero(Tg), one(Tg) / r2)
            ci = coeffs[i]
            ci_r = ci * r
            ci_3r = 3 * ci_r
            yv   += ci_r * r2
            yd1  += ci_3r * xh[ax1]
            yd2  += ci_3r * xh[ax2]
            factor = xh[ax1] * xh[ax2] * r2_inv
            yd12 += ci_3r * factor
        end
    else
        eps2 = eps(Tg)^2
        @fastmath @inbounds @simd for i in 1:ns
            xh = _phs_diff_Δ(Δx, phys_offsets[i])
            r2 = _phs_sum_sq(xh)
            r  = sqrt(r2)
            ci = coeffs[i]
            r_inv  = ifelse(r2 < eps2, zero(Tg), one(Tg) / r)
            r2_inv = r_inv * r_inv
            fp  = _phs_phi_prime(r, Val{K}())
            fpp = _phs_phi_dprime(r, Val{K}())
            ci_fp_r_inv = ci * fp * r_inv
            yv   += ci * _phs_phi(r, Val{K}())
            yd1  += ci_fp_r_inv * xh[ax1]
            yd2  += ci_fp_r_inv * xh[ax2]
            factor = xh[ax1] * xh[ax2] * r2_inv
            yd12 += (ci * fpp - ci_fp_r_inv) * factor
        end
    end

    poly_exps = _phs_poly_exps_tuple(Val(N), Val(K))
    yv   += _phs_eval_poly(Δx, poly_exps, coeffs, ns)
    yd1  += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, Val(ax1))
    yd2  += _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, Val(ax2))
    yd12 += _phs_eval_poly_deriv2(Δx, poly_exps, coeffs, ns, Val(ax1), Val(ax2))
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
        rhs_buf::Vr,
        coeff_buf::Vc,
    ) where {Tg, Tv, N, K, Vr <: AbstractVector, Vc <: AbstractVector}
    hs_local   = itp.hs
    grid_sizes = ntuple(d -> length(itp.grids[d]), N)
    shift = _phs_compute_shift(base_idx, itp.stencil_lo, itp.stencil_hi, grid_sizes)
    offsets, phi_inv, phys_offsets = if all(iszero, shift) || !haskey(itp.shift_cache, shift)
        itp.stencil_offsets, itp.phi_inv, itp.stencil_phys_offsets
    else
        val = itp.shift_cache[shift]
        val[1], val[2], val[3]
    end
    # Check stencil coefficient cache (task-local; zero-alloc on hit after warm-up).
    # On cache hit, return cached vector directly — callers only read from it.
    coeff_cache = _phs_get_coeff_cache(itp)
    cached = get(coeff_cache, base_idx, nothing)
    cached !== nothing && return offsets, phys_offsets, cached, hs_local

    # Cache miss: build RHS and solve via BLAS gemv.
    M            = size(phi_inv, 1)
    actual_coeff = length(coeff_buf) == M ? coeff_buf : similar(coeff_buf, M)
    actual_rhs   = length(rhs_buf)   == M ? rhs_buf   : similar(rhs_buf,   M)
    ns = length(offsets)
    @inbounds for i in ns + 1:M; actual_rhs[i] = zero(Tg); end  # zero polynomial tail only
    _phs_build_rhs!(actual_rhs, itp.data, base_idx, offsets, grid_sizes)
    LinearAlgebra.mul!(actual_coeff, phi_inv, actual_rhs)

    # Store a copy in the cache (bounded to prevent unbounded memory growth).
    if length(coeff_cache) < _PHS_COEFF_CACHE_MAX
        coeff_cache[base_idx] = copy(actual_coeff)
    end

    return offsets, phys_offsets, actual_coeff, hs_local
end

"""
    _phs_eval_from_coeffs(coeffs, phys_offsets, query, base_coords, ::Val{K}, ops) -> scalar

Evaluate the local PHS interpolant given precomputed coefficients.
Dispatches to the appropriate `_phs_eval_coeffs_*` function based on `ops`.
"""
@inline function _phs_eval_from_coeffs(
        coeffs::AbstractVector,
        phys_offsets::Vector{<:NTuple{N, Tg}},
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        ::Val{K},
        ops::O,
    ) where {Tg, N, K, O <: Tuple{Vararg{AbstractEvalOp, N}}}
    total_order = sum(deriv_order(ops[d]) for d in 1:N)
    if total_order == 0
        return _phs_eval_coeffs_value(coeffs, phys_offsets, query, base_coords, Val{K}())
    elseif total_order == 1
        axis_val = _phs_get_deriv1_axis_val(O)
        return _phs_eval_coeffs_deriv1(coeffs, phys_offsets, query, base_coords, Val{K}(), axis_val)
    elseif total_order == 2
        ax1_val, ax2_val = _phs_get_deriv2_axes_val(O)
        return _phs_eval_coeffs_deriv2(coeffs, phys_offsets, query, base_coords, Val{K}(), ax1_val, ax2_val)
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
        ops::O,
        rhs_buf::Vr,
        coeff_buf::Vc,
    ) where {Tg, Tv, N, K, O <: Tuple{Vararg{AbstractEvalOp, N}}, Vr <: AbstractVector, Vc <: AbstractVector}
    offsets, phys_offsets, coeff, hs_local = _phs_solve_stencil!(itp, base_idx, rhs_buf, coeff_buf)
    base_coords = _phs_base_coords(itp, base_idx)
    return _phs_eval_from_coeffs(coeff, phys_offsets, query, base_coords, Val{K}(), ops)
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
        ops::O,
    ) where {Tg, Tv, N, K, O <: Tuple{Vararg{AbstractEvalOp, N}}}
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
            offsets_nb, phys_offsets, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)
            f = Tv(_phs_eval_coeffs_value(coeff_nb, phys_offsets, query, nb_coords, Val{K}()))
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
        grad_ax = findfirst(d -> deriv_order(ops[d]) == 1, 1:N)::Int
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
            offsets_nb, phys_offsets, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)
            if d_dist > eps(Tg)
                f, df = _phs_eval_coeffs_value_and_deriv1(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val(grad_ax))
                f  = Tv(f)
                df = Tv(df)
                dir = (Tg(query[grad_ax]) - nb_coords[grad_ax]) / d_dist
                sum_w  += w
                sum_wy += w * f
                sum_N1 += wp * dir * f + w * df
                sum_W1 += wp * dir
            else
                f = Tv(_phs_eval_coeffs_value(coeff_nb, phys_offsets, query, nb_coords, Val{K}()))
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
        ax1       = findfirst(d -> n_deriv_arr[d] > 0, 1:N)::Int
        ax2_maybe = ax1 < N ? findnext(d -> n_deriv_arr[d] > 0, 1:N, ax1 + 1) : nothing
        ax2       = ax2_maybe !== nothing ? ax2_maybe : ax1
        is_diag   = (ax1 == ax2)

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

            offsets_nb, phys_offsets, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)

            if d_dist > eps(Tg)
                inv_d_dist = one(Tg) / d_dist
                wp_inv = wp * inv_d_dist
                dx1  = Tg(query[ax1]) - nb_coords[ax1]
                da1  = dx1 * inv_d_dist
                wxi1 = wp_inv * dx1

                if is_diag
                    f, f1, f2 = _phs_eval_coeffs_value_and_deriv1_and_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val(ax1), Val(ax1))
                    f = Tv(f); f1 = Tv(f1); f2 = Tv(f2)
                    sum_w += w; sum_wy += w * f
                    sum_N1 += wxi1 * f + w * f1
                    sum_W1 += wxi1
                    da1_sqr = da1 * da1
                    wxixi = da1_sqr * (wpp - wp_inv) + wp_inv
                    sum_N2 += wxixi * f + 2 * wxi1 * f1 + w * f2
                    sum_W2 += wxixi
                else
                    dx2  = Tg(query[ax2]) - nb_coords[ax2]
                    da2  = dx2 * inv_d_dist
                    wxi2 = wp_inv * dx2
                    f, f1, f1b, f2 = _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val(ax1), Val(ax2))
                    f = Tv(f); f1 = Tv(f1); f1b = Tv(f1b); f2 = Tv(f2)
                    sum_w += w; sum_wy += w * f
                    sum_N1 += wxi1 * f + w * f1
                    sum_W1 += wxi1
                    sum_N1b += wxi2 * f + w * f1b
                    sum_W1b += wxi2
                    da1_da2 = da1 * da2
                    wxixi = da1_da2 * (wpp - wp_inv)
                    sum_N2 += wxixi * f + wxi1 * f1b + wxi2 * f1 + w * f2
                    sum_W2 += wxixi
                end
            else
                # d≈0: blend-weight derivatives ≈ 0, only stencil contribution survives
                f  = Tv(_phs_eval_coeffs_value(coeff_nb, phys_offsets, query, nb_coords, Val{K}()))
                f2 = Tv(_phs_eval_coeffs_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val(ax1), Val(ax2)))
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
        ops::O,
    ) where {Tg, Tv, N, K, O <: Tuple{Vararg{AbstractEvalOp, N}}}
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
            offsets_nb, phys_offsets, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)
            f = Tv(_phs_eval_coeffs_value(coeff_nb, phys_offsets, query, nb_coords, Val{K}()))
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
        grad_ax = findfirst(d -> deriv_order(ops[d]) == 1, 1:N)::Int
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
            offsets_nb, phys_offsets, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)
            if d_dist > eps(Tg)
                f, f_ξ = _phs_eval_coeffs_value_and_deriv1(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val(grad_ax))
                f = Tv(f); f_ξ = Tv(f_ξ)
                g  = exp(f)
                dir = (Tg(query[grad_ax]) - nb_coords[grad_ax]) / d_dist
                sum_w  += w
                sum_wg += w * g
                sum_N1 += wp * dir * g + w * g * f_ξ
                sum_W1 += wp * dir
            else
                f = Tv(_phs_eval_coeffs_value(coeff_nb, phys_offsets, query, nb_coords, Val{K}()))
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
        ax1       = findfirst(d -> n_deriv_arr[d] > 0, 1:N)::Int
        ax2_maybe = ax1 < N ? findnext(d -> n_deriv_arr[d] > 0, 1:N, ax1 + 1) : nothing
        ax2       = ax2_maybe !== nothing ? ax2_maybe : ax1
        is_diag   = (ax1 == ax2)

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

            offsets_nb, phys_offsets, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)

            if d_dist > eps(Tg)
                inv_d_dist = one(Tg) / d_dist
                wp_inv = wp * inv_d_dist
                dx1  = Tg(query[ax1]) - nb_coords[ax1]
                da1  = dx1 * inv_d_dist
                wxi1 = wp_inv * dx1

                if is_diag
                    f, f_d1, f_d2 = _phs_eval_coeffs_value_and_deriv1_and_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val(ax1), Val(ax1))
                    f = Tv(f); f_d1 = Tv(f_d1); f_d2 = Tv(f_d2)
                    g    = exp(f)
                    dg1  = g * f_d1
                    d2g  = g * (f_d2 + f_d1 * f_d1)
                    da1_sqr = da1 * da1
                    wxixi = da1_sqr * (wpp - wp_inv) + wp_inv
                    sum_w += w; sum_wg += w * g
                    sum_N1 += wxi1 * g + w * dg1
                    sum_W1 += wxi1
                    sum_N2 += wxixi * g + 2 * wxi1 * dg1 + w * d2g
                    sum_W2 += wxixi
                else
                    dx2   = Tg(query[ax2]) - nb_coords[ax2]
                    da2   = dx2 * inv_d_dist
                    wxi2  = wp_inv * dx2
                    f, f_d1, f_d1b, f_d2 = _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val(ax1), Val(ax2))
                    f = Tv(f); f_d1 = Tv(f_d1); f_d1b = Tv(f_d1b); f_d2 = Tv(f_d2)
                    g    = exp(f)
                    dg1  = g * f_d1
                    dg2  = g * f_d1b
                    d2g  = g * (f_d2 + f_d1 * f_d1b)
                    da1_da2 = da1 * da2
                    wxixi = da1_da2 * (wpp - wp_inv)
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
                f    = Tv(_phs_eval_coeffs_value(coeff_nb, phys_offsets, query, nb_coords, Val{K}()))
                g    = exp(f)
                f_d2 = Tv(_phs_eval_coeffs_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val(ax1), Val(ax2)))
                # For the d≈0 diagonal case, f_d1 contributes to d2g; since it's rare, compute separately
                f_d1_sq = is_diag ? Tv(_phs_eval_coeffs_deriv1(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val(ax1)))^2 : zero(Tv)
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

# ======================================================
# Fused blend functions for _phs_eval_with_transform
# ======================================================
#
# These replace 2–4 separate calls to _phs_eval_blended_G with a single
# pass over the blend neighbourhood, eliminating redundant stencil solves.
# For sequential batch queries (typical 1D paths), the stencil coefficient
# cache in _phs_solve_stencil! makes each neighbour a cheap copyto! instead
# of a full BLAS GEMV.

# Static type-level helpers for resolving derivative axes from the Tuple type O:
@inline function _phs_get_deriv1_axis(::Type{O}) where {O <: Tuple}
    for d in 1:fieldcount(O)
        if fieldtype(O, d) <: DerivOp{1}
            return d
        end
    end
    return 1
end

@generated function _phs_get_deriv1_axis_val(::Type{O}) where {O <: Tuple}
    for d in 1:fieldcount(O)
        if fieldtype(O, d) <: DerivOp{1}
            return :(Val{$d}())
        end
    end
    return :(Val{1}())
end

@inline function _phs_get_deriv2_axes(::Type{O}) where {O <: Tuple}
    ax1 = 0
    ax2 = 0
    for d in 1:fieldcount(O)
        T = fieldtype(O, d)
        if T <: DerivOp{1}
            if ax1 == 0
                ax1 = d
            else
                ax2 = d
            end
        elseif T <: DerivOp{2}
            ax1 = d
            ax2 = d
        end
    end
    return ax1, ax2
end

@generated function _phs_get_deriv2_axes_val(::Type{O}) where {O <: Tuple}
    ax1 = 0
    ax2 = 0
    for d in 1:fieldcount(O)
        T = fieldtype(O, d)
        if T <: DerivOp{1}
            if ax1 == 0
                ax1 = d
            else
                ax2 = d
            end
        elseif T <: DerivOp{2}
            ax1 = d
            ax2 = d
        end
    end
    return :(Val{$ax1}(), Val{$ax2}())
end

"""
    _phs_eval_blended_G_with_grad(itp, query, grad_ax) -> (G, G_ξ)

Evaluate G = blend(exp(fᵢ)) and ∂G/∂x_{grad_ax} in a single pass over the
blend neighbourhood, replacing the two separate `_phs_eval_blended_G` calls
that `_phs_eval_with_transform` previously made for gradient queries.
"""
@inline function _phs_eval_blended_G_with_grad(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        query::NTuple{N, <:Real},
        grad_ax::Int,
    ) where {Tg, Tv, N, K}
    return _phs_eval_blended_G_with_grad(itp, query, Val(grad_ax))
end

@with_pool pool function _phs_eval_blended_G_with_grad(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        query::NTuple{N, <:Real},
        ::Val{grad_ax},
    ) where {Tg, Tv, N, K, grad_ax}
    blend_a   = itp.blend_a
    blend_a3  = itp.blend_a3
    base_idx0 = _phs_find_base_node(itp, query)
    r_idx     = itp.blend_r_idx
    grid_sizes = ntuple(d -> length(itp.grids[d]), N)

    M         = size(itp.phi_inv, 1)
    rhs_buf   = acquire!(pool, Tg, M)
    coeff_buf = acquire!(pool, Tg, M)

    lo_idx = ntuple(d -> max(1, base_idx0[d] - r_idx[d]), N)
    hi_idx = ntuple(d -> min(grid_sizes[d], base_idx0[d] + r_idx[d]), N)
    ranges = ntuple(d -> lo_idx[d]:hi_idx[d], N)

    sum_w  = zero(Tg)
    sum_wg = zero(Tv)
    sum_N1 = zero(Tv)   # ∂N/∂x_{grad_ax}
    sum_W1 = zero(Tg)   # ∂W/∂x_{grad_ax}

    for nb_ci in CartesianIndices(ranges)
        nb_idx    = Tuple(nb_ci)
        nb_coords = _phs_base_coords(itp, nb_idx)
        d2 = zero(Tg)
        @inbounds for dim in 1:N; Δ = Tg(query[dim]) - nb_coords[dim]; d2 += Δ * Δ; end
        d_dist = sqrt(d2)
        w, wp = _phs_blend_weight_and_prime(d_dist, blend_a, blend_a3)
        w < eps(Tg) && continue

        offsets_nb, phys_offsets, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)

        if d_dist > eps(Tg)
            f, f_ξ = _phs_eval_coeffs_value_and_deriv1(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val{grad_ax}())
            f = Tv(f); f_ξ = Tv(f_ξ)
            g   = exp(f)
            dir = (Tg(query[grad_ax]) - nb_coords[grad_ax]) / d_dist
            sum_w  += w
            sum_wg += w * g
            sum_N1 += wp * dir * g + w * g * f_ξ
            sum_W1 += wp * dir
        else
            f = Tv(_phs_eval_coeffs_value(coeff_nb, phys_offsets, query, nb_coords, Val{K}()))
            g = exp(f)
            sum_w  += w
            sum_wg += w * g
        end
    end

    sum_w < eps(Tg) && return zero(Tv), zero(Tv)
    G   = sum_wg / sum_w
    G_ξ = Tv((sum_N1 - G * sum_W1) / sum_w)
    return G, G_ξ
end

"""
    _phs_eval_blended_G_with_hess(itp, query, ax1, ax2) -> (G, G_ax1, G_ax2, G_ax1ax2)

Evaluate G and its requested first- and second-order blend derivatives in a
single pass over the blend neighbourhood.  For the diagonal case (ax1 == ax2),
G_ax2 == G_ax1.  Replaces four separate `_phs_eval_blended_G` calls that
`_phs_eval_with_transform` previously made for Hessian queries.
"""
@inline function _phs_eval_blended_G_with_hess(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        query::NTuple{N, <:Real},
        ax1::Int,
        ax2::Int,
    ) where {Tg, Tv, N, K}
    return _phs_eval_blended_G_with_hess(itp, query, Val(ax1), Val(ax2))
end

@with_pool pool function _phs_eval_blended_G_with_hess(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        query::NTuple{N, <:Real},
        ::Val{ax1},
        ::Val{ax2},
    ) where {Tg, Tv, N, K, ax1, ax2}
    blend_a   = itp.blend_a
    blend_a3  = itp.blend_a3
    base_idx0 = _phs_find_base_node(itp, query)
    r_idx     = itp.blend_r_idx
    grid_sizes = ntuple(d -> length(itp.grids[d]), N)

    M         = size(itp.phi_inv, 1)
    rhs_buf   = acquire!(pool, Tg, M)
    coeff_buf = acquire!(pool, Tg, M)

    lo_idx = ntuple(d -> max(1, base_idx0[d] - r_idx[d]), N)
    hi_idx = ntuple(d -> min(grid_sizes[d], base_idx0[d] + r_idx[d]), N)
    ranges = ntuple(d -> lo_idx[d]:hi_idx[d], N)

    is_diag  = (ax1 == ax2)

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

        offsets_nb, phys_offsets, coeff_nb, hs_nb = _phs_solve_stencil!(itp, nb_idx, rhs_buf, coeff_buf)

        if d_dist > eps(Tg)
            inv_d_dist = one(Tg) / d_dist
            wp_inv = wp * inv_d_dist
            dx1  = Tg(query[ax1]) - nb_coords[ax1]
            da1  = dx1 * inv_d_dist
            wxi1 = wp_inv * dx1
            if is_diag
                f, f_d1, f_d2 = _phs_eval_coeffs_value_and_deriv1_and_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val{ax1}(), Val{ax1}())
                f = Tv(f); f_d1 = Tv(f_d1); f_d2 = Tv(f_d2)
                g     = exp(f)
                dg1   = g * f_d1
                d2g   = g * (f_d2 + f_d1 * f_d1)
                da1_sqr = da1 * da1
                wxixi = da1_sqr * (wpp - wp_inv) + wp_inv
                sum_w  += w; sum_wg += w * g
                sum_N1 += wxi1 * g + w * dg1
                sum_W1 += wxi1
                sum_N2 += wxixi * g + 2 * wxi1 * dg1 + w * d2g
                sum_W2 += wxixi
            else
                dx2   = Tg(query[ax2]) - nb_coords[ax2]
                da2   = dx2 * inv_d_dist
                wxi2  = wp_inv * dx2
                f, f_d1, f_d1b, f_d2 = _phs_eval_coeffs_value_and_two_deriv1_and_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val{ax1}(), Val{ax2}())
                f = Tv(f); f_d1 = Tv(f_d1); f_d1b = Tv(f_d1b); f_d2 = Tv(f_d2)
                g     = exp(f)
                dg1   = g * f_d1
                dg2   = g * f_d1b
                d2g   = g * (f_d2 + f_d1 * f_d1b)
                da1_da2 = da1 * da2
                wxixi = da1_da2 * (wpp - wp_inv)
                sum_w  += w; sum_wg += w * g
                sum_N1  += wxi1 * g + w * dg1
                sum_W1  += wxi1
                sum_N1b += wxi2 * g + w * dg2
                sum_W1b += wxi2
                sum_N2  += wxixi * g + wxi1 * dg2 + wxi2 * dg1 + w * d2g
                sum_W2  += wxixi
            end
        else
            # d≈0: weight prime/dprime ≈ 0; only stencil contributions survive.
            f    = Tv(_phs_eval_coeffs_value(coeff_nb, phys_offsets, query, nb_coords, Val{K}()))
            g    = exp(f)
            f_d1 = Tv(_phs_eval_coeffs_deriv1(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val{ax1}()))
            if is_diag
                f_d2   = Tv(_phs_eval_coeffs_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val{ax1}(), Val{ax1}()))
                d2g    = g * (f_d2 + f_d1 * f_d1)
                sum_w  += w; sum_wg += w * g
                sum_N1 += w * g * f_d1
                sum_N2 += w * d2g
            else
                f_d1b    = Tv(_phs_eval_coeffs_deriv1(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val{ax2}()))
                f_d2     = Tv(_phs_eval_coeffs_deriv2(coeff_nb, phys_offsets, query, nb_coords, Val{K}(), Val{ax1}(), Val{ax2}()))
                d2g      = g * (f_d2 + f_d1 * f_d1b)
                sum_w  += w; sum_wg += w * g
                sum_N1  += w * g * f_d1
                sum_N1b += w * g * f_d1b
                sum_N2  += w * d2g
            end
        end
    end

    sum_w < eps(Tg) && return zero(Tv), zero(Tv), zero(Tv), zero(Tv)
    W     = sum_w
    G     = sum_wg / W
    G_ax1 = Tv((sum_N1 - G * sum_W1) / W)
    if is_diag
        G_ax2 = G_ax1
        d2G   = sum_N2 / W - 2 * sum_N1 * sum_W1 / W^2 - G * sum_W2 / W + 2 * G * (sum_W1 / W)^2
    else
        G_ax2 = Tv((sum_N1b - G * sum_W1b) / W)
        d2G   = sum_N2 / W - (sum_N1 * sum_W1b + sum_N1b * sum_W1) / W^2 -
                G * sum_W2 / W + 2 * G * sum_W1 * sum_W1b / W^2
    end
    return G, G_ax1, G_ax2, Tv(d2G)
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
"""
@generated function _phs_eval_ref(ref, query, ops)
    if hasmethod(ref, Tuple{query, ops})
        return quote
            @inline
            ref(query, ops)
        end
    else
        return quote
            @inline
            ref(query; deriv = ops)
        end
    end
end

@inline function _phs_eval_ref_deriv1(ref, query, ax::Int, ::Val{N}, ::Type{Tg}) where {N, Tg}
    return _phs_eval_ref_deriv1(ref, query, Val(ax), Val(N), Tg)
end

@inline function _phs_eval_ref_deriv1(ref, query, ::Val{ax}, ::Val{N}, ::Type{Tg}) where {N, Tg, ax}
    ops = ntuple(d -> d == ax ? DerivOp{1}() : EvalValue(), Val(N))
    return Tg(_phs_eval_ref(ref, query, ops))
end

function _phs_eval_with_transform(
        itp::PHSInterpolantND{Tg, Tv, N, K},
        query::NTuple{N, <:Real},
        ops::O,
    ) where {Tg, Tv, N, K, O <: Tuple{Vararg{AbstractEvalOp, N}}}
    total_deriv = sum(deriv_order(ops[d]) for d in 1:N)
    ref = itp.transform.reference

    if total_deriv == 0
        ops_val = ntuple(_ -> EvalValue(), Val(N))
        G    = Tg(_phs_eval_blended_G(itp, query, ops_val))
        rho0 = Tg(ref(query))
        return Tv(rho0 * G)
    end

    if total_deriv == 1
        ax_val = _phs_get_deriv1_axis_val(O)
        # Single fused pass: compute G and G_ξ together
        G, G_ξ   = _phs_eval_blended_G_with_grad(itp, query, ax_val)
        rho0     = Tg(ref(query))
        rho0_ξ   = _phs_eval_ref_deriv1(ref, query, ax_val, Val(N), Tg)
        return Tv(rho0_ξ * G + rho0 * G_ξ)
    end

    if total_deriv == 2
        ax1_val, ax2_val = _phs_get_deriv2_axes_val(O)

        # Single fused pass: compute G, G_ξ, G_ζ (= G_ξ for diagonal), G_ξζ together
        G, G_ξ, G_ζ, G_ξζ = _phs_eval_blended_G_with_hess(itp, query, ax1_val, ax2_val)

        rho0    = Tg(ref(query))
        rho0_ξ  = _phs_eval_ref_deriv1(ref, query, ax1_val, Val(N), Tg)
        rho0_ξζ = Tg(_phs_eval_ref(ref, query, ops))

        if ax1_val === ax2_val
            # Diagonal case: ρ̃_ξξ = ρ₀_ξξ·G + 2·ρ₀_ξ·G_ξ + ρ₀·G_ξξ
            # (G_ζ == G_ξ and rho0_ζ == rho0_ξ — no extra blend call needed)
            return Tv(rho0_ξζ * G + 2 * rho0_ξ * G_ξ + rho0 * G_ξζ)
        else
            rho0_ζ = _phs_eval_ref_deriv1(ref, query, ax2_val, Val(N), Tg)
            # Leibniz rule: ρ̃_ξζ = ρ₀_ξζ·G + ρ₀_ξ·G_ζ + ρ₀_ζ·G_ξ + ρ₀·G_ξζ
            return Tv(rho0_ξζ * G + rho0_ξ * G_ζ + rho0_ζ * G_ξ + rho0 * G_ξζ)
        end
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
        ops::O,
    ) where {N, O <: Tuple{Vararg{AbstractEvalOp, N}}}
    return _phs_eval_dispatch(itp, itp.transform, query, ops)
end
