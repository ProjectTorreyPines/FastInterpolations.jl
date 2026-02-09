# Cubic spline integration kernels (private)
#
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Spline: S(u) = a·u³ + b·u² + c·u + d
#   a = (zR - zL) / (6h)
#   b = zL / 2
#   c = (yR - yL)/h - h·(2zL + zR)/6
#   d = yL

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
@inline function _cubic_integral_kernel(
    ::_EvalIntegralPartial,
    zL::Tv, zR::Tv, yL::Tv, yR::Tv,
    h::Tg, u0::Td, u1::Td
) where {Tv, Tg<:AbstractFloat, Td<:Real}
    a = (zR - zL) / (6h)
    b = zL / 2
    c = (yR - yL) / h - h * (2zL + zR) / 6
    d = yL
    return (a / 4) * (u1^4 - u0^4) +
           (b / 3) * (u1^3 - u0^3) +
           (c / 2) * (u1^2 - u0^2) +
            d      * (u1 - u0)
end

# --- Full-cell integral: ∫_0^h S(u) du = h/2·(yL+yR) - h³/24·(zL+zR) ---
@inline function _cubic_integral_kernel(
    ::_EvalIntegralCell,
    zL::Tv, zR::Tv, yL::Tv, yR::Tv, h::Tg
) where {Tv, Tg<:AbstractFloat}
    return h / 2 * (yL + yR) - h^3 / 24 * (zL + zR)
end

# ═══════════════════════════════════════════════════════════════
# Linear integration kernels
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Piecewise linear: S(u) = yL + (yR - yL)/h · u
# ═══════════════════════════════════════════════════════════════

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
@inline function _linear_integral_kernel(
    ::_EvalIntegralPartial,
    yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td
) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return yL * (u1 - u0) + (yR - yL) / (2h) * (u1^2 - u0^2)
end

# --- Full-cell integral: ∫_0^h S(u) du = h/2·(yL + yR) ---
@inline function _linear_integral_kernel(
    ::_EvalIntegralCell,
    yL::Tv, yR::Tv, h::Tg
) where {Tv, Tg<:AbstractFloat}
    return h * (yL + yR) / 2
end

# ═══════════════════════════════════════════════════════════════
# Quadratic integration kernels
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Piecewise quadratic: S(u) = a·u² + d·u + y₀
# (a, d are pre-computed coefficients stored in QuadraticInterpolant)
# ═══════════════════════════════════════════════════════════════

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
@inline function _quadratic_integral_kernel(
    ::_EvalIntegralPartial,
    a::Tv, d::Tv, y0::Tv, u0::Td, u1::Td
) where {Tv, Td<:Real}
    return (a / 3) * (u1^3 - u0^3) + (d / 2) * (u1^2 - u0^2) + y0 * (u1 - u0)
end

# --- Full-cell integral: ∫_0^h S(u) du = a/3·h³ + d/2·h² + y₀·h ---
@inline function _quadratic_integral_kernel(
    ::_EvalIntegralCell,
    a::Tv, d::Tv, y0::Tv, h::Tg
) where {Tv, Tg<:AbstractFloat}
    return (a / 3) * h^3 + (d / 2) * h^2 + y0 * h
end

# ═══════════════════════════════════════════════════════════════
# Constant integration kernels
# Piecewise constant: value depends on side mode (:left, :right, :nearest)
# ═══════════════════════════════════════════════════════════════

# --- :left side — always use left value yL ---
@inline function _constant_integral_kernel(
    ::_EvalIntegralPartial,
    yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::Val{:left}
) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return yL * (u1 - u0)
end

# --- :right side — always use right value yR ---
@inline function _constant_integral_kernel(
    ::_EvalIntegralPartial,
    yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::Val{:right}
) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return yR * (u1 - u0)
end

# --- :nearest side — split at midpoint h/2 ---
@inline function _constant_integral_kernel(
    ::_EvalIntegralPartial,
    yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::Val{:nearest}
) where {Tv, Tg<:AbstractFloat, Td<:Real}
    mid = h / 2
    if u1 <= mid
        return yL * (u1 - u0)
    elseif u0 >= mid
        return yR * (u1 - u0)
    else
        return yL * (mid - u0) + yR * (u1 - mid)
    end
end

# ═══════════════════════════════════════════════════════════════
# ND Hermite integration kernels
#
# Antiderivatives of the four cubic Hermite basis functions on [0,1]:
#   H₀₀(t) = 2t³ - 3t² + 1     →  ∫H₀₀ dt = t⁴/2 - t³ + t
#   H₁₀(t) = t³ - 2t² + t      →  ∫H₁₀ dt = t⁴/4 - 2t³/3 + t²/2
#   H₀₁(t) = -2t³ + 3t²        →  ∫H₀₁ dt = -t⁴/2 + t³
#   H₁₁(t) = t³ - t²           →  ∫H₁₁ dt = t⁴/4 - t³/3
#
# Used by the @generated ND tensor-product cell integral kernel.
# ═══════════════════════════════════════════════════════════════

@inline _IH00(t) = t - t^3 + t^4 / 2
@inline _IH10(t) = t^2 / 2 - 2t^3 / 3 + t^4 / 4
@inline _IH01(t) = t^3 - t^4 / 2
@inline _IH11(t) = -t^3 / 3 + t^4 / 4

# --- 1D Hermite integral over [u0, u1] in local coordinates ---
# Computes ∫_{u0}^{u1} P(x) dx where P is the cubic Hermite polynomial:
#   P(t) = fL·H₀₀(t) + fR·H₀₁(t) + h·(dfL·H₁₀(t) + dfR·H₁₁(t))
# with t = u/h, dx = h dt
@inline function _hermite_integral_kernel_1d(
    fL, fR, dfL, dfR,
    h::Tg, inv_h::Tg,
    u0::Real, u1::Real
) where {Tg<:AbstractFloat}
    t0 = u0 * inv_h
    t1 = u1 * inv_h
    dH00 = _IH00(t1) - _IH00(t0)
    dH10 = _IH10(t1) - _IH10(t0)
    dH01 = _IH01(t1) - _IH01(t0)
    dH11 = _IH11(t1) - _IH11(t0)
    return h * (fL * dH00 + fR * dH01 + h * (dfL * dH10 + dfR * dH11))
end

# ═══════════════════════════════════════════════════════════════
# ND Linear integration kernel
#
# Multilinear basis weights integrated over [u0, u1]:
#   w₀(u0,u1,h) = ∫(1 - u/h) du = (u1-u0) - (u1²-u0²)/(2h)
#   w₁(u0,u1,h) = ∫(u/h) du     = (u1²-u0²)/(2h)
# ═══════════════════════════════════════════════════════════════

@inline _w0_int(u0, u1, h) = (u1 - u0) - (u1^2 - u0^2) / (2h)
@inline _w1_int(u0, u1, h) = (u1^2 - u0^2) / (2h)

@inline @generated function _integrate_linear_nd_cell(
    data::Array{Tv, N},
    idx::NTuple{N, Int},
    hs::NTuple{N},
    ulo::NTuple{N},
    uhi::NTuple{N}
) where {Tv, N}
    nc = 1 << N
    terms = Expr[]
    for c in 0:(nc - 1)
        bits = ntuple(d -> (c >> (d - 1)) & 1, N)
        idx_parts = [:(idx[$d] + $(bits[d])) for d in 1:N]
        ws = [bits[d] == 0 ?
              :(_w0_int(ulo[$d], uhi[$d], hs[$d])) :
              :(_w1_int(ulo[$d], uhi[$d], hs[$d])) for d in 1:N]
        w = foldl((a, b) -> :($a * $b), ws)
        push!(terms, :(@inbounds data[$(idx_parts...)] * $w))
    end
    sum_expr = foldl((a, b) -> :($a + $b), terms)
    quote
        Base.@_inline_meta
        @inbounds $sum_expr
    end
end

# ═══════════════════════════════════════════════════════════════
# ND Quadratic integration kernel
#
# 1D quadratic: S(u) = a·u² + dfL·u + fL
#   where a = ((fR-fL)/h - dfL)/h = (s - dfL)·inv_h
# Integral: ∫_{u0}^{u1} S(u) du = a/3·(u1³-u0³) + dfL/2·(u1²-u0²) + fL·(u1-u0)
# ═══════════════════════════════════════════════════════════════

@inline function _quadratic_integral_kernel_nd(fL, fR, dfL, h, inv_h, u0, u1)
    s = (fR - fL) * inv_h
    a = (s - dfL) * inv_h
    return (a / 3) * (u1^3 - u0^3) + (dfL / 2) * (u1^2 - u0^2) + fL * (u1 - u0)
end

# @generated tensor-product cell integral for quadratic ND
# Mirrors _eval_nd_quad_cell but replaces point-eval with integral kernel.
# Reads 3 values per dimension (fL, fR, dfL — no dfR).
@inline @generated function _integrate_nd_quad_cell(
    partials::Array{Tv, NP1},
    indices::NTuple{N, Int},
    hs::NTuple{N, Tg},
    inv_hs::NTuple{N, Tg},
    ulos::NTuple{N, <:Real},
    uhis::NTuple{N, <:Real}
) where {Tv, Tg, N, NP1}
    NP1 == N + 1 || error("NP1 must equal N+1")

    stmts = Expr[]

    for (prefix, source) in [("idx_", :indices), ("h_", :hs), ("inv_h_", :inv_hs),
                              ("ulo_", :ulos), ("uhi_", :uhis)]
        syms = ntuple(d -> Symbol(prefix, d), N)
        lhs = Expr(:tuple, syms...)
        push!(stmts, :($lhs = $source))
    end

    for stage in 1:N
        num_corners = 1 << (N - stage)
        num_derivs = 1 << (N - stage)

        for corner in 0:(num_corners - 1)
            for deriv in 0:(num_derivs - 1)
                out_var = _varname(stage, corner, deriv)

                if stage == 1
                    function make_partial_access_qi(c_dim1::Int, d_dim1::Int)
                        corner_full = c_dim1 | (corner << 1)
                        deriv_full = d_dim1 | (deriv << 1)
                        p_idx = _partial_index(deriv_full)
                        offsets = _corner_offset_expr(corner_full, N)
                        idx_exprs = [:($(Symbol("idx_", d)) + $(offsets[d])) for d in 1:N]
                        return :(partials[$p_idx, $(idx_exprs...)])
                    end

                    fL = make_partial_access_qi(0, 0)
                    fR = make_partial_access_qi(1, 0)
                    dfL = make_partial_access_qi(0, 1)
                else
                    prev_stage = stage - 1
                    fL = _varname(prev_stage, 0 | (corner << 1), 0 | (deriv << 1))
                    fR = _varname(prev_stage, 1 | (corner << 1), 0 | (deriv << 1))
                    dfL = _varname(prev_stage, 0 | (corner << 1), 1 | (deriv << 1))
                end

                h = Symbol("h_", stage)
                inv_h = Symbol("inv_h_", stage)
                ulo = Symbol("ulo_", stage)
                uhi = Symbol("uhi_", stage)

                kernel_call = :(_quadratic_integral_kernel_nd($fL, $fR, $dfL, $h, $inv_h, $ulo, $uhi))
                push!(stmts, :($out_var = $kernel_call))
            end
        end
    end

    final_var = _varname(N, 0, 0)
    push!(stmts, :(return $final_var))

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
    end
end

# ═══════════════════════════════════════════════════════════════
# ND Constant integration kernel
#
# Per-axis weight functions for side-dependent integration:
#   :left   → all weight to left corner
#   :right  → all weight to right corner
#   :nearest → split at midpoint h/2
# ═══════════════════════════════════════════════════════════════

@inline _cw0(u0, u1, h, ::Val{:left}) = u1 - u0
@inline _cw1(u0, u1, h, ::Val{:left}) = zero(u1 - u0)
@inline _cw0(u0, u1, h, ::Val{:right}) = zero(u1 - u0)
@inline _cw1(u0, u1, h, ::Val{:right}) = u1 - u0
@inline _cw0(u0, u1, h, ::Val{:nearest}) = max(zero(u0), min(u1, h / 2) - u0)
@inline _cw1(u0, u1, h, ::Val{:nearest}) = max(zero(u0), u1 - max(u0, h / 2))

@inline @generated function _integrate_constant_nd_cell(
    data::Array{Tv, N},
    idx::NTuple{N, Int},
    hs::NTuple{N},
    ulo::NTuple{N},
    uhi::NTuple{N},
    sides::NTuple{N, SideVal}
) where {Tv, N}
    nc = 1 << N
    terms = Expr[]
    for c in 0:(nc - 1)
        bits = ntuple(d -> (c >> (d - 1)) & 1, N)
        idx_parts = [:(idx[$d] + $(bits[d])) for d in 1:N]
        ws = [bits[d] == 0 ?
              :(_cw0(ulo[$d], uhi[$d], hs[$d], sides[$d])) :
              :(_cw1(ulo[$d], uhi[$d], hs[$d], sides[$d])) for d in 1:N]
        w = foldl((a, b) -> :($a * $b), ws)
        push!(terms, :(@inbounds data[$(idx_parts...)] * $w))
    end
    sum_expr = foldl((a, b) -> :($a + $b), terms)
    quote
        Base.@_inline_meta
        @inbounds $sum_expr
    end
end
