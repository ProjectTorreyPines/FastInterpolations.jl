# Cubic spline integration kernels (private)
#
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Spline: S(u) = a·u³ + b·u² + c·u + d
#   a = (zR - zL) / (6h)
#   b = zL / 2
#   c = (yR - yL)/h - h·(2zL + zR)/6
#   d = yL

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
# Horner form of antiderivative: F(u) = u·@evalpoly(u, d, c2, b3, a4)
# with pre-absorbed coefficients a4=a/4, b3=b/3, c2=c/2.
@inline function _cubic_integral_kernel(
        ::_EvalIntegralPartial,
        zL::Tz, zR::Tz, yL::Ty, yR::Ty,
        h::Tg, u0::Td, u1::Td
    ) where {Tz, Ty, Tg <: Real, Td <: Real}
    inv_h = inv(h)
    a4 = (zR - zL) * (inv_h * inv(Tg(24)))   # a/4 = (zR-zL)/(24h)
    b3 = inv(Tg(6)) * zL                     # b/3 = zL/6
    c2 = _fielddiff(Tz, yR, yL) * (inv_h / 2) - (h * inv(Tg(12))) * (2zL + zR)  # c/2
    d = yL
    return u1 * @evalpoly(u1, d, c2, b3, a4) -
        u0 * @evalpoly(u0, d, c2, b3, a4)
end

# --- Full-cell integral: ∫_0^h S(u) du = h/2·(yL+yR) - h³/24·(zL+zR) ---
@inline function _cubic_integral_kernel(
        ::_EvalIntegralCell,
        zL::Tz, zR::Tz, yL::Ty, yR::Ty, h::Tg
    ) where {Tz, Ty, Tg <: Real}
    h2 = h * h
    return (h / 2) * muladd(-(h2 * inv(Tg(12))), zL + zR, _fieldsum(Tz, yL, yR))
end

# ═══════════════════════════════════════════════════════════════
# Linear integration kernels
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Piecewise linear: S(u) = yL + (yR - yL)/h · u
# ═══════════════════════════════════════════════════════════════

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
# Uses u1²-u0² = (u1-u0)(u1+u0) factorization + muladd.
@inline function _linear_integral_kernel(
        ::_EvalIntegralPartial,
        yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td
    ) where {Tv, Tg <: Real, Td <: Real}
    du = u1 - u0
    Tc = _promote_eltype(_coeff_op, Tg, Tv)
    half_slope = _fielddiff(Tc, yR, yL) * inv(2h)
    return du * muladd(half_slope, u1 + u0, yL)
end

# --- Full-cell integral: ∫_0^h S(u) du = h/2·(yL + yR) ---
@inline function _linear_integral_kernel(
        ::_EvalIntegralCell,
        yL::Tv, yR::Tv, h::Tg
    ) where {Tv, Tg <: Real}
    Tc = _promote_eltype(_coeff_op, Tg, Tv)
    return (h / 2) * _fieldsum(Tc, yL, yR)
end

# ═══════════════════════════════════════════════════════════════
# Quadratic integration kernels
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Piecewise quadratic: S(u) = a·u² + d·u + y₀
# (a, d are pre-computed coefficients stored in QuadraticInterpolant)
# ═══════════════════════════════════════════════════════════════

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
# Horner form: F(u) = u·@evalpoly(u, y0, d/2, a/3)
@inline function _quadratic_integral_kernel(
        ::_EvalIntegralPartial,
        a::Ta, d::Td2, y0::Ty, u0::Td, u1::Td
    ) where {Ta, Td2, Ty, Td <: Real}
    a_3 = inv(Td(3)) * a
    d_2 = inv(Td(2)) * d
    return u1 * @evalpoly(u1, y0, d_2, a_3) -
        u0 * @evalpoly(u0, y0, d_2, a_3)
end

# --- Full-cell integral: ∫_0^h S(u) du = a/3·h³ + d/2·h² + y₀·h ---
@inline function _quadratic_integral_kernel(
        ::_EvalIntegralCell,
        a::Ta, d::Td2, y0::Ty, h::Tg
    ) where {Ta, Td2, Ty, Tg <: Real}
    return h * @evalpoly(h, y0, inv(Tg(2)) * d, inv(Tg(3)) * a)
end

# ═══════════════════════════════════════════════════════════════
# Constant integration kernels
# Piecewise constant: value depends on side mode (LeftSide, RightSide, NearestSide)
# ═══════════════════════════════════════════════════════════════

# --- LeftSide — always use left value yL ---
@inline function _constant_integral_kernel(
        ::_EvalIntegralPartial,
        yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::LeftSide
    ) where {Tv, Tg <: Real, Td <: Real}
    return yL * (u1 - u0)
end

# --- RightSide — always use right value yR ---
@inline function _constant_integral_kernel(
        ::_EvalIntegralPartial,
        yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::RightSide
    ) where {Tv, Tg <: Real, Td <: Real}
    return yR * (u1 - u0)
end

# --- NearestSide — split at midpoint h/2 ---
@inline function _constant_integral_kernel(
        ::_EvalIntegralPartial,
        yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::NearestSide
    ) where {Tv, Tg <: Real, Td <: Real}
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

# Horner form: F(t) = t · @evalpoly(t, c₁, c₂, c₃) — eliminates explicit t^3, t^4.
@inline _IH00(t) = t * @evalpoly(t, 1, 0, -1, 1 / 2)
@inline _IH10(t) = t * @evalpoly(t, 0, 1 / 2, -2 / 3, 1 / 4)
@inline _IH01(t) = t * @evalpoly(t, 0, 0, 1, -1 / 2)
@inline _IH11(t) = t * @evalpoly(t, 0, 0, -1 / 3, 1 / 4)

# --- 1D Hermite integral over [u0, u1] in local coordinates ---
# Computes ∫_{u0}^{u1} P(x) dx where P is the cubic Hermite polynomial:
#   P(t) = fL·H₀₀(t) + fR·H₀₁(t) + h·(dfL·H₁₀(t) + dfR·H₁₁(t))
# with t = u/h, dx = h dt
@inline function _hermite_integral_kernel_1d(
        fL, fR, dfL, dfR,
        h::Tg, inv_h::Tg,
        u0::Real, u1::Real
    ) where {Tg}
    t0 = u0 * inv_h
    t1 = u1 * inv_h
    dH00 = _IH00(t1) - _IH00(t0)
    dH10 = _IH10(t1) - _IH10(t0)
    dH01 = _IH01(t1) - _IH01(t0)
    dH11 = _IH11(t1) - _IH11(t0)
    inner = muladd(dfR, dH11, dfL * dH10)      # dfL·ΔH₁₀ + dfR·ΔH₁₁
    outer = muladd(fR, dH01, fL * dH00)        # fL·ΔH₀₀ + fR·ΔH₀₁
    return h * muladd(h, inner, outer)
end

# @generated tensor-product cell integral kernel for ND cubic.
# Mirrors _eval_nd_cell but replaces _hermite_kernel_1d with
# _hermite_integral_kernel_1d for each dimension collapse stage.
@inline @generated function _integrate_nd_cubic_cell(
        partials::Array{Tv, NP1},
        indices::NTuple{N, Int},
        hs::NTuple{N, Tg},
        inv_hs::NTuple{N, Tg},
        ulos::NTuple{N, <:Real},
        uhis::NTuple{N, <:Real}
    ) where {Tv, Tg, N, NP1}
    NP1 == N + 1 || error("NP1 must equal N+1")

    stmts = Expr[]

    # Unpack tuples
    for (prefix, source) in [
            ("idx_", :indices), ("h_", :hs), ("inv_h_", :inv_hs),
            ("ulo_", :ulos), ("uhi_", :uhis),
        ]
        syms = ntuple(d -> Symbol(prefix, d), N)
        lhs = Expr(:tuple, syms...)
        push!(stmts, :($lhs = $source))
    end

    # Collapse each dimension via integral kernel
    for stage in 1:N
        num_corners = 1 << (N - stage)
        num_derivs = 1 << (N - stage)

        for corner in 0:(num_corners - 1)
            for deriv in 0:(num_derivs - 1)
                out_var = _varname(stage, corner, deriv)

                if stage == 1
                    function make_partial_access(c_dim1::Int, d_dim1::Int)
                        corner_full = c_dim1 | (corner << 1)
                        deriv_full = d_dim1 | (deriv << 1)
                        p_idx = _partial_index(deriv_full)
                        offsets = _corner_offset_expr(corner_full, N)
                        idx_exprs = [:($(Symbol("idx_", d)) + $(offsets[d])) for d in 1:N]
                        return :(partials[$p_idx, $(idx_exprs...)])
                    end

                    fL = make_partial_access(0, 0)
                    fR = make_partial_access(1, 0)
                    dfL = make_partial_access(0, 1)
                    dfR = make_partial_access(1, 1)
                else
                    prev_stage = stage - 1
                    fL = _varname(prev_stage, 0 | (corner << 1), 0 | (deriv << 1))
                    fR = _varname(prev_stage, 1 | (corner << 1), 0 | (deriv << 1))
                    dfL = _varname(prev_stage, 0 | (corner << 1), 1 | (deriv << 1))
                    dfR = _varname(prev_stage, 1 | (corner << 1), 1 | (deriv << 1))
                end

                h = Symbol("h_", stage)
                inv_h = Symbol("inv_h_", stage)
                ulo = Symbol("ulo_", stage)
                uhi = Symbol("uhi_", stage)

                kernel_call = :(_hermite_integral_kernel_1d($fL, $fR, $dfL, $dfR, $h, $inv_h, $ulo, $uhi))
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
# ND Linear integration kernel
#
# Multilinear basis weights integrated over [u0, u1]:
#   w₀(u0,u1,h) = ∫(1 - u/h) du = (u1-u0) - (u1²-u0²)/(2h)
#   w₁(u0,u1,h) = ∫(u/h) du     = (u1²-u0²)/(2h)
# ═══════════════════════════════════════════════════════════════

# Factored form: u1²-u0² = (u1-u0)(u1+u0), then muladd to avoid explicit squaring.
@inline function _w0_int(u0, u1, h)
    du = u1 - u0
    su = u1 + u0
    return du * muladd(-su, inv(2h), one(su))   # du·(1 - (u1+u0)/(2h))
end
@inline function _w1_int(u0, u1, h)
    du = u1 - u0
    return du * (u1 + u0) / (2h)
end

@inline @generated function _integrate_linear_nd_cell(
        data::AbstractArray{Tv, N},
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
        ws = [
            bits[d] == 0 ?
                :(_w0_int(ulo[$d], uhi[$d], hs[$d])) :
                :(_w1_int(ulo[$d], uhi[$d], hs[$d])) for d in 1:N
        ]
        w = foldl((a, b) -> :($a * $b), ws)
        push!(terms, :(@inbounds data[$(idx_parts...)] * $w))
    end
    sum_expr = foldl((a, b) -> :($a + $b), terms)
    return quote
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

# Horner form: F(u) = u · @evalpoly(u, fL, dfL/2, a/3)
@inline function _quadratic_integral_kernel_nd(fL::Tv, fR, dfL, h::Tg, inv_h::Tinv, u0, u1) where {Tg, Tinv, Tv}
    # `Tg` = grid type (from `h`); `inv_h::Tinv` is its reciprocal (`inv(Int)::Float`, so
    # `Tinv` ≠ `Tg` on Int grids). Witness the coeff field through the grid `Tg` (the
    # `_coeff_op` convention — `inv(h)` floats Int grids), matching `_hermite_kernel_1d`.
    s = _fielddiff(_promote_eltype(_coeff_op, Tg, Tv), fR, fL) * inv_h
    a_3 = (s - dfL) * (inv_h * inv(oftype(h, 3)))  # a/3
    d_2 = inv(oftype(h, 2)) * dfL   # Tg * Tv
    return u1 * @evalpoly(u1, fL, d_2, a_3) -
        u0 * @evalpoly(u0, fL, d_2, a_3)
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

    for (prefix, source) in [
            ("idx_", :indices), ("h_", :hs), ("inv_h_", :inv_hs),
            ("ulo_", :ulos), ("uhi_", :uhis),
        ]
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
#   LeftSide    → all weight to left corner
#   RightSide   → all weight to right corner
#   NearestSide → split at midpoint h/2
# ═══════════════════════════════════════════════════════════════

@inline _cw0(u0, u1, h, ::LeftSide) = u1 - u0
@inline _cw1(u0, u1, h, ::LeftSide) = zero(u1 - u0)
@inline _cw0(u0, u1, h, ::RightSide) = zero(u1 - u0)
@inline _cw1(u0, u1, h, ::RightSide) = u1 - u0
@inline _cw0(u0, u1, h, ::NearestSide) = max(zero(u0), min(u1, h / 2) - u0)
@inline _cw1(u0, u1, h, ::NearestSide) = max(zero(u0), u1 - max(u0, h / 2))

@inline @generated function _integrate_constant_nd_cell(
        data::AbstractArray{Tv, N},
        idx::NTuple{N, Int},
        hs::NTuple{N},
        ulo::NTuple{N},
        uhi::NTuple{N},
        sides::Tuple{Vararg{AbstractSide, N}}
    ) where {Tv, N}
    nc = 1 << N
    terms = Expr[]
    for c in 0:(nc - 1)
        bits = ntuple(d -> (c >> (d - 1)) & 1, N)
        idx_parts = [:(idx[$d] + $(bits[d])) for d in 1:N]
        ws = [
            bits[d] == 0 ?
                :(_cw0(ulo[$d], uhi[$d], hs[$d], sides[$d])) :
                :(_cw1(ulo[$d], uhi[$d], hs[$d], sides[$d])) for d in 1:N
        ]
        w = foldl((a, b) -> :($a * $b), ws)
        push!(terms, :(@inbounds data[$(idx_parts...)] * $w))
    end
    sum_expr = foldl((a, b) -> :($a + $b), terms)
    return quote
        Base.@_inline_meta
        @inbounds $sum_expr
    end
end
