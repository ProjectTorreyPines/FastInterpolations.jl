# ========================================
# Hermite 1D Oneshot API
# ========================================
# hermite_interp / hermite_interp! — user-supplied slopes go directly to kernel.
# No cache, no solve, no wrapper.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         TYPED CORE — all grid types                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar oneshot
# ========================================

"""
    hermite_interp(x, y, dy, xq; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch(), hint=nothing)

Cubic Hermite interpolation at a single query point using user-supplied slopes.

Returns interpolated value (or derivative, if `deriv` is set).
C\$^1\$ continuous — slopes are used directly, no global spline solve.
"""
@inline function hermite_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {Tg, Tv, Tq <: Real}
    x = _prepare_grid(x)
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(dy) == length(x) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
    @boundscheck length(x) >= 2 || throw(ArgumentError("Hermite interpolation requires at least 2 points, got $(length(x))"))

    searcher = _resolve_search(x, xq, search, hint)
    extrap = _materialize_extrap(x, extrap)
    return _hermite_eval_at_point(x, y, dy, xq, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — in-place
# ========================================

"""
    hermite_interp!(output, x, y, dy, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch(), hint=nothing)

In-place cubic Hermite interpolation using user-supplied slopes.
"""
function hermite_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        x_query::AbstractVector{Tq};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {Tg, Tv, Tq <: Real}
    x = _prepare_grid(x)
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(dy) == length(x) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    searcher = _resolve_search(x, x_query, search, hint)
    extrap = _materialize_extrap(x, extrap)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — allocating
# ========================================

"""
    hermite_interp(x, y, dy, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch(), hint=nothing)

Cubic Hermite interpolation at multiple query points using user-supplied slopes.
Returns `Vector` of interpolated values.
"""
function hermite_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        x_query::AbstractVector{Tq};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {Tg, Tv, Tq <: Real}
    Tr = _output_eltype(Tv, _promote_grid_float(Tg, Tv), Tq, eltype(dy))
    output = Vector{Tr}(undef, length(x_query))
    hermite_interp!(output, x, y, dy, x_query; extrap = extrap, deriv = deriv, search = search, hint = hint)
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  INPUT PROMOTION HELPER                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Joint promotion of (x, y, dy) — grid type Tg considers all three inputs,
# so e.g. x::Float32 + y::Float32 + dy::Float64 → Tg=Float64 (no precision loss).
@inline function _promote_hermite_inputs(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        dy::AbstractVector{TDY},
    ) where {TX, TY, TDY}
    Tg_y = _promote_grid_float(TX, TY)
    Tg = TDY <: _PromotableValue ? promote_type(Tg_y, float(_real_eltype(TDY))) : Tg_y
    x_p = _to_float(x, Tg)
    # Only promote values when Tg is a standard float type — duck-typed Tg (Dual etc.)
    # leaves y/dy as-is; Julia arithmetic promotion handles the rest in kernels.
    if TY <: _PromotableValue && Tg <: AbstractFloat
        y_p = _promote_value_type(y, Tg)[2]
    else
        y_p = y
    end
    if TDY <: _PromotableValue && Tg <: AbstractFloat
        dy_p = _promote_value_type(dy, Tg)[2]
    else
        dy_p = dy
    end
    return x_p, y_p, dy_p
end
