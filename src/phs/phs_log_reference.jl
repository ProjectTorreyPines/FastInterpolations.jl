# ========================================
# PHS log reference — ρ₀ for the log transform
# ========================================
#
# `log_reference` is `nothing` (off), a nonzero constant (used as given), or a callable
# ρ₀(q). Everything about ρ₀ lives here: the internal types, how evaluation reads its
# value and derivatives, and how the constructor resolves and validates it. The
# transform evaluation itself (ρ̃ = ρ₀·G with Leibniz derivatives) is in phs_eval.jl.

# ---- Types ----

# Log-transform state. `reference` is the constant ρ₀ as given (any nonzero Real) or a
# callable ρ₀(q) (e.g. an interpolant).
struct _PHSLogTransform{R}
    reference::R
end

# ---- Reference ρ₀ access: constant or callable ----

@inline _phs_ref_value(c::Real, q) = c
@inline _phs_ref_value(r, q) = r(q)

# A constant has zero derivatives; an order-0 `ops` tuple still asks for the value.
# A callable answers derivatives through the `deriv` keyword, as interpolants do.
@inline _phs_ref_deriv(c::Real, q, ops) = sum(deriv_order, ops) == 0 ? c : zero(c)
@inline _phs_ref_deriv(r, q, ops) = r(q; deriv = ops)

@inline function _phs_eval_ref_deriv1(ref, query, ax::Int, ::Val{N}, ::Type{Tg}) where {N, Tg}
    return _phs_eval_ref_deriv1(ref, query, Val(ax), Val(N), Tg)
end

@inline function _phs_eval_ref_deriv1(ref, query, ::Val{ax}, ::Val{N}, ::Type{Tg}) where {N, Tg, ax}
    ops = ntuple(d -> d == ax ? DerivOp{1}() : EvalValue(), Val(N))
    return Tg(_phs_ref_deriv(ref, query, ops))
end

# ---- Resolution and validation (constructor): `log_reference` → (transform, stored data) ----

# Every method annotates `grids` identically, so the first argument alone decides
# dispatch (a typed `grids` on only the callable method would be ambiguous).
_resolve_phs_log_reference(
    ::Nothing, grids::NTuple{N, AbstractVector}, data, ::Type{Tv}, ::Type{Tg}
) where {N, Tv, Tg} = (nothing, Array{Tv}(data))

_resolve_phs_log_reference(
    ::Bool, grids::NTuple{N, AbstractVector}, data, ::Type{Tv}, ::Type{Tg}
) where {N, Tv, Tg} = throw(
    ArgumentError("log_reference must be a nonzero number or a callable ρ₀(q), got a Bool; use `nothing` to disable the log transform")
)

function _resolve_phs_log_reference(
        c::Real, grids::NTuple{N, AbstractVector}, data, ::Type{Tv}, ::Type{Tg}
    ) where {N, Tv, Tg}
    c_g = Tg(c)   # evaluation uses ρ₀ at grid precision
    (iszero(c_g) || !isfinite(c_g)) && throw(
        ArgumentError("log_reference must be nonzero and finite at grid precision $Tg, got $c")
    )
    return _PHSLogTransform(c), _phs_log_data(data, c, Tv, Tg)
end

_resolve_phs_log_reference(
    ::AbstractArray, grids::NTuple{N, AbstractVector}, data, ::Type{Tv}, ::Type{Tg}
) where {N, Tv, Tg} = throw(
    ArgumentError(
        "log_reference cannot be an array: ρ₀ is also needed between grid nodes, and a " *
            "reference sampled on the data grid adds no information. Pass a nonzero constant " *
            "or a callable ρ₀(q), e.g. an interpolant built from finer data"
    )
)

function _resolve_phs_log_reference(
        ref,
        grids::NTuple{N, AbstractVector},
        data,
        ::Type{Tv},
        ::Type{Tg},
    ) where {N, Tv, Tg}
    hasmethod(ref, Tuple{NTuple{N, Tg}}) || throw(
        ArgumentError("log_reference must be a nonzero number or a callable ρ₀(q) on an NTuple{$N} point, got $(typeof(ref))")
    )
    rho0 = [_phs_ref_value(ref, ntuple(d -> grids[d][I[d]], Val(N))) for I in CartesianIndices(size(data))]
    return _PHSLogTransform(ref), _phs_log_data(data, rho0, Tv, Tg)
end

# Stored log data `log(ρ/ρ₀)`, checked at the precision evaluation uses: each ρ₀ must be
# a real that stays nonzero and finite as `Tg`, and every ratio positive and finite.
function _phs_log_data(data, rho0, ::Type{Tv}, ::Type{Tg}) where {Tv, Tg}
    Tv <: Real || throw(ArgumentError("log_reference needs real-valued data, got eltype $Tv"))
    out = Array{Tv}(undef, size(data))
    for I in CartesianIndices(data)
        v = rho0 isa AbstractArray ? rho0[I] : rho0
        v isa Real || _throw_phs_log_ref(Tuple(I), v, "is not a real number")
        v_g = Tg(v)
        (iszero(v_g) || !isfinite(v_g)) &&
            _throw_phs_log_ref(Tuple(I), v, "is zero or not finite at grid precision $Tg")
        r = data[I] / v_g
        (r > zero(r) && isfinite(r)) || _throw_phs_log_ratio(Tuple(I), data[I], v, r)
        out[I] = log(r)
    end
    return out
end

@noinline _throw_phs_log_ref(idx, v, why) = throw(
    ArgumentError("log_reference: ρ₀ at grid node $idx $why (got $v)")
)

@noinline _throw_phs_log_ratio(idx, ρ, ρ0, r) = throw(
    ArgumentError("log_reference needs data/ρ₀ > 0 and finite at every grid node; at node $idx data = $ρ, ρ₀ = $ρ0 (ratio $r)")
)
