# ========================================
# Shared ND Adjoint Helpers
# ========================================
#
# Error helpers, output sizing, and periodic finalization shared by all
# AbstractAdjointND subtypes. Loaded early (core) so both 1D and ND
# adjoint code can reference these.

# ── Error helpers (@noinline cold path) ──

@noinline _throw_adjoint_dim_mismatch(label::String, got::Int, expected::Int) =
    throw(DimensionMismatch("$label length ($got) must match expected ($expected)"))

@noinline _throw_adjoint_size_mismatch(got::Tuple, expected::Tuple) =
    throw(DimensionMismatch("f_bar size $got must match output size $expected"))

# ── Output size (exclusive periodic axes shrink by 1) ──

@inline function _adjoint_output_size(adj::AbstractAdjointND{<:Any, N}) where {N}
    gs = _grid_size(adj)
    bcs = _adjoint_bcs(adj)
    return ntuple(Val(N)) do d
        bcs[d] isa PeriodicBC{:exclusive} ? gs[d] - 1 : gs[d]
    end
end

# ── Exclusive periodic detection ──

@inline _has_exclusive_periodic(bcs::Tuple) =
    any(bp -> bp isa PeriodicBC{:exclusive}, bcs)

# ── Periodic finalization (fold exclusive endpoint + truncate) ──

function _adjoint_nd_finalize(
        f_bar::AbstractArray{<:Any, N},
        adj::AbstractAdjointND{<:Any, N}
    ) where {N}
    bcs = _adjoint_bcs(adj)
    if _has_exclusive_periodic(bcs)
        gs = _grid_size(adj)
        for d in 1:N
            if bcs[d] isa PeriodicBC{:exclusive}
                selectdim(f_bar, d, 1) .+= selectdim(f_bar, d, gs[d])
            end
        end
        out_size = _adjoint_output_size(adj)
        ranges = ntuple(d -> 1:out_size[d], Val(N))
        return f_bar[ranges...]
    end
    return f_bar
end

# ── Default exclusive periodic in-place (no pool; subtypes may override) ──

function _adjoint_apply_exclusive_nd!(
        f_bar::AbstractArray{<:Any, N},
        adj::AbstractAdjointND{Tg, N},
        y_bar,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tg, N}
    Tv = promote_type(eltype(f_bar), Tg)
    gs = _grid_size(adj)
    f_work = zeros(Tv, gs...)
    _adjoint_nd_apply!(f_work, adj, y_bar, ops)
    bcs = _adjoint_bcs(adj)
    for d in 1:N
        if bcs[d] isa PeriodicBC{:exclusive}
            selectdim(f_work, d, 1) .+= selectdim(f_work, d, gs[d])
        end
    end
    out_size = _adjoint_output_size(adj)
    ranges = ntuple(d -> 1:out_size[d], Val(N))
    f_bar .= view(f_work, ranges...)
    return nothing
end


# ========================================
# Shared ND Adjoint Callable Interface
# ========================================
#
# All ND adjoint callables defined once on AbstractAdjointND.
# Per-type adjoint files only need: _adjoint_nd_apply! and accessors.
#
# Subtypes must implement:
#   _n_queries(adj)::Int
#   _grid_size(adj)::NTuple{N,Int}
#   _adjoint_bcs(adj)
#   _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
#
# Subtypes may override for performance:
#   _adjoint_apply_exclusive_nd!(f_bar, adj, y_bar, ops)  (pool-based)

# ── Allocating: adj(y_bar::AbstractVector) ──

function (adj::AbstractAdjointND{Tg, N})(
        y_bar::AbstractVector;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        _extra...
    ) where {Tg, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    Tv = promote_type(eltype(y_bar), Tg)
    f_bar = zeros(Tv, _grid_size(adj)...)
    _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    return _adjoint_nd_finalize(f_bar, adj)
end

# ── In-place: adj(f_bar, y_bar::AbstractVector) ──

function (adj::AbstractAdjointND{Tg, N})(
        f_bar::AbstractArray{Tv, N}, y_bar::AbstractVector;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        _extra...
    ) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    out_size = _adjoint_output_size(adj)
    size(f_bar) == out_size || _throw_adjoint_size_mismatch(size(f_bar), out_size)
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    if _has_exclusive_periodic(_adjoint_bcs(adj))
        _adjoint_apply_exclusive_nd!(f_bar, adj, y_bar, ops)
    else
        fill!(f_bar, zero(Tv))
        _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    end
    return f_bar
end

# ── Allocating scalar: adj(y_bar::Real) ──

function (adj::AbstractAdjointND{Tg, N})(
        y_bar::Real;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        _extra...
    ) where {Tg, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    _n_queries(adj) == 1 || _throw_adjoint_dim_mismatch("y_bar", 1, _n_queries(adj))
    Tv = promote_type(typeof(y_bar), Tg)
    f_bar = zeros(Tv, _grid_size(adj)...)
    _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    return _adjoint_nd_finalize(f_bar, adj)
end

# ── Allocating tuple: adj(y_bar::Tuple{Vararg{Real}}) ──

function (adj::AbstractAdjointND{Tg, N})(
        y_bar::Tuple{Vararg{Real}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        _extra...
    ) where {Tg, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    Tv = promote_type(eltype(y_bar), Tg)
    f_bar = zeros(Tv, _grid_size(adj)...)
    _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    return _adjoint_nd_finalize(f_bar, adj)
end

# ── In-place scalar: adj(f_bar, y_bar::Real) ──

function (adj::AbstractAdjointND{Tg, N})(
        f_bar::AbstractArray{Tv, N},
        y_bar::Real;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        _extra...
    ) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    out_size = _adjoint_output_size(adj)
    size(f_bar) == out_size || _throw_adjoint_size_mismatch(size(f_bar), out_size)
    _n_queries(adj) == 1 || _throw_adjoint_dim_mismatch("y_bar", 1, _n_queries(adj))
    if _has_exclusive_periodic(_adjoint_bcs(adj))
        _adjoint_apply_exclusive_nd!(f_bar, adj, y_bar, ops)
    else
        fill!(f_bar, zero(Tv))
        _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    end
    return f_bar
end

# ── In-place tuple: adj(f_bar, y_bar::Tuple{Vararg{Real}}) ──

function (adj::AbstractAdjointND{Tg, N})(
        f_bar::AbstractArray{Tv, N},
        y_bar::Tuple{Vararg{Real}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        _extra...
    ) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    out_size = _adjoint_output_size(adj)
    size(f_bar) == out_size || _throw_adjoint_size_mismatch(size(f_bar), out_size)
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    if _has_exclusive_periodic(_adjoint_bcs(adj))
        _adjoint_apply_exclusive_nd!(f_bar, adj, y_bar, ops)
    else
        fill!(f_bar, zero(Tv))
        _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    end
    return f_bar
end
