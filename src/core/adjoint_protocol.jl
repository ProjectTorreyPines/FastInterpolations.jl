# ========================================
# Shared Adjoint Callable Interface (1D + ND)
# ========================================
#
# All adjoint callables defined once on AbstractAdjoint1D / AbstractAdjointND.
# Per-type adjoint files only need core apply functions and accessors.
#
# ── 1D Subtypes must implement:
#   _n_queries(adj)::Int
#   _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)
#
# ── 1D Subtypes inherit (assuming `adj.bc::AbstractBC` and `adj.grid_size::Int`):
#   _adjoint_output_length(adj)::Int
#   _adjoint_internal_length(adj)::Int
#   _adjoint_1d_has_seam_fold(adj)::Bool
#   _adjoint_1d_finalize(f_bar, adj)
# Override these only if the subtype lacks the assumed fields (e.g.,
# `CubicAdjoint` reads `length(adj.cache.x)` instead of `adj.grid_size`).
#
# ── ND Subtypes must implement:
#   _n_queries(adj)::Int
#   _grid_size(adj)::NTuple{N,Int}
#   _adjoint_bcs(adj)
#   _adjoint_nd_apply!(f_bar, adj, y_bar, ops)

# ========================================
# Shared Error Helpers (@noinline cold path)
# ========================================

@noinline _throw_adjoint_dim_mismatch(label::String, got::Int, expected::Int) =
    throw(DimensionMismatch("$label length ($got) must match expected ($expected)"))

@noinline _throw_adjoint_size_mismatch(got::Tuple, expected::Tuple) =
    throw(DimensionMismatch("f_bar size $got must match output size $expected"))

@noinline _throw_adjoint_grid_too_small(n::Int) =
    throw(ArgumentError("adjoint requires at least 2 grid points, got $n"))

@noinline _throw_adjoint_grid_too_small(d::Int, n::Int) =
    throw(ArgumentError("adjoint requires at least 2 grid points on axis $d, got $n"))

# ========================================
# Shared OOB Skip Helpers (Compile-Time Dispatch)
# ========================================
#
# Zero-cost OOB handling by dispatching on extrap type:
# - FillExtrap: OOB → filled constant independent of f → zero gradient
# - ClampExtrap: OOB → f[boundary], but derivative is zero for constant extrap
# - NoExtrap/ExtendExtrap/WrapExtrap: no OOB skip (NoExtrap throws at construction)

# EvalValue OOB skip: only FillExtrap needs skip (fill value not function of f)
@inline _is_oob_skip(state::UInt8, ::FillExtrap) = state != IN_DOMAIN
@inline _is_oob_skip(::UInt8, ::AbstractExtrap) = false

# EvalDeriv OOB skip: both Clamp and Fill have zero derivative outside domain
@inline _is_oob_skip_deriv(state::UInt8, ::_ClampOrFill) = state != IN_DOMAIN
@inline _is_oob_skip_deriv(::UInt8, ::AbstractExtrap) = false

# ╔══════════════════════════════════════╗
# ║           1D Adjoint Protocol        ║
# ╚══════════════════════════════════════╝

# ── 1D Defaults ──
# Assume `adj.bc::AbstractBC` and `adj.grid_size::Int` fields. Concrete subtypes
# without those fields (e.g., `CubicAdjoint` uses `length(adj.cache.x)`) override.

@inline _adjoint_output_length(adj::AbstractAdjoint1D) =
    _is_periodic_seam_folded(adj.bc) ? adj.grid_size - 1 : adj.grid_size

@inline _adjoint_internal_length(adj::AbstractAdjoint1D) = adj.grid_size

@inline _adjoint_1d_has_seam_fold(adj::AbstractAdjoint1D) =
    _is_periodic_seam_folded(adj.bc)

# Dispatches on `adj.bc`. Subtypes without `.bc` (`HermiteAdjoint1D`) override directly.
@inline _adjoint_1d_finalize(f_bar::AbstractVector, adj::AbstractAdjoint1D) =
    _adjoint_1d_finalize(adj.bc, f_bar, adj)

@inline _adjoint_1d_finalize(::AbstractBC, f_bar::AbstractVector, ::AbstractAdjoint1D) = f_bar

# Seam-fold + in-place shrink. `resize!` on `Vector` is O(1) (no copy);
# slicing `f_bar[1:n-1]` would heap-allocate a copy on every call.
# `:extended` shares the seam-fold finalization with `:exclusive` (same body).
@inline function _adjoint_1d_finalize(
        ::PeriodicBC{:exclusive}, f_bar::Vector, adj::AbstractAdjoint1D,
    )
    n_internal = _adjoint_internal_length(adj)
    @inbounds f_bar[1] += f_bar[n_internal]
    resize!(f_bar, n_internal - 1)
    return f_bar
end
@inline function _adjoint_1d_finalize(
        ::PeriodicBC{:extended}, f_bar::Vector, adj::AbstractAdjoint1D,
    )
    n_internal = _adjoint_internal_length(adj)
    @inbounds f_bar[1] += f_bar[n_internal]
    resize!(f_bar, n_internal - 1)
    return f_bar
end

# ── Size / Introspection ──

Base.size(adj::AbstractAdjoint1D) = (_adjoint_output_length(adj), _n_queries(adj))
Base.size(adj::AbstractAdjoint1D, d::Integer) = size(adj)[d]

# ========================================
# 1D Allocating Callables
# ========================================

function (adj::AbstractAdjoint1D{Tg})(
        y_bar::AbstractVector; deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    Tv = _output_eltype(eltype(y_bar), Tg)
    f_bar = zeros(Tv, _adjoint_internal_length(adj))
    _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)
    return _adjoint_1d_finalize(f_bar, adj)
end

function (adj::AbstractAdjoint1D{Tg})(
        y_bar::Real; deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    _n_queries(adj) == 1 || _throw_adjoint_dim_mismatch("y_bar", 1, _n_queries(adj))
    Tv = _output_eltype(typeof(y_bar), Tg)
    f_bar = zeros(Tv, _adjoint_internal_length(adj))
    _adjoint_1d_apply!(f_bar, adj, (y_bar,), deriv)
    return _adjoint_1d_finalize(f_bar, adj)
end

function (adj::AbstractAdjoint1D{Tg})(
        y_bar::Tuple{Vararg{Real}}; deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    Tv = _output_eltype(eltype(y_bar), Tg)
    f_bar = zeros(Tv, _adjoint_internal_length(adj))
    _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)
    return _adjoint_1d_finalize(f_bar, adj)
end

# ========================================
# 1D In-Place Callables
# ========================================

function (adj::AbstractAdjoint1D{Tg})(
        f_bar::AbstractVector, y_bar::AbstractVector;
        deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    n_out = _adjoint_output_length(adj)
    length(f_bar) == n_out || _throw_adjoint_dim_mismatch("f_bar", length(f_bar), n_out)
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    if _adjoint_1d_has_seam_fold(adj)
        _adjoint_1d_apply_exclusive_inplace!(f_bar, adj, y_bar, deriv)
    else
        fill!(f_bar, zero(eltype(f_bar)))
        _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)
    end
    return f_bar
end

function (adj::AbstractAdjoint1D{Tg})(
        f_bar::AbstractVector, y_bar::Real;
        deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    n_out = _adjoint_output_length(adj)
    length(f_bar) == n_out || _throw_adjoint_dim_mismatch("f_bar", length(f_bar), n_out)
    _n_queries(adj) == 1 || _throw_adjoint_dim_mismatch("y_bar", 1, _n_queries(adj))
    if _adjoint_1d_has_seam_fold(adj)
        _adjoint_1d_apply_exclusive_inplace!(f_bar, adj, (y_bar,), deriv)
    else
        fill!(f_bar, zero(eltype(f_bar)))
        _adjoint_1d_apply!(f_bar, adj, (y_bar,), deriv)
    end
    return f_bar
end

function (adj::AbstractAdjoint1D{Tg})(
        f_bar::AbstractVector, y_bar::Tuple{Vararg{Real}};
        deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    n_out = _adjoint_output_length(adj)
    length(f_bar) == n_out || _throw_adjoint_dim_mismatch("f_bar", length(f_bar), n_out)
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    if _adjoint_1d_has_seam_fold(adj)
        _adjoint_1d_apply_exclusive_inplace!(f_bar, adj, y_bar, deriv)
    else
        fill!(f_bar, zero(eltype(f_bar)))
        _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)
    end
    return f_bar
end

# ========================================
# 1D Exclusive Periodic In-Place (Generic)
# ========================================

@with_pool pool function _adjoint_1d_apply_exclusive_inplace!(
        f_bar::AbstractVector{Tv}, adj::AbstractAdjoint1D, y_bar, deriv::DerivOp
    ) where {Tv}
    n_internal = _adjoint_internal_length(adj)
    f_work = zeros!(pool, Tv, n_internal)
    _adjoint_1d_apply!(f_work, adj, y_bar, deriv)
    @inbounds f_work[1] += f_work[n_internal]
    n_out = _adjoint_output_length(adj)
    @inbounds for k in 1:n_out
        f_bar[k] = f_work[k]
    end
    return nothing
end

# ========================================
# 1D Matrix Materialization (Debug/Verification)
# ========================================

"""
    Matrix(adj::AbstractAdjoint1D; deriv=EvalValue()) -> Matrix

Materialize the 1D adjoint as a dense matrix `Wᵀ` of size `(n_grid, n_query)`.

Each column `q` of `Wᵀ` is computed by probing with a unit vector `eₑ`:
`Wᵀ[:, q] = adj(eₑ)`, i.e., the grid-space sensitivity when only query point `q`
has unit sensitivity.

This is an O(n_grid × n_query) operation intended for debugging and verification,
not for production use.

# Example
```julia
adj = linear_adjoint(x, xq)
Wᵀ = Matrix(adj)                          # (n_grid × n_query)
W  = Matrix(adj)'                          # (n_query × n_grid)

@assert Wᵀ * y_bar ≈ adj(y_bar)           # matrix-vector == operator
@assert W * f ≈ itp.(xq)                  # forward matrix works too
```
"""
@with_pool pool function Base.Matrix(adj::AbstractAdjoint1D{Tg}; deriv::DerivOp = EvalValue()) where {Tg}
    n_out, n_query = size(adj)
    W_T = zeros(Tg, n_out, n_query)
    e_q = zeros!(pool, Tg, n_query)
    @inbounds for q in 1:n_query
        e_q[q] = one(Tg)
        adj(view(W_T, :, q), e_q; deriv = deriv)
        e_q[q] = zero(Tg)
    end
    return W_T
end

# ╔══════════════════════════════════════╗
# ║           ND Adjoint Protocol        ║
# ╚══════════════════════════════════════╝

# ── Output size (axis is always n+1 inclusive layout; seam-fold BCs shrink user dim by 1) ──
@inline _adjoint_output_size(adj::AbstractAdjointND{<:Any, N}) where {N} =
    map(
    (bc, n) -> _is_periodic_seam_folded(bc) ? n - 1 : n,
    _adjoint_bcs(adj), _grid_size(adj)
)

# ── Seam-fold detection (covers :exclusive AND :extended) ──

@inline _has_seam_fold(bcs::Tuple) = any(_is_periodic_seam_folded, bcs)

# ── Periodic finalization (fold seam-fold axes + truncate to user dim) ──

function _adjoint_nd_finalize(
        f_bar::AbstractArray{<:Any, N},
        adj::AbstractAdjointND{<:Any, N}
    ) where {N}
    bcs = _adjoint_bcs(adj)
    if _has_seam_fold(bcs)
        gs = _grid_size(adj)
        for d in 1:N
            if _is_periodic_seam_folded(bcs[d])
                selectdim(f_bar, d, 1) .+= selectdim(f_bar, d, gs[d])
            end
        end
        out_size = _adjoint_output_size(adj)
        ranges = ntuple(d -> 1:out_size[d], Val(N))
        return f_bar[ranges...]
    end
    return f_bar
end

# ── Seam-fold in-place (pool-allocated work buffer) ──

# Per-axis seam fold: literal dimension `d` ensures `selectdim(_, d, _)`
# specializes at compile time — runtime `d` produces SubArrays whose type
# depends on the dim, forcing per-call boxing on heterogeneous BC tuples.
# `:extended` shares the seam-fold mechanism with `:exclusive` (same body).
@inline _seam_fold_axis!(f_work, ::Val{d}, gs_d::Int, ::PeriodicBC{:exclusive}) where {d} =
    (selectdim(f_work, d, 1) .+= selectdim(f_work, d, gs_d); nothing)
@inline _seam_fold_axis!(f_work, ::Val{d}, gs_d::Int, ::PeriodicBC{:extended}) where {d} =
    (selectdim(f_work, d, 1) .+= selectdim(f_work, d, gs_d); nothing)
@inline _seam_fold_axis!(f_work, ::Val{d}, ::Int, ::AbstractBC) where {d} = nothing

# Compile-time-unrolled seam-fold loop (one branch per axis, all literal `d`).
@generated function _apply_seam_fold!(
        f_work::AbstractArray{T, N},
        bcs::NTuple{N, AbstractBC},
        gs::NTuple{N, Int}
    ) where {T, N}
    body = [:(_seam_fold_axis!(f_work, Val($d), gs[$d], bcs[$d])) for d in 1:N]
    return quote
        $(body...)
        return nothing
    end
end

# Compile-time-unrolled work→user trim view (replaces `view(f_work, ranges...)`
# where `ranges = ntuple(d -> 1:out_size[d], Val(N))` — splatting an `ntuple`
# of UnitRanges produces a runtime-dim view whose result type the compiler
# can't fold, causing minor boxing on `f_bar .= view(...)`).
@generated function _view_first_n(
        f_work::AbstractArray{T, N},
        out_size::NTuple{N, Int}
    ) where {T, N}
    args = [:(1:out_size[$d]) for d in 1:N]
    return :(view(f_work, $(args...)))
end

@with_pool pool function _adjoint_apply_exclusive_nd!(
        f_bar::AbstractArray{Tv, N},
        adj::AbstractAdjointND{Tg, N},
        y_bar,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N}
    gs = _grid_size(adj)
    f_work = zeros!(pool, Tv, gs...)
    _adjoint_nd_apply!(f_work, adj, y_bar, ops)
    _apply_seam_fold!(f_work, _adjoint_bcs(adj), gs)
    f_bar .= _view_first_n(f_work, _adjoint_output_size(adj))
    return nothing
end

# ========================================
# ND Allocating Callables
# ========================================

# ── Allocating: adj(y_bar::AbstractVector) ──

function (adj::AbstractAdjointND{Tg, N})(
        y_bar::AbstractVector;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        _extra...
    ) where {Tg, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    Tv = _output_eltype(eltype(y_bar), Tg)
    f_bar = zeros(Tv, _grid_size(adj)...)
    _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    return _adjoint_nd_finalize(f_bar, adj)
end

# ── Allocating scalar: adj(y_bar::Real) ──

function (adj::AbstractAdjointND{Tg, N})(
        y_bar::Real;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        _extra...
    ) where {Tg, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    _n_queries(adj) == 1 || _throw_adjoint_dim_mismatch("y_bar", 1, _n_queries(adj))
    Tv = _output_eltype(typeof(y_bar), Tg)
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
    Tv = _output_eltype(eltype(y_bar), Tg)
    f_bar = zeros(Tv, _grid_size(adj)...)
    _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    return _adjoint_nd_finalize(f_bar, adj)
end

# ========================================
# ND In-Place Callables
# ========================================

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
    if _has_seam_fold(_adjoint_bcs(adj))
        _adjoint_apply_exclusive_nd!(f_bar, adj, y_bar, ops)
    else
        fill!(f_bar, zero(Tv))
        _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    end
    return f_bar
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
    if _has_seam_fold(_adjoint_bcs(adj))
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
    if _has_seam_fold(_adjoint_bcs(adj))
        _adjoint_apply_exclusive_nd!(f_bar, adj, y_bar, ops)
    else
        fill!(f_bar, zero(Tv))
        _adjoint_nd_apply!(f_bar, adj, y_bar, ops)
    end
    return f_bar
end

# ========================================
# ND Matrix Materialization (Debug/Verification)
# ========================================

"""
    Matrix(adj::AbstractAdjointND; deriv=EvalValue()) -> Matrix

Materialize the ND adjoint operator as a dense matrix `Wᵀ` of size
`(prod(grid_sizes), n_query)`.

This is an O(n_grid × n_query) operation intended for debugging and verification.
"""
@with_pool pool function Base.Matrix(
        adj::AbstractAdjointND{Tg, N};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {Tg, N}
    out_size = _adjoint_output_size(adj)
    n_out = prod(out_size)
    n_query = _n_queries(adj)
    W_T = zeros(Tg, n_out, n_query)
    e_q = zeros!(pool, Tg, n_query)
    f_bar = zeros!(pool, Tg, out_size...)
    @inbounds for q in 1:n_query
        e_q[q] = one(Tg)
        adj(f_bar, e_q; deriv = deriv)
        W_T[:, q] .= vec(f_bar)
        e_q[q] = zero(Tg)
    end
    return W_T
end
