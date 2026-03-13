# ========================================
# Shared 1D Adjoint Callable Interface
# ========================================
#
# All 1D adjoint callables defined once on AbstractAdjoint.
# Per-type adjoint files only need: _adjoint_1d_apply! and accessors.
#
# Subtypes must implement:
#   _n_queries(adj)::Int                    — number of baked query points
#   _adjoint_output_length(adj)::Int        — user-facing output length
#   _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)  — core scatter/solve (accumulates)
#
# Subtypes may override (for periodic BC):
#   _adjoint_internal_length(adj)::Int              — alloc size (default: output length)
#   _adjoint_1d_has_exclusive_periodic(adj)::Bool    — default: false
#   _adjoint_1d_finalize(f_bar, adj)                — default: identity
#
# The exclusive periodic in-place handler is provided generically;
# concrete types do NOT need to override it.

# ── Defaults (non-periodic types inherit these as-is) ──

@inline _adjoint_internal_length(adj::AbstractAdjoint) = _adjoint_output_length(adj)
@inline _adjoint_1d_has_exclusive_periodic(::AbstractAdjoint) = false
@inline _adjoint_1d_finalize(f_bar::AbstractVector, ::AbstractAdjoint) = f_bar

# ── Size / Introspection ──

Base.size(adj::AbstractAdjoint) = (_adjoint_output_length(adj), _n_queries(adj))
Base.size(adj::AbstractAdjoint, d::Integer) = size(adj)[d]

# ========================================
# Allocating Callables
# ========================================

function (adj::AbstractAdjoint{Tg})(
        y_bar::AbstractVector; deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    Tv = promote_type(eltype(y_bar), Tg)
    f_bar = zeros(Tv, _adjoint_internal_length(adj))
    _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)
    return _adjoint_1d_finalize(f_bar, adj)
end

function (adj::AbstractAdjoint{Tg})(
        y_bar::Real; deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    _n_queries(adj) == 1 || _throw_adjoint_dim_mismatch("y_bar", 1, _n_queries(adj))
    Tv = promote_type(typeof(y_bar), Tg)
    f_bar = zeros(Tv, _adjoint_internal_length(adj))
    _adjoint_1d_apply!(f_bar, adj, (y_bar,), deriv)
    return _adjoint_1d_finalize(f_bar, adj)
end

function (adj::AbstractAdjoint{Tg})(
        y_bar::Tuple{Vararg{Real}}; deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    Tv = promote_type(eltype(y_bar), Tg)
    f_bar = zeros(Tv, _adjoint_internal_length(adj))
    _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)
    return _adjoint_1d_finalize(f_bar, adj)
end

# ========================================
# In-Place Callables
# ========================================

function (adj::AbstractAdjoint{Tg})(
        f_bar::AbstractVector, y_bar::AbstractVector;
        deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    n_out = _adjoint_output_length(adj)
    length(f_bar) == n_out || _throw_adjoint_dim_mismatch("f_bar", length(f_bar), n_out)
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    if _adjoint_1d_has_exclusive_periodic(adj)
        _adjoint_1d_apply_exclusive_inplace!(f_bar, adj, y_bar, deriv)
    else
        fill!(f_bar, zero(eltype(f_bar)))
        _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)
    end
    return f_bar
end

function (adj::AbstractAdjoint{Tg})(
        f_bar::AbstractVector, y_bar::Real;
        deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    n_out = _adjoint_output_length(adj)
    length(f_bar) == n_out || _throw_adjoint_dim_mismatch("f_bar", length(f_bar), n_out)
    _n_queries(adj) == 1 || _throw_adjoint_dim_mismatch("y_bar", 1, _n_queries(adj))
    if _adjoint_1d_has_exclusive_periodic(adj)
        _adjoint_1d_apply_exclusive_inplace!(f_bar, adj, (y_bar,), deriv)
    else
        fill!(f_bar, zero(eltype(f_bar)))
        _adjoint_1d_apply!(f_bar, adj, (y_bar,), deriv)
    end
    return f_bar
end

function (adj::AbstractAdjoint{Tg})(
        f_bar::AbstractVector, y_bar::Tuple{Vararg{Real}};
        deriv::DerivOp = EvalValue(), _extra...
    ) where {Tg}
    n_out = _adjoint_output_length(adj)
    length(f_bar) == n_out || _throw_adjoint_dim_mismatch("f_bar", length(f_bar), n_out)
    nq = _n_queries(adj)
    length(y_bar) == nq || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), nq)
    if _adjoint_1d_has_exclusive_periodic(adj)
        _adjoint_1d_apply_exclusive_inplace!(f_bar, adj, y_bar, deriv)
    else
        fill!(f_bar, zero(eltype(f_bar)))
        _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)
    end
    return f_bar
end

# ========================================
# Exclusive Periodic In-Place (Generic)
# ========================================
#
# Generic default: allocates a pool work buffer at internal length,
# runs core apply, folds endpoint, copies to user f_bar.
# Works for any AbstractAdjoint with exclusive periodic BC.

@with_pool pool function _adjoint_1d_apply_exclusive_inplace!(
        f_bar::AbstractVector{Tv}, adj::AbstractAdjoint, y_bar, deriv::DerivOp
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
# Matrix Materialization (Debug/Verification)
# ========================================

"""
    Matrix(adj::AbstractAdjoint; deriv=EvalValue()) -> Matrix

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
@with_pool pool function Base.Matrix(adj::AbstractAdjoint{Tg}; deriv::DerivOp = EvalValue()) where {Tg}
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
