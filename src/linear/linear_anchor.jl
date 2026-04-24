# ========================================
# Linear Anchored Query
# ========================================
# Precomputed geometry for ultra-fast linear interpolation evaluation
# at a fixed query point. Enables speedup by eliminating interval search
# for repeated evaluations.
#
# Include order: ops.jl → utils.jl → linear_kernels.jl → linear_interp.jl → linear_anchor.jl
#
# ========================================
# _LinearAnchoredQuery Type (Internal)
# ========================================

"""
    _LinearAnchoredQuery{Tg, Tq}

Precomputed geometry for ultra-fast linear interpolation at a fixed query point.
Internal API: no runtime grid validation; callers must ensure the anchor
matches the interpolant grid.

# Type Parameters
- `Tg`: Grid type — normally Float32/Float64, but unconstrained for duck-typed grids (e.g. ForwardDiff.Dual)
- `Tq`: Query type (widened to `promote_type(Tq, Tg)` by the outer constructor)

# Fields
- `idxL`: Left cell index (`1 ≤ idxL ≤ n-1` normally; equals `n` at periodic-exclusive seam)
- `idxR`: Right cell index (`idxL + 1` normally; wraps to `1` at periodic-exclusive seam)
- `xq`: Original query point (or wrapped value for periodic), preserves original precision
- `state`: Domain state (`IN_DOMAIN`, `OOB_LEFT`, or `OOB_RIGHT`)
- `xL`: Left grid point of the interval (avoids re-indexing x[idxL] which triggers TwicePrecision on Range)
- `h`: Interval width (xR - xL)
- `inv_h`: Precomputed reciprocal (1/h) for fast derivative computation
- `alpha`: Normalized position within interval: (xq - xL) / h, preserves precision

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
aq = _anchor_query(x, 0.35, Val(:linear))

itp1 = linear_interp(x, sin.(2π .* x))
itp2 = linear_interp(x, cos.(2π .* x))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=DerivOp(1))     # Reuses same anchor for derivative
```

# Performance
Anchored evaluation is faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search.

# Efficiency
- `alpha` for EvalValue: `muladd(alpha, yR - yL, yL)` (no division)
- `inv_h` for EvalDeriv1: `(yR - yL) * inv_h` (no division)
"""
struct _LinearAnchoredQuery{Tg, Tq <: Real}
    # Corner-index stencil: `stencil[1]` is the left index (idxL), `stencil[2]`
    # is the right index (idxR). For non-periodic cells `idxR == idxL + 1`; for
    # periodic-exclusive seam cells `idxR == 1` (wrap). Unified across all
    # wrap-aware methods via `_IdxStencil{K}` (src/core/idx_stencil.jl).
    # Legacy `aq.idxL` / `aq.idxR` accessors are preserved via `getproperty`
    # below — every existing call site reads through the virtual property.
    stencil::_IdxStencil{2}
    xq::Tq                     # query point (possibly wrapped)
    state::UInt8               # IN_DOMAIN / OOB_LEFT / OOB_RIGHT
    xL::Tg                     # left grid point
    h::Tg                      # interval width
    inv_h::Tg                  # precomputed 1/h
    alpha::Tq                  # normalized position: (xq - xL) / h
end

# ──────────────────────────────────────────────────────────────
# Virtual property accessors — legacy `aq.idxL` / `aq.idxR` ergonomics
# ──────────────────────────────────────────────────────────────
# Preserves every existing call site (kernel eval, adjoint scatter, Series
# one-shot, tests) without a single source change downstream.
#
# Dispatch pattern: `getproperty(aq, s::Symbol)` delegates to `_get_lin_prop(aq, Val(s))`,
# one method per property symbol. This forces compile-time specialization — each
# property returns a concrete type and the call inlines to a single `getfield`
# (+ tuple index for `:idxL` / `:idxR`). A single-method `getproperty` with
# `s === :idxL && ...` branches is *not* reliably inlined inside hot loops:
# the union of possible return types (Int, Tq, UInt8, Tg, _IdxStencil{2})
# defeats return-type inference and causes boxing. Val-dispatch sidesteps this.
@inline Base.getproperty(aq::_LinearAnchoredQuery, s::Symbol) = _get_lin_prop(aq, Val(s))
@inline _get_lin_prop(aq::_LinearAnchoredQuery, ::Val{:idxL}) = getfield(aq, :stencil)[1]
@inline _get_lin_prop(aq::_LinearAnchoredQuery, ::Val{:idxR}) = getfield(aq, :stencil)[2]
@inline _get_lin_prop(aq::_LinearAnchoredQuery, ::Val{s}) where {s} = getfield(aq, s)
@inline Base.propertynames(::_LinearAnchoredQuery) =
    (:stencil, :idxL, :idxR, :xq, :state, :xL, :h, :inv_h, :alpha)

# Outer constructor (positional pair): infers Tq from alpha's arithmetic type
# and promotes xq to match. Keeps the legacy 8-arg call shape used by every
# existing caller (Series one-shot, adjoint fixup, `_anchor_query`).
#
# For Float grids:  alpha::Tq, xq::Tq — no conversion (identity).
# For Dual grids + Float query:  alpha::Dual (from grid arithmetic),
#   xq promoted to Dual via convert (zero partials = "query has no grid sensitivity").
@inline function _LinearAnchoredQuery(idxL::Int, idxR::Int, xq, state::UInt8, xL::Tg, h::Tg, inv_h::Tg, alpha) where {Tg}
    Ta = typeof(alpha)
    xq_p = convert(Ta, xq)
    return _LinearAnchoredQuery{Tg, Ta}(_pair(idxL, idxR), xq_p, state, xL, h, inv_h, alpha)
end

# Stencil-native outer constructor — preferred for new code.
@inline function _LinearAnchoredQuery(stencil::_IdxStencil{2}, xq, state::UInt8, xL::Tg, h::Tg, inv_h::Tg, alpha) where {Tg}
    Ta = typeof(alpha)
    xq_p = convert(Ta, xq)
    return _LinearAnchoredQuery{Tg, Ta}(stencil, xq_p, state, xL, h, inv_h, alpha)
end

# ========================================
# Anchored Kernel Overloads
# ========================================
# These _linear_kernel methods receive the anchor directly and extract
# the appropriate precomputed value (alpha or inv_h) based on the operation.
# Leverages Julia's dispatch system for clean, unified API.

"""
    _linear_kernel(::EvalValue, yL, yR, aq::_LinearAnchoredQuery)

Evaluate linear interpolation using anchor's precomputed alpha.
No division: uses muladd(alpha, yR - yL, yL).
"""
@inline function _linear_kernel(::EvalValue, yL::Tv, yR::Tv, aq::_LinearAnchoredQuery) where {Tv}
    return muladd(aq.alpha, yR - yL, yL)
end

"""
    _linear_kernel(::EvalDeriv1, yL, yR, aq::_LinearAnchoredQuery)

Evaluate first derivative using anchor's precomputed inv_h.
No division: uses (yR - yL) * inv_h.
"""
@inline function _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, aq::_LinearAnchoredQuery{Tg}) where {Tg, Tv}
    return (yR - yL) * aq.inv_h
end

"""Second derivative of linear is always zero."""
@inline function _linear_kernel(::EvalDeriv2, yL::Tv, ::Tv, ::_LinearAnchoredQuery{Tg}) where {Tg, Tv}
    return 0 * yL
end

"""Third derivative of linear is always zero."""
@inline function _linear_kernel(::EvalDeriv3, yL::Tv, ::Tv, ::_LinearAnchoredQuery{Tg}) where {Tg, Tv}
    return 0 * yL
end

"""Generic fallback: N-th derivative of linear (anchored) is zero for N ≥ 2."""
@inline function _linear_kernel(::DerivOp{N}, yL::Tv, ::Tv, ::_LinearAnchoredQuery{Tg}) where {N, Tg, Tv}
    return 0 * yL
end

# ========================================
# Anchor Construction
# ========================================

"""
    _anchor_query(x::AbstractVector{T}, xq::T, ::Val{:linear}; wrap::Bool=false) -> _LinearAnchoredQuery

Create an anchored query for ultra-fast linear interpolation at a fixed point.

# Arguments
- `x`: Grid points (must match grid used for interpolant construction)
- `xq`: Query point (scalar)
- `::Val{:linear}`: Type tag to distinguish from cubic anchor
- `wrap`: If true, wrap `xq` to domain [x[1], x[end]) before anchoring.
          Used for `extrap=WrapExtrap()` mode.

# Returns
`_LinearAnchoredQuery{T}` with precomputed geometry.

# Example
```julia
x = range(0.0, 1.0, 101)
itp1 = linear_interp(collect(x), sin.(2π .* x))
itp2 = linear_interp(collect(x), cos.(2π .* x))

aq = _anchor_query(collect(x), 0.35, Val(:linear))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=DerivOp(1))     # Reuses same anchor for derivative
```
"""
# Unified scalar anchor construction. The outer constructor of _LinearAnchoredQuery
# handles Tg×Tq type promotion, so no _promote_for_anchor call is needed here.
@inline function _anchor_query(
        x::AbstractVector{Tg},
        xq,
        ::Val{:linear},
        wrap::Bool = false,
        searcher::P = DEFAULT_SEARCHER
    ) where {Tg, P <: Searcher}
    return _linear_anchor_query_impl(x, xq, wrap, _resolve_searcher_for_grid(x, searcher))
end

"""
    _anchor_query(x::AbstractVector{Tg}, xq::AbstractVector, ::Val{:linear}; wrap::Bool=false)

Create anchored queries for multiple query points with precision preservation.

# Precision Preservation
The outer `_LinearAnchoredQuery` constructor promotes via `promote_type(Tq, Tg)`,
so wider precision is preserved when `Tq` differs from `Tg`.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.75], Val(:linear))

itp1 = linear_interp(x, sin.(2π .* x))
itp2 = linear_interp(x, cos.(2π .* x))

vals1 = itp1(aq_vec)  # Batch evaluation
vals2 = itp2(aq_vec)  # Reuse same anchors
```
"""
function _anchor_query(
        x::AbstractVector{Tg},
        xq::AbstractVector{Tq},
        ::Val{:linear},
        wrap::Bool = false,
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {Tg, Tq <: Real, P <: Searcher}
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)
    # Tq_promoted accounts for Tg×Tq arithmetic: promote_type(Tq, Tg) gives the
    # widened query type that the outer constructor will use for alpha and xq.
    Tq_promoted = promote_type(Tq, Tg)
    output = Vector{_LinearAnchoredQuery{Tg, Tq_promoted}}(undef, length(xq))

    @inbounds for k in eachindex(xq)
        output[k] = _linear_anchor_query_impl(x, xq[k], wrap, searcher_resolved)
    end
    return output
end

"""
    _fill_anchors!(buffer, x, xq, ::Val{:linear}; wrap=false) -> buffer

Fill a caller-allocated buffer with anchored queries for linear interpolation.
In-place version of `_anchor_query(x, xq, Val(:linear))` — zero-allocation as long as
the caller reuses `buffer`. Writes `length(xq)` entries.

# Arguments
- `buffer::Vector{_LinearAnchoredQuery{Tg,Tq}}`: Caller-allocated buffer (length >= length(xq))
- `x::AbstractVector{Tg}`: Grid points (must match interpolant's grid)
- `xq::AbstractVector`: Query points (any Real type)
- `::Val{:linear}`: Type tag for linear interpolation
- `wrap::Bool=false`: If true, wrap query points to domain [x[1], x[end])

# Precision Preservation
The outer `_LinearAnchoredQuery` constructor promotes via `promote_type(S, Tg)`,
so wider precision is preserved when `S` differs from `Tg`.
"""
@inline function _fill_anchors!(
        buffer::AbstractVector{_LinearAnchoredQuery{Tg, Tq}},
        x::AbstractVector{Tg},
        xq::AbstractVector{S},
        ::Val{:linear},
        wrap::Bool = false,
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {Tg, Tq <: Real, S <: Real, P <: Searcher}
    @assert length(buffer) >= length(xq) "Buffer too small: $(length(buffer)) < $(length(xq))"
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)

    @inbounds for k in eachindex(xq)
        buffer[k] = _linear_anchor_query_impl(x, xq[k], wrap, searcher_resolved)
    end
    return buffer
end

"""
    _linear_anchor_query_impl(x, xq, wrap, policy) -> _LinearAnchoredQuery

Internal implementation of _anchor_query for linear interpolation.

# Arguments
- `x`: Grid points (type Tg)
- `xq`: Query point (type Tq, can differ from Tg for precision preservation)
- `wrap`: Whether to wrap query point to domain
- `policy`: Search policy for interval search (default: DEFAULT_SEARCHER)

# AD Support
When `xq` is a ForwardDiff.Dual, the returned anchor preserves the Dual type
in `xq` and `alpha` fields. The interval search uses `_extract_primal(xq)` for comparisons.
"""
@inline function _linear_anchor_query_impl(
        x::AbstractVector{Tg},
        xq::Tq,
        wrap::Bool,
        policy::P = DEFAULT_SEARCHER
    ) where {Tg, Tq <: Real, P <: Searcher}
    loc = _anchor_loc(x, xq, wrap, policy)

    # Compute geometry (linear-internal concern)
    h = _get_h(x, loc.xR, loc.xL)
    inv_h = _get_inv_h(x, loc.xR, loc.xL)
    alpha = (loc.xq - loc.xL) * inv_h

    # `_anchor_loc` never returns a periodic-exclusive seam pair — it operates
    # on a fixed grid with at most wrap-to-domain remapping — so `idxR = idxL+1`
    # here. Periodic-exclusive seam anchors are constructed via `_LinearAnchoredQuery(...)`
    # directly in the exclusive periodic one-shot helpers (bypassing `_anchor_loc`).
    return _LinearAnchoredQuery(loc.idx, loc.idx + 1, loc.xq, loc.state, loc.xL, h, inv_h, alpha)
end

# ========================================
# LinearInterpolant Evaluation with Anchor
# ========================================

"""
    (itp::LinearInterpolant)(aq::_LinearAnchoredQuery; deriv::DerivOp=EvalValue())

Evaluate linear interpolant at anchored query point. Ultra-fast path that
skips interval search.

# Arguments
- `aq`: Pre-computed anchored query from `_anchor_query`
- `deriv`: Derivative order (0=value, 1=first derivative)

# Example
```julia
itp = linear_interp(x, y)
aq = _anchor_query(x, 0.5, Val(:linear))
val = itp(aq)           # Value
d1 = itp(aq; deriv=DerivOp(1))   # First derivative
```
"""
@inline function (itp::LinearInterpolant{Tg})(aq::_LinearAnchoredQuery{Tg}; deriv::DerivOp = EvalValue()) where {Tg}
    return _linear_eval_with_anchor(itp, aq, deriv)
end

@inline function _linear_eval_with_anchor(
        itp::LinearInterpolant{Tg},
        aq::_LinearAnchoredQuery{Tg},
        op::O
    ) where {Tg, O <: AbstractEvalOp}
    # Handle extrapolation based on mode and side
    return _linear_anchor_dispatch(itp, aq, op, itp.extrap)
end

# ========================================
# Shared Raw-Vector Anchor Eval
# ========================================
# Canonical evaluation functions that take raw y vector (not interpolant struct).
# Used by both interpolant anchor dispatch AND series one-shot evaluation.

# Default case (extension, wrap, inbounds): direct kernel evaluation
@inline function _linear_eval_at_anchor(
        y::AbstractVector,
        aq::_LinearAnchoredQuery,
        op::AbstractEvalOp,
        ::AbstractExtrap
    )
    @inbounds return _linear_kernel(op, y[aq.idxL], y[aq.idxR], aq)
end

# No extrapolation: throw DomainError if outside domain
@inline function _linear_eval_at_anchor(
        y::AbstractVector,
        aq::_LinearAnchoredQuery,
        op::AbstractEvalOp,
        ::NoExtrap
    )
    aq.state != IN_DOMAIN && throw(DomainError(aq.xq, "query point outside domain"))
    @inbounds return _linear_kernel(op, y[aq.idxL], y[aq.idxR], aq)
end

# Clamp/Fill extrapolation: boundary value if OOB
@inline function _linear_eval_at_anchor(
        y::AbstractVector,
        aq::_LinearAnchoredQuery,
        op::AbstractEvalOp,
        extrap::_ClampOrFill
    )
    if aq.state != IN_DOMAIN
        y_bnd = aq.state == OOB_LEFT ? first(y) : last(y)
        return _eval_extrapolation(op, y_bnd, extrap, aq.xq)
    end
    @inbounds return _linear_kernel(op, y[aq.idxL], y[aq.idxR], aq)
end

# ========================================
# Interpolant Anchor Dispatch (thin wrappers)
# ========================================

# Default case (extension, wrap): delegate to shared
@inline _linear_anchor_dispatch(
    itp::LinearInterpolant, aq::_LinearAnchoredQuery, op::AbstractEvalOp, ext::AbstractExtrap
) = _linear_eval_at_anchor(itp.y, aq, op, ext)

# No extrapolation: enriched error message with domain bounds, then delegate
@inline function _linear_anchor_dispatch(
        itp::LinearInterpolant{Tg},
        aq::_LinearAnchoredQuery{Tg, Tq},
        op::AbstractEvalOp,
        ::NoExtrap
    ) where {Tg, Tq <: Real}
    if aq.state != IN_DOMAIN
        x_min, x_max = first(itp.x), last(itp.x)
        throw(DomainError(aq.xq, "query point outside domain [$x_min, $x_max]"))
    end
    @inbounds return _linear_kernel(op, itp.y[aq.idxL], itp.y[aq.idxR], aq)
end

# Clamp/Fill: delegate to shared
@inline _linear_anchor_dispatch(
    itp::LinearInterpolant, aq::_LinearAnchoredQuery, op::AbstractEvalOp, ext::_ClampOrFill
) = _linear_eval_at_anchor(itp.y, aq, op, ext)

# ========================================
# Vector Evaluation with Anchors
# ========================================

"""
    (itp::LinearInterpolant)(aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg}}; deriv::DerivOp=EvalValue())

Evaluate linear interpolant at multiple anchored query points.
Returns newly allocated vector with output type promoted from Tv and Tq.
"""
function (itp::LinearInterpolant{Tg, Tv})(
        aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg, Tq}};
        deriv::DerivOp = EvalValue()
    ) where {Tg, Tv, Tq <: Real}
    T_out = _output_eltype(Tv, Tg, Tq)
    output = Vector{T_out}(undef, length(aq_vec))
    @inbounds for i in eachindex(aq_vec)
        output[i] = _linear_eval_with_anchor(itp, aq_vec[i], deriv)
    end
    return output
end

"""
    (itp::LinearInterpolant)(output::AbstractVector, aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg}}; deriv::DerivOp=EvalValue())

In-place evaluation at multiple anchored query points. Zero allocation.
"""
function (itp::LinearInterpolant{Tg})(
        output::AbstractVector,
        aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg}};
        deriv::DerivOp = EvalValue()
    ) where {Tg}
    @assert length(output) == length(aq_vec) "output length must match aq_vec length"
    @inbounds for i in eachindex(aq_vec)
        output[i] = _linear_eval_with_anchor(itp, aq_vec[i], deriv)
    end
    return output
end
