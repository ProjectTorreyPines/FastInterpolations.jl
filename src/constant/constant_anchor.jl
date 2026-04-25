# ========================================
# Constant Anchored Query
# ========================================
# Precomputed geometry for ultra-fast constant interpolation evaluation
# at a fixed query point. Enables speedup by eliminating interval search
# for repeated evaluations.
#
# Include order: ops.jl → utils.jl → constant_kernels.jl → constant_interp.jl → constant_anchor.jl
#
# ========================================
# _ConstantAnchoredQuery Type (Internal)
# ========================================

"""
    _ConstantAnchoredQuery{Tg, Tq}

Precomputed geometry for ultra-fast constant interpolation at a fixed query point.
Internal API: no runtime grid validation; callers must ensure the anchor
matches the interpolant grid.

# Type Parameters
- `Tg`: Grid element type (stored in `h`)
- `Tq <: Real`: Query point type (stored in `xq`, `dL`); may widen `Tg` (e.g. `Dual` for AD)

# Fields
- `stencil::_IdxStencil{2}`: Corner-index stencil; `stencil[1]` is the left index, `stencil[2]` the right
  (legacy `aq.idxL` / `aq.idxR` virtual properties read through `getproperty` — see below).
  For non-periodic cells `idxR == idxL + 1`; at periodic-exclusive seam `idxL == n`, `idxR == 1` (wrap).
- `xq`: Original query point (or wrapped value for periodic)
- `state`: Domain state (`IN_DOMAIN`, `OOB_LEFT`, or `OOB_RIGHT`)
- `h`: Interval width (used by all side modes via `_compute_single_offset`)
- `dL`: Offset from left boundary (used by all side modes via `_compute_single_offset`)

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
aq = _anchor_query(x, 0.35, Val(:constant))

itp1 = constant_interp(x, sin.(2π .* x))
itp2 = constant_interp(x, cos.(2π .* x))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq)              # Reuses same anchor
```

# Performance
Anchored evaluation is faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search.
"""
struct _ConstantAnchoredQuery{Tg, Tq <: Real}
    # Corner-index stencil: `stencil[1]` is the left index (idxL),
    # `stencil[2]` is the right index (idxR). For non-periodic cells
    # `idxR == idxL + 1`; for periodic-exclusive seam cells `idxR == 1` (wrap).
    # Unified across all wrap-aware methods via `_IdxStencil{K}`
    # (src/core/idx_stencil.jl). Legacy `aq.idxL` / `aq.idxR` accessors are
    # preserved via `getproperty` below.
    stencil::_IdxStencil{2}
    xq::Tq                     # query point (possibly wrapped, may be Dual for AD)
    state::UInt8               # IN_DOMAIN / OOB_LEFT / OOB_RIGHT
    h::Tg                      # interval width
    dL::Tq                     # offset from left boundary (same type as xq for AD)
end

# ──────────────────────────────────────────────────────────────
# Virtual property accessors — legacy `aq.idxL` / `aq.idxR` ergonomics
# ──────────────────────────────────────────────────────────────
# Val-dispatch pattern (see linear_anchor.jl for rationale): one method per
# property symbol so the compiler specializes each access to a single `getfield`
# (+ tuple index) with concrete return type — no boxing from union-wide
# `getproperty` return.
@inline Base.getproperty(aq::_ConstantAnchoredQuery, s::Symbol) = _get_const_prop(aq, Val(s))
@inline _get_const_prop(aq::_ConstantAnchoredQuery, ::Val{:idxL}) = getfield(aq, :stencil)[1]
@inline _get_const_prop(aq::_ConstantAnchoredQuery, ::Val{:idxR}) = getfield(aq, :stencil)[2]
@inline _get_const_prop(aq::_ConstantAnchoredQuery, ::Val{s}) where {s} = getfield(aq, s)
@inline Base.propertynames(::_ConstantAnchoredQuery) =
    (:stencil, :idxL, :idxR, :xq, :state, :h, :dL)

# Stencil-native outer — infers `Tg, Tq` from arg types so callers can write
# `_ConstantAnchoredQuery(_IdxPair(idxL, idxR), xq, state, h, dL)` without
# specifying type params. Mirrors Linear's stencil-native outer.
@inline _ConstantAnchoredQuery(stencil::_IdxStencil{2}, xq::Tq, state::UInt8, h::Tg, dL::Tq) where {Tg, Tq} =
    _ConstantAnchoredQuery{Tg, Tq}(stencil, xq, state, h, dL)

# ========================================
# Anchor Construction
# ========================================

"""
    _anchor_query(x::AbstractVector{T}, xq::T, ::Val{:constant}; wrap::Bool=false) -> _ConstantAnchoredQuery

Create an anchored query for ultra-fast constant interpolation at a fixed point.

# Arguments
- `x`: Grid points (must match grid used for interpolant construction)
- `xq`: Query point (scalar)
- `::Val{:constant}`: Type tag to distinguish from other anchor types
- `wrap`: If true, wrap `xq` to domain [x[1], x[end]) before anchoring.
          Used for `extrap=WrapExtrap()` mode.

# Returns
`_ConstantAnchoredQuery{T}` with precomputed geometry.

# Example
```julia
x = range(0.0, 1.0, 101)
itp1 = constant_interp(collect(x), sin.(2π .* x))
itp2 = constant_interp(collect(x), cos.(2π .* x))

aq = _anchor_query(collect(x), 0.35, Val(:constant))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq)              # Reuses same anchor
```
"""
# Unified scalar anchor construction. _constant_anchor_query_impl handles
# Tg conversion internally, so no separate Real wrapper is needed.
@inline function _anchor_query(
        x::AbstractVector{T},
        xq,
        ::Val{:constant},
        wrap::Bool = false,
        searcher::P = DEFAULT_SEARCHER
    ) where {T, P <: Searcher}
    return _constant_anchor_query_impl(x, T(xq), wrap, _resolve_searcher_for_grid(x, searcher))
end

"""
    _anchor_query(x::AbstractVector{T}, xq::AbstractVector, ::Val{:constant}; wrap::Bool=false) -> Vector{_ConstantAnchoredQuery{T}}

Create anchored queries for multiple query points.

Internal API: No runtime grid validation. Caller must ensure `x` matches
the grid used for interpolant construction.

# Arguments
- `x`: Grid points (must match interpolant's grid)
- `xq`: Query points (any Real type, auto-promoted to T)
- `::Val{:constant}`: Type tag
- `wrap`: If true, wrap query points to domain [x[1], x[end]) before anchoring.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.75], Val(:constant))

itp1 = constant_interp(x, sin.(2π .* x))
itp2 = constant_interp(x, cos.(2π .* x))

vals1 = itp1(aq_vec)  # Batch evaluation
vals2 = itp2(aq_vec)  # Reuse same anchors
```
"""
function _anchor_query(
        x::AbstractVector{T},
        xq::AbstractVector{S},
        ::Val{:constant},
        wrap::Bool = false,
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {T, S <: Real, P <: Searcher}
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)
    output = Vector{_ConstantAnchoredQuery{T, T}}(undef, length(xq))

    @inbounds for k in eachindex(xq)
        output[k] = _constant_anchor_query_impl(x, T(xq[k]), wrap, searcher_resolved)
    end
    return output
end

"""
    _fill_anchors!(buffer, x, xq, ::Val{:constant}; wrap=false) -> buffer

Fill a caller-allocated buffer with anchored queries for constant interpolation.
In-place version of `_anchor_query(x, xq, Val(:constant))` — zero-allocation as long as
the caller reuses `buffer`. Writes `length(xq)` entries.

# Arguments
- `buffer::Vector{_ConstantAnchoredQuery{T,T}}`: Caller-allocated buffer (length >= length(xq))
- `x::AbstractVector{T}`: Grid points (must match interpolant's grid)
- `xq::AbstractVector`: Query points (any Real type, auto-promoted to T)
- `::Val{:constant}`: Type tag for constant interpolation
- `wrap::Bool=false`: If true, wrap query points to domain [x[1], x[end])

# Returns
The same `buffer` object, filled with anchored queries.
"""
@inline function _fill_anchors!(
        buffer::AbstractVector{_ConstantAnchoredQuery{T, T}},
        x::AbstractVector{T},
        xq::AbstractVector{S},
        ::Val{:constant},
        wrap::Bool = false,
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {T, S <: Real, P <: Searcher}
    @assert length(buffer) >= length(xq) "Buffer too small: $(length(buffer)) < $(length(xq))"
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)

    @inbounds for k in eachindex(xq)
        buffer[k] = _constant_anchor_query_impl(x, T(xq[k]), wrap, searcher_resolved)
    end
    return buffer
end

"""
    _constant_anchor_query_impl(x, xq, wrap, policy) -> _ConstantAnchoredQuery

Internal implementation of _anchor_query for constant interpolation.

# Arguments
- `x`: Grid points
- `xq`: Query point
- `wrap`: Whether to wrap query point to domain
- `policy`: Search policy for interval search (default: DEFAULT_SEARCHER)
"""
@inline function _constant_anchor_query_impl(
        x::AbstractVector{T},
        xq::Tq,
        wrap::Bool,
        policy::P = DEFAULT_SEARCHER
    ) where {T, Tq <: Real, P <: Searcher}
    loc = _anchor_loc(x, xq, wrap, policy)

    # Compute geometry (constant-internal concern)
    h = _get_h(x, loc.xL, loc.xR)
    dL = loc.xq - loc.xL
    # Promote xq to match dL type (Float64 query + Dual grid → dL is Dual)
    xq_promoted = oftype(dL, loc.xq)

    # `_anchor_loc` never returns a periodic-exclusive seam pair, so
    # `idxR = idxL + 1` here. Seam-pair anchors are constructed directly in
    # the exclusive periodic series helper via `_ConstantAnchoredQuery(...)`.
    return _ConstantAnchoredQuery(_IdxPair(loc.idx, loc.idx + 1), xq_promoted, loc.state, h, dL)
end

# ========================================
# ConstantInterpolant Evaluation with Anchor
# ========================================

"""
    (itp::ConstantInterpolant)(aq::_ConstantAnchoredQuery; deriv::DerivOp=EvalValue())

Evaluate constant interpolant at anchored query point. Ultra-fast path that
skips interval search.

# Arguments
- `aq`: Pre-computed anchored query from `_anchor_query`
- `deriv`: Derivative order (0=value, 1=first derivative, 2=second derivative)
           Note: derivatives are always 0 for constant interpolation.

# Example
```julia
itp = constant_interp(x, y)
aq = _anchor_query(x, 0.5, Val(:constant))
val = itp(aq)           # Value
```
"""
@inline function (itp::ConstantInterpolant{T})(aq::_ConstantAnchoredQuery{T}; deriv::DerivOp = EvalValue()) where {T}
    return _constant_eval_with_anchor(itp, aq, deriv)
end

@inline function _constant_eval_with_anchor(
        itp::ConstantInterpolant{T},
        aq::_ConstantAnchoredQuery{T},
        op::O
    ) where {T, O <: AbstractEvalOp}
    # Handle extrapolation based on mode and side
    return _constant_anchor_dispatch(itp, aq, op, itp.extrap)
end

# ========================================
# Shared Raw-Vector Anchor Eval
# ========================================
# Canonical evaluation functions that take raw y vector + explicit params.
# Used by both interpolant anchor dispatch AND series one-shot evaluation.

# Default case (extension, wrap, inbounds): kernel with right-boundary check
@inline function _constant_eval_at_anchor(
        y::AbstractVector, x_last, aq::_ConstantAnchoredQuery,
        op::AbstractEvalOp, side_param::AbstractSide, ::AbstractExtrap
    )
    aq.xq == x_last && return (op isa EvalValue ? (@inbounds y[end]) : 0 * first(y))
    @inbounds return _constant_kernel(op, y[aq.idxL], y[aq.idxR], aq.h, aq.dL, side_param)
end

# No extrapolation: throw DomainError if outside domain
@inline function _constant_eval_at_anchor(
        y::AbstractVector, x_last, aq::_ConstantAnchoredQuery,
        op::AbstractEvalOp, side_param::AbstractSide, ::NoExtrap
    )
    aq.state != IN_DOMAIN && throw(DomainError(aq.xq, "query point outside domain"))
    aq.xq == x_last && return (op isa EvalValue ? (@inbounds y[end]) : 0 * first(y))
    @inbounds return _constant_kernel(op, y[aq.idxL], y[aq.idxR], aq.h, aq.dL, side_param)
end

# Clamp/Fill extrapolation: boundary value if OOB
@inline function _constant_eval_at_anchor(
        y::AbstractVector, x_last, aq::_ConstantAnchoredQuery,
        op::AbstractEvalOp, side_param::AbstractSide, extrap::_ClampOrFill
    )
    if aq.state != IN_DOMAIN
        y_bnd = aq.state == OOB_LEFT ? first(y) : last(y)
        return _eval_extrapolation(op, y_bnd, extrap, aq.xq)
    end
    aq.xq == x_last && return (op isa EvalValue ? (@inbounds y[end]) : 0 * first(y))
    @inbounds return _constant_kernel(op, y[aq.idxL], y[aq.idxR], aq.h, aq.dL, side_param)
end

# ExtendExtrap → ClampExtrap for constant (zero slope → extend = clamp)
@inline function _constant_eval_at_anchor(
        y::AbstractVector, x_last, aq::_ConstantAnchoredQuery,
        op::AbstractEvalOp, side_param::AbstractSide, ::ExtendExtrap
    )
    return _constant_eval_at_anchor(y, x_last, aq, op, side_param, ClampExtrap())
end

# ========================================
# Interpolant Anchor Dispatch (thin wrappers)
# ========================================

# No extrapolation: enriched error message with domain bounds
@inline function _constant_anchor_dispatch(
        itp::ConstantInterpolant{T},
        aq::_ConstantAnchoredQuery{T},
        op::O,
        ::NoExtrap
    ) where {T, O <: AbstractEvalOp}
    if aq.state != IN_DOMAIN
        x_min, x_max = first(itp.x), last(itp.x)
        throw(DomainError(aq.xq, "query point outside domain [$x_min, $x_max]"))
    end
    if aq.xq == last(itp.x)
        return op isa EvalValue ? (@inbounds itp.y[end]) : zero(T)
    end
    @inbounds return _constant_kernel(op, itp.y[aq.idxL], itp.y[aq.idxR], aq.h, aq.dL, itp.side)
end

# Inside domain or extension mode: delegate to shared
@inline _constant_anchor_dispatch(
    itp::ConstantInterpolant, aq::_ConstantAnchoredQuery, op::AbstractEvalOp, ext::AbstractExtrap
) = _constant_eval_at_anchor(itp.y, last(itp.x), aq, op, itp.side, ext)

# Clamp/Fill: delegate to shared
@inline _constant_anchor_dispatch(
    itp::ConstantInterpolant, aq::_ConstantAnchoredQuery, op::AbstractEvalOp, ext::_ClampOrFill
) = _constant_eval_at_anchor(itp.y, last(itp.x), aq, op, itp.side, ext)

# ========================================
# Vector Evaluation with Anchors
# ========================================

"""
    (itp::ConstantInterpolant)(aq_vec::AbstractVector{_ConstantAnchoredQuery{T}}; deriv::DerivOp=EvalValue())

Evaluate constant interpolant at multiple anchored query points.
Returns newly allocated vector.
"""
function (itp::ConstantInterpolant{T})(
        aq_vec::AbstractVector{<:_ConstantAnchoredQuery{T}};
        deriv::DerivOp = EvalValue()
    ) where {T}
    output = Vector{T}(undef, length(aq_vec))
    @inbounds for i in eachindex(aq_vec)
        output[i] = _constant_eval_with_anchor(itp, aq_vec[i], deriv)
    end
    return output
end

"""
    (itp::ConstantInterpolant)(output::AbstractVector, aq_vec::AbstractVector{<:_ConstantAnchoredQuery{T}}; deriv::DerivOp=EvalValue())

In-place evaluation at multiple anchored query points. Zero allocation.
"""
function (itp::ConstantInterpolant{T})(
        output::AbstractVector,
        aq_vec::AbstractVector{<:_ConstantAnchoredQuery{T}};
        deriv::DerivOp = EvalValue()
    ) where {T}
    @assert length(output) == length(aq_vec) "output length must match aq_vec length"
    @inbounds for i in eachindex(aq_vec)
        output[i] = _constant_eval_with_anchor(itp, aq_vec[i], deriv)
    end
    return output
end
