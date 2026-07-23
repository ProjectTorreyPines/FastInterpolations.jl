# ========================================
# Linear Interpolation Oneshot API
# ========================================
# Zero-allocation linear interpolation functions.
# Type definitions in linear_types.jl.
# Callable methods (2-arg form) in linear_interpolant.jl.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HOT PATH - OPTIMIZED CORE                         ║
# ║                  All arguments have same FT<:AbstractFloat                ║
# ║                      Zero type conversion overhead                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Vector interpolation (in-place, zero-allocation)
# ========================================

"""
    linear_interp!(output, x, y, x_targets; bc=NoBC(), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

Zero-allocation linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: Search algorithm determined by `search` parameter

# Arguments
- `output`: Pre-allocated output vector (must be floating-point type)
- `bc::AbstractBC`: Boundary condition. Default `NoBC()` (no BC). Pass
  `PeriodicBC(endpoint=:inclusive)` or `PeriodicBC(endpoint=:exclusive, period=L)`
  for periodic interpolation (extrap is forced to `WrapExtrap()` in that case).
- `extrap::AbstractExtrap`: `NoExtrap()` (default, throws DomainError), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `deriv::DerivOp`: Derivative order (`EvalValue()` default, `DerivOp(1)` first derivative, `DerivOp(2)` second derivative)
- `search::AbstractSearchPolicy`: Search algorithm for interval finding
  - `BinarySearch()` (default): O(log n) binary search, stateless
  - `LinearBinarySearch(linear_window=0)`: O(1) if hint valid, O(log n) fallback
  - `LinearBinarySearch(linear_window=8)`: Linear search within window, then binary fallback

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
out = Vector{Float64}(undef, 2)
linear_interp!(out, rho, y, [0.55, 0.77])  # throws error if outside domain
linear_interp!(out, rho, y, [-0.1, 1.2]; extrap=ExtendExtrap())  # linear extrapolation

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
output = zeros(1000)
linear_interp!(output, x_vec, y_vec, sorted_queries; search=LinearBinarySearch(linear_window=8))
```

# Implementation Note
- Optimized core works with `AbstractFloat` types (calls optimized scalar version)
- Integer/Real inputs automatically promoted via wrapper methods
"""
function linear_interp! end

# Unified in-place entry point. Handles promotion internally via _promote_itp_inputs,
# so no separate Real/Mixed-type wrapper is needed (same pattern as the scalar API).
function linear_interp!(
        output::AbstractArray,
        x::AbstractVector,
        y::AbstractVector,
        x_targets::AbstractArray;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    )
    @assert length(y) == length(x) "x and y must have same length"
    _check_query_output_size(output, x_targets)

    # Surface-level BC-aware resolvers (zero-alloc reference wrapping):
    #   `_resolve_axis(x, bc)` shapes the axis (Range→`_CachedRange`, Vector→passthrough,
    #     Vector+`:exclusive`→`_ExclusivePeriodicAxis`, Range+`:exclusive`→length-(n+1)
    #     `_CachedRange`).
    #   `_resolve_data(y, bc)` shapes the data (passthrough, or `_ExclusivePeriodicData`
    #     for `:exclusive`).
    # BC info lives in the axis type after resolution → searcher uses `NoBC()`,
    # the seam is handled by the wrapper (or by the naturally-extended Range).
    # Value-matched Tg keeps the loop interior bit-consistent with the scalar path.
    x_eff = _resolve_axis(x, bc, _promote_grid_float(eltype(x), eltype(y)))
    y_eff = _resolve_data(y, bc)
    extrap_eff = _resolve_extrap(extrap, bc, x_eff, y_eff)
    searcher = _resolve_search(x_eff, x_targets, search, hint)
    return _linear_interp_loop!(output, x_eff, y_eff, x_targets, extrap_eff, deriv, searcher)
end

# Internal loop with function-barrier pattern. `_check_domain` returns
# `Union{InBounds, E}` for Clamp/Fill/Wrap (in-bounds promotion); passing
# through a function-barrier call lets Julia's union-splitting specialize
# the inner loop per concrete `ev` (no per-iteration union dispatch). For
# `NoExtrap` paths the return type is already `InBounds`, so this is a
# no-op. Supports mixed types: Tg for grid, Tv for values.
@inline function _linear_interp_loop!(
        output::AbstractArray,
        x::AbstractVector{Tg},
        y::AbstractVector,
        x_targets::AbstractArray,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, O <: AbstractEvalOp, S <: Searcher}
    extrap_eff = _check_domain(x, x_targets, extrap)
    return _linear_interp_loop_inner!(output, x, y, x_targets, extrap_eff, op, searcher)
end

@inline function _linear_interp_loop_inner!(
        output::AbstractArray,
        x::AbstractVector{Tg},
        y::AbstractVector,
        x_targets::AbstractArray,
        extrap::E,
        op::O,
        searcher::S
    ) where {Tg, E <: AbstractExtrap, O <: AbstractEvalOp, S <: Searcher}
    @inbounds for i in eachindex(x_targets, output)
        output[i] = _linear_eval_at_point(x, y, x_targets[i], extrap, op, searcher)
    end
    return output
end

# AbstractRange WrapExtrap specialization removed — identical logic to the
# AbstractVector version above.  _CachedRange <: AbstractRange <: AbstractVector,
# so the AbstractVector dispatch handles all Range inputs.

# NOTE: the former AbstractRange{Tg}-specific `linear_interp!` overload (which resolved
# ambiguity with the old Real wrappers) has been removed. The unified `linear_interp!`
# above handles promotion for all grid types (Float, Int, Dual, Range) via
# `_promote_itp_inputs`, eliminating the need for type-specific overloads.

# ========================================
# Scalar interpolation (zero-allocation)
# ========================================

"""
    linear_interp(x, y, xq::Number; bc=NoBC(), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch()) -> AbstractFloat

Zero-allocation scalar linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: Search algorithm determined by `search` parameter

# Arguments
- `xq::Number`: Single interpolation query point
- `bc::AbstractBC`: Boundary condition. Default `NoBC()` (no BC). Pass
  `PeriodicBC(endpoint=:inclusive)` or `PeriodicBC(endpoint=:exclusive, period=L)`
  for periodic interpolation (extrap is forced to `WrapExtrap()` in that case).
- `extrap::AbstractExtrap`: `NoExtrap()` (default, throws DomainError), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `deriv::DerivOp`: Derivative order (`EvalValue()` default, `DerivOp(1)` first derivative)
- `search::AbstractSearchPolicy`: Search algorithm for interval finding
  - `BinarySearch()` (default): O(log n) binary search, stateless
  - `LinearBinarySearch(linear_window=0)`: O(1) if hint valid, O(log n) fallback
  - `LinearBinarySearch(linear_window=8)`: Linear search within window, then binary fallback

# Returns
- Always returns a floating-point type (Integer inputs auto-promoted to Float)

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
value = linear_interp(rho, y, 0.55)  # Returns Float64, zero allocation
value = linear_interp(rho, y, 1.5; extrap=WrapExtrap())  # wraps to domain

# Integer inputs auto-promoted to Float
x_int = 0:10
y_int = x_int.^2
value = linear_interp(x_int, y_int, 5.5)  # Returns Float64 (not Int)
```

# Implementation Note
- Optimized core works with `AbstractFloat` types only (zero conversion overhead)
- Integer/Real inputs automatically promoted to Float via wrapper methods
- Uses Val dispatch for extrapolation to eliminate runtime branches
"""

# ========================================
# Internal evaluation with op parameter
# ========================================
#
# TYPE PARAMETERS:
# - Tg: Grid type (AbstractFloat) - for x coordinates
# - Tv: Value type (unconstrained) - for y values
# - Tq: Query type - typically Tg but can be left unconstrained for AD support
#
# These functions form the core evaluation logic that supports:
# - Any value type Tv via duck typing (+, -, scalar *)
# - AD support (xq can be Dual{Tg})

"""
    _linear_eval_at_point(x, y, xq, extrap, op, searcher)

Core linear interpolation evaluation using kernel function and search policy.
Supports value (EvalValue), first derivative (EvalDeriv1), and second derivative (EvalDeriv2).

# Type Parameters
- `Tg`: Grid type (AbstractFloat)
- `Tv`: Value type (unconstrained)
- `Tq`: Query type (can be Tg or Dual{Tg} for AD support)

# AD Support
For ForwardDiff compatibility, `xq` can be a Dual type:
- Search uses `_extract_primal(xq)` to find interval index
- Interpolation arithmetic uses original `xq` to propagate derivatives
"""
# ========================================
# Core eval: extrap dispatch → search → kernel (no intermediate layers)
# ========================================
# Cell geometry enters at the value-matched width: `inv_h` converts to the
# compile-time `typeof(inv(oneunit(Tw)))` (the `_CachedRange` Tinv idiom) — a no-op
# for float/duck/Unitful axes, and the value-width float for an Int axis whose
# `inv(h::Int)` would otherwise widen everything to Float64. α = (q−L)·inv_h keeps
# query blood (Dual partials) and mirrors the persistent + ND `_locate_cell` form.

# Core in-bounds path: search + kernel. All non-InBounds extrap overloads
# delegate here after their preprocessing — see cubic_eval.jl for the same
# `InBounds = core fast path` pattern. WrapExtrap uses a periodic-seam-aware
# core overload below (`_raw(y)`) to avoid the data wrapper's cyclic branch.
@inline function _linear_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        e::InBounds,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    # Compile-time cell-geometry type: reciprocal spacing at the value-matched width.
    Tinv = _promote_eltype(_inv_op, _promote_grid_float(Tg, Tv))
    xq = _resolve_grididx(xq, x)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq, e)
    inv_h = convert(Tinv, _get_inv_h(x, idx))
    α = _alpha_of(xq, xL, inv_h)
    @inbounds return _linear_kernel(op, y[idx], y[idx_R], inv_h, α)
end

# NoExtrap / ExtendExtrap / others matching AbstractExtrap: domain check (NoExtrap throws
# on OOB; ExtendExtrap is a no-op and may arrive OOB). Runs the standard two-sided-clamp
# search + kernel HERE — it must NOT delegate to the lean `::InBounds` core, whose one-sided
# clamp would return idx ≤ 0 on an OOB-left ExtendExtrap query. Passing `extrap` routes to
# the clamping `search_interval` overload; the boundary cell (idx ∈ [1, n-1]) lets `α`
# extrapolate past the ends.
@inline function _linear_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq = _resolve_grididx(xq, x)
    # NoExtrap → InBounds for the search once the domain check passes (lean search);
    # ExtendExtrap passes through and keeps the two-sided-clamp search (it may arrive OOB).
    Tinv = _promote_eltype(_inv_op, _promote_grid_float(Tg, Tv))
    extrap_eff = _check_domain(x, xq, extrap)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq, extrap_eff)
    inv_h = convert(Tinv, _get_inv_h(x, idx))
    α = _alpha_of(xq, xL, inv_h)
    @inbounds return _linear_kernel(op, y[idx], y[idx_R], inv_h, α)
end

# ClampExtrap / FillExtrap (all ops): boundary check → extrap value or delegate.
# An OOB query under flat extrapolation freezes at the boundary node, so the result IS the
# boundary sample (value) or zero (derivative) — short-circuit, skipping search + kernel. The
# saving scales with the OOB fraction (each OOB query is ~half a full eval), so this dominates
# an always-clamp-then-evaluate path on extrapolation-heavy batches.
@inline function _linear_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        extrap::_ClampOrFill,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    # Promote to the coordinate type Tc so the OOB extrap VALUE carries the grid
    # carrier (Dual grid → Dual fill via _eval_extrapolation), matching the
    # in-domain kernel. Identity on Float64; Int grids stay Int. Classify on primal.
    xq = _promote_coord(_resolve_grididx(xq, x), eltype(x))
    xq_primal = _extract_primal(xq)
    st = _oob_state(x, xq_primal)
    st == OOB_LEFT && return _eval_extrapolation(op, first(y), extrap, xq)
    st == OOB_RIGHT && return _eval_extrapolation(op, last(y), extrap, xq)
    return _linear_eval_at_point(x, y, xq, InBounds(), op, searcher)
end

# WrapExtrap: wrap query to domain → search + kernel using `_raw(y)`.
# Pass original xq (may be Dual) to _wrap_to_domain to preserve AD derivatives.
# `search_interval` already resolves seam wrap-around (idx_R = 1 for the seam
# cell), so `yi[idx_R]` is safe via the raw inner — no need for the data
# wrapper's per-call cyclic branch. Cannot delegate to the generic InBounds
# core because that uses `y[idx]` (data wrapper); inline the kernel here.
@inline function _linear_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        ::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    Tinv = _promote_eltype(_inv_op, _promote_grid_float(Tg, Tv))
    xq_wrapped = _wrap_to_domain(_resolve_grididx(xq, x), x)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq_wrapped)
    inv_h = convert(Tinv, _get_inv_h(x, idx))
    α = _alpha_of(xq_wrapped, xL, inv_h)
    yi = _raw(y)
    @inbounds return _linear_kernel(op, yi[idx], yi[idx_R], inv_h, α)
end

# Public scalar one-shot API.
# Zero-alloc: _resolve_axis returns Vector as-is, Range → _CachedRange (stack).
# Kernel arithmetic auto-promotes Int×Float via _get_h float() wrappers.
@inline function linear_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Number, Tv, Tq <: Number}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    # Same surface-level resolution as the in-place vector form. Zero-alloc:
    # `_resolve_axis` returns either `x` (Vector passthrough), a stack-allocated
    # `_CachedRange`, or an `_ExclusivePeriodicAxis` reference wrapper.
    # `_resolve_data` is reference-only. The Tg is value-matched: an Int/OneTo
    # grid beside Float32 data floats to Float32, not the blind Float64.
    x_eff = _resolve_axis(x, bc, _promote_grid_float(Tg, Tv))
    y_eff = _resolve_data(y, bc)
    extrap_eff = _resolve_extrap(extrap, bc, x_eff, y_eff)
    searcher = _resolve_search(x_eff, xq, search, hint)
    return _linear_eval_at_point(x_eff, y_eff, xq, extrap_eff, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        GENERIC WRAPPERS - CONVENIENCE                     ║
# ║              Auto-promote Real types to Float (type conversion)           ║
# ║                     Integer inputs → Float64 outputs                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Vector interpolation - Allocating hot path
# Unified allocating vector one-shot. Calls the unified linear_interp! which
# handles promotion internally. Output type includes Tg for duck grids.

function linear_interp(
        x::AbstractVector,
        y::AbstractVector,
        x_targets::AbstractArray;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    )
    Tg = _promote_grid_float(eltype(x), eltype(y))
    # Deriv-aware: an nth derivative lives in value/gridᴺ space, so scale the
    # value eltype through the same `_deriv_eltype` fold as the persistent path
    # (identity for `EvalValue`, so the value case is unchanged).
    T_out = _deriv_eltype(_promote_eltype(_interp_op, Tg, eltype(y), eltype(x_targets)), Tg, deriv)
    output = _alloc_query_output(T_out, x_targets)
    linear_interp!(output, x, y, x_targets; bc, extrap, deriv, search, hint)
    return output
end

# NOTE: the former in-place and allocating vector wrappers (`linear_interp!` and
# `linear_interp` with `{Tg<:Real, Tv, Tq<:Real}`) have been removed. Their
# `_promote_itp_inputs` calls are now performed by the unified `linear_interp!`
# above and the typed `linear_interp(x, y, x_targets)` below. This prevents
# dispatch ambiguity and infinite recursion on duck grids.
