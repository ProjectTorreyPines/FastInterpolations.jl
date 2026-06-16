# ============================================================================
# Unified API — 1D bare-vector entry points
# ============================================================================
# `interp`/`interp!` for genuine 1D data, where the grid and values are bare
# `AbstractVector`s rather than the ND `(grids, data)` tuple form. These route
# directly to the dedicated 1D functions (`cubic_interp`, `linear_interp`, …):
#
#     interp(x, y; method=CubicInterp())            === cubic_interp(x, y)
#     interp(x, y, q; method=CubicInterp())         === cubic_interp(x, y, q)
#     interp(x, y, qs; method=CubicInterp())        === cubic_interp(x, y, qs)
#     interp!(out, x, y, qs; method=CubicInterp())  === cubic_interp!(out, x, y, qs)
#
# Routing is a single per-method trait, `_interp1d_route`, mirroring the
# `_build_hetero_axis_package` idiom: each method maps to its dedicated
# (allocating, in-place) function pair plus the method-specific keyword options
# (`bc` / `side` / `tension`). The four public forms below consume that trait
# generically, so the kwarg wiring for each form is written exactly once and a
# new method family is a one-line addition.
#
# Zero-cost: `_interp1d_route(method)` dispatches on the concrete `method` type,
# so the returned tuple is fully concrete (function singletons carry no data;
# the options NamedTuple holds the method's fields). The compiler scalar-replaces
# the tuple and constant-folds the `opts...` keyword splat — the wrapper emits
# the same code, with the same allocation profile, as calling the dedicated
# function by hand.
#
# Non-ambiguous with the ND forms (hetero_oneshot.jl / hetero_interpolant.jl):
# those require `NTuple{N,AbstractVector}` as the first positional argument,
# while these take a bare `AbstractVector` — dispatch separates them cleanly.
#
# Only kwargs common to every dedicated 1D function are exposed. `coeffs`
# (Hermite-family only) and batch-form `hint` (Hermite-family only) are left to
# each dedicated function's own defaults rather than forwarded conditionally.

# ----------------------------------------------------------------------------
# Per-method routing trait: (allocating fn, in-place fn, method-specific kwargs)
# ----------------------------------------------------------------------------
@inline _interp1d_route(m::CubicInterp)     = (cubic_interp,     cubic_interp!,     (; bc = m.bc))
@inline _interp1d_route(m::LinearInterp)    = (linear_interp,    linear_interp!,    (; bc = m.bc))
@inline _interp1d_route(m::QuadraticInterp) = (quadratic_interp, quadratic_interp!, (; bc = m.bc))
@inline _interp1d_route(m::ConstantInterp)  = (constant_interp,  constant_interp!,  (; side = m.side, bc = m.bc))
@inline _interp1d_route(m::PchipInterp)     = (pchip_interp,     pchip_interp!,     (; bc = m.bc))
@inline _interp1d_route(m::CardinalInterp)  = (cardinal_interp,  cardinal_interp!,  (; bc = m.bc, tension = m.tension))
@inline _interp1d_route(m::AkimaInterp)     = (akima_interp,     akima_interp!,     (; bc = m.bc))

# ----------------------------------------------------------------------------
# Public forms — each consumes the routing trait generically.
# ----------------------------------------------------------------------------

"""
    interp(x::AbstractVector, y::AbstractVector; method, extrap=NoExtrap(), search=AutoSearch())

Build a 1D interpolant. Equivalent to calling the dedicated 1D constructor for
`method` directly (e.g. `method=CubicInterp()` → `cubic_interp(x, y)`), with the
method's boundary condition / side / tension forwarded.

```julia
itp = interp(x, y; method = CubicInterp())              # === cubic_interp(x, y)
itp = interp(x, y; method = LinearInterp(bc = PeriodicBC()))
itp = interp(x, y; method = CardinalInterp(tension = 0.5))
```
"""
@inline function interp(
        x::AbstractVector,
        y::AbstractVector;
        method::AbstractInterpMethod,
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
    )
    fn, _, opts = _interp1d_route(method)
    return fn(x, y; opts..., extrap = extrap, search = search)
end

"""
    interp(x::AbstractVector, y::AbstractVector, q::Real; method, deriv=EvalValue(), extrap=NoExtrap(), search=AutoSearch(), hint=nothing)

One-shot 1D interpolation at a single point. Equivalent to the dedicated 1D
one-shot call (e.g. `cubic_interp(x, y, q)`), with the method's options forwarded.
"""
@inline function interp(
        x::AbstractVector,
        y::AbstractVector,
        q::Real;
        method::AbstractInterpMethod,
        deriv::DerivOp = EvalValue(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    )
    fn, _, opts = _interp1d_route(method)
    return fn(x, y, q; opts..., deriv = deriv, extrap = extrap, search = search, hint = hint)
end

"""
    interp(x::AbstractVector, y::AbstractVector, queries::AbstractVector{<:Real}; method, deriv=EvalValue(), extrap=NoExtrap(), search=AutoSearch())

Allocating one-shot 1D interpolation at multiple points. Returns a `Vector`.
Equivalent to the dedicated 1D batch call (e.g. `cubic_interp(x, y, queries)`).
"""
@inline function interp(
        x::AbstractVector,
        y::AbstractVector,
        queries::AbstractVector{<:Real};
        method::AbstractInterpMethod,
        deriv::DerivOp = EvalValue(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
    )
    fn, _, opts = _interp1d_route(method)
    return fn(x, y, queries; opts..., deriv = deriv, extrap = extrap, search = search)
end

"""
    interp!(output::AbstractVector, x::AbstractVector, y::AbstractVector, queries::AbstractVector{<:Real}; method, deriv=EvalValue(), extrap=NoExtrap(), search=AutoSearch())

In-place one-shot 1D interpolation at multiple points. Writes into `output`.
Equivalent to the dedicated 1D in-place call (e.g. `cubic_interp!(output, x, y, queries)`).
"""
@inline function interp!(
        output::AbstractVector,
        x::AbstractVector,
        y::AbstractVector,
        queries::AbstractVector{<:Real};
        method::AbstractInterpMethod,
        deriv::DerivOp = EvalValue(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
    )
    _, fn!, opts = _interp1d_route(method)
    return fn!(output, x, y, queries; opts..., deriv = deriv, extrap = extrap, search = search)
end
