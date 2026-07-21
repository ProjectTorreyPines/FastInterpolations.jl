# Internal utility functions for FastInterpolations.jl

# ── @noinline throw helpers (keep cold error paths out of hot code) ──

@noinline _throw_length_mismatch(na::Int, nb::Int, a::String = "x", b::String = "y") =
    throw(ArgumentError("$a and $b must have same length, got $na and $nb"))

@noinline _throw_grid_too_small(n::Int) =
    throw(ArgumentError("x must have at least 2 elements, got $n"))

# ── Branchless clamp ──
# Float: `min(max(x, lo), hi)` lowers to 2 ops (`fmax; fmin`); `Base.clamp(x, lo, hi)`
# lowers to 4 (`fcmp; fcsel; fcmp; fcsel`) — also branchless (conditional-select, NOT
# branches), but a longer dependency chain. So min/max is a constant ~2-op win on the
# critical path, independent of query distribution: there is no branch to mispredict, so
# the margin is flat across OOB ratios (measured on arm64, incl. 0% OOB). Int: both forms
# emit identical branchless code (`cmp; csinc; cmp; csel`), so the search-index clamps are
# unaffected — the helper is used there only to keep one consistent call site.
@inline _clamp(x, lo, hi) = min(max(x, lo), hi)
# A GridIdx is in-domain by construction (bounds-checked at resolution), so clamping it to
# the grid domain is a no-op: skip the `min/max` entirely and pass it through. This also
# keeps the value a GridIdx so the downstream `search_interval` takes the index
# short-circuit instead of promoting it to a plain, searched coordinate.
@inline _clamp(xq::GridIdx, ::Any, ::Any) = xq

# ========================================
# Interval Search (IN search.jl)
# ========================================
# Interval search functions are defined in src/core/search.jl:
#   - _search_direct: O(1) direct calculation for uniform grids (AbstractRange)
#   - _search_binary: O(log n) binary search for non-uniform grids (AbstractVector)
#   - _search_interval: dispatcher that routes to the appropriate implementation

# ========================================
# Type Conversion Helpers
# ========================================

# _to_float for Range types (→ _CachedRange) is defined in cached_range.jl.
# _to_float_adding_endpoint is also defined in cached_range.jl.

"""
    _to_float(x::AbstractVector{T}, ::Type{T}) where {T}

Identity conversion — return as-is when element type already matches target type.
Handles both standard Float types and duck types (e.g. `Vector{Dual{Float64}}`
with target `Dual{Float64}`). Zero-allocation in all cases.
"""
_to_float(x::AbstractVector{T}, ::Type{T}) where {T} = x

"""
    _to_float(x::AbstractVector, ::Type{T}) where {T}

Convert a Vector to target type (element-wise broadcast). For duck types
(e.g. `Dual{Int}→Dual{Float64}`), `T.(x)` dispatches to ForwardDiff's `convert`.
"""
_to_float(x::AbstractVector, ::Type{T}) where {T} = T.(x)

# ========================================
# Value Type Helpers (for Complex support)
# ========================================

"""
    _real_eltype(::Type{T}) where {T<:Real} -> Type

Extract the real base type from an element type.
For Real types, returns the type itself.
For Complex{T}, returns T.

This is TYPE-BASED (works with eltype(y) in wrappers).
"""
@inline _real_eltype(::Type{T}) where {T <: Real} = T
@inline _real_eltype(::Type{Complex{T}}) where {T <: Real} = T
# Duck-typing fallback: custom types return themselves (no float base extraction)
@inline _real_eltype(::Type{T}) where {T} = T

"""
    _promote_grid_float(::Type{Tg}, ::Type{Tv}) -> Type{<:AbstractFloat}

Compute the grid float type, optionally widened by value precision.

- **Promotable values** (`_PromotableValue`): grid widens to accommodate value precision.
  Example: `Float32` grid + `Float64` values → `Float64` grid.
  This prevents per-element conversion overhead in hot evaluation paths.
- **Duck types** (Dual, Measurement, etc.): grid uses only its own type.
  Value type must NOT contaminate the grid (e.g., grid coordinates should never
  carry derivative partials from `ForwardDiff.Dual`).

# Examples
```julia
_promote_grid_float(Int, Float64)    # → Float64 (standard widening)
_promote_grid_float(Float32, Int)    # → Float32 (Int doesn't widen Float32)
_promote_grid_float(Float32, Float64)# → Float64 (value precision wins)
_promote_grid_float(Float64, Dual)   # → Float64 (duck: grid ignores Dual)
_promote_grid_float(Int, Dual)       # → Float64 (duck: float(Int) only)
```
"""
@inline function _promote_grid_float(::Type{Tg}, ::Type{Tv}) where {Tg, Tv}
    if Tv <: _PromotableValue
        return float(promote_type(Tg, _real_eltype(Tv)))
    else
        return float(Tg)
    end
end

# Build-entry guard: a grid axis must be orderable (search/sort call `isless`).
# `Real` fast path folds to a no-op; the generic arm runs one `hasmethod` at
# build time and turns e.g. a Complex grid into an actionable error instead of a
# deep search-internal `MethodError`. Necessary, not sufficient (an ordered type
# may still lack grid arithmetic) — see the duck-grid contract docs.
@inline _check_grid_orderable(::Type{<:Real}) = nothing
@noinline function _check_grid_orderable(::Type{Tg}) where {Tg}
    hasmethod(isless, Tuple{Tg, Tg}) || throw(
        ArgumentError(
            "grid axis eltype $(Tg) does not support ordering (`isless`) — " *
                "interpolation grids must be sortable (e.g. Complex is not a valid grid eltype)"
        )
    )
    return nothing
end

# Solver-family ND builds store mixed derivative-mask orders in ONE homogeneous
# array — unit-heterogeneous by construction (f vs ∂²f differ even on same-unit
# axes). Reject unit-carrying grids with an actionable error; `Real` folds away.
@inline _check_nd_solver_grid(::Type{<:Real}) = nothing
@noinline _check_nd_solver_grid(::Type{Tg}) where {Tg} = throw(
    ArgumentError(
        "PreCompute ND coefficient builds (Cubic/Quadratic/Hermite axes) do not " *
            "support unit-carrying grids yet (grid eltype $(Tg)) — the nodal-" *
            "derivative store mixes derivative orders of different dimensions. " *
            "Use LinearInterp/ConstantInterp ND, integrate per-fiber 1-D, or strip units."
    )
)

"""
    _value_type(::Type{Ty}, ::Type{Tg}) -> Type

Determine the output value type from y element type and grid type.
- Standard numerics (Integer, AbstractFloat, Rational, Complex) → Tg or Complex{Tg}
- Duck types (Dual, Measurement, etc.) → preserved as-is
"""
@inline _value_type(::Type{T}, ::Type{Tg}) where {T <: _PromotableValue, Tg <: AbstractFloat} = Tg
@inline _value_type(::Type{Complex{T}}, ::Type{Tg}) where {T <: Real, Tg <: AbstractFloat} = Complex{Tg}
# Duck-typing fallback for Tv: custom value types preserved as-is
@inline _value_type(::Type{T}, ::Type{Tg}) where {T, Tg <: AbstractFloat} = T
# Duck-typing fallback for Tg: when grid is duck-typed (Dual, Measurement, etc.),
# values are not promoted to grid type (no grid-parameter partials in y).
@inline _value_type(::Type{T}, ::Type{Tg}) where {T, Tg} = T

# Inference probe for `_promote_eltype` duck fallback. Standard kernels
# (Linear/Cubic/Quadratic/Hermite) produce `Tv + α·Tv` shapes; Constant's
# `Tv * one(Tq)` lives in the same promotion space.
@inline _kernel_shape_op(yv, q) = yv * q + yv

"""
    _promote_eltype(::Type{Tv}, types...) -> Type

Generic output-eltype probe via the universal arithmetic kernel shape
`y*q + y` (`_kernel_shape_op`). Currently used by:

- Internal coefficient eltype (Cubic `Tz`, Quadratic `Tc`).
- Adjoint allocators (`adjoint_protocol.jl`).
- Hetero ND legacy paths and a few series callsites.

Concrete `promote_type` gets Int→Float upgrade (arithmetic kernels divide
— Int chains widen naturally); duck carriers (e.g. `SVector × Dual`) fall
through to `Base.promote_op` on `_kernel_shape_op`, with final fallback
to `Tv` if the op is undefined.

For method-aware output-buffer sizing (Linear/Cubic/Quadratic/Constant/
Hermite), prefer the kernel-op overload below — it predicts the method's
exact kernel return type via `Base.promote_op`.
"""
@inline function _promote_eltype(::Type{Tv}, types::Type...) where {Tv}
    Tr = promote_type(Tv, types...)
    if isconcretetype(Tr)
        return (Tr <: _PromotableValue && !(Tr <: AbstractFloat)) ? float(Tr) : Tr
    end
    Tq = length(types) == 0 ? Tv : promote_type(types...)
    Top = Base.promote_op(_kernel_shape_op, Tv, Tq)
    (Top === Union{} || Top === Any) && return Tv
    return Top
end

"""
    _promote_eltype(kernel_op, ::Type{Tv}, types...) -> Type

Method-aware output element type via `Base.promote_op` on the method's own
kernel shape. Lets Julia inference predict the kernel's exact return type
— no hand-coded Float upgrade, no `_PromotableValue` enumeration. Use this
overload from a method that declares its kernel shape (e.g., Constant's
`_select_op(xL, yv, xq) = yv * one(xq - xL)`).
"""
@inline function _promote_eltype(kernel_op::F, ::Type{Tv}, types::Type...) where {F, Tv}
    Top = Base.promote_op(kernel_op, Tv, types...)
    (Top === Union{} || Top === Any) && return Tv
    return Top
end

# Type-witness OPS for `_promote_eltype` — small expressions whose return type (via
# `Base.promote_op`) equals the real computation's element type. They are NOT the real
# kernels; they exist only to drive inference, capturing the spacing reciprocal that
# floats Int. Args named/ordered `(grid, value[, query])` → `_promote_eltype(op, Tg, Tv[, Tq])`.
# (Constant's selection op `_select_op(xL, yv, xq) = yv * one(xq - xL)` lives in
# constant_interpolant.jl — no division, so it keeps Int.)
#
# `_interp_op` (3-arg): interpolation eval — value weighted by the query offset `dL/h`
# → the OUTPUT eltype (query-dependent). Models `yv + yv*(dL*inv_h)`; `dL/h ≡ dL*inv_h`
# in type when `h` is the (floated) grid type, which it always is at eltype sites.
@inline _interp_op(h::Tg, yv::Tv, dL::Tq) where {Tg, Tv, Tq} = yv + yv * (dL / h)

# `_coeff_op` (2-arg): FIRST-ORDER divided difference `Δy/h`, accumulated by the
# solve → the order-1 COEFFICIENT eltype (slopes: hermite/pchip/akima/cardinal `dy`,
# secants). Modeled as `yv*inv(h) + yv*inv(h)`: `* inv(h)` (NOT `/ h`) mirrors the
# real solve multiplying a precomputed `inv_h` — duck-safe (`*(Tv, Tg)` not `/`) and
# floats Int grids (`inv(Int)::Float64`); the `+` mirrors the solve summing scaled
# values. Dimensionally HOMOGENEOUS (every term `Y/X`) so unit-carrying grids infer
# a concrete type. QUERY-FREE: coefficients are solved before any query.
@inline _coeff_op(h::Tg, yv::Tv) where {Tg, Tv} = yv * inv(h) + yv * inv(h)

# `_coeff_op2` (2-arg): SECOND-ORDER coefficient witness (`Y/X²` — cubic spline `z`;
# quadratic curvature `a` when its storage splits). Same duck/float/homogeneity
# contract as `_coeff_op`, one more `inv(h)` power.
@inline _coeff_op2(h::Tg, yv::Tv) where {Tg, Tv} =
    yv * (inv(h) * inv(h)) + yv * (inv(h) * inv(h))

# `_inv_op` (1-arg): reciprocal-spacing eltype — `inv(h)` for an axis already at the
# value-matched width (compose: `_promote_eltype(_inv_op, _promote_grid_float(Tg, Tv))`).
# Floats Int (`inv(Int)::Float64` never survives a narrow value space upstream), keeps
# Unitful inverse units and duck carriers.
@inline _inv_op(h) = inv(h)

# Grid-precision DIMENSIONLESS constant `1/n` (kernel coefficients like 1/24):
# `Tg(n)` would demand a unit for unit-carrying grids — `one(Tg)` keeps the
# float width while staying dimensionless. Real arm is codegen-identical.
@inline _inv_const(::Type{Tg}, n::Int) where {Tg <: Real} = inv(Tg(n))
@inline _inv_const(::Type{Tg}, n::Int) where {Tg} = inv(one(Tg) * n)

# `_integrate_op` (3-arg): the definite-integral element type — value × spacing.
# ∫ ≈ Σ yᵢ·hᵢ is dimensionally distinct from the eval witnesses (which weight the value
# by the dimensionless offset `dL/h`). `span` is the integration length (`b2 - xL` for a
# partial cell, the cell width `h` for a full cell). The `oneunit(h)*inv(h)` factor is
# load-bearing: dimensionless by construction (units cancel → every term is `Y·X`,
# so unit-carrying grids infer a concrete type), it floats Int grids
# (`oneunit(Int)*inv(Int)::Float64`) and lifts Dual (`inv(Dual)::Dual`), so Tout is
# correct for all-Int integrate (the kernels divide) and for AD-wrt-grid/bounds.
# Duck-safe: `yv` sees only `*`/`+`, a subset of what the integral kernels require.
@inline _integrate_op(h::Tg, yv::Tv, span::Ts) where {Tg, Tv, Ts} =
    yv * span + yv * (span * (oneunit(h) * inv(h)))

# ── Wrap-free field arithmetic at unavoidable difference/sum sites ──
# `Tc` is the method's coefficient/output field type (e.g. `eltype` of a coeff
# array, or `_promote_eltype(_coeff_op, Tg, Tv)`) — never a forced `Float`.
# Fast path (operand already `Tc`): plain `a ± b`, zero overhead, identical for
# floats and duck/AD types. Promote path (narrow operand, e.g. UInt8/N0f8): widen
# into the field BEFORE the `±` so modular/overflow wrap cannot occur.
# Result is PINNED to `Tc` via `convert` — don't swap to `promote(a, b)` (cf. search.jl
# `_lt`/`_le`, which `promote` for a Bool): a promoted pair can be narrower than `Tc`
# and re-introduce the wrap.
@inline _fielddiff(::Type{Tc}, a::Tc, b::Tc) where {Tc} = a - b
@inline _fielddiff(::Type{Tc}, a, b) where {Tc} = convert(Tc, a) - convert(Tc, b)
@inline _fieldsum(::Type{Tc}, a::Tc, b::Tc) where {Tc} = a + b
@inline _fieldsum(::Type{Tc}, a, b) where {Tc} = convert(Tc, a) + convert(Tc, b)

# ── Secant helpers (cached-inverse, axis-aware) ──────────────────────────────
# Single-cell forward secant (y[i+1]-y[i]) / h_i. Routes through `_get_inv_h`, so a
# `_CachedVector`/`_CachedRange` axis uses the cached reciprocal (no division) and a
# `_UnitStep` range folds the multiply to identity. Raw `AbstractVector` computes
# `inv(h)` on the fly (≤1 ULP vs `/h`, perf-neutral: the extra multiply runs free in
# the divider's shadow).
#
# Width-first form: `Tw` is the value-matched coordinate width from the caller's
# surface (`_promote_grid_float(Tg, Tv)`) — the reciprocal is born at `Tw`, so a raw
# Int axis stops minting `inv(Int)::Float64` beside narrower data. The width-less
# forms delegate with `Tw = eltype(x)`: bit-identical to the historic raw behavior.
@inline function _forward_secant(::Type{Tw}, x, y, i) where {Tw}
    # Value-space widen: the diff stays in value units; the 1/X dimension
    # enters via the cached reciprocal (coeff-space Tc would convert y).
    Tc = _promote_eltype(_interp_op, Tw, eltype(y), Tw)
    return @inbounds _fielddiff(Tc, y[i + 1], y[i]) * _get_inv_h(Tw, x, i)
end
@inline _forward_secant(x, y, i) = _forward_secant(eltype(x), x, y, i)

# Backward secant at i is the forward secant of the previous cell (denominator h_{i-1}).
@inline _backward_secant(::Type{Tw}, x, y, i) where {Tw} = _forward_secant(Tw, x, y, i - 1)
@inline _backward_secant(x, y, i) = _forward_secant(eltype(x), x, y, i - 1)

# Centered (2-cell-span) secant (y[i+1]-y[i-1]) / (x[i+1]-x[i-1]) via `_get_inv_2cell`.
@inline function _centered_secant(::Type{Tw}, x, y, i) where {Tw}
    Tc = _promote_eltype(_interp_op, Tw, eltype(y), Tw)   # value-space (see above)
    return @inbounds _fielddiff(Tc, y[i + 1], y[i - 1]) * _get_inv_2cell(Tw, x, i)
end
@inline _centered_secant(x, y, i) = _centered_secant(eltype(x), x, y, i)

"""
    _promote_query_eltype(::Type{Tv}, q::Tuple) -> Type

Compute the promoted output element type by folding `promote_type` over `Tv`
and the element types of the tuple `q`. Recursive on `Base.tail` for compile-time
type specialization — each step sees concrete types and collapses to a constant
through Julia's normal inference (no @generated body needed, which would suffer
from world-age issues when promotion rules for `q`'s types are defined in an
extension module loaded after FastInterpolations).

Used by the OnTheFly ND `_collapse_dims` entry points where the pool buffer
type must include the query eltype (for ForwardDiff.Dual compatibility) but
the computation must remain zero-cost for plain-Float64 queries.
"""
@inline _promote_query_eltype(::Type{Tv}, ::Tuple{}) where {Tv} = Tv
@inline function _promote_query_eltype(::Type{Tv}, q::Tuple) where {Tv}
    return _promote_query_eltype(promote_type(Tv, typeof(first(q))), Base.tail(q))
end

"""
    _promote_value_type(y, ::Type{Tg}) -> (Tv, y_converted)

Promote y-values to appropriate type based on grid type Tg.

# Rules
1. If eltype(y) === Tg → no conversion (identity)
2. If eltype(y) <: Real → convert to Tg
3. If eltype(y) <: Complex → convert to Complex{Tg}
4. Otherwise → convert to promote_type(eltype(y), Tg)

# Returns
Tuple of (Tv::Type, y_converted::AbstractVector{Tv})
"""
@inline function _promote_value_type(y::AbstractVector{Tv_raw}, ::Type{Tg}) where {Tv_raw, Tg <: AbstractFloat}
    if Tv_raw === Tg
        # Already matching float type - no conversion
        return Tg, y
    elseif Tv_raw <: Real
        # Real (including Int, Float32) → promote to Tg
        return Tg, Tg.(y)
    elseif Tv_raw <: Complex
        # Complex{anything} → Complex{Tg}
        Tv = Complex{Tg}
        # Fast-path: already Complex{Tg} (e.g., ComplexF64 with Tg=Float64)
        if Tv_raw === Tv
            return Tv, y
        end
        return Tv, Tv.(y)
    else
        # Other Number types → promote
        Tv = promote_type(Tv_raw, Tg)
        return Tv, Tv.(y)
    end
end

# ========================================
# Unified Input Promotion API
# ========================================

"""
    _promote_itp_inputs(x, y) -> (x_typed, y_typed)

Promote grid (x) and values (y) to compatible types for interpolation.

# Behavior
- Grid (x) is always converted to AbstractFloat via `_to_float`
- Values (y) handling depends on element type:
  - Standard numerics (`<: _PromotableValue`): promoted to match grid float type
  - Custom/duck types: preserved as-is (zero-copy)

# Standard Path (Real, AbstractFloat, Complex)
- Computes target grid type: `Tg = float(promote_type(TX, _real_eltype(TY)))`
- Converts x via `_to_float` (Range structure preserved)
- Promotes y via `_promote_value_type` (handles numeric widening)

# Duck-Typing Path (custom number types)
- Grid type: `Tg = float(TX)` (no y influence)
- y returned unchanged — custom types preserved for generic kernel arithmetic

# Zero-Overhead Guarantee
- `@inline` + compile-time `TY <: _PromotableValue` check → dead branch eliminated
- Returns inputs unchanged when types already match (zero allocation)

# Examples
```julia
x = [0.0, 1.0, 2.0]; y_int = [1, 2, 3]
x_p, y_p = _promote_itp_inputs(x, y_int)    # y_p is Float64[] (promoted)

# Custom types preserved
x_p, y_p = _promote_itp_inputs(x, custom_y)  # y_p stays custom type
```
"""
@inline function _promote_itp_inputs(
        x::AbstractVector{TX},
        y::AbstractVector{TY}
    ) where {TX, TY}
    Tg = _promote_grid_float(TX, TY)
    x_typed = _to_float(x, Tg)
    # Value promotion: only when BOTH the grid target AND the value type are
    # standard numerics. When Tg is a duck type (e.g. Dual), promoting y to Tg
    # would inject derivative partials into values that carry none — semantically
    # wrong. In that case y passes through unchanged, same as the Tv duck path.
    if TY <: _PromotableValue && Tg <: AbstractFloat
        _, y_typed = _promote_value_type(y, Tg)
        return x_typed, y_typed
    else
        return x_typed, y
    end
end

"""
    _promote_itp_inputs(x, y, xq::AbstractVector) -> (x_typed, y_typed, xq_typed)

Promote grid (x), values (y), and vector query (xq) to compatible Float types.

# Arguments
- `x`: Grid coordinates (any Real type)
- `y`: Values at grid points
- `xq`: Query points (AbstractVector or AbstractRange)

# Returns
- `x_typed`: Grid converted to AbstractFloat
- `y_typed`: Values converted to compatible type
- `xq_typed`: Query converted to grid type (Range structure preserved via `_to_float`)

# Fast-paths
- When types already match, returns inputs unchanged (zero allocation)
- Range inputs remain Range (not converted to Vector)

# Note
For scalar queries, use the 2-arg version and pass the query directly
to preserve ForwardDiff.Dual types for automatic differentiation.
"""
@inline function _promote_itp_inputs(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        xq::AbstractVector{TQ}
    ) where {TX, TY, TQ <: Real}
    x_typed, y_typed = _promote_itp_inputs(x, y)
    xq_typed = _promote_query_typed(xq, eltype(x_typed))
    return x_typed, y_typed, xq_typed
end

# ========================================
# Query & Adjoint Promotion Helpers
"""
    _store_grid(x, ::Type{Tg}) -> stored grid

Single-allocation grid storage for interpolant constructors.
- `AbstractVector`: `_convert_copy(x, Tg)` — promote + copy in one step (no double alloc)
- `AbstractRange`: `_to_float(x, Tg)` — `_CachedRange` (stack alloc, preserves O(1) search)

Note: this is the lightweight normalization — Vector inputs stay as plain
`Vector{Tg}`, no spacing cache. Method-family constructors that want
per-cell h/inv_h caching wrap explicitly via `_CachedVector(_store_grid(x, Tg))`
during the build step (so the cache miss path pays the wrap cost just once,
without inflating cache-hit lookups).
"""
@inline _store_grid(x::AbstractVector, ::Type{Tg}) where {Tg} = _convert_copy(x, Tg)
@inline _store_grid(x::AbstractRange, ::Type{Tg}) where {Tg} = _to_float(x, Tg)

"""
    _convert_copy(v::AbstractVector, ::Type{T}) -> Vector{T}

Copy with optional type conversion in a single allocation.
Same-type: equivalent to `copy(v)`. Different-type: equivalent to `Vector{T}(v)`.

Used in interpolant inner constructors to merge promotion + immutability copy.
"""
@inline _convert_copy(v::AbstractVector{T}, ::Type{T}) where {T} = copy(v)
@inline _convert_copy(v::AbstractVector, ::Type{T}) where {T} = Vector{T}(v)

# ========================================

"""
    _promote_query_typed(xq::AbstractVector, ::Type{Tg}) -> AbstractVector

Widen query vector to `promote_type(Tg, Tq)` — never narrows query precision.
Duck-typed queries (`Dual`, `Measurement`, …) pass through unchanged.
"""
@inline function _promote_query_typed(xq::AbstractVector{Tq}, ::Type{Tg}) where {Tq <: Real, Tg}
    if Tq <: _PromotableValue
        return _to_float(xq, promote_type(Tg, Tq))
    else
        return xq
    end
end

"""
    _promote_adjoint_inputs(x, xq) -> (x_promoted, xq_promoted, Tg)

Promote grid and query vectors for adjoint construction.

Shared pattern across all 1D adjoint builders: cubic, linear, quadratic,
constant, pchip, cardinal, akima.

!!! note "Adjoint queries must be plain real numbers"
    Adjoint constructors do not support `ForwardDiff.Dual` (or other non-grid-
    eltype) *query* points: the anchor structs pin the query type to the grid
    type `Tg` (e.g. `_LinearAnchoredQuery{Tg, Tg}`) and the Hermite-family weight
    kernels are homogeneous in `Tg`, so a Dual query fails to construct. This is
    a long-standing limitation, independent of the `_oob_state`/`_clamp_to_grid`
    boundary helpers (which themselves preserve duck-type). For query-position
    derivatives, differentiate the *forward* eval — it preserves `Dual` queries
    end-to-end. (`constant_adjoint` happens to accept Dual queries because it
    keeps a separate query-type param, but that is not a guaranteed contract.)
"""
@inline function _promote_adjoint_inputs(
        x::AbstractVector,
        xq::AbstractVector
    )
    Tg = _promote_grid_float(eltype(x), eltype(xq))
    x_p = _to_float(x, Tg)
    xq_p = _promote_query_typed(xq, Tg)
    return x_p, xq_p, Tg
end


# ========================================
# AD Support Helpers
# ========================================


"""
    _extract_primal(xq) -> AbstractFloat

Extract the primal (real) value from a query point for index search.

For regular floats, returns as-is.
For ForwardDiff.Dual (when loaded), returns the primal value.

# Usage in Search
This allows AD types to be used for interpolation queries:
- Use `_extract_primal(xq)` ONLY for index search (comparisons)
- Use original `xq` for arithmetic (preserves AD derivatives)

# AD Extension
ForwardDiff support is added via:
```julia
@inline _extract_primal(xq::ForwardDiff.Dual) = ForwardDiff.value(xq)
```
"""
@inline _extract_primal(x) = x  # identity fallback; ForwardDiff ext specializes for Dual
# GridIdx <: Real: _extract_primal(g::GridIdx) returns g (identity fallback).

"""
    _effective_autocache(autocache, Tg) -> Bool

Disable autocache for non-standard grid types (e.g. ForwardDiff.Dual).
Enabled for `_PromotableValue` types (AbstractFloat, Integer, Rational) which
have stable grid identity (cache hit rate > 0). Dual grids are ephemeral
(created fresh each AD call), so autocache is disabled for them.
Resolves at specialization time — zero runtime cost on the Float hot path.
"""
@inline _effective_autocache(ac::Bool, ::Type{Tg}) where {Tg} = ac & (Tg <: _PromotableValue)
# Arithmetic then auto-promotes GridIdx → g.val via promote_rule.

"""
    _coord_eltype(::Type{Tq}, ::Type{Tg}) -> Type

Canonical coordinate element type — the structural twin of [`_promote_eltype`](@ref):
`Base.promote_op` of the coordinate operation, with a `promote_type` fallback when
inference can't resolve the op.

The coordinate operation is **subtraction** (`xq - xL`, comparisons) — *not* the
kernel's `/h`. Subtraction does not widen `Int`, so `Int - Int === Int`: an `Int`
grid + `Int` query keeps an `Int` coordinate (the kernel floats the *output* via
`inv_h`, not the coordinate). `Int - Float === Float` and `Float - Dual === Dual`
fall out for free, and any duck type defining `-` participates without a
`promote_rule`. No `float()` patch — coordinates must not over-float.

This is method-agnostic on purpose: the ND coordinate gateway (`_extrap_axis` /
`_handle_all_extraps`) has no method information, and the only per-method widening
(`/h`) belongs to the value path (`_promote_eltype`), not the coordinate.
"""
@inline function _coord_eltype(::Type{Tq}, ::Type{Tg}) where {Tq, Tg}
    Tc = Base.promote_op(-, Tq, Tg)
    return (Tc === Union{} || Tc === Any) ? promote_type(Tq, Tg) : Tc
end

"""
    _promote_coord(xq, ::Type{Tg}) -> promoted_xq

Promote a query into the grid's coordinate space via the canonical
[`_coord_eltype`](@ref) rule (grid ⊕ query), once, at the eval surface — so the
in-domain kernel and the OOB/fill paths share one concrete coordinate type. The
`convert` is identity on the Float64 hot path, a no-op for a matched `Int` grid,
and a zero-partial lift for a Float query on a `Dual` grid.

# Behavior
- AbstractFloat × AbstractFloat: preserves the wider precision.
- Int/Rational × Float grid: converted to the grid float type.
- Float query × Dual grid: lifts to a zero-partial Dual (AD-with-respect-to-grid).
- Dual query × Float grid: stays Dual (`convert` is identity, or lifts the value type).
- `GridIdx`: passed through unchanged (auto-promotes via its `promote_rule` downstream).

# Example
```julia
_promote_coord(0, Float64)      # → 0.0 (Float64)
_promote_coord(1//2, Float64)   # → 0.5 (Float64)
_promote_coord(0.5, Float32)    # → 0.5 (Float64, preserves precision)
_promote_coord(0.5f0, Float32)  # → 0.5f0 (Float32)
```
"""
@inline _promote_coord(xq::GridIdx, ::Type{Tg}) where {Tg} = xq
@inline _promote_coord(xq, ::Type{Tg}) where {Tg} = convert(_coord_eltype(typeof(xq), Tg), xq)


# ========================================
# Domain Validation Helpers
# ========================================

# `dim == 0` → axis-agnostic message (1D, or when the axis is unknown); `dim > 0`
# names the offending axis (ND scalar / GriddedQuery, which carry the index).
@noinline function _throw_domain_error(xi, x_min, x_max, dim::Int = 0)
    at = dim == 0 ? "query point" : "query point on axis $dim"
    throw(DomainError(xi, "$at outside interpolation domain [$x_min, $x_max]"))
end

# _CachedRange overload: pull the physical endpoints (`lo`/`hi`, the exact span for the
# error message — distinct from the possibly-widened bracket the hot check clamps
# against) inside this @noinline cold path, so the in-domain hot path never
# materializes them — guaranteed, not LLVM-sink-dependent.
@noinline _throw_domain_error(xi, x::_CachedRange, dim::Int = 0) =
    _throw_domain_error(xi, _extract_primal(x.lo), _extract_primal(x.hi), dim)

# Generic vector overload (Vector / `_CachedVector`): physical endpoints via first/last.
@noinline _throw_domain_error(xi, x::AbstractVector, dim::Int = 0) =
    _throw_domain_error(xi, _extract_primal(first(x)), _extract_primal(last(x)), dim)

# NoExtrap domain check. OOB iff the branchless `_clamp` (min/max) alters the query:
# `_clamp(xip,lo,hi) != xip` is 1 compare + 1 branch vs `(xip<lo || xip>hi)`'s two
# branches — a saving that compounds across axes in ND's `_validate_nd_domain`.
# `min`/`max` promote, so the clamp result `c` carries the bound type; comparing
# against `oftype(c, xip)` aligns the RHS to that type so the idiom stays a true
# value test. Without it, a query whose `==` vs its float-promotion is non-reflexive
# (`Irrational`: `Float64(π) != π`; inexact `Rational`) is spuriously flagged OOB.
# On the Float64 hot path `oftype(c, xip)` is identity, so this is free.
# Compares the extracted primal, so a Dual query at the boundary classifies by value,
# not partial sign (cf. `_is_all_inbounds`/`_oob_state`).
# Scalar domain check. Throws `DomainError` for a NoExtrap OOB query, and RETURNS the extrap to
# hand the interval search: once the check establishes in-domain, NoExtrap is equivalent to
# `InBounds()` FOR THE SEARCH, so it promotes to the lean search
# (`search_interval(..., ::InBounds)`) instead of re-paying the boundary guards. Every other mode
# passes through unchanged (ExtendExtrap keeps the two-sided-clamp guarded search — it legitimately
# arrives OOB). The promotion is thus the OUTPUT of the check: an eval core reaches the lean search
# only by threading this return, never from a bare `search_interval(..., ::NoExtrap)` (which stays
# guarded), so an unchecked NoExtrap search can never hit the one-sided lean clamp.
#
# The throw is UNCONDITIONAL (not wrapped in `@boundscheck`): `_validate_domain` calls this inside
# an `@inbounds for` and relies on it always throwing, and the scalar eval surface never wraps a
# core in `@inbounds` — so the cores' former `@boundscheck _check_domain` never actually elided.
# Mirrors the vector/batch `_check_domain` (which likewise returns `InBounds()` / the extrap).
"Scalar domain check for NoExtrap: throws DomainError if OOB, else promotes to InBounds."
@inline function _check_domain(x::AbstractVector, xi, ::NoExtrap, dim::Int = 0)
    xip = _extract_primal(xi)
    x_min, x_max = _extract_primal(first(x)), _extract_primal(last(x))
    c = _clamp(xip, x_min, x_max)
    (c != oftype(c, xip)) && _throw_domain_error(xi, x_min, x_max, dim)
    return InBounds()
end

# _CachedRange: bounds via `_domain_bounds` — `lo`/`hi` for exact tags (shared
# with the search's `lo` load), the widened bracket only for `_WidenedDomain`.
@inline function _check_domain(x::_CachedRange, xi, ::NoExtrap, dim::Int = 0)
    xip = _extract_primal(xi)
    lo, hi = _domain_bounds(x)
    c = _clamp(xip, _extract_primal(lo), _extract_primal(hi))
    (c != oftype(c, xip)) && _throw_domain_error(xi, x, dim)
    return InBounds()
end

# Axis-tagged scalar/batch check: only NoExtrap throws, so only it needs the axis
# index for the message; every other mode ignores `dim` and takes the plain check.
# Lets `_validate_nd_domain` thread the axis without touching the no-op overloads.
@inline _check_domain_axis(x, xi, extrap::AbstractExtrap, dim::Int) = _check_domain(x, xi, extrap)
@inline _check_domain_axis(x, xi, extrap::NoExtrap, dim::Int) = _check_domain(x, xi, extrap, dim)

"No-op scalar domain check for non-NoExtrap modes: returns the extrap unchanged (InBounds stays lean)."
@inline _check_domain(::AbstractVector, ::Any, extrap::AbstractExtrap) = extrap

"GridIdx is in-domain by construction (bounds-checked at resolution time)."
# GridIdx must NOT promote to InBounds: it has its own `search_interval(s, x, ::GridIdx)` fast path
# (index short-circuit, O(1), no coordinate search). Promoting NoExtrap→InBounds would route it into
# the coordinate lean `search_interval(..., ::InBounds)` (GridIdx <: Real), losing the short-circuit
# (O(log n) reads). Return the extrap unchanged so it keeps the GridIdx fast path.
@inline _check_domain(::AbstractVector, ::GridIdx, extrap::AbstractExtrap) = extrap
# Disambiguation: GridIdx <: Real creates ambiguity with the _CachedRange × NoExtrap methods.
@inline _check_domain(::_CachedRange, ::GridIdx, extrap::NoExtrap) = extrap
@inline _check_domain(::AbstractVector, ::GridIdx, extrap::NoExtrap) = extrap

# ----------------------------------------
# Vector domain checks: validate batch, return InBounds() for per-element elision.
# The @boundscheck wraps only the validation logic; `return InBounds()` always
# executes to maintain type stability. With --check-bounds=no, the O(n) min/max
# scan is skipped but the extrap conversion still happens.
# ----------------------------------------

"""
Vector domain check for NoExtrap: validate batch, return `InBounds()`.

Delegates the batch in-domain test to `_is_all_inbounds`, which dispatches
on axis type (a `_WidenedDomain` `_CachedRange` uses its widened `domain_lo/hi`;
exact `_CachedRange`s and Vectors use `lo/hi` / `first/last`) and is
partial-sign-safe under ForwardDiff. Throw message uses `first(x)/last(x)` —
exact endpoints, not the widened bracket.
"""
@inline function _check_domain(x::AbstractVector, xi::AbstractArray, ::NoExtrap, dim::Int = 0)
    @boundscheck _is_all_inbounds(x, xi) || _throw_batch_oob(x, xi, dim)
    return InBounds()
end

# Disambiguation diagonal: `_CachedRange` (scalar arm's axis is unbounded in `xi`)
# × Real-batch query — same batch body as the generic-axis method above.
@inline function _check_domain(x::_CachedRange, xi::AbstractArray, ::NoExtrap, dim::Int = 0)
    @boundscheck _is_all_inbounds(x, xi) || _throw_batch_oob(x, xi, dim)
    return InBounds()
end

# Unit-step axis: fused 3-outcome check — throw (OOB), `InBounds(last = :exclusive)`
# (`maximum < last` proven → the batch loop rides the no-cap search), else closed.
# Generic axes keep the closed-only method above (no Union; nothing to win there).
# Extrema classify on primals: a Dual max whose VALUE == `last` must not tie-break
# on its partial into a false exclusive promotion (→ no-cap OOB read).
# Only the throws are `@boundscheck`-elidable; the promotion is build-mode independent.
@inline function _check_domain(
        x::_CachedRange{T, Tinv, Tag}, xi::AbstractArray{<:Real}, ::NoExtrap, dim::Int = 0
    ) where {T, Tinv, Tag <: _AbstractUnitStep}
    isempty(xi) && return InBounds()
    lo, hi = _domain_bounds(x)
    @boundscheck _ge(_extract_primal(minimum(xi)), _extract_primal(lo)) || _throw_batch_oob(x, xi, dim)
    mx = _extract_primal(maximum(xi))
    hip = _extract_primal(hi)
    @boundscheck _le(mx, hip) || _throw_batch_oob(x, xi, dim)
    return _lt(mx, hip) ? InBounds(last = :exclusive) : InBounds()
end

@noinline function _throw_batch_oob(x::AbstractVector, xi::AbstractArray, dim::Int = 0)
    qmin, qmax = minimum(xi), maximum(xi)
    x_min = _extract_primal(first(x))
    x_max = _extract_primal(last(x))
    _throw_domain_error(qmin < x_min ? qmin : qmax, x_min, x_max, dim)
end

"No-op vector domain check for non-NoExtrap modes: pass-through extrap."
@inline _check_domain(::AbstractVector, ::AbstractArray, extrap::AbstractExtrap) = extrap

# Closed-domain batch fast path: every OOB policy (`ClampExtrap`, `FillExtrap`,
# `WrapExtrap`) treats `[first(x), last(x)]` as the in-domain interval, so they
# share one batch promotion to `InBounds()`.
@inline function _check_domain(
        x::AbstractVector, xi::AbstractArray,
        e::Union{ClampExtrap, FillExtrap, WrapExtrap}
    )
    return _is_all_inbounds(x, xi) ? InBounds() : e
end

# Unit-step twin: original extrap (any OOB) / exclusive-last (strictly below `last`)
# / closed (touches `last`). Primal extrema — see the NoExtrap twin.
@inline function _check_domain(
        x::_CachedRange{T, Tinv, Tag}, xi::AbstractArray{<:Real},
        e::Union{ClampExtrap, FillExtrap, WrapExtrap}
    ) where {T, Tinv, Tag <: _AbstractUnitStep}
    isempty(xi) && return InBounds()
    lo, hi = _domain_bounds(x)
    _ge(_extract_primal(minimum(xi)), _extract_primal(lo)) || return e
    mx = _extract_primal(maximum(xi))
    hip = _extract_primal(hi)
    _le(mx, hip) || return e
    return _lt(mx, hip) ? InBounds(last = :exclusive) : InBounds()
end

# Safe domain bounds — single axis-dispatched source of truth for every in-domain
# test (`_is_all_inbounds`, `_is_inbounds`, `_oob_state`), so they never disagree
# at a boundary query. Exact-domain axes → `first/last` (field reads on
# `_CachedRange`; `_OneTo` folds its lower bound to the literal `one(T)`); only a
# `_WidenedDomain` range → its `domain_lo/hi`, ≈1 ULP wider than the stored
# `lo/hi` on the x86_64 fast path, keeping a query at the true endpoint in-domain
# instead of falsely OOB.
@inline _domain_bounds(x::AbstractVector) = (first(x), last(x))
@inline _domain_bounds(x::_CachedRange) = (first(x), last(x))
@inline _domain_bounds(x::_CachedRange{T, Tinv, _WidenedDomain}) where {T, Tinv} =
    (x.domain_lo, x.domain_hi)                                        # widened bracket

# Scalar in-domain test. `_extract_primal` on bounds and query keeps it partial-
# sign independent for Dual grids (no-op on plain-Float `_CachedRange` fields).
@inline function _is_inbounds(x::AbstractVector, xq::Real)
    lo, hi = _domain_bounds(x)
    xqp = _extract_primal(xq)
    # `_le` promote-compare: dodge Base's exact mixed `<=(Int, Float)` on an
    # Int/Rational grid (no-op on a Float grid). See ordering helpers in search.jl.
    return _le(_extract_primal(lo), xqp) && _le(xqp, _extract_primal(hi))
end

# Clamp a query to the grid's physical span `[first(x), last(x)]`, never the
# widened `_domain_bounds` bracket — adjoint anchor builders use it for valid OOB
# boundary geometry while classification reads the widened bounds (keeps the
# acceptance cushion out of geometry).
@inline _clamp_to_grid(xq::Real, x::AbstractVector) =
    _clamp(xq, _extract_primal(first(x)), _extract_primal(last(x)))

"""
True iff every element of `queries` lies in the closed domain
`[first(x), last(x)]`. Enables batch-level fast paths that elide per-query
domain handling (e.g. `_wrap_to_domain` for PeriodicBC, which only needs
to apply when a query is strictly outside `[first, last]`) when no
element is OOB.

Uses two `&&`-chained reductions rather than `extrema`:
- pre-1.13 `extrema` carries a (min, max) tuple dep across the loop that
  blocks LLVM auto-vectorization (~30× slower on Vector{Float64} N=1000)
- `&&` short-circuits when `minimum` is already OOB, skipping the
  `maximum` scan entirely — strictly ≤ `extrema`'s work in all cases

1.13 fixes the SIMD issue, but the short-circuit advantage remains for
the OOB slow-path, so this form stays preferred even post-1.10-LTS.
"""
# `_extract_primal` on the bounds: ForwardDiff's `Real <= Dual` tie-breaks on the
# partial sign at equal primals, so a Float query at the boundary against a Dual
# grid endpoint must classify on primal alone (see test/ext/test_linear_dual_grid.jl).
# Routes through `_domain_bounds` (widened bracket in one place); `&&` short-circuits.
@inline function _is_all_inbounds(x::AbstractVector, queries::AbstractArray)
    isempty(queries) && return true
    lo, hi = _domain_bounds(x)
    # `_ge`/`_le` promote-compare (see search.jl): dodge Base's exact mixed
    # `>=(Float, Int)` on an Int/Rational grid. Amortized over the min/max scan
    # (one compare per batch), but free on a Float grid. Extrema classify on
    # primals: a Dual query whose VALUE == a bound must not tie-break on its
    # partial sign into a false OOB verdict (identity on the Float64 hot path).
    return _ge(_extract_primal(minimum(queries)), _extract_primal(lo)) &&
        _le(_extract_primal(maximum(queries)), _extract_primal(hi))
end

# ========================================
# Extrapolation value helpers (shared by all interpolation methods)
# ========================================
# _eval_extrapolation: return the appropriate value for an OOB query with ClampExtrap/FillExtrap.
# Dispatches on op (EvalValue vs derivatives) and extrap type (Clamp vs Fill).
#
# _promote_extrap_val:  promotes an extrap result value to match kernel return type.
# _promote_extrap_zero: carrier-aware zero for the OOB deriv-zero path under
#                       flat extrapolation (Clamp/Fill, any method family).
#                       The leading `0 * val` preserves NaN/Inf at the boundary
#                       sample (IEEE: `0 * NaN = NaN`); `zero(xq) * zero(val)`
#                       carries the query carrier (Dual, etc.) into the result.
# These arithmetic forms are LOAD-BEARING — do NOT fold them to `oftype`/`convert` to drop the
# `+ 0.0`: the trailing add also normalizes signed zero (`-0.0 → +0.0`) and keeps Unitful
# dimension-correctness (`oftype(zero(val)+zero(xq), val)` throws on `Quantity` + plain query).
# Guarded by test/test_extrap_carrier_guards.jl.
# Named _promote_extrap_val (not _promote_extrap) to avoid collision with the struct
# promoter in eval_ops.jl which promotes FillExtrap fill_value at construction time.
# `zero(xq) * inv(oneunit(xq))` = the DIMENSIONLESS query-carrier zero: same
# carrier type as `zero(xq)` (Dual, etc.), but unit-free so `* zero(val)` stays
# in value dimensions (a raw `zero(xq)` factor would make the term query×value).
@inline _promote_extrap_val(val::Number, xq::Number) = val + zero(xq) * inv(oneunit(xq)) * zero(val)
# AbstractArray Tv (e.g. `SVector` y) — broadcast the carrier-propagating
# pattern so scalar OOB matches in-domain kernel's `y * one(dL)` shape and
# agrees with batch path's trait-sized buffer.
@inline _promote_extrap_val(val::AbstractArray, xq::Number) = val .+ zero(xq) .* inv(oneunit(xq)) .* zero(eltype(val))
@inline _promote_extrap_val(val, xq) = val
@inline _promote_extrap_zero(val::Number, xq::Number) = 0 * val + zero(xq) * inv(oneunit(xq)) * zero(val)
@inline _promote_extrap_zero(val::AbstractArray, xq::Number) = 0 .* val .+ zero(xq) .* inv(oneunit(xq)) .* zero(eltype(val))
@inline _promote_extrap_zero(val, xq) = 0 * val

# _extrap_oob_data: per-extrap "what data sits in the OOB cell".
#   ClampExtrap → `y_bnd`         (boundary y is extended into the OOB region).
#   FillExtrap  → `e.fill_value`  (fill_value is the OOB cell's data).
# `@inline` + singleton dispatch — LLVM specializes per concrete extrap type
# and dead-branch-eliminates; zero overhead on the OOB cold path.
@inline _extrap_oob_data(::ClampExtrap, y_bnd) = y_bnd
@inline _extrap_oob_data(e::FillExtrap, _) = e.fill_value

# OOB evaluation under flat extrapolation (Clamp / Fill): the OOB cell's
# data is fetched via `_extrap_oob_data` (ClampExtrap → `y_bnd`, FillExtrap
# → `fill_value`), then promoted by the op-specific kernel:
#   `EvalValue` → `data + carrier`         (value path)
#   `DerivOp`   → `0 * data + carrier`     (deriv path — 0 × OOB cell data)
# This `data → promote` split makes the deriv path's `_extrap_oob_data`
# call read naturally: we're not asking "what's the derivative" but "what's
# the cell data" — the `0 *` happens inside `_promote_extrap_zero`.
@inline _eval_extrapolation(::EvalValue, y_bnd, ext::Union{ClampExtrap, FillExtrap}, xq) =
    _promote_extrap_val(_extrap_oob_data(ext, y_bnd), xq)
@inline _eval_extrapolation(::DerivOp, y_bnd, ext::Union{ClampExtrap, FillExtrap}, xq) =
    _promote_extrap_zero(_extrap_oob_data(ext, y_bnd), xq)
