# ========================================
# Coefficient Strategy Resolution Policy
# ========================================
# Centralized AutoCoeffs resolution — single source of truth for default
# strategy selection across all call sites (scalar, vector, interpolant, ND).
#
# Pattern mirrors AutoSearch: runtime O(1) check → concrete strategy → zero-cost dispatch.
# All overloads are @inline for constant propagation and dead branch elimination.
#
# Include order: after coeff_types.jl (AutoCoeffs) and interp_method_types.jl (PchipInterp etc.)

# ── Passthrough: explicit strategies are never modified ──
@inline _resolve_coeffs(c::PreCompute, args...) = c
@inline _resolve_coeffs(c::OnTheFly, args...) = c

# ── AutoCoeffs: 1D scalar query → always OnTheFly ──
# Single query: O(1) local slopes strictly beats O(n) bulk slopes.
@inline _resolve_coeffs(::AutoCoeffs, ::AbstractVector, ::Real) = OnTheFly()

# ── AutoCoeffs: 1D vector query → runtime length check ──
# Crossover: PreCompute O(n + K) vs OnTheFly O(2K). At K ≈ n, PreCompute wins.
@inline function _resolve_coeffs(::AutoCoeffs, x::AbstractVector, xq::AbstractVector)
    return length(xq) > length(x) ? PreCompute() : OnTheFly()
end

# ── AutoCoeffs: interpolant construction (no query) → PreCompute for reuse ──
# Interpolant is built once, called many times — amortize the slope computation.
@inline _resolve_coeffs(::AutoCoeffs) = PreCompute()

# ── AutoCoeffs: ND → method type check ──
# Linear/Constant have no slopes — always use specialized ND types.
# Local Hermite methods → OnTheFly (no ND PreCompute for Hermite yet).
# Global methods (Cubic, Quadratic) → PreCompute (specialized ND types with integrate/adjoint).
# NOTE: N≥3 rule removed — would route Cubic/Quadratic to HeteroInterpolantND which lacks
# integrate/adjoint support. Can be re-added when HeteroInterpolantND gains these features.
@inline function _resolve_coeffs(::AutoCoeffs, ::Val{N}, methods) where {N}
    _all_trivial_methods(methods) && return PreCompute()
    _has_any_local_method(methods) && return OnTheFly()
    return PreCompute()
end

# ── Method traits ──
# Local methods: slopes computed from local stencil (no global solve)
@inline _is_local_method(::PchipInterp) = true
@inline _is_local_method(::CardinalInterp) = true
@inline _is_local_method(::AkimaInterp) = true
@inline _is_local_method(::AbstractInterpMethod) = false
@inline _has_any_local_method(methods::Tuple) = any(_is_local_method, methods)

# Trivial methods: no slopes/partials needed (Linear, Constant)
@inline _is_trivial_method(::LinearInterp) = true
@inline _is_trivial_method(::ConstantInterp) = true
@inline _is_trivial_method(::AbstractInterpMethod) = false
@inline _all_trivial_methods(methods::Tuple) = all(_is_trivial_method, methods)

# ── ND oneshot resolution (separate from interpolant resolution) ──
# Centralized (method, query)-aware policy: resolved once at the user API entry
# point alongside the other resolves (extrap/search/deriv) and threaded down.
# All branches fold at compile time when `methods` is a concrete tuple type.
#
# Strategy matrix:
#   methods\queries          | scalar (Tuple{Vararg{Real}}) | batch
#   -------------------------|------------------------------|---------
#   all trivial (Lin/Const)  | PreCompute (no slopes)       | PreCompute
#   has local Hermite        | OnTheFly (per-fiber 1D)      | OnTheFly (per-query loop)
#   global only (Cubic/Quad) | OnTheFly (skip 2^N partials) | PreCompute (amortize solve)
#
# Rationale for the local-Hermite batch case: PreCompute backend
# (`_compute_nd_partials_hetero!`) does not implement `_extract_bc(::PchipInterp)`
# etc., so the only working path for batch + Hermite is a per-query loop calling
# `_interp_nd_oneshot_onthefly`, dispatched inside `_interp_nd_oneshot_batch_dispatch!`.
#
# ForwardDiff.Dual queries (scalar) flow through OnTheFly directly — `_collapse_dims`
# promotes its pool buffer via `_promote_query_eltype(Tv, q_eval)`.
#
# Separate function from `_resolve_coeffs` (1D) avoids dispatch ambiguity with
# the interpolant-construction overloads.
@inline _resolve_coeffs_nd_oneshot(c::PreCompute, _, _) = c
@inline _resolve_coeffs_nd_oneshot(c::OnTheFly, _, _) = c
@inline function _resolve_coeffs_nd_oneshot(::AutoCoeffs, ::Tuple{Vararg{Real}}, methods)
    _all_trivial_methods(methods) && return PreCompute()
    return OnTheFly()
end
# Batch (non-scalar) fallback: PreCompute amortizes the build, except when the
# method tuple contains a local Hermite method which has no PreCompute backend.
@inline function _resolve_coeffs_nd_oneshot(::AutoCoeffs, queries, methods)
    _has_any_local_method(methods) && return OnTheFly()
    return PreCompute()
end
