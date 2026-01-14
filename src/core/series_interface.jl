# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  SERIES INTERPOLANT INTERFACE                             ║
# ║         Required traits and default callable implementations              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: abstract_types.jl → ... → series_utils.jl → series_interface.jl
#
# Design: Option B (Default + Override)
# - Required traits must be implemented by all concrete types
# - Default callables work for Composition pattern
# - Concrete types can override for specialized performance (e.g., Cubic)
#

# ========================================
# Required Trait Functions
# ========================================

"""
    n_series(sitp::AbstractSeriesInterpolant) -> Int

Return the number of y-series in the interpolant.

**Required**: All concrete types must implement this.
"""
n_series(sitp::AbstractSeriesInterpolant) =
    error("n_series not implemented for $(typeof(sitp))")

"""
    _should_wrap(sitp::AbstractSeriesInterpolant) -> Bool

Check if wrap mode is active for periodic boundary handling.

**Required**: All concrete types must implement this.
"""
_should_wrap(sitp::AbstractSeriesInterpolant) =
    error("_should_wrap not implemented for $(typeof(sitp))")

"""
    _get_grid(sitp::AbstractSeriesInterpolant) -> AbstractVector

Return the x-grid vector.

**Required**: All concrete types must implement this.
"""
_get_grid(sitp::AbstractSeriesInterpolant) =
    error("_get_grid not implemented for $(typeof(sitp))")

"""
    _get_extrap(sitp::AbstractSeriesInterpolant) -> ExtrapVal

Return the extrapolation mode.

**Required**: All concrete types must implement this.
"""
_get_extrap(sitp::AbstractSeriesInterpolant) =
    error("_get_extrap not implemented for $(typeof(sitp))")

"""
    _make_anchor(sitp::AbstractSeriesInterpolant, xq::T) -> AnchoredQuery

Build an anchor for a query point.

**Required**: All concrete types must implement this.
"""
_make_anchor(sitp::AbstractSeriesInterpolant, xq) =
    error("_make_anchor not implemented for $(typeof(sitp))")

"""
    _eval_series_at_anchor!(output, sitp::AbstractSeriesInterpolant, aq, op) -> output

Evaluate all series at the given anchor point.

**Required**: All concrete types must implement this.

# Arguments
- `output::AbstractVector{T}`: Pre-allocated output buffer
- `sitp`: The series interpolant
- `aq`: Anchored query (from `_make_anchor`)
- `op::AbstractEvalOp`: Evaluation operation (EvalValue, EvalDeriv1, etc.)
"""
_eval_series_at_anchor!(output, sitp::AbstractSeriesInterpolant, aq, op) =
    error("_eval_series_at_anchor! not implemented for $(typeof(sitp))")

# ========================================
# Helper: Derivative Order to EvalOp
# ========================================

"""
    _deriv_op(deriv::Int) -> AbstractEvalOp

Convert derivative order integer to compile-time EvalOp type.
"""
@inline function _deriv_op(deriv::Int)
    deriv == 0 && return EvalValue()
    deriv == 1 && return EvalDeriv1()
    deriv == 2 && return EvalDeriv2()
    throw(ArgumentError("deriv must be 0, 1, or 2; got $deriv"))
end
