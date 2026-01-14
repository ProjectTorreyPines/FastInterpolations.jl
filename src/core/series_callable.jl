# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  SERIES CALLABLE IMPLEMENTATIONS                          ║
# ║         Default callable methods for AbstractSeriesInterpolant            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: ... → series_utils.jl → series_interface.jl → series_matrix.jl → series_callable.jl
#
# Design: "Default + Override" Pattern
# - These default implementations use trait functions
# - Concrete types (e.g., CubicSeriesInterpolant) can override for SIMD optimization
# - Falls back to composition-style evaluation via _eval_series_at_anchor!
#
# Required traits (from series_interface.jl):
#   n_series(sitp) → Int
#   _should_wrap(sitp) → Bool
#   _get_grid(sitp) → AbstractVector
#   _get_extrap(sitp) → ExtrapVal
#   _make_anchor(sitp, xq) → AnchoredQuery
#   _eval_series_at_anchor!(output, sitp, aq, op) → output
#
# Validation helpers from series_utils.jl:
#   _validate_scalar_output(output, n_series)
#   _validate_series_outputs(outputs, n_series, n_query)
#

# ════════════════════════════════════════════════════════════════════════════
# SIGNATURE 1: SCALAR OUT-OF-PLACE
# ════════════════════════════════════════════════════════════════════════════

"""
    (sitp::AbstractSeriesInterpolant)(xq::Real; deriv=0)

Evaluate series interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.

# Default Implementation
Allocates output buffer and delegates to in-place form.
Concrete types may override for specialized performance.
"""
function (sitp::AbstractSeriesInterpolant{T})(xq::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    out = Vector{T}(undef, n_series(sitp))
    return sitp(out, xq; deriv=deriv)
end

# ════════════════════════════════════════════════════════════════════════════
# SIGNATURE 2: SCALAR IN-PLACE
# ════════════════════════════════════════════════════════════════════════════

"""
    (sitp::AbstractSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=0)

Evaluate series interpolant at scalar query point (in-place).

# Default Implementation
Uses trait functions `_make_anchor` and `_eval_series_at_anchor!` for dispatch.
Concrete types may override for specialized SIMD performance.
"""
function (sitp::AbstractSeriesInterpolant{T})(
    output::AbstractVector{T},
    xq::S;
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    _validate_scalar_output(output, n_series(sitp))

    xq_typed = T(xq)

    # Build anchor using trait
    aq = _make_anchor(sitp, xq_typed)

    # Dispatch on derivative order and evaluate
    @_dispatch_deriv deriv => op begin
        _eval_series_at_anchor!(output, sitp, aq, op)
    end

    return output
end

# ════════════════════════════════════════════════════════════════════════════
# SIGNATURE 3: VECTOR OUT-OF-PLACE
# ════════════════════════════════════════════════════════════════════════════

"""
    (sitp::AbstractSeriesInterpolant)(xq::AbstractVector; deriv=0)

Evaluate series interpolant at multiple query points (out-of-place).

Returns a vector of vectors: one vector per y-series, each containing
results for all query points.

# Default Implementation
Allocates output buffers and delegates to in-place form.
"""
function (sitp::AbstractSeriesInterpolant{T})(
    xq::AbstractVector{S};
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    xq_typed = _to_float(xq, T)
    n_query = length(xq_typed)
    n_ser = n_series(sitp)

    outputs = [Vector{T}(undef, n_query) for _ in 1:n_ser]
    sitp(outputs, xq_typed; deriv=deriv)

    return outputs
end

# ════════════════════════════════════════════════════════════════════════════
# SIGNATURE 4: VECTOR IN-PLACE
# ════════════════════════════════════════════════════════════════════════════

"""
    (sitp::AbstractSeriesInterpolant)(outputs, xq::AbstractVector; deriv=0)

Evaluate series interpolant at multiple query points (in-place, zero allocation).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points
- `deriv`: Derivative order (0, 1, or 2)

# Default Implementation
Loops over query points, evaluating each via _eval_series_at_anchor!.
Uses task-local pool for temporary buffer to achieve zero allocation.
Concrete types may override for optimized batch processing.
"""
@with_pool pool function (sitp::AbstractSeriesInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{T};
    deriv::Int=0
) where {T<:AbstractFloat}
    n_query = length(xq)
    n_ser = n_series(sitp)

    _validate_series_outputs(outputs, n_ser, n_query)

    # Acquire temporary buffer for single-point evaluation from pool
    tmp = acquire!(pool, T, n_ser)

    # Evaluate each query point using traits
    @_dispatch_deriv deriv => op begin
        @inbounds for j in 1:n_query
            xq_j = xq[j]
            aq = _make_anchor(sitp, xq_j)

            # Evaluate all series at this anchor into temp buffer
            _eval_series_at_anchor!(tmp, sitp, aq, op)

            # Copy results into output vectors
            for k in 1:n_ser
                outputs[k][j] = tmp[k]
            end
        end
    end

    return outputs
end

# Real type wrapper for vector in-place
function (sitp::AbstractSeriesInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{S};
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    xq_typed = _to_float(xq, T)
    return sitp(outputs, xq_typed; deriv=deriv)
end
