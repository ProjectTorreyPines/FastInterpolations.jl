# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    SERIES INTERPOLANT UTILITIES                           ║
# ║              Validation helpers for AbstractSeriesInterpolant             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: abstract_types.jl → ... → series_utils.jl
#

# ========================================
# Input Validation
# ========================================

"""
    _validate_series_inputs(x, ys)

Validate input y-series dimensions match x-grid.

# Throws
- `ArgumentError` if `ys` is empty
- `DimensionMismatch` if any y-series has different length than `x`
"""
function _validate_series_inputs(
    x::AbstractVector,
    ys::AbstractVector{<:AbstractVector}
)
    isempty(ys) && throw(ArgumentError("ys must not be empty"))
    n = length(x)
    for (k, y) in enumerate(ys)
        length(y) != n && throw(DimensionMismatch(
            "y-series $k has length $(length(y)), expected $n"
        ))
    end
    return nothing
end

# ========================================
# Output Validation
# ========================================

"""
    _validate_series_outputs(outputs, n_series, n_query)

Validate output buffer dimensions for vector evaluation.

# Throws
- `DimensionMismatch` if `outputs` length != `n_series`
- `DimensionMismatch` if any output buffer length != `n_query`
"""
function _validate_series_outputs(
    outputs::AbstractVector{<:AbstractVector},
    n_series::Int,
    n_query::Int
)
    length(outputs) != n_series && throw(DimensionMismatch(
        "outputs length $(length(outputs)) must match n_series $n_series"
    ))
    for (k, out) in enumerate(outputs)
        length(out) != n_query && throw(DimensionMismatch(
            "outputs[$k] length $(length(out)) must match n_query $n_query"
        ))
    end
    return nothing
end

"""
    _validate_scalar_output(output, n_series)

Validate scalar output buffer dimension.

# Throws
- `DimensionMismatch` if `output` length != `n_series`
"""
@inline function _validate_scalar_output(
    output::AbstractVector,
    n_series::Int
)
    length(output) == n_series || throw(DimensionMismatch(
        "output length $(length(output)) must match n_series $n_series"
    ))
    return nothing
end
