# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    MULTI-Y CONSTANT INTERPOLATION                         ║
# ║        Multiple y-data series sharing the same x-grid                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Uses composition approach: wraps existing ConstantInterpolant objects.
# Key optimization: anchor computed once, applied to all series.
#
# Include order: ... → constant_anchor.jl → multi_constant_interp.jl
#

# ========================================
# Type Definition
# ========================================

"""
    ConstantMultiInterpolant{T,I}

Container for multiple constant interpolants sharing the same x-grid.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `I`: Container type for interpolants (Vector, etc.)

# Fields
- `itps::I`: Container of `ConstantInterpolant{T}` objects

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
y1, y2, y3 = sin.(2π .* x), cos.(2π .* x), exp.(-x)

mitp = constant_interp(x, [y1, y2, y3])

# Scalar evaluation
vals = mitp(0.5)            # Returns Vector{Float64} of length 3
mitp(output, 0.5)           # In-place

# Vector evaluation
vals = mitp([0.1, 0.5, 0.9])    # Returns Vector of Vectors
mitp([out1, out2, out3], xq)    # In-place (zero allocation)
```

# Performance
- Anchor computed once, reused for all series
- In-place container evaluation is zero-allocation
"""
struct ConstantMultiInterpolant{T<:AbstractFloat, I<:AbstractArray{<:ConstantInterpolant{T}}} <: AbstractMultiInterpolant{T}
    itps::I
end

# ========================================
# Helper Functions
# ========================================

"""Reference interpolant for grid access and mode checks."""
@inline _ref_itp(mitp::ConstantMultiInterpolant) = first(mitp.itps)

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(mitp::ConstantMultiInterpolant) = _ref_itp(mitp).mode === Val(:wrap)

# ========================================
# Constructors
# ========================================

"""
    constant_interp(x, ys::AbstractVector{<:AbstractVector}; side=:nearest, extrap=:none)

Create a multi-Y constant interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `ys`: Vector of y-value vectors (all same length as x)
- `side`: Side for discontinuities (:left, :right, :nearest)
- `extrap`: Extrapolation mode (:none, :constant, :extension, :wrap)

# Returns
`ConstantMultiInterpolant` object.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)
y3 = exp.(-x)

mitp = constant_interp(x, [y1, y2, y3])
vals = mitp(0.5)
```
"""
function constant_interp(
    x::AbstractVector{T},
    ys::AbstractVector{<:AbstractVector{T}};
    side::Symbol=:nearest,
    extrap::Symbol=:none
) where {T<:AbstractFloat}
    # Validate input
    @assert !isempty(ys) "ys must not be empty"
    n = length(x)
    @assert all(y -> length(y) == n, ys) "All y-series must have same length as x"

    # Build first interpolant to get concrete type
    itp0 = constant_interp(x, first(ys); side=side, extrap=extrap)

    # Build remaining interpolants
    itps = similar(ys, typeof(itp0))
    idx = firstindex(itps)
    itps[idx] = itp0

    for (k, y) in Iterators.drop(pairs(ys), 1)
        itps[k] = constant_interp(x, y; side=side, extrap=extrap)
    end

    return ConstantMultiInterpolant{T, typeof(itps)}(itps)
end

# Matrix input: columns as y-series
"""
    constant_interp(x, Y::AbstractMatrix; side=:nearest, extrap=:none)

Create a multi-Y constant interpolant from a matrix where each column is a y-series.

# Arguments
- `x::AbstractVector`: x-coordinates (length n)
- `Y::AbstractMatrix`: n×m matrix, each column is a y-series
- `side`, `extrap`: Same as vector form

# Example
```julia
x = collect(range(0.0, 1.0, 101))
Y = hcat(sin.(2π .* x), cos.(2π .* x))  # 101×2 matrix

mitp = constant_interp(x, Y)
```
"""
function constant_interp(
    x::AbstractVector{T},
    Y::AbstractMatrix{T};
    side::Symbol=:nearest,
    extrap::Symbol=:none
) where {T<:AbstractFloat}
    ys = [Y[:, k] for k in axes(Y, 2)]
    return constant_interp(x, ys; side=side, extrap=extrap)
end

# Real type wrappers (auto-promote to Float)
function constant_interp(
    x::AbstractVector{Tx},
    ys::AbstractVector{<:AbstractVector{Ty}};
    side::Symbol=:nearest,
    extrap::Symbol=:none
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    ys_float = [_to_float(y, T) for y in ys]
    return constant_interp(x_float, ys_float; side=side, extrap=extrap)
end

function constant_interp(
    x::AbstractVector{Tx},
    Y::AbstractMatrix{Ty};
    side::Symbol=:nearest,
    extrap::Symbol=:none
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    Y_float = T.(Y)
    return constant_interp(x_float, Y_float; side=side, extrap=extrap)
end

# ========================================
# Scalar Evaluation
# ========================================

"""
    (mitp::ConstantMultiInterpolant)(xq::Real)

Evaluate multi-Y interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.
"""
function (mitp::ConstantMultiInterpolant{T})(xq::S) where {T<:AbstractFloat, S<:Real}
    ref = _ref_itp(mitp)
    xq_typed = T(xq)

    # Build anchor once
    aq = _anchor_query(ref.x, xq_typed, Val(:constant); wrap=_should_wrap(mitp))

    # Evaluate all series
    output = Vector{T}(undef, length(mitp.itps))
    @inbounds for k in eachindex(mitp.itps)
        output[k] = mitp.itps[k](aq)
    end
    return output
end

"""
    (mitp::ConstantMultiInterpolant)(output::AbstractVector, xq::Real)

Evaluate multi-Y interpolant at scalar query point (in-place).
"""
function (mitp::ConstantMultiInterpolant{T})(
    output::AbstractVector{T},
    xq::S
) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == length(mitp.itps) "output length must match number of series"

    ref = _ref_itp(mitp)
    xq_typed = T(xq)

    # Build anchor once
    aq = _anchor_query(ref.x, xq_typed, Val(:constant); wrap=_should_wrap(mitp))

    # Evaluate all series
    @inbounds for k in eachindex(mitp.itps, output)
        output[k] = mitp.itps[k](aq)
    end
    return output
end

# ========================================
# Vector Evaluation
# ========================================

"""
    (mitp::ConstantMultiInterpolant)(xq::AbstractVector)

Evaluate multi-Y interpolant at multiple query points (out-of-place).

Returns a vector of vectors: one vector per y-series, each containing results for all query points.
"""
function (mitp::ConstantMultiInterpolant{T})(
    xq::AbstractVector{S}
) where {T<:AbstractFloat, S<:Real}
    ref = _ref_itp(mitp)
    xq_typed = _to_float(xq, T)

    # Build anchors once
    aq_vec = _anchor_query(ref.x, xq_typed, Val(:constant); wrap=_should_wrap(mitp))

    # Allocate outputs
    outputs = [Vector{T}(undef, length(xq_typed)) for _ in 1:length(mitp.itps)]

    # Evaluate all series using in-place anchored evaluation
    @inbounds for k in eachindex(mitp.itps)
        mitp.itps[k](outputs[k], aq_vec)
    end
    return outputs
end

"""
    (mitp::ConstantMultiInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector)

Evaluate multi-Y interpolant at multiple query points (in-place, zero allocation).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points

This is the KILLER FEATURE: zero-allocation batch evaluation for hot loops.
"""
function (mitp::ConstantMultiInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{T}
) where {T<:AbstractFloat}
    @assert length(outputs) == length(mitp.itps) "outputs length must match number of series"
    @assert all(out -> length(out) == length(xq), outputs) "all output buffers must match xq length"

    ref = _ref_itp(mitp)

    # Build anchors once
    aq_vec = _anchor_query(ref.x, xq, Val(:constant); wrap=_should_wrap(mitp))

    # Evaluate all series using in-place anchored evaluation
    @inbounds for k in eachindex(mitp.itps, outputs)
        mitp.itps[k](outputs[k], aq_vec)
    end
    return outputs
end

# Real type wrapper for in-place vector
function (mitp::ConstantMultiInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{S}
) where {T<:AbstractFloat, S<:Real}
    xq_typed = _to_float(xq, T)
    return mitp(outputs, xq_typed)
end

"""
    (mitp::ConstantMultiInterpolant)(outputs, aq_vec::AbstractVector{<:_ConstantAnchoredQuery})

Evaluate multi-Y interpolant with pre-built anchors (TRUE zero-allocation).

For maximum performance in hot loops, pre-build anchors once and reuse:
```julia
x = ...
mitp = constant_interp(x, [y1, y2, y3])
xq = [0.1, 0.2, 0.3, ...]

# Pre-build anchors (allocates once)
aq_vec = FastInterpolations._anchor_query(x, xq, Val(:constant))

# Zero-allocation loop
outputs = [similar(xq) for _ in 1:3]
for _ in 1:1000
    mitp(outputs, aq_vec)  # Zero allocation!
end
```
"""
function (mitp::ConstantMultiInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    aq_vec::AbstractVector{<:_ConstantAnchoredQuery{T}}
) where {T<:AbstractFloat}
    @assert length(outputs) == length(mitp.itps) "outputs length must match number of series"
    @assert all(out -> length(out) == length(aq_vec), outputs) "all output buffers must match aq_vec length"

    # Evaluate all series using in-place anchored evaluation
    @inbounds for k in eachindex(mitp.itps, outputs)
        mitp.itps[k](outputs[k], aq_vec)
    end
    return outputs
end
