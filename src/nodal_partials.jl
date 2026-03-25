# ========================================
# Nodal Partial Derivative Access (Public API)
# ========================================
#
# Provides `nodal_partials(itp, order)` for zero-copy access to precomputed
# partial derivatives at grid nodes. Abstracts over the internal indexing
# schemes (bit-encoded for homogeneous ND, mixed-radix for heterogeneous ND).

# ========================================
# Internal Helpers
# ========================================

"""
    _order_to_bitindex(order::NTuple{N, Integer}) -> Int

Convert a binary derivative order tuple to a bit-encoded linear index.

For homogeneous ND (Cubic/Quadratic), partials use bit-encoding:
`p = 1 + Σ(order[k] × 2^(k-1))`.

# Examples
- `(0, 0)` → 1 (function value)
- `(1, 0)` → 2 (∂f/∂x)
- `(0, 1)` → 3 (∂f/∂y)
- `(1, 1)` → 4 (∂²f/∂x∂y)
"""
@inline function _order_to_bitindex(order::NTuple{N, Integer}) where {N}
    p = 1
    for k in 1:N
        p += order[k] << (k - 1)
    end
    return p
end

"""
    _order_to_hetero_index(order, methods) -> Int

Convert a binary derivative order tuple to a compact mixed-radix index.

For heterogeneous ND, partials use mixed-radix indexing where non-derivative
axes (Linear/Constant/NoInterp) have size 1 and are skipped.
"""
@inline function _order_to_hetero_index(
        order::NTuple{N, Integer}, methods::Tuple{Vararg{AbstractInterpMethod, N}}
    ) where {N}
    p = 1
    stride = 1
    for k in 1:N
        p += order[k] * stride
        stride *= _deriv_size(methods[k])
    end
    return p
end

# ========================================
# Validation
# ========================================

@inline function _validate_binary_order(order::NTuple{N, Integer}) where {N}
    for d in 1:N
        o = order[d]
        (o == 0 || o == 1) || throw(
            ArgumentError(
                "derivative order at axis $d must be 0 or 1, got $o"
            )
        )
    end
    return nothing
end

function _validate_hetero_order(
        order::NTuple{N, Integer}, methods::Tuple{Vararg{AbstractInterpMethod, N}}
    ) where {N}
    _validate_binary_order(order)
    for d in 1:N
        if order[d] == 1 && _deriv_size(methods[d]) == 1
            throw(
                ArgumentError(
                    "axis $d uses $(methods[d]) which does not store nodal derivatives — " *
                    "only CubicInterp/QuadraticInterp axes support derivative extraction"
                )
            )
        end
    end
    return nothing
end

# ========================================
# Public API
# ========================================

"""
    nodal_partials(itp, order::NTuple{N, Integer}) -> AbstractArray{Tv, N}

Return a zero-copy view of precomputed partial derivatives at grid nodes.

The `order` tuple specifies which axes are differentiated: `0` means no
differentiation, `1` means first derivative along that axis.

# Examples
```julia
itp = interp((x, y), data)  # CubicInterpolantND (2D)

nodal_partials(itp, (0, 0))  # f(xᵢ, yⱼ)      — function values
nodal_partials(itp, (1, 0))  # ∂f/∂x(xᵢ, yⱼ)  — x-derivative at nodes
nodal_partials(itp, (0, 1))  # ∂f/∂y(xᵢ, yⱼ)  — y-derivative at nodes
nodal_partials(itp, (1, 1))  # ∂²f/∂x∂y        — mixed partial at nodes
```

For heterogeneous interpolants, only axes using `CubicInterp` or
`QuadraticInterp` support `order[d] = 1`. Requesting a derivative on a
`LinearInterp`/`ConstantInterp`/`NoInterp` axis throws `ArgumentError`.

# Supported types
- `CubicInterpolantND` — all `2^N` derivative combinations
- `QuadraticInterpolantND` — all `2^N` combinations (mixed partials are zero)
- `HeteroInterpolantND` with `PreCompute()` — per-axis validation
"""
function nodal_partials(
        itp::CubicInterpolantND{Tg, Tv, N}, order::NTuple{N, Integer}
    ) where {Tg, Tv, N}
    _validate_binary_order(order)
    p = _order_to_bitindex(order)
    return @view itp.nodal_derivs.partials[p, ntuple(_ -> :, Val(N))...]
end

function nodal_partials(
        itp::QuadraticInterpolantND{Tg, Tv, N}, order::NTuple{N, Integer}
    ) where {Tg, Tv, N}
    _validate_binary_order(order)
    p = _order_to_bitindex(order)
    return @view itp.nodal_derivs.partials[p, ntuple(_ -> :, Val(N))...]
end

# HeteroInterpolantND — PreCompute (D <: _HeteroPartials)
function nodal_partials(
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, D}, order::NTuple{N, Integer}
    ) where {Tg, Tv, N, G, S, M, E, P, D <: _HeteroPartials}
    _validate_hetero_order(order, itp.methods)
    p = _order_to_hetero_index(order, itp.methods)
    return @view itp.data.partials[p, ntuple(_ -> :, Val(N))...]
end

# ========================================
# Error Methods (unsupported types)
# ========================================

# HeteroInterpolantND — OnTheFly (D <: Array)
function nodal_partials(
        ::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array}, ::NTuple{N, Integer}
    ) where {Tg, Tv, N, G, S, M, E, P}
    throw(
        ArgumentError(
            "nodal_partials requires PreCompute() strategy — " *
            "use `interp(...; coeffs=PreCompute())` to enable nodal derivative storage"
        )
    )
end

# LinearInterpolantND
function nodal_partials(::LinearInterpolantND, ::NTuple{N, Integer}) where {N}
    throw(
        ArgumentError(
            "LinearInterpolantND does not store nodal partial derivatives"
        )
    )
end

# ConstantInterpolantND
function nodal_partials(::ConstantInterpolantND, ::NTuple{N, Integer}) where {N}
    throw(
        ArgumentError(
            "ConstantInterpolantND does not store nodal partial derivatives"
        )
    )
end

# DimensionMismatch fallback (M != N)
function nodal_partials(
        itp::AbstractInterpolantND{Tg, Tv, N}, order::NTuple{M, Integer}
    ) where {Tg, Tv, N, M}
    throw(
        DimensionMismatch(
            "order tuple has $M elements but interpolant is $N-dimensional"
        )
    )
end
