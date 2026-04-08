# ========================================
# Nodal Partial Derivative Access (Public API)
# ========================================
#
# Provides `nodal_partials(itp, order)` for zero-copy access to precomputed
# partial derivatives at grid nodes. Abstracts over the internal indexing
# schemes (bit-encoded for homogeneous ND, mixed-radix for heterogeneous ND).

# ========================================
# Internal Helpers — Index Computation
# ========================================

@inline function _order_to_bitindex(order::NTuple{N, Integer}) where {N}
    p = 1
    for k in 1:N
        p += order[k] << (k - 1)
    end
    return p
end

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
# Internal Helpers — Per-Type Accessors
# ========================================

# --- Partials array accessor ---

@inline _partials_array(itp::CubicInterpolantND) = itp.nodal_derivs.partials
@inline _partials_array(itp::QuadraticInterpolantND) = itp.nodal_derivs.partials

@inline function _partials_array(itp::HeteroInterpolantND)
    if itp.data isa _HeteroPartials
        return itp.data.partials
    elseif _has_any_local_method(itp.methods)
        _throw_nodal_partials_hermite_nd(itp.methods)
    else
        throw(
            ArgumentError(
                "nodal_partials requires PreCompute() strategy — " *
                    "use `interp(...; coeffs=PreCompute())` to enable nodal derivative storage"
            )
        )
    end
end

# Dedicated error for the "local Hermite ND + nodal_partials" combo.
# Following the user advice to "use PreCompute" would dead-end: PreCompute
# is rejected upstream by `_validate_nd_coeffs` for Hermite family ND.
# This message points users at the correct workaround (per-axis 1D access)
# and the tracking TODO.
@noinline function _throw_nodal_partials_hermite_nd(methods)
    local_names = String[]
    for m in methods
        if m isa PchipInterp || m isa CardinalInterp || m isa AkimaInterp
            push!(local_names, string(typeof(m)))
        end
    end
    throw(
        ArgumentError(
            "nodal_partials is not yet implemented for Hermite family ND " *
                "(found $(join(unique(local_names), ", "))). The Hermite ND PreCompute " *
                "backend has not been written yet — tracked in " *
                "claudedocs/TODO/hermite_nd_precompute.md. " *
                "Workaround: compute per-axis 1D slopes via `pchip_interp` / " *
                "`cardinal_interp` / `akima_interp` with `deriv=DerivOp(1)`, " *
                "or switch the relevant axes to CubicInterp / QuadraticInterp " *
                "which do support nodal_partials."
        )
    )
end

@noinline _partials_array(::LinearInterpolantND) = throw(
    ArgumentError("LinearInterpolantND does not store nodal partial derivatives")
)

@noinline _partials_array(::ConstantInterpolantND) = throw(
    ArgumentError("ConstantInterpolantND does not store nodal partial derivatives")
)

# --- Validate + compute index ---

@inline function _validate_and_index(
        ::Union{CubicInterpolantND, QuadraticInterpolantND},
        order::NTuple{N, Integer},
    ) where {N}
    _validate_binary_order(order)
    return _order_to_bitindex(order)
end

@inline function _validate_and_index(
        itp::HeteroInterpolantND, order::NTuple{N, Integer}
    ) where {N}
    _validate_hetero_order(order, itp.methods)
    return _order_to_hetero_index(order, itp.methods)
end

# Linear/Constant: _partials_array already throws before _validate_and_index is reached,
# but define a fallback for completeness
@inline function _validate_and_index(
        ::Union{LinearInterpolantND, ConstantInterpolantND},
        order::NTuple{N, Integer},
    ) where {N}
    _validate_binary_order(order)
    return 1
end

# --- Validation helpers ---

@inline function _validate_binary_order(order::NTuple{N, Integer}) where {N}
    for d in 1:N
        o = order[d]
        (o == 0 || o == 1) || throw(
            ArgumentError("derivative order at axis $d must be 0 or 1, got $o")
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
itp = interp((x, y), data; method=CubicInterp())

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
function nodal_partials end

function nodal_partials(
        itp::AbstractInterpolantND{Tg, Tv, N}, order::NTuple{N, Integer}
    ) where {Tg, Tv, N}
    partials = _partials_array(itp)
    p = _validate_and_index(itp, order)
    return @view partials[p, ntuple(_ -> :, Val(N))...]
end

# DimensionMismatch fallback (M != N)
function nodal_partials(
        ::AbstractInterpolantND{Tg, Tv, N}, ::NTuple{M, Integer}
    ) where {Tg, Tv, N, M}
    throw(
        DimensionMismatch(
            "order tuple has $M elements but interpolant is $N-dimensional"
        )
    )
end
