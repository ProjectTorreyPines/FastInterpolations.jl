# ========================================
# HermitePartials — multiindex-keyed user-input constructor
# ========================================
#
# Public entry for assembling a `HermitePartials{N, Tv, K, A}` from user
# `multiindex => array` pairs. All validation + element-type promotion lives
# here, so downstream code (build, eval) can assume a well-formed container
# with a single common `Tv`.

"""
    HermitePartials(pairs::Pair{<:NTuple{N, Int}, <:AbstractArray}...) -> HermitePartials

Construct user-supplied full mixed partial derivatives for ND cubic Hermite
interpolation.

Each `pair` is `multiindex => array`, where `multiindex::NTuple{N, Int}` encodes
the derivative order per axis. For Phase 1a, each entry must be `0` or `1`.

# Required pairs (count `2^N - 1`)
Exactly the non-zero multiindices in `{0, 1}^N`. Order is irrelevant — the
constructor reorders to canonical mask order internally. The zero multiindex
`(0, …, 0)` (function value) is **not** allowed here; it is supplied via the
`data` argument of `hermite_interp`.

# Element-type handling
Arrays may have different element types. A common `Tv` is derived via
`promote_type` across all pairs, and each array is converted to
`Array{Tv, N}` (zero-copy when already matching). The data argument of
`hermite_interp` is promoted separately, then a final round of promotion
across (grid, data, partials) yields the storage `Tv`.

# Example (N=2)
```julia
partials = HermitePartials(
    (1, 0) => dfdx,
    (0, 1) => dfdy,
    (1, 1) => d2fdxdy,
)
```
"""
function HermitePartials(pairs::Pair{<:NTuple{N, Int}, <:AbstractArray}...) where {N}
    N >= 1 || throw(ArgumentError("HermitePartials requires N >= 1"))
    K = (1 << N) - 1
    length(pairs) == K || throw(
        ArgumentError(
            "HermitePartials{N=$N} requires exactly $K (mask=1..$K) partials, got $(length(pairs))",
        )
    )

    _validate_hermite_full_pairs(Val(N), pairs)

    # Common element type across all arrays.
    Tv = promote_type(map(p -> eltype(last(p)), pairs)...)

    # Reorder by mask + coerce to a single concrete eltype/Array container.
    ordered = _reorder_pairs_by_mask(Val(N), pairs, Tv)

    A = typeof(first(ordered))
    return HermitePartials{N, Tv, K, A}(ordered)
end

# ========================================
# Validation helpers
# ========================================

@noinline function _throw_bad_multiindex_entry(mi::NTuple{N, Int}, slot::Int, val::Int) where {N}
    throw(
        ArgumentError(
            "HermitePartials: multiindex $(mi) has entry $val at axis $slot; " *
                "Phase 1a requires every entry ∈ {0, 1}",
        )
    )
end

@noinline function _throw_zero_multiindex(N::Int)
    throw(
        ArgumentError(
            "HermitePartials: multiindex $(ntuple(_ -> 0, N)) (function value) " *
                "is not allowed — pass data as the `data` argument to `hermite_interp` instead",
        )
    )
end

@noinline function _throw_duplicate_multiindex(mi)
    throw(ArgumentError("HermitePartials: duplicate multiindex $(mi)"))
end

@noinline function _throw_missing_multiindex(mi, mask::Int)
    throw(
        ArgumentError(
            "HermitePartials: missing multiindex $(mi) (mask=$mask); " *
                "supply every non-zero entry of {0,1}^N",
        )
    )
end

@noinline function _throw_size_mismatch_partials(mask_ref, sz_ref, mask_cur, sz_cur)
    throw(
        DimensionMismatch(
            "HermitePartials: array for mask=$mask_cur has size $sz_cur, " *
                "but mask=$mask_ref has size $sz_ref — all partials must match",
        )
    )
end

function _validate_hermite_full_pairs(
        ::Val{N},
        pairs::NTuple{K, Pair{<:NTuple{N, Int}, <:AbstractArray}},
    ) where {N, K}
    K == (1 << N) - 1 || throw(
        ArgumentError(
            "HermitePartials{N=$N}: expected $((1 << N) - 1) pairs, got $K",
        )
    )
    seen = fill(false, 1 << N)   # index 1 ↔ mask 0 (function-value sentinel; never set here)
    sz_ref = size(last(pairs[1]))
    mask_ref = _multiindex_to_mask(first(pairs[1]))
    @inbounds for (k, p) in enumerate(pairs)
        mi = first(p)
        arr = last(p)
        for d in 1:N
            entry = mi[d]
            (entry == 0 || entry == 1) || _throw_bad_multiindex_entry(mi, d, entry)
        end
        m = _multiindex_to_mask(mi)
        m == 0 && _throw_zero_multiindex(N)
        seen[m + 1] && _throw_duplicate_multiindex(mi)
        seen[m + 1] = true
        ndims(arr) == N || throw(
            DimensionMismatch(
                "HermitePartials: array for multiindex $(mi) is $(ndims(arr))-D, expected $N-D",
            )
        )
        if k > 1
            sz_cur = size(arr)
            sz_cur == sz_ref || _throw_size_mismatch_partials(mask_ref, sz_ref, m, sz_cur)
        end
    end
    @inbounds for m in 1:((1 << N) - 1)
        if !seen[m + 1]
            mi = ntuple(d -> (m >> (d - 1)) & 1, Val(N))
            _throw_missing_multiindex(mi, m)
        end
    end
    return nothing
end

# Reorder pairs into canonical mask order (1, 2, ..., 2^N-1) and coerce each
# array to `Array{Tv, N}`. `convert` is a no-op when the input already
# satisfies the target type (zero-copy alias); otherwise allocates one new
# dense Array per mismatching entry. Wrapper types (OffsetArray, SubArray,
# ReshapedArray, ...) are materialized to plain `Array` because the
# downstream packed buffer uses 1-based indexing.
function _reorder_pairs_by_mask(
        ::Val{N},
        pairs::NTuple{K, Pair{<:NTuple{N, Int}, <:AbstractArray}},
        ::Type{Tv},
    ) where {N, K, Tv}
    slots = Vector{Array{Tv, N}}(undef, K)
    @inbounds for p in pairs
        m = _multiindex_to_mask(first(p))
        slots[m] = convert(Array{Tv, N}, last(p))
    end
    return ntuple(i -> slots[i], Val(K))
end
