# ========================================
# ND Cubic Hermite Interpolation Types (User-Supplied Partials)
# ========================================
#
# Type definitions for ND cubic Hermite interpolation with user-supplied
# partial derivatives at grid nodes. Phase 1a: user provides every non-zero
# element of `{0, 1}^N` ("full mixed partials").
#
# Mirrors the storage conventions of `QuadraticInterpolantND` and
# `CubicInterpolantND`: data + all 2^N - 1 mixed partials are packed into a
# single `_NodalDerivativesND{Tv, N, N+1}` of shape `(2^N, n_ext_1, ..., n_ext_N)`,
# enabling reuse of the existing tensor-product `_eval_nd_cell` kernel.
#
# Future-extension note: alternative input modes (e.g. user supplies only
# first-order ∂f/∂xᵢ and the build estimates mixed terms internally) would
# reintroduce a "Completeness" type parameter on `HermitePartials` and a new
# public constructor — neither affects the current storage layout or
# evaluation kernel.

# ========================================
# HermitePartials — user-input container
# ========================================

"""
    HermitePartials{N, Tv, K, A}

User-supplied partial derivatives for ND cubic Hermite interpolation.

Holds `K` arrays in canonical mask order: `partials[m]` is the partial
derivative for mask `m ∈ 1:K`, where bit `d-1` set in `m` means
"differentiate once with respect to axis `d`". The function value itself
(mask `0`) is **not** stored here — it is supplied as the separate `data`
argument to `hermite_interp`.

Phase 1a: the only public constructor is `HermitePartials(...)`, which
populates every non-zero multiindex in `{0, 1}^N` (so `K == 2^N - 1`).

# Type parameters
- `N` : spatial dimension
- `Tv`: value element type (common eltype across all stored arrays)
- `K` : number of stored arrays
- `A` : concrete element-array type (`<: AbstractArray{Tv, N}`)
"""
struct HermitePartials{N, Tv, K, A <: AbstractArray{Tv, N}}
    partials::NTuple{K, A}
end

# ========================================
# Multiindex → mask conversion (internal)
# ========================================

# Convert a derivative-order multiindex (each entry ∈ {0, 1} for Phase 1a)
# to the canonical bitmask: bit `d-1` ← `mi[d] & 1`. Compile-time foldable.
#
# Examples (N=2):
#   (0, 0) → 0  (data slot — not stored on HermitePartials)
#   (1, 0) → 1  (∂f/∂x)
#   (0, 1) → 2  (∂f/∂y)
#   (1, 1) → 3  (∂²f/∂x∂y)
@inline _multiindex_to_mask(mi::NTuple{N, Int}) where {N} =
    sum(ntuple(d -> (mi[d] & 1) << (d - 1), Val(N)))

# ========================================
# CubicHermiteInterpolantND — the persistent interpolant
# ========================================

"""
    CubicHermiteInterpolantND{Tg, Tv, N, NP1, G, B, E, P}

N-dimensional cubic Hermite interpolant with user-supplied partial
derivatives. Stores function values and all `2^N - 1` mixed partials packed
in a single `_NodalDerivativesND{Tv, N, NP1}`, enabling O(4^N) tensor-product
cubic Hermite evaluation via the shared `_eval_nd_cell` kernel.

# Type parameters
- `Tg` : grid coordinate type
- `Tv` : value type
- `N`  : spatial dimensions
- `NP1`: `N + 1` (partials-array dimensionality)
- `G`  : tuple of wrapped axes (post-extension, post-`_cache_axis`)
- `B`  : tuple of per-axis BCs (`:exclusive → :extended` after build extension)
- `E`  : tuple of per-axis extrapolation modes
- `P`  : tuple of per-axis search policies

# Fields
- `grids`        : per-axis wrapped vectors (`_CachedRange` / `_CachedVector`)
- `nodal_derivs` : packed `(2^N, n_ext_1, ..., n_ext_N)` partials
- `bcs`          : per-axis BC tuple — `:exclusive` is promoted to `:extended`
                   after build-time extension; `:inclusive` and `NoBC` pass through
- `extraps`      : per-axis extrap policies
- `searches`     : per-axis search policies
"""
struct CubicHermiteInterpolantND{
        Tg,
        Tv,
        N,
        NP1,
        G <: Tuple{Vararg{AbstractVector, N}},
        B <: Tuple{Vararg{AbstractBC, N}},
        E <: Tuple{Vararg{AbstractExtrap, N}},
        P <: Tuple{Vararg{AbstractSearchPolicy, N}},
    } <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    nodal_derivs::_NodalDerivativesND{Tv, N, NP1}
    bcs::B
    extraps::E
    searches::P

    # Inner constructor: caller provides post-extension grids, packed
    # nodal_derivs, post-extension bcs, resolved extraps, and searches.
    # Wrapping `grids` via `_cache_axis(grid, bc, Tg)` ensures `_get_h` /
    # `_get_inv_h` are O(1) cached lookups during eval.
    function CubicHermiteInterpolantND(
            grids::Tuple{Vararg{AbstractVector{Tg}, N}},
            nodal_derivs::_NodalDerivativesND{Tv, N, NP1},
            bcs::Tuple{Vararg{AbstractBC, N}},
            extraps::Tuple{Vararg{AbstractExtrap, N}},
            searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        ) where {Tg, Tv, N, NP1}
        NP1 == N + 1 || throw(ArgumentError("NP1 must equal N+1"))
        grids_c = map((g, bc) -> _convert_copy(_cache_axis(g, bc, Tg), Tg), grids, bcs)
        return new{Tg, Tv, N, NP1, typeof(grids_c), typeof(bcs), typeof(extraps), typeof(searches)}(
            grids_c, nodal_derivs, bcs, extraps, searches,
        )
    end
end

# ========================================
# Type introspection
# ========================================

Base.ndims(::CubicHermiteInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = N
Base.size(itp::CubicHermiteInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    ntuple(d -> length(itp.grids[d]), Val(N))
Base.axes(itp::CubicHermiteInterpolantND) = itp.grids

# Convenience: number of stored partial slots per grid point (data + all
# mixed partials = 2^N).
num_partials(::CubicHermiteInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = 1 << N
num_partials(::Type{<:CubicHermiteInterpolantND{Tg, Tv, N}}) where {Tg, Tv, N} = 1 << N
