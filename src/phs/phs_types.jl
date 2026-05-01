# ========================================
# PHSInterpolantND Type Definition
# ========================================
#
# N-dimensional polyharmonic spline (PHS) interpolant with:
#   - Local stencil-based radial basis function interpolation
#   - Weighted blending across neighbouring base-node interpolants for C² continuity
#   - Optional log-density smoothing transformation (f = ln(ρ/ρ₀))
#
# Type Parameters:
#   Tg  — Grid/coordinate float type (Float32 or Float64)
#   Tv  — Value type (Float, Complex{Float}, any duck-typed scalar)
#   N   — Number of dimensions
#   K   — PHS degree (compile-time Int for Val{K} dispatch to radial kernels)

"""
    PHSLogTransform{N, Tr}

Optional log-density smoothing transformation container.
When active, `data` stored in `PHSInterpolantND` contains `log(ρ/ρ₀)` and
evaluation applies the inverse transform (Eqs. 21–23 from the paper) to
recover the interpolated density and its derivatives.

# Fields
- `reference`: An `AbstractInterpolantND` providing ρ₀(x), ∇ρ₀(x), and ∇²ρ₀(x)
"""
struct PHSLogTransform{N, Tr <: AbstractInterpolantND}
    reference::Tr
end

"""
    PHSInterpolantND{Tg, Tv, N, K}

N-dimensional local polyharmonic spline interpolant.

Implements the method from the paper, combining:
1. Local stencil-based PHS interpolation (φ(r) = r^K, K odd)
2. Weighted blending across neighbouring base-node interpolants for C² continuity
3. Optional log-density smoothing transform

A single canonical stencil geometry (and its Φ⁻¹) is precomputed once from the
mean grid spacings.  At boundary nodes the same Φ⁻¹ is reused with clamped data
indices — identical to the reference Fortran implementation.

# Type Parameters
- `Tg`: Grid float type
- `Tv`: Value type (Float, Complex, duck-typed scalar)
- `N`:  Number of dimensions
- `K`:  PHS degree (1, 3, 5, …)

# Fields
- `grids`:           Per-axis grid vectors
- `spacings`:        Per-axis spacing (ScalarSpacing for uniform, VectorSpacing for non-uniform)
- `data`:            N-D data array (or `log(ρ/ρ₀)` when transform is active)
- `stencil_offsets`: Single canonical stencil: `stencil_size^N` integer offsets from origin
- `phi_inv`:         Single Φ⁻¹ matrix for the canonical stencil
- `hs`:              Per-axis mean grid spacing used to build `stencil_offsets`/`phi_inv`
- `blend_a`:         Blending range parameter (≥ max grid spacing × blend_factor)
- `blend_r_idx`:     Per-axis half-width of blend neighbourhood in index space
- `transform`:       Nothing, or PHSLogTransform for log-density mode
- `extraps`:         Per-axis extrapolation modes
- `searches`:        Per-axis search policies (used for OOB checking only)

# Performance
- **Construction**: O(M³) for one Φ⁻¹ (M = stencil_size^N + N + 1)
- **Query**: O(n_blend × N_stencil × M) where n_blend = number of neighbours within blend_a
- **Memory**: O(M²) for Φ⁻¹ plus O(prod(grid_sizes)) for data

# Thread-Safety
Immutable after construction; safe for concurrent read access.
All mutable workspace lives in thread-local `AdaptiveArrayPools`.
"""
struct PHSInterpolantND{
        Tg,
        Tv,
        N,
        K,
        G <: Tuple{Vararg{AbstractVector, N}},
        S <: Tuple{Vararg{AbstractGridSpacing, N}},
        T,   # Nothing or PHSLogTransform
        E <: Tuple{Vararg{AbstractExtrap, N}},
        P <: Tuple{Vararg{AbstractSearchPolicy, N}},
    } <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    spacings::S
    data::Array{Tv, N}
    stencil_offsets::Vector{NTuple{N, Int}}   # canonical stencil (stencil_size^N offsets)
    phi_inv::Matrix{Tg}                       # canonical Φ⁻¹ (shift = 0, used for interior nodes)
    stencil_lo::NTuple{N, Int}                # per-axis min canonical offset (for fast shift computation)
    stencil_hi::NTuple{N, Int}                # per-axis max canonical offset
    shift_cache::Dict{NTuple{N, Int}, Tuple{Vector{NTuple{N, Int}}, Matrix{Tg}}}  # boundary shift variants
    hs::NTuple{N, Tg}                         # mean grid spacing per axis
    blend_a::Tg
    blend_r_idx::NTuple{N, Int}   # ceil(blend_a / h_min) per axis
    transform::T
    extraps::E
    searches::P
end

# ----------------------------------------
# Type Introspection (protocol with AbstractInterpolantND)
# ----------------------------------------

@inline Base.size(itp::PHSInterpolantND) = map(length, itp.grids)
@inline Base.ndims(::PHSInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = N
@inline grid_type(::PHSInterpolantND{Tg}) where {Tg} = Tg
@inline value_type(::PHSInterpolantND{Tg, Tv}) where {Tg, Tv} = Tv
@inline eval_type(itp::PHSInterpolantND{Tg, Tv}) where {Tg, Tv} = promote_type(Tv, Tg)

# Required by AbstractInterpolantND — expose per-axis grid/extrap/search
@inline _grid(itp::PHSInterpolantND, ::Val{D}) where {D} = itp.grids[D]
@inline _extrap(itp::PHSInterpolantND, ::Val{D}) where {D} = itp.extraps[D]
@inline _search(itp::PHSInterpolantND, ::Val{D}) where {D} = itp.searches[D]
@inline Base.axes(itp::PHSInterpolantND) = itp.grids
