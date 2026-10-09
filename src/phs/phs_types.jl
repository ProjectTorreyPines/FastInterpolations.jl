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
- `data`:            N-D data array (or `log(ρ/ρ₀)` when transform is active)
- `stencil_offsets`: Single canonical stencil: `stencil_size^N` integer offsets from origin
- `phi_inv`:         Single Φ⁻¹ matrix for the canonical stencil
- `hs`:              Per-axis mean grid spacing used to build `stencil_offsets`/`phi_inv`
- `blend_a`:         Blending range parameter (≥ max grid spacing × blend_factor)
- `blend_r_idx`:     Per-axis half-width of blend neighbourhood in index space
- `transform`:       `nothing`, or the log-transform state built from `log_reference`
- `extraps`:         Per-axis extrapolation modes
- `searches`:        Per-axis search policies (used for OOB checking only)

# Performance
- **Construction**: O(M³) for one Φ⁻¹ (M = stencil_size^N + N + 1)
- **Query**: O(n_blend × N_stencil × M) where n_blend = number of neighbours within blend_a
- **Memory**: O(M²) for Φ⁻¹ plus O(prod(grid_sizes)) for data

# Thread-Safety
Safe to call concurrently from externally-threaded loops: the mutable per-thread
coefficient cache is indexed by `Threads.threadid()` so each thread uses its own slot.
"""
struct PHSInterpolantND{
        Tg,
        Tv,
        N,
        K,
        G <: Tuple{Vararg{AbstractVector, N}},
        T,   # Nothing or _PHSLogTransform
        E <: Tuple{Vararg{AbstractExtrap, N}},
        P <: Tuple{Vararg{AbstractSearchPolicy, N}},
    } <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    data::Array{Tv, N}
    stencil_offsets::Vector{NTuple{N, Int}}   # canonical stencil (stencil_size^N offsets)
    stencil_phys_offsets::Vector{NTuple{N, Tg}} # precomputed physical coordinate offsets
    phi_inv::Matrix{Tg}                       # canonical Φ⁻¹ (shift = 0, used for interior nodes)
    stencil_lo::NTuple{N, Int}                # per-axis min canonical offset (for fast shift computation)
    stencil_hi::NTuple{N, Int}                # per-axis max canonical offset
    shift_cache::Dict{NTuple{N, Int}, Tuple{Vector{NTuple{N, Int}}, Matrix{Tg}, Vector{NTuple{N, Tg}}}}  # boundary shift variants
    hs::NTuple{N, Tg}                         # mean grid spacing per axis
    blend_a::Tg
    blend_a3::Tg                  # blend_a^3, precomputed for weight function kernels
    blend_r_idx::NTuple{N, Int}   # ceil(blend_a / h_d) per axis (h_d = that axis's spacing)
    transform::T
    extraps::E
    searches::P
    coeff_caches::Vector{Dict{NTuple{N, Int}, Vector{Tg}}}
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
