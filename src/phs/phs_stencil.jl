# ========================================
# PHS Stencil Construction
# ========================================
#
# Precomputes per-stencil-geometry Φ⁻¹ matrices (inverted via Symmetric).
# At query time, coefficients are obtained by mul!(coeffs, phi_inv, rhs) — a
# single BLAS gemv, fully vectorized, rather than a triangular solve.
#
# For uniform grids (ScalarSpacing), all interior nodes share the same stencil
# geometry, so only a handful of unique Φ⁻¹ matrices are needed in total.
# Non-uniform grids may produce O(N_grid) unique geometries (documented limitation).
#
# Stencil layout:
#   The N_stencil stencil nodes are the closest in Euclidean distance (in physical
#   space, scaled by per-axis spacing h_d) to the base node.  Integer offsets are
#   stored so the physical coords can be recovered as  x_base + offset .* hs.
#
# Matrix layout of Φ (size M × M, M = N_stencil + N_dim + 1):
#   [  F   | C  ]   rows/cols  1 : N_stencil     ← RBF–RBF block
#   [  Cᵀ  | 0  ]   rows/cols  N_stencil+1 : M   ← polynomial consistency constraints
#
#   F_ij = φ(|x_i − x_j|)   (pairwise distances between stencil nodes)
#   C_i  = (1, x_i₁, x_i₂, …, x_iN)   (polynomial augmentation row i)

# ----------------------------------------
# Stencil offset selection
# ----------------------------------------

"""
    _phs_stencil_offsets(N, stencil_size, hs::NTuple{N,T}) -> Vector{NTuple{N,Int}}

Choose the `stencil_size^N` integer offsets closest to the origin in
physical (scaled) Euclidean distance, centred on the base node.

Candidate pool: all offsets in the box `[-R, R]^N` where R is chosen so
that the pool is large enough to guarantee `stencil_size^N` candidates:
    R = stencil_size  (gives (2R+1)^N candidates, always ≥ stencil_size^N)
"""
function _phs_stencil_offsets(N::Int, stencil_size::Int, hs::NTuple)
    R = stencil_size  # half-width of candidate box
    # Build all offsets in [-R, R]^N
    ranges = ntuple(_ -> -R:R, N)
    candidates = vec(collect(Iterators.product(ranges...)))  # Vector{NTuple{N,Int}}

    # Sort by scaled Euclidean distance from origin
    target = stencil_size^N
    sort!(candidates; by = off -> sum(d -> (off[d] * hs[d])^2, 1:N))

    # Keep closest target; include the origin (offset == 0), which is always first
    return candidates[1:min(target, length(candidates))]
end

# ----------------------------------------
# Φ matrix construction and inversion
# ----------------------------------------

"""
    _phs_build_phi_inv(offsets, hs, degree) -> Matrix{T}

Build the (N_stencil + N_dim + 1) × (N_stencil + N_dim + 1) collocation matrix Φ
and return its inverse via `inv(Symmetric(Φ))`.

Φ is symmetric positive definite (for well-separated distinct stencil nodes),
so `Symmetric` tells LinearAlgebra to exploit that structure.
"""
function _phs_build_phi_inv(
        offsets::Vector{<:NTuple{N, Int}},
        hs::NTuple{N, T},
        degree::Int,
    ) where {N, T <: AbstractFloat}
    ns = length(offsets)
    poly_deg = (degree - 1) ÷ 2
    poly_exps = _phs_all_exponents(Val(N), poly_deg)  # same ordering as _phs_poly_exps_tuple
    n_poly = length(poly_exps)
    M = ns + n_poly
    Phi = zeros(T, M, M)

    # --- F block: Fij = φ(|x_i − x_j|) ---
    @inbounds for j in 1:ns, i in j:ns
        if i == j
            # φ(0) = 0 for odd-degree PHS, so diagonal is 0 — no need to set
        else
            d2 = zero(T)
            for dim in 1:N
                Δ = T(offsets[i][dim] - offsets[j][dim]) * hs[dim]
                d2 += Δ * Δ
            end
            f = _phs_phi(sqrt(d2), degree)
            Phi[i, j] = f
            Phi[j, i] = f
        end
    end

    # --- C block: polynomial basis at each stencil node ---
    # poly_exps ordering must match _phs_poly_exps_tuple used at eval time
    @inbounds for i in 1:ns
        xi = ntuple(d -> T(offsets[i][d]) * hs[d], N)
        for (k, α) in enumerate(poly_exps)
            val = one(T)
            for d in 1:N
                α[d] != 0 && (val *= xi[d]^α[d])
            end
            Phi[ns + k, i] = val
            Phi[i, ns + k] = val
        end
    end

    # Lower-right 0-block is already zero from initialization
    return inv(LinearAlgebra.Symmetric(Phi))
end

# ----------------------------------------
# Stencil deduplication key
# ----------------------------------------

"""
    _phs_stencil_key(offsets) -> UInt64

Hash a vector of offset NTuples to a UInt64 for Dict lookup.
Uses Julia's built-in hash (polynomial + prime, collision-safe for small N_stencil).
"""
@inline _phs_stencil_key(offsets::Vector) = hash(offsets)

# ----------------------------------------
# Clamped stencil for boundary nodes
# ----------------------------------------

"""
    _phs_clamp_offsets(offsets, base_idx, grid_sizes) -> Vector{NTuple{N,Int}}

For a base node at `base_idx`, clamp each stencil offset so that the resulting
absolute index stays within `1:grid_sizes[d]` for every dimension.

This shifts the stencil at boundary nodes so all nodes remain inside the domain
(the stencil is no longer centred at the base node near boundaries, but it
remains a valid local interpolation stencil).
"""
function _phs_clamp_offsets(
        offsets::Vector{<:NTuple{N, Int}},
        base_idx::NTuple{N, Int},
        grid_sizes::NTuple{N, Int},
    ) where {N}
    # Compute the bounding box of unclamped indices per dimension
    # and the shift needed to bring them inside [1, grid_sizes[d]]
    lo = ntuple(d -> minimum(off -> base_idx[d] + off[d], offsets), N)
    hi = ntuple(d -> maximum(off -> base_idx[d] + off[d], offsets), N)

    shift = ntuple(N) do d
        s = 0
        lo[d] + s < 1 && (s = 1 - lo[d])
        hi[d] + s > grid_sizes[d] && (s = grid_sizes[d] - hi[d])
        s
    end

    any(!=(0), shift) || return offsets   # no clamping needed
    return [ntuple(d -> off[d] + shift[d], N) for off in offsets]
end

# ----------------------------------------
# Shift computation (O(N), used at eval time)
# ----------------------------------------

"""
    _phs_compute_shift(base_idx, stencil_lo, stencil_hi, grid_sizes) -> NTuple{N,Int}

Compute the per-axis effective clip (clamped offset floor) for boundary nodes.
For each axis d, returns `max(0, 1 - base_idx[d])` (left clip) or
`min(0, grid_sizes[d] - base_idx[d] - stencil_hi[d])` (right clip):
i.e., how many canonical negative offsets would go out-of-bounds to the left,
and how far positive offsets exceed the grid to the right.

Returns `(0,...,0)` for interior nodes; non-zero otherwise.
O(N) — no allocation.
"""
@inline function _phs_compute_shift(
        base_idx::NTuple{N, Int},
        stencil_lo::NTuple{N, Int},
        stencil_hi::NTuple{N, Int},
        grid_sizes::NTuple{N, Int},
    ) where {N}
    return ntuple(N) do d
        lo_abs = base_idx[d] + stencil_lo[d]   # absolute index of leftmost offset
        hi_abs = base_idx[d] + stencil_hi[d]   # absolute index of rightmost offset
        lo_clip = lo_abs < 1 ? 1 - lo_abs : 0  # how far the left edge exceeds the boundary
        hi_clip = hi_abs > grid_sizes[d] ? hi_abs - grid_sizes[d] : 0  # how far right edge exceeds
        lo_clip > 0 ? lo_clip : (hi_clip > 0 ? -hi_clip : 0)
    end
end

# ----------------------------------------
# Boundary shift cache
# ----------------------------------------

"""
    _phs_build_boundary_shift_cache(canonical_offsets, hs, degree)
        -> Dict{NTuple{N,Int}, Tuple{Vector{NTuple{N,Int}}, Matrix{Tg}}}

Precompute Φ⁻¹ for every unique boundary shift vector (stencil shifted as a
block so it stays inside the grid).  Only built if the estimated total memory
is ≤ 100 MB; otherwise returns an empty Dict and boundary nodes fall back to
the canonical Φ⁻¹ (Fortran approach, acceptable when queries are interior).

For small-to-moderate stencils (stencil_size ≤ 6, N ≤ 3) the cache is always
built and guarantees exact polynomial reproduction everywhere.
"""
function _phs_build_boundary_shift_cache(
        canonical_offsets::Vector{<:NTuple{N, Int}},
        hs::NTuple{N, Tg},
        degree::Int,
        stencil_size::Int,
    ) where {N, Tg}
    ns  = length(canonical_offsets)
    poly_deg = (degree - 1) ÷ 2
    n_poly   = length(_phs_all_exponents(Val(N), poly_deg))
    M        = ns + n_poly
 
    R = stencil_size   # canonical half-width
 
    min_off = ntuple(d -> minimum(off -> off[d], canonical_offsets), N)
    max_off = ntuple(d -> maximum(off -> off[d], canonical_offsets), N)
 
    # Unique clip amounts per dimension:
    # lo_clip in [0, -min_off[d]], hi_clip in [0, max_off[d]]
    # We encode as clip = lo_clip if lo_clip > 0 else -hi_clip
    clip_options = ntuple(N) do d
        left  = -min_off[d] > 0 ? collect(1:-min_off[d]) : Int[]
        right = max_off[d]  > 0 ? collect(-max_off[d]:-1) : Int[]
        [0; left; right]
    end
 
    # Estimate total cache size
    n_shifts     = prod(length, clip_options) - 1
    mem_estimate = n_shifts * M * M * sizeof(Tg)
 
    cache = Dict{NTuple{N, Int}, Tuple{Vector{NTuple{N, Int}}, Matrix{Tg}, Vector{NTuple{N, Tg}}}}()
    mem_estimate > 100_000_000 && return cache
 
    target = ns   # keep same number of stencil nodes
    for clip_combo in Iterators.product(clip_options...)
        clip = NTuple{N, Int}(clip_combo)
        all(iszero, clip) && continue   # canonical stored separately
 
        # Valid offset range for this clip pattern.
        # clip[d] > 0 means left boundary: abs offsets must be ≥ -R+clip[d], i.e. lo = -R+clip[d] ... R
        # clip[d] < 0 means right boundary: abs offsets must be ≤ R+clip[d], i.e. lo = -R ... R+clip[d]
        lo_off = ntuple(d -> clip[d] > 0 ? min_off[d] + clip[d] : min_off[d], N)
        hi_off = ntuple(d -> clip[d] < 0 ? max_off[d] + clip[d] : max_off[d], N)
        ranges_per_dim = ntuple(d -> lo_off[d]:hi_off[d], N)
        candidates = vec(collect(Iterators.product(ranges_per_dim...)))
        sort!(candidates; by = off -> sum(d -> (Tg(off[d]) * hs[d])^2, 1:N))
        valid_offsets = candidates[1:min(target, length(candidates))]
        shifted_phys_offsets = [ntuple(d -> Tg(off[d]) * hs[d], Val(N)) for off in valid_offsets]
 
        cache[clip] = (valid_offsets, _phs_build_phi_inv(valid_offsets, hs, degree), shifted_phys_offsets)
    end
 
    return cache
end

# ----------------------------------------
# Single canonical stencil (+ boundary shift cache)
# ----------------------------------------

"""
    _phs_build_stencil(grids, spacings, stencil_size, degree)
        -> (offsets, phi_inv, hs, stencil_lo, stencil_hi, shift_cache)

Build the canonical stencil and precompute all boundary shift variants.

Returns:
- `offsets      :: Vector{NTuple{N,Int}}` — canonical offsets
- `phi_inv      :: Matrix{Tg}` — canonical Φ⁻¹
- `hs           :: NTuple{N,Tg}` — mean grid spacing per axis
- `stencil_lo   :: NTuple{N,Int}` — per-axis min canonical offset
- `stencil_hi   :: NTuple{N,Int}` — per-axis max canonical offset
- `shift_cache  :: Dict{NTuple{N,Int}, ...}` — shifted (offsets, Φ⁻¹) per boundary shift
"""
function _phs_build_stencil(
        grids::NTuple{N, AbstractVector{Tg}},
        spacings::NTuple{N, <:AbstractGridSpacing},
        stencil_size::Int,
        degree::Int,
    ) where {N, Tg}

    # Mean h per axis (for uniform grids this is exact)
    hs = ntuple(N) do d
        g = grids[d]
        Tg((last(g) - first(g)) / (length(g) - 1))
    end

    offsets     = _phs_stencil_offsets(N, stencil_size, hs)
    phi_inv     = _phs_build_phi_inv(offsets, hs, degree)
    stencil_lo  = ntuple(d -> minimum(off -> off[d], offsets), N)
    stencil_hi  = ntuple(d -> maximum(off -> off[d], offsets), N)
    shift_cache = _phs_build_boundary_shift_cache(offsets, hs, degree, stencil_size)

    return offsets, phi_inv, hs, stencil_lo, stencil_hi, shift_cache
end

