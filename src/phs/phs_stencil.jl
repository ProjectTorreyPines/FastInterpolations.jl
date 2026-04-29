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
    M = ns + N + 1
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

    # --- C block: C_i = (1, xi₁, xi₂, …, xiN) ---
    # Row N_stencil+1 is the constant-term constraint
    @inbounds for i in 1:ns
        Phi[ns + 1, i] = one(T)
        Phi[i, ns + 1] = one(T)
        for dim in 1:N
            xi_dim = T(offsets[i][dim]) * hs[dim]
            Phi[ns + 1 + dim, i] = xi_dim
            Phi[i, ns + 1 + dim] = xi_dim
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
# Full stencil precomputation for all grid nodes
# ----------------------------------------

"""
    _phs_build_all_stencils(grids, spacings, stencil_size, degree)
        -> (stencil_map, node_key)

Precompute Φ⁻¹ for every unique stencil geometry that appears across all grid nodes.

Returns:
- `stencil_map :: Dict{UInt64, Tuple{offsets, phi_inv}}` — hash → (offsets, Φ⁻¹)
- `node_key    :: Array{UInt64, N}` — maps each grid-node CartesianIndex to its hash

For a uniform grid, the interior and each boundary-layer type share the same
geometry, so `length(stencil_map) ≪ prod(grid_sizes)`.

Stencil design guarantee: every node's stencil includes offset (0,…,0), i.e. the
node itself, so that local stencil interpolants pass through their base-node data
value.  For boundary nodes the candidate set is restricted to valid grid indices,
so the stencil shifts toward the interior rather than wrapping outside the domain.
"""
function _phs_build_all_stencils(
        grids::NTuple{N, AbstractVector{Tg}},
        spacings::NTuple{N, <:AbstractGridSpacing},
        stencil_size::Int,
        degree::Int,
    ) where {N, Tg}
    grid_sizes = ntuple(d -> length(grids[d]), N)

    # Representative grid spacings (use mean h per axis for stencil geometry)
    # For uniform grids this is exact; for non-uniform it gives good neighbour selection.
    hs = ntuple(N) do d
        g = grids[d]
        Tg((last(g) - first(g)) / (length(g) - 1))
    end

    target = stencil_size^N
    R      = stencil_size   # candidate half-width: same as _phs_stencil_offsets

    # Interior offsets (for fast-path interior nodes where the full [-R,R]^N box fits)
    interior_offsets = _phs_stencil_offsets(N, stencil_size, hs)

    # Dict: hash → (offsets, phi_inv)
    stencil_map = Dict{UInt64, Tuple{Vector{NTuple{N, Int}}, Matrix{Tg}}}()

    # node_key: one UInt64 per grid node
    node_key = Array{UInt64, N}(undef, grid_sizes...)

    for ci in CartesianIndices(node_key)
        base_idx = Tuple(ci)

        # Compute per-axis physical spacing for this base node
        # (for non-uniform grids, use local h at the base node)
        hs_local = ntuple(N) do d
            n = grid_sizes[d]
            i = base_idx[d]
            if spacings[d] isa ScalarSpacing
                spacings[d].h
            else
                # Use mean of left/right intervals (or boundary one-sided)
                il = max(1, i - 1)
                ir = min(n - 1, i)
                Tg((grids[d][ir + 1] - grids[d][il]) / (ir - il + 1))
            end
        end

        # Is this node far enough from every boundary that the full [-R,R]^N box fits?
        is_interior = all(d -> base_idx[d] > R && base_idx[d] + R <= grid_sizes[d], 1:N)

        offsets = if is_interior
            # Re-use the pre-computed interior stencil directly (fast path)
            interior_offsets
        else
            # Boundary node: build candidates restricted to valid grid offsets.
            # Clamp each axis to [1 - base_idx[d], grid_sizes[d] - base_idx[d]] ∩ [-R, R].
            # This guarantees (0,…,0) is always a candidate (base node in its own stencil).
            lo_off = ntuple(d -> max(1 - base_idx[d], -R), N)
            hi_off = ntuple(d -> min(grid_sizes[d] - base_idx[d], R), N)
            ranges_per_dim = ntuple(d -> lo_off[d]:hi_off[d], N)
            candidates = vec(collect(Iterators.product(ranges_per_dim...)))
            sort!(candidates; by = off -> sum(d -> (Tg(off[d]) * hs_local[d])^2, 1:N))
            candidates[1:min(target, length(candidates))]
        end

        key = _phs_stencil_key(offsets)
        node_key[ci] = key

        if !haskey(stencil_map, key)
            # Build physical positions relative to origin (offsets * hs_local)
            phi_inv = _phs_build_phi_inv(offsets, hs_local, degree)
            stencil_map[key] = (offsets, phi_inv)
        end
    end

    return stencil_map, node_key
end
