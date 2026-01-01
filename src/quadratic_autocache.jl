# ========================================
# Quadratic Autocache (RCU Pattern)
# ========================================
#
# Automatic caching layer for quadratic spline interpolation.
# Simpler than cubic: no BC-dependent cache variations.
#
# The cache stores x-grid preprocessing (h, inv_h) which is
# independent of the BC type. BC affects coefficient computation,
# not the cache itself.

# ===============================================================
# Cache Entry Type
# ===============================================================

"""
    QuadraticCacheEntry{T, X}

Cache entry for quadratic spline interpolation.

# Fields
- `id::UInt`: Object identity for fast lookup
- `x::X`: Grid points for equality check
- `cache::QuadraticSplineCache{T,X}`: Cached preprocessing result
"""
mutable struct QuadraticCacheEntry{T<:AbstractFloat, X<:AbstractVector{T}}
    id::UInt
    x::X
    cache::QuadraticSplineCache{T, X}
end

# ===============================================================
# RCU Snapshot (Immutable Bank State)
# ===============================================================

"""
    QuadraticBankSnapshot{E}

Immutable snapshot of bank state for RCU pattern.
"""
struct QuadraticBankSnapshot{E<:QuadraticCacheEntry}
    store::Vector{E}
    count::Int
    ring::Int  # 1-based next eviction index
end

# Empty snapshot constructor
function QuadraticBankSnapshot{E}(cache_size::Int) where {E<:QuadraticCacheEntry}
    store = E[]
    sizehint!(store, cache_size)
    QuadraticBankSnapshot{E}(store, 0, 1)
end

# ===============================================================
# Generic Cache Bank (RCU-style)
# ===============================================================

"""
    QuadraticCacheBank{E}

Cache bank for quadratic spline entries using RCU pattern.

# Thread Safety
- Read path: Lock-free via `@atomic :acquire` snapshot load
- Write path: Lock + copy-on-write + `@atomic :release` publish
"""
mutable struct QuadraticCacheBank{E<:QuadraticCacheEntry}
    @atomic snapshot::QuadraticBankSnapshot{E}
end

function QuadraticCacheBank{E}(cache_size::Int) where {E<:QuadraticCacheEntry}
    QuadraticCacheBank{E}(QuadraticBankSnapshot{E}(cache_size))
end

# ===============================================================
# RCU Registry (Lock-Free Bank Lookup)
# ===============================================================

"""
    QuadraticRegistrySnapshot

Type alias for the registry's immutable snapshot.
Simple association list: [(BankType, BankInstance), ...]
"""
const QuadraticRegistrySnapshot = Vector{Pair{DataType, Any}}

"""
    QuadraticGlobalRegistry

Atomic registry wrapper for RCU-style lock-free bank lookup.
"""
mutable struct QuadraticGlobalRegistry
    @atomic snapshot::QuadraticRegistrySnapshot
end

QuadraticGlobalRegistry() = QuadraticGlobalRegistry(QuadraticRegistrySnapshot())

# Global registry and lock
const _QUADRATIC_REGISTRY = QuadraticGlobalRegistry()
const _QUADRATIC_CACHE_LOCK = ReentrantLock()

# Load-time constant for cache size
const _QUADRATIC_CACHE_SIZE = @load_preference("quadratic_cache_size", 16)::Int

# ===============================================================
# Public API
# ===============================================================

"""
    set_quadratic_cache_size!(n::Int)

Set maximum cache size for quadratic splines for future Julia sessions.

**Requires Julia restart** to take effect.
"""
function set_quadratic_cache_size!(n::Int)
    n > 0 || throw(ArgumentError("Cache size must be positive"))
    @set_preferences!("quadratic_cache_size" => n)
    @info "Quadratic cache size preference saved. Restart Julia to apply (current session uses $(get_quadratic_cache_size()))."
    return n
end

"""
    get_quadratic_cache_size()

Get current maximum cache size for quadratic splines.
"""
get_quadratic_cache_size() = _QUADRATIC_CACHE_SIZE

"""
    clear_quadratic_cache!()

Clear all cached quadratic x-grids.

# Thread Safety (RCU)
Atomically replaces registry snapshot with empty vector.
"""
function clear_quadratic_cache!()
    lock(_QUADRATIC_CACHE_LOCK) do
        @atomic :release _QUADRATIC_REGISTRY.snapshot = QuadraticRegistrySnapshot()
    end
    return nothing
end

# ===============================================================
# Internal: RCU Registry Lookup
# ===============================================================

"""
Lock-free registry lookup using RCU pattern.
"""
@inline function _quadratic_registry_lookup(registry::QuadraticGlobalRegistry, ::Type{B}) where {B}
    snap = @atomic :acquire registry.snapshot

    # Linear scan (N < 20 → faster than hash lookup)
    for (TypeKey, Bank) in snap
        if TypeKey === B
            return Bank::B
        end
    end

    return nothing
end

"""
RCU-style bank retrieval with copy-on-write for new bank creation.
"""
@inline function _get_quadratic_bank(registry::QuadraticGlobalRegistry, ::Type{QuadraticCacheBank{E}}) where {E<:QuadraticCacheEntry}
    BankType = QuadraticCacheBank{E}

    # === RCU Read Path (Lock-Free) ===
    bank = _quadratic_registry_lookup(registry, BankType)
    bank !== nothing && return bank

    # === RCU Write Path (Lock + Copy-on-Write) ===
    lock(_QUADRATIC_CACHE_LOCK)
    try
        # Double-check after acquiring lock
        bank = _quadratic_registry_lookup(registry, BankType)
        bank !== nothing && return bank

        # Create new bank
        new_bank = QuadraticCacheBank{E}(_QUADRATIC_CACHE_SIZE)

        # Copy-on-write: copy snapshot → push → publish
        old_snap = @atomic :monotonic registry.snapshot
        new_snap = copy(old_snap)
        push!(new_snap, BankType => new_bank)
        @atomic :release registry.snapshot = new_snap

        return new_bank
    finally
        unlock(_QUADRATIC_CACHE_LOCK)
    end
end

# ===============================================================
# Internal: RCU Lookup (Lock-Free Read Path)
# ===============================================================

"""
Lock-free cache lookup on an immutable snapshot.
Uses 2-pass algorithm: identity check (fast) → equality check (slow).
"""
@inline function _quadratic_rcu_lookup(snap::QuadraticBankSnapshot{E}, id::UInt, x::X) where {E, X}
    store = snap.store
    count = snap.count

    # [Pass 1] Identity check (fast path)
    @inbounds for i in 1:count
        entry = store[i]
        if entry.id === id
            return entry.cache
        end
    end

    # [Pass 2] Equality check (snapshot is immutable, no self-healing)
    @inbounds for i in 1:count
        entry = store[i]
        if isequal(entry.x, x)
            return entry.cache
        end
    end

    return nothing
end

# ===============================================================
# Internal: Lookup/Insert (RCU Pattern)
# ===============================================================

"""
Core lookup/insert logic using RCU pattern.
"""
@inline function _quadratic_lookup_or_insert!(bank::QuadraticCacheBank{E}, x::X) where {E<:QuadraticCacheEntry, X}
    id = objectid(x)

    # === RCU Read Path (Lock-Free) ===
    snap = @atomic :acquire bank.snapshot
    found = _quadratic_rcu_lookup(snap, id, x)
    found !== nothing && return found

    # === RCU Write Path (Lock + Copy-on-Write) ===
    lock(_QUADRATIC_CACHE_LOCK)
    try
        # Double-check after acquiring lock
        snap = @atomic :monotonic bank.snapshot
        found = _quadratic_rcu_lookup(snap, id, x)
        found !== nothing && return found

        # Build new cache
        new_cache = QuadraticSplineCache(x)
        new_entry = E(id, x, new_cache)

        # Copy-on-write: create new snapshot with added entry
        new_store = copy(snap.store)
        new_count = snap.count
        new_ring = snap.ring

        if new_count < _QUADRATIC_CACHE_SIZE
            push!(new_store, new_entry)
            new_count += 1
        else
            # Ring buffer eviction
            new_store[new_ring] = new_entry
            new_ring = (new_ring % _QUADRATIC_CACHE_SIZE) + 1
        end

        # Atomic publish
        new_snap = QuadraticBankSnapshot{E}(new_store, new_count, new_ring)
        @atomic :release bank.snapshot = new_snap

        return new_cache
    finally
        unlock(_QUADRATIC_CACHE_LOCK)
    end
end

# ===============================================================
# Internal API: _get_quadratic_cache
# ===============================================================

"""
    _get_quadratic_cache(x; autocache=true) -> QuadraticSplineCache

Get or create a cached QuadraticSplineCache for the given x-grid.

# Arguments
- `x`: Grid points
- `autocache::Bool=true`: If false, always create new cache (no caching)

# Thread Safety
- Lock-free on cache hit (RCU pattern)
- Lock-protected on cache miss
"""
@inline function _get_quadratic_cache(x::X; autocache::Bool=true) where {T<:AbstractFloat, X<:AbstractVector{T}}
    # Skip caching if disabled
    if !autocache
        return QuadraticSplineCache(x)
    end

    # Get bank for this (T, X) combination
    EntryType = QuadraticCacheEntry{T, X}
    bank = _get_quadratic_bank(_QUADRATIC_REGISTRY, QuadraticCacheBank{EntryType})

    # Lookup or insert
    return _quadratic_lookup_or_insert!(bank, x)
end

# Fallback for non-float element types (convert to Float64)
@inline function _get_quadratic_cache(x::X; autocache::Bool=true) where {X<:AbstractVector}
    T = eltype(x)
    if T <: AbstractFloat
        return _get_quadratic_cache(x; autocache=autocache)
    else
        # Convert to Float64
        x_float = Float64.(x)
        return _get_quadratic_cache(x_float; autocache=autocache)
    end
end
