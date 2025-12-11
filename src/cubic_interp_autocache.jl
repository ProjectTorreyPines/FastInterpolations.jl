"""
    cubic_interp_autocache.jl

Automatic caching layer for cubic spline interpolation.
Transparently reuses LU factorization for repeated x-grids.

# Implementation Details

- True LRU eviction with timestamp-based ordering
- Per-bucket size limit to prevent hash collision DoS
- Type-parametric buckets for Float32/Float64 compatibility
- Defensive x-grid copying in CubicSplineCache constructor
- Thread-safe with ReentrantLock on cache access
"""

# ===============================================================
# Cache Infrastructure
# ===============================================================

# Cache handle for LRU tracking
struct CacheHandle{T<:AbstractFloat}
    cache::CubicSplineCache{T}
    timestamp::UInt64
    hash::UInt64
end

# Global cache store: hash -> Vector{CacheHandle}
const _CUBIC_CACHE_STORE = Dict{UInt64, Vector{CacheHandle}}()
const _CACHE_LOCK = ReentrantLock()
const _CACHE_SIZE_REF = Ref{Int}(16)  # Max total caches across all buckets
const _MAX_BUCKET_SIZE = 4  # Cap per-bucket to prevent collision DoS
const _TIMESTAMP_COUNTER = Ref{UInt64}(0)

# Statistics
const _CACHE_STATS = Ref((hits=0, misses=0, evictions=0, collisions=0))

# ===============================================================
# Public API
# ===============================================================

"""
    set_cubic_cache_size!(n::Int)

Set maximum number of cached x-grids across all hash buckets.

# Arguments
- `n::Int`: Maximum total cache entries (default: 16)

# Example
```julia
set_cubic_cache_size!(32)  # Increase cache size
```
"""
function set_cubic_cache_size!(n::Int)
    n > 0 || throw(ArgumentError("Cache size must be positive"))
    _CACHE_SIZE_REF[] = n
    return n
end

"""
    get_cubic_cache_size()

Get current maximum cache size.
"""
get_cubic_cache_size() = _CACHE_SIZE_REF[]

"""
    clear_cubic_cache!()

Clear all cached x-grids and reset statistics.

Useful for benchmarking or memory management.

# Example
```julia
clear_cubic_cache!()
```
"""
function clear_cubic_cache!()
    lock(_CACHE_LOCK) do
        empty!(_CUBIC_CACHE_STORE)
        _TIMESTAMP_COUNTER[] = 0
        _CACHE_STATS[] = (hits=0, misses=0, evictions=0, collisions=0)
    end
    return nothing
end

"""
    cubic_cache_stats()

Return cache hit/miss/collision statistics for debugging.

# Returns
- `NamedTuple`: (hits, misses, evictions, collisions, size, efficiency)

# Example
```julia
stats = cubic_cache_stats()
println("Cache efficiency: \$(stats.efficiency)%")
println("Hash collisions: \$(stats.collisions)")
```
"""
function cubic_cache_stats()
    stats = _CACHE_STATS[]
    total_size = lock(_CACHE_LOCK) do
        sum(length(bucket) for bucket in values(_CUBIC_CACHE_STORE); init=0)
    end
    total = stats.hits + stats.misses
    efficiency = total > 0 ? round(100 * stats.hits / total, digits=1) : 0.0

    return (
        hits = stats.hits,
        misses = stats.misses,
        evictions = stats.evictions,
        collisions = stats.collisions,
        size = total_size,
        efficiency = efficiency
    )
end

# ===============================================================
# Internal Cache Management
# ===============================================================

"""
    count_total_caches()

Count total number of caches across all buckets.
Must be called within lock.
"""
function count_total_caches()
    return sum(length(bucket) for bucket in values(_CUBIC_CACHE_STORE); init=0)
end

"""
    find_oldest_cache()

Find (hash, index) of oldest cache by timestamp for LRU eviction.
Must be called within lock.

Returns (hash, index, timestamp) or nothing if cache is empty.
"""
function find_oldest_cache()
    oldest_hash = UInt64(0)
    oldest_idx = 0
    oldest_ts = typemax(UInt64)

    for (hash, bucket) in _CUBIC_CACHE_STORE
        for (idx, handle) in enumerate(bucket)
            if handle.timestamp < oldest_ts
                oldest_ts = handle.timestamp
                oldest_hash = hash
                oldest_idx = idx
            end
        end
    end

    return oldest_idx > 0 ? (oldest_hash, oldest_idx, oldest_ts) : nothing
end

"""
    evict_oldest_cache!()

Evict the oldest cache entry by timestamp (true LRU).
Must be called within lock.
"""
function evict_oldest_cache!()
    result = find_oldest_cache()
    if result !== nothing
        hash, idx, _ = result
        bucket = _CUBIC_CACHE_STORE[hash]
        deleteat!(bucket, idx)

        # Remove bucket if empty
        if isempty(bucket)
            delete!(_CUBIC_CACHE_STORE, hash)
        end

        # Update stats
        stats = _CACHE_STATS[]
        _CACHE_STATS[] = (hits=stats.hits,
                         misses=stats.misses,
                         evictions=stats.evictions + 1,
                         collisions=stats.collisions)
    end
end

"""
    get_cubic_cache(x::AbstractVector{T}) where {T<:AbstractFloat}

Internal: Lookup or create cache for x-grid with true LRU eviction.

Thread-safe with ReentrantLock. Returns existing cache if x matches
(using hash + isequal verification), otherwise creates new cache.

# Implementation Details

- **Hash collision handling**: Bucket array with isequal() verification
- **True LRU eviction**: Timestamp-based tracking across all caches
- **Per-bucket cap**: Limits bucket size to prevent DoS from collisions
- **Type safety**: Parametric buckets handle Float32/Float64/BigFloat
- **Defensive copy**: x is copied via collect() in CubicSplineCache constructor

# Collision Mitigation

- Per-bucket size capped at $_MAX_BUCKET_SIZE entries
- Global cache size limits total entries across all buckets
- Oldest cache evicted by timestamp when limits exceeded
"""
function get_cubic_cache(x::AbstractVector{T}) where {T<:AbstractFloat}
    x_hash = hash(x, UInt(0))

    lock(_CACHE_LOCK) do
        # Check if hash bucket exists
        if haskey(_CUBIC_CACHE_STORE, x_hash)
            bucket = _CUBIC_CACHE_STORE[x_hash]

            # Search bucket for matching x (handle hash collisions)
            for (i, handle) in enumerate(bucket)
                if isequal(handle.cache.x, x)
                    # Cache hit! Update timestamp for LRU
                    _TIMESTAMP_COUNTER[] += 1
                    new_handle = CacheHandle(handle.cache, _TIMESTAMP_COUNTER[], x_hash)
                    bucket[i] = new_handle

                    # Update stats
                    stats = _CACHE_STATS[]
                    _CACHE_STATS[] = (hits=stats.hits + 1,
                                     misses=stats.misses,
                                     evictions=stats.evictions,
                                     collisions=stats.collisions)

                    return handle.cache
                end
            end

            # Hash collision - track for statistics
            stats = _CACHE_STATS[]
            _CACHE_STATS[] = (hits=stats.hits,
                             misses=stats.misses + 1,
                             evictions=stats.evictions,
                             collisions=stats.collisions + 1)

            # Check per-bucket size limit before adding
            if length(bucket) >= _MAX_BUCKET_SIZE
                # Remove oldest from this bucket
                oldest_idx = argmin([h.timestamp for h in bucket])
                deleteat!(bucket, oldest_idx)

                stats = _CACHE_STATS[]
                _CACHE_STATS[] = (hits=stats.hits,
                                 misses=stats.misses,
                                 evictions=stats.evictions + 1,
                                 collisions=stats.collisions)
            end

            # Add new cache to bucket
            _TIMESTAMP_COUNTER[] += 1
            new_cache = CubicSplineCache(x)  # Defensive copy via collect(x)
            handle = CacheHandle(new_cache, _TIMESTAMP_COUNTER[], x_hash)
            push!(bucket, handle)

            # Check global size limit
            if count_total_caches() > _CACHE_SIZE_REF[]
                evict_oldest_cache!()
            end

            return new_cache
        else
            # Cache miss - new hash bucket
            stats = _CACHE_STATS[]
            _CACHE_STATS[] = (hits=stats.hits,
                             misses=stats.misses + 1,
                             evictions=stats.evictions,
                             collisions=stats.collisions)

            # Check global size limit before creating new bucket
            if count_total_caches() >= _CACHE_SIZE_REF[]
                evict_oldest_cache!()
            end

            # Create new cache and bucket
            _TIMESTAMP_COUNTER[] += 1
            new_cache = CubicSplineCache(x)  # Defensive copy via collect(x)
            handle = CacheHandle(new_cache, _TIMESTAMP_COUNTER[], x_hash)
            _CUBIC_CACHE_STORE[x_hash] = [handle]

            return new_cache
        end
    end
end
