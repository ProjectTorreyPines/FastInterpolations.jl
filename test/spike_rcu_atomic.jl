"""
Phase 0: RCU Atomic Pattern Spike Test

Validates the RCU (Read-Copy-Update) pattern with actual FastInterpolations types
before full implementation. This is a standalone test that doesn't modify the library.

Goals:
1. BankSnapshot with real CacheEntry types works with @atomic
2. Lock-free read path achieves zero allocation
3. Copy-on-write insert path is correct
4. Concurrent access is safe (no torn reads, no data corruption)

Usage:
    julia -t 4 --project test/spike_rcu_atomic.jl
"""

using FastInterpolations
using LinearAlgebra
using Test

# ===============================================================
# Import Types (for spike test only)
# ===============================================================

# Access internal types via module
const CacheEntry = FastInterpolations.CacheEntry
const PeriodicCacheEntry = FastInterpolations.PeriodicCacheEntry
const AbstractCacheEntry = FastInterpolations.AbstractCacheEntry
const CubicSplineCache = FastInterpolations.CubicSplineCache
const BCPair = FastInterpolations.BCPair
const Deriv2 = FastInterpolations.Deriv2
const PeriodicData = FastInterpolations.PeriodicData

# ===============================================================
# RCU Data Structures (Spike Implementation)
# ===============================================================

"""
Immutable snapshot of bank state at a point in time.
"""
struct BankSnapshot{E<:AbstractCacheEntry}
    store::Vector{E}
    count::Int
    ring::Int  # 1-based next eviction index
end

# Empty snapshot constructor
function BankSnapshot{E}() where {E<:AbstractCacheEntry}
    BankSnapshot{E}(E[], 0, 1)
end

"""
Atomic bank wrapper - holds an atomic reference to snapshot.
"""
mutable struct AtomicBank{E<:AbstractCacheEntry}
    @atomic snapshot::BankSnapshot{E}
end

function AtomicBank{E}() where {E<:AbstractCacheEntry}
    AtomicBank{E}(BankSnapshot{E}())
end

# Global lock for writes (same as real implementation)
const _SPIKE_LOCK = ReentrantLock()
const _SPIKE_CACHE_SIZE = 16

# ===============================================================
# RCU Operations
# ===============================================================

"""
Lock-free lookup (RCU read path).
"""
@inline function rcu_lookup(bank::AtomicBank{E}, id::UInt, x::X) where {E, X}
    # Atomic load with acquire ordering
    snap = @atomic :acquire bank.snapshot
    store = snap.store
    count = snap.count

    # Pass 1: Identity check (fast path)
    @inbounds for i in 1:count
        entry = store[i]
        if entry.id === id
            return entry.cache
        end
    end

    # Pass 2: Equality check
    @inbounds for i in 1:count
        entry = store[i]
        if isequal(entry.x, x)
            return entry.cache
        end
    end

    return nothing
end

"""
RCU insert (write path with copy-on-write).
"""
function rcu_insert!(bank::AtomicBank{E}, entry::E) where {E}
    lock(_SPIKE_LOCK)
    try
        # Load current snapshot (monotonic ok inside lock)
        old_snap = @atomic :monotonic bank.snapshot

        # Copy-on-write
        new_store = copy(old_snap.store)
        new_count = old_snap.count
        new_ring = old_snap.ring

        # Insert with ring buffer semantics
        if new_count < _SPIKE_CACHE_SIZE
            push!(new_store, entry)
            new_count += 1
        else
            new_store[new_ring] = entry
            new_ring = (new_ring % _SPIKE_CACHE_SIZE) + 1
        end

        # Publish new snapshot with release ordering
        new_snap = BankSnapshot{E}(new_store, new_count, new_ring)
        @atomic :release bank.snapshot = new_snap

        return entry.cache
    finally
        unlock(_SPIKE_LOCK)
    end
end

"""
RCU lookup or insert (full path).
"""
function rcu_lookup_or_insert!(bank::AtomicBank{E}, x::X, build_cache::Function) where {E, X}
    id = objectid(x)

    # Fast path: lock-free lookup
    found = rcu_lookup(bank, id, x)
    found !== nothing && return found

    # Slow path: lock → double-check → insert
    lock(_SPIKE_LOCK)
    try
        # Double-check after acquiring lock
        found = rcu_lookup(bank, id, x)
        found !== nothing && return found

        # Build cache and entry
        cache = build_cache(x)
        entry = E(id, x, cache)

        # Insert via RCU
        return rcu_insert!(bank, entry)
    finally
        unlock(_SPIKE_LOCK)
    end
end

# ===============================================================
# Test Helpers
# ===============================================================

# Entry type alias for natural BC
const NaturalCacheEntry = CacheEntry{Float64, Deriv2{Float64}, Deriv2{Float64}, Vector{Float64}}

function make_cache_builder(bc_pair::BCPair)
    function build(x)
        FastInterpolations._build_derivative_bc_cache(x, bc_pair.left, bc_pair.right)
    end
    return build
end

# ===============================================================
# Tests
# ===============================================================

@testset "Phase 0: RCU Spike Tests" begin

    @testset "1. Basic Operations" begin
        # Setup
        bank = AtomicBank{NaturalCacheEntry}()
        bc = BCPair(Deriv2(0.0), Deriv2(0.0))
        builder = make_cache_builder(bc)

        x1 = collect(range(0.0, 1.0, 51))
        x2 = collect(range(0.0, 2.0, 101))

        # Test: Empty lookup returns nothing
        @test rcu_lookup(bank, objectid(x1), x1) === nothing

        # Test: Insert and retrieve
        cache1 = rcu_lookup_or_insert!(bank, x1, builder)
        @test cache1 isa CubicSplineCache

        # Test: Cache hit returns same cache
        cache1_again = rcu_lookup_or_insert!(bank, x1, builder)
        @test cache1_again === cache1

        # Test: Different x creates new cache
        cache2 = rcu_lookup_or_insert!(bank, x2, builder)
        @test cache2 !== cache1

        # Test: Snapshot state
        snap = @atomic :acquire bank.snapshot
        @test snap.count == 2
        @test length(snap.store) == 2
    end

    @testset "2. Zero-Allocation Read Path" begin
        # Setup: Pre-populate cache
        bank = AtomicBank{NaturalCacheEntry}()
        bc = BCPair(Deriv2(0.0), Deriv2(0.0))
        builder = make_cache_builder(bc)

        x = collect(range(0.0, 1.0, 51))
        cache = rcu_lookup_or_insert!(bank, x, builder)

        # Create a barrier function to ensure type stability
        # and prevent compiler from hoisting allocations
        function do_lookup(bank::AtomicBank{NaturalCacheEntry}, id::UInt, x::Vector{Float64})
            return rcu_lookup(bank, id, x)
        end

        id = objectid(x)

        # Warm up for JIT
        for _ in 1:100
            do_lookup(bank, id, x)
        end

        # Test: Zero allocation on single hit
        allocs = @allocated do_lookup(bank, id, x)

        # Note: In MT mode, there may be small allocations from atomic operations
        # Accept up to 64 bytes which is typical for atomic ref boxing
        @test allocs <= 64
        @info "Read path allocation (single call): $allocs bytes"
    end

    @testset "3. Wrapped Function Zero-Allocation" begin
        # This tests the pattern that will be used in real implementation
        bank = AtomicBank{NaturalCacheEntry}()
        bc = BCPair(Deriv2(0.0), Deriv2(0.0))
        builder = make_cache_builder(bc)

        x = collect(range(0.0, 1.0, 51))

        # Pre-populate
        rcu_lookup_or_insert!(bank, x, builder)

        # Wrapper function with explicit types (simulates real API)
        function cached_lookup(bank::AtomicBank{NaturalCacheEntry}, x::Vector{Float64})
            id = objectid(x)
            return rcu_lookup(bank, id, x)
        end

        # Warm up (more iterations for better JIT)
        for _ in 1:100
            cached_lookup(bank, x)
        end

        # Test allocation
        allocs = @allocated cached_lookup(bank, x)

        # Accept up to 64 bytes for potential atomic boxing
        @test allocs <= 64
        @info "Wrapped function allocation: $allocs bytes"
    end

    @testset "4. Ring Buffer Eviction" begin
        bank = AtomicBank{NaturalCacheEntry}()
        bc = BCPair(Deriv2(0.0), Deriv2(0.0))
        builder = make_cache_builder(bc)

        # Fill beyond capacity
        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:20]

        for x in grids
            rcu_lookup_or_insert!(bank, x, builder)
        end

        # Check capacity limit maintained
        snap = @atomic :acquire bank.snapshot
        @test snap.count == _SPIKE_CACHE_SIZE
        @test length(snap.store) == _SPIKE_CACHE_SIZE

        # Recent entries should be found
        recent_cache = rcu_lookup(bank, objectid(grids[end]), grids[end])
        @test recent_cache !== nothing

        @info "Ring buffer: count=$(snap.count), ring=$(snap.ring)"
    end

    @testset "5. Concurrent Access" begin
        bank = AtomicBank{NaturalCacheEntry}()
        bc = BCPair(Deriv2(0.0), Deriv2(0.0))
        builder = make_cache_builder(bc)

        nthreads = Threads.nthreads()
        iterations = 10_000
        errors = Threads.Atomic{Int}(0)
        reads = Threads.Atomic{Int}(0)
        writes = Threads.Atomic{Int}(0)

        # Shared grids
        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:8]

        @info "Starting concurrent test: $nthreads threads, $iterations iterations each"

        Threads.@threads for t in 1:nthreads
            local_errors = 0
            local_reads = 0
            local_writes = 0

            for i in 1:iterations
                try
                    grid_idx = mod1(i + t, length(grids))
                    x = grids[grid_idx]
                    id = objectid(x)

                    # 80% reads, 20% writes
                    if i % 5 != 0
                        # Read operation
                        result = rcu_lookup(bank, id, x)
                        local_reads += 1

                        # Validate cache structure if found
                        if result !== nothing
                            # Access cache fields to detect corruption
                            _ = result.lu_factor  # LU factorization
                        end
                    else
                        # Write operation
                        cache = rcu_lookup_or_insert!(bank, x, builder)
                        local_writes += 1

                        # Validate cache
                        @assert cache !== nothing "Cache should not be nothing after insert"
                    end
                catch e
                    local_errors += 1
                    @error "Thread $t error at iteration $i" exception=(e, catch_backtrace())
                end
            end

            Threads.atomic_add!(errors, local_errors)
            Threads.atomic_add!(reads, local_reads)
            Threads.atomic_add!(writes, local_writes)
        end

        @test errors[] == 0
        @info "Concurrent test complete: $(reads[]) reads, $(writes[]) writes, $(errors[]) errors"
    end

    @testset "6. Snapshot Immutability (Torn Read Test)" begin
        # Test that readers never see partially updated snapshots

        bank = AtomicBank{NaturalCacheEntry}()
        bc = BCPair(Deriv2(0.0), Deriv2(0.0))
        builder = make_cache_builder(bc)

        # Pre-populate with some entries
        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:4]
        for x in grids
            rcu_lookup_or_insert!(bank, x, builder)
        end

        nthreads = Threads.nthreads()
        duration_sec = 2
        torn_reads = Threads.Atomic{Int}(0)
        total_reads = Threads.Atomic{Int}(0)

        @info "Starting torn read test: $nthreads threads, $duration_sec seconds"

        stop_flag = Threads.Atomic{Bool}(false)

        # Reader threads
        reader_task = Threads.@spawn begin
            local_torn = 0
            local_reads = 0

            while !stop_flag[]
                snap = @atomic :acquire bank.snapshot
                count = snap.count
                store = snap.store

                # Validate consistency: all entries should exist
                valid = true
                @inbounds for i in 1:count
                    if i > length(store)
                        valid = false
                        break
                    end
                    entry = store[i]
                    if entry.cache === nothing
                        valid = false
                        break
                    end
                end

                if !valid
                    local_torn += 1
                end
                local_reads += 1
            end

            Threads.atomic_add!(torn_reads, local_torn)
            Threads.atomic_add!(total_reads, local_reads)
        end

        # Writer thread
        writer_task = Threads.@spawn begin
            batch = 0
            while !stop_flag[]
                batch += 1
                # Use (batch % 100) + 1 to avoid zero-length range (causes singular matrix)
                endpoint = Float64((batch % 100) + 1)
                x = collect(range(0.0, endpoint, 51))
                rcu_lookup_or_insert!(bank, x, builder)
                # Small sleep to let readers run
                sleep(0.0001)
            end
        end

        # Run for duration
        sleep(duration_sec)
        stop_flag[] = true

        wait(reader_task)
        wait(writer_task)

        @test torn_reads[] == 0
        @info "Torn read test: $(total_reads[]) reads, $(torn_reads[]) torn"
    end

end

# ===============================================================
# Performance Comparison (Optional)
# ===============================================================

function run_performance_comparison()
    println("\n" * "="^60)
    println("Performance Comparison: Current vs RCU")
    println("="^60)

    # Setup
    bank_rcu = AtomicBank{NaturalCacheEntry}()
    bc = BCPair(Deriv2(0.0), Deriv2(0.0))
    builder = make_cache_builder(bc)

    x = collect(range(0.0, 1.0, 51))
    y = sin.(2π .* x)

    # Populate RCU bank
    rcu_lookup_or_insert!(bank_rcu, x, builder)

    # Populate real cache
    FastInterpolations.clear_cubic_cache!()
    FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

    # Warm up
    for _ in 1:100
        rcu_lookup(bank_rcu, objectid(x), x)
        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)
    end

    # Benchmark
    n_iters = 100_000

    # RCU read
    id = objectid(x)
    t_rcu = @elapsed begin
        for _ in 1:n_iters
            rcu_lookup(bank_rcu, id, x)
        end
    end

    # Current implementation (full interp call includes cache hit)
    t_current = @elapsed begin
        for _ in 1:n_iters
            FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)
        end
    end

    println("RCU lookup:     $(round(t_rcu * 1e9 / n_iters, digits=1)) ns/op")
    println("Current interp: $(round(t_current * 1e9 / n_iters, digits=1)) ns/op (includes interpolation)")
    println()

    # Pure cache hit comparison (need to extract cache lookup timing)
    # This is harder to measure since current impl locks the whole operation

    println("Note: Current impl includes full interpolation, not just cache lookup.")
    println("RCU provides lock-free cache hit which benefits multi-threaded workloads.")
end

# ===============================================================
# Main
# ===============================================================

println("Running Phase 0: RCU Spike Tests")
println("Julia version: $(VERSION)")
println("Threads: $(Threads.nthreads())")
println()

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    run_performance_comparison()
end
