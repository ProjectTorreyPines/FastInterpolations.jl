"""
Phase 1: RCU Bank Unit Tests

Tests for BankSnapshot and RCU-style CacheBank implementation.
These tests validate the lock-free read path and copy-on-write semantics.

Run with:
    julia --project -e 'using Pkg; Pkg.test(test_args=["test_rcu_bank.jl"])'
    julia -t 4 --project -e 'using Pkg; Pkg.test(test_args=["test_rcu_bank.jl"])'
"""

using FastInterpolations
using Test

# ===============================================================
# Test Group 1: BankSnapshot Structure
# ===============================================================

@testset "BankSnapshot Structure" begin
    @testset "Empty snapshot creation" begin
        # BankSnapshot should be creatable with empty store
        snap = FastInterpolations.BankSnapshot(
            FastInterpolations.CacheEntry{Float64, FastInterpolations.Deriv2{Float64}, FastInterpolations.Deriv2{Float64}, Vector{Float64}}[],
            0,
            1
        )
        @test snap.count == 0
        @test snap.ring == 1
        @test isempty(snap.store)
    end

    @testset "Snapshot immutability" begin
        # BankSnapshot should be an immutable struct
        @test !Base.ismutabletype(FastInterpolations.BankSnapshot)
    end

    @testset "Snapshot with entries" begin
        # Create a cache entry
        x = collect(range(0.0, 1.0, 51))
        bc = FastInterpolations.BCPair(FastInterpolations.Deriv2(0.0), FastInterpolations.Deriv2(0.0))
        cache = FastInterpolations._build_derivative_bc_cache(x, bc.left, bc.right)

        EntryType = FastInterpolations.CacheEntry{Float64, FastInterpolations.Deriv2{Float64}, FastInterpolations.Deriv2{Float64}, Vector{Float64}}
        entry = EntryType(objectid(x), x, cache)

        snap = FastInterpolations.BankSnapshot([entry], 1, 1)
        @test snap.count == 1
        @test length(snap.store) == 1
        @test snap.store[1].cache === cache
    end
end

# ===============================================================
# Test Group 2: CacheBank with Atomic Snapshot
# ===============================================================

@testset "CacheBank Atomic Snapshot" begin
    @testset "Bank has atomic snapshot field" begin
        # CacheBank should have @atomic snapshot field
        EntryType = FastInterpolations.CacheEntry{Float64, FastInterpolations.Deriv2{Float64}, FastInterpolations.Deriv2{Float64}, Vector{Float64}}
        bank = FastInterpolations.CacheBank{EntryType}()

        # Atomic load should work
        snap = @atomic :acquire bank.snapshot
        @test snap isa FastInterpolations.BankSnapshot
        @test snap.count == 0
    end

    @testset "Bank constructor creates empty snapshot" begin
        EntryType = FastInterpolations.CacheEntry{Float64, FastInterpolations.Deriv2{Float64}, FastInterpolations.Deriv2{Float64}, Vector{Float64}}
        bank = FastInterpolations.CacheBank{EntryType}()

        snap = @atomic :acquire bank.snapshot
        @test snap.count == 0
        @test snap.ring == 1
        @test isempty(snap.store)
    end
end

# ===============================================================
# Test Group 3: Lock-Free Lookup
# ===============================================================

@testset "Lock-Free Lookup" begin
    @testset "Lookup on empty bank returns nothing" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Force cache creation
        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        # Lookup with different x should return nothing
        x2 = collect(range(0.0, 2.0, 51))
        bc = FastInterpolations.BCPair(FastInterpolations.Deriv2(0.0), FastInterpolations.Deriv2(0.0))
        bank = FastInterpolations._get_derivative_bank(x, bc)

        snap = @atomic :acquire bank.snapshot
        result = FastInterpolations._rcu_lookup(snap, objectid(x2), x2)
        @test result === nothing
    end

    @testset "Lookup by identity (fast path)" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        cache = FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        # Lookup same x by identity
        bc = FastInterpolations.BCPair(FastInterpolations.Deriv2(0.0), FastInterpolations.Deriv2(0.0))
        bank = FastInterpolations._get_derivative_bank(x, bc)

        snap = @atomic :acquire bank.snapshot
        result = FastInterpolations._rcu_lookup(snap, objectid(x), x)
        @test result !== nothing
    end

    @testset "Lookup by equality (slow path)" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        # Create different x with same values
        x_copy = collect(range(0.0, 1.0, 51))
        @test x_copy !== x  # Different object
        @test x_copy == x   # Same values

        bc = FastInterpolations.BCPair(FastInterpolations.Deriv2(0.0), FastInterpolations.Deriv2(0.0))
        bank = FastInterpolations._get_derivative_bank(x, bc)

        snap = @atomic :acquire bank.snapshot
        result = FastInterpolations._rcu_lookup(snap, objectid(x_copy), x_copy)
        @test result !== nothing
    end
end

# ===============================================================
# Test Group 4: RCU Insert (Copy-on-Write)
# ===============================================================

@testset "RCU Insert" begin
    @testset "Insert creates new snapshot" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        bc = FastInterpolations.BCPair(FastInterpolations.Deriv2(0.0), FastInterpolations.Deriv2(0.0))

        # Get bank before insert
        bank = FastInterpolations._get_derivative_bank(x, bc)
        snap_before = @atomic :acquire bank.snapshot
        @test snap_before.count == 0

        # Insert via normal API
        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        # Snapshot should be updated
        snap_after = @atomic :acquire bank.snapshot
        @test snap_after.count == 1
        @test snap_after !== snap_before  # Different snapshot object
    end

    @testset "Multiple inserts increment count" begin
        FastInterpolations.clear_cubic_cache!()

        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:5]
        y = sin.(2π .* grids[1])

        for x in grids
            FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)
        end

        bc = FastInterpolations.BCPair(FastInterpolations.Deriv2(0.0), FastInterpolations.Deriv2(0.0))
        bank = FastInterpolations._get_derivative_bank(grids[1], bc)

        snap = @atomic :acquire bank.snapshot
        @test snap.count == 5
    end

    @testset "Ring buffer eviction at capacity" begin
        FastInterpolations.clear_cubic_cache!()

        cache_size = FastInterpolations.get_cubic_cache_size()
        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:(cache_size + 5)]
        y = sin.(2π .* grids[1])

        for x in grids
            FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)
        end

        bc = FastInterpolations.BCPair(FastInterpolations.Deriv2(0.0), FastInterpolations.Deriv2(0.0))
        bank = FastInterpolations._get_derivative_bank(grids[1], bc)

        snap = @atomic :acquire bank.snapshot
        @test snap.count == cache_size  # Capped at max
        @test length(snap.store) == cache_size
    end
end

# ===============================================================
# Test Group 5: Allocation Test
# ===============================================================

@testset "Zero-Allocation Read Path" begin
    @testset "Cache hit allocates minimally" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Populate cache
        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        # Warm up
        for _ in 1:100
            FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)
        end

        # Test allocation on cache hit
        allocs = @allocated FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        # Should be very low (atomic boxing overhead only)
        # Note: Full interp includes result allocation, but cache lookup should be minimal
        @test allocs <= 256  # Generous threshold for full interp
        @info "Full interp cache hit allocation: $allocs bytes"
    end
end

# ===============================================================
# Test Group 6: Thread Safety
# ===============================================================

if Threads.nthreads() > 1
    @testset "Concurrent Read Safety" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Pre-populate
        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        errors = Threads.Atomic{Int}(0)
        reads = Threads.Atomic{Int}(0)

        Threads.@threads for _ in 1:1000
            try
                result = FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)
                # sin(2π * 0.5) = sin(π) ≈ 0, use atol for near-zero comparison
                @assert isapprox(result, 0.0; atol=1e-10)
                Threads.atomic_add!(reads, 1)
            catch
                Threads.atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
        @test reads[] == 1000
    end

    @testset "Concurrent Read/Write Safety" begin
        FastInterpolations.clear_cubic_cache!()

        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:8]
        y = sin.(2π .* grids[1])

        errors = Threads.Atomic{Int}(0)

        Threads.@threads for i in 1:1000
            try
                grid_idx = mod1(i, length(grids))
                FastInterpolations.cubic_interp(grids[grid_idx], y, 0.5; autocache=true)
            catch
                Threads.atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end
else
    @testset "Thread Safety (skipped)" begin
        @test_broken false  # Marker for skipped tests
        @warn "Thread-safety tests require multiple threads. Run with: julia -t 4"
    end
end

# ===============================================================
# Test Group 7: Periodic BC Support
# ===============================================================

@testset "Periodic BC RCU" begin
    @testset "Periodic bank uses atomic snapshot" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 2π, 51))
        y = sin.(x)

        FastInterpolations.cubic_interp(x, y, π; bc=FastInterpolations.PeriodicBC(), autocache=true)

        bank = FastInterpolations._get_periodic_bank(x)
        snap = @atomic :acquire bank.snapshot
        @test snap.count == 1
    end
end
