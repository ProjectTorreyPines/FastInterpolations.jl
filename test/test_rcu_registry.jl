"""
Phase 2: RCU Registry Unit Tests

Tests for GlobalRegistry and RCU-style registry implementation.
These tests validate lock-free bank lookup and atomic registry updates.

Run with:
    julia --project -e 'using Pkg; Pkg.test(test_args=["test_rcu_registry.jl"])'
    julia -t 4 --project -e 'using Pkg; Pkg.test(test_args=["test_rcu_registry.jl"])'
"""

using FastInterpolations
using Test

# ===============================================================
# Test Group 1: GlobalRegistry Structure
# ===============================================================

@testset "GlobalRegistry Structure" begin
    @testset "RegistrySnapshot is Vector of Pairs" begin
        # RegistrySnapshot should be a type alias for Vector{Pair{DataType, Any}}
        snap = FastInterpolations.RegistrySnapshot()
        @test snap isa Vector{Pair{DataType, Any}}
        @test isempty(snap)
    end

    @testset "GlobalRegistry has atomic snapshot field" begin
        # GlobalRegistry should have @atomic snapshot field
        registry = FastInterpolations.GlobalRegistry(FastInterpolations.RegistrySnapshot())

        # Atomic load should work
        snap = @atomic :acquire registry.snapshot
        @test snap isa FastInterpolations.RegistrySnapshot
        @test isempty(snap)
    end

    @testset "GlobalRegistry constructor creates empty snapshot" begin
        registry = FastInterpolations.GlobalRegistry()
        snap = @atomic :acquire registry.snapshot
        @test isempty(snap)
    end
end

# ===============================================================
# Test Group 2: Global Registries Exist
# ===============================================================

@testset "Global Registries" begin
    @testset "Derivative registry exists and is GlobalRegistry" begin
        @test isdefined(FastInterpolations, :_DERIVATIVE_REGISTRY)
        @test FastInterpolations._DERIVATIVE_REGISTRY isa FastInterpolations.GlobalRegistry
    end

    @testset "Periodic registry exists and is GlobalRegistry" begin
        @test isdefined(FastInterpolations, :_PERIODIC_REGISTRY)
        @test FastInterpolations._PERIODIC_REGISTRY isa FastInterpolations.GlobalRegistry
    end
end

# ===============================================================
# Test Group 3: Lock-Free Registry Lookup
# ===============================================================

@testset "Lock-Free Registry Lookup" begin
    @testset "Lookup on empty registry returns nothing" begin
        FastInterpolations.clear_cubic_cache!()

        # Create test registry
        registry = FastInterpolations.GlobalRegistry()

        # Lookup non-existent bank type
        DummyBankType = FastInterpolations.CacheBank{FastInterpolations.CacheEntry{Float64, FastInterpolations.Deriv2{Float64}, FastInterpolations.Deriv2{Float64}, Vector{Float64}}}

        result = FastInterpolations._registry_lookup(registry, DummyBankType)
        @test result === nothing
    end

    @testset "Lookup finds existing bank" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Force bank creation via interpolation
        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        # Lookup should find the bank
        BankType = FastInterpolations.CacheBank{FastInterpolations.CacheEntry{Float64, FastInterpolations.Deriv2{Float64}, FastInterpolations.Deriv2{Float64}, Vector{Float64}}}

        result = FastInterpolations._registry_lookup(FastInterpolations._DERIVATIVE_REGISTRY, BankType)
        @test result !== nothing
        @test result isa BankType
    end

    @testset "Lookup by type identity (fast path)" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        # Same type should be found
        BankType = FastInterpolations.CacheBank{FastInterpolations.CacheEntry{Float64, FastInterpolations.Deriv2{Float64}, FastInterpolations.Deriv2{Float64}, Vector{Float64}}}

        snap = @atomic :acquire FastInterpolations._DERIVATIVE_REGISTRY.snapshot

        # Type should be found in snapshot
        found = false
        for (TypeKey, _) in snap
            if TypeKey === BankType
                found = true
                break
            end
        end
        @test found
    end
end

# ===============================================================
# Test Group 4: RCU Registry Insert
# ===============================================================

@testset "RCU Registry Insert" begin
    @testset "Insert creates new snapshot" begin
        FastInterpolations.clear_cubic_cache!()

        snap_before = @atomic :acquire FastInterpolations._DERIVATIVE_REGISTRY.snapshot
        @test isempty(snap_before)

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        snap_after = @atomic :acquire FastInterpolations._DERIVATIVE_REGISTRY.snapshot
        @test length(snap_after) == 1
        @test snap_after !== snap_before  # Different snapshot object
    end

    @testset "Multiple bank types create multiple entries" begin
        FastInterpolations.clear_cubic_cache!()

        # Float64 with Vector and natural BC (Deriv2)
        x1 = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x1)
        FastInterpolations.cubic_interp(x1, y1, 0.5; autocache=true)

        # Float64 with different BC (Deriv1 instead of Deriv2) - use BCPair
        x2 = collect(range(0.0, 1.0, 51))
        y2 = cos.(2π .* x2)
        bc_pair = FastInterpolations.BCPair(FastInterpolations.Deriv1(1.0), FastInterpolations.Deriv1(-1.0))
        FastInterpolations.cubic_interp(x2, y2, 0.5; bc=bc_pair, autocache=true)

        snap = @atomic :acquire FastInterpolations._DERIVATIVE_REGISTRY.snapshot
        @test length(snap) == 2
    end

    @testset "Periodic BC uses separate registry" begin
        FastInterpolations.clear_cubic_cache!()

        # Derivative BC
        x1 = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x1)
        FastInterpolations.cubic_interp(x1, y1, 0.5; autocache=true)

        # Periodic BC
        x2 = collect(range(0.0, 2π, 51))
        y2 = sin.(x2)
        FastInterpolations.cubic_interp(x2, y2, π; bc=FastInterpolations.PeriodicBC(), autocache=true)

        deriv_snap = @atomic :acquire FastInterpolations._DERIVATIVE_REGISTRY.snapshot
        periodic_snap = @atomic :acquire FastInterpolations._PERIODIC_REGISTRY.snapshot

        @test length(deriv_snap) == 1
        @test length(periodic_snap) == 1
    end
end

# ===============================================================
# Test Group 5: clear_cubic_cache! with Atomic Registry
# ===============================================================

@testset "clear_cubic_cache! Atomic" begin
    @testset "Clear empties both registries" begin
        # Populate caches
        x1 = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x1)
        FastInterpolations.cubic_interp(x1, y1, 0.5; autocache=true)

        x2 = collect(range(0.0, 2π, 51))
        y2 = sin.(x2)
        FastInterpolations.cubic_interp(x2, y2, π; bc=FastInterpolations.PeriodicBC(), autocache=true)

        # Verify populated
        @test !isempty(@atomic :acquire FastInterpolations._DERIVATIVE_REGISTRY.snapshot)
        @test !isempty(@atomic :acquire FastInterpolations._PERIODIC_REGISTRY.snapshot)

        # Clear
        FastInterpolations.clear_cubic_cache!()

        # Verify empty
        @test isempty(@atomic :acquire FastInterpolations._DERIVATIVE_REGISTRY.snapshot)
        @test isempty(@atomic :acquire FastInterpolations._PERIODIC_REGISTRY.snapshot)
    end

    @testset "Clear atomically replaces snapshot" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        snap_before = @atomic :acquire FastInterpolations._DERIVATIVE_REGISTRY.snapshot

        FastInterpolations.clear_cubic_cache!()

        snap_after = @atomic :acquire FastInterpolations._DERIVATIVE_REGISTRY.snapshot

        @test snap_after !== snap_before  # Different object
        @test isempty(snap_after)
        @test !isempty(snap_before)  # Old snapshot unchanged
    end
end

# ===============================================================
# Test Group 6: Zero-Allocation Registry Lookup
# ===============================================================

@testset "Zero-Allocation Registry Lookup" begin
    @testset "Registry lookup allocates minimally" begin
        FastInterpolations.clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Populate cache
        FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)

        BankType = FastInterpolations.CacheBank{FastInterpolations.CacheEntry{Float64, FastInterpolations.Deriv2{Float64}, FastInterpolations.Deriv2{Float64}, Vector{Float64}}}

        # Warm up
        for _ in 1:100
            FastInterpolations._registry_lookup(FastInterpolations._DERIVATIVE_REGISTRY, BankType)
        end

        # Test allocation
        allocs = @allocated FastInterpolations._registry_lookup(FastInterpolations._DERIVATIVE_REGISTRY, BankType)

        # Should be minimal (atomic boxing overhead only)
        @test allocs <= 64
        @info "Registry lookup allocation: $allocs bytes"
    end
end

# ===============================================================
# Test Group 7: Thread Safety
# ===============================================================

if Threads.nthreads() > 1
    @testset "Concurrent Registry Access" begin
        @testset "Concurrent bank creation is safe" begin
            errors = Threads.Atomic{Int}(0)

            for _ in 1:100
                FastInterpolations.clear_cubic_cache!()

                grids = [collect(range(0.0, Float64(i), 51)) for i in 1:4]
                y = sin.(2π .* grids[1])

                Threads.@threads for i in 1:4
                    try
                        FastInterpolations.cubic_interp(grids[mod1(i, 4)], y, 0.5; autocache=true)
                    catch
                        Threads.atomic_add!(errors, 1)
                    end
                end
            end

            @test errors[] == 0
        end

        @testset "Concurrent lookup/insert is safe" begin
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

        @testset "Concurrent clear/insert is safe" begin
            errors = Threads.Atomic{Int}(0)

            # Run multiple rounds
            for _ in 1:50
                x = collect(range(0.0, 1.0, 51))
                y = sin.(2π .* x)

                # Clear and insert concurrently
                Threads.@threads for i in 1:4
                    try
                        if i == 1
                            FastInterpolations.clear_cubic_cache!()
                        else
                            FastInterpolations.cubic_interp(x, y, 0.5; autocache=true)
                        end
                    catch
                        Threads.atomic_add!(errors, 1)
                    end
                end
            end

            @test errors[] == 0
        end
    end
else
    @testset "Thread Safety (skipped)" begin
        @test_broken false  # Marker for skipped tests
        @warn "Thread-safety tests require multiple threads. Run with: julia -t 4"
    end
end
