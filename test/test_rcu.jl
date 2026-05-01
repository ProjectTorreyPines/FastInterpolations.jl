"""
RCU (Read-Copy-Update) Unit Tests

Tests for the thread-safe autocache implementation:
- BankSnapshot and CacheBank (lock-free cache entry storage)
- GlobalRegistry (lock-free bank type lookup)
"""

# ###################################################################
#                         BANK TESTS
# ###################################################################

@testitem "RCU Bank" begin
    # Common type aliases for tests
    FI = FastInterpolations
    EntryType = FI.CacheEntry{Float64, FI.Deriv2{Float64}, FI.Deriv2{Float64}, Vector{Float64}, FI.CubicSplineCache{Float64, FI._CachedVector{Float64, Float64}, FI.ThomasFactorization{Float64, Vector{Float64}}, FI.BCPair{FI.Deriv2{Float64}, FI.Deriv2{Float64}}}}
    BankType = FI.CacheBank{EntryType}


    # ===============================================================
    # BankSnapshot Structure
    # ===============================================================
    @testset "BankSnapshot Structure" begin
        @testset "Empty snapshot creation" begin
            snap = FI.BankSnapshot(EntryType[], 0, 1)
            @test snap.count == 0
            @test snap.ring == 1
            @test isempty(snap.store)
        end

        @testset "Snapshot immutability" begin
            @test !Base.ismutabletype(FI.BankSnapshot)
        end

        @testset "Snapshot with entries" begin
            x = collect(range(0.0, 1.0, 51))
            bc = FI.BCPair(FI.Deriv2(0.0), FI.Deriv2(0.0))
            cache = FI._build_derivative_bc_cache(x, bc.left, bc.right)

            entry = EntryType(objectid(x), x, cache)
            snap = FI.BankSnapshot([entry], 1, 1)

            @test snap.count == 1
            @test length(snap.store) == 1
            @test snap.store[1].cache === cache
        end
    end

    # ===============================================================
    # CacheBank with Atomic Snapshot
    # ===============================================================
    @testset "CacheBank Atomic Snapshot" begin
        @testset "Bank has atomic snapshot field" begin
            bank = BankType()
            snap = @atomic :acquire bank.snapshot
            @test snap isa FI.BankSnapshot
            @test snap.count == 0
        end

        @testset "Bank constructor creates empty snapshot" begin
            bank = BankType()
            snap = @atomic :acquire bank.snapshot
            @test snap.count == 0
            @test snap.ring == 1
            @test isempty(snap.store)
        end
    end

    # ===============================================================
    # Lock-Free Bank Lookup
    # ===============================================================
    @testset "Lock-Free Lookup" begin
        @testset "Lookup on empty bank returns nothing" begin
            FI.clear_cubic_cache!()

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            FI.cubic_interp(x, y, 0.5; bc = ZeroCurvBC(), autocache = true)

            x2 = collect(range(0.0, 2.0, 51))
            bc = FI.BCPair(FI.Deriv2(0.0), FI.Deriv2(0.0))
            bank = FI._get_derivative_bank(x, bc)

            snap = @atomic :acquire bank.snapshot
            # `_rcu_lookup` 4th arg verifies the cache's BC against the looked-up
            # one (codex P1). Derivative banks fall through `_verify_cache_match
            # (::Any, ::Any) = true`, so any sentinel works here.
            result = FI._rcu_lookup(snap, objectid(x2), x2, bc)
            @test result === nothing
        end

        @testset "Lookup by identity (fast path)" begin
            FI.clear_cubic_cache!()

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            FI.cubic_interp(x, y, 0.5; bc = ZeroCurvBC(), autocache = true)

            bc = FI.BCPair(FI.Deriv2(0.0), FI.Deriv2(0.0))
            bank = FI._get_derivative_bank(x, bc)

            snap = @atomic :acquire bank.snapshot
            result = FI._rcu_lookup(snap, objectid(x), x, bc)
            @test result !== nothing
        end

        @testset "Lookup by equality (slow path)" begin
            FI.clear_cubic_cache!()

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            FI.cubic_interp(x, y, 0.5; bc = ZeroCurvBC(), autocache = true)

            x_copy = collect(range(0.0, 1.0, 51))
            @test x_copy !== x  # Different object
            @test x_copy == x   # Same values

            bc = FI.BCPair(FI.Deriv2(0.0), FI.Deriv2(0.0))
            bank = FI._get_derivative_bank(x, bc)

            snap = @atomic :acquire bank.snapshot
            result = FI._rcu_lookup(snap, objectid(x_copy), x_copy, bc)
            @test result !== nothing
        end
    end

    # ===============================================================
    # RCU Bank Insert (Copy-on-Write)
    # ===============================================================
    @testset "RCU Insert" begin
        @testset "Insert creates new snapshot" begin
            FI.clear_cubic_cache!()

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            bc = FI.BCPair(FI.Deriv2(0.0), FI.Deriv2(0.0))

            bank = FI._get_derivative_bank(x, bc)
            snap_before = @atomic :acquire bank.snapshot
            @test snap_before.count == 0

            FI.cubic_interp(x, y, 0.5; bc = ZeroCurvBC(), autocache = true)

            snap_after = @atomic :acquire bank.snapshot
            @test snap_after.count == 1
            @test snap_after !== snap_before
        end

        @testset "Multiple inserts increment count" begin
            FI.clear_cubic_cache!()

            grids = [collect(range(0.0, Float64(i), 51)) for i in 1:5]
            y = sin.(2π .* grids[1])

            for x in grids
                FI.cubic_interp(x, y, 0.5; bc = ZeroCurvBC(), autocache = true)
            end

            bc = FI.BCPair(FI.Deriv2(0.0), FI.Deriv2(0.0))
            bank = FI._get_derivative_bank(grids[1], bc)

            snap = @atomic :acquire bank.snapshot
            @test snap.count == 5
        end

        @testset "Ring buffer eviction at capacity" begin
            FI.clear_cubic_cache!()

            cache_size = FI.get_cubic_cache_size()
            grids = [collect(range(0.0, Float64(i), 51)) for i in 1:(cache_size + 5)]
            y = sin.(2π .* grids[1])

            for x in grids
                FI.cubic_interp(x, y, 0.5; bc = ZeroCurvBC(), autocache = true)
            end

            bc = FI.BCPair(FI.Deriv2(0.0), FI.Deriv2(0.0))
            bank = FI._get_derivative_bank(grids[1], bc)

            snap = @atomic :acquire bank.snapshot
            @test snap.count == cache_size
            @test length(snap.store) == cache_size
        end
    end

    # ===============================================================
    # Bank Zero-Allocation Read Path
    # ===============================================================
    @testset "Zero-Allocation Read Path" begin
        @testset "Cache hit allocates minimally" begin
            FI.clear_cubic_cache!()

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            FI.cubic_interp(x, y, 0.5; autocache = true)

            # Warm up
            for _ in 1:100
                FI.cubic_interp(x, y, 0.5; autocache = true)
            end

            allocs = @allocated FI.cubic_interp(x, y, 0.5; autocache = true)
            @test allocs <= 256
            @info "Bank: Full interp cache hit allocation: $allocs bytes (threshold: 256 bytes)"
        end
    end

    # ===============================================================
    # Bank Thread Safety
    # ===============================================================
    if Threads.nthreads() > 1
        @testset "Concurrent Access" begin
            @testset "Concurrent read safety" begin
                FI.clear_cubic_cache!()

                x = collect(range(0.0, 1.0, 51))
                y = sin.(2π .* x)
                FI.cubic_interp(x, y, 0.5; autocache = true)

                errors = Threads.Atomic{Int}(0)
                reads = Threads.Atomic{Int}(0)

                Threads.@threads for _ in 1:1000
                    try
                        result = FI.cubic_interp(x, y, 0.5; autocache = true)
                        @assert isapprox(result, 0.0; atol = 1.0e-10)
                        Threads.atomic_add!(reads, 1)
                    catch
                        Threads.atomic_add!(errors, 1)
                    end
                end

                @test errors[] == 0
                @test reads[] == 1000
            end

            @testset "Concurrent read/write safety" begin
                FI.clear_cubic_cache!()

                grids = [collect(range(0.0, Float64(i), 51)) for i in 1:8]
                y = sin.(2π .* grids[1])
                errors = Threads.Atomic{Int}(0)

                Threads.@threads for i in 1:1000
                    try
                        grid_idx = mod1(i, length(grids))
                        FI.cubic_interp(grids[grid_idx], y, 0.5; autocache = true)
                    catch
                        Threads.atomic_add!(errors, 1)
                    end
                end

                @test errors[] == 0
            end
        end
    else
        @testset "Bank Thread Safety (skipped - need julia -t N)" begin
            @test_skip "Need multiple threads"
        end
    end

    # ===============================================================
    # Periodic BC Support
    # ===============================================================
    @testset "Periodic BC RCU" begin
        @testset "Periodic bank uses atomic snapshot" begin
            FI.clear_cubic_cache!()

            x = collect(range(0.0, 2π, 51))
            y = sin.(x)
            y[end] = y[1]
            FI.cubic_interp(x, y, π; bc = FI.PeriodicBC(), autocache = true)

            bank = FI._get_periodic_bank(x, FI.PeriodicBC())
            snap = @atomic :acquire bank.snapshot
            @test snap.count == 1
        end
    end

end  # RCU Bank


# ###################################################################
#                       REGISTRY TESTS
# ###################################################################

@testitem "RCU Registry" begin
    # Common type aliases for tests
    FI = FastInterpolations
    EntryType = FI.CacheEntry{Float64, FI.Deriv2{Float64}, FI.Deriv2{Float64}, Vector{Float64}, FI.CubicSplineCache{Float64, FI._CachedVector{Float64, Float64}, FI.ThomasFactorization{Float64, Vector{Float64}}, FI.BCPair{FI.Deriv2{Float64}, FI.Deriv2{Float64}}}}
    BankType = FI.CacheBank{EntryType}


    # ===============================================================
    # GlobalRegistry Structure
    # ===============================================================
    @testset "GlobalRegistry Structure" begin
        @testset "RegistrySnapshot is Vector of Pairs" begin
            snap = FI.RegistrySnapshot()
            @test snap isa Vector{Pair{DataType, Any}}
            @test isempty(snap)
        end

        @testset "GlobalRegistry has atomic snapshot field" begin
            registry = FI.GlobalRegistry(FI.RegistrySnapshot())
            snap = @atomic :acquire registry.snapshot
            @test snap isa FI.RegistrySnapshot
            @test isempty(snap)
        end

        @testset "GlobalRegistry constructor creates empty snapshot" begin
            registry = FI.GlobalRegistry()
            snap = @atomic :acquire registry.snapshot
            @test isempty(snap)
        end
    end

    # ===============================================================
    # Global Registries Exist
    # ===============================================================
    @testset "Global Registries" begin
        @testset "Derivative registry exists" begin
            @test isdefined(FI, :_DERIVATIVE_REGISTRY)
            @test FI._DERIVATIVE_REGISTRY isa FI.GlobalRegistry
        end

        @testset "Periodic registry exists" begin
            @test isdefined(FI, :_PERIODIC_REGISTRY)
            @test FI._PERIODIC_REGISTRY isa FI.GlobalRegistry
        end
    end

    # ===============================================================
    # Lock-Free Registry Lookup
    # ===============================================================
    @testset "Lock-Free Lookup" begin
        @testset "Lookup on empty registry returns nothing" begin
            FI.clear_cubic_cache!()
            registry = FI.GlobalRegistry()
            result = FI._registry_lookup(registry, BankType)
            @test result === nothing
        end

        @testset "Lookup finds existing bank" begin
            FI.clear_cubic_cache!()

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            FI.cubic_interp(x, y, 0.5; bc = ZeroCurvBC(), autocache = true)

            result = FI._registry_lookup(FI._DERIVATIVE_REGISTRY, BankType)
            @test result !== nothing
            @test result isa BankType
        end

        @testset "Lookup by type identity (fast path)" begin
            FI.clear_cubic_cache!()

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            FI.cubic_interp(x, y, 0.5; bc = ZeroCurvBC(), autocache = true)

            snap = @atomic :acquire FI._DERIVATIVE_REGISTRY.snapshot

            found = any(TypeKey === BankType for (TypeKey, _) in snap)
            @test found
        end
    end

    # ===============================================================
    # RCU Registry Insert
    # ===============================================================
    @testset "RCU Insert" begin
        @testset "Insert creates new snapshot" begin
            FI.clear_cubic_cache!()

            snap_before = @atomic :acquire FI._DERIVATIVE_REGISTRY.snapshot
            @test isempty(snap_before)

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            FI.cubic_interp(x, y, 0.5; autocache = true)

            snap_after = @atomic :acquire FI._DERIVATIVE_REGISTRY.snapshot
            @test length(snap_after) == 1
            @test snap_after !== snap_before
        end

        @testset "Multiple bank types create multiple entries" begin
            FI.clear_cubic_cache!()

            # ZeroCurvBC (Deriv2)
            x1 = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x1)
            FI.cubic_interp(x1, y1, 0.5; bc = ZeroCurvBC(), autocache = true)

            # Custom Deriv1 BC
            x2 = collect(range(0.0, 1.0, 51))
            y2 = cos.(2π .* x2)
            bc_pair = FI.BCPair(FI.Deriv1(1.0), FI.Deriv1(-1.0))
            FI.cubic_interp(x2, y2, 0.5; bc = bc_pair, autocache = true)

            snap = @atomic :acquire FI._DERIVATIVE_REGISTRY.snapshot
            @test length(snap) == 2
        end

        @testset "Periodic BC uses separate registry" begin
            FI.clear_cubic_cache!()

            # Derivative BC (explicit ZeroCurvBC)
            x1 = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x1)
            FI.cubic_interp(x1, y1, 0.5; bc = ZeroCurvBC(), autocache = true)

            # Periodic BC
            x2 = collect(range(0.0, 2π, 51))
            y2 = sin.(x2)
            y2[end] = y2[1]
            FI.cubic_interp(x2, y2, π; bc = FI.PeriodicBC(), autocache = true)

            deriv_snap = @atomic :acquire FI._DERIVATIVE_REGISTRY.snapshot
            periodic_snap = @atomic :acquire FI._PERIODIC_REGISTRY.snapshot

            @test length(deriv_snap) == 1
            @test length(periodic_snap) == 1
        end
    end

    # ===============================================================
    # clear_cubic_cache! with Atomic Registry
    # ===============================================================
    @testset "clear_cubic_cache! Atomic" begin
        @testset "Clear empties both registries" begin
            x1 = collect(range(0.0, 1.0, 51))
            y1 = sin.(2π .* x1)
            FI.cubic_interp(x1, y1, 0.5; autocache = true)

            x2 = collect(range(0.0, 2π, 51))
            y2 = sin.(x2)
            y2[end] = y2[1]
            FI.cubic_interp(x2, y2, π; bc = FI.PeriodicBC(), autocache = true)

            @test !isempty(@atomic :acquire FI._DERIVATIVE_REGISTRY.snapshot)
            @test !isempty(@atomic :acquire FI._PERIODIC_REGISTRY.snapshot)

            FI.clear_cubic_cache!()

            @test isempty(@atomic :acquire FI._DERIVATIVE_REGISTRY.snapshot)
            @test isempty(@atomic :acquire FI._PERIODIC_REGISTRY.snapshot)
        end

        @testset "Clear atomically replaces snapshot" begin
            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            FI.cubic_interp(x, y, 0.5; autocache = true)

            snap_before = @atomic :acquire FI._DERIVATIVE_REGISTRY.snapshot

            FI.clear_cubic_cache!()

            snap_after = @atomic :acquire FI._DERIVATIVE_REGISTRY.snapshot

            @test snap_after !== snap_before
            @test isempty(snap_after)
            @test !isempty(snap_before)  # Old snapshot unchanged (RCU property)
        end
    end

    # ===============================================================
    # Registry Zero-Allocation Lookup
    # ===============================================================
    @testset "Zero-Allocation Lookup" begin
        @testset "Registry lookup allocates minimally" begin
            FI.clear_cubic_cache!()

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            FI.cubic_interp(x, y, 0.5; autocache = true)

            # Warm up
            for _ in 1:100
                FI._registry_lookup(FI._DERIVATIVE_REGISTRY, BankType)
            end

            allocs = @allocated FI._registry_lookup(FI._DERIVATIVE_REGISTRY, BankType)
            @test allocs <= 64
            @info "Registry: lookup allocation: $allocs bytes"
        end
    end

    # ===============================================================
    # Registry Thread Safety
    # ===============================================================
    if Threads.nthreads() > 1
        @testset "Concurrent Access" begin
            @testset "Concurrent bank creation is safe" begin
                errors = Threads.Atomic{Int}(0)

                for _ in 1:100
                    FI.clear_cubic_cache!()

                    grids = [collect(range(0.0, Float64(i), 51)) for i in 1:4]
                    y = sin.(2π .* grids[1])

                    Threads.@threads for i in 1:4
                        try
                            FI.cubic_interp(grids[mod1(i, 4)], y, 0.5; autocache = true)
                        catch
                            Threads.atomic_add!(errors, 1)
                        end
                    end
                end

                @test errors[] == 0
            end

            @testset "Concurrent lookup/insert is safe" begin
                FI.clear_cubic_cache!()

                grids = [collect(range(0.0, Float64(i), 51)) for i in 1:8]
                y = sin.(2π .* grids[1])
                errors = Threads.Atomic{Int}(0)

                Threads.@threads for i in 1:1000
                    try
                        grid_idx = mod1(i, length(grids))
                        FI.cubic_interp(grids[grid_idx], y, 0.5; autocache = true)
                    catch
                        Threads.atomic_add!(errors, 1)
                    end
                end

                @test errors[] == 0
            end

            @testset "Concurrent clear/insert is safe" begin
                errors = Threads.Atomic{Int}(0)

                for _ in 1:50
                    x = collect(range(0.0, 1.0, 51))
                    y = sin.(2π .* x)

                    Threads.@threads for i in 1:4
                        try
                            if i == 1
                                FI.clear_cubic_cache!()
                            else
                                FI.cubic_interp(x, y, 0.5; autocache = true)
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
        @testset "Registry Thread Safety (skipped - need julia -t N)" begin
            @test_skip "Need multiple threads"
        end
    end

end  # RCU Registry
