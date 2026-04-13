# Import internal functions for testing
import FastInterpolations: _get_cubic_cache, _get_derivative_cache_impl

# =============================================================================
# TESTSET 1: Basic Cache Operations
# =============================================================================
@testset "Cubic Cache: Basic Operations" begin
    clear_cubic_cache!()

    @testset "Basic auto-cache reuse" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        y3 = exp.(-x)
        x_query = [0.25, 0.5, 0.75]

        clear_cubic_cache!()

        # First call - cache miss
        result1 = cubic_interp(x, y1, x_query)

        # Second call with same x - cache hit
        result2 = cubic_interp(x, y2, x_query)

        # Third call with same x - another cache hit
        result3 = cubic_interp(x, y3, x_query)

        # Results should be different (different y values)
        @test result1 != result2
        @test result2 != result3
        @test result1 != result3

        # But all should be finite and reasonable
        @test all(isfinite, result1)
        @test all(isfinite, result2)
        @test all(isfinite, result3)
    end

    @testset "Multiple x-grids" begin
        clear_cubic_cache!()

        # Create 3 different x-grids
        x1 = collect(range(0.0, 1.0, 51))
        x2 = collect(range(0.0, 2.0, 51))
        x3 = collect(range(0.0, 3.0, 51))

        y = sin.(2π .* x1)
        x_query = [0.25, 0.5, 0.75]

        # Each grid creates a new cache entry
        r1 = cubic_interp(x1, y, x_query)
        r2 = cubic_interp(x2, y, x_query)
        r3 = cubic_interp(x3, y, x_query)

        # Reuse first grid - should hit cache and give same result
        r1_again = cubic_interp(x1, y, x_query)
        @test r1 ≈ r1_again
    end

    @testset "Cache size limit" begin
        clear_cubic_cache!()
        set_cubic_cache_size!(3)  # Small cache for testing

        # Create 4 different x-grids (exceeds cache size)
        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:4]
        y = ones(51)
        x_query = [0.25]

        # Fill cache to limit (3 grids)
        for i in 1:3
            cubic_interp(grids[i], y, x_query)
        end

        # Add 4th grid - should evict oldest
        result4 = cubic_interp(grids[4], y, x_query)
        @test isfinite(result4[1])

        # Reset cache size to default
        set_cubic_cache_size!(16)
    end

    @testset "autocache parameter control" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5]

        # With autocache=true (default)
        result1 = cubic_interp(x, y, x_query)

        # Another call with autocache=true - should hit cache
        result2 = cubic_interp(x, y, x_query; autocache = true)

        # With autocache=false - creates fresh cache each time
        result3 = cubic_interp(x, y, x_query; autocache = false)

        # Results should be identical
        @test result1 ≈ result2
        @test result2 ≈ result3
    end

    @testset "Manual cache control" begin
        clear_cubic_cache!()

        # Cache size is a load-time constant via Preferences.jl
        # get_cubic_cache_size() returns the current (immutable) value
        @test get_cubic_cache_size() == 8  # Default (reduced from 16 for faster worst-case scan)

        # set_cubic_cache_size! saves to Preferences but doesn't change current session
        # (change takes effect after Julia restart)
        @test set_cubic_cache_size!(32) == 32  # Returns the requested value
        @test get_cubic_cache_size() == 8      # Still 8 until restart

        # Clean up: reset preference back to default for future sessions
        set_cubic_cache_size!(8)

        # Test clear works without error
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        cubic_interp(x, y, [0.5])

        clear_cubic_cache!()  # Should complete without error
    end
end

# =============================================================================
# TESTSET 2: Stress Tests & Edge Cases
# =============================================================================
@testset "Cubic Cache: Stress Tests & Edge Cases" begin
    @testset "Hash collision handling (stress test)" begin
        clear_cubic_cache!()

        # Create many similar grids to increase collision probability
        n_grids = 50
        grids = [collect(range(0.0, 1.0, 51)) .+ (i * 1.0e-10) for i in 1:n_grids]
        y = ones(51)
        x_query = [0.5]

        # All should cache successfully and produce valid results
        for grid in grids
            result = cubic_interp(grid, y, x_query)
            @test isfinite(result[1])
        end
    end

    @testset "Scalar query point with autocache" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Scalar query with autocache
        result1 = cubic_interp(x, y, 0.5)
        result2 = cubic_interp(x, y, 0.75)

        # Disable autocache for scalar query
        result3 = cubic_interp(x, y, 0.25; autocache = false)

        @test isa(result1, Float64)
        @test isa(result2, Float64)
        @test isa(result3, Float64)
        @test isfinite(result1)
        @test isfinite(result2)
        @test isfinite(result3)
    end

    @testset "Correctness under heavy reuse" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        x_query = [0.2, 0.4, 0.6, 0.8]

        # Generate 20 different y functions
        for k in 1:20
            y = @. sin(k * 2π * x) + 0.5 * cos(k * 4π * x)

            result = cubic_interp(x, y, x_query)

            # Verify all results are finite
            @test all(isfinite, result)
        end
    end
end

# =============================================================================
# TESTSET 3: Type Support (Integer, Float32, AbstractRange, AbstractVector)
# =============================================================================
@testset "Cubic Cache: Type Support" begin
    @testset "Auto-cache with Integer inputs" begin
        clear_cubic_cache!()

        x_int = 0:10
        y_int = [sin(2π * i / 10) for i in x_int]
        x_query_float = [2.5, 5.5, 7.3]

        # First call with integer inputs (should create cache)
        result1 = cubic_interp(x_int, y_int, x_query_float)

        # Second call with same integer x (should reuse cache)
        y_int2 = [cos(2π * i / 10) for i in x_int]
        result2 = cubic_interp(x_int, y_int2, x_query_float)

        # Verify results are correct
        @test all(isfinite, result1)
        @test all(isfinite, result2)
        @test result1 != result2
    end

    @testset "Autocache=false with Integer inputs" begin
        clear_cubic_cache!()

        x_int = 0:10
        y_int = [sin(2π * i / 10) for i in x_int]
        x_query_float = [2.5, 5.5, 7.3]

        result = cubic_interp(x_int, y_int, x_query_float; autocache = false)
        @test result isa Vector{Float64}
        @test all(isfinite, result)
    end

    @testset "AbstractRange fallback paths" begin
        clear_cubic_cache!()

        # Test with UnitRange (not StepRangeLen)
        x_unitrange = 1:51
        y = sin.(2π .* collect(x_unitrange) ./ 51)

        # This should trigger the AbstractRange{Float64} fallback
        result = cubic_interp(Float64.(x_unitrange), Float64.(y), 25.0)
        @test !isnan(result)
    end

    @testset "Float32 cache operations" begin
        clear_cubic_cache!()
        old_size = get_cubic_cache_size()
        set_cubic_cache_size!(2)  # Small cache for testing

        y = Float32.(ones(51))

        # Fill cache with Float32 Range grids
        x1_range = range(Float32(0), Float32(1), 51)
        x2_range = range(Float32(0), Float32(2), 51)
        x3_range = range(Float32(0), Float32(3), 51)

        r1 = cubic_interp(x1_range, y, Float32(0.5))
        r2 = cubic_interp(x2_range, y, Float32(0.5))
        r3 = cubic_interp(x3_range, y, Float32(0.5))  # Should trigger eviction

        @test isfinite(r1)
        @test isfinite(r2)
        @test isfinite(r3)

        # Fill cache with Float32 Vector grids
        x1_vec = Float32.(collect(range(0.0, 1.0, 51)))
        x2_vec = Float32.(collect(range(0.0, 2.0, 51)))
        x3_vec = Float32.(collect(range(0.0, 3.0, 51)))

        r4 = cubic_interp(x1_vec, y, Float32(0.5))
        r5 = cubic_interp(x2_vec, y, Float32(0.5))
        r6 = cubic_interp(x3_vec, y, Float32(0.5))  # Should trigger eviction

        @test isfinite(r4)
        @test isfinite(r5)
        @test isfinite(r6)

        set_cubic_cache_size!(old_size)
    end

    @testset "Float32 self-healing path" begin
        clear_cubic_cache!()

        x1 = Float32.(collect(range(0.0, 1.0, 51)))
        y = Float32.(sin.(2π .* x1))

        # Prime cache
        result1 = cubic_interp(x1, y, Float32(0.5))

        # Create equal but different object
        x2 = Float32.(collect(range(0.0, 1.0, 51)))
        @test x1 == x2
        @test objectid(x1) != objectid(x2)

        # This should trigger Pass 2 (equality check) for Float32
        result2 = cubic_interp(x2, y, Float32(0.5))

        # Results should be the same
        @test result1 ≈ result2
    end

    @testset "Range eviction paths" begin
        clear_cubic_cache!()
        old_size = get_cubic_cache_size()
        set_cubic_cache_size!(2)

        y = ones(51)

        # Test Float64 Range eviction
        grids = [range(0.0, Float64(i), 51) for i in 1:4]
        for grid in grids
            result = cubic_interp(grid, y, 0.5)
            @test isfinite(result)
        end

        set_cubic_cache_size!(old_size)
    end

    @testset "Range cache hit" begin
        clear_cubic_cache!()

        x = range(0.0, 1.0, 51)
        y = sin.(2π .* collect(x))

        # Prime cache
        result1 = cubic_interp(x, y, 0.5)

        # Second call should hit cache (same objectid for Range)
        result2 = cubic_interp(x, y, 0.5)

        @test result1 ≈ result2
    end

    # =========================================================================
    # AbstractVector API Compatibility Tests
    # =========================================================================

    @testset "_get_cubic_cache accepts AbstractVector (views, SubArrays)" begin
        clear_cubic_cache!()

        x_full = collect(range(0.0, 1.0, 101))
        y_full = sin.(2π .* x_full)

        # SubArray (view) should work
        x_view = @view x_full[1:51]
        y_view = @view y_full[1:51]

        cache = _get_cubic_cache(x_view, ZeroCurvBC())
        @test cache isa CubicSplineCache

        # Cache should store collected Vector, not the view
        result = cubic_interp(collect(x_view), collect(y_view), 0.5)
        @test isfinite(result)
    end

    @testset "_get_cubic_cache accepts Float32 views" begin
        clear_cubic_cache!()

        x_full = Float32.(collect(range(0.0, 1.0, 101)))
        x_view = @view x_full[1:51]

        cache = _get_cubic_cache(x_view, ZeroCurvBC())
        @test cache isa CubicSplineCache{Float32}
    end

    @testset "_get_cubic_cache fallback for other Real types" begin
        clear_cubic_cache!()

        # Integer range → should convert to Float64
        x_int = 0:10
        cache_int = _get_cubic_cache(x_int, ZeroCurvBC())
        @test cache_int isa CubicSplineCache{Float64}

        # Float16 vector → kept as Float16 (native AbstractFloat)
        x_f16 = Float16.(collect(range(0.0, 1.0, 11)))
        cache_f16 = _get_cubic_cache(x_f16, ZeroCurvBC())
        @test cache_f16 isa CubicSplineCache{Float16}
    end

    @testset "Non-float grid cache: Integer, Rational, Periodic, Duck-type" begin
        clear_cubic_cache!()

        # ── Integer Vector: derivative + periodic cache ──
        x_int_v = [0, 1, 2, 3, 4]
        y_int = sin.(Float64.(x_int_v))

        # Derivative cache (Integer dispatch)
        cache_int_d = _get_cubic_cache(x_int_v, ZeroCurvBC())
        @test cache_int_d isa CubicSplineCache{Float64}

        # Cache hit on repeat (cross-type isequal: Float64 entry vs Int input)
        cache_int_d2 = _get_cubic_cache(x_int_v, ZeroCurvBC())
        @test cache_int_d2 === cache_int_d

        # Periodic cache (Integer dispatch)
        x_periodic_int = [0, 1, 2, 3, 4, 5, 6]
        y_periodic = sin.(2π .* Float64.(x_periodic_int) ./ 6)
        cache_int_p = _get_cubic_cache(x_periodic_int, PeriodicBC())
        @test cache_int_p isa CubicSplineCache{Float64}

        # ── Rational Vector: derivative + periodic cache ──
        x_rat = Rational{Int}[0 // 1, 1 // 1, 2 // 1, 3 // 1, 4 // 1]

        cache_rat_d = _get_cubic_cache(x_rat, ZeroCurvBC())
        @test cache_rat_d isa CubicSplineCache{Float64}

        x_rat_periodic = Rational{Int}[0 // 1, 1 // 1, 2 // 1, 3 // 1, 4 // 1, 5 // 1, 6 // 1]
        cache_rat_p = _get_cubic_cache(x_rat_periodic, PeriodicBC())
        @test cache_rat_p isa CubicSplineCache{Float64}

        # ── Integer with BCPair, PointBC, ZeroSlopeBC ──
        cache_int_bp = _get_cubic_cache(x_int_v, BCPair(Deriv2(0.0), Deriv2(0.0)))
        @test cache_int_bp isa CubicSplineCache{Float64}

        cache_int_pb = _get_cubic_cache(x_int_v, Deriv2(0.0))
        @test cache_int_pb isa CubicSplineCache{Float64}

        cache_int_zs = _get_cubic_cache(x_int_v, ZeroSlopeBC())
        @test cache_int_zs isa CubicSplineCache{Float64}

        # ── Integer with autocache=true (3-arg path) ──
        clear_cubic_cache!()
        bc_pair = BCPair(Deriv2(0.0), Deriv2(0.0))
        cache_int_ac = _get_cubic_cache(x_int_v, bc_pair, true)
        @test cache_int_ac isa CubicSplineCache{Float64}
        # Second call should hit cache
        cache_int_ac2 = _get_cubic_cache(x_int_v, bc_pair, true)
        @test cache_int_ac2 === cache_int_ac

        # ── Integer with autocache=false (3-arg, build fresh) ──
        cache_int_no = _get_cubic_cache(x_int_v, bc_pair, false)
        @test cache_int_no isa CubicSplineCache{Float64}
        @test cache_int_no !== cache_int_ac  # fresh, not cached

        # ── Correctness: Int grid cubic_interp matches Float grid ──
        x_flt = Float64.(x_int_v)
        ref = cubic_interp(x_flt, y_int, 1.5; extrap = ExtendExtrap())
        val = cubic_interp(x_int_v, y_int, 1.5; extrap = ExtendExtrap())
        @test val ≈ ref
    end

    @testset "_get_cubic_cache keyword API" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))

        # Keyword API should work
        cache1 = _get_cubic_cache(x)  # default bc=CubicFit()
        cache2 = _get_cubic_cache(x; bc = CubicFit())
        cache3 = _get_cubic_cache(x; bc = PeriodicBC())

        @test cache1 isa CubicSplineCache
        @test cache2 isa CubicSplineCache
        @test cache3 isa CubicSplineCache

        # CubicFit and periodic caches should be different types
        @test typeof(cache1) == typeof(cache2)
        @test typeof(cache1) != typeof(cache3)
    end

    @testset "_get_cubic_cache typed BC API (type-stable path)" begin
        clear_cubic_cache!()

        x64 = collect(range(0.0, 1.0, 51))
        x32 = Float32.(x64)
        x_range = range(0.0, 1.0, 51)

        # All input types should work with Val API
        c1 = _get_cubic_cache(x64, ZeroCurvBC())
        c2 = _get_cubic_cache(x32, ZeroCurvBC())
        c3 = _get_cubic_cache(x_range, ZeroCurvBC())
        c4 = _get_cubic_cache(x64, PeriodicBC())

        @test c1 isa CubicSplineCache{Float64}
        @test c2 isa CubicSplineCache{Float32}
        @test c3 isa CubicSplineCache{Float64}  # Range normalizes to Float64
        @test c4 isa CubicSplineCache{Float64}

        # Periodic cache should have PeriodicData BC
        @test c4.bc_config !== nothing
    end
end

# =============================================================================
# TESTSET 4: Boundary Condition Coverage
# =============================================================================
@testset "Cubic Cache: Boundary Condition Coverage" begin
    @testset "_get_cubic_cache with ZeroSlopeBC (typed API)" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))

        # ZeroSlopeBC typed API - previously uncovered
        cache = _get_cubic_cache(x, ZeroSlopeBC())
        @test cache isa CubicSplineCache{Float64}

        # Cache should be created with BCPair(Deriv1(0), Deriv1(0))
        @test cache.bc_config isa BCPair{Deriv1{Float64}, Deriv1{Float64}}

        # Should work for Float32 as well
        x32 = Float32.(x)
        cache32 = _get_cubic_cache(x32, ZeroSlopeBC())
        @test cache32 isa CubicSplineCache{Float32}
        @test cache32.bc_config isa BCPair{Deriv1{Float32}, Deriv1{Float32}}

        # Range input
        x_range = range(0.0, 1.0, 51)
        cache_range = _get_cubic_cache(x_range, ZeroSlopeBC())
        @test cache_range isa CubicSplineCache{Float64}
    end

    @testset "_get_cubic_cache with PointBC (convenience API)" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))

        # Deriv1 PointBC - applies symmetrically to both ends
        # Note: LU factorization depends only on BC TYPE (Deriv1 vs Deriv2), not values.
        # Cache stores structural zeros — solver uses caller's original BC values at solve time.
        cache_d1 = _get_cubic_cache(x, Deriv1(0.5))
        @test cache_d1 isa CubicSplineCache{Float64}
        @test cache_d1.bc_config isa BCPair{Deriv1{Float64}, Deriv1{Float64}}
        # Pool caches store structural zeros (values are irrelevant to LU factorization)
        @test cache_d1.bc_config.left.val == 0.0
        @test cache_d1.bc_config.right.val == 0.0

        # Deriv2 PointBC - applies symmetrically to both ends
        cache_d2 = _get_cubic_cache(x, Deriv2(1.0))
        @test cache_d2 isa CubicSplineCache{Float64}
        @test cache_d2.bc_config isa BCPair{Deriv2{Float64}, Deriv2{Float64}}
        # Pool caches store structural zeros (values are irrelevant to LU factorization)
        @test cache_d2.bc_config.left.val == 0.0
        @test cache_d2.bc_config.right.val == 0.0

        # Float32 with PointBC
        x32 = Float32.(x)
        cache_d1_32 = _get_cubic_cache(x32, Deriv1(Float32(0.5)))
        @test cache_d1_32 isa CubicSplineCache{Float32}
    end

    @testset "Int Vector fallback paths" begin
        clear_cubic_cache!()

        # Integer Vector with BCPair - should convert to Float64
        x_int = collect(0:10)
        cache = _get_cubic_cache(x_int, BCPair(Deriv1(0.0), Deriv2(0.0)))
        @test cache isa CubicSplineCache{Float64}

        # Integer Vector with periodic BC - should convert to Float64
        clear_cubic_cache!()
        cache_periodic = _get_cubic_cache(x_int, PeriodicBC())
        @test cache_periodic isa CubicSplineCache{Float64}

        # Integer Range with periodic BC - should convert to Float64
        clear_cubic_cache!()
        x_int_range = 0:10
        cache_periodic_range = _get_cubic_cache(x_int_range, PeriodicBC())
        @test cache_periodic_range isa CubicSplineCache{Float64}
    end

    @testset "Periodic cache self-healing path" begin
        clear_cubic_cache!()

        # Create first grid and cache
        x1 = collect(range(0.0, 2π, 51))
        y = sin.(x1)
        y[end] = y[1]  # Ensure exact periodicity
        result1 = cubic_interp(x1, y, 0.5; bc = PeriodicBC())

        # Create equal but different object (different objectid)
        x2 = collect(range(0.0, 2π, 51))
        @test x1 == x2
        @test objectid(x1) != objectid(x2)

        # This should trigger Pass 2 (equality check) and self-healing
        result2 = cubic_interp(x2, y, 0.5; bc = PeriodicBC())

        # Results should be the same
        @test result1 ≈ result2

        # Now x2 should trigger Pass 1 (identity check) due to self-healing
        result3 = cubic_interp(x2, y, 0.5; bc = PeriodicBC())
        @test result2 ≈ result3
    end
end

# =============================================================================
# TESTSET 5: Mutation Safety & Cache Invalidation
# =============================================================================
@testset "Cubic Cache: Mutation Safety" begin
    @testset "Vector x mutation safety" begin
        # Verify autocache detects in-place mutation and rebuilds correctly.
        # Key invariant: before !== after (result changed) AND after == fresh_build
        clear_cubic_cache!()

        x = collect(range(0.0, 10.0, 11))  # [0,1,2,...,10]
        y = sin.(x)
        xq = [1.5, 4.5, 7.5]

        # Before mutation
        before = cubic_interp(x, y, xq; autocache = true)

        # Mutation 1: middle position
        x[6] = 5.5
        after = cubic_interp(x, y, xq; autocache = true)
        fresh = cubic_interp(x, y, xq; autocache = false)
        @test before != after  # Result must change after mutation
        @test after == fresh   # Autocache must match fresh build

        # Mutation 2: another position (sequential)
        before = after
        x[3] = 2.2
        after = cubic_interp(x, y, xq; autocache = true)
        fresh = cubic_interp(x, y, xq; autocache = false)
        @test before != after
        @test after == fresh

        # Mutation 3: end position
        before = after
        x[11] = 11.0
        after = cubic_interp(x, y, xq; autocache = true)
        fresh = cubic_interp(x, y, xq; autocache = false)
        @test before != after
        @test after == fresh
    end

    @testset "Float32 and PeriodicBC mutation safety" begin
        # Float32
        clear_cubic_cache!()
        x32 = Float32.(collect(range(0.0, 4.0, 9)))
        y32 = Float32.(sin.(x32))
        before = cubic_interp(x32, y32, Float32(2.0); autocache = true)
        x32[5] = Float32(2.2)
        after = cubic_interp(x32, y32, Float32(2.0); autocache = true)
        fresh = cubic_interp(x32, y32, Float32(2.0); autocache = false)
        @test before != after && after == fresh

        # PeriodicBC
        clear_cubic_cache!()
        xp = collect(range(0.0, 2π, 9))
        yp = sin.(xp)
        yp[end] = yp[1]  # Ensure exact periodicity
        before = cubic_interp(xp, yp, 1.0; bc = PeriodicBC(), autocache = true)
        xp[5] += 0.1
        after = cubic_interp(xp, yp, 1.0; bc = PeriodicBC(), autocache = true)
        fresh = cubic_interp(xp, yp, 1.0; bc = PeriodicBC(), autocache = false)
        @test before != after && after == fresh
    end

    @testset "AbstractRange immutability (should already pass)" begin
        # Test 1.2: Verify Range inputs don't have mutation issue (baseline)
        # Expected: PASS (ranges are already immutable in Julia)

        @testset "StepRangeLen immutability" begin
            clear_cubic_cache!()

            x = range(0.0, 4.0, 5)  # StepRangeLen
            y = sin.(collect(x))
            xq = 2.0

            # Prime cache
            result1 = cubic_interp(x, y, xq; autocache = true)

            # Ranges are immutable - creating a new range with same values
            # should hit cache via isequal check
            x2 = range(0.0, 4.0, 5)
            result2 = cubic_interp(x2, y, xq; autocache = true)

            # Both should produce consistent results
            @test result1 ≈ result2
        end

        @testset "LinRange immutability" begin
            clear_cubic_cache!()

            x = LinRange(0.0, 4.0, 5)
            y = cos.(collect(x))
            xq = [1.0, 2.0, 3.0]

            # Prime cache
            result1 = cubic_interp(x, y, xq; autocache = true)

            # Fresh computation should match
            fresh = cubic_interp(x, y, xq; autocache = false)

            @test result1 ≈ fresh
        end

        @testset "Range with different representations" begin
            clear_cubic_cache!()

            # Create equivalent ranges with different constructors
            x1 = range(0.0, 10.0, 11)
            x2 = LinRange(0.0, 10.0, 11)

            y = exp.(-collect(x1) ./ 10)
            xq = 5.0

            # Both should produce same result
            result1 = cubic_interp(x1, y, xq; autocache = true)
            result2 = cubic_interp(x2, y, xq; autocache = true)

            @test result1 ≈ result2
        end

        @testset "Float32 Range immutability" begin
            clear_cubic_cache!()

            x = range(Float32(0), Float32(4), 5)
            y = Float32.(sin.(collect(x)))
            xq = Float32(2.0)

            result1 = cubic_interp(x, y, xq; autocache = true)
            result2 = cubic_interp(x, y, xq; autocache = true)

            @test result1 ≈ result2
        end
    end

    # =========================================================================
    # Explicit Cache Invalidation Tests
    # =========================================================================

    @testset "Explicit cache object invalidation after mutation" begin
        # Verify that mutation creates a NEW cache object, not reusing old one
        clear_cubic_cache!()

        x = collect(range(0.0, 10.0, 11))
        y = sin.(x)

        # Prime cache and get cache object
        cache_before = _get_cubic_cache(x, ZeroCurvBC())
        cache_id_before = objectid(cache_before)

        # Mutate x in-place
        x[6] = 5.5

        # Get cache again - should be a DIFFERENT cache object
        cache_after = _get_cubic_cache(x, ZeroCurvBC())
        cache_id_after = objectid(cache_after)

        # Key assertion: cache objects must be different after mutation
        @test cache_id_before != cache_id_after

        # Additional verification: the new cache should have the mutated x stored
        # (via snapshot mechanism)
        @test cache_after.x[6] == 5.5
    end

    @testset "Cache invalidation with PeriodicBC" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 2π, 9))
        cache_before = _get_cubic_cache(x, PeriodicBC())

        x[5] += 0.1
        cache_after = _get_cubic_cache(x, PeriodicBC())

        @test objectid(cache_before) != objectid(cache_after)
        @test cache_after.x[5] ≈ x[5]
    end

    # =========================================================================
    # CubicInterpolant Immutability Tests
    # =========================================================================

    @testset "CubicInterpolant immutability under x mutation" begin
        # Once an interpolant is created, it should return identical results
        # regardless of subsequent mutations to the original x array
        clear_cubic_cache!()

        x = collect(range(0.0, 4.0, 9))
        y = sin.(x)
        xq = [0.5, 1.5, 2.5, 3.5]

        # Create interpolant
        itp = cubic_interp(x, y)

        # Evaluate before mutation
        result_before = itp.(xq)

        # Mutate original x (should NOT affect interpolant)
        x[5] = 100.0  # Drastic change

        # Evaluate after mutation - should be IDENTICAL
        result_after = itp.(xq)

        @test result_before == result_after  # Exact equality, not ≈
    end

    @testset "CubicInterpolant immutability under y mutation" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 4.0, 9))
        y = cos.(x)
        xq = [0.5, 1.5, 2.5, 3.5]

        itp = cubic_interp(x, y)
        result_before = itp.(xq)

        # Mutate original y (should NOT affect interpolant)
        y[5] = 999.0

        result_after = itp.(xq)

        @test result_before == result_after
    end

    @testset "CubicInterpolant immutability with PeriodicBC" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 2π, 17))
        y = sin.(x)
        y[end] = y[1]  # Ensure exact periodicity
        xq = [π / 4, π / 2, π, 3π / 2]

        itp = cubic_interp(x, y; bc = PeriodicBC())
        result_before = itp.(xq)

        x[9] = 100.0  # Mutate middle point
        y[1] = 999.0  # Mutate first point

        result_after = itp.(xq)

        @test result_before == result_after
    end
end

# =============================================================================
# TESTSET 6: Analytic Correctness After Mutation
# =============================================================================
@testset "Cubic Cache: Analytic Correctness" begin
    @testset "Analytic correctness: cubic polynomial with ZeroSlopeBC after mutation" begin
        # Cubic splines with CLAMPED BC (exact endpoint derivatives) reproduce
        # cubic polynomials EXACTLY. Zero-Curvature BC only reproduces linears.
        # This verifies autocache produces correct results after mutation.

        clear_cubic_cache!()

        # Define a cubic polynomial: f(x) = 2x³ - 3x² + x - 1
        # f'(x) = 6x² - 6x + 1
        f(x) = 2x^3 - 3x^2 + x - 1
        df(x) = 6x^2 - 6x + 1

        # Initial grid
        x = collect(range(0.0, 5.0, 11))
        y = f.(x)
        xq = [0.5, 1.25, 2.75, 4.0]

        # Use Zero-Slope BC with exact derivatives for polynomial exactness
        bc = BCPair(Deriv1(df(x[1])), Deriv1(df(x[end])))

        # Prime cache
        result1 = cubic_interp(x, y, xq; bc = bc, autocache = true)

        # Verify initial result is exact (cubic spline reproduces cubics with exact-derivative BC)
        expected1 = f.(xq)
        @test result1 ≈ expected1 atol = 1.0e-12

        # MUTATE x to create a different grid (non-uniform spacing)
        x[3] = 0.8   # was 1.0
        x[6] = 2.3   # was 2.5
        x[9] = 4.2   # was 4.0

        # Recompute y for mutated x, update BC for new endpoints
        y .= f.(x)
        bc_new = BCPair(Deriv1(df(x[1])), Deriv1(df(x[end])))

        # After mutation, autocache should rebuild and give CORRECT results
        result2 = cubic_interp(x, y, xq; bc = bc_new, autocache = true)
        expected2 = f.(xq)  # Same expected values (polynomial is exact)

        @test result2 ≈ expected2 atol = 1.0e-12

        # Cross-check: fresh computation should match
        fresh = cubic_interp(x, y, xq; bc = bc_new, autocache = false)
        @test result2 ≈ fresh atol = 1.0e-14
    end

    @testset "Analytic correctness: quadratic polynomial with ZeroSlopeBC" begin
        # Quadratic: f(x) = x² - 2x + 3, f'(x) = 2x - 2
        # Zero-Slope BC with correct derivatives should give exact results

        clear_cubic_cache!()

        f(x) = x^2 - 2x + 3
        df(x) = 2x - 2

        x = collect(range(-1.0, 3.0, 9))
        y = f.(x)
        xq = [-0.5, 0.5, 1.5, 2.5]

        # Use Deriv1 BC with exact derivatives
        bc = BCPair(Deriv1(df(x[1])), Deriv1(df(x[end])))

        result1 = cubic_interp(x, y, xq; bc = bc, autocache = true)
        expected = f.(xq)
        @test result1 ≈ expected atol = 1.0e-12

        # Mutate grid
        x[4] = 0.3  # was 0.0
        x[6] = 1.3  # was 1.5
        y .= f.(x)

        # Update BC for new endpoints (endpoints unchanged, so BC same)
        result2 = cubic_interp(x, y, xq; bc = bc, autocache = true)
        @test result2 ≈ expected atol = 1.0e-12
    end

    @testset "Analytic correctness: linear function with ZeroCurvBC" begin
        # Linear: f(x) = 3x + 2
        # Zero-Curvature BC (f''=0 at endpoints) is exact for linear functions

        clear_cubic_cache!()

        f(x) = 3x + 2

        x = collect(range(0.0, 10.0, 11))
        y = f.(x)
        xq = [0.5, 3.33, 7.77, 9.5]

        result1 = cubic_interp(x, y, xq; autocache = true)
        @test result1 ≈ f.(xq) atol = 1.0e-13

        # Mutate to non-uniform grid
        x[2] = 0.5
        x[5] = 3.7
        x[8] = 7.2
        y .= f.(x)

        result2 = cubic_interp(x, y, xq; autocache = true)
        @test result2 ≈ f.(xq) atol = 1.0e-13

        # Verify autocache matches fresh
        @test result2 ≈ cubic_interp(x, y, xq; autocache = false) atol = 1.0e-14
    end

    @testset "Analytic correctness: Float32 with mutation" begin
        clear_cubic_cache!()

        f(x) = x^2 - x + 1

        x = Float32.(collect(range(0.0, 4.0, 9)))
        y = f.(x)
        xq = Float32.([0.5, 1.5, 2.5, 3.5])

        result1 = cubic_interp(x, y, xq; autocache = true)
        @test result1 ≈ f.(xq) atol = 1.0e-5  # Float32 precision

        x[5] = Float32(2.1)
        y .= f.(x)

        result2 = cubic_interp(x, y, xq; autocache = true)
        @test result2 ≈ f.(xq) atol = 1.0e-5
    end

    @testset "Analytic correctness: PeriodicBC with sine function" begin
        # sin(x) is periodic, and with correct periodic BC should be very accurate
        # Key test: after mutation, autocache result MUST match fresh computation

        clear_cubic_cache!()

        x = collect(range(0.0, 2π, 33))  # Dense grid for accuracy
        y = sin.(x)
        y[end] = y[1]  # Ensure exact periodicity
        xq = [π / 6, π / 3, π / 2, 2π / 3, π, 4π / 3, 3π / 2, 5π / 3]

        result1 = cubic_interp(x, y, xq; bc = PeriodicBC(), autocache = true)
        expected = sin.(xq)
        @test result1 ≈ expected atol = 1.0e-5  # Cubic spline approximation error

        # Key correctness check: autocache matches fresh for initial grid
        fresh1 = cubic_interp(x, y, xq; bc = PeriodicBC(), autocache = false)
        @test result1 ≈ fresh1 atol = 1.0e-14

        # Mutate interior points (keep endpoints for periodicity)
        x[10] = x[10] + 0.02
        x[20] = x[20] - 0.02
        y .= sin.(x)
        y[end] = y[1]  # Re-ensure exact periodicity after mutation

        result2 = cubic_interp(x, y, xq; bc = PeriodicBC(), autocache = true)
        # After mutation, still interpolating sin, should be close to expected
        @test result2 ≈ expected atol = 1.0e-4  # Slightly looser due to grid perturbation

        # CRITICAL: autocache must match fresh computation EXACTLY after mutation
        # This is the key test - if stale cache was used, this would fail
        fresh2 = cubic_interp(x, y, xq; bc = PeriodicBC(), autocache = false)
        @test result2 ≈ fresh2 atol = 1.0e-14
    end
end

# =============================================================================
# _get_derivative_cache_impl: AbstractRange fallback normalizes to _CachedRange
# =============================================================================
@testset "Cubic Cache: _get_derivative_cache_impl AbstractRange fallback" begin
    clear_cubic_cache!()

    x_range = range(-1.0, 2.0, 31)   # StepRangeLen, NOT _CachedRange
    bc_pair = BCPair(Deriv1(0.0), Deriv1(0.0))

    # Direct call with raw StepRangeLen triggers AbstractRange fallback
    cache = _get_derivative_cache_impl(x_range, bc_pair)
    @test cache isa CubicSplineCache
    @test cache.x isa FastInterpolations._CachedRange{Float64}
    @test collect(cache.x) ≈ collect(x_range) rtol = 8eps(Float64)

    # Second call with same range params → cache hit via _CachedRange bank
    cache2 = _get_derivative_cache_impl(range(-1.0, 2.0, 31), bc_pair)
    @test cache2 === cache  # same cached object

    # Different BC type → different cache
    bc_pair2 = BCPair(Deriv2(0.0), Deriv2(0.0))
    cache3 = _get_derivative_cache_impl(x_range, bc_pair2)
    @test cache3 !== cache
    @test cache3.x isa FastInterpolations._CachedRange{Float64}
end
