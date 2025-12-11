@testset "CubicInterpCallable" begin
    # Setup
    x = collect(range(0.0, 1.0, 11))
    y = sin.(2π .* x)
    x_query = [0.15, 0.35, 0.55, 0.75, 0.95]

    # Reference: 3-argument form
    ref = cubic_interp(x, y, x_query)

    # Create callable interpolator
    itp = cubic_interp(x, y)

    @testset "Type and structure" begin
        @test itp isa CubicInterpCallable
        @test itp.cache isa CubicSplineCache
        @test itp.y === y
        @test length(itp.cache.x) == length(x)
    end

    @testset "Scalar call" begin
        for xi in x_query
            val = itp(xi)
            ref_val = cubic_interp(x, y, xi)
            @test val == ref_val  # Bit-exact
        end
    end

    @testset "Vector call" begin
        result = itp(x_query)
        @test result == ref  # Bit-exact
    end

    @testset "Broadcasting" begin
        result = itp.(x_query)
        @test result == ref  # Bit-exact
    end

    @testset "Broadcast fusion" begin
        coef = 2.5
        # Fused broadcast
        result = @. coef * itp(x_query)
        # Reference (direct calculation)
        ref_fused = coef .* cubic_interp(x, y, x_query)
        @test result == ref_fused  # Bit-exact

        # Complex fusion
        other = exp.(x_query)
        result = @. coef * itp(x_query) / other
        ref_complex = (coef .* cubic_interp(x, y, x_query)) ./ other
        @test result == ref_complex  # Bit-exact
    end

    @testset "Reuse interpolator" begin
        # Create once, use multiple times
        itp_reuse = cubic_interp(x, y)

        x_query1 = [0.1, 0.2, 0.3]
        x_query2 = [0.6, 0.7, 0.8]
        x_query3 = [0.25, 0.55, 0.85]

        result1 = itp_reuse.(x_query1)
        result2 = itp_reuse.(x_query2)
        result3 = itp_reuse.(x_query3)

        ref1 = cubic_interp(x, y, x_query1)
        ref2 = cubic_interp(x, y, x_query2)
        ref3 = cubic_interp(x, y, x_query3)

        @test result1 == ref1
        @test result2 == ref2
        @test result3 == ref3
    end

    @testset "Auto-cache behavior" begin
        # With autocache=true (default)
        itp1 = cubic_interp(x, y)
        itp2 = cubic_interp(x, y)  # Should reuse cache from auto-cache

        # Both should produce identical results
        result1 = itp1(x_query)
        result2 = itp2(x_query)
        @test result1 == result2  # Bit-exact

        # With autocache=false
        itp3 = cubic_interp(x, y; autocache=false)
        result3 = itp3(x_query)
        @test result3 == ref  # Bit-exact
    end

    @testset "Different y vectors, same x" begin
        # Simulate the gm1-gm9 use case
        y_fields = [sin.(i .* 2π .* x) for i in 1:5]

        # Create interpolators
        itps = [cubic_interp(x, y_i) for y_i in y_fields]

        # Evaluate all
        x_query_common = [0.2, 0.5, 0.8]
        results = [itp_i.(x_query_common) for itp_i in itps]

        # Reference
        refs = [cubic_interp(x, y_i, x_query_common) for y_i in y_fields]

        for (result, ref_i) in zip(results, refs)
            @test result == ref_i  # Bit-exact
        end
    end

    @testset "Type promotion" begin
        # Integer inputs should work
        x_int = 0:10
        y_int = sin.(2π .* x_int / 10)

        itp_int = cubic_interp(x_int, y_int)
        result = itp_int(5.5)

        @test result isa Float64
        @test isfinite(result)
    end

    @testset "Edge cases" begin
        # Create fresh interpolator to avoid cache interference
        itp_fresh = cubic_interp(x, y; autocache=false)

        # Query at grid points
        for xi in x
            val = itp_fresh(xi)
            ref_val = cubic_interp(x, y, xi; autocache=false)
            @test val == ref_val
        end

        # Extrapolation
        val_left = itp_fresh(-0.1)
        val_right = itp_fresh(1.1)
        ref_left = cubic_interp(x, y, -0.1; autocache=false)
        ref_right = cubic_interp(x, y, 1.1; autocache=false)
        @test val_left == ref_left
        @test val_right == ref_right
        @test isfinite(val_left)
        @test isfinite(val_right)
    end

    @testset "Allocation efficiency" begin
        x_test = collect(range(0.0, 1.0, 101))
        y_test = sin.(2π .* x_test)
        x_query_test = collect(range(0.1, 0.9, 20))

        # IMAS callable: construct once, evaluate multiple times
        itp_test = cubic_interp(x_test, y_test; autocache=false)

        # Warmup
        _ = itp_test(x_query_test)

        # Measure allocations for evaluation
        allocs = @allocated itp_test(x_query_test)

        # Should be relatively small (just output array allocation)
        @test allocs < 10000
    end

    @testset "Cache reuse benefit for multiple y vectors" begin
        x_grid = collect(range(0.0, 1.0, 101))
        y_fields = [sin.(k .* 2π .* x_grid) for k in 1:5]
        x_query_grid = [0.2, 0.4, 0.6, 0.8]

        # Create callable interpolators (auto-cache reuses LU factorization)
        function bench_callable(x, y_fields, x_query)
            results = []
            for y_i in y_fields
                itp_i = cubic_interp(x, y_i)  # Auto-cache reuses LU!
                push!(results, itp_i(x_query))
            end
            return results
        end

        # Warmup
        clear_cubic_cache!()
        _ = bench_callable(x_grid, y_fields, x_query_grid)

        # Verify correctness by checking all results are finite
        clear_cubic_cache!()
        results = bench_callable(x_grid, y_fields, x_query_grid)

        for result in results
            @test all(isfinite, result)
        end

        # Verify cache was used (4 hits after 1 miss)
        stats = cubic_cache_stats()
        @test stats.misses == 1
        @test stats.hits == 4
    end
end
