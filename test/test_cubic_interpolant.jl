# ALLOC_THRESHOLD is defined in runtests.jl

@testset "CubicInterpolant" begin
    # Setup
    x = collect(range(0.0, 1.0, 11))
    y = sin.(2π .* x)
    x_query = [0.15, 0.35, 0.55, 0.75, 0.95]

    # Reference: 3-argument form
    ref = cubic_interp(x, y, x_query)

    # Create callable interpolant
    itp = cubic_interp(x, y)

    @testset "Type and structure" begin
        @test itp isa CubicInterpolant
        @test itp.cache isa CubicSplineCache
        @test itp.y == y  # Values equal (copy is made internally)
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
        itp3 = cubic_interp(x, y; autocache = false)
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
        itp_fresh = cubic_interp(x, y; autocache = false)

        # Query at grid points
        for xi in x
            val = itp_fresh(xi)
            ref_val = cubic_interp(x, y, xi; autocache = false)
            @test val == ref_val
        end

        # Extrapolation (requires extrap=ExtendExtrap())
        itp_extrap = cubic_interp(x, y; autocache = false, extrap = ExtendExtrap())
        val_left = itp_extrap(-0.1)
        val_right = itp_extrap(1.1)
        ref_left = cubic_interp(x, y, -0.1; autocache = false, extrap = ExtendExtrap())
        ref_right = cubic_interp(x, y, 1.1; autocache = false, extrap = ExtendExtrap())
        @test val_left == ref_left
        @test val_right == ref_right
        @test isfinite(val_left)
        @test isfinite(val_right)
    end

    @testset "Zero-allocation - Scalar call" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        xi = 0.55

        # Create callable (pre-computes z coefficients)
        itp = cubic_interp(x, y; autocache = false)

        # Warmup
        _ = itp(xi)

        # Scalar call - zero allocation on 1.12+ (uses pre-computed z)
        allocs = @allocated itp(xi)
        @test allocs <= ALLOC_THRESHOLD

        # Different query point - still zero allocation
        allocs = @allocated itp(0.75)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Allocation efficiency - Vector call" begin
        x_test = collect(range(0.0, 1.0, 101))
        y_test = sin.(2π .* x_test)
        x_query_test = collect(range(0.1, 0.9, 20))

        # Create callable (pre-computes z coefficients)
        itp_test = cubic_interp(x_test, y_test; autocache = false)

        # Warmup
        _ = itp_test(x_query_test)

        # Vector call allocates output array only
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
    end
end

@testset "Cubic Interpolation - Range/Vector Input Handling" begin
    # CubicSplineCache now preserves Range structure for O(1) index lookup
    # This testset documents behavior for both input types

    @testset "Range input → Range preserved in direct cache" begin
        x_range = range(0.0, 1.0, 11)  # StepRangeLen{Float64}
        y = sin.(2π .* collect(x_range))

        # Direct CubicSplineCache normalizes Range → _CachedRange for O(1) lookup
        cache = CubicSplineCache(x_range)
        @test cache.x isa FastInterpolations._CachedRange
        @test collect(cache.x) ≈ collect(x_range) rtol = 8eps(Float64)

        # Verify correctness
        result = cubic_interp(cache, y, [0.5])
        @test result[1] ≈ sin(2π * 0.5) atol = 0.01
    end

    @testset "Range input via autocache → Range preserved for O(1) lookup" begin
        clear_cubic_cache!()
        x_range = range(0.0, 1.0, 11)
        y = sin.(2π .* collect(x_range))

        # Range normalized to _CachedRange for O(1) index lookup (cached inv_h)
        itp = cubic_interp(x_range, y; autocache = true)
        @test itp.cache.x isa FastInterpolations._CachedRange

        # Verify correctness
        @test itp(0.5) ≈ sin(2π * 0.5) atol = 0.01

        # Verify cache hit works with equal _CachedRange (isbits → isequal comparison)
        clear_cubic_cache!()
        result1 = cubic_interp(x_range, y, 0.5; autocache = true)  # First call: miss
        result2 = cubic_interp(range(0.0, 1.0, 11), y, 0.5; autocache = true)  # Same params → isequal hit
        @test result1 ≈ result2  # Same results from cache hit
    end

    @testset "Vector input → stored as Vector" begin
        x_vec = collect(range(0.0, 1.0, 11))  # Vector{Float64}
        y = sin.(2π .* x_vec)

        itp = cubic_interp(x_vec, y; autocache = false)

        # Vector remains Vector
        @test itp.cache.x isa Vector{Float64}

        # Verify correctness
        @test itp(0.5) ≈ sin(2π * 0.5) atol = 0.01
    end

    @testset "Integer Range → Float64 Range conversion" begin
        x_int = 0:10  # UnitRange{Int}
        y_int = [sin(2π * i / 10) for i in x_int]

        # Direct cache preserves Range structure
        cache = CubicSplineCache(range(0.0, 10.0, 11))
        @test cache.x isa AbstractRange

        # Via cubic_interp with autocache=false
        itp = cubic_interp(x_int, y_int; autocache = false)
        @test itp.cache.x isa AbstractRange  # Range preserved via _to_float
        @test eltype(itp.cache.x) == Float64

        # Verify correctness
        @test itp(5.0) ≈ sin(2π * 0.5) atol = 0.01
    end

    @testset "Float32 Range → Float32 Range conversion" begin
        x_f32 = range(Float32(0.0), Float32(1.0), 11)
        y_f32 = sin.(Float32(2π) .* collect(x_f32))

        # Direct cache preserves Range
        cache = CubicSplineCache(x_f32)
        @test cache.x isa AbstractRange
        @test eltype(cache.x) == Float32

        # Via cubic_interp with autocache=false
        itp = cubic_interp(x_f32, y_f32; autocache = false)
        @test itp.cache.x isa AbstractRange  # Range preserved
        @test eltype(itp.cache.x) == Float32

        # Verify correctness
        @test itp(Float32(0.5)) ≈ sin(Float32(2π) * Float32(0.5)) atol = 0.01f0
    end

    @testset "Range vs Vector produce nearly identical results" begin
        # Same mathematical grid, different Julia types
        x_range = range(0.0, 1.0, 51)
        x_vec = collect(x_range)
        y = sin.(2π .* x_vec)

        itp_range = cubic_interp(x_range, y; autocache = false)
        itp_vec = cubic_interp(x_vec, y; autocache = false)

        # Query points
        xi_test = [0.1, 0.25, 0.5, 0.75, 0.9]

        # Results should be nearly identical (tiny FP differences due to O(1) vs O(log n) lookup)
        for xi in xi_test
            @test itp_range(xi) ≈ itp_vec(xi) rtol = 1.0e-14
        end

        # Vectorized evaluation
        @test all(itp_range.(xi_test) .≈ itp_vec.(xi_test))
    end

    @testset "CubicSplineCache type parametrization" begin
        x_range = range(0.0, 1.0, 21)
        x_vec = collect(x_range)

        # Range input → parametric type with Range
        cache_range = CubicSplineCache(x_range)
        @test cache_range.x isa AbstractRange
        @test typeof(cache_range).parameters[2] <: AbstractRange

        # Vector input → parametric type with Vector
        cache_vec = CubicSplineCache(x_vec)
        @test cache_vec.x isa Vector{Float64}
        @test typeof(cache_vec).parameters[2] == Vector{Float64}

        # Both produce correct results
        y = sin.(2π .* x_vec)
        @test cubic_interp(cache_range, y, [0.5])[1] ≈ cubic_interp(cache_vec, y, [0.5])[1]
    end

    @testset "CubicInterpolant Real scalar wrapper" begin
        x = range(0.0, 1.0, 51)
        y = Float64.(sin.(2π .* x))
        itp = cubic_interp(x, y)

        # Test with Int scalar (converts to Float64)
        val_int = itp(1)  # Int input
        @test val_int ≈ sin(2π * 1.0) atol = 1.0e-6

        # Test with Float32 scalar
        val_f32 = itp(Float32(0.5))
        @test val_f32 ≈ sin(2π * 0.5) atol = 1.0e-6
    end

    @testset "CubicInterpolant vector with type conversion" begin
        x = range(0.0, 1.0, 51)
        y = Float64.(sin.(2π .* x))
        itp = cubic_interp(x, y)

        # Test with Float32 vector (type conversion)
        x_query_f32 = Float32[0.25, 0.5, 0.75]
        result = itp(x_query_f32)
        @test result ≈ sin.(2π .* x_query_f32) atol = 1.0e-5

        # Test with Int vector
        x_query_int = [0, 1]  # Int inputs
        result_int = itp(Float64.(x_query_int))  # Manual conversion needed for Int
        @test length(result_int) == 2
    end

    @testset "CubicInterpolant in-place methods" begin
        x = range(0.0, 1.0, 51)
        y = Float64.(sin.(2π .* x))
        itp = cubic_interp(x, y)

        x_query = [0.25, 0.5, 0.75]
        output = zeros(3)

        # Test in-place with matching types
        itp(output, x_query)
        @test output ≈ sin.(2π .* x_query) atol = 1.0e-6

        # Test in-place with type conversion (Float32 input)
        x_query_f32 = Float32[0.25, 0.5, 0.75]
        output2 = zeros(3)
        itp(output2, x_query_f32)
        @test output2 ≈ sin.(2π .* x_query_f32) atol = 1.0e-5
    end
end
