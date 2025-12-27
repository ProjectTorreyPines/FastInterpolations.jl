@testset "Cubic Spline - Core Functionality" begin

    @testset "Basic correctness" begin
        for n in [10, 50, 101]
            x = collect(range(0.0, 1.0, n))
            y = @. sin(2π * x) + 0.5 * cos(4π * x)
            x_query = [0.15, 0.37, 0.62, 0.89]

            cache = CubicSplineCache(x)
            result = cubic_interp(cache, y, x_query)

            @test result isa Vector{Float64}
            @test length(result) == length(x_query)
            @test all(isfinite, result)
        end
    end

    @testset "Smooth interpolation" begin
        x = collect(range(0.0, 2π, 31))
        y = @. sin(x) + 0.3 * cos(3x)
        x_query = collect(range(0.1, 2π - 0.1, 50))

        cache = CubicSplineCache(x)
        result = cubic_interp(cache, y, x_query)

        @test all(isfinite, result)

        # Results should be in reasonable range
        y_min, y_max = extrema(y)
        @test all(y_min - 0.5 .<= result .<= y_max + 0.5)
    end

    @testset "LU Reuse: Varying y values" begin
        x = collect(range(0.0, 10.0, 51))
        cache = CubicSplineCache(x)

        y_functions = [
            x -> sin.(x),
            x -> cos.(x),
            x -> exp.(-0.1 .* x),
            x -> x .^ 2,
            x -> 1.0 ./ (1.0 .+ x),
        ]

        x_query = collect(range(0.5, 9.5, 20))

        for y_func in y_functions
            y = y_func(x)
            result = cubic_interp(cache, y, x_query)
            @test all(isfinite, result)
        end
    end

    @testset "LU Reuse: Varying x_query values" begin
        x = collect(range(0.0, 5.0, 31))
        y = @. exp(-x) * sin(2 * x)
        cache = CubicSplineCache(x)

        query_sets = [
            [0.5, 1.5, 2.5, 3.5, 4.5],
            [0.1, 0.9, 1.7, 2.3, 3.1, 4.2],
            collect(range(0.2, 4.8, 15)),
            [2.5],  # Single point
        ]

        for x_query in query_sets
            result = cubic_interp(cache, y, x_query)
            @test all(isfinite, result)
            @test length(result) == length(x_query)
        end
    end

    @testset "Edge Cases" begin
        # Minimal grid (with extrapolation outside domain)
        x_small = [0.0, 0.5, 1.0]
        y_small = [1.0, 2.0, 1.5]
        x_query_small = [-0.5, 0.25, 0.75, 1.5]

        cache_small = CubicSplineCache(x_small)
        result = cubic_interp(cache_small, y_small, x_query_small; extrap=:extension)
        @test all(isfinite, result)

        # Query at grid points (should return close to exact values)
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        cache = CubicSplineCache(x)

        result = cubic_interp(cache, y, x)
        @test result ≈ y

        # Query at boundaries
        x_boundary = [x[1], x[end]]
        result_boundary = cubic_interp(cache, y, x_boundary)
        @test result_boundary ≈ [y[1], y[end]]
    end

    @testset "Extrapolation :none - DomainError" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)

        # Default extrapolation is :none, should throw DomainError
        @test_throws DomainError cubic_interp(x, y, -0.1)
        @test_throws DomainError cubic_interp(x, y, 1.1)

        # Explicit :none also throws
        @test_throws DomainError cubic_interp(x, y, -0.5; extrap=:none)
        @test_throws DomainError cubic_interp(x, y, 1.5; extrap=:none)

        # Vector query - first out-of-domain point throws
        @test_throws DomainError cubic_interp(x, y, [-0.1, 0.5])
        @test_throws DomainError cubic_interp(x, y, [0.5, 1.1])

        # With cache - also throws
        cache = CubicSplineCache(x)
        @test_throws DomainError cubic_interp(cache, y, -0.1)
        @test_throws DomainError cubic_interp(cache, y, 1.1)
        @test_throws DomainError cubic_interp(cache, y, [-0.1])
        @test_throws DomainError cubic_interp(cache, y, [1.1])

        # In-place version also throws
        output = zeros(1)
        @test_throws DomainError cubic_interp!(output, x, y, [-0.1])
        @test_throws DomainError cubic_interp!(output, x, y, [1.1])
        @test_throws DomainError cubic_interp!(output, cache, y, [-0.1])
        @test_throws DomainError cubic_interp!(output, cache, y, [1.1])

        # Callable interpolant (default :none)
        itp = cubic_interp(x, y)
        @test_throws DomainError itp(-0.1)
        @test_throws DomainError itp(1.1)

        # Interior points should work fine
        @test isfinite(cubic_interp(x, y, 0.25))
        @test isfinite(cubic_interp(x, y, 0.75))

        # Boundary points should work
        @test cubic_interp(x, y, 0.0) ≈ y[1]
        @test cubic_interp(x, y, 1.0) ≈ y[end]
    end

    @testset "Extrapolation :constant" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)

        # Left boundary - returns y[1]
        result_left = cubic_interp(x, y, -0.5; extrap=:constant)
        @test result_left ≈ y[1]

        # Right boundary - returns y[end]
        result_right = cubic_interp(x, y, 1.5; extrap=:constant)
        @test result_right ≈ y[end]

        # Vector query
        result = cubic_interp(x, y, [-0.5, 0.5, 1.5]; extrap=:constant)
        @test result[1] ≈ y[1]
        @test result[3] ≈ y[end]

        # With cache
        cache = CubicSplineCache(x)
        @test cubic_interp(cache, y, -0.5; extrap=:constant) ≈ y[1]
        @test cubic_interp(cache, y, 1.5; extrap=:constant) ≈ y[end]

        # Callable interpolant with :constant
        itp = cubic_interp(x, y; extrap=:constant)
        @test itp(-0.5) ≈ y[1]
        @test itp(1.5) ≈ y[end]
    end

    @testset "Scalar Query Points" begin
        x = collect(range(0.0, 2.0, 21))
        y = @. exp(-x) * cos(3 * x)
        cache = CubicSplineCache(x)

        x_scalar = 0.73
        result_scalar = cubic_interp(cache, y, x_scalar)

        result_vector = cubic_interp(cache, y, [x_scalar])
        @test result_scalar == result_vector[1]
    end

    @testset "In-place vs Allocating" begin
        x = collect(range(0.0, 1.0, 31))
        y = @. 1.0 - 0.5 * x^2
        x_query = [0.2, 0.4, 0.6, 0.8]
        cache = CubicSplineCache(x)

        result_alloc = cubic_interp(cache, y, x_query)

        output = Vector{Float64}(undef, length(x_query))
        cubic_interp!(output, cache, y, x_query)

        @test output == result_alloc
    end

    @testset "Type Stability" begin
        # Float32
        x32 = Float32.(range(0.0, 1.0, 21))
        y32 = Float32.(sin.(2π .* x32))
        x_query32 = Float32.([0.25, 0.75])

        cache32 = CubicSplineCache(x32)
        result32 = cubic_interp(cache32, y32, x_query32)

        @test eltype(result32) === Float32
        @test !any(isnan, result32)

        # Float64
        x64 = Float64.(range(0.0, 1.0, 21))
        y64 = Float64.(sin.(2π .* x64))
        x_query64 = Float64.([0.25, 0.75])

        cache64 = CubicSplineCache(x64)
        result64 = cubic_interp(cache64, y64, x_query64)

        @test eltype(result64) === Float64
    end

    @testset "Allocation reduction with cache reuse" begin
        x = collect(range(0.0, 1.0, 51))
        cache = CubicSplineCache(x)

        y_vectors = [sin.(i .* x) for i in 1:9]
        x_query = collect(range(0.1, 0.9, 20))

        # Measure allocation with cache reuse
        alloc_with_cache = @allocated begin
            for y in y_vectors
                result = cubic_interp(cache, y, x_query)
            end
        end

        # Measure allocation without cache (creating new each time)
        alloc_without_cache = @allocated begin
            for y in y_vectors
                cache_new = CubicSplineCache(x)
                result = cubic_interp(cache_new, y, x_query)
            end
        end

        # Cache reuse should significantly reduce allocations
        @test alloc_with_cache < alloc_without_cache
    end

    @testset "Zero-allocation - In-place with cache" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)
        cache = CubicSplineCache(x)

        # Warmup
        cubic_interp!(output, cache, y, x_query)

        # In-place with explicit cache - MUST be zero allocation
        allocs = @allocated cubic_interp!(output, cache, y, x_query)
        @test allocs == 0
    end

    @testset "One-shot Convenience Function" begin
        x = collect(range(0.0, 1.0, 21))
        y = @. exp(-x) * sin(4π * x)
        x_query = [0.23, 0.67]

        # One-shot (constructs cache internally)
        result = cubic_interp(x, y, x_query)

        # Compare with explicit cache construction
        cache = CubicSplineCache(x)
        result_cache = cubic_interp(cache, y, x_query)

        @test result == result_cache
    end

    @testset "Knot passage - Interpolation passes through data points" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        cache = CubicSplineCache(x)

        for (xi, yi) in zip(x, y)
            @test cubic_interp(cache, y, xi) ≈ yi
        end
    end

    @testset "Regression test - Reference values" begin
        # Captured reference values to detect unintended changes
        # These values are from the current implementation (Natural boundary condition)

        # sin(2π*x) on [0,1] with 11 points
        x = collect(range(0.0, 1.0, 11))
        y_sin = sin.(2π .* x)

        @test cubic_interp(x, y_sin, 0.15) ≈ 0.8086551555800082
        @test cubic_interp(x, y_sin, 0.33) ≈ 0.8760692639427288
        @test cubic_interp(x, y_sin, 0.67) ≈ -0.876069263942729
        @test cubic_interp(x, y_sin, 0.89) ≈ -0.6373475686689355

        # exp(-x) on [0,1] with 11 points
        y_exp = exp.(-x)

        @test cubic_interp(x, y_exp, 0.25) ≈ 0.7788333860015754
        @test cubic_interp(x, y_exp, 0.5) ≈ 0.6065306597126333
        @test cubic_interp(x, y_exp, 0.75) ≈ 0.47237845915507926
    end
end

@testset "Cubic Spline - Type Auto-Promotion" begin

    @testset "Integer input → Float output (allocating)" begin
        x_int = 0:10
        y_int = [sin(2π * i / 10) for i in x_int]
        x_query_float = [2.5, 5.5, 7.3]

        x_float = collect(Float64.(x_int))
        y_float = Float64.(y_int)
        x_query_ref = Float64.(x_query_float)

        result = cubic_interp(x_int, y_int, x_query_float)
        @test result isa Vector{Float64}
        @test all(isfinite, result)

        result_ref = cubic_interp(x_float, y_float, x_query_ref)
        @test result == result_ref
    end

    @testset "Integer input → Float output (scalar)" begin
        x_int = 0:10
        y_int = [sin(2π * i / 10) for i in x_int]

        result_scalar = cubic_interp(x_int, y_int, 5.5)
        @test result_scalar isa Float64
        @test isfinite(result_scalar)

        x_float = collect(Float64.(x_int))
        y_float = Float64.(y_int)

        result_ref = cubic_interp(x_float, y_float, 5.5)
        @test result_scalar == result_ref

        # Integer query point
        result_int_query = cubic_interp(x_int, y_int, 5)
        @test result_int_query isa Float64
        @test isfinite(result_int_query)
    end

    @testset "Float32 support" begin
        x_f32 = range(Float32(0.0), Float32(1.0), length=11)
        y_f32 = sin.(Float32(2π) .* x_f32)
        x_query_f32 = Float32[0.25, 0.5, 0.75]

        result = cubic_interp(x_f32, y_f32, x_query_f32)
        @test result isa Vector{Float32}
        @test all(isfinite, result)

        # Compare with Float64 reference (relaxed tolerance)
        x_f64 = range(0.0, 1.0, length=11)
        y_f64 = sin.(2π .* x_f64)
        x_query_f64 = [0.25, 0.5, 0.75]
        result_f64 = cubic_interp(x_f64, y_f64, x_query_f64)

        @test Float64.(result) ≈ result_f64 rtol=1e-6
    end

    @testset "Mixed Real types (Int x/y, Float query)" begin
        x_int = 0:10
        y_int = [sin(2π * i / 10) for i in x_int]
        x_query_float = [2.5, 5.5, 7.3]
        x_query_int = [3, 7]

        result = cubic_interp(x_int, y_int, x_query_float)
        @test result isa Vector{Float64}

        result_int = cubic_interp(x_int, y_int, x_query_int)
        @test result_int isa Vector{Float64}
        @test all(isfinite, result_int)
    end

    @testset "Extrapolation with Integer inputs" begin
        x_int = 0:10
        y_int = [sin(2π * i / 10) for i in x_int]
        x_extrap = [-1.0, 11.0]

        result = cubic_interp(x_int, y_int, x_extrap; extrap=:extension)
        @test result isa Vector{Float64}
        @test all(isfinite, result)

        x_float = collect(Float64.(x_int))
        y_float = Float64.(y_int)

        result_ref = cubic_interp(x_float, y_float, x_extrap; extrap=:extension)
        @test result == result_ref
    end

    @testset "cubic_interp! with autocache (x, y, x_query)" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        clear_cubic_cache!()

        # Test autocache=true path (default)
        cubic_interp!(output, x, y, x_query)
        @test output ≈ sin.(2π .* x_query) atol=1e-6

        # Test autocache=false path
        cubic_interp!(output, x, y, x_query; autocache=false)
        @test output ≈ sin.(2π .* x_query) atol=1e-6

        # Test with Range
        x_range = range(0.0, 1.0, 51)
        y_range = sin.(2π .* x_range)
        cubic_interp!(output, x_range, y_range, x_query)
        @test output ≈ sin.(2π .* x_query) atol=1e-6
    end

    @testset "cubic_interp! Real type wrappers" begin
        # Integer inputs
        x = 0:10
        y = collect(x).^2
        x_query = [2.5, 5.5, 7.5]
        output = zeros(3)

        # In-place vector query with Real types
        cubic_interp!(output, x, y, x_query)
        @test length(output) == 3

        # Scalar query with Real types
        output_scalar = zeros(1)
        cubic_interp!(output_scalar, x, y, 5.5)
        @test length(output_scalar) == 1
    end

    @testset "_to_float Vector path" begin
        # This tests _to_float for Vector conversion
        x_int = collect(0:10)
        y_int = x_int.^2

        # This should use _to_float for Vector conversion
        itp = cubic_interp(x_int, y_int)
        @test itp(5.0) ≈ 25.0 atol=1

        itp_lin = linear_interp(x_int, y_int)
        @test itp_lin(5.0) ≈ 25.0 atol=1
    end
end

@testset "Cubic Spline - Uncovered Paths" begin

    @testset "cubic_interp! with cache and scalar query" begin
        # Lines 154-164: cubic_interp!(output, cache, y, x_query::T)
        x = collect(range(0.0, 1.0, 21))
        y = sin.(2π .* x)
        cache = CubicSplineCache(x)
        output = zeros(1)

        cubic_interp!(output, cache, y, 0.5)
        @test output[1] ≈ sin(π) atol=1e-6

        # Test with different extrap modes
        cubic_interp!(output, cache, y, 0.25; extrap=:none)
        @test isfinite(output[1])

        cubic_interp!(output, cache, y, -0.1; extrap=:constant)
        @test output[1] ≈ y[1]

        cubic_interp!(output, cache, y, 1.1; extrap=:extension)
        @test isfinite(output[1])
    end

    @testset "cubic_interp! with x,y,scalar and bc=PeriodicBC()" begin
        # Lines 179-182: bc=PeriodicBC() branch in scalar cubic_interp!
        x = collect(range(0.0, 2π, 21))
        y = sin.(x)
        y[end] = y[1]  # Ensure periodic
        output = zeros(1)

        cubic_interp!(output, x, y, π; bc=PeriodicBC())
        @test output[1] ≈ 0.0 atol=0.1

        # Test with autocache=false
        cubic_interp!(output, x, y, π; bc=PeriodicBC(), autocache=false)
        @test output[1] ≈ 0.0 atol=0.1
    end

    @testset "cubic_interp! with x,y,scalar and autocache=false" begin
        # Line 186: autocache=false branch
        x = collect(range(0.0, 1.0, 21))
        y = sin.(2π .* x)
        output = zeros(1)

        cubic_interp!(output, x, y, 0.5; autocache=false)
        @test output[1] ≈ sin(π) atol=1e-6
    end

    @testset "cubic_interp with vector query and bc=PeriodicBC()" begin
        # Lines 250-254: bc=PeriodicBC() in allocating vector API
        x = collect(range(0.0, 2π, 31))
        y = sin.(x)
        y[end] = y[1]  # Ensure periodic
        x_query = [π/2, π, 3π/2]

        result = cubic_interp(x, y, x_query; bc=PeriodicBC())
        @test result[1] ≈ 1.0 atol=0.1  # sin(π/2) ≈ 1
        @test result[2] ≈ 0.0 atol=0.1  # sin(π) ≈ 0
        @test result[3] ≈ -1.0 atol=0.1  # sin(3π/2) ≈ -1

        # Test with autocache=false
        result2 = cubic_interp(x, y, x_query; bc=PeriodicBC(), autocache=false)
        @test result2 ≈ result
    end

    @testset "Real wrapper 2-arg form with bc=PeriodicBC()" begin
        # Lines 424-427: bc=PeriodicBC() in Real wrapper 2-arg form
        x_int = 0:20
        y_int = [sin(2π * i / 20) for i in x_int]
        # y_int[end] ≈ y_int[1] (both ≈ 0)

        itp = cubic_interp(x_int, y_int; bc=PeriodicBC())
        @test itp isa CubicInterpolant
        @test itp(5.0) ≈ sin(π/2) atol=0.1  # sin(2π*5/20) = sin(π/2)

        # Also test autocache=false path
        itp2 = cubic_interp(x_int, y_int; bc=PeriodicBC(), autocache=false)
        @test itp2(5.0) ≈ itp(5.0) atol=1e-10
    end
end
