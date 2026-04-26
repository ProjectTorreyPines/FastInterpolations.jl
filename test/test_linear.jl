# ALLOC_THRESHOLD is defined in test/setup.jl

@testitem "Linear Interpolation - Core Functionality" setup = [AllocConstants] begin

    @testset "Uniform grid (AbstractRange) - Interior points" begin
        x = 0.0:0.01:1.0
        y = sin.(x)
        x_targets = [0.25, 0.5, 0.75]

        # Allocating version
        result_alloc = linear_interp(x, y, x_targets)
        @test result_alloc isa Vector{Float64}
        @test length(result_alloc) == 3
        @test all(isfinite, result_alloc)

        # In-place version
        result_inplace = similar(x_targets)
        linear_interp!(result_inplace, x, y, x_targets)
        @test result_inplace == result_alloc
    end

    @testset "Uniform grid (AbstractRange) - Extrapolation :extension" begin
        x = 0.0:0.1:1.0
        y = 2.0 .* collect(x) .+ 1.0  # Linear function y = 2x + 1
        x_targets = [-0.2, -0.1, 1.1, 1.2]

        result = linear_interp(x, y, x_targets; extrap = ExtendExtrap())

        # Verify linear extrapolation works correctly
        # For y = 2x + 1, extrapolated values should follow the same line
        @test result[1] ≈ 2.0 * (-0.2) + 1.0
        @test result[2] ≈ 2.0 * (-0.1) + 1.0
        @test result[3] ≈ 2.0 * 1.1 + 1.0
        @test result[4] ≈ 2.0 * 1.2 + 1.0
    end

    @testset "Uniform grid (AbstractRange) - Extrapolation :constant" begin
        x = 0.0:0.1:1.0
        y = sin.(x)
        x_targets = [-0.2, -0.1, 1.1, 1.2]

        result = linear_interp(x, y, x_targets; extrap = ClampExtrap())

        # For constant extrapolation, values outside bounds should match boundary values
        @test result[1] == y[1]
        @test result[2] == y[1]
        @test result[3] == y[end]
        @test result[4] == y[end]
    end

    @testset "Non-uniform grid (Vector) - Interior points" begin
        x = [0.0, 0.1, 0.3, 0.6, 1.0]
        y = sin.(x)
        x_targets = [0.05, 0.2, 0.45, 0.8]

        # Allocating version
        result_alloc = linear_interp(x, y, x_targets)
        @test result_alloc isa Vector{Float64}
        @test length(result_alloc) == 4
        @test all(isfinite, result_alloc)

        # In-place version
        result_inplace = similar(x_targets)
        linear_interp!(result_inplace, x, y, x_targets)
        @test result_inplace == result_alloc

        # Test scalar version
        result_scalar = linear_interp(x, y, 0.2)
        @test result_scalar isa Float64
    end

    @testset "Non-uniform grid (Vector) - Extrapolation :extension" begin
        x = [0.0, 0.5, 1.0]
        y = 2x .+ 1
        x_targets = [-0.25, 1.5]

        result = linear_interp(x, y, x_targets; extrap = ExtendExtrap())

        # Verify linear extrapolation
        @test result[1] ≈ 2.0 * (-0.25) + 1.0
        @test result[2] ≈ 2.0 * 1.5 + 1.0
    end

    @testset "Non-uniform grid (Vector) - Extrapolation :constant" begin
        x = [0.0, 0.5, 1.0]
        y = [1.0, 3.0, 5.0]
        x_targets = [-0.25, 1.5]

        result = linear_interp(x, y, x_targets; extrap = ClampExtrap())

        # Constant extrapolation
        @test result[1] == y[1]
        @test result[2] == y[end]
    end

    @testset "Extrapolation :none - DomainError" begin
        x = [0.0, 0.5, 1.0]
        y = [1.0, 3.0, 5.0]

        # Default extrapolation is NoExtrap(), should throw DomainError
        @test_throws DomainError linear_interp(x, y, -0.1)
        @test_throws DomainError linear_interp(x, y, 1.1)

        # Explicit :none also throws
        @test_throws DomainError linear_interp(x, y, -0.5; extrap = NoExtrap())
        @test_throws DomainError linear_interp(x, y, 1.5; extrap = NoExtrap())

        # Vector query - first out-of-domain point throws
        @test_throws DomainError linear_interp(x, y, [-0.1, 0.5])
        @test_throws DomainError linear_interp(x, y, [0.5, 1.1])

        # In-place version also throws
        output = zeros(1)
        @test_throws DomainError linear_interp!(output, x, y, [-0.1])
        @test_throws DomainError linear_interp!(output, x, y, [1.1])

        # Callable interpolant (default :none)
        itp = linear_interp(x, y)
        @test_throws DomainError itp(-0.1)
        @test_throws DomainError itp(1.1)

        # Interior points should work fine
        @test linear_interp(x, y, 0.25) ≈ 2.0
        @test linear_interp(x, y, 0.75) ≈ 4.0

        # Boundary points should work
        @test linear_interp(x, y, 0.0) ≈ 1.0
        @test linear_interp(x, y, 1.0) ≈ 5.0
    end


    @testset "Edge cases - Exact matches at grid points" begin
        x = 0.0:0.1:1.0
        y = collect(x) .^ 2
        x_targets = [0.0, 0.3, 0.5, 0.7, 1.0]

        # Use :extension to allow boundary evaluation (1.0 may need slight extension)
        result = linear_interp(x, y, x_targets; extrap = ExtendExtrap())

        # Exact matches should give exact values
        @test result[1] ≈ 0.0^2
        @test result[3] ≈ 0.5^2
        @test result[5] ≈ 1.0^2
    end

    @testset "Edge cases - Boundary points" begin
        x = 0.0:0.1:1.0
        y = sin.(x)
        x_targets = [0.0, 1.0]

        result = linear_interp(x, y, x_targets)

        # Boundaries should give exact values
        @test result[1] ≈ sin(0.0)
        @test result[2] ≈ sin(1.0)
    end

    @testset "Zero-allocation verification" begin
        x_range = 0.0:0.01:1.0
        x_vec = collect(x_range)
        y = sin.(x_range)
        xi = 0.55
        x_targets = [0.25, 0.5, 0.75]
        output = similar(x_targets)

        # Warmup all paths
        linear_interp(x_range, y, xi)
        linear_interp(x_vec, y, xi)
        linear_interp!(output, x_range, y, x_targets)
        linear_interp!(output, x_vec, y, x_targets)

        # Scalar query with Range - MUST be zero allocation
        allocs = @allocated linear_interp(x_range, y, xi)
        @test allocs <= ALLOC_THRESHOLD

        # Scalar query with Vector - MUST be zero allocation
        allocs = @allocated linear_interp(x_vec, y, xi)
        @test allocs <= ALLOC_THRESHOLD

        # In-place with Range - MUST be zero allocation
        allocs = @allocated linear_interp!(output, x_range, y, x_targets)
        @test allocs <= ALLOC_THRESHOLD

        # In-place with Vector - MUST be zero allocation
        allocs = @allocated linear_interp!(output, x_vec, y, x_targets)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-allocation - Callable" begin
        x = 0.0:0.01:1.0
        y = sin.(x)
        xi = 0.55

        itp = linear_interp(x, y)
        itp(xi)  # Warmup

        # Callable scalar call - MUST be zero allocation
        allocs = @allocated itp(xi)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Type stability" begin
        x = 0.0:0.1:1.0
        y = sin.(x)
        x_targets = [0.25, 0.5, 0.75]

        # Allocating version
        result = @inferred linear_interp(x, y, x_targets)
        @test result isa Vector{Float64}

        # In-place version
        output = similar(x_targets)
        result_inplace = @inferred linear_interp!(output, x, y, x_targets)
        @test result_inplace === output
    end

    @testset "Analytical solution - Linear function exact" begin
        # Linear interpolation should be EXACT for linear functions
        x = 0.0:0.1:1.0

        for (a, b) in [(2.0, 3.0), (-1.5, 0.0), (0.0, 5.0)]
            y = a .* collect(x) .+ b

            for xi in [0.05, 0.15, 0.33, 0.77, 0.99]
                expected = a * xi + b
                @test linear_interp(x, y, xi) ≈ expected
            end
        end
    end

    @testset "Knot passage - Interpolation passes through data points" begin
        x = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
        y = sin.(x)

        itp = linear_interp(x, y)

        for (xi, yi) in zip(x, y)
            @test itp(xi) ≈ yi
        end
    end

    @testset "Regression test - Reference values" begin
        # Captured reference values to detect unintended changes
        x = 0.0:0.1:1.0
        y = sin.(x)

        @test linear_interp(x, y, 0.15) ≈ 0.14925137372094466
        @test linear_interp(x, y, 0.33) ≈ 0.3236896473555328
        @test linear_interp(x, y, 0.67) ≈ 0.6203451230848943
        @test linear_interp(x, y, 0.89) ≈ 0.7767298277546874
    end
end

@testitem "Linear Interpolation - Callable (2-Argument Form)" begin

    @testset "Basic callable functionality" begin
        x = 0.0:0.1:1.0
        y = sin.(x)

        itp = linear_interp(x, y)
        @test itp isa LinearInterpolant

        # Test scalar call
        val = itp(0.5)
        @test val isa Float64

        # Compare with 3-argument form
        expected = linear_interp(x, y, 0.5)
        @test val == expected
    end

    @testset "Vector call (direct)" begin
        x = 0.0:0.1:1.0
        y = sin.(x)
        x_targets = [0.25, 0.5, 0.75]

        itp = linear_interp(x, y)

        # Test direct vector call
        result_vector = itp(x_targets)
        @test result_vector isa Vector{Float64}

        # Compare with 3-argument form
        expected = linear_interp(x, y, x_targets)
        @test result_vector == expected
    end

    @testset "Broadcast functionality" begin
        x = 0.0:0.1:1.0
        y = sin.(x)
        x_targets = [0.25, 0.5, 0.75]

        itp = linear_interp(x, y)

        # Test explicit broadcast
        result_broadcast = itp.(x_targets)
        @test result_broadcast isa Vector{Float64}

        # Compare with 3-argument form
        expected = linear_interp(x, y, x_targets)
        @test result_broadcast == expected
    end

    @testset "Fused broadcast" begin
        x = 0.0:0.1:1.0
        y = sin.(x)
        rho = [0.25, 0.5, 0.75]
        coef = 2.0
        other = ones(3)

        itp1 = linear_interp(x, y)
        itp2 = linear_interp(x, cos.(x))

        # Fused broadcast
        result = @. coef * itp1(rho) * itp2(rho) * other
        @test result isa Vector{Float64}

        # Verify correctness
        vals1 = linear_interp(x, y, rho)
        vals2 = linear_interp(x, cos.(x), rho)
        expected = @. coef * vals1 * vals2 * other
        @test result == expected
    end

    @testset "Reusability" begin
        x = 0.0:0.1:1.0
        y = sin.(x)

        # Use :extension to handle floating point boundary issues
        itp = linear_interp(x, y; extrap = ExtendExtrap())

        rho1 = [0.25, 0.5]
        rho2 = [0.75, 0.85]
        rho3 = 0.3

        result1 = itp.(rho1)
        result2 = itp.(rho2)
        result3 = itp(rho3)

        @test result1 == linear_interp(x, y, rho1; extrap = ExtendExtrap())
        @test result2 == linear_interp(x, y, rho2; extrap = ExtendExtrap())
        @test result3 == linear_interp(x, y, rho3; extrap = ExtendExtrap())
    end

    @testset "Extrapolation :extension" begin
        x = [0.0, 0.5, 1.0]
        y = 2.0 .* x .+ 1.0
        x_targets = [-0.25, 1.5]

        itp = linear_interp(x, y; extrap = ExtendExtrap())

        result = itp.(x_targets)
        @test result[1] ≈ 2.0 * (-0.25) + 1.0
        @test result[2] ≈ 2.0 * 1.5 + 1.0

        expected = linear_interp(x, y, x_targets; extrap = ExtendExtrap())
        @test result == expected
    end

    @testset "Extrapolation :constant" begin
        x = [0.0, 0.5, 1.0]
        y = [1.0, 3.0, 5.0]
        x_targets = [-0.25, 1.5]

        itp = linear_interp(x, y; extrap = ClampExtrap())

        result = itp.(x_targets)
        @test result[1] == y[1]
        @test result[2] == y[end]

        expected = linear_interp(x, y, x_targets; extrap = ClampExtrap())
        @test result == expected
    end

    @testset "Non-uniform grid" begin
        x = [0.0, 0.1, 0.3, 0.6, 1.0]
        y = sin.(x)
        x_targets = [0.05, 0.2, 0.45, 0.8]

        itp = linear_interp(x, y)
        result = itp.(x_targets)

        # Persistent uses cached `inv_h * α`; oneshot uses direct
        # `(q-L)/(R-L)`. Both within 1 ULP — tolerance allows the
        # documented design difference.
        expected_3arg = linear_interp(x, y, x_targets)
        @test result ≈ expected_3arg rtol = 4 * eps(Float64)
    end

    @testset "Type stability" begin
        # Float64
        x_f64 = 0.0:0.1:1.0
        y_f64 = sin.(x_f64)
        itp_f64 = @inferred linear_interp(x_f64, y_f64)
        @test itp_f64 isa LinearInterpolant{Float64}

        val_f64 = @inferred itp_f64(0.5)
        @test val_f64 isa Float64

        # Float32
        x_f32 = range(Float32(0.0), Float32(1.0), length = 11)
        y_f32 = sin.(x_f32)
        itp_f32 = @inferred linear_interp(x_f32, y_f32)
        @test itp_f32 isa LinearInterpolant{Float32}

        val_f32 = @inferred itp_f32(Float32(0.5))
        @test val_f32 isa Float32
    end

    @testset "Integer input auto-promotion" begin
        x_int = 0:10
        y_int = [i^2 for i in x_int]

        itp = linear_interp(x_int, y_int)
        @test itp isa LinearInterpolant{Float64}

        val = itp(5.5)
        @test val isa Float64
        @test val == 25.0 + 0.5 * (36.0 - 25.0)

        val_int = itp(5)
        @test val_int isa Float64
        @test val_int == 25.0
    end

    @testset "Memory efficiency" begin
        x = 0.0:0.1:1.0
        y = sin.(x)

        itp = linear_interp(x, y)

        # Callable should be small
        @test sizeof(itp) <= 80

        x_targets = [0.25, 0.5, 0.75]
        _ = itp.(x_targets)  # Warmup

        allocs = @allocated itp.(x_targets)
        @test allocs < 1000  # Should only allocate output array
    end
end

@testitem "Linear Interpolation - Type Auto-Promotion" begin

    @testset "Integer input → Float output" begin
        x_int = 0:10
        y_int = [i^2 for i in x_int]
        x_targets = [2.5, 5.5, 7.3]

        result = linear_interp(x_int, y_int, x_targets)
        @test result isa Vector{Float64}

        # Verify correctness
        @test result[1] ≈ 4.0 + 0.5 * (9.0 - 4.0)
        @test result[2] ≈ 25.0 + 0.5 * (36.0 - 25.0)
        @test result[3] ≈ 49.0 + 0.3 * (64.0 - 49.0)

        # Test scalar version
        result_scalar = linear_interp(x_int, y_int, 5.5)
        @test result_scalar isa Float64
        @test result_scalar ≈ 25.0 + 0.5 * (36.0 - 25.0)

        # Test in-place version
        output = Vector{Float64}(undef, length(x_targets))
        linear_interp!(output, x_int, y_int, x_targets)
        @test output == result
    end

    @testset "Float32 support" begin
        x_f32 = range(Float32(0.0), Float32(1.0), length = 11)
        y_f32 = sin.(x_f32)
        x_targets_f32 = Float32[0.25, 0.5, 0.75]

        result = linear_interp(x_f32, y_f32, x_targets_f32)
        @test result isa Vector{Float32}

        # Compare to Float64 reference
        x_f64 = range(0.0, 1.0, length = 11)
        y_f64 = sin.(x_f64)
        x_targets_f64 = [0.25, 0.5, 0.75]
        result_f64 = linear_interp(x_f64, y_f64, x_targets_f64)

        @test Float64.(result) ≈ result_f64 rtol = 1.0e-6

        # Test scalar version
        result_scalar = linear_interp(x_f32, y_f32, Float32(0.5))
        @test result_scalar isa Float32
    end

    @testset "Mixed Real types" begin
        x_int = 0:5
        y_int = [i^2 for i in x_int]
        x_targets_float = [1.5, 3.2]

        result = linear_interp(x_int, y_int, x_targets_float)
        @test result isa Vector{Float64}
        @test result[1] ≈ 1.0 + 0.5 * (4.0 - 1.0)
        @test result[2] ≈ 9.0 + 0.2 * (16.0 - 9.0)

        result_int_target = linear_interp(x_int, y_int, 3)
        @test result_int_target isa Float64
        @test result_int_target ≈ 9.0
    end

    @testset "Extrapolation with Integer inputs" begin
        x_int = 0:5
        y_int = [2 * i + 1 for i in x_int]

        x_targets = [-1.0, 6.0]
        result_ext = linear_interp(x_int, y_int, x_targets; extrap = ExtendExtrap())
        @test result_ext isa Vector{Float64}
        @test result_ext[1] ≈ 2.0 * (-1.0) + 1.0
        @test result_ext[2] ≈ 2.0 * 6.0 + 1.0

        result_const = linear_interp(x_int, y_int, x_targets; extrap = ClampExtrap())
        @test result_const[1] ≈ y_int[1]
        @test result_const[2] ≈ y_int[end]
    end
end

@testitem "Linear Interpolation - Range Preservation (O(1) Path)" begin
    # Test that LinearInterpolant preserves AbstractRange structure
    # This enables O(1) index lookup vs O(log n) binary search

    @testset "Range input → Range stored (O(1) path)" begin
        x_range = 0.0:0.1:1.0  # StepRangeLen{Float64}
        y = sin.(x_range)

        itp = linear_interp(x_range, y)

        # CRITICAL: x must remain AbstractRange for O(1) lookup
        @test itp.x isa AbstractRange
        @test itp.x isa FastInterpolations._CachedRange{Float64}

        # Verify correctness
        @test itp(0.5) ≈ sin(0.5) atol = 0.01
    end

    @testset "Vector input → Vector stored (O(log n) path)" begin
        x_vec = collect(0.0:0.1:1.0)  # Vector{Float64}
        y = sin.(x_vec)

        itp = linear_interp(x_vec, y)

        # Vector should remain Vector
        @test itp.x isa Vector{Float64}
        @test !(itp.x isa AbstractRange)

        # Verify correctness
        @test itp(0.5) ≈ sin(0.5) atol = 0.01
    end

    @testset "Integer Range → Float64 Range preserved" begin
        x_int = 0:10  # UnitRange{Int}
        y_int = [i^2 for i in x_int]

        itp = linear_interp(x_int, y_int)

        # Integer Range should be converted to Float64 Range (not Vector!)
        @test itp.x isa AbstractRange
        @test eltype(itp.x) == Float64

        # Verify correctness
        @test itp(5.5) ≈ 25.0 + 0.5 * (36.0 - 25.0)
    end

    @testset "Float32 Range preserved" begin
        x_f32 = range(Float32(0.0), Float32(1.0), 11)  # StepRangeLen{Float32}
        y_f32 = sin.(x_f32)

        itp = linear_interp(x_f32, y_f32)

        # Float32 Range should be preserved
        @test itp.x isa AbstractRange
        @test eltype(itp.x) == Float32

        # Verify correctness
        @test itp(Float32(0.5)) ≈ sin(Float32(0.5)) atol = 0.01f0
    end

    @testset "Mixed types: Range x + Vector y → Range preserved" begin
        x_range = 0.0:0.1:1.0
        y_vec = collect(sin.(x_range))  # Explicitly Vector

        itp = linear_interp(x_range, y_vec)

        # x should remain Range even if y is Vector
        @test itp.x isa AbstractRange

        # Verify correctness
        @test itp(0.5) ≈ sin(0.5) atol = 0.01
    end

    @testset "linear_interp! AbstractVector Real wrappers" begin
        # Test with Integer Vector
        x = collect(0:10)
        y = x .^ 2
        x_query = [2.5, 5.5, 7.5]
        output = zeros(3)

        linear_interp!(output, x, y, x_query)
        @test output[1] ≈ 2.5^2 atol = 1
        @test output[2] ≈ 5.5^2 atol = 1

        # Test with constant extrapolation
        linear_interp!(output, x, y, x_query; extrap = ClampExtrap())
        @test length(output) == 3
    end

    @testset "linear_interp scalar AbstractVector Real wrapper" begin
        x = collect(0:10)
        y = x .^ 2

        val = linear_interp(x, y, 5.5)
        @test val ≈ 5.5^2 atol = 1
    end

    @testset "LinearInterpolant in-place methods" begin
        x = range(0.0, 1.0, 51)
        y = Float64.(sin.(2π .* x))
        itp = linear_interp(x, y)

        x_query = [0.25, 0.5, 0.75]
        output = zeros(3)

        # Test in-place with matching types
        itp(output, x_query)
        @test length(output) == 3

        # Test in-place with type conversion
        x_query_f32 = Float32[0.25, 0.5, 0.75]
        output2 = zeros(3)
        itp(output2, x_query_f32)
        @test length(output2) == 3
    end

    @testset "LinearInterpolant vector with type conversion" begin
        x = range(0.0, 1.0, 51)
        y = Float64.(sin.(2π .* x))
        itp = linear_interp(x, y)

        # Test with Float32 vector
        x_query_f32 = Float32[0.25, 0.5, 0.75]
        result = itp(x_query_f32)
        @test length(result) == 3
    end
end
