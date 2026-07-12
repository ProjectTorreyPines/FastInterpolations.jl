# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  QUADRATIC SERIES INTERPOLANT TESTS                        ║
# ║         Tests for QuadraticSeriesInterpolant with direct matrix storage    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Phase E.3: Tests verify QuadraticSeriesInterpolant works correctly using the
# shared series infrastructure from Phases A-D.
#

@testitem "QuadraticSeriesInterpolant" setup = [AllocConstants] begin
    FI = FastInterpolations


    # ========================================
    # Constructor Tests
    # ========================================

    @testset "construction" begin
        x = collect(0.0:0.1:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        ys = [y1, y2]

        @testset "from vector of vectors" begin
            sitp = quadratic_interp(x, Series(ys))
            @test sitp isa FI.QuadraticSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 2
        end

        @testset "from matrix" begin
            Y = hcat(y1, y2)
            sitp = quadratic_interp(x, Series(Y))
            @test sitp isa FI.QuadraticSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 2
        end

        @testset "validates input dimensions" begin
            bad_y = [y1[1:(end - 1)], y2]  # Wrong length
            @test_throws Exception quadratic_interp(x, Series(bad_y))
        end

        @testset "single series" begin
            sitp = quadratic_interp(x, Series(y1))
            @test FI.n_series(sitp) == 1
        end

        @testset "boundary conditions preserved" begin
            # Test with different BC types
            sitp_left = quadratic_interp(x, Series(ys); bc = FI.Left(FI.QuadraticFit()))
            @test sitp_left isa FI.QuadraticSeriesInterpolant{Float64}

            sitp_right = quadratic_interp(x, Series(ys); bc = FI.Right(FI.QuadraticFit()))
            @test sitp_right isa FI.QuadraticSeriesInterpolant{Float64}
        end
    end

    # ========================================
    # Trait Implementation Tests
    # ========================================

    @testset "trait implementations" begin
        x = collect(0.0:0.1:1.0)
        sitp = quadratic_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))

        @test FI.n_series(sitp) == 2
        @test FI._get_grid(sitp) ≈ x
        @test FI._get_extrap(sitp) isa FI.AbstractExtrap
        @test FI._should_wrap(sitp) == false
        @test FI._method_kind(typeof(sitp)) === Val(:quadratic)
    end

    # ========================================
    # Evaluation Accuracy Tests
    # ========================================

    @testset "evaluation accuracy" begin
        # Quadratic interpolation should be exact for quadratic data
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y1 = x .^ 2              # Quadratic: should be exact
        y2 = 2.0 .* x .+ 1.0    # Linear: should also be exact
        sitp = quadratic_interp(x, Series(y1, y2))

        @testset "at grid points" begin
            for (i, xi) in enumerate(x)
                result = sitp(xi)
                @test result[1] ≈ y1[i] atol = 1.0e-10
                @test result[2] ≈ y2[i] atol = 1.0e-10
            end
        end

        @testset "interpolation midpoints" begin
            # Quadratic interpolation should produce good results at midpoints
            result = sitp(0.5)
            @test result isa Vector{Float64}
            @test length(result) == 2
            # For quadratic y=x², at x=0.5 we expect y≈0.25
            @test result[1] ≈ 0.25 atol = 0.1  # Quadratic spline may not be exact
            # For linear y=2x+1, at x=0.5 we expect y=2.0
            @test result[2] ≈ 2.0 atol = 0.1
        end
    end

    # ========================================
    # Zero-Allocation Tests
    # ========================================

    @testset "zero allocation" begin
        @testset "scalar in-place" begin
            function measure_scalar()
                x = collect(0.0:0.1:1.0)
                sitp = quadratic_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))
                output = zeros(2)
                sitp(output, 0.5)  # Warmup
                sitp(output, 0.5)  # Warmup
                return @allocated sitp(output, 0.5)
            end
            @test measure_scalar() <= ALLOC_THRESHOLD
        end

        @testset "vector in-place" begin
            function measure_vector()
                x = collect(0.0:0.1:1.0)
                sitp = quadratic_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))
                xq = collect(0.0:0.05:1.0)
                outputs = [zeros(length(xq)) for _ in 1:2]
                sitp(outputs, xq)  # Warmup
                sitp(outputs, xq)  # Warmup
                return @allocated sitp(outputs, xq)
            end
            @test measure_vector() <= ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Extrapolation Tests
    # ========================================

    @testset "extrapolation modes" begin
        x = collect(0.0:0.1:1.0)
        ys = [sin.(2π .* x)]

        @testset "extrap=NoExtrap() throws" begin
            sitp = quadratic_interp(x, Series(ys); extrap = NoExtrap())
            @test_throws DomainError sitp(-0.1)
            @test_throws DomainError sitp(1.1)
        end

        @testset "domain error message format" begin
            sitp = quadratic_interp(x, Series(ys); extrap = NoExtrap())
            err = try
                sitp(-0.5)
                nothing
            catch e
                e
            end
            @test err isa DomainError
            @test occursin("outside interpolation domain", string(err))
        end

        @testset "extrap=ClampExtrap() returns boundary" begin
            sitp = quadratic_interp(x, Series(ys); extrap = ClampExtrap())
            @test sitp(-0.1)[1] ≈ sin(0.0) atol = 1.0e-6
            @test sitp(1.1)[1] ≈ sin(2π) atol = 1.0e-6
        end

        @testset "extrap=ExtendExtrap() extrapolates" begin
            sitp = quadratic_interp(x, Series(ys); extrap = ExtendExtrap())
            @test sitp(-0.1) isa Vector{Float64}
            @test sitp(1.1) isa Vector{Float64}
        end
    end

    # ========================================
    # Type Stability
    # ========================================

    @testset "type stability" begin
        x32 = Float32.(collect(0.0:0.1:1.0))
        ys32 = [Float32.(sin.(2π .* Float64.(x32)))]

        x64 = collect(0.0:0.1:1.0)
        ys64 = [sin.(2π .* x64)]

        sitp32 = quadratic_interp(x32, Series(ys32))
        sitp64 = quadratic_interp(x64, Series(ys64))

        @test sitp32(0.5f0) isa Vector{Float32}
        @test sitp64(0.5) isa Vector{Float64}

        @test_nowarn @inferred sitp64(0.5)
    end

    # ========================================
    # Callable Signatures
    # ========================================

    @testset "callable signatures" begin
        x = collect(0.0:0.1:1.0)
        sitp = quadratic_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))

        @testset "scalar out-of-place" begin
            result = sitp(0.5)
            @test length(result) == 2
            @test result isa Vector{Float64}
        end

        @testset "scalar in-place" begin
            output = zeros(2)
            result = sitp(output, 0.5)
            @test result === output
        end

        @testset "vector out-of-place" begin
            xq = [0.25, 0.5, 0.75]
            results = sitp(xq)
            @test length(results) == 2  # n_series
            @test all(r -> length(r) == 3, results)
        end

        @testset "vector in-place" begin
            xq = [0.25, 0.5, 0.75]
            outputs = [zeros(3), zeros(3)]
            result = sitp(outputs, xq)
            @test result === outputs
        end
    end

    # ========================================
    # Derivative Support
    # ========================================

    @testset "derivative evaluation" begin
        x = collect(0.0:0.1:1.0)
        y = x .^ 2  # Simple quadratic for testing
        sitp = quadratic_interp(x, Series(y, 2.0 .* y))

        @testset "first derivative" begin
            # For y = x², dy/dx = 2x at x=0.5 → expect ~1.0
            result = sitp(0.5; deriv = DerivOp(1))
            @test length(result) == 2
            @test result[1] ≈ 1.0 atol = 0.2  # Some tolerance for spline approximation
        end

        @testset "second derivative" begin
            # For y = x², d²y/dx² = 2 (constant)
            result = sitp(0.5; deriv = DerivOp(2))
            @test length(result) == 2
            @test result[1] ≈ 2.0 atol = 0.5  # Larger tolerance for second derivative
        end

        @testset "derivative in-place" begin
            output = zeros(2)
            sitp(output, 0.5; deriv = DerivOp(1))
            @test output[1] ≈ 1.0 atol = 0.2
        end
    end

    # ========================================
    # Consistency with Single Interpolant
    # ========================================

    @testset "consistency with existing implementation" begin
        x = collect(0.0:0.01:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        ys = [y1, y2]

        # Both should give same results
        sitp = quadratic_interp(x, Series(ys))

        # Compare at multiple points
        xq = [0.15, 0.35, 0.55, 0.75, 0.95]
        for xqi in xq
            result = sitp(xqi)
            # Create reference single interpolants and compare
            ref1 = quadratic_interp(x, y1)(xqi)
            ref2 = quadratic_interp(x, y2)(xqi)
            @test result[1] ≈ ref1 atol = 1.0e-10
            @test result[2] ≈ ref2 atol = 1.0e-10
        end
    end

    # ========================================
    # Coefficient Matrix Storage
    # ========================================

    @testset "coefficient storage" begin
        x = collect(0.0:0.1:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = quadratic_interp(x, Series(y1, y2))

        @testset "coefficient matrices exist" begin
            # Should have a, d coefficient matrices
            @test hasfield(typeof(sitp), :a)
            @test hasfield(typeof(sitp), :d)
            @test sitp.a isa Matrix{Float64}
            @test sitp.d isa Matrix{Float64}
        end

        @testset "coefficient dimensions" begin
            n = length(x)
            # Coefficients should be (n_points × n_series) for series-contiguous
            @test size(sitp.a) == (n, 2)
            @test size(sitp.d) == (n, 2)
        end
    end

    # ========================================
    # Type Promotion Tests (coverage for Real type wrappers)
    # ========================================

    @testset "type promotion" begin
        @testset "Integer x with Float y vectors (promotes to Float64)" begin
            x_int = collect(1:10)  # Integer vector
            y1 = sin.(Float64.(x_int))
            y2 = cos.(Float64.(x_int))

            sitp = quadratic_interp(x_int, Series(y1, y2))
            @test sitp isa FI.QuadraticSeriesInterpolant{Float64}

            result = sitp(5.5)
            @test length(result) == 2
            @test all(isfinite, result)
        end

        @testset "Integer x with Integer y vectors (promotes to Float64)" begin
            x_int = collect(1:10)
            y1_int = collect(1:10)
            y2_int = collect(10:-1:1)

            sitp = quadratic_interp(x_int, Series(y1_int, y2_int))
            @test sitp isa FI.QuadraticSeriesInterpolant{Float64}

            result = sitp(5.5)
            @test length(result) == 2
        end

        @testset "Integer x with Integer y matrix (promotes to Float64)" begin
            x_int = collect(1:10)
            Y_int = [i * j for i in 1:10, j in 1:3]

            sitp = quadratic_interp(x_int, Series(Y_int))
            @test sitp isa FI.QuadraticSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 3

            result = sitp(5.5)
            @test length(result) == 3
            @test all(isfinite, result)
        end

        @testset "In-place vector with type-promoted xq" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            sitp = quadratic_interp(x, Series(y1, y2))

            # Float32 query points with Float64 interpolant
            xq_f32 = Float32[0.1, 0.3, 0.5, 0.7, 0.9]
            out1 = Vector{Float64}(undef, 5)
            out2 = Vector{Float64}(undef, 5)
            outputs = [out1, out2]

            result = sitp(outputs, xq_f32)
            @test result === outputs
            @test all(isfinite, out1)
            @test all(isfinite, out2)

            # Verify values match Float64 path
            xq_f64 = Float64.(xq_f32)
            ref = sitp(xq_f64)
            @test out1 ≈ ref[1] atol = 1.0e-10
            @test out2 ≈ ref[2] atol = 1.0e-10
        end
    end

    # ========================================
    # Matrix Dimension Validation Tests
    # ========================================

    @testset "matrix dimension validation" begin
        x = collect(0.0:0.1:1.0)

        @testset "Matrix input with row dimension mismatch" begin
            Y_wrong = rand(5, 3)  # 5 rows but x has 11 points
            @test_throws Exception quadratic_interp(x, Series(Y_wrong))
        end
    end

    # ========================================
    # Vector Extrapolation Branches Tests
    # ========================================

    @testset "vector extrapolation branches" begin
        x = collect(0.0:0.1:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        @testset "vector NoExtrap() extrapolation throws" begin
            sitp = quadratic_interp(x, Series(y1, y2); extrap = NoExtrap())
            xq = [-0.1, 0.5, 1.1]

            @test_throws DomainError sitp(xq)
        end

        @testset "vector ClampExtrap() extrapolation" begin
            sitp = quadratic_interp(x, Series(y1, y2); extrap = ClampExtrap())
            xq = [-0.1, 0.5, 1.1]

            outputs = [zeros(3), zeros(3)]
            sitp(outputs, xq)

            # Out-of-domain should clamp to boundary values
            @test outputs[1][1] ≈ y1[1] atol = 1.0e-10
            @test outputs[1][3] ≈ y1[end] atol = 1.0e-10
        end

        @testset "vector ExtendExtrap() extrapolation" begin
            sitp = quadratic_interp(x, Series(y1, y2); extrap = ExtendExtrap())
            xq = [-0.1, 0.5, 1.1]

            outputs = [zeros(3), zeros(3)]
            sitp(outputs, xq)

            @test !any(isnan, outputs[1])
            @test !any(isnan, outputs[2])
        end

        @testset "vector ClampExtrap() extrapolation with derivatives" begin
            sitp = quadratic_interp(x, Series(y1, y2); extrap = ClampExtrap())
            xq = [-0.1, 0.5, 1.1]

            # deriv=DerivOp(1) outside domain should be zero for constant extrap
            outputs_d1 = [zeros(3), zeros(3)]
            sitp(outputs_d1, xq; deriv = DerivOp(1))
            @test outputs_d1[1][1] ≈ 0.0 atol = 1.0e-10  # Left boundary
            @test outputs_d1[1][3] ≈ 0.0 atol = 1.0e-10  # Right boundary

            # deriv=DerivOp(2) outside domain should be zero
            outputs_d2 = [zeros(3), zeros(3)]
            sitp(outputs_d2, xq; deriv = DerivOp(2))
            @test outputs_d2[1][1] ≈ 0.0 atol = 1.0e-10
            @test outputs_d2[1][3] ≈ 0.0 atol = 1.0e-10
        end
    end

    # ========================================
    # Scalar Constant Extrapolation with Derivatives
    # ========================================

    @testset "scalar constant extrap inside domain" begin
        # Test that ClampExtrap() extrap still works correctly inside domain
        x = collect(0.0:0.1:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp_const = quadratic_interp(x, Series(y1, y2); extrap = ClampExtrap())
        sitp_none = quadratic_interp(x, Series(y1, y2); extrap = NoExtrap())

        @testset "value inside domain same as NoExtrap() extrap" begin
            result_const = sitp_const(0.5)
            result_none = sitp_none(0.5)
            @test result_const ≈ result_none atol = 1.0e-10
        end

        @testset "deriv inside domain same as NoExtrap() extrap" begin
            result_const = sitp_const(0.5; deriv = DerivOp(1))
            result_none = sitp_none(0.5; deriv = DerivOp(1))
            @test result_const ≈ result_none atol = 1.0e-10
        end
    end

    @testset "scalar constant extrap derivatives" begin
        x = collect(0.0:0.1:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = quadratic_interp(x, Series(y1, y2); extrap = ClampExtrap())

        @testset "deriv=DerivOp(1) outside domain returns zero" begin
            result_below = sitp(-0.1; deriv = DerivOp(1))
            @test result_below[1] ≈ 0.0 atol = 1.0e-10
            @test result_below[2] ≈ 0.0 atol = 1.0e-10

            result_above = sitp(1.1; deriv = DerivOp(1))
            @test result_above[1] ≈ 0.0 atol = 1.0e-10
            @test result_above[2] ≈ 0.0 atol = 1.0e-10
        end

        @testset "deriv=DerivOp(2) outside domain returns zero" begin
            result_below = sitp(-0.1; deriv = DerivOp(2))
            @test result_below[1] ≈ 0.0 atol = 1.0e-10
            @test result_below[2] ≈ 0.0 atol = 1.0e-10

            result_above = sitp(1.1; deriv = DerivOp(2))
            @test result_above[1] ≈ 0.0 atol = 1.0e-10
            @test result_above[2] ≈ 0.0 atol = 1.0e-10
        end

        @testset "value outside domain returns boundary" begin
            result_below = sitp(-0.1)
            @test result_below[1] ≈ y1[1] atol = 1.0e-10
            @test result_below[2] ≈ y2[1] atol = 1.0e-10

            result_above = sitp(1.1)
            @test result_above[1] ≈ y1[end] atol = 1.0e-10
            @test result_above[2] ≈ y2[end] atol = 1.0e-10
        end

        @testset "in-place scalar with constant extrap and derivatives" begin
            out = zeros(2)

            # deriv=DerivOp(0) outside domain
            sitp(out, -0.1)
            @test out[1] ≈ y1[1] atol = 1.0e-10

            # deriv=DerivOp(1) outside domain
            sitp(out, -0.1; deriv = DerivOp(1))
            @test out[1] ≈ 0.0 atol = 1.0e-10

            # deriv=DerivOp(2) outside domain
            sitp(out, -0.1; deriv = DerivOp(2))
            @test out[1] ≈ 0.0 atol = 1.0e-10
        end
    end

    # ========================================
    # Pre-built Anchor Evaluation Tests
    # ========================================

    @testset "pre-built anchor evaluation" begin
        x = collect(0.0:0.1:1.0)
        y1, y2 = sin.(2π .* x), cos.(2π .* x)
        sitp = quadratic_interp(x, Series(y1, y2))
        xq = [0.15, 0.35, 0.75]

        # Build anchors manually
        aq_vec = FI._anchor_query(x, xq, Val(:quadratic))

        # Allocate outputs
        outputs = [zeros(length(xq)) for _ in 1:2]

        # Evaluate with pre-built anchors
        sitp(outputs, aq_vec)

        # Compare with direct evaluation
        expected = sitp(xq)
        @test outputs[1] ≈ expected[1] atol = 1.0e-12
        @test outputs[2] ≈ expected[2] atol = 1.0e-12

        # Test with derivatives
        outputs_d1 = [zeros(length(xq)) for _ in 1:2]
        sitp(outputs_d1, aq_vec; deriv = DerivOp(1))
        expected_d1 = sitp(xq; deriv = DerivOp(1))
        @test outputs_d1[1] ≈ expected_d1[1] atol = 1.0e-12
    end

end  # testset "QuadraticSeriesInterpolant"
