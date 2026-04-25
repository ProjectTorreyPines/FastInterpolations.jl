# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  CONSTANT SERIES INTERPOLANT TESTS                        ║
# ║         Tests for ConstantSeriesInterpolant with direct matrix storage    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Phase E.2: Tests verify ConstantSeriesInterpolant works correctly using the
# shared series infrastructure from Phases A-D.
#

@testitem "ConstantSeriesInterpolant" setup=[AllocConstants] begin
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
            sitp = constant_interp(x, Series(ys))
            @test sitp isa FI.ConstantSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 2
        end

        @testset "from matrix" begin
            Y = hcat(y1, y2)
            sitp = constant_interp(x, Series(Y))
            @test sitp isa FI.ConstantSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 2
        end

        @testset "validates input dimensions" begin
            bad_y = [y1[1:(end - 1)], y2]  # Wrong length
            @test_throws Exception constant_interp(x, Series(bad_y))
        end

        @testset "single series" begin
            sitp = constant_interp(x, Series(y1))
            @test FI.n_series(sitp) == 1
        end

        @testset "side parameter preserved" begin
            for side_mode in (NearestSide(), LeftSide(), RightSide())
                sitp = constant_interp(x, Series(ys); side = side_mode)
                @test sitp.side === side_mode
            end
        end
    end

    # ========================================
    # Trait Implementation Tests
    # ========================================

    @testset "trait implementations" begin
        x = collect(0.0:0.1:1.0)
        sitp = constant_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))

        @test FI.n_series(sitp) == 2
        @test FI._get_grid(sitp) ≈ x
        @test FI._get_extrap(sitp) isa FI.AbstractExtrap
        @test FI._should_wrap(sitp) == false
        @test FI._method_kind(typeof(sitp)) === Val(:constant)
    end

    # ========================================
    # Evaluation Accuracy Tests
    # ========================================

    @testset "evaluation accuracy" begin
        # Constant interpolation should be exact for constant data
        x = [0.0, 1.0, 2.0, 3.0]
        y1 = [1.0, 2.0, 3.0, 4.0]  # step function
        y2 = [5.0, 5.0, 5.0, 5.0]  # constant
        sitp = constant_interp(x, Series(y1, y2); side = LeftSide())

        @testset "at grid points" begin
            for i in 1:(length(x) - 1)
                result = sitp(x[i])
                @test result[1] == y1[i]
                @test result[2] == y2[i]
            end
        end

        @testset "midpoints with side=LeftSide()" begin
            result = sitp(0.5)
            @test result[1] == 1.0  # Left value in [0,1)
            @test result[2] == 5.0  # Constant
        end

        @testset "midpoints with side=NearestSide()" begin
            sitp_nearest = constant_interp(x, Series(y1, y2); side = NearestSide())
            # At 0.4, closer to 0 -> y[1]=1.0
            result = sitp_nearest(0.4)
            @test result[1] == 1.0
            # At 0.6, closer to 1 -> y[2]=2.0
            result = sitp_nearest(0.6)
            @test result[1] == 2.0
        end
    end

    # ========================================
    # Zero-Allocation Tests
    # ========================================

    @testset "zero allocation" begin
        @testset "scalar in-place" begin
            function measure_scalar()
                x = collect(0.0:0.1:1.0)
                sitp = constant_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))
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
                sitp = constant_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))
                xq = collect(0.0:0.05:1.0)
                outputs = [zeros(length(xq)) for _ in 1:2]
                sitp(outputs, xq)  # Warmup
                sitp(outputs, xq)  # Warmup
                return @allocated sitp(outputs, xq)
            end
            allocs = measure_vector()
            if VERSION >= v"1.12"
                @test allocs <= ALLOC_THRESHOLD
            else
                @test allocs <= 5000 # Allow higher allocation for older Julia versions
            end
        end
    end

    # ========================================
    # Extrapolation Tests
    # ========================================

    @testset "extrapolation modes" begin
        x = collect(0.0:0.1:1.0)
        ys = [sin.(2π .* x)]

        @testset "extrap=NoExtrap() throws" begin
            sitp = constant_interp(x, Series(ys); extrap = NoExtrap())
            @test_throws DomainError sitp(-0.1)
            @test_throws DomainError sitp(1.1)
        end

        @testset "extrap=ClampExtrap() returns boundary" begin
            sitp = constant_interp(x, Series(ys); extrap = ClampExtrap())
            @test sitp(-0.1)[1] ≈ sin(0.0) atol = 1.0e-6
            @test sitp(1.1)[1] ≈ sin(2π) atol = 1.0e-6
        end

        @testset "extrap=ExtendExtrap() extrapolates" begin
            sitp = constant_interp(x, Series(ys); extrap = ExtendExtrap())
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

        sitp32 = constant_interp(x32, Series(ys32))
        sitp64 = constant_interp(x64, Series(ys64))

        @test sitp32(0.5f0) isa Vector{Float32}
        @test sitp64(0.5) isa Vector{Float64}

        @test_nowarn @inferred sitp64(0.5)
    end

    # ========================================
    # Precompute Transpose Tests
    # ========================================

    @testset "precompute_transpose!" begin
        x = collect(0.0:0.1:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        @testset "returns self for method chaining" begin
            sitp = constant_interp(x, Series(y1, y2))
            result = precompute_transpose!(sitp)
            @test result === sitp
        end

        @testset "computes transpose matrix" begin
            sitp = constant_interp(x, Series(y1, y2))
            # Before precompute, snapshot should be nothing
            @test FI._get_snapshot(sitp._transpose) === nothing

            precompute_transpose!(sitp)
            # After precompute, snapshot should be a matrix
            snapshot = FI._get_snapshot(sitp._transpose)
            @test snapshot !== nothing
            @test snapshot isa Matrix{Float64}
            @test size(snapshot) == (2, 11)  # (n_series × n_points)
        end

        @testset "scalar evaluation works after precompute" begin
            sitp = constant_interp(x, Series(y1, y2))
            precompute_transpose!(sitp)

            result = sitp(0.5)
            @test length(result) == 2
            @test all(isfinite, result)

            # Verify values match non-precomputed path
            sitp_ref = constant_interp(x, Series(y1, y2))
            @test result ≈ sitp_ref(0.5) atol = 1.0e-10
        end

        @testset "method chaining pattern" begin
            sitp = precompute_transpose!(constant_interp(x, Series(y1, y2)))
            @test sitp isa FI.ConstantSeriesInterpolant{Float64}
            @test FI._get_snapshot(sitp._transpose) !== nothing
        end
    end

    # ========================================
    # Callable Signatures
    # ========================================

    @testset "callable signatures" begin
        x = collect(0.0:0.1:1.0)
        sitp = constant_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))

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
    # Consistency with Single Interpolant
    # ========================================

    @testset "consistency with existing implementation" begin
        x = collect(0.0:0.01:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        ys = [y1, y2]

        # Both should give same results
        sitp = constant_interp(x, Series(ys); side = NearestSide())

        # Compare at multiple points
        xq = [0.15, 0.35, 0.55, 0.75, 0.95]
        for xqi in xq
            result = sitp(xqi)
            # Constant interpolation returns nearest value
            idx, _, xL, xR = FI._search_interval(x, xqi)
            h = xR - xL
            dL = xqi - xL
            expected1 = dL <= h / 2 ? y1[idx] : y1[idx + 1]
            expected2 = dL <= h / 2 ? y2[idx] : y2[idx + 1]
            @test result[1] ≈ expected1 atol = 1.0e-10
            @test result[2] ≈ expected2 atol = 1.0e-10
        end
    end

    # ========================================
    # Derivative Support (Phase 4)
    # ========================================

    @testset "derivative support" begin
        x = [0.0, 1.0, 2.0, 3.0]
        ys = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]
        sitp = constant_interp(x, Series(ys))

        @testset "deriv=DerivOp(0) returns values (existing behavior)" begin
            @test sitp(0.5; deriv = DerivOp(0)) == sitp(0.5)
            @test sitp(1.5; deriv = DerivOp(0)) == sitp(1.5)
        end

        @testset "deriv=DerivOp(1) returns zeros" begin
            @test sitp(0.5; deriv = DerivOp(1)) == [0.0, 0.0]
            @test sitp(1.5; deriv = DerivOp(1)) == [0.0, 0.0]
            @test sitp(2.5; deriv = DerivOp(1)) == [0.0, 0.0]
        end

        @testset "deriv=DerivOp(2) returns zeros" begin
            @test sitp(0.5; deriv = DerivOp(2)) == [0.0, 0.0]
            @test sitp(1.5; deriv = DerivOp(2)) == [0.0, 0.0]
        end

        @testset "vector evaluation with deriv" begin
            xq = [0.5, 1.5, 2.5]
            results_d1 = sitp(xq; deriv = DerivOp(1))
            @test all(r -> r == zeros(3), results_d1)

            results_d2 = sitp(xq; deriv = DerivOp(2))
            @test all(r -> r == zeros(3), results_d2)
        end
    end

    @testset "derivative with extrapolation" begin
        x = [0.0, 1.0, 2.0]
        ys = [[1.0, 2.0, 3.0]]
        sitp = constant_interp(x, Series(ys); extrap = ClampExtrap())

        @testset "outside boundaries derivative is zero" begin
            @test sitp(-0.5; deriv = DerivOp(1)) == [0.0]
            @test sitp(2.5; deriv = DerivOp(1)) == [0.0]
            @test sitp(-0.5; deriv = DerivOp(2)) == [0.0]
            @test sitp(2.5; deriv = DerivOp(2)) == [0.0]
        end
    end

    @testset "x_max boundary value preserved" begin
        x = [0.0, 1.0, 2.0]
        ys = [[1.0, 2.0, 3.0]]
        sitp = constant_interp(x, Series(ys))

        @testset "x_max returns last value (CRITICAL)" begin
            # CRITICAL: x_max should return last value, not second-to-last
            @test sitp(2.0) == [3.0]  # Must be 3.0, not 2.0
            @test sitp(2.0; deriv = DerivOp(0)) == [3.0]
            @test sitp(2.0; deriv = DerivOp(1)) == [0.0]
            @test sitp(2.0; deriv = DerivOp(2)) == [0.0]
        end

        @testset "x_max derivative in vector query" begin
            xq = [1.5, 2.0]  # Include x_max in vector query
            outputs_d1 = [zeros(2)]
            sitp(outputs_d1, xq; deriv = DerivOp(1))
            @test outputs_d1[1][2] ≈ 0.0 atol = 1.0e-10  # deriv at x_max should be 0
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

            sitp = constant_interp(x_int, Series(y1, y2))
            @test sitp isa FI.ConstantSeriesInterpolant{Float64}

            result = sitp(5.5)
            @test length(result) == 2
            @test all(isfinite, result)
        end

        @testset "Integer x with Integer y vectors (promotes to Float64)" begin
            x_int = collect(1:10)
            y1_int = collect(1:10)
            y2_int = collect(10:-1:1)

            sitp = constant_interp(x_int, Series(y1_int, y2_int))
            @test sitp isa FI.ConstantSeriesInterpolant{Float64}

            result = sitp(5.5)
            @test length(result) == 2
        end

        @testset "Integer x with Integer y matrix (promotes to Float64)" begin
            x_int = collect(1:10)
            Y_int = [i * j for i in 1:10, j in 1:3]

            sitp = constant_interp(x_int, Series(Y_int))
            @test sitp isa FI.ConstantSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 3

            result = sitp(5.5)
            @test length(result) == 3
            @test all(isfinite, result)
        end

        @testset "In-place vector with type-promoted xq" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            sitp = constant_interp(x, Series(y1, y2))

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
            @test_throws DimensionMismatch constant_interp(x, Series(Y_wrong))
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
            sitp = constant_interp(x, Series(y1, y2); extrap = NoExtrap())
            xq = [-0.1, 0.5, 1.1]

            @test_throws DomainError sitp(xq)
        end

        @testset "vector ClampExtrap() extrapolation" begin
            sitp = constant_interp(x, Series(y1, y2); extrap = ClampExtrap())
            xq = [-0.1, 0.5, 1.1]

            outputs = [zeros(3), zeros(3)]
            sitp(outputs, xq)

            # Out-of-domain should clamp to boundary values
            @test outputs[1][1] ≈ y1[1] atol = 1.0e-10
            @test outputs[1][3] ≈ y1[end] atol = 1.0e-10
        end

        @testset "vector ExtendExtrap() extrapolation" begin
            sitp = constant_interp(x, Series(y1, y2); extrap = ExtendExtrap())
            xq = [-0.1, 0.5, 1.1]

            outputs = [zeros(3), zeros(3)]
            sitp(outputs, xq)

            @test !any(isnan, outputs[1])
            @test !any(isnan, outputs[2])
        end

        @testset "vector ClampExtrap() extrapolation with derivatives" begin
            sitp = constant_interp(x, Series(y1, y2); extrap = ClampExtrap())
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

end  # testset "ConstantSeriesInterpolant"
