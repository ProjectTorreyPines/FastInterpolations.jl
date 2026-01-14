# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  CONSTANT SERIES INTERPOLANT TESTS                        ║
# ║         Tests for ConstantSeriesInterpolant with direct matrix storage    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Phase E.2: Tests verify ConstantSeriesInterpolant works correctly using the
# shared series infrastructure from Phases A-D.
#

using Test
using FastInterpolations
const FI = FastInterpolations

@testset "ConstantSeriesInterpolant" begin

    # ========================================
    # Constructor Tests
    # ========================================

    @testset "construction" begin
        x = collect(0.0:0.1:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        ys = [y1, y2]

        @testset "from vector of vectors" begin
            sitp = constant_interp(x, ys)
            @test sitp isa FI.ConstantSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 2
        end

        @testset "from matrix" begin
            Y = hcat(y1, y2)
            sitp = constant_interp(x, Y)
            @test sitp isa FI.ConstantSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 2
        end

        @testset "validates input dimensions" begin
            bad_y = [y1[1:end-1], y2]  # Wrong length
            @test_throws Exception constant_interp(x, bad_y)
        end

        @testset "single series" begin
            sitp = constant_interp(x, [y1])
            @test FI.n_series(sitp) == 1
        end

        @testset "side parameter preserved" begin
            for side_mode in (:nearest, :left, :right)
                sitp = constant_interp(x, ys; side=side_mode)
                @test sitp.side === Val(side_mode)
            end
        end
    end

    # ========================================
    # Trait Implementation Tests
    # ========================================

    @testset "trait implementations" begin
        x = collect(0.0:0.1:1.0)
        sitp = constant_interp(x, [sin.(2π .* x), cos.(2π .* x)])

        @test FI.n_series(sitp) == 2
        @test FI._get_grid(sitp) ≈ x
        @test FI._get_extrap(sitp) isa FI.ExtrapVal
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
        sitp = constant_interp(x, [y1, y2]; side=:left)

        @testset "at grid points" begin
            for i in 1:(length(x)-1)
                result = sitp(x[i])
                @test result[1] == y1[i]
                @test result[2] == y2[i]
            end
        end

        @testset "midpoints with side=:left" begin
            result = sitp(0.5)
            @test result[1] == 1.0  # Left value in [0,1)
            @test result[2] == 5.0  # Constant
        end

        @testset "midpoints with side=:nearest" begin
            sitp_nearest = constant_interp(x, [y1, y2]; side=:nearest)
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
        x = collect(0.0:0.1:1.0)
        sitp = constant_interp(x, [sin.(2π .* x), cos.(2π .* x)])

        @testset "scalar in-place" begin
            output = zeros(2)
            sitp(output, 0.5)  # Warmup
            sitp(output, 0.5)  # Warmup
            allocs = @allocated sitp(output, 0.5)
            @test allocs <= ALLOC_THRESHOLD
        end

        @testset "vector in-place" begin
            xq = collect(0.0:0.05:1.0)
            outputs = [zeros(length(xq)) for _ in 1:2]
            sitp(outputs, xq)  # Warmup
            sitp(outputs, xq)  # Warmup
            allocs = @allocated sitp(outputs, xq)
            if VERSION >= v"1.12"
                @test allocs == 0
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

        @testset "extrap=:none throws" begin
            sitp = constant_interp(x, ys; extrap=:none)
            @test_throws DomainError sitp(-0.1)
            @test_throws DomainError sitp(1.1)
        end

        @testset "extrap=:constant returns boundary" begin
            sitp = constant_interp(x, ys; extrap=:constant)
            @test sitp(-0.1)[1] ≈ sin(0.0) atol=1e-6
            @test sitp(1.1)[1] ≈ sin(2π) atol=1e-6
        end

        @testset "extrap=:extension extrapolates" begin
            sitp = constant_interp(x, ys; extrap=:extension)
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

        sitp32 = constant_interp(x32, ys32)
        sitp64 = constant_interp(x64, ys64)

        @test sitp32(0.5f0) isa Vector{Float32}
        @test sitp64(0.5) isa Vector{Float64}

        @test_nowarn @inferred sitp64(0.5)
    end

    # ========================================
    # Callable Signatures
    # ========================================

    @testset "callable signatures" begin
        x = collect(0.0:0.1:1.0)
        sitp = constant_interp(x, [sin.(2π .* x), cos.(2π .* x)])

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
        sitp = constant_interp(x, ys; side=:nearest)

        # Compare at multiple points
        xq = [0.15, 0.35, 0.55, 0.75, 0.95]
        for xqi in xq
            result = sitp(xqi)
            # Constant interpolation returns nearest value
            idx, xL, xR = FI._find_interval(x, xqi)
            h = xR - xL
            dL = xqi - xL
            expected1 = dL <= h / 2 ? y1[idx] : y1[idx+1]
            expected2 = dL <= h / 2 ? y2[idx] : y2[idx+1]
            @test result[1] ≈ expected1 atol=1e-10
            @test result[2] ≈ expected2 atol=1e-10
        end
    end

    # ========================================
    # Derivative Support (Phase 4)
    # ========================================

    @testset "derivative support" begin
        x = [0.0, 1.0, 2.0, 3.0]
        ys = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]
        sitp = constant_interp(x, ys)

        @testset "deriv=0 returns values (existing behavior)" begin
            @test sitp(0.5; deriv=0) == sitp(0.5)
            @test sitp(1.5; deriv=0) == sitp(1.5)
        end

        @testset "deriv=1 returns zeros" begin
            @test sitp(0.5; deriv=1) == [0.0, 0.0]
            @test sitp(1.5; deriv=1) == [0.0, 0.0]
            @test sitp(2.5; deriv=1) == [0.0, 0.0]
        end

        @testset "deriv=2 returns zeros" begin
            @test sitp(0.5; deriv=2) == [0.0, 0.0]
            @test sitp(1.5; deriv=2) == [0.0, 0.0]
        end

        @testset "vector evaluation with deriv" begin
            xq = [0.5, 1.5, 2.5]
            results_d1 = sitp(xq; deriv=1)
            @test all(r -> r == zeros(3), results_d1)

            results_d2 = sitp(xq; deriv=2)
            @test all(r -> r == zeros(3), results_d2)
        end
    end

    @testset "derivative with extrapolation" begin
        x = [0.0, 1.0, 2.0]
        ys = [[1.0, 2.0, 3.0]]
        sitp = constant_interp(x, ys; extrap=:constant)

        @testset "outside boundaries derivative is zero" begin
            @test sitp(-0.5; deriv=1) == [0.0]
            @test sitp(2.5; deriv=1) == [0.0]
            @test sitp(-0.5; deriv=2) == [0.0]
            @test sitp(2.5; deriv=2) == [0.0]
        end
    end

    @testset "x_max boundary value preserved" begin
        x = [0.0, 1.0, 2.0]
        ys = [[1.0, 2.0, 3.0]]
        sitp = constant_interp(x, ys)

        @testset "x_max returns last value (CRITICAL)" begin
            # CRITICAL: x_max should return last value, not second-to-last
            @test sitp(2.0) == [3.0]  # Must be 3.0, not 2.0
            @test sitp(2.0; deriv=0) == [3.0]
            @test sitp(2.0; deriv=1) == [0.0]
            @test sitp(2.0; deriv=2) == [0.0]
        end
    end

end  # testset "ConstantSeriesInterpolant"
