# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     SERIES KERNEL DISPATCH TESTS                          ║
# ║         Tests for parameterized SIMD kernels in series_kernels.jl         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Phase D: Tests verify that method-parameterized kernels work correctly
# and specialize at compile time for optimal SIMD performance.
#

using Test
using FastInterpolations
const FI = FastInterpolations

@testset "Series Kernel Dispatch" begin

    # ========================================
    # Method Kind Trait
    # ========================================

    @testset "method kind trait" begin
        x = collect(0.0:0.1:1.0)
        ys = [sin.(2π .* x)]

        sitp_cubic = cubic_interp(x, ys)
        @test FI._method_kind(typeof(sitp_cubic)) === Val(:cubic)

        # Note: Linear, Constant, Quadratic tests will be added in Phase E
        # when those SeriesInterpolant types exist
    end

    # ========================================
    # Kernel Evaluation - Cubic
    # ========================================

    @testset "kernel evaluation - cubic" begin
        T = Float64
        # Cubic uses 4 coefficients: yL, yR, zL, zR
        coeffs = (1.0, 2.0, 0.1, 0.2)
        # Cubic uses 4 weights: wyL, wyR, wzL, wzR
        weights = (0.3, 0.7, 0.05, 0.05)

        result = FI._eval_kernel(coeffs, weights, Val(:cubic))

        # Manual calculation: wyR*yR + wyL*yL + wzR*zR + wzL*zL
        # = 0.7*2.0 + 0.3*1.0 + 0.05*0.2 + 0.05*0.1
        # = 1.4 + 0.3 + 0.01 + 0.005 = 1.715
        expected = 0.7 * 2.0 + 0.3 * 1.0 + 0.05 * 0.2 + 0.05 * 0.1
        @test result ≈ expected
    end

    # ========================================
    # Kernel Evaluation - Linear
    # ========================================

    @testset "kernel evaluation - linear" begin
        # Linear uses 2 coefficients: yL, yR
        coeffs = (1.0, 2.0)
        # Linear uses 2 weights: wyL, wyR
        weights = (0.3, 0.7)

        result = FI._eval_kernel(coeffs, weights, Val(:linear))

        # Manual calculation: wyR*yR + wyL*yL = 0.7*2.0 + 0.3*1.0 = 1.7
        expected = 0.7 * 2.0 + 0.3 * 1.0
        @test result ≈ expected
    end

    # ========================================
    # Kernel Evaluation - Constant
    # ========================================

    @testset "kernel evaluation - constant" begin
        # Constant uses 1 coefficient: y
        coeffs = (42.0,)
        # Constant has no weights (empty tuple)
        weights = ()

        result = FI._eval_kernel(coeffs, weights, Val(:constant))
        @test result == 42.0
    end

    # ========================================
    # Kernel Evaluation - Quadratic
    # ========================================

    @testset "kernel evaluation - quadratic" begin
        # Quadratic uses 3 coefficients: y0, y1, y2
        coeffs = (1.0, 4.0, 9.0)
        # Quadratic uses 3 weights: w0, w1, w2
        weights = (0.2, 0.5, 0.3)

        result = FI._eval_kernel(coeffs, weights, Val(:quadratic))

        # Manual calculation: w0*y0 + w1*y1 + w2*y2
        # = 0.2*1.0 + 0.5*4.0 + 0.3*9.0 = 0.2 + 2.0 + 2.7 = 4.9
        expected = 0.2 * 1.0 + 0.5 * 4.0 + 0.3 * 9.0
        @test result ≈ expected
    end

    # ========================================
    # Type Stability
    # ========================================

    @testset "type stability" begin
        # Float32 tests
        coeffs32 = (1.0f0, 2.0f0)
        weights32 = (0.3f0, 0.7f0)

        result32 = FI._eval_kernel(coeffs32, weights32, Val(:linear))
        @test result32 isa Float32

        # Float64 tests
        coeffs64 = (1.0, 2.0)
        weights64 = (0.3, 0.7)

        result64 = FI._eval_kernel(coeffs64, weights64, Val(:linear))
        @test result64 isa Float64

        # Verify inferred types (no Union types)
        @test_nowarn @inferred FI._eval_kernel(coeffs64, weights64, Val(:linear))
        @test_nowarn @inferred FI._eval_kernel((1.0, 2.0, 0.1, 0.2), (0.3, 0.7, 0.05, 0.05), Val(:cubic))
    end

    # ========================================
    # Zero Allocation
    # ========================================

    @testset "zero allocation" begin
        coeffs = (1.0, 2.0, 0.1, 0.2)
        weights = (0.3, 0.7, 0.05, 0.05)

        # Warmup
        FI._eval_kernel(coeffs, weights, Val(:cubic))

        # Should have zero allocations
        allocs = @allocated FI._eval_kernel(coeffs, weights, Val(:cubic))
        @test allocs == 0

        # Linear kernel
        FI._eval_kernel((1.0, 2.0), (0.3, 0.7), Val(:linear))
        allocs_linear = @allocated FI._eval_kernel((1.0, 2.0), (0.3, 0.7), Val(:linear))
        @test allocs_linear == 0

        # Constant kernel
        FI._eval_kernel((42.0,), (), Val(:constant))
        allocs_const = @allocated FI._eval_kernel((42.0,), (), Val(:constant))
        @test allocs_const == 0
    end

    # ========================================
    # Coefficient Gathering - Point Layout
    # ========================================

    @testset "coefficient gathering - point layout" begin
        # Point-contiguous layout: n_series × n_points
        # This is the layout used for scalar SIMD evaluation
        y_point = Float64[1.0 2.0 3.0 4.0;   # series 1, points 1-4
                          5.0 6.0 7.0 8.0]   # series 2, points 1-4

        @testset "linear gathers 2 coefficients" begin
            # At idx=2, we need y[k, 2] and y[k, 3] for series k
            # For series 1: yL=2.0, yR=3.0
            coeffs = FI._gather_coefficients_point((y_point,), 1, 2, Val(:linear))
            @test coeffs == (2.0, 3.0)

            # For series 2: yL=6.0, yR=7.0
            coeffs2 = FI._gather_coefficients_point((y_point,), 2, 2, Val(:linear))
            @test coeffs2 == (6.0, 7.0)
        end

        @testset "constant gathers 1 coefficient" begin
            # For constant, we just need y[k, idx]
            coeffs = FI._gather_coefficients_point((y_point,), 1, 2, Val(:constant))
            @test coeffs == (2.0,)
        end

        @testset "cubic gathers 4 coefficients" begin
            # Cubic needs y and z matrices
            z_point = Float64[0.1 0.2 0.3 0.4;
                              0.5 0.6 0.7 0.8]

            # At idx=2: yL=y[k,2], yR=y[k,3], zL=z[k,2], zR=z[k,3]
            coeffs = FI._gather_coefficients_point((y_point, z_point), 1, 2, Val(:cubic))
            @test coeffs == (2.0, 3.0, 0.2, 0.3)
        end

        @testset "quadratic gathers 3 coefficients" begin
            # Quadratic needs 3 points: y[k, idx-1], y[k, idx], y[k, idx+1]
            # (or similar pattern depending on boundary handling)
            coeffs = FI._gather_coefficients_point((y_point,), 1, 2, Val(:quadratic))
            @test length(coeffs) == 3
        end
    end

    # ========================================
    # Integration: Kernel with Real Interpolant
    # ========================================

    @testset "kernel integration with CubicSeriesInterpolant" begin
        x = collect(0.0:0.1:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        # Verify method kind
        @test FI._method_kind(typeof(sitp)) === Val(:cubic)

        # Evaluation should still work correctly
        result = sitp(0.5)
        @test length(result) == 2
        @test result[1] ≈ sin(π) atol=1e-2
        @test result[2] ≈ cos(π) atol=1e-2
    end

end  # testset "Series Kernel Dispatch"
