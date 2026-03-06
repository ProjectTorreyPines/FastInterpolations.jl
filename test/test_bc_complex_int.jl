# ========================================
# Complex{Int} Type Promotion Regression Tests
# ========================================
#
# Bug: Type promotion condition `Tv_real <: AbstractFloat` skipped Complex{Int}
# Fix: _ensure_promoted_xy helper handles all Real-based types including Int
#
# Coverage: Single interpolant creation (cubic, linear, constant, quadratic)

using Test
using FastInterpolations

@testset "Complex{Int} Type Promotion" begin
    # Common test data
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_cint = Complex{Int}[1 + 2im, 3 + 4im, 5 + 6im, 7 + 8im, 9 + 10im]
    y_cf64 = ComplexF64.(y_cint)

    # ========================================
    # Cubic Interpolant + BC Types
    # ========================================
    @testset "cubic_interp" begin
        # Core BC types (representative selection)
        @testset "BC=$bc" for bc in [
                ZeroCurvBC(), Deriv1(0.0), Deriv2(0.5),
                ZeroSlopeBC(), LinearFit(), QuadraticFit(),
            ]
            itp = cubic_interp(x, y_cint; bc = bc)
            @test itp isa CubicInterpolant{Float64, ComplexF64}
            @test itp(1.5) isa ComplexF64
        end

        # BCPair (asymmetric)
        @testset "BCPair" begin
            itp = cubic_interp(x, y_cint; bc = BCPair(Deriv1(0.0), Deriv2(0.0)))
            @test itp isa CubicInterpolant{Float64, ComplexF64}
        end

        # Accuracy check: Complex{Int} vs ComplexF64 should match
        @testset "accuracy" begin
            itp_int = cubic_interp(x, y_cint; bc = Deriv1(0.0))
            itp_f64 = cubic_interp(x, y_cf64; bc = Deriv1(0.0))
            @test isapprox(itp_int(1.5), itp_f64(1.5); rtol = 1.0e-10)
        end
    end

    # ========================================
    # Other Interpolant Types
    # ========================================
    @testset "linear_interp" begin
        itp = linear_interp(x, y_cint)
        @test itp isa LinearInterpolant{Float64, ComplexF64}
        @test itp(1.5) isa ComplexF64
    end

    @testset "constant_interp" begin
        itp = constant_interp(x, y_cint)
        @test itp isa ConstantInterpolant{Float64, ComplexF64}
        @test itp(1.5) isa ComplexF64
    end

    @testset "quadratic_interp" begin
        itp = quadratic_interp(x, y_cint)
        @test itp isa QuadraticInterpolant{Float64, ComplexF64}
        @test itp(1.5) isa ComplexF64
    end

    # ========================================
    # Edge Cases
    # ========================================
    @testset "edge cases" begin
        # Integer range grid
        @testset "Int range grid" begin
            itp = cubic_interp(0:4, y_cint; bc = Deriv1(0.0))
            @test itp isa CubicInterpolant{Float64, ComplexF64}
        end

        # Complex{Int32}
        @testset "Complex{Int32}" begin
            y_int32 = Complex{Int32}.(y_cint)
            itp = cubic_interp(x, y_int32; bc = Deriv1(0.0))
            @test itp isa CubicInterpolant{Float64, ComplexF64}
        end

        # Non-integer BC values
        @testset "non-integer BC" begin
            itp = cubic_interp(x, y_cint; bc = Deriv1(1.5))
            @test itp isa CubicInterpolant{Float64, ComplexF64}
        end

        # Plain Int (not Complex{Int})
        @testset "plain Int values" begin
            y_int = [1, 2, 3, 4, 5]
            itp = linear_interp(x, y_int)
            @test itp isa LinearInterpolant{Float64, Float64}
            @test itp(1.5) isa Float64
        end
    end
end
