# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         ENZYME AD TESTS                                   ║
# ║         Tests for Enzyme LLVM-level AD support in interpolants            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Enzyme performs automatic differentiation at the LLVM IR level, which can be
# faster than source-level AD but has compatibility limitations with certain
# Julia patterns.
#
# SUPPORTED:
#   - Single interpolant evaluation (Linear, Constant, Quadratic, Cubic)
#   - One-shot API (Linear)
#   - Complex values via real()/imag()
#
# NOT SUPPORTED:
#   - Enzyme.gradient API (different interface)
#   - Series interpolants (array mutation)
#

using Test
using FastInterpolations

# Try to import Enzyme, skip all tests if not available
const ENZYME_AVAILABLE = try
    @eval import Enzyme
    true
catch
    false
end

if !ENZYME_AVAILABLE
    @testset "Enzyme AD Support (skipped - not installed)" begin
        @test_skip "Enzyme not available"
    end
else

@testset "Enzyme AD Support" begin

    # ════════════════════════════════════════════════════════════════════════
    # ENZYME COMPATIBILITY CHECK
    # ════════════════════════════════════════════════════════════════════════

    @testset "Enzyme Compatibility Status" begin
        x = collect(0.0:0.5:2.0)
        y = sin.(x)
        itp = linear_interp(x, y; extrap=:extension)

        @testset "basic autodiff works" begin
            f(xq) = itp(xq)
            x0 = 0.73
            result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(x0))
            @test isfinite(result[1][1])

            # Cross-validate with analytical
            analytical = itp(x0; deriv=1)
            @test result[1][1] ≈ analytical atol=1e-10
        end

        @testset "gradient API (broken - different interface)" begin
            # Enzyme.gradient API may not work with all function signatures
            @test_broken begin
                xqv = [0.3, 0.8, 1.2]
                g(v) = sum(itp.(v))
                grad = Enzyme.gradient(Enzyme.Reverse, g, xqv)
                all(isfinite, grad)
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # LINEAR INTERPOLATION
    # ════════════════════════════════════════════════════════════════════════

    @testset "Linear - Enzyme" begin
        x = collect(0.0:0.5:5.0)
        y_linear = 2.0 .* x .+ 1.0

        @testset "Single Interpolant - Real" begin
            itp = linear_interp(x, y_linear; extrap=:extension)

            f(xq) = itp(xq)
            result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.25))
            @test abs(result[1][1] - 2.0) < 1e-10  # Expected slope = 2
        end

        @testset "One-shot API" begin
            f(xq) = linear_interp(x, y_linear, xq)
            result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.25))
            @test abs(result[1][1] - 2.0) < 1e-10
        end

        @testset "Series Interpolant (broken - array mutation)" begin
            y1 = sin.(x)
            y2 = cos.(x)
            sitp = linear_interp(x, [y1, y2]; extrap=:extension)

            @test_broken begin
                f(xq) = sum(sitp(xq))
                result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(1.5))
                isfinite(result[1][1])
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CONSTANT INTERPOLATION
    # ════════════════════════════════════════════════════════════════════════

    # Note: Constant interpolation has Enzyme LLVM compilation issues on Julia < 1.12
    # due to phi node handling in the generated IR
    if VERSION >= v"1.12"
        @testset "Constant - Enzyme" begin
            x = collect(0.0:1.0:5.0)
            y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

            @testset "Single Interpolant" begin
                itp = constant_interp(x, y; side=:left, extrap=:extension)

                f(xq) = itp(xq)
                result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.5))
                @test abs(result[1][1]) < 1e-10  # Derivative should be 0
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # QUADRATIC INTERPOLATION
    # ════════════════════════════════════════════════════════════════════════

    @testset "Quadratic - Enzyme" begin
        x = collect(0.0:0.5:5.0)
        y_quad = x .^ 2

        @testset "Single Interpolant" begin
            itp = quadratic_interp(x, y_quad; extrap=:extension)

            f(xq) = itp(xq)
            xq = 2.25
            result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(xq))
            @test abs(result[1][1] - 2.0 * xq) < 1e-10  # d/dx(x²) = 2x
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CUBIC INTERPOLATION
    # ════════════════════════════════════════════════════════════════════════

    @testset "Cubic - Enzyme" begin
        x = collect(0.0:0.5:5.0)
        y_cubic = x .^ 3

        @testset "Single Interpolant" begin
            itp = cubic_interp(x, y_cubic; extrap=:extension)

            f(xq) = itp(xq)
            xq = 2.25
            result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(xq))
            # d/dx(x³) = 3x², approximate check
            @test abs(result[1][1] - 3.0 * xq^2) / (3.0 * xq^2) < 0.05
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # COMPLEX VALUES
    # ════════════════════════════════════════════════════════════════════════

    @testset "Complex Values - Enzyme" begin
        x = collect(0.0:0.5:5.0)
        y_complex = (2.0 + 1.0im) .* x .+ (1.0 - 1.0im)

        @testset "Linear Complex (via real)" begin
            itp = linear_interp(x, y_complex; extrap=:extension)

            f(xq) = real(itp(xq))
            result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.25))
            @test abs(result[1][1] - 2.0) < 1e-10  # Real part derivative
        end

        @testset "Linear Complex (via imag)" begin
            itp = linear_interp(x, y_complex; extrap=:extension)

            f(xq) = imag(itp(xq))
            result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.25))
            @test abs(result[1][1] - 1.0) < 1e-10  # Imag part derivative
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CROSS-VALIDATION WITH FORWARDDIFF
    # ════════════════════════════════════════════════════════════════════════

    @testset "Cross-validation with ForwardDiff" begin
        using ForwardDiff

        x = collect(0.0:0.5:5.0)
        y = sin.(x)
        itp = cubic_interp(x, y; extrap=:extension)

        @testset "Enzyme matches ForwardDiff" begin
            for xq in [0.5, 1.5, 2.5, 3.5]
                f(q) = itp(q)

                # Enzyme
                en_result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(xq))
                en_deriv = en_result[1][1]

                # ForwardDiff
                fd_deriv = ForwardDiff.derivative(f, xq)

                @test en_deriv ≈ fd_deriv atol=1e-10
            end
        end
    end

end  # testset "Enzyme AD Support"

end  # if ENZYME_AVAILABLE
