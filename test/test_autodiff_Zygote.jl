# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         ZYGOTE AD TESTS                                   ║
# ║         Tests for Zygote reverse-mode AD support in interpolants          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Zygote uses source-to-source transformation for reverse-mode AD.
# Unlike ForwardDiff (forward-mode), Zygote computes gradients efficiently
# when output dimension is smaller than input dimension.
#
# KNOWN LIMITATIONS:
#   - Zygote does NOT support complex-valued output (must use real/imag)
#   - Zygote does NOT support in-place array mutation (series interpolants fail)
#   - Zygote does NOT support one-shot API (interpolant construction mutates arrays)
#   - Constant interpolation gradient returns `nothing` (not 0)
#
# SUPPORTED CASES:
#   - Pre-constructed single interpolants (Linear, Quadratic, Cubic)
#   - Real-valued output only
#   - Scalar query points
#

using Test
using FastInterpolations
using Zygote
using ForwardDiff  # For cross-validation

# Numeric finite difference helper for validation
function central_fd(f, x; eps=1e-6)
    (f(x + eps) - f(x - eps)) / (2eps)
end

@testset "Zygote AD Support" begin

    # ════════════════════════════════════════════════════════════════════════
    # LINEAR INTERPOLATION (Fully Supported)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Linear - Zygote" begin
        x = collect(0.0:0.5:5.0)
        y_linear = 2.0 .* x .+ 1.0  # y = 2x + 1, slope = 2
        y_sin = sin.(x)
        y_complex = (2.0 + 1.0im) .* x .+ (1.0 - 1.0im)

        @testset "Single Interpolant - Real" begin
            itp = linear_interp(x, y_linear; extrap=ExtendExtrap())

            @testset "gradient matches ForwardDiff" begin
                for xq in [0.25, 1.0, 2.5, 3.75, 4.5]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    fd_grad = ForwardDiff.derivative(itp, xq)
                    @test zy_grad ≈ fd_grad atol=1e-10
                    @test zy_grad ≈ 2.0 atol=1e-10  # Known slope
                end
            end

            @testset "gradient matches finite difference" begin
                for xq in [0.25, 1.5, 3.0]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    num_grad = central_fd(itp, xq)
                    @test zy_grad ≈ num_grad atol=1e-5
                end
            end

            @testset "value preserved in pullback" begin
                for xq in [0.5, 2.0, 4.0]
                    val, pb = Zygote.pullback(itp, xq)
                    @test val ≈ itp(xq) atol=1e-10
                end
            end
        end

        @testset "Single Interpolant - Complex via real/imag" begin
            # Zygote doesn't support complex output directly
            # Must use real() or imag() for scalar output
            itp = linear_interp(x, y_complex; extrap=ExtendExtrap())

            @testset "complex gradient via real/imag" begin
                xq = 2.25

                # Gradient of real part
                zy_real_grad = Zygote.gradient(q -> real(itp(q)), xq)[1]
                @test zy_real_grad ≈ 2.0 atol=1e-10  # d/dx(2x+1)

                # Gradient of imaginary part
                zy_imag_grad = Zygote.gradient(q -> imag(itp(q)), xq)[1]
                @test zy_imag_grad ≈ 1.0 atol=1e-10  # d/dx(x-1)

                # Cross-validate with ForwardDiff
                fd_grad = ForwardDiff.derivative(itp, xq)
                @test zy_real_grad ≈ real(fd_grad) atol=1e-10
                @test zy_imag_grad ≈ imag(fd_grad) atol=1e-10
            end
        end

        @testset "Vector input gradient (broadcast)" begin
            itp = linear_interp(x, y_sin; extrap=ExtendExtrap())
            xqv = [0.3, 0.8, 1.2]

            g(v) = sum(itp.(v))
            zy_grad = Zygote.gradient(g, xqv)[1]
            fd_grad = ForwardDiff.gradient(g, xqv)

            @test zy_grad ≈ fd_grad atol=1e-5
        end

        # One-shot API works for Linear!
        @testset "One-shot API - Real" begin
            xq = 2.25
            zy_grad = Zygote.gradient(q -> linear_interp(x, y_linear, q), xq)[1]
            analytical = linear_interp(x, y_linear, xq; deriv=DerivOp(1))
            @test zy_grad ≈ analytical atol=1e-10
            @test zy_grad ≈ 2.0 atol=1e-10
        end

        @testset "One-shot API - Complex" begin
            xq = 2.25
            zy_real = Zygote.gradient(q -> real(linear_interp(x, y_complex, q)), xq)[1]
            zy_imag = Zygote.gradient(q -> imag(linear_interp(x, y_complex, q)), xq)[1]
            analytical = linear_interp(x, y_complex, xq; deriv=DerivOp(1))
            @test zy_real ≈ real(analytical) atol=1e-10
            @test zy_imag ≈ imag(analytical) atol=1e-10
        end

        @testset "Series Interpolant (broken - array mutation)" begin
            y1 = sin.(x)
            y2 = cos.(x)
            sitp = linear_interp(x, Series(y1, y2); extrap=ExtendExtrap())

            @test_broken begin
                f(xq) = sum(sitp(xq))
                zy_grad = Zygote.gradient(f, 1.5)[1]
                isfinite(zy_grad)
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CONSTANT INTERPOLATION (Limited Support)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Constant - Zygote" begin
        x = collect(0.0:1.0:5.0)
        y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

        @testset "Single Interpolant - gradient is nothing (not 0)" begin
            # Zygote returns `nothing` for constant functions, not 0
            itp = constant_interp(x, y; side=LeftSide(), extrap=ExtendExtrap())

            for xq in [0.5, 1.5, 2.5, 3.5]
                zy_grad = Zygote.gradient(itp, xq)[1]
                # Zygote returns nothing for constant, treat as 0
                @test (zy_grad === nothing) || isapprox(zy_grad, 0.0; atol=1e-10)
            end
        end

        @testset "All side modes" begin
            for side_mode in [LeftSide(), RightSide(), NearestSide()]
                itp = constant_interp(x, y; side=side_mode, extrap=ExtendExtrap())
                zy_grad = Zygote.gradient(itp, 2.5)[1]
                @test (zy_grad === nothing) || isapprox(zy_grad, 0.0; atol=1e-10)
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # QUADRATIC INTERPOLATION (Single Interpolant Supported)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Quadratic - Zygote" begin
        x = collect(0.0:0.5:5.0)
        y_quad = x .^ 2
        y_complex = (1.0 + 1.0im) .* x .^ 2

        @testset "Single Interpolant - Real" begin
            itp = quadratic_interp(x, y_quad; extrap=ExtendExtrap())

            @testset "gradient matches ForwardDiff" begin
                for xq in [0.25, 1.0, 2.5, 3.75, 4.5]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    fd_grad = ForwardDiff.derivative(itp, xq)
                    @test zy_grad ≈ fd_grad atol=1e-10
                end
            end

            @testset "exact quadratic reproduction" begin
                # Quadratic spline reproduces y=x² exactly
                for xq in [0.25, 1.25, 2.25, 3.25]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    expected = 2.0 * xq  # d/dx(x²) = 2x
                    @test zy_grad ≈ expected atol=1e-10
                end
            end
        end

        @testset "Single Interpolant - Complex via real/imag" begin
            itp = quadratic_interp(x, y_complex; extrap=ExtendExtrap())
            xq = 2.25

            zy_real = Zygote.gradient(q -> real(itp(q)), xq)[1]
            zy_imag = Zygote.gradient(q -> imag(itp(q)), xq)[1]
            fd_grad = ForwardDiff.derivative(itp, xq)

            @test zy_real ≈ real(fd_grad) atol=1e-10
            @test zy_imag ≈ imag(fd_grad) atol=1e-10
            @test zy_real ≈ 2.0 * xq atol=1e-10
            @test zy_imag ≈ 2.0 * xq atol=1e-10
        end

        @testset "One-shot API (broken - array mutation)" begin
            xq = 2.25
            @test_broken begin
                zy_grad = Zygote.gradient(q -> quadratic_interp(x, y_quad, q), xq)[1]
                zy_grad ≈ 2.0 * xq
            end
        end

        @testset "Series Interpolant (broken - array mutation)" begin
            y1 = x .^ 2
            y2 = 2.0 .* x .^ 2
            sitp = quadratic_interp(x, Series(y1, y2); extrap=ExtendExtrap())

            @test_broken begin
                f(xq) = sum(sitp(xq))
                zy_grad = Zygote.gradient(f, 1.5)[1]
                isfinite(zy_grad)
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CUBIC INTERPOLATION (Single Interpolant Supported)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Cubic - Zygote" begin
        x = collect(0.0:0.5:5.0)
        y_cubic = x .^ 3
        y_sin = sin.(x)
        y_complex = (1.0 + 1.0im) .* x .^ 3

        @testset "Single Interpolant - Real" begin
            itp = cubic_interp(x, y_cubic; extrap=ExtendExtrap())

            @testset "gradient matches ForwardDiff" begin
                for xq in [0.25, 1.0, 2.5, 3.75, 4.5]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    fd_grad = ForwardDiff.derivative(itp, xq)
                    @test zy_grad ≈ fd_grad atol=1e-10
                end
            end

            @testset "cubic data accuracy" begin
                # Cubic spline approximates cubic polynomial derivatives well
                for xq in [1.25, 2.25, 3.25]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    expected = 3.0 * xq^2  # d/dx(x³) = 3x²
                    @test zy_grad ≈ expected rtol=0.05
                end
            end
        end

        @testset "Single Interpolant - Sine data" begin
            itp = cubic_interp(x, y_sin; extrap=ExtendExtrap())
            xq = 1.5
            zy_grad = Zygote.gradient(itp, xq)[1]
            @test zy_grad ≈ cos(1.5) rtol=0.01  # d/dx(sin(x)) = cos(x)
        end

        @testset "Single Interpolant - Complex via real/imag" begin
            itp = cubic_interp(x, y_complex; extrap=ExtendExtrap())
            xq = 2.25

            zy_real = Zygote.gradient(q -> real(itp(q)), xq)[1]
            zy_imag = Zygote.gradient(q -> imag(itp(q)), xq)[1]
            fd_grad = ForwardDiff.derivative(itp, xq)

            @test zy_real ≈ real(fd_grad) atol=1e-10
            @test zy_imag ≈ imag(fd_grad) atol=1e-10
            expected_deriv = 3.0 * xq^2
            @test zy_real ≈ expected_deriv rtol=0.05
            @test zy_imag ≈ expected_deriv rtol=0.05
        end

        @testset "Different BC types" begin
            for bc in [ZeroCurvBC(), ZeroSlopeBC()]
                itp = cubic_interp(x, y_cubic; bc=bc, extrap=ExtendExtrap())
                xq = 2.25
                zy_grad = Zygote.gradient(itp, xq)[1]
                fd_grad = ForwardDiff.derivative(itp, xq)
                @test zy_grad ≈ fd_grad atol=1e-10
            end
        end

        @testset "One-shot API (broken - Zygote compilation)" begin
            xq = 2.25
            @test_broken begin
                zy_grad = Zygote.gradient(q -> cubic_interp(x, y_cubic, q), xq)[1]
                isfinite(zy_grad)
            end
        end

        @testset "Series Interpolant (broken - array mutation)" begin
            y1 = sin.(x)
            y2 = cos.(x)
            sitp = cubic_interp(x, Series(y1, y2); extrap=ExtendExtrap())

            @test_broken begin
                f(xq) = sum(sitp(xq))
                zy_grad = Zygote.gradient(f, 1.5)[1]
                isfinite(zy_grad)
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CUBIC ND — Vector query gradient (issue #60: was broken by DerivOp API change)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Cubic ND - Zygote" begin
        x = range(0.0, 1.0, 20)
        y = range(0.0, 1.0, 20)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data; extrap=ExtendExtrap())

        @testset "gradient via vector query" begin
            q = [0.5, 0.5]
            grad = Zygote.gradient(itp, q)[1]
            fd_grad = ForwardDiff.gradient(itp, q)
            @test grad ≈ fd_grad atol=1e-8
        end

        @testset "withgradient via vector query (issue #60)" begin
            q = [0.5, 0.5]
            result = Zygote.withgradient(itp, q)
            @test result.val ≈ itp(q) atol=1e-12
            @test result.grad[1] ≈ ForwardDiff.gradient(itp, q) atol=1e-8
        end

        @testset "gradient via lambda with vector query" begin
            q = [0.5, 0.5]
            grad = Zygote.gradient(x -> itp(x), q)[1]
            fd_grad = ForwardDiff.gradient(itp, q)
            @test grad ≈ fd_grad atol=1e-8
        end

        @testset "gradient via tuple query" begin
            q = (0.5, 0.5)
            grad = Zygote.gradient(x -> itp(x), q)[1]
            @test grad isa Tuple
            fd_grad = ForwardDiff.gradient(itp, [q...])
            @test collect(grad) ≈ fd_grad atol=1e-8
        end

        @testset "gradient in loss function" begin
            q = [0.5, 0.5]
            loss(x) = itp(x)^2
            zy_grad = Zygote.gradient(loss, q)[1]
            fd_grad = ForwardDiff.gradient(loss, q)
            @test zy_grad ≈ fd_grad atol=1e-8
        end

        @testset "3D gradient" begin
            z = range(0.0, 1.0, 10)
            data3 = [sin(xi) * cos(yj) * zk for xi in x, yj in y, zk in z]
            itp3 = cubic_interp((x, y, z), data3; extrap=ExtendExtrap())
            q = [0.4, 0.5, 0.6]
            grad = Zygote.gradient(itp3, q)[1]
            fd_grad = ForwardDiff.gradient(itp3, q)
            @test grad ≈ fd_grad atol=1e-7
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # withgradient — 1D query-coordinate derivatives
    # ════════════════════════════════════════════════════════════════════════

    @testset "withgradient - 1D query" begin
        x = collect(0.0:0.5:5.0)

        @testset "Linear" begin
            itp = linear_interp(x, 2.0 .* x .+ 1.0; extrap=ExtendExtrap())
            xq = 2.25
            result = Zygote.withgradient(itp, xq)
            @test result.val ≈ itp(xq) atol=1e-12
            @test result.grad[1] ≈ ForwardDiff.derivative(itp, xq) atol=1e-10
        end

        @testset "Quadratic" begin
            itp = quadratic_interp(x, x .^ 2; extrap=ExtendExtrap())
            xq = 2.25
            result = Zygote.withgradient(itp, xq)
            @test result.val ≈ itp(xq) atol=1e-12
            @test result.grad[1] ≈ 2.0 * xq atol=1e-10
        end

        @testset "Cubic" begin
            itp = cubic_interp(x, sin.(x); extrap=ExtendExtrap())
            xq = 1.5
            result = Zygote.withgradient(itp, xq)
            @test result.val ≈ itp(xq) atol=1e-12
            @test result.grad[1] ≈ ForwardDiff.derivative(itp, xq) atol=1e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # GRADIENT COMPOSITION (Loss functions)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Gradient Composition" begin
        x = collect(0.0:0.5:5.0)
        y = x .^ 2

        @testset "Loss function gradient" begin
            itp = linear_interp(x, y; extrap=ExtendExtrap())

            function loss(params)
                xq = params[1]
                return itp(xq)^2
            end

            params = [2.25]
            zy_grad = Zygote.gradient(loss, params)[1]
            fd_grad = ForwardDiff.gradient(loss, params)

            @test zy_grad ≈ fd_grad atol=1e-10

            # Manual calculation: d/d(xq)[itp(xq)²] = 2 * itp(xq) * itp'(xq)
            val = itp(2.25)
            deriv = itp(2.25; deriv=DerivOp(1))
            expected_grad = 2 * val * deriv
            @test zy_grad[1] ≈ expected_grad atol=1e-10
        end

        @testset "Vector input gradient" begin
            itp = linear_interp(x, y; extrap=ExtendExtrap())
            xqv = [0.3, 0.8, 1.2]

            g(v) = sum(itp.(v))
            zy_grad = Zygote.gradient(g, xqv)[1]
            fd_grad = ForwardDiff.gradient(g, xqv)

            @test zy_grad ≈ fd_grad atol=1e-5
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # FLOAT32 SUPPORT
    # ════════════════════════════════════════════════════════════════════════

    @testset "Float32 Support" begin
        x32 = Float32.(collect(0.0:0.5:5.0))
        y32 = 2.0f0 .* x32 .+ 1.0f0

        @testset "Linear Float32" begin
            itp = linear_interp(x32, y32; extrap=ExtendExtrap())
            xq = 2.25f0
            zy_grad = Zygote.gradient(itp, xq)[1]
            @test zy_grad ≈ 2.0f0 atol=1e-5
            @test zy_grad isa Float32
        end

        @testset "Quadratic Float32" begin
            y32_quad = x32 .^ 2
            itp = quadratic_interp(x32, y32_quad; extrap=ExtendExtrap())
            xq = 2.25f0
            zy_grad = Zygote.gradient(itp, xq)[1]
            @test zy_grad ≈ 2.0f0 * xq atol=1e-4
            @test zy_grad isa Float32
        end

        @testset "Cubic Float32" begin
            y32_cubic = x32 .^ 3
            itp = cubic_interp(x32, y32_cubic; extrap=ExtendExtrap())
            xq = 2.25f0
            zy_grad = Zygote.gradient(itp, xq)[1]
            expected = 3.0f0 * xq^2
            @test zy_grad ≈ expected rtol=0.05
            @test zy_grad isa Float32
        end
    end

end  # testset "Zygote AD Support"
