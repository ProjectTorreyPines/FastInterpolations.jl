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
# PLATFORM NOTES:
#   - Windows + Julia 1.12: Enzyme has known LLVM codegen issues causing Access
#     Violations (exit code 0xC0000005). Tests are skipped on this combination;
#     users should verify Enzyme compatibility manually if needed.
#

using Test
using FastInterpolations

# Skip Enzyme tests on Windows + Julia 1.12 due to LLVM codegen issues
const SKIP_ENZYME = Sys.iswindows() && VERSION >= v"1.12"

if SKIP_ENZYME
    @testset "Enzyme AD Support (skipped on Windows + Julia 1.12)" begin
        @test_skip "Enzyme has known LLVM issues on Windows + Julia 1.12 - verify manually if needed"
    end
else

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
                itp = linear_interp(x, y; extrap = ExtendExtrap())

                @testset "basic autodiff works" begin
                    f(xq) = itp(xq)
                    x0 = 0.73
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(x0))
                    @test isfinite(result[1][1])

                    # Cross-validate with analytical
                    analytical = itp(x0; deriv = DerivOp(1))
                    @test result[1][1] ≈ analytical atol = 1.0e-10
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
                    itp = linear_interp(x, y_linear; extrap = ExtendExtrap())

                    f(xq) = itp(xq)
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.25))
                    @test abs(result[1][1] - 2.0) < 1.0e-10  # Expected slope = 2
                end

                @testset "One-shot API" begin
                    f(xq) = linear_interp(x, y_linear, xq)
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.25))
                    @test abs(result[1][1] - 2.0) < 1.0e-10
                end

                @testset "Series Interpolant (broken - array mutation)" begin
                    y1 = sin.(x)
                    y2 = cos.(x)
                    sitp = linear_interp(x, Series(y1, y2); extrap = ExtendExtrap())

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
                        itp = constant_interp(x, y; side = LeftSide(), extrap = ExtendExtrap())

                        f(xq) = itp(xq)
                        result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.5))
                        @test abs(result[1][1]) < 1.0e-10  # Derivative should be 0
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
                    itp = quadratic_interp(x, y_quad; extrap = ExtendExtrap())

                    f(xq) = itp(xq)
                    xq = 2.25
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(xq))
                    @test abs(result[1][1] - 2.0 * xq) < 1.0e-10  # d/dx(x²) = 2x
                end
            end

            # ════════════════════════════════════════════════════════════════════════
            # CUBIC INTERPOLATION
            # ════════════════════════════════════════════════════════════════════════

            @testset "Cubic - Enzyme" begin
                x = collect(0.0:0.5:5.0)
                y_cubic = x .^ 3

                @testset "Single Interpolant" begin
                    itp = cubic_interp(x, y_cubic; extrap = ExtendExtrap())

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
                    itp = linear_interp(x, y_complex; extrap = ExtendExtrap())

                    f(xq) = real(itp(xq))
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.25))
                    @test abs(result[1][1] - 2.0) < 1.0e-10  # Real part derivative
                end

                @testset "Linear Complex (via imag)" begin
                    itp = linear_interp(x, y_complex; extrap = ExtendExtrap())

                    f(xq) = imag(itp(xq))
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(2.25))
                    @test abs(result[1][1] - 1.0) < 1.0e-10  # Imag part derivative
                end

                @testset "Quadratic Complex (via real)" begin
                    y_qc = (1.0 + 1.0im) .* x .^ 2
                    itp = quadratic_interp(x, y_qc; extrap = ExtendExtrap())
                    xq = 2.25

                    f(q) = real(itp(q))
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(xq))
                    @test abs(result[1][1] - 2.0 * xq) < 1.0e-10
                end

                @testset "Quadratic Complex (via imag)" begin
                    y_qc = (1.0 + 1.0im) .* x .^ 2
                    itp = quadratic_interp(x, y_qc; extrap = ExtendExtrap())
                    xq = 2.25

                    f(q) = imag(itp(q))
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(xq))
                    @test abs(result[1][1] - 2.0 * xq) < 1.0e-10
                end

                @testset "Cubic Complex (via real)" begin
                    y_cc = (1.0 + 1.0im) .* x .^ 3
                    itp = cubic_interp(x, y_cc; extrap = ExtendExtrap())
                    xq = 2.25

                    f(q) = real(itp(q))
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(xq))
                    expected = 3.0 * xq^2
                    @test abs(result[1][1] - expected) / expected < 0.05
                end

                @testset "Cubic Complex (via imag)" begin
                    y_cc = (1.0 + 1.0im) .* x .^ 3
                    itp = cubic_interp(x, y_cc; extrap = ExtendExtrap())
                    xq = 2.25

                    f(q) = imag(itp(q))
                    result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(xq))
                    expected = 3.0 * xq^2
                    @test abs(result[1][1] - expected) / expected < 0.05
                end
            end

            # ════════════════════════════════════════════════════════════════════════
            # CROSS-VALIDATION WITH FORWARDDIFF
            # ════════════════════════════════════════════════════════════════════════

            @testset "Cross-validation with ForwardDiff" begin
                using ForwardDiff

                x = collect(0.0:0.5:5.0)
                y = sin.(x)
                itp = cubic_interp(x, y; extrap = ExtendExtrap())

                @testset "Enzyme matches ForwardDiff" begin
                    for xq in [0.5, 1.5, 2.5, 3.5]
                        f(q) = itp(q)

                        # Enzyme
                        en_result = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(xq))
                        en_deriv = en_result[1][1]

                        # ForwardDiff
                        fd_deriv = ForwardDiff.derivative(f, xq)

                        @test en_deriv ≈ fd_deriv atol = 1.0e-10
                    end
                end
            end

            # ════════════════════════════════════════════════════════════════════════
            # CUBIC DATA-ADJOINT (∂f/∂y) via EnzymeRules
            # ════════════════════════════════════════════════════════════════════════
            # Tests the native EnzymeRules for cubic_interp(x, f, xq) that use
            # CubicAdjoint for the reverse pass. Differentiates w.r.t. DATA (f).

            @testset "Cubic data-adjoint (∂f/∂y) — Enzyme via EnzymeRules" begin
                using ForwardDiff

                x = collect(0.0:0.5:5.0)
                f_data = sin.(x)
                xq_vec = [0.75, 1.25, 2.75, 3.5, 4.25]
                y_obs = cos.(xq_vec)

                @testset "Vector query — L2 loss" begin
                    loss_enz(y, y_obs, x, xq) = sum(abs2, cubic_interp(x, y, xq) .- y_obs)
                    df = zeros(length(f_data))
                    Enzyme.autodiff(
                        Enzyme.Reverse, loss_enz, Enzyme.Active,
                        Enzyme.Duplicated(copy(f_data), df),
                        Enzyme.Const(y_obs), Enzyme.Const(x), Enzyme.Const(xq_vec)
                    )
                    g_fd = ForwardDiff.gradient(
                        y -> sum(abs2, cubic_interp(x, y, xq_vec) .- y_obs), f_data
                    )
                    @test df ≈ g_fd atol = 1.0e-10
                end

                @testset "Scalar query" begin
                    loss_scalar(y, x, xq) = cubic_interp(x, y, xq)
                    df = zeros(length(f_data))
                    Enzyme.autodiff(
                        Enzyme.Reverse, loss_scalar, Enzyme.Active,
                        Enzyme.Duplicated(copy(f_data), df),
                        Enzyme.Const(x), Enzyme.Const(1.25)
                    )
                    g_fd = ForwardDiff.gradient(y -> cubic_interp(x, y, 1.25), f_data)
                    @test df ≈ g_fd atol = 1.0e-10
                end

                @testset "Different BC — ZeroCurvBC" begin
                    loss_bc(y, x, xq) = sum(cubic_interp(x, y, xq; bc = ZeroCurvBC()))
                    df = zeros(length(f_data))
                    Enzyme.autodiff(
                        Enzyme.Reverse, loss_bc, Enzyme.Active,
                        Enzyme.Duplicated(copy(f_data), df),
                        Enzyme.Const(x), Enzyme.Const(xq_vec)
                    )
                    g_fd = ForwardDiff.gradient(
                        y -> sum(cubic_interp(x, y, xq_vec; bc = ZeroCurvBC())), f_data
                    )
                    @test df ≈ g_fd atol = 1.0e-10
                end
            end

        end  # testset "Enzyme AD Support"

    end  # if ENZYME_AVAILABLE

end  # if !SKIP_ENZYME
