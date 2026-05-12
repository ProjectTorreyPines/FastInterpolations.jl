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
function central_fd(f, x; eps = 1.0e-6)
    return (f(x + eps) - f(x - eps)) / (2eps)
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
            itp = linear_interp(x, y_linear; extrap = ExtendExtrap())

            @testset "gradient matches ForwardDiff" begin
                for xq in [0.25, 1.0, 2.5, 3.75, 4.5]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    fd_grad = ForwardDiff.derivative(itp, xq)
                    @test zy_grad ≈ fd_grad atol = 1.0e-10
                    @test zy_grad ≈ 2.0 atol = 1.0e-10  # Known slope
                end
            end

            @testset "gradient matches finite difference" begin
                for xq in [0.25, 1.5, 3.0]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    num_grad = central_fd(itp, xq)
                    @test zy_grad ≈ num_grad atol = 1.0e-5
                end
            end

            @testset "value preserved in pullback" begin
                for xq in [0.5, 2.0, 4.0]
                    val, pb = Zygote.pullback(itp, xq)
                    @test val ≈ itp(xq) atol = 1.0e-10
                end
            end
        end

        @testset "Single Interpolant - Complex via real/imag" begin
            # Zygote doesn't support complex output directly
            # Must use real() or imag() for scalar output
            itp = linear_interp(x, y_complex; extrap = ExtendExtrap())

            @testset "complex gradient via real/imag" begin
                xq = 2.25

                # Gradient of real part
                zy_real_grad = Zygote.gradient(q -> real(itp(q)), xq)[1]
                @test zy_real_grad ≈ 2.0 atol = 1.0e-10  # d/dx(2x+1)

                # Gradient of imaginary part
                zy_imag_grad = Zygote.gradient(q -> imag(itp(q)), xq)[1]
                @test zy_imag_grad ≈ 1.0 atol = 1.0e-10  # d/dx(x-1)

                # Cross-validate with ForwardDiff
                fd_grad = ForwardDiff.derivative(itp, xq)
                @test zy_real_grad ≈ real(fd_grad) atol = 1.0e-10
                @test zy_imag_grad ≈ imag(fd_grad) atol = 1.0e-10
            end
        end

        @testset "Vector input gradient (broadcast)" begin
            itp = linear_interp(x, y_sin; extrap = ExtendExtrap())
            xqv = [0.3, 0.8, 1.2]

            g(v) = sum(itp.(v))
            zy_grad = Zygote.gradient(g, xqv)[1]
            fd_grad = ForwardDiff.gradient(g, xqv)

            @test zy_grad ≈ fd_grad atol = 1.0e-5
        end

        # One-shot API works for Linear!
        @testset "One-shot API - Real" begin
            xq = 2.25
            zy_grad = Zygote.gradient(q -> linear_interp(x, y_linear, q), xq)[1]
            analytical = linear_interp(x, y_linear, xq; deriv = DerivOp(1))
            @test zy_grad ≈ analytical atol = 1.0e-10
            @test zy_grad ≈ 2.0 atol = 1.0e-10
        end

        @testset "One-shot API - Complex" begin
            xq = 2.25
            zy_real = Zygote.gradient(q -> real(linear_interp(x, y_complex, q)), xq)[1]
            zy_imag = Zygote.gradient(q -> imag(linear_interp(x, y_complex, q)), xq)[1]
            analytical = linear_interp(x, y_complex, xq; deriv = DerivOp(1))
            @test zy_real ≈ real(analytical) atol = 1.0e-10
            @test zy_imag ≈ imag(analytical) atol = 1.0e-10
        end

        @testset "Series Interpolant (broken - array mutation)" begin
            y1 = sin.(x)
            y2 = cos.(x)
            sitp = linear_interp(x, Series(y1, y2); extrap = ExtendExtrap())

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
            itp = constant_interp(x, y; side = LeftSide(), extrap = ExtendExtrap())

            for xq in [0.5, 1.5, 2.5, 3.5]
                zy_grad = Zygote.gradient(itp, xq)[1]
                # Zygote returns nothing for constant, treat as 0
                @test (zy_grad === nothing) || isapprox(zy_grad, 0.0; atol = 1.0e-10)
            end
        end

        @testset "All side modes" begin
            for side_mode in [LeftSide(), RightSide(), NearestSide()]
                itp = constant_interp(x, y; side = side_mode, extrap = ExtendExtrap())
                zy_grad = Zygote.gradient(itp, 2.5)[1]
                @test (zy_grad === nothing) || isapprox(zy_grad, 0.0; atol = 1.0e-10)
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
            itp = quadratic_interp(x, y_quad; extrap = ExtendExtrap())

            @testset "gradient matches ForwardDiff" begin
                for xq in [0.25, 1.0, 2.5, 3.75, 4.5]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    fd_grad = ForwardDiff.derivative(itp, xq)
                    @test zy_grad ≈ fd_grad atol = 1.0e-10
                end
            end

            @testset "exact quadratic reproduction" begin
                # Quadratic spline reproduces y=x² exactly
                for xq in [0.25, 1.25, 2.25, 3.25]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    expected = 2.0 * xq  # d/dx(x²) = 2x
                    @test zy_grad ≈ expected atol = 1.0e-10
                end
            end
        end

        @testset "Single Interpolant - Complex via real/imag" begin
            itp = quadratic_interp(x, y_complex; extrap = ExtendExtrap())
            xq = 2.25

            zy_real = Zygote.gradient(q -> real(itp(q)), xq)[1]
            zy_imag = Zygote.gradient(q -> imag(itp(q)), xq)[1]
            fd_grad = ForwardDiff.derivative(itp, xq)

            @test zy_real ≈ real(fd_grad) atol = 1.0e-10
            @test zy_imag ≈ imag(fd_grad) atol = 1.0e-10
            @test zy_real ≈ 2.0 * xq atol = 1.0e-10
            @test zy_imag ≈ 2.0 * xq atol = 1.0e-10
        end

        @testset "One-shot API - Real scalar (∂/∂xq via rrule)" begin
            xq = 2.25
            zy_grad = Zygote.gradient(q -> quadratic_interp(x, y_quad, q), xq)[1]
            @test zy_grad ≈ 2.0 * xq atol = 1.0e-10
        end

        @testset "Series Interpolant (broken - array mutation)" begin
            y1 = x .^ 2
            y2 = 2.0 .* x .^ 2
            sitp = quadratic_interp(x, Series(y1, y2); extrap = ExtendExtrap())

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
            itp = cubic_interp(x, y_cubic; extrap = ExtendExtrap())

            @testset "gradient matches ForwardDiff" begin
                for xq in [0.25, 1.0, 2.5, 3.75, 4.5]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    fd_grad = ForwardDiff.derivative(itp, xq)
                    @test zy_grad ≈ fd_grad atol = 1.0e-10
                end
            end

            @testset "cubic data accuracy" begin
                # Cubic spline approximates cubic polynomial derivatives well
                for xq in [1.25, 2.25, 3.25]
                    zy_grad = Zygote.gradient(itp, xq)[1]
                    expected = 3.0 * xq^2  # d/dx(x³) = 3x²
                    @test zy_grad ≈ expected rtol = 0.05
                end
            end
        end

        @testset "Single Interpolant - Sine data" begin
            itp = cubic_interp(x, y_sin; extrap = ExtendExtrap())
            xq = 1.5
            zy_grad = Zygote.gradient(itp, xq)[1]
            @test zy_grad ≈ cos(1.5) rtol = 0.01  # d/dx(sin(x)) = cos(x)
        end

        @testset "Single Interpolant - Complex via real/imag" begin
            itp = cubic_interp(x, y_complex; extrap = ExtendExtrap())
            xq = 2.25

            zy_real = Zygote.gradient(q -> real(itp(q)), xq)[1]
            zy_imag = Zygote.gradient(q -> imag(itp(q)), xq)[1]
            fd_grad = ForwardDiff.derivative(itp, xq)

            @test zy_real ≈ real(fd_grad) atol = 1.0e-10
            @test zy_imag ≈ imag(fd_grad) atol = 1.0e-10
            expected_deriv = 3.0 * xq^2
            @test zy_real ≈ expected_deriv rtol = 0.05
            @test zy_imag ≈ expected_deriv rtol = 0.05
        end

        @testset "Different BC types" begin
            for bc in [ZeroCurvBC(), ZeroSlopeBC()]
                itp = cubic_interp(x, y_cubic; bc = bc, extrap = ExtendExtrap())
                xq = 2.25
                zy_grad = Zygote.gradient(itp, xq)[1]
                fd_grad = ForwardDiff.derivative(itp, xq)
                @test zy_grad ≈ fd_grad atol = 1.0e-10
            end
        end

        @testset "One-shot API — scalar xq via rrule" begin
            xq = 2.25
            zy_grad = Zygote.gradient(q -> cubic_interp(x, y_cubic, q), xq)[1]
            @test isfinite(zy_grad)
            # Analytic: derivative of x³ interpolant at xq = 3xq²
            @test zy_grad ≈ 3.0 * xq^2 rtol = 0.05
        end

        @testset "Series Interpolant (broken - array mutation)" begin
            y1 = sin.(x)
            y2 = cos.(x)
            sitp = cubic_interp(x, Series(y1, y2); extrap = ExtendExtrap())

            @test_broken begin
                f(xq) = sum(sitp(xq))
                zy_grad = Zygote.gradient(f, 1.5)[1]
                isfinite(zy_grad)
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CUBIC DATA-ADJOINT (∂f/∂y) via CubicAdjoint rrule
    # ════════════════════════════════════════════════════════════════════════
    # Tests the ChainRulesCore rrule for cubic_interp(x, f, xq) that uses
    # CubicAdjoint for the pullback. This differentiates w.r.t. DATA (f),
    # not w.r.t. query coordinates (xq).

    @testset "Cubic data-adjoint (∂f/∂y) — Zygote via rrule" begin
        x = collect(0.0:0.5:5.0)
        f_data = sin.(x)
        xq_vec = [0.75, 1.25, 2.75, 3.5, 4.25]
        y_obs = cos.(xq_vec)

        @testset "Scalar query — ∂f/∂y" begin
            g_zy = Zygote.gradient(y -> cubic_interp(x, y, 1.25), f_data)[1]
            g_fd = ForwardDiff.gradient(y -> cubic_interp(x, y, 1.25), f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "Vector query — ∂f/∂y" begin
            g_zy = Zygote.gradient(y -> sum(cubic_interp(x, y, xq_vec)), f_data)[1]
            g_fd = ForwardDiff.gradient(y -> sum(cubic_interp(x, y, xq_vec)), f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "L2 loss function" begin
            loss(y) = sum(abs2, cubic_interp(x, y, xq_vec) .- y_obs)
            g_zy = Zygote.gradient(loss, f_data)[1]
            g_fd = ForwardDiff.gradient(loss, f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "Different BC types" begin
            for (name, bc) in [("ZeroCurvBC", ZeroCurvBC()), ("ZeroSlopeBC", ZeroSlopeBC())]
                g_zy = Zygote.gradient(y -> sum(cubic_interp(x, y, xq_vec; bc = bc)), f_data)[1]
                g_fd = ForwardDiff.gradient(y -> sum(cubic_interp(x, y, xq_vec; bc = bc)), f_data)
                @test g_zy ≈ g_fd atol = 1.0e-10
            end
        end

        @testset "Float32" begin
            x32 = Float32.(x)
            f32 = Float32.(f_data)
            xq32 = Float32.(xq_vec)
            g_zy = Zygote.gradient(y -> sum(cubic_interp(x32, y, xq32)), f32)[1]
            g_fd = ForwardDiff.gradient(y -> sum(cubic_interp(x32, y, xq32)), f32)
            @test g_zy ≈ g_fd atol = 1.0e-4
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CUBIC ND DATA-ADJOINT (∂/∂data) via CubicAdjointND rrule
    # ════════════════════════════════════════════════════════════════════════
    # Tests the ChainRulesCore rrule for cubic_interp(grids, data, queries)
    # that uses CubicAdjointND for the pullback. Differentiates w.r.t. DATA,
    # not query coordinates. Cross-validates against CubicAdjointND directly.

    @testset "Cubic ND data-adjoint (∂/∂data) — Zygote via rrule" begin
        nx, ny = 15, 12
        x = range(0.0, 1.0, nx)
        y = range(0.0, 1.0, ny)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
        n_query = 20
        xq = sort(rand(n_query)) .* 0.96 .+ 0.02
        yq = sort(rand(n_query)) .* 0.96 .+ 0.02

        @testset "SoA batch — CubicFit (default)" begin
            g_zy = Zygote.gradient(d -> sum(cubic_interp((x, y), d, (xq, yq))), data)[1]
            adj = cubic_adjoint((x, y), (xq, yq))
            g_adj = adj(ones(n_query))
            @test g_zy ≈ g_adj atol = 1.0e-10
        end

        @testset "SoA batch — ZeroCurvBC" begin
            bc = ZeroCurvBC()
            g_zy = Zygote.gradient(d -> sum(cubic_interp((x, y), d, (xq, yq); bc = bc)), data)[1]
            adj = cubic_adjoint((x, y), (xq, yq); bc = bc)
            g_adj = adj(ones(n_query))
            @test g_zy ≈ g_adj atol = 1.0e-10
        end

        @testset "SoA batch — PeriodicBC (inclusive)" begin
            x_p = range(0.0, 2π, nx)
            y_p = range(0.0, 2π, ny)
            data_p = [sin(xi) + cos(yj) for xi in x_p, yj in y_p]
            # Enforce exact endpoint equality for inclusive periodic validation
            data_p[end, :] .= data_p[1, :]
            data_p[:, end] .= data_p[:, 1]
            data_p[end, end] = data_p[1, 1]
            bc = PeriodicBC()
            xq_p = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
            yq_p = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
            g_zy = Zygote.gradient(d -> sum(cubic_interp((x_p, y_p), d, (xq_p, yq_p); bc = bc)), data_p)[1]
            adj = cubic_adjoint((x_p, y_p), (xq_p, yq_p); bc = bc)
            g_adj = adj(ones(n_query))
            @test g_zy ≈ g_adj atol = 1.0e-10
        end

        @testset "SoA batch — deriv=DerivOp(1)" begin
            g_zy = Zygote.gradient(
                d -> sum(cubic_interp((x, y), d, (xq, yq); deriv = DerivOp(1))), data
            )[1]
            adj = cubic_adjoint((x, y), (xq, yq))
            g_adj = adj(ones(n_query); deriv = DerivOp(1))
            @test g_zy ≈ g_adj atol = 1.0e-10
        end

        @testset "SoA batch — per-axis deriv (DerivOp(1), EvalValue())" begin
            g_zy = Zygote.gradient(
                d -> sum(cubic_interp((x, y), d, (xq, yq); deriv = (DerivOp(1), EvalValue()))), data
            )[1]
            adj = cubic_adjoint((x, y), (xq, yq))
            g_adj = adj(ones(n_query); deriv = (DerivOp(1), EvalValue()))
            @test g_zy ≈ g_adj atol = 1.0e-10
        end

        @testset "Single point — default" begin
            g_zy = Zygote.gradient(d -> cubic_interp((x, y), d, (0.5, 0.5)), data)[1]
            adj = cubic_adjoint((x, y), ([0.5], [0.5]))
            g_adj = adj([1.0])
            @test g_zy ≈ g_adj atol = 1.0e-10
        end

        @testset "L2 loss function" begin
            y_obs = randn(n_query)
            loss(d) = sum(abs2, cubic_interp((x, y), d, (xq, yq)) .- y_obs)
            g_zy = Zygote.gradient(loss, data)[1]
            # Manual: ∂L/∂data = 2 * Wᵀ * (W*data - y_obs)
            residual = cubic_interp((x, y), data, (xq, yq)) .- y_obs
            adj = cubic_adjoint((x, y), (xq, yq))
            g_adj = adj(2.0 .* residual)
            @test g_zy ≈ g_adj atol = 1.0e-10
        end

        @testset "N=3" begin
            nz = 8
            z = range(0.0, 1.0, nz)
            data3 = randn(nx, ny, nz)
            zq = sort(rand(n_query)) .* 0.96 .+ 0.02
            g_zy = Zygote.gradient(d -> sum(cubic_interp((x, y, z), d, (xq, yq, zq))), data3)[1]
            adj = cubic_adjoint((x, y, z), (xq, yq, zq))
            g_adj = adj(ones(n_query))
            @test g_zy ≈ g_adj atol = 1.0e-10
        end

        @testset "Float32" begin
            x32, y32 = Float32.(x), Float32.(y)
            data32 = Float32.(data)
            xq32, yq32 = Float32.(xq), Float32.(yq)
            g_zy = Zygote.gradient(d -> sum(cubic_interp((x32, y32), d, (xq32, yq32))), data32)[1]
            @test g_zy isa Matrix{Float32}
            adj = cubic_adjoint((x32, y32), (xq32, yq32))
            g_adj = adj(ones(Float32, n_query))
            @test g_zy ≈ g_adj atol = 1.0f-4
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Issue #60 — topology optimization patterns (∂/∂data via one-shot)
    # ════════════════════════════════════════════════════════════════════════
    # GitHub issue #60 requests Zygote.withgradient(f, data, x0) support.
    # The interpolant API (itp = cubic_interp(grids, data); itp(x0)) only
    # provides ∂/∂x0 via rrule. For ∂/∂data, use the one-shot API:
    #   cubic_interp(grids, data, query; deriv=...)
    # which has an rrule backed by CubicAdjointND.

    @testset "Issue #60 — topology optimization (∂/∂data via one-shot)" begin
        Nx = Ny = 5
        x = range(0, 1, length = Nx)
        y = range(0, 1, length = Ny)
        data = [0.1xi + 0.2yj + sin(xi) * cos(yj) for xi in x, yj in y]
        x0 = (0.2, 0.3)

        @testset "f0: withgradient — basic value + data gradient" begin
            # Issue example 1: f0(data) = cubic_interp((x,y), data)(x0)
            f0(d) = cubic_interp((x, y), d, x0)

            result = Zygote.withgradient(f0, data)
            @test result.val ≈ cubic_interp((x, y), data, x0)
            @test result.grad[1] !== nothing

            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            @test result.grad[1] ≈ adj([1.0]) atol = 1.0e-10
        end

        @testset "f1: ∂/∂data of ‖∇f(x0)‖² — gradient-of-gradient" begin
            # Issue example 2: f1(data) = sum(abs2, gradient(itp, x0))
            # One-shot equivalent with per-axis DerivOp:
            function f1(d)
                fx = cubic_interp((x, y), d, x0; deriv = (DerivOp(1), EvalValue()))
                fy = cubic_interp((x, y), d, x0; deriv = (EvalValue(), DerivOp(1)))
                return fx^2 + fy^2
            end

            result = Zygote.withgradient(f1, data)
            @test isfinite(result.val)
            @test size(result.grad[1]) == size(data)

            # Manual: ∂L/∂data = 2fx·(∂fx/∂data) + 2fy·(∂fy/∂data)
            fx = cubic_interp((x, y), data, x0; deriv = (DerivOp(1), EvalValue()))
            fy = cubic_interp((x, y), data, x0; deriv = (EvalValue(), DerivOp(1)))
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            g_expected = adj([2fx]; deriv = (DerivOp(1), EvalValue())) .+
                adj([2fy]; deriv = (EvalValue(), DerivOp(1)))
            @test result.grad[1] ≈ g_expected atol = 1.0e-10
        end

        @testset "f2: ∂/∂data of ‖H(f)(x0)‖²_F — gradient-of-hessian" begin
            # Issue example 3: f2(data) = sum(abs2, hessian(itp, x0))
            # One-shot with 2nd-order per-axis DerivOp:
            function f2(d)
                fxx = cubic_interp((x, y), d, x0; deriv = (DerivOp(2), EvalValue()))
                fxy = cubic_interp((x, y), d, x0; deriv = (DerivOp(1), DerivOp(1)))
                fyy = cubic_interp((x, y), d, x0; deriv = (EvalValue(), DerivOp(2)))
                return fxx^2 + 2 * fxy^2 + fyy^2  # ‖H‖²_F for symmetric H
            end

            result = Zygote.withgradient(f2, data)
            @test isfinite(result.val)
            @test size(result.grad[1]) == size(data)

            fxx = cubic_interp((x, y), data, x0; deriv = (DerivOp(2), EvalValue()))
            fxy = cubic_interp((x, y), data, x0; deriv = (DerivOp(1), DerivOp(1)))
            fyy = cubic_interp((x, y), data, x0; deriv = (EvalValue(), DerivOp(2)))
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            g_expected = adj([2fxx]; deriv = (DerivOp(2), EvalValue())) .+
                adj([4fxy]; deriv = (DerivOp(1), DerivOp(1))) .+
                adj([2fyy]; deriv = (EvalValue(), DerivOp(2)))
            @test result.grad[1] ≈ g_expected atol = 1.0e-10
        end

        @testset "batch: withgradient L2 loss over query grid" begin
            # Typical topology optimization: minimize residual over many points
            xq = [0.2, 0.4, 0.6, 0.8]
            yq = [0.3, 0.5, 0.3, 0.7]
            target = [0.5, -0.3, 0.1, 0.8]

            loss(d) = sum(abs2, cubic_interp((x, y), d, (xq, yq)) .- target)

            result = Zygote.withgradient(loss, data)
            @test result.val ≈ loss(data)
            @test isfinite(result.val)

            residual = cubic_interp((x, y), data, (xq, yq)) .- target
            adj = cubic_adjoint((x, y), (xq, yq))
            @test result.grad[1] ≈ adj(2.0 .* residual) atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CUBIC ND — Vector query gradient (issue #60: was broken by DerivOp API change)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Cubic ND - Zygote" begin
        x = range(0.0, 1.0, 20)
        y = range(0.0, 1.0, 20)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data; extrap = ExtendExtrap())

        @testset "gradient via vector query" begin
            q = [0.5, 0.5]
            grad = Zygote.gradient(itp, q)[1]
            fd_grad = ForwardDiff.gradient(itp, q)
            @test grad ≈ fd_grad atol = 1.0e-8
        end

        @testset "withgradient via vector query (issue #60)" begin
            q = [0.5, 0.5]
            result = Zygote.withgradient(itp, q)
            @test result.val ≈ itp(q) atol = 1.0e-12
            @test result.grad[1] ≈ ForwardDiff.gradient(itp, q) atol = 1.0e-8
        end

        @testset "gradient via lambda with vector query" begin
            q = [0.5, 0.5]
            grad = Zygote.gradient(x -> itp(x), q)[1]
            fd_grad = ForwardDiff.gradient(itp, q)
            @test grad ≈ fd_grad atol = 1.0e-8
        end

        @testset "gradient via tuple query" begin
            q = (0.5, 0.5)
            grad = Zygote.gradient(x -> itp(x), q)[1]
            @test grad isa Tuple
            fd_grad = ForwardDiff.gradient(itp, [q...])
            @test collect(grad) ≈ fd_grad atol = 1.0e-8
        end

        @testset "gradient in loss function" begin
            q = [0.5, 0.5]
            loss(x) = itp(x)^2
            zy_grad = Zygote.gradient(loss, q)[1]
            fd_grad = ForwardDiff.gradient(loss, q)
            @test zy_grad ≈ fd_grad atol = 1.0e-8
        end

        @testset "3D gradient" begin
            z = range(0.0, 1.0, 10)
            data3 = [sin(xi) * cos(yj) * zk for xi in x, yj in y, zk in z]
            itp3 = cubic_interp((x, y, z), data3; extrap = ExtendExtrap())
            q = [0.4, 0.5, 0.6]
            grad = Zygote.gradient(itp3, q)[1]
            fd_grad = ForwardDiff.gradient(itp3, q)
            @test grad ≈ fd_grad atol = 1.0e-7
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # withgradient — 1D query-coordinate derivatives
    # ════════════════════════════════════════════════════════════════════════

    @testset "withgradient - 1D query" begin
        x = collect(0.0:0.5:5.0)

        @testset "Linear" begin
            itp = linear_interp(x, 2.0 .* x .+ 1.0; extrap = ExtendExtrap())
            xq = 2.25
            result = Zygote.withgradient(itp, xq)
            @test result.val ≈ itp(xq) atol = 1.0e-12
            @test result.grad[1] ≈ ForwardDiff.derivative(itp, xq) atol = 1.0e-10
        end

        @testset "Quadratic" begin
            itp = quadratic_interp(x, x .^ 2; extrap = ExtendExtrap())
            xq = 2.25
            result = Zygote.withgradient(itp, xq)
            @test result.val ≈ itp(xq) atol = 1.0e-12
            @test result.grad[1] ≈ 2.0 * xq atol = 1.0e-10
        end

        @testset "Cubic" begin
            itp = cubic_interp(x, sin.(x); extrap = ExtendExtrap())
            xq = 1.5
            result = Zygote.withgradient(itp, xq)
            @test result.val ≈ itp(xq) atol = 1.0e-12
            @test result.grad[1] ≈ ForwardDiff.derivative(itp, xq) atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # GRADIENT COMPOSITION (Loss functions)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Gradient Composition" begin
        x = collect(0.0:0.5:5.0)
        y = x .^ 2

        @testset "Loss function gradient" begin
            itp = linear_interp(x, y; extrap = ExtendExtrap())

            function loss(params)
                xq = params[1]
                return itp(xq)^2
            end

            params = [2.25]
            zy_grad = Zygote.gradient(loss, params)[1]
            fd_grad = ForwardDiff.gradient(loss, params)

            @test zy_grad ≈ fd_grad atol = 1.0e-10

            # Manual calculation: d/d(xq)[itp(xq)²] = 2 * itp(xq) * itp'(xq)
            val = itp(2.25)
            deriv = itp(2.25; deriv = DerivOp(1))
            expected_grad = 2 * val * deriv
            @test zy_grad[1] ≈ expected_grad atol = 1.0e-10
        end

        @testset "Vector input gradient" begin
            itp = linear_interp(x, y; extrap = ExtendExtrap())
            xqv = [0.3, 0.8, 1.2]

            g(v) = sum(itp.(v))
            zy_grad = Zygote.gradient(g, xqv)[1]
            fd_grad = ForwardDiff.gradient(g, xqv)

            @test zy_grad ≈ fd_grad atol = 1.0e-5
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # FLOAT32 SUPPORT
    # ════════════════════════════════════════════════════════════════════════

    @testset "Float32 Support" begin
        x32 = Float32.(collect(0.0:0.5:5.0))
        y32 = 2.0f0 .* x32 .+ 1.0f0

        @testset "Linear Float32" begin
            itp = linear_interp(x32, y32; extrap = ExtendExtrap())
            xq = 2.25f0
            zy_grad = Zygote.gradient(itp, xq)[1]
            @test zy_grad ≈ 2.0f0 atol = 1.0e-5
            @test zy_grad isa Float32
        end

        @testset "Quadratic Float32" begin
            y32_quad = x32 .^ 2
            itp = quadratic_interp(x32, y32_quad; extrap = ExtendExtrap())
            xq = 2.25f0
            zy_grad = Zygote.gradient(itp, xq)[1]
            @test zy_grad ≈ 2.0f0 * xq atol = 1.0e-4
            @test zy_grad isa Float32
        end

        @testset "Cubic Float32" begin
            y32_cubic = x32 .^ 3
            itp = cubic_interp(x32, y32_cubic; extrap = ExtendExtrap())
            xq = 2.25f0
            zy_grad = Zygote.gradient(itp, xq)[1]
            expected = 3.0f0 * xq^2
            @test zy_grad ≈ expected rtol = 0.05
            @test zy_grad isa Float32
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CUBIC ND — Interpolant API ∂/∂data (constructor + eval/gradient/hessian/laplacian rrules)
    # ════════════════════════════════════════════════════════════════════════
    # Tests the natural API pattern:
    #   itp = cubic_interp(grids, data)
    #   loss = f(itp(x0))
    #   Zygote.gradient(data -> f(cubic_interp(grids, data)(x0)), data)

    @testset "Cubic ND interpolant API — ∂/∂data via rrule chain" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 2.0, 15)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        x0 = (0.7, 0.9)

        # ── eval: ∂/∂data of itp(x0) ──
        @testset "eval — sum loss" begin
            function f_eval(d)
                itp = cubic_interp((x, y), d)
                return itp(x0)
            end
            result = Zygote.withgradient(f_eval, data)
            @test result.val ≈ f_eval(data)

            # Cross-validate: same as one-shot adjoint with Δy=1
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([1.0])
            @test result.grad[1] ≈ expected atol = 1.0e-10
        end

        # ── eval: ∂/∂data of L2 loss ──
        @testset "eval — L2 loss" begin
            target_val = 0.5
            function f_eval_l2(d)
                itp = cubic_interp((x, y), d)
                return (itp(x0) - target_val)^2
            end
            result = Zygote.withgradient(f_eval_l2, data)
            @test result.val ≈ f_eval_l2(data)

            # Manual: ∂L/∂data = 2(y - target) * Wᵀ [1]
            y_val = cubic_interp((x, y), data, x0)
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([2.0 * (y_val - target_val)])
            @test result.grad[1] ≈ expected atol = 1.0e-10
        end

        # ── gradient: ∂/∂data of ‖∇f(x0)‖² ──
        @testset "gradient — ‖∇f‖² loss" begin
            function f_grad_norm(d)
                itp = cubic_interp((x, y), d)
                g = FastInterpolations.gradient(itp, x0)
                return g[1]^2 + g[2]^2
            end
            result = Zygote.withgradient(f_grad_norm, data)
            @test result.val ≈ f_grad_norm(data)

            # Cross-validate with one-shot adjoint
            fx = cubic_interp((x, y), data, x0; deriv = (DerivOp(1), EvalValue()))
            fy = cubic_interp((x, y), data, x0; deriv = (EvalValue(), DerivOp(1)))
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([2fx]; deriv = (DerivOp(1), EvalValue())) .+
                adj([2fy]; deriv = (EvalValue(), DerivOp(1)))
            @test result.grad[1] ≈ expected atol = 1.0e-10
        end

        # ── gradient: ∂/∂data of single partial ──
        @testset "gradient — single partial ∂f/∂x" begin
            function f_partial_x(d)
                itp = cubic_interp((x, y), d)
                g = FastInterpolations.gradient(itp, x0)
                return g[1]  # ∂f/∂x only
            end
            result = Zygote.withgradient(f_partial_x, data)

            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([1.0]; deriv = (DerivOp(1), EvalValue()))
            @test result.grad[1] ≈ expected atol = 1.0e-10
        end

        # ── hessian: ∂/∂data of ‖H(f)(x0)‖²_F ──
        @testset "hessian — Frobenius norm loss" begin
            function f_hess_frob(d)
                itp = cubic_interp((x, y), d)
                H = FastInterpolations.hessian(itp, x0)
                return sum(abs2, H)
            end
            result = Zygote.withgradient(f_hess_frob, data)
            @test result.val ≈ f_hess_frob(data)

            # Cross-validate with one-shot adjoint
            H = FastInterpolations.hessian(cubic_interp((x, y), data), x0)
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            # ∂/∂data of sum(H.^2) = sum_ij 2*H[i,j] * adj(1; deriv=ops_ij)
            expected = zeros(size(data))
            # Diagonal
            expected .+= adj([2H[1, 1]]; deriv = (DerivOp(2), EvalValue()))
            expected .+= adj([2H[2, 2]]; deriv = (EvalValue(), DerivOp(2)))
            # Off-diagonal (symmetry: H[1,2]=H[2,1], cotangent = 2H[1,2] + 2H[2,1] = 4H[1,2])
            expected .+= adj([4H[1, 2]]; deriv = (DerivOp(1), DerivOp(1)))
            @test result.grad[1] ≈ expected atol = 1.0e-9
        end

        # ── hessian: ∂/∂data of single element ──
        @testset "hessian — single element ∂²f/∂x²" begin
            function f_hess_xx(d)
                itp = cubic_interp((x, y), d)
                H = FastInterpolations.hessian(itp, x0)
                return H[1, 1]
            end
            result = Zygote.withgradient(f_hess_xx, data)

            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([1.0]; deriv = (DerivOp(2), EvalValue()))
            @test result.grad[1] ≈ expected atol = 1.0e-10
        end

        # ── laplacian: ∂/∂data of ∇²f(x0) ──
        @testset "laplacian — ∂/∂data of ∇²f" begin
            function f_lap(d)
                itp = cubic_interp((x, y), d)
                return FastInterpolations.laplacian(itp, x0)
            end
            result = Zygote.withgradient(f_lap, data)
            @test result.val ≈ f_lap(data)

            # ∇²f = ∂²f/∂x² + ∂²f/∂y², so ∂/∂data = adj(1; d²/dx²) + adj(1; d²/dy²)
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([1.0]; deriv = (DerivOp(2), EvalValue())) .+
                adj([1.0]; deriv = (EvalValue(), DerivOp(2)))
            @test result.grad[1] ≈ expected atol = 1.0e-10
        end

        # ── laplacian: ∂/∂data of L2 loss on ∇²f ──
        @testset "laplacian — (∇²f)² loss" begin
            function f_lap_sq(d)
                itp = cubic_interp((x, y), d)
                lap = FastInterpolations.laplacian(itp, x0)
                return lap^2
            end
            result = Zygote.withgradient(f_lap_sq, data)
            @test result.val ≈ f_lap_sq(data)

            lap_val = FastInterpolations.laplacian(cubic_interp((x, y), data), x0)
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([2lap_val]; deriv = (DerivOp(2), EvalValue())) .+
                adj([2lap_val]; deriv = (EvalValue(), DerivOp(2)))
            @test result.grad[1] ≈ expected atol = 1.0e-9
        end

        # ── 3D test ──
        @testset "3D eval — ∂/∂data" begin
            z = range(0.0, 2.0, 10)
            data3 = [sin(xi) * cos(yj) * (1.0 + zk) for xi in x, yj in y, zk in z]
            x0_3d = (0.7, 0.9, 0.5)

            function f_3d(d)
                itp = cubic_interp((x, y, z), d)
                return itp(x0_3d)
            end
            result = Zygote.withgradient(f_3d, data3)
            @test result.val ≈ f_3d(data3)

            adj = cubic_adjoint((x, y, z), ([x0_3d[1]], [x0_3d[2]], [x0_3d[3]]))
            expected = adj([1.0])
            @test result.grad[1] ≈ expected atol = 1.0e-10
        end

        # ── eval: also returns ∂/∂query ──
        @testset "eval — ∂/∂query still works" begin
            itp = cubic_interp((x, y), data)
            q = (0.7, 0.9)
            grad = Zygote.gradient(x -> itp(x), q)[1]
            fd_grad = ForwardDiff.gradient(itp, [q...])
            @test collect(grad) ≈ fd_grad atol = 1.0e-8
        end

        # ── eval: ∂/∂data via vector query matches tuple query ──
        @testset "eval — ∂/∂data via vector query" begin
            x0_vec = [x0[1], x0[2]]
            function f_eval_vec(d)
                itp = cubic_interp((x, y), d)
                return itp(x0_vec)
            end
            result_vec = Zygote.withgradient(f_eval_vec, data)

            # Must match tuple query result
            function f_eval_tuple(d)
                itp = cubic_interp((x, y), d)
                return itp(x0)
            end
            result_tuple = Zygote.withgradient(f_eval_tuple, data)
            @test result_vec.grad[1] ≈ result_tuple.grad[1] atol = 1.0e-12

            # Cross-validate with one-shot adjoint
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([1.0])
            @test result_vec.grad[1] ≈ expected atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # value_gradient — Zygote rrule tests
    # ════════════════════════════════════════════════════════════════════════

    @testset "value_gradient — Zygote rrule" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 2.0, 15)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        x0 = (0.7, 0.9)

        # ── ∂/∂query: extract value from value_gradient ──
        @testset "∂/∂query — value part" begin
            itp = cubic_interp((x, y), data)
            grad = Zygote.gradient(q -> FastInterpolations.value_gradient(itp, q)[1], x0)[1]
            fd_grad = ForwardDiff.gradient(itp, [x0...])
            @test collect(grad) ≈ fd_grad atol = 1.0e-8
        end

        # ── ∂/∂query: extract gradient norm from value_gradient ──
        @testset "∂/∂query — gradient norm" begin
            itp = cubic_interp((x, y), data)
            grad = Zygote.gradient(q -> sum(abs2, FastInterpolations.value_gradient(itp, q)[2]), x0)[1]
            # Cross-validate: same as ∂/∂query of sum(abs2, gradient(itp, q))
            grad_ref = Zygote.gradient(q -> sum(abs2, FastInterpolations.gradient(itp, q)), x0)[1]
            @test collect(grad) ≈ collect(grad_ref) atol = 1.0e-8
        end

        # ── ∂/∂data: value part via CubicND rrule ──
        @testset "∂/∂data — value" begin
            function f_vg_val(d)
                itp = cubic_interp((x, y), d)
                val, _ = FastInterpolations.value_gradient(itp, x0)
                return val
            end
            result = Zygote.withgradient(f_vg_val, data)
            @test result.val ≈ f_vg_val(data)

            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([1.0])
            @test result.grad[1] ≈ expected atol = 1.0e-10
        end

        # ── ∂/∂data: gradient norm via CubicND rrule ──
        @testset "∂/∂data — ‖∇f‖²" begin
            function f_vg_gnorm(d)
                itp = cubic_interp((x, y), d)
                _, g = FastInterpolations.value_gradient(itp, x0)
                return g[1]^2 + g[2]^2
            end
            result = Zygote.withgradient(f_vg_gnorm, data)

            # Cross-validate with existing gradient rrule path
            function f_g_gnorm(d)
                itp = cubic_interp((x, y), d)
                g = FastInterpolations.gradient(itp, x0)
                return g[1]^2 + g[2]^2
            end
            result_ref = Zygote.withgradient(f_g_gnorm, data)
            @test result.grad[1] ≈ result_ref.grad[1] atol = 1.0e-10
        end

        # ── ∂/∂data: combined loss v² + ‖∇f‖² ──
        @testset "∂/∂data — combined v² + ‖∇f‖²" begin
            function f_combined(d)
                itp = cubic_interp((x, y), d)
                val, g = FastInterpolations.value_gradient(itp, x0)
                return val^2 + g[1]^2 + g[2]^2
            end
            result = Zygote.withgradient(f_combined, data)
            @test isfinite(result.val)

            # Manual: ∂L/∂data = 2v·adj(1) + 2gx·adj(1;dx) + 2gy·adj(1;dy)
            itp = cubic_interp((x, y), data)
            v, g = FastInterpolations.value_gradient(itp, x0)
            adj = cubic_adjoint((x, y), ([x0[1]], [x0[2]]))
            expected = adj([2v]) .+
                adj([2g[1]]; deriv = (DerivOp(1), EvalValue())) .+
                adj([2g[2]]; deriv = (EvalValue(), DerivOp(1)))
            @test result.grad[1] ≈ expected atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # value_gradient — non-Cubic (AbstractInterpolantND rrule path)
    # ════════════════════════════════════════════════════════════════════════

    @testset "value_gradient — non-Cubic ∂/∂query" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 2.0, 15)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        x0 = (0.7, 0.9)

        # Linear: ∂/∂query via value part
        @testset "LinearInterpolantND — value" begin
            itp = linear_interp((x, y), data)
            grad = Zygote.gradient(q -> FastInterpolations.value_gradient(itp, q)[1], x0)[1]
            fd_grad = ForwardDiff.gradient(q -> itp(Tuple(q)), [x0...])
            @test collect(grad) ≈ fd_grad atol = 1.0e-8
        end

        # Quadratic: ∂/∂query via gradient norm
        @testset "QuadraticInterpolantND — gradient norm" begin
            itp = quadratic_interp((x, y), data)
            grad = Zygote.gradient(q -> sum(abs2, FastInterpolations.value_gradient(itp, q)[2]), x0)[1]
            grad_ref = Zygote.gradient(q -> sum(abs2, FastInterpolations.gradient(itp, q)), x0)[1]
            @test collect(grad) ≈ collect(grad_ref) atol = 1.0e-8
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # LINEAR DATA-ADJOINT (∂f/∂y) via LinearAdjoint rrule
    # ════════════════════════════════════════════════════════════════════════

    @testset "Linear data-adjoint (∂f/∂y) — Zygote via rrule" begin
        x = collect(0.0:0.5:5.0)
        f_data = 2.0 .* x .+ 1.0
        xq_vec = [0.75, 1.25, 2.75, 3.5, 4.25]
        y_obs = 2.0 .* xq_vec .+ 0.5  # different from f_data values

        @testset "Scalar query — ∂f/∂y" begin
            g_zy = Zygote.gradient(y -> linear_interp(x, y, 1.25), f_data)[1]
            g_fd = ForwardDiff.gradient(y -> linear_interp(x, y, 1.25), f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "Vector query — ∂f/∂y" begin
            g_zy = Zygote.gradient(y -> sum(linear_interp(x, y, xq_vec)), f_data)[1]
            g_fd = ForwardDiff.gradient(y -> sum(linear_interp(x, y, xq_vec)), f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "L2 loss function" begin
            loss(y) = sum(abs2, linear_interp(x, y, xq_vec) .- y_obs)
            g_zy = Zygote.gradient(loss, f_data)[1]
            g_fd = ForwardDiff.gradient(loss, f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "Float32" begin
            x32 = Float32.(x)
            f32 = Float32.(f_data)
            xq32 = Float32.(xq_vec)
            g_zy = Zygote.gradient(y -> sum(linear_interp(x32, y, xq32)), f32)[1]
            g_fd = ForwardDiff.gradient(y -> sum(linear_interp(x32, y, xq32)), f32)
            @test g_zy ≈ g_fd atol = 1.0e-4
        end
    end

    # Direct 1D one-shot rrule under `bc=PeriodicBC(...)` for all 5 non-cubic
    # method families. The `kwargs...` propagation in the generic `_InterpMethod`
    # rrule is the entire mechanism that makes the periodic adjoints reachable
    # from Zygote — a future refactor dropping it would not be caught by the
    # NoBC-default tests above.
    @testset "1D one-shot ∂f/∂y — bc=PeriodicBC via rrule" begin
        period = 1.0
        nx = 12
        h = period / nx
        x = collect(range(0.0, step = h, length = nx))
        # Monotone f closed by f[1] = f[end+period]'s constraint via vcat.
        f_data = collect(range(-1.0, 1.0, length = nx))
        # Boundary-touching queries guarantee the cyclic stencil path is exercised.
        xq_vec = vcat(0.5 * h, period - 0.5 * h, period .* rand(4))
        bc_inc = PeriodicBC()
        bc_exc = PeriodicBC(endpoint = :exclusive, period = period)
        # For :inclusive, use a closed-cycle grid (length nx+1, f[end]=f[1]).
        x_inc = vcat(x, x[1] + period)
        f_inc = vcat(f_data, f_data[1])

        for (label, fn, bc, xg, fg) in [
                ("Linear", linear_interp, bc_exc, x, f_data),
                ("Constant", constant_interp, bc_exc, x, f_data),
                ("PCHIP", pchip_interp, bc_exc, x, f_data),
                ("Cardinal", cardinal_interp, bc_exc, x, f_data),
                ("Akima", akima_interp, bc_exc, x, f_data),
                ("Linear-inc", linear_interp, bc_inc, x_inc, f_inc),
                ("PCHIP-inc", pchip_interp, bc_inc, x_inc, f_inc),
                ("Akima-inc", akima_interp, bc_inc, x_inc, f_inc),
            ]
            @testset "$label" begin
                g_zy = Zygote.gradient(y -> sum(fn(xg, y, xq_vec; bc = bc)), fg)[1]
                g_fd = ForwardDiff.gradient(y -> sum(fn(xg, y, xq_vec; bc = bc)), fg)
                @test g_zy ≈ g_fd atol = 1.0e-10
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # QUADRATIC DATA-ADJOINT (∂f/∂y) via QuadraticAdjoint rrule
    # ════════════════════════════════════════════════════════════════════════

    @testset "Quadratic data-adjoint (∂f/∂y) — Zygote via rrule" begin
        x = collect(0.0:0.5:5.0)
        f_data = x .^ 2
        xq_vec = [0.75, 1.25, 2.75, 3.5, 4.25]
        y_obs = xq_vec .^ 2 .+ 0.1  # different from interpolated values

        @testset "Scalar query — ∂f/∂y" begin
            g_zy = Zygote.gradient(y -> quadratic_interp(x, y, 1.25), f_data)[1]
            g_fd = ForwardDiff.gradient(y -> quadratic_interp(x, y, 1.25), f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "Vector query — ∂f/∂y" begin
            g_zy = Zygote.gradient(y -> sum(quadratic_interp(x, y, xq_vec)), f_data)[1]
            g_fd = ForwardDiff.gradient(y -> sum(quadratic_interp(x, y, xq_vec)), f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "L2 loss function" begin
            loss(y) = sum(abs2, quadratic_interp(x, y, xq_vec) .- y_obs)
            g_zy = Zygote.gradient(loss, f_data)[1]
            g_fd = ForwardDiff.gradient(loss, f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "Float32" begin
            x32 = Float32.(x)
            f32 = Float32.(f_data)
            xq32 = Float32.(xq_vec)
            g_zy = Zygote.gradient(y -> sum(quadratic_interp(x32, y, xq32)), f32)[1]
            g_fd = ForwardDiff.gradient(y -> sum(quadratic_interp(x32, y, xq32)), f32)
            @test g_zy ≈ g_fd atol = 1.0e-4
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # CONSTANT DATA-ADJOINT (∂f/∂y) via ConstantAdjoint rrule
    # ════════════════════════════════════════════════════════════════════════

    @testset "Constant data-adjoint (∂f/∂y) — Zygote via rrule" begin
        x = collect(0.0:1.0:5.0)
        f_data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        xq_vec = [0.3, 0.8, 1.7, 2.5, 3.9]
        y_obs = [5.0, 15.0, 25.0, 35.0, 45.0]  # arbitrary target

        @testset "Scalar query — ∂f/∂y" begin
            g_zy = Zygote.gradient(y -> constant_interp(x, y, 1.7), f_data)[1]
            g_fd = ForwardDiff.gradient(y -> constant_interp(x, y, 1.7), f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "Vector query — ∂f/∂y" begin
            g_zy = Zygote.gradient(y -> sum(constant_interp(x, y, xq_vec)), f_data)[1]
            g_fd = ForwardDiff.gradient(y -> sum(constant_interp(x, y, xq_vec)), f_data)
            @test g_zy ≈ g_fd atol = 1.0e-10
        end

        @testset "All side modes" begin
            for side in [LeftSide(), RightSide(), NearestSide()]
                g_zy = Zygote.gradient(y -> sum(constant_interp(x, y, xq_vec; side = side)), f_data)[1]
                g_fd = ForwardDiff.gradient(y -> sum(constant_interp(x, y, xq_vec; side = side)), f_data)
                @test g_zy ≈ g_fd atol = 1.0e-10
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # VEC QUERY ∂/∂xq — one-shot API (linear / quadratic / cubic)
    # ════════════════════════════════════════════════════════════════════════
    # For y = interp(x, f, xq_vec), ∂L/∂xq[i] = ∂L/∂y[i] * d[i]
    # where d[i] = derivative of the interpolant at xq[i].
    # This is element-wise (diagonal Jacobian) since y[i] only depends on xq[i].
    # Validated against ForwardDiff.

    # NOTE: ForwardDiff.gradient is NOT used here — the one-shot vec API has a
    # Float64 type barrier that rejects Dual numbers. Instead, validate against
    # the analytical derivative (interp(...; deriv=DerivOp(1))), which is what
    # the rrule computes internally, plus finite differences for the L2 loss case.

    @testset "Vec query ∂/∂xq — one-shot (linear)" begin
        x = collect(0.0:0.5:5.0)
        f_data = sin.(x)
        xq_vec = [0.3, 1.1, 2.4, 3.7, 4.2]

        # sum loss: ∂/∂xq[i] = d[i] where d = derivative of interpolant
        g_zy = Zygote.gradient(q -> sum(linear_interp(x, f_data, q)), xq_vec)[1]
        d_expected = linear_interp(x, f_data, xq_vec; deriv = DerivOp(1))
        @test g_zy ≈ d_expected atol = 1.0e-10

        # L2 loss: ∂/∂xq[i] = 2*(y[i] - y_obs[i]) * d[i]
        y_obs = zeros(length(xq_vec))
        y_pred = linear_interp(x, f_data, xq_vec)
        loss(q) = sum(abs2, linear_interp(x, f_data, q) .- y_obs)
        g_zy2 = Zygote.gradient(loss, xq_vec)[1]
        g_expected = 2 .* (y_pred .- y_obs) .* d_expected
        @test g_zy2 ≈ g_expected atol = 1.0e-10
    end

    @testset "Vec query ∂/∂xq — one-shot (quadratic)" begin
        x = collect(0.0:0.5:5.0)
        f_data = x .^ 2
        xq_vec = [0.3, 1.1, 2.4, 3.7, 4.2]

        g_zy = Zygote.gradient(q -> sum(quadratic_interp(x, f_data, q)), xq_vec)[1]
        d_expected = quadratic_interp(x, f_data, xq_vec; deriv = DerivOp(1))
        @test g_zy ≈ d_expected atol = 1.0e-10

        # Analytic: d/dxq of x^2 interpolant = 2*xq
        @test g_zy ≈ 2.0 .* xq_vec atol = 0.05
    end

    @testset "Vec query ∂/∂xq — one-shot (cubic)" begin
        x = collect(0.0:0.5:5.0)
        f_data = sin.(x)
        xq_vec = [0.3, 1.1, 2.4, 3.7, 4.2]

        g_zy = Zygote.gradient(q -> sum(cubic_interp(x, f_data, q)), xq_vec)[1]
        d_expected = cubic_interp(x, f_data, xq_vec; deriv = DerivOp(1))
        @test g_zy ≈ d_expected atol = 1.0e-10

        # Analytic: d/dxq of sin interpolant ≈ cos(xq)
        @test g_zy ≈ cos.(xq_vec) atol = 0.01
    end

    # ════════════════════════════════════════════════════════════════════════
    # Non-Cubic ND interpolant API — ∂/∂data via rrule chain
    # ════════════════════════════════════════════════════════════════════════
    # Validates that generalized rrules (constructor + eval + gradient +
    # hessian + laplacian + value_gradient) work for all 4 interpolant types.

    @testset "Non-cubic ND interpolant API — ∂/∂data via rrule chain" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 2.0, 15)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        x0 = (0.7, 0.9)

        for (interp_fn, adj_fn, label) in [
                (linear_interp, linear_adjoint, "Linear"),
                (quadratic_interp, quadratic_adjoint, "Quadratic"),
                (constant_interp, constant_adjoint, "Constant"),
            ]
            @testset "$label" begin
                # ── constructor + eval: ∂/∂data of itp(x0) ──
                @testset "eval — ∂/∂data" begin
                    function f_eval(d)
                        itp = interp_fn((x, y), d)
                        return itp(x0)
                    end
                    result = Zygote.withgradient(f_eval, data)
                    @test result.val ≈ f_eval(data)

                    # Cross-validate with one-shot adjoint
                    adj = adj_fn((x, y), ([x0[1]], [x0[2]]))
                    expected = adj([1.0])
                    @test result.grad[1] ≈ expected atol = 1.0e-10
                end

                # ── eval: L2 loss ──
                @testset "eval — L2 loss" begin
                    target_val = 0.5
                    function f_l2(d)
                        itp = interp_fn((x, y), d)
                        return (itp(x0) - target_val)^2
                    end
                    result = Zygote.withgradient(f_l2, data)
                    y_val = interp_fn((x, y), data, x0)
                    adj = adj_fn((x, y), ([x0[1]], [x0[2]]))
                    expected = adj([2.0 * (y_val - target_val)])
                    @test result.grad[1] ≈ expected atol = 1.0e-10
                end

                # ── eval: ∂/∂data via vector query (must match tuple query) ──
                @testset "eval — ∂/∂data via vector query" begin
                    x0_vec = [x0[1], x0[2]]
                    function f_eval_vec(d)
                        itp = interp_fn((x, y), d)
                        return itp(x0_vec)
                    end
                    result_vec = Zygote.withgradient(f_eval_vec, data)
                    @test result_vec.val ≈ f_eval_vec(data)

                    # Must match tuple query ∂/∂data exactly
                    function f_eval_tuple(d)
                        itp = interp_fn((x, y), d)
                        return itp(x0)
                    end
                    result_tuple = Zygote.withgradient(f_eval_tuple, data)
                    @test result_vec.grad[1] ≈ result_tuple.grad[1] atol = 1.0e-12

                    # Cross-validate with one-shot adjoint
                    adj = adj_fn((x, y), ([x0[1]], [x0[2]]))
                    expected = adj([1.0])
                    @test result_vec.grad[1] ≈ expected atol = 1.0e-10
                end

                # ── gradient: ∂/∂data of ‖∇f(x0)‖² ──
                @testset "gradient — ‖∇f‖² loss" begin
                    function f_grad_norm(d)
                        itp = interp_fn((x, y), d)
                        g = FastInterpolations.gradient(itp, x0)
                        return g[1]^2 + g[2]^2
                    end
                    result = Zygote.withgradient(f_grad_norm, data)
                    @test isfinite(result.val)
                    @test size(result.grad[1]) == size(data)

                    # Cross-validate: adjoint with DerivOp
                    fx = interp_fn((x, y), data, x0; deriv = (DerivOp(1), EvalValue()))
                    fy = interp_fn((x, y), data, x0; deriv = (EvalValue(), DerivOp(1)))
                    adj = adj_fn((x, y), ([x0[1]], [x0[2]]))
                    expected = adj([2fx]; deriv = (DerivOp(1), EvalValue())) .+
                        adj([2fy]; deriv = (EvalValue(), DerivOp(1)))
                    @test result.grad[1] ≈ expected atol = 1.0e-10
                end

                # ── hessian: ∂/∂data of ‖H(f)‖²_F ──
                @testset "hessian — ∂/∂data" begin
                    function f_hess(d)
                        itp = interp_fn((x, y), d)
                        H = FastInterpolations.hessian(itp, x0)
                        return sum(abs2, H)
                    end
                    result = Zygote.withgradient(f_hess, data)
                    @test isfinite(result.val)
                    @test size(result.grad[1]) == size(data)
                end

                # ── laplacian: ∂/∂data of (∇²f)² ──
                @testset "laplacian — ∂/∂data" begin
                    function f_lap(d)
                        itp = interp_fn((x, y), d)
                        return FastInterpolations.laplacian(itp, x0)^2
                    end
                    result = Zygote.withgradient(f_lap, data)
                    @test isfinite(result.val)
                    @test size(result.grad[1]) == size(data)
                end

                # ── value_gradient: ∂/∂data ──
                @testset "value_gradient — ∂/∂data" begin
                    function f_vg(d)
                        itp = interp_fn((x, y), d)
                        v, g = FastInterpolations.value_gradient(itp, x0)
                        return v^2 + g[1]^2 + g[2]^2
                    end
                    result = Zygote.withgradient(f_vg, data)
                    @test isfinite(result.val)
                    @test size(result.grad[1]) == size(data)
                end
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Linear / Constant ND adjoint with PeriodicBC — full backward AD round-trip
    # ════════════════════════════════════════════════════════════════════════
    # Verifies that Zygote.gradient through `linear_interp` / `constant_interp`
    # with `bc=PeriodicBC(...)` produces a gradient matching the direct adjoint
    # (which routes through the new `bc` kwarg on `linear_adjoint` /
    # `constant_adjoint`). For `:exclusive` BCs, also explicitly verifies the
    # n→(n+1)→n round-trip: the user passes data of shape `size(f) == size(data)`
    # (the n-point grid), and the gradient comes back at the SAME shape — the
    # internal length-(n+1) virtual extension is fully encapsulated.

    @testset "Linear ND ∂/∂data — PeriodicBC{:inclusive}" begin
        nx, ny = 12, 9
        n_query = 30
        x = range(0.0, 1.0, nx)
        y = range(0.0, 2.0, ny)
        data = randn(nx, ny)
        # `:inclusive` requires exact endpoint match
        data[end, :] .= data[1, :]
        data[:, end] .= data[:, 1]
        xq = rand(n_query)
        yq = rand(n_query) .* 2
        bc = (PeriodicBC(), PeriodicBC())

        g_zy = Zygote.gradient(d -> sum(linear_interp((x, y), d, (xq, yq); bc = bc)), data)[1]
        adj = linear_adjoint((x, y), (xq, yq); bc = bc)
        g_adj = adj(ones(n_query))

        @test size(g_zy) == size(data)   # inclusive: closed form, no virtual extension
        @test g_zy ≈ g_adj atol = 1.0e-10
    end

    @testset "Linear ND ∂/∂data — PeriodicBC{:exclusive} (n→n+1→n round-trip)" begin
        nx, ny = 11, 8
        n_query = 30
        # User-supplied n-point (half-open) grid + n-point data
        x = collect(range(0.0, step = 1.0 / nx, length = nx))
        y_grid = collect(range(0.0, step = 2.0 / ny, length = ny))
        data = randn(nx, ny)
        xq = rand(n_query)
        yq = rand(n_query) .* 2
        bc = (
            PeriodicBC(endpoint = :exclusive, period = 1.0),
            PeriodicBC(endpoint = :exclusive, period = 2.0),
        )

        g_zy = Zygote.gradient(
            d -> sum(linear_interp((x, y_grid), d, (xq, yq); bc = bc)), data
        )[1]
        adj = linear_adjoint((x, y_grid), (xq, yq); bc = bc)
        g_adj = adj(ones(n_query))

        # Critical shape check: gradient back at the user's n-point shape.
        # Internally LinearAdjointND scattered into an n+1-sized work buffer
        # and folded the seam (`f_work[1, :] += f_work[end, :]`); the trim
        # should hand back exactly `size(data) == (nx, ny)`.
        @test size(g_zy) == (nx, ny)
        @test size(g_adj) == (nx, ny)
        @test g_zy ≈ g_adj atol = 1.0e-10
    end

    @testset "Linear ND ∂/∂data — PeriodicBC{:exclusive} × NoBC mixed" begin
        nx, ny = 10, 8
        n_query = 25
        x = collect(range(0.0, step = 1.0 / nx, length = nx))
        y = range(0.0, 1.0, ny)
        data = randn(nx, ny)
        xq = rand(n_query)
        yq = rand(n_query)
        bc = (PeriodicBC(endpoint = :exclusive, period = 1.0), NoBC())

        g_zy = Zygote.gradient(d -> sum(linear_interp((x, y), d, (xq, yq); bc = bc)), data)[1]
        adj = linear_adjoint((x, y), (xq, yq); bc = bc)
        g_adj = adj(ones(n_query))

        @test size(g_zy) == (nx, ny)   # axis-1 trim n+1→n; axis-2 unchanged
        @test g_zy ≈ g_adj atol = 1.0e-10
    end

    @testset "Constant ND ∂/∂data — PeriodicBC{:inclusive}" begin
        nx, ny = 12, 9
        n_query = 30
        x = range(0.0, 1.0, nx)
        y = range(0.0, 2.0, ny)
        data = randn(nx, ny)
        data[end, :] .= data[1, :]
        data[:, end] .= data[:, 1]
        xq = rand(n_query)
        yq = rand(n_query) .* 2
        bc = (PeriodicBC(), PeriodicBC())

        g_zy = Zygote.gradient(d -> sum(constant_interp((x, y), d, (xq, yq); bc = bc)), data)[1]
        adj = constant_adjoint((x, y), (xq, yq); bc = bc)
        g_adj = adj(ones(n_query))

        @test size(g_zy) == size(data)
        @test g_zy ≈ g_adj atol = 1.0e-12   # constant: single-point scatter, exact
    end

    @testset "Constant ND ∂/∂data — PeriodicBC{:exclusive} (n→n+1→n round-trip)" begin
        nx, ny = 11, 8
        n_query = 25
        x = collect(range(0.0, step = 1.0 / nx, length = nx))
        y_grid = collect(range(0.0, step = 2.0 / ny, length = ny))
        data = randn(nx, ny)
        xq = rand(n_query)
        yq = rand(n_query) .* 2
        bc = (
            PeriodicBC(endpoint = :exclusive, period = 1.0),
            PeriodicBC(endpoint = :exclusive, period = 2.0),
        )

        g_zy = Zygote.gradient(
            d -> sum(constant_interp((x, y_grid), d, (xq, yq); bc = bc)), data
        )[1]
        adj = constant_adjoint((x, y_grid), (xq, yq); bc = bc)
        g_adj = adj(ones(n_query))

        @test size(g_zy) == (nx, ny)
        @test size(g_adj) == (nx, ny)
        @test g_zy ≈ g_adj atol = 1.0e-12
    end

    @testset "Cubic ND ∂/∂data — PeriodicBC{:exclusive} (n→n+1→n round-trip)" begin
        # Mirrors the existing `:inclusive` test at L393-408 but for `:exclusive`
        # — verifies the Sherman-Morrison transpose path AND the n→n+1→n trim.
        nx, ny = 10, 8
        n_query = 25
        x = collect(range(0.0, step = 2π / nx, length = nx))
        y_grid = collect(range(0.0, step = 2π / ny, length = ny))
        data = [sin(xi) + cos(yj) for xi in x, yj in y_grid]
        xq = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
        yq = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
        bc = (
            PeriodicBC(endpoint = :exclusive, period = 2π),
            PeriodicBC(endpoint = :exclusive, period = 2π),
        )

        g_zy = Zygote.gradient(d -> sum(cubic_interp((x, y_grid), d, (xq, yq); bc = bc)), data)[1]
        adj = cubic_adjoint((x, y_grid), (xq, yq); bc = bc)
        g_adj = adj(ones(n_query))

        @test size(g_zy) == (nx, ny)
        @test size(g_adj) == (nx, ny)
        @test g_zy ≈ g_adj atol = 1.0e-10
    end

    @testset "Unified API one-shot ∂/∂data — homogeneous Linear+PeriodicBC (batch)" begin
        nx, ny = 10, 8
        n_query = 25
        x = collect(range(0.0, 1.0, nx))
        y_grid = collect(range(0.0, 1.0, ny))
        data = [sin(2π * xi) + cos(2π * yj) for xi in x, yj in y_grid]
        xq = sort(rand(n_query)) .* 0.96 .+ 0.02
        yq = sort(rand(n_query)) .* 0.96 .+ 0.02
        bc = PeriodicBC()
        methods = (LinearInterp(bc = bc), LinearInterp(bc = bc))

        g_zy = Zygote.gradient(
            d -> sum(interp((x, y_grid), d, (xq, yq); method = methods)), data
        )[1]
        # Reference via direct `linear_adjoint` (same shape, same bc tuple).
        adj_ref = linear_adjoint((x, y_grid), (xq, yq); bc = (bc, bc))
        g_ref = adj_ref(ones(n_query))

        @test size(g_zy) == (nx, ny)
        @test g_zy ≈ g_ref atol = 1.0e-10
    end

    @testset "Unified API one-shot ∂/∂data — heterogeneous Linear+Cubic (batch)" begin
        nx, ny = 12, 10
        n_query = 20
        # `:exclusive` requires n distinct samples on [first, first+period) —
        # i.e. last(x) < first(x) + period. Build x with `step=period/nx` so
        # x[end] = (nx-1) * period/nx < period.
        x = collect(range(0.0, step = 2π / nx, length = nx))
        y_grid = collect(range(0.0, 1.0, ny))
        data = [sin(xi) * yj for xi in x, yj in y_grid]
        xq = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
        yq = sort(rand(n_query)) .* 0.96 .+ 0.02
        methods = (
            LinearInterp(bc = PeriodicBC(endpoint = :exclusive, period = 2π)),
            CubicInterp(),
        )

        g_zy = Zygote.gradient(
            d -> sum(interp((x, y_grid), d, (xq, yq); method = methods)), data
        )[1]
        adj_ref = hetero_adjoint((x, y_grid), (xq, yq); methods = methods)
        g_ref = adj_ref(ones(n_query))

        @test size(g_zy) == (nx, ny)
        @test g_zy ≈ g_ref atol = 1.0e-10
    end

    @testset "Unified API one-shot ∂/∂data — single-point tuple query" begin
        nx, ny = 10, 8
        x = collect(range(0.0, 1.0, nx))
        y_grid = collect(range(0.0, 1.0, ny))
        data = [sin(2π * xi) + cos(2π * yj) for xi in x, yj in y_grid]
        bc = PeriodicBC()
        methods = (LinearInterp(bc = bc), LinearInterp(bc = bc))

        g_zy = Zygote.gradient(d -> interp((x, y_grid), d, (0.35, 0.7); method = methods), data)[1]
        adj_ref = linear_adjoint((x, y_grid), ([0.35], [0.7]); bc = (bc, bc))
        g_ref = adj_ref(ones(1))

        @test size(g_zy) == (nx, ny)
        @test g_zy ≈ g_ref atol = 1.0e-10
    end

    # Regression: the batch rrule previously slurped all kwargs and forwarded
    # them to the pullback callable, so construction-time kwargs (`extrap`)
    # would leak into the adjoint apply path. Fix lifts `deriv` out explicitly
    # and forwards only `deriv = deriv` to `adj(...)`.
    @testset "Unified API one-shot ∂/∂data — extrap kwarg does not leak to pullback" begin
        nx, ny = 10, 8
        x = collect(range(0.0, 1.0, nx))
        y_grid = collect(range(0.0, 1.0, ny))
        data = [sin(2π * xi) + cos(2π * yj) for xi in x, yj in y_grid]
        bc = PeriodicBC()
        methods = (LinearInterp(bc = bc), LinearInterp(bc = bc))
        xq = collect(range(0.05, 0.95, 5))
        yq = collect(range(0.05, 0.95, 5))

        # ExtendExtrap is a construction-time arg — must reach the forward but
        # NOT be forwarded by the pullback (apply path does not accept it).
        g_zy = Zygote.gradient(
            d -> sum(interp((x, y_grid), d, (xq, yq); method = methods, extrap = ExtendExtrap())),
            data
        )[1]
        adj_ref = linear_adjoint((x, y_grid), (xq, yq); bc = (bc, bc), extrap = ExtendExtrap())
        g_ref = adj_ref(ones(length(xq)))

        @test size(g_zy) == (nx, ny)
        @test g_zy ≈ g_ref atol = 1.0e-10
    end

end  # testset "Zygote AD Support"
