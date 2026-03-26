# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║          HETEROGENEOUS ND — AD INTEGRATION TESTS (Zygote + ForwardDiff) ║
# ║      ∂/∂query, ∂/∂data, loss functions, analytic cross-validation      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Tests Zygote reverse-mode and ForwardDiff forward-mode AD for HeteroInterpolantND.
# Cross-validates against analytical derivatives (gradient/hessian) and adjoint operators.

using Test
using FastInterpolations
using Zygote
using ForwardDiff
using LinearAlgebra

@testset "Hetero ND — AD Integration" begin

    # ════════════════════════════════════════════════════════════════════════
    # FORWARDDIFF ∂/∂query — Hetero ND
    # ════════════════════════════════════════════════════════════════════════

    @testset "ForwardDiff ∂/∂query" begin
        x = range(0.0, 1.0, 15)
        y = range(0.0, 1.0, 12)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]

        for (label, methods) in [
                ("Cubic×Linear", (CubicInterp(), LinearInterp())),
                ("Quadratic×Constant", (QuadraticInterp(), ConstantInterp())),
                ("Cubic×Quadratic", (CubicInterp(), QuadraticInterp())),
                ("Linear×Linear", (LinearInterp(), LinearInterp())),
            ]
            @testset "$label" begin
                itp = interp((x, y), data; method = methods)
                q = [0.37, 0.63]
                fd_grad = ForwardDiff.gradient(itp, q)
                ana_grad = collect(FastInterpolations.gradient(itp, Tuple(q)))
                @test fd_grad ≈ ana_grad atol = 1.0e-10
            end
        end

        @testset "3D Cubic×Linear×Quad" begin
            z = range(0.0, 1.0, 8)
            data3 = randn(15, 12, 8)
            methods = (CubicInterp(), LinearInterp(), QuadraticInterp())
            itp3 = interp((x, y, z), data3; method = methods)
            q = [0.3, 0.5, 0.7]
            fd_grad = ForwardDiff.gradient(itp3, q)
            ana_grad = collect(FastInterpolations.gradient(itp3, Tuple(q)))
            @test fd_grad ≈ ana_grad atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # ZYGOTE ∂/∂query — Hetero ND
    # ════════════════════════════════════════════════════════════════════════

    @testset "Zygote ∂/∂query" begin
        x = range(0.0, 1.0, 15)
        y = range(0.0, 1.0, 12)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]

        for (label, methods) in [
                ("Cubic×Linear", (CubicInterp(), LinearInterp())),
                ("Cubic×Quadratic", (CubicInterp(), QuadraticInterp())),
                ("Quadratic×Linear", (QuadraticInterp(), LinearInterp())),
            ]
            @testset "$label — vector query" begin
                itp = interp((x, y), data; method = methods)
                q = [0.37, 0.63]
                zy_grad = Zygote.gradient(itp, q)[1]
                fd_grad = ForwardDiff.gradient(itp, q)
                @test zy_grad ≈ fd_grad atol = 1.0e-8
            end

            @testset "$label — tuple query" begin
                itp = interp((x, y), data; method = methods)
                zy_grad = Zygote.gradient(xq -> itp((xq, 0.5)), 0.37)[1]
                fd_grad = ForwardDiff.derivative(xq -> itp((xq, 0.5)), 0.37)
                @test zy_grad ≈ fd_grad atol = 1.0e-10
            end
        end

        @testset "Loss function: L2" begin
            methods = (CubicInterp(), LinearInterp())
            itp = interp((x, y), data; method = methods)
            target = 0.42
            loss(q) = (itp(q) - target)^2
            q = [0.37, 0.63]
            zy_grad = Zygote.gradient(loss, q)[1]
            fd_grad = ForwardDiff.gradient(loss, q)
            @test zy_grad ≈ fd_grad atol = 1.0e-8
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # ZYGOTE ∂/∂data — via interp() constructor rrule
    # ════════════════════════════════════════════════════════════════════════

    @testset "Zygote ∂/∂data — interp() constructor" begin
        x = range(0.0, 1.0, 12)
        y = range(0.0, 1.0, 10)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
        x0 = (0.37, 0.63)

        @testset "Cubic×Linear — basic eval" begin
            methods = (CubicInterp(), LinearInterp())
            g = Zygote.gradient(d -> interp((x, y), d; method = methods)(x0), data)[1]
            @test size(g) == size(data)
            @test sum(g) ≈ 1.0 atol = 1.0e-10  # partition of unity

            # Cross-validate with adjoint operator
            adj = hetero_adjoint((x, y), (x0,); methods = methods)
            @test g ≈ adj(1.0) atol = 1.0e-10
        end

        @testset "Quadratic×Linear — basic eval" begin
            methods = (QuadraticInterp(), LinearInterp())
            g = Zygote.gradient(d -> interp((x, y), d; method = methods)(x0), data)[1]
            adj = hetero_adjoint((x, y), (x0,); methods = methods)
            @test g ≈ adj(1.0) atol = 1.0e-10
        end

        @testset "Cubic×Quadratic — L2 loss" begin
            methods = (CubicInterp(), QuadraticInterp())
            target = 0.5
            loss(d) = (interp((x, y), d; method = methods)(x0) - target)^2
            g = Zygote.gradient(loss, data)[1]

            # Manual: ∂L/∂data = 2(y - target) * adj(1)
            itp = interp((x, y), data; method = methods)
            y_val = itp(x0)
            adj = hetero_adjoint((x, y), (x0,); methods = methods)
            expected = adj(2.0 * (y_val - target))
            @test g ≈ expected atol = 1.0e-10
        end

        @testset "Multiple query points — SoA batch" begin
            methods = (CubicInterp(), LinearInterp())
            xq = [0.2, 0.5, 0.8]
            yq = [0.3, 0.5, 0.7]
            nq = length(xq)

            g = Zygote.gradient(
                d -> sum(interp((x, y), d; method = methods)((xq[i], yq[i])) for i in 1:nq),
                data,
            )[1]

            # Cross-validate: sum of individual adjoint contributions
            adj = hetero_adjoint((x, y), (xq, yq); methods = methods)
            expected = adj(ones(nq))
            @test g ≈ expected atol = 1.0e-10
        end

        @testset "3D Cubic×Linear×Quad" begin
            z = range(0.0, 1.0, 8)
            data3 = randn(12, 10, 8)
            methods = (CubicInterp(), LinearInterp(), QuadraticInterp())
            q3 = (0.3, 0.5, 0.7)

            g3 = Zygote.gradient(d -> interp((x, y, z), d; method = methods)(q3), data3)[1]
            adj3 = hetero_adjoint((x, y, z), (q3,); methods = methods)
            @test g3 ≈ adj3(1.0) atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # ZYGOTE ∂/∂data — gradient/hessian/laplacian rrules with hetero
    # ════════════════════════════════════════════════════════════════════════

    @testset "Zygote ∂/∂data — gradient rrule" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 2.0, 12)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        x0 = (0.7, 0.9)
        methods = (CubicInterp(), QuadraticInterp())

        # ∂/∂data of ‖∇f(x0)‖²
        function f_grad_norm(d)
            itp = interp((x, y), d; method = methods)
            g = FastInterpolations.gradient(itp, x0)
            return g[1]^2 + g[2]^2
        end
        result = Zygote.withgradient(f_grad_norm, data)
        @test result.val ≈ f_grad_norm(data)

        # Cross-validate
        itp = interp((x, y), data; method = methods)
        fx = itp(x0; deriv = (DerivOp(1), EvalValue()))
        fy = itp(x0; deriv = (EvalValue(), DerivOp(1)))
        adj = hetero_adjoint((x, y), (x0,); methods = methods)
        expected = adj(2fx; deriv = (DerivOp(1), EvalValue())) .+
            adj(2fy; deriv = (EvalValue(), DerivOp(1)))
        @test result.grad[1] ≈ expected atol = 1.0e-9
    end

    @testset "Zygote ∂/∂data — hessian rrule" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 2.0, 12)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        x0 = (0.7, 0.9)
        methods = (CubicInterp(), QuadraticInterp())

        function f_hess_xx(d)
            itp = interp((x, y), d; method = methods)
            H = FastInterpolations.hessian(itp, x0)
            return H[1, 1]
        end
        result = Zygote.withgradient(f_hess_xx, data)
        adj = hetero_adjoint((x, y), (x0,); methods = methods)
        expected = adj(1.0; deriv = (DerivOp(2), EvalValue()))
        @test result.grad[1] ≈ expected atol = 1.0e-9
    end

    @testset "Zygote ∂/∂data — laplacian rrule" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 2.0, 12)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        x0 = (0.7, 0.9)
        methods = (CubicInterp(), QuadraticInterp())

        function f_lap(d)
            itp = interp((x, y), d; method = methods)
            return FastInterpolations.laplacian(itp, x0)
        end
        result = Zygote.withgradient(f_lap, data)
        adj = hetero_adjoint((x, y), (x0,); methods = methods)
        expected = adj(1.0; deriv = (DerivOp(2), EvalValue())) .+
            adj(1.0; deriv = (EvalValue(), DerivOp(2)))
        @test result.grad[1] ≈ expected atol = 1.0e-9
    end

    # ════════════════════════════════════════════════════════════════════════
    # PERIODIC BC with AD
    # ════════════════════════════════════════════════════════════════════════

    @testset "PeriodicBC + Zygote ∂/∂data" begin
        x_p = range(0.0, 2π, 20)
        y = range(0.0, 1.0, 10)
        data = [sin(xi) * yj for xi in x_p, yj in y]
        x0 = (1.5, 0.6)
        methods = (CubicInterp(bc = PeriodicBC()), LinearInterp())

        g = Zygote.gradient(d -> interp((x_p, y), d; method = methods)(x0), data)[1]
        adj = hetero_adjoint((x_p, y), (x0,); methods = methods)
        @test g ≈ adj(1.0) atol = 1.0e-10
    end

    # ════════════════════════════════════════════════════════════════════════
    # HOMOGENEOUS VIA interp() — verify interp() rrule doesn't break existing
    # ════════════════════════════════════════════════════════════════════════

    @testset "Homogeneous via interp() — ∂/∂data unchanged" begin
        x = range(0.0, 1.0, 12)
        y = range(0.0, 1.0, 10)
        data = randn(12, 10)
        x0 = (0.5, 0.5)

        @testset "CubicInterp via interp()" begin
            g_interp = Zygote.gradient(d -> interp((x, y), d; method = CubicInterp())(x0), data)[1]
            g_direct = Zygote.gradient(d -> cubic_interp((x, y), d)(x0), data)[1]
            @test g_interp ≈ g_direct atol = 1.0e-12
        end

        @testset "LinearInterp via interp()" begin
            g_interp = Zygote.gradient(d -> interp((x, y), d; method = LinearInterp())(x0), data)[1]
            g_direct = Zygote.gradient(d -> linear_interp((x, y), d)(x0), data)[1]
            @test g_interp ≈ g_direct atol = 1.0e-12
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # FLOAT32 AD
    # ════════════════════════════════════════════════════════════════════════

    @testset "Float32 AD" begin
        x32 = Float32.(range(0.0, 1.0, 12))
        y32 = Float32.(range(0.0, 1.0, 10))
        data32 = Float32[sin(xi) * cos(yj) for xi in x32, yj in y32]
        methods = (CubicInterp(), LinearInterp())

        @testset "ForwardDiff gradient" begin
            itp = interp((x32, y32), data32; method = methods)
            q = Float32[0.37, 0.63]
            fd = ForwardDiff.gradient(itp, q)
            ana = collect(FastInterpolations.gradient(itp, Tuple(q)))
            @test fd ≈ ana atol = 1.0f-4
        end

        @testset "Zygote ∂/∂data" begin
            g = Zygote.gradient(
                d -> interp((x32, y32), d; method = methods)((0.37f0, 0.63f0)), data32
            )[1]
            @test g isa Matrix{Float32}
            @test isapprox(sum(g), 1.0f0, atol = 1.0f-5)
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # ENZYME ∂/∂data — tested in test_autodiff_Enzyme.jl
    # ════════════════════════════════════════════════════════════════════════
    # Enzyme ND struct API tests (including hetero) are in test_autodiff_Enzyme.jl
    # under the ENZYME_ND_STRUCT_SUPPORTED guard (Julia ≥ 1.11, Linux x64).
    # The hetero path uses the same EnzymeRules (interp() constructor + eval)
    # as homogeneous types — see "Hetero (Cubic×Linear)" testset there.

end # @testset "Hetero ND — AD Integration"
