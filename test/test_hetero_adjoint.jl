# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║               HETEROGENEOUS ADJOINT (HeteroAdjointND) TESTS             ║
# ║  Matrix consistency, forward-adjoint verification, derivative adjoint   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Tests the HeteroAdjointND operator:
#   adj = hetero_adjoint(grids, queries; methods=...)
#   f_bar = adj(y_bar)           # W^T * y_bar
#   Matrix(adj) materializes W^T
#
# Cross-validates:
#   W' * f ≈ itp.(queries)  (forward-adjoint consistency)
#   adj(1) partition of unity: sum(adj(1.0)) ≈ 1.0 for EvalValue
#   derivative adjoint: Matrix(adj; deriv=...) vs analytical gradient

@testitem "HeteroAdjointND" setup = [AllocConstants] begin
    using LinearAlgebra

    # ════════════════════════════════════════════════════════════════════════
    # FORWARD-ADJOINT CONSISTENCY: W' * f ≈ itp.(queries)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Forward-adjoint consistency (W' * f)" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
        f_vec = vec(data)
        xq = [0.15, 0.35, 0.55, 0.75, 0.95]
        yq = [0.1, 0.3, 0.5, 0.7, 0.9]

        for (label, methods) in [
                ("Cubic×Linear", (CubicInterp(), LinearInterp())),
                ("Linear×Cubic", (LinearInterp(), CubicInterp())),
                ("Quadratic×Constant", (QuadraticInterp(), ConstantInterp())),
                ("Cubic×Quadratic", (CubicInterp(), QuadraticInterp())),
                ("Linear×Linear", (LinearInterp(), LinearInterp())),
                ("Constant×Constant", (ConstantInterp(), ConstantInterp())),
                ("Cubic×Constant", (CubicInterp(), ConstantInterp())),
                ("Quadratic×Linear", (QuadraticInterp(), LinearInterp())),
            ]
            @testset "$label" begin
                itp = interp((x, y), data; method = methods)
                adj = hetero_adjoint((x, y), (xq, yq); methods = methods)
                W_T = Matrix(adj)

                for q in eachindex(xq)
                    fwd = itp((xq[q], yq[q]))
                    mat_fwd = dot(W_T[:, q], f_vec)
                    @test fwd ≈ mat_fwd atol = 1.0e-12
                end
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # 3D CONSISTENCY
    # ════════════════════════════════════════════════════════════════════════

    @testset "3D forward-adjoint consistency" begin
        x = range(0.0, 1.0, 8)
        y = range(0.0, 1.0, 6)
        z = range(0.0, 1.0, 5)
        data = randn(8, 6, 5)
        f_vec = vec(data)
        xq = [0.2, 0.5, 0.8]
        yq = [0.3, 0.5, 0.7]
        zq = [0.2, 0.6, 0.9]

        for (label, methods) in [
                ("Cubic×Linear×Quad", (CubicInterp(), LinearInterp(), QuadraticInterp())),
                ("Linear×Cubic×Constant", (LinearInterp(), CubicInterp(), ConstantInterp())),
                ("Quad×Quad×Linear", (QuadraticInterp(), QuadraticInterp(), LinearInterp())),
            ]
            @testset "$label" begin
                itp = interp((x, y, z), data; method = methods)
                adj = hetero_adjoint((x, y, z), (xq, yq, zq); methods = methods)
                W_T = Matrix(adj)
                for q in eachindex(xq)
                    fwd = itp((xq[q], yq[q], zq[q]))
                    mat_fwd = dot(W_T[:, q], f_vec)
                    @test fwd ≈ mat_fwd atol = 1.0e-12
                end
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # PARTITION OF UNITY: sum(adj(1.0)) ≈ 1.0
    # ════════════════════════════════════════════════════════════════════════

    @testset "Partition of unity" begin
        x = range(0.0, 1.0, 12)
        y = range(0.0, 1.0, 10)

        for (label, methods) in [
                ("Cubic×Linear", (CubicInterp(), LinearInterp())),
                ("Quadratic×Constant", (QuadraticInterp(), ConstantInterp())),
                ("Linear×Quadratic", (LinearInterp(), QuadraticInterp())),
            ]
            @testset "$label" begin
                adj = hetero_adjoint((x, y), ((0.37, 0.63),); methods = methods)
                f_bar = adj(1.0)
                @test sum(f_bar) ≈ 1.0 atol = 1.0e-12
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # GOLDEN RULE: ⟨Wf, ȳ⟩ = ⟨f, W^Tȳ⟩  (adjoint inner product identity)
    # ════════════════════════════════════════════════════════════════════════
    # The defining property of the adjoint: for ARBITRARY y_bar and data,
    # the inner products must match. This is stronger than per-column checks
    # because it tests the full operator with random cotangent vectors.

    @testset "Golden rule: ⟨Wf, ȳ⟩ = ⟨f, W^Tȳ⟩" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        data = randn(10, 8)
        f_vec = vec(data)
        xq = [0.15, 0.35, 0.55, 0.75, 0.95]
        yq = [0.1, 0.3, 0.5, 0.7, 0.9]
        nq = length(xq)

        for (label, methods) in [
                ("Cubic×Linear", (CubicInterp(), LinearInterp())),
                ("Quadratic×Constant", (QuadraticInterp(), ConstantInterp())),
                ("Cubic×Quadratic", (CubicInterp(), QuadraticInterp())),
                ("Linear×Linear", (LinearInterp(), LinearInterp())),
                ("Linear×Cubic", (LinearInterp(), CubicInterp())),
            ]
            @testset "$label" begin
                itp = interp((x, y), data; method = methods)
                adj = hetero_adjoint((x, y), (xq, yq); methods = methods)

                # Compute Wf (forward interpolation at all query points)
                Wf = [itp((xq[q], yq[q])) for q in 1:nq]

                # Random cotangent vector
                y_bar = randn(nq)

                # ⟨Wf, ȳ⟩ — inner product in query space
                lhs = dot(Wf, y_bar)

                # ⟨f, W^Tȳ⟩ — inner product in data space
                f_bar = adj(y_bar)
                rhs = dot(f_vec, vec(f_bar))

                @test lhs ≈ rhs atol = 1.0e-10
            end
        end

        # 3D golden rule
        @testset "3D Cubic×Linear×Quad" begin
            z = range(0.0, 1.0, 5)
            data3 = randn(10, 8, 5)
            f_vec3 = vec(data3)
            zq = [0.2, 0.5, 0.8, 0.6, 0.4]
            methods = (CubicInterp(), LinearInterp(), QuadraticInterp())
            itp3 = interp((x, y, z), data3; method = methods)
            adj3 = hetero_adjoint((x, y, z), (xq, yq, zq); methods = methods)

            Wf3 = [itp3((xq[q], yq[q], zq[q])) for q in 1:nq]
            y_bar3 = randn(nq)
            lhs3 = dot(Wf3, y_bar3)
            rhs3 = dot(f_vec3, vec(adj3(y_bar3)))
            @test lhs3 ≈ rhs3 atol = 1.0e-10
        end

        # Golden rule with derivative ops
        @testset "Derivative: ⟨W_dx f, ȳ⟩ = ⟨f, W_dx^T ȳ⟩" begin
            methods = (CubicInterp(), LinearInterp())
            itp = interp((x, y), data; method = methods)
            adj = hetero_adjoint((x, y), (xq, yq); methods = methods)

            # ∂/∂x derivative operator
            Wf_dx = [FastInterpolations.gradient(itp, (xq[q], yq[q]))[1] for q in 1:nq]
            y_bar = randn(nq)
            lhs_dx = dot(Wf_dx, y_bar)
            rhs_dx = dot(f_vec, vec(adj(y_bar; deriv = (DerivOp(1), EvalValue()))))
            @test lhs_dx ≈ rhs_dx atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # DERIVATIVE ADJOINT: W^T with deriv ops
    # ════════════════════════════════════════════════════════════════════════

    @testset "Derivative adjoint" begin
        x = range(0.0, 1.0, 12)
        y = range(0.0, 1.0, 10)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
        f_vec = vec(data)
        xq = [0.2, 0.5, 0.8]
        yq = [0.3, 0.5, 0.7]

        @testset "Cubic×Linear — ∂/∂x via DerivOp(1)" begin
            methods = (CubicInterp(), LinearInterp())
            itp = interp((x, y), data; method = methods)
            adj = hetero_adjoint((x, y), (xq, yq); methods = methods)
            W_T_dx = Matrix(adj; deriv = (DerivOp(1), EvalValue()))

            for q in eachindex(xq)
                grad = FastInterpolations.gradient(itp, (xq[q], yq[q]))
                mat_dx = dot(W_T_dx[:, q], f_vec)
                @test grad[1] ≈ mat_dx atol = 1.0e-10
            end
        end

        @testset "Cubic×Quadratic — ∂/∂y via DerivOp(1)" begin
            methods = (CubicInterp(), QuadraticInterp())
            itp = interp((x, y), data; method = methods)
            adj = hetero_adjoint((x, y), (xq, yq); methods = methods)
            W_T_dy = Matrix(adj; deriv = (EvalValue(), DerivOp(1)))

            for q in eachindex(xq)
                grad = FastInterpolations.gradient(itp, (xq[q], yq[q]))
                mat_dy = dot(W_T_dy[:, q], f_vec)
                @test grad[2] ≈ mat_dy atol = 1.0e-10
            end
        end

        @testset "Cubic×Quadratic — ∂²/∂x² via DerivOp(2)" begin
            methods = (CubicInterp(), QuadraticInterp())
            itp = interp((x, y), data; method = methods)
            adj = hetero_adjoint((x, y), (xq, yq); methods = methods)
            W_T_dxx = Matrix(adj; deriv = (DerivOp(2), EvalValue()))

            for q in eachindex(xq)
                H = FastInterpolations.hessian(itp, (xq[q], yq[q]))
                mat_dxx = dot(W_T_dxx[:, q], f_vec)
                @test H[1, 1] ≈ mat_dxx atol = 1.0e-9
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # IN-PLACE ADJOINT: adj(f_bar, y_bar) — zero-allocation path
    # ════════════════════════════════════════════════════════════════════════

    @testset "In-place adjoint" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        methods = (CubicInterp(), LinearInterp())
        adj = hetero_adjoint((x, y), ((0.5, 0.5),); methods = methods)

        f_bar_alloc = adj(1.0)
        f_bar_ip = zeros(size(f_bar_alloc))
        adj(f_bar_ip, 1.0)
        @test f_bar_ip ≈ f_bar_alloc atol = 1.0e-14
    end

    # ════════════════════════════════════════════════════════════════════════
    # PERIODIC BC (Cubic axis with PeriodicBC in hetero context)
    # ════════════════════════════════════════════════════════════════════════

    @testset "PeriodicBC — Cubic(periodic)×Linear" begin
        x_p = range(0.0, 2π, 20)
        y = range(0.0, 1.0, 10)
        data = [sin(xi) * yj for xi in x_p, yj in y]
        f_vec = vec(data)
        xq = [0.5, 2.0, 4.0, 5.5]
        yq = [0.2, 0.4, 0.6, 0.8]

        methods = (CubicInterp(bc = PeriodicBC()), LinearInterp())
        itp = interp((x_p, y), data; method = methods)
        adj = hetero_adjoint((x_p, y), (xq, yq); methods = methods)
        W_T = Matrix(adj)

        for q in eachindex(xq)
            fwd = itp((xq[q], yq[q]))
            mat_fwd = dot(W_T[:, q], f_vec)
            @test fwd ≈ mat_fwd atol = 1.0e-10
        end
    end

    @testset "PeriodicBC{:exclusive} — Cubic(periodic_excl)×Quadratic" begin
        # Exclusive periodic: grid does NOT include right endpoint → adjoint must extend grid
        # and _adjoint_output_size must truncate (n → n-1 on periodic axis)
        n_p = 20
        x_p = range(0.0, 2π, n_p + 1)[1:n_p]  # exclusive: omits endpoint
        y = range(0.0, 1.0, 8)
        data = [sin(xi) * (1 + yj) for xi in x_p, yj in y]
        f_vec = vec(data)
        xq = [0.5, 2.0, 4.0, 5.5]
        yq = [0.2, 0.4, 0.6, 0.8]

        methods = (CubicInterp(bc = PeriodicBC(endpoint = :exclusive)), QuadraticInterp())
        itp = interp((x_p, y), data; method = methods)
        adj = hetero_adjoint((x_p, y), (xq, yq); methods = methods)

        # Output size should match input data size (exclusive periodic shrinks by 1)
        f_bar = adj(ones(length(xq)))
        @test size(f_bar) == size(data)

        # Forward-adjoint consistency
        W_T = Matrix(adj)
        for q in eachindex(xq)
            fwd = itp((xq[q], yq[q]))
            mat_fwd = dot(W_T[:, q], f_vec)
            @test fwd ≈ mat_fwd atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # MINCURVFIT BC (Quadratic axis with MinCurvFit in hetero context)
    # ════════════════════════════════════════════════════════════════════════

    @testset "QuadraticInterp(bc=MinCurvFit()) in hetero" begin
        x = range(0.0, 1.0, 12)
        y = range(0.0, 1.0, 10)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
        f_vec = vec(data)
        xq = [0.2, 0.5, 0.8]
        yq = [0.3, 0.5, 0.7]

        methods = (CubicInterp(), QuadraticInterp(bc = MinCurvFit()))
        itp = interp((x, y), data; method = methods)
        adj = hetero_adjoint((x, y), (xq, yq); methods = methods)
        W_T = Matrix(adj)

        for q in eachindex(xq)
            fwd = itp((xq[q], yq[q]))
            mat_fwd = dot(W_T[:, q], f_vec)
            @test fwd ≈ mat_fwd atol = 1.0e-10
        end

        # Golden rule with random y_bar
        Wf = [itp((xq[q], yq[q])) for q in eachindex(xq)]
        y_bar = randn(length(xq))
        @test dot(Wf, y_bar) ≈ dot(f_vec, vec(adj(y_bar))) atol = 1.0e-10
    end

    # ════════════════════════════════════════════════════════════════════════
    # DERIVATIVE ON NON-DERIVATIVE AXES (DerivOp on Linear/Constant axis)
    # ════════════════════════════════════════════════════════════════════════

    @testset "DerivOp on linear/constant axis" begin
        x = range(0.0, 1.0, 12)
        y = range(0.0, 1.0, 10)
        data = [xi * yj for xi in x, yj in y]  # f(x,y) = xy — exact for linear
        f_vec = vec(data)
        xq = [0.2, 0.5, 0.8]
        yq = [0.3, 0.5, 0.7]

        @testset "DerivOp(1) on Linear axis — Cubic×Linear" begin
            methods = (CubicInterp(), LinearInterp())
            itp = interp((x, y), data; method = methods)
            adj = hetero_adjoint((x, y), (xq, yq); methods = methods)

            # ∂f/∂y on the LINEAR axis: for f=xy, ∂f/∂y = x
            W_T_dy = Matrix(adj; deriv = (EvalValue(), DerivOp(1)))
            for q in eachindex(xq)
                grad = FastInterpolations.gradient(itp, (xq[q], yq[q]))
                @test dot(W_T_dy[:, q], f_vec) ≈ grad[2] atol = 1.0e-10
                @test dot(W_T_dy[:, q], f_vec) ≈ xq[q] atol = 1.0e-10  # analytic
            end
        end

        @testset "DerivOp(1) on Constant axis — Linear×Constant (zero derivative)" begin
            methods = (LinearInterp(), ConstantInterp())
            adj = hetero_adjoint((x, y), (xq, yq); methods = methods)

            # Constant axis: derivative is zero → all weights should be zero
            W_T_dy = Matrix(adj; deriv = (EvalValue(), DerivOp(1)))
            @test maximum(abs.(W_T_dy)) < 1.0e-15
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # OOB / EXTRAPOLATION
    # ════════════════════════════════════════════════════════════════════════

    @testset "Extrapolation" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        data = rand(10, 8)
        f_vec = vec(data)
        methods = (CubicInterp(), LinearInterp())

        @testset "ClampExtrap" begin
            adj = hetero_adjoint(
                (x, y), ((1.5, 0.5),);
                methods = methods, extrap = ClampExtrap()
            )
            itp = interp((x, y), data; method = methods, extrap = ClampExtrap())
            W_T = Matrix(adj)
            @test itp((1.5, 0.5)) ≈ dot(W_T[:, 1], f_vec) atol = 1.0e-12
        end

        @testset "FillExtrap — all weights zero" begin
            adj = hetero_adjoint(
                (x, y), ((-0.1, 0.5),);
                methods = methods, extrap = FillExtrap(0.0)
            )
            W_T = Matrix(adj)
            @test maximum(abs.(W_T)) < 1.0e-15
        end

        @testset "Per-axis extrap" begin
            adj = hetero_adjoint(
                (x, y), ((1.5, 0.5),);
                methods = methods, extrap = (ClampExtrap(), NoExtrap())
            )
            itp = interp((x, y), data; method = methods, extrap = (ClampExtrap(), NoExtrap()))
            W_T = Matrix(adj)
            @test itp((1.5, 0.5)) ≈ dot(W_T[:, 1], f_vec) atol = 1.0e-12
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # FLOAT32 SUPPORT
    # ════════════════════════════════════════════════════════════════════════

    @testset "Float32" begin
        x32 = Float32.(range(0.0, 1.0, 10))
        y32 = Float32.(range(0.0, 1.0, 8))
        data32 = Float32[sin(xi) * cos(yj) for xi in x32, yj in y32]
        f_vec32 = vec(data32)
        methods = (CubicInterp(), LinearInterp())

        itp32 = interp((x32, y32), data32; method = methods)
        adj32 = hetero_adjoint((x32, y32), ((0.3f0, 0.5f0),); methods = methods)
        W_T32 = Matrix(adj32)
        @test eltype(W_T32) == Float32
        @test itp32((0.3f0, 0.5f0)) ≈ dot(W_T32[:, 1], f_vec32) atol = 1.0f-5
    end

    # ════════════════════════════════════════════════════════════════════════
    # ANALYTIC COMPARISON — polynomial test functions
    # ════════════════════════════════════════════════════════════════════════

    @testset "Analytic — bilinear f(x,y)=x*y" begin
        # Bilinear: exact for Cubic×Linear and Linear×Linear
        x = range(0.0, 1.0, 5)
        y = range(0.0, 1.0, 5)
        data = [xi * yj for xi in x, yj in y]
        f_vec = vec(data)
        q = (0.37, 0.63)

        for methods in [
                (CubicInterp(), LinearInterp()),
                (LinearInterp(), LinearInterp()),
                (LinearInterp(), CubicInterp()),
            ]
            itp = interp((x, y), data; method = methods)
            adj = hetero_adjoint((x, y), (q,); methods = methods)

            # Value should be exact: f(0.37, 0.63) = 0.37 * 0.63
            @test itp(q) ≈ q[1] * q[2] atol = 1.0e-12

            # Gradient: ∂f/∂x = y, ∂f/∂y = x
            W_T_dx = Matrix(adj; deriv = (DerivOp(1), EvalValue()))
            W_T_dy = Matrix(adj; deriv = (EvalValue(), DerivOp(1)))
            @test dot(W_T_dx[:, 1], f_vec) ≈ q[2] atol = 1.0e-10
            @test dot(W_T_dy[:, 1], f_vec) ≈ q[1] atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # ALLOCATION TESTS — in-place path
    # ════════════════════════════════════════════════════════════════════════
    # Val(d) recursive dispatch ensures all heterogeneous tuple indexing
    # (methods[d], caches[d], bcs[d], etc.) uses compile-time d → concrete types.
    # All combos are zero-alloc on the in-place path (pool warmup excluded).

    @testset "Allocation — in-place apply" begin
        x_a = range(0.0, 1.0, 15)
        y_a = range(0.0, 1.0, 12)
        xq_a = [0.2, 0.5, 0.8]
        yq_a = [0.3, 0.5, 0.7]
        y_bar_a = randn(3)

        function _test_hetero_adjoint_alloc(grids, queries, f_bar, y_bar; methods)
            adj = hetero_adjoint(grids, queries; methods = methods)
            adj(f_bar, y_bar)  # warmup
            adj(f_bar, y_bar)  # warmup
            return @allocated adj(f_bar, y_bar)
        end

        @testset "Cubic×Linear (zero alloc)" begin
            fb = zeros(length(x_a), length(y_a))
            allocs = _test_hetero_adjoint_alloc(
                (x_a, y_a), (xq_a, yq_a), fb, y_bar_a;
                methods = (CubicInterp(), LinearInterp())
            )
            @test allocs <= ND_ALLOC_THRESHOLD
        end

        @testset "Cubic×Cubic (zero alloc)" begin
            fb = zeros(length(x_a), length(y_a))
            allocs = _test_hetero_adjoint_alloc(
                (x_a, y_a), (xq_a, yq_a), fb, y_bar_a;
                methods = (CubicInterp(), CubicInterp())
            )
            @test allocs <= ND_ALLOC_THRESHOLD
        end

        @testset "Linear×Linear (zero alloc)" begin
            fb = zeros(length(x_a), length(y_a))
            allocs = _test_hetero_adjoint_alloc(
                (x_a, y_a), (xq_a, yq_a), fb, y_bar_a;
                methods = (LinearInterp(), LinearInterp())
            )
            @test allocs <= ND_ALLOC_THRESHOLD
        end

        @testset "Quadratic×Constant (zero alloc)" begin
            fb = zeros(length(x_a), length(y_a))
            allocs = _test_hetero_adjoint_alloc(
                (x_a, y_a), (xq_a, yq_a), fb, y_bar_a;
                methods = (QuadraticInterp(), ConstantInterp())
            )
            @test allocs <= ND_ALLOC_THRESHOLD
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # ANALYTIC COMPARISON
    # ════════════════════════════════════════════════════════════════════════

    @testset "Analytic — cubic f(x,y)=x³y" begin
        # Cubic in x, linear in y: exact for Cubic×Linear
        x = range(0.0, 1.0, 20)
        y = range(0.0, 1.0, 5)
        data = [xi^3 * yj for xi in x, yj in y]
        f_vec = vec(data)
        q = (0.37, 0.63)

        methods = (CubicInterp(), LinearInterp())
        itp = interp((x, y), data; method = methods)
        adj = hetero_adjoint((x, y), (q,); methods = methods)

        # Gradient: ∂f/∂x = 3x²y, ∂f/∂y = x³
        W_T_dx = Matrix(adj; deriv = (DerivOp(1), EvalValue()))
        W_T_dy = Matrix(adj; deriv = (EvalValue(), DerivOp(1)))
        @test dot(W_T_dx[:, 1], f_vec) ≈ 3 * q[1]^2 * q[2] atol = 1.0e-5
        @test dot(W_T_dy[:, 1], f_vec) ≈ q[1]^3 atol = 1.0e-8
    end

    # ════════════════════════════════════════════════════════════════════════
    # 4D FORWARD-ADJOINT CONSISTENCY
    # ════════════════════════════════════════════════════════════════════════
    # Validates mixed-radix decomposition for higher dimensions (prod(sizes)
    # with sizes = (2,1,2,1) for Cubic×Linear×Quadratic×Constant).

    @testset "4D forward-adjoint consistency" begin
        g1 = range(0.0, 1.0, 6)
        g2 = range(0.0, 1.0, 5)
        g3 = range(0.0, 1.0, 5)
        g4 = range(0.0, 1.0, 4)
        data4 = randn(6, 5, 5, 4)
        f_vec = vec(data4)
        q1 = [0.3, 0.7]
        q2 = [0.4, 0.6]
        q3 = [0.2, 0.8]
        q4 = [0.3, 0.7]
        nq = 2

        methods_4d = (CubicInterp(), LinearInterp(), QuadraticInterp(), ConstantInterp())
        itp = interp((g1, g2, g3, g4), data4; method = methods_4d)
        adj = hetero_adjoint((g1, g2, g3, g4), (q1, q2, q3, q4); methods = methods_4d)
        W_T = Matrix(adj)

        @testset "Forward-adjoint per query" begin
            for q in 1:nq
                fwd = itp((q1[q], q2[q], q3[q], q4[q]))
                mat_fwd = dot(W_T[:, q], f_vec)
                @test fwd ≈ mat_fwd atol = 1.0e-12
            end
        end

        @testset "Golden rule: ⟨Wf, ȳ⟩ = ⟨f, W^Tȳ⟩" begin
            Wf = [itp((q1[q], q2[q], q3[q], q4[q])) for q in 1:nq]
            y_bar = randn(nq)
            lhs = dot(Wf, y_bar)
            rhs = dot(f_vec, vec(adj(y_bar)))
            @test lhs ≈ rhs atol = 1.0e-10
        end

        @testset "Zero alloc — in-place" begin
            function _test_4d_alloc(adj, f_bar, y_bar)
                adj(f_bar, y_bar)
                adj(f_bar, y_bar)
                return @allocated adj(f_bar, y_bar)
            end
            fb = zeros(6, 5, 5, 4)
            @test _test_4d_alloc(adj, fb, [1.0, 2.0]) <= ND_ALLOC_THRESHOLD
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # COVERAGE: show methods
    # ════════════════════════════════════════════════════════════════════════

    @testset "show — HeteroAdjointND" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        adj = hetero_adjoint((x, y), ([0.5], [0.5]); methods = (CubicInterp(), LinearInterp()))

        compact_str = sprint(show, adj)
        @test occursin("HeteroAdjointND", compact_str)
        @test occursin("Float64", compact_str)
        @test occursin("Cubic×Linear", compact_str)

        verbose_str = sprint(show, MIME("text/plain"), adj)
        @test occursin("HeteroAdjointND", verbose_str)
        @test occursin("Grids:", verbose_str)
        @test occursin("Method:", verbose_str)
    end

    # ════════════════════════════════════════════════════════════════════════
    # COVERAGE: scalar y_bar → ND adjoint (scalar dispatch path)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Scalar y_bar (single query)" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        methods = (CubicInterp(), LinearInterp())
        adj = hetero_adjoint((x, y), ((0.5, 0.5),); methods = methods)

        f_bar_scalar = adj(1.0)
        f_bar_vec = adj([1.0])
        @test f_bar_scalar ≈ f_bar_vec atol = 1.0e-14
    end

    # ════════════════════════════════════════════════════════════════════════
    # COVERAGE: OOB weight fixup — FillExtrap zeros ALL weights
    # ════════════════════════════════════════════════════════════════════════

    @testset "FillExtrap — per-axis OOB baking" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        methods = (CubicInterp(), LinearInterp())

        # Both axes OOB (x=-0.1, y=1.5) with FillExtrap → all weights zero
        adj = hetero_adjoint(
            (x, y), ((-0.1, 1.5),);
            methods = methods, extrap = FillExtrap(0.0)
        )
        W_T = Matrix(adj)
        @test maximum(abs.(W_T)) < 1.0e-15

        # Only one axis OOB (x in-bounds, y OOB)
        adj2 = hetero_adjoint(
            (x, y), ((0.5, 1.5),);
            methods = methods, extrap = FillExtrap(0.0)
        )
        W_T2 = Matrix(adj2)
        @test maximum(abs.(W_T2)) < 1.0e-15

        # ClampExtrap — derivative weights zeroed, value weight preserved
        adj_clamp = hetero_adjoint(
            (x, y), ((1.5, 0.5),);
            methods = methods, extrap = ClampExtrap()
        )
        W_T_clamp = Matrix(adj_clamp)
        @test sum(abs.(W_T_clamp)) > 0  # value weights non-zero

        # Derivative at clamped OOB → all zero (w1 zeroed)
        W_T_dx = Matrix(adj_clamp; deriv = (DerivOp(1), EvalValue()))
        @test maximum(abs.(W_T_dx)) < 1.0e-15
    end

    # ════════════════════════════════════════════════════════════════════════
    # COVERAGE: ConstantInterp weight functions (all side types)
    # ════════════════════════════════════════════════════════════════════════

    @testset "ConstantInterp sides — weight coverage" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        data = [xi + yj for xi in x, yj in y]
        f_vec = vec(data)

        for side in [LeftSide(), RightSide(), NearestSide()]
            @testset "$(typeof(side))" begin
                methods = (LinearInterp(), ConstantInterp(side = side))
                itp = interp((x, y), data; method = methods)
                adj = hetero_adjoint((x, y), ([0.5], [0.5]); methods = methods)
                W_T = Matrix(adj)
                @test itp((0.5, 0.5)) ≈ dot(W_T[:, 1], f_vec) atol = 1.0e-12
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # COVERAGE: PeriodicBC{:exclusive} with Vector grids (vcat branch)
    # ════════════════════════════════════════════════════════════════════════

    @testset "PeriodicBC{:exclusive} — Vector grid (non-Range)" begin
        n_p = 12
        x_p_range = range(0.0, 2π, n_p + 1)[1:n_p]
        x_p = collect(x_p_range)  # Vector, not Range → triggers vcat branch
        y = range(0.0, 1.0, 8)
        data = [sin(xi) * yj for xi in x_p, yj in y]
        f_vec = vec(data)

        bc_excl = PeriodicBC(endpoint = :exclusive, period = 2π)
        methods = (CubicInterp(bc = bc_excl), LinearInterp())
        itp = interp((x_p, y), data; method = methods)
        adj = hetero_adjoint((x_p, y), ([1.0], [0.5]); methods = methods)
        W_T = Matrix(adj)
        @test size(adj(1.0)) == size(data)
        @test itp((1.0, 0.5)) ≈ dot(W_T[:, 1], f_vec) atol = 1.0e-10
    end

    # ════════════════════════════════════════════════════════════════════════
    # COVERAGE: PeriodicBC{:exclusive} on a Linear axis — full forward+adjoint
    # — exercises the `LinearInterp(bc=...)` tag-struct field end-to-end:
    #   (1) Hetero forward wraps via `_cache_axis_for_method`.
    #   (2) Hetero adjoint wraps via `_cache_axis` in `grids_ext`, the generic
    #       seam-fold post-apply hook trims back to user-`n` via
    #       `_adjoint_output_size`.
    # Mixed Linear+Cubic so we cover both wrap strategies in a single call.
    # ════════════════════════════════════════════════════════════════════════

    @testset "PeriodicBC{:exclusive} on Linear axis (mixed Linear+Cubic)" begin
        nx, ny = 12, 10
        n_query = 18
        x = collect(range(0.0, step = 2π / nx, length = nx))   # exclusive n-point
        y = range(0.0, 1.0, ny)
        data = [sin(xi) * yj for xi in x, yj in y]
        f_vec = vec(data)

        bc_x = PeriodicBC(endpoint = :exclusive, period = 2π)
        methods = (LinearInterp(bc = bc_x), CubicInterp(bc = ZeroSlopeBC()))

        # Forward — verify wrap-around eval works through the unified API
        itp = interp((x, y), data; method = methods)
        # Query past inner-grid last knot (= 11·2π/12 ≈ 5.76) into the seam
        # region [x[end], x[1]+period).
        xq_wrap = x[end] + 0.34
        v_wrap = itp((xq_wrap, 0.5))
        α = (xq_wrap - x[end]) / (2π - x[end])
        v_expected = (1 - α) * (sin(x[end]) * 0.5) + α * 0.0
        @test v_wrap ≈ v_expected atol = 1.0e-10

        # Adjoint — dot-product identity ⟨W·f, ȳ⟩ = ⟨f, Wᵀ·ȳ⟩.
        # Queries cover both inner and seam (wrap) regions to exercise the
        # full periodic adjoint path.
        xq = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
        yq = sort(rand(n_query)) .* 0.96 .+ 0.02
        adj = hetero_adjoint((x, y), (xq, yq); methods = methods)
        # Adjoint output shape = user's n-point shape (n+1 → n trim happens)
        @test size(adj(zeros(n_query))) == size(data)
        W_T = Matrix(adj)
        @test size(W_T) == (length(data), n_query)
        for k in 1:n_query
            @test itp((xq[k], yq[k])) ≈ dot(W_T[:, k], f_vec) atol = 1.0e-10
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # COVERAGE: PolyFit BC validation error path
    # ════════════════════════════════════════════════════════════════════════

    @testset "PolyFit BC validation" begin
        # CubicFit needs 4 points, QuadraticFit needs 3, LinearFit needs 2
        y = range(0.0, 1.0, 10)

        # Too few points for CubicFit (need 4, give 3)
        x_small = range(0.0, 1.0, 3)
        @test_throws ArgumentError hetero_adjoint(
            (x_small, y), ([0.5], [0.5]);
            methods = (CubicInterp(bc = CubicFit()), LinearInterp())
        )

        # Enough points — should work
        x_ok = range(0.0, 1.0, 5)
        adj = hetero_adjoint(
            (x_ok, y), ([0.5], [0.5]);
            methods = (CubicInterp(bc = CubicFit()), LinearInterp())
        )
        @test adj(1.0) isa Matrix
    end

    # ════════════════════════════════════════════════════════════════════════
    # COVERAGE: Cubic×Cubic mixed-partial path (p_src > 1, mixed_cache_d)
    # ════════════════════════════════════════════════════════════════════════

    @testset "Mixed-partial path (Cubic×Cubic, p_src > 1)" begin
        x = range(0.0, 1.0, 12)
        y = range(0.0, 1.0, 10)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
        f_vec = vec(data)

        # Cubic×Cubic triggers sizes=(2,2) → stride_d=2 → p_src ∈ {1,2}
        methods = (CubicInterp(), CubicInterp())
        itp = interp((x, y), data; method = methods)
        adj = hetero_adjoint((x, y), ([0.37], [0.63]); methods = methods)

        # Forward-adjoint consistency (exercises p_src=1 AND p_src=2)
        W_T = Matrix(adj)
        @test itp((0.37, 0.63)) ≈ dot(W_T[:, 1], f_vec) atol = 1.0e-12

        # Mixed partial: ∂²f/∂x∂y (requires p_src=2 for mixed cache)
        W_T_dxdy = Matrix(adj; deriv = (DerivOp(1), DerivOp(1)))
        H = FastInterpolations.hessian(itp, (0.37, 0.63))
        @test dot(W_T_dxdy[:, 1], f_vec) ≈ H[1, 2] atol = 1.0e-9
    end

end # @testset "HeteroAdjointND"
