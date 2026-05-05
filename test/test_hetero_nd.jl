@testitem "HeteroInterpolantND" begin
    # ========================================
    # Test Setup — separable functions for exact verification
    # ========================================
    x = range(0.0, 2π, 30)
    y = range(0.0, π, 25)
    z = range(0.0, 1.0, 20)

    g(xi) = sin(xi)
    h(yj) = cos(yj)
    k(zk) = zk^2 + 1.0

    data_2d = [g(xi) * h(yj) for xi in x, yj in y]
    data_3d = [g(xi) * h(yj) * k(zk) for xi in x, yj in y, zk in z]

    qx, qy, qz = 1.7, 0.8, 0.45

    # ========================================
    # 1. Homogeneous equivalence — all-cubic
    # ========================================
    @testset "Homogeneous: all-cubic matches CubicInterpolantND" begin
        itp_ref = cubic_interp((x, y), data_2d)
        itp_tp = interp((x, y), data_2d; method = (CubicInterp(), CubicInterp()))

        @test itp_tp isa CubicInterpolantND  # auto-dispatched to existing type
        @test itp_tp((qx, qy)) ≈ itp_ref((qx, qy)) rtol = 1.0e-12
    end

    # ========================================
    # 2. Homogeneous equivalence — all-linear
    # ========================================
    @testset "Homogeneous: all-linear matches LinearInterpolantND" begin
        itp_ref = linear_interp((x, y), data_2d)
        itp_tp = interp((x, y), data_2d; method = (LinearInterp(), LinearInterp()))

        @test itp_tp((qx, qy)) ≈ itp_ref((qx, qy)) rtol = 1.0e-12
    end

    # ========================================
    # 3. Homogeneous equivalence — all-quadratic
    # ========================================
    @testset "Homogeneous: all-quadratic matches QuadraticInterpolantND" begin
        itp_ref = quadratic_interp((x, y), data_2d)
        itp_tp = interp((x, y), data_2d; method = (QuadraticInterp(), QuadraticInterp()))

        @test itp_tp((qx, qy)) ≈ itp_ref((qx, qy)) rtol = 1.0e-12
    end

    # ========================================
    # 4. Homogeneous equivalence — all-constant
    # ========================================
    @testset "Homogeneous: all-constant matches ConstantInterpolantND" begin
        itp_ref = constant_interp((x, y), data_2d)
        itp_tp = interp((x, y), data_2d; method = (ConstantInterp(), ConstantInterp()))

        @test itp_tp((qx, qy)) ≈ itp_ref((qx, qy)) rtol = 1.0e-12
    end

    # ========================================
    # 5. Heterogeneous: Cubic × Linear (2D)
    # ========================================
    @testset "Heterogeneous: Cubic × Linear on separable function" begin
        itp_tp = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))

        # For separable f(x,y) = g(x)*h(y), tensor product gives:
        # itp(qx,qy) = cubic_interp(x,g)(qx) * linear_interp(y,h)(qy)
        g_vals = [g(xi) for xi in x]
        h_vals = [h(yj) for yj in y]
        expected = cubic_interp(x, g_vals)(qx) * linear_interp(y, h_vals)(qy)

        @test itp_tp((qx, qy)) ≈ expected rtol = 1.0e-12
    end

    # ========================================
    # 6. Heterogeneous: Linear × Cubic (2D, swapped)
    # ========================================
    @testset "Heterogeneous: Linear × Cubic (swapped axes)" begin
        itp_tp = interp((x, y), data_2d; method = (LinearInterp(), CubicInterp()))

        g_vals = [g(xi) for xi in x]
        h_vals = [h(yj) for yj in y]
        expected = linear_interp(x, g_vals)(qx) * cubic_interp(y, h_vals)(qy)

        @test itp_tp((qx, qy)) ≈ expected rtol = 1.0e-12
    end

    # ========================================
    # 7. Heterogeneous: 3D Cubic × Linear × Quadratic
    # ========================================
    @testset "Heterogeneous: 3D Cubic × Linear × Quadratic" begin
        itp_tp = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), LinearInterp(), QuadraticInterp()),
        )

        g_vals = [g(xi) for xi in x]
        h_vals = [h(yj) for yj in y]
        k_vals = [k(zk) for zk in z]
        expected = cubic_interp(x, g_vals)(qx) *
            linear_interp(y, h_vals)(qy) *
            quadratic_interp(z, k_vals)(qz)

        @test itp_tp((qx, qy, qz)) ≈ expected rtol = 1.0e-10
    end

    # ========================================
    # 8-9. Derivatives on heterogeneous (Cubic × Linear)
    # ========================================
    @testset "Derivatives: ∂f/∂x on Cubic × Linear" begin
        itp_tp = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))

        g_vals = [g(xi) for xi in x]
        h_vals = [h(yj) for yj in y]
        # ∂f/∂x = g'(x) * h(y) via cubic derivative × linear value
        expected = cubic_interp(x, g_vals)(qx; deriv = DerivOp(1)) *
            linear_interp(y, h_vals)(qy)

        result = itp_tp((qx, qy); deriv = (DerivOp(1), DerivOp(0)))
        @test result ≈ expected rtol = 1.0e-10
    end

    @testset "Derivatives: ∂f/∂y on Cubic × Linear" begin
        itp_tp = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))

        g_vals = [g(xi) for xi in x]
        h_vals = [h(yj) for yj in y]
        # ∂f/∂y = g(x) * h'(y) via cubic value × linear derivative
        expected = cubic_interp(x, g_vals)(qx) *
            linear_interp(y, h_vals)(qy; deriv = DerivOp(1))

        result = itp_tp((qx, qy); deriv = (DerivOp(0), DerivOp(1)))
        @test result ≈ expected rtol = 1.0e-10
    end

    # ========================================
    # 10. gradient() compatibility
    # ========================================
    @testset "gradient() on HeteroInterpolantND" begin
        itp_tp = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))

        grad = gradient(itp_tp, (qx, qy))

        dfdx = itp_tp((qx, qy); deriv = (DerivOp(1), DerivOp(0)))
        dfdy = itp_tp((qx, qy); deriv = (DerivOp(0), DerivOp(1)))

        @test grad[1] ≈ dfdx rtol = 1.0e-12
        @test grad[2] ≈ dfdy rtol = 1.0e-12
    end

    # ========================================
    # 11. Extrapolation per axis
    # ========================================
    @testset "Per-axis extrapolation" begin
        itp_clamp_noextrap = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()),
            extrap = (ClampExtrap(), NoExtrap()),
        )

        # OOB on axis 1 → clamps (no error)
        @test itp_clamp_noextrap((-1.0, qy)) isa Float64

        # OOB on axis 2 → throws (NoExtrap)
        @test_throws Exception itp_clamp_noextrap((qx, -1.0))
    end

    # ========================================
    # 12. Minimum grid size validation
    # ========================================
    @testset "Grid size validation" begin
        tiny_grid = range(0.0, 1.0, 2)
        tiny_data = [1.0, 2.0]

        # Cubic needs ≥4 points
        @test_throws ArgumentError interp(
            (tiny_grid,), reshape(tiny_data, 2);
            method = (CubicInterp(),),
        )

        # Linear needs ≥2 points — should work
        itp = interp((tiny_grid,), reshape(tiny_data, 2); method = (LinearInterp(),))
        @test itp((0.5,)) isa Float64
    end

    # ========================================
    # 13. Vararg callable
    # ========================================
    @testset "Vararg callable form" begin
        itp_tp = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))

        # itp(qx, qy) == itp((qx, qy))
        @test itp_tp(qx, qy) ≈ itp_tp((qx, qy)) rtol = 1.0e-15
    end

    # ========================================
    # 14. Show method
    # ========================================
    @testset "Show method" begin
        itp_tp = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))

        # 1-arg show (compact)
        str = sprint(show, itp_tp)
        @test occursin("HeteroInterpolantND", str)
        @test occursin("Cubic", str)
        @test occursin("Linear", str)
        @test occursin("30×25", str)

        # 2-arg show (REPL text/plain)
        str_full = sprint(show, MIME("text/plain"), itp_tp)
        @test occursin("HeteroInterpolantND", str_full)
        @test occursin("Axis 1: Cubic", str_full)
        @test occursin("Axis 2: Linear", str_full)

        # With non-default extrap (shows extrap label)
        itp_ext = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()),
            extrap = (ClampExtrap(), NoExtrap()),
        )
        str_ext = sprint(show, MIME("text/plain"), itp_ext)
        @test occursin("ClampExtrap", str_ext)
        @test occursin("NoExtrap", str_ext)   # Per-axis detail shows all extraps

        # All 4 method name variants
        itp_q = interp(
            (x, y), data_2d;
            method = (QuadraticInterp(), ConstantInterp()),
        )
        str_q = sprint(show, itp_q)
        @test occursin("Quadratic", str_q)
        @test occursin("Constant", str_q)
    end

    # ========================================
    # 15-17. Analytic Exactness Tests
    # ========================================
    # Cubic reproduces ≤3rd-order polynomials exactly,
    # Linear reproduces ≤1st-order exactly,
    # Quadratic reproduces ≤2nd-order exactly.
    # For separable f(x,y) = pₘ(x) * pₙ(y), the tensor product
    # must be exact when each axis method matches its polynomial degree.

    @testset "Exactness: Cubic × Linear on p3(x) * p1(y)" begin
        p3(xi) = 2xi^3 - 3xi^2 + xi - 1
        p1(yj) = 4yj + 7
        dp3(xi) = 6xi^2 - 6xi + 1

        xg = range(-1.0, 3.0, 20)
        yg = range(0.0, 5.0, 15)
        data_exact = [p3(xi) * p1(yj) for xi in xg, yj in yg]
        itp_e = interp((xg, yg), data_exact; method = (CubicInterp(), LinearInterp()))

        for qxi in range(-0.9, 2.9, 15), qyj in range(0.1, 4.9, 15)
            @test itp_e((qxi, qyj)) ≈ p3(qxi) * p1(qyj) atol = 1.0e-10
            @test itp_e((qxi, qyj); deriv = (DerivOp(1), DerivOp(0))) ≈
                dp3(qxi) * p1(qyj) atol = 1.0e-8
            @test itp_e((qxi, qyj); deriv = (DerivOp(0), DerivOp(1))) ≈
                p3(qxi) * 4.0 atol = 1.0e-10
        end
    end

    @testset "Exactness: Linear × Cubic on p1(x) * p3(y)" begin
        p1(xi) = 3xi + 2
        p3(yj) = yj^3 - 2yj^2 + yj

        xg = range(-1.0, 3.0, 20)
        yg = range(0.0, 5.0, 15)
        data_exact = [p1(xi) * p3(yj) for xi in xg, yj in yg]
        itp_e = interp((xg, yg), data_exact; method = (LinearInterp(), CubicInterp()))

        for qxi in range(-0.9, 2.9, 12), qyj in range(0.1, 4.9, 12)
            @test itp_e((qxi, qyj)) ≈ p1(qxi) * p3(qyj) atol = 1.0e-10
        end
    end

    @testset "Exactness: 3D Cubic × Linear × Quadratic" begin
        p3(xi) = xi^3 - xi
        p1(yj) = 2yj + 1
        p2(zk) = zk^2 - zk
        f3d(xi, yj, zk) = p3(xi) * p1(yj) * p2(zk)

        xg = range(-1.0, 3.0, 20)
        yg = range(0.0, 5.0, 15)
        zg = range(0.0, 2.0, 12)
        data_3d_exact = [f3d(xi, yj, zk) for xi in xg, yj in yg, zk in zg]
        itp_e = interp(
            (xg, yg, zg), data_3d_exact;
            method = (CubicInterp(), LinearInterp(), QuadraticInterp()),
        )

        for qxi in range(-0.5, 2.5, 8), qyj in range(0.5, 4.5, 8), qzk in range(0.1, 1.9, 8)
            @test itp_e((qxi, qyj, qzk)) ≈ f3d(qxi, qyj, qzk) atol = 1.0e-8
        end
    end

    # ========================================
    # 18. Auto-dispatch: homogeneous → existing types
    # ========================================
    @testset "Auto-dispatch: homo → existing types" begin
        @test interp((x, y), data_2d; method = (CubicInterp(), CubicInterp())) isa
            CubicInterpolantND
        @test interp((x, y), data_2d; method = (LinearInterp(), LinearInterp())) isa
            LinearInterpolantND
        @test interp((x, y), data_2d; method = (QuadraticInterp(), QuadraticInterp())) isa
            QuadraticInterpolantND
        @test interp((x, y), data_2d; method = (ConstantInterp(), ConstantInterp())) isa
            ConstantInterpolantND

        # Hetero → HeteroInterpolantND
        @test interp((x, y), data_2d; method = (CubicInterp(), LinearInterp())) isa
            HeteroInterpolantND
    end

    # ========================================
    # 19. Single broadcast: methods=CubicInterp()
    # ========================================
    @testset "Single broadcast: methods=single value" begin
        @test interp((x, y), data_2d; method = CubicInterp()) isa CubicInterpolantND
        @test interp((x, y), data_2d; method = LinearInterp()) isa LinearInterpolantND

        # Result equivalence
        itp_nd = interp((x, y), data_2d; method = CubicInterp())
        itp_direct = cubic_interp((x, y), data_2d)
        @test itp_nd((qx, qy)) ≈ itp_direct((qx, qy))
    end

    # ========================================
    # 20. Per-axis BC forwarding
    # ========================================
    @testset "Per-axis BC forwarding" begin
        itp_nd = interp(
            (x, y), data_2d;
            method = (CubicInterp(CubicFit()), CubicInterp(ZeroCurvBC())),
        )
        itp_direct = cubic_interp((x, y), data_2d; bc = (CubicFit(), ZeroCurvBC()))

        @test itp_nd isa CubicInterpolantND
        @test itp_nd((qx, qy)) ≈ itp_direct((qx, qy))
    end

    # ========================================
    # 21. hessian() / laplacian() on OnTheFly
    # ========================================
    @testset "hessian/laplacian on OnTheFly Cubic × Linear" begin
        itp_tp = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))

        H = hessian(itp_tp, (qx, qy))
        @test size(H) == (2, 2)
        @test H[1, 2] ≈ H[2, 1] rtol = 1.0e-12   # symmetry

        # H[2,2] = ∂²f/∂y² on linear axis → must be 0
        @test H[2, 2] == 0.0

        # H[1,1] = ∂²f/∂x² on cubic axis → non-zero
        @test H[1, 1] != 0.0

        # laplacian = tr(H)
        lap = laplacian(itp_tp, (qx, qy))
        @test lap ≈ H[1, 1] + H[2, 2] rtol = 1.0e-12
    end

    # ========================================
    # 22. 1D interp (N=1 edge case)
    # ========================================
    @testset "1D interp dispatches to existing 1D types" begin
        x1d = range(0.0, 5.0, 20)
        data_1d = [sin(xi) for xi in x1d]

        # Cubic
        itp_c = interp((x1d,), reshape(data_1d, :); method = (CubicInterp(),))
        itp_ref = cubic_interp(x1d, data_1d)
        @test itp_c((2.3,)) ≈ itp_ref(2.3) rtol = 1.0e-14

        # Linear
        itp_l = interp((x1d,), reshape(data_1d, :); method = (LinearInterp(),))
        itp_lref = linear_interp(x1d, data_1d)
        @test itp_l((2.3,)) ≈ itp_lref(2.3) rtol = 1.0e-14
    end

    # ========================================
    # 23. Float32 data and grids
    # ========================================
    @testset "Float32 data and grids" begin
        x32 = range(0.0f0, 2.0f0 * Float32(π), 30)
        y32 = range(0.0f0, Float32(π), 25)
        data32 = [sin(xi) * cos(yj) for xi in x32, yj in y32]

        itp32 = interp((x32, y32), data32; method = (CubicInterp(), LinearInterp()))
        val = itp32((1.0f0, 0.5f0))
        @test val isa Float32
        @test val ≈ sin(1.0f0) * cos(0.5f0) atol = 0.01f0
    end

    # ========================================
    # 24. Mixed grid types (Range + Vector)
    # ========================================
    @testset "Mixed grid types: Range × Vector" begin
        x_range = range(0.0, 2π, 30)                         # AbstractRange
        y_vec = collect(range(0.0, π, 25))                    # Vector{Float64}

        data_mixed = [sin(xi) * cos(yj) for xi in x_range, yj in y_vec]

        itp_mixed = interp(
            (x_range, y_vec), data_mixed;
            method = (CubicInterp(), LinearInterp())
        )

        # Reference: separate 1D interps
        g_vals = [sin(xi) for xi in x_range]
        h_vals = [cos(yj) for yj in y_vec]
        expected = cubic_interp(x_range, g_vals)(qx) * linear_interp(y_vec, h_vals)(qy)

        @test itp_mixed((qx, qy)) ≈ expected rtol = 1.0e-12
    end

    # ========================================
    # OnTheFly HeteroInterpolantND Coverage
    # ========================================
    # Exercises: _locate_cell/<:Array, _eval_at_cell/<:Array, _first_hint(::Tuple),
    # _tail_hints(::Tuple), _oneshot_eval_1d(::QuadraticInterp, ...)

    @testset "OnTheFly: gradient (locate_cell + eval_at_cell)" begin
        itp_otf = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        g = gradient(itp_otf, (qx, qy))
        # Reference: PreCompute gradient
        itp_ref = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))
        g_ref = gradient(itp_ref, (qx, qy))
        @test g[1] ≈ g_ref[1] rtol = 1.0e-10
        @test g[2] ≈ g_ref[2] rtol = 1.0e-10
    end

    @testset "OnTheFly: hessian" begin
        itp_otf = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        H = hessian(itp_otf, (qx, qy))
        itp_ref = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))
        H_ref = hessian(itp_ref, (qx, qy))
        @test H ≈ H_ref rtol = 1.0e-8
    end

    @testset "OnTheFly: laplacian" begin
        itp_otf = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        L = laplacian(itp_otf, (qx, qy))
        itp_ref = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))
        L_ref = laplacian(itp_ref, (qx, qy))
        @test L ≈ L_ref rtol = 1.0e-8
    end

    @testset "OnTheFly: value_gradient" begin
        itp_otf = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        val, g = value_gradient(itp_otf, (qx, qy))
        @test val ≈ itp_otf((qx, qy)) rtol = 1.0e-14
        g_ref = gradient(itp_otf, (qx, qy))
        @test g[1] ≈ g_ref[1] rtol = 1.0e-14
        @test g[2] ≈ g_ref[2] rtol = 1.0e-14
    end

    @testset "OnTheFly: gradient! in-place" begin
        itp_otf = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        G = zeros(2)
        gradient!(G, itp_otf, (qx, qy))
        g_ref = gradient(itp_otf, (qx, qy))
        @test G[1] ≈ g_ref[1] rtol = 1.0e-14
        @test G[2] ≈ g_ref[2] rtol = 1.0e-14
    end

    @testset "OnTheFly: hessian! in-place" begin
        itp_otf = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        H = zeros(2, 2)
        hessian!(H, itp_otf, (qx, qy))
        H_ref = hessian(itp_otf, (qx, qy))
        @test H ≈ H_ref rtol = 1.0e-14
    end

    @testset "OnTheFly: with hints (_first_hint/::Tuple, _tail_hints/::Tuple)" begin
        itp_otf = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        hint = (Ref(1), Ref(1))
        val = itp_otf((qx, qy); hint = hint)
        ref = itp_otf((qx, qy))
        @test val ≈ ref rtol = 1.0e-14
        # Hints should be updated
        @test hint[1][] > 1
        @test hint[2][] >= 1
    end

    @testset "OnTheFly: QuadraticInterp fiber eval" begin
        itp_q = interp(
            (x, y), data_2d;
            method = (QuadraticInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        val = itp_q((qx, qy))
        # Reference: separate 1D one-shot interps on separable data
        ref_x = quadratic_interp(x, [sin(xi) for xi in x])(qx)
        ref_y = linear_interp(y, [cos(yj) for yj in y])(qy)
        @test val ≈ ref_x * ref_y rtol = 1.0e-10
    end
end

# ════════════════════════════════════════════════════════════════
# PR1 (`refac/cleanup_nd_spacing`) lock-down: spacings field
# removed from forward struct. Asserts field absence,
# type-parameter count, type stability, and zero-allocation
# persistent eval. Hetero is constructed via `interp(... ; coeffs=OnTheFly())`
# (PreCompute path dispatches to specialized homogeneous structs).
# ════════════════════════════════════════════════════════════════
@testitem "HeteroInterpolantND — spacings cleanup lock-down" setup = [AllocConstants] begin
    using FastInterpolations: interp, LinearInterp, PchipInterp, OnTheFly

    @testset "spacings field removed" begin
        x = 0.0:1.0:3.0
        y = 0.0:1.0:3.0
        data = [Float64(i + j) for i in 1:4, j in 1:4]
        # Heterogeneous methods → HeteroInterpolantND (homogeneous would dispatch to
        # the specialized struct).
        itp = interp((x, y), data; method = (PchipInterp(), LinearInterp()), coeffs = OnTheFly())

        @test !hasfield(typeof(itp), :spacings)
        # Was 9 (Tg, Tv, N, G, S, M, E, P, D), now 8 (drops S)
        @test length(typeof(itp).parameters) == 8
        @test isfinite(itp((1.5, 1.5)))
    end

    @testset "type stability (@inferred)" begin
        x_rng = 0.0:1.0:3.0
        x_vec = [0.0, 1.0, 2.0, 3.0]
        data = [Float64(i + j) for i in 1:4, j in 1:4]

        itp_rng = interp((x_rng, x_rng), data; method = (PchipInterp(), LinearInterp()), coeffs = OnTheFly())
        itp_vec = interp((x_vec, x_vec), data; method = (PchipInterp(), LinearInterp()), coeffs = OnTheFly())

        @test (@inferred itp_rng((0.5, 0.5))) isa Float64
        @test (@inferred itp_vec((0.5, 0.5))) isa Float64
    end

    @testset "zero-alloc persistent eval" begin
        x = 0.0:1.0:3.0
        y = 0.0:1.0:3.0
        data = [Float64(i + j) for i in 1:4, j in 1:4]
        itp = interp((x, y), data; method = (PchipInterp(), LinearInterp()), coeffs = OnTheFly())
        itp((0.5, 0.5))
        itp((0.5, 0.5))

        @test (@allocated itp((0.5, 0.5))) <= ALLOC_THRESHOLD
    end
end
