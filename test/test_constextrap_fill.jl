@testitem "FillExtrap Fill Value" begin
    # ────────────────────────────────────────────
    # Construction & Type Hierarchy
    # ────────────────────────────────────────────
    @testset "FillExtrap type hierarchy" begin
        # FillExtrap is a concrete type under AbstractExtrap
        @test ClampExtrap <: AbstractExtrap
        @test FillExtrap <: AbstractExtrap
        @test !(ClampExtrap <: FillExtrap)
        @test !(FillExtrap <: ClampExtrap)

        # No-arg factory: boundary clamp
        e0 = ClampExtrap()
        @test e0 isa ClampExtrap

        # Float value: auto-promote Int → Float64
        e1 = FillExtrap(0)
        @test e1.fill_value === 0.0
        @test e1 isa FillExtrap{Float64}

        # Float64
        e2 = FillExtrap(NaN)
        @test isnan(e2.fill_value)
        @test e2 isa FillExtrap{Float64}

        # Float32
        e3 = FillExtrap(0.0f0)
        @test e3.fill_value === 0.0f0
        @test e3 isa FillExtrap{Float32}

        # Kwarg form
        e4 = FillExtrap(; fill_value = NaN)
        @test isnan(e4.fill_value)
        @test e4 isa FillExtrap{Float64}

        # Kwarg with nothing → FillExtrap{Nothing}
        e5 = FillExtrap(; fill_value = nothing)
        @test e5 isa FillExtrap{Nothing}

        # Direct construction
        @test ClampExtrap() isa ClampExtrap
        @test FillExtrap(NaN) isa FillExtrap{Float64}
        @test FillExtrap(0.0f0) isa FillExtrap{Float32}
    end

    @testset "_promote_extrap" begin
        using FastInterpolations: _promote_extrap

        # Non-FillExtrap passes through
        @test _promote_extrap(NoExtrap(), Float64) === NoExtrap()
        @test _promote_extrap(ExtendExtrap(), Float64) === ExtendExtrap()

        # ClampExtrap passes through regardless of Tv
        e_clamp = ClampExtrap()
        @test _promote_extrap(e_clamp, Float64) === e_clamp
        @test _promote_extrap(e_clamp, Float32) === e_clamp

        # Fill value gets converted to Tv
        e_f64 = FillExtrap(0.0)
        e_promoted = _promote_extrap(e_f64, Float32)
        @test e_promoted isa FillExtrap{Float32}
        @test e_promoted.fill_value === 0.0f0

        # NaN promotion
        e_nan = FillExtrap(NaN)
        e_nan32 = _promote_extrap(e_nan, Float32)
        @test e_nan32 isa FillExtrap{Float32}
        @test isnan(e_nan32.fill_value)
    end

    # ────────────────────────────────────────────
    # Evaluation: all 4 interpolation types
    # ────────────────────────────────────────────
    x = collect(range(0.0, 5.0, length = 11))
    y = sin.(x)
    x_below = -1.0
    x_above = 6.0

    @testset "Cubic with fill value" begin
        # NaN fill
        itp_nan = cubic_interp(x, y; extrap = FillExtrap(NaN))
        @test isnan(itp_nan(x_below))
        @test isnan(itp_nan(x_above))
        # In-domain should work normally
        @test itp_nan(2.5) ≈ cubic_interp(x, y; extrap = ClampExtrap())(2.5)

        # Zero fill
        itp_zero = cubic_interp(x, y; extrap = FillExtrap(0.0))
        @test itp_zero(x_below) === 0.0
        @test itp_zero(x_above) === 0.0

        # Custom fill
        itp_42 = cubic_interp(x, y; extrap = FillExtrap(42.0))
        @test itp_42(x_below) === 42.0
        @test itp_42(x_above) === 42.0

        # Derivative contract under FillExtrap (cell-local "fill_value-as-data"):
        # OOB cell's "data" is `fill_value`, so deriv = `0 * fill_value`. Finite
        # fill → 0; NaN fill propagates (`0 * NaN = NaN`).
        @test isnan(itp_nan(x_below; deriv = DerivOp(1)))
        @test isnan(itp_nan(x_above; deriv = DerivOp(1)))
        @test isnan(itp_nan(x_below; deriv = DerivOp(2)))
        @test isnan(itp_nan(x_above; deriv = DerivOp(2)))
        # Finite fill (zero / 42) → deriv 0.
        @test iszero(itp_zero(x_below; deriv = DerivOp(1)))
        @test iszero(itp_42(x_above; deriv = DerivOp(1)))
    end

    @testset "Linear with fill value" begin
        itp = linear_interp(x, y; extrap = FillExtrap(NaN))
        @test isnan(itp(x_below))
        @test isnan(itp(x_above))
        @test itp(2.5) ≈ linear_interp(x, y; extrap = ClampExtrap())(2.5)

        # NaN fill_value × deriv → NaN (fill_value-as-data contract)
        @test isnan(itp(x_below; deriv = DerivOp(1)))
    end

    @testset "Quadratic with fill value" begin
        itp = quadratic_interp(x, y; extrap = FillExtrap(-1.0))
        @test itp(x_below) === -1.0
        @test itp(x_above) === -1.0
        @test itp(2.5) ≈ quadratic_interp(x, y; extrap = ClampExtrap())(2.5)

        # Derivative = 0
        @test iszero(itp(x_below; deriv = DerivOp(1)))
    end

    @testset "Constant with fill value" begin
        itp = constant_interp(x, y; extrap = FillExtrap(99.0))
        @test itp(x_below) === 99.0
        @test itp(x_above) === 99.0
        @test itp(2.5) ≈ constant_interp(x, y; extrap = ClampExtrap())(2.5)
    end

    # ────────────────────────────────────────────
    # Oneshot paths
    # ────────────────────────────────────────────
    @testset "Oneshot paths" begin
        @test cubic_interp(x, y, x_below; extrap = FillExtrap(NaN)) |> isnan
        @test linear_interp(x, y, x_below; extrap = FillExtrap(NaN)) |> isnan
        @test quadratic_interp(x, y, x_below; extrap = FillExtrap(NaN)) |> isnan
        @test constant_interp(x, y, x_below; extrap = FillExtrap(NaN)) |> isnan
    end

    # ────────────────────────────────────────────
    # Series interpolation
    # ────────────────────────────────────────────
    @testset "Series with fill value" begin
        y_series = hcat(sin.(x), cos.(x))
        s = Series(y_series)
        out = Vector{Float64}(undef, 2)

        # Cubic series
        sitp = cubic_interp(x, s; extrap = FillExtrap(NaN))
        sitp(out, x_below)
        @test all(isnan, out)
        sitp(out, x_above)
        @test all(isnan, out)
        # In-domain should work
        sitp(out, 2.5)
        @test !any(isnan, out)

        # Linear series
        sitp_l = linear_interp(x, s; extrap = FillExtrap(0.0))
        sitp_l(out, x_below)
        @test all(==(0.0), out)

        # Quadratic series
        sitp_q = quadratic_interp(x, s; extrap = FillExtrap(-1.0))
        sitp_q(out, x_below)
        @test all(==(-1.0), out)

        # Constant series
        sitp_c = constant_interp(x, s; extrap = FillExtrap(42.0))
        sitp_c(out, x_below)
        @test all(==(42.0), out)
    end

    # ────────────────────────────────────────────
    # Boundary clamp backward compat
    # ────────────────────────────────────────────
    @testset "Boundary clamp unchanged" begin
        itp_clamp = cubic_interp(x, y; extrap = ClampExtrap())
        # Below domain → y[1]
        @test itp_clamp(x_below) ≈ y[1]
        # Above domain → y[end]
        @test itp_clamp(x_above) ≈ y[end]
    end

    # ────────────────────────────────────────────
    # ND fill-value evaluation
    # ────────────────────────────────────────────
    @testset "ND fill-value evaluation" begin
        xg = range(0.0, 1.0, length = 6)
        yg = range(0.0, 2.0, length = 8)
        data2d = [xi^2 + yj for xi in xg, yj in yg]

        # -- All 4 types with NaN fill --
        @testset "All types 2D with NaN fill" begin
            itp_c = cubic_interp((xg, yg), data2d; extrap = FillExtrap(NaN))
            itp_l = linear_interp((xg, yg), data2d; extrap = FillExtrap(NaN))
            itp_q = quadratic_interp((xg, yg), data2d; extrap = FillExtrap(NaN))
            itp_k = constant_interp((xg, yg), data2d; extrap = FillExtrap(NaN))

            # In-domain: unchanged behavior
            in_pt = (0.5, 1.0)
            @test itp_c(in_pt) isa Float64
            @test !isnan(itp_c(in_pt))
            @test !isnan(itp_l(in_pt))
            @test !isnan(itp_q(in_pt))
            @test !isnan(itp_k(in_pt))

            # OOB x only
            @test isnan(itp_c((-0.1, 1.0)))
            @test isnan(itp_l((-0.1, 1.0)))
            @test isnan(itp_q((-0.1, 1.0)))
            @test isnan(itp_k((-0.1, 1.0)))

            # OOB y only
            @test isnan(itp_c((0.5, 2.5)))
            @test isnan(itp_l((0.5, 2.5)))

            # OOB both
            @test isnan(itp_c((-0.1, 2.5)))
            @test isnan(itp_l((1.5, -0.5)))
        end

        # -- Zero fill --
        @testset "Zero fill 2D" begin
            itp = cubic_interp((xg, yg), data2d; extrap = FillExtrap(0.0))
            @test itp((-0.1, 1.0)) === 0.0
            @test itp((0.5, 2.5)) === 0.0
            # In-domain unchanged
            ref = cubic_interp((xg, yg), data2d; extrap = ClampExtrap())
            @test itp((0.5, 1.0)) ≈ ref((0.5, 1.0))
        end

        # -- Derivative under FillExtrap: cell-local "fill_value-as-data" --
        @testset "ND derivatives under FillExtrap" begin
            # NaN fill_value → deriv = 0 * NaN = NaN at OOB.
            itp_nan = cubic_interp((xg, yg), data2d; extrap = FillExtrap(NaN))
            @test isnan(itp_nan((-0.1, 1.0); deriv = DerivOp(1)))
            @test isnan(itp_nan((0.5, 2.5); deriv = (DerivOp(1), EvalValue())))
            @test isnan(itp_nan((-0.1, 1.0); deriv = (EvalValue(), DerivOp(1))))

            # Finite fill_value → deriv = 0 * fill_value = 0 at OOB.
            itp_zero = cubic_interp((xg, yg), data2d; extrap = FillExtrap(0.0))
            @test iszero(itp_zero((-0.1, 1.0); deriv = DerivOp(1)))
            @test iszero(itp_zero((0.5, 2.5); deriv = (DerivOp(1), EvalValue())))
            @test iszero(itp_zero((-0.1, 1.0); deriv = (EvalValue(), DerivOp(1))))
        end

        # -- Per-axis heterogeneous extrap --
        @testset "Per-axis mixed extrap" begin
            # Fill on x, clamp on y
            itp = cubic_interp(
                (xg, yg), data2d;
                extrap = (FillExtrap(NaN), ClampExtrap())
            )
            # OOB on x → fill
            @test isnan(itp((-0.1, 1.0)))
            # OOB on y → clamp (not fill)
            @test !isnan(itp((0.5, 2.5)))
            # In-domain → normal
            @test !isnan(itp((0.5, 1.0)))
        end

        # -- Conflicting fill values → ArgumentError --
        @testset "Conflicting fill values" begin
            @test_throws ArgumentError cubic_interp(
                (xg, yg), data2d;
                extrap = (FillExtrap(NaN), FillExtrap(0.0))
            )
        end

        # -- Boundary clamp backward compat --
        @testset "ND boundary clamp unchanged" begin
            itp = cubic_interp((xg, yg), data2d; extrap = ClampExtrap())
            @test itp((0.5, 0.5)) isa Float64
            # Clamp behavior: OOB queries get clamped to boundary
            @test !isnan(itp((-0.1, 1.0)))
        end
    end

    # ────────────────────────────────────────────
    # ND oneshot fill value
    # ────────────────────────────────────────────
    @testset "ND oneshot fill value" begin
        xg = range(0.0, 1.0, length = 6)
        yg = range(0.0, 2.0, length = 8)
        data2d = [xi^2 + yj for xi in xg, yj in yg]

        @test isnan(cubic_interp((xg, yg), data2d, (-0.1, 1.0); extrap = FillExtrap(NaN)))
        @test isnan(linear_interp((xg, yg), data2d, (-0.1, 1.0); extrap = FillExtrap(NaN)))
        @test isnan(quadratic_interp((xg, yg), data2d, (-0.1, 1.0); extrap = FillExtrap(NaN)))
        @test isnan(constant_interp((xg, yg), data2d, (-0.1, 1.0); extrap = FillExtrap(NaN)))

        # In-domain oneshot
        @test !isnan(cubic_interp((xg, yg), data2d, (0.5, 1.0); extrap = FillExtrap(NaN)))
    end

    # ────────────────────────────────────────────
    # ND batch fill value
    # ────────────────────────────────────────────
    @testset "ND batch fill value" begin
        xg = range(0.0, 1.0, length = 6)
        yg = range(0.0, 2.0, length = 8)
        data2d = [xi^2 + yj for xi in xg, yj in yg]

        itp = cubic_interp((xg, yg), data2d; extrap = FillExtrap(NaN))
        out = Vector{Float64}(undef, 3)

        # SoA batch: mix of in-domain and OOB
        xs = [0.5, -0.1, 0.8]
        ys = [1.0, 1.0, 2.5]
        itp(out, (xs, ys))
        @test !isnan(out[1])  # in-domain
        @test isnan(out[2])   # OOB x
        @test isnan(out[3])   # OOB y

        # AoS batch
        pts = [(0.5, 1.0), (-0.1, 1.0), (0.8, 2.5)]
        itp(out, pts)
        @test !isnan(out[1])
        @test isnan(out[2])
        @test isnan(out[3])
    end

    # ────────────────────────────────────────────
    # ND 3D fill value
    # ────────────────────────────────────────────
    @testset "3D fill value" begin
        xg = range(0.0, 1.0, length = 4)
        yg = range(0.0, 1.0, length = 4)
        zg = range(0.0, 1.0, length = 4)
        data3d = [xi + yj + zk for xi in xg, yj in yg, zk in zg]

        itp = linear_interp((xg, yg, zg), data3d; extrap = FillExtrap(NaN))
        @test !isnan(itp((0.5, 0.5, 0.5)))
        @test isnan(itp((-0.1, 0.5, 0.5)))
        @test isnan(itp((0.5, -0.1, 0.5)))
        @test isnan(itp((0.5, 0.5, 1.5)))
    end

    # ────────────────────────────────────────────
    # ND type promotion
    # ────────────────────────────────────────────
    @testset "ND fill type promotion" begin
        xg = range(0.0f0, 1.0f0, length = 5)
        yg = range(0.0f0, 1.0f0, length = 5)
        data = Float32[xi + yj for xi in xg, yj in yg]

        # Float64 fill → promoted to Float32
        itp = linear_interp((xg, yg), data; extrap = FillExtrap(0.0))
        val = itp((-0.1f0, 0.5f0))
        @test val === 0.0f0
        @test val isa Float32
    end

    # ────────────────────────────────────────────
    # Integral path
    # ────────────────────────────────────────────
    @testset "Integral with fill value" begin
        itp_nan = cubic_interp(x, y; extrap = FillExtrap(NaN))
        itp_zero = cubic_interp(x, y; extrap = FillExtrap(0.0))
        itp_clamp = cubic_interp(x, y; extrap = ClampExtrap())

        # NaN fill → NaN integral when bounds extend outside domain
        @test isnan(integrate(itp_nan, -1.0, 1.0))

        # Zero fill → no contribution from outside domain
        I_zero = integrate(itp_zero, -1.0, 1.0)
        I_in = integrate(itp_zero, 0.0, 1.0)
        @test I_zero ≈ I_in   # outside part contributes 0.0

        # Clamp still works (backward compat)
        I_clamp = integrate(itp_clamp, -1.0, 1.0)
        @test I_clamp ≈ y[1] * 1.0 + I_in   # left tail = y[1] × 1.0
    end

    # ────────────────────────────────────────────
    # Display
    # ────────────────────────────────────────────
    @testset "show formatting" begin
        itp_clamp = cubic_interp(x, y; extrap = ClampExtrap())
        itp_nan = cubic_interp(x, y; extrap = FillExtrap(NaN))
        itp_zero = cubic_interp(x, y; extrap = FillExtrap(0.0))

        # Use text/plain MIME for full multiline display (compact show omits extrap)
        s_clamp = sprint(show, MIME("text/plain"), itp_clamp)
        s_nan = sprint(show, MIME("text/plain"), itp_nan)
        s_zero = sprint(show, MIME("text/plain"), itp_zero)

        @test occursin("ClampExtrap", s_clamp)
        @test !occursin("FillExtrap(", s_clamp)  # no parens for boundary clamp
        @test occursin("FillExtrap(NaN)", s_nan)
        @test occursin("FillExtrap(0.0)", s_zero)
    end

    # ────────────────────────────────────────────
    # Factory
    # ────────────────────────────────────────────
    @testset "Extrap factory :clamp and :fill" begin
        e1 = Extrap(:clamp)
        @test e1 isa ClampExtrap

        e2 = Extrap(:fill; fill_value = NaN)
        @test e2 isa FillExtrap{Float64}
        @test isnan(e2.fill_value)

        e3 = Extrap(:fill; fill_value = 0.0)
        @test e3 isa FillExtrap{Float64}
        @test e3.fill_value === 0.0

        # :fill without value → ArgumentError
        @test_throws ArgumentError Extrap(:fill)

        # :constant deprecated but still works
        e4 = @test_deprecated Extrap(:constant)
        @test e4 isa ClampExtrap
    end

    # ────────────────────────────────────────────
    # Type promotion at construction
    # ────────────────────────────────────────────
    @testset "Fill value type promotion at construction" begin
        # Int fill with Float64 data → fill promoted to Float64
        itp = cubic_interp(x, y; extrap = FillExtrap(0))
        @test itp(x_below) === 0.0
        @test itp(x_below) isa Float64

        # Float32 grid/data with Float64 fill → fill promoted to match Tv
        x32 = collect(range(0.0f0, 5.0f0, length = 11))
        y32 = sin.(x32)
        itp32 = cubic_interp(x32, y32; extrap = FillExtrap(0.0))
        @test itp32(-1.0f0) isa Float32
        @test itp32(-1.0f0) === 0.0f0
    end

    # ────────────────────────────────────────────
    # P2: Cache constructor promotes fill extrap
    # ────────────────────────────────────────────
    @testset "Cache constructor promotes FillExtrap to Tv" begin
        x32 = collect(range(0.0f0, 5.0f0, length = 11))
        y32 = sin.(x32)
        cache = CubicSplineCache(x32)

        # Full API path: promotes Float64 fill → Float32
        itp_full = cubic_interp(x32, y32; extrap = FillExtrap(0.0))
        @test itp_full.extrap isa FillExtrap{Float32}
        @test itp_full(-1.0f0) === 0.0f0

        # Cache path: should also promote
        itp_cache = cubic_interp(cache, y32; extrap = FillExtrap(0.0))
        @test itp_cache.extrap isa FillExtrap{Float32}
        @test itp_cache(-1.0f0) === 0.0f0
    end

    # ────────────────────────────────────────────
    # P1: Vector-calculus OOB with FillExtrap — fill value is the OOB authority
    # ────────────────────────────────────────────
    @testset "Vector-calculus OOB with FillExtrap(NaN)" begin
        # The fill value IS the OOB region's data (test_oob_zero_carrier_contract):
        # a NaN fill poisons OOB derivatives, matching the 1D and ND eval paths.
        xg = range(0.0, 1.0, length = 6)
        yg = range(0.0, 1.0, length = 6)
        data = [sin(x + y) for x in xg, y in yg]
        itp = cubic_interp((xg, yg), data; extrap = FillExtrap(NaN))

        # In-domain query: derivatives should be nonzero
        q_in = (0.5, 0.5)
        g_in = gradient(itp, q_in)
        @test all(isfinite, g_in)
        @test !all(iszero, g_in)

        # OOB query: the NaN fill flows into every derivative
        q_oob = (-0.2, 0.5)
        g_oob = gradient(itp, q_oob)
        @test all(isnan, g_oob)

        # gradient! OOB
        G = zeros(2)
        gradient!(G, itp, q_oob)
        @test all(isnan, G)

        # hessian OOB
        H = hessian(itp, q_oob)
        @test all(isnan, H)

        # hessian! OOB
        H2 = ones(2, 2)
        hessian!(H2, itp, q_oob)
        @test all(isnan, H2)

        # laplacian OOB
        @test isnan(laplacian(itp, q_oob))

        # Both axes OOB
        q_oob2 = (-0.1, 1.5)
        @test all(isnan, gradient(itp, q_oob2))
        @test isnan(laplacian(itp, q_oob2))

        # Vector API passes through correctly
        g_vec = gradient(itp, [-0.2, 0.5])
        @test all(isnan, g_vec)

        # A FINITE fill keeps clean derivative zeros OOB
        itp0 = cubic_interp((xg, yg), data; extrap = FillExtrap(0.0))
        @test all(iszero, gradient(itp0, q_oob))
        @test all(iszero, hessian(itp0, q_oob))
        @test iszero(laplacian(itp0, q_oob))
    end

    # ────────────────────────────────────────────
    # ClampExtrap ND derivative correctness
    # ────────────────────────────────────────────
    # ClampExtrap clamps the query to the domain boundary and evaluates normally.
    # Derivatives at the clamped point are the REAL derivatives at the boundary,
    # not zeroed out. For f(x,y) = x² + y²: at boundary x=0, ∂f/∂x = 0, ∂²f/∂x² = 2.
    @testset "ClampExtrap ND derivative correctness" begin
        xg = range(0.0, 1.0, length = 11)
        yg = range(0.0, 1.0, length = 11)
        # f(x,y) = x² + y² → ∂f/∂x = 2x, ∂f/∂y = 2y, ∂²f/∂x² = ∂²f/∂y² = 2
        data = [xi^2 + yj^2 for xi in xg, yj in yg]
        itp = cubic_interp((xg, yg), data; extrap = ClampExtrap())

        # In-domain: gradient should be correct
        g_in = gradient(itp, (0.5, 0.5))
        @test g_in[1] ≈ 1.0 atol = 0.05   # ∂f/∂x ≈ 2*0.5
        @test g_in[2] ≈ 1.0 atol = 0.05   # ∂f/∂y ≈ 2*0.5

        # OOB on x-axis → clamped to x=0: ∂f/∂x = 2*0 ≈ 0, ∂f/∂y = 2*0.5 ≈ 1
        g_oob_x = gradient(itp, (-0.2, 0.5))
        @test g_oob_x[1] ≈ 0.0 atol = 0.05  # ∂f/∂x at x=0
        @test g_oob_x[2] ≈ 1.0 atol = 0.05  # ∂f/∂y at y=0.5

        # OOB on y-axis → clamped to y=1: ∂f/∂x = 2*0.5 ≈ 1, ∂f/∂y = 2*1 ≈ 2
        g_oob_y = gradient(itp, (0.5, 1.5))
        @test g_oob_y[1] ≈ 1.0 atol = 0.05  # ∂f/∂x at x=0.5
        @test g_oob_y[2] ≈ 2.0 atol = 0.05  # ∂f/∂y at y=1

        # Both axes OOB → clamped to (0, 1): ∂f/∂x = 0, ∂f/∂y = 2
        g_oob_both = gradient(itp, (-0.2, 1.5))
        @test g_oob_both[1] ≈ 0.0 atol = 0.05
        @test g_oob_both[2] ≈ 2.0 atol = 0.05

        # gradient! same behavior
        G = zeros(2)
        gradient!(G, itp, (-0.2, 0.5))
        @test G[1] ≈ 0.0 atol = 0.05
        @test G[2] ≈ 1.0 atol = 0.05

        # Hessian at clamped boundary: ∂²f/∂x² = 2, ∂²f/∂y² = 2, mixed ≈ 0
        H = hessian(itp, (-0.2, 0.5))
        @test H[1, 1] ≈ 2.0 atol = 0.1   # ∂²f/∂x² at x=0
        @test H[1, 2] ≈ 0.0 atol = 0.1   # mixed partial ≈ 0
        @test H[2, 1] ≈ 0.0 atol = 0.1   # symmetric
        @test H[2, 2] ≈ 2.0 atol = 0.1   # ∂²f/∂y² at y=0.5

        # hessian! same behavior
        H2 = ones(2, 2)
        hessian!(H2, itp, (-0.2, 0.5))
        @test H2[1, 1] ≈ 2.0 atol = 0.1
        @test H2[1, 2] ≈ 0.0 atol = 0.1
        @test H2[2, 2] ≈ 2.0 atol = 0.1

        # Laplacian: sum of all second derivatives (both contribute)
        lap = laplacian(itp, (-0.2, 0.5))
        @test lap ≈ 4.0 atol = 0.2  # ∂²f/∂x² + ∂²f/∂y² = 2 + 2

        lap_both_oob = laplacian(itp, (-0.2, 1.5))
        @test lap_both_oob ≈ 4.0 atol = 0.2  # clamped to (0,1): still 2 + 2
    end

    # ────────────────────────────────────────────
    # Mixed ClampExtrap/FillExtrap per-axis
    # ────────────────────────────────────────────
    @testset "Mixed ClampExtrap/FillExtrap per-axis derivatives" begin
        xg = range(0.0, 1.0, length = 11)
        yg = range(0.0, 1.0, length = 11)
        data = [xi^2 + yj^2 for xi in xg, yj in yg]

        # axis 1: ClampExtrap, axis 2: FillExtrap(NaN)
        itp = cubic_interp(
            (xg, yg), data;
            extrap = (ClampExtrap(), FillExtrap(NaN))
        )

        # In-domain → normal
        @test isfinite(itp((0.5, 0.5)))
        g_in = gradient(itp, (0.5, 0.5))
        @test all(isfinite, g_in)

        # OOB on axis 1 (Clamped) → clamp value, derivatives at boundary
        val_oob_x = itp((-0.2, 0.5))
        @test isfinite(val_oob_x)  # clamped, not NaN
        g_oob_x = gradient(itp, (-0.2, 0.5))
        @test g_oob_x[1] ≈ 0.0 atol = 0.05  # ∂f/∂x at x=0 boundary
        @test isfinite(g_oob_x[2])  # in-domain axis

        # OOB on axis 2 (Fill) → NaN value; the NaN fill poisons the derivatives
        val_oob_y = itp((0.5, 1.5))
        @test isnan(val_oob_y)  # FillExtrap → NaN
        g_oob_y = gradient(itp, (0.5, 1.5))
        @test all(isnan, g_oob_y)  # fill value carries the OOB zeros

        # Hessian: OOB on Fill axis → NaN fill flows through
        H = hessian(itp, (0.5, 1.5))
        @test all(isnan, H)

        # Hessian: OOB on Clamp axis only → real derivatives at boundary
        H2 = hessian(itp, (-0.2, 0.5))
        @test H2[1, 1] ≈ 2.0 atol = 0.1  # ∂²f/∂x² at x=0
        @test H2[1, 2] ≈ 0.0 atol = 0.1  # mixed partial ≈ 0
        @test H2[2, 1] ≈ 0.0 atol = 0.1  # symmetric
        @test H2[2, 2] ≈ 2.0 atol = 0.1  # ∂²f/∂y² at y=0.5
    end

    # ────────────────────────────────────────────
    # 1D derivative correctness with FillExtrap
    # ────────────────────────────────────────────
    # Cell-local "fill_value-as-data" contract: OOB cell's data IS `fill_value`,
    # so deriv = `0 * fill_value` — finite fill_value → 0, NaN propagates.
    @testset "1D derivative correctness: FillExtrap OOB" begin
        x = collect(range(0.0, 1.0, length = 21))
        y = @. sin(2π * x)

        # Finite fill_value → deriv exactly 0 at OOB.
        for (label, fill_val) in [("zero", 0.0), ("42", 42.0)]
            @testset "$label fill ($fill_val) → deriv 0" begin
                itp_c = cubic_interp(x, y; extrap = FillExtrap(fill_val))
                itp_l = linear_interp(x, y; extrap = FillExtrap(fill_val))
                itp_q = quadratic_interp(x, y; extrap = FillExtrap(fill_val))

                for itp in (itp_c, itp_l, itp_q)
                    for xq_oob in (-0.5, 1.5)
                        @test itp(xq_oob; deriv = DerivOp(1)) == 0.0
                    end
                    # In-domain at x=0.1: sin'(2π*0.1) = 2π*cos(0.2π) ≈ 5.08
                    @test abs(itp(0.1; deriv = DerivOp(1))) > 1.0
                end

                # 2nd and 3rd order derivatives (cubic only)
                for xq_oob in (-0.5, 1.5)
                    @test itp_c(xq_oob; deriv = DerivOp(2)) == 0.0
                    @test itp_c(xq_oob; deriv = DerivOp(3)) == 0.0
                end
            end
        end

        # NaN fill_value → deriv propagates NaN at OOB (cell-local-as-data).
        @testset "NaN fill → deriv NaN at OOB" begin
            itp_c = cubic_interp(x, y; extrap = FillExtrap(NaN))
            itp_l = linear_interp(x, y; extrap = FillExtrap(NaN))
            itp_q = quadratic_interp(x, y; extrap = FillExtrap(NaN))

            for itp in (itp_c, itp_l, itp_q)
                for xq_oob in (-0.5, 1.5)
                    @test isnan(itp(xq_oob; deriv = DerivOp(1)))
                end
                # In-domain unchanged.
                @test abs(itp(0.1; deriv = DerivOp(1))) > 1.0
            end
            for xq_oob in (-0.5, 1.5)
                @test isnan(itp_c(xq_oob; deriv = DerivOp(2)))
                @test isnan(itp_c(xq_oob; deriv = DerivOp(3)))
            end
        end
    end

    # ────────────────────────────────────────────
    # integrate correctness with FillExtrap
    # ────────────────────────────────────────────
    # OOB region integral = fill_value × interval_length
    @testset "integrate correctness: FillExtrap tails" begin
        x = collect(range(0.0, 1.0, length = 31))
        y = @. x^2 + 1.0  # y[1] = 1.0, y[end] = 2.0

        @testset "FillExtrap(0.0) — zero tails" begin
            itp = linear_interp(x, y; extrap = FillExtrap(0.0))
            # Pure left tail: integral of 0 over [-0.4, 0]
            @test integrate(itp, -0.4, 0.0) ≈ 0.0 atol = 1.0e-14
            # Pure right tail: integral of 0 over [1, 1.6]
            @test integrate(itp, 1.0, 1.6) ≈ 0.0 atol = 1.0e-14
            # Mixed: left tail (zero) + in-domain part
            in_part = integrate(linear_interp(x, y; extrap = NoExtrap()), 0.0, 0.5)
            @test integrate(itp, -0.3, 0.5) ≈ 0.0 * 0.3 + in_part atol = 1.0e-12
        end

        @testset "FillExtrap(42.0) — nonzero tails" begin
            itp = linear_interp(x, y; extrap = FillExtrap(42.0))
            # Pure left tail: 42 × 0.4
            @test integrate(itp, -0.4, 0.0) ≈ 42.0 * 0.4 atol = 1.0e-12
            # Pure right tail: 42 × 0.6
            @test integrate(itp, 1.0, 1.6) ≈ 42.0 * 0.6 atol = 1.0e-12
            # Spanning both tails + in-domain
            in_part = integrate(linear_interp(x, y; extrap = NoExtrap()), 0.0, 1.0)
            @test integrate(itp, -0.5, 1.5) ≈ 42.0 * 0.5 + in_part + 42.0 * 0.5 atol = 1.0e-12
        end

        @testset "FillExtrap(NaN) — NaN tails" begin
            itp = cubic_interp(x, y; extrap = FillExtrap(NaN))
            # Pure OOB → NaN × length = NaN
            @test isnan(integrate(itp, -0.5, -0.1))
            @test isnan(integrate(itp, 1.1, 1.5))
            # Mixed: in-domain + NaN tail → NaN
            @test isnan(integrate(itp, 0.5, 1.5))
            # Fully in-domain → finite (NaN doesn't leak)
            @test isfinite(integrate(itp, 0.1, 0.9))
        end

        @testset "FillExtrap signed orientation" begin
            itp = linear_interp(x, y; extrap = FillExtrap(42.0))
            # Reversed bounds → negated result
            @test integrate(itp, 0.0, -0.4) ≈ -(42.0 * 0.4) atol = 1.0e-12
        end

        @testset "FillExtrap cubic tails" begin
            itp = cubic_interp(x, y; extrap = FillExtrap(5.0))
            @test integrate(itp, -0.5, 0.0) ≈ 5.0 * 0.5 atol = 1.0e-12
            @test integrate(itp, 1.0, 1.3) ≈ 5.0 * 0.3 atol = 1.0e-12
        end

        @testset "FillExtrap constant interp tails" begin
            itp = constant_interp(x, y; extrap = FillExtrap(7.0))
            @test integrate(itp, -0.4, 0.0) ≈ 7.0 * 0.4 atol = 1.0e-12
            @test integrate(itp, 1.0, 1.6) ≈ 7.0 * 0.6 atol = 1.0e-12
        end
    end
end

@testitem "FillExtrap: a float fill on integer data lives in the output space" begin
    # The kernels already float Int data against a float grid (`itp(2.2)::Float64`),
    # but several persistent entries promoted the fill against the RAW data eltype
    # → `convert(Int, 0.5)` InexactError at construction. Real axes, plain Int data.
    x = [0.0, 1.0, 2.5, 3.0, 4.5]
    y = [0.0, 1.0, 2.0, 3.5]
    yint = [1, 2, 3, 4, 5]
    Zint = [i + j for i in 1:5, j in 1:4]
    q_oob, q_in = 9.9, 2.2

    @testset "1D persistent: every family accepts it" begin
        for mk in (linear_interp, constant_interp, cubic_interp, quadratic_interp, pchip_interp)
            itp = mk(x, yint; extrap = FillExtrap(0.5))
            @test itp(q_oob) === 0.5
            @test itp(q_in) isa Float64
        end
    end

    @testset "1D one-shot mirrors" begin
        @test cubic_interp(x, yint, q_oob; extrap = FillExtrap(0.5)) === 0.5
        @test linear_interp(x, yint, q_oob; extrap = FillExtrap(0.5)) === 0.5
        @test constant_interp(x, yint, q_oob; extrap = FillExtrap(0.5)) === 0.5
    end

    @testset "ND persistent + gridded" begin
        for m in (LinearInterp(), ConstantInterp())
            itp = interp((x, y), Zint; method = m, extrap = FillExtrap(0.5))
            @test itp((q_oob, 1.0)) === 0.5
            # Gridded (axis-target) path shares the fill promotion.
            @test itp(([q_oob], [1.0]))[1] === 0.5
        end
        @test cubic_interp((x, y), Zint; extrap = FillExtrap(0.5))((q_oob, 1.0)) === 0.5
    end
end
