using Test
using FastInterpolations

@testset "ConstExtrap Fill Value" begin
    # ────────────────────────────────────────────
    # Construction & Type Hierarchy
    # ────────────────────────────────────────────
    @testset "ConstExtrap type hierarchy" begin
        # ConstExtrap is a factory function, not a type
        @test ClampedExtrap <: AbstractExtrap
        @test FillExtrap <: AbstractExtrap
        @test !(ClampedExtrap <: FillExtrap)
        @test !(FillExtrap <: ClampedExtrap)

        # No-arg factory: boundary clamp
        e0 = ConstExtrap()
        @test e0 isa ClampedExtrap

        # Float value: auto-promote Int → Float64
        e1 = ConstExtrap(0)
        @test e1.value === 0.0
        @test e1 isa FillExtrap{Float64}

        # Float64
        e2 = ConstExtrap(NaN)
        @test isnan(e2.value)
        @test e2 isa FillExtrap{Float64}

        # Float32
        e3 = ConstExtrap(0.0f0)
        @test e3.value === 0.0f0
        @test e3 isa FillExtrap{Float32}

        # Kwarg form
        e4 = ConstExtrap(; value=NaN)
        @test isnan(e4.value)
        @test e4 isa FillExtrap{Float64}

        # Kwarg no-arg → boundary clamp
        e5 = ConstExtrap(; value=nothing)
        @test e5 isa ClampedExtrap

        # Direct construction
        @test ClampedExtrap() isa ClampedExtrap
        @test FillExtrap(NaN) isa FillExtrap{Float64}
        @test FillExtrap(0.0f0) isa FillExtrap{Float32}
    end

    @testset "_promote_extrap" begin
        using FastInterpolations: _promote_extrap

        # Non-ConstExtrap passes through
        @test _promote_extrap(NoExtrap(), Float64) === NoExtrap()
        @test _promote_extrap(ExtendExtrap(), Float64) === ExtendExtrap()

        # ClampedExtrap passes through regardless of Tv
        e_clamp = ClampedExtrap()
        @test _promote_extrap(e_clamp, Float64) === e_clamp
        @test _promote_extrap(e_clamp, Float32) === e_clamp

        # Fill value gets converted to Tv
        e_f64 = FillExtrap(0.0)
        e_promoted = _promote_extrap(e_f64, Float32)
        @test e_promoted isa FillExtrap{Float32}
        @test e_promoted.value === 0.0f0

        # NaN promotion
        e_nan = FillExtrap(NaN)
        e_nan32 = _promote_extrap(e_nan, Float32)
        @test e_nan32 isa FillExtrap{Float32}
        @test isnan(e_nan32.value)
    end

    # ────────────────────────────────────────────
    # Evaluation: all 4 interpolation types
    # ────────────────────────────────────────────
    x = collect(range(0.0, 5.0, length=11))
    y = sin.(x)
    x_below = -1.0
    x_above = 6.0

    @testset "Cubic with fill value" begin
        # NaN fill
        itp_nan = cubic_interp(x, y; extrap=ConstExtrap(NaN))
        @test isnan(itp_nan(x_below))
        @test isnan(itp_nan(x_above))
        # In-domain should work normally
        @test itp_nan(2.5) ≈ cubic_interp(x, y; extrap=ConstExtrap())(2.5)

        # Zero fill
        itp_zero = cubic_interp(x, y; extrap=ConstExtrap(0.0))
        @test itp_zero(x_below) === 0.0
        @test itp_zero(x_above) === 0.0

        # Custom fill
        itp_42 = cubic_interp(x, y; extrap=ConstExtrap(42.0))
        @test itp_42(x_below) === 42.0
        @test itp_42(x_above) === 42.0

        # Derivatives should return 0, not NaN (even with NaN fill)
        # Note: 0 * negative_y gives -0.0, so use iszero() not === 0.0
        @test iszero(itp_nan(x_below; deriv=DerivOp(1)))
        @test iszero(itp_nan(x_above; deriv=DerivOp(1)))
        @test iszero(itp_nan(x_below; deriv=DerivOp(2)))
        @test iszero(itp_nan(x_above; deriv=DerivOp(2)))
    end

    @testset "Linear with fill value" begin
        itp = linear_interp(x, y; extrap=ConstExtrap(NaN))
        @test isnan(itp(x_below))
        @test isnan(itp(x_above))
        @test itp(2.5) ≈ linear_interp(x, y; extrap=ConstExtrap())(2.5)

        # Derivative = 0
        @test iszero(itp(x_below; deriv=DerivOp(1)))
    end

    @testset "Quadratic with fill value" begin
        itp = quadratic_interp(x, y; extrap=ConstExtrap(-1.0))
        @test itp(x_below) === -1.0
        @test itp(x_above) === -1.0
        @test itp(2.5) ≈ quadratic_interp(x, y; extrap=ConstExtrap())(2.5)

        # Derivative = 0
        @test iszero(itp(x_below; deriv=DerivOp(1)))
    end

    @testset "Constant with fill value" begin
        itp = constant_interp(x, y; extrap=ConstExtrap(99.0))
        @test itp(x_below) === 99.0
        @test itp(x_above) === 99.0
        @test itp(2.5) ≈ constant_interp(x, y; extrap=ConstExtrap())(2.5)
    end

    # ────────────────────────────────────────────
    # Oneshot paths
    # ────────────────────────────────────────────
    @testset "Oneshot paths" begin
        @test cubic_interp(x, y, x_below; extrap=ConstExtrap(NaN)) |> isnan
        @test linear_interp(x, y, x_below; extrap=ConstExtrap(NaN)) |> isnan
        @test quadratic_interp(x, y, x_below; extrap=ConstExtrap(NaN)) |> isnan
        @test constant_interp(x, y, x_below; extrap=ConstExtrap(NaN)) |> isnan
    end

    # ────────────────────────────────────────────
    # Series interpolation
    # ────────────────────────────────────────────
    @testset "Series with fill value" begin
        y_series = hcat(sin.(x), cos.(x))
        s = Series(y_series)
        out = Vector{Float64}(undef, 2)

        # Cubic series
        sitp = cubic_interp(x, s; extrap=ConstExtrap(NaN))
        sitp(out, x_below)
        @test all(isnan, out)
        sitp(out, x_above)
        @test all(isnan, out)
        # In-domain should work
        sitp(out, 2.5)
        @test !any(isnan, out)

        # Linear series
        sitp_l = linear_interp(x, s; extrap=ConstExtrap(0.0))
        sitp_l(out, x_below)
        @test all(==(0.0), out)

        # Quadratic series
        sitp_q = quadratic_interp(x, s; extrap=ConstExtrap(-1.0))
        sitp_q(out, x_below)
        @test all(==(-1.0), out)

        # Constant series
        sitp_c = constant_interp(x, s; extrap=ConstExtrap(42.0))
        sitp_c(out, x_below)
        @test all(==(42.0), out)
    end

    # ────────────────────────────────────────────
    # Boundary clamp backward compat
    # ────────────────────────────────────────────
    @testset "Boundary clamp unchanged" begin
        itp_clamp = cubic_interp(x, y; extrap=ConstExtrap())
        # Below domain → y[1]
        @test itp_clamp(x_below) ≈ y[1]
        # Above domain → y[end]
        @test itp_clamp(x_above) ≈ y[end]
    end

    # ────────────────────────────────────────────
    # ND fill-value evaluation
    # ────────────────────────────────────────────
    @testset "ND fill-value evaluation" begin
        xg = range(0.0, 1.0, length=6)
        yg = range(0.0, 2.0, length=8)
        data2d = [xi^2 + yj for xi in xg, yj in yg]

        # -- All 4 types with NaN fill --
        @testset "All types 2D with NaN fill" begin
            itp_c = cubic_interp((xg, yg), data2d; extrap=ConstExtrap(NaN))
            itp_l = linear_interp((xg, yg), data2d; extrap=ConstExtrap(NaN))
            itp_q = quadratic_interp((xg, yg), data2d; extrap=ConstExtrap(NaN))
            itp_k = constant_interp((xg, yg), data2d; extrap=ConstExtrap(NaN))

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
            itp = cubic_interp((xg, yg), data2d; extrap=ConstExtrap(0.0))
            @test itp((-0.1, 1.0)) === 0.0
            @test itp((0.5, 2.5)) === 0.0
            # In-domain unchanged
            ref = cubic_interp((xg, yg), data2d; extrap=ConstExtrap())
            @test itp((0.5, 1.0)) ≈ ref((0.5, 1.0))
        end

        # -- Derivatives return zero (not fill value) --
        @testset "ND derivatives return zero" begin
            itp = cubic_interp((xg, yg), data2d; extrap=ConstExtrap(NaN))
            @test iszero(itp((-0.1, 1.0); deriv=DerivOp(1)))
            @test iszero(itp((0.5, 2.5); deriv=(DerivOp(1), EvalValue())))
            @test iszero(itp((-0.1, 1.0); deriv=(EvalValue(), DerivOp(1))))
        end

        # -- Per-axis heterogeneous extrap --
        @testset "Per-axis mixed extrap" begin
            # Fill on x, clamp on y
            itp = cubic_interp((xg, yg), data2d;
                extrap=(ConstExtrap(NaN), ConstExtrap()))
            # OOB on x → fill
            @test isnan(itp((-0.1, 1.0)))
            # OOB on y → clamp (not fill)
            @test !isnan(itp((0.5, 2.5)))
            # In-domain → normal
            @test !isnan(itp((0.5, 1.0)))
        end

        # -- Conflicting fill values → ArgumentError --
        @testset "Conflicting fill values" begin
            @test_throws ArgumentError cubic_interp((xg, yg), data2d;
                extrap=(ConstExtrap(NaN), ConstExtrap(0.0)))
        end

        # -- Boundary clamp backward compat --
        @testset "ND boundary clamp unchanged" begin
            itp = cubic_interp((xg, yg), data2d; extrap=ConstExtrap())
            @test itp((0.5, 0.5)) isa Float64
            # Clamp behavior: OOB queries get clamped to boundary
            @test !isnan(itp((-0.1, 1.0)))
        end
    end

    # ────────────────────────────────────────────
    # ND oneshot fill value
    # ────────────────────────────────────────────
    @testset "ND oneshot fill value" begin
        xg = range(0.0, 1.0, length=6)
        yg = range(0.0, 2.0, length=8)
        data2d = [xi^2 + yj for xi in xg, yj in yg]

        @test isnan(cubic_interp((xg, yg), data2d, (-0.1, 1.0); extrap=ConstExtrap(NaN)))
        @test isnan(linear_interp((xg, yg), data2d, (-0.1, 1.0); extrap=ConstExtrap(NaN)))
        @test isnan(quadratic_interp((xg, yg), data2d, (-0.1, 1.0); extrap=ConstExtrap(NaN)))
        @test isnan(constant_interp((xg, yg), data2d, (-0.1, 1.0); extrap=ConstExtrap(NaN)))

        # In-domain oneshot
        @test !isnan(cubic_interp((xg, yg), data2d, (0.5, 1.0); extrap=ConstExtrap(NaN)))
    end

    # ────────────────────────────────────────────
    # ND batch fill value
    # ────────────────────────────────────────────
    @testset "ND batch fill value" begin
        xg = range(0.0, 1.0, length=6)
        yg = range(0.0, 2.0, length=8)
        data2d = [xi^2 + yj for xi in xg, yj in yg]

        itp = cubic_interp((xg, yg), data2d; extrap=ConstExtrap(NaN))
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
        xg = range(0.0, 1.0, length=4)
        yg = range(0.0, 1.0, length=4)
        zg = range(0.0, 1.0, length=4)
        data3d = [xi + yj + zk for xi in xg, yj in yg, zk in zg]

        itp = linear_interp((xg, yg, zg), data3d; extrap=ConstExtrap(NaN))
        @test !isnan(itp((0.5, 0.5, 0.5)))
        @test isnan(itp((-0.1, 0.5, 0.5)))
        @test isnan(itp((0.5, -0.1, 0.5)))
        @test isnan(itp((0.5, 0.5, 1.5)))
    end

    # ────────────────────────────────────────────
    # ND type promotion
    # ────────────────────────────────────────────
    @testset "ND fill type promotion" begin
        xg = range(0.0f0, 1.0f0, length=5)
        yg = range(0.0f0, 1.0f0, length=5)
        data = Float32[xi + yj for xi in xg, yj in yg]

        # Float64 fill → promoted to Float32
        itp = linear_interp((xg, yg), data; extrap=ConstExtrap(0.0))
        val = itp((-0.1f0, 0.5f0))
        @test val === 0.0f0
        @test val isa Float32
    end

    # ────────────────────────────────────────────
    # Integral path
    # ────────────────────────────────────────────
    @testset "Integral with fill value" begin
        itp_nan = cubic_interp(x, y; extrap=ConstExtrap(NaN))
        itp_zero = cubic_interp(x, y; extrap=ConstExtrap(0.0))
        itp_clamp = cubic_interp(x, y; extrap=ConstExtrap())

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
        itp_clamp = cubic_interp(x, y; extrap=ConstExtrap())
        itp_nan = cubic_interp(x, y; extrap=ConstExtrap(NaN))
        itp_zero = cubic_interp(x, y; extrap=ConstExtrap(0.0))

        # Use text/plain MIME for full multiline display (compact show omits extrap)
        s_clamp = sprint(show, MIME("text/plain"), itp_clamp)
        s_nan = sprint(show, MIME("text/plain"), itp_nan)
        s_zero = sprint(show, MIME("text/plain"), itp_zero)

        @test occursin("ConstExtrap", s_clamp)
        @test !occursin("ConstExtrap(", s_clamp)  # no parens for boundary clamp
        @test occursin("FillExtrap(NaN)", s_nan)
        @test occursin("FillExtrap(0.0)", s_zero)
    end

    # ────────────────────────────────────────────
    # Factory
    # ────────────────────────────────────────────
    @testset "Extrap factory with value" begin
        e1 = Extrap(:constant)
        @test e1 isa ClampedExtrap

        e2 = Extrap(:constant; value=NaN)
        @test e2 isa FillExtrap{Float64}
        @test isnan(e2.value)

        e3 = Extrap(:constant; value=0.0)
        @test e3 isa FillExtrap{Float64}
        @test e3.value === 0.0
    end

    # ────────────────────────────────────────────
    # Type promotion at construction
    # ────────────────────────────────────────────
    @testset "Fill value type promotion at construction" begin
        # Int fill with Float64 data → fill promoted to Float64
        itp = cubic_interp(x, y; extrap=ConstExtrap(0))
        @test itp(x_below) === 0.0
        @test itp(x_below) isa Float64

        # Float32 grid/data with Float64 fill → fill promoted to match Tv
        x32 = collect(range(0.0f0, 5.0f0, length=11))
        y32 = sin.(x32)
        itp32 = cubic_interp(x32, y32; extrap=ConstExtrap(0.0))
        @test itp32(-1.0f0) isa Float32
        @test itp32(-1.0f0) === 0.0f0
    end

    # ────────────────────────────────────────────
    # P2: Cache constructor promotes fill extrap
    # ────────────────────────────────────────────
    @testset "Cache constructor promotes FillExtrap to Tv" begin
        x32 = collect(range(0.0f0, 5.0f0, length=11))
        y32 = sin.(x32)
        cache = CubicSplineCache(x32)

        # Full API path: promotes Float64 fill → Float32
        itp_full = cubic_interp(x32, y32; extrap=FillExtrap(0.0))
        @test itp_full.extrap isa FillExtrap{Float32}
        @test itp_full(-1.0f0) === 0.0f0

        # Cache path: should also promote
        itp_cache = cubic_interp(cache, y32; extrap=FillExtrap(0.0))
        @test itp_cache.extrap isa FillExtrap{Float32}
        @test itp_cache(-1.0f0) === 0.0f0
    end

    # ────────────────────────────────────────────
    # P1: Vector-calculus OOB guard for FillExtrap
    # ────────────────────────────────────────────
    @testset "Vector-calculus OOB guard with FillExtrap" begin
        # 2D cubic interpolant with NaN fill
        xg = range(0.0, 1.0, length=6)
        yg = range(0.0, 1.0, length=6)
        data = [sin(x + y) for x in xg, y in yg]
        itp = cubic_interp((xg, yg), data; extrap=FillExtrap(NaN))

        # In-domain query: derivatives should be nonzero
        q_in = (0.5, 0.5)
        g_in = gradient(itp, q_in)
        @test all(isfinite, g_in)
        @test !all(iszero, g_in)

        # OOB query: all derivatives should be zero
        q_oob = (-0.2, 0.5)
        g_oob = gradient(itp, q_oob)
        @test all(iszero, g_oob)

        # gradient! OOB
        G = zeros(2)
        gradient!(G, itp, q_oob)
        @test all(iszero, G)

        # hessian OOB
        H = hessian(itp, q_oob)
        @test all(iszero, H)

        # hessian! OOB
        H2 = ones(2, 2)
        hessian!(H2, itp, q_oob)
        @test all(iszero, H2)

        # laplacian OOB
        @test laplacian(itp, q_oob) == 0.0

        # Both axes OOB
        q_oob2 = (-0.1, 1.5)
        @test all(iszero, gradient(itp, q_oob2))
        @test laplacian(itp, q_oob2) == 0.0

        # Vector API passes through correctly
        g_vec = gradient(itp, [-0.2, 0.5])
        @test all(iszero, g_vec)
    end

    # ────────────────────────────────────────────
    # ClampedExtrap ND derivative correctness
    # ────────────────────────────────────────────
    @testset "ClampedExtrap ND derivative correctness" begin
        xg = range(0.0, 1.0, length=11)
        yg = range(0.0, 1.0, length=11)
        # f(x,y) = x² + y² → ∂f/∂x = 2x, ∂f/∂y = 2y
        data = [xi^2 + yj^2 for xi in xg, yj in yg]
        itp = cubic_interp((xg, yg), data; extrap=ClampedExtrap())

        # In-domain: gradient should be correct
        g_in = gradient(itp, (0.5, 0.5))
        @test g_in[1] ≈ 1.0 atol=0.05   # ∂f/∂x ≈ 2*0.5
        @test g_in[2] ≈ 1.0 atol=0.05   # ∂f/∂y ≈ 2*0.5

        # OOB on x-axis only → ∂f/∂x = 0, ∂f/∂y computed at clamped boundary
        g_oob_x = gradient(itp, (-0.2, 0.5))
        @test g_oob_x[1] == 0.0  # OOB axis → zero derivative
        @test g_oob_x[2] ≈ 1.0 atol=0.05  # in-domain axis → normal ∂f/∂y at (0,0.5)

        # OOB on y-axis only → ∂f/∂x computed at clamped boundary, ∂f/∂y = 0
        g_oob_y = gradient(itp, (0.5, 1.5))
        @test g_oob_y[1] ≈ 1.0 atol=0.05  # in-domain axis → normal ∂f/∂x at (0.5,1)
        @test g_oob_y[2] == 0.0  # OOB axis → zero derivative

        # Both axes OOB → all derivatives zero
        g_oob_both = gradient(itp, (-0.2, 1.5))
        @test all(iszero, g_oob_both)

        # gradient! same behavior
        G = zeros(2)
        gradient!(G, itp, (-0.2, 0.5))
        @test G[1] == 0.0
        @test G[2] ≈ 1.0 atol=0.05

        # Hessian: OOB axis rows/columns should be zero
        H = hessian(itp, (-0.2, 0.5))
        @test H[1, 1] == 0.0   # ∂²f/∂x² on OOB axis
        @test H[1, 2] == 0.0   # mixed partial involving OOB axis
        @test H[2, 1] == 0.0   # symmetric
        @test H[2, 2] ≈ 2.0 atol=0.1  # ∂²f/∂y² on in-domain axis

        # hessian! same behavior
        H2 = ones(2, 2)
        hessian!(H2, itp, (-0.2, 0.5))
        @test H2[1, 1] == 0.0
        @test H2[1, 2] == 0.0
        @test H2[2, 2] ≈ 2.0 atol=0.1

        # Laplacian: only in-domain axes contribute
        lap = laplacian(itp, (-0.2, 0.5))
        @test lap ≈ 2.0 atol=0.1  # only ∂²f/∂y² contributes

        lap_both_oob = laplacian(itp, (-0.2, 1.5))
        @test lap_both_oob == 0.0
    end

    # ────────────────────────────────────────────
    # Mixed ClampedExtrap/FillExtrap per-axis
    # ────────────────────────────────────────────
    @testset "Mixed ClampedExtrap/FillExtrap per-axis derivatives" begin
        xg = range(0.0, 1.0, length=11)
        yg = range(0.0, 1.0, length=11)
        data = [xi^2 + yj^2 for xi in xg, yj in yg]

        # axis 1: ClampedExtrap, axis 2: FillExtrap(NaN)
        itp = cubic_interp((xg, yg), data;
            extrap=(ClampedExtrap(), FillExtrap(NaN)))

        # In-domain → normal
        @test isfinite(itp((0.5, 0.5)))
        g_in = gradient(itp, (0.5, 0.5))
        @test all(isfinite, g_in)

        # OOB on axis 1 (Clamped) → clamp value, derivatives masked
        val_oob_x = itp((-0.2, 0.5))
        @test isfinite(val_oob_x)  # clamped, not NaN
        g_oob_x = gradient(itp, (-0.2, 0.5))
        @test g_oob_x[1] == 0.0  # ClampedExtrap OOB axis → zero
        @test isfinite(g_oob_x[2])  # in-domain axis

        # OOB on axis 2 (Fill) → NaN value, all derivatives zero
        val_oob_y = itp((0.5, 1.5))
        @test isnan(val_oob_y)  # FillExtrap → NaN
        g_oob_y = gradient(itp, (0.5, 1.5))
        @test all(iszero, g_oob_y)  # FillExtrap total short-circuit

        # Hessian: OOB on Fill axis → all zero
        H = hessian(itp, (0.5, 1.5))
        @test all(iszero, H)

        # Hessian: OOB on Clamp axis only → row/col 1 zero, H[2,2] nonzero
        H2 = hessian(itp, (-0.2, 0.5))
        @test H2[1, 1] == 0.0
        @test H2[1, 2] == 0.0
        @test H2[2, 1] == 0.0
        @test H2[2, 2] ≈ 2.0 atol=0.1
    end
end
