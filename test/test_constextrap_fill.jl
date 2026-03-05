using Test
using FastInterpolations

@testset "ConstExtrap Fill Value" begin
    # ────────────────────────────────────────────
    # Construction & Type Promotion
    # ────────────────────────────────────────────
    @testset "ConstExtrap construction" begin
        # No-arg: boundary clamp (Nothing)
        e0 = ConstExtrap()
        @test e0.value === nothing
        @test e0 isa ConstExtrap{Nothing}

        # Float value: auto-promote Int → Float64
        e1 = ConstExtrap(0)
        @test e1.value === 0.0
        @test e1 isa ConstExtrap{Float64}

        # Float64
        e2 = ConstExtrap(NaN)
        @test isnan(e2.value)
        @test e2 isa ConstExtrap{Float64}

        # Float32
        e3 = ConstExtrap(0.0f0)
        @test e3.value === 0.0f0
        @test e3 isa ConstExtrap{Float32}

        # Kwarg form
        e4 = ConstExtrap(; value=NaN)
        @test isnan(e4.value)
        @test e4 isa ConstExtrap{Float64}

        # Kwarg no-arg → boundary clamp
        e5 = ConstExtrap(; value=nothing)
        @test e5.value === nothing
        @test e5 isa ConstExtrap{Nothing}
    end

    @testset "_promote_extrap" begin
        using FastInterpolations: _promote_extrap

        # Non-ConstExtrap passes through
        @test _promote_extrap(NoExtrap(), Float64) === NoExtrap()
        @test _promote_extrap(ExtendExtrap(), Float64) === ExtendExtrap()

        # ConstExtrap{Nothing} passes through regardless of Tv
        e_nothing = ConstExtrap()
        @test _promote_extrap(e_nothing, Float64) === e_nothing
        @test _promote_extrap(e_nothing, Float32) === e_nothing

        # Fill value gets converted to Tv
        e_f64 = ConstExtrap(0.0)
        e_promoted = _promote_extrap(e_f64, Float32)
        @test e_promoted isa ConstExtrap{Float32}
        @test e_promoted.value === 0.0f0

        # NaN promotion
        e_nan = ConstExtrap(NaN)
        e_nan32 = _promote_extrap(e_nan, Float32)
        @test e_nan32 isa ConstExtrap{Float32}
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
        @test occursin("ConstExtrap(NaN)", s_nan)
        @test occursin("ConstExtrap(0.0)", s_zero)
    end

    # ────────────────────────────────────────────
    # Factory
    # ────────────────────────────────────────────
    @testset "Extrap factory with value" begin
        e1 = Extrap(:constant)
        @test e1 isa ConstExtrap{Nothing}

        e2 = Extrap(:constant; value=NaN)
        @test e2 isa ConstExtrap{Float64}
        @test isnan(e2.value)

        e3 = Extrap(:constant; value=0.0)
        @test e3 isa ConstExtrap{Float64}
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
end
