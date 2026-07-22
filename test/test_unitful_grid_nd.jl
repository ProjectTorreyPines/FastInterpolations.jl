# ========================================
# Unitful grid axis: ND (ducktype-grid phase 5)
# ========================================
# 5a: all axes share ONE concrete unit — common-Tg machinery, all families.
# 5b: mixed-unit axes (x in s, y in m) — Linear/Constant only (per-axis
#     eltypes; solver-family PreCompute ND is excluded → friendly error).
# The 5b Linear case is the NumInt ND-Unitful regression this plan restores.

@testitem "Unitful ND 5a: same-unit axes, all families" begin
    using Unitful
    using InteractiveUtils: @which

    xs = [0.0, 1.0, 2.0, 3.0] .* u"s"
    ys = [0.0, 0.5, 1.0, 1.5, 2.0] .* u"s"
    xf = [0.0, 1.0, 2.0, 3.0]
    yf = [0.0, 0.5, 1.0, 1.5, 2.0]
    data = [Float64(i + j) for i in 1:4, j in 1:5] .* u"W"
    dataf = [Float64(i + j) for i in 1:4, j in 1:5]

    @testset "Linear ND: build/eval/integrate" begin
        itp = interp((xs, ys), data; method = LinearInterp())
        tw = interp((xf, yf), dataf; method = LinearInterp())
        @test itp((1.5u"s", 0.75u"s")) ≈ tw((1.5, 0.75)) * u"W"
        @test integrate(itp) ≈ integrate(tw) * u"W*s^2"
        @test integrate(itp, (0.5u"s", 0.25u"s"), (2.5u"s", 1.75u"s")) ≈
            integrate(tw, (0.5, 0.25), (2.5, 1.75)) * u"W*s^2"
    end

    @testset "one-shot ND integrate" begin
        v = integrate((xs, ys), data; method = LinearInterp())
        @test v ≈ integrate((xf, yf), dataf; method = LinearInterp()) * u"W*s^2"
    end

    @testset "@which pins: point-tuple vs SoA batch" begin
        itp = interp((xs, ys), data; method = LinearInterp())
        m_point = @which itp((1.5u"s", 0.75u"s"))
        m_soa = @which itp(([1.5, 2.0] .* u"s", [0.75, 1.0] .* u"s"))
        @test m_point !== m_soa
        # SoA batch works end-to-end
        out = itp(([1.5, 2.0] .* u"s", [0.75, 1.0] .* u"s"))
        @test eltype(out) === typeof(1.0u"W")
        @test out[1] ≈ itp((1.5u"s", 0.75u"s"))
    end
end

@testitem "Unitful ND 5b: mixed-unit axes (Linear/Constant) — NumInt regression" begin
    using Unitful

    # The regressed NumInt case: different units per axis + unit values.
    xs = [0.0, 1.0, 2.0] .* u"s"
    ym = [0.0, 0.5, 1.0, 1.5] .* u"m"
    xf = [0.0, 1.0, 2.0]
    yf = [0.0, 0.5, 1.0, 1.5]
    data = [Float64(i * j) for i in 1:3, j in 1:4] .* u"W"
    dataf = [Float64(i * j) for i in 1:3, j in 1:4]

    @testset "Linear: full-domain integrate (trapezoid ≡ NumInt)" begin
        itp = interp((xs, ym), data; method = LinearInterp())
        tw = interp((xf, yf), dataf; method = LinearInterp())
        @test integrate(itp) ≈ integrate(tw) * u"W*s*m"
        @test itp((1.5u"s", 0.75u"m")) ≈ tw((1.5, 0.75)) * u"W"
    end

    @testset "Constant: mixed-unit" begin
        itp = interp((xs, ym), data; method = ConstantInterp())
        tw = interp((xf, yf), dataf; method = ConstantInterp())
        @test integrate(itp) ≈ integrate(tw) * u"W*s*m"
    end

    @testset "solver families: friendly error (deferred)" begin
        @test_throws ArgumentError interp((xs, ym), data; method = CubicInterp())
    end
end

@testitem "Unitful ND: inference stability (@inferred, review pin F14)" begin
    using Unitful
    using Test: @inferred

    xs = [0.0, 1.0, 2.0] .* u"s"
    ym = [0.0, 0.5, 1.0, 1.5] .* u"m"
    xs2 = [0.0, 0.5, 1.0, 1.5] .* u"s"
    data = [Float64(i * j) for i in 1:3, j in 1:4] .* u"W"
    TW = typeof(1.0u"W")

    @testset "same-unit ND: eval / SoA batch / integrate" begin
        itp = interp((xs, xs2), data; method = LinearInterp())
        @test (@inferred itp((1.5u"s", 0.75u"s"))) isa TW
        @test (@inferred itp(([1.5, 2.0] .* u"s", [0.75, 1.0] .* u"s"))) isa Vector{TW}
        @test (@inferred integrate(itp)) isa typeof(1.0u"W*s^2")
    end

    @testset "mixed-unit ND (abstract-Tg): eval / integrate / gradient" begin
        itp = interp((xs, ym), data; method = LinearInterp())
        @test (@inferred itp((1.5u"s", 0.75u"m"))) isa TW
        @test (@inferred integrate(itp)) isa typeof(1.0u"W*s*m")
        g = @inferred gradient(itp, (1.5u"s", 0.75u"m"))
        @test g isa Tuple{typeof(1.0u"W/s"), typeof(1.0u"W/m")}
    end
end

@testitem "Unitful ND: 1-tuple adapters, vector-point, unit-Range axes (review F13)" begin
    using Unitful

    xu = [0.0, 1.0, 2.5, 3.0, 4.0] .* u"s"
    xf = [0.0, 1.0, 2.5, 3.0, 4.0]
    yu = [1.0, 2.0, 4.0, 8.0, 5.0] .* u"W"
    yf = [1.0, 2.0, 4.0, 8.0, 5.0]

    @testset "1-tuple adapter: unit SCALAR query returns a scalar" begin
        # `q::Real` forwarders didn't match Quantity — the call fell to a
        # generic arm and SILENTLY returned a 1-element Vector. Pin the shape.
        for f in (linear_interp, constant_interp, cubic_interp, quadratic_interp, pchip_interp, akima_interp, cardinal_interp)
            v = f((xu,), yu, 1.5u"s")
            @test v isa Unitful.Quantity
            @test v ≈ f((xf,), yf, 1.5) * u"W"
        end
    end

    @testset "1-tuple adapter: unit batch query" begin
        qs = [0.5, 1.5] .* u"s"
        @test linear_interp((xu,), yu, qs) ≈ linear_interp((xf,), yf, [0.5, 1.5]) .* u"W"
        @test pchip_interp((xu,), yu, qs) ≈ pchip_interp((xf,), yf, [0.5, 1.5]) .* u"W"
    end

    @testset "ND vector-point query (ForwardDiff-style) ≡ tuple query" begin
        gy = [0.0, 0.5, 1.0] .* u"m"
        data = [Float64(i * j) for i in 1:5, j in 1:3] .* u"W"
        itp = interp((xu, gy), data; method = LinearInterp())
        @test itp([1.5u"s", 0.25u"m"]) === itp((1.5u"s", 0.25u"m"))
    end

    @testset "ND unit-Range axes (allowlisted `_CachedRange` arms: generic siblings serve)" begin
        xr = 0.0u"s":1.0u"s":2.0u"s"
        yr = 0.0u"m":0.5u"m":1.0u"m"
        data = [Float64(i * j) for i in 1:3, j in 1:3] .* u"W"
        itp = interp((xr, yr), data; method = LinearInterp())
        tw = interp((0.0:1.0:2.0, 0.0:0.5:1.0), ustrip.(data); method = LinearInterp())
        @test itp((1.5u"s", 0.25u"m")) ≈ tw((1.5, 0.25)) * u"W"
        @test integrate(itp) ≈ integrate(tw) * u"W*s*m"
    end
end

@testitem "Unitful ND: vector-calculus vector query (review F12/C10)" begin
    using Unitful

    xs = [0.0, 1.0, 2.0] .* u"s"
    ym = [0.0, 0.5, 1.0] .* u"m"
    data = [Float64(i * j) for i in 1:3, j in 1:3] .* u"W"
    itp = interp((xs, ym), data; method = LinearInterp())

    # The documented Vector-query overloads rejected Vector{Quantity} while
    # the tuple form worked — mixed-unit coords make the vector eltype
    # abstract, which is fine for a point query.
    g_tuple = gradient(itp, (1.5u"s", 0.5u"m"))
    @test all(gradient(itp, [1.5u"s", 0.5u"m"]) .≈ g_tuple)
    @test value_gradient(itp, [1.5u"s", 0.5u"m"])[1] ≈ itp((1.5u"s", 0.5u"m"))
end

@testitem "Unitful ND: hetero engine rejects unit grids friendly (review F11)" begin
    using Unitful

    # The per-axis (hetero) ND engine — which also backs PCHIP/Akima/Cardinal ND —
    # missed the solver-grid guard: unit grids died in deep MethodErrors
    # (`_collapse_dims`, `_build_nd_coeffs_hetero`, ctor). Pin the friendly error
    # for both builders (OnTheFly + PreCompute) and both unit layouts.
    xs = [0.0, 1.0, 2.0] .* u"s"
    xs2 = [0.0, 0.5, 1.0, 1.5] .* u"s"
    ym = [0.0, 0.5, 1.0, 1.5] .* u"m"
    data = [Float64(i * j) for i in 1:3, j in 1:4] .* u"W"

    for build in (
            () -> interp((xs, ym), data; method = (LinearInterp(), ConstantInterp())),
            () -> interp((xs, xs2), data; method = (LinearInterp(), CubicInterp())),
            () -> pchip_interp((xs, xs2), data),
            () -> cardinal_interp((xs, ym), data),
        )
        err = try
            build()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("unit-carrying", sprint(showerror, err))
    end
end

@testitem "Unitful ND: zero-alloc hot path, mixed-unit abstract-Tg (review pin F6)" setup = [AllocConstants] begin
    using Unitful

    # Mixed-unit axes exercise the abstract-Tg per-axis machinery — the alloc
    # contract must hold there too (devirtualized axis maps, no closure boxes).
    xs = [0.0, 1.0, 2.0] .* u"s"
    ym = [0.0, 0.5, 1.0, 1.5] .* u"m"
    data = [Float64(i * j) for i in 1:3, j in 1:4] .* u"W"
    itp = interp((xs, ym), data; method = LinearInterp())
    q = (1.5u"s", 0.75u"m")

    itp(q)
    integrate(itp)   # warmup
    @test (@allocated itp(q)) <= ND_ALLOC_THRESHOLD
    @test (@allocated integrate(itp)) <= ND_ALLOC_THRESHOLD
end

@testitem "Unitful ND: Constant derivative carries grid⁻ᴺ units (Range-duck audit)" begin
    using Unitful
    Wps = typeof(1.0u"W/s")
    Wps2 = typeof(1.0u"W/s^2")

    # Same as 1D but for the ND `_constant_nd_evaluate` deriv path (`kernel * 0` dropped
    # the per-axis grid⁻ᴺ scale). Same-unit axes give a CONCRETE `W/s` buffer, so the
    # batch deriv CRASHED (DimensionError) — mixed-unit hid it behind an abstract eltype.
    data = [Float64(i * j) for i in 1:4, j in 1:4] .* u"W"
    for (gname, xg, yg) in (
            ("Vector", [0.0, 1.0, 2.0, 3.0] .* u"s", [0.0, 1.0, 2.0, 3.0] .* u"s"),
            ("LinRange", LinRange(0.0u"s", 3.0u"s", 4), LinRange(0.0u"s", 3.0u"s", 4)),
        )
        itp = interp((xg, yg), data; method = ConstantInterp())
        @testset "$gname: scalar ∂/∂x units" begin
            d1 = itp((1.5u"s", 1.5u"s"); deriv = (DerivOp(1), DerivOp(0)))
            d2 = itp((1.5u"s", 1.5u"s"); deriv = (DerivOp(1), DerivOp(1)))
            @test d1 isa Wps
            @test iszero(d1)
            @test d2 isa Wps2
            @test iszero(d2)
        end
        @testset "$gname: batch deriv (was DimensionError)" begin
            out = itp(([1.0, 2.0] .* u"s", [1.0, 2.0] .* u"s"); deriv = (DerivOp(1), DerivOp(0)))
            @test eltype(out) === Wps
            @test all(iszero, out)
        end
    end
end
