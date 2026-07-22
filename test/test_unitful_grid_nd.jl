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
