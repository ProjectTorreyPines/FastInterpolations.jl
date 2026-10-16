# ═══════════════════════════════════════════════════════════════════════════════
# test_bigfloat_support.jl
#
# Arbitrary-precision (BigFloat / BigInt / Rational{BigInt}) regression pins.
#
# 1. Integer-parameterized ranges: `length` follows the range's integer type, so
#    `big.(0:16) ./ 16`, `big(0):big(16)`, `big(0):big(2):big(32)`, and
#    `Base.OneTo(big(17))` report a `BigInt` length. The `_CachedRange` normalizer
#    must accept it (stored as `len::Int`) and behave exactly like the Vector twin.
#
# 2. Precision leakage: every Hermite-family rule (PCHIP / Cardinal / Akima /
#    user slopes) reproduces linear data exactly, so its integral of
#    `f(x) = x/3 + 1/7` must match the closed form at BigFloat precision. A
#    Float64 literal coefficient (e.g. `2/3`) in the antiderivative basis caps
#    the error at ~1e-18 instead of ~1e-77. The ND cubic bounded integrate's
#    partial-cell weights share that basis.
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "BigInt-length ranges build like their Vector twin" begin
    f(x) = x / 3 + big(1) // 7
    xq = [big(1) / 7, big(3) / 7, big(5) / 7]

    # Each grid's `length` is a BigInt; values span [0, 1] or [0, 16] / [0, 32].
    grids = (
        big.(0:16) ./ 16,                                    # StepRangeLen{BigFloat,…,BigInt}
        big(0):big(16),                                      # UnitRange{BigInt}
        big(0):big(2):big(32),                               # StepRange{BigInt,BigInt}
        (big(0) // 1):(big(1) // 16):(big(1) // 1),          # StepRange{Rational{BigInt}}
        Base.OneTo(big(17)),                                 # OneTo{BigInt}
    )
    for g in grids
        @test length(g) isa BigInt
        v = collect(g)
        q = first(v) .+ (last(v) - first(v)) .* xq
        for mk in (linear_interp, cubic_interp, pchip_interp, quadratic_interp, constant_interp)
            y = BigFloat.(f.(v))
            itp_r = mk(g, y)
            itp_v = mk(v, y)
            @test itp_r(q) == itp_v(q)
            @test eltype(itp_r(q)) === BigFloat
            @test mk(g, y, q) == mk(v, y, q)    # one-shot path
        end
    end

    # ND: one BigInt-length axis beside a BigInt UnitRange axis.
    gx = big.(0:8) ./ 8
    gy = big(0):big(10)
    F = [x / 3 + y / 7 + big(1) // 11 for x in gx, y in gy]
    q2 = (big(1) / 3, big(37) / 7)
    @test cubic_interp((gx, gy), F)(q2) ≈ q2[1] / 3 + q2[2] / 7 + big(1) // 11 atol = 1.0e-70
    @test linear_interp((gx, gy), F)(q2) == linear_interp((collect(gx), collect(gy)), F)(q2)

    # Slicing a normalized range with a BigInt index range keeps the range form.
    r = FastInterpolations._to_float(0.0:0.25:2.0, Float64)
    s = r[big(2):big(4)]
    @test s isa FastInterpolations._CachedRange
    @test length(s) === 3
    @test collect(s) == [0.25, 0.5, 0.75]
end

@testitem "Hermite-family integrate keeps BigFloat precision" begin
    tol = 1.0e-70                 # BigFloat(256) eps ≈ 1e-77; a Float64 leak gives ~1e-18
    F(x) = x^2 / 6 + x / 7        # antiderivative of x/3 + 1/7
    g = collect(range(big(0), big(1); length = 17))
    y = g ./ 3 .+ big(1) // 7
    a, b = big(1) / 7, big(5) / 7

    builders = (
        pchip_interp,
        cardinal_interp,
        akima_interp,
        (x, v) -> hermite_interp(x, v, fill(big(1) / 3, length(x))),
    )
    for mk in builders, grid in (g, range(big(0), big(1); length = 17))
        itp = mk(grid, y)
        @test abs(integrate(itp, big(0), big(1)) - F(big(1))) < tol      # full cells
        @test abs(integrate(itp, a, b) - (F(b) - F(a))) < tol            # partial cells
        @test maximum(abs.(cumulative_integrate(itp) .- F.(g))) < tol
    end

    # ND cubic bounded integrate: its partial-cell node weights share the same
    # antiderivative basis (full-domain uses closed-form weights instead).
    gx = collect(range(big(0), big(1); length = 9))
    gy = collect(range(big(0), big(2); length = 11))
    Fd = [x / 3 + y / 7 + big(1) // 11 for x in gx, y in gy]
    G(x, y) = x^2 * y / 6 + x * y^2 / 14 + x * y / 11   # ∫∫ (x/3 + y/7 + 1/11)
    lo, hi = (big(1) / 7, big(2) / 9), (big(5) / 7, big(13) / 9)
    exact = G(hi...) - G(lo[1], hi[2]) - G(hi[1], lo[2]) + G(lo...)
    for itp in (cubic_interp((gx, gy), Fd), cubic_interp((range(big(0), big(1); length = 9), gy), Fd))
        @test abs(integrate(itp, lo, hi) - exact) < tol
    end

    # Float32 stays Float32 (no silent widening of the result type).
    g32 = Float32.(0:16) ./ 16
    @test integrate(pchip_interp(g32, g32 ./ 3 .+ 1.0f0 / 7), 0.0f0, 1.0f0) isa Float32
end
