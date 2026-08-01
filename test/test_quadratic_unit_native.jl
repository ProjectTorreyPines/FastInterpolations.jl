# ========================================
# Quadratic 1D native duck (unit) build — refac/duck-thomas Phase 4 contracts
# ========================================
# Internal-seam RED (the public unit API routes through the strip-twin today):
# `_compute_quadratic_coeffs` must produce d in [Y/X] and a in [Y/X²] natively.
# Surface/Series testitems are EQUIVALENCE pins — green via the twin now, and
# they must stay green unchanged when the twin is deleted.

@testitem "quadratic unit native: coeff kernels (internal seam)" begin
    using Unitful
    const FI = FastInterpolations

    xu = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
    yw = [1.0, 2.0, 0.5, 3.0, 2.5] .* u"W"
    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [1.0, 2.0, 0.5, 3.0, 2.5]

    cases = [
        ("Left(QuadraticFit)", Left(QuadraticFit()), Left(QuadraticFit())),
        ("Right(QuadraticFit)", Right(QuadraticFit()), Right(QuadraticFit())),
        ("MinCurvFit", MinCurvFit(), MinCurvFit()),
        ("Left(Deriv1)", Left(Deriv1(0.7u"W/s")), Left(Deriv1(0.7))),
        ("Right(Deriv2)", Right(Deriv2(-0.3u"W/s^2")), Right(Deriv2(-0.3))),
    ]

    for (name, bcu, bcf) in cases
        @testset "$name" begin
            xw = FI._policy_axis(xu, FI.NoBC(), eltype(xu), FI.StorePolicy())
            du, au = FI._compute_quadratic_coeffs(xw, yw, bcu)
            @test eltype(du) === typeof(1.0u"W/s")
            @test eltype(au) === typeof(1.0u"W/s^2")

            xwf = FI._policy_axis(xf, FI.NoBC(), Float64, FI.StorePolicy())
            df, af = FI._compute_quadratic_coeffs(xwf, yf, bcf)
            @test all(i -> ustrip(u"W/s", du[i]) === df[i], eachindex(df))
            @test all(i -> ustrip(u"W/s^2", au[i]) === af[i], eachindex(af))
        end
    end
end

@testitem "quadratic unit one-shot ≡ persistent (fork-free contract)" begin
    using Unitful

    xu = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
    yw = [1.0, 2.0, 0.5, 3.0, 2.5] .* u"W"
    y2 = [0.5, 1.5, 2.0, 0.0, 1.0] .* u"W"
    q = 2.2u"s"
    qv = [0.35, 2.2, 4.1] .* u"s"

    for (nm, bc) in (
            ("default", Left(QuadraticFit())),
            ("Left(Deriv1)", Left(Deriv1(0.4u"W/s"))),
            ("Right(Deriv2)", Right(Deriv2(-0.3u"W/s^2"))),
            ("MinCurvFit", MinCurvFit()),
        )
        @testset "$nm" begin
            itp = quadratic_interp(xu, yw; bc = bc)
            @test quadratic_interp(xu, yw, q; bc = bc) === itp(q)
            vv = quadratic_interp(xu, yw, qv; bc = bc)
            pv = itp(qv)
            @test all(i -> vv[i] === pv[i], eachindex(pv))
        end
    end

    @testset "Series one-shot" begin
        sitp = quadratic_interp(xu, Series(yw, y2))
        sv = quadratic_interp(xu, Series(yw, y2), q)
        sp = sitp(q)
        @test all(i -> sv[i] === sp[i], eachindex(sp))
    end
end

@testitem "quadratic unit: surface + Series equivalence (pre/post twin deletion)" begin
    using Unitful
    const FI = FastInterpolations

    xu = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
    yw = [1.0, 2.0, 0.5, 3.0, 2.5] .* u"W"
    y2 = [0.5, 1.5, 2.0, 0.0, 1.0] .* u"W"
    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [1.0, 2.0, 0.5, 3.0, 2.5]

    @testset "scalar surface" begin
        itp = quadratic_interp(xu, yw)
        ref = quadratic_interp(xf, yf)
        @test itp(2.2u"s") === ref(2.2) * u"W"
        @test eltype(itp.d) === typeof(1.0u"W/s")
        @test eltype(itp.a) === typeof(1.0u"W/s^2")
        itp_d2 = quadratic_interp(xu, yw; bc = Right(Deriv2(-0.3u"W/s^2")))
        ref_d2 = quadratic_interp(xf, yf; bc = Right(Deriv2(-0.3)))
        @test itp_d2(0.35u"s") === ref_d2(0.35) * u"W"
    end

    @testset "Series" begin
        sitp = quadratic_interp(xu, Series(yw, y2))
        v = sitp(2.2u"s")
        ref = quadratic_interp(xf, yf)
        @test v[1] === ref(2.2) * u"W"
        @test eltype(sitp.a) === typeof(1.0u"W/s^2")
        @test eltype(sitp.d) === typeof(1.0u"W/s")
    end
end
