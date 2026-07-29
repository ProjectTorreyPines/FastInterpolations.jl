# ========================================
# Cubic 1D native duck (unit) build — refac/duck-thomas Phase 2 contracts
# ========================================
# The public unit API routes through the strip-twin today, so these tests pin
# the NATIVE build/solve seam directly (internal-path RED strategy). After the
# twin is deleted, `cubic_interp` itself lands on these paths.
#
# Unit conventions: SI-coherent u"s"/u"W" so stripped-vs-native mantissas are
# bit-identical (conversion factor exactly 1.0) — pins use `===` on ustrip.

@testitem "cubic unit native: derivative-BC builder + solve (internal seam)" begin
    using Unitful
    const FI = FastInterpolations

    xu = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
    yw = [1.0, 2.0, 0.5, 3.0, 2.5] .* u"W"
    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [1.0, 2.0, 0.5, 3.0, 2.5]

    # (name, unit-payload BC pair, stripped-payload BC pair)
    cases = [
        ("Deriv1", (FI.Deriv1(0.7u"W/s"), FI.Deriv1(0.7u"W/s")), (FI.Deriv1(0.7), FI.Deriv1(0.7))),
        ("Deriv2", (FI.Deriv2(-0.3u"W/s^2"), FI.Deriv2(-0.3u"W/s^2")), (FI.Deriv2(-0.3), FI.Deriv2(-0.3))),
        ("Deriv3", (FI.Deriv3(0.2u"W/s^3"), FI.Deriv3(0.2u"W/s^3")), (FI.Deriv3(0.2), FI.Deriv3(0.2))),
        ("mixed D1/D2", (FI.Deriv1(0.7u"W/s"), FI.Deriv2(-0.3u"W/s^2")), (FI.Deriv1(0.7), FI.Deriv2(-0.3))),
    ]

    for (name, (lu, ru), (lf, rf)) in cases
        @testset "$name" begin
            cache_u = FI._build_derivative_bc_cache(xu, lu, ru)

            # Witness-typed factorization spaces (Phase 1 contract through the builder):
            # l dimensionless, du grid-space, inv_d reciprocal-grid-space.
            @test eltype(cache_u.x) === typeof(1.0u"s")
            @test eltype(cache_u.thomas.dl) === Float64
            @test eltype(cache_u.thomas.du) === typeof(1.0u"s")
            @test eltype(cache_u.thomas.inv_d) === typeof(inv(1.0u"s"))

            Tz = FI._promote_eltype(FI._coeff_op2, eltype(cache_u.x), eltype(yw))
            zu = Vector{Tz}(undef, length(yw))
            FI._solve_system!(zu, cache_u, yw, FI.BCPair(lu, ru))

            # Reference: stripped Real system (identical mantissas, SI-coherent).
            cache_f = FI._build_derivative_bc_cache(xf, lf, rf)
            zf = Vector{Float64}(undef, length(yf))
            FI._solve_system!(zf, cache_f, yf, FI.BCPair(lf, rf))

            @test all(i -> ustrip(u"W/s^2", zu[i]) === zf[i], eachindex(zf))
        end
    end
end

@testitem "cubic unit native: impl seam end-to-end (normalized + payload BCs)" begin
    using Unitful
    const FI = FastInterpolations

    xu = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
    yw = [1.0, 2.0, 0.5, 3.0, 2.5] .* u"W"
    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [1.0, 2.0, 0.5, 3.0, 2.5]

    cases = [
        ("CubicFit", CubicFit(), CubicFit()),
        ("ZeroCurvBC", ZeroCurvBC(), ZeroCurvBC()),
        ("ZeroSlopeBC", ZeroSlopeBC(), ZeroSlopeBC()),
        ("Deriv1", Deriv1(0.7u"W/s"), Deriv1(0.7)),
        ("Deriv2", Deriv2(-0.3u"W/s^2"), Deriv2(-0.3)),
        ("BCPair mixed", BCPair(Deriv1(0.7u"W/s"), Deriv2(-0.3u"W/s^2")), BCPair(Deriv1(0.7), Deriv2(-0.3))),
    ]

    for (name, bcu, bcf) in cases
        @testset "$name" begin
            itp_u = FI._cubic_interp_impl(xu, yw, bcu, NoExtrap(), false, AutoSearch())
            ref = cubic_interp(xf, yf; bc = bcf, autocache = false)

            @test eltype(itp_u.cache.x) === typeof(1.0u"s")
            @test all(i -> ustrip(u"W/s^2", itp_u.z[i]) === ref.z[i], eachindex(ref.z))
            @test itp_u(2.2u"s") === ref(2.2) * u"W"
            @test itp_u(0.35u"s") === ref(0.35) * u"W"
        end
    end

    @testset "H9: normalized zero-BC payloads live in payload space" begin
        # ZeroCurvBC → Deriv2 payload is [Y/X²]; ZeroSlopeBC → Deriv1 payload is
        # [Y/X]. A value-space (`0 * first(y)`) zero would make the RHS rule
        # `bc.val * oneunit(Tg)` dimensionally wrong — pin the stored payloads.
        itp_c = FI._cubic_interp_impl(xu, yw, ZeroCurvBC(), NoExtrap(), false, AutoSearch())
        @test itp_c.bc.left.val === 0.0u"W/s^2"
        @test itp_c.bc.right.val === 0.0u"W/s^2"

        itp_s = FI._cubic_interp_impl(xu, yw, ZeroSlopeBC(), NoExtrap(), false, AutoSearch())
        @test itp_s.bc.left.val === 0.0u"W/s"
        @test itp_s.bc.right.val === 0.0u"W/s"
    end
end

@testitem "cubic Real release-parity pins (per BC, bit-exact)" begin
    # Literals generated from the pre-Phase-2 branch (Real path proven
    # bit-identical to the v0.4.17 release by the Phase 1 A/B) — Phase 2+ edits
    # to rows/RHS/normalize must not move a single bit on Real grids.
    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [1.0, 2.0, 0.5, 3.0, 2.5]

    pins = [
        (
            CubicFit(),
            [-10.093793472445158, -3.612413055109683, 10.770572498662384, -3.327340823970036, -16.0506153023007],
            -0.10312808988764004, 2.166403347378277,
        ),
        (
            ZeroCurvBC(),
            [0.0, -6.150537634408604, 12.501792114695343, -9.562724014336917, 0.0],
            -0.10735483870967825, 1.6648306451612904,
        ),
        (
            ZeroSlopeBC(),
            [6.910112359550562, -7.820224719101122, 13.46067415730337, -12.224719101123595, 6.779026217228464],
            -0.09069662921348325, 1.317983848314607,
        ),
        (
            Deriv1(0.7),
            [4.558426966292134, -7.3168539325842685, 13.350561797752809, -12.853932584269666, 8.493632958801498],
            -0.11504719101123588, 1.4393448735955057,
        ),
        (
            Deriv2(-0.3),
            [-0.3, -6.077956989247312, 12.459856630824373, -9.444982078853046, -0.3],
            -0.10805161290322544, 1.6798841733870968,
        ),
        (
            Deriv3(0.2),
            [-5.093015873015873, -4.893015873015873, 11.705396825396825, -6.964126984126984, -6.664126984126984],
            -0.1118857142857143, 1.9190930555555554,
        ),
        (
            BCPair(Deriv1(0.7), Deriv2(-0.3)),
            [4.4784848484848485, -7.156969696969696, 12.87090909090909, -9.496363636363638, -0.3],
            -0.07475636363636377, 1.436162178030303,
        ),
    ]

    for (bc, z_pin, v22, v035) in pins
        @testset "$(typeof(bc))" begin
            itp = cubic_interp(xf, yf; bc = bc, autocache = false)
            @test all(i -> itp.z[i] === z_pin[i], eachindex(z_pin))
            @test itp(2.2) === v22
            @test itp(0.35) === v035
        end
    end
end

@testitem "cubic Series unit native: self-consistent cache + payload spaces + periodic" begin
    using Unitful
    const FI = FastInterpolations

    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [1.0, 2.0, 0.5, 3.0, 2.5]
    y2f = [0.5, 1.5, 2.0, 0.0, 1.0]
    xu = xf .* u"s"
    yw = yf .* u"W"
    y2 = y2f .* u"W"

    @testset "default BC: equivalence + native factorization on the cache" begin
        sitp = cubic_interp(xu, Series(yw, y2))
        ref = cubic_interp(xf, Series(yf, y2f))
        v = sitp(2.2u"s")
        vr = ref(2.2)
        @test v[1] === vr[1] * u"W"
        @test v[2] === vr[2] * u"W"
        @test eltype(sitp.z) === typeof(1.0u"W/s^2")
        # The cache must be self-consistent: a unit axis carries a unit-typed
        # factorization (the twin paired a Float64 thomas with a unit axis and
        # forbade reuse — that lie ends here).
        @test eltype(sitp.cache.thomas.du) === typeof(1.0u"s")
        @test eltype(sitp.cache.thomas.inv_d) === typeof(inv(1.0u"s"))
    end

    @testset "ZeroCurvBC: payload zero lives in [Y/X²]" begin
        sitp = cubic_interp(xu, Series(yw, y2); bc = ZeroCurvBC())
        ref = cubic_interp(xf, Series(yf, y2f); bc = ZeroCurvBC())
        @test sitp(0.35u"s")[1] === ref(0.35)[1] * u"W"
        @test sitp.bc.left.val === 0.0u"W/s^2"
    end

    @testset "per-series BC array (unit payloads)" begin
        bcs = [
            BCPair(Deriv1(0.7u"W/s"), Deriv1(0.7u"W/s")),
            BCPair(Deriv2(-0.3u"W/s^2"), Deriv2(-0.3u"W/s^2")),
        ]
        bcs_f = [
            BCPair(Deriv1(0.7), Deriv1(0.7)),
            BCPair(Deriv2(-0.3), Deriv2(-0.3)),
        ]
        sitp = cubic_interp(xu, Series(yw, y2); bc = bcs)
        ref = cubic_interp(xf, Series(yf, y2f); bc = bcs_f)
        v = sitp(2.2u"s")
        vr = ref(2.2)
        @test v[1] === vr[1] * u"W"
        @test v[2] === vr[2] * u"W"
    end

    @testset "PeriodicBC Series (new capability — twin rejected this)" begin
        ywp = [1.0, 2.0, 0.5, 3.0, 1.0] .* u"W"
        y2p = [0.5, 1.5, 2.0, 0.0, 0.5] .* u"W"
        sitp = cubic_interp(xu, Series(ywp, y2p); bc = PeriodicBC())
        ref = cubic_interp(xf, Series([1.0, 2.0, 0.5, 3.0, 1.0], [0.5, 1.5, 2.0, 0.0, 0.5]); bc = PeriodicBC())
        v = sitp(2.2u"s")
        vr = ref(2.2)
        @test v[1] === vr[1] * u"W"
        @test v[2] === vr[2] * u"W"
    end
end

@testitem "cubic unit native: periodic S-M build + solve" begin
    using Unitful
    const FI = FastInterpolations

    # Sherman-Morrison is dimensionally clean end-to-end: u is the
    # DIMENSIONLESS structural vector [1,0,…,1], q = A'⁻¹u lives in [1/X]
    # (the inv_d space), vᵀq is dimensionless, and the correction
    # z − (vᵀy/(1+vᵀq))·q stays in [Y/X²]. Real grids collapse u and q to one
    # eltype — the historic single-buffer in-place build.
    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [1.0, 2.0, 0.5, 3.0, 1.0]          # closed cycle
    xu = xf .* u"s"
    yw = yf .* u"W"

    @testset "inclusive: bit parity + witness spaces" begin
        itp_u = cubic_interp(xu, yw; bc = PeriodicBC(), autocache = false)
        ref = cubic_interp(xf, yf; bc = PeriodicBC(), autocache = false)

        @test eltype(itp_u.cache.q) === typeof(inv(1.0u"s"))
        @test eltype(itp_u.cache.thomas.dl) === Float64
        @test eltype(itp_u.cache.thomas.inv_d) === typeof(inv(1.0u"s"))
        @test all(i -> ustrip(u"W/s^2", itp_u.z[i]) === ref.z[i], eachindex(ref.z))
        @test itp_u(2.2u"s") === ref(2.2) * u"W"
        @test itp_u(6.0u"s") === ref(6.0) * u"W"   # wrap extrapolation
    end

    @testset "exclusive: explicit unit period" begin
        xe = [0.0, 1.0, 2.5, 3.0] .* u"s"
        ye = [1.0, 2.0, 0.5, 3.0] .* u"W"
        itp_u = cubic_interp(xe, ye; bc = PeriodicBC(endpoint = :exclusive, period = 4.5u"s"), autocache = false)
        ref = cubic_interp([0.0, 1.0, 2.5, 3.0], [1.0, 2.0, 0.5, 3.0]; bc = PeriodicBC(endpoint = :exclusive, period = 4.5), autocache = false)
        @test itp_u(2.2u"s") === ref(2.2) * u"W"
        @test itp_u(4.0u"s") === ref(4.0) * u"W"   # seam cell
    end

    @testset "periodic bank: unit grids hit" begin
        FI.clear_cubic_cache!()
        i1 = cubic_interp(xu, yw; bc = PeriodicBC())
        i2 = cubic_interp(xu, yw; bc = PeriodicBC())
        @test i2.cache === i1.cache
    end
end
