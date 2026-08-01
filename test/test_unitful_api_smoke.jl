# ========================================
# Unitful sibling-path contract (post PR #199 smoke-sweep fixes)
# ========================================
# PR #199 relaxed the grid axis to Tg <: Number; forward eval + integrate were
# the focus and work. These pin the *sibling* public paths that were left on the
# old <:Real bound and crashed on Unitful grids — all forward-mode, all expected
# to work (triage: claudedocs/2026-07-24-unitful-api-smoke-triage.md):
#   A  show(::MIME"text/plain")   — _format_num value fallback
#   B  1D DerivativeView eval     — deriv1/2/3(itp)(q::Quantity)
#   E  ND GriddedQuery eval       — _anchor_loc coord promotion
#   G  Series eval                — query bound widened to Number
# Adjoints / mixed-unit hessian remain design-excluded (kept <:Real) — their
# friendly-error contract is pinned separately. Solver-family Series builds
# are unit-native since the duck Thomas core (see test_cubic_unit_native.jl).

@testitem "Unitful A: show(text/plain) renders for every built family" begin
    using Unitful
    xr = range(1.0u"s", 10.0u"s", length = 9)
    xv = collect(xr)
    y = (xr ./ u"s") .^ 2 .* u"m"
    dy = (2 .* (xr ./ u"s")) .* (u"m" / u"s")

    builders = Any[
        ("linear", linear_interp(xr, y)),
        ("constant", constant_interp(xr, y)),
        ("quadratic", quadratic_interp(xr, y)),
        ("cubic", cubic_interp(xr, y)),
        ("pchip", pchip_interp(xr, y)),
        ("cardinal", cardinal_interp(xr, y)),
        ("akima", akima_interp(xr, y)),
        ("hermite", hermite_interp(xr, y, dy)),
        ("linear/vec", linear_interp(xv, y)),
        ("series", linear_interp(xr, Series(hcat(y, 2 .* y)))),
    ]
    @testset "$name text/plain shows units, no throw" for (name, itp) in builders
        s = sprint(show, MIME("text/plain"), itp)   # would MethodError pre-fix
        @test occursin("s", s)                       # grid bound renders with its unit
    end

    # ND grid row (Linear ND, same-unit — build supported)
    gs = range(1.0u"s", 4.0u"s", length = 4)
    gs2 = range(1.0u"s", 4.0u"s", length = 4)
    nd = [Float64(i + j) for i in 1:4, j in 1:4]
    itp_nd = interp((gs, gs2), nd; method = LinearInterp())
    @test occursin("s", sprint(show, MIME("text/plain"), itp_nd))
end

@testitem "Unitful B: 1D DerivativeView eval == keyword deriv, unit-typed" begin
    using Unitful
    xr = range(1.0u"s", 10.0u"s", length = 9)
    y = (xr ./ u"s") .^ 2 .* u"m"
    q = 2.5u"s"

    @testset "$name deriv1 view matches deriv=DerivOp(1)" for (name, itp) in Any[
            ("linear", linear_interp(xr, y)),
            ("cubic", cubic_interp(xr, y)),
            ("pchip", pchip_interp(xr, y)),
            ("akima", akima_interp(xr, y)),
        ]
        d1 = deriv1(itp)(q)                       # would MethodError pre-fix
        @test d1 == itp(q; deriv = DerivOp(1))
        @test d1 isa Quantity                     # m/s, dimensionful
        @test unit(d1) == u"m" / u"s"
        # vector query path (AbstractArray branch) also widened
        @test deriv1(itp)([q, 3.5u"s"]) == [itp(q; deriv = DerivOp(1)), itp(3.5u"s"; deriv = DerivOp(1))]
    end

    # cubic 2nd derivative (m/s²)
    itp = cubic_interp(xr, y)
    @test deriv2(itp)(q) == itp(q; deriv = DerivOp(2))
    @test unit(deriv2(itp)(q)) == u"m" / u"s"^2
end

@testitem "Unitful E: ND GriddedQuery eval == point-wise (same-unit)" begin
    using Unitful
    gs = range(1.0u"s", 4.0u"s", length = 4)
    gs2 = range(1.0u"s", 4.0u"s", length = 4)
    nd = [Float64(i + j) for i in 1:4, j in 1:4]

    @testset "$name ND gridded matches comprehension" for (name, itp) in Any[
            ("linear", interp((gs, gs2), nd; method = LinearInterp())),
            ("constant", interp((gs, gs2), nd; method = ConstantInterp())),
        ]
        tx = [1.5u"s", 2.5u"s", 3.5u"s"]
        ty = [1.5u"s", 2.5u"s"]
        C = itp(GriddedQuery((tx, ty)))          # would MethodError pre-fix
        ref = [itp((x, y)) for x in tx, y in ty]
        @test size(C) == (length(tx), length(ty))
        @test all(C .== ref)
    end
end

@testitem "Unitful G: Series eval == per-series scalar, unit-typed" begin
    using Unitful
    xr = range(1.0u"s", 10.0u"s", length = 9)
    y1 = (xr ./ u"s") .^ 2 .* u"m"
    y2 = 2 .* y1
    q = 2.5u"s"

    sitp = linear_interp(xr, Series(hcat(y1, y2)))
    out = sitp(q)                                 # would MethodError pre-fix
    @test out[1] == linear_interp(xr, y1)(q)
    @test out[2] == linear_interp(xr, y2)(q)
    @test eltype(out) <: Quantity
    @test unit(out[1]) == u"m"

    # vector query → Vector{Vector} (outer: series, inner: query points)
    outv = sitp([q, 3.5u"s"])
    @test outv[1][1] == linear_interp(xr, y1)(q)
    @test outv[2][2] == linear_interp(xr, y2)(3.5u"s")
end

# ── Codex-review follow-ups: complete the sibling-path coverage ──

@testitem "Unitful #1: Series eval is batch-size- and API-form-independent" begin
    using Unitful
    xr = range(1.0u"s", 10.0u"s", length = 9)
    y1 = (xr ./ u"s") .^ 2 .* u"m"
    y2 = 2 .* y1
    Y = hcat(y1, y2)
    sitp = linear_interp(xr, Series(Y))

    # NQ > 16 selects the `_linear_series_kq!` large-batch kernel → `_fill_series_anchors!`
    q17 = collect(range(1.5u"s", 8.5u"s", length = 17))
    big = sitp(q17)                                  # would MethodError pre-fix (S<:Real)
    @test big[1] == [linear_interp(xr, y1)(qi) for qi in q17]
    @test big[2] == [linear_interp(xr, y2)(qi) for qi in q17]

    # One-shot Series forms (no persistent interpolant)
    q = 2.5u"s"
    @test linear_interp(xr, Series(Y), q) == sitp(q)                 # scalar one-shot
    @test linear_interp(xr, Series(Y), [q, 3.5u"s"]) == sitp([q, 3.5u"s"])  # vector one-shot
end

@testitem "Unitful #1b: Constant + Quadratic Series eval (parity with Linear)" begin
    using Unitful
    xr = range(1.0u"s", 10.0u"s", length = 9)
    y1 = (xr ./ u"s") .^ 2 .* u"m"
    y2 = 2 .* y1
    Y = hcat(y1, y2)
    q = 2.5u"s"

    @testset "constant series" begin
        sitp = constant_interp(xr, Series(Y))
        out = sitp(q)
        @test out[1] == constant_interp(xr, y1)(q)
        @test out[2] == constant_interp(xr, y2)(q)
        @test out == constant_interp(xr, Series(Y), q)        # one-shot parity

        qs = [q, 3.5u"s"]
        outv = sitp(qs)
        @test outv[1] == [constant_interp(xr, y1)(qi) for qi in qs]
        @test outv[2] == [constant_interp(xr, y2)(qi) for qi in qs]
    end
    # ── Solver families (Quadratic, Cubic) under units: Series vs scalar ──
    # These two do NOT run the same arithmetic. The solve is unit-hostile by
    # storage, so both nondimensionalize — but at different points: the Series
    # build strips the whole matrix once (`_quadratic_series_units` /
    # `_cubic_series_units`), the scalar build strips each vector
    # (`_quadratic_interp_units` / `_cubic_interp_units`). Same value, last-ULP
    # apart by construction, and which way it rounds depends on the LLVM version.
    # So Series-vs-scalar asserts use `rtol`; Series-vs-Series asserts (one-shot
    # vs persistent, in-place vs out-of-place) stay exact — there the two sides
    # really are the same computation, which is the point of the assert.
    SERIES_RTOL = 1.0e-15
    @testset "quadratic series" begin
        sitp = quadratic_interp(xr, Series(Y))
        out = sitp(q)
        @test out[1] ≈ quadratic_interp(xr, y1)(q) rtol = SERIES_RTOL
        @test out[2] ≈ quadratic_interp(xr, y2)(q) rtol = SERIES_RTOL
        out_inplace = similar(out)
        @test sitp(out_inplace, q) == out

        qs = [q, 3.5u"s"]
        outv = sitp(qs)
        @test outv[1] ≈ [quadratic_interp(xr, y1)(qi) for qi in qs] rtol = SERIES_RTOL
        @test outv[2] ≈ [quadratic_interp(xr, y2)(qi) for qi in qs] rtol = SERIES_RTOL
        outv_inplace = [similar(v) for v in outv]
        @test sitp(outv_inplace, qs) == outv

        d = sitp(q; deriv = DerivOp(1))
        @test d[1] ≈ quadratic_interp(xr, y1)(q; deriv = DerivOp(1)) rtol = SERIES_RTOL
        @test unit(eltype(d)) == u"m" / u"s"

        @test quadratic_interp(xr, Series(Y), q) == out
        @test quadratic_interp(xr, Series(Y), qs) == outv
        one_inplace = similar(out)
        @test quadratic_interp!(one_inplace, xr, Series(Y), q) == out
        onev_inplace = [similar(v) for v in outv]
        @test quadratic_interp!(onev_inplace, xr, Series(Y), qs) == outv

        for bc in (Left(QuadraticFit()), Right(QuadraticFit()))
            sitp_bc = quadratic_interp(xr, Series(Y); bc = bc)
            @test sitp_bc(q) ≈ [quadratic_interp(xr, y; bc = bc)(q) for y in (y1, y2)] rtol = SERIES_RTOL
            @test sitp_bc(q; deriv = DerivOp(1)) ≈
                [quadratic_interp(xr, y; bc = bc)(q; deriv = DerivOp(1)) for y in (y1, y2)] rtol = SERIES_RTOL
        end
    end
    # Cubic series build mirrors the scalar `_cubic_interp_units` strip→solve→
    # reattach (`z` is order 2 → `Y/X²`): the Thomas solve is unit-hostile by
    # STORAGE, so it runs on the nondimensionalized twin. Parity with quadratic,
    # including the `rtol` rule above.
    @testset "cubic series" begin
        sitp = cubic_interp(xr, Series(Y))
        out = sitp(q)
        @test out[1] ≈ cubic_interp(xr, y1)(q) rtol = SERIES_RTOL
        @test out[2] ≈ cubic_interp(xr, y2)(q) rtol = SERIES_RTOL

        qs = [q, 3.5u"s"]
        outv = sitp(qs)
        @test outv[1] ≈ [cubic_interp(xr, y1)(qi) for qi in qs] rtol = SERIES_RTOL
        @test outv[2] ≈ [cubic_interp(xr, y2)(qi) for qi in qs] rtol = SERIES_RTOL

        d = sitp(q; deriv = DerivOp(1))
        @test d[1] ≈ cubic_interp(xr, y1)(q; deriv = DerivOp(1)) rtol = SERIES_RTOL
        @test unit(eltype(d)) == u"m" / u"s"
        d2 = sitp(q; deriv = DerivOp(2))
        @test d2[1] ≈ cubic_interp(xr, y1)(q; deriv = DerivOp(2)) rtol = SERIES_RTOL
        @test unit(eltype(d2)) == u"m" / u"s"^2

        @test cubic_interp(xr, Series(Y), q) == out          # one-shot parity
        @test cubic_interp(xr, Series(Y), qs) == outv

        # In-place is a separate code path from the out-of-place delegation
        # (mirrors the quadratic testset above): persistent + one-shot, both shapes.
        out_inplace = similar(out)
        @test sitp(out_inplace, q) == out
        outv_inplace = [similar(v) for v in outv]
        @test sitp(outv_inplace, qs) == outv
        one_inplace = similar(out)
        @test cubic_interp!(one_inplace, xr, Series(Y), q) == out
        onev_inplace = [similar(v) for v in outv]
        @test cubic_interp!(onev_inplace, xr, Series(Y), qs) == outv

        # BC payloads carry derivative units — `_strip_bc_units` must rescale them.
        # `CubicFit` fits a polynomial at the boundary, so this is the assert most
        # exposed to reassociation: it is the one that broke on Ubuntu LTS.
        for bc in (CubicFit(), ZeroCurvBC(), Deriv1(2.0u"m" / u"s"))
            sitp_bc = cubic_interp(xr, Series(Y); bc = bc)
            @test sitp_bc(q) ≈ [cubic_interp(xr, y; bc = bc)(q) for y in (y1, y2)] rtol = SERIES_RTOL
        end

        # PeriodicBC stays unsupported under units (same friendly error as scalar).
        @test_throws ArgumentError cubic_interp(xr, Series(Y); bc = PeriodicBC())
    end
end

@testitem "Unitful #4: in-place DerivativeView accepts Number queries" begin
    using Unitful
    xr = range(1.0u"s", 10.0u"s", length = 9)
    y = (xr ./ u"s") .^ 2 .* u"m"
    itp = cubic_interp(xr, y)
    qs = [2.5u"s", 3.5u"s"]

    # 1D in-place vector deriv (parent supports it; only the view wrapper was <:Real).
    # Caller allocates a derivative-typed buffer (m/s), since deriv1 yields m/s.
    out = Vector{typeof(1.0u"m" / u"s")}(undef, 2)
    deriv1(itp)(out, qs)                              # would MethodError pre-fix
    @test out == deriv1(itp)(qs)
    @test unit(out[1]) == u"m" / u"s"
end

@testitem "Unitful #2: Series derivative allocates in derivative units" begin
    using Unitful
    xr = range(1.0u"s", 10.0u"s", length = 9)
    y1 = (xr ./ u"s") .^ 2 .* u"m"
    y2 = 2 .* y1
    Y = hcat(y1, y2)
    q = 2.5u"s"

    @testset "$(name)" for (name, sitp) in Any[
            ("NoExtrap", linear_interp(xr, Series(Y))),
            ("Clamp", linear_interp(xr, Series(Y); extrap = ClampExtrap())),
        ]
        d = sitp(q; deriv = DerivOp(1))                  # would DimensionError pre-fix
        @test d[1] == linear_interp(xr, y1)(q; deriv = DerivOp(1))
        @test unit(d[1]) == u"m" / u"s"
        # vector query derivative
        dv = sitp([q, 3.5u"s"]; deriv = DerivOp(1))
        @test dv[1][1] == linear_interp(xr, y1)(q; deriv = DerivOp(1))
        # deriv1 view over the series (delegates to the deriv kwarg above)
        @test deriv1(sitp)(q) == d
    end

    sitp_ref = linear_interp(xr, Series(Y))
    d1 = linear_interp(xr, Series(Y), q; deriv = DerivOp(1))
    @test d1 == sitp_ref(q; deriv = DerivOp(1))
    @test unit(eltype(d1)) == u"m" / u"s"

    qs = [q, 3.5u"s"]
    d1v = linear_interp(xr, Series(Y), qs; deriv = DerivOp(1))
    @test d1v == sitp_ref(qs; deriv = DerivOp(1))
    @test unit(eltype(d1v[1])) == u"m" / u"s"

    sc = linear_interp(xr, Series(Y); extrap = ClampExtrap())
    dc = sc(20.0u"s"; deriv = DerivOp(1))
    @test unit(eltype(dc)) == u"m" / u"s"
    @test all(iszero, dc)

    cs = constant_interp(xr, Series(Y))
    cd = cs(q; deriv = DerivOp(1))
    @test unit(eltype(cd)) == u"m" / u"s"
    @test all(iszero, cd)

    cd1 = constant_interp(xr, Series(Y), q; deriv = DerivOp(1))
    @test unit(eltype(cd1)) == u"m" / u"s"
    @test all(iszero, cd1)

    cdv = constant_interp(xr, Series(Y), [q, 3.5u"s"]; deriv = DerivOp(1))
    @test unit(eltype(cdv[1])) == u"m" / u"s"
    @test all(iszero, Iterators.flatten(cdv))
end

@testitem "Unitful #3: ND GriddedQuery derivative is op-aware and unit-typed" begin
    using Unitful
    gs = range(1.0u"s", 4.0u"s", length = 4)
    gs2 = range(1.0u"s", 4.0u"s", length = 4)
    nd = [Float64(i + j) for i in 1:4, j in 1:4] .* u"W"
    tx = [1.5u"s", 2.5u"s"]
    ty = [1.5u"s", 2.5u"s"]
    gq = GriddedQuery((tx, ty))

    li = interp((gs, gs2), nd; method = LinearInterp())
    C = li(gq; deriv = DerivOp(1, 0))                    # would DimensionError pre-fix
    ref = [li((x, y); deriv = DerivOp(1, 0)) for x in tx, y in ty]
    @test all(C .== ref)
    @test unit(eltype(C)) == u"W" / u"s"                 # ∂/∂x → W/s

    C1 = linear_interp((gs, gs2), nd, gq; deriv = DerivOp(1, 0))
    @test C1 == ref
    @test unit(eltype(C1)) == u"W" / u"s"

    ci = interp((gs, gs2), nd; method = ConstantInterp())
    Cc = ci(gq; deriv = DerivOp(1, 0))
    @test unit(eltype(Cc)) == u"W" / u"s"
    @test all(iszero, Cc)

    Cc1 = constant_interp((gs, gs2), nd, gq; deriv = DerivOp(1, 0))
    @test unit(eltype(Cc1)) == u"W" / u"s"
    @test all(iszero, Cc1)
end

@testitem "Unitful derivative zeros use grid units" begin
    using Unitful

    xh = 0.0u"hr":1.0u"hr":3.0u"hr"
    y = [0.0, 1.0, 4.0, 9.0] .* u"W"
    itp = linear_interp(xh, y; extrap = ClampExtrap())

    din = itp(3600.0u"s"; deriv = DerivOp(1))
    dout = itp(18000.0u"s"; deriv = DerivOp(1))
    @test unit(din) == u"W" / u"hr"
    @test unit(dout) == u"W" / u"hr"
    @test iszero(dout)
end
