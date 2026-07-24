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
# Adjoints / mixed-unit hessian / solver-family series build are design-excluded
# (kept <:Real) — their friendly-error contract is pinned separately.

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
    end
    # Quadratic series under units is a KNOWN LIMITATION (documented, not fixed):
    # its per-column coeff solve (`_compute_quadratic_coeffs`) allocates slope `d`
    # (Y/X) and curvature `a` (Y/X²) through one shared buffer — the same solver-
    # storage class as cubic-series build. The persistent SCALAR build escapes via
    # `_quadratic_interp_units` (nondimensionalized), but that is not wired to the
    # series/one-shot path. Pinned so a future fix flips this RED.
    @testset "quadratic series — documented solver-storage limitation" begin
        @test_throws Unitful.DimensionError quadratic_interp(xr, Series(Y))(q)
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
    # KNOWN LIMITATION (documented, not fixed): the OOB/stateful derivative path
    # emits a *value*-unit zero (m) rather than the derivative-scaled zero (m/s) —
    # the OOB helper doesn't carry grid-unit × order. In-domain deriv is correct.
    sc = linear_interp(xr, Series(Y); extrap = ClampExtrap())
    @test_broken unit(sc(20.0u"s"; deriv = DerivOp(1))[1]) == u"m" / u"s"
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

    # KNOWN LIMITATION (documented, not fixed): Constant's derivative is a zero,
    # but the gridded zero kernel emits it in value units (W) not derivative units
    # (W/s) — the zero payload lacks grid-unit × order. Linear (above) is correct.
    ci = interp((gs, gs2), nd; method = ConstantInterp())
    Cc = ci(gq; deriv = DerivOp(1, 0))
    @test_broken unit(eltype(Cc)) == u"W" / u"s"
end
