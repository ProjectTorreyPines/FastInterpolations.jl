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

    @testset "solver families on mixed-unit axes: quadratic builds (scaled store)" begin
        # Both solver persistents admit unit axes now; quadratic is the probe
        # here (3-point axis-1 — cubic's PolyFit min-points needs 4, covered by
        # the dedicated P-testitems below).
        @test interp((xs, ym), data; method = QuadraticInterp()) isa QuadraticInterpolantND
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

    # Scaled-store families: the dimensionless-twin machinery (build/eval/restore/
    # integrate/one-shot) must stay concretely inferred end to end.
    xs4 = [0.0, 1.0, 2.5, 3.0] .* u"s"
    ym4 = [0.0, 1.0, 2.0, 3.5] .* u"m"
    F4 = [Float64(i + j) for i in 1:4, j in 1:4] .* u"W"
    q4 = (1.5u"s", 0.75u"m")
    d10 = (DerivOp(1), DerivOp(0))

    @testset "solver families (scaled store): eval / deriv / integrate / one-shot" begin
        itp = cubic_interp((xs4, ym4), F4)
        @test (@inferred itp(q4)) isa TW
        @test (@inferred itp(q4; deriv = d10)) isa typeof(1.0u"W/s")
        @test (@inferred integrate(itp)) isa typeof(1.0u"W*s*m")
        @test (@inferred integrate(itp, (0.5u"s", 0.25u"m"), (2.5u"s", 3.0u"m"))) isa
            typeof(1.0u"W*s*m")
        @test (@inferred quadratic_interp((xs4, ym4), F4)(q4)) isa TW
        @test (@inferred cubic_interp((xs4, ym4), F4, q4)) isa TW
        @test (@inferred cubic_interp((xs4, ym4), F4, q4; coeffs = PreCompute())) isa TW
        @test (@inferred cubic_interp((xs4, ym4), F4, q4; deriv = d10)) isa typeof(1.0u"W/s")
        @test (@inferred cubic_interp((xs4, ym4), F4, [q4, q4])) isa Vector{TW}
    end

    @testset "FillExtrap keeps the value type concrete (in-domain + OOB + deriv)" begin
        q_oob = (9.9u"s", 0.75u"m")
        itpf = cubic_interp((xs4, ym4), F4; extrap = FillExtrap(NaN * u"W"))
        @test (@inferred itpf(q4)) isa TW
        @test (@inferred itpf(q_oob)) isa TW
        @test (@inferred itpf(q_oob; deriv = d10)) isa typeof(1.0u"W/s")
        @test (
            @inferred cubic_interp(
                (xs4, ym4), F4, q_oob;
                extrap = FillExtrap(NaN * u"W"), coeffs = PreCompute()
            )
        ) isa TW
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
        # The gate condition is `Tg <: Real` — the message must name that, with
        # units as the canonical example (not the condition itself).
        @test occursin("non-Real", sprint(showerror, err))
    end
end

@testitem "ND gates: units-free duck Number grid gets the accurate refusal" begin
    # A units-free duck Number hits the same `<: Real` gate — the error must be
    # the actionable ArgumentError naming the actual eltype, not a units claim.
    # (Testitem name must NOT contain the pinned phrase: ReTestItems derives the
    # module name from it, and qualified type names would smuggle it into `msg`.)
    struct _OrderedDuckNum <: Number
        v::Float64
    end
    Base.isless(a::_OrderedDuckNum, b::_OrderedDuckNum) = isless(a.v, b.v)
    xd = [_OrderedDuckNum(0.0), _OrderedDuckNum(1.0), _OrderedDuckNum(2.0), _OrderedDuckNum(3.0)]
    F = [Float64(i + j) for i in 1:4, j in 1:4]

    for build in (
            () -> cubic_interp((xd, xd), F),        # solver gate (cubic entry)
            () -> quadratic_interp((xd, xd), F),    # solver gate (quadratic entry)
            () -> pchip_interp((xd, xd), F),        # hetero gate
        )
        err = try
            build()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("non-Real", msg)
        @test occursin("_OrderedDuckNum", msg)
    end
end

@testitem "Unitful ND: cubic PreCompute build — scaled [Y]-homogeneous store (P1)" begin
    using Unitful
    using ForwardDiff: Dual

    # Store contract: slot p holds ∂ᵏf · Π oneunit(axisᵢ)ᵏ — every 2^N slot in
    # the value space [Y], so the single homogeneous partials array survives
    # unit grids. Axis reparameterization is exact (t = x·inv(oneunit)), so the
    # fiber solves are the SAME arithmetic as the 1D unit-native path.
    x = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
    y = [0.0, 1.0, 2.0, 3.5] .* u"m"
    F = [
        (
                sin(ustrip(u"s", xi)) + 2.0 * ustrip(u"m", yj) +
                0.4 * ustrip(u"s", xi) * ustrip(u"m", yj)
            ) * u"W"
            for xi in x, yj in y
    ]

    itp = cubic_interp((x, y), F)
    P = itp.nodal_derivs.partials
    @test eltype(P) === typeof(1.0u"W")

    @testset "slot 1 = f verbatim" begin
        @test all(P[1, i, j] === F[i, j] for i in eachindex(x), j in eachindex(y))
    end

    @testset "slot 2/3 = axis-fiber nodal derivative × oneunit(axis)" begin
        for j in eachindex(y)
            itp1 = cubic_interp(x, F[:, j])
            for i in eachindex(x)
                @test P[2, i, j] ≈ itp1(x[i]; deriv = DerivOp(1)) * oneunit(eltype(x)) rtol = 1.0e-12
            end
        end
        for i in eachindex(x)
            itp2 = cubic_interp(y, F[i, :])
            for j in eachindex(y)
                @test P[3, i, j] ≈ itp2(y[j]; deriv = DerivOp(1)) * oneunit(eltype(y)) rtol = 1.0e-12
            end
        end
    end

    @testset "slot 4 = mixed partial, [Y]-typed finite" begin
        @test all(isfinite, ustrip.(P[4, :, :]))
    end

    @testset "payload-free zero BCs mint unit-correct boundary rows" begin
        itp_z = cubic_interp((x, y), F; bc = (ZeroSlopeBC(), ZeroCurvBC()))
        Pz = itp_z.nodal_derivs.partials
        @test iszero(Pz[2, 1, 1]) && iszero(Pz[2, end, 1])   # ∂f/∂x₁ = 0 at both x-edges
        @test Pz[2, 1, 1] isa typeof(1.0u"W")
    end

    @testset "typed per-axis payload BC scales into the store" begin
        itp_p = cubic_interp((x, y), F; bc = (Deriv1(0.25u"W/s"), ZeroCurvBC()))
        Pp = itp_p.nodal_derivs.partials
        for j in eachindex(y)
            @test Pp[2, 1, j] ≈ 0.25u"W/s" * oneunit(eltype(x)) rtol = 1.0e-12
        end
    end

    @testset "Dual grids stay on the Real passthrough (no reparameterization)" begin
        xd = Dual.(ustrip.(u"s", x), 1.0)
        yd = Dual.(ustrip.(u"m", y), 0.0)
        Fd = ustrip.(u"W", F)
        itp_d = cubic_interp((xd, yd), Fd)
        Pd = itp_d.nodal_derivs.partials
        @test eltype(Pd) <: Dual
        itp_d1 = cubic_interp(xd, Fd[:, 2])
        @test Pd[2, 3, 2] ≈ itp_d1(xd[3]; deriv = DerivOp(1)) rtol = 1.0e-12
    end
end

@testitem "Unitful ND: cubic PreCompute eval/deriv/OOB (P2)" setup = [AllocConstants] begin
    using Unitful
    using Test: @inferred

    # The dimensionless solve runs the SAME arithmetic as the Float64 twin, so
    # unit results are the twin's values re-tagged: value [Y], deriv [Y/Xᵏ] via
    # the canonical `_nd_deriv_scale` restoration at the cell surface.
    x = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
    y = [0.0, 1.0, 2.0, 3.5] .* u"m"
    xf = ustrip.(u"s", x)
    yf = ustrip.(u"m", y)
    F = [(sin(xi) + 2.0 * yj + 0.4 * xi * yj) * u"W" for xi in xf, yj in yf]
    Ff = ustrip.(u"W", F)
    itp = cubic_interp((x, y), F)
    tw = cubic_interp((xf, yf), Ff)
    q = (2.2u"s", 1.3u"m")
    qf = (2.2, 1.3)

    @testset "value ≡ Float64 twin × unit" begin
        @test itp(q) ≈ tw(qf) * u"W" rtol = 1.0e-14
        @test (@inferred itp(q)) isa typeof(1.0u"W")
        @test itp((0.0u"s", 0.0u"m")) ≈ tw((0.0, 0.0)) * u"W" rtol = 1.0e-14
        @test itp((4.5u"s", 3.5u"m")) ≈ tw((4.5, 3.5)) * u"W" rtol = 1.0e-14
    end

    @testset "deriv orders restore per-axis grid⁻ᵏ units" begin
        @test itp(q; deriv = (DerivOp(1), DerivOp(0))) ≈
            tw(qf; deriv = (DerivOp(1), DerivOp(0))) * u"W/s" rtol = 1.0e-14
        @test itp(q; deriv = (DerivOp(0), DerivOp(1))) ≈
            tw(qf; deriv = (DerivOp(0), DerivOp(1))) * u"W/m" rtol = 1.0e-14
        @test itp(q; deriv = (DerivOp(1), DerivOp(1))) ≈
            tw(qf; deriv = (DerivOp(1), DerivOp(1))) * u"W/(s*m)" rtol = 1.0e-14
        @test itp(q; deriv = (DerivOp(2), DerivOp(0))) ≈
            tw(qf; deriv = (DerivOp(2), DerivOp(0))) * u"W/s^2" rtol = 1.0e-14
    end

    @testset "vector calculus unlocks (mixed: grad/hess ✓, laplacian guarded)" begin
        g = gradient(itp, q)
        gt = gradient(tw, qf)
        @test g[1] ≈ gt[1] * u"W/s" rtol = 1.0e-14
        @test g[2] ≈ gt[2] * u"W/m" rtol = 1.0e-14
        H = hessian(itp, q)
        Ht = hessian(tw, qf)
        @test H[1, 2] ≈ Ht[1, 2] * u"W/(s*m)" rtol = 1.0e-13
        @test_throws ArgumentError laplacian(itp, q)
    end

    @testset "same-unit axes: laplacian works" begin
        ys = [0.0, 1.0, 2.0, 3.5] .* u"s"
        itp_s = cubic_interp((x, ys), F)
        tw_s = cubic_interp((xf, yf), Ff)
        @test laplacian(itp_s, (2.2u"s", 1.3u"s")) ≈
            laplacian(tw_s, (2.2, 1.3)) * u"W/s^2" rtol = 1.0e-13
    end

    @testset "OOB: NoExtrap DomainError / Clamp value / Fill-NaN rule" begin
        @test_throws DomainError itp((9.9u"s", 1.0u"m"))
        itp_c = cubic_interp((x, y), F; extrap = ClampExtrap())
        tw_c = cubic_interp((xf, yf), Ff; extrap = ClampExtrap())
        @test itp_c((9.9u"s", 1.0u"m")) ≈ tw_c((9.9, 1.0)) * u"W" rtol = 1.0e-14
        itp_f = cubic_interp((x, y), F; extrap = FillExtrap(NaN * u"W"))
        @test isnan(itp_f((9.9u"s", 1.0u"m")))
        @test isnan(itp_f((9.9u"s", 1.0u"m"); deriv = (DerivOp(1), DerivOp(0))))
    end

    @testset "unit Range axes + batch SoA" begin
        xr = (0.0:1.0:4.0) .* u"s"
        yr = (0.0:1.0:3.0) .* u"m"
        Fr = [(xi + 2.0 * yj) * u"W" for xi in 0.0:1.0:4.0, yj in 0.0:1.0:3.0]
        itp_r = cubic_interp((xr, yr), Fr)
        tw_r = cubic_interp((0.0:1.0:4.0, 0.0:1.0:3.0), ustrip.(u"W", Fr))
        @test itp_r((2.2u"s", 1.3u"m")) ≈ tw_r((2.2, 1.3)) * u"W" rtol = 1.0e-14
        out = itp(([1.1, 2.2] .* u"s", [0.5, 1.5] .* u"m"))
        @test eltype(out) === typeof(1.0u"W")
        @test out[2] ≈ itp((2.2u"s", 1.5u"m")) rtol = 1.0e-14
    end

    @testset "alloc: unit scalar eval hot path" begin
        # Function barriers: the testitem-global `itp` boxes the KWARG call form
        # under `@allocated` (16 B measurement artifact, not a path allocation).
        measure_val(itp, q) = @allocated itp(q)
        measure_der(itp, q) = @allocated itp(q; deriv = (DerivOp(1), DerivOp(0)))
        measure_val(itp, q)
        measure_der(itp, q)
        @test measure_val(itp, q) <= ND_ALLOC_THRESHOLD
        @test measure_der(itp, q) <= ND_ALLOC_THRESHOLD
    end
end

@testitem "Unitful ND: quadratic PreCompute build+eval (P3 mirror)" begin
    using Unitful

    x = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
    y = [0.0, 1.0, 2.0, 3.5] .* u"m"
    xf = ustrip.(u"s", x)
    yf = ustrip.(u"m", y)
    F = [(sin(xi) + 2.0 * yj + 0.4 * xi * yj) * u"W" for xi in xf, yj in yf]
    Ff = ustrip.(u"W", F)
    itp = quadratic_interp((x, y), F)
    tw = quadratic_interp((xf, yf), Ff)
    q = (2.2u"s", 1.3u"m")
    qf = (2.2, 1.3)

    @testset "store slots in [Y]; axis-fiber oracle" begin
        P = itp.nodal_derivs.partials
        @test eltype(P) === typeof(1.0u"W")
        itp1 = quadratic_interp(x, F[:, 2])
        @test P[2, 3, 2] ≈ itp1(x[3]; deriv = DerivOp(1)) * oneunit(eltype(x)) rtol = 1.0e-12
    end

    @testset "eval/deriv ≡ twin with restored units" begin
        @test itp(q) ≈ tw(qf) * u"W" rtol = 1.0e-14
        @test itp(q; deriv = (DerivOp(1), DerivOp(0))) ≈
            tw(qf; deriv = (DerivOp(1), DerivOp(0))) * u"W/s" rtol = 1.0e-14
        @test itp(q; deriv = (DerivOp(1), DerivOp(1))) ≈
            tw(qf; deriv = (DerivOp(1), DerivOp(1))) * u"W/(s*m)" rtol = 1.0e-14
        g = gradient(itp, q)
        gt = gradient(tw, qf)
        @test g[1] ≈ gt[1] * u"W/s" rtol = 1.0e-14
        @test g[2] ≈ gt[2] * u"W/m" rtol = 1.0e-14
    end

    @testset "Left/Right payload BCs scale in" begin
        bcp = (Left(Deriv1(0.25u"W/s")), MinCurvFit())
        itp_p = quadratic_interp((x, y), F; bc = bcp)
        itp1p = quadratic_interp(x, F[:, 2]; bc = Left(Deriv1(0.25u"W/s")))
        @test itp_p.nodal_derivs.partials[2, 1, 2] ≈
            itp1p(x[1]; deriv = DerivOp(1)) * oneunit(eltype(x)) rtol = 1.0e-12
    end
end

@testitem "Unitful ND: solver-family integrate — full + bounded (P4)" begin
    using Unitful

    # The separable engine runs on the exact dimensionless twins; the volume
    # element Π oneunit(axis) restores ∫…dx from ∫…dt, so unit results are the
    # Float64 twin's values re-tagged with [Y·X₁·X₂].
    x = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
    y = [0.0, 1.0, 2.0, 3.5] .* u"m"
    xf = ustrip.(u"s", x)
    yf = ustrip.(u"m", y)
    F = [(sin(xi) + 2.0 * yj + 0.4 * xi * yj) * u"W" for xi in xf, yj in yf]
    Ff = ustrip.(u"W", F)
    lo = (0.5u"s", 0.5u"m")
    hi = (3.5u"s", 3.0u"m")
    lof = (0.5, 0.5)
    hif = (3.5, 3.0)
    TWSM = typeof(1.0u"W*s*m")

    for (name, mk) in (("cubic", cubic_interp), ("quadratic", quadratic_interp))
        @testset "$name: vec axes" begin
            itp = mk((x, y), F)
            tw = mk((xf, yf), Ff)
            @test integrate(itp) ≈ integrate(tw) * u"W*s*m" rtol = 1.0e-13
            @test integrate(itp) isa TWSM
            @test integrate(itp, lo, hi) ≈ integrate(tw, lof, hif) * u"W*s*m" rtol = 1.0e-13
        end
    end

    @testset "cubic: unit-Range axes" begin
        xr = (0.0:1.0:4.0) .* u"s"
        yr = (0.0:1.0:3.0) .* u"m"
        Fr = [(xi + 2.0 * yj) * u"W" for xi in 0.0:1.0:4.0, yj in 0.0:1.0:3.0]
        itp_r = cubic_interp((xr, yr), Fr)
        tw_r = cubic_interp((0.0:1.0:4.0, 0.0:1.0:3.0), ustrip.(u"W", Fr))
        @test integrate(itp_r) ≈ integrate(tw_r) * u"W*s*m" rtol = 1.0e-13
        @test integrate(itp_r, lo, (3.5u"s", 2.5u"m")) ≈
            integrate(tw_r, lof, (3.5, 2.5)) * u"W*s*m" rtol = 1.0e-13
    end
end

@testitem "Unitful ND: composition gaps — 3D, PolyFit{4}, Real-zero BCPair, periodic, in-place" begin
    using Unitful

    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [0.0, 1.0, 2.0, 3.5]
    x = xf .* u"s"
    y = yf .* u"m"
    F2 = [(sin(xi) + 2.0 * yj + 0.4 * xi * yj) * u"W" for xi in xf, yj in yf]
    F2f = ustrip.(u"W", F2)

    @testset "3D mixed-unit: build/eval/deriv/integrate vs twin" begin
        zf = [0.0, 0.5, 1.0, 2.0]
        z = zf .* u"kg"
        F3 = [
            (sin(xi) + 2.0 * yj + 0.3 * zk + 0.1 * xi * yj * zk) * u"W"
                for xi in xf, yj in yf, zk in zf
        ]
        F3f = ustrip.(u"W", F3)
        itp = cubic_interp((x, y, z), F3)
        tw = cubic_interp((xf, yf, zf), F3f)
        q = (2.2u"s", 1.3u"m", 0.7u"kg")
        qf = (2.2, 1.3, 0.7)
        @test itp(q) ≈ tw(qf) * u"W" rtol = 1.0e-14
        @test itp(q; deriv = (DerivOp(1), DerivOp(0), DerivOp(0))) ≈
            tw(qf; deriv = (DerivOp(1), DerivOp(0), DerivOp(0))) * u"W/s" rtol = 1.0e-14
        @test itp(q; deriv = (DerivOp(1), DerivOp(1), DerivOp(0))) ≈
            tw(qf; deriv = (DerivOp(1), DerivOp(1), DerivOp(0))) * u"W/(s*m)" rtol = 1.0e-13
        @test integrate(itp) ≈ integrate(tw) * u"W*s*m*kg" rtol = 1.0e-13
        @test integrate(itp, (0.5u"s", 0.5u"m", 0.2u"kg"), (3.5u"s", 3.0u"m", 1.5u"kg")) ≈
            integrate(tw, (0.5, 0.5, 0.2), (3.5, 3.0, 1.5)) * u"W*s*m*kg" rtol = 1.0e-13
    end

    @testset "per-axis PolyFit{4} on unit axes ≡ twin" begin
        itp = cubic_interp((x, y), F2; bc = (PolyFit{4}(), CubicFit()))
        tw = cubic_interp((xf, yf), F2f; bc = (PolyFit{4}(), CubicFit()))
        @test itp((2.2u"s", 1.3u"m")) ≈ tw((2.2, 1.3)) * u"W" rtol = 1.0e-14
        @test itp((0.2u"s", 3.3u"m")) ≈ tw((0.2, 3.3)) * u"W" rtol = 1.0e-14
    end

    @testset "Real-zero BCPair payloads beside unit ND data (rehydrate composition)" begin
        bcs = (BCPair(Deriv2(0.0), Deriv2(0.0)), ZeroCurvBC())
        itp = cubic_interp((x, y), F2; bc = bcs)
        tw = cubic_interp((xf, yf), F2f; bc = bcs)
        @test itp((2.2u"s", 1.3u"m")) ≈ tw((2.2, 1.3)) * u"W" rtol = 1.0e-14
        # structurally ≡ the payload-free zero-curvature axis
        ref = cubic_interp((x, y), F2; bc = (ZeroCurvBC(), ZeroCurvBC()))
        @test itp((2.2u"s", 1.3u"m")) === ref((2.2u"s", 1.3u"m"))
    end

    @testset "exclusive periodic unit axis composes with the twin build" begin
        xpf = [0.0, 1.0, 2.0, 3.0]
        xp = xpf .* u"s"
        Fp = [(sin(2π * xi / 4.0) + 2.0 * yj) * u"W" for xi in xpf, yj in yf]
        Fpf = ustrip.(u"W", Fp)
        itp = cubic_interp(
            (xp, y), Fp;
            bc = (PeriodicBC(endpoint = :exclusive, period = 4.0u"s"), ZeroCurvBC())
        )
        tw = cubic_interp(
            (xpf, yf), Fpf;
            bc = (PeriodicBC(endpoint = :exclusive, period = 4.0), ZeroCurvBC())
        )
        @test itp((3.7u"s", 1.3u"m")) ≈ tw((3.7, 1.3)) * u"W" rtol = 1.0e-14
        @test itp((0.3u"s", 1.3u"m")) ≈ tw((0.3, 1.3)) * u"W" rtol = 1.0e-14
    end

    @testset "gradient! into an eltype-compatible store" begin
        itp = cubic_interp((x, y), F2)
        q = (2.2u"s", 1.3u"m")
        g_ref = gradient(itp, q)
        buf = Vector{Any}(undef, 2)
        gradient!(buf, itp, q)
        @test buf[1] === g_ref[1] && buf[2] === g_ref[2]
    end
end

@testitem "Unitful ND: one-shot solver families mirror the persistent build (P5b)" begin
    using Unitful

    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [0.0, 1.0, 2.0, 3.5]
    x = xf .* u"s"
    y = yf .* u"m"
    F = [(sin(xi) + 2.0 * yj + 0.4 * xi * yj) * u"W" for xi in xf, yj in yf]
    q = (2.2u"s", 1.3u"m")
    qs = [(2.2u"s", 1.3u"m"), (0.4u"s", 3.1u"m")]
    d10 = (DerivOp(1), DerivOp(0))
    d01 = (DerivOp(0), DerivOp(1))

    # The persistent interpolant is the reference — its unit forward path is pinned above.
    ref = cubic_interp((x, y), F)

    @testset "cubic scalar one-shot (AutoCoeffs → pool + explicit PreCompute)" begin
        @test cubic_interp((x, y), F, q) ≈ ref(q) rtol = 1.0e-14
        @test cubic_interp((x, y), F, q; coeffs = PreCompute()) ≈ ref(q) rtol = 1.0e-14
        r10 = cubic_interp((x, y), F, q; deriv = d10)
        @test unit(r10) === u"W/s"
        @test r10 ≈ ref(q; deriv = d10) rtol = 1.0e-13
        r11 = cubic_interp((x, y), F, q; deriv = (DerivOp(1), DerivOp(1)))
        @test unit(r11) === u"W" / (u"s" * u"m")
        @test r11 ≈ ref(q; deriv = (DerivOp(1), DerivOp(1))) rtol = 1.0e-13
    end

    @testset "cubic batch one-shot: op-aware output eltype + in-place" begin
        out = cubic_interp((x, y), F, qs)
        @test eltype(out) === typeof(ref(q))
        @test out[1] ≈ ref(qs[1]) rtol = 1.0e-14
        @test out[2] ≈ ref(qs[2]) rtol = 1.0e-14
        outd = cubic_interp((x, y), F, qs; deriv = d10)
        @test eltype(outd) === typeof(ref(q; deriv = d10))
        @test outd[2] ≈ ref(qs[2]; deriv = d10) rtol = 1.0e-13
        buf = Vector{typeof(ref(q))}(undef, 2)
        cubic_interp!(buf, (x, y), F, qs)
        @test buf[2] ≈ ref(qs[2]) rtol = 1.0e-14
    end

    @testset "quadratic scalar + batch one-shot" begin
        refq = quadratic_interp((x, y), F)
        @test quadratic_interp((x, y), F, q) ≈ refq(q) rtol = 1.0e-14
        @test quadratic_interp((x, y), F, q; coeffs = PreCompute()) ≈ refq(q) rtol = 1.0e-14
        rq = quadratic_interp((x, y), F, q; deriv = d01)
        @test unit(rq) === u"W/m"
        @test rq ≈ refq(q; deriv = d01) rtol = 1.0e-13
        outq = quadratic_interp((x, y), F, qs)
        @test eltype(outq) === typeof(refq(q))
        @test outq[1] ≈ refq(qs[1]) rtol = 1.0e-14
        outqd = quadratic_interp((x, y), F, qs; deriv = d01)
        @test eltype(outqd) === typeof(refq(q; deriv = d01))
        @test outqd[2] ≈ refq(qs[2]; deriv = d01) rtol = 1.0e-13
    end

    @testset "same-unit axes (concrete Tg) + unit Range axis" begin
        ys = yf .* u"s"
        Fs = [(xi + 2.0 * yj) * u"W" for xi in xf, yj in yf]
        qsame = (2.2u"s", 1.3u"s")
        @test cubic_interp((x, ys), Fs, qsame) ≈ cubic_interp((x, ys), Fs)(qsame) rtol = 1.0e-14
        @test quadratic_interp((x, ys), Fs, qsame) ≈
            quadratic_interp((x, ys), Fs)(qsame) rtol = 1.0e-14
        xr = (0.0:1.0:4.0) * u"s"   # unit StepRangeLen → pooled _CachedRange arm
        Fr = [(xi + 2.0 * yj) * u"W" for xi in 0.0:1.0:4.0, yj in yf]
        @test cubic_interp((xr, y), Fr, q) ≈ cubic_interp((xr, y), Fr)(q) rtol = 1.0e-14
    end

    @testset "exclusive periodic unit axis through the pooled one-shot" begin
        xpf = [0.0, 1.0, 2.0, 3.0]
        xp = xpf .* u"s"
        Fp = [(sin(2π * xi / 4.0) + 2.0 * yj) * u"W" for xi in xpf, yj in yf]
        bcs = (PeriodicBC(endpoint = :exclusive, period = 4.0u"s"), ZeroCurvBC())
        refp = cubic_interp((xp, y), Fp; bc = bcs)
        @test cubic_interp((xp, y), Fp, (3.7u"s", 1.3u"m"); bc = bcs, coeffs = PreCompute()) ≈
            refp((3.7u"s", 1.3u"m")) rtol = 1.0e-14
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

@testitem "Unitful ND: zero-alloc hot path, same-unit concrete-Tg (review pin F21)" setup = [AllocConstants] begin
    using Unitful

    # 5a same-unit axes route through the concrete-Tg "common-Tg machinery" — a distinct
    # path from F6's abstract per-axis one, so its alloc contract needs its own pin. Both
    # unit-supporting ND families (Linear/Constant) must stay 0-alloc.
    xs = [0.0, 1.0, 2.0, 3.0] .* u"s"
    ys = [0.0, 0.5, 1.0, 1.5, 2.0] .* u"s"   # same unit → concrete Tg
    data = [Float64(i + j) for i in 1:4, j in 1:5] .* u"W"
    q = (1.5u"s", 0.75u"s")

    for (nm, method) in (("Linear", LinearInterp()), ("Constant", ConstantInterp()))
        @testset "$nm: eval / integrate" begin
            itp = interp((xs, ys), data; method = method)
            itp(q)
            integrate(itp)   # warmup
            @test (@allocated itp(q)) <= ND_ALLOC_THRESHOLD
            @test (@allocated integrate(itp)) <= ND_ALLOC_THRESHOLD
        end
    end
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

@testitem "Unitful ND: unit LinRange axes on the _CachedRange path" setup = [AllocConstants] begin
    using Unitful
    FI = FastInterpolations

    # F13 covers ND StepRange axes; a `LinRange` of Quantities must wrap the same way
    # (`_CachedRange` per axis) and stay value-correct + zero-alloc on the hot path.
    gu1 = LinRange(0.0u"s", 3.0u"s", 4)
    gu2 = LinRange(0.0u"m", 2.0u"m", 5)
    gf1 = LinRange(0.0, 3.0, 4)
    gf2 = LinRange(0.0, 2.0, 5)
    data = [Float64(i * j) for i in 1:4, j in 1:5] .* u"W"
    dataf = [Float64(i * j) for i in 1:4, j in 1:5]

    itp = interp((gu1, gu2), data; method = LinearInterp())
    tw = interp((gf1, gf2), dataf; method = LinearInterp())

    @test itp.grids[1] isa FI._CachedRange
    @test itp.grids[2] isa FI._CachedRange
    @test itp((1.5u"s", 0.75u"m")) ≈ tw((1.5, 0.75)) * u"W"
    @test integrate(itp) ≈ integrate(tw) * u"W*s*m"

    q = (1.5u"s", 0.75u"m")
    itp(q)
    integrate(itp)   # warmup
    @test (@allocated itp(q)) <= ND_ALLOC_THRESHOLD
    @test (@allocated integrate(itp)) <= ND_ALLOC_THRESHOLD
end

@testitem "Unitful ND: one-shot builders reject unit grids with a friendly error (review pin F17)" begin
    using Unitful

    # Hetero ND one-shots (pchip/akima/cardinal) gate unit-carrying grids with an
    # actionable ArgumentError (`_check_nd_hetero_grid`) instead of deep, non-actionable
    # errors (`TypeError: Quantity … is not a valid key`). Cubic/quadratic one-shots
    # reparameterize and are pinned in the P5b mirror testitem above.
    xs = collect(1.0:5.0) .* u"s"
    ys = collect(1.0:5.0) .* u"m"
    data = [Float64(i + j) for i in 1:5, j in 1:5] .* u"W"
    qsc = (2.5u"s", 2.5u"m")
    qb = [(2.5u"s", 2.5u"m"), (3.5u"s", 3.5u"m")]

    # (name, scalar one-shot, batch one-shot)
    fams = [
        ("pchip", () -> pchip_interp((xs, ys), data, qsc), () -> pchip_interp((xs, ys), data, qb)),
        ("akima", () -> akima_interp((xs, ys), data, qsc), () -> akima_interp((xs, ys), data, qb)),
        ("cardinal", () -> cardinal_interp((xs, ys), data, qsc), () -> cardinal_interp((xs, ys), data, qb)),
    ]
    for (nm, fsc, fb) in fams
        @testset "$nm one-shot: scalar + batch → ArgumentError" begin
            @test_throws ArgumentError fsc()
            @test_throws ArgumentError fb()
        end
    end
end

@testitem "Unitful ND: solver/hetero guards fire on same-unit (concrete-Tg) axes (review pin F20)" begin
    using Unitful

    # Existing guard-throw coverage uses *mixed-unit* axes (u"s", u"m") → an abstract
    # promoted grid eltype. Same-unit axes promote to a *concrete* `Quantity{Float64,𝐓,…}`
    # Tg — a distinct dispatch. Pin that this concrete-Tg still reaches the guard (a future
    # `_promote_grid_eltype` narrowing must not leak it to the Real fast path), for both the
    # persistent builders and the one-shot entries.
    xs = collect(1.0:5.0) .* u"s"
    xs2 = collect(0.5:0.5:2.5) .* u"s"     # same unit → concrete Tg
    data = [Float64(i + j) for i in 1:5, j in 1:5] .* u"W"
    q = (2.5u"s", 1.5u"s")

    @testset "solver families (Cubic/Quadratic): persistent + one-shot" begin
        # Solver persistents admit unit axes since the scaled-store build; the
        # concrete-Tg dispatch pins flip to build-success, and the one-shot
        # mirrors (P5b) now agree with the persistent reference.
        @test cubic_interp((xs, xs2), data) isa CubicInterpolantND
        @test quadratic_interp((xs, xs2), data) isa QuadraticInterpolantND
        @test cubic_interp((xs, xs2), data, q) ≈ cubic_interp((xs, xs2), data)(q) rtol = 1.0e-14
        @test quadratic_interp((xs, xs2), data, q) ≈
            quadratic_interp((xs, xs2), data)(q) rtol = 1.0e-14
    end

    @testset "hetero families (PCHIP): persistent + one-shot" begin
        @test_throws ArgumentError pchip_interp((xs, xs2), data)
        @test_throws ArgumentError pchip_interp((xs, xs2), data, q)
    end
end

@testitem "Unitful ND: FillExtrap OOB derivative units (review pin P1-fill)" begin
    using Unitful

    # FillExtrap OOB short-circuits derive the result from the value-space fill (`W`) and a
    # non-derivative-aware zero, ignoring the requested derivative orders. On unit grids the
    # scalar returns the wrong (value) units and the allocating batch throws DimensionError
    # (a `W/s` buffer can't store the `W` zero). Fold the axis units per derivative order.
    xs = [0.0, 1.0, 2.0, 3.0] .* u"s"
    ys = [0.0, 0.5, 1.0, 1.5, 2.0] .* u"s"   # same unit → well-defined gradient/Hessian/Laplacian
    data = [Float64(i + j) for i in 1:4, j in 1:5] .* u"W"
    itp = interp((xs, ys), data; method = LinearInterp(), extrap = FillExtrap(0.0u"W"))
    q_oob = (5.0u"s", 0.75u"s")   # axis-1 OOB → fill

    @testset "eval OOB deriv carries grid⁻ⁿ units" begin
        @test unit(itp(q_oob; deriv = (DerivOp(1), DerivOp(0)))) === unit(1.0u"W" / 1.0u"s")
        @test unit(itp(q_oob; deriv = (DerivOp(1), DerivOp(1)))) === unit(1.0u"W" / 1.0u"s"^2)
        @test iszero(ustrip(itp(q_oob; deriv = (DerivOp(1), DerivOp(0)))))
        @test eltype(itp([q_oob, q_oob]; deriv = (DerivOp(1), DerivOp(0)))) === typeof(1.0u"W/s")
    end

    @testset "vector-calculus OOB zeros are derivative-scaled" begin
        g = gradient(itp, q_oob)
        @test all(c -> unit(c) === unit(1.0u"W" / 1.0u"s"), g)
        G = [1.0u"W/s", 1.0u"W/s"]
        gradient!(G, itp, q_oob)
        @test all(c -> unit(c) === unit(1.0u"W" / 1.0u"s"), G)
        H = hessian(itp, q_oob)
        @test all(c -> unit(c) === unit(1.0u"W" / 1.0u"s"^2), H)
        @test unit(laplacian(itp, q_oob)) === unit(1.0u"W" / 1.0u"s"^2)
    end
end

@testitem "Unitful ND: mixed-unit exclusive-periodic axes (review pin P1-periodic-nd)" begin
    using Unitful
    using Test: @inferred

    # `_prepare_periodic_nd_impl` promoted all axes to one `float(_promote_grid_eltype)` type;
    # mixed units (s, m) collapse to an abstract `Quantity{Float64}` → `zero(Quantity{Float64})`
    # threw during the exclusive-periodic extension. Float each axis independently.
    xs = (0.0:1.0:2.0) .* u"s"
    ym = (0.0:1.0:2.0) .* u"m"
    xs2 = (0.0:1.0:2.0) .* u"s"
    data = [Float64(i + j) for i in 1:3, j in 1:3] .* u"W"
    dataf = [Float64(i + j) for i in 1:3, j in 1:3]
    bcx = (PeriodicBC(endpoint = :exclusive), PeriodicBC(endpoint = :exclusive))

    @testset "mixed-unit builds + evals" begin
        itp = linear_interp((xs, ym), data; bc = bcx)
        @test itp((0.5u"s", 0.5u"m")) isa typeof(1.0u"W")
    end
    @testset "same-unit: value parity + inference (no per-axis boxing regression)" begin
        iu = linear_interp((xs, xs2), data; bc = bcx)
        ir = linear_interp((0.0:1.0:2.0, 0.0:1.0:2.0), dataf; bc = bcx)
        @test iu((0.5u"s", 0.5u"s")) ≈ ir((0.5, 0.5)) * u"W"
        @test (@inferred iu((0.5u"s", 0.5u"s"))) isa typeof(1.0u"W")   # concrete grid ⇒ no box
    end
end

@testitem "Unitful ND: Real BC payloads validate in their true derivative space" begin
    using Unitful

    # Real data on unit axes: a nonzero Real `Deriv1` payload is dimensionally
    # incomplete ([Y/X] carries axis units) — the 1D builders reject it with an
    # actionable error, and the reparam scale-in must match. The structural zero
    # keeps minting in the true space before scaling into the [Y] store.
    xf = [0.0, 1.0, 2.5, 3.0, 4.5]
    yf = [0.0, 1.0, 2.0, 3.5]
    xs = xf .* u"s"
    ym = yf .* u"m"
    Freal = [(sin(xi) + 2.0 * yj) for xi in xf, yj in yf]
    q = (2.2u"s", 1.3u"m")

    @testset "nonzero Real payload beside Real data + unit axes → ArgumentError" begin
        err = try
            cubic_interp((xs, ym), Freal; bc = (BCPair(Deriv1(0.25), Deriv1(0.25)), ZeroCurvBC()))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("derivative space", sprint(showerror, err))
    end

    @testset "structural zero still mints (Real data + unit axes)" begin
        ref = cubic_interp((xs, ym), Freal; bc = (ZeroCurvBC(), ZeroCurvBC()))
        itp = cubic_interp((xs, ym), Freal; bc = (BCPair(Deriv2(0.0), Deriv2(0.0)), ZeroCurvBC()))
        @test itp(q) === ref(q)
    end

    @testset "typed payloads in the true [Y/X] space stay accepted" begin
        itp = cubic_interp(
            (xs, ym), Freal;
            bc = (BCPair(Deriv1(0.25u"s^-1"), Deriv1(0.25u"s^-1")), ZeroCurvBC())
        )
        @test itp(q) isa Float64
    end
end
