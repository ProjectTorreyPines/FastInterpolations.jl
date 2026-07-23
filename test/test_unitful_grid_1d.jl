# ========================================
# Unitful grid axis: 1D families (ducktype-grid phases 2-4)
# ========================================
# Phase 2: Linear + Constant full pipeline (eval, integrate, cumulative,
#          bounded, one-shot) on Quantity grids — unit-asserted outputs.
# Phases 3-4 extend this file per family (cubic/quadratic, then local-slope).

@testitem "Unitful 1D: Linear full pipeline" begin
    using Unitful

    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"

    # Real canary: the Float64 twin must stay bit-identical to the phase-0 pin.
    xf = [0.0, 1.0, 2.0, 3.0, 4.0]
    yf = [0.0, 1.0, 0.5, 2.0, 1.0]
    @test integrate(linear_interp(xf, yf)) === 4.0

    itp = linear_interp(xu, yw)

    @testset "eval: value units, dimensionless weights" begin
        v = itp(1.5u"s")
        @test v === 0.75u"W"
        @test itp(0.0u"s") === 0.0u"W"          # exact node hit
        @test itp(4.0u"s") === 1.0u"W"          # right endpoint
    end

    @testset "integrate: Tg·Tv units" begin
        full = integrate(itp)
        @test full === 4.0u"W*s"
        part = integrate(itp, 0.5u"s", 2.5u"s")
        @test part isa typeof(1.0u"W*s")
        @test part ≈ integrate(linear_interp(xf, yf), 0.5, 2.5) * u"W*s"
    end

    @testset "cumulative_integrate: vector of Tg·Tv" begin
        cum = cumulative_integrate(itp)
        @test eltype(cum) === typeof(1.0u"W*s")
        @test cum[end] === integrate(itp)
        @test cum[1] === 0.0u"W*s"
    end

    @testset "one-shot forms" begin
        @test integrate(xu, yw; method = LinearInterp()) === 4.0u"W*s"
        cum = cumulative_integrate(xu, yw; method = LinearInterp())
        @test eltype(cum) === typeof(1.0u"W*s")
        @test cum[end] === 4.0u"W*s"
    end

    @testset "range grid" begin
        ru = (0.0:1.0:4.0) .* u"s"
        itr = linear_interp(ru, yw)
        @test itr(1.5u"s") === 0.75u"W"
        @test integrate(itr) === 4.0u"W*s"
    end

    @testset "review pins: one-shot 3-arg scalar, deriv eval" begin
        # F1: one-shot 3-arg scalar (was Tq<:Real — MethodError for Quantity)
        @test linear_interp(xu, yw, 1.5u"s") === 0.75u"W"
        # F3: EvalDeriv1 kernel value-space diff (was coeff-space convert)
        d1 = itp(1.5u"s"; deriv = DerivOp(1))
        @test d1 === -0.5u"W/s"
    end

    @testset "extrapolation modes" begin
        @test_throws DomainError linear_interp(xu, yw)(-1.0u"s")             # NoExtrap OOB
        @test linear_interp(xu, yw; extrap = ClampExtrap())(-1.0u"s") === 0.0u"W"
        @test linear_interp(xu, yw; extrap = ClampExtrap())(9.0u"s") === 1.0u"W"
    end
end

@testitem "Unitful 1D: Constant full pipeline" begin
    using Unitful

    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"

    itp = constant_interp(xu, yw)

    @testset "eval" begin
        # Nearest-node oracle, not just the type (review: type-only assert).
        @test itp(1.4u"s") === 1.0u"W"    # nearest node x=1s → y[2]
        @test itp(1.6u"s") === 0.5u"W"    # nearest node x=2s → y[3]
    end

    @testset "StorePolicy(copy=false): unit grid/data alias round-trip" begin
        itp_ref = constant_interp(xu, yw; store = StorePolicy(copy = false))
        @test itp_ref.y === yw                      # data aliased, not copied
        @test itp_ref(1.4u"s") === itp(1.4u"s")     # same results as owning build
    end

    @testset "integrate / cumulative" begin
        full = integrate(itp)
        @test full isa typeof(1.0u"W*s")
        # Constant (nearest) oracle via the Float64 twin
        xf = [0.0, 1.0, 2.0, 3.0, 4.0]
        yf = [0.0, 1.0, 0.5, 2.0, 1.0]
        @test full ≈ integrate(constant_interp(xf, yf)) * u"W*s"
        cum = cumulative_integrate(itp)
        @test eltype(cum) === typeof(1.0u"W*s")
        @test cum[end] === full
    end

    @testset "one-shot" begin
        v = integrate(xu, yw; method = ConstantInterp())
        @test v isa typeof(1.0u"W*s")
    end

    @testset "bounded integrate (review pin)" begin
        # F2: impl helper kept x0::Real — MethodError for Quantity bounds
        xf = [0.0, 1.0, 2.0, 3.0, 4.0]
        yf = [0.0, 1.0, 0.5, 2.0, 1.0]
        @test integrate(itp, 0.5u"s", 2.5u"s") ≈
            integrate(constant_interp(xf, yf), 0.5, 2.5) * u"W*s"
    end
end

@testitem "Unitful 1D: dispatch-selection pins (@which)" begin
    using Unitful
    using InteractiveUtils: @which

    # Guards the scalar-vs-batch method split under the unbounded query params:
    # a Quantity scalar must take the scalar arm, a Vector{Quantity} the
    # AbstractVector batch arm (capture here would be invisible to Aqua).
    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"
    itp = linear_interp(xu, yw)

    m_scalar = @which itp(1.5u"s")
    m_batch = @which itp([1.5, 2.5] .* u"s")
    @test m_scalar !== m_batch
    batch_qtype = Base.unwrap_unionall(m_batch.sig).parameters[end]
    scalar_qtype = Base.unwrap_unionall(m_scalar.sig).parameters[end]
    @test batch_qtype <: AbstractVector
    @test !(scalar_qtype <: AbstractArray)
    # batch actually works end-to-end
    out = itp([1.5, 2.5] .* u"s")
    @test out == [0.75, 1.25] .* u"W"
    @test eltype(out) === typeof(1.0u"W")
end

# ========================================
# Phase 3 — Cubic + Quadratic (solver families)
# ========================================

@testitem "Unitful 1D: Cubic full pipeline" begin
    using Unitful

    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"
    xf = [0.0, 1.0, 2.0, 3.0, 4.0]
    yf = [0.0, 1.0, 0.5, 2.0, 1.0]

    itp = cubic_interp(xu, yw)
    tw = cubic_interp(xf, yf)   # Float64 twin (oracle)

    @testset "build: z carries Y/X² units" begin
        @test eltype(itp.z) === typeof(1.0u"W/s^2")
    end

    @testset "eval" begin
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
        @test itp(0.0u"s") ≈ tw(0.0) * u"W" atol = 1.0e-12u"W"
        d1 = itp(1.5u"s"; deriv = DerivOp(1))
        @test d1 ≈ tw(1.5; deriv = DerivOp(1)) * u"W/s"
    end

    @testset "integrate / cumulative / bounded / one-shot" begin
        @test integrate(itp) ≈ integrate(tw) * u"W*s"
        @test integrate(itp, 0.5u"s", 2.5u"s") ≈ integrate(tw, 0.5, 2.5) * u"W*s"
        cum = cumulative_integrate(itp)
        @test eltype(cum) === typeof(1.0u"W*s")
        @test cum[end] ≈ integrate(itp)
        @test integrate(xu, yw; method = CubicInterp()) ≈ integrate(tw) * u"W*s"
    end

    @testset "range grid" begin
        ru = (0.0:1.0:4.0) .* u"s"
        @test cubic_interp(ru, yw)(1.5u"s") ≈ tw(1.5) * u"W"
    end
end

@testitem "Unitful 1D: Quadratic full pipeline" begin
    using Unitful

    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"
    xf = [0.0, 1.0, 2.0, 3.0, 4.0]
    yf = [0.0, 1.0, 0.5, 2.0, 1.0]

    itp = quadratic_interp(xu, yw)
    tw = quadratic_interp(xf, yf)

    @testset "build: a=Y/X², d=Y/X coefficient types" begin
        @test eltype(itp.d) === typeof(1.0u"W/s")
        @test eltype(itp.a) === typeof(1.0u"W/s^2")
    end

    @testset "eval / integrate" begin
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
        @test integrate(itp) ≈ integrate(tw) * u"W*s"
        @test integrate(itp, 0.5u"s", 2.5u"s") ≈ integrate(tw, 0.5, 2.5) * u"W*s"
    end
end

@testitem "Unitful 1D: Quadratic BC unit-strip (MinCurvFit + side selectors, review pin F18)" begin
    using Unitful

    # `_quadratic_interp_units` strips units, solves the dimensionless twin, then
    # reattaches per-order units. `MinCurvFit` — a payload-free marker BC valid on
    # Real grids — hit the throwing `_strip_bc_units` catch-all because it lacked the
    # identity strip the other marker BCs get. Side selectors `Left/Right(QuadraticFit())`
    # were also unexercised on unit grids.
    xu = collect(1.0:6.0) .* u"s"
    yw = [1.0, 2.0, 4.0, 7.0, 5.0, 3.0] .* u"W"
    xf = collect(1.0:6.0)
    yf = [1.0, 2.0, 4.0, 7.0, 5.0, 3.0]
    TW = typeof(1.0u"W")

    @testset "MinCurvFit: unit grid ≡ Real twin" begin
        itp = quadratic_interp(xu, yw; bc = MinCurvFit())
        tw = quadratic_interp(xf, yf; bc = MinCurvFit())
        @test itp(2.5u"s") isa TW
        @test itp(2.5u"s") ≈ tw(2.5) * u"W"
        @test eltype(itp.a) === typeof(1.0u"W/s^2")
        @test eltype(itp.d) === typeof(1.0u"W/s")
    end

    @testset "side selectors: Left/Right(QuadraticFit()) unit-strip" begin
        for bc in (Left(QuadraticFit()), Right(QuadraticFit()))
            itp = quadratic_interp(xu, yw; bc = bc)
            tw = quadratic_interp(xf, yf; bc = bc)
            @test itp(2.5u"s") ≈ tw(2.5) * u"W"
        end
    end
end

# ========================================
# Phase 4 — local-slope families (PCHIP/Akima/Cardinal/Hermite)
# ========================================

@testitem "Unitful 1D: local-slope families" begin
    using Unitful

    xu = [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"
    yw = [0.0, 1.0, 0.5, 2.0, 1.0] .* u"W"
    xf = [0.0, 1.0, 2.0, 3.0, 4.0]
    yf = [0.0, 1.0, 0.5, 2.0, 1.0]

    @testset "PCHIP" begin
        itp = pchip_interp(xu, yw)
        tw = pchip_interp(xf, yf)
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
        @test integrate(itp) ≈ integrate(tw) * u"W*s"
    end

    @testset "Akima" begin
        itp = akima_interp(xu, yw)
        tw = akima_interp(xf, yf)
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
        @test integrate(itp) ≈ integrate(tw) * u"W*s"
    end

    @testset "Cardinal" begin
        itp = cardinal_interp(xu, yw)
        tw = cardinal_interp(xf, yf)
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
        @test integrate(itp) ≈ integrate(tw) * u"W*s"
    end

    @testset "Hermite (explicit slopes, Y/X units)" begin
        dyu = [1.0, 0.5, 0.0, -0.5, -1.0] .* u"W/s"
        dyf = [1.0, 0.5, 0.0, -0.5, -1.0]
        itp = hermite_interp(xu, yw, dyu)
        tw = hermite_interp(xf, yf, dyf)
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
        @test integrate(itp) ≈ integrate(tw) * u"W*s"
    end
end

@testitem "Unitful 1D: Cardinal persistent tension stays dimensionless (review pin F19)" begin
    using Unitful

    # `tension` is a dimensionless shape parameter (field type `Tt`), independent of the
    # grid's units. Default-tension (0.0) unit tests can't catch a `Tt → Tg` regression —
    # a unit-carrying zero still compares equal. Pin a *nonzero* tension on a unit grid
    # for both coefficient strategies (PreCompute and OnTheFly).
    xu = collect(1.0:6.0) .* u"s"
    yw = [1.0, 2.0, 4.0, 7.0, 5.0, 3.0] .* u"W"
    xf = collect(1.0:6.0)
    yf = [1.0, 2.0, 4.0, 7.0, 5.0, 3.0]

    for (label, coeffs) in (("PreCompute", PreCompute()), ("OnTheFly", OnTheFly()))
        @testset "tension=0.3 [$label]" begin
            itp = cardinal_interp(xu, yw; tension = 0.3, coeffs = coeffs)
            tw = cardinal_interp(xf, yf; tension = 0.3, coeffs = coeffs)
            @test itp.tension isa Float64          # dimensionless, not a Quantity
            @test itp.tension == 0.3
            @test itp(2.5u"s") ≈ tw(2.5) * u"W"
        end
    end
end

@testitem "Unitful 1D: zero-alloc hot path (review pin F6)" setup = [AllocConstants] begin
    using Unitful

    # The point of the relaxation is that the 0-alloc contract SURVIVES the
    # abstraction — pin it for unit grids exactly as the Real files do.
    xs = [0.0, 1.0, 2.5, 3.0] .* u"s"
    yw = [1.0, 2.0, 4.0, 8.0] .* u"W"
    q = 1.5u"s"

    for (nm, itp) in (("linear", linear_interp(xs, yw)), ("cubic", cubic_interp(xs, yw)))
        @testset "$nm: eval / integrate" begin
            itp(q)
            integrate(itp)   # warmup
            @test (@allocated itp(q)) <= ALLOC_THRESHOLD
            @test (@allocated integrate(itp)) <= ALLOC_THRESHOLD
        end
    end
end

@testitem "Unitful 1D: cubic PeriodicBC rejection is friendly (review pin F7)" begin
    using Unitful

    # `_cubic_interp_units` deliberately rejects PeriodicBC (strip→solve→reattach
    # has no periodic factorization path yet) — pin the ERROR QUALITY, not just
    # that it throws: message must name the feature and the workaround.
    xs = [0.0, 1.0, 2.5, 3.0] .* u"s"
    yw = [1.0, 2.0, 4.0, 8.0] .* u"W"
    err = try
        cubic_interp(xs, yw; bc = PeriodicBC())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("PeriodicBC", msg)
    @test occursin("unit-carrying", msg)
    @test occursin("ustrip", msg)   # actionable workaround named

    @testset "unhandled BC type: catch-all is actionable, not a solve MethodError" begin
        struct _F7UnknownBC <: FastInterpolations.AbstractBC end
        err2 = try
            FastInterpolations._strip_bc_units(_F7UnknownBC(), 1.0u"W", 1.0u"s")
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("ustrip", sprint(showerror, err2))
    end
end

@testitem "Unitful 1D: batch eval across families (review pin F8)" begin
    using Unitful

    # Batch (Vector-of-Quantity) eval was only exercised for Linear: the
    # persistent vector loops of Cubic/Quadratic/hermite-family kept
    # `AbstractArray{<:Real}` query bounds → MethodError on unit queries.
    # Default AutoSearch exercises batch search-policy resolution too.
    xs = [0.0, 1.0, 2.5, 3.0, 4.0] .* u"s"
    xf = [0.0, 1.0, 2.5, 3.0, 4.0]
    yw = [1.0, 2.0, 4.0, 8.0, 5.0] .* u"W"
    yf = [1.0, 2.0, 4.0, 8.0, 5.0]
    qf = [0.25, 0.75, 1.25, 2.6, 3.9]
    qs = qf .* u"s"

    dyf = [1.0, 0.5, 0.0, -0.5, -1.0]
    dyu = dyf .* u"W/s"

    fams = [
        ("linear", linear_interp(xs, yw), linear_interp(xf, yf)),
        ("cubic", cubic_interp(xs, yw), cubic_interp(xf, yf)),
        ("quadratic", quadratic_interp(xs, yw), quadratic_interp(xf, yf)),
        ("pchip", pchip_interp(xs, yw), pchip_interp(xf, yf)),
        ("akima", akima_interp(xs, yw), akima_interp(xf, yf)),
        ("cardinal", cardinal_interp(xs, yw), cardinal_interp(xf, yf)),
        ("hermite", hermite_interp(xs, yw, dyu), hermite_interp(xf, yf, dyf)),
    ]
    for (nm, itp, tw) in fams
        @testset "$nm: allocating + in-place batch" begin
            out = itp(qs)
            @test eltype(out) === typeof(1.0u"W")
            @test out ≈ tw(qf) .* u"W"
            buf = similar(out)
            itp(buf, qs)
            @test buf ≈ out
        end
    end
end

@testitem "Unitful 1D: Fill/Wrap extrapolation (review pin F9)" begin
    using Unitful

    # Extrap coverage was NoExtrap+Clamp only; the relaxation touched
    # Fill/Wrap bounded signatures broadly — exercise both OOB on unit grids.
    xs = [0.0, 1.0, 2.5, 3.0] .* u"s"
    xf = [0.0, 1.0, 2.5, 3.0]
    yw = [1.0, 2.0, 4.0, 8.0] .* u"W"
    yf = [1.0, 2.0, 4.0, 8.0]

    @testset "FillExtrap: value-typed fill" begin
        itp = linear_interp(xs, yw; extrap = FillExtrap(0.0u"W"))
        @test itp(5.0u"s") === 0.0u"W"
        @test itp(1.5u"s") ≈ linear_interp(xf, yf)(1.5) * u"W"   # in-domain untouched
    end

    @testset "FillExtrap: dimensionless fill vs unit values errors loudly" begin
        # The mismatch surfaces at BUILD (fill value promoted against Tv) —
        # earlier than eval, which is the better failure point.
        @test_throws Unitful.DimensionError linear_interp(xs, yw; extrap = FillExtrap(0.0))
    end

    @testset "WrapExtrap: periodic OOB matches Float64 twin" begin
        itp = linear_interp(xs, yw; extrap = WrapExtrap())
        tw = linear_interp(xf, yf; extrap = WrapExtrap())
        @test itp(4.5u"s") ≈ tw(4.5) * u"W"
        @test itp(-0.5u"s") ≈ tw(-0.5) * u"W"
    end
end

@testitem "Unitful 1D: Codex review batch (review pin F12)" begin
    using Unitful

    xu = [0.0, 1.0, 2.5, 3.0, 4.0] .* u"s"
    xf = [0.0, 1.0, 2.5, 3.0, 4.0]
    yu = [1.0, 2.0, 4.0, 8.0, 5.0] .* u"W"
    yf = [1.0, 2.0, 4.0, 8.0, 5.0]
    qs = [0.5, 1.5] .* u"s"

    @testset "C1: batch deriv buffer carries derivative units" begin
        itp = linear_interp(xu, yu)
        tw = linear_interp(xf, yf)
        out = itp(qs; deriv = DerivOp(1))
        @test eltype(out) === typeof(1.0u"W/s")
        @test out ≈ tw([0.5, 1.5]; deriv = DerivOp(1)) .* u"W/s"
        # shaped-array form shares the fix
        out2 = itp(reshape(qs, 1, 2); deriv = DerivOp(1))
        @test eltype(out2) === typeof(1.0u"W/s")
    end

    @testset "C2: solver one-shots route through the units path" begin
        for (f, tf) in ((cubic_interp, cubic_interp), (quadratic_interp, quadratic_interp))
            @test f(xu, yu, 1.5u"s") ≈ tf(xf, yf, 1.5) * u"W"
            @test f(xu, yu, qs) ≈ tf(xf, yf, [0.5, 1.5]) .* u"W"
        end
    end

    @testset "C3: AutoCoeffs scalar one-shot (local-slope families)" begin
        @test pchip_interp(xu, yu, 1.5u"s") ≈ pchip_interp(xf, yf, 1.5) * u"W"
        @test akima_interp(xu, yu, 1.5u"s") ≈ akima_interp(xf, yf, 1.5) * u"W"
    end

    @testset "C4: cardinal one-shot tension stays dimensionless" begin
        for coeffs in (OnTheFly(), PreCompute(), AutoCoeffs())
            @test cardinal_interp(xu, yu, 1.5u"s"; coeffs) ≈
                cardinal_interp(xf, yf, 1.5) * u"W"
        end
    end

    @testset "C5: bounded one-shot integrate facade" begin
        @test integrate(xu, yu, 0.5u"s", 2.5u"s"; method = LinearInterp()) ≈
            integrate(xf, yf, 0.5, 2.5; method = LinearInterp()) * u"W*s"
    end

    @testset "C6: OnTheFly persistent bounded integrate" begin
        itp = pchip_interp(xu, yu; coeffs = OnTheFly())
        tw = pchip_interp(xf, yf; coeffs = OnTheFly())
        @test integrate(itp, 0.5u"s", 2.5u"s") ≈ integrate(tw, 0.5, 2.5) * u"W*s"
    end

    @testset "C7: cardinal on a unit Range axis (2-cell reciprocal)" begin
        itp = cardinal_interp(0.0u"s":1.0u"s":4.0u"s", yu)
        tw = cardinal_interp(0.0:1.0:4.0, yf)
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
    end

    @testset "C8: OOB deriv under Clamp/Fill carries derivative units" begin
        itp = linear_interp(xu, yu; extrap = ClampExtrap())
        v_in = itp(1.5u"s"; deriv = DerivOp(1))
        v_oob = itp(5.0u"s"; deriv = DerivOp(1))
        @test typeof(v_oob) === typeof(v_in)   # W/s both sides of the boundary
        @test v_oob === 0.0u"W/s"
        itf = linear_interp(xu, yu; extrap = FillExtrap(0.0u"W"))
        @test typeof(itf(5.0u"s"; deriv = DerivOp(1))) === typeof(v_in)
    end

    @testset "C9: Int grid/data ClampExtrap keeps Int OOB (Real regression)" begin
        ci = constant_interp([0, 1, 2, 3, 4], [10, 20, 40, 80, 50]; extrap = ClampExtrap())
        @test ci(2) === 40
        @test ci(7) === 50    # was 50.0 (Float64) — carrier minted inv(oneunit(Int))
    end
end

@testitem "Unitful 1D: inference stability (@inferred, review pin F14)" begin
    using Unitful
    using Test: @inferred

    # Runtime-type pins (F1–F13) freeze WHAT comes out; these freeze that the
    # compiler can PROVE it — duck paths must be as inferable as the Real ones
    # (same standard as the Real files' "Type stability" testsets).
    xu = [0.0, 1.0, 2.5, 3.0, 4.0] .* u"s"
    yu = [1.0, 2.0, 4.0, 8.0, 5.0] .* u"W"
    yf = [1.0, 2.0, 4.0, 8.0, 5.0]
    TW = typeof(1.0u"W")
    TWs = typeof(1.0u"W/s")
    TWi = typeof(1.0u"W*s")
    qs = [0.5, 1.5] .* u"s"

    @testset "persistent eval / deriv / batch" begin
        for f in (linear_interp, cubic_interp, quadratic_interp, pchip_interp, akima_interp, cardinal_interp)
            itp = f(xu, yu)
            @test (@inferred itp(1.5u"s")) isa TW
            @test (@inferred itp(qs)) isa Vector{TW}
        end
        litp = linear_interp(xu, yu)
        @test (@inferred litp(1.5u"s"; deriv = DerivOp(1))) isa TWs
        @test (@inferred litp(qs; deriv = DerivOp(1))) isa Vector{TWs}
    end

    @testset "integrate family" begin
        litp = linear_interp(xu, yu)
        citp = cubic_interp(xu, yu)
        @test (@inferred integrate(litp)) isa TWi
        @test (@inferred integrate(citp)) isa TWi
        @test (@inferred integrate(litp, 0.5u"s", 2.5u"s")) isa TWi
        @test (@inferred cumulative_integrate(litp)) isa Vector{TWi}
    end

    @testset "one-shot forms (incl. gated solver routes)" begin
        @test (@inferred linear_interp(xu, yu, 1.5u"s")) isa TW
        @test (@inferred cubic_interp(xu, yu, 1.5u"s")) isa TW
        @test (@inferred quadratic_interp(xu, yu, 1.5u"s")) isa TW
        @test (@inferred pchip_interp(xu, yu, 1.5u"s")) isa TW
        @test (@inferred integrate(xu, yu; method = LinearInterp())) isa TWi
    end

    @testset "OOB extrap: same inferred type as in-domain" begin
        itc = linear_interp(xu, yu; extrap = ClampExtrap())
        @test (@inferred itc(5.0u"s")) isa TW
        @test (@inferred itc(5.0u"s"; deriv = DerivOp(1))) isa TWs
        ci = constant_interp([0, 1, 2, 3, 4], [10, 20, 40, 80, 50]; extrap = ClampExtrap())
        @test (@inferred ci(7)) === 50
    end

    @testset "unit grid + unitless values" begin
        itp = linear_interp(xu, yf)
        @test (@inferred itp(1.5u"s")) isa Float64
        @test (@inferred integrate(itp)) isa typeof(1.0u"s")
    end
end

@testitem "Unitful 1D: unit grid + unitless values (review pin F5)" begin
    using Unitful

    # `_promote_grid_float` assumed Tg <: Real: a duck grid + PROMOTABLE value
    # hit `float(promote_type(Quantity, Float64))` → abstract Quantity → throw.
    # Duck grids must pass through raw — never promoted against the value type,
    # never `float`ed (mirrors the ND gate `Tg_raw <: Real ? ... : Tg_raw`).
    FI = FastInterpolations
    xs = [0.0, 1.0, 2.5, 3.0] .* u"s"
    xf = [0.0, 1.0, 2.5, 3.0]
    yf = [1.0, 2.0, 4.0, 8.0]   # unitless values on a unit grid

    @testset "witness: duck Tg passes through raw" begin
        Tq = typeof(1.0u"s")
        @test FI._promote_grid_float(Tq, Float64) === Tq
        @test FI._promote_grid_float(Tq, Float32) === Tq
        # Real grids keep value-precision widening (unchanged contract)
        @test FI._promote_grid_float(Float32, Float64) === Float64
        @test FI._promote_grid_float(Int, Float64) === Float64
    end

    @testset "build/eval/integrate: Linear, Constant, Cubic, PCHIP" begin
        for method in (LinearInterp(), ConstantInterp(), CubicInterp(), PchipInterp())
            itp = interp(xs, yf; method)
            tw = interp(xf, yf; method)
            @test itp(1.5u"s") ≈ tw(1.5)
            @test integrate(itp) ≈ integrate(tw) * u"s"
        end
    end

    @testset "one-shot eval" begin
        @test interp(xs, yf, 1.5u"s"; method = LinearInterp()) ≈
            interp(xf, yf, 1.5; method = LinearInterp())
    end
end

@testitem "Unitful 1D: LinearSearch + spacing accessors (review pin F4)" begin
    using Unitful
    using InteractiveUtils: @which

    FI = FastInterpolations
    # Non-uniform spacing so cached-reciprocal paths differ from a naive span.
    xs = [0.0, 1.0, 2.5, 3.0] .* u"s"
    xf = [0.0, 1.0, 2.5, 3.0]
    yw = [1.0, 2.0, 4.0, 8.0] .* u"W"
    yf = [1.0, 2.0, 4.0, 8.0]

    @testset "F4: LinearSearch eval on unit grid (was MethodError)" begin
        # `_search_linear!` kept `where {T <: Real}` while its siblings were
        # relaxed — LinearSearch() + unit Vector grid threw at eval time.
        itp = linear_interp(xs, yw; search = LinearSearch())
        tw = linear_interp(xf, yf; search = LinearSearch())
        @test itp(1.5u"s") ≈ tw(1.5) * u"W"
        # Monotone batch is LinearSearch's intended workload (hint walk,
        # including the backward-walk branch).
        qf = [0.25, 0.75, 1.25, 2.6, 2.9, 0.5]
        @test itp(qf .* u"s") ≈ tw(qf) .* u"W"
    end

    @testset "F4b: 4-arg `_get_h`/`_get_inv_h` accept unit endpoints" begin
        # Search-result forms `(x, idx, xL, xR)` kept `::Real` endpoints.
        # Relaxation must cover the WHOLE overload family: a partially relaxed
        # raw fallback would silently steal wrapped-axis dispatch instead.
        cv = FI._CachedVector(xs)
        cr = FI._CachedRange(0.0u"s":0.5u"s":3.0u"s")
        @test FI._get_inv_h(xs, 2, xs[2], xs[3]) == inv(xs[3] - xs[2])
        @test FI._get_inv_h(cv, 2, xs[2], xs[3]) == cv.inv_h[2]
        @test FI._get_inv_h(cr, 1, cr[1], cr[2]) == cr.inv_h
    end

    @testset "F4c: width-first `(Tw, x, idx, xL, xR)` accept unit endpoints" begin
        Tw = typeof(1.0u"s")
        cv = FI._CachedVector(xs)
        cr = FI._CachedRange(0.0u"s":0.5u"s":3.0u"s")
        @test FI._get_inv_h(Tw, xs, 2, xs[2], xs[3]) == inv(xs[3] - xs[2])
        @test FI._get_inv_h(Tw, cv, 2, xs[2], xs[3]) == cv.inv_h[2]
        @test FI._get_inv_h(Tw, cr, 1, cr[1], cr[2]) == cr.inv_h
    end

    @testset "F4d: exclusive-periodic seam keeps its OWN overloads" begin
        # The `_ExclusivePeriodicAxis` seam-aware overloads (`_get_h`,
        # `_get_inv_h`, `_alpha_of`) kept `::Real` args. Unit args then either
        # threw (`_get_inv_h`) or SILENTLY fell through to the generic
        # `::AbstractVector` fallback (`_get_h`, `_alpha_of`) — losing the
        # seam-cell branch. Pin dispatch with @which, not just values.
        g = FI._ExclusivePeriodicAxis(xs, 4.0u"s")
        seam = length(g.inner)
        xL, xR = xs[end], g._x_max
        @test FI._get_h(g, seam, xL, xR) == xR - xL
        @test FI._get_inv_h(g, seam, xL, xR) == inv(xR - xL)
        @test FI._get_inv_h(typeof(xR), g, seam, xL, xR) == inv(xR - xL)
        # Interior cell delegates to the wrapped inner axis (cached reciprocal).
        @test FI._get_inv_h(g, 2, xs[2], xs[3]) == FI._get_inv_h(g.inner, 2, xs[2], xs[3])

        m_h = @which FI._get_h(g, seam, xL, xR)
        m_a = @which FI._alpha_of(3.5u"s", xL, xR, g)
        @test occursin("_ExclusivePeriodicAxis", string(m_h.sig))
        @test occursin("_ExclusivePeriodicAxis", string(m_a.sig))
        @test FI._alpha_of(3.5u"s", xL, xR, g) == 0.5
    end
end

@testitem "Unitful 1D: Constant derivative carries grid⁻¹ units (Range-duck audit)" begin
    using Unitful
    Wps = typeof(1.0u"W/s")
    Wps2 = typeof(1.0u"W/s^2")

    yw = [1.0, 4.0, 9.0, 16.0, 25.0] .* u"W"

    # The Constant deriv kernel returned `0 * y * one(dL)` — value units (W), dropping
    # the derivative's grid⁻¹ scale. Scalar gave wrong units (W not W/s); the batch loop
    # then CRASHED (DimensionError) storing a W value into a W/s buffer. Not Range-
    # specific (Vector too), but the Range-duck audit surfaced it.
    for (gname, xg) in (
            ("Vector", [0.0, 1.0, 2.0, 3.0, 4.0] .* u"s"),
            ("LinRange", LinRange(0.0u"s", 4.0u"s", 5)),
            ("StepRangeLen", 0.0u"s":1.0u"s":4.0u"s"),
        )
        itp = constant_interp(xg, yw)
        @testset "$gname: scalar deriv units" begin
            d1 = itp(1.5u"s"; deriv = DerivOp(1))
            d2 = itp(1.5u"s"; deriv = DerivOp(2))
            @test d1 isa Wps            # value 0, units W/s
            @test iszero(d1)
            @test d2 isa Wps2
            @test iszero(d2)
        end
        @testset "$gname: batch deriv (was DimensionError)" begin
            out = itp([0.5, 1.5, 2.5] .* u"s"; deriv = DerivOp(1))
            @test eltype(out) === Wps
            @test all(iszero, out)
        end
    end
end

@testitem "Unitful 1D: unit Range grids on the _CachedRange path (LinRange/StepRangeLen)" begin
    using Unitful
    using Test: @inferred
    FI = FastInterpolations

    # Existing unit coverage builds Vector grids (`collect`). A `LinRange`/`StepRangeLen`
    # of Quantities must wrap into a `_CachedRange` (concrete `h`/`inv_h`) and drive the
    # same zero-alloc, inference-stable, value-correct path as a Real Range.
    xf = LinRange(1.0, 10.0, 10)
    yf = [Float64(i^2) for i in 1:10]
    yw = yf .* u"W"
    TW = typeof(1.0u"W")
    qf = [1.4, 3.7, 5.5, 8.2, 9.9]

    for (gname, xu) in (
            ("LinRange", LinRange(1.0u"s", 10.0u"s", 10)),
            ("StepRangeLen", 1.0u"s":1.0u"s":10.0u"s"),
        )
        @testset "$gname: _CachedRange wrap + value/inference/alloc parity" begin
            for (nm, f) in (
                    ("linear", linear_interp), ("cubic", cubic_interp),
                    ("quadratic", quadratic_interp), ("pchip", pchip_interp),
                    ("akima", akima_interp), ("constant", constant_interp),
                )
                iu = f(xu, yw)
                ir = f(xf, yf)
                # Axis is a _CachedRange (not silently demoted to a Vector).
                # `_itp_grid` is the family-uniform accessor (Cubic stores `cache.x`).
                @test FI._itp_grid(iu) isa FI._CachedRange
                # Value parity with the Real-Range interpolant.
                @test iu(3.5u"s") ≈ ir(3.5) * u"W"
                @test iu(qf .* u"s") ≈ ir(qf) .* u"W"
                # Inference-stable, concrete unit output.
                @test (@inferred iu(3.5u"s")) isa TW
                # Zero-alloc scalar eval (warm).
                iu(3.5u"s")
                @test (@allocated iu(3.5u"s")) == 0
            end
        end
    end

    @testset "integrate parity on unit LinRange" begin
        iu = linear_interp(LinRange(1.0u"s", 10.0u"s", 10), yw)
        ir = linear_interp(xf, yf)
        @test integrate(iu) ≈ integrate(ir) * u"W*s"
        @test (@inferred integrate(iu)) isa typeof(1.0u"W*s")
    end
end

@testitem "Unitful 1D: one-shot batch + deriv carries grid⁻ᴺ units (review pin F16)" begin
    using Unitful

    # The persistent call path sizes its deriv batch buffer via the deriv-aware
    # `_promote_eltype(itp, Tq, deriv)` fold, but the *allocating one-shot* batch
    # entries (`fam_interp(x, y, x_query; deriv=…)`) sized the output in value space
    # and threw `DimensionError` on unit grids — a first derivative lives in
    # value/grid space (W/s), not value space (W).
    xs = [0.0, 1.0, 2.5, 3.0, 4.0] .* u"s"
    xf = [0.0, 1.0, 2.5, 3.0, 4.0]
    yw = [1.0, 2.0, 4.0, 8.0, 5.0] .* u"W"
    yf = [1.0, 2.0, 4.0, 8.0, 5.0]
    qf = [0.75, 1.25, 2.6, 3.9]
    qs = qf .* u"s"
    d1 = DerivOp(1)
    TWs = typeof(1.0u"W/s")     # value/grid units for a first derivative

    dyf = [1.0, 0.5, 0.0, -0.5, -1.0]
    dyu = dyf .* u"W/s"

    # (name, unit one-shot deriv closure, Real-twin deriv closure)
    fams = [
        ("linear", () -> linear_interp(xs, yw, qs; deriv = d1), () -> linear_interp(xf, yf, qf; deriv = d1)),
        ("constant", () -> constant_interp(xs, yw, qs; deriv = d1), () -> constant_interp(xf, yf, qf; deriv = d1)),
        ("quadratic", () -> quadratic_interp(xs, yw, qs; deriv = d1), () -> quadratic_interp(xf, yf, qf; deriv = d1)),
        ("cubic", () -> cubic_interp(xs, yw, qs; deriv = d1), () -> cubic_interp(xf, yf, qf; deriv = d1)),
        ("pchip", () -> pchip_interp(xs, yw, qs; deriv = d1), () -> pchip_interp(xf, yf, qf; deriv = d1)),
        ("akima", () -> akima_interp(xs, yw, qs; deriv = d1), () -> akima_interp(xf, yf, qf; deriv = d1)),
        ("cardinal", () -> cardinal_interp(xs, yw, qs; deriv = d1), () -> cardinal_interp(xf, yf, qf; deriv = d1)),
        ("hermite", () -> hermite_interp(xs, yw, dyu, qs; deriv = d1), () -> hermite_interp(xf, yf, dyf, qf; deriv = d1)),
    ]
    for (nm, fu, fr) in fams
        @testset "$nm: one-shot batch deriv → W/s" begin
            out = fu()
            @test eltype(out) === TWs
            @test out ≈ fr() .* u"W/s"
        end
    end

    @testset "second derivative → W/s² (N-fold witness)" begin
        d2 = DerivOp(2)
        out = quadratic_interp(xs, yw, qs; deriv = d2)
        @test eltype(out) === typeof(1.0u"W/s^2")
        @test out ≈ quadratic_interp(xf, yf, qf; deriv = d2) .* u"W/s^2"
    end
end

@testitem "Unitful 1D: higher zero-derivative units (review pin P1-2)" begin
    using Unitful

    # A zero higher-derivative (linear N≥2, quadratic N≥3, cubic/hermite-family N≥4)
    # must live in value/gridᴺ space, not value space — like Constant's deriv. Otherwise
    # a unit-grid deriv batch throws DimensionError storing value-unit 0 into a value/gridᴺ
    # buffer, and the scalar returns the wrong (value) units.
    xu = collect(1.0:6.0) .* u"s"
    yw = [1.0, 2.0, 4.0, 7.0, 5.0, 3.0] .* u"W"
    qb = [2.5, 3.5] .* u"s"
    dyu = fill(0.5u"W/s", length(xu))

    # (name, itp, first zero-derivative order) — scalar carries W/sⁿ, batch stays storable.
    cases = [
        ("linear", linear_interp(xu, yw), 2),
        ("quadratic", quadratic_interp(xu, yw), 3),
        ("cubic", cubic_interp(xu, yw), 4),
        ("pchip", pchip_interp(xu, yw), 4),
        ("akima", akima_interp(xu, yw), 4),
        ("hermite", hermite_interp(xu, yw, dyu), 4),
    ]
    for (nm, itp, n0) in cases
        @testset "$nm zero-deriv scalar units (n0=$n0, n0+1)" begin
            for n in (n0, n0 + 1)
                v = itp(2.5u"s"; deriv = DerivOp(n))
                @test unit(v) === unit(1.0u"W" / 1.0u"s"^n)   # value/gridⁿ
                @test iszero(ustrip(v))
            end
        end
    end

    @testset "batch zero-deriv is storable (no DimensionError)" begin
        @test eltype(linear_interp(xu, yw, qb; deriv = DerivOp(2))) === typeof(1.0u"W/s^2")
        @test eltype(quadratic_interp(xu, yw, qb; deriv = DerivOp(3))) === typeof(1.0u"W/s^3")
        @test eltype(cubic_interp(xu, yw, qb; deriv = DerivOp(4))) === typeof(1.0u"W/s^4")
        @test eltype(pchip_interp(xu, yw, qb; deriv = DerivOp(4))) === typeof(1.0u"W/s^4")
    end
end
