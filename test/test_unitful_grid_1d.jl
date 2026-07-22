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
