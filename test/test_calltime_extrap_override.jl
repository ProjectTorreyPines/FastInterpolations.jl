# Call-time `extrap` override on a persistent interpolant. The stored extrap is
# the construction-time contract; the ONLY permitted per-call override is
# `InBounds` (an in-domain fast-path assertion, not an extrapolation contract).
# Any other explicit extrap errors.

@testitem "call-time extrap override — 1D scalar+vector" begin
    x = collect(range(0.0, 10.0, 21))
    y = sin.(x)
    xq = 5.0
    xqv = collect(range(0.5, 9.5, 50))

    # Interpolants built with NoExtrap (the default contract). A call-time
    # InBounds override must reproduce the default in-domain result exactly,
    # since InBounds only skips the domain check for guaranteed-in-domain queries.
    for build in (linear_interp, quadratic_interp, cubic_interp, constant_interp)
        itp = build(x, y)                      # stored extrap = NoExtrap()

        @testset "$build: InBounds override == default (in-domain)" begin
            @test itp(xq; extrap = InBounds()) == itp(xq)
            @test itp(xqv; extrap = InBounds()) == itp(xqv)
            out = similar(xqv)
            itp(out, xqv; extrap = InBounds())
            @test out == itp(xqv)
        end

        @testset "$build: InBounds(last=:exclusive) works in-domain" begin
            @test itp(xq; extrap = InBounds(last = :exclusive)) == itp(xq)
        end

        @testset "$build: omitting extrap is unchanged" begin
            @test itp(xq; extrap = nothing) == itp(xq)
        end

        @testset "$build: disallowed extrap override errors" begin
            @test_throws ArgumentError itp(xq; extrap = ClampExtrap())
            @test_throws ArgumentError itp(xq; extrap = ExtendExtrap())
            @test_throws ArgumentError itp(xqv; extrap = FillExtrap(0.0))
            # even re-passing the same stored mode errors (only omission uses stored)
            @test_throws ArgumentError itp(xq; extrap = NoExtrap())
            # genuine junk is cut at the kwarg type boundary (not a friendly error)
            @test_throws TypeError itp(xq; extrap = 5)
        end
    end
end

@testitem "call-time extrap override — ND scalar" begin
    xg = collect(range(0.0, 10.0, 11))
    yg = collect(range(0.0, 5.0, 9))
    data = [sin(xi) * cos(yj) for xi in xg, yj in yg]
    q = (5.0, 2.5)                                     # in-domain 2D query

    for build in (linear_interp, quadratic_interp, cubic_interp, constant_interp)
        itp = build((xg, yg), data)                    # stored extraps = (NoExtrap(), NoExtrap())

        @testset "$build: InBounds override == default (in-domain)" begin
            @test itp(q; extrap = InBounds()) == itp(q)             # broadcast all axes
            @test itp(q...; extrap = InBounds()) == itp(q)          # vararg form
        end

        @testset "$build: per-axis tuple (nothing keeps stored)" begin
            @test itp(q; extrap = (InBounds(), nothing)) == itp(q)  # axis 1 fast, axis 2 stored
            @test itp(q; extrap = (InBounds(), InBounds())) == itp(q)
        end

        @testset "$build: disallowed override errors" begin
            @test_throws ArgumentError itp(q; extrap = ClampExtrap())
            @test_throws ArgumentError itp(q; extrap = (ClampExtrap(), InBounds()))
            @test_throws ArgumentError itp(q; extrap = (NoExtrap(), NoExtrap()))
            @test_throws ArgumentError itp(q; extrap = (InBounds(),))  # wrong arity (1 elem for 2D)
            @test_throws TypeError itp(q; extrap = 5)                 # scalar junk cut at kwarg boundary
            @test_throws TypeError itp(q; extrap = (5, nothing))      # junk tuple element cut as TypeError
        end
    end
end

@testitem "call-time extrap override — ND batch" begin
    xg = collect(range(0.0, 10.0, 11))
    yg = collect(range(0.0, 5.0, 9))
    data = [sin(xi) * cos(yj) for xi in xg, yj in yg]
    qxs = collect(range(0.5, 9.5, 7))
    qys = collect(range(0.3, 4.7, 7))
    soa = (qxs, qys)                                   # SoA: tuple of coord vectors
    aos = [(qxs[i], qys[i]) for i in eachindex(qxs)]   # AoS: vector of query tuples

    for build in (linear_interp, quadratic_interp, cubic_interp, constant_interp)
        itp = build((xg, yg), data)

        @testset "$build: batch InBounds override == default (SoA + AoS)" begin
            @test itp(soa; extrap = InBounds()) == itp(soa)
            @test itp(aos; extrap = InBounds()) == itp(aos)
            out = similar(qxs)
            itp(out, soa; extrap = InBounds())
            @test out == itp(soa)
        end

        @testset "$build: batch per-axis + disallowed" begin
            @test itp(soa; extrap = (InBounds(), nothing)) == itp(soa)
            @test_throws ArgumentError itp(soa; extrap = ClampExtrap())
        end
    end
end

@testitem "call-time extrap override — Hermite ND + NoInterp Hetero" begin
    # Two families whose construction differs from the tensor-product loops above.
    x = collect(range(0.0, 1.0, 6))
    y = collect(range(0.0, 1.0, 5))
    data = [sin(xi) * cos(yj) + xi for xi in x, yj in y]
    q = (0.5, 0.5)
    soa = (collect(range(0.1, 0.9, 7)), collect(range(0.15, 0.85, 7)))

    @testset "Hermite ND (user partials)" begin
        dfdx = [cos(xi) * cos(yj) + 1 for xi in x, yj in y]
        dfdy = [-sin(xi) * sin(yj) for xi in x, yj in y]
        d2 = [-cos(xi) * sin(yj) for xi in x, yj in y]
        p = HermitePartials((1, 0) => dfdx, (0, 1) => dfdy, (1, 1) => d2)
        itp = hermite_interp((x, y), data, p)
        @test itp(q; extrap = InBounds()) == itp(q)                # broadcast all axes
        @test itp(q; extrap = (InBounds(), nothing)) == itp(q)     # per-axis
        @test itp(soa; extrap = InBounds()) == itp(soa)            # batch
        @test_throws ArgumentError itp(q; extrap = ClampExtrap())
        @test_throws ArgumentError itp(q; extrap = (InBounds(),))  # wrong arity
    end

    @testset "NoInterp-axis Hetero: InBounds is a value-safe no-op" begin
        # The NoInterp branch reads `itp.extraps` directly — the override is accepted
        # (validated) but must not change the result; a disallowed mode still throws.
        itp = interp((x, y), data; method = (CubicInterp(), NoInterp()))
        for k in (1, 3, 5)
            qn = (0.5, GridIdx(k))
            @test itp(qn; extrap = InBounds()) == itp(qn)
            @test itp(qn; extrap = (InBounds(), nothing)) == itp(qn)
        end
        @test_throws ArgumentError itp((0.5, GridIdx(2)); extrap = ClampExtrap())
    end
end

@testitem "call-time extrap override — HeteroND (all forms)" setup = [AllocConstants] begin
    using FastInterpolations: _resolve_extrap_override_nd, HeteroInterpolantND
    using ForwardDiff

    x = collect(range(0.0, 1.0, 6))
    y = collect(range(0.0, 1.0, 5))
    data = [sin(xi) * cos(yj) + xi for xi in x, yj in y]
    q = (0.5, 0.5)                                       # in-domain 2D query
    qxs = collect(range(0.1, 0.9, 7))
    qys = collect(range(0.15, 0.85, 7))
    soa = (qxs, qys)                                     # SoA: tuple of coord vectors
    aos = [(qxs[i], qys[i]) for i in eachindex(qxs)]     # AoS: vector of query tuples

    # Three HeteroInterpolantND realizations, each a distinct eval path:
    #  - mixed global-solve OnTheFly (CubicInterp × LinearInterp)
    #  - windowed OnTheFly with a local-Hermite axis (Pchip → OnTheFly Hermite ND)
    #  - PreCompute (`_HeteroPartials`) — direct ND kernel, gets the full fast-path
    builds = (
        "mixed-OTF" => interp((x, y), data; method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()),
        "windowed-OTF" => interp((x, y), data; method = (PchipInterp(), LinearInterp()), coeffs = OnTheFly()),
        "PreCompute" => interp((x, y), data; method = (CubicInterp(), LinearInterp()), coeffs = PreCompute()),
    )

    @testset "$name: InBounds override == default (in-domain)" for (name, itp) in builds
        @test itp isa HeteroInterpolantND
        # scalar tuple + vararg forms; single InBounds broadcasts to all axes
        @test itp(q; extrap = InBounds()) == itp(q)
        @test itp(q...; extrap = InBounds()) == itp(q)
        @test itp(q; extrap = InBounds(last = :exclusive)) == itp(q)
        # per-axis tuple: `nothing` keeps that axis's stored mode
        @test itp(q; extrap = (InBounds(), nothing)) == itp(q)
        @test itp(q; extrap = (InBounds(), InBounds())) == itp(q)
        # batch: SoA, AoS, in-place
        @test itp(soa; extrap = InBounds()) == itp(soa)
        @test itp(aos; extrap = InBounds()) == itp(aos)
        out = similar(qxs)
        itp(out, soa; extrap = InBounds())
        @test out == itp(soa)
        @test itp(soa; extrap = (InBounds(), nothing)) == itp(soa)
    end

    @testset "$name: disallowed override errors" for (name, itp) in builds
        @test_throws ArgumentError itp(q; extrap = ClampExtrap())
        @test_throws ArgumentError itp(q; extrap = (ClampExtrap(), InBounds()))
        @test_throws ArgumentError itp(q; extrap = (NoExtrap(), NoExtrap()))  # same-mode also errors
        @test_throws ArgumentError itp(soa; extrap = ClampExtrap())
        @test_throws TypeError itp(q; extrap = 5)                             # scalar junk cut at kwarg boundary
        @test_throws TypeError itp(q; extrap = (5, nothing))                  # junk tuple element → TypeError
    end

    @testset "type stability & allocation" begin
        itp = builds[1].second                           # mixed-OTF
        # @inferred + isa pins: concrete Float64 return, no Union in the hot path
        @test @inferred(itp(q; extrap = InBounds())) isa Float64
        @test @inferred(itp(q; extrap = InBounds(last = :exclusive))) isa Float64
        @test @inferred(itp(q; extrap = (InBounds(), nothing))) isa Float64
        # resolution must infer a CONCRETE per-axis tuple (guards the per-axis
        # extrap-union boxing trap); default returns the stored tuple, same object
        @test @inferred(_resolve_extrap_override_nd(itp, (InBounds(), nothing))) isa
            Tuple{<:InBounds, <:AbstractExtrap}
        @test @inferred(_resolve_extrap_override_nd(itp, nothing)) === itp.extraps

        # allocation-free: bind the query INSIDE the barrier (a tuple passed as an
        # argument boxes 16B at the call ABI — a measurement artifact, not the eval).
        # Default path was measured 0-alloc for all three forms; the override must match.
        ovr(f) = (p = (0.5, 0.5); f(p; extrap = InBounds()))
        dfl(f) = (p = (0.5, 0.5); f(p))
        for (_, itpb) in builds
            ovr(itpb)
            dfl(itpb)                                     # warm up each specialization
            @test @allocated(ovr(itpb)) <= ALLOC_THRESHOLD
            @test @allocated(dfl(itpb)) <= ALLOC_THRESHOLD   # default unchanged
        end
    end

    @testset "Dual queries (AD) through InBounds override" begin
        itp = builds[1].second
        g_ib = ForwardDiff.gradient(p -> itp((p[1], p[2]); extrap = InBounds()), [0.5, 0.5])
        g_df = ForwardDiff.gradient(p -> itp((p[1], p[2])), [0.5, 0.5])
        @test g_ib ≈ g_df
    end
end

@testitem "call-time extrap override — type stability & allocation" setup = [AllocConstants] begin
    using FastInterpolations: _resolve_extrap_override
    x = collect(range(0.0, 10.0, 21))
    y = sin.(x)
    itp1 = cubic_interp(x, y)

    xg = collect(range(0.0, 10.0, 11))
    yg = collect(range(0.0, 5.0, 9))
    data = [sin(xi) * cos(yj) for xi in xg, yj in yg]
    itpN = cubic_interp((xg, yg), data)

    @testset "inference (concrete return, no Union)" begin
        @test @inferred(itp1(5.0; extrap = InBounds())) isa Float64
        @test @inferred(itp1(5.0; extrap = InBounds(last = :exclusive))) isa Float64
        @test @inferred(itpN((5.0, 2.5); extrap = InBounds())) isa Float64
        @test @inferred(itpN((5.0, 2.5); extrap = (InBounds(), nothing))) isa Float64
        # ND per-axis resolution map must infer a concrete tuple (no Union leak)
        @test @inferred(map(_resolve_extrap_override, (NoExtrap(), NoExtrap()), (InBounds(), nothing))) isa Tuple
    end

    @testset "allocation-free vs default (function barrier)" begin
        # Bind the query INSIDE the barrier. Passing an NTuple as an argument to the
        # measured function boxes 16 bytes at the call ABI boundary — a measurement
        # artifact, not the eval; with the query bound internally the eval is
        # allocation-free. `ALLOC_THRESHOLD` is 0 on Julia ≥ 1.12, with LTS slack.
        ovr1(itp) = itp(5.0; extrap = InBounds())
        ovrN(itp) = (q = (5.0, 2.5); itp(q; extrap = InBounds()))
        ovr1(itp1); ovrN(itpN)                     # warm up
        @test @allocated(ovr1(itp1)) <= ALLOC_THRESHOLD
        @test @allocated(ovrN(itpN)) <= ALLOC_THRESHOLD
    end
end

@testitem "call-time extrap override — Dual queries (AD)" begin
    using ForwardDiff
    x = collect(range(0.0, 10.0, 21))
    y = sin.(x)
    itp1 = cubic_interp(x, y)
    # 1D: derivative through an InBounds-override eval matches the default eval
    @test ForwardDiff.derivative(t -> itp1(t; extrap = InBounds()), 5.0) ≈
        ForwardDiff.derivative(t -> itp1(t), 5.0)

    xg = collect(range(0.0, 10.0, 11))
    yg = collect(range(0.0, 5.0, 9))
    data = [sin(xi) * cos(yj) for xi in xg, yj in yg]
    itpN = cubic_interp((xg, yg), data)
    g_ib = ForwardDiff.gradient(p -> itpN((p[1], p[2]); extrap = InBounds()), [5.0, 2.5])
    g_df = ForwardDiff.gradient(p -> itpN((p[1], p[2])), [5.0, 2.5])
    @test g_ib ≈ g_df
end

@testitem "call-time extrap override — InBounds parity across stored extraps" begin
    x = collect(range(0.0, 10.0, 21))
    y = sin.(x)
    xq = 5.0
    xqv = collect(range(0.5, 9.5, 40))

    # For an in-domain query, InBounds must reproduce the default result regardless
    # of which extrap the interpolant was built with (Clamp/Fill/Wrap/Extend).
    @testset "stored extrap = $(nameof(typeof(e)))" for e in
        (ClampExtrap(), FillExtrap(0.0), WrapExtrap(), ExtendExtrap())
        itp = cubic_interp(x, y; extrap = e)
        @test itp(xq; extrap = InBounds()) == itp(xq)
        @test itp(xqv; extrap = InBounds()) == itp(xqv)
    end

    @testset "periodic interpolant (stored WrapExtrap)" begin
        xp = collect(range(0.0, 2π, 21))
        yp = sin.(xp)                       # yp[1] ≈ yp[end] ≈ 0 (inclusive periodic)
        itp = cubic_interp(xp, yp; bc = PeriodicBC())
        @test itp(3.0; extrap = InBounds()) == itp(3.0)              # in-domain, no wrap needed
        @test itp(collect(range(0.1, 6.0, 30)); extrap = InBounds()) ==
            itp(collect(range(0.1, 6.0, 30)))
    end
end
