# Endpoint-contract tests for the parameterized `InBounds{First, Last}` extrap.
#
# `InBounds(last = :exclusive)` promises `first(x) ≤ xq < last(x)`; unit-step range
# axes then drop the direct search's top-cell cap (`min(·, len−1)` — see
# `_search_direct_inbounds(x, xq, ::InBounds{First, :exclusive})`). Pinned here:
#   1. constructor surface: defaults, kwarg forms, validation errors;
#   2. closed `InBounds()` semantics are UNCHANGED (incl. `xq == last(x)`);
#   3. exclusive-last is BIT-IDENTICAL (`===`) to closed for strictly-interior
#      queries on unit-step grids — 1D scalar/batch across families, 2D
#      persistent + one-shot, mixed per-axis contracts;
#   4. GridIdx keeps its index short-circuit under an endpoint contract;
#   5. float-step ranges have NO no-cap arm (their cap doubles as a rounding
#      guard) — exclusive-last delegates to the capped closed search there, so
#      it stays bit-identical even at `xq == last(x)`.
# `xq == last(x)` under exclusive-last on a UNIT-STEP axis is a violated promise
# (undefined, like `Base.@inbounds`) and is deliberately never exercised.

@testitem "InBounds endpoint constructors and validation" begin
    @test InBounds() isa InBounds{:inclusive, :inclusive}
    @test InBounds(last = :exclusive) isa InBounds{:inclusive, :exclusive}
    @test InBounds(first = :exclusive) isa InBounds{:exclusive, :inclusive}
    @test InBounds(first = :exclusive, last = :exclusive) isa InBounds{:exclusive, :exclusive}

    # Singleton identity — internal promotion sites compare `=== InBounds()`.
    @test InBounds() === InBounds{:inclusive, :inclusive}()

    @test_throws ArgumentError InBounds(first = :closed)
    @test_throws ArgumentError InBounds(last = :open)
    # Inner constructor rejects raw bad parameters (Symbol-typed and otherwise).
    @test_throws ErrorException InBounds{:bad, :inclusive}()
    @test_throws ErrorException InBounds{:inclusive, 1}()

    # Compact kwarg-form show (round-trips as constructor calls; GridIdx/DerivOp precedent).
    @test repr(InBounds()) == "InBounds()"
    @test repr(InBounds(last = :exclusive)) == "InBounds(last = :exclusive)"
    @test repr(InBounds(first = :exclusive)) == "InBounds(first = :exclusive)"
    @test repr(InBounds(first = :exclusive, last = :exclusive)) ==
        "InBounds(first = :exclusive, last = :exclusive)"
end

@testitem "batch domain check promotes to exclusive-last on unit-step axes (proven)" begin
    using FastInterpolations: _check_domain

    # `_CachedRange` instances via the resolved ND grids (same extraction as the
    # perf probes) — one per tag family.
    cr(g) = linear_interp((g, g), zeros(length(g), length(g))).grids[1]
    g_us = cr(1:16)                            # _UnitStep
    g_ot = cr(Base.OneTo(16))                  # _OneTo (<: _AbstractUnitStep)
    g_fs = cr(range(0.0, 1.0; length = 17))    # float-step → generic method (gate)

    @testset "NoExtrap batch: strict → exclusive, touch-hi → closed, OOB → throw" begin
        @test _check_domain(g_us, [2.0, 5.5, prevfloat(16.0)], NoExtrap()) ===
            InBounds(last = :exclusive)
        @test _check_domain(g_ot, [1.0, 15.5], NoExtrap()) === InBounds(last = :exclusive)
        @test _check_domain(g_us, [2.0, 16.0], NoExtrap()) === InBounds()   # touches last
        @test _check_domain(g_us, Float64[], NoExtrap()) === InBounds()     # empty → closed
        @test_throws DomainError _check_domain(g_us, [0.5, 5.0], NoExtrap())
        @test_throws DomainError _check_domain(g_us, [2.0, 16.5], NoExtrap())
    end

    @testset "float-step axis keeps the closed-only promotion (gate)" begin
        @test _check_domain(g_fs, [0.1, 0.9], NoExtrap()) === InBounds()
        @test _check_domain(g_fs, [0.1, 0.9], ClampExtrap()) === InBounds()
        # Inference pin: the literal-false strict claim must constant-fold, keeping
        # the non-unit-step promotion CONCRETE closed (no Union).
        @test Base.promote_op(_check_domain, typeof(g_fs), Vector{Float64}, NoExtrap) ==
            InBounds{:inclusive, :inclusive}
    end

    @testset "Clamp/Fill/Wrap batch: strict → exclusive, touch-hi → closed, OOB → original" begin
        @test _check_domain(g_us, [2.0, 15.0], ClampExtrap()) === InBounds(last = :exclusive)
        @test _check_domain(g_us, [2.0, 16.0], ClampExtrap()) === InBounds()
        @test _check_domain(g_us, [0.0, 5.0], ClampExtrap()) === ClampExtrap()
        @test _check_domain(g_us, [2.0, 15.0], FillExtrap(0.0)) === InBounds(last = :exclusive)
        @test _check_domain(g_us, [2.0, 17.0], FillExtrap(0.0)) === FillExtrap(0.0)
        @test _check_domain(g_us, [2.0, 15.0], WrapExtrap()) === InBounds(last = :exclusive)
    end
end

@testitem "Dual-query batch touching last must NOT promote to exclusive (no-cap OOB guard)" begin
    using FastInterpolations: _check_domain
    using ForwardDiff
    Dual = ForwardDiff.Dual

    # `Dual(16,-1) < 16.0 === true` (partial tie-break at equal primals). A Dual batch
    # whose max VALUE == last(x) must classify on primal, else it falsely promotes to
    # exclusive and the no-cap search reads y[len+1] OOB.
    cr(g) = linear_interp((g, g), zeros(length(g), length(g))).grids[1]
    g = cr(1:16)

    # max value == 16 == last(x), partial < 0 (the tie-break trap)
    xi_touch = [Dual{:t}(2.0, 0.3), Dual{:t}(8.0, 0.1), Dual{:t}(16.0, -1.0)]
    @test _check_domain(g, xi_touch, NoExtrap()) === InBounds()            # NOT exclusive
    @test _check_domain(g, xi_touch, ClampExtrap()) === InBounds()

    # strictly-interior Dual batch (max primal < last) still promotes
    xi_interior = [Dual{:t}(2.0, 1.0), Dual{:t}(15.0, -1.0)]
    @test _check_domain(g, xi_interior, NoExtrap()) === InBounds(last = :exclusive)

    # end-to-end: gradient through a Dual batch touching last(x) must not OOB
    x = 1:16
    y = collect(1.0:16.0)
    itp = linear_interp(x, y)
    qs = [3.0, 16.0]                                    # includes the exact endpoint
    J = ForwardDiff.jacobian(q -> itp(q), qs)          # batch value+partials, no OOB read
    @test J ≈ [1.0 0.0; 0.0 1.0]
end

@testitem "batch NoExtrap end-to-end rides the promoted contract, bit-identical" begin
    x = 1:16
    y = [sin(0.4i) for i in 1:16]
    qs = [1.0, 2.25, 7.5, 15.999, prevfloat(16.0)]      # strictly interior batch
    qs_hi = [2.0, 9.5, 16.0]                            # touches the right endpoint

    for f in (linear_interp, cubic_interp, quadratic_interp, constant_interp, pchip_interp)
        itp_ne = f(x, y)                                        # NoExtrap default
        itp_ib = f(x, y; extrap = InBounds())
        itp_ex = f(x, y; extrap = InBounds(last = :exclusive))
        @test itp_ne(qs) == itp_ex(qs)      # promoted batch == explicit exclusive
        @test itp_ne(qs) == itp_ib(qs)      # and still bit-identical to closed
        @test itp_ne(qs_hi) == itp_ib(qs_hi)                    # closed fallback at last
        @test itp_ne(qs_hi)[end] === itp_ib(qs_hi)[end]
    end

    # ND SoA per-axis: x-axis strictly interior (promotes), y-axis touches its
    # last (stays closed) — the mixed per-axis tuple must stay bit-identical.
    axx, axy = 1:7, 2:9
    data = [0.1i + 0.3j + 0.01i * j for i in 1:7, j in 1:8]
    qx = [1.5, 3.25, 6.9]
    qy = [2.5, 9.0, 8.5]
    out_ne = similar(qx)
    out_ib = similar(qx)
    linear_interp!(out_ne, (axx, axy), data, (qx, qy))
    linear_interp!(out_ib, (axx, axy), data, (qx, qy); extrap = (InBounds(), InBounds()))
    @test out_ne == out_ib
end

@testitem "1D exclusive-last === closed on unit-step grids (all families)" begin
    x1 = 1:16                     # _UnitStep with lo == 1 → index-space arm
    x2 = 3:18                     # _UnitStep with offset lo
    excl = InBounds(last = :exclusive)

    # Strictly-interior queries incl. the sharpest no-cap edge prevfloat(last).
    qs(x) = [
        float(first(x)), first(x) + 0.25, 0.5 * (first(x) + last(x)),
        last(x) - 1.5, prevfloat(float(last(x))),
    ]

    for x in (x1, x2)
        y = [sin(0.4i) + 0.01i for i in 1:length(x)]
        for f in (linear_interp, cubic_interp, quadratic_interp, constant_interp, pchip_interp)
            itp_c = f(x, y; extrap = InBounds())
            itp_e = f(x, y; extrap = excl)
            for q in qs(x)
                @test itp_e(q) === itp_c(q)
            end
            # Batch: `_check_domain` passes a user InBounds through untouched,
            # so the per-element search sees the exclusive contract.
            @test itp_e(qs(x)) == itp_c(qs(x))
            # One-shot scalar form.
            @test f(x, y, first(x) + 0.75; extrap = excl) ===
                f(x, y, first(x) + 0.75; extrap = InBounds())
        end
    end
end

@testitem "closed InBounds endpoint behavior unchanged (=== NoExtrap incl. xq == last)" begin
    x = 1:12
    y = [sin(0.5i) for i in 1:12]
    for f in (linear_interp, cubic_interp)
        itp_ib = f(x, y; extrap = InBounds())
        itp_ne = f(x, y)                       # NoExtrap reference
        for q in (1.0, 4.3, 11.999, 12.0)      # incl. the closed right endpoint
            @test itp_ib(q) === itp_ne(q)
        end
    end
end

@testitem "2D exclusive-last === closed (persistent + one-shot + batch + mixed axes)" begin
    axx, axy = 1:7, 2:9
    data = [sin(0.3i) + cos(0.2j) + 0.01i * j for i in 1:length(axx), j in 1:length(axy)]

    closed = (InBounds(), InBounds())
    excl = (InBounds(last = :exclusive), InBounds(last = :exclusive))
    mixed = (InBounds(last = :exclusive), InBounds())

    qs = ((1.0, 2.0), (1.25, 4.75), (3.4, 6.2), (6.999, 8.5), (prevfloat(7.0), prevfloat(9.0)))

    @testset "persistent: linear + cubic" begin
        for f in (linear_interp, cubic_interp)
            itp_c = f((axx, axy), data; extrap = closed)
            itp_e = f((axx, axy), data; extrap = excl)
            for q in qs
                @test itp_e(q) === itp_c(q)
            end
        end
    end

    @testset "mixed per-axis: exclusive x-axis, closed y-axis may touch its endpoint" begin
        itp_c = linear_interp((axx, axy), data; extrap = closed)
        itp_m = linear_interp((axx, axy), data; extrap = mixed)
        for q in ((1.5, 9.0), (prevfloat(7.0), 9.0), (2.2, 5.5))
            @test itp_m(q) === itp_c(q)
        end
    end

    @testset "one-shot scalar (all families) + batch (linear!)" begin
        for f in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
            for q in qs
                @test f((axx, axy), data, q; extrap = excl) ===
                    f((axx, axy), data, q; extrap = closed)
            end
        end
        qxs = Float64[q[1] for q in qs]
        qys = Float64[q[2] for q in qs]
        o_e = similar(qxs)
        o_c = similar(qxs)
        linear_interp!(o_e, (axx, axy), data, (qxs, qys); extrap = excl)
        linear_interp!(o_c, (axx, axy), data, (qxs, qys); extrap = closed)
        @test o_e == o_c
    end
end

@testitem "GridIdx keeps the index short-circuit under an exclusive contract" begin
    using FastInterpolations: GridIdx

    x = 1:10
    y = collect(1.0:10.0)
    itp_c = linear_interp(x, y; extrap = InBounds())
    itp_e = linear_interp(x, y; extrap = InBounds(last = :exclusive))
    # GridIdx is in-domain by construction and coordinate-free — the endpoint
    # contract must not reroute it into the coordinate lean. Includes the last
    # node: the short-circuit never runs the no-cap coordinate arithmetic.
    for k in (1, 3, 10)
        @test itp_e(GridIdx(k)) === itp_c(GridIdx(k))
        @test itp_e(GridIdx(k)) === y[k]
    end

    axx, axy = 1:7, 2:9
    data = [0.1i + 0.3j + 0.01i * j for i in 1:length(axx), j in 1:length(axy)]
    excl = (InBounds(last = :exclusive), InBounds(last = :exclusive))
    closed = (InBounds(), InBounds())
    for q in ((GridIdx(2), GridIdx(3)), (GridIdx(7), GridIdx(8)))
        @test linear_interp((axx, axy), data, q; extrap = excl) ===
            linear_interp((axx, axy), data, q; extrap = closed)
    end
end

@testitem "float-step ranges: exclusive-last delegates to the capped closed search" begin
    x = range(0.0, 1.0; length = 17)           # float-step _CachedRange, NOT unit-step
    y = sin.(x)
    itp_c = linear_interp(x, y; extrap = InBounds())
    itp_e = linear_interp(x, y; extrap = InBounds(last = :exclusive))
    # Same (capped) code path ⇒ bit-identical for every in-domain query, INCLUDING
    # xq == last(x). This pins the safety property that float-step ranges never get
    # a no-cap arm: their muladd index arithmetic can overshoot len by rounding, so
    # the cap is a rounding guard there, not just endpoint semantics.
    for q in (0.0, 0.123, 0.5, prevfloat(1.0), 1.0)
        @test itp_e(q) === itp_c(q)
    end
end

@testitem "AD: Dual query under exclusive-last === closed (value + derivative)" begin
    using ForwardDiff

    x = 1:16
    y = [sin(0.4i) for i in 1:16]
    itp_c = cubic_interp(x, y; extrap = InBounds())
    itp_e = cubic_interp(x, y; extrap = InBounds(last = :exclusive))
    for q in (1.5, 7.25, prevfloat(16.0))
        @test itp_e(q) === itp_c(q)
        @test ForwardDiff.derivative(itp_e, q) === ForwardDiff.derivative(itp_c, q)
    end
end
