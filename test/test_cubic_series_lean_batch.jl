# Equivalence gate for the lean payload-anchor batch surfaces.
# Oracles are the RETAINED full-anchor building blocks, composed exactly like
# the pre-migration batch entries — so these tests pass BEFORE the wiring
# (validating the oracle reconstruction) and must still pass bit-identically
# (`===` elementwise) AFTER it. Design: docs/design/cubic_series_payload_anchor.md §8

@testsnippet LeanBatchOracles begin
    const FI = FastInterpolations

    # Pre-migration persistent batch recipe: full anchors + _eval_series_with_extrap
    function persistent_oracle(sitp, xq::AbstractVector, op)
        Tg = eltype(sitp.cache.x)
        Tq_w = FI._coord_eltype(eltype(xq), Tg)
        aq_vec = Vector{FI._CubicAnchoredQuery{Tg, Tq_w, FI._interval_type(sitp.cache.x)}}(undef, length(xq))
        searcher = FI._resolve_search(sitp.cache.x, xq, sitp.search_policy, nothing)
        FI._fill_anchors!(aq_vec, sitp.cache.x, xq, Val(:cubic), FI._should_wrap(sitp), searcher)
        n_pts = FI.n_points(sitp)
        x_min, x_max = Tg(first(sitp.cache.x)), Tg(last(sitp.cache.x))
        K = FI.n_series(sitp)
        return [
            [
                    FI._eval_series_with_extrap(
                        sitp.y, sitp.z, n_pts, x_min, x_max, k, aq_vec[j], sitp.extrap, op
                    )
                    for j in eachindex(xq)
                ]
                for k in 1:K
        ]
    end

    # Elementwise egal compare (NaN === NaN holds under ===)
    function assert_egal(got, want)
        for k in eachindex(want), j in eachindex(want[k])
            got[k][j] === want[k][j] || return (k, j, got[k][j], want[k][j])
        end
        return nothing
    end
end

@testitem "persistent lean batch === full-anchor recipe (op × extrap × values)" setup = [LeanBatchOracles] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), 2.0)      # -0.0 boundary exercises signed-zero OOB
    y2 = collect(range(2.0, 3.0, 11))
    y_nan = [1.0, 2.0, NaN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

    xq_oob = [-0.5, 0.05, 0.5, 0.999, 1.5]
    xq_in = [0.05, 0.37, 0.94]

    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3), DerivOp(5))
    for (extrap, xq) in (
            (NoExtrap(), xq_in),
            (ExtendExtrap(), xq_oob),
            (ClampExtrap(), xq_oob),
            (FillExtrap(NaN), xq_oob),
            (FillExtrap(7.5), xq_oob),
            (WrapExtrap(), xq_oob),
            (InBounds(), xq_in),
        )
        for ydata in ((y1, y2), (y_nan, y2))
            sitp = cubic_interp(x, Series(ydata...); extrap = extrap)
            for op in ops
                want = persistent_oracle(sitp, xq, op)
                got = [Vector{Float64}(undef, length(xq)) for _ in 1:2]
                sitp(got, xq; deriv = op)
                @test assert_egal(got, want) === nothing
            end
        end
    end
end

@testitem "persistent lean batch: mixed precision + Range grid + periodic bc" setup = [LeanBatchOracles] begin
    # F32 query on F64 grid / F64 query on F32 grid
    for (Tg, Tq) in ((Float64, Float32), (Float32, Float64), (Float32, Float32))
        x = collect(Tg, range(zero(Tg), one(Tg), 11))
        y1 = collect(Tg, range(1, 2, 11))
        y2 = collect(Tg, range(2, 3, 11))
        xq = Tq[-0.5, 0.25, 1.5]
        for extrap in (ClampExtrap(), ExtendExtrap())
            sitp = cubic_interp(x, Series(y1, y2); extrap = extrap)
            for op in (EvalValue(), DerivOp(1))
                want = persistent_oracle(sitp, xq, op)
                got = [similar(want[k]) for k in 1:2]
                sitp(got, xq; deriv = op)
                @test assert_egal(got, want) === nothing
            end
        end
    end

    # Range grid
    xr = range(0.0, 1.0, 11)
    sitp_r = cubic_interp(xr, Series(collect(range(1.0, 2.0, 11)), collect(range(2.0, 3.0, 11))); extrap = ClampExtrap())
    xq = [-0.5, 0.37, 1.5]
    for op in (EvalValue(), DerivOp(2))
        want = persistent_oracle(sitp_r, xq, op)
        got = [similar(want[k]) for k in 1:2]
        sitp_r(got, xq; deriv = op)
        @test assert_egal(got, want) === nothing
    end

    # PeriodicBC persistent (inclusive endpoints)
    xp = collect(range(0.0, 1.0, 11))
    yp = sin.(2π .* xp)
    yp[end] = yp[1]
    sitp_p = cubic_interp(xp, Series(yp, 2 .* yp); bc = PeriodicBC(), extrap = ExtendExtrap())
    xqp = [0.05, 0.5, 0.94]
    for op in (EvalValue(), DerivOp(1))
        want = persistent_oracle(sitp_p, xqp, op)
        got = [similar(want[k]) for k in 1:2]
        sitp_p(got, xqp; deriv = op)
        @test assert_egal(got, want) === nothing
    end
end

@testitem "persistent lean batch: Complex series values" setup = [LeanBatchOracles] begin
    x = collect(range(0.0, 1.0, 11))
    yc = complex.(collect(range(1.0, 2.0, 11)), collect(range(-1.0, 1.0, 11)))
    xq = [-0.5, 0.37, 1.5]
    for extrap in (ClampExtrap(), ExtendExtrap())
        sitp = cubic_interp(x, Series(yc, 2 .* yc); extrap = extrap)
        for op in (EvalValue(), DerivOp(1))
            want = persistent_oracle(sitp, xq, op)
            got = [similar(want[k]) for k in 1:2]
            sitp(got, xq; deriv = op)
            @test assert_egal(got, want) === nothing
        end
    end
end

@testitem "one-shot lean batch === scalar one-shot path (op × extrap × precision)" begin
    # The scalar one-shot path keeps full anchors in this PR — it is the
    # unchanged reference for the vector batch (scalar/vector symmetry contract).
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), 2.0)
    y2 = collect(range(2.0, 3.0, 11))
    xq_oob = [-0.5, 0.05, 0.5, 0.999, 1.5]
    xq_in = [0.05, 0.37, 0.94]

    for (extrap, xq) in (
            (NoExtrap(), xq_in),
            (ExtendExtrap(), xq_oob),
            (ClampExtrap(), xq_oob),
            (FillExtrap(NaN), xq_oob),
            (FillExtrap(7.5), xq_oob),
            (WrapExtrap(), xq_oob),
        )
        for op in (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3), DerivOp(5))
            got = cubic_interp(x, Series(y1, y2), xq; extrap = extrap, deriv = op)
            for j in eachindex(xq)
                want = cubic_interp(x, Series(y1, y2), xq[j]; extrap = extrap, deriv = op)
                @test got[1][j] === want[1]
                @test got[2][j] === want[2]
            end
        end
    end

    # mixed precision both directions
    for (Tg, Tq) in ((Float64, Float32), (Float32, Float64))
        xg = collect(Tg, range(zero(Tg), one(Tg), 11))
        ya = collect(Tg, range(1, 2, 11))
        xqm = Tq[0.25, 0.75]
        got = cubic_interp(xg, Series(ya, 2 .* ya), xqm; extrap = ClampExtrap())
        for j in eachindex(xqm)
            want = cubic_interp(xg, Series(ya, 2 .* ya), xqm[j]; extrap = ClampExtrap())
            @test got[1][j] === want[1]
            @test got[2][j] === want[2]
        end
    end
end

@testitem "one-shot lean batch: exclusive-periodic seam === scalar path" begin
    # exclusive grid: last point omitted; queries past x[end] wrap through the seam
    x = collect(range(0.0, 0.9, 10))            # period 1.0, seam cell (x[10], virtual 1.0)
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    xqs = [0.02, 0.55, 0.93, 0.99]              # last two live in the seam cell

    for bc in (PeriodicBC(endpoint = :exclusive, period = 1.0),)
        for op in (EvalValue(), DerivOp(1), DerivOp(2))
            got = cubic_interp(x, Series(y1, y2), xqs; bc = bc, deriv = op)
            for j in eachindex(xqs)
                want = cubic_interp(x, Series(y1, y2), xqs[j]; bc = bc, deriv = op)
                @test got[1][j] === want[1]
                @test got[2][j] === want[2]
            end
        end
    end
end

@testitem "persistent NoExtrap mixed precision: DomainError before any output write" begin
    # RED before wiring: today this raises MethodError (same-typed thrower)
    # AFTER partially mutating outputs (throw happens mid-eval of series 1).
    x32 = collect(Float32, range(0.0f0, 1.0f0, 11))
    y1 = collect(Float32, range(1, 2, 11))
    y2 = collect(Float32, range(2, 3, 11))
    sitp = cubic_interp(x32, Series(y1, y2); extrap = NoExtrap())

    outputs = [fill(123.0, 3), fill(123.0, 3)]
    xq64 = [0.25, 0.75, 1.5]                    # last one OOB
    err = try
        sitp(outputs, xq64)
        nothing
    catch e
        e
    end
    @test err isa DomainError
    @test err.val == 1.5
    # throw happens at anchor build — before ANY output element is written
    @test all(v -> v === 123.0, outputs[1])
    @test all(v -> v === 123.0, outputs[2])
end

@testitem "persistent lean batch: zero-alloc after pool warmup" begin
    x = collect(range(0.0, 1.0, 11))
    y1 = collect(range(1.0, 2.0, 11))
    y2 = collect(range(2.0, 3.0, 11))
    xq = collect(range(0.05, 0.95, 16))

    run!(sitp, outputs, xq, op) = sitp(outputs, xq; deriv = op)

    for extrap in (ExtendExtrap(), ClampExtrap())
        sitp = cubic_interp(x, Series(y1, y2); extrap = extrap)
        outputs = [similar(xq) for _ in 1:2]
        for op in (EvalValue(), DerivOp(1), DerivOp(2))
            run!(sitp, outputs, xq, op)
            run!(sitp, outputs, xq, op)
            @test (@allocated run!(sitp, outputs, xq, op)) == 0
        end
    end
end
