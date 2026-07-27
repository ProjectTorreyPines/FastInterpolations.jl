# Equivalence gate for the lean payload-anchor batch surfaces.
# The reference is the pre-migration full-anchor series formula, ported inline in
# the snippet (`_ref_with_extrap`) after the src kernels were removed — full
# anchors are still built via the retained `_fill_anchors!` (used by the adjoints).
# The lean batch must match it bit-for-bit on 1.12+, within a small ULP budget on
# codegen (LTS) that schedules FMA contraction differently (see `assert_egal`).
# Design: docs/design/cubic_series_payload_anchor.md §8

@testsnippet LeanBatchOracles begin
    const FI = FastInterpolations

    # Independent reference for the lean batch: full anchors (still built via the
    # retained `_fill_anchors!`, used by the adjoints) fed the pre-migration series
    # formula, ported here self-contained after the src kernels were removed. Keeps
    # the exact series OOB formula (`_constant_extrap_boundary_value`, `val*one(xq)`)
    # so the signed-zero / NaN pins still bite against the shipped lean path.
    function _ref_anchored(y, z, k, aq, op)
        idx = aq.idx
        if op isa EvalValue || op isa FI.EvalDeriv1
            wyL, wyR, wzL, wzR = op isa EvalValue ? aq.w0 : aq.w1
            return muladd(wyR, y[idx + 1, k], muladd(wyL, y[idx, k], muladd(wzR, z[idx + 1, k], wzL * z[idx, k])))
        elseif op isa FI.EvalDeriv2 || op isa FI.EvalDeriv3
            wzL, wzR = op isa FI.EvalDeriv2 ? aq.w2 : aq.w3
            return muladd(wzR, z[idx + 1, k], wzL * z[idx, k])
        else
            return 0 * y[idx, k]                       # DerivOp{N≥4}
        end
    end
    function _ref_with_extrap(y, z, n_pts, x_min, x_max, k, aq, extrap, op)
        (aq.state == FI.IN_DOMAIN || extrap isa ExtendExtrap || extrap isa WrapExtrap) &&
            return _ref_anchored(y, z, k, aq, op)
        # Deriv OOB zeros carry the grid's reciprocal-spacing unit (identity for
        # Real grids) — mirror the src oneunit witness off the grid-typed `x_min`.
        extrap isa FI._ClampOrFill &&
            return FI._constant_extrap_boundary_value(
            y, aq.state, n_pts, k, op, extrap, typeof(aq.xq),
            FI._deriv_oneunit(oneunit(x_min), op)
        )
        return FI._throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
    function persistent_oracle(sitp, xq::AbstractVector, op)
        Tg = eltype(sitp.cache.x)
        Tq_w = FI._coord_eltype(eltype(xq), Tg)
        aq_vec = Vector{FI._CubicAdjointAnchor{Tg, Tq_w, FI._interval_type(sitp.cache.x)}}(undef, length(xq))
        searcher = FI._resolve_search(sitp.cache.x, xq, sitp.search_policy, nothing)
        FI._fill_anchors!(aq_vec, sitp.cache.x, xq, Val(:cubic), FI._should_wrap(sitp), searcher)
        n_pts = FI.n_points(sitp)
        x_min, x_max = Tg(first(sitp.cache.x)), Tg(last(sitp.cache.x))
        K = FI.n_series(sitp)
        return [
            [
                    _ref_with_extrap(sitp.y, sitp.z, n_pts, x_min, x_max, k, aq_vec[j], sitp.extrap, op)
                    for j in eachindex(xq)
                ]
                for k in 1:K
        ]
    end

    # Scalar equivalence compare. The lean path and its full-anchor reference are
    # algebraically identical, so on 1.12+ they match bit-for-bit (`===`, which
    # also makes `NaN === NaN` and the signed-zero OOB pins bite). On some codegen
    # (LTS) FMA contraction is scheduled differently between the two call sites,
    # perturbing cancellation-heavy derivatives by a few ULP (observed ≤4). Accept
    # a small ULP budget for nonzero finite pairs only — zeros and non-finites
    # still require egal, so signed-zero / NaN-propagation regressions are caught.
    function egal_or_ulp(g, w)
        g === w && return true
        return isfinite(g) && isfinite(w) && !iszero(g) && !iszero(w) &&
            abs(g - w) <= 16 * eps(max(abs(g), abs(w)))
    end

    # Elementwise wrapper over a Vector{Vector} pair; returns the first offending
    # (k, j, got, want) or nothing.
    function assert_egal(got, want)
        for k in eachindex(want), j in eachindex(want[k])
            egal_or_ulp(got[k][j], want[k][j]) || return (k, j, got[k][j], want[k][j])
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

@testitem "one-shot lean batch === scalar one-shot path (op × extrap × precision)" setup = [LeanBatchOracles] begin
    # Scalar and vector one-shot now share the lean anchor build + raw-vector
    # adapter; this pins their agreement (scalar/vector symmetry contract).
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
            (InBounds(), xq_in),
        )
        for op in (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3), DerivOp(5))
            got = cubic_interp(x, Series(y1, y2), xq; extrap = extrap, deriv = op)
            for j in eachindex(xq)
                want = cubic_interp(x, Series(y1, y2), xq[j]; extrap = extrap, deriv = op)
                @test egal_or_ulp(got[1][j], want[1])
                @test egal_or_ulp(got[2][j], want[2])
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
            @test egal_or_ulp(got[1][j], want[1])
            @test egal_or_ulp(got[2][j], want[2])
        end
    end
end

@testitem "one-shot lean batch: exclusive-periodic seam === scalar path" setup = [LeanBatchOracles] begin
    # exclusive grid: last point omitted; queries past x[end] wrap through the seam
    x = collect(range(0.0, 0.9, 10))            # period 1.0, seam cell (x[10], virtual 1.0)
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    xqs = [0.02, 0.55, 0.93, 0.99]              # last two live in the seam cell

    # ops span the deriv2 (NTuple{2}), deriv3 and ≥4 (zero) payload arms so each
    # bare-payload type is exercised through the periodic-seam builder.
    for bc in (PeriodicBC(endpoint = :exclusive, period = 1.0),)
        for op in (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3), DerivOp(5))
            got = cubic_interp(x, Series(y1, y2), xqs; bc = bc, deriv = op)
            for j in eachindex(xqs)
                want = cubic_interp(x, Series(y1, y2), xqs[j]; bc = bc, deriv = op)
                @test egal_or_ulp(got[1][j], want[1])
                @test egal_or_ulp(got[2][j], want[2])
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

@testitem "persistent lean batch: zero-alloc after pool warmup" setup = [AllocConstants] begin
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
            @test (@allocated run!(sitp, outputs, xq, op)) <= ALLOC_THRESHOLD
        end
    end
end
