# Phase 4 (Quadratic port) equivalence gate for the lean coefficient kernels.
# The lean quadratic path (build one shared `_AxisAnchor` baking `dL`, then the
# op-threaded kernel reading the stored y/a/d coefficients per series) must match
# the CURRENT quadratic path up to FMA-scheduling ULP (muladd reorders). RED
# first: the lean symbols do not exist yet.
#
# Quadratic is coefficient-based (like cubic) but the coefficients y/a/d are
# precomputed at build (like linear reads stored y): the anchor bakes only the
# op-independent `dL`, and the op-threaded kernel picks the formula. ExtendExtrap
# extends the boundary polynomial (bare kernel) — NO Clamp normalization (unlike
# constant). The point/matrix lean is checked against the persistent BATCH
# reference; the raw-vector lean against the one-shot path (each shares its OOB
# convention + its deriv2/deriv3 formula with the matching lean surface).
# Design: docs/design/series_lean_ports_plan.md

@testsnippet QuadraticPointKernelOracle begin
    const FI = FastInterpolations
    import ForwardDiff
    const FD = ForwardDiff

    # A zero result's sign bit is a don't-care where the current surfaces disagree:
    # the persistent point/matrix path preserves it (`_fill_*`, `* one`, matching
    # Constant/Linear), while the batch/one-shot path normalizes it via
    # `_eval_extrapolation`'s `+ 0` (e.g. Clamp-OOB deriv of a tiny-negative
    # boundary → `-0.0` vs `+0.0`). The lean unifies to preserve; treat both zeros
    # as equal so the gate pins the value, not the historically-divergent sign.
    egal_or_ulp(g, w) = g === w || (iszero(g) && iszero(w)) || (
        isfinite(g) && isfinite(w) && !iszero(g) && !iszero(w) &&
            abs(g - w) <= 16 * eps(max(abs(g), abs(w)))
    )
    dual_match(g::FD.Dual, w::FD.Dual) =
        egal_or_ulp(FD.value(g), FD.value(w)) && all(egal_or_ulp.(Tuple(FD.partials(g)), Tuple(FD.partials(w))))

    # lean point surface: one shared anchor + point-layout op-threaded kernel.
    function lean_point_eval(sitp, xq, op)
        Tg = eltype(sitp.x)
        xqp = FI._promote_coord(xq, Tg)
        Tq_w = FI._coord_eltype(typeof(xq), Tg)
        A = FI._quadratic_series_anchor_type(sitp.extrap, sitp.x, Tq_w)
        searcher = FI._resolve_search(sitp.x, xq, sitp.search_policy, nothing)
        a = FI._build_series_anchor(FI.QuadraticInterp(), A, sitp.x, xqp, sitp.extrap, FI._should_wrap(sitp), searcher)
        yp, ap, dp = FI._ensure_point_layout!(sitp)
        T_out = FI._promote_eltype(FI._interp_op, Tg, eltype(sitp.y), typeof(xqp))
        out = Vector{T_out}(undef, FI.n_series(sitp))
        FI._quadratic_series_eval!(out, yp, ap, dp, a, op, sitp.extrap)
        return out
    end

    function point_eq(sitp, xq, op, cmp = egal_or_ulp)
        s = lean_point_eval(sitp, xq, op)
        b = sitp([xq]; deriv = op)
        length(s) == length(b) || return false
        for k in eachindex(s)
            cmp(s[k], b[k][1]) || return false
        end
        return true
    end

    # lean matrix surface: series-contiguous per-series kernel (batch layout).
    function lean_matrix_eval(sitp, xq, op)
        Tg = eltype(sitp.x)
        xqp = FI._promote_coord(xq, Tg)
        Tq_w = FI._coord_eltype(typeof(xq), Tg)
        A = FI._quadratic_series_anchor_type(sitp.extrap, sitp.x, Tq_w)
        searcher = FI._resolve_search(sitp.x, xq, sitp.search_policy, nothing)
        a = FI._build_series_anchor(FI.QuadraticInterp(), A, sitp.x, xqp, sitp.extrap, FI._should_wrap(sitp), searcher)
        return [FI._quadratic_series_eval(sitp.y, sitp.a, sitp.d, k, a, op, sitp.extrap) for k in 1:FI.n_series(sitp)]
    end

    function matrix_eq(sitp, xq, op, cmp = egal_or_ulp)
        s = lean_matrix_eval(sitp, xq, op)
        b = sitp([xq]; deriv = op)
        length(s) == length(b) || return false
        for k in eachindex(s)
            cmp(s[k], b[k][1]) || return false
        end
        return true
    end
end

@testitem "quadratic lean point kernel ≡ current batch (ops × extraps)" setup = [QuadraticPointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = sin.(2π .* x); y2 = collect(range(2.0, 3.0, 11))
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))
    for extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = quadratic_interp(x, Series(y1, y2); extrap = extrap)
        dom = extrap isa InBounds ? (0.05, 0.37, 0.94) : (0.05, 0.37, 0.94, -0.5, 1.5)
        for xq in dom, op in ops
            @test point_eq(sitp, xq, op)
        end
    end
    sN = quadratic_interp(x, Series(y1, y2); extrap = NoExtrap())
    for xq in (0.05, 0.37, 0.94), op in ops
        @test point_eq(sN, xq, op)
    end
end

@testitem "quadratic lean point kernel ≡ current: mixed precision, Complex, Dual" setup = [QuadraticPointKernelOracle] begin
    for (Tg, Tq) in ((Float64, Float32), (Float32, Float64), (Float32, Float32))
        xg = collect(Tg, range(zero(Tg), one(Tg), 11))
        ya = collect(Tg, sin.(2π .* range(0, 1, 11))); yb = collect(Tg, range(2, 3, 11))
        for extrap in (ClampExtrap(), ExtendExtrap(), FillExtrap(Tg(9)))
            sitp = quadratic_interp(xg, Series(ya, yb); extrap = extrap)
            for xq in Tq.((0.37, -0.5, 1.5)), op in (EvalValue(), DerivOp(1), DerivOp(2))
                @test point_eq(sitp, xq, op)
            end
        end
    end
    # Complex values
    x = collect(range(0.0, 1.0, 11))
    yc = complex.(sin.(2π .* x), collect(range(-1.0, 1.0, 11)))
    for extrap in (ClampExtrap(), ExtendExtrap())
        sitp = quadratic_interp(x, Series(yc, 2 .* yc); extrap = extrap)
        for xq in (0.37, -0.5, 1.5), op in (EvalValue(), DerivOp(1))
            @test point_eq(sitp, xq, op)
        end
    end
    # Dual query (in-domain + Clamp OOB)
    z1 = sin.(2π .* x); z2 = collect(range(2.0, 3.0, 11))
    sCd = quadratic_interp(x, Series(z1, z2); extrap = ClampExtrap())
    for xq in (0.37, -0.5, 1.5), op in (EvalValue(), DerivOp(1), DerivOp(2))
        @test point_eq(sCd, FD.Dual(xq, 1.0), op, dual_match)
    end
end

@testitem "quadratic lean MATRIX kernel ≡ current batch (ops × extraps)" setup = [QuadraticPointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = sin.(2π .* x); y2 = collect(range(2.0, 3.0, 11))
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))
    for extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = quadratic_interp(x, Series(y1, y2); extrap = extrap)
        dom = extrap isa InBounds ? (0.05, 0.37, 0.94) : (0.05, 0.37, 0.94, -0.5, 1.5)
        for xq in dom, op in ops
            @test matrix_eq(sitp, xq, op)
        end
    end
    for (Tg, Tq) in ((Float64, Float32), (Float32, Float64))
        xg = collect(Tg, range(zero(Tg), one(Tg), 11))
        ya = collect(Tg, sin.(2π .* range(0, 1, 11))); yb = collect(Tg, range(2, 3, 11))
        for extrap in (ClampExtrap(), ExtendExtrap(), FillExtrap(Tg(9)))
            sM = quadratic_interp(xg, Series(ya, yb); extrap = extrap)
            for xq in Tq.((0.37, -0.5, 1.5)), op in (EvalValue(), DerivOp(1), DerivOp(2))
                @test matrix_eq(sM, xq, op)
            end
        end
    end
end

@testitem "quadratic lean RAW-VECTOR kernel ≡ current one-shot (ops × extraps)" setup = [QuadraticPointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = sin.(2π .* x); y2 = collect(range(2.0, 3.0, 11))
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))
    # Persistent sitp shares the solver+bc with the one-shot, so its stored
    # y/a/d columns equal the one-shot's solved coefficients — use them to drive
    # the lean raw-vector kernel and compare against the one-shot result.
    for extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = quadratic_interp(x, Series(y1, y2); extrap = extrap)
        dom = extrap isa InBounds ? (0.05, 0.37, 0.94) : (0.05, 0.37, 0.94, -0.5, 1.5)
        for xq in dom, op in ops
            Tg = eltype(x)
            xqp = FI._promote_coord(xq, Tg)
            Tq_w = FI._coord_eltype(typeof(xq), Tg)
            A = FI._quadratic_series_anchor_type(extrap, x, Tq_w)
            searcher = FI._resolve_search(x, xq, AutoSearch(), nothing)
            a = FI._build_series_anchor(FI.QuadraticInterp(), A, x, xqp, extrap, extrap isa WrapExtrap, searcher)
            lean = [FI._quadratic_series_eval(sitp.y[:, k], sitp.a[:, k], sitp.d[:, k], a, op, extrap) for k in 1:2]
            cur = quadratic_interp(x, Series(y1, y2), xq; extrap = extrap, deriv = op)
            for k in 1:2
                @test egal_or_ulp(lean[k], cur[k])
            end
        end
    end
end
