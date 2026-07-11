# Phase 2 (Linear port) equivalence gate for the lean point-contiguous scalar kernel.
# The lean linear path (build one shared `_AxisAnchor`, then `_linear_series_eval!`
# over the interpolant's point-contiguous layout) must match the CURRENT linear
# path (`sitp([xq])`, still the old `_LinearAnchoredQuery` chain) up to FMA-
# scheduling ULP. RED first: `_linear_series_anchor_type` / `_linear_series_eval!`
# do not exist yet → this file fails to load until the lean layer lands.
# Design: docs/design/series_lean_ports_plan.md

@testsnippet LinearPointKernelOracle begin
    const FI = FastInterpolations
    import ForwardDiff
    const FD = ForwardDiff

    egal_or_ulp(g, w) = g === w || (
        isfinite(g) && isfinite(w) && !iszero(g) && !iszero(w) &&
            abs(g - w) <= 16 * eps(max(abs(g), abs(w)))
    )
    dual_match(g::FD.Dual, w::FD.Dual) =
        egal_or_ulp(FD.value(g), FD.value(w)) && all(egal_or_ulp.(Tuple(FD.partials(g)), Tuple(FD.partials(w))))

    # The lean scalar surface, replicated: one shared anchor + point-layout eval.
    function lean_point_eval(sitp, xq, op)
        Tg = eltype(sitp.x)
        xqp = FI._promote_coord(xq, Tg)
        Tq_w = FI._coord_eltype(typeof(xq), Tg)
        A = FI._linear_series_anchor_type(op, sitp.extrap, sitp.x, Tq_w)
        searcher = FI._resolve_search(sitp.x, xq, sitp.search_policy, nothing)
        a = FI._build_series_anchor(FI.LinearInterp(), A, sitp.x, xqp, sitp.extrap, FI._should_wrap(sitp), searcher)
        yp = FI._ensure_point_layout!(sitp)
        T_out = FI._promote_eltype(FI._interp_op, Tg, eltype(sitp.y), typeof(xqp))
        out = Vector{T_out}(undef, FI.n_series(sitp))
        FI._linear_series_eval!(out, yp, a, sitp.extrap)
        return out
    end

    # lean point path ≡ current path, elementwise
    function point_eq_current(sitp, xq, op, cmp = egal_or_ulp)
        s = lean_point_eval(sitp, xq, op)
        b = sitp([xq]; deriv = op)
        length(s) == length(b) || return false
        for k in eachindex(s)
            cmp(s[k], b[k][1]) || return false
        end
        return true
    end

    # The lean batch surface, replicated: fill one shared anchor, then the
    # series-contiguous matrix kernel per series k (the layout the batch loops use).
    function lean_matrix_eval(sitp, xq, op)
        Tg = eltype(sitp.x)
        xqp = FI._promote_coord(xq, Tg)
        Tq_w = FI._coord_eltype(typeof(xq), Tg)
        A = FI._linear_series_anchor_type(op, sitp.extrap, sitp.x, Tq_w)
        searcher = FI._resolve_search(sitp.x, xq, sitp.search_policy, nothing)
        a = FI._build_series_anchor(FI.LinearInterp(), A, sitp.x, xqp, sitp.extrap, FI._should_wrap(sitp), searcher)
        y = sitp.y
        return [FI._linear_series_eval(y, k, a, sitp.extrap) for k in 1:FI.n_series(sitp)]
    end

    # lean matrix (series-contiguous) path ≡ current path, elementwise
    function matrix_eq_current(sitp, xq, op, cmp = egal_or_ulp)
        s = lean_matrix_eval(sitp, xq, op)
        b = sitp([xq]; deriv = op)
        length(s) == length(b) || return false
        for k in eachindex(s)
            cmp(s[k], b[k][1]) || return false
        end
        return true
    end
end

@testitem "linear lean point kernel ≡ current (ops × extraps, incl. signed zero)" setup = [LinearPointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), 2.0)          # signed-zero boundary
    y2 = collect(range(2.0, 3.0, 11))
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))

    for extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = linear_interp(x, Series(y1, y2); extrap = extrap)
        dom = extrap isa InBounds ? (0.05, 0.37, 0.94) : (0.05, 0.37, 0.94, -0.5, 1.5)
        for xq in dom, op in ops
            @test point_eq_current(sitp, xq, op)
        end
    end
    # NoExtrap: in-domain only (OOB throws)
    sN = linear_interp(x, Series(y1, y2); extrap = NoExtrap())
    for xq in (0.05, 0.37, 0.94), op in ops
        @test point_eq_current(sN, xq, op)
    end
end

@testitem "linear lean point kernel ≡ current: mixed precision, Complex, Dual" setup = [LinearPointKernelOracle] begin
    for (Tg, Tq) in ((Float64, Float32), (Float32, Float64), (Float32, Float32))
        xg = collect(Tg, range(zero(Tg), one(Tg), 11))
        ya = collect(Tg, range(1, 2, 11)); yb = collect(Tg, range(2, 3, 11))
        for extrap in (ClampExtrap(), ExtendExtrap(), FillExtrap(Tg(9)))
            sitp = linear_interp(xg, Series(ya, yb); extrap = extrap)
            for xq in Tq.((0.37, -0.5, 1.5)), op in (EvalValue(), DerivOp(1))
                @test point_eq_current(sitp, xq, op)
            end
        end
    end

    # Complex values
    x = collect(range(0.0, 1.0, 11))
    yc = complex.(collect(range(1.0, 2.0, 11)), collect(range(-1.0, 1.0, 11)))
    for extrap in (ClampExtrap(), ExtendExtrap())
        sitp = linear_interp(x, Series(yc, 2 .* yc); extrap = extrap)
        for xq in (0.37, -0.5, 1.5), op in (EvalValue(), DerivOp(1))
            @test point_eq_current(sitp, xq, op)
        end
    end

    # Dual query (in-domain + Clamp OOB)
    z1 = collect(range(1.0, 2.0, 11)); z2 = collect(range(2.0, 3.0, 11))
    sCd = linear_interp(x, Series(z1, z2); extrap = ClampExtrap())
    for xq in (0.37, -0.5, 1.5), op in (EvalValue(), DerivOp(1))
        @test point_eq_current(sCd, FD.Dual(xq, 1.0), op, dual_match)
    end
end

@testitem "linear lean MATRIX kernel ≡ current (batch layout: ops × extraps, incl. signed zero)" setup = [LinearPointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), 2.0)
    y2 = collect(range(2.0, 3.0, 11))
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))
    for extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = linear_interp(x, Series(y1, y2); extrap = extrap)
        dom = extrap isa InBounds ? (0.05, 0.37, 0.94) : (0.05, 0.37, 0.94, -0.5, 1.5)
        for xq in dom, op in ops
            @test matrix_eq_current(sitp, xq, op)
        end
    end
    # mixed precision (F32 grid + F64 query) — the deriv1 inv_h-width case
    for (Tg, Tq) in ((Float64, Float32), (Float32, Float64), (Float32, Float32))
        xg = collect(Tg, range(zero(Tg), one(Tg), 11))
        ya = collect(Tg, range(1, 2, 11)); yb = collect(Tg, range(2, 3, 11))
        for extrap in (ClampExtrap(), ExtendExtrap(), FillExtrap(Tg(9)))
            sM = linear_interp(xg, Series(ya, yb); extrap = extrap)
            for xq in Tq.((0.37, -0.5, 1.5)), op in (EvalValue(), DerivOp(1))
                @test matrix_eq_current(sM, xq, op)
            end
        end
    end
end

@testitem "linear lean RAW-VECTOR kernel ≡ current one-shot (ops × extraps, signed zero)" setup = [LinearPointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), 2.0)
    y2 = collect(range(2.0, 3.0, 11))
    vecs = (y1, y2)
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))
    for extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        dom = extrap isa InBounds ? (0.05, 0.37, 0.94) : (0.05, 0.37, 0.94, -0.5, 1.5)
        for xq in dom, op in ops
            Tg = eltype(x)
            xqp = FI._promote_coord(xq, Tg)
            Tq_w = FI._coord_eltype(typeof(xq), Tg)
            A = FI._linear_series_anchor_type(op, extrap, x, Tq_w)
            searcher = FI._resolve_search(x, xq, AutoSearch(), nothing)
            a = FI._build_series_anchor(FI.LinearInterp(), A, x, xqp, extrap, extrap isa WrapExtrap, searcher)
            lean = [FI._linear_series_eval(v, a, extrap) for v in vecs]
            cur = linear_interp(x, Series(y1, y2), xq; extrap = extrap, deriv = op)
            for k in 1:2
                @test egal_or_ulp(lean[k], cur[k])
            end
        end
    end
end
