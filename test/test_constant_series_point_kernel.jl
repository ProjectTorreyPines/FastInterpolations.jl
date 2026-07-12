# Phase 3 (Constant port) equivalence gate for the lean gather kernels.
# The lean constant path (build one shared `_AxisAnchor` carrying a baked
# `select_right`, then gather over the interpolant's layout) must match the
# CORRECT constant behavior BIT-FOR-BIT (constant is a pure gather — no FMA
# reorder, so `===` not ULP). RED first: the lean symbols do not exist yet → this
# file fails to load until the lean layer lands.
#
# REFERENCE CHOICE. Constant's four current surfaces disagree on a few edge cases
# — pre-existing bugs the lean port unifies away (see constant_series_payloads.jl):
#   • ExtendExtrap OOB: the 1D scalar / one-shot paths (and the docstring) treat
#     Extend ≡ Clamp; the persistent Series path extended via the bare kernel
#     (wrong node). Lean → Clamp; the reference sitp is built with `ref_extrap`.
#   • domain-max under a Dual query: batch/one-shot `aq.xq == x_max` (raw ==) miss
#     the Dual → `y[n-1]` for LeftSide; lean is correct (`y[n]`). Excluded from the
#     batch comparison below; pinned explicitly in the "domain-max" item.
#   • mixed-precision OOB: the persistent SCALAR `_eval_series_at_anchor!` pins
#     `output::AbstractVector{Tv}`, so it MethodErrors on a widened output; lean's
#     `out::AbstractVector` is unconstrained.
# The point/matrix lean is compared against the persistent BATCH `ref([xq])` — it
# shares the persistent OOB helpers (`_fill_*`/`_constant_extrap_boundary_value`,
# so signed-zero and mixed precision match) and lacks the scalar's widening bug.
# Raw-vector lean is checked against the one-shot path (correct Extend).
# Design: docs/design/series_lean_ports_plan.md

@testsnippet ConstantPointKernelOracle begin
    const FI = FastInterpolations
    import ForwardDiff
    const FD = ForwardDiff

    egal(g, w) = g === w || (isnan(g) && isnan(w))
    dual_egal(g::FD.Dual, w::FD.Dual) =
        egal(FD.value(g), FD.value(w)) && all(egal.(Tuple(FD.partials(g)), Tuple(FD.partials(w))))

    # Extend ≡ Clamp for constant — the reference sitp is built with this
    # normalization so the correct-surface comparison holds on the Extend edge.
    ref_extrap(e) = e isa ExtendExtrap ? ClampExtrap() : e

    # The lean scalar surface, replicated: one shared anchor + point-layout gather.
    function lean_point_eval(sitp, xq, op)
        Tg = eltype(sitp.x)
        xqp = FI._promote_coord(xq, Tg)
        Tq_w = FI._coord_eltype(typeof(xq), Tg)
        A = FI._constant_series_anchor_type(op, sitp.extrap, sitp.x, Tq_w)
        searcher = FI._resolve_search(sitp.x, xq, sitp.search_policy, nothing)
        m = FI.ConstantInterp(sitp.side)
        a = FI._build_series_anchor(m, A, sitp.x, xqp, sitp.extrap, FI._should_wrap(sitp), searcher)
        yp = FI._ensure_point_layout!(sitp)
        T_out = FI._promote_eltype(FI._select_op, Tg, eltype(sitp.y), typeof(xqp))
        out = Vector{T_out}(undef, FI.n_series(sitp))
        FI._constant_series_eval!(out, yp, a, sitp.extrap)
        return out
    end

    # lean point path ≡ correct batch reference (`ref` built with normalized extrap)
    function point_eq(sitp, ref, xq, op, cmp = egal)
        s = lean_point_eval(sitp, xq, op)
        b = ref([xq]; deriv = op)
        length(s) == length(b) || return false
        for k in eachindex(s)
            cmp(s[k], b[k][1]) || return false
        end
        return true
    end

    # The lean batch surface, replicated: one shared anchor, series-contiguous
    # matrix gather per series k (the layout the Q×K / K×Q batch loops use).
    function lean_matrix_eval(sitp, xq, op)
        Tg = eltype(sitp.x)
        xqp = FI._promote_coord(xq, Tg)
        Tq_w = FI._coord_eltype(typeof(xq), Tg)
        A = FI._constant_series_anchor_type(op, sitp.extrap, sitp.x, Tq_w)
        searcher = FI._resolve_search(sitp.x, xq, sitp.search_policy, nothing)
        m = FI.ConstantInterp(sitp.side)
        a = FI._build_series_anchor(m, A, sitp.x, xqp, sitp.extrap, FI._should_wrap(sitp), searcher)
        y = sitp.y
        return [FI._constant_series_eval(y, k, a, sitp.extrap) for k in 1:FI.n_series(sitp)]
    end

    function matrix_eq(sitp, ref, xq, op, cmp = egal)
        s = lean_matrix_eval(sitp, xq, op)
        b = ref([xq]; deriv = op)
        length(s) == length(b) || return false
        for k in eachindex(s)
            cmp(s[k], b[k][1]) || return false
        end
        return true
    end
end

@testitem "constant lean point gather ≡ correct batch (ops × extraps × sides, signed/NaN closing cell)" setup = [ConstantPointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), NaN)          # signed-zero + NaN closing cell
    y2 = collect(range(2.0, 3.0, 11))
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))
    sides = (NearestSide(), LeftSide(), RightSide())

    for side in sides, extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = constant_interp(x, Series(y1, y2); side = side, extrap = extrap)
        ref = constant_interp(x, Series(y1, y2); side = side, extrap = ref_extrap(extrap))
        dom = extrap isa InBounds ? (0.0, 0.05, 0.5, 0.94, 1.0) : (0.0, 0.05, 0.5, 0.94, 1.0, -0.5, 1.5)
        for xq in dom, op in ops
            @test point_eq(sitp, ref, xq, op)
        end
    end
    # NoExtrap: in-domain only (OOB throws)
    for side in sides
        sN = constant_interp(x, Series(y1, y2); side = side, extrap = NoExtrap())
        for xq in (0.0, 0.05, 0.5, 0.94, 1.0), op in ops
            @test point_eq(sN, sN, xq, op)
        end
    end
end

@testitem "constant lean point gather ≡ correct batch: mixed precision, Complex, Dual, Int" setup = [ConstantPointKernelOracle] begin
    sides = (NearestSide(), LeftSide(), RightSide())
    for (Tg, Tq) in ((Float64, Float32), (Float32, Float64), (Float32, Float32))
        xg = collect(Tg, range(zero(Tg), one(Tg), 11))
        ya = collect(Tg, range(1, 2, 11)); yb = collect(Tg, range(2, 3, 11))
        for side in sides, extrap in (ClampExtrap(), ExtendExtrap(), FillExtrap(Tg(9)))
            sitp = constant_interp(xg, Series(ya, yb); side = side, extrap = extrap)
            ref = constant_interp(xg, Series(ya, yb); side = side, extrap = ref_extrap(extrap))
            for xq in Tq.((0.37, 0.5, 1.0, -0.5, 1.5)), op in (EvalValue(), DerivOp(1))
                @test point_eq(sitp, ref, xq, op)
            end
        end
    end

    # Int grid + Int query — the "coordinate must not over-float" carrier case.
    xi = collect(0:10)
    yi1 = collect(10:20); yi2 = collect(20:30)
    for side in sides
        sI = constant_interp(xi, Series(yi1, yi2); side = side, extrap = ClampExtrap())
        for xq in (3, 5, 10, -2, 12), op in (EvalValue(), DerivOp(1))
            @test point_eq(sI, sI, xq, op)
        end
    end

    # Complex values
    x = collect(range(0.0, 1.0, 11))
    yc = complex.(collect(range(1.0, 2.0, 11)), collect(range(-1.0, 1.0, 11)))
    for side in sides, extrap in (ClampExtrap(), ExtendExtrap())
        sitp = constant_interp(x, Series(yc, 2 .* yc); side = side, extrap = extrap)
        ref = constant_interp(x, Series(yc, 2 .* yc); side = side, extrap = ref_extrap(extrap))
        for xq in (0.37, 0.5, 1.0, -0.5, 1.5), op in (EvalValue(), DerivOp(1))
            @test point_eq(sitp, ref, xq, op)
        end
    end

    # Dual query (in-domain + Clamp OOB). Exclude the exact domain-max (xq=1.0):
    # the batch reference mis-handles Dual there — pinned in the next item.
    z1 = collect(range(1.0, 2.0, 11)); z2 = collect(range(2.0, 3.0, 11))
    for side in sides
        sCd = constant_interp(x, Series(z1, z2); side = side, extrap = ClampExtrap())
        for xq in (0.37, 0.5, -0.5, 1.5), op in (EvalValue(), DerivOp(1))
            @test point_eq(sCd, sCd, FD.Dual(xq, 1.0), op, dual_egal)
        end
    end
end

@testitem "constant lean domain-max = y[n] for every side (Float + Dual)" setup = [ConstantPointKernelOracle] begin
    # At xq == last(x) every side collapses to the closing grid value y[n] (LeftSide's
    # floor is overridden). Pinned directly (the batch/one-shot references get the
    # Dual case wrong; the lean is correct for both Float and Dual).
    x = collect(range(0.0, 1.0, 11))
    y1 = collect(range(1.0, 2.0, 11)); y2 = collect(range(2.0, 3.0, 11))   # y1[11]=2.0, y2[11]=3.0
    for side in (NearestSide(), LeftSide(), RightSide()), extrap in (ClampExtrap(), ExtendExtrap(), InBounds(), WrapExtrap())
        sitp = constant_interp(x, Series(y1, y2); side = side, extrap = extrap)
        vF = lean_point_eval(sitp, 1.0, EvalValue())
        @test vF[1] == 2.0 && vF[2] == 3.0
        vD = lean_point_eval(sitp, FD.Dual(1.0, 1.0), EvalValue())
        @test FD.value(vD[1]) == 2.0 && FD.value(vD[2]) == 3.0
        @test FD.partials(vD[1])[1] == 0.0 && FD.partials(vD[2])[1] == 0.0
        # derivative is zero everywhere (incl. the closing cell)
        dF = lean_point_eval(sitp, 1.0, DerivOp(1))
        @test dF[1] == 0.0 && dF[2] == 0.0
    end
end

@testitem "constant lean MATRIX gather ≡ correct batch (batch layout: ops × extraps × sides)" setup = [ConstantPointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), NaN)
    y2 = collect(range(2.0, 3.0, 11))
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))
    sides = (NearestSide(), LeftSide(), RightSide())
    for side in sides, extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = constant_interp(x, Series(y1, y2); side = side, extrap = extrap)
        ref = constant_interp(x, Series(y1, y2); side = side, extrap = ref_extrap(extrap))
        dom = extrap isa InBounds ? (0.0, 0.05, 0.5, 0.94, 1.0) : (0.0, 0.05, 0.5, 0.94, 1.0, -0.5, 1.5)
        for xq in dom, op in ops
            @test matrix_eq(sitp, ref, xq, op)
        end
    end
    # mixed precision + Int grid (carrier width)
    for (Tg, Tq) in ((Float64, Float32), (Float32, Float64))
        xg = collect(Tg, range(zero(Tg), one(Tg), 11))
        ya = collect(Tg, range(1, 2, 11)); yb = collect(Tg, range(2, 3, 11))
        for side in sides, extrap in (ClampExtrap(), ExtendExtrap(), FillExtrap(Tg(9)))
            sM = constant_interp(xg, Series(ya, yb); side = side, extrap = extrap)
            ref = constant_interp(xg, Series(ya, yb); side = side, extrap = ref_extrap(extrap))
            for xq in Tq.((0.37, 0.5, 1.0, -0.5, 1.5)), op in (EvalValue(), DerivOp(1))
                @test matrix_eq(sM, ref, xq, op)
            end
        end
    end
end

@testitem "constant lean RAW-VECTOR gather ≡ current one-shot (ops × extraps × sides)" setup = [ConstantPointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), NaN)
    y2 = collect(range(2.0, 3.0, 11))
    vecs = (y1, y2)
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))
    sides = (NearestSide(), LeftSide(), RightSide())
    # The one-shot path already routes Extend → Clamp (matches lean); Float
    # queries only, so its Dual-domain-max quirk is never exercised here.
    for side in sides, extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        dom = extrap isa InBounds ? (0.0, 0.05, 0.5, 0.94, 1.0) : (0.0, 0.05, 0.5, 0.94, 1.0, -0.5, 1.5)
        for xq in dom, op in ops
            Tg = eltype(x)
            xqp = FI._promote_coord(xq, Tg)
            Tq_w = FI._coord_eltype(typeof(xq), Tg)
            A = FI._constant_series_anchor_type(op, extrap, x, Tq_w)
            searcher = FI._resolve_search(x, xq, AutoSearch(), nothing)
            m = FI.ConstantInterp(side)
            a = FI._build_series_anchor(m, A, x, xqp, extrap, extrap isa WrapExtrap, searcher)
            lean = [FI._constant_series_eval(v, a, extrap) for v in vecs]
            cur = constant_interp(x, Series(y1, y2), xq; side = side, extrap = extrap, deriv = op)
            for k in 1:2
                @test egal(lean[k], cur[k])
            end
        end
    end
end
