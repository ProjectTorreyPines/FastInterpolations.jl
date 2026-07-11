# Phase 2 equivalence gate for the lean point-contiguous scalar kernel.
# The lean point path (build one shared _AxisAnchor, then `_cubic_series_eval!`
# over the interpolant's point-contiguous layout) matches the already-lean BATCH
# path up to FMA-scheduling ULP — the two kernels share weights/indices and differ
# only in load pattern, so the point-vs-series load order can reorder FMA
# contraction (hence `egal_or_ulp`, egal for zeros/non-finites, ≤16 ULP for finite
# nonzeros; it is NOT universally bit-identical near heavy cancellation). This
# validates the kernel BEFORE Phase 3 wires the scalar entries to it. DerivOp≥4's
# per-k zero arm matches batch exactly (resolving the master scalar(-0.0)/batch(+0.0)
# divergence pinned sign-agnostically in Phase 1).

@testsnippet PointKernelOracle begin
    const FI = FastInterpolations
    # Owned `const` alias (not `import`): only owned snippet bindings propagate
    # into testitem modules under ReTestItems parallel workers; a bare `import`
    # here leaves `FD.Dual` undefined in the testitem body (CI-only, hidden
    # locally by the non-parallel splice). Aliasing the module (not the names)
    # avoids shadowing the common `value`/`partials` identifiers.
    import ForwardDiff
    const FD = ForwardDiff

    egal_or_ulp(g, w) = g === w || (
        isfinite(g) && isfinite(w) && !iszero(g) && !iszero(w) &&
            abs(g - w) <= 16 * eps(max(abs(g), abs(w)))
    )
    dual_match(g::FD.Dual, w::FD.Dual) =
        egal_or_ulp(FD.value(g), FD.value(w)) && all(egal_or_ulp.(Tuple(FD.partials(g)), Tuple(FD.partials(w))))

    # Replicates exactly what Phase 3 will place in the scalar entries: one shared
    # anchor (built from the statically-typed extrap) + the point-layout eval.
    function lean_point_eval(sitp, xq, op)
        Tg = eltype(sitp.cache.x)
        xqp = FI._promote_coord(xq, Tg)
        Tq_w = FI._coord_eltype(typeof(xq), Tg)
        A = FI._cubic_series_anchor_type(op, sitp.extrap, sitp.cache.x, Tq_w)
        searcher = FI._resolve_search(sitp.cache.x, xq, sitp.search_policy, nothing)
        a = FI._build_series_anchor(A, sitp.cache.x, xqp, sitp.extrap, FI._should_wrap(sitp), searcher)
        yp, zp = FI._ensure_point_layout!(sitp)
        T_out = FI._promote_eltype(FI._interp_op, Tg, eltype(sitp.y), typeof(xqp))
        out = Vector{T_out}(undef, FI.n_series(sitp))
        FI._cubic_series_eval!(out, yp, zp, a, sitp.extrap)
        return out
    end

    # lean point path ≡ batch path, elementwise
    function point_eq_batch(sitp, xq, op, cmp = egal_or_ulp)
        s = lean_point_eval(sitp, xq, op)
        b = sitp([xq]; deriv = op)
        length(s) == length(b) || return false
        for k in eachindex(s)
            cmp(s[k], b[k][1]) || return false
        end
        return true
    end
end

@testitem "lean point kernel ≡ batch (ops × extraps, incl. DerivOp≥4 signed zero)" setup = [PointKernelOracle] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), 2.0)          # signed-zero boundary
    y2 = collect(range(2.0, 3.0, 11))
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3), DerivOp(4), DerivOp(5))

    for extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = cubic_interp(x, Series(y1, y2); extrap = extrap)
        dom = extrap isa InBounds ? (0.05, 0.37, 0.94) : (0.05, 0.37, 0.94, -0.5, 1.5)
        for xq in dom, op in ops
            @test point_eq_batch(sitp, xq, op)
        end
    end
    # NoExtrap: in-domain only (OOB throws — covered in scalar_unify RED pins)
    sN = cubic_interp(x, Series(y1, y2); extrap = NoExtrap())
    for xq in (0.05, 0.37, 0.94), op in ops
        @test point_eq_batch(sN, xq, op)
    end
end

@testitem "lean point kernel ≡ batch: mixed precision, Complex, Dual" setup = [PointKernelOracle] begin
    # mixed precision (this is the OOB case that MethodError'd on the old scalar path)
    for (Tg, Tq) in ((Float64, Float32), (Float32, Float64), (Float32, Float32))
        xg = collect(Tg, range(zero(Tg), one(Tg), 11))
        ya = collect(Tg, range(1, 2, 11)); yb = collect(Tg, range(2, 3, 11))
        for extrap in (ClampExtrap(), ExtendExtrap(), FillExtrap(Tg(9)))
            sitp = cubic_interp(xg, Series(ya, yb); extrap = extrap)
            for xq in Tq.((0.37, -0.5, 1.5)), op in (EvalValue(), DerivOp(1), DerivOp(2))
                @test point_eq_batch(sitp, xq, op)
            end
        end
    end

    # Complex values
    x = collect(range(0.0, 1.0, 11))
    yc = complex.(collect(range(1.0, 2.0, 11)), collect(range(-1.0, 1.0, 11)))
    for extrap in (ClampExtrap(), ExtendExtrap())
        sitp = cubic_interp(x, Series(yc, 2 .* yc); extrap = extrap)
        for xq in (0.37, -0.5, 1.5), op in (EvalValue(), DerivOp(1))
            @test point_eq_batch(sitp, xq, op)
        end
    end

    # Dual query (in-domain + Clamp OOB — the case that MethodError'd on old scalar)
    z1 = collect(range(1.0, 2.0, 11)); z2 = collect(range(2.0, 3.0, 11))
    sCd = cubic_interp(x, Series(z1, z2); extrap = ClampExtrap())
    for xq in (0.37, -0.5, 1.5), op in (EvalValue(), DerivOp(1))
        @test point_eq_batch(sCd, FD.Dual(xq, 1.0), op, dual_match)
    end
end
