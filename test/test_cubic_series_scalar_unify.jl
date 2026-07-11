# Oracle + RED pins for unifying the cubic Series SCALAR surfaces onto the lean
# `_AxisAnchor` payload anchors (plan: docs/plans/PLAN_cubic_series_payload_anchor_scalar_cleanup.md).
#
# Oracle = the already-validated BATCH path (PR #187): a scalar query at `xq`,
# series k, must equal the length-1 batch query at `xq`, series k. These tests
# pass against the CURRENT (unwired) scalar path — validating the symmetry
# relation — and must still hold once the scalar path is rewired to the lean
# point-contiguous kernel. The RED pins below capture the mixed-precision / Dual
# OOB bugs the rewiring fixes; they are `@test_broken` until Phase 3 flips them.

@testsnippet ScalarUnifyOracles begin
    const FI = FastInterpolations
    import ForwardDiff: Dual, value, partials

    # egal-or-tiny-ULP (LTS FMA-scheduling slack), egal for zeros/non-finites so
    # signed-zero / NaN still bite. Mirrors test_cubic_series_lean_batch.jl.
    egal_or_ulp(g, w) = g === w || (
        isfinite(g) && isfinite(w) && !iszero(g) && !iszero(w) &&
            abs(g - w) <= 16 * eps(max(abs(g), abs(w)))
    )

    # Dual-aware element compare: value + each partial through egal_or_ulp.
    dual_match(g::Dual, w::Dual) =
        egal_or_ulp(value(g), value(w)) &&
        all(egal_or_ulp.(Tuple(partials(g)), Tuple(partials(w))))

    # scalar (out-of-place) ≡ length-1 batch, elementwise across the K series.
    function scalar_eq_batch(sitp, xq, op, cmp = egal_or_ulp)
        s = sitp(xq; deriv = op)
        b = sitp([xq]; deriv = op)
        length(s) == length(b) || return false
        for k in eachindex(s)
            cmp(s[k], b[k][1]) || return false
        end
        return true
    end

    throws_domain(f) = try
        (f(); false)
    catch e
        e isa DomainError
    end
    returns_ok(f) = try
        (f(); true)
    catch
        false
    end
end

@testitem "scalar ≡ batch: in-domain + F64-grid OOB (ops × extraps × value types)" setup = [ScalarUnifyOracles] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), 2.0)          # -0.0 exercises signed-zero OOB
    y2 = collect(range(2.0, 3.0, 11))
    # DerivOp≥4 (definitional zero) has a non-contractual sign that diverges
    # scalar (-0.0) vs batch (+0.0) on master; covered sign-agnostically in the
    # dedicated testitem below (Phase 3 unification makes it bit-identical).
    ops = (EvalValue(), DerivOp(1), DerivOp(2), DerivOp(3))

    # In-domain: every extrap (incl. NoExtrap / InBounds) at matched F64 precision.
    for extrap in (NoExtrap(), ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = cubic_interp(x, Series(y1, y2); extrap = extrap)
        for xq in (0.05, 0.37, 0.94), op in ops
            @test scalar_eq_batch(sitp, xq, op)
        end
    end

    # OOB at matched F64 precision — the OOB extraps that return a value (not NoExtrap).
    for extrap in (ExtendExtrap(), ClampExtrap(), FillExtrap(NaN), FillExtrap(7.5), WrapExtrap())
        sitp = cubic_interp(x, Series(y1, y2); extrap = extrap)
        for xq in (-0.5, 1.5), op in ops
            @test scalar_eq_batch(sitp, xq, op)
        end
    end
end

@testitem "scalar ≡ batch: DerivOp≥4 zero payload (signed, bit-identical)" setup = [ScalarUnifyOracles] begin
    # A ≥4 derivative of a cubic is exactly 0. Now that scalar routes through the
    # lean per-k zero arm, its signed zero matches batch bit-for-bit (on master
    # they diverged scalar(-0.0)/batch(+0.0) via different kernels).
    x = collect(range(0.0, 1.0, 11))
    y1 = vcat(-0.0, collect(1.0:9.0), 2.0)
    y2 = collect(range(2.0, 3.0, 11))
    for extrap in (NoExtrap(), ExtendExtrap(), ClampExtrap(), FillExtrap(7.5), WrapExtrap(), InBounds())
        sitp = cubic_interp(x, Series(y1, y2); extrap = extrap)
        dom = (extrap isa NoExtrap || extrap isa InBounds) ? (0.05, 0.37, 0.94) : (0.05, 0.37, 0.94, -0.5, 1.5)
        for xq in dom, op in (DerivOp(4), DerivOp(5))
            @test scalar_eq_batch(sitp, xq, op)
        end
    end
end

@testitem "scalar ≡ batch: matched-precision F32 grid + Complex values" setup = [ScalarUnifyOracles] begin
    # matched F32 grid / F32 query (the MIXED-precision case is the RED pin below)
    x32 = collect(Float32, range(0.0f0, 1.0f0, 11))
    y1 = collect(Float32, range(1, 2, 11))
    y2 = collect(Float32, range(2, 3, 11))
    for extrap in (ClampExtrap(), ExtendExtrap(), FillExtrap(9.0f0))
        sitp = cubic_interp(x32, Series(y1, y2); extrap = extrap)
        for xq in (0.37f0, -0.5f0, 1.5f0), op in (EvalValue(), DerivOp(1), DerivOp(2))
            @test scalar_eq_batch(sitp, xq, op)
        end
    end

    # Complex series values (in-domain + OOB), F64 grid.
    x = collect(range(0.0, 1.0, 11))
    yc = complex.(collect(range(1.0, 2.0, 11)), collect(range(-1.0, 1.0, 11)))
    for extrap in (ClampExtrap(), ExtendExtrap())
        sitp = cubic_interp(x, Series(yc, 2 .* yc); extrap = extrap)
        for xq in (0.37, -0.5, 1.5), op in (EvalValue(), DerivOp(1))
            @test scalar_eq_batch(sitp, xq, op)
        end
    end
end

@testitem "scalar ≡ batch: Dual-query in-domain (value + partials)" setup = [ScalarUnifyOracles] begin
    x = collect(range(0.0, 1.0, 11))
    y1 = collect(range(1.0, 2.0, 11))
    y2 = collect(range(2.0, 3.0, 11))
    for extrap in (ClampExtrap(), ExtendExtrap())
        sitp = cubic_interp(x, Series(y1, y2); extrap = extrap)
        for xq in (0.05, 0.37, 0.94), op in (EvalValue(), DerivOp(1), DerivOp(2))
            @test scalar_eq_batch(sitp, Dual(xq, 1.0), op, dual_match)
        end
    end
end

@testitem "persistent scalar: zero-alloc after warmup (lean point path)" setup = [AllocConstants] begin
    # extrap/op baked in as literals + two warmups (LTS won't const-prop through
    # args under the @testset try/catch, and the first call of each specialization
    # pays a one-time JIT cost a single warmup misses). See test_cubic_oneshot_series.jl.
    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    function m_clamp_oob(x, y1, y2)
        sitp = cubic_interp(x, Series(y1, y2); extrap = ClampExtrap())
        out = zeros(2)
        sitp(out, 1.5; deriv = DerivOp(1))
        sitp(out, 1.5; deriv = DerivOp(1))
        return @allocated sitp(out, 1.5; deriv = DerivOp(1))
    end
    function m_fill_indom(x, y1, y2)
        sitp = cubic_interp(x, Series(y1, y2); extrap = FillExtrap(9.0))
        out = zeros(2)
        sitp(out, 0.37; deriv = EvalValue())
        sitp(out, 0.37; deriv = EvalValue())
        return @allocated sitp(out, 0.37; deriv = EvalValue())
    end
    @test m_clamp_oob(x, y1, y2) <= ALLOC_THRESHOLD
    @test m_fill_indom(x, y1, y2) <= ALLOC_THRESHOLD
end

@testitem "scalar OOB now correct: mixed-precision / Dual (was MethodError on master)" setup = [ScalarUnifyOracles] begin
    # (a) NoExtrap mixed precision OOB: now throws DomainError (was MethodError).
    x32 = collect(Float32, range(0.0f0, 1.0f0, 11))
    y1 = collect(Float32, range(1, 2, 11))
    y2 = collect(Float32, range(2, 3, 11))
    sN = cubic_interp(x32, Series(y1, y2); extrap = NoExtrap())
    @test throws_domain(() -> sN(1.5))                            # 1.5::Float64 OOB on F32 grid

    # (b) Clamp/Fill mixed precision OOB: now returns the batch value.
    sC = cubic_interp(x32, Series(y1, y2); extrap = ClampExtrap())
    sF = cubic_interp(x32, Series(y1, y2); extrap = FillExtrap(9.0))
    @test scalar_eq_batch(sC, 1.5, EvalValue())
    @test scalar_eq_batch(sF, 1.5, EvalValue())

    # (c) Dual-query Clamp OOB: now returns the batch value.
    x = collect(range(0.0, 1.0, 11))
    z1 = collect(range(1.0, 2.0, 11))
    z2 = collect(range(2.0, 3.0, 11))
    sCd = cubic_interp(x, Series(z1, z2); extrap = ClampExtrap())
    @test scalar_eq_batch(sCd, Dual(1.5, 1.0), EvalValue(), dual_match)
end
