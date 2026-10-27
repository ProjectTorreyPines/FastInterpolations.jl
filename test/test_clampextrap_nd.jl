# ClampExtrap ND forward-evaluation contract.
#
# ClampExtrap is a *flat* extension that clamps each coordinate independently:
#     f_ext(x, y) = f(clamp(x, loₓ, hiₓ), clamp(y, lo_y, hi_y))
# For a query OOB along x but in-domain along y, f_ext(x, y) = f(loₓ, y):
#   - it is CONSTANT in x          ⇒ ∂f/∂x = 0       (the OOB axis collapses)
#   - it still VARIES in y          ⇒ ∂f/∂y ≠ 0       (the in-domain axis keeps its slope)
# So the rule is axis-specific: a partial is zero iff it differentiates along an
# OOB axis; partials along in-domain axes survive. A blanket "OOB ⇒ zero deriv"
# (the 1D intuition) would wrongly zero the in-domain slope too.
#
# The ND *forward* path currently differentiates the clamped boundary cell, so the
# pure derivative along an OOB axis returns the boundary slope instead of 0. The
# in-domain-axis slopes are already correct. The defect is pinned with `@test_broken`.

@testitem "ClampExtrap ND forward contract" begin
    gx = 1:5
    gy = 1:5
    A = [Float64(i + j) for i in gx, j in gy]   # f(x,y) = x + y, ∂x ≡ ∂y ≡ 1 in-domain
    itp = linear_interp((gx, gy), A; extrap = ClampExtrap())

    @testset "value at OOB axis (clamped)" begin
        @test itp((0.0, 3.0)) ≈ 4.0    # x OOB-left → clamp x→1, f(1,3) = 4
        @test itp((0.0, 0.0)) ≈ 2.0    # both OOB → clamp to (1,1), f(1,1) = 2
        @test itp((2.5, 3.0)) ≈ 5.5    # in-domain, unaffected
    end

    @testset "in-domain-axis slope preserved under OOB on another axis" begin
        # Clamping one axis must NOT zero the derivative along the other axis.
        @test itp((0.0, 3.0); deriv = DerivOp(0, 1)) ≈ 1.0   # x OOB, ∂y still 1
        @test itp((3.0, 0.0); deriv = DerivOp(1, 0)) ≈ 1.0   # y OOB, ∂x still 1
        @test itp((2.5, 3.0); deriv = DerivOp(0, 1)) ≈ 1.0   # in-domain, ∂y = 1
    end

    @testset "deriv along an OOB axis must collapse to zero (KNOWN BUG)" begin
        # 1D reference: the OOB-axis derivative is correctly 0.
        i1 = linear_interp(gx, Float64.(collect(gx)); extrap = ClampExtrap())
        @test i1(0.0; deriv = DerivOp(1)) == 0.0

        # ND: pure derivative ALONG the OOB axis should be 0 (flat), currently 1.0.
        @test_broken itp((0.0, 3.0); deriv = DerivOp(1, 0)) == 0.0   # x OOB → ∂x
        @test_broken itp((3.0, 0.0); deriv = DerivOp(0, 1)) == 0.0   # y OOB → ∂y
    end
end

# A query that `_oob_state` classifies IN_DOMAIN must pass through the Clamp coordinate
# handler unchanged (value AND partials), exactly like `InBounds`. Batch paths promote an
# all-in-domain axis to `InBounds`, so any deviation makes the result depend on the path
# and on the other queries in the batch. A plain `min/max` clamp got two in-domain Dual cases
# wrong: a Dual AT an endpoint (the `max` tie at `lo` keeps the constant, dropping ∂), and a
# Dual inside a `_WidenedDomain` bracket but outside the physical `first/last` (clamped, ∂ = 0).
# Plain floats keep the 2-op clamp; its 1-ulp widened-sliver residual is pinned broken below.
@testitem "ClampExtrap ND: in-domain queries pass through unchanged" setup = [Basic] begin
    using ForwardDiff, Unitful
    using ForwardDiff: Dual, value, partials

    # f = x + 2y: every family with a slope reproduces it exactly → ∂ along axis a is slopes[a].
    x = collect(0.0:4.0)
    data = [a + 2b for a in x, b in x]
    slopes = (1.0, 2.0)
    hetero(g, d, q...; kw...) = interp(g, d, q...; method = (CubicInterp(), LinearInterp()), kw...)
    families = (
        ("linear", linear_interp, true), ("quadratic", quadratic_interp, true),
        ("cubic", cubic_interp, true), ("pchip", pchip_interp, true),
        ("hetero", hetero, true), ("constant", constant_interp, false),
    )
    clamp_ = ClampExtrap()
    ∂(v) = partials(v)[1]
    agree(a, b) = isclose(value(a), value(b); nulps = PATH_ULPS)
    # `a` = the axis carrying the Dual; `u` goes on axis a, `o` on the other axis.
    axes_tuple(a, u, o) = a == 1 ? (u, o) : (o, u)

    @testset "Dual at an endpoint — $name" for (name, F, has_slope) in families
        itp = F((x, x), data; extrap = clamp_)
        oneshot(q) = F((x, x), data, q; extrap = clamp_)
        for a in 1:2, e in (first(x), last(x))
            dq = Dual(e, 1.0)
            oob_same = Dual(e == first(x) ? -1.0 : 5.0, 1.0)
            for other in (1.5, -1.0)                       # other axis inside / OOB
                ref = itp(axes_tuple(a, dq, other))
                paths = (
                    itp(axes_tuple(a, [dq], [other]))[1],
                    oneshot(axes_tuple(a, [dq], [other]))[1],
                    # an OOB query on the same axis keeps that axis on the Clamp handler
                    itp(axes_tuple(a, [dq, oob_same], [other, other]))[1],
                    oneshot(axes_tuple(a, [dq, oob_same], [other, other]))[1],
                )
                @test all(p -> agree(p, ref), paths)
                if has_slope
                    @test isclose(∂(ref), slopes[a]; nulps = PATH_ULPS)
                    @test all(p -> isclose(∂(p), slopes[a]; nulps = PATH_ULPS), paths)
                end
            end
        end
    end

    # `prevfloat(lo)` / `nextfloat(hi)` are inside the widened bracket (IN_DOMAIN) but outside
    # the physical `[first, last]` (on x86_64 a user's exact `first(r)` can land there).
    w = FI._cached_range(FI._WidenedDomain(), 1.0, 5.0, 1.0, 1.0, 5)
    @testset "widened-bracket sliver — $name" for (name, F, has_slope) in families
        dw = [a + 2b for a in w, b in x]
        itp = F((w, x), dw; extrap = clamp_)
        for q in (prevfloat(first(w)), nextfloat(last(w))), other in (1.5, -1.0)
            # Dual: ∂ survives iff the coordinate is left unchanged (a clamp zeroes it).
            dq = Dual(q, 1.0)
            ref = itp((dq, other))
            paths = (itp(([dq], [other]))[1], F((w, x), dw, ([dq], [other]); extrap = clamp_)[1])
            @test all(p -> agree(p, ref), paths)
            if has_slope
                @test isclose(∂(ref), slopes[1]; nulps = PATH_ULPS)
                @test all(p -> isclose(∂(p), slopes[1]; nulps = PATH_ULPS), paths)
            end
            # Plain float: the 2-op clamp snaps the sliver onto the endpoint; on smooth data that
            # differs from the InBounds-promoted batch only at ULP level.
            @test isclose(itp((q, other)), itp(([q], [other]))[1]; nulps = PATH_ULPS)
        end
        # Known residual: steep data exposes that 1-ulp snap (scalar clamps, the promoted batch
        # passes through).
        if has_slope
            ds = [(a - first(w)) * 1.0e15 + 2b for a in w, b in x]
            itps = F((w, x), ds; extrap = clamp_)
            for other in (1.5, -1.0)
                q = prevfloat(first(w))
                @test_broken isclose(itps((q, other)), itps(([q], [other]))[1]; nulps = PATH_ULPS)
            end
        end
    end

    # Genuinely OOB queries still clamp to the physical endpoint, with ∂ = 0 on that axis.
    @testset "OOB still clamps — $name" for (name, F, _) in families
        itp = F((x, x), data; extrap = clamp_)
        lo_v = itp((Dual(-1.0, 1.0), 1.5))
        hi_v = itp((Dual(5.0, 1.0), 1.5))
        @test isclose(value(lo_v), itp((first(x), 1.5)); nulps = PATH_ULPS) && ∂(lo_v) == 0
        @test isclose(value(hi_v), itp((last(x), 1.5)); nulps = PATH_ULPS) && ∂(hi_v) == 0
    end

    # Other carriers `_extract_primal` strips: inferred, and scalar == batch in every partial at both
    # endpoints (grid sensitivities and nested tags included).
    flat(v::Real) = [v]
    flat(v::Dual) = vcat(flat(value(v)), (flat(p) for p in partials(v))...)
    tg = typeof(ForwardDiff.Tag(sin, Float64))
    tq = typeof(ForwardDiff.Tag(cos, Float64))
    carriers = (
        ("Float query, Dual grid", Dual{tg}.(x, 1.0), identity),
        ("tagged Dual query, Dual grid", Dual{tg}.(x, 1.0), e -> Dual{tq}(e, 1.0)),
        ("nested Dual query", x, e -> Dual{tq}(Dual{tg}(e, 1.0), 1.0)),
        ("Dual query, Float32 grid", Float32.(x), e -> Dual(e, 1.0)),
    )
    @testset "$cname — $name" for (cname, g, mkq) in carriers, (name, F) in (("linear", linear_interp), ("cubic", cubic_interp))
        itp = F((g, x), data; extrap = clamp_)
        for e in (0.0, 4.0)
            q = mkq(e)
            ref = @inferred itp((q, 1.5))
            b = (@inferred itp(([q], [1.5])))[1]
            @test isclose(flat(ref), flat(b); nulps = PATH_ULPS)
        end
    end

    # Through a unit wrapper too (`Quantity{<:Dual}`, stripped by the Unitful extension): an endpoint
    # Dual seeded outward keeps its slope, and the result type is inferred.
    itpu = linear_interp((x .* u"s", x .* u"m"), data; extrap = clamp_)
    for (e, sd) in ((first(x), -1.0), (last(x), 1.0))
        v = @inferred itpu((Dual(e, sd) * u"s", 1.5u"m"))
        @test v isa Dual && isclose(∂(v), sd * slopes[1]; nulps = PATH_ULPS)
    end
end
