# ════════════════════════════════════════════════════════════════════════════
# Range-grid boundary OOB classification must use the axis-aware widened bounds
# (`domain_lo`/`domain_hi`), not raw `first`/`last`: on the x86_64 `_CachedRange`
# fast path the stored `lo`/`hi` can round 1 ULP inward of the true endpoint, so a
# query at the true endpoint (== `domain_hi`) is otherwise flagged OOB and
# FillExtrap leaks the fill. The batch path already classifies on `domain_lo/hi`;
# these tests pin every scalar/one-shot/ND path to it. The synthetic `_CachedRange`
# below forces the inward rounding, so the bug reproduces on any architecture.
# ════════════════════════════════════════════════════════════════════════════

@testsnippet InwardCR begin
    using FastInterpolations: _CachedRange

    # Right boundary: stored `hi = prevfloat(true_hi)` (1 ULP inward),
    # `domain_hi = nextfloat(hi) = true_hi` (the cushion recovers it).
    function inward_cr_right(lo, true_hi, n)
        hi = prevfloat(true_hi)
        h = (hi - lo) / (n - 1)
        return _CachedRange{Float64, Float64}(lo, hi, h, inv(h), n, lo, nextfloat(hi))
    end

    # Left boundary: stored `lo = nextfloat(true_lo)` (1 ULP inward),
    # `domain_lo = prevfloat(lo) = true_lo`.
    function inward_cr_left(true_lo, hi, n)
        lo = nextfloat(true_lo)
        h = (hi - lo) / (n - 1)
        return _CachedRange{Float64, Float64}(lo, hi, h, inv(h), n, prevfloat(lo), hi)
    end

    # Both endpoints inward (the real x86 case): stored `lo`/`hi` round toward
    # the interior, `domain_lo`/`domain_hi` recover the true endpoints.
    function inward_cr_both(true_lo, true_hi, n)
        lo = nextfloat(true_lo)
        hi = prevfloat(true_hi)
        h = (hi - lo) / (n - 1)
        return _CachedRange{Float64, Float64}(lo, hi, h, inv(h), n, prevfloat(lo), nextfloat(hi))
    end
end

@testitem "Boundary FillExtrap — scalar persistent == batch (right endpoint)" setup = [InwardCR] begin
    using FastInterpolations
    n = 5
    cr = inward_cr_right(0.0, 2.0, n)
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    fv = -999.0
    true_hi = 2.0   # == cr.domain_hi, but > cr.hi
    for m in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, cardinal_interp, akima_interp,
        )
        itp = m(cr, y; extrap = FillExtrap(fv))
        scalar = itp(true_hi)
        batch = itp([true_hi])[1]
        @test scalar != fv                      # fill must NOT leak at the true boundary
        @test scalar ≈ y[end]  atol = 1.0e-7    # returns the boundary value
        @test scalar ≈ batch   atol = 1.0e-9    # scalar agrees with the correct batch path
    end
end

@testitem "Boundary FillExtrap — scalar persistent == batch (left endpoint)" setup = [InwardCR] begin
    using FastInterpolations
    n = 5
    cr = inward_cr_left(1.0, 3.0, n)
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    fv = -999.0
    true_lo = 1.0   # == cr.domain_lo, but < cr.lo
    for m in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, cardinal_interp, akima_interp,
        )
        itp = m(cr, y; extrap = FillExtrap(fv))
        scalar = itp(true_lo)
        batch = itp([true_lo])[1]
        @test scalar != fv
        @test scalar ≈ y[begin]  atol = 1.0e-7
        @test scalar ≈ batch     atol = 1.0e-9
    end
end

@testitem "Boundary FillExtrap — one-shot == batch (right endpoint)" setup = [InwardCR] begin
    using FastInterpolations
    n = 5
    cr = inward_cr_right(0.0, 2.0, n)
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    fv = -999.0
    true_hi = 2.0
    for m in (
            linear_interp, cubic_interp, quadratic_interp, constant_interp,
            pchip_interp, cardinal_interp, akima_interp,
        )
        oneshot = m(cr, y, true_hi; extrap = FillExtrap(fv))
        batch = m(cr, y, [true_hi]; extrap = FillExtrap(fv))[1]
        @test oneshot != fv
        @test oneshot ≈ y[end]  atol = 1.0e-7
        @test oneshot ≈ batch   atol = 1.0e-9
    end
end

@testitem "Boundary FillExtrap — ND corner query returns the corner, not fill" setup = [InwardCR] begin
    using FastInterpolations
    n = 5
    crx = inward_cr_right(0.0, 2.0, n)
    cry = inward_cr_right(0.0, 2.0, n)
    data = [10.0 * i + j for i in 1:n, j in 1:n]
    fv = -999.0
    corner = (2.0, 2.0)   # both axes at their true endpoint
    for m in (linear_interp, cubic_interp, constant_interp)
        itp = m((crx, cry), data; extrap = FillExtrap(fv))
        scalar = itp(corner)
        @test scalar != fv
        @test scalar ≈ data[end, end]  atol = 1.0e-7
    end
end

@testitem "Boundary FillExtrap(0) — ND full-grid boundary" setup = [InwardCR] begin
    using FastInterpolations
    # Both axes round inward (x86 fast path), fill = 0. A boundary grid point
    # queried at its true value must interpolate, not collapse to the 0 fill —
    # i.e. no boundary row/column may go all-zero.
    n = 5
    crx = inward_cr_both(0.0, 2.0, n)
    cry = inward_cr_both(0.0, 2.0, n)
    data = [10.0 * i + j for i in 1:n, j in 1:n]   # every entry nonzero
    lo, hi = 0.0, 2.0                              # == domain_lo / domain_hi
    mid = crx[3]                                   # an interior grid point
    for m in (linear_interp, cubic_interp, constant_interp)
        itp = m((crx, cry), data; extrap = FillExtrap(0.0))
        # Four corners return the corner density (not the 0 fill).
        @test itp((lo, lo)) ≈ data[1, 1]  atol = 1.0e-7
        @test itp((lo, hi)) ≈ data[1, n]  atol = 1.0e-7
        @test itp((hi, lo)) ≈ data[n, 1]  atol = 1.0e-7
        @test itp((hi, hi)) ≈ data[n, n]  atol = 1.0e-7
        # Boundary edges (one axis at the true endpoint) — the "first row/column"
        # case: must interpolate, not collapse to the 0 fill.
        @test itp((lo, mid)) != 0.0
        @test itp((hi, mid)) != 0.0
        @test itp((mid, lo)) != 0.0
        @test itp((mid, hi)) != 0.0
    end
end

@testitem "Boundary FillExtrap — scalar and batch must not disagree" setup = [InwardCR] begin
    using FastInterpolations
    # The sharpest statement of the bug: the same interpolant returns the fill
    # value for a scalar query but the boundary value for the batch of one.
    n = 5
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    fv = -999.0
    crR = inward_cr_right(0.0, 2.0, n)
    itpR = linear_interp(crR, y; extrap = FillExtrap(fv))
    @test itpR(2.0) ≈ itpR([2.0])[1]  atol = 1.0e-9
    crL = inward_cr_left(1.0, 3.0, n)
    itpL = linear_interp(crL, y; extrap = FillExtrap(fv))
    @test itpL(1.0) ≈ itpL([1.0])[1]  atol = 1.0e-9
end

@testitem "Boundary WrapExtrap — true endpoint is in-domain (returns y[end])" setup = [InwardCR] begin
    using FastInterpolations
    # WrapExtrap shares the same `first/last` classification: a query at the
    # true endpoint must be treated as the closed-domain boundary (y[end]),
    # not wrapped back to y[1].
    n = 5
    cr = inward_cr_right(0.0, 2.0, n)
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    scalar = linear_interp(cr, y, 2.0; extrap = WrapExtrap())
    batch = linear_interp(cr, y, [2.0]; extrap = WrapExtrap())[1]
    @test scalar ≈ y[end]  atol = 1.0e-7
    @test scalar ≈ batch   atol = 1.0e-9
end

@testitem "Boundary derivative — slope at the true endpoint, not zero" setup = [InwardCR] begin
    using FastInterpolations
    # ClampExtrap value is correct either way (y[end]), but the DERIVATIVE
    # exposes the misclassification: OOB-clamp yields a flat (zero) derivative,
    # while the correct in-domain classification yields the boundary slope.
    n = 5
    cr = inward_cr_right(0.0, 2.0, n)
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    g = linear_interp(cr, y, 2.0; extrap = ClampExtrap(), deriv = DerivOp(1))
    @test g != 0.0
    @test g ≈ (y[end] - y[end - 1]) / cr.h  atol = 1.0e-6
end

@testitem "Boundary Series — true endpoint returns the boundary curve" setup = [InwardCR] begin
    using FastInterpolations
    # Series interpolants classify via `_anchor_loc`, so a query at the true
    # `_CachedRange` endpoint returns the boundary curve (matches the exact
    # Vector grid), not an OOB result. Regression guard for the inherited fix.
    n = 5
    Y = [10.0 * i + j for i in 1:n, j in 1:2]
    crL = inward_cr_left(1.0, 3.0, n)
    refL = collect(range(1.0, 3.0, n))
    crR = inward_cr_right(0.0, 2.0, n)
    refR = collect(range(0.0, 2.0, n))
    for m in (linear_interp, quadratic_interp, cubic_interp, constant_interp)
        @test m(crL, Series(Y))(1.0) ≈ m(refL, Series(Y))(1.0)  atol = 1.0e-7
        @test m(crR, Series(Y))(2.0) ≈ m(refR, Series(Y))(2.0)  atol = 1.0e-7
    end
end

@testitem "Boundary exclusive-periodic — true endpoint uses first cell, not seam" setup = [InwardCR] begin
    using FastInterpolations
    # Inner `_CachedRange` with an inward-rounded left endpoint (lo = nextfloat(0),
    # domain_lo = 0). Under `:exclusive` PeriodicBC the inner's widening does not
    # reach `_ExclusivePeriodicAxis` classification, so the true left endpoint was
    # wrapped to the seam cell. Value is y[1] either way; the derivative is the
    # sharp guard — it must be the FIRST-cell slope, not the seam-cell slope.
    cr = inward_cr_left(0.0, 1.0, 2)
    xref = [0.0, 1.0]
    y = [10.0, 30.0]
    bc = PeriodicBC(endpoint = :exclusive, period = 2.0)
    @test linear_interp(cr, y, 0.0; bc = bc) ≈
        linear_interp(xref, y, 0.0; bc = bc)  atol = 1.0e-9
    @test linear_interp(cr, y, 0.0; bc = bc, deriv = DerivOp(1)) ≈
        linear_interp(xref, y, 0.0; bc = bc, deriv = DerivOp(1))  atol = 1.0e-9
end
