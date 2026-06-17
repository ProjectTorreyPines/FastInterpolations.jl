# ════════════════════════════════════════════════════════════════════════════
# The reverse-mode adjoint must classify a range-grid boundary query the same way
# the forward eval does. Previously the adjoints derived OOB state from raw
# `first`/`last`, so a query at the true endpoint was flagged OOB → its scatter
# contribution skipped → an all-zero sensitivity row under FillExtrap.
#
# Contract: at an in-domain boundary query `∂out/∂y` sums to 1 (value interp
# reproduces constants). Uses the synthetic widened `_CachedRange` (`InwardCR`).
# ════════════════════════════════════════════════════════════════════════════

@testitem "Adjoint boundary FillExtrap — sensitivity row sums to 1 at the true endpoint" setup = [InwardCR] begin
    using FastInterpolations
    n = 5
    crR = inward_cr_right(0.0, 2.0, n)   # query true endpoint 2.0 (> stored hi)
    crL = inward_cr_left(1.0, 3.0, n)    # query true endpoint 1.0 (< stored lo)
    y = collect(10.0:10.0:50.0)
    fv = FillExtrap(0.0)

    rowsum(adj) = sum(vec(Base.Matrix(adj)))

    # Data-independent adjoints (linear in y, no y argument).
    @test rowsum(linear_adjoint(crR, 2.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(linear_adjoint(crL, 1.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(cubic_adjoint(crR, 2.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(cubic_adjoint(crL, 1.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(quadratic_adjoint(crR, 2.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(quadratic_adjoint(crL, 1.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(constant_adjoint(crR, 2.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(constant_adjoint(crL, 1.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(cardinal_adjoint(crR, 2.0; tension = 0.0, extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(cardinal_adjoint(crL, 1.0; tension = 0.0, extrap = fv)) ≈ 1.0  atol = 1.0e-9

    # Data-dependent adjoints (slope-from-data; need y).
    @test rowsum(pchip_adjoint(crR, y, 2.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(pchip_adjoint(crL, y, 1.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(akima_adjoint(crR, y, 2.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(akima_adjoint(crL, y, 1.0; extrap = fv)) ≈ 1.0  atol = 1.0e-9
end

@testitem "ND adjoint boundary FillExtrap — sensitivity sums to 1 at the corner" setup = [InwardCR] begin
    using FastInterpolations
    n = 5
    crx = inward_cr_both(0.0, 2.0, n)
    cry = inward_cr_both(0.0, 2.0, n)
    fv = FillExtrap(0.0)
    rowsum(adj) = sum(Base.Matrix(adj))
    corner = (2.0, 2.0)   # both axes at the true endpoint

    # 2D adjoint ∂out/∂data at an in-domain corner must sum to 1 (value
    # interpolation reproduces constants). The per-axis `is_oob` flag (computed
    # from raw first/last) misclassified the corner → weights zeroed → sum 0.
    @test rowsum(linear_adjoint((crx, cry), corner; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(cubic_adjoint((crx, cry), corner; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(constant_adjoint((crx, cry), corner; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(quadratic_adjoint((crx, cry), corner; extrap = fv)) ≈ 1.0  atol = 1.0e-9
    # Heterogeneous per-axis methods route through the shared ND anchor builder.
    @test rowsum(
        hetero_adjoint(
            (crx, cry), corner;
            methods = (LinearInterp(), CubicInterp()), extrap = fv
        )
    ) ≈ 1.0  atol = 1.0e-9

    # Left corner (both axes round inward from below).
    crxL = inward_cr_left(1.0, 3.0, n)
    cryL = inward_cr_left(1.0, 3.0, n)
    @test rowsum(linear_adjoint((crxL, cryL), (1.0, 1.0); extrap = fv)) ≈ 1.0  atol = 1.0e-9
    @test rowsum(cubic_adjoint((crxL, cryL), (1.0, 1.0); extrap = fv)) ≈ 1.0  atol = 1.0e-9
end

@testitem "Adjoint boundary NoExtrap — true endpoint accepted (no DomainError)" setup = [InwardCR] begin
    using FastInterpolations
    n = 5
    crR = inward_cr_right(0.0, 2.0, n)
    crL = inward_cr_left(1.0, 3.0, n)
    y = collect(10.0:10.0:50.0)
    no = NoExtrap()
    # Forward NoExtrap accepts the true endpoint (it checks `domain_lo/hi`); the
    # adjoint validation must agree, not throw a DomainError one ULP early.
    @test (linear_adjoint(crR, 2.0; extrap = no); true)
    @test (linear_adjoint(crL, 1.0; extrap = no); true)
    @test (constant_adjoint(crR, 2.0; extrap = no); true)
    @test (cubic_adjoint(crR, 2.0; extrap = no); true)
    @test (quadratic_adjoint(crR, 2.0; extrap = no); true)
    @test (cardinal_adjoint(crR, 2.0; tension = 0.0, extrap = no); true)
    @test (pchip_adjoint(crR, y, 2.0; extrap = no); true)
    @test (akima_adjoint(crR, y, 2.0; extrap = no); true)
end

@testitem "Adjoint boundary — reverse matches forward (Matrix·y ≈ itp) for linear-in-y methods" setup = [InwardCR] begin
    using FastInterpolations
    n = 5
    cr = inward_cr_right(0.0, 2.0, n)
    y = collect(10.0:10.0:50.0)
    fv = FillExtrap(-7.0)   # distinctive fill: a leak would show as ≈ -7, not y[end]

    # For methods whose interpolant is linear in y, the adjoint IS the forward
    # linear map, so Matrix(adj)·y must equal the forward value at the boundary.
    @test sum(vec(Base.Matrix(linear_adjoint(cr, 2.0; extrap = fv))) .* y) ≈
        linear_interp(cr, y, 2.0; extrap = fv)  atol = 1.0e-9
    @test sum(vec(Base.Matrix(cubic_adjoint(cr, 2.0; extrap = fv))) .* y) ≈
        cubic_interp(cr, y, 2.0; extrap = fv)  atol = 1.0e-9
    @test sum(vec(Base.Matrix(constant_adjoint(cr, 2.0; extrap = fv))) .* y) ≈
        constant_interp(cr, y, 2.0; extrap = fv)  atol = 1.0e-9
end

@testitem "Adjoint golden-rule (transpose identity) on widened _CachedRange boundary" setup = [InwardCR] begin
    using FastInterpolations
    using LinearAlgebra: dot
    # Golden rule for an interpolant linear in the data f: the adjoint IS Wᵀ, so
    #   ⟨W·f, ȳ⟩ == ⟨f, Wᵀȳ⟩   ⇔   dot(itp.(xq), ȳ) == dot(f, adj(ȳ)).
    # Checked on the synthetic widened `_CachedRange` (stored lo/hi rounded inward,
    # domain_lo/hi recover the true endpoints) with the query set INCLUDING the
    # true endpoint (the 1-ULP sliver). This pins the fused ClampExtrap/FillExtrap
    # adjoint builder to a map consistent with the forward at the widened boundary.
    # (atol 1e-7 absorbs the intended ~1e-14 forward-sliver-vs-adjoint-snap gap:
    #  forward leaves the sliver query at t⪆1, the adjoint clamps it to t==1.)
    n = 5
    f = [10.0, 20.0, 30.0, 40.0, 50.0]
    ȳ = [1.0, -2.0, 3.0]

    for (cr, ep) in ((inward_cr_right(0.0, 2.0, n), 2.0), (inward_cr_left(1.0, 3.0, n), 1.0))
        xq = [ep, cr[2], cr[3]]          # true-endpoint sliver + two interior points
        for extrap in (FillExtrap(-7.0), ClampExtrap())
            for (madj, mfwd) in (
                    (linear_adjoint, linear_interp),
                    (cubic_adjoint, cubic_interp),
                    (constant_adjoint, constant_interp),
                    (quadratic_adjoint, quadratic_interp),
                )
                itp = mfwd(cr, f; extrap = extrap)
                adj = madj(cr, xq; extrap = extrap)
                @test dot(itp.(xq), ȳ) ≈ dot(f, adj(ȳ))  atol = 1.0e-7
            end
            # cardinal carries a tension kwarg (tension 0 ⇒ Catmull–Rom, linear in f)
            itpc = cardinal_interp(cr, f; tension = 0.0, extrap = extrap)
            adjc = cardinal_adjoint(cr, xq; tension = 0.0, extrap = extrap)
            @test dot(itpc.(xq), ȳ) ≈ dot(f, adjc(ȳ))  atol = 1.0e-7
        end
    end
end
