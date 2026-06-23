# Type-promotion contract for the integrate path: the `_integrate_op` witness floats
# Int, lifts Dual, and stays duck-safe; integrate over Int/Dual grids returns the
# right element type. Phase 1 of the integrate promotion modernization.
#
# Type stability is pinned with `@inferred` (return type is inferred + concrete) combined
# with `isa` (the concrete type is the expected one); AD-wrt-grid correctness is pinned by
# matching the ForwardDiff partial against a central finite difference.

@testitem "_integrate_op witness eltype matrix" begin
    using FastInterpolations: _integrate_op, _promote_eltype
    using ForwardDiff: Dual
    D = Dual{Nothing, Float64, 1}
    # (Tg, Tv, Tspan) → Tout
    @test _promote_eltype(_integrate_op, Float64, Float64, Float64) === Float64
    @test _promote_eltype(_integrate_op, Int, Int, Int) === Float64          # floats Int
    @test _promote_eltype(_integrate_op, D, Float64, Float64) === D          # AD wrt grid
    @test _promote_eltype(_integrate_op, Float64, Float64, D) === D          # AD wrt bounds
    @test _promote_eltype(_integrate_op, Float64, Int, Float64) === Float64  # Int data floats
end

@testitem "integrate — Constant 1D over all-Int grid floats to Float64" begin
    x  = collect(1:10)            # Vector{Int} grid
    y  = collect(1:10) .^ 2       # Vector{Int} data
    xf = float.(x); yf = float.(y)
    for side in (LeftSide(), RightSide(), NearestSide())
        itp_i = constant_interp(x,  y;  side = side)
        itp_f = constant_interp(xf, yf; side = side)
        # bounded (Int bounds) — @inferred pins type stability, isa pins the concrete type
        r = @inferred integrate(itp_i, 2, 7)
        @test r isa Float64
        @test r ≈ integrate(itp_f, 2.0, 7.0)
        # full-domain
        rf = @inferred integrate(itp_i)
        @test rf isa Float64
        @test rf ≈ integrate(itp_f)
        # cumulative
        rc = @inferred cumulative_integrate(itp_i)
        @test rc isa Vector{Float64}
        @test rc ≈ cumulative_integrate(itp_f)
    end
end

@testitem "integrate — Dual grid (AD wrt nodes) returns concrete Dual for non-Hermite 1D" begin
    using ForwardDiff: Dual, value
    mkdual(r) = [Dual{Nothing}(Float64(v), 1.0) for v in r]
    g  = mkdual(0.5:1.0:9.5)
    gf = value.(g)                       # Float64 reference grid
    y  = collect(Float64, 1:10)
    for mk in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
        itp  = mk(g,  y)
        itpf = mk(gf, y)
        I = @inferred integrate(itp, 1.0, 5.0)        # @inferred pins concrete-Dual stability
        @test I isa Dual{Nothing, Float64, 1}
        @test value(I) ≈ integrate(itpf, 1.0, 5.0)
    end
end

@testitem "integrate — Dual-grid partial matches finite difference (AD-wrt-grid correctness)" begin
    using ForwardDiff: Dual, value, partials
    xb = collect(0.5:1.0:9.5)
    y  = collect(Float64, 1:10)
    # I(s) = ∫_{1.3}^{4.7} of the interpolant built on the grid scaled by s. Bounds avoid
    # nodes (0.5,1.5,…) and Constant NearestSide midpoints (1,2,…) so I(s) is smooth at s=1
    # for every family (no subgradient kink), making the AD partial comparable to a finite diff.
    Iof(mk, s) = integrate(mk(s .* xb, y), 1.3, 4.7)
    hfd = 1.0e-6
    for mk in (linear_interp, cubic_interp, quadratic_interp, constant_interp)
        Id = Iof(mk, Dual{:t}(1.0, 1.0))                        # AD: dI/ds at s = 1
        fd = (Iof(mk, 1.0 + hfd) - Iof(mk, 1.0 - hfd)) / (2hfd)  # central difference
        @test value(Id) ≈ Iof(mk, 1.0)
        @test partials(Id)[1] ≈ fd rtol = 1.0e-4
    end
end

@testitem "integrate — non-Constant 1D Int grids stay correct (regression pin)" begin
    x = collect(1:10); y = float.(collect(1:10) .^ 2)
    xf = float.(x)
    for mk in (linear_interp, cubic_interp, quadratic_interp,
               pchip_interp, cardinal_interp, akima_interp)
        @test (@inferred integrate(mk(x, y), 2.0, 7.0)) ≈ integrate(mk(xf, y), 2.0, 7.0)
    end
end

@testitem "integrate — ND all-Int grid floats to Float64 (ConstantND InexactError fix)" begin
    xg = collect(1:6)
    data_i = [i + 2j for i in 1:6, j in 1:6]      # Int data
    data_f = float.(data_i)
    xf = float.(xg)
    # ConstantND: the headline InexactError case.
    ci = constant_interp((xg, xg), data_i)
    cf = constant_interp((xf, xf), data_f)
    rb = @inferred integrate(ci, (2, 2), (5, 5))
    @test rb isa Float64
    @test rb ≈ integrate(cf, (2.0, 2.0), (5.0, 5.0))
    rfd = @inferred integrate(ci)
    @test rfd isa Float64
    @test rfd ≈ integrate(cf)
    # Cubic/Linear/Quadratic ND already float at construction — regression pin.
    for mk in (linear_interp, cubic_interp, quadratic_interp)
        ii = mk((xg, xg), data_f); ff = mk((xf, xf), data_f)
        @test (@inferred integrate(ii, (2.0, 2.0), (5.0, 5.0))) ≈ integrate(ff, (2.0, 2.0), (5.0, 5.0))
    end
end

@testitem "integrate — Float-path values + zero-alloc unchanged after witness routing" setup = [AllocConstants] begin
    x = collect(range(0.0, 1.0, length = 21))
    y = @. 3x - 1
    # Affine reference: ∫_a^b (3x-1) dx = [1.5x² - x]_a^b
    a, b = 0.15, 0.85
    itp = linear_interp(x, y; extrap = NoExtrap())
    @test integrate(itp, a, b) ≈ (1.5b^2 - b) - (1.5a^2 - a) atol = 1.0e-12

    # Zero-alloc on the Float path, measured in a function barrier (avoids @testset
    # try/catch polluting @allocated).
    probe(it, p, q) = (it(p); integrate(it, p, q); @allocated integrate(it, p, q))
    probe(itp, a, b)                       # warmup/compile
    @test probe(itp, a, b) <= ALLOC_THRESHOLD
end
