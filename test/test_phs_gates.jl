# ============================================================================
# PHS (polyharmonic spline) — experimental namespace and input gates
# ============================================================================
#
# PHS ships under `FastInterpolations.Experimental` (not covered by SemVer). Its
# constructor and callables reject inputs the implementation cannot handle
# correctly — non-uniform axes, stencils spanning more nodes than an axis, blends
# that leave cell centres uncovered, unsupported extrapolation, derivative orders
# ≥ 3 — with an ArgumentError instead of returning wrong values.
# ============================================================================

@testitem "PHS gates — names live in FastInterpolations.Experimental" begin
    for s in (:phs_interp, :phs_interp!, :PHSInterpolantND)
        @test !Base.isexported(FastInterpolations, s)
        @test Base.isexported(FastInterpolations.Experimental, s)
        @test getfield(FastInterpolations.Experimental, s) === getfield(FastInterpolations, s)
    end
    # the default `using FastInterpolations` of a testitem does not bring PHS in
    @test !isdefined(@__MODULE__, :phs_interp)
end

# ── G1: grid axes must be uniformly spaced ──────────────────────────────────
# One stencil geometry (and one Φ⁻¹) is built from the mean spacing of each axis,
# so a non-uniform axis would be evaluated as if it were uniform. Ranges are
# accepted as is; other vectors must match their mean spacing up to rounding.

@testitem "PHS gates — uniform axes accepted (ranges and rounding-level vectors)" begin
    using FastInterpolations.Experimental
    f(x) = sin(x) + 0.3x
    for g in (
            range(0.0, 2.0, 21),
            LinRange(0.0, 2.0, 21),
            collect(range(0.0, 2pi, 41)),
            collect(range(1.0e6, 1.0e6 + 1, 101)),
            collect(range(0.0f0, 1.0f0, 33)),
            collect(range(1000.0f0, 1100.0f0, 41)),
        )
        itp = phs_interp((g,), f.(g))
        @test itp((g[7],)) ≈ f(g[7]) rtol = 1.0e-5
    end
    @test phs_interp((1:10,), Float64.(1:10))((4.5,)) ≈ 4.5
    gx = range(0.0, 1.0, 9)
    gy = collect(range(0.0, 2.0, 11))
    @test phs_interp((gx, gy), [a + b for a in gx, b in gy]; stencil_size = 4)((0.3, 1.1)) ≈ 1.4
end

@testitem "PHS gates — non-uniform axes rejected" begin
    using FastInterpolations.Experimental
    u = collect(range(0.0, 1.0, 21))
    h = step(range(0.0, 1.0, 21))
    bump(r) = (x = copy(u); x[2:(end - 1)] .+= r * h .* (-1) .^ (1:19); x)
    for x in (bump(0.01), bump(1.0e-6), u .^ 2, (1 .- cos.(range(0, pi, 21))) ./ 2)
        @test_throws ArgumentError phs_interp((x,), sin.(x))
    end
    # one non-uniform axis is enough
    x2 = bump(0.01)
    @test_throws ArgumentError phs_interp((range(0.0, 1.0, 9), x2), zeros(9, 21))
    # the error points at the escape hatch for grids that are uniform up to rounding
    err = try
        phs_interp((u .^ 2,), u)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("range(", sprint(showerror, err))
end

# ── G2: stencil_size within [minimum, axis length] ─────────────────────────────
# A stencil larger than an axis cannot be shifted inside the grid and silently loses
# polynomial reproduction; stencils that are too small give singular local systems.

@testitem "PHS gates — stencil_size bounds" begin
    using FastInterpolations.Experimental
    lin(p) = 1.0 + sum(k * p[k] for k in eachindex(p))
    g5 = range(0.0, 1.0, 5)
    @test_throws ArgumentError phs_interp((g5,), lin.(tuple.(g5)); stencil_size = 8)
    g12 = range(0.0, 1.0, 12)
    g6 = range(0.0, 1.0, 6)
    @test_throws ArgumentError phs_interp((g12, g6), zeros(12, 6); stencil_size = 8)
    # below the minimum: singular local systems in 1-D/2-D, s = 1 everywhere
    g10 = range(0.0, 1.0, 10)
    @test_throws ArgumentError phs_interp((g10,), zeros(10); stencil_size = 2)
    @test_throws ArgumentError phs_interp((g10, g10), zeros(10, 10); stencil_size = 2)
    @test_throws ArgumentError phs_interp((g5, g5, g5), zeros(5, 5, 5); stencil_size = 1)
    # an axis shorter than the minimum stencil
    g2 = range(0.0, 1.0, 2)
    @test_throws ArgumentError phs_interp((g2,), [0.0, 1.0])

    maxerr(itp, N) = maximum(abs(itp(q) - lin(q)) for q in (ntuple(k -> 0.013 + 0.97 * mod(0.37j + 0.21k, 1.0), N) for j in 1:40))
    # the minimum and the axis length themselves are accepted and reproduce linears
    g3 = range(0.0, 1.0, 3)
    @test maxerr(phs_interp((g3, g3, g3), [lin((a, b, c)) for a in g3, b in g3, c in g3]; stencil_size = 2), 3) < 1.0e-12
    @test maxerr(phs_interp((g6,), lin.(tuple.(g6)); stencil_size = 6), 1) < 1.0e-12
    # the default (8) is checked like an explicit size
    @test_throws ArgumentError phs_interp((g5,), lin.(tuple.(g5)))
    # the stencil is the nodes nearest in physical distance, so on an anisotropic grid
    # it stretches along the fine axis: 5^2 nodes fit 20 × 6 by count but not by extent
    gc = range(0.0, 19.0, 20)
    gf = range(0.0, 0.5, 6)
    err = try
        phs_interp((gc, gf), [a + b for a in gc, b in gf]; stencil_size = 5)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("spans", sprint(showerror, err))
end

# ── G4: blend_factor must cover every cell centre ───────────────────────────────
# A query is blended from base nodes within a = blend_factor · h_max. The cell centre
# is the point farthest from every node, at d_c = ½‖h‖₂; if the blend weight there is
# below eps the evaluator returns exactly zero.

@testitem "PHS gates — blend_factor validation" begin
    using FastInterpolations.Experimental
    x = range(0.0, 2pi, 41)
    y = sin.(x)
    for bf in (-1.0, 0.0, NaN, Inf, 0.3, 0.5)
        @test_throws ArgumentError phs_interp((x,), y; blend_factor = bf)
    end
    mid = (x[10] + x[11]) / 2
    @test phs_interp((x,), y; blend_factor = 0.6)((mid,)) ≈ sin(mid) rtol = 1.0e-3

    x4 = range(0.0, 3.0, 7)
    d4 = [sum(sin, Tuple(I) .* step(x4) .- step(x4)) for I in CartesianIndices(ntuple(_ -> 7, 4))]
    @test_throws ArgumentError phs_interp(ntuple(_ -> x4, 4), d4; stencil_size = 3, blend_factor = 1.0)
    c4 = ntuple(_ -> (x4[3] + x4[4]) / 2, 4)
    @test phs_interp(ntuple(_ -> x4, 4), d4; stencil_size = 3, blend_factor = 1.2)(c4) ≈ sum(sin, c4) rtol = 1.0e-2

    # anisotropic spacing: hx = 1, hy = 0.25 → d_c = ½√(1 + 1/16) ≈ 0.5154 (in units of h_max)
    gx = range(0.0, 10.0, 11)
    gy = range(0.0, 2.5, 11)
    dxy = [a + 2b for a in gx, b in gy]
    @test_throws ArgumentError phs_interp((gx, gy), dxy; stencil_size = 3, blend_factor = 0.51)
    @test phs_interp((gx, gy), dxy; stencil_size = 3, blend_factor = 0.53)((4.5, 1.125)) ≈ 6.75 rtol = 1.0e-8

    # the default 1.0 leaves 4-D cell centres uncovered, so it is rejected there too
    @test_throws ArgumentError phs_interp(ntuple(_ -> x4, 4), d4; stencil_size = 3)

    # the blend radius in index units must fit an Int
    @test_throws ArgumentError phs_interp((x,), y; blend_factor = 1.0e20)
    x32 = range(0.0f0, 2.0f0 * Float32(pi), 41)
    @test_throws ArgumentError phs_interp((x32,), sin.(x32); blend_factor = 1.0e39)
end

# ── G8: real floating-point grids and data only ─────────────────────────────────
# Coefficient caches and pool buffers are typed by the grid float type, so complex,
# unitful or dual-number data cannot be evaluated.

@testitem "PHS gates — element types" begin
    using FastInterpolations.Experimental
    using Unitful, ForwardDiff
    x = range(0.0, 2pi, 30)
    y = cos.(x)
    @test_throws ArgumentError phs_interp((x,), complex.(y, y))
    @test_throws ArgumentError phs_interp((collect(x) .* u"m",), y)
    @test_throws ArgumentError phs_interp((x,), y .* u"K")
    @test_throws ArgumentError phs_interp((x,), ForwardDiff.Dual.(y, 1.0))
    # integer data, mixed float widths and Float32 grids are fine
    @test phs_interp((x,), round.(Int, 10 .* y))((1.0,)) isa Float64
    @test phs_interp((x,), Float32.(y))((1.0,)) ≈ cos(1.0) rtol = 1.0e-3
    x32 = range(0.0f0, 2.0f0 * Float32(pi), 30)
    @test phs_interp((x32,), cos.(x32))((1.0f0,)) ≈ cos(1.0f0) rtol = 1.0e-3
end

# ── G5/G6: extrapolation modes and batch domain checks ──────────────────────────
# Only NoExtrap, FillExtrap and InBounds are implemented; the others used to be
# accepted and return 0.0 out of domain. A call-time `extrap` override used to be
# swallowed silently, and the batch path skipped the NoExtrap domain check.

@testitem "PHS gates — unsupported extrapolation rejected at construction" begin
    using FastInterpolations.Experimental
    x = range(0.0, 2pi, 30)
    y = cos.(x)
    for e in (ClampExtrap(), WrapExtrap(), ExtendExtrap())
        @test_throws ArgumentError phs_interp((x,), y; extrap = e)
    end
    g = range(0.0, 1.0, 9)
    d = [a + b for a in g, b in g]
    @test_throws ArgumentError phs_interp((g, g), d; stencil_size = 4, extrap = (NoExtrap(), ClampExtrap()))
    for e in (NoExtrap(), FillExtrap(-7.0), InBounds(), (FillExtrap(0.0), NoExtrap()))
        @test phs_interp((g, g), d; stencil_size = 4, extrap = e)((0.3, 0.4)) ≈ 0.7
    end
end

# A call-time override follows the package-wide rule: only `InBounds()` (an in-domain
# fast path) is accepted; any other mode is an ArgumentError instead of being ignored.
@testitem "PHS gates — call-time extrap override: InBounds only, on every entry point" begin
    using FastInterpolations.Experimental
    x = range(0.0, 2pi, 30)
    itp = phs_interp((x,), cos.(x))
    g = range(0.0, 1.0, 9)
    itp2 = phs_interp((g, g), [a + b for a in g, b in g]; stencil_size = 4)
    xs = [0.5, 1.5]
    for e in (ClampExtrap(), FillExtrap(0.0), NoExtrap())
        @test_throws ArgumentError itp((1.0,); extrap = e)
        @test_throws ArgumentError itp(zeros(2), (xs,); extrap = e)
        @test_throws ArgumentError itp((xs,); extrap = e)
        @test_throws ArgumentError itp(GriddedQuery((xs,)); extrap = e)
        @test_throws ArgumentError itp2([0.3, 0.4]; extrap = e)
        @test_throws ArgumentError itp2(0.3, 0.4; extrap = e)
    end
    @test itp((1.0,); extrap = InBounds()) == itp((1.0,))
    @test itp((xs,); extrap = InBounds()) == itp((xs,))
    @test itp2((0.3, 0.4); extrap = InBounds()) == itp2((0.3, 0.4))
    @test itp((1.0,); extrap = nothing) ≈ cos(1.0) rtol = 1.0e-3
end

@testitem "PHS gates — batch queries check the domain like scalar queries" setup = [AllocConstants] begin
    using FastInterpolations.Experimental
    x = range(0.0, 2pi, 30)
    itp = phs_interp((x,), cos.(x))
    # the batch domain gate, like every method's, is elided under --check-bounds=no
    checks_on = Base.JLOptions().check_bounds != 2
    if checks_on
        @test_throws DomainError itp(zeros(3), ([0.5, 1.5, 8.0],))
        @test_throws DomainError itp(([0.5, 8.0],))
        @test_throws DomainError itp([(0.5,), (8.0,)])
    end
    g = range(0.0, 1.0, 9)
    itp2 = phs_interp((g, g), [a + b for a in g, b in g]; stencil_size = 4)
    checks_on && @test_throws DomainError itp2(([0.3, 0.4], [0.5, 1.2]))
    # FillExtrap batches still fill
    itpf = phs_interp((x,), cos.(x); extrap = FillExtrap(-7.0))
    @test itpf(([0.5, 8.0],)) ≈ [cos(0.5), -7.0] rtol = 1.0e-4

    # a rejected call leaves the pool clean: the next batch call does not allocate
    out = zeros(2)
    qs = ([0.5, 1.5],)
    itp(out, qs)
    itp(out, qs)
    checks_on && @test_throws DomainError itp(out, ([0.5, 8.0],))
    @test_throws ArgumentError itp(out, qs; extrap = ClampExtrap())
    @test (@allocations itp(out, qs)) <= ALLOC_THRESHOLD
end

# ── G7: derivative order ────────────────────────────────────────────────────────
# Derivatives of total order ≥ 3 are not implemented and used to return 0.0; a
# degree-3 PHS blended with an exponential bump has nonzero derivatives of every order.

@testitem "PHS gates — derivative order above 2 rejected" begin
    using FastInterpolations.Experimental
    x = range(0.0, 2pi, 30)
    itp = phs_interp((x,), sin.(x))
    @test_throws ArgumentError itp((1.0,); deriv = DerivOp(3))
    @test_throws ArgumentError itp(([0.5, 1.0],); deriv = DerivOp(3))
    @test_throws ArgumentError itp(zeros(2), ([0.5, 1.0],); deriv = DerivOp(4))
    @test_throws ArgumentError phs_interp((x,), sin.(x), (1.0,); deriv = DerivOp(3))
    @test_throws ArgumentError phs_interp((x,), 2.0 .+ sin.(x); log_reference = 1.0)((1.0,); deriv = DerivOp(3))
    g = range(0.0, 1.0, 9)
    itp2 = phs_interp((g, g), [a * b for a in g, b in g]; stencil_size = 4)
    @test_throws ArgumentError itp2((0.3, 0.4); deriv = (DerivOp(2), DerivOp(1)))
    g3 = range(0.0, 1.0, 6)
    itp3 = phs_interp((g3, g3, g3), [a + b + c for a in g3, b in g3, c in g3]; stencil_size = 3)
    @test_throws ArgumentError itp3((0.3, 0.4, 0.5); deriv = (DerivOp(1), DerivOp(1), DerivOp(1)))
    # orders up to 2 still evaluate
    @test itp2((0.3, 0.4); deriv = (DerivOp(1), DerivOp(1))) ≈ 1.0 atol = 0.1
    @test itp((1.0,); deriv = DerivOp(2)) ≈ -sin(1.0) atol = 5.0e-2
end

# ── Log transform with 2nd derivatives: blend candidates fit the kernel buffer ──
# The log-transform second-derivative kernel collects blend nodes into a fixed
# 27-slot buffer; a wider candidate box used to overflow it (BoundsError, or an
# out-of-bounds write with bounds checks off). Values and first derivatives are
# unaffected, as are second derivatives with blend_factor ≤ 1 in 3-D.

@testitem "PHS gates — log transform 2nd derivatives need ≤ 27 blend candidates" begin
    using FastInterpolations.Experimental
    x = range(0.0, 3.0, 13)
    data = [exp(-((a - 1.5)^2 + (b - 1.5)^2 + (c - 1.5)^2)) + 0.5 for a in x, b in x, c in x]
    q = (1.3, 1.37, 1.21)
    d2 = (DerivOp(2), DerivOp(0), DerivOp(0))
    wide = phs_interp((x, x, x), data; stencil_size = 4, blend_factor = 2.0, log_reference = 1.0)
    @test_throws ArgumentError wide(q; deriv = d2)
    @test_throws ArgumentError wide(zeros(1), ([q[1]], [q[2]], [q[3]]); deriv = d2)
    # the one-shot forms default to blend_factor = 2.0
    @test_throws ArgumentError phs_interp((x, x, x), data, q; stencil_size = 4, log_reference = 1.0, deriv = d2)
    @test isfinite(wide(q))
    @test isfinite(wide(q; deriv = (DerivOp(1), DerivOp(0), DerivOp(0))))
    narrow = phs_interp((x, x, x), data; stencil_size = 4, log_reference = 1.0)
    @test isfinite(narrow(q; deriv = d2))
    # without the log transform the kernel has no buffer and wide blends are fine
    @test isfinite(phs_interp((x, x, x), data; stencil_size = 4, blend_factor = 2.0)(q; deriv = d2))
end
