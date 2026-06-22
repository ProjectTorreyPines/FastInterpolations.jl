# ═══════════════════════════════════════════════════════════════════════════════
# test_dual_grid_coord_promotion.jl
#
# Coordinate-promotion contract for a DUAL GRID (AD-with-respect-to-grid: the grid
# nodes carry partials). A plain-Float query on a Dual grid MUST promote to a
# concrete Dual coordinate — never leave a Union{Float,Dual} (which leaks to a
# public `Any` for the Hermite family ND). Complements test_query_grid_promotion.jl
# (Float/Int grids). Pins design invariants I1 (concrete), I2 (coordinate rides the
# grid), I6 (duck-friendly).
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "Dual grid (AD-w.r.t-grid) — ND _handle_all_extraps stays concrete" begin
    using ForwardDiff: Dual
    using FastInterpolations: _handle_all_extraps

    # Float-backed Dual grid: each node carries a unit partial (∂node/∂param).
    mkdual(r) = [Dual{Nothing}(Float64(v), 1.0) for v in r]
    gd = (mkdual(0.5:1.0:9.5), mkdual(0.5:1.0:9.5))

    exs = (
        NoExtrap(), ClampExtrap(), FillExtrap(fill_value = 0.0),
        ExtendExtrap(), WrapExtrap(), InBounds(),
    )
    # Float query on a Dual grid: the returned coordinate tuple must be CONCRETE
    # (no Union{Float64, Dual} per axis).  RED before the rule fix.
    for ex in exs
        e2 = (ex, ex)
        rt = Base.return_types(_handle_all_extraps, Tuple{Tuple{Float64, Float64}, typeof(gd), typeof(e2)})
        @test length(rt) == 1 && isconcretetype(rt[1])
    end
end

@testitem "Dual grid (AD-w.r.t-grid) — public ND return concrete, no `Any` leak (all methods)" begin
    using ForwardDiff: Dual
    mkdual(r) = [Dual{Nothing}(Float64(v), 1.0) for v in r]
    g = mkdual(0.5:1.0:9.5)
    data = [Float64(i + j) for i in 1:10, j in 1:10]
    builders = (
        linear_interp, constant_interp, cubic_interp,
        pchip_interp, cardinal_interp, akima_interp,
    )
    exs = (ClampExtrap(), FillExtrap(fill_value = 0.0), ExtendExtrap(), WrapExtrap())
    for mk in builders, ex in exs
        itp = mk((g, g), data; extrap = ex)
        rt = Base.return_types(itp, Tuple{Float64, Float64})
        @test length(rt) == 1 && isconcretetype(rt[1])     # I1: concrete, no Any
        @test (@inferred itp(3.0, 4.0)) isa Dual           # I2: coordinate rides the grid
    end
end

@testitem "Dual grid (AD-w.r.t-grid) — 1D return concrete (all methods)" begin
    using ForwardDiff: Dual
    mkdual(r) = [Dual{Nothing}(Float64(v), 1.0) for v in r]
    g = mkdual(0.5:1.0:9.5)
    y = collect(Float64, 1:10)
    builders = (
        linear_interp, constant_interp, cubic_interp, quadratic_interp,
        pchip_interp, cardinal_interp, akima_interp,
    )
    exs = (ClampExtrap(), FillExtrap(fill_value = 0.0), ExtendExtrap(), WrapExtrap())
    for mk in builders, ex in exs
        itp = mk(g, y; extrap = ex)
        rt = Base.return_types(itp, Tuple{Float64})
        @test length(rt) == 1 && isconcretetype(rt[1])
        @test (@inferred itp(3.0)) isa Dual                # in-bounds
        @test (@inferred itp(-5.0)) isa Dual               # OOB (extrap)
    end
end

@testitem "AD-w.r.t-grid gradient is correct (end-to-end ForwardDiff through the grid)" begin
    using ForwardDiff
    # Differentiate the interpolated value w.r.t. a uniform grid shift δ: x = x0 .+ δ.
    x0 = collect(0.5:1.0:9.5)
    y = collect(Float64, 1:10)
    f(δ) = linear_interp(x0 .+ δ, y; extrap = ClampExtrap())(3.0)
    d = ForwardDiff.derivative(f, 0.0)
    # Linear interp on a unit-spaced grid: shifting the grid by δ moves the sample
    # point relative to the grid by -δ, so df/dδ = -(local slope). Slope here is 1.0.
    @test isfinite(d)
    @test d ≈ -1.0 rtol = 1.0e-9
end

# ── Perf invariant I5: Float64 hot path is a compile-time no-op + zero alloc ──
# The eval-surface promotion (`map(_promote_coord, query, map(eltype, grids))`)
# and the per-method `_promote_coord(xq, eltype(x))` must be identity on Float64
# and must not allocate. Int grids must stay Int (not over-promoted).
@testitem "Float64 hot path is a compile-time no-op + zero alloc (I5)" begin
    using FastInterpolations: _coord_eltype

    # _coord_eltype(::Type{Tq}, ::Type{Tg}) — type-level identities.
    @test _coord_eltype(Float64, Float64) === Float64
    @test _coord_eltype(Int, Float64) === Float64      # Float grid + Int query floats (bonus precursor)
    @test _coord_eltype(Int, Int) === Int              # Int grid stays Int (kernel floats the output)
    @test _coord_eltype(Float32, Float64) === Float64

    # Zero-alloc on the scalar eval hot path — inside a function barrier (NOT a
    # @testset: @testset's try/catch makes locals unstable and pollutes @allocated).
    function _alloc_probe()
        g = 0.5:1.0:9.5
        y = collect(Float64, 1:10)
        data = [Float64(i + j) for i in 1:10, j in 1:10]
        itp1 = cubic_interp(g, y; extrap = ClampExtrap())
        itpN = cubic_interp((g, g), data; extrap = ClampExtrap())
        itp1(3.0); itpN(3.0, 4.0); itp1(3)            # warmup / compile
        a1 = @allocated itp1(3.0)
        aN = @allocated itpN(3.0, 4.0)
        a1i = @allocated itp1(3)                       # Int query promotes → must still be 0
        aNi = @allocated itpN(3, 4)                    # ND Int query
        return a1, aN, a1i, aNi
    end
    a1, aN, a1i, aNi = _alloc_probe()
    @test a1 == 0
    @test aN == 0
    @test a1i == 0
    @test aNi == 0
end

# ── Batch on a Dual grid: the allocating buffer is Vector{To}, so values
# convert-on-store → a concrete Vector{Dual} (no escaping Union). Locks that the
# batch path is correct for AD-w.r.t-grid (1D + ND), matching the scalar path. ──
@testitem "Dual grid (AD-w.r.t-grid) — batch returns concrete Vector{Dual} (1D + ND)" begin
    using ForwardDiff: Dual, value
    mkdual(r) = [Dual{Nothing}(Float64(v), 1.0) for v in r]
    g = mkdual(0.5:1.0:9.5)
    y = collect(Float64, 1:10)
    data = [Float64(i + j) for i in 1:10, j in 1:10]

    builders1d = (
        linear_interp, cubic_interp, constant_interp, quadratic_interp,
        pchip_interp, cardinal_interp, akima_interp,
    )
    for mk in builders1d, ex in (ClampExtrap(), FillExtrap(fill_value = 0.0))
        itp = mk(g, y; extrap = ex)
        out = itp([3.0, 4.0, -5.0, 15.0])        # in-domain + OOB-left + OOB-right
        @test out isa Vector && isconcretetype(eltype(out)) && eltype(out) <: Dual
        @test length(out) == 4
        @test value(out[1]) ≈ value(itp(3.0))    # batch == scalar (primal)
        @test value(out[3]) ≈ value(itp(-5.0))   # OOB primal matches
    end

    buildersnd = (
        linear_interp, cubic_interp, constant_interp,
        pchip_interp, cardinal_interp, akima_interp,
    )
    for mk in buildersnd, ex in (ClampExtrap(), FillExtrap(fill_value = 0.0))
        itp = mk((g, g), data; extrap = ex)
        xs = [3.0, -5.0, 15.0]; ys = [4.0, 4.0, 4.0]   # SoA: tuple of coordinate vectors
        out = itp((xs, ys))
        @test out isa Vector && isconcretetype(eltype(out)) && eltype(out) <: Dual
        @test length(out) == 3
        @test value(out[1]) ≈ value(itp(3.0, 4.0))
    end
end

# ── Derived quantities on a Dual grid: vector calculus (ND) must return a
# concrete carrier (the audit flagged hand-coded promote_type in vector_calculus.jl).
# RED would show Union/Any or a wrong eltype.
# NOTE: `integrate`/`cumulative_integrate` are constrained to `Tg <: AbstractFloat`,
# so a Dual grid throws a clean `ArgumentError` (an unimplemented-feature gap, NOT a
# leak — no silent Union/Any). AD-w.r.t-grid through an integral is out of scope here. ──
@testitem "Dual grid (AD-w.r.t-grid) — vector calculus stays concrete; integrate gap is a clean throw" begin
    using ForwardDiff: Dual
    mkdual(r) = [Dual{Nothing}(Float64(v), 1.0) for v in r]
    g = mkdual(0.5:1.0:9.5)
    y = collect(Float64, 1:10)
    data = [Float64(i + j) for i in 1:10, j in 1:10]

    # ND gradient / hessian / laplacian on a Dual grid → concrete Dual carrier.
    for mk in (linear_interp, cubic_interp)
        itp = mk((g, g), data)
        grad = gradient(itp, (3.0, 4.0))
        @test isconcretetype(eltype(grad)) && eltype(grad) <: Dual
        H = hessian(itp, (3.0, 4.0))
        @test isconcretetype(eltype(H)) && eltype(H) <: Dual
        L = laplacian(itp, (3.0, 4.0))
        @test L isa Dual && isconcretetype(typeof(L))
    end

    # integrate on a Dual grid: a clean ArgumentError (feature gap), not a silent leak.
    itp1 = linear_interp(g, y)
    @test_throws ArgumentError integrate(itp1, 1.0, 5.0)
end

# ── Quadratic ND completeness (the public-ND test above omits quadratic). ──
@testitem "Dual grid (AD-w.r.t-grid) — quadratic ND concrete" begin
    using ForwardDiff: Dual
    mkdual(r) = [Dual{Nothing}(Float64(v), 1.0) for v in r]
    g = mkdual(0.5:1.0:9.5)
    data = [Float64(i + j) for i in 1:10, j in 1:10]
    for ex in (ClampExtrap(), FillExtrap(fill_value = 0.0), ExtendExtrap())
        itp = quadratic_interp((g, g), data; extrap = ex)
        rt = Base.return_types(itp, Tuple{Float64, Float64})
        @test length(rt) == 1 && isconcretetype(rt[1])
        @test (@inferred itp(3.0, 4.0)) isa Dual
    end
end
