# ========================================
# Cache bank duck widening — refac/duck-thomas Phase 2b contracts
# ========================================
# The autocache pool banks any `<:Number` grid eltype whose `isequal` is
# grid-faithful (`_grid_bankable`, default true) — unit grids and Duals
# included. Banks are segregated by ENTRY TYPE (exact grid eltype), so
# cross-unit `isequal(1.0m, 100.0cm) == true` can never produce a cross-typed
# hit, and the pool's linear isequal scan never touches `hash` (Unitful and
# ForwardDiff both bend the isequal/hash contract, in opposite directions).

@testitem "cache bank: unit grids bank and hit (Vector + Range)" begin
    using Unitful
    const FI = FastInterpolations

    yw = [1.0, 2.0, 0.5, 3.0, 2.5] .* u"W"

    @testset "Vector grid" begin
        FI.clear_cubic_cache!()
        xu = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
        i1 = cubic_interp(xu, yw)
        i2 = cubic_interp(xu, yw)
        @test i2.cache === i1.cache            # pass-1: objectid hint
        xu_fresh = [0.0, 1.0, 2.5, 3.0, 4.5] .* u"s"
        i3 = cubic_interp(xu_fresh, yw)
        @test i3.cache === i1.cache            # pass-2: isequal (fresh array)
        i4 = cubic_interp(xu, yw; autocache = false)
        @test i4.cache !== i1.cache            # explicit opt-out stays fresh
        @test i4(2.2u"s") === i1(2.2u"s")      # values agree either way
    end

    @testset "Range grid (_CachedRange bank)" begin
        FI.clear_cubic_cache!()
        yr = [1.0, 2.0, 0.5, 3.0, 2.5, 1.0, 4.0, 2.0, 3.5, 0.5] .* u"W"
        r1 = cubic_interp(1.0u"s":1.0u"s":10.0u"s", yr)
        r2 = cubic_interp(1.0u"s":1.0u"s":10.0u"s", yr)
        @test r2.cache === r1.cache            # isbits range: deterministic key
    end
end

@testitem "cache bank: cross-unit isolation (m vs cm, type-stable returns)" begin
    using Unitful
    const FI = FastInterpolations

    # `isequal([1.0m], [100.0cm])` is TRUE in Unitful — the pool must never
    # compare them: different eltypes land in different (type-keyed) banks.
    # (Passes pre-widening too — this is the regression guard for the widened
    # pool, pinning that a build's return type depends only on its inputs.)
    FI.clear_cubic_cache!()
    y = [1.0, 2.0, 0.5, 3.0] .* u"W"
    xm = [0.0, 1.0, 2.0, 3.0] .* u"m"
    xcm = [0.0, 100.0, 200.0, 300.0] .* u"cm"   # same physical values
    @test isequal(xm, xcm)                       # the trap is real…
    im_ = cubic_interp(xm, y)
    icm = cubic_interp(xcm, y)
    @test im_.cache !== icm.cache                # …and never taken
    @test eltype(im_.cache.x) === typeof(1.0u"m")
    @test eltype(icm.cache.x) === typeof(1.0u"cm")
end

@testitem "cache bank: Dual grids bank by full isequal (primal + partials)" begin
    using ForwardDiff
    using ForwardDiff: Dual
    const FI = FastInterpolations

    # ForwardDiff's array `isequal` distinguishes partials (probe-verified), so
    # banking Duals is CORRECT: same-seed rebuilds hit, different seeds miss.
    # This pin watches that upstream semantic — if ForwardDiff ever made
    # `isequal` primal-only, the miss case would silently become a wrong-hit.
    FI.clear_cubic_cache!()
    mk(p) = [Dual(0.0, p), Dual(1.0, 0.0), Dual(2.5, 0.0), Dual(3.0, 0.0)]
    y = [1.0, 2.0, 0.5, 3.0]

    x1 = mk(1.0)
    d1 = cubic_interp(x1, y)
    d2 = cubic_interp(x1, y)
    @test d2.cache === d1.cache                # pass-1: same array
    d3 = cubic_interp(mk(1.0), y)
    @test d3.cache === d1.cache                # pass-2: same primal AND partials
    d4 = cubic_interp(mk(2.0), y)
    @test d4.cache !== d1.cache                # same primal, different seed → miss
    @test d4.z != d1.z                         # …and the partials really differ
end
