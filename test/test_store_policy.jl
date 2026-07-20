# Tests for StorePolicy (copy vs reference / zero-copy storage).
# Coverage: linear/constant (1D + ND), cubic 1D, local-Hermite family
# (akima/pchip/cardinal/cubic-hermite) 1D, hetero ND OnTheFly. Reference mode is
# type-transparent: aliases when the stored eltype is unchanged (Float/Range/dense
# -> zero-copy); non-float falls back to copy.

@testitem "Store Policy - tag, factory, helpers" begin
    using FastInterpolations: StorePolicy, copies_grid, copies_values,
        _own_or_ref_axis, _own_or_ref_values, _own_or_ref_data, _cache_axis

    @testset "tag and factory" begin
        @test StorePolicy() === StorePolicy{true, true, true}()
        @test StorePolicy(copy = false) === StorePolicy{false, false, true}()
        @test StorePolicy(copy_values = false) === StorePolicy{true, false, true}()
        @test StorePolicy(copy_grid = false) === StorePolicy{false, true, true}()
        @test StorePolicy(copy = false, copy_grid = true) === StorePolicy{true, false, true}()

        @test copies_grid(StorePolicy()) === true
        @test copies_grid(StorePolicy(copy = false)) === false
        @test copies_values(StorePolicy()) === true
        @test copies_values(StorePolicy(copy = false)) === false
    end

    @testset "value helper: copy vs alias vs eltype-mismatch fallback" begin
        y = [1.0, 2.0, 3.0]
        yc = _own_or_ref_values(y, Float64, StorePolicy())
        @test yc == y && yc !== y                                   # copy → fresh
        @test _own_or_ref_values(y, Float64, StorePolicy(copy = false)) === y   # ref + match → alias
        yi = [1, 2, 3]
        yf = _own_or_ref_values(yi, Float64, StorePolicy(copy = false))
        @test yf == [1.0, 2.0, 3.0] && eltype(yf) === Float64 && yf !== yi      # ref + mismatch → copy
    end

    @testset "ND data helper: dense/view alias vs copy" begin
        d = [1.0 2.0; 3.0 4.0]
        @test _own_or_ref_data(d, StorePolicy(copy = false)) === d
        @test _own_or_ref_data(d, StorePolicy()) == d && _own_or_ref_data(d, StorePolicy()) !== d
        dv = @view d[:, :]
        @test _own_or_ref_data(dv, StorePolicy(copy = false)) === dv            # ref aliases the view
        @test _own_or_ref_data(dv, StorePolicy()) isa Array{Float64, 2}         # copy materializes
    end

    @testset "grid axis helper: alias vs own-copy" begin
        x = [0.0, 1.0, 2.0, 3.0]
        xc = _cache_axis(x, NoBC(), Float64)
        @test _own_or_ref_axis(xc, Float64, StorePolicy(copy = false)) === xc
        @test _own_or_ref_axis(xc, Float64, StorePolicy()) !== xc
    end
end

@testitem "Store Policy - 1D reference" begin
    x = collect(range(0.0, 1.0, 60))
    y = sin.(2π .* x)
    qs = range(0.05, 0.95, 31)

    @testset "linear (alias + type-stable + alloc saved)" begin
        ic = linear_interp(x, y)
        ir = linear_interp(x, y; store = StorePolicy(copy = false))
        @test ir.y === y                      # value aliased
        @test ir.x.inner === x                # grid inner aliased too (Vector grid, copy_grid=false)
        @test ic.y !== y
        @test ic.x.inner !== x                # default owns a private grid copy
        @test typeof(ic) === typeof(ir)
        @test all(ic(q) ≈ ir(q) for q in qs)

        build_ref(xx, yy) = linear_interp(xx, yy; store = StorePolicy(copy = false))
        @test (@inferred build_ref(x, y)) isa LinearInterpolant
        build_ref(x, y)
        @test (@allocated build_ref(x, y)) < (@allocated linear_interp(x, y))
    end

    @testset "constant" begin
        ic = constant_interp(x, y)
        ir = constant_interp(x, y; store = StorePolicy(copy = false))
        @test ir.y === y && ic.y !== y
        @test typeof(ic) === typeof(ir)
        @test all(ic(q) ≈ ir(q) for q in qs)
    end

    @testset "cubic (value-ref; grid stays owned via cache)" begin
        ic = cubic_interp(x, y)
        ir = cubic_interp(x, y; store = StorePolicy(copy = false))
        @test ir.y === y && ic.y !== y
        @test typeof(ic) === typeof(ir)
        @test all(ic(q) ≈ ir(q) for q in qs)
        @test all(ic(q; deriv = DerivOp(1)) ≈ ir(q; deriv = DerivOp(1)) for q in qs)

        # parametric y::Y → can alias a view (Phase 3 generalization)
        yv = @view y[:]
        irv = cubic_interp(x, yv; store = StorePolicy(copy = false))
        @test irv.y === yv && irv.y isa SubArray
        @test all(irv(q) ≈ ic(q) for q in qs)
    end

    @testset "local-Hermite slope family (akima/pchip/cardinal)" begin
        for f in (akima_interp, pchip_interp, cardinal_interp)
            ic = f(x, y)
            ir = f(x, y; store = StorePolicy(copy = false))
            @test ir.y === y && ic.y !== y
            @test typeof(ic) === typeof(ir)
            @test all(ic(q) ≈ ir(q) for q in qs)
        end
    end

    @testset "cubic-hermite (user-supplied dy)" begin
        dy = (2π) .* cos.(2π .* x)
        hc = hermite_interp(x, y, dy)
        hr = hermite_interp(x, y, dy; store = StorePolicy(copy = false))
        @test hr.y === y && hc.y !== y
        @test typeof(hc) === typeof(hr)
        @test all(hc(q) ≈ hr(q) for q in qs)
    end

    @testset "slope family via OnTheFly coeffs (alias on OnTheFly inner ctor)" begin
        # Default coeffs route through the PreCompute (dy) inner ctor; coeffs=OnTheFly()
        # routes through the distinct OnTheFly inner ctor, whose own ref branch needs
        # exercising with an aliased value vector.
        for f in (akima_interp, pchip_interp, cardinal_interp)
            ic = f(x, y; coeffs = OnTheFly())
            ir = f(x, y; coeffs = OnTheFly(), store = StorePolicy(copy = false))
            @test ir.y === y && ic.y !== y
            @test typeof(ic) === typeof(ir)
            @test all(ic(q) ≈ ir(q) for q in qs)
        end
    end

    @testset "cubic periodic builder (value-ref through periodic path)" begin
        # Non-periodic cubic ref is covered above; the periodic builder is a separate
        # store-threaded path (_build_interpolant_periodic).
        xp = collect(range(0.0, 1.0, 40))
        yp = sin.(2π .* xp)
        yp[end] = yp[1]                       # periodic-consistent endpoints
        ic = cubic_interp(xp, yp; bc = PeriodicBC())
        ir = cubic_interp(xp, yp; bc = PeriodicBC(), store = StorePolicy(copy = false))
        @test typeof(ic) === typeof(ir)
        @test all(ic(q) ≈ ir(q) for q in range(0.05, 0.95, 25))
    end
end

@testitem "Store Policy - quadratic 1D reference" begin
    # Quadratic was added to StorePolicy alongside the other 1D methods so the
    # unified `interp(x, y; method=QuadraticInterp())` can build copy-free too.
    x = collect(range(0.0, 1.0, 60))
    y = @. sin(2π * x) + 0.2 * x
    qs = range(0.05, 0.95, 31)

    ic = quadratic_interp(x, y)
    ir = quadratic_interp(x, y; store = StorePolicy(copy = false))
    @test ir.y === y                       # value aliased
    @test ir.x.inner === x                 # grid inner aliased (Vector grid, copy_grid=false)
    @test ic.y !== y
    @test ic.x.inner !== x                  # default owns a private grid copy
    @test typeof(ic) === typeof(ir)
    @test all(ic(q) ≈ ir(q) for q in qs)
    @test all(ic(q; deriv = DerivOp(1)) ≈ ir(q; deriv = DerivOp(1)) for q in qs)

    build_ref(xx, yy) = quadratic_interp(xx, yy; store = StorePolicy(copy = false))
    @test (@inferred build_ref(x, y)) isa QuadraticInterpolant
    build_ref(x, y)
    @test (@allocated build_ref(x, y)) < (@allocated quadratic_interp(x, y))

    # parametric y::Y → can alias a view
    yv = @view y[:]
    irv = quadratic_interp(x, yv; store = StorePolicy(copy = false))
    @test irv.y === yv && irv.y isa SubArray
    @test all(irv(q) ≈ ic(q) for q in qs)
end

@testitem "Store Policy - unified 1D interp(x, y; method, store)" begin
    using FastInterpolations: LinearInterp, ConstantInterp, QuadraticInterp,
        CubicInterp, PchipInterp, CardinalInterp, AkimaInterp

    x = collect(range(0.0, 1.0, 50))
    y = @. cos(3x) + 0.1 * x
    qs = range(0.05, 0.95, 25)
    ref = StorePolicy(copy = false)

    # Every 1D method routed by the unified API must thread `store` through to its
    # dedicated constructor and alias the user's value vector (zero-copy).
    for m in (
            LinearInterp(), ConstantInterp(), QuadraticInterp(), CubicInterp(),
            PchipInterp(), CardinalInterp(), AkimaInterp(),
        )
        ic = interp(x, y; method = m)
        ir = interp(x, y; method = m, store = ref)
        @test ir.y === y                    # value aliased through the unified API
        @test ic.y !== y                    # default still owns a private copy
        @test typeof(ic) === typeof(ir)
        @test all(ic(q) ≈ ir(q) for q in qs)
    end
end

@testitem "Store Policy - ND reference" begin
    using FastInterpolations: CubicInterp, LinearInterp

    @testset "linear ND (image: Range axes + dense data)" begin
        m, n = 40, 60
        data = [sin(i / 7) * cos(j / 9) for i in 1:m, j in 1:n]
        grids = (1:m, 1:n)
        ic = linear_interp(grids, data)
        ir = linear_interp(grids, data; store = StorePolicy(copy = false))
        pts = [(3.4, 5.6), (10.2, 41.9), (39.0, 1.0), (1.0, 60.0)]
        @test all(ic(p) ≈ ir(p) for p in pts)
        @test ir.data === data && ic.data !== data
        @test typeof(ic) === typeof(ir)
        @test ic((10.2, 41.9); deriv = DerivOp(1, 0)) ≈ ir((10.2, 41.9); deriv = DerivOp(1, 0))
        build_ref(g, d) = linear_interp(g, d; store = StorePolicy(copy = false))
        build_ref(grids, data)
        @test (@allocated build_ref(grids, data)) < (@allocated linear_interp(grids, data))
    end

    @testset "constant ND" begin
        m, n = 32, 24
        data = Float64[i + 2j for i in 1:m, j in 1:n]
        grids = (1:m, 1:n)
        ic = constant_interp(grids, data)
        ir = constant_interp(grids, data; store = StorePolicy(copy = false))
        pts = [(3.4, 5.6), (10.2, 19.9), (32.0, 1.0)]
        @test all(ic(p) ≈ ir(p) for p in pts)
        @test ir.data === data && ic.data !== data
        @test typeof(ic) === typeof(ir)
    end

    @testset "view/SubArray aliasing (parametric data::D)" begin
        big = rand(50, 50)
        dview = @view big[1:40, 1:30]
        grids = (collect(1.0:40.0), collect(1.0:30.0))
        pts = [(3.4, 5.6), (10.2, 19.9), (39.0, 1.0)]
        lo, hi = (3.0, 4.0), (35.0, 25.0)

        itp = linear_interp(grids, dview; store = StorePolicy(copy = false))
        @test itp.data === dview && itp.data isa SubArray
        base = linear_interp(grids, collect(dview))
        @test all(itp(p) ≈ base(p) for p in pts)
        @test itp((10.2, 19.9); deriv = DerivOp(1, 0)) ≈ base((10.2, 19.9); deriv = DerivOp(1, 0))
        @test integrate(itp, lo, hi) ≈ integrate(base, lo, hi)

        citp = constant_interp(grids, dview; store = StorePolicy(copy = false))
        cbase = constant_interp(grids, collect(dview))
        @test citp.data === dview
        @test all(citp(p) ≈ cbase(p) for p in pts)
        @test integrate(citp, lo, hi) ≈ integrate(cbase, lo, hi)

        @test linear_interp(grids, dview).data isa Array{Float64, 2}   # copy still materializes
    end

    @testset "hetero ND OnTheFly (data alias)" begin
        m, n = 30, 24
        data = rand(m, n)
        grids = (collect(1.0:m), collect(1.0:n))
        meth = (CubicInterp(), LinearInterp())
        ic = interp(grids, data; method = meth, coeffs = OnTheFly())
        ir = interp(grids, data; method = meth, coeffs = OnTheFly(), store = StorePolicy(copy = false))
        @test ir.data === data && ic.data !== data
        pts = [(3.4, 5.6), (10.2, 19.9), (29.0, 1.0)]
        @test all(ic(p) ≈ ir(p) for p in pts)
    end

    @testset "unsupported ND reference → warn + copy (and cubic-ND OnTheFly honors)" begin
        g = (1:10, 1:10)
        d = rand(10, 10)
        ref = StorePolicy(copy = false)
        # cubic ND PreCompute: cannot alias (derived partials) → warns once, still correct
        @test cubic_interp(g, d; store = ref)((3.3, 4.4)) ≈ cubic_interp(g, d)((3.3, 4.4))
        # cubic ND OnTheFly routes through hetero → data-ref honored
        @test cubic_interp(g, d; coeffs = OnTheFly(), store = ref).data === d
        # interp(...; PreCompute) homogeneous dispatch: warns once, still correct
        @test interp(g, d; method = CubicInterp(), coeffs = PreCompute(), store = ref)((3.3, 4.4)) ≈
            interp(g, d; method = CubicInterp(), coeffs = PreCompute())((3.3, 4.4))
    end
end

@testitem "Store Policy - cross-cutting" begin
    @testset "Int input → type-transparent copy fallback" begin
        xi = collect(0:9)
        yi = collect(10:19)
        ic = linear_interp(xi, yi)
        ir = linear_interp(xi, yi; store = StorePolicy(copy = false))
        @test typeof(ic) === typeof(ir)
        @test all(ic(q) ≈ ir(q) for q in 0.5:1.0:8.5)
    end

    @testset "mixed component: alias values, copy grid" begin
        x = collect(range(0.0, 1.0, 32))
        y = exp.(x)
        itp = linear_interp(x, y; store = StorePolicy(copy_values = false))
        @test itp.y === y
        plain = linear_interp(x, y)
        @test all(itp(q) ≈ plain(q) for q in 0.1:0.1:0.9)
    end

    @testset ":exclusive PeriodicBC degrades to copy, still correct" begin
        xp = collect(1.0:6.0)
        yp = [1.0, 2.0, 3.0, 4.0, 3.0, 1.0]
        ir = linear_interp(
            xp, yp; bc = PeriodicBC(endpoint = :exclusive, period = 6.0),
            store = StorePolicy(copy = false)
        )
        icp = linear_interp(xp, yp; bc = PeriodicBC(endpoint = :exclusive, period = 6.0))
        @test all(ir(q) ≈ icp(q) for q in 1.5:1.0:5.5)
    end

    @testset "1D view aliasing (parametric Y)" begin
        ybig = sin.(range(0.0, 2.0, 128))
        yview = @view ybig[1:64]
        xv = collect(range(0.0, 1.0, 64))
        itp = linear_interp(xv, yview; store = StorePolicy(copy = false))
        @test itp.y === yview
        vc = linear_interp(xv, collect(yview))
        @test all(itp(q) ≈ vc(q) for q in 0.05:0.1:0.95)
    end
end

@testitem "Store Policy - slope-family inner constructors" begin
    # Direct coverage of every inner ctor of akima/pchip/cardinal under both copy
    # (default) and reference store. The `::Type{PreCompute}` inner ctor has no
    # public factory route (the *_interp factory builds slopes inline and calls the
    # `dy` ctor), so it needs a direct unit test; the `dy` and OnTheFly ctors are
    # exercised explicitly here too so each path is unambiguously covered.
    using FastInterpolations: AkimaInterpolant1D, PchipInterpolant1D, CardinalInterpolant1D

    x = collect(range(0.0, 1.0, 24))
    y = @. sin(2π * x) + 0.3 * x^2
    qs = range(0.03, 0.97, 23)
    ref = StorePolicy(copy = false)

    @testset "akima" begin
        fac = akima_interp(x, y)
        # PreCompute inner ctor (auto slopes) — copy + reference
        @test all(AkimaInterpolant1D(x, y, PreCompute, NoExtrap(), AutoSearch())(q) ≈ fac(q) for q in qs)
        pcr = AkimaInterpolant1D(x, y, PreCompute, NoExtrap(), AutoSearch(); store = ref)
        @test pcr.y === y && all(pcr(q) ≈ fac(q) for q in qs)
        # dy inner ctor (caller-supplied slopes) — copy + reference
        @test all(AkimaInterpolant1D(x, y, fac.dy, NoExtrap(), AutoSearch())(q) ≈ fac(q) for q in qs)
        @test AkimaInterpolant1D(x, y, fac.dy, NoExtrap(), AutoSearch(); store = ref).y === y
        # OnTheFly inner ctor (via factory route) — copy + reference
        @test all(akima_interp(x, y; coeffs = OnTheFly())(q) ≈ fac(q) for q in qs)
        @test akima_interp(x, y; coeffs = OnTheFly(), store = ref).y === y
    end

    @testset "pchip" begin
        fac = pchip_interp(x, y)
        @test all(PchipInterpolant1D(x, y, PreCompute, NoExtrap(), AutoSearch())(q) ≈ fac(q) for q in qs)
        pcr = PchipInterpolant1D(x, y, PreCompute, NoExtrap(), AutoSearch(); store = ref)
        @test pcr.y === y && all(pcr(q) ≈ fac(q) for q in qs)
        @test all(PchipInterpolant1D(x, y, fac.dy, NoExtrap(), AutoSearch())(q) ≈ fac(q) for q in qs)
        @test PchipInterpolant1D(x, y, fac.dy, NoExtrap(), AutoSearch(); store = ref).y === y
        @test all(pchip_interp(x, y; coeffs = OnTheFly())(q) ≈ fac(q) for q in qs)
        @test pchip_interp(x, y; coeffs = OnTheFly(), store = ref).y === y
    end

    @testset "cardinal" begin
        fac = cardinal_interp(x, y)                       # tension = 0.0 default
        @test all(CardinalInterpolant1D(x, y, PreCompute, NoExtrap(), AutoSearch(), 0.0)(q) ≈ fac(q) for q in qs)
        pcr = CardinalInterpolant1D(x, y, PreCompute, NoExtrap(), AutoSearch(), 0.0; store = ref)
        @test pcr.y === y && all(pcr(q) ≈ fac(q) for q in qs)
        @test all(CardinalInterpolant1D(x, y, fac.dy, NoExtrap(), AutoSearch(), 0.0)(q) ≈ fac(q) for q in qs)
        @test CardinalInterpolant1D(x, y, fac.dy, NoExtrap(), AutoSearch(), 0.0; store = ref).y === y
        @test all(cardinal_interp(x, y; coeffs = OnTheFly())(q) ≈ fac(q) for q in qs)
        @test cardinal_interp(x, y; coeffs = OnTheFly(), store = ref).y === y
    end
end

# cache_axis=false: store the axis RAW — no `_CachedVector` h/inv_h precompute
# (n-1 divisions + 2 allocations). Effective for Vector axes only: a Range's
# `_CachedRange` is a stack-only scalar struct with nothing to skip. Eval on a
# raw axis falls back to on-the-fly h/inv(h); intended for one-shot /
# integrate-heavy use where construction cost dominates.
@testitem "Store Policy - cache_axis (raw axis storage)" begin
    using FastInterpolations: StorePolicy, caches_axis, copies_grid, copies_values,
        _grid_1d, _CachedRange, _CachedVector

    @testset "tag, factory, accessor" begin
        @test StorePolicy() === StorePolicy{true, true, true}()
        @test StorePolicy(cache_axis = false) === StorePolicy{true, true, false}()
        @test StorePolicy(copy = false) === StorePolicy{false, false, true}()
        @test StorePolicy(copy = false, cache_axis = false) === StorePolicy{false, false, false}()
        @test caches_axis(StorePolicy()) === true
        @test caches_axis(StorePolicy(cache_axis = false)) === false
    end

    x = [0.0, 0.5, 1.5, 3.0, 4.0]                    # non-uniform → per-cell h matters
    y = [1.0, 2.0, 4.0, 8.0, 16.0]

    @testset "vector axis: raw storage" begin
        itp_alias = linear_interp(x, y; store = StorePolicy(copy = false, cache_axis = false))
        @test _grid_1d(itp_alias) === x               # pure pass-through
        itp_owned = linear_interp(x, y; store = StorePolicy(cache_axis = false))
        g = _grid_1d(itp_owned)
        @test g isa Vector{Float64} && !(g isa _CachedVector)
        @test g == x && g !== x                       # owned copy, still raw
    end

    @testset "range axis: keeps free _CachedRange" begin
        xr = range(0.0, 4.0, length = 5)
        itp = linear_interp(xr, y; store = StorePolicy(cache_axis = false))
        @test _grid_1d(itp) isa _CachedRange          # stack-only — nothing to skip
    end

    @testset "eltype mismatch still converts (never aliases)" begin
        xi = [0, 1, 2, 4, 8]
        itp = linear_interp(xi, y; store = StorePolicy(copy = false, cache_axis = false))
        g = _grid_1d(itp)
        @test g isa Vector{Float64} && g == xi
    end

    @testset "eval / deriv / integrate parity vs cached build" begin
        for ctor in (linear_interp, quadratic_interp)
            itp_raw = ctor(x, y; store = StorePolicy(copy = false, cache_axis = false))
            itp_ref = ctor(x, y)
            for q in (0.25, 1.0, 2.2, 3.7)
                @test itp_raw(q) ≈ itp_ref(q) atol = 1.0e-12
                @test itp_raw(q; deriv = DerivOp(1)) ≈ itp_ref(q; deriv = DerivOp(1)) atol = 1.0e-12
            end
            @test integrate(itp_raw) ≈ integrate(itp_ref) atol = 1.0e-12
            @test integrate(itp_raw, 0.3, 3.3) ≈ integrate(itp_ref, 0.3, 3.3) atol = 1.0e-12
        end
        itp_a = akima_interp(x, y; store = StorePolicy(copy = false, cache_axis = false))
        itp_ar = akima_interp(x, y)
        for q in (0.25, 1.0, 2.2, 3.7)
            @test itp_a(q) ≈ itp_ar(q) atol = 1.0e-12
        end
        itp_c = constant_interp(x, y; store = StorePolicy(copy = false, cache_axis = false))
        itp_cr = constant_interp(x, y)
        @test itp_c(2.2) ≈ itp_cr(2.2)
        @test integrate(itp_c) ≈ integrate(itp_cr) atol = 1.0e-12
    end

    @testset "ND: raw vector axes (linear/constant)" begin
        xg = [0.0, 0.5, 1.5, 3.0, 4.0]
        yg = [0.0, 1.0, 2.5, 4.0]
        data = [xi + 2yj for xi in xg, yj in yg]
        itp = linear_interp((xg, yg), data; store = StorePolicy(copy = false, cache_axis = false))
        @test itp.grids[1] === xg && itp.grids[2] === yg      # raw pass-through per axis
        itp_ref = linear_interp((xg, yg), data)
        @test itp(1.2, 2.2) ≈ itp_ref(1.2, 2.2) atol = 1.0e-12
        @test integrate(itp) ≈ integrate(itp_ref) atol = 1.0e-12

        itp_c = constant_interp((xg, yg), data; store = StorePolicy(copy = false, cache_axis = false))
        @test itp_c.grids[1] === xg
        itp_cref = constant_interp((xg, yg), data)
        @test itp_c(1.2, 2.2) ≈ itp_cref(1.2, 2.2)
        @test integrate(itp_c) ≈ integrate(itp_cref) atol = 1.0e-12
    end

    @testset "ND: Range axes keep free _CachedRange (cache_axis=false)" begin
        xr = range(0.0, 1.0, length = 5)
        yr = range(0.0, 2.0, length = 4)
        data = [xi + 2yj for xi in xr, yj in yr]
        itp = linear_interp((xr, yr), data; store = StorePolicy(cache_axis = false))
        @test itp.grids[1] isa _CachedRange && itp.grids[2] isa _CachedRange  # nothing to skip
        itp_ref = linear_interp((xr, yr), data)
        @test integrate(itp) ≈ integrate(itp_ref) atol = 1.0e-12
    end

    @testset "pre-wrapped axis passes through (cache_axis=false, copy=false)" begin
        # A `_CachedVector` axis reaching the ctor again (grid reuse) is stored as-is
        # under copy=false — the raw-axis policy short-circuits before re-wrapping.
        x = [0.0, 0.5, 1.5, 3.0, 4.0]
        y = [1.0, 2.0, 4.0, 8.0, 16.0]
        cv = _grid_1d(linear_interp(x, y))                    # a _CachedVector
        @test cv isa _CachedVector
        itp = linear_interp(cv, y; store = StorePolicy(copy = false, cache_axis = false))
        @test _grid_1d(itp) === cv                            # 1D passthrough by identity

        data = [xi + 2yj for xi in x, yj in y[1:4]]
        itp2 = linear_interp((x, y[1:4]), data)
        cvx = itp2.grids[1]
        @test cvx isa _CachedVector
        itp2b = linear_interp((cvx, itp2.grids[2]), data; store = StorePolicy(copy = false, cache_axis = false))
        @test itp2b.grids[1] === cvx                          # ND passthrough by identity
    end

    @testset "hetero ND data: eltype mismatch promote-copies" begin
        # `_own_or_ref_data(data, Tv, store)` copies to `Array{Tv}` when the input
        # eltype differs from the promoted value type — here Int data → Float64.
        x = range(0.0, 1.0, length = 6)
        y = range(0.0, 1.0, length = 5)
        di = [Int(round(10 * (xi + yj))) for xi in x, yj in y]
        itp = interp((x, y), di; method = (LinearInterp(), AkimaInterp()))
        @test eltype(itp.data) === Float64                    # promoted, not aliased
        itp_ref = interp((x, y), Float64.(di); method = (LinearInterp(), AkimaInterp()))
        @test itp(0.4, 0.6) ≈ itp_ref(0.4, 0.6) atol = 1.0e-12
    end
end
