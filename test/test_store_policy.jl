# Tests for StorePolicy (copy vs reference / zero-copy storage).
# Phase 1: linear + constant, 1D + ND, dense-array aliasing, type-transparent.

@testitem "Store Policy - tag and factory" begin
    using FastInterpolations: StorePolicy, copies_grid, copies_values

    @test StorePolicy() === StorePolicy{true, true}()
    @test StorePolicy(copy = false) === StorePolicy{false, false}()
    @test StorePolicy(copy_values = false) === StorePolicy{true, false}()
    @test StorePolicy(copy_grid = false) === StorePolicy{false, true}()
    @test StorePolicy(copy = false, copy_grid = true) === StorePolicy{true, false}()

    @test copies_grid(StorePolicy()) === true
    @test copies_grid(StorePolicy(copy = false)) === false
    @test copies_values(StorePolicy()) === true
    @test copies_values(StorePolicy(copy = false)) === false
end

@testitem "Store Policy - storage helpers (copy vs alias)" begin
    using FastInterpolations: StorePolicy, _own_or_ref_axis, _own_or_ref_values,
        _own_or_ref_data, _cache_axis

    # 1D values: copy → fresh, reference+match → alias, reference+mismatch → copy
    y = [1.0, 2.0, 3.0]
    yc = _own_or_ref_values(y, Float64, StorePolicy())
    @test yc == y
    @test yc !== y
    @test _own_or_ref_values(y, Float64, StorePolicy(copy = false)) === y
    yi = [1, 2, 3]
    yf = _own_or_ref_values(yi, Float64, StorePolicy(copy = false))
    @test yf == [1.0, 2.0, 3.0]
    @test eltype(yf) === Float64
    @test yf !== yi

    # ND data: dense alias vs materialize
    d = [1.0 2.0; 3.0 4.0]
    @test _own_or_ref_data(d, StorePolicy(copy = false)) === d
    @test _own_or_ref_data(d, StorePolicy()) == d
    @test _own_or_ref_data(d, StorePolicy()) !== d
    dv = @view d[:, :]
    @test _own_or_ref_data(dv, StorePolicy(copy = false)) == d
    @test _own_or_ref_data(dv, StorePolicy(copy = false)) isa Array{Float64, 2}

    # grid axis wrapper: alias vs own-copy
    x = [0.0, 1.0, 2.0, 3.0]
    xc = _cache_axis(x, NoBC(), Float64)
    @test _own_or_ref_axis(xc, Float64, StorePolicy(copy = false)) === xc
    @test _own_or_ref_axis(xc, Float64, StorePolicy()) !== xc
end

@testitem "Store Policy - linear 1D reference" begin
    x = collect(range(0.0, 1.0, 64))
    y = sin.(x)

    itp_copy = linear_interp(x, y)
    itp_ref = linear_interp(x, y; store = StorePolicy(copy = false))

    qs = range(0.05, 0.95, 37)
    @test all(itp_copy(q) ≈ itp_ref(q) for q in qs)

    @test itp_ref.y === y          # values aliased
    @test itp_copy.y !== y         # default owns a copy
    @test typeof(itp_copy) === typeof(itp_ref)   # type-transparent for Float input

    # type-stable construction (literal store inside a function → constprop)
    build_ref(xx, yy) = linear_interp(xx, yy; store = StorePolicy(copy = false))
    @test (@inferred build_ref(x, y)) isa LinearInterpolant

    # reference saves the O(n) value copy
    build_ref(x, y)
    @test (@allocated build_ref(x, y)) < (@allocated linear_interp(x, y))
end

@testitem "Store Policy - constant 1D reference" begin
    x = collect(range(0.0, 1.0, 50))
    y = cos.(x)

    itp_copy = constant_interp(x, y)
    itp_ref = constant_interp(x, y; store = StorePolicy(copy = false))

    qs = range(0.02, 0.98, 41)
    @test all(itp_copy(q) ≈ itp_ref(q) for q in qs)
    @test itp_ref.y === y
    @test itp_copy.y !== y
    @test typeof(itp_copy) === typeof(itp_ref)
end

@testitem "Store Policy - linear ND reference (image case)" begin
    m, n = 40, 60
    data = [sin(i / 7) * cos(j / 9) for i in 1:m, j in 1:n]
    grids = (1:m, 1:n)

    itp_copy = linear_interp(grids, data)
    itp_ref = linear_interp(grids, data; store = StorePolicy(copy = false))

    pts = [(3.4, 5.6), (10.2, 41.9), (39.0, 1.0), (1.0, 60.0)]
    @test all(itp_copy(p) ≈ itp_ref(p) for p in pts)

    @test itp_ref.data === data          # the dominant win: data aliased
    @test itp_copy.data !== data
    @test typeof(itp_copy) === typeof(itp_ref)

    # gradient still correct under reference
    g_copy = itp_copy((10.2, 41.9); deriv = DerivOp(1, 0))
    g_ref = itp_ref((10.2, 41.9); deriv = DerivOp(1, 0))
    @test g_copy ≈ g_ref

    build_ref(g, d) = linear_interp(g, d; store = StorePolicy(copy = false))
    build_ref(grids, data)
    @test (@allocated build_ref(grids, data)) < (@allocated linear_interp(grids, data))
end

@testitem "Store Policy - constant ND reference" begin
    m, n = 32, 24
    data = Float64[i + 2j for i in 1:m, j in 1:n]
    grids = (1:m, 1:n)

    itp_copy = constant_interp(grids, data)
    itp_ref = constant_interp(grids, data; store = StorePolicy(copy = false))

    pts = [(3.4, 5.6), (10.2, 19.9), (32.0, 1.0)]
    @test all(itp_copy(p) ≈ itp_ref(p) for p in pts)
    @test itp_ref.data === data
    @test itp_copy.data !== data
    @test typeof(itp_copy) === typeof(itp_ref)
end

@testitem "Store Policy - cross-cutting (Int fallback, mixed, exclusive, view)" begin
    # Int input → type-transparent copy fallback (types match copy mode)
    xi = collect(0:9)
    yi = collect(10:19)
    itp_i_copy = linear_interp(xi, yi)
    itp_i_ref = linear_interp(xi, yi; store = StorePolicy(copy = false))
    @test typeof(itp_i_copy) === typeof(itp_i_ref)
    @test all(itp_i_copy(q) ≈ itp_i_ref(q) for q in 0.5:1.0:8.5)

    # mixed component: alias values, copy grid
    x = collect(range(0.0, 1.0, 32))
    y = exp.(x)
    itp_mix = linear_interp(x, y; store = StorePolicy(copy_values = false))
    @test itp_mix.y === y
    itp_plain = linear_interp(x, y)
    @test all(itp_mix(q) ≈ itp_plain(q) for q in 0.1:0.1:0.9)

    # :exclusive PeriodicBC degrades to copy internally, still correct
    xp = collect(1.0:6.0)
    yp = [1.0, 2.0, 3.0, 4.0, 3.0, 1.0]
    itp_excl_ref = linear_interp(
        xp, yp; bc = PeriodicBC(endpoint = :exclusive, period = 6.0),
        store = StorePolicy(copy = false)
    )
    itp_excl_copy = linear_interp(xp, yp; bc = PeriodicBC(endpoint = :exclusive, period = 6.0))
    @test all(itp_excl_ref(q) ≈ itp_excl_copy(q) for q in 1.5:1.0:5.5)

    # 1D view aliasing (Y is parametric → view stored directly)
    ybig = sin.(range(0.0, 2.0, 128))
    yview = @view ybig[1:64]
    xv = collect(range(0.0, 1.0, 64))
    itp_v = linear_interp(xv, yview; store = StorePolicy(copy = false))
    @test itp_v.y === yview
    itp_vc = linear_interp(xv, collect(yview))
    @test all(itp_v(q) ≈ itp_vc(q) for q in 0.05:0.1:0.95)
end
