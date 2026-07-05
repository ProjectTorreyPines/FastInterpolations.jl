# Value-match contract for 1D ONE-SHOT paths, mirroring test_nd_output_type_promotion.jl.
# Arithmetic kernels: output = promote_type(value-matched grid float, data, query),
# where the value-matched grid float = float(promote(grid eltype, data eltype)) — an
# Int/OneTo grid beside Float32 data floats to Float32, not the blind Float64.
# Selection kernel (constant): pure natural promote_type(grid, data, query).
# Persistent 1D already conforms (via the 3-arg `_cache_axis`) — guarded here too.

@testitem "1D scalar one-shot output types: arithmetic value-match, @inferred" begin
    # Direct function-value calls (NOT closures): a closure capturing testitem-level
    # locals boxes them as `Any`, which poisons `@inferred` for every cell.
    data32 = Float32[sin(0.4i) + 0.5 for i in 1:7]
    dy32 = Float32[0.4cos(0.4i) for i in 1:7]

    gridspecs = (
        ("IntRange", Int, () -> 1:7),
        ("IntOneTo", Int, () -> Base.OneTo(7)),
        ("IntVec", Int, () -> collect(1:7)),
        ("F32range", Float32, () -> 1.0f0:1.0f0:7.0f0),
        ("F32vec", Float32, () -> collect(Float32, 1:7)),
    )
    fns = (linear_interp, quadratic_interp, cubic_interp, pchip_interp, akima_interp, cardinal_interp)

    for (gname, Te, gbuild) in gridspecs, f in fns, q in (2.4f0, 2.4)
        Tq = typeof(q)
        Tg = float(promote_type(Te, Float32))
        expected = promote_type(Tg, Float32, Tq)
        g = gbuild()
        @testset "$(nameof(f)) $gname Tq=$Tq → $expected" begin
            if f in (cubic_interp, pchip_interp, akima_interp, cardinal_interp) &&
                    g isa Vector && Te === Int && expected !== Float64
                # KNOWN-RED (follow-up): raw Int-VECTOR axes pass through untyped (the
                # zero-alloc one-shot contract forbids converting them). linear/hermite
                # width-type the kernel inv_h (`_cell_geom`/`_typed_inv_h`) and quadratic
                # pool-converts, but the slope families (pchip/akima/cardinal) compute
                # local slopes reading x internally, and cubic needs an eltype-aware
                # spline-cache key — both deferred.
                @test_broken f(g, data32, q) isa expected
            else
                @test f(g, data32, q) isa expected
                @test (@inferred f(g, data32, q)) isa expected
            end
        end
    end

    # hermite carries user slopes — same value-match rule with value space y ∪ dy.
    for (gname, Te, gbuild) in gridspecs, q in (2.4f0, 2.4)
        Tq = typeof(q)
        expected = promote_type(float(promote_type(Te, Float32)), Float32, Tq)
        g = gbuild()
        @testset "hermite $gname Tq=$Tq → $expected" begin
            @test hermite_interp(g, data32, dy32, q) isa expected
            @test (@inferred hermite_interp(g, data32, dy32, q)) isa expected
        end
    end
end

@testitem "1D Series one-shot output eltypes: value-matched (guard)" begin
    # The Series wrappers already value-match their axes at the entry (arith families
    # via `_promote_grid_float(Tg, _series_eltype(s))`; constant via raw Tg) — pin it.
    data32a = Float32[sin(0.4i) for i in 1:7]
    data32b = Float32[cos(0.4i) for i in 1:7]
    for g in (1:7, collect(1:7))
        s = Series(data32a, data32b)
        @test eltype(linear_interp(g, s, 2.4f0)) === Float32
        @test eltype(quadratic_interp(g, s, 2.4f0)) === Float32
        @test eltype(cubic_interp(g, s, 2.4f0)) === Float32
        @test eltype(constant_interp(g, s, 2.4f0)) === Float32
    end
end

@testitem "1D persistent parity guard: same value-matched output types" begin
    data32 = Float32[sin(0.4i) + 0.5 for i in 1:7]
    dy32 = Float32[0.4cos(0.4i) for i in 1:7]

    gridspecs = (
        ("IntRange", Int, () -> 1:7),
        ("IntVec", Int, () -> collect(1:7)),
        ("F32vec", Float32, () -> collect(Float32, 1:7)),
    )
    builders = (
        ("linear", g -> linear_interp(g, data32)),
        ("quadratic", g -> quadratic_interp(g, data32)),
        ("cubic", g -> cubic_interp(g, data32)),
        ("pchip", g -> pchip_interp(g, data32)),
        ("akima", g -> akima_interp(g, data32)),
        ("cardinal", g -> cardinal_interp(g, data32)),
        ("hermite", g -> hermite_interp(g, data32, dy32)),
    )

    for (gname, Te, gbuild) in gridspecs, (mname, build) in builders, q in (2.4f0, 2.4)
        Tq = typeof(q)
        expected = promote_type(float(promote_type(Te, Float32)), Float32, Tq)
        itp = build(gbuild())
        @testset "$mname $gname Tq=$Tq → $expected" begin
            @test itp(q) isa expected
        end
    end
end

@testitem "1D batch one-shot output eltypes: value-matched" begin
    data32 = Float32[sin(0.4i) + 0.5 for i in 1:7]
    dy32 = Float32[0.4cos(0.4i) for i in 1:7]
    qs32 = Float32[2.4, 3.1]

    fns = (
        ("linear", g -> linear_interp(g, data32, qs32)),
        ("quadratic", g -> quadratic_interp(g, data32, qs32)),
        ("cubic", g -> cubic_interp(g, data32, qs32)),
        ("pchip", g -> pchip_interp(g, data32, qs32)),
        ("akima", g -> akima_interp(g, data32, qs32)),
        ("cardinal", g -> cardinal_interp(g, data32, qs32)),
        ("hermite", g -> hermite_interp(g, data32, dy32, qs32)),
    )

    for (gname, g) in (("IntRange", 1:7), ("IntVec", collect(1:7))), (mname, f) in fns
        @testset "$mname $gname → Vector{Float32}" begin
            @test eltype(f(g)) === Float32
        end
    end
end

@testitem "1D constant: selection kernel stays natural (no float forcing)" begin
    data32 = Float32[sin(0.4i) + 0.5 for i in 1:7]
    dataI = [2i for i in 1:7]

    for (gname, g) in (("IntRange", 1:7), ("IntOneTo", Base.OneTo(7)), ("IntVec", collect(1:7)))
        @testset "constant $gname" begin
            # F32 data: natural promote(Int, F32, Tq)
            @test constant_interp(g, data32, 2.4f0) isa Float32
            @test constant_interp(g, data32, 2.4) isa Float64
            @test constant_interp(g, data32)(2.4f0) isa Float32       # persistent parity
            # all-Int: nearest-neighbor selection returns Int for Int query
            @test constant_interp(g, dataI, 3) isa Int
            @test constant_interp(g, dataI, 2.4f0) isa Float32
            @test constant_interp(g, dataI)(3) isa Int                # persistent parity
            # batch eltype follows the same natural rule
            @test eltype(constant_interp(g, data32, Float32[2.4, 3.1])) === Float32
        end
    end
end

@testitem "1D hermite mixed y/dy widths: value space = y ∪ dy" begin
    data32 = Float32[sin(0.4i) + 0.5 for i in 1:7]
    dy64 = [0.4cos(0.4i) for i in 1:7]              # Float64 slopes beside Float32 data

    for (gname, g) in (("IntRange", 1:7), ("F32vec", collect(Float32, 1:7)))
        @testset "hermite $gname F32 data + F64 dy → Float64" begin
            @test hermite_interp(g, data32, dy64, 2.4f0) isa Float64
        end
    end
end

@testitem "1D Int-data arithmetic: float-forces to Float64 (unchanged legacy)" begin
    dataI = [2i for i in 1:7]
    # All-Int value space: float(promote(Int, Int)) = Float64 — arithmetic kernels divide.
    for f in (linear_interp, cubic_interp, pchip_interp)
        @test f(1:7, dataI, 2.4f0) isa Float64
        @test f(1:7, dataI, 3) isa Float64
    end
end
