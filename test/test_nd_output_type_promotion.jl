# ND output-type contract: the result eltype must follow the natural
# `promote_type(grid, data, query)` and be `@inferred`-stable, for every method ×
# grid container × (data, query) float combination. An Int/OneTo grid value-matches
# the data float (Int grid + Float32 data → Float32 grid), so e.g.
# F32 grid+data+query → Float32, while a Float64 query still promotes → Float64.

@testitem "ND persistent eval output type = promote_type(grid, data, query), @inferred" begin
    FI = FastInterpolations

    builders = (
        ("linear", (g, d) -> linear_interp(g, d)),
        ("cubic", (g, d) -> cubic_interp(g, d)),
        ("quadratic", (g, d) -> quadratic_interp(g, d)),
        ("constant", (g, d) -> constant_interp(g, d)),
    )
    gridspecs = (
        ("F32vec", Float32, () -> (collect(Float32, 1:7), collect(Float32, 1:7))),
        ("F64vec", Float64, () -> (collect(Float64, 1:7), collect(Float64, 1:7))),
        ("IntOneTo", :match, () -> (Base.OneTo(7), Base.OneTo(7))),
        ("IntVec", :match, () -> (collect(1:7), collect(1:7))),
    )

    for (mname, build) in builders, (gname, gTg, gbuild) in gridspecs,
            Tv in (Float32, Float64), Tq in (Float32, Float64)

        gx, gy = gbuild()
        xs = gx isa Base.OneTo ? (1:7) : gx
        ys = gy isa Base.OneTo ? (1:7) : gy
        data = Tv.([sin(3 * float(x)) + cos(2 * float(y)) for x in xs, y in ys])
        q = (Tq(2.4), Tq(3.6))
        Tg = gTg === :match ? promote_type(Float32, Tv) : gTg   # Int/OneTo grid value-matches data float
        expected = promote_type(Tg, Tv, Tq)
        itp = build((gx, gy), data)

        @testset "$mname $gname Tv=$Tv Tq=$Tq → $expected" begin
            @test itp(q) isa expected
            @test (@inferred itp(q)) isa expected
        end
    end
end

@testitem "ND one-shot output type = promote_type(grid, data, query), @inferred" begin
    FI = FastInterpolations

    methods = (
        ("linear", (g, d, q) -> linear_interp(g, d, q)),
        ("cubic", (g, d, q) -> cubic_interp(g, d, q)),
        ("quadratic", (g, d, q) -> quadratic_interp(g, d, q)),
        ("constant", (g, d, q) -> constant_interp(g, d, q)),
    )
    gridspecs = (
        ("F32vec", Float32, () -> (collect(Float32, 1:7), collect(Float32, 1:7))),
        ("F64vec", Float64, () -> (collect(Float64, 1:7), collect(Float64, 1:7))),
        ("IntOneTo", :match, () -> (Base.OneTo(7), Base.OneTo(7))),
        ("IntVec", :match, () -> (collect(1:7), collect(1:7))),
    )

    for (mname, mfn) in methods, (gname, gTg, gbuild) in gridspecs,
            Tv in (Float32, Float64), Tq in (Float32, Float64)

        gx, gy = gbuild()
        xs = gx isa Base.OneTo ? (1:7) : gx
        ys = gy isa Base.OneTo ? (1:7) : gy
        data = Tv.([sin(3 * float(x)) + cos(2 * float(y)) for x in xs, y in ys])
        q = (Tq(2.4), Tq(3.6))
        Tg = gTg === :match ? promote_type(Float32, Tv) : gTg   # Int/OneTo grid value-matches data float
        expected = promote_type(Tg, Tv, Tq)

        @testset "$mname $gname Tv=$Tv Tq=$Tq → $expected" begin
            @test mfn((gx, gy), data, q) isa expected
            @test (@inferred mfn((gx, gy), data, q)) isa expected
        end
    end
end

@testitem "Hermite ND output type = promote_type(grid, data∪partials, query), @inferred" begin
    FI = FastInterpolations

    gridspecs = (
        ("F32vec", Float32, () -> (collect(Float32, 1:5), collect(Float32, 1:5))),
        ("F64vec", Float64, () -> (collect(Float64, 1:5), collect(Float64, 1:5))),
        ("IntOneTo", :match, () -> (Base.OneTo(5), Base.OneTo(5))),
        ("IntVec", :match, () -> (collect(1:5), collect(1:5))),
    )

    for (gname, gTg, gbuild) in gridspecs, Tv in (Float32, Float64), Tq in (Float32, Float64)
        gx, gy = gbuild()
        xs, ys = 1:5, 1:5
        data = Tv.([sin(0.5a) * cos(0.5b) for a in xs, b in ys])
        p = HermitePartials(
            (1, 0) => Tv.([0.5cos(0.5a) * cos(0.5b) for a in xs, b in ys]),
            (0, 1) => Tv.([-0.5sin(0.5a) * sin(0.5b) for a in xs, b in ys]),
            (1, 1) => Tv.([-0.25cos(0.5a) * sin(0.5b) for a in xs, b in ys]),
        )
        q = (Tq(2.4), Tq(3.6))
        Tg = gTg === :match ? promote_type(Float32, Tv) : gTg   # Int/OneTo grid value-matches data float
        expected = promote_type(Tg, Tv, Tq)

        @testset "oneshot $gname Tv=$Tv Tq=$Tq → $expected" begin
            @test hermite_interp((gx, gy), data, p, q) isa expected
            @test (@inferred hermite_interp((gx, gy), data, p, q)) isa expected
        end
        itp = hermite_interp((gx, gy), data, p)
        @testset "persistent $gname Tv=$Tv Tq=$Tq → $expected" begin
            @test itp(q) isa expected
            @test (@inferred itp(q)) isa expected
        end
    end
end

@testitem "cubic explicit PreCompute value-matches all axis containers" begin
    # The scalar PreCompute backend promotes grids exactly like the batch path
    # (`_nd_promote_grids`): Ranges → isbits `_CachedRange{Tg}` (spline caches memoise
    # via value-deterministic objectid), same-eltype Vectors pass by identity, and a
    # mismatched Vector converts (the caches match it by content). So Int/OneTo/Vector
    # grids + Float32 data all return Float32, matching the OnTheFly default.
    g = Base.OneTo(7)
    data = Float32.([sin(3.0x) + cos(2.0y) for x in 1:7, y in 1:7])
    q = (2.4f0, 3.6f0)
    v_otf = cubic_interp((g, g), data, q)
    @test v_otf isa Float32                                               # default (OnTheFly)
    v_pc = cubic_interp((g, g), data, q; coeffs = PreCompute())
    @test v_pc isa Float32
    @test v_pc ≈ v_otf rtol = 1.0f-5

    gv = collect(1:7)
    @test cubic_interp((gv, gv), data, q; coeffs = PreCompute()) isa Float32
    @test cubic_interp((g, gv), data, q; coeffs = PreCompute()) isa Float32  # mixed containers
end

@testitem "ND Int-data output types: arithmetic float-forces, selection stays natural" begin
    FI = FastInterpolations
    dataI = [x + 2y for x in 1:7, y in 1:7]

    gridspecs = (
        ("IntOneTo", Int, () -> (Base.OneTo(7), Base.OneTo(7))),
        ("IntVec", Int, () -> (collect(1:7), collect(1:7))),
        ("F32vec", Float32, () -> (collect(Float32, 1:7), collect(Float32, 1:7))),
        ("F64vec", Float64, () -> (collect(Float64, 1:7), collect(Float64, 1:7))),
    )
    arith = (
        ("linear", linear_interp),
        ("cubic", cubic_interp),
        ("quadratic", quadratic_interp),
    )

    for (gname, Te, gbuild) in gridspecs, q in ((2.4f0, 3.6f0), (2.4, 3.6), (2, 3))
        Tq = typeof(q[1])
        gx, gy = gbuild()

        # Arithmetic kernels divide → Int floats: Tg = float(promote(grid, Int)).
        Tg = float(promote_type(Te, Int))
        expected = promote_type(Tg, Tq)
        for (mname, f) in arith
            @testset "$mname $gname Int data Tq=$Tq → $expected" begin
                @test f((gx, gy), dataI, q) isa expected
                @test f((gx, gy), dataI)(q) isa expected     # one-shot ≡ persistent type
            end
        end

        # Selection kernel (constant): no x·y arithmetic → pure natural promotion,
        # NO float forcing (all-Int in → Int out); one-shot must match persistent.
        expected_sel = promote_type(Te, Int, Tq)
        @testset "constant $gname Int data Tq=$Tq → $expected_sel" begin
            @test constant_interp((gx, gy), dataI, q) isa expected_sel
            @test constant_interp((gx, gy), dataI)(q) isa expected_sel
        end
    end
end
