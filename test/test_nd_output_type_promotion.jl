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

@testitem "KNOWN-RED: cubic explicit PreCompute Int-grid narrow-float (1D-phase scope)" begin
    # `coeffs = PreCompute()` routes through the 1D `_get_cubic_cache` machinery, which is
    # still data-unaware (floats an Int grid to Float64) — the value-matched witness (F32)
    # then rejects the F64 eval. Flips when the 1D one-shot value-match phase lands.
    g = Base.OneTo(7)
    data = Float32.([sin(3.0x) + cos(2.0y) for x in 1:7, y in 1:7])
    q = (2.4f0, 3.6f0)
    @test cubic_interp((g, g), data, q) isa Float32                       # default (OnTheFly): green
    @test_broken cubic_interp((g, g), data, q; coeffs = PreCompute()) isa Float32
end
