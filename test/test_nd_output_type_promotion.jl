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
