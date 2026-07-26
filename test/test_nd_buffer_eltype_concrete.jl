# ========================================
# ND output buffer eltype must be CONCRETE
# ========================================
# The only place the library has to know the result type up front is the buffer
# it allocates before filling. Getting it wrong is silent: the values are right,
# but an abstract eltype boxes every element and poisons every downstream
# `similar`/`zeros`.
#
# `AbstractInterpolantND{Tg, …}` carries ONE `Tg`. On mixed-unit axes (`s` × `m`)
# no concrete common grid type exists, so `Tg` degrades to `Quantity{Float64}` —
# and the ND eltype trait, which reads that single `Tg`, degrades with it. The
# scalar path is unaffected (its type comes from the arithmetic), which is why
# this hid: only the allocating batch surfaces show it.
#
# The trait must agree with what the scalar call actually returns.

@testitem "ND buffer eltype: mixed-unit axes still allocate a concrete buffer" begin
    using Unitful

    V = [Float64(i + j) for i in 1:5, j in 1:4]u"K"
    itp = linear_interp(((0.0:1.0:4.0)u"s", (0.0:1.0:3.0)u"m"), V)
    qs = [(1.5u"s", 1.5u"m"), (2.5u"s", 0.5u"m")]

    @testset "value" begin
        out = itp(qs)
        @test isconcretetype(eltype(out))
        @test eltype(out) === typeof(itp(1.5u"s", 1.5u"m"))
        @test unit(eltype(out)) === u"K"
    end

    @testset "derivatives" begin
        for ops in ((1, 0), (0, 1), (1, 1), (2, 0))
            d = DerivOp(ops...)
            out = itp(qs; deriv = d)
            @test isconcretetype(eltype(out))
            @test eltype(out) === typeof(itp(1.5u"s", 1.5u"m"; deriv = d))
        end
    end
end

@testitem "ND buffer eltype: the trait agrees with the scalar result" begin
    using Unitful
    using FastInterpolations: _promote_eltype

    V = [Float64(i + j) for i in 1:5, j in 1:4]u"K"
    cases = (
        ("Real", (0.0:1.0:4.0, 0.0:1.0:3.0), (1.5, 1.5), [Float64(i + j) for i in 1:5, j in 1:4]),
        ("same-unit", ((0.0:1.0:4.0)u"s", (0.0:1.0:3.0)u"s"), (1.5u"s", 1.5u"s"), V),
        ("same-dim", ((0.0:1.0:4.0)u"cm", (0.0:1.0:3.0)u"m"), (1.5u"cm", 1.5u"m"), V),
        ("mixed-unit", ((0.0:1.0:4.0)u"s", (0.0:1.0:3.0)u"m"), (1.5u"s", 1.5u"m"), V),
    )

    for (name, grids, q, data) in cases
        itp = linear_interp(grids, data)
        @testset "$name" begin
            @test _promote_eltype(itp, typeof(first(q))) === typeof(itp(q...))
            for ops in ((1, 0), (0, 1), (1, 1))
                d = DerivOp(ops...)
                @test _promote_eltype(itp, typeof(first(q)), d) === typeof(itp(q...; deriv = d))
            end
        end
    end
end
