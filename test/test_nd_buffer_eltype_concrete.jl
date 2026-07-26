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

# The one-shot surfaces compute the same witness from the interpolant's inputs
# rather than from a built struct. They joined the axes into ONE grid type, so on
# mixed-unit axes the buffer came out abstract — and unlike the persistent path,
# the ND batch kernels are pinned to a concrete output, so this is not a silent
# boxing but an internal `MethodError` on a public call.
@testitem "ND buffer eltype: one-shot surfaces match the persistent buffer" begin
    using Unitful

    V = [Float64(i + j) for i in 1:5, j in 1:4]u"K"
    grids = ((0.0:1.0:4.0)u"s", (0.0:1.0:3.0)u"m")
    q = (1.5u"s", 1.5u"m")
    qs = [q, (2.5u"s", 0.5u"m")]

    for (name, f, M) in (
            ("linear", linear_interp, LinearInterp()),
            ("constant", constant_interp, ConstantInterp()),
        )
        itp = f(grids, V)                       # persistent = the reference
        ref = itp(qs)
        @testset "$name" begin
            @test isconcretetype(eltype(ref))

            out = f(grids, V, qs)               # one-shot batch
            @test eltype(out) === eltype(ref)
            @test out ≈ ref

            out2 = interp(grids, V, qs; method = M)
            @test eltype(out2) === eltype(ref)
            @test out2 ≈ ref

            out3 = interp(grids, V, qs; method = (M, M))
            @test eltype(out3) === eltype(ref)

            # in-place with a correctly-typed buffer must also route
            buf = similar(ref)
            @test f(grids, V, qs) ≈ ref
            f isa typeof(linear_interp) && (linear_interp!(buf, grids, V, qs); @test buf ≈ ref)

            # derivatives keep the same story
            d = DerivOp(1, 0)
            @test eltype(f(grids, V, qs; deriv = d)) === eltype(itp(qs; deriv = d))
        end
    end
end

@testitem "ND buffer eltype: one-shot on Real / same-unit axes is unchanged" begin
    using Unitful

    # The invariant, stated once: the one-shot batch buffer is exactly what the
    # persistent scalar call returns. Everything below is a case of it.
    D32 = [Float32(i + j) for i in 1:5, j in 1:4]
    Di = [i + j for i in 1:5, j in 1:4]
    Df = [Float64(i + j) for i in 1:5, j in 1:4]
    V = [Float64(i + j) for i in 1:5, j in 1:4]u"K"

    cases = (
        ("Int grid, F32 data, F32 query", (0:4, 0:3), D32, (1.5f0, 1.5f0)),
        ("Int grid, F32 data, F64 query", (0:4, 0:3), D32, (1.5, 1.5)),
        ("Int grid, Int data, Int query", (0:4, 0:3), Di, (1, 1)),
        ("Int grid, F64 data", (0:4, 0:3), Df, (1.5, 1.5)),
        ("F64 grid", (0.0:1.0:4.0, 0.0:1.0:3.0), Df, (1.5, 1.5)),
        ("same-unit axes", ((0.0:1.0:4.0)u"s", (0.0:1.0:3.0)u"s"), V, (1.5u"s", 1.5u"s")),
    )
    for (name, g, d, q) in cases, f in (linear_interp, constant_interp)
        @testset "$name / $(nameof(f))" begin
            @test eltype(f(g, d, [q])) === typeof(f(g, d)(q))
        end
    end

    # The two families must keep disagreeing where they should: Constant SELECTS
    # (Int data over an Int grid stays Int), Linear BLENDS (it widens to Float).
    @test eltype(constant_interp((0:4, 0:3), Di, [(1, 1)])) === Int
    @test eltype(linear_interp((0:4, 0:3), Di, [(1, 1)])) === Float64
    # A Float32 grid+data pair must not be widened by the fold itself.
    @test eltype(linear_interp((0:4, 0:3), D32, [(1.5f0, 1.5f0)])) === Float32
end
