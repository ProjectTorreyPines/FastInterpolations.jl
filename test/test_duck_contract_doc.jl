# ========================================
# The published `Tv` duck-type contract, pinned
# ========================================
# docs/src/guides/custom_value_types.md tells users exactly which operations
# their value type must define. That page is a public promise, so each of its
# claims gets a test here — including the two "needs more" carve-outs, which
# exist only because the plain 4-op contract measurably is NOT enough there.
#
# Two value types, so the extra `Number` methods of the second cannot leak into
# the first's method table.

@testitem "Tv contract: 4 vector-space ops suffice on a Real grid" begin
    struct Vec4
        v::Float64
    end
    Base.:+(a::Vec4, b::Vec4) = Vec4(a.v + b.v)
    Base.:-(a::Vec4, b::Vec4) = Vec4(a.v - b.v)
    Base.:*(s::Real, a::Vec4) = Vec4(s * a.v)
    Base.:*(a::Vec4, s::Real) = Vec4(a.v * s)

    x = collect(0.0:1.0:4.0)
    y = Vec4.([1.0, 2.0, 4.0, 8.0, 16.0])

    for f in (constant_interp, linear_interp, quadratic_interp, cubic_interp)
        itp = f(x, y)
        @test itp(1.5) isa Vec4
        @test itp([1.5, 2.5]) isa Vector{Vec4}
        @test itp(1.5; deriv = DerivOp(1)) isa Vec4
        @test itp(1.5; deriv = DerivOp(3)) isa Vec4          # fabricated zero
        @test integrate(itp, 1.0, 4.0) isa Vec4              # BOUNDED integration
    end

    # ND, Series and vector calculus are covered by the same 4 ops.
    itp2 = linear_interp((x, x[1:4]), [Vec4(i + j) for i in 1:5, j in 1:4])
    @test itp2(1.5, 1.5) isa Vec4
    @test gradient(itp2, 1.5, 1.5) isa Tuple{Vec4, Vec4}
    s = linear_interp(x, Series([Vec4(i + j) for i in 1:5, j in 1:2]))
    @test s(1.5) isa Vector{Vec4}

    # Documented carve-out: full-domain / cumulative integration seeds an
    # accumulator, so it needs `zero(::Type{Tv})` on top of the 4 ops.
    itp = linear_interp(x, y)
    @test_throws MethodError integrate(itp)
    @test_throws MethodError cumulative_integrate(itp)
end

@testitem "Tv contract: a non-Real grid needs the `Number` scalar forms too" begin
    using Unitful

    struct Vec4U
        v::Float64
    end
    Base.:+(a::Vec4U, b::Vec4U) = Vec4U(a.v + b.v)
    Base.:-(a::Vec4U, b::Vec4U) = Vec4U(a.v - b.v)
    Base.:*(s::Real, a::Vec4U) = Vec4U(s * a.v)
    Base.:*(a::Vec4U, s::Real) = Vec4U(a.v * s)

    xu = (0.0:1.0:4.0)u"s"
    y = Vec4U.([1.0, 2.0, 4.0, 8.0, 16.0])
    itp = linear_interp(xu, y)

    # Values need only the `Real` forms — interpolation weights are dimensionless
    # even on a unit grid.
    @test itp(1.5u"s") isa Vec4U
    @test itp([1.5u"s", 2.5u"s"]) isa Vector{Vec4U}

    # Derivatives are not: they carry `coord⁻ᴺ`, so the scalar is a `Quantity`.
    @test_throws MethodError itp(1.5u"s"; deriv = DerivOp(1))
    @test_throws MethodError itp(1.5u"s"; deriv = DerivOp(2))
end

@testitem "Tv contract: adding the `Number` scalar forms unlocks unit derivatives" begin
    using Unitful

    struct Vec4N{T}
        v::T
    end
    Base.:+(a::Vec4N, b::Vec4N) = Vec4N(a.v + b.v)
    Base.:-(a::Vec4N, b::Vec4N) = Vec4N(a.v - b.v)
    Base.:*(s::Number, a::Vec4N) = Vec4N(s * a.v)
    Base.:*(a::Vec4N, s::Number) = Vec4N(a.v * s)

    xu = (0.0:1.0:4.0)u"s"
    y = Vec4N.([1.0, 2.0, 4.0, 8.0, 16.0])
    itp = linear_interp(xu, y)

    @test itp(1.5u"s") isa Vec4N
    d1 = itp(1.5u"s"; deriv = DerivOp(1))
    @test d1 isa Vec4N
    @test unit(d1.v) === u"s"^-1                 # value/coord¹ — the grid's unit
    d2 = itp(1.5u"s"; deriv = DerivOp(2))        # fabricated zero, still scaled
    @test unit(d2.v) === u"s"^-2
end
