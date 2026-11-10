# The all-Cardinal `_collapse_dims` fold must equal the tensor product of 1D cardinal splines,
# which the public 1D API computes along whole fibers.

@testitem "Cardinal ND cell fold matches nested 1D" begin
    using StaticArrays

    function nested(x, y, z, A, q, tension, deriv)
        kw(d) = (; tension, deriv = DerivOp(d), extrap = ExtendExtrap())
        along_x = [cardinal_interp(x, A[:, j, k], q[1]; kw(deriv[1])...) for j in eachindex(y), k in eachindex(z)]
        along_y = [cardinal_interp(y, along_x[:, k], q[2]; kw(deriv[2])...) for k in eachindex(z)]
        return cardinal_interp(z, along_y, q[3]; kw(deriv[3])...)
    end

    grid(n, ::Val{:range}) = range(0.0, 2.0, length = n)
    grid(n, ::Val{:nonuniform}) = cumsum([0.0; [0.3 + abs(sin(3.1i)) for i in 1:(n - 1)]])

    @testset "n=$n $kind tension=$tension $T" for n in (2, 3, 4, 5, 8),
            kind in (:range, :nonuniform), tension in (0.0, 0.3), T in (Float64, SVector{2, Float64})

        x = grid(n, Val(kind))
        y = grid(max(n - 1, 2), Val(kind))
        z = grid(n + 1, Val(kind))
        scalar(a, b, c) = sin(1.3a + 0.2) * cos(0.7b) + 0.3c^2
        A = [T <: SVector ? T(scalar(a, b, c), a - c) : scalar(a, b, c) for a in x, b in y, c in z]
        itp = interp((x, y, z), A; method = CardinalInterp(tension), extrap = ExtendExtrap())

        # Interior, knots, both grid ends and past them, where the end cells extrapolate.
        for fx in (0.37, 0.0, 1.0, -0.2, 1.2), fy in (0.61, 1.0), fz in (0.13, 0.0), deriv in ((0, 0, 0), (1, 0, 2))
            at(g, f) = first(g) + f * (last(g) - first(g))
            q = (at(x, fx), at(y, fy), at(z, fz))
            @test itp(q; deriv = map(DerivOp, deriv)) ≈ nested(x, y, z, A, q, tension, deriv) rtol = 1.0e-12 atol = 1.0e-12
        end
    end
end

@testitem "Cardinal ND cell fold reads only the cell stencil" setup = [AllocConstants] begin
    x = range(0.0, 2.0, length = 9)
    y = range(0.0, 1.0, length = 7)
    A = [sin(a) * cos(b) for a in x, b in y]
    A[4, :] .= NaN
    itp = interp((x, y), A; method = CardinalInterp())

    # x[4] is inside the fixed 4-point window of the end cell but outside its stencil.
    @test isfinite(itp((0.1, 0.3)))
    @test isnan(itp((0.5, 0.3)))

    probe(itp, q) = @allocated itp(q)
    probe(itp, (0.1, 0.3))
    @test probe(itp, (0.1, 0.3)) <= ND_ALLOC_THRESHOLD
end
