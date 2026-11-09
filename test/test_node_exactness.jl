# Node exactness on Vector grids: a query that lands exactly on a grid node returns the
# data value exactly (`==`). Interior nodes sit at the LEFT end of their cell (t = 0); the last
# node sits at the right end of the last cell, exact only when t = h·inv(h) rounds to 1.
# The contract is value equality, not identical bits: arithmetic kernels (linear, Hermite,
# cubic) add exact-zero terms, which turns a -0.0 node value into +0.0.
# Range grids are out of scope: their nodes are recomputed as `first + (i-1)·step`.

@testitem "node exactness: cubic interior nodes on every evaluation path" begin
    using Random
    rng = Xoshiro(207)
    grids = (
        collect(-1:0.1:1),                     # non-dyadic uniform spacing
        cumsum(rand(rng, 24) .+ 0.05),         # nonuniform
        cumsum(10 .^ (3 .* rand(rng, 24))),    # spacing over three decades
    )
    bcs = ("CubicFit" => CubicFit(), "ZeroCurvBC" => ZeroCurvBC(), "ZeroSlopeBC" => ZeroSlopeBC(), "PeriodicBC" => PeriodicBC())
    @testset "$name, n=$(length(x))" for x in grids, (name, bc) in bcs
        n = length(x)
        I = 1:(n - 1)
        y = randn(rng, n)
        y2 = randn(rng, n)
        if bc isa PeriodicBC
            y[end] = y[1]
            y2[end] = y2[1]
        end
        itp = cubic_interp(x, y; bc = bc)
        @test count(i -> itp(x[i]) != y[i], I) == 0           # scalar
        @test itp.(x)[I] == y[I]                              # broadcast
        @test itp(x)[I] == y[I]                               # batch
        out = similar(y)
        itp(out, x)
        @test out[I] == y[I]                                  # in-place batch
        @test cubic_interp(x, y, x; bc = bc)[I] == y[I]       # one-shot

        sitp = cubic_interp(x, Series(y, y2); bc = bc)
        @test count(i -> sitp(x[i]) != [y[i], y2[i]], I) == 0 # Series scalar
        sb = sitp(x)
        @test sb[1][I] == y[I]                                # Series batch
        @test sb[2][I] == y2[I]
        so = cubic_interp(x, Series(y, y2), x; bc = bc)
        @test so[1][I] == y[I]                                # Series one-shot
        @test so[2][I] == y2[I]
    end

    @testset "Float32" begin
        x = Float32.(collect(-1:0.1:1))
        y = randn(rng, Float32, length(x))
        I = 1:(length(x) - 1)
        itp = cubic_interp(x, y)
        @test count(i -> itp(x[i]) != y[i], I) == 0
        @test itp(x)[I] == y[I]
    end
end

@testitem "node exactness: every family at interior nodes (1D + ND)" begin
    using Random
    rng = Xoshiro(2071)
    x = cumsum(rand(rng, 20) .+ 0.05)
    n = length(x)
    I = 1:(n - 1)
    y = randn(rng, n)
    @testset "$(nameof(f))" for f in (
            constant_interp, linear_interp, quadratic_interp, cubic_interp,
            pchip_interp, akima_interp, cardinal_interp,
        )
        itp = f(x, y)
        @test count(i -> itp(x[i]) != y[i], I) == 0
        @test itp(x)[I] == y[I]
    end
    @testset "hermite_interp" begin
        itp = hermite_interp(x, y, randn(rng, n))
        @test count(i -> itp(x[i]) != y[i], I) == 0
    end

    x2 = cumsum(rand(rng, 7) .+ 0.05)
    Y = randn(rng, n, 7)
    @testset "ND $(nameof(f))" for f in (constant_interp, linear_interp, quadratic_interp, cubic_interp)
        itp = f((x, x2), Y)
        @test count(ij -> itp((x[ij[1]], x2[ij[2]])) != Y[ij[1], ij[2]], Iterators.product(I, 1:6)) == 0
    end
end

@testitem "node exactness: last node when the last cell has h * inv(h) == 1" begin
    using Random
    rng = Xoshiro(2072)
    x = collect(-1:0.1:1)
    h = x[end] - x[end - 1]
    @assert h * inv(h) == 1     # t = h·inv_h lands exactly on 1 for this grid
    # quadratic is excluded: its power form a·h² + d·h + y[n-1] is not exact at a right end
    @testset "$(nameof(f))" for f in (
            constant_interp, linear_interp, cubic_interp,
            pchip_interp, akima_interp, cardinal_interp,
        )
        bad = 0
        for _ in 1:20
            y = randn(rng, length(x))
            itp = f(x, y)
            bad += (itp(x[end]) != y[end]) + (itp(x)[end] != y[end])
        end
        @test bad == 0
    end
end
