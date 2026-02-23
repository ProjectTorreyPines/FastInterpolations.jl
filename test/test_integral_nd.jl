using Test
using FastInterpolations

@testset "integrate nd parity (linear/quadratic/constant)" begin
    x = collect(range(0.0, 1.0, length=21))
    y = collect(range(0.0, 2.0, length=17))

    @testset "linear nd exact on multilinear field" begin
        data = [2xi - 3yj + 1 for xi in x, yj in y]
        itp = linear_interp((x, y), data; extrap=(NoExtrap(), NoExtrap()))
        lo, hi = (0.2, 0.3), (0.9, 1.4)
        expected =
            ((hi[1]^2 - lo[1]^2) * (hi[2] - lo[2])) -
            ((3/2) * (hi[2]^2 - lo[2]^2) * (hi[1] - lo[1])) +
            ((hi[1] - lo[1]) * (hi[2] - lo[2]))
        @test integrate(itp, lo, hi) ≈ expected atol=1e-10
    end

    @testset "quadratic nd exact on separable field" begin
        data = [xi^2 + 2yj^2 for xi in x, yj in y]
        itp = quadratic_interp((x, y), data; extrap=(NoExtrap(), NoExtrap()))
        lo, hi = (0.1, 0.4), (0.8, 1.6)
        expected =
            ((hi[1]^3 - lo[1]^3)/3) * (hi[2] - lo[2]) +
            2 * ((hi[2]^3 - lo[2]^3)/3) * (hi[1] - lo[1])
        @test integrate(itp, lo, hi) ≈ expected atol=1e-8
    end

    @testset "constant nd finite by side mode" begin
        data = [sin(xi) + cos(yj) for xi in x, yj in y]
        for side in ((LeftSide(), LeftSide()), (RightSide(), RightSide()), (NearestSide(), NearestSide()))
            itp = constant_interp((x, y), data; side=side, extrap=(NoExtrap(), NoExtrap()))
            @test isfinite(integrate(itp, (0.2, 0.3), (0.8, 1.7)))
        end
    end
end
