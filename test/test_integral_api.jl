using Test
using FastInterpolations

@testset "integrate scaffold api" begin
    x = collect(range(0.0, 1.0, length = 11))
    y = @. x^2
    itp_lin = linear_interp(x, y)
    itp_cub = cubic_interp(x, y)

    @testset "1d integrate returns value (all methods)" begin
        for itp in (itp_lin, itp_cub)
            @test integrate(itp) isa Real
            @test integrate(itp, 0.2, 0.7) isa Real
        end
    end

    @testset "normalization helpers" begin
        s1, a1, b1 = FastInterpolations._normalize_bounds_1d(0.2, 0.8)
        @test s1 == 1
        @test a1 == 0.2
        @test b1 == 0.8

        s2, a2, b2 = FastInterpolations._normalize_bounds_1d(0.8, 0.2)
        @test s2 == -1
        @test a2 == 0.2
        @test b2 == 0.8

        s3, a3, b3 = FastInterpolations._normalize_bounds_1d(0.5, 0.5)
        @test s3 == 0
        @test a3 == 0.5
        @test b3 == 0.5
    end

    @testset "nd normalization sign" begin
        sign, lo, hi = FastInterpolations._normalize_bounds_nd((2.0, -1.0), (0.0, 3.0))
        @test sign == -1
        @test lo == (0.0, -1.0)
        @test hi == (2.0, 3.0)
    end
end
