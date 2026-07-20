@testitem "integrate scaffold api" begin
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

# One-shot quadrature: integrate(x, y; method) — unified-API routing (the same
# `_interp1d_route` trait as `interp(x, y; method=…)`), built internally with
# `StorePolicy(copy=false, cache_axis=false)` so construction cost ≈ 0.
@testitem "one-shot integrate(x, y; method)" begin
    x_vec = collect(range(0.0, 2.0, length = 21))
    x_rng = range(0.0, 2.0, length = 21)
    y = @. x_vec^2 + 1.0

    @testset "matches the persistent build" begin
        for x in (x_vec, x_rng)
            @test integrate(x, y; method = LinearInterp()) ≈ integrate(linear_interp(x, y)) atol = 1.0e-12
            @test integrate(x, y; method = CubicInterp()) ≈ integrate(cubic_interp(x, y)) atol = 1.0e-12
            @test integrate(x, y; method = QuadraticInterp()) ≈ integrate(quadratic_interp(x, y)) atol = 1.0e-12
        end
    end

    @testset "method options forwarded" begin
        @test integrate(x_vec, y; method = ConstantInterp(side = LeftSide())) ≈
            integrate(constant_interp(x_vec, y; side = LeftSide())) atol = 1.0e-12
        @test integrate(x_vec, y; method = CubicInterp(bc = ZeroCurvBC())) ≈
            integrate(cubic_interp(x_vec, y; bc = ZeroCurvBC())) atol = 1.0e-12
    end

    @testset "contract: method is required; unsupported methods reject" begin
        @test_throws UndefKeywordError integrate(x_vec, y)
        @test_throws ArgumentError integrate(x_vec, y; method = NoInterp())
    end

    @testset "inputs are not mutated" begin
        xb = copy(x_vec)
        yb = copy(y)
        integrate(x_vec, y; method = LinearInterp())
        @test x_vec == xb && y == yb
    end
end
