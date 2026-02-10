using Test
using FastInterpolations

@testset "integrate nd cubic in-domain" begin
    x = collect(range(0.0, 2.0, length=31))
    y = collect(range(-1.0, 1.0, length=29))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    itp = cubic_interp((x, y), data; extrap=(:none, :none))

    @testset "full-domain parity" begin
        lo = (first(x), first(y))
        hi = (last(x), last(y))
        @test integrate(itp) ≈ integrate(itp, lo, hi) atol=1e-10
    end

    @testset "sign rule" begin
        lo, hi = (0.2, -0.7), (1.4, 0.3)
        @test integrate(itp, lo, hi) ≈ -integrate(itp, (hi[1], lo[2]), (lo[1], hi[2])) atol=1e-10
        @test integrate(itp, lo, hi) ≈ integrate(itp, hi, lo) atol=1e-10
    end

    @testset "partition additivity" begin
        lo, hi = (0.1, -0.4), (1.7, 0.8)
        xm = 0.9
        lhs = integrate(itp, lo, hi)
        rhs = integrate(itp, lo, (xm, hi[2])) + integrate(itp, (xm, lo[2]), hi)
        @test lhs ≈ rhs atol=1e-10
    end

    @testset "domain check" begin
        @test_throws DomainError integrate(itp, (-0.1, -0.3), (0.5, 0.2))
        @test_throws DomainError integrate(itp, (0.2, -0.2), (2.1, 0.4))
    end
end
