using Test
using FastInterpolations

@testset "integrate cubic 1d in-domain" begin
    x = collect(range(0.0, 1.0, length = 41))
    y = @. x^3 - 2x + 1
    itp = cubic_interp(x, y; extrap = NoExtrap())

    @testset "full-domain parity" begin
        @test integrate(itp) ≈ integrate(itp, first(x), last(x)) atol = 1.0e-12
    end

    @testset "antisymmetry" begin
        a, b = 0.2, 0.8
        @test integrate(itp, a, b) ≈ -integrate(itp, b, a) atol = 1.0e-12
    end

    @testset "additivity" begin
        a, m, b = 0.1, 0.55, 0.9
        @test integrate(itp, a, b) ≈ integrate(itp, a, m) + integrate(itp, m, b) atol = 1.0e-12
    end

    @testset "cell full formula parity" begin
        total = zero(eltype(y))
        h = itp.cache.spacing.h
        for i in 1:length(h)
            hi = h[i]
            total += hi / 2 * (itp.y[i] + itp.y[i + 1]) - hi^3 / 24 * (itp.z[i] + itp.z[i + 1])
        end
        @test integrate(itp) ≈ total atol = 1.0e-12
    end

    @testset "domain check" begin
        @test_throws DomainError integrate(itp, -0.1, 0.2)
        @test_throws DomainError integrate(itp, 0.2, 1.1)
    end
end
