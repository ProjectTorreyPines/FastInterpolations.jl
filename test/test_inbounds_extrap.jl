using Test
using FastInterpolations

@testset "InBounds() extrap type" begin
    x = collect(range(0.0, 10.0, 21))
    y = sin.(x)
    xq_scalar = 5.0
    xq_vec = collect(range(0.5, 9.5, 50))

    # ========================================
    # Scalar API: extrap=InBounds() skips domain check
    # ========================================
    @testset "Scalar in-domain: all methods" begin
        @test linear_interp(x, y, xq_scalar; extrap = InBounds()) ≈
              linear_interp(x, y, xq_scalar; extrap = NoExtrap())
        @test quadratic_interp(x, y, xq_scalar; extrap = InBounds()) ≈
              quadratic_interp(x, y, xq_scalar; extrap = NoExtrap())
        @test cubic_interp(x, y, xq_scalar; extrap = InBounds()) ≈
              cubic_interp(x, y, xq_scalar; extrap = NoExtrap())
        @test constant_interp(x, y, xq_scalar; extrap = InBounds()) ≈
              constant_interp(x, y, xq_scalar; extrap = NoExtrap())
    end

    # ========================================
    # Vector API: extrap=InBounds() matches NoExtrap for in-domain queries
    # ========================================
    @testset "Vector in-domain: all methods" begin
        @test linear_interp(x, y, xq_vec; extrap = InBounds()) ≈
              linear_interp(x, y, xq_vec; extrap = NoExtrap())
        @test quadratic_interp(x, y, xq_vec; extrap = InBounds()) ≈
              quadratic_interp(x, y, xq_vec; extrap = NoExtrap())
        @test cubic_interp(x, y, xq_vec; extrap = InBounds()) ≈
              cubic_interp(x, y, xq_vec; extrap = NoExtrap())
        @test constant_interp(x, y, xq_vec; extrap = InBounds()) ≈
              constant_interp(x, y, xq_vec; extrap = NoExtrap())
    end

    # ========================================
    # Interpolant path: InBounds() also works
    # ========================================
    @testset "Interpolant scalar + vector" begin
        itp = linear_interp(x, y; extrap = InBounds())
        @test itp(xq_scalar) ≈ linear_interp(x, y, xq_scalar)
        @test itp(xq_vec) ≈ linear_interp(x, y, xq_vec)

        itp_c = cubic_interp(x, y; extrap = InBounds())
        @test itp_c(xq_scalar) ≈ cubic_interp(x, y, xq_scalar)
        @test itp_c(xq_vec) ≈ cubic_interp(x, y, xq_vec)
    end

    # ========================================
    # NoExtrap → InBounds batch conversion: vector NoExtrap must work
    # (internally converts to InBounds in the loop)
    # ========================================
    @testset "Batch NoExtrap → InBounds conversion" begin
        itp = linear_interp(x, y)  # default NoExtrap
        out1 = itp(xq_vec)
        out2 = similar(xq_vec)
        itp(out2, xq_vec)
        @test out1 ≈ out2
    end

    # ========================================
    # Type stability: InBounds() interpolants are @inferred
    # ========================================
    @testset "Type stability" begin
        @test @inferred(linear_interp(x, y, xq_scalar; extrap = InBounds())) isa Float64
        @test @inferred(cubic_interp(x, y, xq_scalar; extrap = InBounds())) isa Float64

        itp = linear_interp(x, y; extrap = InBounds())
        @test @inferred(itp(xq_scalar)) isa Float64
    end
end
