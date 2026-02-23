using Test
using FastInterpolations

@testset "Series Integration" begin
    x = collect(range(0.0, 2.0, length=21))
    y1 = sin.(x)
    y2 = cos.(x)
    y3 = @. x^2 - x

    @testset "CubicSeriesInterpolant" begin
        sitp = cubic_interp(x, [y1, y2, y3])
        itp1 = cubic_interp(x, y1)
        itp2 = cubic_interp(x, y2)
        itp3 = cubic_interp(x, y3)

        @testset "bounded integration matches scalar" begin
            result = integrate(sitp, 0.3, 1.7)
            @test result isa Vector
            @test length(result) == 3
            @test result[1] ≈ integrate(itp1, 0.3, 1.7)
            @test result[2] ≈ integrate(itp2, 0.3, 1.7)
            @test result[3] ≈ integrate(itp3, 0.3, 1.7)
        end

        @testset "full-domain integration" begin
            result = integrate(sitp)
            @test result isa Vector
            @test result[1] ≈ integrate(itp1)
            @test result[2] ≈ integrate(itp2)
            @test result[3] ≈ integrate(itp3)
        end

        @testset "reversed bounds" begin
            fwd = integrate(sitp, 0.3, 1.7)
            rev = integrate(sitp, 1.7, 0.3)
            @test rev ≈ -fwd
        end

        @testset "equal bounds" begin
            result = integrate(sitp, 0.5, 0.5)
            @test all(iszero, result)
        end
    end

    @testset "LinearSeriesInterpolant" begin
        sitp = linear_interp(x, [y1, y2])
        itp1 = linear_interp(x, y1)
        itp2 = linear_interp(x, y2)

        @testset "bounded integration matches scalar" begin
            result = integrate(sitp, 0.3, 1.7)
            @test result[1] ≈ integrate(itp1, 0.3, 1.7)
            @test result[2] ≈ integrate(itp2, 0.3, 1.7)
        end

        @testset "full-domain integration" begin
            result = integrate(sitp)
            @test result[1] ≈ integrate(itp1)
            @test result[2] ≈ integrate(itp2)
        end

        @testset "reversed bounds" begin
            @test integrate(sitp, 1.7, 0.3) ≈ -integrate(sitp, 0.3, 1.7)
        end
    end

    @testset "QuadraticSeriesInterpolant" begin
        sitp = quadratic_interp(x, [y1, y2])
        itp1 = quadratic_interp(x, y1)
        itp2 = quadratic_interp(x, y2)

        @testset "bounded integration matches scalar" begin
            result = integrate(sitp, 0.3, 1.7)
            @test result[1] ≈ integrate(itp1, 0.3, 1.7)
            @test result[2] ≈ integrate(itp2, 0.3, 1.7)
        end

        @testset "full-domain integration" begin
            result = integrate(sitp)
            @test result[1] ≈ integrate(itp1)
            @test result[2] ≈ integrate(itp2)
        end

        @testset "reversed bounds" begin
            @test integrate(sitp, 1.7, 0.3) ≈ -integrate(sitp, 0.3, 1.7)
        end
    end

    @testset "ConstantSeriesInterpolant" begin
        for side in (LeftSide(), RightSide(), NearestSide())
            @testset "side=$side" begin
                sitp = constant_interp(x, [y1, y2]; side=side)
                itp1 = constant_interp(x, y1; side=side)
                itp2 = constant_interp(x, y2; side=side)

                @testset "bounded integration matches scalar" begin
                    result = integrate(sitp, 0.3, 1.7)
                    @test result[1] ≈ integrate(itp1, 0.3, 1.7)
                    @test result[2] ≈ integrate(itp2, 0.3, 1.7)
                end

                @testset "full-domain integration" begin
                    result = integrate(sitp)
                    @test result[1] ≈ integrate(itp1)
                    @test result[2] ≈ integrate(itp2)
                end
            end
        end
    end

    @testset "Extrapolation modes" begin
        x = collect(range(0.0, 1.0, length=11))
        y1 = sin.(x)
        y2 = cos.(x)

        @testset "extrap=NoExtrap() throws DomainError" begin
            sitp = cubic_interp(x, [y1, y2]; extrap=NoExtrap())
            @test_throws DomainError integrate(sitp, -0.1, 0.5)
            @test_throws DomainError integrate(sitp, 0.5, 1.1)
        end

        @testset "extrap=ConstExtrap()" begin
            sitp = cubic_interp(x, [y1, y2]; extrap=ConstExtrap())
            itp1 = cubic_interp(x, y1; extrap=ConstExtrap())
            itp2 = cubic_interp(x, y2; extrap=ConstExtrap())
            result = integrate(sitp, -0.5, 1.5)
            @test result[1] ≈ integrate(itp1, -0.5, 1.5)
            @test result[2] ≈ integrate(itp2, -0.5, 1.5)
        end

        @testset "extrap=ExtendExtrap()" begin
            sitp = cubic_interp(x, [y1, y2]; extrap=ExtendExtrap())
            itp1 = cubic_interp(x, y1; extrap=ExtendExtrap())
            itp2 = cubic_interp(x, y2; extrap=ExtendExtrap())
            result = integrate(sitp, -0.1, 1.1)
            @test result[1] ≈ integrate(itp1, -0.1, 1.1)
            @test result[2] ≈ integrate(itp2, -0.1, 1.1)
        end

        @testset "extrap=WrapExtrap()" begin
            sitp = cubic_interp(x, [y1, y2]; extrap=WrapExtrap())
            itp1 = cubic_interp(x, y1; extrap=WrapExtrap())
            itp2 = cubic_interp(x, y2; extrap=WrapExtrap())
            result = integrate(sitp, -0.5, 2.5)
            @test result[1] ≈ integrate(itp1, -0.5, 2.5)
            @test result[2] ≈ integrate(itp2, -0.5, 2.5)
        end

        @testset "linear extrap=ConstExtrap()" begin
            sitp = linear_interp(x, [y1, y2]; extrap=ConstExtrap())
            itp1 = linear_interp(x, y1; extrap=ConstExtrap())
            itp2 = linear_interp(x, y2; extrap=ConstExtrap())
            result = integrate(sitp, -0.5, 1.5)
            @test result[1] ≈ integrate(itp1, -0.5, 1.5)
            @test result[2] ≈ integrate(itp2, -0.5, 1.5)
        end
    end
end
