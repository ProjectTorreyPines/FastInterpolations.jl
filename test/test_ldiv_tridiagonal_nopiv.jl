using Test
using FastInterpolations

const FI = FastInterpolations

@testset "Tridiagonal NoPivot _ldiv_tridiagonal_nopiv!" begin
    @testset "Derivative-BC cache: custom solve matches LU backslash" begin
        for T in (Float64, Float32)
            x = collect(range(T(0), T(1), 51))
            cache = FI.CubicSplineCache(x; bc=NaturalBC())

            rhs = rand(T, length(x))
            ref = cache.lu_factor \ rhs

            rhs2 = copy(rhs)
            FI._ldiv_tridiagonal_nopiv!(rhs2, cache.lu_factor, cache.inv_d)

            @test rhs2 ≈ ref rtol=eps(T) * 200
        end
    end

    @testset "Periodic cache: custom solve matches LU backslash (A')" begin
        for T in (Float64, Float32)
            x = collect(range(T(0), T(1), 51))
            cache = FI.CubicSplineCache(x; bc=PeriodicBC())

            # Periodic cache stores A' of size n = length(x)-1
            rhs = rand(T, length(cache.inv_d))
            ref = cache.lu_factor \ rhs

            rhs2 = copy(rhs)
            FI._ldiv_tridiagonal_nopiv!(rhs2, cache.lu_factor, cache.inv_d)

            @test rhs2 ≈ ref rtol=eps(T) * 200
        end
    end
end
