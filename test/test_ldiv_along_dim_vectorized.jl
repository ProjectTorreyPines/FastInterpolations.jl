using Test
using FastInterpolations

const FI = FastInterpolations

@testset "Batch Thomas _ldiv_along_dim!(z, lu, inv_d, Val{D})" begin
    @testset "Val(1) throws ArgumentError" begin
        x = collect(range(0.0, 1.0, 20))
        cache = FI.CubicSplineCache(x; bc=NaturalBC())
        z = rand(20, 5)

        @test_throws ArgumentError FI._ldiv_along_dim!(z, cache.lu_factor, cache.inv_d, Val(1))
    end

    @testset "Val(2) correctness: batch == sequential" begin
        for T in (Float64, Float32)
            for n in (10, 50, 101)
                x = collect(range(T(0), T(1), n))
                cache = FI.CubicSplineCache(x; bc=NaturalBC())

                n_batch = 20
                z_batch = rand(T, n_batch, n)
                z_seq = copy(z_batch)

                # Sequential reference: solve each RHS row independently
                @inbounds for i in 1:n_batch
                    FI._ldiv_tridiagonal_nopiv!(view(z_seq, i, :), cache.lu_factor, cache.inv_d)
                end

                FI._ldiv_along_dim!(z_batch, cache.lu_factor, cache.inv_d, Val(2))

                @test z_batch ≈ z_seq rtol=eps(T) * 200
            end
        end
    end

    @testset "Periodic system size consistency" begin
        x = collect(range(0.0, 1.0, 51))
        cache = FI.CubicSplineCache(x; bc=PeriodicBC())

        n_sys = length(cache.inv_d) # = length(x)-1
        z = rand(8, n_sys)
        z_ref = copy(z)

        @inbounds for i in 1:size(z, 1)
            FI._ldiv_tridiagonal_nopiv!(view(z_ref, i, :), cache.lu_factor, cache.inv_d)
        end

        FI._ldiv_along_dim!(z, cache.lu_factor, cache.inv_d, Val(2))
        @test z ≈ z_ref rtol=1e-10
    end
end

