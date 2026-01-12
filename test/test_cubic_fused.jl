# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      CUBIC FUSED INTERPOLANT TESTS                         ║
# ║         Tests for CubicMultiInterpolantFused with interleaved layout       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

using Test
using FastInterpolations

@testset "CubicMultiInterpolantFused" begin

    # ========================================
    # Phase 1: Type Definition Tests
    # ========================================
    @testset "Phase 1: Type Definition" begin

        @testset "Type hierarchy" begin
            # CubicMultiInterpolantFused should be subtype of AbstractMultiInterpolant
            @test CubicMultiInterpolantFused <: AbstractMultiInterpolant
        end

        @testset "Type can be constructed manually" begin
            T = Float64
            n_points = 5
            n_series = 3

            # Create minimal valid data
            x = collect(range(0.0, 1.0, n_points))
            spacing = FastInterpolations._create_spacing(x)
            y = Matrix{T}(undef, n_series, n_points)
            z = Matrix{T}(undef, n_series, n_points)

            # Fill with simple data
            for i in 1:n_series
                y[i, :] .= range(0.0, 1.0, n_points) .* i
                z[i, :] .= 0.0  # Natural BC
            end

            # BCPair for natural BC
            bc_config = BCPair(Deriv2(0.0), Deriv2(0.0))
            extrap = Val(:none)

            # Construct directly
            mitp = CubicMultiInterpolantFused(
                x, spacing, y, z, bc_config, extrap, n_series, n_points
            )

            @test mitp isa CubicMultiInterpolantFused
            @test mitp isa AbstractMultiInterpolant{Float64}
        end

        @testset "Field accessors return expected types" begin
            T = Float64
            n_points = 5
            n_series = 3

            x = collect(range(0.0, 1.0, n_points))
            spacing = FastInterpolations._create_spacing(x)
            y = ones(T, n_series, n_points)
            z = zeros(T, n_series, n_points)
            bc_config = BCPair(Deriv2(0.0), Deriv2(0.0))
            extrap = Val(:none)

            mitp = CubicMultiInterpolantFused(
                x, spacing, y, z, bc_config, extrap, n_series, n_points
            )

            # Test field access
            @test mitp.x === x
            @test mitp.spacing === spacing
            @test mitp.y === y
            @test mitp.z === z
            @test mitp.bc_config === bc_config
            @test mitp.extrap === extrap
            @test mitp.n_series == n_series
            @test mitp.n_points == n_points
        end

        @testset "Type parameters are correctly inferred" begin
            # Float64 with Vector grid
            x64 = collect(range(0.0, 1.0, 5))
            spacing64 = FastInterpolations._create_spacing(x64)
            y64 = ones(Float64, 3, 5)
            z64 = zeros(Float64, 3, 5)
            bc64 = BCPair(Deriv2(0.0), Deriv2(0.0))

            mitp64 = CubicMultiInterpolantFused(
                x64, spacing64, y64, z64, bc64, Val(:none), 3, 5
            )
            @test mitp64 isa CubicMultiInterpolantFused{Float64}

            # Float32 with Range grid
            x32 = range(0.0f0, 1.0f0, 5)
            spacing32 = FastInterpolations._create_spacing(x32)
            y32 = ones(Float32, 2, 5)
            z32 = zeros(Float32, 2, 5)
            bc32 = BCPair(Deriv2(0.0f0), Deriv2(0.0f0))

            mitp32 = CubicMultiInterpolantFused(
                x32, spacing32, y32, z32, bc32, Val(:constant), 2, 5
            )
            @test mitp32 isa CubicMultiInterpolantFused{Float32}
        end

    end  # Phase 1 testset

end  # Main testset
