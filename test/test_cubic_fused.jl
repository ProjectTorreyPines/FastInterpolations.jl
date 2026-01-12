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

    # ========================================
    # Phase 2: Vector{Vector} Constructor Tests
    # ========================================
    @testset "Phase 2: Constructor" begin

        @testset "Basic construction with Vector{Vector}" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-x)

            mitp = cubic_interp_fused(x, [y1, y2, y3])

            @test mitp isa CubicMultiInterpolantFused{Float64}
            @test mitp.n_series == 3
            @test mitp.n_points == 11
        end

        @testset "Coefficients match single CubicInterpolant" begin
            x = collect(range(0.0, 1.0, 21))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Build fused interpolant
            mitp = cubic_interp_fused(x, [y1, y2])

            # Build individual interpolants
            itp1 = cubic_interp(x, y1)
            itp2 = cubic_interp(x, y2)

            # Check z coefficients match (within floating point tolerance)
            @test mitp.z[1, :] ≈ itp1.z atol=1e-14
            @test mitp.z[2, :] ≈ itp2.z atol=1e-14

            # Check y values match
            @test mitp.y[1, :] ≈ y1 atol=1e-14
            @test mitp.y[2, :] ≈ y2 atol=1e-14
        end

        @testset "Works with AbstractRange grid" begin
            x = range(0.0, 1.0, 11)  # Range, not collected
            y1 = sin.(2π .* collect(x))
            y2 = cos.(2π .* collect(x))

            mitp = cubic_interp_fused(x, [y1, y2])

            @test mitp isa CubicMultiInterpolantFused
            # Verify Range type is preserved for O(1) lookup
            @test mitp.x isa AbstractRange
        end

        @testset "Validation: mismatched y lengths throw DimensionMismatch" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x[1:5])  # Wrong length!

            @test_throws DimensionMismatch cubic_interp_fused(x, [y1, y2])
        end

        @testset "Validation: empty ys throws ArgumentError" begin
            x = collect(range(0.0, 1.0, 11))
            ys = Vector{Float64}[]

            @test_throws ArgumentError cubic_interp_fused(x, ys)
        end

        @testset "Works with bc keyword" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Test ClampedBC
            mitp_clamped = cubic_interp_fused(x, [y1, y2]; bc=ClampedBC())
            @test mitp_clamped isa CubicMultiInterpolantFused

            # Test explicit NaturalBC
            mitp_natural = cubic_interp_fused(x, [y1, y2]; bc=NaturalBC())
            @test mitp_natural isa CubicMultiInterpolantFused
        end

        @testset "Works with extrap keyword" begin
            x = collect(range(0.0, 1.0, 11))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            for extrap in [:none, :constant, :extension]
                mitp = cubic_interp_fused(x, [y1, y2]; extrap=extrap)
                @test mitp.extrap === Val(extrap)
            end
        end

    end  # Phase 2 testset

end  # Main testset
