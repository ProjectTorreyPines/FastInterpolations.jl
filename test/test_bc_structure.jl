# ========================================
# BC Structure Trait Tests
# ========================================
# Tests for bc_structure() trait function that returns structural identity
# of boundary conditions for cache key purposes.
#
# Note: bc_structure is @inline, so we must explicitly call it to get coverage.

using Test
using FastInterpolations
using FastInterpolations: bc_structure

@testset "bc_structure Trait" begin

    # ========================================
    # Deriv1/Deriv2/Deriv3 - Basic PointBC Types
    # ========================================
    @testset "Deriv1 bc_structure" begin
        # Float64
        @test bc_structure(Deriv1(0.0)) === Val(:deriv1)
        @test bc_structure(Deriv1(0.5)) === Val(:deriv1)
        @test bc_structure(Deriv1(100.0)) === Val(:deriv1)

        # Float32
        @test bc_structure(Deriv1(0.0f0)) === Val(:deriv1)

        # Complex - same structure as Float64!
        @test bc_structure(Deriv1(1.0 + 2.0im)) === Val(:deriv1)
        @test bc_structure(Deriv1(0.0 + 0.0im)) === Val(:deriv1)

        # Different values yield same structure
        @test bc_structure(Deriv1(0.0)) === bc_structure(Deriv1(999.0))
        @test bc_structure(Deriv1(0.0)) === bc_structure(Deriv1(1.0im))
    end

    @testset "Deriv2 bc_structure" begin
        # Float64
        @test bc_structure(Deriv2(0.0)) === Val(:deriv2)
        @test bc_structure(Deriv2(1.5)) === Val(:deriv2)

        # Float32
        @test bc_structure(Deriv2(0.0f0)) === Val(:deriv2)

        # Complex
        @test bc_structure(Deriv2(0.0 + 1.0im)) === Val(:deriv2)

        # Different values yield same structure
        @test bc_structure(Deriv2(0.0)) === bc_structure(Deriv2(-20.0))
    end

    @testset "Deriv3 bc_structure" begin
        # Float64
        @test bc_structure(Deriv3(0.0)) === Val(:deriv3)
        @test bc_structure(Deriv3(6.0)) === Val(:deriv3)

        # Float32
        @test bc_structure(Deriv3(0.0f0)) === Val(:deriv3)

        # Complex
        @test bc_structure(Deriv3(1.0 + 0.0im)) === Val(:deriv3)
    end

    # ========================================
    # PolyFit - Lazy Polynomial Fitting BCs
    # ========================================
    @testset "PolyFit bc_structure" begin
        # Different degrees have different structures
        @test bc_structure(LinearFit()) === Val(:polyfit_1)
        @test bc_structure(QuadraticFit()) === Val(:polyfit_2)
        @test bc_structure(CubicFit()) === Val(:polyfit_3)

        # Using PolyFit{D} directly
        @test bc_structure(PolyFit{1}()) === Val(:polyfit_1)
        @test bc_structure(PolyFit{2}()) === Val(:polyfit_2)
        @test bc_structure(PolyFit{3}()) === Val(:polyfit_3)
        @test bc_structure(PolyFit{4}()) === Val(:polyfit_4)
        @test bc_structure(PolyFit{5}()) === Val(:polyfit_5)

        # Aliases match generic form
        @test bc_structure(LinearFit()) === bc_structure(PolyFit{1}())
        @test bc_structure(QuadraticFit()) === bc_structure(PolyFit{2}())
        @test bc_structure(CubicFit()) === bc_structure(PolyFit{3}())
    end

    # ========================================
    # Singleton BC Types
    # ========================================
    @testset "ZeroCurvBC bc_structure" begin
        @test bc_structure(ZeroCurvBC()) === Val(:natural)
        # Multiple instances have same structure
        @test bc_structure(ZeroCurvBC()) === bc_structure(ZeroCurvBC())
    end

    @testset "ZeroSlopeBC bc_structure" begin
        @test bc_structure(ZeroSlopeBC()) === Val(:clamped)
        @test bc_structure(ZeroSlopeBC()) === bc_structure(ZeroSlopeBC())
    end

    @testset "PeriodicBC bc_structure" begin
        @test bc_structure(PeriodicBC()) === Val(:periodic)
        @test bc_structure(PeriodicBC()) === bc_structure(PeriodicBC())
    end

    @testset "MinCurvFit bc_structure" begin
        @test bc_structure(MinCurvFit()) === Val(:mincurvfit)
        @test bc_structure(MinCurvFit()) === bc_structure(MinCurvFit())
    end

    # ========================================
    # BCPair - Compound BC Type
    # ========================================
    @testset "BCPair bc_structure" begin
        # BCPair returns a tuple of left/right structures
        bc_d1d2 = BCPair(Deriv1(0.5), Deriv2(1.0))
        @test bc_structure(bc_d1d2) === (Val(:deriv1), Val(:deriv2))

        # Symmetric BCPair
        bc_d1d1 = BCPair(Deriv1(0.0), Deriv1(0.0))
        @test bc_structure(bc_d1d1) === (Val(:deriv1), Val(:deriv1))

        bc_d2d2 = BCPair(Deriv2(0.0), Deriv2(0.0))
        @test bc_structure(bc_d2d2) === (Val(:deriv2), Val(:deriv2))

        # BCPair with Deriv3
        bc_d3d3 = BCPair(Deriv3(0.0), Deriv3(0.0))
        @test bc_structure(bc_d3d3) === (Val(:deriv3), Val(:deriv3))

        bc_d3d1 = BCPair(Deriv3(0.0), Deriv1(0.0))
        @test bc_structure(bc_d3d1) === (Val(:deriv3), Val(:deriv1))

        # BCPair with PolyFit
        bc_pf = BCPair(CubicFit(), CubicFit())
        @test bc_structure(bc_pf) === (Val(:polyfit_3), Val(:polyfit_3))

        bc_mixed_pf = BCPair(LinearFit(), QuadraticFit())
        @test bc_structure(bc_mixed_pf) === (Val(:polyfit_1), Val(:polyfit_2))

        # BCPair mixing PolyFit with Deriv
        bc_pf_d2 = BCPair(CubicFit(), Deriv2(0.0))
        @test bc_structure(bc_pf_d2) === (Val(:polyfit_3), Val(:deriv2))

        # Float64 and ComplexF64 BCPairs with same structure have same bc_structure
        bc_f64 = BCPair(Deriv1(0.5), Deriv2(1.0))
        bc_c64 = BCPair(Deriv1(0.5 + 0.0im), Deriv2(1.0 + 0.0im))
        @test bc_structure(bc_f64) === bc_structure(bc_c64)
    end

    # ========================================
    # Left/Right - Endpoint Wrappers (Quadratic)
    # ========================================
    @testset "Left bc_structure" begin
        # Left unwraps to inner BC structure
        @test bc_structure(Left(Deriv1(0.5))) === Val(:deriv1)
        @test bc_structure(Left(Deriv2(0.0))) === Val(:deriv2)
        @test bc_structure(Left(Deriv3(0.0))) === Val(:deriv3)

        # Left with PolyFit
        @test bc_structure(Left(LinearFit())) === Val(:polyfit_1)
        @test bc_structure(Left(QuadraticFit())) === Val(:polyfit_2)
        @test bc_structure(Left(CubicFit())) === Val(:polyfit_3)

        # Left with Complex values has same structure
        @test bc_structure(Left(Deriv1(1.0 + 2.0im))) === Val(:deriv1)
    end

    @testset "Right bc_structure" begin
        # Right unwraps to inner BC structure
        @test bc_structure(Right(Deriv1(2.0))) === Val(:deriv1)
        @test bc_structure(Right(Deriv2(0.0))) === Val(:deriv2)
        @test bc_structure(Right(Deriv3(0.0))) === Val(:deriv3)

        # Right with PolyFit
        @test bc_structure(Right(LinearFit())) === Val(:polyfit_1)
        @test bc_structure(Right(QuadraticFit())) === Val(:polyfit_2)
        @test bc_structure(Right(CubicFit())) === Val(:polyfit_3)

        # Right with Complex values has same structure
        @test bc_structure(Right(Deriv1(1.0 + 2.0im))) === Val(:deriv1)
    end

    # ========================================
    # Structure Identity Invariants
    # ========================================
    @testset "Structure Identity Invariants" begin
        # Key property: Different Tv types with same BC structure share structure
        # This enables cache sharing between Float64 and ComplexF64 interpolants

        # Deriv1: Float64 vs ComplexF64
        @test bc_structure(Deriv1{Float64}(0.5)) === bc_structure(Deriv1{ComplexF64}(1.0im))

        # Deriv2: Float64 vs ComplexF64
        @test bc_structure(Deriv2{Float64}(0.0)) === bc_structure(Deriv2{ComplexF64}(0.0im))

        # Deriv3: Float64 vs ComplexF64
        @test bc_structure(Deriv3{Float64}(6.0)) === bc_structure(Deriv3{ComplexF64}(6.0 + 0.0im))

        # BCPair: Float64 vs ComplexF64
        @test bc_structure(BCPair(Deriv1{Float64}(0.5), Deriv2{Float64}(0.0))) ===
              bc_structure(BCPair(Deriv1{ComplexF64}(0.5im), Deriv2{ComplexF64}(0.0im)))

        # Left/Right: Float64 vs ComplexF64
        @test bc_structure(Left(Deriv1{Float64}(0.0))) ===
              bc_structure(Left(Deriv1{ComplexF64}(0.0im)))
    end

    # ========================================
    # Different BC Types Have Different Structures
    # ========================================
    @testset "Different BC Types Have Different Structures" begin
        # Each BC type has a unique structure symbol
        @test bc_structure(Deriv1(0.0)) !== bc_structure(Deriv2(0.0))
        @test bc_structure(Deriv1(0.0)) !== bc_structure(Deriv3(0.0))
        @test bc_structure(Deriv2(0.0)) !== bc_structure(Deriv3(0.0))

        # Singletons are distinct
        @test bc_structure(ZeroCurvBC()) !== bc_structure(ZeroSlopeBC())
        @test bc_structure(ZeroCurvBC()) !== bc_structure(PeriodicBC())
        @test bc_structure(ZeroSlopeBC()) !== bc_structure(PeriodicBC())
        @test bc_structure(MinCurvFit()) !== bc_structure(ZeroCurvBC())

        # PolyFit degrees are distinct
        @test bc_structure(PolyFit{1}()) !== bc_structure(PolyFit{2}())
        @test bc_structure(PolyFit{2}()) !== bc_structure(PolyFit{3}())

        # Deriv vs PolyFit (even though PolyFit materializes to Deriv1)
        @test bc_structure(Deriv1(0.0)) !== bc_structure(CubicFit())
    end

end
