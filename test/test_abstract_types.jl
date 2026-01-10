# Test suite for Abstract Type Hierarchy
# Phase 1 of AbstractMultiInterpolant implementation

using Test
using FastInterpolations

@testset "Abstract Types Hierarchy" begin

    # ========================================
    # Abstract Types Exist and Are Abstract
    # ========================================
    @testset "AbstractInterpolant{T} exists and is abstract" begin
        @test isdefined(FastInterpolations, :AbstractInterpolant)
        @test isabstracttype(FastInterpolations.AbstractInterpolant)
        @test isabstracttype(FastInterpolations.AbstractInterpolant{Float64})
    end

    @testset "AbstractMultiInterpolant{T} exists and is abstract" begin
        @test isdefined(FastInterpolations, :AbstractMultiInterpolant)
        @test isabstracttype(FastInterpolations.AbstractMultiInterpolant)
        @test isabstracttype(FastInterpolations.AbstractMultiInterpolant{Float64})
    end

    # ========================================
    # Single Interpolants Subtype AbstractInterpolant
    # ========================================
    @testset "LinearInterpolant subtypes AbstractInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(x)
        itp = linear_interp(x, y)

        @test itp isa FastInterpolations.AbstractInterpolant
        @test itp isa FastInterpolations.AbstractInterpolant{Float64}
        @test LinearInterpolant <: FastInterpolations.AbstractInterpolant
    end

    @testset "ConstantInterpolant subtypes AbstractInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(x)
        itp = constant_interp(x, y)

        @test itp isa FastInterpolations.AbstractInterpolant
        @test itp isa FastInterpolations.AbstractInterpolant{Float64}
        @test ConstantInterpolant <: FastInterpolations.AbstractInterpolant
    end

    @testset "QuadraticInterpolant subtypes AbstractInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(x)
        itp = quadratic_interp(x, y)

        @test itp isa FastInterpolations.AbstractInterpolant
        @test itp isa FastInterpolations.AbstractInterpolant{Float64}
        @test QuadraticInterpolant <: FastInterpolations.AbstractInterpolant
    end

    @testset "CubicInterpolant subtypes AbstractInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(x)
        itp = cubic_interp(x, y)

        @test itp isa FastInterpolations.AbstractInterpolant
        @test itp isa FastInterpolations.AbstractInterpolant{Float64}
        @test CubicInterpolant <: FastInterpolations.AbstractInterpolant
    end

    # ========================================
    # CubicMultiInterpolant Exists and Subtypes AbstractMultiInterpolant
    # ========================================
    @testset "CubicMultiInterpolant subtypes AbstractMultiInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y1, y2 = sin.(x), cos.(x)
        mitp = cubic_interp(x, [y1, y2])

        @test mitp isa FastInterpolations.AbstractMultiInterpolant
        @test mitp isa FastInterpolations.AbstractMultiInterpolant{Float64}

        # Check that CubicMultiInterpolant is exported and usable
        @test isdefined(FastInterpolations, :CubicMultiInterpolant)
    end

    # ========================================
    # Backward Compatibility: MultiCubicInterpolant Alias
    # ========================================
    @testset "MultiCubicInterpolant alias still works" begin
        x = collect(range(0.0, 1.0, 11))
        y1, y2 = sin.(x), cos.(x)
        mitp = cubic_interp(x, [y1, y2])

        # MultiCubicInterpolant should still be available
        @test isdefined(FastInterpolations, :MultiCubicInterpolant)
        @test mitp isa MultiCubicInterpolant

        # Both names should refer to the same type
        @test MultiCubicInterpolant === FastInterpolations.CubicMultiInterpolant

        # Functionality should work the same
        @test length(mitp(0.5)) == 2
    end

    # ========================================
    # Float32 Support
    # ========================================
    @testset "Float32 type parameter propagation" begin
        x = collect(range(0.0f0, 1.0f0, 11))
        y = sin.(x)

        itp_linear = linear_interp(x, y)
        @test itp_linear isa FastInterpolations.AbstractInterpolant{Float32}

        itp_constant = constant_interp(x, y)
        @test itp_constant isa FastInterpolations.AbstractInterpolant{Float32}

        itp_quad = quadratic_interp(x, y)
        @test itp_quad isa FastInterpolations.AbstractInterpolant{Float32}

        itp_cubic = cubic_interp(x, y)
        @test itp_cubic isa FastInterpolations.AbstractInterpolant{Float32}

        y1, y2 = sin.(x), cos.(x)
        mitp = cubic_interp(x, [y1, y2])
        @test mitp isa FastInterpolations.AbstractMultiInterpolant{Float32}
    end

    # ========================================
    # Type Hierarchy Structure
    # ========================================
    @testset "type hierarchy structure" begin
        # AbstractInterpolant should be at top of single interpolant hierarchy
        @test LinearInterpolant{Float64} <: FastInterpolations.AbstractInterpolant{Float64}
        @test ConstantInterpolant{Float64} <: FastInterpolations.AbstractInterpolant{Float64}
        @test QuadraticInterpolant{Float64} <: FastInterpolations.AbstractInterpolant{Float64}
        @test CubicInterpolant{Float64} <: FastInterpolations.AbstractInterpolant{Float64}

        # AbstractMultiInterpolant should be at top of multi interpolant hierarchy
        @test FastInterpolations.CubicMultiInterpolant{Float64} <: FastInterpolations.AbstractMultiInterpolant{Float64}
    end

end
