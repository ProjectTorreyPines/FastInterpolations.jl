# Test suite for Abstract Type Hierarchy

@testitem "Abstract Types Hierarchy" begin

    # ========================================
    # Abstract Types Exist and Are Abstract
    # ========================================
    @testset "AbstractInterpolant{T} exists and is abstract" begin
        @test isdefined(FastInterpolations, :AbstractInterpolant)
        @test isabstracttype(FastInterpolations.AbstractInterpolant)
        @test isabstracttype(FastInterpolations.AbstractInterpolant{Float64})
    end

    @testset "AbstractSeriesInterpolant{T} exists and is abstract" begin
        @test isdefined(FastInterpolations, :AbstractSeriesInterpolant)
        @test isabstracttype(FastInterpolations.AbstractSeriesInterpolant)
        @test isabstracttype(FastInterpolations.AbstractSeriesInterpolant{Float64})
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
    # SeriesInterpolants Subtype AbstractSeriesInterpolant
    # ========================================
    @testset "CubicSeriesInterpolant subtypes AbstractSeriesInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y1, y2 = sin.(x), cos.(x)
        sitp = cubic_interp(x, Series(y1, y2))

        @test sitp isa FastInterpolations.AbstractSeriesInterpolant
        @test sitp isa FastInterpolations.AbstractSeriesInterpolant{Float64}
        @test isdefined(FastInterpolations, :CubicSeriesInterpolant)
    end

    @testset "LinearSeriesInterpolant subtypes AbstractSeriesInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y1, y2 = sin.(x), cos.(x)
        sitp = linear_interp(x, Series(y1, y2))

        @test sitp isa FastInterpolations.AbstractSeriesInterpolant
        @test sitp isa FastInterpolations.AbstractSeriesInterpolant{Float64}
        @test isdefined(FastInterpolations, :LinearSeriesInterpolant)
    end

    @testset "ConstantSeriesInterpolant subtypes AbstractSeriesInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y1, y2 = sin.(x), cos.(x)
        sitp = constant_interp(x, Series(y1, y2))

        @test sitp isa FastInterpolations.AbstractSeriesInterpolant
        @test sitp isa FastInterpolations.AbstractSeriesInterpolant{Float64}
        @test isdefined(FastInterpolations, :ConstantSeriesInterpolant)
    end

    @testset "QuadraticSeriesInterpolant subtypes AbstractSeriesInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y1, y2 = sin.(x), cos.(x)
        sitp = quadratic_interp(x, Series(y1, y2))

        @test sitp isa FastInterpolations.AbstractSeriesInterpolant
        @test sitp isa FastInterpolations.AbstractSeriesInterpolant{Float64}
        @test isdefined(FastInterpolations, :QuadraticSeriesInterpolant)
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
        sitp = cubic_interp(x, Series(y1, y2))
        @test sitp isa FastInterpolations.AbstractSeriesInterpolant{Float32}
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

        # AbstractSeriesInterpolant should be at top of series interpolant hierarchy
        @test CubicSeriesInterpolant{Float64} <: FastInterpolations.AbstractSeriesInterpolant{Float64}
        @test LinearSeriesInterpolant{Float64} <: FastInterpolations.AbstractSeriesInterpolant{Float64}
        @test ConstantSeriesInterpolant{Float64} <: FastInterpolations.AbstractSeriesInterpolant{Float64}
        @test QuadraticSeriesInterpolant{Float64} <: FastInterpolations.AbstractSeriesInterpolant{Float64}
    end

end
