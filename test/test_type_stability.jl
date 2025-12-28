# Type stability tests using @inferred
using Test
using FastInterpolations
using FastInterpolations: BCPair, Deriv1, Deriv2, PeriodicBC, NaturalBC, ClampedBC, CubicSplineCache

@testset "Type Stability" begin
    x = collect(range(0.0, 2π, 10))
    y = sin.(x)
    x_query = [0.25, 0.5, 0.75]

    # =========================================================================
    # 2-arg form: CubicInterpolant construction is now type-stable
    # =========================================================================
    @testset "cubic_interp 2-arg @inferred" begin
        # All BC types should be inferrable
        @test @inferred(cubic_interp(x, y)) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=NaturalBC())) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=ClampedBC())) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=PeriodicBC())) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=BCPair(Deriv1(0.0), Deriv2(0.0)))) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=Deriv2(0.0))) isa CubicInterpolant  # PointBC
    end

    @testset "cubic_interp 2-arg extrap variations" begin
        # Different extrap values should all return the SAME type
        itp_none = @inferred cubic_interp(x, y; extrap=:none)
        itp_ext = @inferred cubic_interp(x, y; extrap=:extension)
        itp_const = @inferred cubic_interp(x, y; extrap=:constant)
        itp_wrap = @inferred cubic_interp(x, y; extrap=:wrap)

        # All should be the same concrete type (no E parameter anymore)
        @test typeof(itp_none) === typeof(itp_ext)
        @test typeof(itp_none) === typeof(itp_const)
        @test typeof(itp_none) === typeof(itp_wrap)

        # Type parameters should be {Float64, CubicSplineCache{...}}
        @test itp_none isa CubicInterpolant{Float64}
        @test itp_ext isa CubicInterpolant{Float64}
    end

    @testset "cubic_interp 2-arg BC + extrap combinations" begin
        # Various BC + extrap combinations
        @test @inferred(cubic_interp(x, y; bc=NaturalBC(), extrap=:extension)) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=ClampedBC(), extrap=:constant)) isa CubicInterpolant
        @test @inferred(cubic_interp(x, y; bc=BCPair(Deriv1(0.5), Deriv2(-0.5)), extrap=:wrap)) isa CubicInterpolant

        # Periodic BC always uses :wrap internally
        itp_periodic = @inferred cubic_interp(x, y; bc=PeriodicBC())
        @test itp_periodic.extrap === Val(:wrap)
    end

    # =========================================================================
    # 4-arg form: Vector/scalar return types
    # =========================================================================
    @testset "cubic_interp 4-arg Typed BC" begin
        @test @inferred(cubic_interp(x, y, x_query; bc=NaturalBC())) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; bc=ClampedBC())) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; bc=PeriodicBC())) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; bc=BCPair(Deriv1(0.0), Deriv2(0.0)))) isa Vector{Float64}

        # Scalar query
        @test @inferred(cubic_interp(x, y, 0.5; bc=NaturalBC())) isa Float64
        @test @inferred(cubic_interp(x, y, 0.5; bc=ClampedBC())) isa Float64
    end

    @testset "cubic_interp 4-arg extrap variations" begin
        # All extrap values return same Vector type
        @test @inferred(cubic_interp(x, y, x_query; extrap=:none)) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; extrap=:extension)) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; extrap=:constant)) isa Vector{Float64}
        @test @inferred(cubic_interp(x, y, x_query; extrap=:wrap)) isa Vector{Float64}

        # Scalar
        @test @inferred(cubic_interp(x, y, 0.5; extrap=:extension)) isa Float64
    end

    @testset "cubic_interp! in-place" begin
        out = similar(x_query)
        @test @inferred(cubic_interp!(out, x, y, x_query)) === out
        @test @inferred(cubic_interp!(out, x, y, x_query; bc=ClampedBC())) === out
        @test @inferred(cubic_interp!(out, x, y, x_query; extrap=:extension)) === out
    end

    # =========================================================================
    # Cache-based API
    # =========================================================================
    @testset "Cache-based API" begin
        cache = CubicSplineCache(x; bc=NaturalBC())
        @test @inferred(cubic_interp(cache, y, 0.5; extrap=:none)) isa Float64
        @test @inferred(cubic_interp(cache, y, 0.5; extrap=:extension)) isa Float64

        out = similar(x_query)
        @test @inferred(cubic_interp!(out, cache, y, x_query; extrap=:none)) === out

        # 2-arg with cache
        itp = @inferred cubic_interp(cache, y)
        @test itp isa CubicInterpolant{Float64}
    end

    # =========================================================================
    # CubicInterpolant callable
    # =========================================================================
    @testset "CubicInterpolant callable" begin
        itp = cubic_interp(x, y; bc=NaturalBC())

        # Scalar call
        @test @inferred(itp(0.5)) isa Float64
        @test @inferred(itp(1)) isa Float64  # Int → Float64 conversion

        # Vector call
        @test @inferred(itp(x_query)) isa Vector{Float64}

        # In-place
        out = similar(x_query)
        @test @inferred(itp(out, x_query)) === out
    end

    @testset "CubicInterpolant with different extrap" begin
        # Create interpolants with different extrap modes
        itp_none = cubic_interp(x, y; extrap=:none)
        itp_ext = cubic_interp(x, y; extrap=:extension)

        # Both should be callable with same return types
        @test @inferred(itp_none(0.5)) isa Float64
        @test @inferred(itp_ext(0.5)) isa Float64
        @test @inferred(itp_none(x_query)) isa Vector{Float64}
        @test @inferred(itp_ext(x_query)) isa Vector{Float64}

        # extrap field is different but type is same
        @test itp_none.extrap === Val(:none)
        @test itp_ext.extrap === Val(:extension)
        @test typeof(itp_none) === typeof(itp_ext)
    end

    # =========================================================================
    # Float32 type stability
    # =========================================================================
    @testset "Float32 type stability" begin
        x32 = Float32.(x)
        y32 = Float32.(y)
        xq32 = Float32.(x_query)

        # 2-arg form
        itp32 = @inferred cubic_interp(x32, y32)
        @test itp32 isa CubicInterpolant{Float32}

        # Callable
        @test @inferred(itp32(Float32(0.5))) isa Float32
        @test @inferred(itp32(xq32)) isa Vector{Float32}

        # 4-arg form
        @test @inferred(cubic_interp(x32, y32, xq32)) isa Vector{Float32}
        @test @inferred(cubic_interp(x32, y32, Float32(0.5))) isa Float32
    end

    # =========================================================================
    # Range vs Vector (O(1) vs O(log n) lookup preservation)
    # =========================================================================
    @testset "Range input preserves type" begin
        x_range = range(0.0, 2π, 10)
        y_range = sin.(collect(x_range))

        # Range should be preserved in cache
        itp_range = @inferred cubic_interp(x_range, y_range)
        @test itp_range.cache.x isa AbstractRange

        # Vector input
        itp_vec = @inferred cubic_interp(collect(x_range), y_range)
        @test itp_vec.cache.x isa Vector

        # Both are type-stable and produce same results
        @test @inferred(itp_range(0.5)) ≈ @inferred(itp_vec(0.5))
    end

    # =========================================================================
    # Linear interpolation type stability
    # =========================================================================
    @testset "linear_interp type stability" begin
        @test @inferred(linear_interp(x, y, x_query)) isa Vector{Float64}
        @test @inferred(linear_interp(x, y, 0.5)) isa Float64

        out = similar(x_query)
        @test @inferred(linear_interp!(out, x, y, x_query)) === out

        # LinearInterpolant
        litp = @inferred LinearInterpolant(x, y)
        @test @inferred(litp(0.5)) isa Float64
        @test @inferred(litp(x_query)) isa Vector{Float64}
    end
end
