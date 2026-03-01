using Test
using FastInterpolations

@testset "Series Wrapper" begin
    # Test data
    x = collect(range(0.0, 1.0, 11))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-x)

    # ────────────────────────────────────────────
    # Series construction
    # ────────────────────────────────────────────
    @testset "Construction" begin
        # Varargs
        s = Series(y1, y2, y3)
        @test s isa Series
        @test s.data isa Tuple

        # Vector of vectors
        s = Series([y1, y2, y3])
        @test s isa Series
        @test s.data isa AbstractVector{<:AbstractVector}

        # Matrix
        Y = hcat(y1, y2, y3)
        s = Series(Y)
        @test s isa Series
        @test s.data isa AbstractMatrix
    end

    # ────────────────────────────────────────────
    # _series_eltype
    # ────────────────────────────────────────────
    @testset "_series_eltype" begin
        @test FastInterpolations._series_eltype(Series(y1, y2)) == Float64
        @test FastInterpolations._series_eltype(Series([y1, y2])) == Float64
        @test FastInterpolations._series_eltype(Series(hcat(y1, y2))) == Float64

        # Mixed precision → promotion
        y_f32 = Float32.(y1)
        @test FastInterpolations._series_eltype(Series(y_f32, y2)) == Float64

        # Homogeneous Float32
        @test FastInterpolations._series_eltype(Series(y_f32, Float32.(y2))) == Float32
    end

    # ────────────────────────────────────────────
    # _build_series_mat correctness
    # ────────────────────────────────────────────
    @testset "_build_series_mat" begin
        n_pts = length(x)
        mat_ref = hcat(y1, y2, y3)

        # Tuple → Matrix
        s_tuple = Series(y1, y2, y3)
        mat, n_ser = FastInterpolations._build_series_mat(s_tuple, n_pts, Float64)
        @test mat ≈ mat_ref
        @test n_ser == 3

        # Vector of vectors → Matrix
        s_vecs = Series([y1, y2, y3])
        mat, n_ser = FastInterpolations._build_series_mat(s_vecs, n_pts, Float64)
        @test mat ≈ mat_ref
        @test n_ser == 3

        # Matrix → owned copy (not same object)
        Y = hcat(y1, y2, y3)
        s_mat = Series(Y)
        mat, n_ser = FastInterpolations._build_series_mat(s_mat, n_pts, Float64)
        @test mat ≈ mat_ref
        @test n_ser == 3
        @test mat !== Y  # must be a copy (owned by interpolant)
    end

    # ────────────────────────────────────────────
    # Type promotion in _build_series_mat
    # ────────────────────────────────────────────
    @testset "Type promotion" begin
        y_f32 = Float32.(y1)
        y_f64 = y2  # Float64
        n_pts = length(x)

        s = Series(y_f32, y_f64)
        mat, _ = FastInterpolations._build_series_mat(s, n_pts, Float64)
        @test eltype(mat) == Float64  # promoted to wider type
    end

    # ────────────────────────────────────────────
    # DimensionMismatch in _build_series_mat
    # ────────────────────────────────────────────
    @testset "DimensionMismatch" begin
        y_short = [1.0, 2.0]
        n_pts = length(x)
        @test_throws DimensionMismatch FastInterpolations._build_series_mat(
            Series(y1, y_short), n_pts, Float64)
        @test_throws DimensionMismatch FastInterpolations._build_series_mat(
            Series(hcat(y_short, y_short)), n_pts, Float64)
    end

    # ────────────────────────────────────────────
    # All 4 *_interp with Series produce correct types
    # ────────────────────────────────────────────
    @testset "Series dispatch: $method" for (method, SType) in [
        (linear_interp, LinearSeriesInterpolant),
        (cubic_interp, CubicSeriesInterpolant),
        (quadratic_interp, QuadraticSeriesInterpolant),
        (constant_interp, ConstantSeriesInterpolant),
    ]
        s = Series(y1, y2, y3)
        sitp = method(x, s)
        @test sitp isa SType
        @test length(sitp(0.5)) == 3  # returns vector of 3 series
    end

    # ────────────────────────────────────────────
    # Numerical equivalence: Series(Matrix) vs Series(y1, y2)
    # ────────────────────────────────────────────
    @testset "Numerical equivalence: $method" for (method, kwargs) in [
        (linear_interp, (;)),
        (cubic_interp, (;)),
        (quadratic_interp, (;)),
        (constant_interp, (;)),
    ]
        Y = hcat(y1, y2)

        sitp_mat = method(x, Series(Y); kwargs...)
        sitp_tuple = method(x, Series(y1, y2); kwargs...)

        xq = 0.37
        @test sitp_mat(xq) ≈ sitp_tuple(xq)

        xq_vec = [0.1, 0.5, 0.9]
        for (v_mat, v_tuple) in zip(sitp_mat(xq_vec), sitp_tuple(xq_vec))
            @test v_mat ≈ v_tuple
        end
    end

    # ────────────────────────────────────────────
    # Old paths (Vec{Vec} and Matrix) are removed — MethodError expected
    # ────────────────────────────────────────────
    @testset "Bare Matrix/Vec{Vec} dispatches removed" begin
        Y = hcat(y1, y2)
        @test_throws MethodError linear_interp(x, Y)
        @test_throws MethodError cubic_interp(x, Y)
        @test_throws MethodError quadratic_interp(x, Y)
        @test_throws MethodError constant_interp(x, Y)

        @test_throws MethodError linear_interp(x, [y1, y2])
        @test_throws MethodError cubic_interp(x, [y1, y2])
        @test_throws MethodError quadratic_interp(x, [y1, y2])
        @test_throws MethodError constant_interp(x, [y1, y2])
    end

    # ────────────────────────────────────────────
    # Integer grid promotion
    # ────────────────────────────────────────────
    @testset "Integer grid promotion" begin
        x_int = collect(1:10)
        y_a = Float64.(x_int) .^ 2
        y_b = Float64.(x_int) .^ 3

        sitp = linear_interp(x_int, Series(y_a, y_b))
        @test sitp isa LinearSeriesInterpolant
        @test sitp(5.5) isa AbstractVector
    end

    # ────────────────────────────────────────────
    # show methods
    # ────────────────────────────────────────────
    @testset "show" begin
        s = Series(y1, y2, y3)
        str = sprint(show, s)
        @test occursin("Series", str)
        @test occursin("3 series", str)
    end
end
