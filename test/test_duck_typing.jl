# Tests for duck-typing support: custom value types (non-Real, non-Complex)
# Verifies that _promote_itp_inputs preserves custom types while still
# promoting standard numerics (Real, Complex) as before.
#
# Three custom types tested:
# 1. ScaledFloat (<: Number, explicit muladd) — mimics Unitful quantities
# 2. MeasuredFloat (<: Number, value ± error) — mimics Measurements.jl
# 3. SVector (∉ Number, generic muladd fallback) — StaticArrays.jl

using Test
using FastInterpolations
using StaticArrays

# ================================================================
# Custom Type 1: ScaledFloat (<: Number)
# ================================================================
# Minimal "ring" type that wraps Float64.
# <: Number means muladd MUST be defined explicitly (Julia's promotion-based
# muladd(::Number, ::Number, ::Number) shadows the generic x*y+z fallback).

struct ScaledFloat <: Number
    val::Float64
end

# Identity constructor — resolves ambiguity between field ctor and (T)(x::T) for Number
ScaledFloat(x::ScaledFloat) = x

Base.zero(::Type{ScaledFloat}) = ScaledFloat(0.0)
Base.zero(::ScaledFloat) = ScaledFloat(0.0)
Base.:+(a::ScaledFloat, b::ScaledFloat) = ScaledFloat(a.val + b.val)
Base.:-(a::ScaledFloat, b::ScaledFloat) = ScaledFloat(a.val - b.val)
Base.:-(a::ScaledFloat) = ScaledFloat(-a.val)
Base.:*(a::Float64, b::ScaledFloat) = ScaledFloat(a * b.val)
Base.:*(a::ScaledFloat, b::Float64) = ScaledFloat(a.val * b)
Base.:*(a::ScaledFloat, b::ScaledFloat) = ScaledFloat(a.val * b.val)
Base.:/(a::ScaledFloat, b::ScaledFloat) = ScaledFloat(a.val / b.val)
Base.:/(a::ScaledFloat, b::Float64) = ScaledFloat(a.val / b)
Base.muladd(a::Float64, b::ScaledFloat, c::ScaledFloat) = ScaledFloat(muladd(a, b.val, c.val))
Base.muladd(a::ScaledFloat, b::Float64, c::ScaledFloat) = ScaledFloat(muladd(a.val, b, c.val))
Base.muladd(a::ScaledFloat, b::ScaledFloat, c::ScaledFloat) = ScaledFloat(muladd(a.val, b.val, c.val))
Base.convert(::Type{ScaledFloat}, x::Real) = ScaledFloat(Float64(x))
Base.convert(::Type{ScaledFloat}, x::ScaledFloat) = x
Base.isapprox(a::ScaledFloat, b::ScaledFloat; kwargs...) = isapprox(a.val, b.val; kwargs...)
Base.:*(a::Integer, b::ScaledFloat) = ScaledFloat(a * b.val)
Base.:*(a::ScaledFloat, b::Integer) = ScaledFloat(a.val * b)
Base.:<(a::ScaledFloat, b::ScaledFloat) = a.val < b.val
Base.:>(a::ScaledFloat, b::ScaledFloat) = a.val > b.val

# ================================================================
# Custom Type 2: MeasuredFloat (<: Number)
# ================================================================
# Value with uncertainty — simplified error propagation.
# Mimics Measurements.jl without the dependency.
# Key difference from ScaledFloat: carries TWO fields (val, err).

struct MeasuredFloat <: Number
    val::Float64
    err::Float64
end

# Single-arg constructors for Tv(scalar) calls in solver/ND paths
MeasuredFloat(x::Real) = MeasuredFloat(Float64(x), 0.0)
MeasuredFloat(x::MeasuredFloat) = x

# Ring operations (quadrature error propagation for additive ops)
Base.zero(::Type{MeasuredFloat}) = MeasuredFloat(0.0, 0.0)
Base.zero(::MeasuredFloat) = MeasuredFloat(0.0, 0.0)
Base.:+(a::MeasuredFloat, b::MeasuredFloat) = MeasuredFloat(a.val + b.val, hypot(a.err, b.err))
Base.:-(a::MeasuredFloat, b::MeasuredFloat) = MeasuredFloat(a.val - b.val, hypot(a.err, b.err))
Base.:-(a::MeasuredFloat) = MeasuredFloat(-a.val, a.err)

# Scalar multiplication (error scales linearly)
Base.:*(a::Float64, b::MeasuredFloat) = MeasuredFloat(a * b.val, abs(a) * b.err)
Base.:*(a::MeasuredFloat, b::Float64) = MeasuredFloat(a.val * b, a.err * abs(b))
Base.:/(a::MeasuredFloat, b::Float64) = MeasuredFloat(a.val / b, a.err / abs(b))

# Integer multiplication (for spline solvers: 2*s, 6*d)
Base.:*(a::Integer, b::MeasuredFloat) = MeasuredFloat(a * b.val, abs(a) * b.err)
Base.:*(a::MeasuredFloat, b::Integer) = MeasuredFloat(a.val * b, a.err * abs(b))

# Self-arithmetic (for coefficient computation and ND solver Tv(scalar) paths)
Base.:*(a::MeasuredFloat, b::MeasuredFloat) = MeasuredFloat(a.val * b.val, hypot(a.val * b.err, b.val * a.err))
Base.:/(a::MeasuredFloat, b::MeasuredFloat) = MeasuredFloat(a.val / b.val, hypot(a.err / b.val, a.val * b.err / b.val^2))

# muladd — REQUIRED for <: Number (promotion fallback fails without promote_rule)
Base.muladd(a::Float64, b::MeasuredFloat, c::MeasuredFloat) = a * b + c
Base.muladd(a::MeasuredFloat, b::Float64, c::MeasuredFloat) = a * b + c
Base.muladd(a::MeasuredFloat, b::MeasuredFloat, c::MeasuredFloat) = a * b + c

# Type conversion and comparison
Base.convert(::Type{MeasuredFloat}, x::Real) = MeasuredFloat(Float64(x), 0.0)
Base.convert(::Type{MeasuredFloat}, x::MeasuredFloat) = x
Base.isapprox(a::MeasuredFloat, b::MeasuredFloat; kwargs...) = isapprox(a.val, b.val; kwargs...)
Base.:<(a::MeasuredFloat, b::MeasuredFloat) = a.val < b.val
Base.:>(a::MeasuredFloat, b::MeasuredFloat) = a.val > b.val


@testset "Duck Typing" begin

    # ================================================================
    # 1D TEST DATA
    # ================================================================
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_scaled = ScaledFloat.([1.0, 4.0, 2.0, 5.0, 3.0])
    y_measured = [MeasuredFloat(v, 0.1) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
    y_svec = [SVector(v, 2v, 3v) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
    xq = 1.5

    # ================================================================
    # ScaledFloat — 1D (all 4 types)
    # ================================================================
    @testset "ScaledFloat 1D" begin
        @testset "Constant" begin
            itp = constant_interp(x, y_scaled)
            @test eltype(itp.y) === ScaledFloat
            @test itp(xq) isa ScaledFloat
        end

        @testset "Linear" begin
            itp = linear_interp(x, y_scaled)
            @test eltype(itp.y) === ScaledFloat
            result = itp(xq)
            @test result isa ScaledFloat
            @test result ≈ ScaledFloat(3.0)
        end

        @testset "Quadratic" begin
            itp = quadratic_interp(x, y_scaled)
            @test eltype(itp.y) === ScaledFloat
            @test itp(xq) isa ScaledFloat
        end

        @testset "Cubic" begin
            itp = cubic_interp(x, y_scaled)
            @test eltype(itp.y) === ScaledFloat
            @test itp(xq) isa ScaledFloat
        end

        @testset "Integer grid" begin
            itp = linear_interp([0, 1, 2, 3, 4], y_scaled)
            @test eltype(itp.x) <: AbstractFloat
            @test eltype(itp.y) === ScaledFloat
            @test itp(1.5) ≈ ScaledFloat(3.0)
        end

        @testset "Deriv BC" begin
            bc = Deriv1(ScaledFloat(0.0))
            @test bc isa Deriv1{ScaledFloat}
            bc2 = Deriv2(ScaledFloat(0.0))
            @test bc2 isa Deriv2{ScaledFloat}
        end
    end

    # ================================================================
    # MeasuredFloat — 1D (all 4 types)
    # ================================================================
    @testset "MeasuredFloat 1D" begin
        @testset "Constant" begin
            itp = constant_interp(x, y_measured)
            @test eltype(itp.y) === MeasuredFloat
            result = itp(xq)
            @test result isa MeasuredFloat
            @test result.err > 0  # uncertainty preserved
        end

        @testset "Linear" begin
            itp = linear_interp(x, y_measured)
            @test eltype(itp.y) === MeasuredFloat
            result = itp(xq)
            @test result isa MeasuredFloat
            @test result ≈ MeasuredFloat(3.0, 0.0)  # value correct
            @test result.err > 0  # uncertainty propagated through interpolation
        end

        @testset "Quadratic" begin
            itp = quadratic_interp(x, y_measured)
            @test eltype(itp.y) === MeasuredFloat
            result = itp(xq)
            @test result isa MeasuredFloat
            @test result.err > 0
        end

        @testset "Cubic" begin
            itp = cubic_interp(x, y_measured)
            @test eltype(itp.y) === MeasuredFloat
            result = itp(xq)
            @test result isa MeasuredFloat
            @test result.err > 0
        end

        @testset "Error propagation sanity" begin
            # Larger input errors → larger output errors
            y_small_err = [MeasuredFloat(v, 0.01) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
            y_large_err = [MeasuredFloat(v, 1.0) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
            itp_small = linear_interp(x, y_small_err)
            itp_large = linear_interp(x, y_large_err)
            @test itp_small(xq).err < itp_large(xq).err
            @test itp_small(xq).val ≈ itp_large(xq).val  # same central value
        end
    end

    # ================================================================
    # SVector — 1D (all 4 types)
    # ================================================================
    @testset "SVector 1D" begin
        @testset "Constant" begin
            itp = constant_interp(x, y_svec)
            @test eltype(itp.y) === SVector{3, Float64}
            @test itp(xq) isa SVector{3, Float64}
        end

        @testset "Linear" begin
            itp = linear_interp(x, y_svec)
            @test eltype(itp.y) === SVector{3, Float64}
            result = itp(xq)
            @test result isa SVector{3, Float64}
            # At xq=1.5: lerp between [4,8,12] and [2,4,6] → [3,6,9]
            @test result ≈ SVector(3.0, 6.0, 9.0)
        end

        @testset "Quadratic" begin
            itp = quadratic_interp(x, y_svec)
            @test eltype(itp.y) === SVector{3, Float64}
            result = itp(xq)
            @test result isa SVector{3, Float64}
            # Each component should be proportional: result[2] ≈ 2*result[1]
            @test result[2] ≈ 2 * result[1]
            @test result[3] ≈ 3 * result[1]
        end

        @testset "Cubic" begin
            itp = cubic_interp(x, y_svec)
            @test eltype(itp.y) === SVector{3, Float64}
            result = itp(xq)
            @test result isa SVector{3, Float64}
            @test result[2] ≈ 2 * result[1]
            @test result[3] ≈ 3 * result[1]
        end

        @testset "Grid point exactness" begin
            itp = linear_interp(x, y_svec)
            for i in eachindex(x)
                @test itp(x[i]) ≈ y_svec[i]
            end
        end
    end

    # ================================================================
    # ScaledFloat — 2D ND
    # ================================================================
    @testset "ScaledFloat 2D" begin
        xg = [0.0, 1.0, 2.0, 3.0]
        yg = [0.0, 1.0, 2.0, 3.0]
        # f(x,y) = x + 2y  (linear function → exact for linear interp)
        data_scaled = [ScaledFloat(xi + 2*yj) for xi in xg, yj in yg]

        @testset "Linear ND" begin
            itp = linear_interp((xg, yg), data_scaled)
            result = itp((0.5, 1.5))
            @test result isa ScaledFloat
            @test result ≈ ScaledFloat(0.5 + 2*1.5)
        end

        @testset "Constant ND" begin
            itp = constant_interp((xg, yg), data_scaled)
            result = itp((0.5, 1.5))
            @test result isa ScaledFloat
        end

        @testset "Quadratic ND" begin
            itp = quadratic_interp((xg, yg), data_scaled)
            result = itp((0.5, 1.5))
            @test result isa ScaledFloat
        end

        @testset "Cubic ND" begin
            # Cubic ND solver uses Tv(scalar) conversions and muladd(Tv, Tv, Tv)
            itp = cubic_interp((xg, yg), data_scaled)
            result = itp((0.5, 1.5))
            @test result isa ScaledFloat
        end
    end

    # ================================================================
    # SVector — 2D ND (vector-valued surface)
    # ================================================================
    @testset "SVector 2D" begin
        xg = [0.0, 1.0, 2.0, 3.0]
        yg = [0.0, 1.0, 2.0, 3.0]
        # Vector-valued: f(x,y) = [x+y, x-y] (linear)
        data_sv = [SVector(xi + yj, xi - yj) for xi in xg, yj in yg]

        @testset "Linear ND" begin
            itp = linear_interp((xg, yg), data_sv)
            result = itp((0.5, 1.5))
            @test result isa SVector{2, Float64}
            @test result ≈ SVector(0.5 + 1.5, 0.5 - 1.5)
        end

        @testset "Constant ND" begin
            itp = constant_interp((xg, yg), data_sv)
            result = itp((0.5, 1.5))
            @test result isa SVector{2, Float64}
        end

        @testset "Quadratic ND" begin
            itp = quadratic_interp((xg, yg), data_sv)
            result = itp((0.5, 1.5))
            @test result isa SVector{2, Float64}
        end

        @testset "Cubic ND" begin
            itp = cubic_interp((xg, yg), data_sv)
            result = itp((0.5, 1.5))
            @test result isa SVector{2, Float64}
        end

        @testset "Grid point exactness (linear)" begin
            itp = linear_interp((xg, yg), data_sv)
            for (i, xi) in enumerate(xg), (j, yj) in enumerate(yg)
                @test itp((xi, yj)) ≈ data_sv[i, j]
            end
        end
    end

    # ================================================================
    # MeasuredFloat — 2D ND
    # ================================================================
    @testset "MeasuredFloat 2D" begin
        xg = [0.0, 1.0, 2.0, 3.0]
        yg = [0.0, 1.0, 2.0, 3.0]
        data_mf = [MeasuredFloat(xi + yj, 0.05) for xi in xg, yj in yg]

        @testset "Linear ND" begin
            itp = linear_interp((xg, yg), data_mf)
            result = itp((0.5, 1.5))
            @test result isa MeasuredFloat
            @test result ≈ MeasuredFloat(2.0, 0.0)
            @test result.err > 0  # uncertainty propagated
        end

        @testset "Cubic ND" begin
            # MeasuredFloat is a scalar <: Number → Tv(scalar) works
            itp = cubic_interp((xg, yg), data_mf)
            result = itp((0.5, 1.5))
            @test result isa MeasuredFloat
            @test result.err > 0
        end
    end

    # ================================================================
    # Series — ScaledFloat (tests BC promotion to Tv_out)
    # ================================================================
    @testset "ScaledFloat Series" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y1_sf = ScaledFloat.([1.0, 4.0, 2.0, 5.0, 3.0])
        y2_sf = ScaledFloat.([2.0, 1.0, 5.0, 3.0, 4.0])

        @testset "Quadratic Series (default BC)" begin
            sitp = quadratic_interp(x, Series(y1_sf, y2_sf))
            result = sitp(1.5)
            @test length(result) == 2
            @test eltype(result) === ScaledFloat
        end

        @testset "Quadratic Series (Deriv1 BC)" begin
            bc = Left(Deriv1(ScaledFloat(0.0)))
            sitp = quadratic_interp(x, Series(y1_sf, y2_sf); bc=bc)
            result = sitp(1.5)
            @test eltype(result) === ScaledFloat
        end

        @testset "Cubic Series (default BC)" begin
            sitp = cubic_interp(x, Series(y1_sf, y2_sf))
            result = sitp(1.5)
            @test length(result) == 2
            @test eltype(result) === ScaledFloat
        end

        @testset "Cubic Series (ZeroCurvBC)" begin
            sitp = cubic_interp(x, Series(y1_sf, y2_sf); bc=ZeroCurvBC())
            result = sitp(1.5)
            @test eltype(result) === ScaledFloat
        end

        @testset "Linear Series" begin
            sitp = linear_interp(x, Series(y1_sf, y2_sf))
            result = sitp(1.5)
            @test length(result) == 2
            @test eltype(result) === ScaledFloat
        end

        @testset "Constant Series" begin
            sitp = constant_interp(x, Series(y1_sf, y2_sf))
            result = sitp(1.5)
            @test length(result) == 2
            @test eltype(result) === ScaledFloat
        end
    end

    # ================================================================
    # Series — MeasuredFloat (tests uncertainty propagation through series)
    # ================================================================
    @testset "MeasuredFloat Series" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y1_mf = [MeasuredFloat(v, 0.1) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
        y2_mf = [MeasuredFloat(v, 0.2) for v in [2.0, 1.0, 5.0, 3.0, 4.0]]

        @testset "Linear Series" begin
            sitp = linear_interp(x, Series(y1_mf, y2_mf))
            result = sitp(1.5)
            @test length(result) == 2
            @test all(r -> r isa MeasuredFloat, result)
            @test all(r -> r.err > 0, result)
        end

        @testset "Quadratic Series" begin
            sitp = quadratic_interp(x, Series(y1_mf, y2_mf))
            result = sitp(1.5)
            @test all(r -> r isa MeasuredFloat, result)
            @test all(r -> r.err > 0, result)
        end

        @testset "Cubic Series" begin
            sitp = cubic_interp(x, Series(y1_mf, y2_mf))
            result = sitp(1.5)
            @test all(r -> r isa MeasuredFloat, result)
            @test all(r -> r.err > 0, result)
        end
    end

    # ================================================================
    # Standard promotion (regression tests)
    # ================================================================
    @testset "Standard promotion unchanged" begin
        @test eltype(linear_interp([0.0, 1.0, 2.0], [10, 20, 30]).y) === Float64
        @test eltype(linear_interp([0.0, 1.0, 2.0], Float32[1, 2, 3]).y) === Float64
        @test eltype(linear_interp([0.0, 1.0, 2.0], ComplexF64[1, 2, 3]).y) === ComplexF64
        @test eltype(linear_interp([0, 1, 2], [1.0, 2.0, 3.0]).x) === Float64
    end

    @testset "Mixed precision widening" begin
        itp = linear_interp(Float32[0, 1, 2, 3, 4], [1.0, 4.0, 2.0, 5.0, 3.0])
        @test eltype(itp.x) === Float64
        @test eltype(itp.y) === Float64
    end

    # ================================================================
    # Internal API
    # ================================================================
    @testset "_promote_itp_inputs" begin
        @testset "preserves custom Tv" begin
            x_in = [0.0, 1.0, 2.0]
            y_in = ScaledFloat.([1.0, 2.0, 3.0])
            x_out, y_out = FastInterpolations._promote_itp_inputs(x_in, y_in)
            @test x_out === x_in
            @test y_out === y_in
            @test eltype(y_out) === ScaledFloat
        end

        @testset "preserves SVector Tv" begin
            x_in = [0.0, 1.0, 2.0]
            y_in = [SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 6.0)]
            x_out, y_out = FastInterpolations._promote_itp_inputs(x_in, y_in)
            @test x_out === x_in
            @test y_out === y_in
            @test eltype(y_out) === SVector{2, Float64}
        end

        @testset "promotes standard types" begin
            x_out, y_out = FastInterpolations._promote_itp_inputs([0.0, 1.0, 2.0], [1, 2, 3])
            @test eltype(y_out) === Float64
        end
    end

end
