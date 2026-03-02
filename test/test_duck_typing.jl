# Tests for duck-typing support: custom value types (non-Real, non-Complex)
# Verifies that _promote_itp_inputs preserves custom types while still
# promoting standard numerics (Real, Complex) as before.

using Test
using FastInterpolations

# ========================================
# Custom Number Type for Testing
# ========================================
# Minimal "ring" type that wraps Float64 — mimics types like Unitful quantities.
# NOT <: Real, so it exercises the duck-typing path.

struct ScaledFloat <: Number
    val::Float64
end

# Ring operations required by interpolation kernels
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
Base.convert(::Type{ScaledFloat}, x::Real) = ScaledFloat(Float64(x))
Base.convert(::Type{ScaledFloat}, x::ScaledFloat) = x
Base.isapprox(a::ScaledFloat, b::ScaledFloat; kwargs...) = isapprox(a.val, b.val; kwargs...)

# Integer arithmetic (needed by spline coefficient solvers that use 2*s, 6*d, etc.)
Base.:*(a::Integer, b::ScaledFloat) = ScaledFloat(a * b.val)
Base.:*(a::ScaledFloat, b::Integer) = ScaledFloat(a.val * b)

# Comparison ops needed for domain checks / extrapolation
Base.:<(a::ScaledFloat, b::ScaledFloat) = a.val < b.val
Base.:>(a::ScaledFloat, b::ScaledFloat) = a.val > b.val

@testset "Duck Typing" begin

    # ────────────────────────────────────────────
    # Test data
    # ────────────────────────────────────────────
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_custom = ScaledFloat.([1.0, 4.0, 2.0, 5.0, 3.0])
    xq = 1.5

    # ────────────────────────────────────────────
    # Constant interpolation with custom Tv
    # ────────────────────────────────────────────
    @testset "Constant - custom Tv preserved" begin
        itp = constant_interp(x, y_custom)
        @test eltype(itp.y) === ScaledFloat
        result = itp(xq)
        @test result isa ScaledFloat
    end

    # ────────────────────────────────────────────
    # Linear interpolation with custom Tv
    # ────────────────────────────────────────────
    @testset "Linear - custom Tv preserved" begin
        itp = linear_interp(x, y_custom)
        @test eltype(itp.y) === ScaledFloat
        result = itp(xq)
        @test result isa ScaledFloat
        # Check value: linear interp at 1.5 between y[2]=4.0 and y[3]=2.0 → 3.0
        @test result ≈ ScaledFloat(3.0)
    end

    # ────────────────────────────────────────────
    # Integer grid + custom Tv
    # ────────────────────────────────────────────
    @testset "Integer grid + custom Tv" begin
        x_int = [0, 1, 2, 3, 4]
        itp = linear_interp(x_int, y_custom)
        @test eltype(itp.x) <: AbstractFloat  # grid promoted to Float
        @test eltype(itp.y) === ScaledFloat    # y preserved
        result = itp(1.5)
        @test result isa ScaledFloat
        @test result ≈ ScaledFloat(3.0)
    end

    # ────────────────────────────────────────────
    # Standard promotion still works (regression)
    # ────────────────────────────────────────────
    @testset "Standard promotion unchanged" begin
        # Int y → Float64
        itp_int = linear_interp([0.0, 1.0, 2.0], [10, 20, 30])
        @test eltype(itp_int.y) === Float64

        # Float32 y on Float64 grid → Float64 (widened)
        itp_f32 = linear_interp([0.0, 1.0, 2.0], Float32[1.0, 2.0, 3.0])
        @test eltype(itp_f32.y) === Float64

        # ComplexF64 preserved
        itp_cplx = linear_interp([0.0, 1.0, 2.0], ComplexF64[1+0im, 2+1im, 3+0im])
        @test eltype(itp_cplx.y) === ComplexF64

        # Int grid + Float64 y → Float64 grid
        itp_intgrid = linear_interp([0, 1, 2], [1.0, 2.0, 3.0])
        @test eltype(itp_intgrid.x) === Float64
    end

    # ────────────────────────────────────────────
    # Mixed precision widening (regression)
    # ────────────────────────────────────────────
    @testset "Mixed precision widening" begin
        # Float32 grid + Float64 y → grid widened to Float64
        itp = linear_interp(Float32[0, 1, 2, 3, 4], [1.0, 4.0, 2.0, 5.0, 3.0])
        @test eltype(itp.x) === Float64
        @test eltype(itp.y) === Float64
    end

    # ────────────────────────────────────────────
    # Quadratic interpolation with custom Tv
    # ────────────────────────────────────────────
    @testset "Quadratic - custom Tv preserved" begin
        itp = quadratic_interp(x, y_custom)
        @test eltype(itp.y) === ScaledFloat
        result = itp(xq)
        @test result isa ScaledFloat
    end

    # ────────────────────────────────────────────
    # Cubic interpolation with custom Tv
    # ────────────────────────────────────────────
    @testset "Cubic - custom Tv preserved" begin
        itp = cubic_interp(x, y_custom)
        @test eltype(itp.y) === ScaledFloat
        result = itp(xq)
        @test result isa ScaledFloat
    end

    # ────────────────────────────────────────────
    # Deriv BC with custom type (via auto-generated constructor)
    # ────────────────────────────────────────────
    @testset "Deriv BC accepts custom types" begin
        bc_val = ScaledFloat(0.0)
        bc = Deriv1(bc_val)
        @test bc isa Deriv1{ScaledFloat}
        @test bc.val === bc_val

        bc2 = Deriv2(bc_val)
        @test bc2 isa Deriv2{ScaledFloat}
    end

    # ────────────────────────────────────────────
    # _promote_itp_inputs directly
    # ────────────────────────────────────────────
    @testset "_promote_itp_inputs preserves custom Tv" begin
        x_in = [0.0, 1.0, 2.0]
        y_in = ScaledFloat.([1.0, 2.0, 3.0])
        x_out, y_out = FastInterpolations._promote_itp_inputs(x_in, y_in)
        @test x_out === x_in           # Float64 grid unchanged
        @test y_out === y_in           # Custom y NOT copied
        @test eltype(y_out) === ScaledFloat
    end

    @testset "_promote_itp_inputs promotes standard types" begin
        x_in = [0.0, 1.0, 2.0]
        y_int = [1, 2, 3]
        x_out, y_out = FastInterpolations._promote_itp_inputs(x_in, y_int)
        @test eltype(y_out) === Float64  # Int → Float64
        @test x_out === x_in             # grid unchanged
    end

end
