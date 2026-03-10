# ═══════════════════════════════════════════════════════════════════════════════
# test_mixed_precision_extrap.jl
#
# Verify that FillExtrap/ClampExtrap return types are promoted correctly
# when data is Float32 but queries are Float64 (or vice versa).
#
# Bug: OOB path returned Float32 fill_value/y_bnd while in-domain kernel
# promoted to Float64 via arithmetic → Union{Float32, Float64} return type.
# Fix: _constant_extrap_result now promotes via _promote_extrap_val idiom.
#
# Note: Constant interpolation is intentionally excluded — it converts query
# to grid type (Tg) before evaluation, so both in-domain and OOB return Tv
# without promotion. No type instability exists for constant interp.
# ═══════════════════════════════════════════════════════════════════════════════

using Test
using FastInterpolations

@testset "Mixed-Precision Extrapolation Type Stability" begin
    # ── Shared test data (Float32 grids + values) ────────────────────────
    x32 = collect(range(0.0f0, 5.0f0, length = 11))
    y32 = sin.(x32)

    # Float64 query points: in-domain and OOB
    xq_in = 2.5       # inside [0, 5]
    xq_lo = -1.0      # below domain
    xq_hi = 6.0       # above domain

    # ── Cubic (oneshot) ──────────────────────────────────────────────────
    @testset "Cubic oneshot" begin
        @testset "FillExtrap" begin
            itp = cubic_interp(x32, y32; extrap = FillExtrap(NaN32))
            # All paths should return Float64 when queried with Float64
            @test @inferred(itp(xq_in)) isa Float64
            @test @inferred(itp(xq_lo)) isa Float64
            @test @inferred(itp(xq_hi)) isa Float64
            # OOB values should be NaN (promoted from NaN32)
            @test isnan(itp(xq_lo))
            @test isnan(itp(xq_hi))
            # In-domain should match Float32 result
            @test itp(xq_in) ≈ Float64(sin(Float32(xq_in))) atol = 0.01
        end

        @testset "ClampExtrap" begin
            itp = cubic_interp(x32, y32; extrap = ClampExtrap())
            @test @inferred(itp(xq_in)) isa Float64
            @test @inferred(itp(xq_lo)) isa Float64
            @test @inferred(itp(xq_hi)) isa Float64
            # OOB clamp to boundary values (promoted to Float64)
            @test itp(xq_lo) ≈ Float64(y32[1])
            @test itp(xq_hi) ≈ Float64(y32[end])
        end

        @testset "FillExtrap derivatives" begin
            itp = cubic_interp(x32, y32; extrap = FillExtrap(NaN32))
            # Derivative of constant fill → zero, promoted to Float64
            @test @inferred(itp(xq_lo; deriv = DerivOp(1))) isa Float64
            @test itp(xq_lo; deriv = DerivOp(1)) == 0.0
            @test @inferred(itp(xq_lo; deriv = DerivOp(2))) isa Float64
            @test itp(xq_lo; deriv = DerivOp(2)) == 0.0
        end

        @testset "Same-type no regression" begin
            # Float64 data + Float64 query: should still work
            x64 = collect(range(0.0, 5.0, length = 11))
            y64 = sin.(x64)
            itp64 = cubic_interp(x64, y64; extrap = FillExtrap(NaN))
            @test @inferred(itp64(2.5)) isa Float64
            @test @inferred(itp64(-1.0)) isa Float64
            @test isnan(itp64(-1.0))
        end
    end

    # ── Cubic (anchored / CubicInterpolant) ──────────────────────────────
    @testset "Cubic anchored" begin
        itp = cubic_interp(x32, y32; extrap = FillExtrap(NaN32))
        aq_in = FastInterpolations._anchor_query(x32, xq_in, Val(:cubic))
        aq_lo = FastInterpolations._anchor_query(x32, xq_lo, Val(:cubic))
        aq_hi = FastInterpolations._anchor_query(x32, xq_hi, Val(:cubic))
        @test @inferred(itp(aq_in)) isa Float64
        @test @inferred(itp(aq_lo)) isa Float64
        @test @inferred(itp(aq_hi)) isa Float64
        @test isnan(itp(aq_lo))
        @test isnan(itp(aq_hi))

        # ClampExtrap anchored path
        itp_c = cubic_interp(x32, y32; extrap = ClampExtrap())
        aq_lo_c = FastInterpolations._anchor_query(x32, xq_lo, Val(:cubic))
        @test @inferred(itp_c(aq_lo_c)) isa Float64
        @test itp_c(aq_lo_c) ≈ Float64(y32[1])
    end

    # ── Linear (oneshot) ─────────────────────────────────────────────────
    @testset "Linear oneshot" begin
        @testset "FillExtrap" begin
            itp = linear_interp(x32, y32; extrap = FillExtrap(NaN32))
            @test @inferred(itp(xq_in)) isa Float64
            @test @inferred(itp(xq_lo)) isa Float64
            @test @inferred(itp(xq_hi)) isa Float64
            @test isnan(itp(xq_lo))
        end

        @testset "ClampExtrap" begin
            itp = linear_interp(x32, y32; extrap = ClampExtrap())
            @test @inferred(itp(xq_in)) isa Float64
            @test @inferred(itp(xq_lo)) isa Float64
            @test @inferred(itp(xq_hi)) isa Float64
        end
    end

    # ── Linear (anchored / LinearInterpolant) ────────────────────────────
    @testset "Linear anchored" begin
        itp = LinearInterpolant(x32, y32; extrap = FillExtrap(NaN32))
        aq_lo = FastInterpolations._anchor_query(x32, xq_lo, Val(:linear))
        aq_hi = FastInterpolations._anchor_query(x32, xq_hi, Val(:linear))
        @test @inferred(itp(aq_lo)) isa Float64
        @test @inferred(itp(aq_hi)) isa Float64
        @test isnan(itp(aq_lo))
    end

    # ── Quadratic (oneshot) ──────────────────────────────────────────────
    @testset "Quadratic oneshot" begin
        @testset "FillExtrap" begin
            itp = quadratic_interp(x32, y32; extrap = FillExtrap(NaN32))
            @test @inferred(itp(xq_in)) isa Float64
            @test @inferred(itp(xq_lo)) isa Float64
            @test @inferred(itp(xq_hi)) isa Float64
            @test isnan(itp(xq_lo))
        end

        @testset "ClampExtrap" begin
            itp = quadratic_interp(x32, y32; extrap = ClampExtrap())
            @test @inferred(itp(xq_in)) isa Float64
            @test @inferred(itp(xq_lo)) isa Float64
            @test @inferred(itp(xq_hi)) isa Float64
        end
    end

    # ── Quadratic (anchored / QuadraticInterpolant) ──────────────────────
    @testset "Quadratic anchored" begin
        itp = quadratic_interp(x32, y32; extrap = FillExtrap(NaN32))
        aq_lo = FastInterpolations._anchor_query(x32, xq_lo, Val(:quadratic))
        aq_hi = FastInterpolations._anchor_query(x32, xq_hi, Val(:quadratic))
        @test @inferred(itp(aq_lo)) isa Float64
        @test @inferred(itp(aq_hi)) isa Float64
        @test isnan(itp(aq_lo))
    end

    # ── Reverse precision: Float64 data + Float32 query ──────────────────
    @testset "Reverse precision (Float64 data, Float32 query)" begin
        x64 = collect(range(0.0, 5.0, length = 11))
        y64 = sin.(x64)
        xq_lo32 = -1.0f0
        xq_hi32 = 6.0f0
        xq_in32 = 2.5f0

        @testset "Cubic" begin
            itp = cubic_interp(x64, y64; extrap = FillExtrap(NaN))
            # Float64 data + Float32 query → kernel promotes to Float64
            @test @inferred(itp(xq_in32)) isa Float64
            @test @inferred(itp(xq_lo32)) isa Float64
            @test @inferred(itp(xq_hi32)) isa Float64
            @test isnan(itp(xq_lo32))
        end

        @testset "Linear" begin
            itp = linear_interp(x64, y64; extrap = ClampExtrap())
            @test @inferred(itp(xq_in32)) isa Float64
            @test @inferred(itp(xq_lo32)) isa Float64
            @test @inferred(itp(xq_hi32)) isa Float64
        end

        @testset "Quadratic" begin
            itp = quadratic_interp(x64, y64; extrap = FillExtrap(NaN))
            @test @inferred(itp(xq_in32)) isa Float64
            @test @inferred(itp(xq_lo32)) isa Float64
            @test @inferred(itp(xq_hi32)) isa Float64
        end
    end

    # ── ND Cubic (2D) ────────────────────────────────────────────────────
    @testset "ND Cubic FillExtrap" begin
        x1_32 = collect(range(0.0f0, 4.0f0, length = 9))
        x2_32 = collect(range(0.0f0, 4.0f0, length = 9))
        data32 = Float32[sin(a + b) for a in x1_32, b in x2_32]
        itp_nd = cubic_interp((x1_32, x2_32), data32; extrap = FillExtrap(NaN32))
        # In-domain query with Float64 (ND takes tuple)
        @test @inferred(itp_nd((2.0, 2.0))) isa Float64
        # OOB query
        @test @inferred(itp_nd((-1.0, 2.0))) isa Float64
        @test isnan(itp_nd((-1.0, 2.0)))
    end
end
