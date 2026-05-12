# ═══════════════════════════════════════════════════════════════════════════════
# test_mixed_precision_extrap.jl
#
# Verify that FillExtrap/ClampExtrap return types are promoted correctly
# when data is Float32 but queries are Float64 (or vice versa).
#
# Bug: OOB path returned Float32 fill_value/y_bnd while in-domain kernel
# promoted to Float64 via arithmetic → Union{Float32, Float64} return type.
# Fix: _eval_extrapolation now promotes via _promote_extrap_val idiom.
#
# Note: Constant interpolation is intentionally excluded — it converts query
# to grid type (Tg) before evaluation, so both in-domain and OOB return Tv
# without promotion. No type instability exists for constant interp.
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "Mixed-Precision Extrapolation Type Stability" begin
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
            itp = cubic_interp(x32, y32; extrap = FillExtrap(Float32(NaN)))
            # All paths should return Float64 when queried with Float64
            @test @inferred(itp(xq_in)) isa Float64
            @test @inferred(itp(xq_lo)) isa Float64
            @test @inferred(itp(xq_hi)) isa Float64
            # OOB values should be NaN (promoted from Float32(NaN))
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
            itp = cubic_interp(x32, y32; extrap = FillExtrap(Float32(NaN)))
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
        itp = cubic_interp(x32, y32; extrap = FillExtrap(Float32(NaN)))
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
            itp = linear_interp(x32, y32; extrap = FillExtrap(Float32(NaN)))
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
        itp = LinearInterpolant(x32, y32; extrap = FillExtrap(Float32(NaN)))
        aq_lo = FastInterpolations._anchor_query(x32, xq_lo, Val(:linear))
        aq_hi = FastInterpolations._anchor_query(x32, xq_hi, Val(:linear))
        @test @inferred(itp(aq_lo)) isa Float64
        @test @inferred(itp(aq_hi)) isa Float64
        @test isnan(itp(aq_lo))
    end

    # ── Quadratic (oneshot) ──────────────────────────────────────────────
    @testset "Quadratic oneshot" begin
        @testset "FillExtrap" begin
            itp = quadratic_interp(x32, y32; extrap = FillExtrap(Float32(NaN)))
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
        itp = quadratic_interp(x32, y32; extrap = FillExtrap(Float32(NaN)))
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
        itp_nd = cubic_interp((x1_32, x2_32), data32; extrap = FillExtrap(Float32(NaN)))
        # In-domain query with Float64 (ND takes tuple)
        @test @inferred(itp_nd((2.0, 2.0))) isa Float64
        # OOB query
        @test @inferred(itp_nd((-1.0, 2.0))) isa Float64
        @test isnan(itp_nd((-1.0, 2.0)))
    end
end

@testitem "Tg promotion preserved on Integer Range + Float32 data" begin
    # Regression: pre-`_cache_axis` migration the persistent path threaded
    # the promoted `Tg` (computed from `_promote_grid_float(eltype(x), eltype(y))`)
    # all the way to the cached axis, so `linear_interp(1:4, Float32[...])`
    # produced `_CachedRange{Float32}` + `Vector{Float32}`. After the migration,
    # `_cache_axis(x::AbstractRange, ::AbstractBC) = _to_float(x, float(eltype(x)))`
    # ignored Tg, defaulted Int eltype to Float64, and the inner ctor's
    # `_promote_grid_float(Float64, Float32)` then widened y to Float64 too.
    # Cubic (still on the legacy `_resolve_axis_copied(x, bc, T)` builder) was
    # unaffected; the migrated families (Linear / PCHIP / Cardinal / Akima +
    # their Series + ND) lost the Float32 promotion contract.
    #
    # Constant is an exception: under the raw-eltype duck-typing policy
    # (`Tg = eltype(x)`, `Tv = eltype(y)`), `constant_interp(1:4, Float32[...])`
    # keeps `Tg = Int`. The selection kernel performs no x·y arithmetic, so
    # there is no precision argument for widening x. Verified explicitly below.
    x_int = 1:4
    y32 = Float32[1.0, 2.0, 3.0, 4.0]

    @testset "1D Linear" begin
        itp_l = linear_interp(x_int, y32)
        @test itp_l.x isa FastInterpolations._CachedRange{Float32}
        @test itp_l.y isa Vector{Float32}
    end

    @testset "1D Constant (raw-eltype: Tg stays Int)" begin
        itp_c = constant_interp(x_int, y32)
        @test itp_c.x isa FastInterpolations._CachedRange{Int, Float64}
        @test itp_c.y isa Vector{Float32}
    end

    @testset "1D Hermite family (PCHIP / Cardinal / Akima)" begin
        itp_p = pchip_interp(x_int, y32)
        @test itp_p.x isa FastInterpolations._CachedRange{Float32}
        @test itp_p.y isa Vector{Float32}

        itp_c = cardinal_interp(x_int, y32; tension = 0.0)
        @test itp_c.x isa FastInterpolations._CachedRange{Float32}
        @test itp_c.y isa Vector{Float32}

        itp_a = akima_interp(x_int, y32)
        @test itp_a.x isa FastInterpolations._CachedRange{Float32}
        @test itp_a.y isa Vector{Float32}
    end

    @testset "Cubic preserves Float32 (control)" begin
        # Cubic uses the legacy `_resolve_axis_copied(x, bc, T)` builder which
        # already threads Tg. Locks in the existing-correct behavior so any
        # future Cubic migration to `_cache_axis` doesn't regress.
        itp = cubic_interp(x_int, y32)
        @test itp.cache.x isa FastInterpolations._CachedRange{Float32}
        @test itp.y isa Vector{Float32}
    end

    @testset "1D Series (Linear)" begin
        sitp_l = linear_interp(x_int, Series(y32, y32))
        @test sitp_l.x isa FastInterpolations._CachedRange{Float32}
        @test sitp_l.y isa Matrix{Float32}
    end

    @testset "1D Series (Constant): raw-eltype keeps Tg = Int" begin
        sitp_c = constant_interp(x_int, Series(y32, y32))
        @test sitp_c.x isa FastInterpolations._CachedRange{Int, Float64}
        @test sitp_c.y isa Matrix{Float32}
    end

    # Note: ND (Linear/Constant) `_nd_promote_grids` computes `Tg =
    # float(promote(grid_eltypes))` *without* considering data eltype, so
    # ND `linear_interp((1:4, 1:5), Float32[...])` has historically widened
    # to Float64. That's a separate (pre-existing) limitation — not in scope
    # of the `_cache_axis` migration.
end
