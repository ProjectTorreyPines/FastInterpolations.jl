@testitem "PeriodicBC Exclusive Endpoint" setup = [AllocConstants] begin
    using FastInterpolations: _prepare_periodic, _prepare_periodic_nd,
        _resolve_exclusive_period,
        _can_infer_period, _is_periodic_bc, endpoint

    # ========================================
    # Construction Tests
    # ========================================
    @testset "PeriodicBC Construction" begin
        @testset "Backward compatibility" begin
            pbc = PeriodicBC()
            @test pbc isa PeriodicBC
            @test pbc isa AbstractBC
            @test endpoint(pbc) === :inclusive
            @test pbc.period === nothing
            @test _is_periodic_bc(pbc)

            # Singleton-like behavior: all default instances are equivalent
            @test PeriodicBC() === PeriodicBC()
        end

        @testset "Exclusive with explicit period" begin
            pbc = PeriodicBC(endpoint = :exclusive, period = 2π)
            @test pbc isa PeriodicBC{:exclusive, Float64}
            @test endpoint(pbc) === :exclusive
            @test pbc.period ≈ 2π
            @test _is_periodic_bc(pbc)

            # Float32 promotion
            pbc32 = PeriodicBC(endpoint = :exclusive, period = 1.0f0)
            @test pbc32 isa PeriodicBC{:exclusive, Float32}
            @test pbc32.period === 1.0f0

            # Integer promotion to Float64
            pbc_int = PeriodicBC(endpoint = :exclusive, period = 6)
            @test pbc_int isa PeriodicBC{:exclusive, Float64}
            @test pbc_int.period === 6.0
        end

        @testset "Exclusive without period (infer from Range)" begin
            pbc = PeriodicBC(endpoint = :exclusive)
            @test pbc isa PeriodicBC{:exclusive, Nothing}
            @test endpoint(pbc) === :exclusive
            @test pbc.period === nothing
        end

        @testset "Construction errors" begin
            # Invalid endpoint symbol
            @test_throws ArgumentError PeriodicBC(endpoint = :bad)
            @test_throws ArgumentError PeriodicBC(endpoint = :open)

            # Inclusive with period → error
            @test_throws ArgumentError PeriodicBC(endpoint = :inclusive, period = 1.0)
            @test_throws ArgumentError PeriodicBC(period = 1.0)  # default is :inclusive

            # Exclusive with non-positive period
            @test_throws ArgumentError PeriodicBC(endpoint = :exclusive, period = 0.0)
            @test_throws ArgumentError PeriodicBC(endpoint = :exclusive, period = -1.0)
        end
    end

    # ========================================
    # Period Resolution Tests
    # ========================================
    @testset "Period Resolution" begin
        @testset "Range grid auto-inference" begin
            x = range(0.0, step = 0.1, length = 10)
            bc = PeriodicBC(endpoint = :exclusive)
            period = _resolve_exclusive_period(x, bc)
            @test period ≈ 1.0  # step(x) * length(x) = 0.1 * 10
        end

        @testset "Range grid with matching explicit period" begin
            x = range(0.0, step = 0.1, length = 10)
            bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
            period = _resolve_exclusive_period(x, bc)
            @test period ≈ 1.0
        end

        @testset "Range grid with conflicting period → error" begin
            x = range(0.0, step = 0.1, length = 10)
            bc = PeriodicBC(endpoint = :exclusive, period = 2.0)  # doesn't match 0.1*10=1.0
            @test_throws ArgumentError _resolve_exclusive_period(x, bc)
        end

        @testset "Vector grid requires explicit period" begin
            x = [0.0, 0.3, 0.7, 1.5]
            bc = PeriodicBC(endpoint = :exclusive)
            @test_throws ArgumentError _resolve_exclusive_period(x, bc)

            # With period → OK
            bc2 = PeriodicBC(endpoint = :exclusive, period = 2π)
            @test _resolve_exclusive_period(x, bc2) ≈ 2π
        end

        @testset "_can_infer_period" begin
            @test _can_infer_period(range(0, 1, 10)) == true
            @test _can_infer_period([0.0, 1.0, 2.0]) == false
        end

        @testset "Mixed-precision period correctly rejected" begin
            # Float32 period on Float64 Range: isapprox with Float32's generous rtol
            # (~3e-4) would accept Float32(1.0002) ≈ 1.0, but grid-precision comparison
            # correctly rejects it.
            x = range(0.0, step = 0.1, length = 10)  # Float64, inferred period=1.0
            bc = PeriodicBC(endpoint = :exclusive, period = Float32(1.0002))
            @test_throws ArgumentError _resolve_exclusive_period(x, bc)

            # Float32 period that genuinely matches → accepted
            bc_ok = PeriodicBC(endpoint = :exclusive, period = Float32(1.0))
            @test _resolve_exclusive_period(x, bc_ok) == Float32(1.0)
        end
    end

    # ========================================
    # Data Extension Tests
    # ========================================
    @testset "Data Extension" begin
        @testset "Inclusive → no-op" begin
            x = [0.0, 1.0, 2.0]
            y = [1.0, 2.0, 1.0]
            bc = PeriodicBC()
            x2, y2 = _prepare_periodic(x, y, bc)
            @test x2 === x
            @test y2 === y
        end

        @testset "Exclusive Range → Range preserved" begin
            x = range(0.0, step = 0.5, length = 4)  # [0, 0.5, 1.0, 1.5]
            y = sin.(x)
            bc = PeriodicBC(endpoint = :exclusive)
            x_ext, y_ext = _prepare_periodic(x, y, bc)
            @test x_ext isa AbstractRange
            @test length(x_ext) == 5
            @test last(x_ext) ≈ 2.0  # step * length = 0.5 * 4
            @test y_ext[end] ≈ y[1]
        end

        @testset "Exclusive Vector" begin
            x = [0.0, 1.0, 3.0, 5.0]
            y = [1.0, 2.0, 3.0, 4.0]
            bc = PeriodicBC(endpoint = :exclusive, period = 2π)
            x_ext, y_ext = _prepare_periodic(x, y, bc)
            @test length(x_ext) == 5
            @test last(x_ext) ≈ 2π
            @test y_ext[end] == y[1]
        end

        @testset "Period validation at build time" begin
            x = [0.0, 1.0, 3.0, 5.0]
            y = [1.0, 2.0, 3.0, 4.0]
            # period too small: virtual endpoint at 4.0 which is < x[end]=5.0
            bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
            @test_throws ArgumentError _prepare_periodic(x, y, bc)
        end
    end

    # ========================================
    # Functional Tests: 2-arg Form (CubicInterpolant)
    # ========================================
    @testset "CubicInterpolant — Exclusive vs Inclusive equivalence" begin
        @testset "Range grid (period inferred)" begin
            N = 64
            dx = 2π / N
            x_excl = range(0.0, step = dx, length = N)
            y_excl = sin.(x_excl)

            x_incl = range(0.0, step = dx, length = N + 1)
            y_incl = sin.(x_incl)
            y_incl[end] = y_incl[1]

            itp_excl = cubic_interp(x_excl, y_excl; bc = PeriodicBC(endpoint = :exclusive))
            itp_incl = cubic_interp(x_incl, y_incl; bc = PeriodicBC())

            # Values should match at multiple query points
            for xq in [0.1, 1.0, π, 2π - 0.01, 3.5]
                @test itp_excl(xq) ≈ itp_incl(xq) atol = 1.0e-14
            end
        end

        @testset "Range grid with explicit period (redundant but valid)" begin
            N = 32
            dx = 2π / N
            x = range(0.0, step = dx, length = N)
            y = cos.(x)

            itp1 = cubic_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))
            itp2 = cubic_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 2π))

            for xq in [0.5, 1.5, π]
                @test itp1(xq) ≈ itp2(xq) atol = 1.0e-14
            end
        end

        @testset "Vector grid with period" begin
            x_incl = [0.0, 0.5, 1.5, 3.0, 5.0, 2π]
            y_incl = sin.(x_incl)
            y_incl[end] = y_incl[1]

            x_excl = x_incl[1:(end - 1)]  # remove last point
            y_excl = y_incl[1:(end - 1)]

            itp_incl = cubic_interp(x_incl, y_incl; bc = PeriodicBC())
            itp_excl = cubic_interp(x_excl, y_excl; bc = PeriodicBC(endpoint = :exclusive, period = 2π))

            for xq in [0.1, 1.0, π, 5.5]
                @test itp_excl(xq) ≈ itp_incl(xq) atol = 1.0e-14
            end
        end
    end

    # ========================================
    # Functional Tests: 4-arg Form (Oneshot)
    # ========================================
    @testset "Oneshot API — Exclusive endpoint" begin
        N = 32
        dx = 2π / N
        x = range(0.0, step = dx, length = N)
        y = sin.(x)

        @testset "Scalar query" begin
            val = cubic_interp(x, y, 1.0; bc = PeriodicBC(endpoint = :exclusive))
            @test val ≈ sin(1.0) atol = 1.0e-4
        end

        @testset "Vector query" begin
            xq = [0.5, 1.5, π]
            vals = cubic_interp(x, y, xq; bc = PeriodicBC(endpoint = :exclusive))
            @test length(vals) == 3
            for i in eachindex(xq)
                @test vals[i] ≈ sin(xq[i]) atol = 1.0e-4
            end
        end

        @testset "In-place vector query" begin
            xq = [0.5, 1.5, π]
            output = zeros(3)
            cubic_interp!(output, x, y, xq; bc = PeriodicBC(endpoint = :exclusive))
            for i in eachindex(xq)
                @test output[i] ≈ sin(xq[i]) atol = 1.0e-4
            end
        end
    end

    @testset "Oneshot API — Exclusive endpoint (Vector grid)" begin
        # Build inclusive reference from a non-uniform grid
        x_incl = [0.0, 0.5, 1.5, 3.0, 5.0, 2π]
        y_incl = sin.(x_incl)
        y_incl[end] = y_incl[1]
        x_excl = x_incl[1:(end - 1)]
        y_excl = y_incl[1:(end - 1)]
        bc_excl = PeriodicBC(endpoint = :exclusive, period = 2π)

        itp_ref = cubic_interp(x_incl, y_incl; bc = PeriodicBC())

        @testset "Scalar query" begin
            for xq in [0.1, 1.0, π, 5.5]
                val = cubic_interp(x_excl, y_excl, xq; bc = bc_excl)
                @test val ≈ itp_ref(xq) atol = 1.0e-14
            end
        end

        @testset "Vector query" begin
            xq = [0.1, 1.0, π, 5.5]
            vals = cubic_interp(x_excl, y_excl, xq; bc = bc_excl)
            for i in eachindex(xq)
                @test vals[i] ≈ itp_ref(xq[i]) atol = 1.0e-14
            end
        end

        @testset "In-place vector query" begin
            xq = [0.1, 1.0, π, 5.5]
            output = zeros(4)
            cubic_interp!(output, x_excl, y_excl, xq; bc = bc_excl)
            for i in eachindex(xq)
                @test output[i] ≈ itp_ref(xq[i]) atol = 1.0e-14
            end
        end
    end

    # ========================================
    # Series Interpolant Tests
    # ========================================
    @testset "CubicSeriesInterpolant — Exclusive endpoint" begin
        N = 32
        dx = 2π / N
        x = range(0.0, step = dx, length = N)
        y1 = sin.(x)
        y2 = cos.(x)

        mitp = cubic_interp(x, Series(y1, y2); bc = PeriodicBC(endpoint = :exclusive))

        vals = mitp(1.0)
        @test vals[1] ≈ sin(1.0) atol = 1.0e-4
        @test vals[2] ≈ cos(1.0) atol = 1.0e-4
    end

    @testset "CubicSeriesInterpolant — Exclusive endpoint (Vector grid)" begin
        # Build inclusive reference from non-uniform grid
        x_incl = [0.0, 0.5, 1.5, 3.0, 5.0, 2π]
        y1_incl = sin.(x_incl)
        y2_incl = cos.(x_incl)
        y1_incl[end] = y1_incl[1]
        y2_incl[end] = y2_incl[1]

        x_excl = x_incl[1:(end - 1)]
        y1_excl = y1_incl[1:(end - 1)]
        y2_excl = y2_incl[1:(end - 1)]

        bc_excl = PeriodicBC(endpoint = :exclusive, period = 2π)

        mitp_incl = cubic_interp(x_incl, Series(y1_incl, y2_incl); bc = PeriodicBC())
        mitp_excl = cubic_interp(x_excl, Series(y1_excl, y2_excl); bc = bc_excl)

        for xq in [0.1, 1.0, π, 5.5]
            v_incl = mitp_incl(xq)
            v_excl = mitp_excl(xq)
            @test v_excl[1] ≈ v_incl[1] atol = 1.0e-14
            @test v_excl[2] ≈ v_incl[2] atol = 1.0e-14
        end
    end

    # ========================================
    # Derivative Tests
    # ========================================
    @testset "Derivatives — Exclusive endpoint" begin
        N = 64
        dx = 2π / N
        x = range(0.0, step = dx, length = N)
        y = sin.(x)

        itp = cubic_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))

        # C2 continuity at wrap point: derivatives should match across boundary
        ε = 1.0e-6
        d1_left = itp(ε; deriv = DerivOp(1))
        d1_right = itp(2π - ε; deriv = DerivOp(1))
        # sin'(0) ≈ sin'(2π) ≈ cos(0) = 1.0
        @test d1_left ≈ d1_right atol = 1.0e-3

        # First derivative accuracy
        @test itp(π / 4; deriv = DerivOp(1)) ≈ cos(π / 4) atol = 1.0e-3

        # Second derivative
        @test itp(π / 2; deriv = DerivOp(2)) ≈ -sin(π / 2) atol = 0.1
    end

    # ========================================
    # Type Stability Tests
    # ========================================
    @testset "Type stability" begin
        N = 16
        dx = 2π / N

        # Inclusive data for baseline
        x_incl = range(0.0, step = dx, length = N + 1)
        y_incl = sin.(x_incl)
        y_incl[end] = y_incl[1]
        @test @inferred(cubic_interp(x_incl, y_incl; bc = PeriodicBC())) isa CubicInterpolant
        @test @inferred(cubic_interp(x_incl, y_incl; bc = PeriodicBC(endpoint = :exclusive))) isa CubicInterpolant

        # Exclusive: isa-based grid extension → construction and evaluation are type-stable
        x_excl = range(0.0, step = dx, length = N)
        y_excl = sin.(x_excl)
        itp = cubic_interp(x_excl, y_excl; bc = PeriodicBC(endpoint = :exclusive))
        @test itp isa CubicInterpolant
        @test @inferred(itp(1.0)) isa Float64
    end

    @testset "_prepare_periodic type stability (isa branch)" begin
        N = 16
        dx = 2π / N

        # Range grid + exclusive: should return Range (compile-time isa narrowing)
        x_range = range(0.0, step = dx, length = N)
        y_range = sin.(x_range)
        bc_excl = PeriodicBC(endpoint = :exclusive)
        result_range = @inferred _prepare_periodic(x_range, y_range, bc_excl)
        x_ext, y_ext = result_range
        @test x_ext isa AbstractRange
        @test length(x_ext) == N + 1
        @test last(x_ext) ≈ first(x_range) + dx * N
        @test y_ext[end] ≈ y_ext[1]

        # Vector grid + exclusive: should return Vector (compile-time isa narrowing)
        x_vec = [0.0, 0.5, 1.5, 3.0, 5.0]
        y_vec = sin.(x_vec)
        bc_vec = PeriodicBC(endpoint = :exclusive, period = 2π)
        result_vec = @inferred _prepare_periodic(x_vec, y_vec, bc_vec)
        x_ext_v, y_ext_v = result_vec
        @test x_ext_v isa Vector
        @test length(x_ext_v) == 6
        @test last(x_ext_v) ≈ 2π
        @test y_ext_v[end] ≈ y_ext_v[1]

        # Inclusive: no-op passthrough (trivially type-stable)
        bc_incl = PeriodicBC()
        result_incl = @inferred _prepare_periodic(x_range, y_range, bc_incl)
        @test result_incl === (x_range, y_range)
    end

    # ========================================
    # Show / Display Tests
    # ========================================
    @testset "Show methods" begin
        @test FastInterpolations._format_bc(PeriodicBC()) == "Periodic"
        @test FastInterpolations._short_bc_name(PeriodicBC()) == "Periodic"

        bc_excl = PeriodicBC(endpoint = :exclusive, period = 2π)
        @test occursin("exclusive", FastInterpolations._format_bc(bc_excl))
        @test FastInterpolations._short_bc_name(bc_excl) == "Periodic(excl)"

        # Series interpolant show (bc_for_solve is PeriodicData, not PeriodicBC)
        N = 16
        dx = 2π / N
        x_incl = range(0.0, step = dx, length = N + 1)
        y_incl = sin.(x_incl)
        y_incl[end] = y_incl[1]
        y_cos = cos.(x_incl)
        y_cos[end] = y_cos[1]
        sitp = cubic_interp(x_incl, Series(y_incl, y_cos); bc = PeriodicBC())
        buf = IOBuffer()
        show(buf, sitp)
        @test occursin("Periodic", String(take!(buf)))
        show(buf, MIME"text/plain"(), sitp)
        @test occursin("Periodic", String(take!(buf)))

        # `itp.bc` is normalized to `:inclusive` post-extension, with period
        # materialized from the cache for introspection.
        @testset "itp.bc reflects post-extension `:inclusive` form (period preserved)" begin
            N_bc = 16
            dx_bc = 2π / N_bc

            # Exclusive input → normalized to `:inclusive`, period preserved
            x_bc = range(0.0, step = dx_bc, length = N_bc)
            y_bc = sin.(x_bc)
            itp_excl = cubic_interp(collect(x_bc), y_bc; bc = PeriodicBC(endpoint = :exclusive, period = 2π))
            @test itp_excl.bc isa PeriodicBC{:inclusive}
            @test itp_excl.bc.period ≈ 2π

            # Exclusive without period (auto-inferred from Range)
            itp_excl_auto = cubic_interp(x_bc, y_bc; bc = PeriodicBC(endpoint = :exclusive))
            @test itp_excl_auto.bc isa PeriodicBC{:inclusive}
            @test itp_excl_auto.bc.period ≈ 2π

            buf_bc = IOBuffer()
            show(buf_bc, MIME"text/plain"(), itp_excl_auto)
            s = String(take!(buf_bc))
            @test occursin("Periodic", s)
            @test occursin("period≈", s)

            # Inclusive input — passthrough; period materialized too
            x_incl_bc = range(0.0, step = dx_bc, length = N_bc + 1)
            y_incl_bc = sin.(x_incl_bc); y_incl_bc[end] = y_incl_bc[1]
            itp_incl = cubic_interp(collect(x_incl_bc), y_incl_bc; bc = PeriodicBC())
            @test itp_incl.bc isa PeriodicBC{:inclusive}
            @test itp_incl.bc.period ≈ 2π

            show(buf_bc, MIME"text/plain"(), itp_incl)
            @test occursin("period≈", String(take!(buf_bc)))
        end

        # Also test exclusive series interpolant show
        x_excl = range(0.0, step = dx, length = N)
        y_excl = sin.(x_excl)
        sitp_excl = cubic_interp(x_excl, Series(y_excl, cos.(x_excl)); bc = PeriodicBC(endpoint = :exclusive))
        show(buf, sitp_excl)
        @test occursin("Periodic", String(take!(buf)))
        show(buf, MIME"text/plain"(), sitp_excl)
        @test occursin("Periodic", String(take!(buf)))
    end

    # ========================================
    # Zero-Allocation Tests (One-Shot, Pool-Based)
    # ========================================
    @testset "One-shot exclusive zero-alloc" begin
        x = range(0.0, step = 2π / 16, length = 16)
        y = sin.(x)
        bc = PeriodicBC(endpoint = :exclusive)

        @testset "scalar (Range grid)" begin
            cubic_interp(x, y, 1.0; bc = bc)  # warmup
            alloc = @allocated cubic_interp(x, y, 1.0; bc = bc)
            @test alloc <= ALLOC_THRESHOLD
        end

        @testset "vector in-place (Range grid)" begin
            xq = [0.5, 1.0, 2.0]
            out = similar(xq)
            cubic_interp!(out, x, y, xq; bc = bc)  # warmup
            alloc = @allocated cubic_interp!(out, x, y, xq; bc = bc)
            @test alloc <= ALLOC_THRESHOLD
        end

        # Vector grid path uses pool-based unsafe_acquire! + copyto!
        x_vec = [0.0, 0.5, 1.5, 3.0, 5.0]
        y_vec = sin.(x_vec)
        bc_vec = PeriodicBC(endpoint = :exclusive, period = 2π)

        @testset "scalar (Vector grid)" begin
            cubic_interp(x_vec, y_vec, 1.0; bc = bc_vec)  # warmup
            alloc = @allocated cubic_interp(x_vec, y_vec, 1.0; bc = bc_vec)
            @test alloc <= ALLOC_THRESHOLD
        end

        @testset "vector in-place (Vector grid)" begin
            xq_v = [0.5, 1.0, 2.0]
            out_v = similar(xq_v)
            cubic_interp!(out_v, x_vec, y_vec, xq_v; bc = bc_vec)  # warmup
            alloc = @allocated cubic_interp!(out_v, x_vec, y_vec, xq_v; bc = bc_vec)
            @test alloc <= ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Edge Cases
    # ========================================
    @testset "Edge cases" begin
        @testset "Vector grid without period → error" begin
            x = [0.0, 1.0, 2.0, 3.0]
            y = sin.(x)
            @test_throws ArgumentError cubic_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))
        end

        @testset "Range grid + conflicting period → error" begin
            x = range(0.0, step = 0.1, length = 10)
            y = sin.(x)
            # 0.1 * 10 = 1.0, but user says period=2.0
            @test_throws ArgumentError cubic_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 2.0))
        end

        @testset "Minimum points (4 for periodic)" begin
            x = range(0.0, step = 1.0, length = 4)
            y = [0.0, 1.0, 0.0, -1.0]
            itp = cubic_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))
            @test itp(0.5) isa Float64
        end

        @testset "Float32 support" begin
            x = range(0.0f0, step = Float32(2π / 16), length = 16)
            y = sin.(x)
            itp = cubic_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))
            @test itp(1.0f0) isa Float32
        end

        @testset "CubicSplineCache rejects exclusive PeriodicBC (direct construction)" begin
            # The internal cache pool handles `:exclusive` correctly via the
            # public oneshot/persistent APIs. Direct cache construction is
            # rejected because the cache-direct eval path
            # (`cubic_interp!(out, cache, y, xq)`) does not thread BC into the
            # searcher, so seam-cell queries would silently mis-evaluate.
            x = range(0.0, step = 0.1, length = 10)
            @test_throws ArgumentError CubicSplineCache(x; bc = PeriodicBC(endpoint = :exclusive))
        end

        @testset "Cache key includes bc.period — distinct periods on same x (codex P1)" begin
            # Same Vector x called twice with different explicit periods: each
            # must build/lookup a distinct cache. Pre-fix: bank partitioned by
            # `(T, X, S, E)` only and `_rcu_lookup` compared `x` only, so the
            # second call silently hit the first cache and used the stale
            # period in its bc_config (wrong seam_h, wrong q vector).
            x = collect([0.0, 0.3, 0.6, 0.85, 1.05, 1.4, 1.85])    # non-uniform
            y = [0.0, 1.0, 0.5, -0.5, 0.3, 0.8, -0.2]
            xq = 1.95   # in seam region — sensitive to seam_h

            v1 = cubic_interp(x, y, xq; bc = PeriodicBC(endpoint = :exclusive, period = 2.5))
            v2 = cubic_interp(x, y, xq; bc = PeriodicBC(endpoint = :exclusive, period = 2.2))
            v3 = cubic_interp(x, y, xq; bc = PeriodicBC(endpoint = :exclusive, period = 2.5))

            # v1 and v3 should match (same period). v1 vs v2 should differ
            # (different seam geometry → different result).
            @test v1 ≈ v3
            @test !isapprox(v1, v2; atol = 1.0e-6)

            # Direct cache pool inspection — distinct cache objects for distinct periods.
            c1 = FastInterpolations._get_cubic_cache(x, PeriodicBC(endpoint = :exclusive, period = 2.5))
            c2 = FastInterpolations._get_cubic_cache(x, PeriodicBC(endpoint = :exclusive, period = 2.2))
            @test c1.bc_config.period ≈ 2.5
            @test c2.bc_config.period ≈ 2.2
            @test c1.bc_config.h_n ≈ 0.65
            @test c2.bc_config.h_n ≈ 0.35
        end

        @testset "Reject non-positive seam width (codex P2.1)" begin
            # Pre-fix: `_build_periodic_cache(x, bc::PeriodicBC{:exclusive})`
            # computed `h_n = period - (last(x) - first(x))` without checking
            # `h_n > 0`. Master path threw via `_extend_exclusive`'s
            # `last(x) < x_end` validation; zero-copy bypass lost that.
            x = collect([0.0, 0.3, 0.6, 0.85, 1.05, 1.4, 1.85])    # span = 1.85
            y = sin.(2π .* x ./ 2.0)

            # period == grid span → h_n == 0 → divide by zero in solver
            @test_throws ArgumentError cubic_interp(x, y, 0.5; bc = PeriodicBC(endpoint = :exclusive, period = 1.85))

            # period < grid span → h_n < 0 → invalid seam geometry
            @test_throws ArgumentError cubic_interp(x, y, 0.5; bc = PeriodicBC(endpoint = :exclusive, period = 1.5))
        end

        @testset "autocache=false routes through internal builder (codex P2.2)" begin
            # Pre-fix: `_get_cubic_cache(x, bc, autocache=false)` routed to
            # `CubicSplineCache(...; bc=bc)` outer constructor, which (post
            # commit 7d9b125e) rejects `:exclusive` direct construction. So
            # the public oneshot API silently broke for `autocache=false`.
            x = collect(range(0.0, 1.0, length = 11))[1:10]
            y = sin.(2π .* x)
            bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

            v_true = cubic_interp(x, y, 0.5; bc = bc, autocache = true)
            v_false = cubic_interp(x, y, 0.5; bc = bc, autocache = false)
            @test v_true ≈ v_false
        end
    end

end

# ========================================
# ND Exclusive Endpoint Tests
# ========================================

@testitem "PeriodicBC Exclusive Endpoint — ND" begin
    using FastInterpolations: _prepare_periodic, _prepare_periodic_nd,
        _resolve_exclusive_period,
        _can_infer_period, _is_periodic_bc, endpoint

    # ========================================
    # _prepare_periodic_nd unit tests
    # ========================================
    @testset "_prepare_periodic_nd" begin
        @testset "No exclusive axes → no-op" begin
            x = range(0.0, 2π, 11)
            y = range(0.0, π, 9)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            bcs = (PeriodicBC(), ZeroCurvBC())

            grids_out, data_out, bcs_out = _prepare_periodic_nd((x, y), data, bcs)
            @test grids_out === (x, y)
            @test data_out === data
            @test bcs_out === bcs
        end

        @testset "Single exclusive axis extends correctly" begin
            N = 8
            x = range(0.0, step = 2π / N, length = N)
            y = range(0.0, π, 5)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            bcs = (PeriodicBC(endpoint = :exclusive), ZeroCurvBC())

            grids_out, data_out, bcs_out = _prepare_periodic_nd((x, y), data, bcs)

            @test length(grids_out[1]) == N + 1         # extended
            @test length(grids_out[2]) == 5              # unchanged
            @test size(data_out) == (N + 1, 5)
            @test data_out[end, :] ≈ data_out[1, :]      # first slice copied
            @test grids_out[2] === y                      # unchanged reference
            # Post-extension: bc is normalized to `:inclusive` (grid is now in
            # closed-cycle inclusive form, so the seam cell is the last real cell).
            # The resolved period is recoverable as `last(grid) - first(grid)`.
            @test bcs_out[1] isa PeriodicBC{:inclusive}
            @test last(grids_out[1]) - first(grids_out[1]) ≈ 2π
            @test bcs_out[2] === ZeroCurvBC()              # unchanged
        end

        @testset "Both axes exclusive" begin
            Nx, Ny = 8, 6
            x = range(0.0, step = 2π / Nx, length = Nx)
            y = range(0.0, step = π / Ny, length = Ny)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            bcs = (PeriodicBC(endpoint = :exclusive), PeriodicBC(endpoint = :exclusive))

            grids_out, data_out, bcs_out = _prepare_periodic_nd((x, y), data, bcs)

            @test size(data_out) == (Nx + 1, Ny + 1)
            @test data_out[end, :] ≈ data_out[1, :]       # dim 1 wrap
            @test data_out[:, end] ≈ data_out[:, 1]        # dim 2 wrap
            @test data_out[end, end] ≈ data_out[1, 1]      # corner
            # Post-extension period is encoded in the grid span (not bc.period).
            @test last(grids_out[1]) - first(grids_out[1]) ≈ 2π
            @test last(grids_out[2]) - first(grids_out[2]) ≈ π
        end

        @testset "Range preserved after extension" begin
            x = range(0.0, step = 0.5, length = 4)
            y = range(0.0, 1.0, 5)
            data = zeros(4, 5)
            bcs = (PeriodicBC(endpoint = :exclusive), ZeroCurvBC())

            result = @inferred _prepare_periodic_nd((x, y), data, bcs)
            grids_out, _, _ = result
            @test grids_out[1] isa AbstractRange
        end
    end

    # ========================================
    # 2D Functional Tests: Exclusive vs Inclusive equivalence
    # ========================================
    @testset "2D CubicInterpolantND — Exclusive vs Inclusive equivalence" begin
        @testset "Both axes periodic (Range, period inferred)" begin
            Nx, Ny = 32, 24
            dx, dy = 2π / Nx, 2π / Ny
            x_excl = range(0.0, step = dx, length = Nx)
            y_excl = range(0.0, step = dy, length = Ny)
            data_excl = [sin(xi) * cos(yj) for xi in x_excl, yj in y_excl]

            x_incl = range(0.0, step = dx, length = Nx + 1)
            y_incl = range(0.0, step = dy, length = Ny + 1)
            data_incl = [sin(xi) * cos(yj) for xi in x_incl, yj in y_incl]
            # Enforce exact periodicity on both axes
            data_incl[end, :] .= data_incl[1, :]
            data_incl[:, end] .= data_incl[:, 1]

            itp_excl = cubic_interp(
                (x_excl, y_excl), data_excl;
                bc = PeriodicBC(endpoint = :exclusive)
            )
            itp_incl = cubic_interp(
                (x_incl, y_incl), data_incl;
                bc = PeriodicBC()
            )

            for (xq, yq) in [(0.5, 0.5), (π, 1.0), (1.0, π), (5.0, 5.0)]
                @test itp_excl((xq, yq)) ≈ itp_incl((xq, yq)) atol = 1.0e-12
            end
        end

        @testset "One axis periodic, one ZeroCurvBC" begin
            Nx = 32
            dx = 2π / Nx
            x_excl = range(0.0, step = dx, length = Nx)
            y = range(0.0, 1.0, 10)
            data_excl = [sin(xi) * yj for xi in x_excl, yj in y]

            x_incl = range(0.0, step = dx, length = Nx + 1)
            data_incl = [sin(xi) * yj for xi in x_incl, yj in y]
            # Enforce exact periodicity on dim 1
            data_incl[end, :] .= data_incl[1, :]

            itp_excl = cubic_interp(
                (x_excl, y), data_excl;
                bc = (PeriodicBC(endpoint = :exclusive), ZeroCurvBC())
            )
            itp_incl = cubic_interp(
                (x_incl, y), data_incl;
                bc = (PeriodicBC(), ZeroCurvBC())
            )

            for (xq, yq) in [(0.5, 0.3), (π, 0.5), (5.0, 0.8)]
                @test itp_excl((xq, yq)) ≈ itp_incl((xq, yq)) atol = 1.0e-12
            end
        end

        @testset "Explicit period (Vector grid)" begin
            x_incl = [0.0, 0.5, 1.5, 3.0, 5.0, 2π]
            y = range(0.0, 1.0, 6)
            data_incl = [sin(xi) * yj for xi in x_incl, yj in y]
            # Enforce exact periodicity on dim 1
            data_incl[end, :] .= data_incl[1, :]

            x_excl = x_incl[1:(end - 1)]
            data_excl = data_incl[1:(end - 1), :]

            itp_incl = cubic_interp(
                (x_incl, y), data_incl;
                bc = (PeriodicBC(), ZeroCurvBC())
            )
            itp_excl = cubic_interp(
                (x_excl, y), data_excl;
                bc = (PeriodicBC(endpoint = :exclusive, period = 2π), ZeroCurvBC())
            )

            for (xq, yq) in [(0.5, 0.3), (π, 0.5), (5.0, 0.8)]
                @test itp_excl((xq, yq)) ≈ itp_incl((xq, yq)) atol = 1.0e-12
            end
        end
    end

    # ========================================
    # Accuracy against analytic function
    # ========================================
    @testset "2D Accuracy" begin
        Nx, Ny = 64, 48
        dx, dy = 2π / Nx, 2π / Ny
        x = range(0.0, step = dx, length = Nx)
        y = range(0.0, step = dy, length = Ny)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp(
            (x, y), data;
            bc = PeriodicBC(endpoint = :exclusive)
        )

        # Value accuracy
        for (xq, yq) in [(0.3, 0.7), (π / 2, π / 3), (4.0, 3.0)]
            @test itp((xq, yq)) ≈ sin(xq) * cos(yq) atol = 1.0e-4
        end
    end

    # ========================================
    # Derivatives
    # ========================================
    @testset "2D Derivatives" begin
        Nx, Ny = 64, 48
        dx, dy = 2π / Nx, 2π / Ny
        x = range(0.0, step = dx, length = Nx)
        y = range(0.0, step = dy, length = Ny)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp(
            (x, y), data;
            bc = PeriodicBC(endpoint = :exclusive)
        )

        xq, yq = π / 4, π / 3

        # ∂/∂x [sin(x)cos(y)] = cos(x)cos(y)
        @test itp((xq, yq); deriv = DerivOp(1, 0)) ≈ cos(xq) * cos(yq) atol = 1.0e-3

        # ∂/∂y [sin(x)cos(y)] = -sin(x)sin(y)
        @test itp((xq, yq); deriv = DerivOp(0, 1)) ≈ -sin(xq) * sin(yq) atol = 1.0e-3
    end

    # ========================================
    # 3D Test
    # ========================================
    @testset "3D Exclusive endpoint" begin
        Nx, Ny, Nz = 16, 12, 10
        x = range(0.0, step = 2π / Nx, length = Nx)
        y = range(0.0, step = 2π / Ny, length = Ny)
        z = range(0.0, 1.0, Nz)
        data = [sin(xi) * cos(yj) * zk for xi in x, yj in y, zk in z]

        itp = cubic_interp(
            (x, y, z), data;
            bc = (
                PeriodicBC(endpoint = :exclusive),
                PeriodicBC(endpoint = :exclusive),
                ZeroCurvBC(),
            )
        )

        @test itp isa CubicInterpolantND
        xq, yq, zq = 1.0, 0.5, 0.3
        @test itp((xq, yq, zq)) ≈ sin(xq) * cos(yq) * zq atol = 1.0e-2
    end

    # ========================================
    # bcs_store preserves endpoint info
    # ========================================
    @testset "bcs_store reflects post-extension `:inclusive` form" begin
        Nx = 16
        x = range(0.0, step = 2π / Nx, length = Nx)
        y = range(0.0, 1.0, 8)
        data = [sin(xi) * yj for xi in x, yj in y]

        itp = cubic_interp(
            (x, y), data;
            bc = (PeriodicBC(endpoint = :exclusive), ZeroCurvBC())
        )

        @test itp.bcs[1] isa PeriodicBC{:inclusive}                   # normalized
        @test itp.bcs[1].period ≈ 2π                                  # materialized from grid span
        @test itp.bcs[2] isa BCPair                                    # ZeroCurvBC → BCPair{Deriv2, Deriv2}
    end

    # ========================================
    # Error Cases
    # ========================================
    @testset "ND Error cases" begin
        @testset "Vector grid without period → error" begin
            x = [0.0, 1.0, 2.0, 3.0]
            y = range(0.0, 1.0, 5)
            data = zeros(4, 5)
            @test_throws ArgumentError cubic_interp(
                (x, y), data;
                bc = (PeriodicBC(endpoint = :exclusive), ZeroCurvBC())
            )
        end

        @testset "Range grid with conflicting period → error" begin
            x = range(0.0, step = 0.1, length = 10)
            y = range(0.0, 1.0, 5)
            data = zeros(10, 5)
            @test_throws ArgumentError cubic_interp(
                (x, y), data;
                bc = (PeriodicBC(endpoint = :exclusive, period = 2.0), ZeroCurvBC())
            )
        end
    end

end
