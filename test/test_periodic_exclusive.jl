using Test
using FastInterpolations
using FastInterpolations: _prepare_periodic, _resolve_exclusive_period, _extend_exclusive,
    _can_infer_period, _is_periodic_bc, endpoint

@testset "PeriodicBC Exclusive Endpoint" begin

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
            pbc = PeriodicBC(endpoint=:exclusive, period=2π)
            @test pbc isa PeriodicBC{:exclusive, Float64}
            @test endpoint(pbc) === :exclusive
            @test pbc.period ≈ 2π
            @test _is_periodic_bc(pbc)

            # Float32 promotion
            pbc32 = PeriodicBC(endpoint=:exclusive, period=1f0)
            @test pbc32 isa PeriodicBC{:exclusive, Float32}
            @test pbc32.period === 1f0

            # Integer promotion to Float64
            pbc_int = PeriodicBC(endpoint=:exclusive, period=6)
            @test pbc_int isa PeriodicBC{:exclusive, Float64}
            @test pbc_int.period === 6.0
        end

        @testset "Exclusive without period (infer from Range)" begin
            pbc = PeriodicBC(endpoint=:exclusive)
            @test pbc isa PeriodicBC{:exclusive, Nothing}
            @test endpoint(pbc) === :exclusive
            @test pbc.period === nothing
        end

        @testset "Construction errors" begin
            # Invalid endpoint symbol
            @test_throws ArgumentError PeriodicBC(endpoint=:bad)
            @test_throws ArgumentError PeriodicBC(endpoint=:open)

            # Inclusive with period → error
            @test_throws ArgumentError PeriodicBC(endpoint=:inclusive, period=1.0)
            @test_throws ArgumentError PeriodicBC(period=1.0)  # default is :inclusive

            # Exclusive with non-positive period
            @test_throws ArgumentError PeriodicBC(endpoint=:exclusive, period=0.0)
            @test_throws ArgumentError PeriodicBC(endpoint=:exclusive, period=-1.0)
        end
    end

    # ========================================
    # Period Resolution Tests
    # ========================================
    @testset "Period Resolution" begin
        @testset "Range grid auto-inference" begin
            x = range(0.0, step=0.1, length=10)
            bc = PeriodicBC(endpoint=:exclusive)
            period = _resolve_exclusive_period(x, bc)
            @test period ≈ 1.0  # step(x) * length(x) = 0.1 * 10
        end

        @testset "Range grid with matching explicit period" begin
            x = range(0.0, step=0.1, length=10)
            bc = PeriodicBC(endpoint=:exclusive, period=1.0)
            period = _resolve_exclusive_period(x, bc)
            @test period ≈ 1.0
        end

        @testset "Range grid with conflicting period → error" begin
            x = range(0.0, step=0.1, length=10)
            bc = PeriodicBC(endpoint=:exclusive, period=2.0)  # doesn't match 0.1*10=1.0
            @test_throws ArgumentError _resolve_exclusive_period(x, bc)
        end

        @testset "Vector grid requires explicit period" begin
            x = [0.0, 0.3, 0.7, 1.5]
            bc = PeriodicBC(endpoint=:exclusive)
            @test_throws ArgumentError _resolve_exclusive_period(x, bc)

            # With period → OK
            bc2 = PeriodicBC(endpoint=:exclusive, period=2π)
            @test _resolve_exclusive_period(x, bc2) ≈ 2π
        end

        @testset "_can_infer_period" begin
            @test _can_infer_period(range(0, 1, 10)) == true
            @test _can_infer_period([0.0, 1.0, 2.0]) == false
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
            x = range(0.0, step=0.5, length=4)  # [0, 0.5, 1.0, 1.5]
            y = sin.(x)
            bc = PeriodicBC(endpoint=:exclusive)
            x_ext, y_ext = _prepare_periodic(x, y, bc)
            @test x_ext isa AbstractRange
            @test length(x_ext) == 5
            @test last(x_ext) ≈ 2.0  # step * length = 0.5 * 4
            @test y_ext[end] ≈ y[1]
        end

        @testset "Exclusive Vector" begin
            x = [0.0, 1.0, 3.0, 5.0]
            y = [1.0, 2.0, 3.0, 4.0]
            bc = PeriodicBC(endpoint=:exclusive, period=2π)
            x_ext, y_ext = _prepare_periodic(x, y, bc)
            @test length(x_ext) == 5
            @test last(x_ext) ≈ 2π
            @test y_ext[end] == y[1]
        end

        @testset "Period validation at build time" begin
            x = [0.0, 1.0, 3.0, 5.0]
            y = [1.0, 2.0, 3.0, 4.0]
            # period too small: virtual endpoint at 4.0 which is < x[end]=5.0
            bc = PeriodicBC(endpoint=:exclusive, period=4.0)
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
            x_excl = range(0.0, step=dx, length=N)
            y_excl = sin.(x_excl)

            x_incl = range(0.0, step=dx, length=N + 1)
            y_incl = sin.(x_incl)

            itp_excl = cubic_interp(x_excl, y_excl; bc=PeriodicBC(endpoint=:exclusive))
            itp_incl = cubic_interp(x_incl, y_incl; bc=PeriodicBC())

            # Values should match at multiple query points
            for xq in [0.1, 1.0, π, 2π - 0.01, 3.5]
                @test itp_excl(xq) ≈ itp_incl(xq) atol=1e-14
            end
        end

        @testset "Range grid with explicit period (redundant but valid)" begin
            N = 32
            dx = 2π / N
            x = range(0.0, step=dx, length=N)
            y = cos.(x)

            itp1 = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))
            itp2 = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive, period=2π))

            for xq in [0.5, 1.5, π]
                @test itp1(xq) ≈ itp2(xq) atol=1e-14
            end
        end

        @testset "Vector grid with period" begin
            x_incl = [0.0, 0.5, 1.5, 3.0, 5.0, 2π]
            y_incl = sin.(x_incl)

            x_excl = x_incl[1:end-1]  # remove last point
            y_excl = y_incl[1:end-1]

            itp_incl = cubic_interp(x_incl, y_incl; bc=PeriodicBC())
            itp_excl = cubic_interp(x_excl, y_excl; bc=PeriodicBC(endpoint=:exclusive, period=2π))

            for xq in [0.1, 1.0, π, 5.5]
                @test itp_excl(xq) ≈ itp_incl(xq) atol=1e-14
            end
        end
    end

    # ========================================
    # Functional Tests: 4-arg Form (Oneshot)
    # ========================================
    @testset "Oneshot API — Exclusive endpoint" begin
        N = 32
        dx = 2π / N
        x = range(0.0, step=dx, length=N)
        y = sin.(x)

        @testset "Scalar query" begin
            val = cubic_interp(x, y, 1.0; bc=PeriodicBC(endpoint=:exclusive))
            @test val ≈ sin(1.0) atol=1e-4
        end

        @testset "Vector query" begin
            xq = [0.5, 1.5, π]
            vals = cubic_interp(x, y, xq; bc=PeriodicBC(endpoint=:exclusive))
            @test length(vals) == 3
            for i in eachindex(xq)
                @test vals[i] ≈ sin(xq[i]) atol=1e-4
            end
        end

        @testset "In-place vector query" begin
            xq = [0.5, 1.5, π]
            output = zeros(3)
            cubic_interp!(output, x, y, xq; bc=PeriodicBC(endpoint=:exclusive))
            for i in eachindex(xq)
                @test output[i] ≈ sin(xq[i]) atol=1e-4
            end
        end
    end

    # ========================================
    # Series Interpolant Tests
    # ========================================
    @testset "CubicSeriesInterpolant — Exclusive endpoint" begin
        N = 32
        dx = 2π / N
        x = range(0.0, step=dx, length=N)
        y1 = sin.(x)
        y2 = cos.(x)

        mitp = cubic_interp(x, [y1, y2]; bc=PeriodicBC(endpoint=:exclusive))

        vals = mitp(1.0)
        @test vals[1] ≈ sin(1.0) atol=1e-4
        @test vals[2] ≈ cos(1.0) atol=1e-4
    end

    # ========================================
    # Derivative Tests
    # ========================================
    @testset "Derivatives — Exclusive endpoint" begin
        N = 64
        dx = 2π / N
        x = range(0.0, step=dx, length=N)
        y = sin.(x)

        itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))

        # C2 continuity at wrap point: derivatives should match across boundary
        ε = 1e-6
        d1_left = itp(ε; deriv=1)
        d1_right = itp(2π - ε; deriv=1)
        # sin'(0) ≈ sin'(2π) ≈ cos(0) = 1.0
        @test d1_left ≈ d1_right atol=1e-3

        # First derivative accuracy
        @test itp(π / 4; deriv=1) ≈ cos(π / 4) atol=1e-3

        # Second derivative
        @test itp(π / 2; deriv=2) ≈ -sin(π / 2) atol=0.1
    end

    # ========================================
    # Type Stability Tests
    # ========================================
    @testset "Type stability" begin
        N = 16
        dx = 2π / N

        # Inclusive data for baseline
        x_incl = range(0.0, step=dx, length=N + 1)
        y_incl = sin.(x_incl)
        @test @inferred(cubic_interp(x_incl, y_incl; bc=PeriodicBC())) isa CubicInterpolant

        # For exclusive, construction may have Union return (Range vs Vector grid branch),
        # but evaluation is always type-stable
        x_excl = range(0.0, step=dx, length=N)
        y_excl = sin.(x_excl)
        itp = cubic_interp(x_excl, y_excl; bc=PeriodicBC(endpoint=:exclusive))
        @test itp isa CubicInterpolant
        @test @inferred(itp(1.0)) isa Float64
    end

    # ========================================
    # Show / Display Tests
    # ========================================
    @testset "Show methods" begin
        @test FastInterpolations._format_bc(PeriodicBC()) == "Periodic"
        @test FastInterpolations._short_bc_name(PeriodicBC()) == "Periodic"

        bc_excl = PeriodicBC(endpoint=:exclusive, period=2π)
        @test occursin("exclusive", FastInterpolations._format_bc(bc_excl))
        @test FastInterpolations._short_bc_name(bc_excl) == "Periodic(excl)"

        # Series interpolant show (bc_for_solve is PeriodicData, not PeriodicBC)
        N = 16
        dx = 2π / N
        x_incl = range(0.0, step=dx, length=N + 1)
        y_incl = sin.(x_incl)
        sitp = cubic_interp(x_incl, [y_incl, cos.(x_incl)]; bc=PeriodicBC())
        buf = IOBuffer()
        show(buf, sitp)
        @test occursin("Periodic", String(take!(buf)))
        show(buf, MIME"text/plain"(), sitp)
        @test occursin("Periodic", String(take!(buf)))

        # itp.bc preserves original user-specified BC with resolved period
        @testset "itp.bc preserves endpoint and resolves period" begin
            N_bc = 16
            dx_bc = 2π / N_bc

            # Exclusive with explicit period
            x_bc = range(0.0, step=dx_bc, length=N_bc)
            y_bc = sin.(x_bc)
            itp_excl = cubic_interp(collect(x_bc), y_bc; bc=PeriodicBC(endpoint=:exclusive, period=2π))
            @test itp_excl.bc isa PeriodicBC{:exclusive}
            @test itp_excl.bc.period ≈ 2π

            # Exclusive without period (auto-inferred from Range) — period should still be resolved
            itp_excl_auto = cubic_interp(x_bc, y_bc; bc=PeriodicBC(endpoint=:exclusive))
            @test itp_excl_auto.bc isa PeriodicBC{:exclusive}
            @test itp_excl_auto.bc.period ≈ 2π

            # show output should reflect exclusive with period
            buf_bc = IOBuffer()
            show(buf_bc, MIME"text/plain"(), itp_excl_auto)
            s = String(take!(buf_bc))
            @test occursin("exclusive", s)
            @test occursin("period≈", s)

            # Inclusive — period should also be resolved from grid
            x_incl_bc = range(0.0, step=dx_bc, length=N_bc + 1)
            y_incl_bc = sin.(x_incl_bc)
            itp_incl = cubic_interp(collect(x_incl_bc), y_incl_bc; bc=PeriodicBC())
            @test itp_incl.bc isa PeriodicBC{:inclusive}
            @test itp_incl.bc.period ≈ 2π

            # show output should show period for inclusive too
            show(buf_bc, MIME"text/plain"(), itp_incl)
            s_incl = String(take!(buf_bc))
            @test occursin("period≈", s_incl)
        end

        # Also test exclusive series interpolant show
        x_excl = range(0.0, step=dx, length=N)
        y_excl = sin.(x_excl)
        sitp_excl = cubic_interp(x_excl, [y_excl, cos.(x_excl)]; bc=PeriodicBC(endpoint=:exclusive))
        show(buf, sitp_excl)
        @test occursin("Periodic", String(take!(buf)))
        show(buf, MIME"text/plain"(), sitp_excl)
        @test occursin("Periodic", String(take!(buf)))
    end

    # ========================================
    # Edge Cases
    # ========================================
    @testset "Edge cases" begin
        @testset "Vector grid without period → error" begin
            x = [0.0, 1.0, 2.0, 3.0]
            y = sin.(x)
            @test_throws ArgumentError cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))
        end

        @testset "Range grid + conflicting period → error" begin
            x = range(0.0, step=0.1, length=10)
            y = sin.(x)
            # 0.1 * 10 = 1.0, but user says period=2.0
            @test_throws ArgumentError cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive, period=2.0))
        end

        @testset "Minimum points (4 for periodic)" begin
            x = range(0.0, step=1.0, length=4)
            y = [0.0, 1.0, 0.0, -1.0]
            itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))
            @test itp(0.5) isa Float64
        end

        @testset "Float32 support" begin
            x = range(0f0, step=Float32(2π / 16), length=16)
            y = sin.(x)
            itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))
            @test itp(1f0) isa Float32
        end
    end

end
