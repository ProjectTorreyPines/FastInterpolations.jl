# ═══════════════════════════════════════════════════════════════════════════════
# test_precision_vector_queries.jl
#
# Phase 1 TDD Tests: Verify scalar/vector query path consistency
#
# These tests verify that vector query paths (`itp(xq_vec)`) produce
# numerically identical results to scalar broadcast (`itp.(xq_vec)`).
#
# The bug: vector paths use `_to_float(xq, Tg)` which downcasts Float64 queries
# to Float32 when the grid is Float32, losing precision in arithmetic.
# ═══════════════════════════════════════════════════════════════════════════════

@testitem "Precision: Vector Query Paths (Scalar/Vector Symmetry)" setup = [AllocConstants] begin
    # Helper: extract diagonal from a matrix (avoid LinearAlgebra dependency)
    __diag(M::AbstractMatrix) = [M[i, i] for i in 1:min(size(M)...)]

    # ═══════════════════════════════════════════════════════════════════════════════
    # Test Configuration
    # ═══════════════════════════════════════════════════════════════════════════════

    # Tolerance for numerical comparison
    # The bug causes errors on the order of Float32 epsilon (~1e-7) relative to Float64
    const PRECISION_RTOL = 1.0e-10  # Tight tolerance - should pass if no downcast


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Callable Interpolant - Scalar vs Vector Consistency
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # For each interpolant type (constant, linear, quadratic, cubic):
    # - Create with Float32 grid and Float32 values
    # - Query with Float64 vector
    # - Compare: itp(xq_vec) vs itp.(xq_vec)
    #
    # Expected: 🔴 FAIL (vector path downcasts Float64→Float32)

    @testset "Constant Interpolant: vector vs broadcast" begin
        # Float32 grid and values
        x = Float32.(collect(range(0.0, 4.0, 5)))  # [0, 1, 2, 3, 4]
        y = Float32[10.0, 20.0, 30.0, 40.0, 50.0]

        itp = constant_interp(x, y)

        # Float64 query points (high precision)
        xq = [0.5, 1.5, 2.5, 3.5]  # Float64 by default

        # Vector call vs broadcast (scalar per-element)
        result_vec = itp(xq)           # Uses vector path → downcasts xq to Float32
        result_broadcast = itp.(xq)    # Uses scalar path

        @testset "Allocating vector call" begin
            @test result_vec ≈ result_broadcast rtol = PRECISION_RTOL
            # Zero-slope arithmetic propagates the Float64 xq carrier:
            # Float32 y * one(Float64) = Float64. Scalar / broadcast / batch agree.
            @test eltype(result_vec) == Float64
        end

        @testset "In-place vector call" begin
            output = Vector{Float64}(undef, length(xq))
            itp(output, xq)
            @test output ≈ result_broadcast rtol = PRECISION_RTOL
        end
    end

    @testset "Linear Interpolant: vector vs broadcast" begin
        # Float32 grid and values (simple function: y = 2x + 1)
        x = Float32.(collect(range(0.0, 4.0, 5)))
        y = Float32.(2.0 .* x .+ 1.0)

        itp = linear_interp(x, y)

        # Float64 query points at non-grid locations
        # These values are chosen to maximize the difference between Float32/Float64 arithmetic
        xq = [0.123456789, 1.23456789, 2.345678901, 3.456789012]

        result_vec = itp(xq)
        result_broadcast = itp.(xq)

        @testset "Allocating vector call" begin
            # Compare actual values
            @test result_vec ≈ result_broadcast rtol = PRECISION_RTOL

            # Also verify against analytical solution (y = 2x + 1)
            expected = 2.0 .* xq .+ 1.0
            @test result_broadcast ≈ expected rtol = PRECISION_RTOL
        end

        @testset "In-place vector call" begin
            output = Vector{Float64}(undef, length(xq))
            itp(output, xq)
            @test output ≈ result_broadcast rtol = PRECISION_RTOL
        end
    end

    @testset "Quadratic Interpolant: vector vs broadcast" begin
        # Float32 grid and quadratic function: y = x^2
        x = Float32.(collect(range(0.0, 4.0, 9)))
        y = Float32.(x .^ 2)

        itp = quadratic_interp(x, y)

        # Float64 query points
        xq = [0.123456789, 1.23456789, 2.345678901, 3.456789012]

        result_vec = itp(xq)
        result_broadcast = itp.(xq)

        @testset "Allocating vector call" begin
            @test result_vec ≈ result_broadcast rtol = PRECISION_RTOL
        end

        @testset "In-place vector call" begin
            output = Vector{Float64}(undef, length(xq))
            itp(output, xq)
            @test output ≈ result_broadcast rtol = PRECISION_RTOL
        end
    end

    @testset "Cubic Interpolant: vector vs broadcast" begin
        # Float32 grid and smooth function
        x = Float32.(collect(range(0.0, 2π, 51)))
        y = Float32.(sin.(x))

        itp = cubic_interp(x, y; autocache = false)

        # Float64 query points
        xq = [0.5, 1.0, π / 2, 2.0, π, 4.0, 5.0, 2π - 0.1]

        result_vec = itp(xq)
        result_broadcast = itp.(xq)

        @testset "Allocating vector call" begin
            @test result_vec ≈ result_broadcast rtol = PRECISION_RTOL
        end

        @testset "In-place vector call" begin
            output = Vector{Float64}(undef, length(xq))
            itp(output, xq)
            @test output ≈ result_broadcast rtol = PRECISION_RTOL
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Series Interpolants - Scalar vs Vector Consistency
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Series interpolants: sitp(xq_vec) should match per-point scalar evaluation
    # Expected: 🔴 FAIL (vector path uses anchors built from Float32-converted xq)

    @testset "Linear Series: vector vs per-point scalar" begin
        # Float32 grid, multiple series
        x = Float32.(collect(range(0.0, 4.0, 9)))
        y1 = Float32.(2.0 .* x .+ 1.0)  # y = 2x + 1
        y2 = Float32.(x .^ 2)            # y = x^2

        sitp = linear_interp(x, Series(y1, y2))

        # Float64 query points
        xq = [0.123456789, 1.23456789, 2.345678901, 3.456789012]

        # Vector call
        result_vec = sitp(xq)  # Returns Vector of Vectors

        # Per-point scalar call (reference)
        result_scalar = [sitp(xi) for xi in xq]

        @testset "Per-series comparison" begin
            for k in 1:2
                vec_series = [result_vec[k][i] for i in eachindex(xq)]
                scalar_series = [result_scalar[i][k] for i in eachindex(xq)]
                @test vec_series ≈ scalar_series rtol = PRECISION_RTOL
            end
        end
    end

    @testset "Constant Series: vector vs per-point scalar" begin
        x = Float32.(collect(range(0.0, 4.0, 5)))
        y1 = Float32[10.0, 20.0, 30.0, 40.0, 50.0]
        y2 = Float32[1.0, 2.0, 3.0, 4.0, 5.0]

        sitp = constant_interp(x, Series(y1, y2))

        xq = [0.5, 1.5, 2.5, 3.5]

        result_vec = sitp(xq)
        result_scalar = [sitp(xi) for xi in xq]

        @testset "Per-series comparison" begin
            for k in 1:2
                vec_series = [result_vec[k][i] for i in eachindex(xq)]
                scalar_series = [result_scalar[i][k] for i in eachindex(xq)]
                @test vec_series ≈ scalar_series rtol = PRECISION_RTOL
            end
        end
    end

    @testset "Quadratic Series: vector vs per-point scalar" begin
        x = Float32.(collect(range(0.0, 4.0, 17)))
        y1 = Float32.(x .^ 2)
        y2 = Float32.(sin.(x))

        sitp = quadratic_interp(x, Series(y1, y2))

        xq = [0.123456789, 1.23456789, 2.345678901, 3.456789012]

        result_vec = sitp(xq)
        result_scalar = [sitp(xi) for xi in xq]

        @testset "Per-series comparison" begin
            for k in 1:2
                vec_series = [result_vec[k][i] for i in eachindex(xq)]
                scalar_series = [result_scalar[i][k] for i in eachindex(xq)]
                @test vec_series ≈ scalar_series rtol = PRECISION_RTOL
            end
        end
    end

    @testset "Cubic Series: vector vs per-point scalar" begin
        # Known limitation: CubicSeriesInterpolant vector path requires out::Vector{Tv}
        # and data::Matrix{Tv} to share the same Tv. When the grid is Float32 and
        # queries are Float64, the output buffer is Vector{Float64} but data is
        # Matrix{Float32}, causing a MethodError in _eval_series_vector!.
        # Fix requires decoupling output type from data type in _eval_series_vector!.
        @test_broken begin
            x = Float32.(collect(range(0.0, 2π, 51)))
            y1 = Float32.(sin.(x))
            y2 = Float32.(cos.(x))
            sitp = cubic_interp(x, Series(y1, y2))
            xq = [0.5, 1.0, π / 2, 2.0, π, 4.0, 5.0]
            result_vec = sitp(xq)
            result_scalar = [sitp(xi) for xi in xq]
            all(
                k -> all(
                    [result_vec[k][i] for i in eachindex(xq)] .≈
                        [result_scalar[i][k] for i in eachindex(xq)]; rtol = PRECISION_RTOL
                ), 1:2
            )
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Edge Cases
    # ═══════════════════════════════════════════════════════════════════════════

    @testset "AbstractRange queries" begin
        # Range queries should also preserve precision
        x = Float32.(collect(range(0.0, 4.0, 9)))
        y = Float32.(2.0 .* x .+ 1.0)

        itp = linear_interp(x, y)

        # Float64 Range query
        xq_range = range(0.5, 3.5, 7)  # StepRangeLen{Float64}

        result_vec = itp(xq_range)
        result_broadcast = itp.(xq_range)

        @test result_vec ≈ result_broadcast rtol = PRECISION_RTOL
    end

    @testset "Extrapolation mode :none with domain check" begin
        # Domain check should use Tg-typed primal, not fail on type mismatch
        x = Float32.(collect(range(0.0, 1.0, 11)))
        y = Float32.(x .^ 2)

        itp = linear_interp(x, y; extrap = NoExtrap())

        # Float64 query in domain
        xq_in = [0.1, 0.5, 0.9]
        @test_nowarn itp(xq_in)

        # Float64 query out of domain
        xq_out = [0.1, 0.5, 1.1]
        @test_throws DomainError itp(xq_out)
    end

    @testset "Float64 grid with Float32 query (reverse scenario)" begin
        # Less common but should also work correctly
        x = collect(range(0.0, 4.0, 9))  # Float64 grid
        y = 2.0 .* x .+ 1.0              # Float64 values

        itp = linear_interp(x, y)

        # Float32 query points
        xq = Float32[0.5, 1.5, 2.5, 3.5]

        result_vec = itp(xq)
        result_broadcast = itp.(xq)

        # In this case, Float32 queries get promoted to Float64 in both paths
        # So they should match
        @test result_vec ≈ result_broadcast rtol = PRECISION_RTOL
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 4: ForwardDiff Dual Vector Safety
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # When ForwardDiff is available, ensure vector-of-Dual usage is either:
    # - Correct: preserves Dual in arithmetic and returns Dual outputs
    # - Safe failure: throws a clear error suggesting scalar/broadcast usage
    #
    # Expected: Should NOT silently strip derivatives via float conversion

    @testset "Dual-vector safety (ForwardDiff extension)" begin
        # Check if ForwardDiff is available by trying to load it
        local ForwardDiff_loaded = false
        try
            @eval Main using ForwardDiff
            ForwardDiff_loaded = true
        catch
            ForwardDiff_loaded = false
        end

        if ForwardDiff_loaded
            @testset "Linear interpolant with Dual vector" begin
                x = Float32.(collect(range(0.0, 4.0, 9)))
                y = Float32.(2.0 .* x .+ 1.0)

                itp = linear_interp(x, y)

                # Create Dual vector for testing
                xq_float = [0.5, 1.5, 2.5, 3.5]
                f(x_vec) = itp(x_vec)
                f_broadcast(x_vec) = itp.(x_vec)

                # The vector path should either:
                # 1. Work correctly (return Dual array with correct derivatives)
                # 2. Throw an informative error

                # Scalar broadcast should definitely work
                J_broadcast = Main.ForwardDiff.jacobian(f_broadcast, xq_float)
                @test size(J_broadcast) == (4, 4)  # Diagonal Jacobian

                # The derivative of linear interpolation is constant within each interval
                # For y = 2x + 1, dy/dx = 2 everywhere
                diag_elements = [J_broadcast[i, i] for i in 1:4]
                @test all(diag_elements .≈ 2.0)

                # Vector path: check behavior (may error or work)
                # We mainly want to ensure it doesn't silently return Float with zero derivatives
                try
                    J_vec = Main.ForwardDiff.jacobian(f, xq_float)
                    # If it works, derivatives should match
                    @test J_vec ≈ J_broadcast rtol = PRECISION_RTOL
                catch e
                    # If it throws, should be an informative error
                    @test e isa Union{MethodError, ArgumentError}
                    @test_skip "Vector path with Dual not supported (expected)"
                end
            end
        else
            @test_skip "ForwardDiff not available"
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 5: Performance Baseline (Typed Hot Paths)
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Verify that typed hot paths (eltype(xq) === Tg) remain zero-allocation.
    # This establishes the baseline for later phases to compare against.

    @testset "Typed hot path: itp(xq::Vector{Tg})" begin
        function measure_linear_alloc()
            # Float64 grid with Float64 query (common case)
            x = collect(range(0.0, 4.0, 51))
            y = sin.(2π .* x)

            itp = linear_interp(x, y)

            xq = [0.5, 1.5, 2.5, 3.5]  # Float64

            # Warmup
            itp(xq)
            itp(xq)

            return @allocated itp(xq)
        end
        # Vector call allocates output array (~32 bytes for 4 Float64 + header)
        expected_output = sizeof(Float64) * 4 + 40
        @test measure_linear_alloc() <= expected_output * 2 + ALLOC_THRESHOLD
    end

    @testset "Typed hot path: itp(output, xq::Vector{Tg}) in-place" begin
        function measure_linear_inplace_alloc()
            x = collect(range(0.0, 4.0, 51))
            y = sin.(2π .* x)

            itp = linear_interp(x, y)

            xq = [0.5, 1.5, 2.5, 3.5]
            output = similar(xq)

            # Warmup
            itp(output, xq)
            itp(output, xq)

            return @allocated itp(output, xq)
        end
        @test measure_linear_inplace_alloc() <= ALLOC_THRESHOLD
    end

    @testset "Typed hot path: sitp(outputs, xq::Vector{Tg}) series in-place" begin
        function measure_series_inplace_alloc()
            x = collect(range(0.0, 4.0, 51))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            sitp = linear_interp(x, Series(y1, y2))
            precompute_transpose!(sitp)

            xq = collect(range(0.25, 3.75, 10))
            outputs = [Vector{Float64}(undef, length(xq)) for _ in 1:2]

            # Warmup
            sitp(outputs, xq)
            sitp(outputs, xq)

            return @allocated sitp(outputs, xq)
        end
        @test measure_series_inplace_alloc() <= ALLOC_THRESHOLD
    end

    @testset "Typed hot path: cubic itp(xq::Vector{Tg})" begin
        function measure_cubic_alloc()
            x = collect(range(0.0, 2π, 51))
            y = sin.(x)

            itp = cubic_interp(x, y; autocache = false)

            xq = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

            # Warmup
            itp(xq)
            itp(xq)

            return @allocated itp(xq)
        end
        expected_output = sizeof(Float64) * 6 + 40
        @test measure_cubic_alloc() <= expected_output * 2 + ALLOC_THRESHOLD
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 6: Precision Loss Demonstration
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Explicit tests demonstrating the precision loss magnitude.
    # These help quantify the bug and verify the fix.

    @testset "Precision loss magnitude (diagnostic)" begin
        # This test explicitly shows the precision loss
        x = Float32.(collect(range(0.0, 1.0, 11)))
        y = Float32.(x .^ 2)

        itp = linear_interp(x, y)

        # Query point with many significant digits
        xq_precise = 0.123456789012345  # ~15 significant digits

        # Scalar call (correct path)
        result_scalar = itp(xq_precise)

        # Vector call (buggy path)
        result_vector = itp([xq_precise])[1]

        # The difference should be on the order of Float32 epsilon when downcast occurs
        diff = abs(result_scalar - result_vector)

        # For now, just record the difference (this test is informational)
        @info "Precision loss test" scalar = result_scalar vector = result_vector diff = diff

        # After fix, this should pass (diff should be essentially zero)
        @test result_scalar ≈ result_vector rtol = PRECISION_RTOL
    end

end
