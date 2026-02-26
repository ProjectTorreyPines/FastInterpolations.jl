using Test
using FastInterpolations
using FastInterpolations: _resolve_search_nd, _resolve_search,
    AutoSearch, Binary, LinearBinary

@testset "ND Per-Axis Adaptive AutoSearch Resolution" begin

    # ========================================
    # 4-arg _resolve_search_nd: per-axis monotonicity check
    # ========================================

    @testset "Scalar AutoSearch + SoA + no hint → per-axis check" begin
        sorted = collect(1.0:100.0)
        random = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
                  7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]

        @testset "both sorted → both LinearBinary" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), (sorted, sorted), nothing)
            @test result[1] isa LinearBinary
            @test result[2] isa LinearBinary
        end

        @testset "both random → both Binary" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), (random, random), nothing)
            @test result[1] isa Binary
            @test result[2] isa Binary
        end

        @testset "axis 1 sorted, axis 2 random → mixed policies" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), (sorted, random), nothing)
            @test result[1] isa LinearBinary
            @test result[2] isa Binary
        end

        @testset "axis 1 random, axis 2 sorted → mixed policies (reversed)" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), (random, sorted), nothing)
            @test result[1] isa Binary
            @test result[2] isa LinearBinary
        end

        @testset "3D: mixed pattern" begin
            descending = collect(100.0:-1.0:1.0)
            result = _resolve_search_nd(AutoSearch(), Val(3), (sorted, random, descending), nothing)
            @test result[1] isa LinearBinary
            @test result[2] isa Binary
            @test result[3] isa LinearBinary
        end
    end

    @testset "Tuple of AutoSearch + SoA + no hint → per-axis check" begin
        sorted = collect(1.0:100.0)
        random = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
                  7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]

        @testset "(AutoSearch, AutoSearch) per-axis independent" begin
            result = _resolve_search_nd((AutoSearch(), AutoSearch()), Val(2),
                                        (sorted, random), nothing)
            @test result[1] isa LinearBinary
            @test result[2] isa Binary
        end
    end

    @testset "Mixed tuple (AutoSearch + explicit) + SoA + no hint" begin
        sorted = collect(1.0:100.0)
        random = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
                  7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]

        @testset "(AutoSearch, Binary) → axis 1 checked, axis 2 passthrough" begin
            result = _resolve_search_nd((AutoSearch(), Binary()), Val(2),
                                        (sorted, sorted), nothing)
            @test result[1] isa LinearBinary   # AutoSearch resolved via monotonicity
            @test result[2] isa Binary         # explicit Binary unchanged
        end

        @testset "(Binary, AutoSearch) → axis 1 passthrough, axis 2 checked" begin
            result = _resolve_search_nd((Binary(), AutoSearch()), Val(2),
                                        (random, sorted), nothing)
            @test result[1] isa Binary         # explicit Binary unchanged
            @test result[2] isa LinearBinary   # AutoSearch resolved via monotonicity
        end
    end

    @testset "hint present → fallback (no monotonicity check)" begin
        sorted = collect(1.0:100.0)
        hints = (Ref(1), Ref(1))

        # With hint, falls through to 3-arg → type-based → LinearBinary for vectors
        result = _resolve_search_nd(AutoSearch(), Val(2), (sorted, sorted), hints)
        @test result[1] isa LinearBinary
        @test result[2] isa LinearBinary
    end

    # ========================================
    # Oneshot SoA: verify adaptive resolution through public API
    # ========================================
    # These tests verify correctness (values) to ensure the search policy
    # changes don't break results. Both Binary and LinearBinary produce
    # identical results — this confirms the plumbing works end-to-end.

    @testset "Oneshot SoA correctness with sorted queries" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        sorted_xq = collect(0.5:0.5:9.5)
        sorted_yq = collect(0.5:0.5:9.5)
        n = length(sorted_xq)
        output = Vector{Float64}(undef, n)

        linear_interp!(output, (xs, ys), data, (sorted_xq, sorted_yq))
        for i in 1:n
            @test output[i] ≈ sorted_xq[i] + sorted_yq[i] atol=1e-12
        end
    end

    @testset "Oneshot SoA correctness with random queries" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        # Random (non-monotone) queries — AutoSearch should pick Binary
        random_xq = [3.2, 1.5, 7.8, 0.3, 9.1, 4.6, 6.0, 2.4, 8.7, 5.5]
        random_yq = [8.1, 2.3, 5.7, 9.0, 0.5, 6.8, 3.4, 7.2, 1.1, 4.9]
        n = length(random_xq)
        output = Vector{Float64}(undef, n)

        linear_interp!(output, (xs, ys), data, (random_xq, random_yq))
        for i in 1:n
            @test output[i] ≈ random_xq[i] + random_yq[i] atol=1e-12
        end
    end

    @testset "Oneshot SoA correctness with mixed sorted/random per axis" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        sorted_xq = collect(0.5:0.5:5.0)
        random_yq = [8.1, 2.3, 5.7, 9.0, 0.5, 6.8, 3.4, 7.2, 1.1, 4.9]
        n = length(sorted_xq)
        output = Vector{Float64}(undef, n)

        linear_interp!(output, (xs, ys), data, (sorted_xq, random_yq))
        for i in 1:n
            @test output[i] ≈ sorted_xq[i] + random_yq[i] atol=1e-12
        end
    end
end
