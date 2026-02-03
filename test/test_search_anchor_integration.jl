using Test
using FastInterpolations
using FastInterpolations: _anchor_query, _fill_anchors!,
    _LinearAnchoredQuery, _ConstantAnchoredQuery, _QuadraticAnchoredQuery, _CubicAnchoredQuery,
    Searcher, HintedBinary, LinearBinary, RefHint

@testset "Search Policy Anchor Integration" begin

    # ========================================
    # Linear Anchor Tests
    # ========================================

    @testset "Linear Anchor with Policy" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Single Query - Default Policy" begin
            aq = _anchor_query(x, 0.5, Val(:linear))
            @test aq isa _LinearAnchoredQuery{Float64,Float64}
            @test aq.idx == 51
        end

        @testset "Vector Query - Policy Used in _fill_anchors!" begin
            xq = collect(range(0.1, 0.9, 9))
            buffer = Vector{_LinearAnchoredQuery{Float64,Float64}}(undef, length(xq))

            # _fill_anchors! should use hinted search internally
            _fill_anchors!(buffer, x, xq, Val(:linear))

            # Verify correct anchors
            for i in eachindex(xq)
                @test buffer[i].idx == round(Int, xq[i] * 100) + 1
            end
        end

        @testset "Monotonic Query Benefit" begin
            # Monotonic queries should benefit from hint caching
            xq_monotonic = collect(range(0.01, 0.99, 99))
            buffer = Vector{_LinearAnchoredQuery{Float64,Float64}}(undef, length(xq_monotonic))

            # This should be faster due to linear bounded search
            _fill_anchors!(buffer, x, xq_monotonic, Val(:linear))

            # Verify all anchors correct
            for i in eachindex(xq_monotonic)
                expected_idx = round(Int, xq_monotonic[i] * 100) + 1
                @test buffer[i].idx == expected_idx || buffer[i].idx == expected_idx - 1
            end
        end
    end

    # ========================================
    # Constant Anchor Tests
    # ========================================

    @testset "Constant Anchor with Policy" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Single Query" begin
            aq = _anchor_query(x, 0.5, Val(:constant))
            @test aq isa _ConstantAnchoredQuery{Float64}
            @test aq.idx == 51
        end

        @testset "Vector Query" begin
            xq = collect(range(0.1, 0.9, 9))
            buffer = Vector{_ConstantAnchoredQuery{Float64}}(undef, length(xq))

            _fill_anchors!(buffer, x, xq, Val(:constant))

            for i in eachindex(xq)
                @test buffer[i] isa _ConstantAnchoredQuery{Float64}
            end
        end
    end

    # ========================================
    # Quadratic Anchor Tests
    # ========================================

    @testset "Quadratic Anchor with Policy" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Single Query" begin
            aq = _anchor_query(x, 0.5, Val(:quadratic))
            @test aq isa _QuadraticAnchoredQuery{Float64}
            @test aq.idx == 51
        end

        @testset "Vector Query" begin
            xq = collect(range(0.1, 0.9, 9))
            buffer = Vector{_QuadraticAnchoredQuery{Float64,Float64}}(undef, length(xq))

            _fill_anchors!(buffer, x, xq, Val(:quadratic))

            for i in eachindex(xq)
                @test buffer[i] isa _QuadraticAnchoredQuery{Float64,Float64}
            end
        end
    end

    # ========================================
    # Cubic Anchor Tests
    # ========================================

    @testset "Cubic Anchor with Policy" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Single Query" begin
            aq = _anchor_query(x, 0.5, Val(:cubic))
            @test aq isa _CubicAnchoredQuery{Float64}
            @test aq.idx == 51
        end

        @testset "Vector Query" begin
            xq = collect(range(0.1, 0.9, 9))
            buffer = Vector{_CubicAnchoredQuery{Float64,Float64}}(undef, length(xq))

            _fill_anchors!(buffer, x, xq, Val(:cubic))

            for i in eachindex(xq)
                @test buffer[i] isa _CubicAnchoredQuery{Float64,Float64}
            end
        end
    end

    # ========================================
    # Backward Compatibility
    # ========================================

    @testset "Backward Compatibility" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Existing API Still Works" begin
            # Single query (no policy exposed)
            aq_linear = _anchor_query(x, 0.5, Val(:linear))
            @test aq_linear.idx == 51

            aq_cubic = _anchor_query(x, 0.5, Val(:cubic))
            @test aq_cubic.idx == 51

            # Vector query
            xq = [0.1, 0.5, 0.9]
            aq_vec = _anchor_query(x, xq, Val(:linear))
            @test length(aq_vec) == 3
        end
    end

    # ========================================
    # Type Stability Tests
    # ========================================

    @testset "Type Stability" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Linear Type Stable" begin
            @test @inferred(_anchor_query(x, 0.5, Val(:linear))) isa _LinearAnchoredQuery{Float64,Float64}
        end

        @testset "Constant Type Stable" begin
            @test @inferred(_anchor_query(x, 0.5, Val(:constant))) isa _ConstantAnchoredQuery{Float64}
        end

        @testset "Quadratic Type Stable" begin
            @test @inferred(_anchor_query(x, 0.5, Val(:quadratic))) isa _QuadraticAnchoredQuery{Float64,Float64}
        end

        @testset "Cubic Type Stable" begin
            @test @inferred(_anchor_query(x, 0.5, Val(:cubic))) isa _CubicAnchoredQuery{Float64,Float64}
        end
    end

end  # @testset "Search Policy Anchor Integration"
