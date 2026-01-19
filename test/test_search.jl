using Test
using FastInterpolations
using FastInterpolations: search_interval, _search_binary, _search_interval,
    SearchPolicy, BinaryAlg, HintedBinaryAlg, LinearBoundedAlg,
    NoHint, RefHint, DEFAULT_SEARCH_POLICY, ScalarSpacing, _create_spacing

@testset "Search Module" begin

    # ========================================
    # Type System Tests
    # ========================================

    @testset "Search Policy Types" begin
        @testset "Type Hierarchy" begin
            @test BinaryAlg <: FastInterpolations.AbstractSearchAlg
            @test HintedBinaryAlg <: FastInterpolations.AbstractSearchAlg
            @test LinearBoundedAlg{8} <: FastInterpolations.AbstractSearchAlg

            @test NoHint <: FastInterpolations.AbstractHint
            @test RefHint <: FastInterpolations.AbstractHint
        end

        @testset "SearchPolicy Construction" begin
            # Default policy
            @test DEFAULT_SEARCH_POLICY isa SearchPolicy{BinaryAlg,NoHint}

            # RefHint construction
            hint = RefHint()
            @test hint.idx[] == 1

            hint2 = RefHint(50)
            @test hint2.idx[] == 50

            # Full policy construction
            policy = SearchPolicy{HintedBinaryAlg,RefHint}(RefHint(Ref(10)))
            @test policy.hint.idx[] == 10
        end
    end

    # ========================================
    # Default Policy (Zero-Overhead) Tests
    # ========================================

    @testset "Default Policy Zero-Overhead" begin
        x_vec = collect(range(0.0, 1.0, 101))
        x_range = range(0.0, 1.0, 101)
        policy = DEFAULT_SEARCH_POLICY

        @testset "Vector Path" begin
            for xi in [0.0, 0.25, 0.5, 0.75, 1.0]
                result_policy = search_interval(policy, x_vec, xi)
                result_direct = _search_binary(x_vec, xi)
                @test result_policy == result_direct
            end
        end

        @testset "Range Path" begin
            for xi in [0.0, 0.25, 0.5, 0.75, 1.0]
                result_policy = search_interval(policy, x_range, xi)
                result_direct = _search_binary(x_range, xi)
                @test result_policy == result_direct
            end
        end

        @testset "Type Inference" begin
            # Must return concrete Tuple type
            @test @inferred(search_interval(policy, x_vec, 0.5)) isa Tuple{Int,Float64,Float64}
            @test @inferred(search_interval(policy, x_range, 0.5)) isa Tuple{Int,Float64,Float64}
        end
    end

    # ========================================
    # Internal Alias Tests
    # ========================================

    @testset "Internal Alias (_search_interval)" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Basic Functionality" begin
            idx, xL, xR = _search_interval(x, 0.5)
            @test idx == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end

        @testset "Equivalence to _search_binary" begin
            for xi in [0.0, 0.1, 0.5, 0.9, 1.0]
                @test _search_interval(x, xi) == _search_binary(x, xi)
            end
        end
    end

    # ========================================
    # Deprecated Alias Tests
    # ========================================

    @testset "Deprecated Alias (_find_interval)" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Backward Compatibility" begin
            idx, xL, xR = FastInterpolations._find_interval(x, 0.5)
            @test idx == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end

        @testset "Equivalence Chain" begin
            # _find_interval → _search_interval → _search_binary
            for xi in [0.0, 0.5, 1.0]
                r1 = FastInterpolations._find_interval(x, xi)
                r2 = _search_interval(x, xi)
                r3 = _search_binary(x, xi)
                @test r1 == r2 == r3
            end
        end
    end

    # ========================================
    # Hinted Binary Search Tests
    # ========================================

    @testset "Hinted Binary with RefHint" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Hint Update" begin
            hint = Ref(1)
            policy = SearchPolicy{HintedBinaryAlg,RefHint}(RefHint(hint))

            # Query far from hint - should update
            idx, _, _ = search_interval(policy, x, 0.75)
            @test idx == 76
            @test hint[] == 76
        end

        @testset "Hint Hit (O(1) path)" begin
            hint = Ref(50)
            policy = SearchPolicy{HintedBinaryAlg,RefHint}(RefHint(hint))

            # Query within hint interval - should hit O(1) path
            # Interval 50 covers [0.49, 0.50)
            idx, xL, xR = search_interval(policy, x, 0.495)
            @test idx == 50
            @test hint[] == 50  # Hint unchanged (was already correct)
        end

        @testset "Monotonic Query Sequence" begin
            hint = Ref(1)
            policy = SearchPolicy{HintedBinaryAlg,RefHint}(RefHint(hint))

            # Monotonically increasing queries
            for xi in range(0.1, 0.9, 10)
                idx, _, _ = search_interval(policy, x, xi)
                @test hint[] == idx
            end
        end

        @testset "Range Ignores Hint" begin
            x_range = range(0.0, 1.0, 101)
            hint = Ref(50)
            policy = SearchPolicy{HintedBinaryAlg,RefHint}(RefHint(hint))

            # Range path always uses O(1) calculation, hint not updated
            idx, _, _ = search_interval(policy, x_range, 0.1)
            @test idx == 11  # Direct O(1) calculation
            @test hint[] == 50  # Hint unchanged for ranges
        end
    end

    # ========================================
    # Linear Bounded Search Tests
    # ========================================

    @testset "LinearBoundedAlg with RefHint" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Linear Search Success" begin
            hint = Ref(50)
            policy = SearchPolicy{LinearBoundedAlg{8},RefHint}(RefHint(hint))

            # Query near hint (within 8 steps)
            idx, _, _ = search_interval(policy, x, 0.55)
            @test idx == 56
            @test hint[] == 56
        end

        @testset "Fallback to Binary" begin
            hint = Ref(10)
            policy = SearchPolicy{LinearBoundedAlg{4},RefHint}(RefHint(hint))

            # Query far from hint (> 4 steps away) - should fall back to binary
            idx, _, _ = search_interval(policy, x, 0.90)
            @test idx == 91
            @test hint[] == 91
        end

        @testset "Backward Linear Search" begin
            hint = Ref(60)
            policy = SearchPolicy{LinearBoundedAlg{8},RefHint}(RefHint(hint))

            # Query backward from hint
            idx, _, _ = search_interval(policy, x, 0.55)
            @test idx == 56
            @test hint[] == 56
        end

        @testset "Range Ignores Hint" begin
            x_range = range(0.0, 1.0, 101)
            hint = Ref(50)
            policy = SearchPolicy{LinearBoundedAlg{8},RefHint}(RefHint(hint))

            idx, _, _ = search_interval(policy, x_range, 0.1)
            @test idx == 11  # Direct O(1)
        end
    end

    # ========================================
    # Spacing-Aware Search Tests
    # ========================================

    @testset "Spacing-aware Search" begin
        x_range = range(0.0, 1.0, 101)
        x_vec = collect(x_range)
        spacing_scalar = _create_spacing(x_range)
        spacing_vector = _create_spacing(x_vec)
        policy = DEFAULT_SEARCH_POLICY

        @testset "ScalarSpacing Path" begin
            idx, xL, xR = search_interval(policy, x_range, spacing_scalar, 0.5)
            @test idx == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end

        @testset "VectorSpacing Path" begin
            idx, xL, xR = search_interval(policy, x_vec, spacing_vector, 0.5)
            @test idx == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end

        @testset "Internal Alias with Spacing" begin
            r1 = _search_interval(x_range, spacing_scalar, 0.5)
            r2 = FastInterpolations._find_interval(x_range, spacing_scalar, 0.5)
            @test r1 == r2
        end
    end

    # ========================================
    # Boundary and Edge Case Tests
    # ========================================

    @testset "Boundary Conditions" begin
        x = collect(range(0.0, 1.0, 101))
        policy = DEFAULT_SEARCH_POLICY

        @testset "Left Boundary" begin
            idx, xL, xR = search_interval(policy, x, 0.0)
            @test idx == 1
            @test xL ≈ 0.0 atol=1e-12
            @test xR ≈ 0.01 atol=1e-12
        end

        @testset "Right Boundary" begin
            idx, xL, xR = search_interval(policy, x, 1.0)
            @test idx == 100  # Last interval
            @test xL ≈ 0.99 atol=1e-12
            @test xR ≈ 1.0 atol=1e-12
        end

        @testset "Below Domain" begin
            idx, xL, xR = search_interval(policy, x, -0.1)
            @test idx == 1  # Clamped to first interval
        end

        @testset "Above Domain" begin
            idx, xL, xR = search_interval(policy, x, 1.1)
            @test idx == 100  # Clamped to last interval
        end

        @testset "Exact Grid Points" begin
            # xi == x[i] for i < n should return idx == i
            for i in 1:99
                xi = x[i]
                idx, _, _ = search_interval(policy, x, xi)
                @test idx == i
            end

            # xi == x[end] should return idx == n-1
            idx, _, _ = search_interval(policy, x, x[end])
            @test idx == 100
        end
    end

    # ========================================
    # Float32 Support Tests
    # ========================================

    @testset "Float32 Support" begin
        x32 = collect(range(0.0f0, 1.0f0, 101))
        policy = DEFAULT_SEARCH_POLICY

        @testset "Basic Float32" begin
            idx, xL, xR = search_interval(policy, x32, 0.5f0)
            @test idx == 51
            @test xL ≈ 0.50f0 atol=1f-6
            @test xR ≈ 0.51f0 atol=1f-6
        end

        @testset "Type Preservation" begin
            _, xL, xR = search_interval(policy, x32, 0.5f0)
            @test xL isa Float32
            @test xR isa Float32
        end
    end

    # ========================================
    # Non-Uniform Grid Tests
    # ========================================

    @testset "Non-Uniform Grids" begin
        # Clustered near boundaries
        x = [0.0, 0.01, 0.02, 0.1, 0.3, 0.7, 0.9, 0.98, 0.99, 1.0]
        policy = DEFAULT_SEARCH_POLICY

        @testset "Search in Varying Intervals" begin
            # Query in dense region
            idx, xL, xR = search_interval(policy, x, 0.015)
            @test idx == 2
            @test xL == 0.01
            @test xR == 0.02

            # Query in sparse region
            idx, xL, xR = search_interval(policy, x, 0.5)
            @test idx == 5
            @test xL == 0.3
            @test xR == 0.7
        end

        @testset "Hinted Search on Non-Uniform" begin
            hint = Ref(1)
            policy_hint = SearchPolicy{HintedBinaryAlg,RefHint}(RefHint(hint))

            # Sequential queries
            queries = [0.005, 0.015, 0.05, 0.2, 0.5, 0.8, 0.95]
            for xi in queries
                idx, _, _ = search_interval(policy_hint, x, xi)
                @test hint[] == idx
            end
        end
    end

    # ========================================
    # Property-Based Tests
    # ========================================

    @testset "Property-Based Tests" begin
        using Random
        Random.seed!(42)

        @testset "Monotonicity Property" begin
            x = sort(rand(100)) .* 10.0
            x[1] = 0.0
            x[end] = 10.0
            policy = DEFAULT_SEARCH_POLICY

            for _ in 1:100
                xi = rand() * 10.0
                idx, xL, xR = search_interval(policy, x, xi)

                # Property: xL ≤ xi ≤ xR (for inside-domain queries)
                if xL <= xi <= xR
                    @test true
                elseif xi < x[1]
                    @test idx == 1
                elseif xi > x[end]
                    @test idx == length(x) - 1
                else
                    # Unexpected case - fail with info
                    @test xL <= xi <= xR
                end
            end
        end
    end

end  # @testset "Search Module"
