using Test
using FastInterpolations
using FastInterpolations: search_interval, _search_binary, _search_direct, _search_interval,
    Searcher, Binary, HintedBinary, LinearBounded,
    NoHint, RefHint, DEFAULT_SEARCHER, ScalarSpacing, _create_spacing, _to_searcher

@testset "Search Module" begin

    # ========================================
    # Type System Tests
    # ========================================

    @testset "Search Policy Types" begin
        @testset "Type Hierarchy" begin
            @test Binary <: FastInterpolations.AbstractSearchPolicy
            @test HintedBinary <: FastInterpolations.AbstractSearchPolicy
            @test LinearBounded{8} <: FastInterpolations.AbstractSearchPolicy

            @test NoHint <: FastInterpolations.AbstractHint
            @test RefHint <: FastInterpolations.AbstractHint
        end

        @testset "Searcher Construction" begin
            # Default searcher
            @test DEFAULT_SEARCHER isa Searcher{Binary,NoHint}

            # RefHint construction
            hint = RefHint()
            @test hint.idx[] == 1

            hint2 = RefHint(50)
            @test hint2.idx[] == 50

            # Full searcher construction
            searcher = Searcher{HintedBinary,RefHint}(RefHint(Ref(10)))
            @test searcher.hint.idx[] == 10
        end
    end

    # ========================================
    # Default Searcher (Zero-Overhead) Tests
    # ========================================

    @testset "Default Searcher Zero-Overhead" begin
        x_vec = collect(range(0.0, 1.0, 101))
        x_range = range(0.0, 1.0, 101)
        searcher = DEFAULT_SEARCHER

        @testset "Vector Path" begin
            for xi in [0.0, 0.25, 0.5, 0.75, 1.0]
                result_policy = search_interval(searcher, x_vec, xi)
                result_direct = _search_binary(x_vec, xi)
                @test result_policy == result_direct
            end
        end

        @testset "Range Path" begin
            for xi in [0.0, 0.25, 0.5, 0.75, 1.0]
                result_policy = search_interval(searcher, x_range, xi)
                result_direct = _search_direct(x_range, xi)
                @test result_policy == result_direct
            end
        end

        @testset "Type Inference" begin
            # Must return concrete Tuple type
            @test @inferred(search_interval(searcher, x_vec, 0.5)) isa Tuple{Int,Float64,Float64}
            @test @inferred(search_interval(searcher, x_range, 0.5)) isa Tuple{Int,Float64,Float64}
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
    # Hinted Binary Search Tests
    # ========================================

    @testset "Hinted Binary with RefHint" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Hint Update" begin
            hint = Ref(1)
            policy = Searcher{HintedBinary,RefHint}(RefHint(hint))

            # Query far from hint - should update
            idx, _, _ = search_interval(policy, x, 0.75)
            @test idx == 76
            @test hint[] == 76
        end

        @testset "Hint Hit (O(1) path)" begin
            hint = Ref(50)
            policy = Searcher{HintedBinary,RefHint}(RefHint(hint))

            # Query within hint interval - should hit O(1) path
            # Interval 50 covers [0.49, 0.50)
            idx, xL, xR = search_interval(policy, x, 0.495)
            @test idx == 50
            @test hint[] == 50  # Hint unchanged (was already correct)
        end

        @testset "Monotonic Query Sequence" begin
            hint = Ref(1)
            policy = Searcher{HintedBinary,RefHint}(RefHint(hint))

            # Monotonically increasing queries
            for xi in range(0.1, 0.9, 10)
                idx, _, _ = search_interval(policy, x, xi)
                @test hint[] == idx
            end
        end

        @testset "Range Ignores Hint" begin
            x_range = range(0.0, 1.0, 101)
            hint = Ref(50)
            policy = Searcher{HintedBinary,RefHint}(RefHint(hint))

            # Range path always uses O(1) calculation, hint not updated
            idx, _, _ = search_interval(policy, x_range, 0.1)
            @test idx == 11  # Direct O(1) calculation
            @test hint[] == 50  # Hint unchanged for ranges
        end
    end

    # ========================================
    # Linear Bounded Search Tests
    # ========================================

    @testset "LinearBounded with RefHint" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Linear Search Success" begin
            hint = Ref(50)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

            # Query near hint (within 8 steps)
            idx, _, _ = search_interval(policy, x, 0.55)
            @test idx == 56
            @test hint[] == 56
        end

        @testset "Fallback to Binary" begin
            hint = Ref(10)
            policy = Searcher{LinearBounded{4},RefHint}(RefHint(hint))

            # Query far from hint (> 4 steps away) - should fall back to binary
            idx, _, _ = search_interval(policy, x, 0.90)
            @test idx == 91
            @test hint[] == 91
        end

        @testset "Backward Linear Search" begin
            hint = Ref(60)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

            # Query backward from hint
            idx, _, _ = search_interval(policy, x, 0.55)
            @test idx == 56
            @test hint[] == 56
        end

        @testset "Range Ignores Hint" begin
            x_range = range(0.0, 1.0, 101)
            hint = Ref(50)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

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
        policy = DEFAULT_SEARCHER

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
            # Verify spacing-aware path via internal alias
            r1 = _search_interval(x_range, spacing_scalar, 0.5)
            idx, xL, xR = r1
            @test idx == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end
    end

    # ========================================
    # Boundary and Edge Case Tests
    # ========================================

    @testset "Boundary Conditions" begin
        x = collect(range(0.0, 1.0, 101))
        policy = DEFAULT_SEARCHER

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
        policy = DEFAULT_SEARCHER

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
        policy = DEFAULT_SEARCHER

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
            policy_hint = Searcher{HintedBinary,RefHint}(RefHint(hint))

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
            policy = DEFAULT_SEARCHER

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

    # ========================================
    # Additional Coverage Tests
    # ========================================
    # These tests specifically target uncovered code paths

    @testset "Coverage: _search_hinted_binary! Direct" begin
        # Import internal function for direct testing
        _search_hinted_binary! = FastInterpolations._search_hinted_binary!

        x = collect(range(0.0, 1.0, 101))

        @testset "O(1) Hint Hit Path" begin
            hint_ref = Ref(50)
            # Query exactly in interval 50: [0.49, 0.50)
            idx, xL, xR = _search_hinted_binary!(x, 0.495, hint_ref)
            @test idx == 50
            @test xL ≈ 0.49 atol=1e-12
            @test xR ≈ 0.50 atol=1e-12
            @test hint_ref[] == 50  # Unchanged (direct hit)
        end

        @testset "Hint Miss - Fallback to Binary" begin
            hint_ref = Ref(10)
            # Query far from hint
            idx, xL, xR = _search_hinted_binary!(x, 0.75, hint_ref)
            @test idx == 76
            @test hint_ref[] == 76  # Updated after binary search
        end

        @testset "Hint at Boundary" begin
            hint_ref = Ref(1)
            # Query in first interval
            idx, _, _ = _search_hinted_binary!(x, 0.005, hint_ref)
            @test idx == 1
            @test hint_ref[] == 1
        end
    end

    @testset "Coverage: LinearBoundedAlg Direct Hit" begin
        # Test the early return path when hint already points to correct interval
        x = collect(range(0.0, 1.0, 101))

        @testset "Direct Hit - Hint Already Correct" begin
            # Hint at 50: interval [0.49, 0.50)
            # Query 0.495 is in interval 50, so direct hit (no search needed)
            hint = Ref(50)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

            idx, xL, xR = search_interval(policy, x, 0.495)
            @test idx == 50
            @test hint[] == 50  # Unchanged, direct hit
            @test xL ≈ 0.49 atol=1e-12
            @test xR ≈ 0.50 atol=1e-12
        end

        @testset "Direct Hit - Multiple Queries Same Interval" begin
            hint = Ref(30)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

            # Multiple queries in interval 30: [0.29, 0.30)
            for xi in [0.291, 0.295, 0.299]
                idx, xL, xR = search_interval(policy, x, xi)
                @test idx == 30
                @test hint[] == 30  # All direct hits
            end
        end
    end

    @testset "Coverage: Backward Linear Search Hit" begin
        # Grid: [0.0, 0.01, 0.02, ..., 1.0] - 101 points, 100 intervals
        # Interval i spans [x[i], x[i+1]) = [(i-1)*0.01, i*0.01)
        x = collect(range(0.0, 1.0, 101))

        @testset "Backward Search Finds Match After 1 Step" begin
            # Hint at index 60: interval [0.59, 0.60)
            # Query 0.585 < x[60]=0.59, so enters backward search
            # After 1 decrement: ix=59, interval [0.58, 0.59) contains 0.585 ✓
            hint = Ref(60)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

            idx, xL, xR = search_interval(policy, x, 0.585)
            @test idx == 59
            @test hint[] == 59  # Updated by backward search
            @test xL ≈ 0.58 atol=1e-12
            @test xR ≈ 0.59 atol=1e-12
        end

        @testset "Backward Search Finds Match After 3 Steps" begin
            # Hint at index 70: interval [0.69, 0.70)
            # Query 0.665 < x[70]=0.69, enters backward search
            # Need to find interval containing 0.665 = interval 67 [0.66, 0.67)
            # Steps: 70→69→68→67 (3 decrements)
            hint = Ref(70)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

            idx, xL, xR = search_interval(policy, x, 0.665)
            @test idx == 67
            @test hint[] == 67
            @test xL ≈ 0.66 atol=1e-12
            @test xR ≈ 0.67 atol=1e-12
        end

        @testset "Backward Search Single Step" begin
            # Most direct case: hint just 1 interval ahead
            # Hint at 52: interval [0.51, 0.52)
            # Query 0.505 < x[52]=0.51, backward 1 step to interval 51 [0.50, 0.51)
            hint = Ref(52)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

            idx, xL, xR = search_interval(policy, x, 0.505)
            @test idx == 51
            @test hint[] == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end
    end

    @testset "Coverage: LinearBoundedAlg with Range" begin
        x_range = range(0.0, 1.0, 101)
        hint = Ref(50)
        policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

        @testset "Range Uses O(1) - Ignores Hint" begin
            # Range path should use direct O(1) calculation
            idx, xL, xR = search_interval(policy, x_range, 0.25)
            @test idx == 26
            @test xL ≈ 0.25 atol=1e-12
            @test xR ≈ 0.26 atol=1e-12
            # Hint should NOT be updated for ranges
            @test hint[] == 50
        end

        @testset "Range Multiple Queries" begin
            for xi in [0.1, 0.3, 0.7, 0.9]
                idx, _, _ = search_interval(policy, x_range, xi)
                expected_idx = round(Int, xi * 100) + 1
                @test idx == expected_idx
            end
        end
    end

    @testset "Coverage: HintedBinaryAlg with Range" begin
        x_range = range(0.0, 1.0, 101)
        hint = Ref(80)
        policy = Searcher{HintedBinary,RefHint}(RefHint(hint))

        @testset "Range Uses O(1) - Ignores Hint" begin
            idx, xL, xR = search_interval(policy, x_range, 0.15)
            @test idx == 16
            @test hint[] == 80  # Hint unchanged for ranges
        end

        @testset "Range with ScalarSpacing" begin
            spacing = _create_spacing(x_range)
            idx, xL, xR = search_interval(policy, x_range, spacing, 0.25)
            @test idx == 26
            @test xL ≈ 0.25 atol=1e-12
            @test xR ≈ 0.26 atol=1e-12
            @test hint[] == 80  # Hint unchanged for ranges with spacing
        end
    end

    @testset "Coverage: LinearBoundedAlg with Range and Spacing" begin
        x_range = range(0.0, 1.0, 101)
        spacing = _create_spacing(x_range)
        hint = Ref(50)
        policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))

        @testset "Range with ScalarSpacing Uses O(1)" begin
            idx, xL, xR = search_interval(policy, x_range, spacing, 0.35)
            @test idx == 36
            @test xL ≈ 0.35 atol=1e-12
            @test xR ≈ 0.36 atol=1e-12
            @test hint[] == 50  # Hint unchanged for ranges with spacing
        end
    end

    @testset "Coverage: Spacing-aware with Default Policy" begin
        x_range = range(0.0, 1.0, 101)
        x_vec = collect(x_range)
        spacing_scalar = _create_spacing(x_range)
        spacing_vector = _create_spacing(x_vec)
        policy = DEFAULT_SEARCHER

        @testset "Direct search_interval with Spacing" begin
            # This specifically tests line 303-304
            idx, xL, xR = search_interval(policy, x_range, spacing_scalar, 0.75)
            @test idx == 76
            @test xL ≈ 0.75 atol=1e-12
            @test xR ≈ 0.76 atol=1e-12
        end

        @testset "Multiple Spacing Queries" begin
            for xi in [0.0, 0.25, 0.5, 0.75, 1.0]
                r1 = search_interval(policy, x_range, spacing_scalar, xi)
                r2 = search_interval(policy, x_vec, spacing_vector, xi)
                # Both should give same index
                @test r1[1] == r2[1]
            end
        end
    end

    @testset "Coverage: Edge Cases in Linear Search" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Linear Search at Domain Boundaries" begin
            # Near left boundary
            hint = Ref(5)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))
            idx, _, _ = search_interval(policy, x, 0.005)
            @test idx == 1
            @test hint[] == 1

            # Near right boundary
            hint2 = Ref(95)
            policy2 = Searcher{LinearBounded{8},RefHint}(RefHint(hint2))
            idx2, _, _ = search_interval(policy2, x, 0.995)
            @test idx2 == 100
            @test hint2[] == 100
        end

        @testset "Linear Search Backward at Left Edge" begin
            # Hint at 3, query at 0.0 - should clamp and handle
            hint = Ref(3)
            policy = Searcher{LinearBounded{8},RefHint}(RefHint(hint))
            idx, _, _ = search_interval(policy, x, 0.0)
            @test idx == 1
        end
    end

    # ========================================
    # LinearBounded Constructor Tests
    # ========================================

    @testset "LinearBounded Constructor" begin
        @testset "Valid max_steps Values" begin
            # All allowed max_steps values
            valid_steps = (1, 2, 4, 8, 16, 32, 64, 128)
            for ms in valid_steps
                policy = LinearBounded(max_steps=ms)
                @test policy isa LinearBounded{ms}

                # Also test positional argument
                policy2 = LinearBounded(ms)
                @test policy2 isa LinearBounded{ms}
            end

            # Default is 8
            @test LinearBounded() isa LinearBounded{8}
        end

        @testset "Invalid max_steps Throws ArgumentError" begin
            invalid_steps = (0, 3, 5, 6, 7, 9, 10, 15, 17, 100, 256)
            for ms in invalid_steps
                @test_throws ArgumentError LinearBounded(max_steps=ms)
                @test_throws ArgumentError LinearBounded(ms)
            end
        end
    end

    @testset "Integrated test" begin
        x = collect(range(0.0, 1.0, 101))
        y = x.^3

        xq = 0.5
        xq_vec = rand(10)

        out_vec1 = similar(xq_vec)
        out_vec2 = similar(xq_vec)
        out_vec3 = similar(xq_vec)
        out_vec4 = similar(xq_vec)
        out_vec5 = similar(xq_vec)

        itp = cubic_interp(x, y)

        itp(out_vec1, xq_vec) # Default search (Binary)
        itp(out_vec2, xq_vec; search=Binary()) # Default search (Binary)
        itp(out_vec3, xq_vec; search=HintedBinary()) # Default search (HintedBinary)
        itp(out_vec4, xq_vec; search=LinearBounded()) # Default search (LinearBounded{8})
        itp(out_vec5, xq_vec; search=LinearBounded{2}()) # Default search (LinearBounded{2})


        @test out_vec1 == out_vec2
        @test out_vec1 == out_vec3
        @test out_vec1 == out_vec4
        @test out_vec1 == out_vec5
    end

    # ========================================
    # Baked-in Default Search Policy Tests
    # ========================================
    # Tests that verify when an interpolant is created with a non-default search policy,
    # that policy is used by default (without explicit search= override at call time).

    @testset "Baked-in Default Search Policy" begin
        x = collect(range(0.0, 1.0, 1001))
        y = sin.(2π .* x)

        @testset "Single Interpolant: stored policy is used by default" begin
            # Create with LinearBounded as default
            itp_lb = linear_interp(x, y; search=LinearBounded())
            @test itp_lb.search_policy isa LinearBounded{8}

            # Create with HintedBinary as default
            itp_hb = linear_interp(x, y; search=HintedBinary())
            @test itp_hb.search_policy isa HintedBinary

            # Create with default (Binary)
            itp_bin = linear_interp(x, y)
            @test itp_bin.search_policy isa Binary
        end

        @testset "Baked-in policy with hint: hint updates when policy supports it" begin
            # Create interpolant with LinearBounded as default
            itp = linear_interp(x, y; search=LinearBounded())
            hint = Ref(500)

            # Call WITHOUT search= override → uses stored LinearBounded → hint should update
            xi = 0.5
            for _ in 1:50
                xi += 1e-3
                yi = itp(xi; hint=hint)  # Uses itp.search_policy (LinearBounded)
            end

            # hint should have tracked the position (~550-560)
            @test hint[] >= 540 && hint[] <= 570
        end

        @testset "Override with Binary + hint auto-upgrades to HintedBinary" begin
            # Create with LinearBounded default, but override with Binary at call time
            itp = linear_interp(x, y; search=LinearBounded())
            hint = Ref(100)

            # Override with Binary + hint → auto-upgrades to HintedBinary behavior
            for xi in range(0.5, 0.6, 10)
                yi = itp(xi; search=Binary(), hint=hint)
            end

            # hint should be updated (auto-upgraded to HintedBinary)
            @test hint[] >= 500 && hint[] <= 610
        end

        @testset "Cubic interpolant baked-in policy" begin
            itp = cubic_interp(x, y; search=LinearBounded(max_steps=4))
            @test itp.search_policy isa LinearBounded{4}

            hint = Ref(200)
            xi = 0.2
            for _ in 1:30
                xi += 2e-3
                yi = itp(xi; hint=hint)  # Uses baked-in LinearBounded{4}
            end

            # hint should track position (~260)
            @test hint[] >= 250 && hint[] <= 280
        end

        @testset "Quadratic interpolant baked-in policy" begin
            itp = quadratic_interp(x, y; search=HintedBinary())
            @test itp.search_policy isa HintedBinary

            hint = Ref(300)
            yi = itp(0.35; hint=hint)
            # HintedBinary updates hint
            @test hint[] >= 340 && hint[] <= 360
        end

        @testset "Constant interpolant baked-in policy" begin
            itp = constant_interp(x, y; search=LinearBounded(max_steps=16))
            @test itp.search_policy isa LinearBounded{16}
        end
    end

    @testset "Series Interpolant Baked-in Default Search" begin
        x = collect(range(0.0, 1.0, 1001))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        @testset "LinearSeriesInterpolant stored policy" begin
            sitp = linear_interp(x, [y1, y2]; search=LinearBounded())
            @test sitp.search_policy isa LinearBounded{8}

            # Scalar call uses stored policy
            hint = Ref(400)
            xi = 0.4
            for _ in 1:30
                xi += 1e-3
                yi = sitp(xi; hint=hint)  # Uses sitp.search_policy
            end
            @test hint[] >= 420 && hint[] <= 440
        end

        @testset "CubicSeriesInterpolant stored policy" begin
            sitp = cubic_interp(x, [y1, y2]; search=HintedBinary())
            @test sitp.search_policy isa HintedBinary

            hint = Ref(600)
            yi = sitp(0.65; hint=hint)
            @test hint[] >= 640 && hint[] <= 660
        end

        @testset "Series vector call with baked-in policy and hint" begin
            sitp = linear_interp(x, [y1, y2]; search=LinearBounded())
            hint = Ref(1)

            # Vector call with sorted queries
            xq = collect(range(0.1, 0.5, 100))
            outputs = sitp(xq; hint=hint)  # Uses stored LinearBounded

            # hint should track to end of query range (~500)
            @test hint[] >= 490 && hint[] <= 510
            @test length(outputs) == 2
            @test length(outputs[1]) == 100
        end

        @testset "Series override with Binary + hint auto-upgrades" begin
            sitp = linear_interp(x, [y1, y2]; search=LinearBounded())
            hint = Ref(250)

            # Override with Binary + hint → auto-upgrades to HintedBinary
            yi = sitp(0.75; search=Binary(), hint=hint)

            # hint should be updated (auto-upgraded to HintedBinary)
            @test hint[] >= 740 && hint[] <= 760
        end
    end

    # ========================================
    # Persistent Hint Tests (ODE/Streaming Pattern)
    # ========================================

    @testset "Persistent Hint via _to_searcher 2-arg" begin
        @testset "hint=nothing creates fresh RefHint" begin
            s1 = _to_searcher(LinearBounded(), nothing)
            @test s1.hint.idx[] == 1

            s2 = _to_searcher(HintedBinary(), nothing)
            @test s2.hint.idx[] == 1
        end

        @testset "hint=Ref uses external Ref" begin
            ext_ref = Ref(50)
            s = _to_searcher(LinearBounded(), ext_ref)
            @test s.hint.idx === ext_ref
            @test s.hint.idx[] == 50
        end

        @testset "Binary with hint auto-upgrades to HintedBinary" begin
            # Binary policy with hint auto-upgrades to HintedBinary behavior
            ext_ref = Ref(100)
            s1 = _to_searcher(Binary(), ext_ref)
            @test s1.hint isa RefHint
            @test s1.hint.idx === ext_ref  # Uses external Ref

            # Binary without hint stays pure Binary
            s2 = _to_searcher(Binary(), nothing)
            @test s2.hint isa NoHint
        end

        @testset "Searcher direct injection passthrough" begin
            # Pre-built Searcher should pass through unchanged
            ext_ref = Ref(42)
            pre_built = Searcher{LinearBounded{8},RefHint}(RefHint(ext_ref))

            s1 = _to_searcher(pre_built)
            @test s1 === pre_built

            s2 = _to_searcher(pre_built, nothing)
            @test s2 === pre_built

            s3 = _to_searcher(pre_built, Ref(99))
            @test s3 === pre_built  # Ignores new hint
        end
    end

    @testset "ODE-style Persistent Hint Pattern" begin
        x = collect(range(0.0, 1.0, 1001))
        y = sin.(2π .* x)

        @testset "LinearInterpolant with persistent hint" begin
            itp = linear_interp(x, y)
            hint = Ref(500)

            # Simulate ODE-style monotonic queries
            xi = 0.5
            for _ in 1:100
                xi += 1e-3
                yi = itp(xi; search=LinearBounded(), hint=hint)
            end

            # hint should be near xi position (0.6 → index ~601)
            @test hint[] >= 590 && hint[] <= 610
        end

        @testset "CubicInterpolant with persistent hint" begin
            itp = cubic_interp(x, y)
            hint = Ref(200)

            xi = 0.2
            for _ in 1:50
                xi += 2e-3
                yi = itp(xi; search=HintedBinary(), hint=hint)
            end

            # hint should track xi position (0.3 → index ~301)
            @test hint[] >= 290 && hint[] <= 310
        end

        @testset "hint=nothing is thread-safe (fresh each call)" begin
            itp = linear_interp(x, y)

            # Without hint kwarg, each call gets fresh state
            y1 = itp(0.1; search=LinearBounded())
            y2 = itp(0.9; search=LinearBounded())

            @test y1 ≈ sin(2π * 0.1) atol=1e-4
            @test y2 ≈ sin(2π * 0.9) atol=1e-4
        end
    end

    @testset "Persistent Hint with One-shot Functions" begin
        x = collect(range(0.0, 1.0, 101))
        y = x.^2

        @testset "linear_interp oneshot" begin
            hint = Ref(1)
            for xi in range(0.1, 0.5, 10)
                yi = linear_interp(x, y, xi; search=LinearBounded(), hint=hint)
                @test yi ≈ xi^2 atol=1e-4
            end
            # hint should have updated
            @test hint[] > 1
        end

        @testset "quadratic_interp oneshot" begin
            hint = Ref(50)
            for xi in range(0.5, 0.9, 10)
                yi = quadratic_interp(x, y, xi; search=HintedBinary(), hint=hint)
                @test yi ≈ xi^2 atol=1e-4
            end
            @test hint[] > 50
        end

        @testset "constant_interp oneshot" begin
            hint = Ref(10)
            yi = constant_interp(x, y, 0.55; search=LinearBounded(), hint=hint)
            @test hint[] == 56 || hint[] == 55  # Depends on side selection
        end
    end

    @testset "Persistent Hint with Series Interpolants" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = x.^2
        y2 = sin.(π .* x)

        @testset "LinearSeriesInterpolant" begin
            sitp = linear_interp(x, [y1, y2])
            hint = Ref(1)

            # Scalar query evaluates all series, returns first by default
            for xi in range(0.1, 0.4, 5)
                yi = sitp(xi; search=LinearBounded(), hint=hint)
            end
            @test hint[] >= 35 && hint[] <= 45

            # Continue with more queries (hint preserved)
            old_hint = hint[]
            for xi in range(0.5, 0.8, 5)
                yi = sitp(xi; search=LinearBounded(), hint=hint)
            end
            @test hint[] > old_hint
        end

        @testset "CubicSeriesInterpolant" begin
            sitp = cubic_interp(x, [y1, y2])
            hint = Ref(50)

            yi = sitp(0.55; search=HintedBinary(), hint=hint)
            @test hint[] >= 54 && hint[] <= 57
        end
    end

    @testset "Searcher Direct Injection (Advanced Usage)" begin
        x = collect(range(0.0, 1.0, 1001))
        y = x.^3
        itp = linear_interp(x, y)

        @testset "Pre-built Searcher avoids _to_searcher overhead" begin
            ext_ref = Ref(500)
            searcher = Searcher{LinearBounded{8},RefHint}(RefHint(ext_ref))

            # Directly inject searcher (no _to_searcher call at runtime)
            xi = 0.5
            for _ in 1:100
                xi += 1e-3
                yi = itp(xi; search=searcher)
            end

            # External ref should be updated
            @test ext_ref[] >= 590 && ext_ref[] <= 610
        end
    end

end  # @testset "Search Module"
