@testitem "Search Module" begin
    using FastInterpolations: search_interval, _search_binary, _search_direct, _search_interval,
        _search_interval_real,
        Searcher, BinarySearch, LinearSearch, LinearBinarySearch, AutoSearch, DirectSearch,
        NoHint, RefHint, DEFAULT_SEARCHER, _to_searcher,
        _resolve_search_policy, _is_likely_monotone, GridIdx

    # ========================================
    # Type System Tests
    # ========================================

    @testset "Search Policy Types" begin
        @testset "Type Hierarchy" begin
            @test BinarySearch <: FastInterpolations.AbstractSearchPolicy
            @test LinearSearch <: FastInterpolations.AbstractSearchPolicy
            @test LinearBinarySearch{8} <: FastInterpolations.AbstractSearchPolicy

            @test NoHint <: FastInterpolations.AbstractHint
            @test RefHint <: FastInterpolations.AbstractHint
        end

        @testset "Searcher Construction" begin
            # Default searcher
            @test DEFAULT_SEARCHER isa Searcher{BinarySearch, NoHint}

            # RefHint construction
            hint = RefHint()
            @test hint.idx[] == 1

            hint2 = RefHint(50)
            @test hint2.idx[] == 50

            # Full searcher construction
            searcher = Searcher{LinearBinarySearch{0}, RefHint}(RefHint(Ref(10)))
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
                i_p, _, xL_p, xR_p = search_interval(searcher, x_vec, xi)
                i_d, xL_d, xR_d = _search_binary(x_vec, xi)
                @test (i_p, xL_p, xR_p) == (i_d, xL_d, xR_d)
            end
        end

        @testset "Range Path" begin
            for xi in [0.0, 0.25, 0.5, 0.75, 1.0]
                i_p, _, xL_p, xR_p = search_interval(searcher, x_range, xi)
                i_d, xL_d, xR_d = _search_direct(x_range, xi)
                @test (i_p, xL_p, xR_p) == (i_d, xL_d, xR_d)
            end
        end

        @testset "Type Inference" begin
            # Must return concrete Tuple type
            @test @inferred(search_interval(searcher, x_vec, 0.5)) isa Tuple{Int, Int, Float64, Float64}
            @test @inferred(search_interval(searcher, x_range, 0.5)) isa Tuple{Int, Int, Float64, Float64}
        end
    end

    # ========================================
    # Internal Alias Tests
    # ========================================

    @testset "Internal Alias (_search_interval)" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Basic Functionality" begin
            idx, _, xL, xR = _search_interval(x, 0.5)
            @test idx == 51
            @test xL ≈ 0.5 atol = 1.0e-12
            @test xR ≈ 0.51 atol = 1.0e-12
        end

        @testset "Equivalence to _search_binary" begin
            for xi in [0.0, 0.1, 0.5, 0.9, 1.0]
                i_si, _, xL_si, xR_si = _search_interval(x, xi)
                i_b, xL_b, xR_b = _search_binary(x, xi)
                @test (i_si, xL_si, xR_si) == (i_b, xL_b, xR_b)
            end
        end
    end

    # ========================================
    # Hinted Binary Search Tests
    # ========================================

    @testset "LinearBinarySearch{0} with RefHint" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Hint Update" begin
            hint = Ref(1)
            policy = Searcher{LinearBinarySearch{0}, RefHint}(RefHint(hint))

            # Query far from hint - should update
            idx, _, _, _ = search_interval(policy, x, 0.75)
            @test idx == 76
            @test hint[] == 76
        end

        @testset "Hint Hit (O(1) path)" begin
            hint = Ref(50)
            policy = Searcher{LinearBinarySearch{0}, RefHint}(RefHint(hint))

            # Query within hint interval - should hit O(1) path
            # Interval 50 covers [0.49, 0.50)
            idx, _, xL, xR = search_interval(policy, x, 0.495)
            @test idx == 50
            @test hint[] == 50  # Hint unchanged (was already correct)
        end

        @testset "Monotonic Query Sequence" begin
            hint = Ref(1)
            policy = Searcher{LinearBinarySearch{0}, RefHint}(RefHint(hint))

            # Monotonically increasing queries
            for xi in range(0.1, 0.9, 10)
                idx, _, _, _ = search_interval(policy, x, xi)
                @test hint[] == idx
            end
        end

        @testset "Range Updates Hint" begin
            x_range = range(0.0, 1.0, 101)
            hint = Ref(50)
            policy = Searcher{LinearBinarySearch{0}, RefHint}(RefHint(hint))

            # Range path: hint checked first, then O(1) fallback + hint update
            idx, _, _, _ = search_interval(policy, x_range, 0.1)
            @test idx == 11  # Direct O(1) calculation
            @test hint[] == 11  # Hint updated to found index
        end
    end

    # ========================================
    # Pure Linear Search Tests
    # ========================================

    @testset "LinearSearch (Pure) with RefHint" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Type Construction" begin
            @test LinearSearch() isa LinearSearch
            @test _to_searcher(LinearSearch()) isa Searcher{LinearSearch, RefHint}
            @test _to_searcher(LinearSearch(), nothing) isa Searcher{LinearSearch, RefHint}

            ext_ref = Ref(42)
            s = _to_searcher(LinearSearch(), ext_ref)
            @test s.hint.idx === ext_ref
        end

        @testset "Monotonic Forward Sequence" begin
            hint = Ref(1)
            policy = Searcher{LinearSearch, RefHint}(RefHint(hint))

            # Strictly increasing queries (LinearSearch's intended use case)
            for xi in range(0.05, 0.95, 20)
                idx, _, xL, xR = search_interval(policy, x, xi)
                @test xL <= xi < xR
                @test hint[] == idx
            end
        end

        @testset "Monotonic Backward Sequence" begin
            hint = Ref(95)
            policy = Searcher{LinearSearch, RefHint}(RefHint(hint))

            # Strictly decreasing queries
            for xi in range(0.95, 0.05, 20)
                idx, _, xL, xR = search_interval(policy, x, xi)
                @test xL <= xi < xR
                @test hint[] == idx
            end
        end

        @testset "Direct Hit Optimization" begin
            hint = Ref(50)
            policy = Searcher{LinearSearch, RefHint}(RefHint(hint))

            # Query in same interval multiple times - O(1) each time
            for xi in [0.495, 0.496, 0.497]
                idx, _, _, _ = search_interval(policy, x, xi)
                @test idx == 50
            end
        end

        @testset "Range Updates Hint" begin
            x_range = range(0.0, 1.0, 101)
            hint = Ref(50)
            policy = Searcher{LinearSearch, RefHint}(RefHint(hint))

            idx, _, _, _ = search_interval(policy, x_range, 0.25)
            @test idx == 26
            @test hint[] == 26  # Hint updated to found index
        end

        @testset "Integration with Interpolants" begin
            y = x .^ 3
            itp = linear_interp(x, y)
            hint = Ref(1)

            # ODE-style monotonic evaluation
            xi = 0.1
            for _ in 1:50
                yi = itp(xi; search = LinearSearch(), hint = hint)
                @test yi ≈ xi^3 atol = 1.0e-4
                xi += 0.016
            end
            @test hint[] >= 85
        end
    end

    # ========================================
    # LinearBinarySearch Tests
    # ========================================

    @testset "LinearBinarySearch with RefHint" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Linear Walk Success" begin
            hint = Ref(50)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))

            # Query near hint (within 8 steps)
            idx, _, _, _ = search_interval(policy, x, 0.55)
            @test idx == 56
            @test hint[] == 56
        end

        @testset "Fallback to BinarySearch" begin
            hint = Ref(10)
            policy = Searcher{LinearBinarySearch{4}, RefHint}(RefHint(hint))

            # Query far from hint (> 4 steps away) - should fall back to binary
            idx, _, _, _ = search_interval(policy, x, 0.9)
            @test idx == 91
            @test hint[] == 91
        end

        @testset "Backward Linear Walk" begin
            hint = Ref(60)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))

            # Query backward from hint
            idx, _, _, _ = search_interval(policy, x, 0.55)
            @test idx == 56
            @test hint[] == 56
        end

        @testset "Range Ignores Hint" begin
            x_range = range(0.0, 1.0, 101)
            hint = Ref(50)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))

            idx, _, _, _ = search_interval(policy, x_range, 0.1)
            @test idx == 11  # Direct O(1)
        end
    end

    # ========================================
    # Boundary and Edge Case Tests
    # ========================================

    @testset "Boundary Conditions" begin
        x = collect(range(0.0, 1.0, 101))
        policy = DEFAULT_SEARCHER

        @testset "Left Boundary" begin
            idx, _, xL, xR = search_interval(policy, x, 0.0)
            @test idx == 1
            @test xL ≈ 0.0 atol = 1.0e-12
            @test xR ≈ 0.01 atol = 1.0e-12
        end

        @testset "Right Boundary" begin
            idx, _, xL, xR = search_interval(policy, x, 1.0)
            @test idx == 100  # Last interval
            @test xL ≈ 0.99 atol = 1.0e-12
            @test xR ≈ 1.0 atol = 1.0e-12
        end

        @testset "Below Domain" begin
            idx, _, xL, xR = search_interval(policy, x, -0.1)
            @test idx == 1  # Clamped to first interval
        end

        @testset "Above Domain" begin
            idx, _, xL, xR = search_interval(policy, x, 1.1)
            @test idx == 100  # Clamped to last interval
        end

        @testset "Exact Grid Points" begin
            # xi == x[i] for i < n should return idx == i
            for i in 1:99
                xi = x[i]
                idx, _, _, _ = search_interval(policy, x, xi)
                @test idx == i
            end

            # xi == x[end] should return idx == n-1
            idx, _, _, _ = search_interval(policy, x, x[end])
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
            idx, _, xL, xR = search_interval(policy, x32, 0.5f0)
            @test idx == 51
            @test xL ≈ 0.5f0 atol = 1.0f-6
            @test xR ≈ 0.51f0 atol = 1.0f-6
        end

        @testset "Type Preservation" begin
            _, _, xL, xR = search_interval(policy, x32, 0.5f0)
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
            idx, _, xL, xR = search_interval(policy, x, 0.015)
            @test idx == 2
            @test xL == 0.01
            @test xR == 0.02

            # Query in sparse region
            idx, _, xL, xR = search_interval(policy, x, 0.5)
            @test idx == 5
            @test xL == 0.3
            @test xR == 0.7
        end

        @testset "Hinted Search on Non-Uniform" begin
            hint = Ref(1)
            policy_hint = Searcher{LinearBinarySearch{0}, RefHint}(RefHint(hint))

            # Sequential queries
            queries = [0.005, 0.015, 0.05, 0.2, 0.5, 0.8, 0.95]
            for xi in queries
                idx, _, _, _ = search_interval(policy_hint, x, xi)
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
                idx, _, xL, xR = search_interval(policy, x, xi)

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

    @testset "Coverage: LinearBinarySearchAlg Direct Hit" begin
        # Test the early return path when hint already points to correct interval
        x = collect(range(0.0, 1.0, 101))

        @testset "Direct Hit - Hint Already Correct" begin
            # Hint at 50: interval [0.49, 0.50)
            # Query 0.495 is in interval 50, so direct hit (no search needed)
            hint = Ref(50)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))

            idx, _, xL, xR = search_interval(policy, x, 0.495)
            @test idx == 50
            @test hint[] == 50  # Unchanged, direct hit
            @test xL ≈ 0.49 atol = 1.0e-12
            @test xR ≈ 0.5 atol = 1.0e-12
        end

        @testset "Direct Hit - Multiple Queries Same Interval" begin
            hint = Ref(30)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))

            # Multiple queries in interval 30: [0.29, 0.30)
            for xi in [0.291, 0.295, 0.299]
                idx, _, xL, xR = search_interval(policy, x, xi)
                @test idx == 30
                @test hint[] == 30  # All direct hits
            end
        end
    end

    @testset "Coverage: Backward Linear Walk Hit" begin
        # Grid: [0.0, 0.01, 0.02, ..., 1.0] - 101 points, 100 intervals
        # Interval i spans [x[i], x[i+1]) = [(i-1)*0.01, i*0.01)
        x = collect(range(0.0, 1.0, 101))

        @testset "Backward Search Finds Match After 1 Step" begin
            # Hint at index 60: interval [0.59, 0.60)
            # Query 0.585 < x[60]=0.59, so enters backward search
            # After 1 decrement: ix=59, interval [0.58, 0.59) contains 0.585 ✓
            hint = Ref(60)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))

            idx, _, xL, xR = search_interval(policy, x, 0.585)
            @test idx == 59
            @test hint[] == 59  # Updated by backward search
            @test xL ≈ 0.58 atol = 1.0e-12
            @test xR ≈ 0.59 atol = 1.0e-12
        end

        @testset "Backward Search Finds Match After 3 Steps" begin
            # Hint at index 70: interval [0.69, 0.70)
            # Query 0.665 < x[70]=0.69, enters backward search
            # Need to find interval containing 0.665 = interval 67 [0.66, 0.67)
            # Steps: 70→69→68→67 (3 decrements)
            hint = Ref(70)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))

            idx, _, xL, xR = search_interval(policy, x, 0.665)
            @test idx == 67
            @test hint[] == 67
            @test xL ≈ 0.66 atol = 1.0e-12
            @test xR ≈ 0.67 atol = 1.0e-12
        end

        @testset "Backward Search Single Step" begin
            # Most direct case: hint just 1 interval ahead
            # Hint at 52: interval [0.51, 0.52)
            # Query 0.505 < x[52]=0.51, backward 1 step to interval 51 [0.50, 0.51)
            hint = Ref(52)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))

            idx, _, xL, xR = search_interval(policy, x, 0.505)
            @test idx == 51
            @test hint[] == 51
            @test xL ≈ 0.5 atol = 1.0e-12
            @test xR ≈ 0.51 atol = 1.0e-12
        end
    end

    @testset "Coverage: LinearBinarySearchAlg with Range" begin
        x_range = range(0.0, 1.0, 101)
        hint = Ref(50)
        policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))

        @testset "Range Updates Hint" begin
            # Range path: hint checked first, then O(1) fallback + hint update
            idx, _, xL, xR = search_interval(policy, x_range, 0.25)
            @test idx == 26
            @test xL ≈ 0.25 atol = 1.0e-12
            @test xR ≈ 0.26 atol = 1.0e-12
            @test hint[] == 26  # Hint updated to found index
        end

        @testset "Range Multiple Queries" begin
            for xi in [0.1, 0.3, 0.7, 0.9]
                idx, _, _, _ = search_interval(policy, x_range, xi)
                expected_idx = round(Int, xi * 100) + 1
                @test idx == expected_idx
            end
        end
    end

    @testset "Coverage: LinearBinarySearch{0} with Range" begin
        x_range = range(0.0, 1.0, 101)
        hint = Ref(80)
        policy = Searcher{LinearBinarySearch{0}, RefHint}(RefHint(hint))

        @testset "Range Updates Hint" begin
            idx, _, xL, xR = search_interval(policy, x_range, 0.15)
            @test idx == 16
            @test hint[] == 16  # Hint updated to found index
        end

    end

    @testset "Coverage: Default-policy Range search hits" begin
        x_range = range(0.0, 1.0, 101)
        x_vec = collect(x_range)
        policy = DEFAULT_SEARCHER

        @testset "Range result" begin
            idx, _, xL, xR = search_interval(policy, x_range, 0.75)
            @test idx == 76
            @test xL ≈ 0.75 atol = 1.0e-12
            @test xR ≈ 0.76 atol = 1.0e-12
        end

        @testset "Range vs Vector agree" begin
            for xi in [0.0, 0.25, 0.5, 0.75, 1.0]
                r1 = search_interval(policy, x_range, xi)
                r2 = search_interval(policy, x_vec, xi)
                @test r1[1] == r2[1]
            end
        end
    end

    @testset "Coverage: Edge Cases in LinearBinarySearch" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "LinearBinarySearch at Domain Boundaries" begin
            # Near left boundary
            hint = Ref(5)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))
            idx, _, _, _ = search_interval(policy, x, 0.005)
            @test idx == 1
            @test hint[] == 1

            # Near right boundary
            hint2 = Ref(95)
            policy2 = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint2))
            idx2, _, _, _ = search_interval(policy2, x, 0.995)
            @test idx2 == 100
            @test hint2[] == 100
        end

        @testset "LinearBinarySearch Backward at Left Edge" begin
            # Hint at 3, query at 0.0 - should clamp and handle
            hint = Ref(3)
            policy = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint))
            idx, _, _, _ = search_interval(policy, x, 0.0)
            @test idx == 1
        end
    end

    # ========================================
    # LinearBinarySearch Constructor Tests
    # ========================================

    @testset "LinearBinarySearch Constructor" begin
        @testset "Valid linear_window Values" begin
            # All allowed linear_window values
            @test LinearBinarySearch(linear_window = 0) isa LinearBinarySearch{0}
            @test LinearBinarySearch(linear_window = 1) isa LinearBinarySearch{1}
            @test LinearBinarySearch(linear_window = 2) isa LinearBinarySearch{2}
            @test LinearBinarySearch(linear_window = 4) isa LinearBinarySearch{4}
            @test LinearBinarySearch(linear_window = 8) isa LinearBinarySearch{8}
            @test LinearBinarySearch(linear_window = 16) isa LinearBinarySearch{16}
            @test LinearBinarySearch(linear_window = 32) isa LinearBinarySearch{32}
            @test LinearBinarySearch(linear_window = 64) isa LinearBinarySearch{64}
            @test LinearBinarySearch(linear_window = 128) isa LinearBinarySearch{128}

            # Test positional argument
            @test LinearBinarySearch(4) isa LinearBinarySearch{4}
            @test LinearBinarySearch(16) isa LinearBinarySearch{16}

            # Default is 8
            @test LinearBinarySearch() isa LinearBinarySearch{8}
        end

        @testset "Invalid linear_window Throws ArgumentError" begin
            invalid_steps = (3, 5, 6, 7, 9, 10, 15, 17, 100, 256)
            for ms in invalid_steps
                @test_throws ArgumentError LinearBinarySearch(linear_window = ms)
                @test_throws ArgumentError LinearBinarySearch(ms)
            end
        end
    end

    @testset "LinearBinarySearch: out-of-range hint is clamped safely" begin
        x = collect(range(0.0, 1.0, 101))
        y = x .^ 2
        itp = linear_interp(x, y; search = LinearBinarySearch())

        # Ref(0) — below valid range [1, n-1]
        bad_hint = Ref(0)
        @test isfinite(itp(0.5; hint = bad_hint))
        @test bad_hint[] >= 1   # hint was clamped then updated to valid index

        # Ref(n) — above valid range [1, n-1]
        big_hint = Ref(200)
        @test isfinite(itp(0.5; hint = big_hint))
        @test big_hint[] <= 100  # valid range [1, n-1]=[1,100]; query 0.5 → idx ≈ 50
    end

    @testset "Integrated test" begin
        x = collect(range(0.0, 1.0, 101))
        y = x .^ 3

        xq = 0.5
        xq_vec = sort(rand(10))  # Sort for LinearSearch() compatibility

        out_vec1 = similar(xq_vec)
        out_vec2 = similar(xq_vec)
        out_vec3 = similar(xq_vec)
        out_vec4 = similar(xq_vec)
        out_vec5 = similar(xq_vec)
        out_vec6 = similar(xq_vec)

        itp = cubic_interp(x, y)

        itp(out_vec1, xq_vec) # Default search (BinarySearch)
        itp(out_vec2, xq_vec; search = BinarySearch()) # BinarySearch
        itp(out_vec3, xq_vec; search = LinearBinarySearch(linear_window = 0)) # LinearBinarySearch{0}
        itp(out_vec4, xq_vec; search = LinearSearch()) # LinearSearch (pure)
        itp(out_vec5, xq_vec; search = LinearBinarySearch()) # LinearBinarySearch{8}
        itp(out_vec6, xq_vec; search = LinearBinarySearch{2}()) # LinearBinarySearch{2}

        @test out_vec1 == out_vec2
        @test out_vec1 == out_vec3
        @test out_vec1 == out_vec4
        @test out_vec1 == out_vec5
        @test out_vec1 == out_vec6
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
            # Create with LinearBinarySearch as default
            itp_lb = linear_interp(x, y; search = LinearBinarySearch())
            @test itp_lb.search_policy isa LinearBinarySearch{8}

            # Create with LinearBinarySearch{0} as default
            itp_hb = linear_interp(x, y; search = LinearBinarySearch(linear_window = 0))
            @test itp_hb.search_policy isa LinearBinarySearch{0}

            # Create with default (AutoSearch)
            itp_auto = linear_interp(x, y)
            @test itp_auto.search_policy isa AutoSearch
        end

        @testset "Baked-in policy with hint: hint updates when policy supports it" begin
            # Create interpolant with LinearBinarySearch as default
            itp = linear_interp(x, y; search = LinearBinarySearch())
            hint = Ref(500)

            # Call WITHOUT search= override → uses stored LinearBinarySearch → hint should update
            xi = 0.5
            for _ in 1:50
                xi += 1.0e-3
                yi = itp(xi; hint = hint)  # Uses itp.search_policy (LinearBinarySearch)
            end

            # hint should have tracked the position (~550-560)
            @test hint[] >= 540 && hint[] <= 570
        end

        @testset "Override with BinarySearch + hint → LB{0} (hint tracks)" begin
            # Create with LinearBinarySearch default, but override with BinarySearch at call time
            itp = linear_interp(x, y; search = LinearBinarySearch())
            hint = Ref(100)

            # Override with BinarySearch + hint → LB{0}: direct-hit check + binary + hint write-back
            for xi in range(0.5, 0.6, 10)
                yi = itp(xi; search = BinarySearch(), hint = hint)
            end

            # hint should be updated (LB{0} writes back after binary search)
            @test hint[] >= 500 && hint[] <= 610
        end

        @testset "Cubic interpolant baked-in policy" begin
            itp = cubic_interp(x, y; search = LinearBinarySearch(linear_window = 4))
            @test itp.search_policy isa LinearBinarySearch{4}

            hint = Ref(200)
            xi = 0.2
            for _ in 1:30
                xi += 2.0e-3
                yi = itp(xi; hint = hint)  # Uses baked-in LinearBinarySearch{4}
            end

            # hint should track position (~260)
            @test hint[] >= 250 && hint[] <= 280
        end

        @testset "Quadratic interpolant baked-in policy" begin
            itp = quadratic_interp(x, y; search = LinearBinarySearch(linear_window = 0))
            @test itp.search_policy isa LinearBinarySearch{0}

            hint = Ref(300)
            yi = itp(0.35; hint = hint)
            # LinearBinarySearch{0} updates hint
            @test hint[] >= 340 && hint[] <= 360
        end

        @testset "Constant interpolant baked-in policy" begin
            itp = constant_interp(x, y; search = LinearBinarySearch(linear_window = 16))
            @test itp.search_policy isa LinearBinarySearch{16}
        end
    end

    @testset "Series Interpolant Baked-in Default Search" begin
        x = collect(range(0.0, 1.0, 1001))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        @testset "LinearSeriesInterpolant stored policy" begin
            sitp = linear_interp(x, Series(y1, y2); search = LinearBinarySearch())
            @test sitp.search_policy isa LinearBinarySearch{8}

            # Scalar call uses stored policy
            hint = Ref(400)
            xi = 0.4
            for _ in 1:30
                xi += 1.0e-3
                yi = sitp(xi; hint = hint)  # Uses sitp.search_policy
            end
            @test hint[] >= 420 && hint[] <= 440
        end

        @testset "CubicSeriesInterpolant stored policy" begin
            sitp = cubic_interp(x, Series(y1, y2); search = LinearBinarySearch(linear_window = 0))
            @test sitp.search_policy isa LinearBinarySearch{0}

            hint = Ref(600)
            yi = sitp(0.65; hint = hint)
            @test hint[] >= 640 && hint[] <= 660
        end

        @testset "Series vector call with baked-in policy and hint" begin
            sitp = linear_interp(x, Series(y1, y2); search = LinearBinarySearch())
            hint = Ref(1)

            # Vector call with sorted queries
            xq = collect(range(0.1, 0.5, 100))
            outputs = sitp(xq; hint = hint)  # Uses stored LinearBinarySearch

            # hint should track to end of query range (~500)
            @test hint[] >= 490 && hint[] <= 510
            @test length(outputs) == 2
            @test length(outputs[1]) == 100
        end

        @testset "Series override with BinarySearch + hint → LB{0} (hint tracks)" begin
            sitp = linear_interp(x, Series(y1, y2); search = LinearBinarySearch())
            hint = Ref(250)

            # Override with BinarySearch + hint → LB{0}: hint write-back
            yi = sitp(0.75; search = BinarySearch(), hint = hint)

            # hint should be updated (LB{0} writes back after binary search)
            @test hint[] >= 740 && hint[] <= 760
        end
    end

    # ========================================
    # _to_searcher 1-arg (used by oneshot functions)
    # ========================================

    @testset "_to_searcher 1-arg coverage" begin
        # These are used by oneshot functions like linear_interp!(output, x, y, xq_vec; search=...)
        s1 = _to_searcher(BinarySearch())
        @test s1.hint isa NoHint

        s3 = _to_searcher(LinearSearch())
        @test s3.hint isa RefHint

        s4 = _to_searcher(LinearBinarySearch())
        @test s4.hint isa RefHint
    end

    # ========================================
    # Persistent Hint Tests (ODE/Streaming Pattern)
    # ========================================

    @testset "Persistent Hint via _to_searcher 2-arg" begin
        @testset "hint=nothing creates fresh RefHint" begin
            s1 = _to_searcher(LinearBinarySearch(), nothing)
            @test s1.hint.idx[] == 1

            s3 = _to_searcher(LinearSearch(), nothing)
            @test s3.hint.idx[] == 1
        end

        @testset "hint=Ref uses external Ref" begin
            ext_ref = Ref(50)
            s = _to_searcher(LinearBinarySearch(), ext_ref)
            @test s.hint.idx === ext_ref
            @test s.hint.idx[] == 50

            ext_ref2 = Ref(75)
            s2 = _to_searcher(LinearSearch(), ext_ref2)
            @test s2.hint.idx === ext_ref2
            @test s2.hint.idx[] == 75
        end

        @testset "BinarySearch with hint → Binary + RefHint (write-back)" begin
            # BinarySearch + hint → Binary stays Binary, hint used for write-back only
            ext_ref = Ref(100)
            s1 = _to_searcher(BinarySearch(), ext_ref)
            @test s1 isa Searcher{BinarySearch, RefHint}
            @test s1.hint isa RefHint
            @test s1.hint.idx === ext_ref  # Uses external Ref

            # BinarySearch without hint stays pure BinarySearch
            s2 = _to_searcher(BinarySearch(), nothing)
            @test s2.hint isa NoHint
        end

        # Searcher direct injection passthrough removed:
        # API sealing (::AbstractSearchPolicy on all user kwargs) makes
        # _to_searcher(::Searcher, ...) unreachable — no test needed.
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
                xi += 1.0e-3
                yi = itp(xi; search = LinearBinarySearch(), hint = hint)
            end

            # hint should be near xi position (0.6 → index ~601)
            @test hint[] >= 590 && hint[] <= 610
        end

        @testset "CubicInterpolant with persistent hint" begin
            itp = cubic_interp(x, y)
            hint = Ref(200)

            xi = 0.2
            for _ in 1:50
                xi += 2.0e-3
                yi = itp(xi; search = LinearBinarySearch(linear_window = 0), hint = hint)
            end

            # hint should track xi position (0.3 → index ~301)
            @test hint[] >= 290 && hint[] <= 310
        end

        @testset "hint=nothing is thread-safe (fresh each call)" begin
            itp = linear_interp(x, y)

            # Without hint kwarg, each call gets fresh state
            y1 = itp(0.1; search = LinearBinarySearch())
            y2 = itp(0.9; search = LinearBinarySearch())

            @test y1 ≈ sin(2π * 0.1) atol = 1.0e-4
            @test y2 ≈ sin(2π * 0.9) atol = 1.0e-4
        end
    end

    @testset "Persistent Hint with One-shot Functions" begin
        x = collect(range(0.0, 1.0, 101))
        y = x .^ 2

        @testset "linear_interp oneshot" begin
            hint = Ref(1)
            for xi in range(0.1, 0.5, 10)
                yi = linear_interp(x, y, xi; search = LinearBinarySearch(), hint = hint)
                @test yi ≈ xi^2 atol = 1.0e-4
            end
            # hint should have updated
            @test hint[] > 1
        end

        @testset "quadratic_interp oneshot" begin
            hint = Ref(50)
            for xi in range(0.5, 0.9, 10)
                yi = quadratic_interp(x, y, xi; search = LinearBinarySearch(linear_window = 0), hint = hint)
                @test yi ≈ xi^2 atol = 1.0e-4
            end
            @test hint[] > 50
        end

        @testset "constant_interp oneshot" begin
            hint = Ref(10)
            yi = constant_interp(x, y, 0.55; search = LinearBinarySearch(), hint = hint)
            @test hint[] == 56 || hint[] == 55  # Depends on side selection
        end
    end

    @testset "Persistent Hint with Series Interpolants" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = x .^ 2
        y2 = sin.(π .* x)

        @testset "LinearSeriesInterpolant" begin
            sitp = linear_interp(x, Series(y1, y2))
            hint = Ref(1)

            # Scalar query evaluates all series, returns first by default
            for xi in range(0.1, 0.4, 5)
                yi = sitp(xi; search = LinearBinarySearch(), hint = hint)
            end
            @test hint[] >= 35 && hint[] <= 45

            # Continue with more queries (hint preserved)
            old_hint = hint[]
            for xi in range(0.5, 0.8, 5)
                yi = sitp(xi; search = LinearBinarySearch(), hint = hint)
            end
            @test hint[] > old_hint
        end

        @testset "CubicSeriesInterpolant" begin
            sitp = cubic_interp(x, Series(y1, y2))
            hint = Ref(50)

            yi = sitp(0.55; search = LinearBinarySearch(linear_window = 0), hint = hint)
            @test hint[] >= 54 && hint[] <= 57
        end
    end

    # ============================================================================
    # AutoSearch Resolution Tests
    # ============================================================================
    # Verifies that AutoSearch resolves to the correct concrete policy at every
    # API layer: unit dispatch, interpolant callable, oneshot, and series.

    @testset "AutoSearch Resolution" begin

        # ========================================
        # Unit Tests: _resolve_search dispatch
        # ========================================
        @testset "_resolve_search dispatch" begin
            auto = AutoSearch()

            # 1D scalar → BinarySearch
            @test _resolve_search_policy(auto, 0.5) isa BinarySearch
            @test _resolve_search_policy(auto, 1) isa BinarySearch       # Int <: Real
            @test _resolve_search_policy(auto, Float32(0.5)) isa BinarySearch

            # 1D vector → LinearBinarySearch
            @test _resolve_search_policy(auto, [0.1, 0.5]) isa LinearBinarySearch
            @test _resolve_search_policy(auto, 0.0:0.1:1.0) isa LinearBinarySearch   # Range <: AbstractVector
            @test _resolve_search_policy(auto, Float32[1.0, 2.0]) isa LinearBinarySearch

            # ND scalar (Tuple of Reals) → BinarySearch
            @test _resolve_search_policy(auto, (0.5, 0.3)) isa BinarySearch
            @test _resolve_search_policy(auto, (0.1, 0.2, 0.3)) isa BinarySearch

            # ND vector (Tuple of Vectors) → LinearBinarySearch
            @test _resolve_search_policy(auto, ([0.1, 0.2], [0.3, 0.4])) isa LinearBinarySearch
            @test _resolve_search_policy(auto, (0.0:0.1:1.0, [0.5])) isa LinearBinarySearch

            # Passthrough: explicit policies are returned unchanged
            @test _resolve_search_policy(BinarySearch(), 0.5) === BinarySearch()
            @test _resolve_search_policy(BinarySearch(), [0.1]) === BinarySearch()
            @test _resolve_search_policy(LinearBinarySearch(), 0.5) isa LinearBinarySearch
            @test _resolve_search_policy(LinearBinarySearch(), [0.1]) isa LinearBinarySearch
            @test _resolve_search_policy(LinearSearch(), [0.1]) === LinearSearch()

            # Tuple of policies: each resolved independently
            mixed = (AutoSearch(), BinarySearch(), AutoSearch())
            resolved_scalar = _resolve_search_policy(mixed, 0.5)
            @test resolved_scalar == (BinarySearch(), BinarySearch(), BinarySearch())
            resolved_vec = _resolve_search_policy(mixed, [0.1, 0.2])
            @test resolved_vec[1] isa LinearBinarySearch
            @test resolved_vec[2] === BinarySearch()
            @test resolved_vec[3] isa LinearBinarySearch

        end

        # ========================================
        # Interpolant API: AutoSearch resolves per query type
        # ========================================
        @testset "Interpolant API" begin
            x = collect(range(0.0, 1.0, 201))
            y = sin.(2π .* x)

            @testset "LinearSearch" begin
                itp = linear_interp(x, y)
                @test itp.search_policy isa AutoSearch

                # Scalar call → correct result (AutoSearch → BinarySearch internally)
                val = itp(0.25)
                @test val ≈ sin(2π * 0.25) atol = 1.0e-3

                # Vector call → correct results (AutoSearch → LinearBinarySearch internally)
                xq = [0.1, 0.3, 0.7]
                vals = itp(xq)
                @test length(vals) == 3
                @test vals ≈ sin.(2π .* xq) atol = 0.02

                # In-place vector call
                out = zeros(3)
                itp(out, xq)
                @test out ≈ vals
            end

            @testset "Cubic" begin
                itp = cubic_interp(x, y)
                @test itp.search_policy isa AutoSearch

                val = itp(0.25)
                @test val ≈ sin(2π * 0.25) atol = 1.0e-5

                xq = [0.1, 0.3, 0.7]
                vals = itp(xq)
                @test length(vals) == 3
                for i in 1:3
                    @test vals[i] ≈ sin(2π * xq[i]) atol = 1.0e-4
                end
            end

            @testset "Quadratic" begin
                itp = quadratic_interp(x, y)
                @test itp.search_policy isa AutoSearch

                val = itp(0.25)
                @test val ≈ sin(2π * 0.25) atol = 1.0e-3

                xq = [0.1, 0.3, 0.7]
                vals = itp(xq)
                @test length(vals) == 3
                for i in 1:3
                    @test vals[i] ≈ sin(2π * xq[i]) atol = 1.0e-2
                end
            end

            @testset "Constant" begin
                itp = constant_interp(x, y)
                @test itp.search_policy isa AutoSearch

                # Constant interpolation snaps to nearest grid point
                val = itp(0.25)
                @test isfinite(val)

                xq = [0.1, 0.3, 0.7]
                vals = itp(xq)
                @test length(vals) == 3
                @test all(isfinite, vals)
            end
        end

        # ========================================
        # Interpolant API: hint behavior verifies resolved policy
        # ========================================
        # LinearBinarySearch with hint tracks sequentially through sorted queries.
        # BinarySearch (no hint) doesn't update the hint.
        # AutoSearch + vector → LinearBinarySearch → hint should track.
        # AutoSearch + scalar → BinarySearch → hint unchanged (no hint used).
        @testset "Interpolant hint tracking reveals resolved policy" begin
            x = collect(range(0.0, 1.0, 1001))
            y = x .^ 2

            itp = linear_interp(x, y)  # default AutoSearch

            # Vector call with hint: AutoSearch → LinearBinarySearch → hint tracks
            hint = Ref(1)
            xq_sorted = collect(range(0.1, 0.9, 100))
            itp(zeros(100), xq_sorted; hint = hint)
            # 1001-point grid, last query at x=0.9 → idx ≈ 901; margin of 50
            @test hint[] >= 850

            # Scalar call without hint: AutoSearch → BinarySearch → correct value
            # y = x^2, so itp(0.5) ≈ 0.25
            @test itp(0.5) ≈ 0.25 atol = 1.0e-6

            # Scalar call with hint: BinarySearch+hint → LB{0} → hint updated via write-back
            hint_scalar = Ref(1)
            itp(0.5; hint = hint_scalar)
            @test hint_scalar[] >= 490 && hint_scalar[] <= 510  # hint moved to ~500
        end

        # ========================================
        # Oneshot API: AutoSearch resolves per query type
        # ========================================
        @testset "Oneshot API" begin
            x = collect(range(0.0, 1.0, 201))
            y = sin.(2π .* x)

            @testset "linear_interp scalar oneshot (AutoSearch default)" begin
                val = linear_interp(x, y, 0.25)
                @test val ≈ sin(2π * 0.25) atol = 1.0e-3
            end

            @testset "linear_interp vector oneshot (AutoSearch default)" begin
                xq = collect(range(0.1, 0.9, 50))
                vals = linear_interp(x, y, xq)
                @test length(vals) == 50
                for i in 1:50
                    @test vals[i] ≈ sin(2π * xq[i]) atol = 1.0e-2
                end
            end

            @testset "linear_interp! vector oneshot (AutoSearch default)" begin
                xq = collect(range(0.1, 0.9, 50))
                out = zeros(50)
                linear_interp!(out, x, y, xq)
                for i in 1:50
                    @test out[i] ≈ sin(2π * xq[i]) atol = 1.0e-2
                end
            end

            @testset "cubic_interp scalar oneshot" begin
                val = cubic_interp(x, y, 0.25)
                @test val ≈ sin(2π * 0.25) atol = 1.0e-5
            end

            @testset "cubic_interp! vector oneshot" begin
                xq = collect(range(0.1, 0.9, 50))
                out = zeros(50)
                cubic_interp!(out, x, y, xq)
                for i in 1:50
                    @test out[i] ≈ sin(2π * xq[i]) atol = 1.0e-3
                end
            end

            @testset "quadratic_interp scalar oneshot" begin
                val = quadratic_interp(x, y, 0.25)
                @test val ≈ sin(2π * 0.25) atol = 1.0e-3
            end

            @testset "quadratic_interp! vector oneshot" begin
                xq = collect(range(0.1, 0.9, 50))
                out = zeros(50)
                quadratic_interp!(out, x, y, xq)
                for i in 1:50
                    @test out[i] ≈ sin(2π * xq[i]) atol = 1.0e-2
                end
            end

            @testset "constant_interp scalar oneshot" begin
                val = constant_interp(x, y, 0.25)
                @test isfinite(val)
            end

            @testset "constant_interp! vector oneshot" begin
                xq = collect(range(0.1, 0.9, 50))
                out = zeros(50)
                constant_interp!(out, x, y, xq)
                @test all(isfinite, out)
            end
        end

        # ========================================
        # Series Interpolant API: AutoSearch resolves per query type
        # ========================================
        @testset "Series API" begin
            x = collect(range(0.0, 1.0, 201))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            @testset "LinearSeries" begin
                sitp = linear_interp(x, Series(y1, y2))
                @test sitp.search_policy isa AutoSearch

                # Scalar call → AutoSearch → BinarySearch
                vals = sitp(0.25)
                @test length(vals) == 2
                @test vals[1] ≈ sin(2π * 0.25) atol = 1.0e-3
                @test vals[2] ≈ cos(2π * 0.25) atol = 1.0e-3

                # Vector call → AutoSearch → LinearBinarySearch
                xq = [0.1, 0.3, 0.7]
                result = sitp(xq)
                @test length(result) == 2       # 2 series
                @test length(result[1]) == 3    # 3 query points
                @test result[1][1] ≈ sin(2π * 0.1) atol = 1.0e-2
                @test result[2][1] ≈ cos(2π * 0.1) atol = 1.0e-2
            end

            @testset "CubicSeries" begin
                sitp = cubic_interp(x, Series(y1, y2))
                @test sitp.search_policy isa AutoSearch

                vals = sitp(0.25)
                @test vals[1] ≈ sin(2π * 0.25) atol = 1.0e-5
                @test vals[2] ≈ cos(2π * 0.25) atol = 1.0e-5

                xq = [0.1, 0.3, 0.7]
                result = sitp(xq)
                @test length(result) == 2
                for i in 1:3
                    @test result[1][i] ≈ sin(2π * xq[i]) atol = 1.0e-3
                end
            end

            @testset "QuadraticSeries" begin
                sitp = quadratic_interp(x, Series(y1, y2))
                @test sitp.search_policy isa AutoSearch

                vals = sitp(0.25)
                @test vals[1] ≈ sin(2π * 0.25) atol = 1.0e-3

                xq = [0.1, 0.3, 0.7]
                result = sitp(xq)
                @test length(result) == 2
                @test length(result[1]) == 3
            end

            @testset "ConstantSeries" begin
                sitp = constant_interp(x, Series(y1, y2))
                @test sitp.search_policy isa AutoSearch

                vals = sitp(0.25)
                @test length(vals) == 2
                @test all(isfinite, vals)

                xq = [0.1, 0.3, 0.7]
                result = sitp(xq)
                @test length(result) == 2
                @test length(result[1]) == 3
            end

            @testset "Series hint tracking (vector → LinearBinarySearch)" begin
                sitp = linear_interp(x, Series(y1, y2))
                hint = Ref(1)
                xq_sorted = collect(range(0.1, 0.9, 100))

                # Vector call: AutoSearch → LinearBinarySearch → hint tracks
                sitp(xq_sorted; hint = hint)
                # 201-point grid, last query at x=0.9 → idx ≈ 181; margin of 21
                @test hint[] >= 160
            end
        end

        # ========================================
        # Explicit User Override: policy honored across all APIs
        # ========================================
        @testset "Explicit user override" begin
            x = collect(range(0.0, 1.0, 201))
            y = sin.(2π .* x)
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            @testset "Interpolant: explicit BinarySearch on vector call" begin
                itp = linear_interp(x, y; search = BinarySearch())
                @test itp.search_policy isa BinarySearch

                # Vector call with BinarySearch baked in → still works correctly
                xq = [0.1, 0.5, 0.9]
                vals = itp(xq)
                for i in 1:3
                    @test vals[i] ≈ sin(2π * xq[i]) atol = 1.0e-2
                end
            end

            @testset "Interpolant: explicit LinearBinarySearch on scalar call" begin
                itp = linear_interp(x, y; search = LinearBinarySearch())
                @test itp.search_policy isa LinearBinarySearch

                # Scalar call with LinearBinarySearch baked in → still works
                val = itp(0.25)
                @test val ≈ sin(2π * 0.25) atol = 1.0e-3
            end

            @testset "Interpolant: call-site override trumps stored policy" begin
                itp = linear_interp(x, y)  # AutoSearch stored
                @test itp.search_policy isa AutoSearch

                # Override with explicit BinarySearch at call-site
                val = itp(0.25; search = BinarySearch())
                @test val ≈ sin(2π * 0.25) atol = 1.0e-3

                # Override with explicit LinearBinarySearch at call-site
                vals = itp([0.1, 0.5]; search = LinearBinarySearch())
                @test length(vals) == 2
            end

            @testset "Oneshot: explicit policy" begin
                xq = collect(range(0.1, 0.9, 50))
                out = zeros(50)

                # Explicit BinarySearch
                linear_interp!(out, x, y, xq; search = BinarySearch())
                @test all(isfinite, out)

                # Explicit LinearBinarySearch
                linear_interp!(out, x, y, xq; search = LinearBinarySearch())
                @test all(isfinite, out)

                # Scalar oneshot with explicit policy
                val = linear_interp(x, y, 0.25; search = LinearBinarySearch())
                @test val ≈ sin(2π * 0.25) atol = 1.0e-3
            end

            @testset "Series: explicit policy" begin
                sitp = linear_interp(x, Series(y1, y2); search = BinarySearch())
                @test sitp.search_policy isa BinarySearch

                # Scalar call
                vals = sitp(0.25)
                @test vals[1] ≈ sin(2π * 0.25) atol = 1.0e-3

                # Vector call
                xq = [0.1, 0.3]
                result = sitp(xq)
                @test length(result) == 2
                @test length(result[1]) == 2
            end

            @testset "Series: call-site override" begin
                sitp = linear_interp(x, Series(y1, y2))  # AutoSearch
                @test sitp.search_policy isa AutoSearch

                # Override at call-site
                vals = sitp(0.25; search = BinarySearch())
                @test length(vals) == 2

                result = sitp([0.1, 0.3]; search = LinearBinarySearch())
                @test length(result) == 2
            end
        end

        # ========================================
        # ND vector calculus: gradient/hessian/laplacian resolve AutoSearch
        # ========================================
        @testset "gradient/hessian/laplacian resolve AutoSearch" begin
            x = collect(range(0.0, 1.0, 21))
            y = collect(range(0.0, 1.0, 21))
            data = [xi^2 + yi^2 for xi in x, yi in y]
            itp = cubic_interp((x, y), data)  # AutoSearch stored on each axis
            @test itp.searches[1] isa AutoSearch
            @test itp.searches[2] isa AutoSearch

            q = (0.5, 0.5)

            # gradient: ∇(x²+y²) = (2x, 2y)
            g = gradient(itp, q)
            @test all(isfinite, g)
            @test g[1] ≈ 2 * 0.5 atol = 0.05
            @test g[2] ≈ 2 * 0.5 atol = 0.05

            # gradient! (in-place)
            G = zeros(2)
            gradient!(G, itp, q)
            @test G ≈ collect(g)

            # hessian: H(x²+y²) = [[2,0],[0,2]]
            h = hessian(itp, q)
            @test all(isfinite, h)
            @test h[1, 1] ≈ 2.0 atol = 0.1
            @test h[2, 2] ≈ 2.0 atol = 0.1
            @test abs(h[1, 2]) < 0.05  # off-diagonal ≈ 0

            # laplacian: ∇²(x²+y²) = 4
            lap = laplacian(itp, q)
            @test isfinite(lap)
            @test lap ≈ 4.0 atol = 0.1
        end

        # ========================================
        # Unit Tests: _is_likely_monotone
        # ========================================
        @testset "_is_likely_monotone" begin
            # Ascending → true
            @test _is_likely_monotone(collect(1.0:100.0))
            @test _is_likely_monotone(collect(1.0:0.5:50.0))

            # Descending → true
            @test _is_likely_monotone(collect(100.0:-1.0:1.0))
            @test _is_likely_monotone(collect(50.0:-0.5:1.0))

            # Random → false (with overwhelming probability)
            rng_data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]
            @test !_is_likely_monotone(rng_data)

            # Too short (< K=8) → false
            @test !_is_likely_monotone([1.0, 2.0, 3.0])
            @test !_is_likely_monotone([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

            # Exactly K=8 elements → true if sorted
            @test _is_likely_monotone(collect(1.0:8.0))
            @test _is_likely_monotone(collect(8.0:-1.0:1.0))

            # Exactly K=8 elements → false if not sorted
            @test !_is_likely_monotone([1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0])

            # Flat (all equal) → trivially monotone
            @test _is_likely_monotone(fill(5.0, 10))

            # Monotone prefix but unsorted later → still true (only checks first K)
            v = collect(1.0:20.0)
            v[15] = 0.0  # break monotonicity beyond K=8
            @test _is_likely_monotone(v)

            # Equal-valued prefix followed by descending → true (non-increasing)
            @test _is_likely_monotone([0.0, 0.0, 0.0, -1.0, -1.0, -2.0, -2.0, -3.0])

            # Equal-valued prefix followed by ascending → true (non-decreasing)
            @test _is_likely_monotone([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0])

            # Equal prefix then direction reversal → false
            @test !_is_likely_monotone([0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 4.0, 2.0])

            # Ascending then descending → false
            @test !_is_likely_monotone([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        end

        # ========================================
        # Unit Tests: _resolve_search 3-arg (adaptive)
        # ========================================
        @testset "_resolve_search 3-arg adaptive" begin
            auto = AutoSearch()
            sorted = collect(1.0:100.0)
            random = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]

            @testset "AutoSearch + sorted + no hint → LinearBinarySearch" begin
                @test _resolve_search_policy(auto, sorted, nothing) isa LinearBinarySearch
            end

            @testset "AutoSearch + random + no hint → BinarySearch" begin
                @test _resolve_search_policy(auto, random, nothing) isa BinarySearch
            end

            @testset "AutoSearch + short vector + no hint → BinarySearch (too short for monotone check)" begin
                @test _resolve_search_policy(auto, [1.0, 2.0, 3.0], nothing) isa BinarySearch
            end

            @testset "AutoSearch + hint present → fallback to _resolve_search (LinearBinarySearch)" begin
                # When hint is present, generic fallback delegates to _resolve_search
                # _resolve_search_policy(AutoSearch(), vector) → LinearBinarySearch
                @test _resolve_search_policy(auto, random, Ref(1)) isa LinearBinarySearch
                @test _resolve_search_policy(auto, sorted, Ref(1)) isa LinearBinarySearch
            end

            @testset "Explicit policy passthrough" begin
                @test _resolve_search_policy(BinarySearch(), sorted, nothing) === BinarySearch()
                @test _resolve_search_policy(BinarySearch(), random, Ref(1)) === BinarySearch()
                @test _resolve_search_policy(LinearBinarySearch(), random, nothing) isa LinearBinarySearch
                @test _resolve_search_policy(LinearSearch(), sorted, nothing) === LinearSearch()
            end

            @testset "Scalar query → fallback to _resolve_search" begin
                # Scalar + no hint: AutoSearch → BinarySearch
                @test _resolve_search_policy(auto, 0.5, nothing) isa BinarySearch
                # Scalar + hint: AutoSearch → LB (hint implies locality intent)
                @test _resolve_search_policy(auto, 0.5, Ref(1)) isa LinearBinarySearch
            end
        end

        # ========================================
        # Correctness Regression: sorted vs random produce same results
        # ========================================
        @testset "Adaptive search correctness (sorted vs random same values)" begin
            x = collect(range(0.0, 10.0, 1001))
            y = sin.(x)
            itp = linear_interp(x, y)

            # Same query points, different orderings
            xq_values = [0.5, 1.3, 2.7, 4.1, 5.5, 6.8, 7.2, 8.9, 9.1, 3.3]
            xq_sorted = sort(xq_values)
            xq_random = xq_values  # unsorted

            vals_sorted = itp(xq_sorted)
            vals_random = itp(xq_random)

            # Results should match value-for-value (reorder sorted results to match random order)
            perm = sortperm(xq_values)
            for (i, p) in enumerate(perm)
                @test vals_sorted[i] ≈ vals_random[p]
            end

            # Also test oneshot API
            vals_sorted_os = linear_interp(x, y, xq_sorted)
            vals_random_os = linear_interp(x, y, xq_random)
            for (i, p) in enumerate(perm)
                @test vals_sorted_os[i] ≈ vals_random_os[p]
            end

            # Cubic interpolant
            itp_c = cubic_interp(x, y)
            vals_sorted_c = itp_c(xq_sorted)
            vals_random_c = itp_c(xq_random)
            for (i, p) in enumerate(perm)
                @test vals_sorted_c[i] ≈ vals_random_c[p]
            end
        end

        # ========================================
        # Zero-allocation: AutoSearch vector eval
        # ========================================
        @testset "AutoSearch vector eval zero-allocation" begin
            function _test_adaptive_alloc_sorted()
                x = collect(range(0.0, 10.0, 1001))
                y = sin.(x)
                itp = linear_interp(x, y)
                xq = collect(range(0.1, 9.9, 1000))
                out = Vector{Float64}(undef, 1000)

                # Warmup
                itp(out, xq)

                # Measure
                alloc = @allocated itp(out, xq)
                return alloc
            end

            function _test_adaptive_alloc_random()
                x = collect(range(0.0, 10.0, 1001))
                y = sin.(x)
                itp = linear_interp(x, y)
                xq = [mod(i * 7.3, 9.8) + 0.1 for i in 1:1000]  # deterministic pseudo-random
                out = Vector{Float64}(undef, 1000)

                # Warmup
                itp(out, xq)

                # Measure
                alloc = @allocated itp(out, xq)
                return alloc
            end

            @test _test_adaptive_alloc_sorted() == 0
            @test _test_adaptive_alloc_random() == 0
        end

    end  # @testset "AutoSearch Resolution"

    # ========================================
    # DirectSearch: Range Grid Short-Circuit
    # ========================================
    @testset "DirectSearch Range Short-Circuit" begin

        @testset "4-arg _resolve_search dispatch" begin
            x_range = 0.0:0.1:1.0
            x_vec = collect(x_range)

            # Range grid → DirectSearch (regardless of policy)
            @test _resolve_search_policy(x_range, 0.5, AutoSearch(), nothing) isa DirectSearch
            @test _resolve_search_policy(x_range, 0.5, BinarySearch(), nothing) isa DirectSearch
            @test _resolve_search_policy(x_range, 0.5, LinearBinarySearch(), nothing) isa DirectSearch
            @test _resolve_search_policy(x_range, [0.1, 0.5], AutoSearch(), Ref(1)) isa DirectSearch

            # Vector grid → delegates to 3-arg (NOT DirectSearch)
            @test !(_resolve_search_policy(x_vec, 0.5, AutoSearch(), nothing) isa DirectSearch)
            @test _resolve_search_policy(x_vec, 0.5, BinarySearch(), nothing) === BinarySearch()

        end

        @testset "_to_searcher(DirectSearch())" begin
            # NoHint variants — carries DirectSearch through
            @test _to_searcher(DirectSearch()) isa Searcher{DirectSearch, NoHint}
            @test _to_searcher(DirectSearch(), nothing) isa Searcher{DirectSearch, NoHint}

            # RefHint variant
            ref = Ref(5)
            s = _to_searcher(DirectSearch(), ref)
            @test s isa Searcher{DirectSearch, RefHint}
            @test s.hint.idx === ref
        end

        @testset "search_interval DirectSearch+NoHint correctness" begin
            x = 0.0:0.25:1.0
            s = _to_searcher(DirectSearch())
            result = search_interval(s, x, 0.3)
            @test result[1] == 2
            result2 = search_interval(s, x, 0.9)
            @test result2[1] == 4
        end

        @testset "search_interval DirectSearch+RefHint correctness" begin
            x = 0.0:0.25:1.0  # 5 points: [0.0, 0.25, 0.5, 0.75, 1.0]
            hint_ref = Ref(1)
            s = _to_searcher(DirectSearch(), hint_ref)

            # search_interval returns (idx_L, idx_R, xL, xR) tuple
            # Query in middle → should find interval 2 (0.25..0.5)
            result = search_interval(s, x, 0.3)
            @test result[1] == 2
            @test hint_ref[] == 2  # hint updated

            # Query near end → should find interval 4 (0.75..1.0)
            result2 = search_interval(s, x, 0.9)
            @test result2[1] == 4
            @test hint_ref[] == 4

            # Query at start → interval 1
            result3 = search_interval(s, x, 0.0)
            @test result3[1] == 1
            @test hint_ref[] == 1
        end

        @testset "Type inference: no Union leakage" begin
            x_range = 0.0:0.1:1.0
            # 4-arg resolve on Range → DirectSearch (concrete, not Union)
            @test @inferred(_resolve_search_policy(x_range, 0.5, AutoSearch(), nothing)) isa DirectSearch
            @test @inferred(_resolve_search_policy(x_range, [0.1], AutoSearch(), Ref(1))) isa DirectSearch

            # _to_searcher on DirectSearch → concrete Searcher types
            @test @inferred(_to_searcher(DirectSearch())) isa Searcher{DirectSearch, NoHint}
            @test @inferred(_to_searcher(DirectSearch(), nothing)) isa Searcher{DirectSearch, NoHint}
            @test @inferred(_to_searcher(DirectSearch(), Ref(1))) isa Searcher{DirectSearch, RefHint}
        end

    end  # @testset "DirectSearch Range Short-Circuit"

    # ========================================
    # GridIdx Short-Circuit Dispatch Tests
    # ========================================
    # GridIdx <: Real creates method ambiguity with every (ConcretePolicy × Real)
    # overload. These tests verify that every GridIdx disambiguation method:
    #   1. dispatches correctly (short-circuits search, returns correct interval)
    #   2. updates hint when RefHint is used
    #   3. clamps idx at right boundary (GridIdx(N) → cell N-1)
    #   4. @boundscheck triggers on out-of-range GridIdx

    @testset "GridIdx search_interval dispatch" begin
        # --- Setup: Vector grid ---
        x_vec = collect(range(0.0, 1.0; length = 11))   # 11 points, 10 cells

        # --- Setup: Range grid ---
        x_range = range(0.0, 1.0; length = 11)

        # Target: interior cell 5 → x[5]=0.4, x[6]=0.5
        k = 5

        # --------------------------------------------------------
        # 3-arg: search_interval(searcher, grid, GridIdx(k))
        # --------------------------------------------------------

        @testset "3-arg: BinarySearch + NoHint + Vector" begin
            s = Searcher{BinarySearch, NoHint}(NoHint())
            idx, _, lo, hi = @inferred search_interval(s, x_vec, GridIdx(k))
            @test idx == k
            @test lo ≈ x_vec[k]
            @test hi ≈ x_vec[k + 1]
        end

        @testset "3-arg: LinearSearch + RefHint + Vector" begin
            hint = RefHint()
            hint.idx[] = 1  # start far away
            s = Searcher{LinearSearch, RefHint}(hint)
            idx, _, lo, hi = @inferred search_interval(s, x_vec, GridIdx(k))
            @test idx == k
            @test lo ≈ x_vec[k]
            @test hi ≈ x_vec[k + 1]
            @test hint.idx[] == k  # hint updated
        end

        @testset "3-arg: LinearBinarySearch + RefHint + Vector" begin
            hint = RefHint()
            hint.idx[] = 1
            s = Searcher{LinearBinarySearch{8}, RefHint}(hint)
            idx, _, lo, hi = @inferred search_interval(s, x_vec, GridIdx(k))
            @test idx == k
            @test lo ≈ x_vec[k]
            @test hi ≈ x_vec[k + 1]
            @test hint.idx[] == k
        end

        @testset "3-arg: DirectSearch + NoHint + Range" begin
            s = Searcher{DirectSearch, NoHint}(NoHint())
            idx, _, lo, hi = @inferred search_interval(s, x_range, GridIdx(k))
            @test idx == k
            @test lo ≈ x_range[k]
            @test hi ≈ x_range[k + 1]
        end

        @testset "3-arg: DirectSearch + RefHint + Range" begin
            hint = RefHint()
            hint.idx[] = 1
            s = Searcher{DirectSearch, RefHint}(hint)
            idx, _, lo, hi = @inferred search_interval(s, x_range, GridIdx(k))
            @test idx == k
            @test lo ≈ x_range[k]
            @test hi ≈ x_range[k + 1]
            @test hint.idx[] == k
        end

        # --------------------------------------------------------
        # Right-boundary clamping: GridIdx(N) → cell N-1
        # --------------------------------------------------------

        @testset "right-boundary clamping" begin
            N = length(x_vec)
            s_nohint = Searcher{BinarySearch, NoHint}(NoHint())
            idx, _, lo, hi = search_interval(s_nohint, x_vec, GridIdx(N))
            @test idx == N - 1
            @test lo ≈ x_vec[N - 1]
            @test hi ≈ x_vec[N]

            # Same for Range grid
            s_direct = Searcher{DirectSearch, NoHint}(NoHint())
            idx_r, _, lo_r, hi_r = search_interval(s_direct, x_range, GridIdx(N))
            @test idx_r == N - 1
            @test lo_r ≈ x_range[N - 1]
            @test hi_r ≈ x_range[N]
        end

        # --------------------------------------------------------
        # @boundscheck: out-of-range GridIdx
        # --------------------------------------------------------

        @testset "bounds checking" begin
            # GridIdx(0) is rejected at construction (ArgumentError), so test idx > length(x)
            s = Searcher{BinarySearch, NoHint}(NoHint())
            @test_throws ArgumentError GridIdx(0)  # constructor guard
            @test_throws BoundsError search_interval(s, x_vec, GridIdx(length(x_vec) + 1))

            s_range = Searcher{DirectSearch, NoHint}(NoHint())
            @test_throws BoundsError search_interval(s_range, x_range, GridIdx(length(x_range) + 1))
        end
    end  # @testset "GridIdx search_interval dispatch"

end  # @testset "Search Module"

@testitem "GridIdx short-circuit survives ClampExtrap value eval" begin
    using FastInterpolations: GridIdx

    # Instrumented axis that counts element reads. A `GridIdx` query carries a
    # pre-resolved index, so its evaluation must touch the grid a number of
    # times that is CONSTANT in the grid length (the index short-circuit). A
    # coordinate search instead reads O(log n) extra elements — observable as a
    # read count that grows with grid size.
    mutable struct CountedAxis{T} <: AbstractVector{T}
        data::Vector{T}
        nread::Base.RefValue{Int}
    end
    CountedAxis(d::AbstractVector) = CountedAxis(collect(d), Ref(0))
    Base.size(v::CountedAxis) = size(v.data)
    Base.IndexStyle(::Type{<:CountedAxis}) = IndexLinear()
    Base.@propagate_inbounds function Base.getindex(v::CountedAxis, i::Int)
        v.nread[] += 1
        return v.data[i]
    end

    # Grid reads for one query of each kind on a length-`n` grid.
    function grid_reads(n)
        x = CountedAxis(range(0.0, 10.0; length = n))
        y = collect(sin.(x.data))
        k = (n + 1) ÷ 2                       # interior node
        # warm up compilation before measuring
        linear_interp(x, y, GridIdx(k); extrap = ClampExtrap())
        linear_interp(x, y, GridIdx(k); extrap = NoExtrap())
        linear_interp(x, y, 5.0; extrap = ClampExtrap())
        reads(call) = (x.nread[] = 0; call(); x.nread[])
        return (
            grididx_clamp = reads(() -> linear_interp(x, y, GridIdx(k); extrap = ClampExtrap())),
            grididx_noex = reads(() -> linear_interp(x, y, GridIdx(k); extrap = NoExtrap())),
            coord_clamp = reads(() -> linear_interp(x, y, 5.0; extrap = ClampExtrap())),
        )
    end

    small = grid_reads(33)
    large = grid_reads(1025)

    @testset "instrument detects a coordinate search" begin
        # Sanity: a real-coordinate ClampExtrap query searches → reads grow with n.
        @test large.coord_clamp > small.coord_clamp
    end

    @testset "GridIdx + ClampExtrap value short-circuits (zero search cost)" begin
        # Reads are constant in grid length ⇒ no coordinate search performed.
        # (A search would read O(log n) more on the larger grid — see coord_clamp.)
        @test large.grididx_clamp == small.grididx_clamp
        # ...and strictly cheaper than an actual coordinate search.
        @test large.grididx_clamp < large.coord_clamp
    end

    @testset "GridIdx + ClampExtrap value is bit-identical to the fast path" begin
        x = collect(range(0.0, 10.0; length = 1025))
        y = sin.(x)
        @test linear_interp(x, y, GridIdx(500); extrap = ClampExtrap()) ===
            linear_interp(x, y, GridIdx(500); extrap = NoExtrap())
        @test linear_interp(x, y, GridIdx(length(x)); extrap = ClampExtrap()) ===
            linear_interp(x, y, GridIdx(length(x)); extrap = NoExtrap())
    end
end
