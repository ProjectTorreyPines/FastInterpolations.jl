@testitem "ND Per-Axis Adaptive AutoSearch Resolution" begin
    using Random: MersenneTwister
    using FastInterpolations: _resolve_search_nd, _resolve_search_policy,
        _is_axis_likely_monotone, _check_mono_nd, _query_extract, _query_length,
        _resolve_oneshot_search_nd,
        AutoSearch, BinarySearch, LinearBinarySearch

    # ========================================
    # 4-arg _resolve_search_nd: per-axis monotonicity check
    # ========================================

    @testset "Scalar AutoSearch + SoA + no hint → per-axis check" begin
        sorted = collect(1.0:100.0)
        random = [
            3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
            7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]

        @testset "both sorted → both LinearBinarySearch" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), (sorted, sorted), nothing)
            @test result[1] isa LinearBinarySearch
            @test result[2] isa LinearBinarySearch
        end

        @testset "both random → both BinarySearch" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), (random, random), nothing)
            @test result[1] isa BinarySearch
            @test result[2] isa BinarySearch
        end

        @testset "axis 1 sorted, axis 2 random → mixed policies" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), (sorted, random), nothing)
            @test result[1] isa LinearBinarySearch
            @test result[2] isa BinarySearch
        end

        @testset "axis 1 random, axis 2 sorted → mixed policies (reversed)" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), (random, sorted), nothing)
            @test result[1] isa BinarySearch
            @test result[2] isa LinearBinarySearch
        end

        @testset "3D: mixed pattern" begin
            descending = collect(100.0:-1.0:1.0)
            result = _resolve_search_nd(AutoSearch(), Val(3), (sorted, random, descending), nothing)
            @test result[1] isa LinearBinarySearch
            @test result[2] isa BinarySearch
            @test result[3] isa LinearBinarySearch
        end
    end

    @testset "Tuple of AutoSearch + SoA + no hint → per-axis check" begin
        sorted = collect(1.0:100.0)
        random = [
            3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
            7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]

        @testset "(AutoSearch, AutoSearch) per-axis independent" begin
            result = _resolve_search_nd(
                (AutoSearch(), AutoSearch()), Val(2),
                (sorted, random), nothing
            )
            @test result[1] isa LinearBinarySearch
            @test result[2] isa BinarySearch
        end
    end

    @testset "Mixed tuple (AutoSearch + explicit) + SoA + no hint" begin
        sorted = collect(1.0:100.0)
        random = [
            3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
            7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]

        @testset "(AutoSearch, BinarySearch) → axis 1 checked, axis 2 passthrough" begin
            result = _resolve_search_nd(
                (AutoSearch(), BinarySearch()), Val(2),
                (sorted, sorted), nothing
            )
            @test result[1] isa LinearBinarySearch   # AutoSearch resolved via monotonicity
            @test result[2] isa BinarySearch         # explicit BinarySearch unchanged
        end

        @testset "(BinarySearch, AutoSearch) → axis 1 passthrough, axis 2 checked" begin
            result = _resolve_search_nd(
                (BinarySearch(), AutoSearch()), Val(2),
                (random, sorted), nothing
            )
            @test result[1] isa BinarySearch         # explicit BinarySearch unchanged
            @test result[2] isa LinearBinarySearch   # AutoSearch resolved via monotonicity
        end
    end

    @testset "hint present → fallback (no monotonicity check)" begin
        sorted = collect(1.0:100.0)
        hints = (Ref(1), Ref(1))

        # With hint, falls through to 3-arg → type-based → LinearBinarySearch for vectors
        result = _resolve_search_nd(AutoSearch(), Val(2), (sorted, sorted), hints)
        @test result[1] isa LinearBinarySearch
        @test result[2] isa LinearBinarySearch
    end

    # ========================================
    # Oneshot batch resolution — `_resolve_oneshot_search_nd`
    # ========================================
    # Single resolution point hoisted outside the per-query loop. Verifies
    # the (policies, hints) shape across (sort/random × hint/no-hint) combos.

    @testset "Oneshot ND batch resolution — `_resolve_oneshot_search_nd`" begin
        sorted = collect(1.0:50.0)
        random = [
            3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
            7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]

        @testset "no-hint, sort × sort → both LB + fresh persistent Refs" begin
            policies, hints = _resolve_oneshot_search_nd(
                AutoSearch(), (sorted, sorted), nothing, Val(2)
            )
            @test policies[1] isa LinearBinarySearch
            @test policies[2] isa LinearBinarySearch
            @test hints[1] isa Base.RefValue{Int}
            @test hints[2] isa Base.RefValue{Int}
            @test hints[1][] == 1 && hints[2][] == 1  # fresh start at idx=1
        end

        @testset "no-hint, rand × rand → both Binary (no walk overhead)" begin
            policies, _hints = _resolve_oneshot_search_nd(
                AutoSearch(), (random, random), nothing, Val(2)
            )
            @test policies[1] isa BinarySearch
            @test policies[2] isa BinarySearch
        end

        @testset "no-hint, mixed → per-axis (LB, Binary)" begin
            policies, _hints = _resolve_oneshot_search_nd(
                AutoSearch(), (sorted, random), nothing, Val(2)
            )
            @test policies[1] isa LinearBinarySearch
            @test policies[2] isa BinarySearch
        end

        @testset "user-supplied hint → preserved, random axis still LB" begin
            # With explicit hint, "caller wants locality" → both axes LB.
            user_hint = (Ref(7), Ref(13))
            policies, hints = _resolve_oneshot_search_nd(
                AutoSearch(), (random, random), user_hint, Val(2)
            )
            @test policies[1] isa LinearBinarySearch
            @test policies[2] isa LinearBinarySearch
            @test hints === user_hint
            @test hints[1][] == 7 && hints[2][] == 13
        end

        @testset "explicit BinarySearch passthrough" begin
            policies, _hints = _resolve_oneshot_search_nd(
                BinarySearch(), (sorted, sorted), nothing, Val(2)
            )
            @test policies[1] isa BinarySearch
            @test policies[2] isa BinarySearch
        end

        @testset "3D mixed: (sort, rand, descending) → (LB, Binary, LB)" begin
            descending = collect(50.0:-1.0:1.0)
            policies, hints = _resolve_oneshot_search_nd(
                AutoSearch(), (sorted, random, descending), nothing, Val(3)
            )
            @test policies[1] isa LinearBinarySearch
            @test policies[2] isa BinarySearch
            @test policies[3] isa LinearBinarySearch
            @test length(hints) == 3
        end
    end

    @testset "Behavioral equivalence: AutoSearch (no hint) ≡ LB + persistent user hint" begin
        # AutoSearch on sorted queries must produce bit-identical output to an
        # explicit `search=LinearBinarySearch()` + user-supplied persistent Ref.
        xs = collect(range(0.0, 10.0, 51))
        ys = collect(range(0.0, 10.0, 51))
        data = [Float64(x + y) for x in xs, y in ys]

        sorted_xq = collect(0.1:0.1:9.9)
        sorted_yq = collect(0.1:0.1:9.9)
        nq = length(sorted_xq)
        out_auto = Vector{Float64}(undef, nq)
        out_lb_hint = Vector{Float64}(undef, nq)

        linear_interp!(out_auto, (xs, ys), data, (sorted_xq, sorted_yq))
        linear_interp!(
            out_lb_hint, (xs, ys), data, (sorted_xq, sorted_yq);
            search = LinearBinarySearch(), hint = (Ref(1), Ref(1))
        )
        @test out_auto == out_lb_hint   # bit-exact: same algorithm, same hints semantics
    end

    @testset "Behavioral equivalence: random AutoSearch ≡ explicit BinarySearch" begin
        # Random queries must produce bit-identical output to explicit BinarySearch.
        xs = collect(range(0.0, 10.0, 51))
        ys = collect(range(0.0, 10.0, 51))
        data = [Float64(x + y) for x in xs, y in ys]

        rng = MersenneTwister(0x1234abcd)
        random_xq = rand(rng, 100) .* 9.9
        random_yq = rand(rng, 100) .* 9.9
        nq = length(random_xq)
        out_auto = Vector{Float64}(undef, nq)
        out_binary = Vector{Float64}(undef, nq)

        linear_interp!(out_auto, (xs, ys), data, (random_xq, random_yq))
        linear_interp!(out_binary, (xs, ys), data, (random_xq, random_yq); search = BinarySearch())
        @test out_auto == out_binary
    end

    # ========================================
    # Oneshot SoA: verify adaptive resolution through public API
    # ========================================
    # These tests verify correctness (values) to ensure the search policy
    # changes don't break results. Both BinarySearch and LinearBinarySearch produce
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
            @test output[i] ≈ sorted_xq[i] + sorted_yq[i] atol = 1.0e-12
        end
    end

    @testset "Oneshot SoA correctness with random queries" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        # Random (non-monotone) queries — AutoSearch should pick BinarySearch
        random_xq = [3.2, 1.5, 7.8, 0.3, 9.1, 4.6, 6.0, 2.4, 8.7, 5.5]
        random_yq = [8.1, 2.3, 5.7, 9.0, 0.5, 6.8, 3.4, 7.2, 1.1, 4.9]
        n = length(random_xq)
        output = Vector{Float64}(undef, n)

        linear_interp!(output, (xs, ys), data, (random_xq, random_yq))
        for i in 1:n
            @test output[i] ≈ random_xq[i] + random_yq[i] atol = 1.0e-12
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
            @test output[i] ≈ sorted_xq[i] + random_yq[i] atol = 1.0e-12
        end
    end

    # ========================================
    # Per-Axis _check_mono_nd (SoA)
    # ========================================
    # Returns NTuple{N, Bool} — per-axis monotonicity flags for AutoSearch axes.
    # Replaces the old _resolve_search_nd_uniform (all-or-nothing).

    @testset "Per-axis: _check_mono_nd (SoA)" begin
        sorted = collect(1.0:100.0)
        random = [
            3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
            7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]

        @testset "both sorted → (true, true)" begin
            mono = _check_mono_nd((AutoSearch(), AutoSearch()), (sorted, sorted))
            @test mono == (true, true)
        end

        @testset "both random → (false, false)" begin
            mono = _check_mono_nd((AutoSearch(), AutoSearch()), (random, random))
            @test mono == (false, false)
        end

        @testset "mixed sorted/random → PER-AXIS (true, false)" begin
            mono = _check_mono_nd((AutoSearch(), AutoSearch()), (sorted, random))
            @test mono == (true, false)
        end

        @testset "3D: all sorted → (true, true, true)" begin
            descending = collect(100.0:-1.0:1.0)
            mono = _check_mono_nd((AutoSearch(), AutoSearch(), AutoSearch()), (sorted, sorted, descending))
            @test mono == (true, true, true)
        end

        @testset "3D: one random → per-axis (true, false, true)" begin
            descending = collect(100.0:-1.0:1.0)
            mono = _check_mono_nd((AutoSearch(), AutoSearch(), AutoSearch()), (sorted, random, descending))
            @test mono == (true, false, true)
        end

        @testset "explicit policies → true (flag ignored)" begin
            mono = _check_mono_nd((BinarySearch(), AutoSearch()), (sorted, sorted))
            @test mono[1] == true   # BinarySearch: flag always true (ignored by _search_axis_adaptive)
            @test mono[2] == true   # AutoSearch: sorted → true
        end
    end

    # ========================================
    # Oneshot Hint Plumbing: correctness + state updates
    # ========================================

    @testset "Oneshot scalar with hints — correctness" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        # Without hint
        val_no_hint = linear_interp((xs, ys), data, (3.5, 4.5))

        # With hint
        hints = (Ref(1), Ref(1))
        val_with_hint = linear_interp((xs, ys), data, (3.5, 4.5); hint = hints)

        @test val_no_hint ≈ val_with_hint
        @test val_no_hint ≈ 8.0 atol = 1.0e-12
    end

    @testset "Oneshot SoA with hints — correctness" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        xqs = collect(0.5:0.5:9.5)
        yqs = collect(0.5:0.5:9.5)
        n = length(xqs)

        # Without hint
        out1 = Vector{Float64}(undef, n)
        linear_interp!(out1, (xs, ys), data, (xqs, yqs))

        # With hint
        hints = (Ref(1), Ref(1))
        out2 = Vector{Float64}(undef, n)
        linear_interp!(out2, (xs, ys), data, (xqs, yqs); hint = hints)

        @test out1 ≈ out2
    end

    @testset "Oneshot AoS with hints — correctness" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        points = [(0.5, 0.5), (3.5, 4.5), (7.5, 8.5)]
        n = length(points)

        out1 = Vector{Float64}(undef, n)
        linear_interp!(out1, (xs, ys), data, points)

        hints = (Ref(1), Ref(1))
        out2 = Vector{Float64}(undef, n)
        linear_interp!(out2, (xs, ys), data, points; hint = hints)

        @test out1 ≈ out2
    end

    @testset "Oneshot constant with hints — correctness" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(floor(x) + floor(y)) for x in xs, y in ys]

        val_no = constant_interp((xs, ys), data, (3.5, 4.5))
        hints = (Ref(1), Ref(1))
        val_yes = constant_interp((xs, ys), data, (3.5, 4.5); hint = hints)
        @test val_no == val_yes
    end

    @testset "Oneshot cubic with hints — correctness" begin
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]

        val_no = cubic_interp((xs, ys), data, (3.5, 4.5))
        hints = (Ref(1), Ref(1))
        val_yes = cubic_interp((xs, ys), data, (3.5, 4.5); hint = hints)
        @test val_no ≈ val_yes atol = 1.0e-12
    end

    @testset "Oneshot quadratic with hints — correctness" begin
        xs = collect(range(0.0, 10.0, 21))
        ys = collect(range(0.0, 10.0, 21))
        data = [sin(x) * cos(y) for x in xs, y in ys]

        val_no = quadratic_interp((xs, ys), data, (3.5, 4.5))
        hints = (Ref(1), Ref(1))
        val_yes = quadratic_interp((xs, ys), data, (3.5, 4.5); hint = hints)
        @test val_no ≈ val_yes atol = 1.0e-12
    end

    # ========================================
    # Allocation Tests: oneshot SoA with function barrier
    # ========================================

    # True function barriers: setup + warmup + @allocated all in one function.
    # This avoids @testset try/catch type-instability artifacts.

    function _alloc_test_linear_soa_sorted()
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        xqs = collect(0.5:0.5:9.5)
        yqs = collect(0.5:0.5:9.5)
        out = Vector{Float64}(undef, length(xqs))
        linear_interp!(out, (xs, ys), data, (xqs, yqs))
        linear_interp!(out, (xs, ys), data, (xqs, yqs))
        @allocated linear_interp!(out, (xs, ys), data, (xqs, yqs))
    end

    function _alloc_test_linear_soa_random()
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        xqs = [3.2, 1.5, 7.8, 0.3, 9.1, 4.6, 6.0, 2.4, 8.7, 5.5]
        yqs = [8.1, 2.3, 5.7, 9.0, 0.5, 6.8, 3.4, 7.2, 1.1, 4.9]
        out = Vector{Float64}(undef, length(xqs))
        linear_interp!(out, (xs, ys), data, (xqs, yqs))
        linear_interp!(out, (xs, ys), data, (xqs, yqs))
        @allocated linear_interp!(out, (xs, ys), data, (xqs, yqs))
    end

    function _alloc_test_linear_soa_with_hint()
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        xqs = collect(0.5:0.5:9.5)
        yqs = collect(0.5:0.5:9.5)
        out = Vector{Float64}(undef, length(xqs))
        hints = (Ref(1), Ref(1))
        linear_interp!(out, (xs, ys), data, (xqs, yqs); hint = hints)
        linear_interp!(out, (xs, ys), data, (xqs, yqs); hint = hints)
        @allocated linear_interp!(out, (xs, ys), data, (xqs, yqs); hint = hints)
    end

    function _alloc_test_constant_soa_sorted()
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]
        xqs = collect(0.5:0.5:9.5)
        yqs = collect(0.5:0.5:9.5)
        out = Vector{Float64}(undef, length(xqs))
        constant_interp!(out, (xs, ys), data, (xqs, yqs))
        constant_interp!(out, (xs, ys), data, (xqs, yqs))
        @allocated constant_interp!(out, (xs, ys), data, (xqs, yqs))
    end

    @testset "Zero-alloc: oneshot SoA in-place (adaptive search)" begin
        @testset "linear sorted (→ LB)" begin
            @test _alloc_test_linear_soa_sorted() == 0
        end
        @testset "linear random (→ BinarySearch)" begin
            @test _alloc_test_linear_soa_random() == 0
        end
        @testset "linear with hint (→ LB, type-based)" begin
            @test _alloc_test_linear_soa_with_hint() == 0
        end
        @testset "constant sorted (→ LB)" begin
            @test _alloc_test_constant_soa_sorted() == 0
        end
    end

    # ========================================
    # Protocol-based monotonicity: _is_axis_likely_monotone
    # ========================================
    # Tests _is_axis_likely_monotone which uses _query_extract to check
    # per-axis monotonicity for ANY query type (AoS, custom, etc.).

    @testset "_is_axis_likely_monotone (protocol-based)" begin
        sorted_aos = [(Float64(i), Float64(i)) for i in 1:20]
        random_aos = [
            (3.0, 8.0), (1.0, 2.0), (4.0, 5.0), (1.0, 9.0),
            (5.0, 0.0), (9.0, 6.0), (2.0, 3.0), (6.0, 7.0),
            (5.0, 1.0), (3.0, 4.0), (7.0, 8.0), (8.0, 10.0),
            (10.0, 11.0), (11.0, 12.0), (12.0, 13.0), (13.0, 14.0),
            (14.0, 15.0), (15.0, 16.0), (16.0, 17.0), (17.0, 18.0),
        ]

        @testset "sorted AoS → true for both axes" begin
            @test _is_axis_likely_monotone(sorted_aos, 1, Val(2)) == true
            @test _is_axis_likely_monotone(sorted_aos, 2, Val(2)) == true
        end

        @testset "random AoS → false for both axes" begin
            @test _is_axis_likely_monotone(random_aos, 1, Val(2)) == false
            @test _is_axis_likely_monotone(random_aos, 2, Val(2)) == false
        end

        @testset "mixed: axis 1 sorted, axis 2 random" begin
            mixed = [(Float64(i), random_aos[i][2]) for i in 1:20]
            @test _is_axis_likely_monotone(mixed, 1, Val(2)) == true
            @test _is_axis_likely_monotone(mixed, 2, Val(2)) == false
        end

        @testset "too few points → false" begin
            short = [(1.0, 2.0), (3.0, 4.0)]
            @test _is_axis_likely_monotone(short, 1, Val(2)) == false
        end

        @testset "descending → true (monotone, not just ascending)" begin
            desc = [(20.0 - i, Float64(i)) for i in 1:20]
            @test _is_axis_likely_monotone(desc, 1, Val(2)) == true
        end
    end

    # ========================================
    # Generic (AoS) _resolve_search_nd: per-axis adaptive
    # ========================================

    @testset "AoS AutoSearch: _resolve_search_nd per-axis adaptive" begin
        sorted_aos = [(Float64(i), Float64(i)) for i in 1:20]
        random_aos = [
            (3.0, 8.0), (1.0, 2.0), (4.0, 5.0), (1.0, 9.0),
            (5.0, 0.0), (9.0, 6.0), (2.0, 3.0), (6.0, 7.0),
            (5.0, 1.0), (3.0, 4.0), (7.0, 8.0), (8.0, 10.0),
            (10.0, 11.0), (11.0, 12.0), (12.0, 13.0), (13.0, 14.0),
            (14.0, 15.0), (15.0, 16.0), (16.0, 17.0), (17.0, 18.0),
        ]

        @testset "both sorted → both LinearBinarySearch" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), sorted_aos, nothing)
            @test result[1] isa LinearBinarySearch
            @test result[2] isa LinearBinarySearch
        end

        @testset "both random → both BinarySearch" begin
            result = _resolve_search_nd(AutoSearch(), Val(2), random_aos, nothing)
            @test result[1] isa BinarySearch
            @test result[2] isa BinarySearch
        end

        @testset "mixed: axis 1 sorted, axis 2 random" begin
            mixed = [(Float64(i), random_aos[i][2]) for i in 1:20]
            result = _resolve_search_nd(AutoSearch(), Val(2), mixed, nothing)
            @test result[1] isa LinearBinarySearch
            @test result[2] isa BinarySearch
        end

        @testset "explicit policy passthrough" begin
            result = _resolve_search_nd((BinarySearch(), AutoSearch()), Val(2), sorted_aos, nothing)
            @test result[1] isa BinarySearch
            @test result[2] isa LinearBinarySearch
        end
    end

    # ========================================
    # Per-Axis _check_mono_nd (AoS)
    # ========================================

    @testset "AoS AutoSearch: _check_mono_nd per-axis" begin
        sorted_aos = [(Float64(i), Float64(i)) for i in 1:20]
        random_aos = [
            (3.0, 8.0), (1.0, 2.0), (4.0, 5.0), (1.0, 9.0),
            (5.0, 0.0), (9.0, 6.0), (2.0, 3.0), (6.0, 7.0),
            (5.0, 1.0), (3.0, 4.0), (7.0, 8.0), (8.0, 10.0),
            (10.0, 11.0), (11.0, 12.0), (12.0, 13.0), (13.0, 14.0),
            (14.0, 15.0), (15.0, 16.0), (16.0, 17.0), (17.0, 18.0),
        ]

        @testset "both sorted → (true, true)" begin
            mono = _check_mono_nd((AutoSearch(), AutoSearch()), sorted_aos)
            @test mono == (true, true)
        end

        @testset "both random → (false, false)" begin
            mono = _check_mono_nd((AutoSearch(), AutoSearch()), random_aos)
            @test mono == (false, false)
        end

        @testset "mixed → PER-AXIS (true, false)" begin
            mixed = [(Float64(i), random_aos[i][2]) for i in 1:20]
            mono = _check_mono_nd((AutoSearch(), AutoSearch()), mixed)
            @test mono == (true, false)
        end

        @testset "explicit policy → true (flag ignored)" begin
            mono = _check_mono_nd((BinarySearch(), AutoSearch()), sorted_aos)
            @test mono[1] == true   # BinarySearch: always true
            @test mono[2] == true   # AutoSearch: sorted → true
        end
    end

    # ========================================
    # AoS correctness: oneshot with sorted AoS queries
    # ========================================

    @testset "Oneshot AoS correctness with sorted queries" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        sorted_points = [(Float64(i) * 0.5, Float64(i) * 0.5) for i in 1:19]
        n = length(sorted_points)
        output = Vector{Float64}(undef, n)

        linear_interp!(output, (xs, ys), data, sorted_points)
        for i in 1:n
            @test output[i] ≈ sorted_points[i][1] + sorted_points[i][2] atol = 1.0e-12
        end
    end

    @testset "Oneshot AoS correctness with random queries" begin
        xs = 0.0:1.0:10.0
        ys = 0.0:1.0:10.0
        data = [Float64(x + y) for x in xs, y in ys]

        random_points = [
            (3.2, 8.1), (1.5, 2.3), (7.8, 5.7), (0.3, 9.0),
            (9.1, 0.5), (4.6, 6.8), (6.0, 3.4), (2.4, 7.2),
            (8.7, 1.1), (5.5, 4.9),
        ]
        n = length(random_points)
        output = Vector{Float64}(undef, n)

        linear_interp!(output, (xs, ys), data, random_points)
        for i in 1:n
            @test output[i] ≈ random_points[i][1] + random_points[i][2] atol = 1.0e-12
        end
    end
end
