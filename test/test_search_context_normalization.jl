using Test
using FastInterpolations
using FastInterpolations: search_interval, Searcher, BinarySearch, LinearSearch, LinearBinarySearch,
    AutoSearch, DirectSearch, NoHint, RefHint, DEFAULT_SEARCHER,
    _to_searcher, _resolve_search, _resolve_search_policy,
    _resolve_searcher_for_grid, ScalarSpacing, _create_spacing

@testset "Search Context Normalization" begin

    # Shared test fixtures
    x_vec = collect(range(0.0, 1.0, 101))
    x_range = range(0.0, 1.0, 101)
    xq_scalar = 0.5
    xq_vec = [0.1, 0.3, 0.5, 0.7, 0.9]
    xq_sorted = collect(range(0.05, 0.95, 20))
    xq_random = [0.8, 0.2, 0.6, 0.1, 0.9]

    # ========================================
    # 1. _resolve_searcher_for_grid equivalence
    # ========================================

    @testset "_resolve_searcher_for_grid (grid adaptation)" begin
        searchers = [
            Searcher{BinarySearch, NoHint}(NoHint()),
            Searcher{LinearBinarySearch{8}, RefHint}(RefHint()),
            Searcher{LinearSearch, RefHint}(RefHint()),
            Searcher{DirectSearch, NoHint}(NoHint()),
        ]

        for s in searchers
            @testset "$(typeof(s)) on Range" begin
                result = _resolve_searcher_for_grid(x_range, s)
                @test result isa Searcher{DirectSearch}
            end
            @testset "$(typeof(s)) on Vector" begin
                result = _resolve_searcher_for_grid(x_vec, s)
                @test typeof(result) === typeof(s)
            end
        end

        @testset "RefHint preservation on Range" begin
            hint_ref = Ref(42)
            s = Searcher{LinearBinarySearch{8}, RefHint}(RefHint(hint_ref))
            result = _resolve_searcher_for_grid(x_range, s)
            @test result isa Searcher{DirectSearch, RefHint}
            @test result.hint.idx === hint_ref
        end
    end

    # ========================================
    # 2. _resolve_search — end-to-end
    # ========================================

    @testset "_resolve_search end-to-end" begin

        @testset "Equivalence to 2-step chain" begin
            # The canonical invariant: _resolve_search must produce
            # the same Searcher type as the 2-step resolve_policy+to_searcher chain.
            for grid in (x_vec, x_range)
                for query in (xq_scalar, xq_sorted)
                    for policy in (AutoSearch(), BinarySearch(), LinearBinarySearch(), LinearSearch())
                        for hint in (nothing, Ref(1))
                            resolved = _resolve_search_policy(grid, query, policy, hint)
                            expected = _to_searcher(resolved, hint)
                            actual = _resolve_search(grid, query, policy, hint)
                            @test typeof(actual) === typeof(expected)
                        end
                    end
                end
            end
        end

        @testset "Range grid → DirectSearch (all policies)" begin
            for policy in (AutoSearch(), BinarySearch(), LinearBinarySearch(), LinearSearch())
                for hint in (nothing, Ref(1))
                    searcher = _resolve_search(x_range, xq_scalar, policy, hint)
                    @test searcher isa Searcher{DirectSearch}
                end
            end
        end

        @testset "Vector grid + explicit BinarySearch + no hint → BinarySearch" begin
            s = _resolve_search(x_vec, xq_scalar, BinarySearch(), nothing)
            @test s isa Searcher{BinarySearch, NoHint}
        end

        @testset "Vector grid + explicit BinarySearch + hint → Binary + RefHint" begin
            s = _resolve_search(x_vec, xq_scalar, BinarySearch(), Ref(1))
            @test s isa Searcher{BinarySearch, RefHint}
        end

        @testset "Vector grid + AutoSearch + scalar → BinarySearch" begin
            s = _resolve_search(x_vec, xq_scalar, AutoSearch(), nothing)
            @test s isa Searcher{BinarySearch, NoHint}
        end

        @testset "Vector grid + AutoSearch + sorted vec → LinearBinarySearch" begin
            s = _resolve_search(x_vec, xq_sorted, AutoSearch(), nothing)
            @test s isa Searcher{LinearBinarySearch{8}, RefHint}
        end
    end

    # ========================================
    # 3. Functional correctness — search results match
    # ========================================

    @testset "search_interval results match after resolution" begin
        spacing_vec = _create_spacing(x_vec)
        spacing_range = _create_spacing(x_range)

        test_points = [0.0, 0.15, 0.5, 0.85, 1.0]

        for xq in test_points
            # 2-step chain
            resolved_v = _resolve_search_policy(x_vec, xq, AutoSearch(), nothing)
            searcher_v = _to_searcher(resolved_v, nothing)
            result_old = search_interval(searcher_v, x_vec, spacing_vec, xq)

            # New context resolver
            searcher_new = _resolve_search(x_vec, xq, AutoSearch(), nothing)
            result_new = search_interval(searcher_new, x_vec, spacing_vec, xq)

            @test result_old == result_new

            # Range grid
            resolved_r = _resolve_search_policy(x_range, xq, AutoSearch(), nothing)
            searcher_r = _to_searcher(resolved_r, nothing)
            result_old_r = search_interval(searcher_r, x_range, spacing_range, xq)

            searcher_new_r = _resolve_search(x_range, xq, AutoSearch(), nothing)
            result_new_r = search_interval(searcher_new_r, x_range, spacing_range, xq)

            @test result_old_r == result_new_r
        end
    end

    # ========================================
    # 4. Type inference checks
    # ========================================

    @testset "Type inference" begin
        @testset "_resolve_search is inferrable" begin
            # Range: always DirectSearch (no Union)
            @test @inferred(_resolve_search(x_range, 0.5, BinarySearch(), nothing)) isa Searcher{DirectSearch, NoHint}
            @test @inferred(_resolve_search(x_range, 0.5, AutoSearch(), nothing)) isa Searcher{DirectSearch, NoHint}
            @test @inferred(_resolve_search(x_range, 0.5, LinearBinarySearch(), Ref(1))) isa Searcher{DirectSearch, RefHint}

            # Vector + explicit policy: concrete type
            @test @inferred(_resolve_search(x_vec, 0.5, BinarySearch(), nothing)) isa Searcher{BinarySearch, NoHint}
            @test @inferred(_resolve_search(x_vec, 0.5, LinearBinarySearch(), nothing)) isa Searcher{LinearBinarySearch{8}, RefHint}
            @test @inferred(_resolve_search(x_vec, 0.5, LinearSearch(), Ref(1))) isa Searcher{LinearSearch, RefHint}
        end

        @testset "_resolve_search_policy is inferrable" begin
            @test @inferred(_resolve_search_policy(x_range, 0.5, AutoSearch(), nothing)) isa DirectSearch
            @test @inferred(_resolve_search_policy(x_range, [0.1], AutoSearch(), Ref(1))) isa DirectSearch
        end
    end

end
