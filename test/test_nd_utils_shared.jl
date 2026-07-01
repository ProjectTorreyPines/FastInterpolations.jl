# ========================================
# Tests for Shared ND Utilities
# ========================================
#
# Phase 1 of ND Constant/Linear implementation.
# These utilities are shared across all ND interpolation methods.
#
# Tests follow TDD protocol:
# - 🔴 RED: Tests written first, expected to fail initially
# - 🟢 GREEN: Minimal implementation to pass
# - 🔵 REFACTOR: Cleanup while staying green

@testitem "Shared ND Utilities" begin
    # Access internal functions for testing
    import FastInterpolations: _resolve_extrap, _resolve_search_nd, _resolve_bcs_nd,
        _resolve_side_nd, _validate_nd_grids,
        _promote_grid_eltype, _convert_grids_typed,
        _check_mode_periodic_compat, _check_modes_periodic_compat,
        _mode_to_modes_with_periodic, _modes_to_modes_with_periodic

    # ========================================
    # _resolve_extrap
    # ========================================
    # NOTE: Old 2-arg _resolve_extrap(extrap, Val(N)) tests removed.
    # Symbol-based extrap was removed in v0.3.0.
    # The 3-arg form _resolve_extrap(extrap, bcs, Val(N)) is tested below
    # in "Typed Extrap resolution".

    # ========================================
    # _resolve_search_nd
    # ========================================
    @testset "_resolve_search_nd" begin
        @testset "broadcast single policy to N-tuple" begin
            result = _resolve_search_nd(BinarySearch(), Val(3))
            @test length(result) == 3
            @test all(s -> s isa BinarySearch, result)

            result = _resolve_search_nd(LinearBinarySearch(), Val(2))
            @test length(result) == 2
            @test all(s -> s isa LinearBinarySearch, result)
        end

        @testset "passthrough matching tuple" begin
            policies = (BinarySearch(), LinearBinarySearch(), LinearSearch())
            result = _resolve_search_nd(policies, Val(3))
            @test result === policies
        end

        @testset "reject wrong-length tuple" begin
            @test_throws ArgumentError _resolve_search_nd((BinarySearch(), LinearBinarySearch()), Val(3))
            @test_throws ArgumentError _resolve_search_nd((BinarySearch(), LinearBinarySearch(), LinearSearch(), LinearBinarySearch(linear_window = 0)), Val(3))
        end
    end

    # ========================================
    # _resolve_bcs_nd
    # ========================================
    @testset "_resolve_bcs_nd" begin
        @testset "broadcast single BC to N-tuple" begin
            result = _resolve_bcs_nd(ZeroCurvBC(), Val(3))
            @test length(result) == 3
            @test all(bc -> bc isa ZeroCurvBC, result)

            result = _resolve_bcs_nd(PolyFit{3}(), Val(2))
            @test length(result) == 2
            @test all(bc -> bc isa PolyFit{3}, result)
        end

        @testset "passthrough matching tuple" begin
            bcs = (ZeroCurvBC(), PolyFit{3}(), ZeroSlopeBC())
            result = _resolve_bcs_nd(bcs, Val(3))
            @test result === bcs
        end

        @testset "reject wrong-length tuple" begin
            @test_throws ArgumentError _resolve_bcs_nd((ZeroCurvBC(), PolyFit{3}()), Val(3))
            @test_throws ArgumentError _resolve_bcs_nd((ZeroCurvBC(),), Val(2))
        end
    end

    # ========================================
    # _resolve_side_nd (NEW - for ConstantInterpolantND)
    # ========================================
    @testset "_resolve_side_nd" begin
        @testset "broadcast single AbstractSide to N-tuple" begin
            result = _resolve_side_nd(NearestSide(), Val(3))
            @test result === (NearestSide(), NearestSide(), NearestSide())

            result = _resolve_side_nd(LeftSide(), Val(2))
            @test result === (LeftSide(), LeftSide())

            result = _resolve_side_nd(RightSide(), Val(4))
            @test result === (RightSide(), RightSide(), RightSide(), RightSide())
        end

        @testset "passthrough matching tuple" begin
            result = _resolve_side_nd((NearestSide(), LeftSide(), RightSide()), Val(3))
            @test result === (NearestSide(), LeftSide(), RightSide())
        end

        @testset "reject wrong-length tuple" begin
            @test_throws ArgumentError _resolve_side_nd((NearestSide(), LeftSide()), Val(3))
            @test_throws ArgumentError _resolve_side_nd((NearestSide(), LeftSide(), RightSide(), NearestSide()), Val(3))
        end

        @testset "reject non-AbstractSide via MethodError" begin
            @test_throws MethodError _resolve_side_nd(:invalid, Val(2))
            @test_throws MethodError _resolve_side_nd(:nearest, Val(2))
            @test_throws MethodError _resolve_side_nd(:center, Val(2))
        end
    end

    # ========================================
    # DerivOp constructors
    # ========================================
    @testset "DerivOp constructors" begin
        @testset "1D: DerivOp(n) returns singleton" begin
            @test DerivOp(0) === DerivOp{0}()
            @test DerivOp(1) === DerivOp{1}()
            @test DerivOp(2) === DerivOp{2}()
            @test DerivOp(3) === DerivOp{3}()
        end

        @testset "ND: DerivOp(n1, n2, ...) returns tuple" begin
            result = DerivOp(1, 0)
            @test result == (DerivOp{1}(), DerivOp{0}())
            @test result[1] isa EvalDeriv1
            @test result[2] isa EvalValue

            result = DerivOp(1, 1)
            @test all(op -> op isa EvalDeriv1, result)

            result = DerivOp(0, 1, 0)
            @test result == (DerivOp{0}(), DerivOp{1}(), DerivOp{0}())
        end

        @testset "backward-compat aliases" begin
            @test DerivOp{0}() isa EvalValue
            @test DerivOp{1}() isa EvalDeriv1
            @test DerivOp{2}() isa EvalDeriv2
            @test DerivOp{3}() isa EvalDeriv3
            @test EvalValue() === DerivOp{0}()
            @test EvalDeriv1() === DerivOp{1}()
        end

        @testset "deriv_order" begin
            @test deriv_order(DerivOp{0}()) == 0
            @test deriv_order(DerivOp{1}()) == 1
            @test deriv_order(DerivOp{2}()) == 2
            @test deriv_order(DerivOp{3}()) == 3
        end
    end

    # ========================================
    # _validate_nd_grids
    # ========================================
    @testset "_validate_nd_grids" begin
        @testset "pass valid grids" begin
            x = range(0, 1, 10)
            y = range(0, 1, 20)
            data = rand(10, 20)
            @test _validate_nd_grids((x, y), data) === nothing
        end

        @testset "pass with mixed grid types (Range + Vector)" begin
            x = range(0, 1, 10)  # Range
            y = [0.0, 0.5, 1.0, 1.5, 2.0]  # Vector
            data = rand(10, 5)
            @test _validate_nd_grids((x, y), data) === nothing
        end

        @testset "reject dimension mismatch" begin
            x = range(0, 1, 10)
            y = range(0, 1, 20)
            data_wrong = rand(10, 15)  # y has 20 points but data dim 2 has 15
            @test_throws DimensionMismatch _validate_nd_grids((x, y), data_wrong)

            data_wrong2 = rand(8, 20)  # x has 10 points but data dim 1 has 8
            @test_throws DimensionMismatch _validate_nd_grids((x, y), data_wrong2)
        end

        @testset "reject grid with < 2 points" begin
            x = [0.0]  # Only 1 point
            y = range(0, 1, 10)
            data = rand(1, 10)
            @test_throws ArgumentError _validate_nd_grids((x, y), data)
        end

        @testset "3D validation" begin
            x = range(0, 1, 5)
            y = range(0, 1, 10)
            z = range(0, 1, 15)
            data = rand(5, 10, 15)
            @test _validate_nd_grids((x, y, z), data) === nothing

            # Wrong z dimension
            data_wrong = rand(5, 10, 12)
            @test_throws DimensionMismatch _validate_nd_grids((x, y, z), data_wrong)
        end
    end

    # ========================================
    # Grid Type Helpers
    # ========================================
    @testset "Grid type helpers" begin
        @testset "_promote_grid_eltype" begin
            # Homogeneous Float64
            grids_f64 = (range(0.0, 1.0, 10), [0.0, 0.5, 1.0])
            @test _promote_grid_eltype(grids_f64) === Float64

            # Mixed Float32 and Float64 → Float64
            grids_mixed = (range(0.0f0, 1.0f0, 10), range(0.0, 1.0, 5))
            @test _promote_grid_eltype(grids_mixed) === Float64

            # Homogeneous Float32
            grids_f32 = (range(0.0f0, 1.0f0, 10), [0.0f0, 0.5f0, 1.0f0])
            @test _promote_grid_eltype(grids_f32) === Float32
        end

    end

    # ========================================
    # Typed Extrap Resolution (3-arg form)
    # ========================================
    @testset "Typed Extrap resolution" begin
        @testset "single mode → Mode tuple, no BCs" begin
            @test _resolve_extrap(NoExtrap(), nothing, Val(2), Float64) === (NoExtrap(), NoExtrap())
            @test _resolve_extrap(ClampExtrap(), nothing, Val(3), Float64) === (ClampExtrap(), ClampExtrap(), ClampExtrap())
            @test _resolve_extrap(ExtendExtrap(), nothing, Val(1), Float64) === (ExtendExtrap(),)
            @test _resolve_extrap(WrapExtrap(), nothing, Val(2), Float64) === (WrapExtrap(), WrapExtrap())
        end

        @testset "mode tuple → Mode tuple, no BCs" begin
            @test _resolve_extrap((NoExtrap(), WrapExtrap()), nothing, Val(2), Float64) === (NoExtrap(), WrapExtrap())
            @test _resolve_extrap((ClampExtrap(), ExtendExtrap(), NoExtrap()), nothing, Val(3), Float64) ===
                (ClampExtrap(), ExtendExtrap(), NoExtrap())
        end

        @testset "wrong-length mode tuple → error" begin
            @test_throws ArgumentError _resolve_extrap((NoExtrap(), WrapExtrap()), nothing, Val(3), Float64)
            @test_throws ArgumentError _resolve_extrap((NoExtrap(),), nothing, Val(2), Float64)
        end

        @testset "single mode + periodic BC override" begin
            bcs = (ZeroCurvBC(), PeriodicBC())
            # NoExtrap: axis 2 (periodic) overridden to WrapExtrap()
            @test _resolve_extrap(NoExtrap(), bcs, Val(2), Float64) === (NoExtrap(), WrapExtrap())
            # WrapExtrap: all axes get WrapExtrap() — compatible with PeriodicBC
            @test _resolve_extrap(WrapExtrap(), bcs, Val(2), Float64) === (WrapExtrap(), WrapExtrap())
        end

        @testset "single mode + periodic BC rejection" begin
            bcs = (ZeroCurvBC(), PeriodicBC())
            @test_throws ArgumentError _resolve_extrap(ClampExtrap(), bcs, Val(2), Float64)
            @test_throws ArgumentError _resolve_extrap(ExtendExtrap(), bcs, Val(2), Float64)
        end

        @testset "per-axis mode tuple + periodic BC" begin
            bcs = (ZeroCurvBC(), PeriodicBC())
            # ExtendExtrap on non-periodic axis is OK; periodic axis gets wrap
            @test _resolve_extrap((ExtendExtrap(), WrapExtrap()), bcs, Val(2), Float64) === (ExtendExtrap(), WrapExtrap())
            # ClampExtrap on periodic axis → error
            @test_throws ArgumentError _resolve_extrap((NoExtrap(), ClampExtrap()), bcs, Val(2), Float64)
        end

        # Symbol-based extrap was fully removed in v0.3.0.
        # Only AbstractExtrap types are accepted.

        @testset "_check_mode_periodic_compat" begin
            bcs = (ZeroCurvBC(), PeriodicBC(), ZeroCurvBC())
            @test _check_mode_periodic_compat(NoExtrap(), bcs, Val(3)) === nothing
            @test _check_mode_periodic_compat(WrapExtrap(), bcs, Val(3)) === nothing
            @test_throws ArgumentError _check_mode_periodic_compat(ClampExtrap(), bcs, Val(3))
            @test_throws ArgumentError _check_mode_periodic_compat(ExtendExtrap(), bcs, Val(3))

            # No periodic BCs — all modes are compatible
            bcs_none = (ZeroCurvBC(), ZeroCurvBC())
            @test _check_mode_periodic_compat(ClampExtrap(), bcs_none, Val(2)) === nothing
        end

        @testset "_check_modes_periodic_compat" begin
            bcs = (ZeroCurvBC(), PeriodicBC())
            @test _check_modes_periodic_compat((NoExtrap(), WrapExtrap()), bcs, Val(2)) === nothing
            @test _check_modes_periodic_compat((ClampExtrap(), WrapExtrap()), bcs, Val(2)) === nothing
            @test_throws ArgumentError _check_modes_periodic_compat((NoExtrap(), ClampExtrap()), bcs, Val(2))
        end

        @testset "@generated periodic override" begin
            bcs = (ZeroCurvBC(), PeriodicBC(), ZeroCurvBC())
            # Single mode broadcast: periodic axis → WrapExtrap(), others → original mode
            @test _mode_to_modes_with_periodic(NoExtrap(), bcs) === (NoExtrap(), WrapExtrap(), NoExtrap())
            @test _mode_to_modes_with_periodic(WrapExtrap(), bcs) === (WrapExtrap(), WrapExtrap(), WrapExtrap())

            # Per-axis mode tuple: periodic axis → WrapExtrap()
            modes = (ExtendExtrap(), WrapExtrap(), ClampExtrap())
            @test _modes_to_modes_with_periodic(modes, bcs) === (ExtendExtrap(), WrapExtrap(), ClampExtrap())
        end
    end
end
@testitem "_search_all_intervals — grid-only overload (no spacings arg)" begin
    using FastInterpolations:
        _search_all_intervals, _ensure_hint_nd, _check_mono_nd,
        AutoSearch, _CachedRange, _CachedVector

    # Setup: 2D Range × Vector with cached wrappers (the post-1D-migration shape)
    rng_grid = FastInterpolations._to_float(0.0:1.0:5.0, Float64)
    vec_grid = FastInterpolations._CachedVector([0.0, 0.5, 1.5, 3.0, 5.0])
    grids = (rng_grid, vec_grid)
    policies = (AutoSearch(), AutoSearch())
    hints = _ensure_hint_nd(nothing, Val(2))
    mono = (false, false)
    q_evals = (2.5, 1.0)

    # New 5-arg overload (no spacings)
    indices, Ls, Rs = _search_all_intervals(q_evals, grids, policies, hints, mono)

    @test indices isa NTuple{2, Int}
    @test Ls isa NTuple{2, <:Real}
    @test Rs isa NTuple{2, <:Real}
    # 2.5 falls in cell 3 of 0:1:5 (between 2.0 and 3.0)
    @test indices[1] == 3
    @test Ls[1] ≈ 2.0
    @test Rs[1] ≈ 3.0
    # 1.0 falls in cell 2 of [0.0, 0.5, 1.5, 3.0, 5.0] (between 0.5 and 1.5)
    @test indices[2] == 2
    @test Ls[2] ≈ 0.5
    @test Rs[2] ≈ 1.5
end

@testitem "_compute_all_local_params — grid-only overload" begin
    using FastInterpolations: _compute_all_local_params, _CachedRange, _CachedVector

    rng_grid = FastInterpolations._to_float(0.0:1.0:5.0, Float64)
    vec_grid = FastInterpolations._CachedVector([0.0, 0.5, 1.5, 3.0, 5.0])
    grids = (rng_grid, vec_grid)
    indices = (3, 2)
    Ls = (2.0, 0.5)
    q_evals = (2.5, 1.0)

    hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids, indices, Ls)

    @test hs[1] ≈ 1.0      # rng cell width
    @test hs[2] ≈ 1.0      # vec cell 2: 1.5 - 0.5
    @test inv_hs[1] ≈ 1.0
    @test inv_hs[2] ≈ 1.0
    @test dLs[1] ≈ 0.5     # 2.5 - 2.0
    @test dLs[2] ≈ 0.5     # 1.0 - 0.5
end

# The 2D `_locate_cell_2d_preamble` was consolidated into the generic-N locate (extrap-aware
# `_handle_all_extraps` + 6-arg `_search_all_intervals`, commit `refactor(nd): collapse
# 2D-specialized locate into extrap-aware generic-N`). Test that surviving path directly.
@testitem "N=2 cell locate (generic path) — grid-only, mixed range+vector" begin
    using FastInterpolations:
        _search_all_intervals, _handle_all_extraps, _ensure_hint_nd, NoExtrap, AutoSearch, _CachedVector

    rng_grid = FastInterpolations._to_float(0.0:1.0:5.0, Float64)
    vec_grid = FastInterpolations._CachedVector([0.0, 0.5, 1.5, 3.0, 5.0])
    grids = (rng_grid, vec_grid)
    extraps = (NoExtrap(), NoExtrap())
    policies = (AutoSearch(), AutoSearch())
    hints = _ensure_hint_nd(nothing, Val(2))
    mono = (false, false)
    query = (2.5, 1.0)

    q_eval = _handle_all_extraps(query, grids, extraps)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, policies, hints, mono, extraps)

    @test q_eval[1] ≈ 2.5
    @test q_eval[2] ≈ 1.0
    @test indices[1] == 3
    @test indices[2] == 2
    @test Ls[1] ≈ 2.0
    @test Ls[2] ≈ 0.5
end

@testitem "N=0 Aqua disambiguators — coverage" begin
    using FastInterpolations: _search_all_intervals, _compute_all_local_params

    # These N=0 methods exist solely to satisfy `Aqua.test_ambiguities` —
    # ND interpolants require N >= 1 at runtime so they are never reached
    # via normal call sites. Cover them by direct invocation so the
    # coverage report doesn't flag them as missing.
    @test _search_all_intervals((), (), (), ()) === ((), (), ())
    @test _search_all_intervals((), (), (), (), ()) === ((), (), ())
    @test _compute_all_local_params((), (), (), ()) === ((), (), ())
end

@testitem "_first_fill_value defensive `error` path" begin
    using FastInterpolations: _first_fill_value

    # Normal callers guard with `_is_fill_oob` so the no-`FillExtrap` arm is
    # unreachable, but the defensive `error("unreachable: ...")` body still
    # ships in the binary. Cover it by direct invocation with extraps tuples
    # that contain no FillExtrap — exercises the loop-then-error branch.
    @test_throws ErrorException _first_fill_value(())
    @test_throws ErrorException _first_fill_value((NoExtrap(), ClampExtrap()))
end

@testitem "_check_mono_nd generic-queries protocol — coverage" begin
    using FastInterpolations: _check_mono_nd, _query_length, _query_extract

    # The third `_check_mono_nd` overload (generic queries protocol) is the
    # fallback for any container that implements `_query_length` /
    # `_query_extract` but is neither an SoA `Tuple{Vararg{AbstractVector,N}}`
    # nor an AoS `AbstractVector` (which hit the more specific overloads).
    # No production caller currently routes through this path, so cover it
    # via a custom container.
    struct MyQ
        pts::Vector{NTuple{2, Float64}}
    end
    FastInterpolations._query_length(q::MyQ) = length(q.pts)
    FastInterpolations._query_extract(q::MyQ, i) = q.pts[i]

    # Short queries (< 8) → fast-path all-false (line 719).
    short = MyQ([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    @test _check_mono_nd((BinarySearch(), BinarySearch()), short) ===
        (false, false)

    # Long queries (≥ 8) → AoS mono-check path (lines 720-721).
    long = MyQ([(Float64(i), Float64(i)) for i in 1:10])
    @test _check_mono_nd((BinarySearch(), BinarySearch()), long) isa
        NTuple{2, Bool}
end
