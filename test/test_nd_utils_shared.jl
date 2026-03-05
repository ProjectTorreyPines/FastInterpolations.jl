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

using Test
using FastInterpolations

# Access internal functions for testing
import FastInterpolations: _resolve_extrap_nd, _resolve_search_nd, _resolve_bcs_nd,
    _resolve_side_nd, _validate_nd_grids,
    _promote_grid_eltype, _convert_grids_typed, _create_spacings_typed,
    _check_mode_periodic_compat, _check_modes_periodic_compat,
    _mode_to_modes_with_periodic, _modes_to_modes_with_periodic

@testset "Shared ND Utilities" begin
    # ========================================
    # _resolve_extrap_nd
    # ========================================
    # NOTE: Old 2-arg _resolve_extrap_nd(extrap, Val(N)) tests removed.
    # Symbol-based extrap was removed in v0.3.0.
    # The 3-arg form _resolve_extrap_nd(extrap, bcs, Val(N)) is tested below
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
            @test_throws ArgumentError _resolve_search_nd((BinarySearch(), LinearBinarySearch(), LinearSearch(), LinearBinarySearch(linear_window=0)), Val(3))
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

        @testset "_create_spacings_typed" begin
            # Range → ScalarSpacing
            x_range = range(0.0, 1.0, 11)
            # Vector → VectorSpacing
            y_vec = [0.0, 0.3, 0.7, 1.0]

            spacings = _create_spacings_typed((x_range, y_vec))
            @test length(spacings) == 2
            # First should be ScalarSpacing (uniform grid)
            # Second should be VectorSpacing (non-uniform)
        end
    end

    # ========================================
    # Typed Extrap Resolution (3-arg form)
    # ========================================
    @testset "Typed Extrap resolution" begin
        @testset "single mode → Mode tuple, no BCs" begin
            @test _resolve_extrap_nd(NoExtrap(), nothing, Val(2), Float64) === (NoExtrap(), NoExtrap())
            @test _resolve_extrap_nd(ClampedExtrap(), nothing, Val(3), Float64) === (ClampedExtrap(), ClampedExtrap(), ClampedExtrap())
            @test _resolve_extrap_nd(ExtendExtrap(), nothing, Val(1), Float64) === (ExtendExtrap(),)
            @test _resolve_extrap_nd(WrapExtrap(), nothing, Val(2), Float64) === (WrapExtrap(), WrapExtrap())
        end

        @testset "mode tuple → Mode tuple, no BCs" begin
            @test _resolve_extrap_nd((NoExtrap(), WrapExtrap()), nothing, Val(2), Float64) === (NoExtrap(), WrapExtrap())
            @test _resolve_extrap_nd((ClampedExtrap(), ExtendExtrap(), NoExtrap()), nothing, Val(3), Float64) ===
                (ClampedExtrap(), ExtendExtrap(), NoExtrap())
        end

        @testset "wrong-length mode tuple → error" begin
            @test_throws ArgumentError _resolve_extrap_nd((NoExtrap(), WrapExtrap()), nothing, Val(3), Float64)
            @test_throws ArgumentError _resolve_extrap_nd((NoExtrap(),), nothing, Val(2), Float64)
        end

        @testset "single mode + periodic BC override" begin
            bcs = (ZeroCurvBC(), PeriodicBC())
            # NoExtrap: axis 2 (periodic) overridden to WrapExtrap()
            @test _resolve_extrap_nd(NoExtrap(), bcs, Val(2), Float64) === (NoExtrap(), WrapExtrap())
            # WrapExtrap: all axes get WrapExtrap() — compatible with PeriodicBC
            @test _resolve_extrap_nd(WrapExtrap(), bcs, Val(2), Float64) === (WrapExtrap(), WrapExtrap())
        end

        @testset "single mode + periodic BC rejection" begin
            bcs = (ZeroCurvBC(), PeriodicBC())
            @test_throws ArgumentError _resolve_extrap_nd(ClampedExtrap(), bcs, Val(2), Float64)
            @test_throws ArgumentError _resolve_extrap_nd(ExtendExtrap(), bcs, Val(2), Float64)
        end

        @testset "per-axis mode tuple + periodic BC" begin
            bcs = (ZeroCurvBC(), PeriodicBC())
            # ExtendExtrap on non-periodic axis is OK; periodic axis gets wrap
            @test _resolve_extrap_nd((ExtendExtrap(), WrapExtrap()), bcs, Val(2), Float64) === (ExtendExtrap(), WrapExtrap())
            # ClampedExtrap on periodic axis → error
            @test_throws ArgumentError _resolve_extrap_nd((NoExtrap(), ClampedExtrap()), bcs, Val(2), Float64)
        end

        # Symbol-based extrap was fully removed in v0.3.0.
        # Only AbstractExtrap types are accepted.

        @testset "_check_mode_periodic_compat" begin
            bcs = (ZeroCurvBC(), PeriodicBC(), ZeroCurvBC())
            @test _check_mode_periodic_compat(NoExtrap(), bcs, Val(3)) === nothing
            @test _check_mode_periodic_compat(WrapExtrap(), bcs, Val(3)) === nothing
            @test_throws ArgumentError _check_mode_periodic_compat(ClampedExtrap(), bcs, Val(3))
            @test_throws ArgumentError _check_mode_periodic_compat(ExtendExtrap(), bcs, Val(3))

            # No periodic BCs — all modes are compatible
            bcs_none = (ZeroCurvBC(), ZeroCurvBC())
            @test _check_mode_periodic_compat(ClampedExtrap(), bcs_none, Val(2)) === nothing
        end

        @testset "_check_modes_periodic_compat" begin
            bcs = (ZeroCurvBC(), PeriodicBC())
            @test _check_modes_periodic_compat((NoExtrap(), WrapExtrap()), bcs, Val(2)) === nothing
            @test _check_modes_periodic_compat((ClampedExtrap(), WrapExtrap()), bcs, Val(2)) === nothing
            @test_throws ArgumentError _check_modes_periodic_compat((NoExtrap(), ClampedExtrap()), bcs, Val(2))
        end

        @testset "@generated periodic override" begin
            bcs = (ZeroCurvBC(), PeriodicBC(), ZeroCurvBC())
            # Single mode broadcast: periodic axis → WrapExtrap(), others → original mode
            @test _mode_to_modes_with_periodic(NoExtrap(), bcs) === (NoExtrap(), WrapExtrap(), NoExtrap())
            @test _mode_to_modes_with_periodic(WrapExtrap(), bcs) === (WrapExtrap(), WrapExtrap(), WrapExtrap())

            # Per-axis mode tuple: periodic axis → WrapExtrap()
            modes = (ExtendExtrap(), WrapExtrap(), ClampedExtrap())
            @test _modes_to_modes_with_periodic(modes, bcs) === (ExtendExtrap(), WrapExtrap(), ClampedExtrap())
        end
    end
end
