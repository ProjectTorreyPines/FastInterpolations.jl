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
    _resolve_deriv_nd, _resolve_side_nd, _validate_nd_grids,
    _promote_grid_eltype, _convert_grids_typed, _create_spacings_typed

@testset "Shared ND Utilities" begin
    # ========================================
    # _resolve_extrap_nd
    # ========================================
    @testset "_resolve_extrap_nd" begin
        @testset "broadcast single symbol to N-tuple" begin
            # Single symbol should broadcast to all axes
            result = _resolve_extrap_nd(:none, Val(3))
            @test result === (:none, :none, :none)

            result = _resolve_extrap_nd(:wrap, Val(2))
            @test result === (:wrap, :wrap)

            result = _resolve_extrap_nd(:constant, Val(4))
            @test result === (:constant, :constant, :constant, :constant)
        end

        @testset "passthrough matching tuple" begin
            # Matching N-tuple should pass through
            result = _resolve_extrap_nd((:none, :wrap, :constant), Val(3))
            @test result === (:none, :wrap, :constant)
        end

        @testset "reject wrong-length tuple" begin
            # Wrong-length tuple should throw ArgumentError
            @test_throws ArgumentError _resolve_extrap_nd((:none, :wrap), Val(3))
            @test_throws ArgumentError _resolve_extrap_nd((:none, :wrap, :constant, :extension), Val(3))
            @test_throws ArgumentError _resolve_extrap_nd((:none,), Val(2))
        end

        @testset "reject invalid symbol" begin
            # Invalid symbol should throw ArgumentError
            @test_throws ArgumentError _resolve_extrap_nd(:invalid, Val(2))
            @test_throws ArgumentError _resolve_extrap_nd((:none, :invalid), Val(2))
        end
    end

    # ========================================
    # _resolve_search_nd
    # ========================================
    @testset "_resolve_search_nd" begin
        @testset "broadcast single policy to N-tuple" begin
            result = _resolve_search_nd(Binary(), Val(3))
            @test length(result) == 3
            @test all(s -> s isa Binary, result)

            result = _resolve_search_nd(LinearBinary(), Val(2))
            @test length(result) == 2
            @test all(s -> s isa LinearBinary, result)
        end

        @testset "passthrough matching tuple" begin
            policies = (Binary(), LinearBinary(), Linear())
            result = _resolve_search_nd(policies, Val(3))
            @test result === policies
        end

        @testset "reject wrong-length tuple" begin
            @test_throws ArgumentError _resolve_search_nd((Binary(), LinearBinary()), Val(3))
            @test_throws ArgumentError _resolve_search_nd((Binary(), LinearBinary(), Linear(), HintedBinary()), Val(3))
        end
    end

    # ========================================
    # _resolve_bcs_nd
    # ========================================
    @testset "_resolve_bcs_nd" begin
        @testset "broadcast single BC to N-tuple" begin
            result = _resolve_bcs_nd(NaturalBC(), Val(3))
            @test length(result) == 3
            @test all(bc -> bc isa NaturalBC, result)

            result = _resolve_bcs_nd(PolyFit{3}(), Val(2))
            @test length(result) == 2
            @test all(bc -> bc isa PolyFit{3}, result)
        end

        @testset "passthrough matching tuple" begin
            bcs = (NaturalBC(), PolyFit{3}(), ClampedBC())
            result = _resolve_bcs_nd(bcs, Val(3))
            @test result === bcs
        end

        @testset "reject wrong-length tuple" begin
            @test_throws ArgumentError _resolve_bcs_nd((NaturalBC(), PolyFit{3}()), Val(3))
            @test_throws ArgumentError _resolve_bcs_nd((NaturalBC(),), Val(2))
        end
    end

    # ========================================
    # _resolve_side_nd (NEW - for ConstantInterpolantND)
    # ========================================
    @testset "_resolve_side_nd" begin
        @testset "broadcast single symbol to N-tuple" begin
            result = _resolve_side_nd(:nearest, Val(3))
            @test result === (:nearest, :nearest, :nearest)

            result = _resolve_side_nd(:left, Val(2))
            @test result === (:left, :left)

            result = _resolve_side_nd(:right, Val(4))
            @test result === (:right, :right, :right, :right)
        end

        @testset "passthrough matching tuple" begin
            result = _resolve_side_nd((:nearest, :left, :right), Val(3))
            @test result === (:nearest, :left, :right)
        end

        @testset "reject wrong-length tuple" begin
            @test_throws ArgumentError _resolve_side_nd((:nearest, :left), Val(3))
            @test_throws ArgumentError _resolve_side_nd((:nearest, :left, :right, :nearest), Val(3))
        end

        @testset "reject invalid symbol" begin
            @test_throws ArgumentError _resolve_side_nd(:invalid, Val(2))
            @test_throws ArgumentError _resolve_side_nd((:nearest, :invalid), Val(2))
            @test_throws ArgumentError _resolve_side_nd(:center, Val(2))
        end
    end

    # ========================================
    # _resolve_deriv_nd
    # ========================================
    @testset "_resolve_deriv_nd" begin
        @testset "Int broadcast to uniform EvalOp tuple" begin
            result = _resolve_deriv_nd(0, Val(3))
            @test length(result) == 3
            @test all(op -> op isa EvalValue, result)

            result = _resolve_deriv_nd(1, Val(2))
            @test length(result) == 2
            @test all(op -> op isa EvalDeriv1, result)

            result = _resolve_deriv_nd(2, Val(4))
            @test length(result) == 4
            @test all(op -> op isa EvalDeriv2, result)

            result = _resolve_deriv_nd(3, Val(2))
            @test length(result) == 2
            @test all(op -> op isa EvalDeriv3, result)
        end

        @testset "Val{Int} compile-time broadcast" begin
            result = _resolve_deriv_nd(Val(0), Val(3))
            @test length(result) == 3
            @test all(op -> op isa EvalValue, result)

            result = _resolve_deriv_nd(Val(1), Val(2))
            @test length(result) == 2
            @test all(op -> op isa EvalDeriv1, result)
        end

        @testset "Val{Tuple} mixed partials" begin
            # ∂f/∂x: deriv=1 on axis 1, deriv=0 on axis 2
            result = _resolve_deriv_nd(Val((1, 0)), Val(2))
            @test result[1] isa EvalDeriv1
            @test result[2] isa EvalValue

            # ∂²f/∂x∂y: deriv=1 on both axes
            result = _resolve_deriv_nd(Val((1, 1)), Val(2))
            @test all(op -> op isa EvalDeriv1, result)

            # ∂²f/∂x²: deriv=2 on axis 1, deriv=0 on axis 2
            result = _resolve_deriv_nd(Val((2, 0)), Val(2))
            @test result[1] isa EvalDeriv2
            @test result[2] isa EvalValue
        end

        @testset "reject invalid deriv values" begin
            @test_throws ArgumentError _resolve_deriv_nd(4, Val(2))
            @test_throws ArgumentError _resolve_deriv_nd(-1, Val(2))

            # Wrong-length Val tuple
            @test_throws ArgumentError _resolve_deriv_nd(Val((1, 0)), Val(3))
            @test_throws ArgumentError _resolve_deriv_nd(Val((1, 0, 1, 0)), Val(3))
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
end
