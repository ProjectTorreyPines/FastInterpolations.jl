# ========================================
# ND Coverage Tests
# ========================================
#
# Targeted tests to improve coverage for:
# - nd_build.jl: Non-batch differentiation, error paths
# - nd_types.jl: Convenience constructors, accessors, @generated alternatives
# - nd_utils.jl: Wrong-sized tuple errors, PolyFit BC helpers
#
# These tests focus on exercising internal functions that aren't
# reached through the normal public API code paths.

using Test
using FastInterpolations

# Import internal functions we need to test
import FastInterpolations:
    # nd_math.jl (Val(1) error methods)
    _ldiv_along_dim!,
    compute_rhs_along_dim!,
    moments_to_derivatives_along_dim!,
    # nd_build.jl
    _differentiate_nd_along_dim!,
    _differentiate_nd_along_dim_batch!,
    _check_periodic_data_nd,
    _get_effective_bc,
    _compute_nd_partials!,
    _build_nd_coeffs,
    # nd_types.jl
    NodalDerivativesND,
    PreCompute,
    OnTheFly,
    _grid,
    _spacing,
    _bc,
    _extrap,
    _search,
    _handle_all_extraps,
    _handle_all_extraps_gen,
    _search_all_intervals,
    _search_all_intervals_gen,
    _compute_all_local_params,
    _compute_all_local_params_gen,
    num_partials,
    # nd_utils.jl
    _resolve_extrap_nd,
    _resolve_search_nd,
    _resolve_bcs_nd,
    _resolve_deriv_nd,
    _int_to_evalop,
    _get_polyfit_bc,
    _make_polyfit,
    _validate_nd_grids,
    _promote_grid_eltype,
    _convert_grids_typed,
    _create_spacings_typed,
    # BC types
    NaturalBC,
    PeriodicBC,
    PolyFit,
    CubicFit,
    BCPair,
    # Eval ops
    EvalValue,
    EvalDeriv1,
    EvalDeriv2,
    EvalDeriv3

@testset "ND Coverage Tests" begin

    # ========================================
    # nd_build.jl Coverage
    # ========================================
    @testset "nd_build.jl" begin

        @testset "_differentiate_nd_along_dim! (non-batch version)" begin
            # This function is never called in normal operation because
            # _differentiate_nd_along_dim_batch! handles all cases.
            # We need to test it directly.

            x = collect(range(0.0, 2π, 11))
            y = collect(range(0.0, π, 9))
            data = [sin(xi) * cos(yj) for xi in x, yj in y]

            # Test along dimension 1
            out1 = similar(data)
            _differentiate_nd_along_dim!(out1, data, x, NaturalBC(), 1)

            # Derivative of sin(x)*cos(y) w.r.t. x is cos(x)*cos(y)
            # Check at interior points
            for j in 2:length(y)-1
                for i in 2:length(x)-1
                    expected = cos(x[i]) * cos(y[j])
                    @test out1[i, j] ≈ expected atol=0.1
                end
            end

            # Test along dimension 2
            out2 = similar(data)
            _differentiate_nd_along_dim!(out2, data, y, NaturalBC(), 2)

            # Derivative of sin(x)*cos(y) w.r.t. y is -sin(x)*sin(y)
            for j in 2:length(y)-1
                for i in 2:length(x)-1
                    expected = -sin(x[i]) * sin(y[j])
                    @test out2[i, j] ≈ expected atol=0.1
                end
            end
        end

        @testset "_differentiate_nd_along_dim! error paths" begin
            x = collect(range(0.0, 1.0, 5))
            y = collect(range(0.0, 1.0, 4))
            data = rand(5, 4)
            out = similar(data)

            # Dimension out of range
            @test_throws ArgumentError _differentiate_nd_along_dim!(out, data, x, NaturalBC(), 0)
            @test_throws ArgumentError _differentiate_nd_along_dim!(out, data, x, NaturalBC(), 3)

            # Size mismatch between out and data
            out_wrong = rand(4, 4)
            @test_throws DimensionMismatch _differentiate_nd_along_dim!(out_wrong, data, x, NaturalBC(), 1)

            # Grid length mismatch
            wrong_grid = collect(range(0.0, 1.0, 6))  # 6 points but dim 1 has 5
            @test_throws DimensionMismatch _differentiate_nd_along_dim!(out, data, wrong_grid, NaturalBC(), 1)
        end

        @testset "_check_periodic_data_nd error path" begin
            # Test with data that violates periodicity
            x = collect(range(0.0, 2π, 11))
            y = collect(range(0.0, π, 9))

            # Data where first and last slices don't match along dimension 1
            data_non_periodic = zeros(11, 9)
            data_non_periodic[1, :] .= 1.0   # First slice
            data_non_periodic[end, :] .= 0.0  # Last slice (different!)

            @test_throws ArgumentError _check_periodic_data_nd(data_non_periodic, 1)

            # Data where first and last slices don't match along dimension 2
            data_non_periodic2 = zeros(11, 9)
            data_non_periodic2[:, 1] .= 1.0
            data_non_periodic2[:, end] .= 0.0

            @test_throws ArgumentError _check_periodic_data_nd(data_non_periodic2, 2)

            # Valid periodic data should not throw
            data_periodic = zeros(11, 9)
            data_periodic[1, :] .= 1.0
            data_periodic[end, :] .= 1.0  # Matches!
            @test _check_periodic_data_nd(data_periodic, 1) === nothing
        end

        @testset "_get_effective_bc edge cases" begin
            grid_short = collect(1.0:3.0)  # 3 points, not enough for CubicFit
            grid_long = collect(1.0:10.0)   # 10 points

            # p_src == 1: always return original BC
            @test _get_effective_bc(NaturalBC(), 1, grid_long) isa NaturalBC
            @test _get_effective_bc(PeriodicBC(), 1, grid_long) isa PeriodicBC

            # p_src > 1 with PeriodicBC: propagate periodic
            @test _get_effective_bc(PeriodicBC(), 2, grid_long) isa PeriodicBC

            # p_src > 1 with short grid (< 4 points): fallback to NaturalBC
            @test _get_effective_bc(NaturalBC(), 2, grid_short) isa NaturalBC

            # p_src > 1 with long grid: use CubicFit
            @test _get_effective_bc(NaturalBC(), 2, grid_long) isa CubicFit
        end
    end

    # ========================================
    # nd_types.jl Coverage
    # ========================================
    @testset "nd_types.jl" begin

        @testset "NodalDerivativesND convenience constructor" begin
            # The convenience constructor NodalDerivativesND{Tv, N}(partials) infers NP1
            partials_2d = rand(4, 5, 6)  # 2^2=4 partials, 5x6 grid

            # Test convenience constructor
            nd = NodalDerivativesND{Float64, 2}(partials_2d)
            @test nd isa NodalDerivativesND{Float64, 2, 3}
            @test nd.partials === partials_2d
        end

        @testset "Strategy types" begin
            @test PreCompute() isa PreCompute
            @test OnTheFly() isa OnTheFly
            @test PreCompute <: FastInterpolations.AbstractCoeffStrategy
            @test OnTheFly <: FastInterpolations.AbstractCoeffStrategy
        end

        @testset "Val-based accessors" begin
            x = range(0.0, 1.0, 10)
            y = range(0.0, 2.0, 15)
            data = rand(10, 15)
            itp = cubic_interp((x, y), data)

            # Test _grid accessor
            @test _grid(itp, Val(1)) == collect(x)
            @test _grid(itp, Val(2)) == collect(y)

            # Test _spacing accessor (this was uncovered!)
            sp1 = _spacing(itp, Val(1))
            sp2 = _spacing(itp, Val(2))
            @test sp1 isa FastInterpolations.AbstractGridSpacing
            @test sp2 isa FastInterpolations.AbstractGridSpacing

            # Test _bc accessor
            @test _bc(itp, Val(1)) isa FastInterpolations.AbstractBC
            @test _bc(itp, Val(2)) isa FastInterpolations.AbstractBC

            # Test _extrap accessor
            @test _extrap(itp, Val(1)) isa Val
            @test _extrap(itp, Val(2)) isa Val

            # Test _search accessor
            @test _search(itp, Val(1)) isa FastInterpolations.AbstractSearchPolicy
            @test _search(itp, Val(2)) isa FastInterpolations.AbstractSearchPolicy
        end

        @testset "@generated alternatives (performance variants)" begin
            x = range(0.0, 1.0, 10)
            y = range(0.0, 2.0, 15)
            data = rand(10, 15)
            itp = cubic_interp((x, y), data)

            grids = itp.grids
            spacings = itp.spacings
            searches = itp.searches
            extraps = itp.extraps

            query = (0.5, 1.0)

            # Test _handle_all_extraps_gen
            result_gen = _handle_all_extraps_gen(query, grids, extraps)
            result_std = _handle_all_extraps(query, grids, extraps)
            @test result_gen == result_std

            # Test _search_all_intervals_gen
            q_evals = (0.5, 1.0)
            (indices_gen, Ls_gen, Rs_gen) = _search_all_intervals_gen(q_evals, grids, spacings, searches)
            (indices_std, Ls_std, Rs_std) = _search_all_intervals(q_evals, grids, spacings, searches)
            @test indices_gen == indices_std
            @test Ls_gen == Ls_std
            @test Rs_gen == Rs_std

            # Test _compute_all_local_params_gen
            indices = indices_std
            Ls = Ls_std
            (hs_gen, inv_hs_gen, dLs_gen) = _compute_all_local_params_gen(q_evals, spacings, indices, Ls)
            (hs_std, inv_hs_std, dLs_std) = _compute_all_local_params(q_evals, spacings, indices, Ls)
            @test hs_gen == hs_std
            @test inv_hs_gen == inv_hs_std
            @test dLs_gen == dLs_std
        end

        @testset "num_partials on type" begin
            x = range(0.0, 1.0, 10)
            y = range(0.0, 2.0, 15)
            data = rand(10, 15)
            itp = cubic_interp((x, y), data)

            # Instance method
            @test num_partials(itp) == 4  # 2^2 = 4

            # Type method (uncovered!)
            @test num_partials(typeof(itp)) == 4
        end
    end

    # ========================================
    # nd_utils.jl Coverage
    # ========================================
    @testset "nd_utils.jl" begin

        @testset "_resolve_extrap_nd wrong-sized tuple error" begin
            # Correct size should work
            @test _resolve_extrap_nd(:none, Val(2)) == (:none, :none)
            @test _resolve_extrap_nd((:none, :constant), Val(2)) == (:none, :constant)

            # Wrong size should throw
            @test_throws ArgumentError _resolve_extrap_nd((:none,), Val(2))  # 1 element for 2D
            @test_throws ArgumentError _resolve_extrap_nd((:none, :none, :none), Val(2))  # 3 for 2D
        end

        @testset "_resolve_search_nd wrong-sized tuple error" begin
            # Note: Search policy types are Binary, Linear, etc. (not exported)
            Binary = FastInterpolations.Binary
            Linear = FastInterpolations.Linear
            bs = Binary()
            ls = Linear()

            # Correct size should work
            @test _resolve_search_nd(bs, Val(2)) == (bs, bs)
            @test _resolve_search_nd((bs, ls), Val(2)) == (bs, ls)

            # Wrong size should throw
            @test_throws ArgumentError _resolve_search_nd((bs,), Val(2))
            @test_throws ArgumentError _resolve_search_nd((bs, bs, bs), Val(2))
        end

        @testset "_resolve_bcs_nd wrong-sized tuple error" begin
            # Correct size should work
            @test _resolve_bcs_nd(NaturalBC(), Val(2)) == (NaturalBC(), NaturalBC())
            @test _resolve_bcs_nd((NaturalBC(), PeriodicBC()), Val(2)) == (NaturalBC(), PeriodicBC())

            # Wrong size should throw
            @test_throws ArgumentError _resolve_bcs_nd((NaturalBC(),), Val(2))
            @test_throws ArgumentError _resolve_bcs_nd((NaturalBC(), NaturalBC(), NaturalBC()), Val(2))
        end

        @testset "_int_to_evalop" begin
            @test _int_to_evalop(Val(0)) isa EvalValue
            @test _int_to_evalop(Val(1)) isa EvalDeriv1
            @test _int_to_evalop(Val(2)) isa EvalDeriv2
            @test _int_to_evalop(Val(3)) isa EvalDeriv3
        end

        @testset "_resolve_deriv_nd runtime Int path" begin
            # Val{Int} path (already tested in comprehensive)
            @test _resolve_deriv_nd(Val((0, 0)), Val(2)) == (EvalValue(), EvalValue())
            @test _resolve_deriv_nd(Val((1, 0)), Val(2)) == (EvalDeriv1(), EvalValue())
            @test _resolve_deriv_nd(Val((0, 1)), Val(2)) == (EvalValue(), EvalDeriv1())
            @test _resolve_deriv_nd(Val((2, 1)), Val(2)) == (EvalDeriv2(), EvalDeriv1())

            # Runtime Int path (uncovered!)
            @test _resolve_deriv_nd(0, Val(2)) == (EvalValue(), EvalValue())
            @test _resolve_deriv_nd(1, Val(2)) == (EvalDeriv1(), EvalDeriv1())
            @test _resolve_deriv_nd(2, Val(2)) == (EvalDeriv2(), EvalDeriv2())
            @test _resolve_deriv_nd(3, Val(2)) == (EvalDeriv3(), EvalDeriv3())

            # Invalid Int should throw
            @test_throws ArgumentError _resolve_deriv_nd(4, Val(2))
            @test_throws ArgumentError _resolve_deriv_nd(-1, Val(2))
        end

        @testset "_resolve_deriv_nd Val tuple size mismatch" begin
            # Wrong-sized tuple in Val
            @test_throws ArgumentError _resolve_deriv_nd(Val((1,)), Val(2))  # 1 element for 2D
            @test_throws ArgumentError _resolve_deriv_nd(Val((1, 2, 3)), Val(2))  # 3 for 2D
        end

        @testset "PolyFit BC helpers" begin
            # _get_polyfit_bc returns the PolyFit BC or constructs one
            @test _get_polyfit_bc(PolyFit{3}(), 2) isa PolyFit{3}  # Already PolyFit, return as-is
            @test _get_polyfit_bc(NaturalBC(), 3) isa PolyFit{3}   # Construct from degree

            # BCPair with PolyFit on one side (BCPair requires PointBC subtypes)
            # PolyFit is a PointBC, so we can use BCPair(PolyFit, PolyFit)
            # or BCPair(PolyFit, Deriv1/Deriv2/Deriv3)
            Deriv1 = FastInterpolations.Deriv1
            bc_pair_left = BCPair(PolyFit{2}(), Deriv1(0.0))
            bc_pair_right = BCPair(Deriv1(0.0), PolyFit{3}())
            @test _get_polyfit_bc(bc_pair_left, 1) isa PolyFit{2}
            @test _get_polyfit_bc(bc_pair_right, 1) isa PolyFit{3}

            # _make_polyfit for various degrees
            @test _make_polyfit(Val(1)) isa PolyFit{1}
            @test _make_polyfit(Val(2)) isa PolyFit{2}
            @test _make_polyfit(Val(3)) isa PolyFit{3}
            @test _make_polyfit(Val(4)) isa PolyFit{4}
            @test _make_polyfit(Val(5)) isa PolyFit{5}

            # Higher degrees use @generated fallback
            @test _make_polyfit(Val(6)) isa PolyFit{6}
            @test _make_polyfit(Val(10)) isa PolyFit{10}
        end

        @testset "_validate_nd_grids" begin
            # Valid case
            grids = (collect(1.0:5.0), collect(1.0:4.0))
            data = rand(5, 4)
            @test _validate_nd_grids(grids, data) === nothing

            # Size mismatch
            data_wrong = rand(4, 4)  # First dim doesn't match
            @test_throws DimensionMismatch _validate_nd_grids(grids, data_wrong)

            # Grid too short (< 2 points)
            grids_short = ([1.0], collect(1.0:4.0))
            data_short = rand(1, 4)
            @test_throws ArgumentError _validate_nd_grids(grids_short, data_short)
        end

        @testset "_promote_grid_eltype" begin
            # Same type
            grids_f64 = (collect(1.0:5.0), collect(1.0:4.0))
            @test _promote_grid_eltype(grids_f64) == Float64

            # Mixed Float32/Float64
            grids_mixed = (collect(Float32.(1:5)), collect(1.0:4.0))
            @test _promote_grid_eltype(grids_mixed) == Float64

            # All Float32
            grids_f32 = (collect(Float32.(1:5)), collect(Float32.(1:4)))
            @test _promote_grid_eltype(grids_f32) == Float32
        end

        @testset "_create_spacings_typed" begin
            # Uniform grids
            grids = (range(0.0, 1.0, 10), range(0.0, 2.0, 15))
            spacings = _create_spacings_typed(grids)
            @test spacings isa Tuple
            @test length(spacings) == 2
            @test all(s -> s isa FastInterpolations.AbstractGridSpacing, spacings)
        end
    end

    # ========================================
    # QuadraticND Build Coverage
    # ========================================
    @testset "quadratic_nd_build.jl" begin
        import FastInterpolations:
            _slope_1d_quadratic!,
            _differentiate_nd_along_dim_quadratic!,
            _get_effective_bc_quadratic,
            _compute_nd_partials_quadratic!,
            _build_nd_coeffs_quadratic

        @testset "_get_effective_bc_quadratic edge cases" begin
            grid_short = collect(1.0:2.0)   # 2 points, not enough for QuadraticFit
            grid_long  = collect(1.0:10.0)  # 10 points

            # p_src == 1: always return original BC
            @test _get_effective_bc_quadratic(Right(QuadraticFit()), 1, grid_long) isa Right
            @test _get_effective_bc_quadratic(MinCurvFit(), 1, grid_long) isa MinCurvFit

            # p_src > 1 with long grid: use Right(QuadraticFit())
            result = _get_effective_bc_quadratic(Right(QuadraticFit()), 2, grid_long)
            @test result isa Right

            # p_src > 1 with short grid (< 3 points): fallback
            result_short = _get_effective_bc_quadratic(Right(QuadraticFit()), 2, grid_short)
            @test result_short isa FastInterpolations.AbstractBC
        end

        @testset "_build_nd_coeffs_quadratic" begin
            x = collect(range(0.0, 1.0, 8))
            y = collect(range(0.0, 1.0, 6))
            data = [xi^2 + yj^2 for xi in x, yj in y]
            bcs = (Right(QuadraticFit()), Right(QuadraticFit()))

            nd = _build_nd_coeffs_quadratic((x, y), data, bcs)
            @test nd isa NodalDerivativesND{Float64, 2, 3}
            @test size(nd.partials) == (4, 8, 6)  # 2^2 partials, 8x6 grid
        end
    end

    # ========================================
    # QuadraticND API Coverage
    # ========================================
    @testset "quadratic_nd_interpolant.jl" begin
        import FastInterpolations:
            _resolve_bcs_nd_quadratic,
            _to_quadratic_bc

        @testset "_resolve_bcs_nd_quadratic" begin
            # Single QuadraticBC → broadcast
            bcs = _resolve_bcs_nd_quadratic(Right(QuadraticFit()), Val(2))
            @test length(bcs) == 2
            @test all(b -> b isa Right, bcs)

            # NTuple pass-through
            bcs_tuple = (Left(QuadraticFit()), Right(QuadraticFit()))
            @test _resolve_bcs_nd_quadratic(bcs_tuple, Val(2)) === bcs_tuple

            # NaturalBC conversion
            bcs_nat = _resolve_bcs_nd_quadratic(NaturalBC(), Val(2))
            @test length(bcs_nat) == 2
            @test all(b -> b isa Right, bcs_nat)

            # PolyFit conversion
            bcs_poly = _resolve_bcs_nd_quadratic(CubicFit(), Val(2))
            @test length(bcs_poly) == 2

            # Heterogeneous AbstractBC tuple
            bcs_hetero = _resolve_bcs_nd_quadratic((NaturalBC(), CubicFit()), Val(2))
            @test length(bcs_hetero) == 2
        end

        @testset "_to_quadratic_bc" begin
            @test _to_quadratic_bc(Right(QuadraticFit())) isa Right
            @test _to_quadratic_bc(MinCurvFit()) isa MinCurvFit
            @test _to_quadratic_bc(NaturalBC()) isa Right
            @test _to_quadratic_bc(CubicFit()) isa Right

            # Unsupported BC
            @test_throws ArgumentError _to_quadratic_bc(PeriodicBC())
            @test_throws ArgumentError _to_quadratic_bc(ClampedBC())
        end
    end

    # ========================================
    # Additional Edge Cases
    # ========================================
    @testset "Additional Edge Cases" begin

        @testset "3D interpolant type introspection" begin
            x = range(0.0, 1.0, 5)
            y = range(0.0, 1.0, 6)
            z = range(0.0, 1.0, 7)
            data = rand(5, 6, 7)
            itp = cubic_interp((x, y, z), data)

            @test ndims(itp) == 3
            @test size(itp) == (5, 6, 7)
            @test axes(itp) == itp.grids
            @test num_partials(itp) == 8  # 2^3 = 8
        end

        @testset "NodalDerivativesND validation" begin
            # NP1 mismatch with array dimension causes MethodError (type constraint in signature)
            partials_3d = rand(4, 5, 6)  # 3D array
            @test_throws MethodError NodalDerivativesND{Float64, 3, 4}(partials_3d)  # Expects 4D array

            # Correct NP1 but wrong N (N+1 ≠ NP1) should throw ArgumentError
            partials_3d_2 = rand(8, 3, 4, 5)  # 4D array with 8 partials (correct for N=3)
            @test_throws ArgumentError NodalDerivativesND{Float64, 2, 4}(partials_3d_2)  # N=2 but NP1=4≠3

            # Wrong number of partials should throw DimensionMismatch
            partials_wrong = rand(3, 5, 6)  # 3 partials but 2D needs 4
            @test_throws DimensionMismatch NodalDerivativesND{Float64, 2, 3}(partials_wrong)
        end

        @testset "Periodic BC with ND construction" begin
            # Valid periodic data (first and last match)
            x = collect(range(0.0, 2π, 11))  # First point equals last for periodic
            y = collect(range(0.0, π, 9))

            # Create valid periodic data for x dimension
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            # Make it periodic: copy first slice to last
            data[end, :] = data[1, :]

            # Should not throw
            itp = cubic_interp((x, y), data, bc=(PeriodicBC(), NaturalBC()))
            @test itp isa FastInterpolations.CubicInterpolantND
        end
    end

    # ========================================
    # nd_math.jl Val(1) Error Methods Coverage
    # ========================================
    #
    # These methods explicitly throw ArgumentError for batch operations along axis 1.
    # Benchmarking showed per-column approach is faster due to view creation overhead.
    # Testing ensures these error paths are covered.
    @testset "nd_math.jl Val(1) Error Methods" begin

        @testset "_ldiv_along_dim! Val(1) error" begin
            # Create minimal test data
            z = rand(5, 10)

            # Create a minimal Thomas factorization mock
            ThomasFactorization = FastInterpolations.ThomasFactorization
            n = 10
            dl = zeros(n - 1)
            du = zeros(n - 1)
            inv_d = ones(n)
            thomas = ThomasFactorization(dl, du, inv_d)

            # Val(1) should throw ArgumentError
            @test_throws ArgumentError _ldiv_along_dim!(z, thomas, Val(1))

            # Check error message content
            try
                _ldiv_along_dim!(z, thomas, Val(1))
            catch e
                @test e isa ArgumentError
                @test occursin("Val", e.msg) && occursin("1", e.msg)
                @test occursin("not supported", e.msg)
            end
        end

        @testset "compute_rhs_along_dim! Val(1) error" begin
            # Create minimal test data
            D = zeros(5, 10)
            data = rand(5, 10)
            x = collect(range(0.0, 1.0, 10))

            # Create spacing using ScalarSpacing (uniform grid)
            ScalarSpacing = FastInterpolations.ScalarSpacing
            h = 1.0 / 9.0  # spacing for 10 points in [0,1]
            spacing = ScalarSpacing(h, 1.0 / h)

            bc = BCPair(FastInterpolations.Deriv2(0.0), FastInterpolations.Deriv2(0.0))

            # Val(1) should throw ArgumentError
            @test_throws ArgumentError compute_rhs_along_dim!(D, data, x, spacing, bc, Val(1))

            # Check error message content
            try
                compute_rhs_along_dim!(D, data, x, spacing, bc, Val(1))
            catch e
                @test e isa ArgumentError
                @test occursin("Val", e.msg) && occursin("1", e.msg)
                @test occursin("not supported", e.msg)
            end
        end

        @testset "moments_to_derivatives_along_dim! Val(1) error" begin
            # Create minimal test data
            out = zeros(5, 10)
            M = rand(5, 10)  # Moments
            data = rand(5, 10)

            # Create spacing using ScalarSpacing
            ScalarSpacing = FastInterpolations.ScalarSpacing
            h = 1.0 / 9.0
            spacing = ScalarSpacing(h, 1.0 / h)

            bc = BCPair(FastInterpolations.Deriv2(0.0), FastInterpolations.Deriv2(0.0))

            # Val(1) should throw ArgumentError
            @test_throws ArgumentError moments_to_derivatives_along_dim!(out, M, data, spacing, bc, Val(1))

            # Check error message content
            try
                moments_to_derivatives_along_dim!(out, M, data, spacing, bc, Val(1))
            catch e
                @test e isa ArgumentError
                @test occursin("Val", e.msg) && occursin("1", e.msg)
                @test occursin("not supported", e.msg)
            end
        end

        @testset "Val(2) batch methods work correctly" begin
            # Ensure Val(2) methods work (not just testing errors)
            # This is a sanity check that our error methods don't break the valid paths

            x = range(0.0, 1.0, 10)
            y = range(0.0, 2.0, 15)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]

            # Creating an interpolant exercises the Val(2) batch paths internally
            itp = cubic_interp((x, y), data)

            # Verify it works
            @test itp((0.5, 1.0)) isa Float64
            @test itp((0.5, 1.0)) ≈ sin(0.5) * cos(1.0) atol=0.1
        end
    end

end

# ========================================
# Targeted Coverage: Val/NTuple deriv branches in ND batch eval
# ========================================
#
# These tests cover the `elseif deriv isa Val` and `else` (NTuple) branches
# in SoA/AoS batch evaluation for ConstantInterpolantND, LinearInterpolantND,
# and the oneshot APIs for linear/quadratic.
# They also cover core/utils.jl dispatch paths not exercised by default tests.

@testset "Val/NTuple deriv branches — ConstantInterpolantND batch" begin
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 2.0, 3.0]
    data = [10.0 * i + j for i in 1:3, j in 1:4]
    itp = constant_interp((x, y), data)

    xs = [0.5, 1.5, 0.5]
    ys = [0.5, 0.5, 1.5]
    ref = zeros(3)
    itp(ref, (xs, ys))

    @testset "SoA batch — Val(0) deriv" begin
        out = zeros(3)
        itp(out, (xs, ys); deriv=Val(0))
        @test out ≈ ref
    end

    @testset "SoA batch — NTuple (0,0) deriv" begin
        out = zeros(3)
        itp(out, (xs, ys); deriv=(0, 0))
        @test out ≈ ref
    end

    @testset "SoA batch — Val(1) deriv (zero for constant)" begin
        out = ones(3)
        itp(out, (xs, ys); deriv=Val(1))
        @test all(iszero, out)
    end

    @testset "SoA batch — NTuple (1,0) deriv (zero for constant)" begin
        out = ones(3)
        itp(out, (xs, ys); deriv=(1, 0))
        @test all(iszero, out)
    end

    queries = [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5)]
    ref_aos = zeros(3)
    itp(ref_aos, queries)

    @testset "AoS batch — Val(0) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=Val(0))
        @test out ≈ ref_aos
    end

    @testset "AoS batch — NTuple (0,0) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=(0, 0))
        @test out ≈ ref_aos
    end

    @testset "AoS batch — Val(1) deriv (zero for constant)" begin
        out = ones(3)
        itp(out, queries; deriv=Val(1))
        @test all(iszero, out)
    end

    @testset "AoS batch — NTuple (1,0) deriv (zero for constant)" begin
        out = ones(3)
        itp(out, queries; deriv=(1, 0))
        @test all(iszero, out)
    end
end

@testset "Val/NTuple deriv branches — LinearInterpolantND batch" begin
    x = collect(range(0.0, 1.0, 5))
    y = collect(range(0.0, 1.0, 5))
    data = [2xi + 3yj for xi in x, yj in y]
    itp = linear_interp((x, y), data)

    xs = [0.25, 0.75, 0.5]
    ys = [0.25, 0.75, 0.5]
    ref = zeros(3)
    itp(ref, (xs, ys))

    @testset "SoA batch — Val(0) deriv" begin
        out = zeros(3)
        itp(out, (xs, ys); deriv=Val(0))
        @test out ≈ ref
    end

    @testset "SoA batch — NTuple (0,0) deriv" begin
        out = zeros(3)
        itp(out, (xs, ys); deriv=(0, 0))
        @test out ≈ ref
    end

    @testset "SoA batch — Val(2) deriv (zero for linear)" begin
        out = ones(3)
        itp(out, (xs, ys); deriv=Val(2))
        @test all(iszero, out)
    end

    @testset "SoA batch — NTuple (2,0) deriv (zero for linear)" begin
        out = ones(3)
        itp(out, (xs, ys); deriv=(2, 0))
        @test all(iszero, out)
    end

    queries = [(0.25, 0.25), (0.75, 0.75), (0.5, 0.5)]
    ref_aos = zeros(3)
    itp(ref_aos, queries)

    @testset "AoS batch — Val(0) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=Val(0))
        @test out ≈ ref_aos
    end

    @testset "AoS batch — NTuple (0,0) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=(0, 0))
        @test out ≈ ref_aos
    end

    @testset "AoS batch — Val((1,0)) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=Val((1, 0)))
        @test all(≈(2.0, atol=1e-10), out)  # ∂/∂x(2x+3y) = 2
    end

    @testset "AoS batch — NTuple (2,0) deriv (zero for linear)" begin
        out = ones(3)
        itp(out, queries; deriv=(2, 0))
        @test all(iszero, out)
    end
end

@testset "Val/NTuple deriv branches — linear_interp oneshot" begin
    grids = (collect(range(0.0, 1.0, 6)), collect(range(0.0, 1.0, 6)))
    data = [2xi + 3yj for xi in grids[1], yj in grids[2]]

    @testset "scalar — NTuple (1,0) deriv (else branch)" begin
        result = linear_interp(grids, data, (0.5, 0.5); deriv=(1, 0))
        @test result ≈ 2.0 atol=1e-10
    end

    @testset "scalar — NTuple (0,1) deriv (else branch)" begin
        result = linear_interp(grids, data, (0.5, 0.5); deriv=(0, 1))
        @test result ≈ 3.0 atol=1e-10
    end

    @testset "linear_interp! SoA — Val(0) deriv (elseif branch)" begin
        xs = [0.25, 0.75]
        ys = [0.25, 0.75]
        out = zeros(2)
        linear_interp!(out, grids, data, (xs, ys); deriv=Val(0))
        ref = [2xi + 3yj for (xi, yj) in zip(xs, ys)]
        @test out ≈ ref atol=1e-10
    end

    @testset "linear_interp! SoA — NTuple (0,0) deriv (else branch)" begin
        xs = [0.25, 0.75]
        ys = [0.25, 0.75]
        out = zeros(2)
        linear_interp!(out, grids, data, (xs, ys); deriv=(0, 0))
        ref = [2xi + 3yj for (xi, yj) in zip(xs, ys)]
        @test out ≈ ref atol=1e-10
    end

    @testset "linear_interp! AoS — Val(0) deriv (elseif branch)" begin
        queries = [(0.25, 0.25), (0.75, 0.75)]
        out = zeros(2)
        linear_interp!(out, grids, data, queries; deriv=Val(0))
        ref = [2xi + 3yj for (xi, yj) in queries]
        @test out ≈ ref atol=1e-10
    end

    @testset "linear_interp! AoS — NTuple (0,0) deriv (else branch)" begin
        queries = [(0.25, 0.25), (0.75, 0.75)]
        out = zeros(2)
        linear_interp!(out, grids, data, queries; deriv=(0, 0))
        ref = [2xi + 3yj for (xi, yj) in queries]
        @test out ≈ ref atol=1e-10
    end
end

@testset "Val/NTuple deriv branches — quadratic_interp oneshot" begin
    grids = (collect(range(0.0, 1.0, 7)), collect(range(0.0, 1.0, 7)))
    data = [2xi + 3yj for xi in grids[1], yj in grids[2]]

    @testset "scalar — NTuple (1,0) deriv (else branch)" begin
        result = quadratic_interp(grids, data, (0.5, 0.5); deriv=(1, 0))
        @test result ≈ 2.0 atol=1e-6
    end

    @testset "scalar — NTuple (0,1) deriv (else branch)" begin
        result = quadratic_interp(grids, data, (0.5, 0.5); deriv=(0, 1))
        @test result ≈ 3.0 atol=1e-6
    end

    @testset "quadratic_interp! SoA — Val(0) deriv (elseif branch)" begin
        xs = [0.25, 0.75]
        ys = [0.25, 0.75]
        out = zeros(2)
        quadratic_interp!(out, grids, data, (xs, ys); deriv=Val(0))
        ref = [2xi + 3yj for (xi, yj) in zip(xs, ys)]
        @test out ≈ ref atol=1e-6
    end

    @testset "quadratic_interp! SoA — NTuple (0,0) deriv (else branch)" begin
        xs = [0.25, 0.75]
        ys = [0.25, 0.75]
        out = zeros(2)
        quadratic_interp!(out, grids, data, (xs, ys); deriv=(0, 0))
        ref = [2xi + 3yj for (xi, yj) in zip(xs, ys)]
        @test out ≈ ref atol=1e-6
    end

    @testset "quadratic_interp! AoS — Val(0) deriv (elseif branch)" begin
        queries = [(0.25, 0.25), (0.75, 0.75)]
        out = zeros(2)
        quadratic_interp!(out, grids, data, queries; deriv=Val(0))
        ref = [2xi + 3yj for (xi, yj) in queries]
        @test out ≈ ref atol=1e-6
    end

    @testset "quadratic_interp! AoS — NTuple (0,0) deriv (else branch)" begin
        queries = [(0.25, 0.25), (0.75, 0.75)]
        out = zeros(2)
        quadratic_interp!(out, grids, data, queries; deriv=(0, 0))
        ref = [2xi + 3yj for (xi, yj) in queries]
        @test out ≈ ref atol=1e-6
    end
end

@testset "core/utils.jl — @_dispatch_extrap_nd paths" begin
    grids = (collect(range(0.0, 1.0, 6)), collect(range(0.0, 1.0, 6)))
    data = [xi + yj for xi in grids[1], yj in grids[2]]

    @testset "fast path 1: :wrap extrap with bcs=nothing (linear oneshot)" begin
        # Exercises the `ntuple(_ -> Val(:wrap), valn)` branch (fast path 1 :wrap)
        result = linear_interp(grids, data, (0.5, 0.5); extrap=:wrap)
        @test result ≈ 1.0 atol=1e-10
    end

    @testset "fallback: non-uniform extraps with bcs=nothing (linear oneshot)" begin
        # Exercises _resolve_mixed_extrap_vals(extraps, ::Nothing) + fallback path
        result = linear_interp(grids, data, (0.5, 0.5); extrap=(:none, :constant))
        @test result ≈ 1.0 atol=1e-10
    end

    @testset "fast path 3: :wrap extrap with mixed BCs (cubic oneshot)" begin
        # bc=(PeriodicBC(), NaturalBC()) + extrap=:wrap → fast path 3 :wrap branch
        # (some periodic, not all → not fast path 2; uniform extrap → fast path 3)
        x = collect(range(0.0, 2π, 9))
        y = collect(range(0.0, 1.0, 9))
        data_p = [cos(xi) + yj for xi in x, yj in y]
        data_p[end, :] = data_p[1, :]  # ensure periodicity in x

        # Query is interior: cos(π/2) + 0.5 ≈ 0.5
        result = cubic_interp((x, y), data_p, (π/2, 0.5);
            bc=(PeriodicBC(), NaturalBC()), extrap=:wrap)
        @test result ≈ 0.5 atol=0.01
    end

    @testset "fallback: non-uniform extraps with mixed BCs (cubic oneshot)" begin
        # bc=(PeriodicBC(), NaturalBC()) + extrap=(:none,:constant) → fallback path
        # Exercises _resolve_mixed_extrap_vals(extraps, bcs::NTuple{N,AbstractBC})
        x = collect(range(0.0, 2π, 9))
        y = collect(range(0.0, 1.0, 9))
        data_p = [cos(xi) + yj for xi in x, yj in y]
        data_p[end, :] = data_p[1, :]

        # Query is interior: cos(π/2) + 0.5 ≈ 0.5 (extrap mode doesn't affect interior)
        result = cubic_interp((x, y), data_p, (π/2, 0.5);
            bc=(PeriodicBC(), NaturalBC()), extrap=(:none, :constant))
        @test result ≈ 0.5 atol=0.01
    end
end

@testset "core/utils.jl — @_dispatch_side_nd mixed-sides fallback" begin
    # Mixed side=(:left, :right) triggers the _to_side_vals fallback in @_dispatch_side_nd
    # (the `else` branch when _is_uniform_side returns false)
    grids = (collect(range(0.0, 2.0, 4)), collect(range(0.0, 3.0, 5)))
    data = [Float64(10i + j) for i in 1:4, j in 1:5]

    result_left  = constant_interp(grids, data, (0.5, 0.5); side=:left)
    result_right = constant_interp(grids, data, (0.5, 0.5); side=:right)
    result_mixed = constant_interp(grids, data, (0.5, 0.5); side=(:left, :right))

    @test result_mixed isa Float64
    @test result_mixed != result_left   # x picks left, y picks right corner
    @test result_mixed != result_right  # different from uniform-right
end
