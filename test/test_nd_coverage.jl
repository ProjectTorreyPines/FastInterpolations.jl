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
    _validate_nd_grids,
    _promote_grid_eltype,
    _convert_grids_typed,
    _create_spacings_typed,
    _throw_ndims_mismatch,
    # BC types
    ZeroCurvBC,
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
            _differentiate_nd_along_dim!(out1, data, x, ZeroCurvBC(), 1)

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
            _differentiate_nd_along_dim!(out2, data, y, ZeroCurvBC(), 2)

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
            @test_throws ArgumentError _differentiate_nd_along_dim!(out, data, x, ZeroCurvBC(), 0)
            @test_throws ArgumentError _differentiate_nd_along_dim!(out, data, x, ZeroCurvBC(), 3)

            # Size mismatch between out and data
            out_wrong = rand(4, 4)
            @test_throws DimensionMismatch _differentiate_nd_along_dim!(out_wrong, data, x, ZeroCurvBC(), 1)

            # Grid length mismatch
            wrong_grid = collect(range(0.0, 1.0, 6))  # 6 points but dim 1 has 5
            @test_throws DimensionMismatch _differentiate_nd_along_dim!(out, data, wrong_grid, ZeroCurvBC(), 1)
        end

        @testset "_get_effective_bc edge cases" begin
            grid_short = collect(1.0:3.0)  # 3 points, not enough for CubicFit
            grid_long = collect(1.0:10.0)   # 10 points

            # p_src == 1: always return original BC
            @test _get_effective_bc(ZeroCurvBC(), 1, grid_long) isa ZeroCurvBC
            @test _get_effective_bc(PeriodicBC(), 1, grid_long) isa PeriodicBC

            # p_src > 1 with PeriodicBC: propagate periodic
            @test _get_effective_bc(PeriodicBC(), 2, grid_long) isa PeriodicBC

            # p_src > 1 with short grid (< 4 points): fallback to ZeroCurvBC
            @test _get_effective_bc(ZeroCurvBC(), 2, grid_short) isa ZeroCurvBC

            # p_src > 1 with long grid: use CubicFit
            @test _get_effective_bc(ZeroCurvBC(), 2, grid_long) isa CubicFit
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
            @test _extrap(itp, Val(1)) isa FastInterpolations.AbstractExtrap
            @test _extrap(itp, Val(2)) isa FastInterpolations.AbstractExtrap

            # Test _search accessor
            @test _search(itp, Val(1)) isa FastInterpolations.AbstractSearchPolicy
            @test _search(itp, Val(2)) isa FastInterpolations.AbstractSearchPolicy
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
            # Correct size should work (3-arg form: extrap, bcs, Val(N))
            @test _resolve_extrap_nd(NoExtrap(), nothing, Val(2), Float64) == (NoExtrap(), NoExtrap())
            @test _resolve_extrap_nd((NoExtrap(), ClampedExtrap()), nothing, Val(2), Float64) == (NoExtrap(), ClampedExtrap())

            # Wrong size should throw
            @test_throws ArgumentError _resolve_extrap_nd((NoExtrap(),), nothing, Val(2), Float64)  # 1 element for 2D
            @test_throws ArgumentError _resolve_extrap_nd((NoExtrap(), NoExtrap(), NoExtrap()), nothing, Val(2), Float64)  # 3 for 2D
        end

        @testset "_resolve_search_nd wrong-sized tuple error" begin
            # Note: Search policy types are BinarySearch, LinearSearch, etc. (not exported)
            BinarySearch = FastInterpolations.BinarySearch
            LinearSearch = FastInterpolations.LinearSearch
            bs = BinarySearch()
            ls = LinearSearch()

            # Correct size should work
            @test _resolve_search_nd(bs, Val(2)) == (bs, bs)
            @test _resolve_search_nd((bs, ls), Val(2)) == (bs, ls)

            # Wrong size should throw
            @test_throws ArgumentError _resolve_search_nd((bs,), Val(2))
            @test_throws ArgumentError _resolve_search_nd((bs, bs, bs), Val(2))
        end

        @testset "_resolve_bcs_nd wrong-sized tuple error" begin
            # Correct size should work
            @test _resolve_bcs_nd(ZeroCurvBC(), Val(2)) == (ZeroCurvBC(), ZeroCurvBC())
            @test _resolve_bcs_nd((ZeroCurvBC(), PeriodicBC()), Val(2)) == (ZeroCurvBC(), PeriodicBC())

            # Wrong size should throw
            @test_throws ArgumentError _resolve_bcs_nd((ZeroCurvBC(),), Val(2))
            @test_throws ArgumentError _resolve_bcs_nd((ZeroCurvBC(), ZeroCurvBC(), ZeroCurvBC()), Val(2))
        end

        @testset "DerivOp constructors" begin
            # 1D constructors
            @test DerivOp(0) === DerivOp{0}()
            @test DerivOp(1) === DerivOp{1}()
            @test DerivOp(2) === DerivOp{2}()
            @test DerivOp(3) === DerivOp{3}()

            # ND constructors return tuple
            @test DerivOp(1, 0) == (DerivOp{1}(), DerivOp{0}())
            @test DerivOp(0, 1) == (DerivOp{0}(), DerivOp{1}())
            @test DerivOp(2, 1) == (DerivOp{2}(), DerivOp{1}())
            @test DerivOp(0, 0, 0) == (DerivOp{0}(), DerivOp{0}(), DerivOp{0}())
            @test DerivOp(1, 2, 3) == (DerivOp{1}(), DerivOp{2}(), DerivOp{3}())

            # deriv_order accessor
            @test deriv_order(DerivOp{0}()) == 0
            @test deriv_order(DerivOp{1}()) == 1
            @test deriv_order(DerivOp{2}()) == 2
            @test deriv_order(DerivOp{3}()) == 3
        end

        @testset "_throw_ndims_mismatch" begin
            @test_throws DimensionMismatch _throw_ndims_mismatch("query vectors", 2, 3)
            # Verify message content
            try
                _throw_ndims_mismatch("derivative orders", 3, 1)
            catch e
                @test e isa DimensionMismatch
                @test occursin("expected 3 derivative orders, got 1", e.msg)
            end
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
    @testset "quadratic_nd_interpolant.jl — lazy BC normalization" begin
        # Quadratic ND now uses _resolve_bcs_nd (shared with cubic) for BC broadcasting.
        # AbstractBC flows through lazily — normalization happens in _slope_1d_quadratic!.

        @testset "_resolve_bcs_nd with quadratic-compatible BCs" begin
            # Single QuadraticBC → broadcast (same as cubic)
            bcs = _resolve_bcs_nd(Right(QuadraticFit()), Val(2))
            @test length(bcs) == 2
            @test all(b -> b isa Right, bcs)

            # NTuple pass-through
            bcs_tuple = (Left(QuadraticFit()), Right(QuadraticFit()))
            @test _resolve_bcs_nd(bcs_tuple, Val(2)) === bcs_tuple

            # ZeroCurvBC stays raw (lazy normalization — NOT eagerly converted)
            bcs_nat = _resolve_bcs_nd(ZeroCurvBC(), Val(2))
            @test length(bcs_nat) == 2
            @test all(b -> b isa ZeroCurvBC, bcs_nat)

            # PolyFit stays raw
            bcs_poly = _resolve_bcs_nd(CubicFit(), Val(2))
            @test length(bcs_poly) == 2
            @test all(b -> b isa CubicFit, bcs_poly)

            # Heterogeneous AbstractBC tuple
            bcs_hetero = _resolve_bcs_nd((ZeroCurvBC(), CubicFit()), Val(2))
            @test length(bcs_hetero) == 2
            @test bcs_hetero[1] isa ZeroCurvBC
            @test bcs_hetero[2] isa CubicFit
        end

        @testset "_normalize_bc for quadratic BC types" begin
            import FastInterpolations: _normalize_bc
            # Left/Right promote inner PointBC values to Tv
            bc_l = _normalize_bc(Left(Deriv2(5)), Float64)
            @test bc_l isa Left
            @test bc_l.bc.val === 5.0

            bc_r = _normalize_bc(Right(Deriv1(3)), Float32)
            @test bc_r isa Right
            @test bc_r.bc.val === 3.0f0

            # MinCurvFit is passthrough
            @test _normalize_bc(MinCurvFit(), Float64) === MinCurvFit()

            # PolyFit inside Left/Right is passthrough (no value to promote)
            bc_p = _normalize_bc(Right(QuadraticFit()), Float64)
            @test bc_p isa Right
            @test bc_p.bc isa QuadraticFit
        end

        @testset "_normalize_bc value-based overloads" begin
            import FastInterpolations: _normalize_bc

            # ZeroCurvBC: value-based uses 0 * sample instead of zero(Tv)
            bc_zc = _normalize_bc(ZeroCurvBC(), 1.0)
            @test bc_zc isa BCPair
            @test bc_zc.left isa Deriv2
            @test bc_zc.left.val === 0.0
            @test bc_zc.right.val === 0.0

            # ZeroSlopeBC: same pattern
            bc_zs = _normalize_bc(ZeroSlopeBC(), 1.0f0)
            @test bc_zs isa BCPair
            @test bc_zs.left isa Deriv1
            @test bc_zs.left.val === 0.0f0

            # Generic fallback: value → typeof(value) → type-based method
            bc_left = _normalize_bc(Left(Deriv2(5)), 1.0)
            @test bc_left isa Left
            @test bc_left.bc.val === 5.0

            # SVector: 0 * SVector produces correct zero vector
            using StaticArrays
            sv = SVector(1.0, 2.0, 3.0)
            bc_sv = _normalize_bc(ZeroCurvBC(), sv)
            @test bc_sv.left.val === SVector(0.0, 0.0, 0.0)
            @test bc_sv.right.val === SVector(0.0, 0.0, 0.0)
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
            itp = cubic_interp((x, y), data, bc=(PeriodicBC(), ZeroCurvBC()))
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
# Targeted Coverage: DerivOp deriv paths in ND batch eval
# ========================================
#
# These tests cover the DerivOp tuple path in SoA/AoS batch evaluation
# for ConstantInterpolantND, LinearInterpolantND, and the oneshot APIs
# for linear/quadratic. Distinct from the `deriv::Int` broadcast path.

@testset "DerivOp deriv paths — ConstantInterpolantND batch" begin
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 2.0, 3.0]
    data = [10.0 * i + j for i in 1:3, j in 1:4]
    itp = constant_interp((x, y), data)

    xs = [0.5, 1.5, 0.5]
    ys = [0.5, 0.5, 1.5]
    ref = zeros(3)
    itp(ref, (xs, ys))

    @testset "SoA batch — Int(0) deriv" begin
        out = zeros(3)
        itp(out, (xs, ys); deriv=DerivOp(0, 0))
        @test out ≈ ref
    end

    @testset "SoA batch — DerivOp(0,0) deriv" begin
        out = zeros(3)
        itp(out, (xs, ys); deriv=DerivOp(0, 0))
        @test out ≈ ref
    end

    @testset "SoA batch — Int(1) deriv (zero for constant)" begin
        out = ones(3)
        itp(out, (xs, ys); deriv=DerivOp(1, 1))
        @test all(iszero, out)
    end

    @testset "SoA batch — DerivOp(1,0) deriv (zero for constant)" begin
        out = ones(3)
        itp(out, (xs, ys); deriv=DerivOp(1, 0))
        @test all(iszero, out)
    end

    queries = [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5)]
    ref_aos = zeros(3)
    itp(ref_aos, queries)

    @testset "AoS batch — Int(0) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=DerivOp(0, 0))
        @test out ≈ ref_aos
    end

    @testset "AoS batch — DerivOp(0,0) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=DerivOp(0, 0))
        @test out ≈ ref_aos
    end

    @testset "AoS batch — Int(1) deriv (zero for constant)" begin
        out = ones(3)
        itp(out, queries; deriv=DerivOp(1, 1))
        @test all(iszero, out)
    end

    @testset "AoS batch — DerivOp(1,0) deriv (zero for constant)" begin
        out = ones(3)
        itp(out, queries; deriv=DerivOp(1, 0))
        @test all(iszero, out)
    end
end

@testset "DerivOp deriv paths — LinearInterpolantND batch" begin
    x = collect(range(0.0, 1.0, 5))
    y = collect(range(0.0, 1.0, 5))
    data = [2xi + 3yj for xi in x, yj in y]
    itp = linear_interp((x, y), data)

    xs = [0.25, 0.75, 0.5]
    ys = [0.25, 0.75, 0.5]
    ref = zeros(3)
    itp(ref, (xs, ys))

    @testset "SoA batch — Int(0) deriv" begin
        out = zeros(3)
        itp(out, (xs, ys); deriv=DerivOp(0, 0))
        @test out ≈ ref
    end

    @testset "SoA batch — DerivOp(0,0) deriv" begin
        out = zeros(3)
        itp(out, (xs, ys); deriv=DerivOp(0, 0))
        @test out ≈ ref
    end

    @testset "SoA batch — Int(2) deriv (zero for linear)" begin
        out = ones(3)
        itp(out, (xs, ys); deriv=DerivOp(2, 2))
        @test all(iszero, out)
    end

    @testset "SoA batch — DerivOp(2,0) deriv (zero for linear)" begin
        out = ones(3)
        itp(out, (xs, ys); deriv=DerivOp(2, 0))
        @test all(iszero, out)
    end

    queries = [(0.25, 0.25), (0.75, 0.75), (0.5, 0.5)]
    ref_aos = zeros(3)
    itp(ref_aos, queries)

    @testset "AoS batch — Int(0) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=DerivOp(0, 0))
        @test out ≈ ref_aos
    end

    @testset "AoS batch — DerivOp(0,0) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=DerivOp(0, 0))
        @test out ≈ ref_aos
    end

    @testset "AoS batch — DerivOp(1,0) deriv" begin
        out = zeros(3)
        itp(out, queries; deriv=DerivOp(1, 0))
        @test all(≈(2.0, atol=1e-10), out)  # ∂/∂x(2x+3y) = 2
    end

    @testset "AoS batch — DerivOp(2,0) deriv (zero for linear)" begin
        out = ones(3)
        itp(out, queries; deriv=DerivOp(2, 0))
        @test all(iszero, out)
    end
end

@testset "DerivOp deriv paths — linear_interp oneshot" begin
    grids = (collect(range(0.0, 1.0, 6)), collect(range(0.0, 1.0, 6)))
    data = [2xi + 3yj for xi in grids[1], yj in grids[2]]

    @testset "scalar — DerivOp(1,0) deriv (else branch)" begin
        result = linear_interp(grids, data, (0.5, 0.5); deriv=DerivOp(1, 0))
        @test result ≈ 2.0 atol=1e-10
    end

    @testset "scalar — NTuple (0,1) deriv (else branch)" begin
        result = linear_interp(grids, data, (0.5, 0.5); deriv=DerivOp(0, 1))
        @test result ≈ 3.0 atol=1e-10
    end

    @testset "linear_interp! SoA — Int(0) deriv (elseif branch)" begin
        xs = [0.25, 0.75]
        ys = [0.25, 0.75]
        out = zeros(2)
        linear_interp!(out, grids, data, (xs, ys); deriv=DerivOp(0, 0))
        ref = [2xi + 3yj for (xi, yj) in zip(xs, ys)]
        @test out ≈ ref atol=1e-10
    end

    @testset "linear_interp! SoA — DerivOp(0,0) deriv (else branch)" begin
        xs = [0.25, 0.75]
        ys = [0.25, 0.75]
        out = zeros(2)
        linear_interp!(out, grids, data, (xs, ys); deriv=DerivOp(0, 0))
        ref = [2xi + 3yj for (xi, yj) in zip(xs, ys)]
        @test out ≈ ref atol=1e-10
    end

    @testset "linear_interp! AoS — Int(0) deriv (elseif branch)" begin
        queries = [(0.25, 0.25), (0.75, 0.75)]
        out = zeros(2)
        linear_interp!(out, grids, data, queries; deriv=DerivOp(0, 0))
        ref = [2xi + 3yj for (xi, yj) in queries]
        @test out ≈ ref atol=1e-10
    end

    @testset "linear_interp! AoS — DerivOp(0,0) deriv (else branch)" begin
        queries = [(0.25, 0.25), (0.75, 0.75)]
        out = zeros(2)
        linear_interp!(out, grids, data, queries; deriv=DerivOp(0, 0))
        ref = [2xi + 3yj for (xi, yj) in queries]
        @test out ≈ ref atol=1e-10
    end
end

@testset "DerivOp deriv paths — quadratic_interp oneshot" begin
    grids = (collect(range(0.0, 1.0, 7)), collect(range(0.0, 1.0, 7)))
    data = [2xi + 3yj for xi in grids[1], yj in grids[2]]

    @testset "scalar — DerivOp(1,0) deriv (else branch)" begin
        result = quadratic_interp(grids, data, (0.5, 0.5); deriv=DerivOp(1, 0))
        @test result ≈ 2.0 atol=1e-6
    end

    @testset "scalar — NTuple (0,1) deriv (else branch)" begin
        result = quadratic_interp(grids, data, (0.5, 0.5); deriv=DerivOp(0, 1))
        @test result ≈ 3.0 atol=1e-6
    end

    @testset "quadratic_interp! SoA — Int(0) deriv (elseif branch)" begin
        xs = [0.25, 0.75]
        ys = [0.25, 0.75]
        out = zeros(2)
        quadratic_interp!(out, grids, data, (xs, ys); deriv=DerivOp(0, 0))
        ref = [2xi + 3yj for (xi, yj) in zip(xs, ys)]
        @test out ≈ ref atol=1e-6
    end

    @testset "quadratic_interp! SoA — DerivOp(0,0) deriv (else branch)" begin
        xs = [0.25, 0.75]
        ys = [0.25, 0.75]
        out = zeros(2)
        quadratic_interp!(out, grids, data, (xs, ys); deriv=DerivOp(0, 0))
        ref = [2xi + 3yj for (xi, yj) in zip(xs, ys)]
        @test out ≈ ref atol=1e-6
    end

    @testset "quadratic_interp! AoS — Int(0) deriv (elseif branch)" begin
        queries = [(0.25, 0.25), (0.75, 0.75)]
        out = zeros(2)
        quadratic_interp!(out, grids, data, queries; deriv=DerivOp(0, 0))
        ref = [2xi + 3yj for (xi, yj) in queries]
        @test out ≈ ref atol=1e-6
    end

    @testset "quadratic_interp! AoS — DerivOp(0,0) deriv (else branch)" begin
        queries = [(0.25, 0.25), (0.75, 0.75)]
        out = zeros(2)
        quadratic_interp!(out, grids, data, queries; deriv=DerivOp(0, 0))
        ref = [2xi + 3yj for (xi, yj) in queries]
        @test out ≈ ref atol=1e-6
    end
end

@testset "ND extrapolation dispatch paths" begin
    grids = (collect(range(0.0, 1.0, 6)), collect(range(0.0, 1.0, 6)))
    data = [xi + yj for xi in grids[1], yj in grids[2]]

    @testset "fast path 1: WrapExtrap() extrap with bcs=nothing (linear oneshot)" begin
        # Uniform WrapExtrap on all dims — exercises periodic wrap fast path
        result = linear_interp(grids, data, (0.5, 0.5); extrap=WrapExtrap())
        @test result ≈ 1.0 atol=1e-10
    end

    @testset "fallback: non-uniform extraps with bcs=nothing (linear oneshot)" begin
        # Mixed extrap types per dim — exercises per-dim extrap dispatch fallback
        result = linear_interp(grids, data, (0.5, 0.5); extrap=(NoExtrap(), ConstExtrap()))
        @test result ≈ 1.0 atol=1e-10
    end

    @testset "fast path 3: WrapExtrap() extrap with mixed BCs (cubic oneshot)" begin
        # bc=(PeriodicBC(), ZeroCurvBC()) + extrap=WrapExtrap()
        # Some periodic, not all → not fast path 2; uniform extrap → fast path 3
        x = collect(range(0.0, 2π, 9))
        y = collect(range(0.0, 1.0, 9))
        data_p = [cos(xi) + yj for xi in x, yj in y]
        data_p[end, :] = data_p[1, :]  # ensure periodicity in x

        # Query is interior: cos(π/2) + 0.5 ≈ 0.5
        result = cubic_interp((x, y), data_p, (π/2, 0.5);
            bc=(PeriodicBC(), ZeroCurvBC()), extrap=WrapExtrap())
        @test result ≈ 0.5 atol=0.01
    end

    @testset "fallback: non-uniform extraps with mixed BCs (cubic oneshot)" begin
        # bc=(PeriodicBC(), ZeroCurvBC()) + extrap=(NoExtrap(),ConstExtrap())
        # Mixed extrap types per dim — exercises per-dim extrap dispatch fallback
        x = collect(range(0.0, 2π, 9))
        y = collect(range(0.0, 1.0, 9))
        data_p = [cos(xi) + yj for xi in x, yj in y]
        data_p[end, :] = data_p[1, :]

        # Query is interior: cos(π/2) + 0.5 ≈ 0.5 (extrap mode doesn't affect interior)
        result = cubic_interp((x, y), data_p, (π/2, 0.5);
            bc=(PeriodicBC(), ZeroCurvBC()), extrap=(NoExtrap(), ConstExtrap()))
        @test result ≈ 0.5 atol=0.01
    end
end

@testset "mixed-sides fallback" begin
    # Mixed side=(LeftSide(), RightSide()) 
    grids = (collect(range(0.0, 2.0, 4)), collect(range(0.0, 3.0, 5)))
    data = [Float64(10i + j) for i in 1:4, j in 1:5]

    result_left  = constant_interp(grids, data, (0.5, 0.5); side=LeftSide())
    result_right = constant_interp(grids, data, (0.5, 0.5); side=RightSide())
    result_mixed = constant_interp(grids, data, (0.5, 0.5); side=(LeftSide(), RightSide()))

    @test result_mixed isa Float64
    @test result_mixed != result_left   # x picks left, y picks right corner
    @test result_mixed != result_right  # different from uniform-right
end
