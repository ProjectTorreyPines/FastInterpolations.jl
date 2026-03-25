using Test
using FastInterpolations
using FastInterpolations: get_task_local_pool

# Allocation threshold (bytes) — tolerates minor LTS/GC overhead.
# On latest Julia, scalar oneshot is truly zero-alloc; LTS may show ≤64 bytes.
# Guarded for standalone execution (runtests.jl defines this globally).
if !@isdefined(ND_ALLOC_THRESHOLD)
    const ND_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240
end

@testset "Cubic ND One-Shot (Pool-Based)" begin

    # ========================================
    # Correctness: One-shot vs Interpolant
    # ========================================

    @testset "Scalar one-shot matches Interpolant (2D, ZeroCurvBC)" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        for (xq, yq) in [(1.0, 0.5), (3.0, 1.2), (0.1, 2.8)]
            val_oneshot = cubic_interp((x, y), data, (xq, yq))
            val_interp = itp((xq, yq))
            @test val_oneshot ≈ val_interp atol = 1.0e-14
        end
    end

    @testset "Scalar one-shot matches Interpolant (2D, CubicFit)" begin
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^3 + yj^2 for xi in x, yj in y]

        itp = cubic_interp((x, y), data; bc = CubicFit())
        val_oneshot = cubic_interp((x, y), data, (1.0, 0.5); bc = CubicFit())
        val_interp = itp((1.0, 0.5))
        @test val_oneshot ≈ val_interp atol = 1.0e-14
    end

    @testset "Scalar one-shot matches Interpolant (3D, ZeroCurvBC)" begin
        x = range(0.0, 2.0, 10)
        y = range(0.0, 1.0, 8)
        z = range(0.0, 3.0, 6)
        data = [xi^2 + yj + zk for xi in x, yj in y, zk in z]

        itp = cubic_interp((x, y, z), data)
        val_oneshot = cubic_interp((x, y, z), data, (1.0, 0.5, 1.5))
        val_interp = itp((1.0, 0.5, 1.5))
        @test val_oneshot ≈ val_interp atol = 1.0e-14
    end

    @testset "Derivative one-shot matches Interpolant" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)
        query = (1.5, 0.8)

        # deriv=DerivOp(0) (value)
        @test cubic_interp((x, y), data, query; deriv = DerivOp(0, 0)) ≈ itp(query; deriv = DerivOp(0, 0)) atol = 1.0e-14

        # deriv=DerivOp(1) (all first derivatives)
        @test cubic_interp((x, y), data, query; deriv = DerivOp(1, 1)) ≈ itp(query; deriv = DerivOp(1, 1)) atol = 1.0e-14

        # Mixed partial: ∂f/∂x
        @test cubic_interp((x, y), data, query; deriv = DerivOp(1, 0)) ≈ itp(query; deriv = DerivOp(1, 0)) atol = 1.0e-14

        # Mixed partial: ∂f/∂y
        @test cubic_interp((x, y), data, query; deriv = DerivOp(0, 1)) ≈ itp(query; deriv = DerivOp(0, 1)) atol = 1.0e-14

        # Mixed partial: ∂²f/∂x∂y
        @test cubic_interp((x, y), data, query; deriv = DerivOp(1, 1)) ≈ itp(query; deriv = DerivOp(1, 1)) atol = 1.0e-14
    end

    @testset "SoA batch one-shot matches Interpolant" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)
        xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
        yqs = [0.2, 0.4, 0.6, 0.8, 1.0]

        vals_oneshot = cubic_interp((x, y), data, (xqs, yqs))
        for k in 1:5
            @test vals_oneshot[k] ≈ itp((xqs[k], yqs[k])) atol = 1.0e-14
        end
    end

    @testset "AoS batch one-shot matches Interpolant" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)
        points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6), (2.0, 0.8), (3.0, 1.0)]

        vals_oneshot = cubic_interp((x, y), data, points)
        for k in 1:5
            @test vals_oneshot[k] ≈ itp(points[k]) atol = 1.0e-14
        end
    end

    @testset "Complex-valued one-shot" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) + im * cos(xi) * sin(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)
        query = (1.5, 0.8)
        val_oneshot = cubic_interp((x, y), data, query)
        val_interp = itp(query)
        @test val_oneshot isa ComplexF64
        @test val_oneshot ≈ val_interp atol = 1.0e-14
    end

    @testset "Heterogeneous grids (Range + Vector)" begin
        x = range(0.0, 2.0, 15)  # Range
        y = collect(range(0.0, 1.0, 10))  # Vector
        data = [xi^2 + yj for xi in x, yj in y]

        itp = cubic_interp((x, y), data; bc = CubicFit())
        val_oneshot = cubic_interp((x, y), data, (1.0, 0.5); bc = CubicFit())
        val_interp = itp((1.0, 0.5))
        @test val_oneshot ≈ val_interp atol = 1.0e-14
    end

    @testset "Extrapolation modes" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 10)
        data = [xi + yj for xi in x, yj in y]
        itp_const = cubic_interp((x, y), data; extrap = ClampExtrap())
        itp_ext = cubic_interp((x, y), data; extrap = ExtendExtrap())

        # Constant extrap
        @test cubic_interp((x, y), data, (1.0, 0.5); extrap = ClampExtrap()) ≈
            itp_const((1.0, 0.5)) atol = 1.0e-14

        # Extension extrap
        @test cubic_interp((x, y), data, (1.0, 0.5); extrap = ExtendExtrap()) ≈
            itp_ext((1.0, 0.5)) atol = 1.0e-14
    end

    # ========================================
    # Periodic BC
    # ========================================

    @testset "Periodic BC (inclusive)" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, 2π, 21)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        # Ensure periodicity: endpoints match for sin/cos on [0,2π]
        data[end, :] .= data[1, :]
        data[:, end] .= data[:, 1]

        itp = cubic_interp((x, y), data; bc = PeriodicBC())
        val_oneshot = cubic_interp((x, y), data, (1.5, 0.8); bc = PeriodicBC())
        val_interp = itp((1.5, 0.8))
        @test val_oneshot ≈ val_interp atol = 1.0e-14
    end

    @testset "Periodic BC (exclusive)" begin
        n = 20
        x = range(0.0, 2π, n + 1)[1:n]  # exclusive: no endpoint
        y = range(0.0, 2π, n + 1)[1:n]
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        bc = PeriodicBC(endpoint = :exclusive)
        itp = cubic_interp((x, y), data; bc = bc)
        val_oneshot = cubic_interp((x, y), data, (1.5, 0.8); bc = bc)
        val_interp = itp((1.5, 0.8))
        @test val_oneshot ≈ val_interp atol = 1.0e-14
    end

    @testset "Mixed periodic/non-periodic BCs" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, 1.0, 11)
        data = [sin(xi) * yj^2 for xi in x, yj in y]
        data[end, :] .= data[1, :]

        bc = (PeriodicBC(), ZeroCurvBC())
        itp = cubic_interp((x, y), data; bc = bc)
        val_oneshot = cubic_interp((x, y), data, (1.5, 0.5); bc = bc)
        val_interp = itp((1.5, 0.5))
        @test val_oneshot ≈ val_interp atol = 1.0e-14
    end

    # ========================================
    # Periodic BC Data Validation (one-shot path)
    # ========================================
    #
    # Verifies that the one-shot API validates periodic data integrity
    # (data[...,1,...] ≈ data[...,end,...]) before entering the pool-based kernel,
    # matching the behaviour of the CubicInterpolant constructor.

    @testset "Periodic BC data validation — one-shot" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, 2π, 21)

        # Canonical valid periodic data: pin endpoints to match
        data_ok = [sin(xi) * cos(yj) for xi in x, yj in y]
        data_ok[end, :] .= data_ok[1, :]
        data_ok[:, end] .= data_ok[:, 1]

        @testset "valid data passes (no error)" begin
            @test_nowarn cubic_interp((x, y), data_ok, (1.5, 0.8); bc = PeriodicBC())
        end

        @testset "non-periodic dim-1 throws ArgumentError (scalar)" begin
            data_bad = copy(data_ok)
            data_bad[1, 5] = 999.0  # data[1,5] ≠ data[end,5]
            @test_throws ArgumentError cubic_interp((x, y), data_bad, (1.5, 0.8); bc = PeriodicBC())
        end

        @testset "non-periodic dim-2 throws ArgumentError (mixed BCs)" begin
            data_bad = copy(data_ok)
            data_bad[3, end] = 999.0  # break dim-2 match while dim-1 is still ok
            bc_mixed = (ZeroCurvBC(), PeriodicBC())
            @test_throws ArgumentError cubic_interp((x, y), data_bad, (1.5, 0.8); bc = bc_mixed)
        end

        @testset "SoA batch also validates" begin
            data_bad = copy(data_ok)
            data_bad[1, 5] = 999.0
            @test_throws ArgumentError cubic_interp((x, y), data_bad, ([0.5, 1.0], [0.5, 1.0]); bc = PeriodicBC())
        end

        @testset "in-place (cubic_interp!) also validates" begin
            data_bad = copy(data_ok)
            data_bad[1, 5] = 999.0
            out = zeros(2)
            @test_throws ArgumentError cubic_interp!(out, (x, y), data_bad, ([0.5, 1.0], [0.5, 1.0]); bc = PeriodicBC())
        end

        @testset "exclusive PeriodicBC: no false positive on valid data" begin
            # For exclusive BC the endpoint is added by the pool extension, so
            # data[end] is NOT expected to match data[1] in the user-supplied array.
            n = 20
            xe = range(0.0, 2π, n + 1)[1:n]
            ye = range(0.0, 2π, n + 1)[1:n]
            data_excl = [sin(xi) * cos(yj) for xi in xe, yj in ye]
            # sin(xe[end]) ≈ sin(19π/10) ≠ 0 = sin(xe[1]) — valid exclusive input
            bc_excl = PeriodicBC(endpoint = :exclusive)
            @test_nowarn cubic_interp((xe, ye), data_excl, (1.5, 0.8); bc = bc_excl)
        end

        @testset "3D: validation across all periodic dims" begin
            x3 = range(0.0, 2π, 11)
            y3 = range(0.0, 2π, 11)
            z3 = range(0.0, 1.0, 6)
            data_3d = [sin(xi) * cos(yj) + zk for xi in x3, yj in y3, zk in z3]
            data_3d[end, :, :] .= data_3d[1, :, :]   # pin dim 1
            data_3d[:, end, :] .= data_3d[:, 1, :]   # pin dim 2
            bc3 = (PeriodicBC(), PeriodicBC(), ZeroCurvBC())

            @testset "valid 3D data passes" begin
                @test_nowarn cubic_interp((x3, y3, z3), data_3d, (1.5, 0.8, 0.5); bc = bc3)
            end

            @testset "broken dim-2 throws in 3D" begin
                data_3d_bad = copy(data_3d)
                data_3d_bad[2, end, 1] = 999.0  # break dim-2 match
                @test_throws ArgumentError cubic_interp((x3, y3, z3), data_3d_bad, (1.5, 0.8, 0.5); bc = bc3)
            end
        end
    end

    # ========================================
    # Allocation Tests
    # ========================================
    #
    # Each test uses a full function barrier: setup + warmup + @allocated
    # all inside one function. This avoids @testset-scope boxing artifacts.

    function _alloc_test_natural()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        cubic_interp((x, y), data, query)
        cubic_interp((x, y), data, query)
        @allocated cubic_interp((x, y), data, query)
    end

    function _alloc_test_natural_deriv()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        cubic_interp((x, y), data, query; deriv = DerivOp(1, 1))
        cubic_interp((x, y), data, query; deriv = DerivOp(1, 1))
        @allocated cubic_interp((x, y), data, query; deriv = DerivOp(1, 1))
    end

    function _alloc_test_cubicfit()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query; bc = CubicFit())
        cubic_interp((x, y), data, query; bc = CubicFit())
        @allocated cubic_interp((x, y), data, query; bc = CubicFit())
    end

    function _alloc_test_periodic_inclusive()
        x = range(0.0, 2π, 21)
        y = range(0.0, 2π, 21)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        data[end, :] .= data[1, :]
        data[:, end] .= data[:, 1]
        query = (1.5, 0.8)
        cubic_interp((x, y), data, query; bc = PeriodicBC())
        cubic_interp((x, y), data, query; bc = PeriodicBC())
        @allocated cubic_interp((x, y), data, query; bc = PeriodicBC())
    end

    function _alloc_test_periodic_exclusive()
        n = 20
        x = range(0.0, 2π, n + 1)[1:n]
        y = range(0.0, 2π, n + 1)[1:n]
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        bc = PeriodicBC(endpoint = :exclusive)
        cubic_interp((x, y), data, query; bc = bc)
        cubic_interp((x, y), data, query; bc = bc)
        @allocated cubic_interp((x, y), data, query; bc = bc)
    end

    @testset "Zero-alloc scalar one-shot (Range grids)" begin
        @test _alloc_test_natural() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc scalar one-shot with deriv (Range grids)" begin
        @test _alloc_test_natural_deriv() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc scalar one-shot (CubicFit BC, Range grids)" begin
        @test _alloc_test_cubicfit() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc scalar one-shot (Periodic BC inclusive, Range grids)" begin
        @test _alloc_test_periodic_inclusive() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc scalar one-shot (Periodic BC exclusive, Range grids)" begin
        @test _alloc_test_periodic_exclusive() <= ND_ALLOC_THRESHOLD
    end

    function _alloc_test_mixed_periodic()
        x = range(0.0, 2π, 21)
        y = range(0.0, 1.0, 11)
        data = [sin(xi) * yj^2 for xi in x, yj in y]
        data[end, :] .= data[1, :]
        query = (1.5, 0.5)
        bc = (PeriodicBC(), ZeroCurvBC())
        cubic_interp((x, y), data, query; bc = bc)
        cubic_interp((x, y), data, query; bc = bc)
        @allocated cubic_interp((x, y), data, query; bc = bc)
    end

    @testset "Zero-alloc scalar one-shot (Mixed periodic/ZeroCurvBC, Range grids)" begin
        # Heterogeneous BC tuple (PeriodicBC, ZeroCurvBC) may show ≤48 bytes
        # from validation path specialization — not a hot-path regression.
        @test _alloc_test_mixed_periodic() <= max(ND_ALLOC_THRESHOLD, 48)
    end

    # ========================================
    # Vector-Grid Allocation Tests
    # ========================================
    #
    # Pool-based spacing: VectorSpacing h/inv_h acquired from pool,
    # zero heap allocation for Vector grids after warmup.

    function _alloc_test_vector_natural()
        x = collect(range(0.0, 2.0, 15))
        y = collect(range(0.0, 1.0, 11))
        data = [xi^3 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query)
        cubic_interp((x, y), data, query)
        @allocated cubic_interp((x, y), data, query)
    end

    function _alloc_test_vector_cubicfit()
        x = collect(range(0.0, 2.0, 15))
        y = collect(range(0.0, 1.0, 11))
        data = [xi^3 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query; bc = CubicFit())
        cubic_interp((x, y), data, query; bc = CubicFit())
        @allocated cubic_interp((x, y), data, query; bc = CubicFit())
    end

    function _alloc_test_vector_deriv()
        x = collect(range(0.0, 2.0, 15))
        y = collect(range(0.0, 1.0, 11))
        data = [xi^3 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query; deriv = DerivOp(1, 0))
        cubic_interp((x, y), data, query; deriv = DerivOp(1, 0))
        @allocated cubic_interp((x, y), data, query; deriv = DerivOp(1, 0))
    end

    @testset "Zero-alloc scalar one-shot (Vector grids, ZeroCurvBC)" begin
        @test _alloc_test_vector_natural() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc scalar one-shot (Vector grids, CubicFit)" begin
        @test _alloc_test_vector_cubicfit() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc scalar one-shot (Vector grids, deriv)" begin
        @test _alloc_test_vector_deriv() <= ND_ALLOC_THRESHOLD
    end

    function _alloc_test_vector_3d()
        x = collect(range(0.0, 2.0, 10))
        y = collect(range(0.0, 1.0, 8))
        z = collect(range(0.0, 3.0, 6))
        data = [xi^3 + yj^2 + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        cubic_interp((x, y, z), data, query)
        cubic_interp((x, y, z), data, query)
        @allocated cubic_interp((x, y, z), data, query)
    end

    @testset "Zero-alloc scalar one-shot (3D Vector grids)" begin
        @test _alloc_test_vector_3d() <= ND_ALLOC_THRESHOLD
    end

    # ========================================
    # Mixed-Grid Allocation Tests (Range + Vector)
    # ========================================
    #
    # Heterogeneous grid tuples (ScalarSpacing + VectorSpacing) must be zero-allocation.
    # Catches ntuple closure boxing on heterogeneous inputs.

    function _alloc_test_mixed_2d()
        x = range(0.0, 2.0, 15)          # Range → ScalarSpacing
        y = collect(range(0.0, 1.0, 11)) # Vector → VectorSpacing
        data = [xi^3 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query)
        cubic_interp((x, y), data, query)
        @allocated cubic_interp((x, y), data, query)
    end

    function _alloc_test_mixed_3d()
        x = range(0.0, 2.0, 10)          # Range → ScalarSpacing
        y = collect(range(0.0, 1.0, 8))  # Vector → VectorSpacing
        z = range(0.0, 3.0, 6)           # Range → ScalarSpacing
        data = [xi^3 + yj^2 + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        cubic_interp((x, y, z), data, query)
        cubic_interp((x, y, z), data, query)
        @allocated cubic_interp((x, y, z), data, query)
    end

    @testset "Zero-alloc scalar one-shot (2D mixed grid: Range + Vector)" begin
        @test _alloc_test_mixed_2d() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc scalar one-shot (3D mixed grid: Range + Vector + Range)" begin
        @test _alloc_test_mixed_3d() <= ND_ALLOC_THRESHOLD
    end

    # ========================================
    # In-Place Batch Allocation Tests
    # ========================================
    #
    # In-place paths write into a pre-allocated output buffer.
    # These must be truly zero-allocation (only output + THRESHOLD).

    function _alloc_test_inplace_soa()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)
        xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
        yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
        out = Vector{Float64}(undef, 5)
        itp(out, (xqs, yqs))
        itp(out, (xqs, yqs))
        @allocated itp(out, (xqs, yqs))
    end

    function _alloc_test_inplace_aos()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)
        points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6), (2.0, 0.8), (3.0, 1.0)]
        out = Vector{Float64}(undef, 5)
        itp(out, points)
        itp(out, points)
        @allocated itp(out, points)
    end

    @testset "In-Place Batch Allocation Tests" begin
        @testset "in-place SoA batch (Range grids)" begin
            @test _alloc_test_inplace_soa() <= ND_ALLOC_THRESHOLD
        end

        @testset "in-place AoS batch (Range grids)" begin
            @test _alloc_test_inplace_aos() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Oneshot In-Place API (cubic_interp!)
    # ========================================

    @testset "Oneshot In-Place (cubic_interp!)" begin
        @testset "SoA correctness" begin
            x = range(0.0, 2π, 21)
            y = range(0.0, π, 11)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
            yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
            ref = cubic_interp((x, y), data, (xqs, yqs))
            out = similar(ref)
            cubic_interp!(out, (x, y), data, (xqs, yqs))
            @test out ≈ ref atol = 1.0e-14
        end

        @testset "AoS correctness" begin
            x = range(0.0, 2π, 21)
            y = range(0.0, π, 11)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6), (2.0, 0.8), (3.0, 1.0)]
            ref = cubic_interp((x, y), data, points)
            out = similar(ref)
            cubic_interp!(out, (x, y), data, points)
            @test out ≈ ref atol = 1.0e-14
        end

        @testset "SoA with deriv" begin
            x = range(0.0, 2π, 21)
            y = range(0.0, π, 11)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            xqs = [0.5, 1.0, 1.5]
            yqs = [0.2, 0.4, 0.6]
            ref = cubic_interp((x, y), data, (xqs, yqs); deriv = DerivOp(1, 1))
            out = similar(ref)
            cubic_interp!(out, (x, y), data, (xqs, yqs); deriv = DerivOp(1, 1))
            @test out ≈ ref atol = 1.0e-14
        end

        @testset "DimensionMismatch on wrong output length" begin
            x = range(0.0, 1.0, 10)
            y = range(0.0, 1.0, 10)
            data = [xi + yj for xi in x, yj in y]
            xqs = [0.5, 0.6, 0.7]
            yqs = [0.5, 0.6, 0.7]
            out = zeros(5)  # wrong length
            @test_throws DimensionMismatch cubic_interp!(out, (x, y), data, (xqs, yqs))
        end
    end

    # Oneshot in-place allocation tests (function barrier pattern)
    function _alloc_test_oneshot_inplace_soa_cubic()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.4, 0.6]
        out = Vector{Float64}(undef, 3)
        cubic_interp!(out, (x, y), data, (xqs, yqs))
        cubic_interp!(out, (x, y), data, (xqs, yqs))
        @allocated cubic_interp!(out, (x, y), data, (xqs, yqs))
    end

    function _alloc_test_oneshot_inplace_aos_cubic()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6)]
        out = Vector{Float64}(undef, 3)
        cubic_interp!(out, (x, y), data, points)
        cubic_interp!(out, (x, y), data, points)
        @allocated cubic_interp!(out, (x, y), data, points)
    end

    @testset "Oneshot In-Place Allocation Tests" begin
        @testset "oneshot in-place SoA (Range grids)" begin
            @test _alloc_test_oneshot_inplace_soa_cubic() <= ND_ALLOC_THRESHOLD
        end

        @testset "oneshot in-place AoS (Range grids)" begin
            @test _alloc_test_oneshot_inplace_aos_cubic() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Oneshot In-Place Allocation Tests (Vector grids)
    # ========================================

    function _alloc_test_oneshot_inplace_soa_cubic_vec()
        x = collect(range(0.0, 2.0, 15))
        y = collect(range(0.0, 1.0, 11))
        data = [xi^3 + yj^2 for xi in x, yj in y]
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.5, 0.8]
        out = Vector{Float64}(undef, 3)
        cubic_interp!(out, (x, y), data, (xqs, yqs))
        cubic_interp!(out, (x, y), data, (xqs, yqs))
        @allocated cubic_interp!(out, (x, y), data, (xqs, yqs))
    end

    function _alloc_test_oneshot_inplace_aos_cubic_vec()
        x = collect(range(0.0, 2.0, 15))
        y = collect(range(0.0, 1.0, 11))
        data = [xi^3 + yj^2 for xi in x, yj in y]
        points = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
        out = Vector{Float64}(undef, 3)
        cubic_interp!(out, (x, y), data, points)
        cubic_interp!(out, (x, y), data, points)
        @allocated cubic_interp!(out, (x, y), data, points)
    end

    @testset "Oneshot In-Place Allocation Tests (Vector grids)" begin
        @testset "oneshot in-place SoA (Vector grids)" begin
            @test _alloc_test_oneshot_inplace_soa_cubic_vec() <= ND_ALLOC_THRESHOLD
        end

        @testset "oneshot in-place AoS (Vector grids)" begin
            @test _alloc_test_oneshot_inplace_aos_cubic_vec() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Pool Rewind Verification
    # ========================================
    # Verify that @with_pool properly rewinds after oneshot API calls.
    # After each call, pool.float64.n_active must return to its pre-call value.

    # ========================================
    # DerivOp deriv paths in cubic_interp! batch
    # ========================================
    #
    # Covers the DerivOp tuple path in SoA and AoS `cubic_interp!` functions.
    # This is distinct from the `deriv::Int` broadcast path.

    @testset "DerivOp deriv paths in cubic_interp! batch" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)
        query = (1.5, 0.8)
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.4, 0.6]
        points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6)]
        out = Vector{Float64}(undef, 3)

        # Scalar: DerivOp tuple deriv → triggers the `else` branch
        val_ntuple = cubic_interp((x, y), data, query; deriv = DerivOp(1, 0))
        val_val = cubic_interp((x, y), data, query; deriv = DerivOp(1, 0))
        @test val_ntuple ≈ val_val atol = 1.0e-14

        # SoA cubic_interp!: DerivOp deriv → triggers `elseif deriv isa Val` branch
        ref_soa = cubic_interp((x, y), data, (xqs, yqs); deriv = DerivOp(1, 0))
        cubic_interp!(out, (x, y), data, (xqs, yqs); deriv = DerivOp(1, 0))
        @test out ≈ ref_soa atol = 1.0e-14

        # SoA cubic_interp!: DerivOp deriv → triggers `else` branch
        ref_soa2 = cubic_interp((x, y), data, (xqs, yqs); deriv = DerivOp(1, 0))
        cubic_interp!(out, (x, y), data, (xqs, yqs); deriv = DerivOp(1, 0))
        @test out ≈ ref_soa2 atol = 1.0e-14

        # AoS cubic_interp!: DerivOp deriv → triggers `elseif deriv isa Val` branch
        ref_aos = cubic_interp((x, y), data, points; deriv = DerivOp(1, 0))
        cubic_interp!(out, (x, y), data, points; deriv = DerivOp(1, 0))
        @test out ≈ ref_aos atol = 1.0e-14

        # AoS cubic_interp!: DerivOp deriv → triggers `else` branch
        ref_aos2 = cubic_interp((x, y), data, points; deriv = DerivOp(1, 0))
        cubic_interp!(out, (x, y), data, points; deriv = DerivOp(1, 0))
        @test out ≈ ref_aos2 atol = 1.0e-14

        # Confirm NTuple result agrees with interpolant
        @test val_ntuple ≈ itp(query; deriv = DerivOp(1, 0)) atol = 1.0e-14
    end

    # ========================================
    # SoA DimensionMismatch for mismatched query lengths
    # ========================================

    @testset "SoA DimensionMismatch for mismatched query vector lengths" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 10)
        data = [xi + yj for xi in x, yj in y]
        out = zeros(3)
        # dim-1 query has 3 pts, dim-2 has 2 pts → should throw
        @test_throws DimensionMismatch cubic_interp!(
            out, (x, y), data, ([0.5, 0.6, 0.7], [0.5, 0.6])
        )
    end

    @testset "Pool rewind after oneshot (cubic)" begin
        xv = collect(range(0.0, 2π, 21))
        yv = collect(range(0.0, π, 11))
        data = [sin(xi) * cos(yj) for xi in xv, yj in yv]
        query = (1.0, 0.5)
        xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
        yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
        pts = [(xqs[i], yqs[i]) for i in 1:5]

        # Warmup all paths
        cubic_interp((xv, yv), data, query)
        cubic_interp((xv, yv), data, (xqs, yqs))
        cubic_interp((xv, yv), data, pts)
        out = Vector{Float64}(undef, 5)
        cubic_interp!(out, (xv, yv), data, (xqs, yqs))
        cubic_interp!(out, (xv, yv), data, pts)

        pool = get_task_local_pool()

        @testset "scalar oneshot" begin
            n_before = pool.float64.n_active
            cubic_interp((xv, yv), data, query)
            @test pool.float64.n_active == n_before
        end

        @testset "SoA batch oneshot" begin
            n_before = pool.float64.n_active
            cubic_interp((xv, yv), data, (xqs, yqs))
            @test pool.float64.n_active == n_before
        end

        @testset "AoS batch oneshot" begin
            n_before = pool.float64.n_active
            cubic_interp((xv, yv), data, pts)
            @test pool.float64.n_active == n_before
        end

        @testset "SoA in-place oneshot" begin
            n_before = pool.float64.n_active
            cubic_interp!(out, (xv, yv), data, (xqs, yqs))
            @test pool.float64.n_active == n_before
        end

        @testset "AoS in-place oneshot" begin
            n_before = pool.float64.n_active
            cubic_interp!(out, (xv, yv), data, pts)
            @test pool.float64.n_active == n_before
        end
    end

end
