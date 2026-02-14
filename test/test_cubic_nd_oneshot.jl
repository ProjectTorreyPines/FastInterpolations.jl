using Test
using FastInterpolations

@testset "Cubic ND One-Shot (Pool-Based)" begin

    # ========================================
    # Correctness: One-shot vs Interpolant
    # ========================================

    @testset "Scalar one-shot matches Interpolant (2D, NaturalBC)" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        for (xq, yq) in [(1.0, 0.5), (3.0, 1.2), (0.1, 2.8)]
            val_oneshot = cubic_interp((x, y), data, (xq, yq))
            val_interp = itp((xq, yq))
            @test val_oneshot ≈ val_interp atol=1e-14
        end
    end

    @testset "Scalar one-shot matches Interpolant (2D, CubicFit)" begin
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^3 + yj^2 for xi in x, yj in y]

        itp = cubic_interp((x, y), data; bc=CubicFit())
        val_oneshot = cubic_interp((x, y), data, (1.0, 0.5); bc=CubicFit())
        val_interp = itp((1.0, 0.5))
        @test val_oneshot ≈ val_interp atol=1e-14
    end

    @testset "Scalar one-shot matches Interpolant (3D, NaturalBC)" begin
        x = range(0.0, 2.0, 10)
        y = range(0.0, 1.0, 8)
        z = range(0.0, 3.0, 6)
        data = [xi^2 + yj + zk for xi in x, yj in y, zk in z]

        itp = cubic_interp((x, y, z), data)
        val_oneshot = cubic_interp((x, y, z), data, (1.0, 0.5, 1.5))
        val_interp = itp((1.0, 0.5, 1.5))
        @test val_oneshot ≈ val_interp atol=1e-14
    end

    @testset "Derivative one-shot matches Interpolant" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)
        query = (1.5, 0.8)

        # deriv=0 (value)
        @test cubic_interp((x, y), data, query; deriv=0) ≈ itp(query; deriv=0) atol=1e-14

        # deriv=1 (all first derivatives)
        @test cubic_interp((x, y), data, query; deriv=1) ≈ itp(query; deriv=1) atol=1e-14

        # Mixed partial: ∂f/∂x
        @test cubic_interp((x, y), data, query; deriv=Val((1,0))) ≈ itp(query; deriv=Val((1,0))) atol=1e-14

        # Mixed partial: ∂f/∂y
        @test cubic_interp((x, y), data, query; deriv=Val((0,1))) ≈ itp(query; deriv=Val((0,1))) atol=1e-14

        # Mixed partial: ∂²f/∂x∂y
        @test cubic_interp((x, y), data, query; deriv=Val((1,1))) ≈ itp(query; deriv=Val((1,1))) atol=1e-14
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
            @test vals_oneshot[k] ≈ itp((xqs[k], yqs[k])) atol=1e-14
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
            @test vals_oneshot[k] ≈ itp(points[k]) atol=1e-14
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
        @test val_oneshot ≈ val_interp atol=1e-14
    end

    @testset "Heterogeneous grids (Range + Vector)" begin
        x = range(0.0, 2.0, 15)  # Range
        y = collect(range(0.0, 1.0, 10))  # Vector
        data = [xi^2 + yj for xi in x, yj in y]

        itp = cubic_interp((x, y), data; bc=CubicFit())
        val_oneshot = cubic_interp((x, y), data, (1.0, 0.5); bc=CubicFit())
        val_interp = itp((1.0, 0.5))
        @test val_oneshot ≈ val_interp atol=1e-14
    end

    @testset "Extrapolation modes" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 10)
        data = [xi + yj for xi in x, yj in y]
        itp_const = cubic_interp((x, y), data; extrap=:constant)
        itp_ext = cubic_interp((x, y), data; extrap=:extension)

        # Constant extrap
        @test cubic_interp((x, y), data, (1.0, 0.5); extrap=:constant) ≈
              itp_const((1.0, 0.5)) atol=1e-14

        # Extension extrap
        @test cubic_interp((x, y), data, (1.0, 0.5); extrap=:extension) ≈
              itp_ext((1.0, 0.5)) atol=1e-14
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

        itp = cubic_interp((x, y), data; bc=PeriodicBC())
        val_oneshot = cubic_interp((x, y), data, (1.5, 0.8); bc=PeriodicBC())
        val_interp = itp((1.5, 0.8))
        @test val_oneshot ≈ val_interp atol=1e-14
    end

    @testset "Periodic BC (exclusive)" begin
        n = 20
        x = range(0.0, 2π, n + 1)[1:n]  # exclusive: no endpoint
        y = range(0.0, 2π, n + 1)[1:n]
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        bc = PeriodicBC(endpoint=:exclusive)
        itp = cubic_interp((x, y), data; bc=bc)
        val_oneshot = cubic_interp((x, y), data, (1.5, 0.8); bc=bc)
        val_interp = itp((1.5, 0.8))
        @test val_oneshot ≈ val_interp atol=1e-14
    end

    @testset "Mixed periodic/non-periodic BCs" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, 1.0, 11)
        data = [sin(xi) * yj^2 for xi in x, yj in y]
        data[end, :] .= data[1, :]

        bc = (PeriodicBC(), NaturalBC())
        itp = cubic_interp((x, y), data; bc=bc)
        val_oneshot = cubic_interp((x, y), data, (1.5, 0.5); bc=bc)
        val_interp = itp((1.5, 0.5))
        @test val_oneshot ≈ val_interp atol=1e-14
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
        cubic_interp((x, y), data, query; deriv=1)
        cubic_interp((x, y), data, query; deriv=1)
        @allocated cubic_interp((x, y), data, query; deriv=1)
    end

    function _alloc_test_cubicfit()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query; bc=CubicFit())
        cubic_interp((x, y), data, query; bc=CubicFit())
        @allocated cubic_interp((x, y), data, query; bc=CubicFit())
    end

    function _alloc_test_periodic_inclusive()
        x = range(0.0, 2π, 21)
        y = range(0.0, 2π, 21)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        data[end, :] .= data[1, :]
        data[:, end] .= data[:, 1]
        query = (1.5, 0.8)
        cubic_interp((x, y), data, query; bc=PeriodicBC())
        cubic_interp((x, y), data, query; bc=PeriodicBC())
        @allocated cubic_interp((x, y), data, query; bc=PeriodicBC())
    end

    function _alloc_test_periodic_exclusive()
        n = 20
        x = range(0.0, 2π, n + 1)[1:n]
        y = range(0.0, 2π, n + 1)[1:n]
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        bc = PeriodicBC(endpoint=:exclusive)
        cubic_interp((x, y), data, query; bc=bc)
        cubic_interp((x, y), data, query; bc=bc)
        @allocated cubic_interp((x, y), data, query; bc=bc)
    end

    @testset "Zero-alloc scalar one-shot (Range grids)" begin
        @test _alloc_test_natural() == 0
    end

    @testset "Zero-alloc scalar one-shot with deriv (Range grids)" begin
        @test _alloc_test_natural_deriv() == 0
    end

    @testset "Zero-alloc scalar one-shot (CubicFit BC, Range grids)" begin
        @test _alloc_test_cubicfit() == 0
    end

    @testset "Zero-alloc scalar one-shot (Periodic BC inclusive, Range grids)" begin
        @test _alloc_test_periodic_inclusive() == 0
    end

    @testset "Zero-alloc scalar one-shot (Periodic BC exclusive, Range grids)" begin
        @test _alloc_test_periodic_exclusive() == 0
    end

    function _alloc_test_mixed_periodic()
        x = range(0.0, 2π, 21)
        y = range(0.0, 1.0, 11)
        data = [sin(xi) * yj^2 for xi in x, yj in y]
        data[end, :] .= data[1, :]
        query = (1.5, 0.5)
        bc = (PeriodicBC(), NaturalBC())
        cubic_interp((x, y), data, query; bc=bc)
        cubic_interp((x, y), data, query; bc=bc)
        @allocated cubic_interp((x, y), data, query; bc=bc)
    end

    @testset "Zero-alloc scalar one-shot (Mixed periodic/NaturalBC, Range grids)" begin
        @test _alloc_test_mixed_periodic() == 0
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
        cubic_interp((x, y), data, query; bc=CubicFit())
        cubic_interp((x, y), data, query; bc=CubicFit())
        @allocated cubic_interp((x, y), data, query; bc=CubicFit())
    end

    function _alloc_test_vector_deriv()
        x = collect(range(0.0, 2.0, 15))
        y = collect(range(0.0, 1.0, 11))
        data = [xi^3 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query; deriv=Val((1, 0)))
        cubic_interp((x, y), data, query; deriv=Val((1, 0)))
        @allocated cubic_interp((x, y), data, query; deriv=Val((1, 0)))
    end

    @testset "Zero-alloc scalar one-shot (Vector grids, NaturalBC)" begin
        @test _alloc_test_vector_natural() == 0
    end

    @testset "Zero-alloc scalar one-shot (Vector grids, CubicFit)" begin
        @test _alloc_test_vector_cubicfit() == 0
    end

    @testset "Zero-alloc scalar one-shot (Vector grids, deriv)" begin
        @test _alloc_test_vector_deriv() == 0
    end

end
