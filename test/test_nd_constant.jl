# ========================================
# Tests for ConstantInterpolantND
# ========================================
#
# Phase 2 of ND Constant/Linear implementation.
# Tests follow TDD protocol:
# - 🔴 RED: Tests written first, expected to fail initially
# - 🟢 GREEN: Minimal implementation to pass
# - 🔵 REFACTOR: Cleanup while staying green

@testitem "ConstantInterpolantND" setup = [AllocConstants] begin
    using FastInterpolations: get_task_local_pool
    # ========================================
    # 2D Constant Exactness
    # ========================================
    @testset "2D constant exactness" begin
        # Create simple 3x4 grid with distinct integer values
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0, 3.0]
        # data[i,j] = 10*i + j (unique per cell)
        data = [
            11.0 12.0 13.0 14.0;
            21.0 22.0 23.0 24.0;
            31.0 32.0 33.0 34.0
        ]

        @testset "side=LeftSide() (default)" begin
            itp = constant_interp((x, y), data; side = LeftSide())

            # At grid points, returns left value of the interval containing the point
            # At (0,0): interval idx=(1,1), data[1,1]=11
            @test itp((0.0, 0.0)) == 11.0  # Origin
            # At (1,2): x in interval 2, y in interval 3, data[2,3]=23
            @test itp((1.0, 2.0)) == 23.0  # Interior point
            # At boundary (2,3): last intervals, data[2,3]=23 (not 34!)
            # Because side=LeftSide() always returns left corner of interval
            @test itp((2.0, 3.0)) == 23.0  # Far corner (in last interval)

            # Between grid points with LeftSide(), always select left neighbor
            @test itp((0.5, 0.5)) == 11.0  # Cell (1,1) - left corner
            @test itp((1.5, 2.5)) == 23.0  # Cell (2,3) - left corner
            @test itp((0.9, 0.9)) == 11.0  # Still left corner
        end

        @testset "side=RightSide()" begin
            itp = constant_interp((x, y), data; side = RightSide())

            # At grid points, still returns left value (dL == 0)
            @test itp((0.0, 0.0)) == 11.0
            @test itp((1.0, 2.0)) == 23.0

            # Between grid points with RightSide(), select right neighbor
            @test itp((0.5, 0.5)) == 22.0  # Right corner of cell (1,1) → data[2,2]
            @test itp((0.1, 0.1)) == 22.0  # Any offset → right
            @test itp((1.5, 2.5)) == 34.0  # Cell (2,3) → data[3,4]
        end

        @testset "side=NearestSide()" begin
            itp = constant_interp((x, y), data; side = NearestSide())

            # At grid points
            @test itp((0.0, 0.0)) == 11.0
            @test itp((1.0, 2.0)) == 23.0

            # Near left boundary (< h/2 = 0.5)
            @test itp((0.3, 0.3)) == 11.0  # Closer to left
            @test itp((0.5, 0.5)) == 11.0  # Exactly h/2 → left (tie-breaker)

            # Near right boundary (> h/2)
            @test itp((0.6, 0.6)) == 22.0  # Closer to right → data[2,2]
            @test itp((0.9, 0.9)) == 22.0  # Near right
        end

        @testset "per-axis side configuration" begin
            # side=(LeftSide(), RightSide()) → left on x-axis, right on y-axis
            itp = constant_interp((x, y), data; side = (LeftSide(), RightSide()))

            @test itp((0.5, 0.5)) == 12.0  # x: left (idx=1), y: right (idx=2) → data[1,2]
            @test itp((1.5, 0.5)) == 22.0  # x: left (idx=2), y: right (idx=2) → data[2,2]
        end
    end

    # ========================================
    # 3D Constant Exactness
    # ========================================
    @testset "3D constant exactness" begin
        x = [0.0, 1.0]
        y = [0.0, 1.0]
        z = [0.0, 1.0]
        # 2x2x2 cube with values = 100*i + 10*j + k
        data = zeros(2, 2, 2)
        for i in 1:2, j in 1:2, k in 1:2
            data[i, j, k] = 100.0 * i + 10.0 * j + k
        end

        itp = constant_interp((x, y, z), data; side = LeftSide())

        # At origin: all indices = 1
        @test itp((0.0, 0.0, 0.0)) == 111.0  # data[1,1,1]
        # At boundary (1,1,1): still in interval [0,1], side=LeftSide() → data[1,1,1]
        @test itp((1.0, 1.0, 1.0)) == 111.0  # data[1,1,1] (left of each interval)

        # Interior points with LeftSide()
        @test itp((0.5, 0.5, 0.5)) == 111.0  # All left → data[1,1,1]
        @test itp((0.9, 0.9, 0.9)) == 111.0  # Still all left
    end

    # ========================================
    # Derivatives Always Zero
    # ========================================
    @testset "derivatives return zero" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        itp = constant_interp((x, y), data)

        # First derivative (any direction)
        @test itp((0.5, 0.5); deriv = DerivOp(1, 1)) == 0.0
        @test itp((0.5, 0.5); deriv = DerivOp(1, 1)) == 0.0

        # Mixed partials
        @test itp((0.5, 0.5); deriv = DerivOp(1, 0)) == 0.0  # ∂f/∂x
        @test itp((0.5, 0.5); deriv = DerivOp(0, 1)) == 0.0  # ∂f/∂y
        @test itp((0.5, 0.5); deriv = DerivOp(1, 1)) == 0.0  # ∂²f/∂x∂y

        # Second derivatives
        @test itp((0.5, 0.5); deriv = DerivOp(2, 2)) == 0.0
        @test itp((0.5, 0.5); deriv = DerivOp(2, 0)) == 0.0

        # Third derivatives
        @test itp((0.5, 0.5); deriv = DerivOp(3, 3)) == 0.0
    end

    # ========================================
    # Vector API (ForwardDiff compatibility)
    # ========================================
    @testset "vector API" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        itp = constant_interp((x, y), data; side = LeftSide())

        # Query with vector instead of tuple
        @test itp([0.5, 0.5]) == itp((0.5, 0.5))
        @test itp([1.5, 1.5]) == itp((1.5, 1.5))
    end

    # ========================================
    # Batch Queries - SoA (Structure of Arrays)
    # ========================================
    @testset "batch SoA queries" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        itp = constant_interp((x, y), data; side = LeftSide())

        # Batch query: tuple of vectors
        xs = [0.5, 1.5, 0.5]
        ys = [0.5, 0.5, 1.5]
        results = itp((xs, ys))

        @test length(results) == 3
        @test results[1] == itp((0.5, 0.5))
        @test results[2] == itp((1.5, 0.5))
        @test results[3] == itp((0.5, 1.5))
    end

    # ========================================
    # Batch Queries - AoS (Array of Structures)
    # ========================================
    @testset "batch AoS queries" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        itp = constant_interp((x, y), data; side = LeftSide())

        # Batch query: vector of tuples
        points = [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5)]
        results = itp(points)

        @test length(results) == 3
        @test results[1] == itp((0.5, 0.5))
        @test results[2] == itp((1.5, 0.5))
        @test results[3] == itp((0.5, 1.5))
    end

    # ========================================
    # Extrapolation Modes
    # ========================================
    @testset "extrapolation modes" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        @testset "extrap=NoExtrap() (domain error)" begin
            itp = constant_interp((x, y), data; extrap = NoExtrap())

            # In domain - OK
            @test itp((0.5, 0.5)) == 1.0

            # Out of domain - should throw
            @test_throws DomainError itp((-0.1, 0.5))
            @test_throws DomainError itp((0.5, -0.1))
            @test_throws DomainError itp((2.1, 0.5))
            @test_throws DomainError itp((0.5, 2.1))
        end

        @testset "extrap=ClampExtrap()" begin
            itp = constant_interp((x, y), data; extrap = ClampExtrap(), side = LeftSide())

            # In domain
            @test itp((0.5, 0.5)) == 1.0

            # Out of domain - clamp to boundary then evaluate
            @test itp((-0.5, 0.5)) == 1.0   # Clamp x to 0 → data[1,1] = 1
            @test itp((0.5, -0.5)) == 1.0   # Clamp y to 0 → data[1,1] = 1
            # Clamp x to 2.0, which is at last interval boundary
            # With side=LeftSide(), we get data[2, 1] = 4.0
            @test itp((2.5, 0.5)) == 4.0    # Clamp x to 2 → interval 2, data[2,1]
            # Clamp y to 2.0, which is at last interval boundary
            # With side=LeftSide(), we get data[1, 2] = 2.0
            @test itp((0.5, 2.5)) == 2.0    # Clamp y to 2 → interval 2, data[1,2]
        end

        @testset "extrap=WrapExtrap()" begin
            itp = constant_interp((x, y), data; extrap = WrapExtrap(), side = LeftSide())

            # In domain
            @test itp((0.5, 0.5)) == 1.0

            # Out of domain - wrap periodically
            @test itp((2.5, 0.5)) == itp((0.5, 0.5))  # x wraps: 2.5 → 0.5
            @test itp((0.5, 2.5)) == itp((0.5, 0.5))  # y wraps: 2.5 → 0.5
        end

        @testset "per-axis extrap configuration" begin
            # extrap=(NoExtrap(), ClampExtrap()) → strict on x, clamp on y
            itp = constant_interp((x, y), data; extrap = (NoExtrap(), ClampExtrap()), side = LeftSide())

            # y clamped to 2.0 → data[1, 2] = 2.0
            @test itp((0.5, 2.5)) == 2.0  # y clamped to last interval
            @test_throws DomainError itp((2.5, 0.5))  # x out of domain
        end
    end

    # ========================================
    # Range Grid Support (O(1) lookup)
    # ========================================
    @testset "Range grid support" begin
        x = range(0.0, 2.0, 3)  # [0, 1, 2]
        y = range(0.0, 2.0, 3)  # [0, 1, 2]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        itp = constant_interp((x, y), data; side = LeftSide())

        @test itp((0.5, 0.5)) == 1.0
        @test itp((1.5, 0.5)) == 4.0
    end

    # ========================================
    # Mixed Grid Types (Range + Vector)
    # ========================================
    @testset "mixed grid types" begin
        x = range(0.0, 2.0, 3)  # Range
        y = [0.0, 0.5, 2.0]     # Non-uniform Vector
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        itp = constant_interp((x, y), data; side = LeftSide())

        @test itp((0.5, 0.3)) == 1.0  # First cell
        @test itp((1.5, 1.0)) == 5.0  # y in [0.5, 2.0) → idx 2
    end

    # ========================================
    # Search Policy Override
    # ========================================
    @testset "search policy" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        # Default BinarySearch
        itp = constant_interp((x, y), data)
        @test itp((0.5, 0.5)) == 1.0

        # LinearBinarySearch
        itp_lb = constant_interp((x, y), data; search = LinearBinarySearch())
        @test itp_lb((0.5, 0.5)) == 1.0

        # Per-axis search
        itp_mixed = constant_interp((x, y), data; search = (BinarySearch(), LinearBinarySearch()))
        @test itp_mixed((0.5, 0.5)) == 1.0
    end

    # ========================================
    # One-shot API
    # ========================================
    @testset "one-shot API" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        # Scalar one-shot
        result = constant_interp((x, y), data, (0.5, 0.5); side = LeftSide())
        @test result == 1.0

        # Batch one-shot (SoA)
        xs = [0.5, 1.5]
        ys = [0.5, 0.5]
        results = constant_interp((x, y), data, (xs, ys); side = LeftSide())
        @test results[1] == 1.0
        @test results[2] == 4.0
    end

    # ========================================
    # Type Introspection
    # ========================================
    @testset "type introspection" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        itp = constant_interp((x, y), data)

        @test grid_type(itp) === Float64
        @test value_type(itp) === Float64
        @test itp isa AbstractInterpolantND{Float64, Float64, 2}
    end

    # ========================================
    # Complex Values
    # ========================================
    @testset "complex values" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = ComplexF64[1 + 1im 2 + 2im 3 + 3im; 4 + 4im 5 + 5im 6 + 6im; 7 + 7im 8 + 8im 9 + 9im]

        itp = constant_interp((x, y), data; side = LeftSide())

        @test itp((0.5, 0.5)) == 1.0 + 1.0im
        @test itp((1.5, 1.5)) == 5.0 + 5.0im

        # Derivatives still zero
        @test itp((0.5, 0.5); deriv = DerivOp(1, 1)) == 0.0 + 0.0im
    end

    # ========================================
    # Integer Grid Promotion (regression)
    # ========================================
    @testset "integer grid promotion" begin
        data = [11.0 12.0 13.0; 21.0 22.0 23.0; 31.0 32.0 33.0]

        # Vector{Int} grids
        @testset "Vector{Int} grids" begin
            x = [0, 1, 2]
            y = [0, 1, 2]
            itp = constant_interp((x, y), data)
            @test itp((0.5, 0.5)) == 11.0
        end

        # UnitRange{Int} grids
        @testset "UnitRange{Int} grids" begin
            itp = constant_interp((0:2, 0:2), data)
            @test itp((0.5, 0.5)) == 11.0
        end

        # Mixed Int + Float grids
        @testset "mixed Int and Float grids" begin
            itp = constant_interp(([0, 1, 2], [0.0, 1.0, 2.0]), data)
            @test itp((0.5, 0.5)) == 11.0
        end

        # One-shot API with integer grids
        @testset "one-shot with integer grids" begin
            @test constant_interp(([0, 1, 2], [0, 1, 2]), data, (0.5, 0.5)) == 11.0
        end
    end

    # ========================================
    # Zero-Allocation One-Shot Tests
    # ========================================
    #
    # Each test uses a full function barrier: setup + warmup + @allocated
    # all inside one function. This avoids @testset-scope boxing artifacts.

    function _alloc_test_constant_default()
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        constant_interp((x, y), data, query)
        constant_interp((x, y), data, query)
        @allocated constant_interp((x, y), data, query)
    end

    function _alloc_test_constant_left()
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        constant_interp((x, y), data, query; side = LeftSide())
        constant_interp((x, y), data, query; side = LeftSide())
        @allocated constant_interp((x, y), data, query; side = LeftSide())
    end

    function _alloc_test_constant_right()
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        constant_interp((x, y), data, query; side = RightSide())
        constant_interp((x, y), data, query; side = RightSide())
        @allocated constant_interp((x, y), data, query; side = RightSide())
    end

    function _alloc_test_constant_extrap_constant()
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        constant_interp((x, y), data, query; extrap = ClampExtrap())
        constant_interp((x, y), data, query; extrap = ClampExtrap())
        @allocated constant_interp((x, y), data, query; extrap = ClampExtrap())
    end

    function _alloc_test_constant_extrap_wrap()
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        constant_interp((x, y), data, query; extrap = WrapExtrap())
        constant_interp((x, y), data, query; extrap = WrapExtrap())
        @allocated constant_interp((x, y), data, query; extrap = WrapExtrap())
    end

    function _alloc_test_constant_mixed_mode()
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        constant_interp((x, y), data, query; extrap = (NoExtrap(), ClampExtrap()))
        constant_interp((x, y), data, query; extrap = (NoExtrap(), ClampExtrap()))
        @allocated constant_interp((x, y), data, query; extrap = (NoExtrap(), ClampExtrap()))
    end

    function _alloc_test_constant_3d()
        x = range(0.0, 2.0, 8)
        y = range(0.0, 1.0, 6)
        z = range(0.0, 3.0, 5)
        data = [xi + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        constant_interp((x, y, z), data, query)
        constant_interp((x, y, z), data, query)
        @allocated constant_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot" begin
        @testset "zero-alloc scalar (Range grids, default)" begin
            @test _alloc_test_constant_default() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, side=LeftSide())" begin
            @test _alloc_test_constant_left() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, side=RightSide())" begin
            @test _alloc_test_constant_right() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, extrap=ClampExtrap())" begin
            @test _alloc_test_constant_extrap_constant() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, extrap=WrapExtrap)" begin
            @test _alloc_test_constant_extrap_wrap() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, per-axis mixed Mode)" begin
            @test _alloc_test_constant_mixed_mode() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D Range grids)" begin
            @test _alloc_test_constant_3d() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Mixed-Grid Allocation Tests (Range + Vector)
    # ========================================
    #
    # Heterogeneous grid tuples (ScalarSpacing + VectorSpacing) must be zero-allocation.
    # Catches ntuple closure boxing on heterogeneous inputs.

    function _alloc_test_constant_mixed_2d()
        x = range(0.0, 2.0, 20)          # Range → ScalarSpacing
        y = collect(range(0.0, 1.0, 15)) # Vector → VectorSpacing
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        constant_interp((x, y), data, query)
        constant_interp((x, y), data, query)
        @allocated constant_interp((x, y), data, query)
    end

    function _alloc_test_constant_mixed_3d()
        x = range(0.0, 2.0, 8)           # Range → ScalarSpacing
        y = collect(range(0.0, 1.0, 6))  # Vector → VectorSpacing
        z = range(0.0, 3.0, 5)           # Range → ScalarSpacing
        data = [xi + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        constant_interp((x, y, z), data, query)
        constant_interp((x, y, z), data, query)
        @allocated constant_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot (Mixed grids: Range + Vector)" begin
        @testset "zero-alloc scalar (2D mixed grid)" begin
            @test _alloc_test_constant_mixed_2d() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D mixed grid)" begin
            @test _alloc_test_constant_mixed_3d() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Zero-Allocation One-Shot (Vector grids)
    # ========================================
    #
    # Vector grids must also be zero-allocation via pool-based spacing.

    function _alloc_test_constant_vector_default()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        constant_interp((x, y), data, query)
        constant_interp((x, y), data, query)
        @allocated constant_interp((x, y), data, query)
    end

    function _alloc_test_constant_vector_left()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        constant_interp((x, y), data, query; side = LeftSide())
        constant_interp((x, y), data, query; side = LeftSide())
        @allocated constant_interp((x, y), data, query; side = LeftSide())
    end

    function _alloc_test_constant_vector_3d()
        x = collect(range(0.0, 2.0, 10))
        y = collect(range(0.0, 1.0, 8))
        z = collect(range(0.0, 3.0, 6))
        data = [xi + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        constant_interp((x, y, z), data, query)
        constant_interp((x, y, z), data, query)
        @allocated constant_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot (Vector grids)" begin
        @testset "zero-alloc scalar (Vector grids, default)" begin
            @test _alloc_test_constant_vector_default() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Vector grids, side=LeftSide())" begin
            @test _alloc_test_constant_vector_left() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D Vector grids)" begin
            @test _alloc_test_constant_vector_3d() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # In-Place Batch Allocation Tests
    # ========================================
    #
    # In-place paths write into a pre-allocated output buffer.
    # These must be truly zero-allocation (only output + THRESHOLD).

    function _alloc_test_constant_inplace_soa()
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi + yj for xi in x, yj in y]
        itp = constant_interp((x, y), data)
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.5, 0.8]
        out = Vector{Float64}(undef, 3)
        itp(out, (xqs, yqs))
        itp(out, (xqs, yqs))
        @allocated itp(out, (xqs, yqs))
    end

    function _alloc_test_constant_inplace_aos()
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi + yj for xi in x, yj in y]
        itp = constant_interp((x, y), data)
        points = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
        out = Vector{Float64}(undef, 3)
        itp(out, points)
        itp(out, points)
        @allocated itp(out, points)
    end

    @testset "In-Place Batch Allocation Tests" begin
        @testset "in-place SoA batch (Range grids)" begin
            @test _alloc_test_constant_inplace_soa() <= ND_ALLOC_THRESHOLD
        end

        @testset "in-place AoS batch (Range grids)" begin
            @test _alloc_test_constant_inplace_aos() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Oneshot In-Place API (constant_interp!)
    # ========================================

    @testset "Oneshot In-Place (constant_interp!)" begin
        @testset "SoA correctness" begin
            x = range(0.0, 2π, 21)
            y = range(0.0, π, 11)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
            yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
            ref = constant_interp((x, y), data, (xqs, yqs))
            out = similar(ref)
            constant_interp!(out, (x, y), data, (xqs, yqs))
            @test out ≈ ref atol = 1.0e-14
        end

        @testset "AoS correctness" begin
            x = range(0.0, 2π, 21)
            y = range(0.0, π, 11)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6), (2.0, 0.8), (3.0, 1.0)]
            ref = constant_interp((x, y), data, points)
            out = similar(ref)
            constant_interp!(out, (x, y), data, points)
            @test out ≈ ref atol = 1.0e-14
        end

        @testset "Deriv fills zeros in-place" begin
            x = range(0.0, 1.0, 10)
            y = range(0.0, 1.0, 10)
            data = [xi + yj for xi in x, yj in y]
            xqs = [0.5, 0.6, 0.7]
            yqs = [0.5, 0.6, 0.7]
            out = ones(3)  # non-zero before call
            constant_interp!(out, (x, y), data, (xqs, yqs); deriv = DerivOp(1, 1))
            @test all(out .== 0.0)
        end

        @testset "DimensionMismatch on wrong output length" begin
            x = range(0.0, 1.0, 10)
            y = range(0.0, 1.0, 10)
            data = [xi + yj for xi in x, yj in y]
            xqs = [0.5, 0.6, 0.7]
            yqs = [0.5, 0.6, 0.7]
            out = zeros(5)
            @test_throws DimensionMismatch constant_interp!(out, (x, y), data, (xqs, yqs))
        end
    end

    function _alloc_test_oneshot_inplace_soa_constant()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.4, 0.6]
        out = Vector{Float64}(undef, 3)
        constant_interp!(out, (x, y), data, (xqs, yqs))
        constant_interp!(out, (x, y), data, (xqs, yqs))
        @allocated constant_interp!(out, (x, y), data, (xqs, yqs))
    end

    function _alloc_test_oneshot_inplace_aos_constant()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6)]
        out = Vector{Float64}(undef, 3)
        constant_interp!(out, (x, y), data, points)
        constant_interp!(out, (x, y), data, points)
        @allocated constant_interp!(out, (x, y), data, points)
    end

    @testset "Oneshot In-Place Allocation Tests" begin
        @testset "oneshot in-place SoA (Range grids)" begin
            @test _alloc_test_oneshot_inplace_soa_constant() <= ND_ALLOC_THRESHOLD
        end

        @testset "oneshot in-place AoS (Range grids)" begin
            @test _alloc_test_oneshot_inplace_aos_constant() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Oneshot In-Place Allocation Tests (Vector grids)
    # ========================================

    function _alloc_test_oneshot_inplace_soa_constant_vec()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi + yj for xi in x, yj in y]
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.5, 0.8]
        out = Vector{Float64}(undef, 3)
        constant_interp!(out, (x, y), data, (xqs, yqs))
        constant_interp!(out, (x, y), data, (xqs, yqs))
        @allocated constant_interp!(out, (x, y), data, (xqs, yqs))
    end

    function _alloc_test_oneshot_inplace_aos_constant_vec()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi + yj for xi in x, yj in y]
        points = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
        out = Vector{Float64}(undef, 3)
        constant_interp!(out, (x, y), data, points)
        constant_interp!(out, (x, y), data, points)
        @allocated constant_interp!(out, (x, y), data, points)
    end

    @testset "Oneshot In-Place Allocation Tests (Vector grids)" begin
        @testset "oneshot in-place SoA (Vector grids)" begin
            @test _alloc_test_oneshot_inplace_soa_constant_vec() <= ND_ALLOC_THRESHOLD
        end

        @testset "oneshot in-place AoS (Vector grids)" begin
            @test _alloc_test_oneshot_inplace_aos_constant_vec() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Pool Rewind Verification
    # ========================================

    @testset "Pool rewind after oneshot (constant)" begin
        xv = collect(range(0.0, 3.0, 10))
        yv = collect(range(0.0, 2.0, 8))
        data = [Float64(xi + yi) for xi in xv, yi in yv]
        query = (1.0, 0.5)
        xqs = [0.5, 1.0, 1.5, 2.0, 2.5]
        yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
        pts = [(xqs[i], yqs[i]) for i in 1:5]

        # Warmup
        constant_interp((xv, yv), data, query)
        constant_interp((xv, yv), data, (xqs, yqs))
        constant_interp((xv, yv), data, pts)
        out = Vector{Float64}(undef, 5)
        constant_interp!(out, (xv, yv), data, (xqs, yqs))
        constant_interp!(out, (xv, yv), data, pts)

        pool = get_task_local_pool()

        @testset "scalar oneshot" begin
            n_before = pool.float64.n_active
            constant_interp((xv, yv), data, query)
            @test pool.float64.n_active == n_before
        end

        @testset "SoA batch oneshot" begin
            n_before = pool.float64.n_active
            constant_interp((xv, yv), data, (xqs, yqs))
            @test pool.float64.n_active == n_before
        end

        @testset "AoS batch oneshot" begin
            n_before = pool.float64.n_active
            constant_interp((xv, yv), data, pts)
            @test pool.float64.n_active == n_before
        end

        @testset "SoA in-place oneshot" begin
            n_before = pool.float64.n_active
            constant_interp!(out, (xv, yv), data, (xqs, yqs))
            @test pool.float64.n_active == n_before
        end

        @testset "AoS in-place oneshot" begin
            n_before = pool.float64.n_active
            constant_interp!(out, (xv, yv), data, pts)
            @test pool.float64.n_active == n_before
        end
    end
end

# ════════════════════════════════════════════════════════════════
# PR1 (`refac/cleanup_nd_spacing`) lock-down: spacings field
# removed from forward struct. Asserts field absence,
# type-parameter count, type stability, and zero-allocation
# persistent eval.
# ════════════════════════════════════════════════════════════════
@testitem "ConstantInterpolantND — spacings cleanup lock-down" setup = [AllocConstants] begin
    using FastInterpolations: constant_interp

    @testset "spacings field removed" begin
        x = 0.0:1.0:3.0
        y = 0.0:1.0:3.0
        data = [Float64(i + j) for i in 1:4, j in 1:4]
        itp = constant_interp((x, y), data)

        @test !hasfield(typeof(itp), :spacings)
        # Was 8 (Tg, Tv, N, G, S, E, SD, P), now 7 (drops S)
        @test length(typeof(itp).parameters) == 7
        @test isfinite(itp((1.5, 1.5)))
    end

    @testset "type stability (@inferred)" begin
        x_rng = 0.0:1.0:3.0
        x_vec = [0.0, 1.0, 2.0, 3.0]
        data = [Float64(i + j) for i in 1:4, j in 1:4]

        itp_rng = constant_interp((x_rng, x_rng), data)
        itp_vec = constant_interp((x_vec, x_vec), data)

        @test (@inferred itp_rng((0.5, 0.5))) isa Float64
        @test (@inferred itp_vec((0.5, 0.5))) isa Float64
    end

    @testset "zero-alloc persistent eval" begin
        x = 0.0:1.0:3.0
        y = 0.0:1.0:3.0
        data = [Float64(i + j) for i in 1:4, j in 1:4]
        itp = constant_interp((x, y), data)
        itp((0.5, 0.5))
        itp((0.5, 0.5))

        @test (@allocated itp((0.5, 0.5))) <= ALLOC_THRESHOLD
    end
end
