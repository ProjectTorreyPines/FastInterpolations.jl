# ========================================
# Tests for ConstantInterpolantND
# ========================================
#
# Phase 2 of ND Constant/Linear implementation.
# Tests follow TDD protocol:
# - 🔴 RED: Tests written first, expected to fail initially
# - 🟢 GREEN: Minimal implementation to pass
# - 🔵 REFACTOR: Cleanup while staying green

using Test
using FastInterpolations

@testset "ConstantInterpolantND" begin
    # ========================================
    # 2D Constant Exactness
    # ========================================
    @testset "2D constant exactness" begin
        # Create simple 3x4 grid with distinct integer values
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0, 3.0]
        # data[i,j] = 10*i + j (unique per cell)
        data = [11.0 12.0 13.0 14.0;
                21.0 22.0 23.0 24.0;
                31.0 32.0 33.0 34.0]

        @testset "side=:left (default)" begin
            itp = constant_interp((x, y), data; side=:left)

            # At grid points, returns left value of the interval containing the point
            # At (0,0): interval idx=(1,1), data[1,1]=11
            @test itp((0.0, 0.0)) == 11.0  # Origin
            # At (1,2): x in interval 2, y in interval 3, data[2,3]=23
            @test itp((1.0, 2.0)) == 23.0  # Interior point
            # At boundary (2,3): last intervals, data[2,3]=23 (not 34!)
            # Because side=:left always returns left corner of interval
            @test itp((2.0, 3.0)) == 23.0  # Far corner (in last interval)

            # Between grid points with :left, always select left neighbor
            @test itp((0.5, 0.5)) == 11.0  # Cell (1,1) - left corner
            @test itp((1.5, 2.5)) == 23.0  # Cell (2,3) - left corner
            @test itp((0.9, 0.9)) == 11.0  # Still left corner
        end

        @testset "side=:right" begin
            itp = constant_interp((x, y), data; side=:right)

            # At grid points, still returns left value (dL == 0)
            @test itp((0.0, 0.0)) == 11.0
            @test itp((1.0, 2.0)) == 23.0

            # Between grid points with :right, select right neighbor
            @test itp((0.5, 0.5)) == 22.0  # Right corner of cell (1,1) → data[2,2]
            @test itp((0.1, 0.1)) == 22.0  # Any offset → right
            @test itp((1.5, 2.5)) == 34.0  # Cell (2,3) → data[3,4]
        end

        @testset "side=:nearest" begin
            itp = constant_interp((x, y), data; side=:nearest)

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
            # side=(:left, :right) → left on x-axis, right on y-axis
            itp = constant_interp((x, y), data; side=(:left, :right))

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

        itp = constant_interp((x, y, z), data; side=:left)

        # At origin: all indices = 1
        @test itp((0.0, 0.0, 0.0)) == 111.0  # data[1,1,1]
        # At boundary (1,1,1): still in interval [0,1], side=:left → data[1,1,1]
        @test itp((1.0, 1.0, 1.0)) == 111.0  # data[1,1,1] (left of each interval)

        # Interior points with :left
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
        @test itp((0.5, 0.5); deriv=1) == 0.0
        @test itp((0.5, 0.5); deriv=Val(1)) == 0.0

        # Mixed partials
        @test itp((0.5, 0.5); deriv=Val((1, 0))) == 0.0  # ∂f/∂x
        @test itp((0.5, 0.5); deriv=Val((0, 1))) == 0.0  # ∂f/∂y
        @test itp((0.5, 0.5); deriv=Val((1, 1))) == 0.0  # ∂²f/∂x∂y

        # Second derivatives
        @test itp((0.5, 0.5); deriv=2) == 0.0
        @test itp((0.5, 0.5); deriv=Val((2, 0))) == 0.0

        # Third derivatives
        @test itp((0.5, 0.5); deriv=3) == 0.0
    end

    # ========================================
    # Vector API (ForwardDiff compatibility)
    # ========================================
    @testset "vector API" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

        itp = constant_interp((x, y), data; side=:left)

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

        itp = constant_interp((x, y), data; side=:left)

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

        itp = constant_interp((x, y), data; side=:left)

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

        @testset "extrap=:none (domain error)" begin
            itp = constant_interp((x, y), data; extrap=:none)

            # In domain - OK
            @test itp((0.5, 0.5)) == 1.0

            # Out of domain - should throw
            @test_throws DomainError itp((-0.1, 0.5))
            @test_throws DomainError itp((0.5, -0.1))
            @test_throws DomainError itp((2.1, 0.5))
            @test_throws DomainError itp((0.5, 2.1))
        end

        @testset "extrap=:constant" begin
            itp = constant_interp((x, y), data; extrap=:constant, side=:left)

            # In domain
            @test itp((0.5, 0.5)) == 1.0

            # Out of domain - clamp to boundary then evaluate
            @test itp((-0.5, 0.5)) == 1.0   # Clamp x to 0 → data[1,1] = 1
            @test itp((0.5, -0.5)) == 1.0   # Clamp y to 0 → data[1,1] = 1
            # Clamp x to 2.0, which is at last interval boundary
            # With side=:left, we get data[2, 1] = 4.0
            @test itp((2.5, 0.5)) == 4.0    # Clamp x to 2 → interval 2, data[2,1]
            # Clamp y to 2.0, which is at last interval boundary
            # With side=:left, we get data[1, 2] = 2.0
            @test itp((0.5, 2.5)) == 2.0    # Clamp y to 2 → interval 2, data[1,2]
        end

        @testset "extrap=:wrap" begin
            itp = constant_interp((x, y), data; extrap=:wrap, side=:left)

            # In domain
            @test itp((0.5, 0.5)) == 1.0

            # Out of domain - wrap periodically
            @test itp((2.5, 0.5)) == itp((0.5, 0.5))  # x wraps: 2.5 → 0.5
            @test itp((0.5, 2.5)) == itp((0.5, 0.5))  # y wraps: 2.5 → 0.5
        end

        @testset "per-axis extrap configuration" begin
            # extrap=(:none, :constant) → strict on x, clamp on y
            itp = constant_interp((x, y), data; extrap=(:none, :constant), side=:left)

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

        itp = constant_interp((x, y), data; side=:left)

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

        itp = constant_interp((x, y), data; side=:left)

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

        # Default Binary
        itp = constant_interp((x, y), data)
        @test itp((0.5, 0.5)) == 1.0

        # LinearBinary
        itp_lb = constant_interp((x, y), data; search=LinearBinary())
        @test itp_lb((0.5, 0.5)) == 1.0

        # Per-axis search
        itp_mixed = constant_interp((x, y), data; search=(Binary(), LinearBinary()))
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
        result = constant_interp((x, y), data, (0.5, 0.5); side=:left)
        @test result == 1.0

        # Batch one-shot (SoA)
        xs = [0.5, 1.5]
        ys = [0.5, 0.5]
        results = constant_interp((x, y), data, (xs, ys); side=:left)
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
        data = ComplexF64[1+1im 2+2im 3+3im; 4+4im 5+5im 6+6im; 7+7im 8+8im 9+9im]

        itp = constant_interp((x, y), data; side=:left)

        @test itp((0.5, 0.5)) == 1.0 + 1.0im
        @test itp((1.5, 1.5)) == 5.0 + 5.0im

        # Derivatives still zero
        @test itp((0.5, 0.5); deriv=1) == 0.0 + 0.0im
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
end
