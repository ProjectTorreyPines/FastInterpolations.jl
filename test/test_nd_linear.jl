# ========================================
# Tests for LinearInterpolantND
# ========================================
#
# Comprehensive test coverage for N-dimensional multilinear interpolation.
# Tests cover: exactness, derivatives, batch queries, extrapolation, grid types, complex values.

using Test
using FastInterpolations
using FastInterpolations: get_task_local_pool

# Allocation threshold (bytes) — tolerates minor LTS/GC overhead.
# Guarded for standalone execution (runtests.jl defines this globally).
if !@isdefined(ND_ALLOC_THRESHOLD)
    const ND_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240
end

@testset "LinearInterpolantND" begin
    # ========================================
    # 2D Bilinear Exactness Tests
    # ========================================
    @testset "2D bilinear exactness" begin
        # Create simple 2D grid
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0, 3.0]

        # Linear function: f(x,y) = 2x + 3y + 1
        # Bilinear should reproduce this exactly
        data = [2.0*xi + 3.0*yj + 1.0 for xi in x, yj in y]

        itp = linear_interp((x, y), data)

        # Test at grid points (exact)
        @test itp((0.0, 0.0)) ≈ 1.0
        @test itp((1.0, 0.0)) ≈ 3.0
        @test itp((0.0, 1.0)) ≈ 4.0
        @test itp((1.0, 1.0)) ≈ 6.0
        @test itp((2.0, 3.0)) ≈ 14.0

        # Test at midpoints (should be exact for linear function)
        @test itp((0.5, 0.5)) ≈ 2.0*0.5 + 3.0*0.5 + 1.0
        @test itp((1.5, 2.5)) ≈ 2.0*1.5 + 3.0*2.5 + 1.0

        # Test random interior points
        @test itp((0.3, 0.7)) ≈ 2.0*0.3 + 3.0*0.7 + 1.0 atol=1e-14
        @test itp((1.8, 2.2)) ≈ 2.0*1.8 + 3.0*2.2 + 1.0 atol=1e-14
    end

    @testset "2D bilinear with product function" begin
        # f(x,y) = x*y is bilinear - exact on cell corners
        x = range(0.0, 2.0, 5)
        y = range(0.0, 3.0, 7)
        data = [xi * yj for xi in x, yj in y]

        itp = linear_interp((x, y), data)

        # Exact at grid points
        @test itp((0.0, 0.0)) ≈ 0.0
        @test itp((1.0, 1.5)) ≈ 1.5
        @test itp((2.0, 3.0)) ≈ 6.0

        # The product function x*y is exactly representable by bilinear interpolation
        # within each cell, so test at cell midpoints
        @test itp((0.25, 0.25)) ≈ 0.25 * 0.25 atol=1e-14
        @test itp((0.75, 1.25)) ≈ 0.75 * 1.25 atol=1e-14
    end

    # ========================================
    # 3D Trilinear Exactness Tests
    # ========================================
    @testset "3D trilinear exactness" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0]
        z = [0.0, 1.0, 2.0]

        # Linear function: f(x,y,z) = x + 2y + 3z
        data = [xi + 2.0*yj + 3.0*zk for xi in x, yj in y, zk in z]

        itp = linear_interp((x, y, z), data)

        # Grid points
        @test itp((0.0, 0.0, 0.0)) ≈ 0.0
        @test itp((1.0, 0.0, 0.0)) ≈ 1.0
        @test itp((0.0, 1.0, 0.0)) ≈ 2.0
        @test itp((0.0, 0.0, 1.0)) ≈ 3.0
        @test itp((2.0, 1.0, 2.0)) ≈ 2.0 + 2.0 + 6.0

        # Interior points
        @test itp((0.5, 0.5, 0.5)) ≈ 0.5 + 1.0 + 1.5 atol=1e-14
        @test itp((1.5, 0.3, 1.7)) ≈ 1.5 + 0.6 + 5.1 atol=1e-14
    end

    # ========================================
    # Derivative Tests
    # ========================================
    @testset "first derivatives" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 3.0, 16)

        # f(x,y) = 2x + 3y + 1
        # ∂f/∂x = 2, ∂f/∂y = 3
        data = [2.0*xi + 3.0*yj + 1.0 for xi in x, yj in y]
        itp = linear_interp((x, y), data)

        # Test ∂f/∂x at various points
        @test itp((0.5, 0.5); deriv=Val((1,0))) ≈ 2.0 atol=1e-12
        @test itp((1.3, 2.1); deriv=Val((1,0))) ≈ 2.0 atol=1e-12

        # Test ∂f/∂y at various points
        @test itp((0.5, 0.5); deriv=Val((0,1))) ≈ 3.0 atol=1e-12
        @test itp((1.3, 2.1); deriv=Val((0,1))) ≈ 3.0 atol=1e-12

        # Test that ∂²f/∂x² = 0 (linear has no second derivative)
        @test itp((0.5, 0.5); deriv=Val((2,0))) ≈ 0.0 atol=1e-14
        @test itp((0.5, 0.5); deriv=Val((0,2))) ≈ 0.0 atol=1e-14
    end

    @testset "3D first derivatives" begin
        x = range(0.0, 1.0, 6)
        y = range(0.0, 1.0, 6)
        z = range(0.0, 1.0, 6)

        # f(x,y,z) = x + 2y + 3z
        data = [xi + 2.0*yj + 3.0*zk for xi in x, yj in y, zk in z]
        itp = linear_interp((x, y, z), data)

        # ∂f/∂x = 1, ∂f/∂y = 2, ∂f/∂z = 3
        @test itp((0.5, 0.5, 0.5); deriv=Val((1,0,0))) ≈ 1.0 atol=1e-12
        @test itp((0.5, 0.5, 0.5); deriv=Val((0,1,0))) ≈ 2.0 atol=1e-12
        @test itp((0.5, 0.5, 0.5); deriv=Val((0,0,1))) ≈ 3.0 atol=1e-12
    end

    @testset "second+ derivatives return zero" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = rand(3, 3)
        itp = linear_interp((x, y), data)

        # Same-axis second derivatives should be zero for linear
        # (linear interpolation has constant slope per axis)
        @test itp((0.5, 0.5); deriv=Val((2,0))) == 0.0
        @test itp((0.5, 0.5); deriv=Val((0,2))) == 0.0
        @test itp((0.5, 0.5); deriv=2) == 0.0

        # Note: Mixed cross-derivatives Val((1,1)) are NOT zero for bilinear!
        # ∂²f/∂x∂y = (f₁₁ - f₁₀ - f₀₁ + f₀₀)/(h·k) which is non-zero in general
    end

    # ========================================
    # API Tests
    # ========================================
    @testset "vector API" begin
        x = range(0.0, 1.0, 5)
        y = range(0.0, 1.0, 5)
        data = [xi + yj for xi in x, yj in y]
        itp = linear_interp((x, y), data)

        # Vector query should work
        @test itp([0.5, 0.5]) ≈ 1.0
        @test itp([0.25, 0.75]) ≈ 1.0
    end

    @testset "batch SoA queries" begin
        x = range(0.0, 1.0, 5)
        y = range(0.0, 1.0, 5)
        data = [xi + yj for xi in x, yj in y]
        itp = linear_interp((x, y), data)

        xs = [0.1, 0.5, 0.9]
        ys = [0.2, 0.5, 0.8]
        results = itp((xs, ys))

        @test results ≈ [0.3, 1.0, 1.7]
    end

    @testset "batch AoS queries" begin
        x = range(0.0, 1.0, 5)
        y = range(0.0, 1.0, 5)
        data = [xi + yj for xi in x, yj in y]
        itp = linear_interp((x, y), data)

        points = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)]
        results = itp(points)

        @test results ≈ [0.3, 1.0, 1.7]
    end

    # ========================================
    # Extrapolation Tests
    # ========================================
    @testset "extrapolation modes" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [xi + yj for xi in x, yj in y]

        @testset "extrap=:none (domain error)" begin
            itp = linear_interp((x, y), data; extrap=:none)
            @test_throws DomainError itp((-0.1, 0.5))
            @test_throws DomainError itp((0.5, 2.1))
            @test_throws DomainError itp((-0.1, -0.1))
        end

        @testset "extrap=:constant" begin
            itp = linear_interp((x, y), data; extrap=:constant)
            # Query beyond domain should clamp to boundary
            @test itp((-0.5, 0.5)) ≈ itp((0.0, 0.5))
            @test itp((2.5, 0.5)) ≈ itp((2.0, 0.5))
            @test itp((0.5, -0.5)) ≈ itp((0.5, 0.0))
            @test itp((0.5, 2.5)) ≈ itp((0.5, 2.0))
        end

        @testset "extrap=:extension" begin
            itp = linear_interp((x, y), data; extrap=:extension)
            # Linear extension beyond domain
            @test itp((-0.5, 0.5)) ≈ -0.5 + 0.5  # Linear extrapolation
            @test itp((2.5, 0.5)) ≈ 2.5 + 0.5
        end

        @testset "extrap=:wrap" begin
            itp = linear_interp((x, y), data; extrap=:wrap)
            # Wrapping should work
            @test itp((2.5, 0.5)) ≈ itp((0.5, 0.5))  # 2.5 wraps to 0.5
        end

        @testset "per-axis extrap configuration" begin
            itp = linear_interp((x, y), data; extrap=(:none, :constant))
            # x has :none, y has :constant
            @test itp((0.5, 2.5)) == itp((0.5, 2.0))  # y clamped
            @test_throws DomainError itp((2.5, 0.5))   # x throws
        end
    end

    # ========================================
    # Grid Type Tests
    # ========================================
    @testset "Range grid support" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 2.0, 21)
        data = [xi * yj for xi in x, yj in y]

        itp = linear_interp((x, y), data)
        @test itp((0.5, 1.0)) ≈ 0.5 atol=1e-12
    end

    @testset "mixed grid types" begin
        x = range(0.0, 1.0, 5)   # Range
        y = [0.0, 0.3, 0.7, 1.0]  # Vector (non-uniform)
        data = [xi + yj for xi in x, yj in y]

        itp = linear_interp((x, y), data)
        @test itp((0.5, 0.5)) ≈ 1.0 atol=1e-10
    end

    @testset "search policy" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(range(0.0, 1.0, 11))
        data = [xi + yj for xi in x, yj in y]

        # Default Binary
        itp_default = linear_interp((x, y), data)
        @test itp_default((0.5, 0.5)) ≈ 1.0

        # Mixed search policies
        itp_mixed = linear_interp((x, y), data; search=(Binary(), LinearBinary{8}()))
        @test itp_mixed((0.5, 0.5)) ≈ 1.0
    end

    # ========================================
    # One-Shot API Tests
    # ========================================
    @testset "one-shot API" begin
        x = range(0.0, 1.0, 5)
        y = range(0.0, 1.0, 5)
        data = [xi + yj for xi in x, yj in y]

        # Scalar one-shot
        val = linear_interp((x, y), data, (0.5, 0.5))
        @test val ≈ 1.0

        # Batch SoA one-shot
        xs = [0.25, 0.75]
        ys = [0.25, 0.75]
        vals = linear_interp((x, y), data, (xs, ys))
        @test vals ≈ [0.5, 1.5]

        # Batch AoS one-shot
        points = [(0.25, 0.25), (0.75, 0.75)]
        vals = linear_interp((x, y), data, points)
        @test vals ≈ [0.5, 1.5]
    end

    # ========================================
    # Type Introspection Tests
    # ========================================
    @testset "type introspection" begin
        x = range(0.0, 1.0, 5)
        y = range(0.0, 1.0, 5)
        data = rand(5, 5)

        itp = linear_interp((x, y), data)

        @test ndims(itp) == 2
        @test size(itp) == (5, 5)
        @test axes(itp) == (x, y)
    end

    # ========================================
    # Complex Value Tests
    # ========================================
    @testset "complex values" begin
        x = range(0.0, 1.0, 5)
        y = range(0.0, 1.0, 5)
        data = [complex(xi, yj) for xi in x, yj in y]

        itp = linear_interp((x, y), data)

        # Test complex interpolation
        @test itp((0.5, 0.5)) ≈ complex(0.5, 0.5)
        @test itp((0.25, 0.75)) ≈ complex(0.25, 0.75)

        # Complex derivatives
        @test itp((0.5, 0.5); deriv=Val((1,0))) ≈ complex(1.0, 0.0)
        @test itp((0.5, 0.5); deriv=Val((0,1))) ≈ complex(0.0, 1.0)
    end

    # ========================================
    # Edge Cases
    # ========================================
    @testset "boundary queries" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        data = [xi + yj for xi in x, yj in y]
        itp = linear_interp((x, y), data)

        # Exact boundary queries
        @test itp((0.0, 0.0)) ≈ 0.0
        @test itp((2.0, 2.0)) ≈ 4.0
        @test itp((0.0, 2.0)) ≈ 2.0
        @test itp((2.0, 0.0)) ≈ 2.0
    end

    @testset "single cell grid" begin
        # Minimal 2x2 grid (one cell)
        x = [0.0, 1.0]
        y = [0.0, 1.0]
        data = [0.0 1.0; 2.0 3.0]  # corners: (0,0)=0, (1,0)=2, (0,1)=1, (1,1)=3
        itp = linear_interp((x, y), data)

        @test itp((0.0, 0.0)) ≈ 0.0
        @test itp((1.0, 0.0)) ≈ 2.0
        @test itp((0.0, 1.0)) ≈ 1.0
        @test itp((1.0, 1.0)) ≈ 3.0
        @test itp((0.5, 0.5)) ≈ 1.5  # center
    end

    # ========================================
    # REPL Display Tests
    # ========================================
    @testset "show methods" begin
        x = range(0.0, 1.0, 5)
        y = range(0.0, 1.0, 5)
        data = rand(5, 5)
        itp = linear_interp((x, y), data)

        # Compact show
        str = sprint(show, itp)
        @test occursin("LinearInterpolantND", str)
        @test occursin("5×5", str)

        # Verbose show
        str_verbose = sprint(show, MIME("text/plain"), itp)
        @test occursin("LinearInterpolantND", str_verbose)
        @test occursin("Extrap:", str_verbose)  # Should have extrapolation info
    end

    # ========================================
    # Integer Grid Promotion (regression)
    # ========================================
    @testset "integer grid promotion" begin
        # f(x,y) = 2x + 3y + 1 — bilinear should reproduce exactly
        _f_lin(xi, yj) = 2.0 * xi + 3.0 * yj + 1.0

        # Vector{Int} grids
        @testset "Vector{Int} grids" begin
            x = [0, 1, 2]
            y = [0, 1, 2]
            data = [_f_lin(xi, yj) for xi in x, yj in y]
            itp = linear_interp((x, y), data)
            @test itp((0.5, 0.5)) ≈ _f_lin(0.5, 0.5)
        end

        # UnitRange{Int} grids
        @testset "UnitRange{Int} grids" begin
            x = 0:2
            y = 0:2
            data = [_f_lin(xi, yj) for xi in x, yj in y]
            itp = linear_interp((x, y), data)
            @test itp((0.5, 0.5)) ≈ _f_lin(0.5, 0.5)
        end

        # Mixed Int + Float grids
        @testset "mixed Int and Float grids" begin
            x = [0, 1, 2]
            y = [0.0, 1.0, 2.0]
            data = [_f_lin(xi, yj) for xi in x, yj in y]
            itp = linear_interp((x, y), data)
            @test itp((0.5, 0.5)) ≈ _f_lin(0.5, 0.5)
        end

        # One-shot API with integer grids
        @testset "one-shot with integer grids" begin
            x = [0, 1, 2]
            y = [0, 1, 2]
            data = [_f_lin(xi, yj) for xi in x, yj in y]
            @test linear_interp((x, y), data, (0.5, 0.5)) ≈ _f_lin(0.5, 0.5)
        end
    end

    # ========================================
    # Zero-Allocation One-Shot Tests
    # ========================================
    #
    # Each test uses a full function barrier: setup + warmup + @allocated
    # all inside one function. This avoids @testset-scope boxing artifacts.

    function _alloc_test_linear_default()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        linear_interp((x, y), data, query)
        linear_interp((x, y), data, query)
        @allocated linear_interp((x, y), data, query)
    end

    function _alloc_test_linear_deriv()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        linear_interp((x, y), data, query; deriv=1)
        linear_interp((x, y), data, query; deriv=1)
        @allocated linear_interp((x, y), data, query; deriv=1)
    end

    function _alloc_test_linear_deriv_val()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        linear_interp((x, y), data, query; deriv=Val((1, 0)))
        linear_interp((x, y), data, query; deriv=Val((1, 0)))
        @allocated linear_interp((x, y), data, query; deriv=Val((1, 0)))
    end

    function _alloc_test_linear_extrap_constant()
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 10)
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        linear_interp((x, y), data, query; extrap=:constant)
        linear_interp((x, y), data, query; extrap=:constant)
        @allocated linear_interp((x, y), data, query; extrap=:constant)
    end

    function _alloc_test_linear_extrap_extension()
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 10)
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        linear_interp((x, y), data, query; extrap=:extension)
        linear_interp((x, y), data, query; extrap=:extension)
        @allocated linear_interp((x, y), data, query; extrap=:extension)
    end

    function _alloc_test_linear_3d()
        x = range(0.0, 2.0, 10)
        y = range(0.0, 1.0, 8)
        z = range(0.0, 3.0, 6)
        data = [xi + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        linear_interp((x, y, z), data, query)
        linear_interp((x, y, z), data, query)
        @allocated linear_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot" begin
        @testset "zero-alloc scalar (Range grids, default)" begin
            @test _alloc_test_linear_default() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, deriv=1)" begin
            @test _alloc_test_linear_deriv() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, deriv=Val)" begin
            @test _alloc_test_linear_deriv_val() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, extrap=:constant)" begin
            @test _alloc_test_linear_extrap_constant() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, extrap=:extension)" begin
            @test _alloc_test_linear_extrap_extension() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D Range grids)" begin
            @test _alloc_test_linear_3d() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Zero-Allocation One-Shot (Vector grids)
    # ========================================
    #
    # Vector grids must also be zero-allocation via pool-based spacing.

    function _alloc_test_linear_vector_default()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        linear_interp((x, y), data, query)
        linear_interp((x, y), data, query)
        @allocated linear_interp((x, y), data, query)
    end

    function _alloc_test_linear_vector_deriv()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        linear_interp((x, y), data, query; deriv=Val((1, 0)))
        linear_interp((x, y), data, query; deriv=Val((1, 0)))
        @allocated linear_interp((x, y), data, query; deriv=Val((1, 0)))
    end

    function _alloc_test_linear_vector_3d()
        x = collect(range(0.0, 2.0, 10))
        y = collect(range(0.0, 1.0, 8))
        z = collect(range(0.0, 3.0, 6))
        data = [xi + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        linear_interp((x, y, z), data, query)
        linear_interp((x, y, z), data, query)
        @allocated linear_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot (Vector grids)" begin
        @testset "zero-alloc scalar (Vector grids, default)" begin
            @test _alloc_test_linear_vector_default() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Vector grids, deriv=Val)" begin
            @test _alloc_test_linear_vector_deriv() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D Vector grids)" begin
            @test _alloc_test_linear_vector_3d() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Mixed-Grid Allocation Tests (Range + Vector)
    # ========================================
    #
    # Heterogeneous grid tuples (ScalarSpacing + VectorSpacing) must be zero-allocation.
    # Catches ntuple closure boxing on heterogeneous inputs.

    function _alloc_test_linear_mixed_2d()
        x = range(0.0, 2.0, 20)          # Range → ScalarSpacing
        y = collect(range(0.0, 1.0, 15)) # Vector → VectorSpacing
        data = [xi + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        linear_interp((x, y), data, query)
        linear_interp((x, y), data, query)
        @allocated linear_interp((x, y), data, query)
    end

    function _alloc_test_linear_mixed_3d()
        x = range(0.0, 2.0, 10)          # Range → ScalarSpacing
        y = collect(range(0.0, 1.0, 8))  # Vector → VectorSpacing
        z = range(0.0, 3.0, 6)           # Range → ScalarSpacing
        data = [xi + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        linear_interp((x, y, z), data, query)
        linear_interp((x, y, z), data, query)
        @allocated linear_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot (Mixed grids: Range + Vector)" begin
        @testset "zero-alloc scalar (2D mixed grid)" begin
            @test _alloc_test_linear_mixed_2d() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D mixed grid)" begin
            @test _alloc_test_linear_mixed_3d() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Integer Data Type Promotion (P1-B verification)
    # ========================================

    @testset "Integer data type promotion" begin
        x = range(0.0, 4.0, 5)
        y = range(0.0, 4.0, 5)
        data = [i + j for i in 0:4, j in 0:4]  # Matrix{Int}
        result = linear_interp((x, y), data, (1.5, 2.5))
        @test result isa Float64
        @test result ≈ 4.0
        results = linear_interp((x, y), data, ([1.5], [2.5]))
        @test eltype(results) == Float64
    end

    # ========================================
    # In-Place Batch Allocation Tests
    # ========================================
    #
    # In-place paths write into a pre-allocated output buffer.
    # These must be truly zero-allocation (only output + THRESHOLD).

    function _alloc_test_linear_inplace_soa()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = linear_interp((x, y), data)
        xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
        yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
        out = Vector{Float64}(undef, 5)
        itp(out, (xqs, yqs))
        itp(out, (xqs, yqs))
        @allocated itp(out, (xqs, yqs))
    end

    function _alloc_test_linear_inplace_aos()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = linear_interp((x, y), data)
        points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6), (2.0, 0.8), (3.0, 1.0)]
        out = Vector{Float64}(undef, 5)
        itp(out, points)
        itp(out, points)
        @allocated itp(out, points)
    end

    @testset "In-Place Batch Allocation Tests" begin
        @testset "in-place SoA batch (Range grids)" begin
            @test _alloc_test_linear_inplace_soa() <= ND_ALLOC_THRESHOLD
        end

        @testset "in-place AoS batch (Range grids)" begin
            @test _alloc_test_linear_inplace_aos() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Oneshot In-Place API (linear_interp!)
    # ========================================

    @testset "Oneshot In-Place (linear_interp!)" begin
        @testset "SoA correctness" begin
            x = range(0.0, 2π, 21)
            y = range(0.0, π, 11)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
            yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
            ref = linear_interp((x, y), data, (xqs, yqs))
            out = similar(ref)
            linear_interp!(out, (x, y), data, (xqs, yqs))
            @test out ≈ ref atol=1e-14
        end

        @testset "AoS correctness" begin
            x = range(0.0, 2π, 21)
            y = range(0.0, π, 11)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6), (2.0, 0.8), (3.0, 1.0)]
            ref = linear_interp((x, y), data, points)
            out = similar(ref)
            linear_interp!(out, (x, y), data, points)
            @test out ≈ ref atol=1e-14
        end

        @testset "SoA with deriv" begin
            x = range(0.0, 2π, 21)
            y = range(0.0, π, 11)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            xqs = [0.5, 1.0, 1.5]
            yqs = [0.2, 0.4, 0.6]
            ref = linear_interp((x, y), data, (xqs, yqs); deriv=1)
            out = similar(ref)
            linear_interp!(out, (x, y), data, (xqs, yqs); deriv=1)
            @test out ≈ ref atol=1e-14
        end

        @testset "DimensionMismatch on wrong output length" begin
            x = range(0.0, 1.0, 10)
            y = range(0.0, 1.0, 10)
            data = [xi + yj for xi in x, yj in y]
            xqs = [0.5, 0.6, 0.7]
            yqs = [0.5, 0.6, 0.7]
            out = zeros(5)
            @test_throws DimensionMismatch linear_interp!(out, (x, y), data, (xqs, yqs))
        end
    end

    function _alloc_test_oneshot_inplace_soa_linear()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.4, 0.6]
        out = Vector{Float64}(undef, 3)
        linear_interp!(out, (x, y), data, (xqs, yqs))
        linear_interp!(out, (x, y), data, (xqs, yqs))
        @allocated linear_interp!(out, (x, y), data, (xqs, yqs))
    end

    function _alloc_test_oneshot_inplace_aos_linear()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        points = [(0.5, 0.2), (1.0, 0.4), (1.5, 0.6)]
        out = Vector{Float64}(undef, 3)
        linear_interp!(out, (x, y), data, points)
        linear_interp!(out, (x, y), data, points)
        @allocated linear_interp!(out, (x, y), data, points)
    end

    @testset "Oneshot In-Place Allocation Tests" begin
        @testset "oneshot in-place SoA (Range grids)" begin
            @test _alloc_test_oneshot_inplace_soa_linear() <= ND_ALLOC_THRESHOLD
        end

        @testset "oneshot in-place AoS (Range grids)" begin
            @test _alloc_test_oneshot_inplace_aos_linear() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Oneshot In-Place Allocation Tests (Vector grids)
    # ========================================

    function _alloc_test_oneshot_inplace_soa_linear_vec()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi + yj for xi in x, yj in y]
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.5, 0.8]
        out = Vector{Float64}(undef, 3)
        linear_interp!(out, (x, y), data, (xqs, yqs))
        linear_interp!(out, (x, y), data, (xqs, yqs))
        @allocated linear_interp!(out, (x, y), data, (xqs, yqs))
    end

    function _alloc_test_oneshot_inplace_aos_linear_vec()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi + yj for xi in x, yj in y]
        points = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
        out = Vector{Float64}(undef, 3)
        linear_interp!(out, (x, y), data, points)
        linear_interp!(out, (x, y), data, points)
        @allocated linear_interp!(out, (x, y), data, points)
    end

    @testset "Oneshot In-Place Allocation Tests (Vector grids)" begin
        @testset "oneshot in-place SoA (Vector grids)" begin
            @test _alloc_test_oneshot_inplace_soa_linear_vec() <= ND_ALLOC_THRESHOLD
        end

        @testset "oneshot in-place AoS (Vector grids)" begin
            @test _alloc_test_oneshot_inplace_aos_linear_vec() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Pool Rewind Verification
    # ========================================

    @testset "Pool rewind after oneshot (linear)" begin
        xv = collect(range(0.0, 2π, 21))
        yv = collect(range(0.0, π, 11))
        data = [sin(xi) * cos(yj) for xi in xv, yj in yv]
        query = (1.0, 0.5)
        xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
        yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
        pts = [(xqs[i], yqs[i]) for i in 1:5]

        # Warmup
        linear_interp((xv, yv), data, query)
        linear_interp((xv, yv), data, (xqs, yqs))
        linear_interp((xv, yv), data, pts)
        out = Vector{Float64}(undef, 5)
        linear_interp!(out, (xv, yv), data, (xqs, yqs))
        linear_interp!(out, (xv, yv), data, pts)

        pool = get_task_local_pool()

        @testset "scalar oneshot" begin
            n_before = pool.float64.n_active
            linear_interp((xv, yv), data, query)
            @test pool.float64.n_active == n_before
        end

        @testset "SoA batch oneshot" begin
            n_before = pool.float64.n_active
            linear_interp((xv, yv), data, (xqs, yqs))
            @test pool.float64.n_active == n_before
        end

        @testset "AoS batch oneshot" begin
            n_before = pool.float64.n_active
            linear_interp((xv, yv), data, pts)
            @test pool.float64.n_active == n_before
        end

        @testset "SoA in-place oneshot" begin
            n_before = pool.float64.n_active
            linear_interp!(out, (xv, yv), data, (xqs, yqs))
            @test pool.float64.n_active == n_before
        end

        @testset "AoS in-place oneshot" begin
            n_before = pool.float64.n_active
            linear_interp!(out, (xv, yv), data, pts)
            @test pool.float64.n_active == n_before
        end
    end
end
