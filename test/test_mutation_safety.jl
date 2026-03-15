# ========================================
# Mutation Safety Tests
# ========================================
#
# TDD red-phase: tests that verify interpolants and adjoints
# are immune to external mutation of input arrays (grids, data).
#
# Issue: https://github.com/ProjectTorreyPines/FastInterpolations.jl/issues/80
#
# Once an interpolant is constructed, its results must NEVER change
# when the caller mutates the original x, y, or data arrays.
# Same principle applies to adjoint operators.

using Test
using FastInterpolations

# ============================================================
# Helper: build common test data
# ============================================================

function _make_1d_test_data(; n = 50)
    x = collect(range(0.0, 10.0, n))
    y = sin.(x)
    return x, y
end

function _make_2d_test_data(; nx = 10, ny = 12)
    x = collect(range(0.0, 2π, nx))
    y = collect(range(0.0, π, ny))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    return (x, y), data
end

function _make_3d_test_data(; nx = 8, ny = 9, nz = 7)
    x = collect(range(0.0, 2π, nx))
    y = collect(range(0.0, π, ny))
    z = collect(range(0.0, 1.0, nz))
    data = [sin(xi) * cos(yj) * (1.0 + zk) for xi in x, yj in y, zk in z]
    return (x, y, z), data
end

# Query point inside domain (avoid boundaries for cleaner tests)
const Q1D = 2.4
const Q2D = (1.5, 0.8)
const Q3D = (1.5, 0.8, 0.4)

# Struct definitions for §13 (must be at top level, not inside @testset)
mutable struct _MutTestFooLinear{T, I}
    xs::T
    ys::T
    interp::I
end

mutable struct _MutTestFooConstant{T, I}
    xs::T
    ys::T
    interp::I
end

@testset "Mutation Safety (issue #80)" begin

    # ============================================================
    # §1  1D Interpolant — y-data mutation
    # ============================================================

    @testset "1D y-data mutation" begin
        @testset "ConstantInterpolant" begin
            x, y = _make_1d_test_data()
            itp = constant_interp(x, y)
            val_before = itp(Q1D)
            y .= 0.0
            val_after = itp(Q1D)
            @test val_before == val_after
        end

        @testset "LinearInterpolant" begin
            x, y = _make_1d_test_data()
            itp = linear_interp(x, y)
            val_before = itp(Q1D)
            y .= 0.0
            val_after = itp(Q1D)
            @test val_before == val_after
        end

        @testset "QuadraticInterpolant" begin
            x, y = _make_1d_test_data()
            itp = quadratic_interp(x, y)
            val_before = itp(Q1D)
            y .= 0.0
            val_after = itp(Q1D)
            @test val_before == val_after
        end

        @testset "CubicInterpolant" begin
            x, y = _make_1d_test_data()
            itp = cubic_interp(x, y)
            val_before = itp(Q1D)
            y .= 0.0
            val_after = itp(Q1D)
            @test val_before == val_after
        end
    end

    # ============================================================
    # §2  1D Interpolant — x-grid mutation (Vector grids)
    # ============================================================
    # Mutate grid points NEAR the query point to ensure the
    # search/interpolation cell is affected.
    # Q1D = 2.4 → falls between index 12 and 13 (step ≈ 0.2041)

    @testset "1D x-grid mutation" begin
        @testset "ConstantInterpolant" begin
            x, y = _make_1d_test_data()
            itp = constant_interp(x, y)
            val_before = itp(Q1D)
            x[12] = 100.0  # mutate grid near query point
            val_after = itp(Q1D)
            @test val_before == val_after
        end

        @testset "LinearInterpolant" begin
            x, y = _make_1d_test_data()
            itp = linear_interp(x, y)
            val_before = itp(Q1D)
            x[12] = 100.0
            val_after = itp(Q1D)
            @test val_before == val_after
        end

        @testset "QuadraticInterpolant" begin
            x, y = _make_1d_test_data()
            itp = quadratic_interp(x, y)
            val_before = itp(Q1D)
            x[12] = 100.0
            val_after = itp(Q1D)
            @test val_before == val_after
        end

        @testset "CubicInterpolant" begin
            x, y = _make_1d_test_data()
            itp = cubic_interp(x, y)
            val_before = itp(Q1D)
            x[12] = 100.0
            val_after = itp(Q1D)
            @test val_before == val_after
        end
    end

    # ============================================================
    # §3  1D Interpolant — y mutation with PeriodicBC
    # ============================================================

    @testset "1D y-data mutation (PeriodicBC)" begin
        @testset "CubicInterpolant" begin
            x = collect(range(0.0, 2π, 50))
            y = sin.(x)
            y[end] = y[1]
            itp = cubic_interp(x, y; bc = PeriodicBC())
            val_before = itp(Q1D)
            y .= 0.0
            val_after = itp(Q1D)
            @test val_before == val_after
        end
    end

    # ============================================================
    # §4  1D Interpolant — y mutation with extrap modes
    # ============================================================
    # Query at 11.0 (past end of domain [0,10]) to expose boundary
    # value mutation. y[end] = sin(10.0) ≈ -0.544 ≠ 0

    @testset "1D y-data mutation (extrap modes)" begin
        for (name, extrap) in [
                ("ExtendExtrap", ExtendExtrap()),
                ("ClampExtrap", ClampExtrap()),
            ]
            @testset "LinearInterpolant $name" begin
                x, y = _make_1d_test_data()
                itp = linear_interp(x, y; extrap = extrap)
                val_before = itp(11.0)  # past end of domain
                y .= 0.0
                val_after = itp(11.0)
                @test val_before == val_after
            end

            @testset "ConstantInterpolant $name" begin
                x, y = _make_1d_test_data()
                itp = constant_interp(x, y; extrap = extrap)
                val_before = itp(11.0)
                y .= 0.0
                val_after = itp(11.0)
                @test val_before == val_after
            end
        end
    end

    # ============================================================
    # §5  ND Interpolant — data mutation (2D)
    # ============================================================

    @testset "ND data mutation (2D)" begin
        @testset "LinearInterpolantND" begin
            grids, data = _make_2d_test_data()
            itp = linear_interp(grids, data)
            val_before = itp(Q2D)
            data .= 0.0
            val_after = itp(Q2D)
            @test val_before == val_after
        end

        @testset "QuadraticInterpolantND" begin
            grids, data = _make_2d_test_data()
            itp = quadratic_interp(grids, data)
            val_before = itp(Q2D)
            data .= 0.0
            val_after = itp(Q2D)
            @test val_before == val_after
        end

        @testset "CubicInterpolantND" begin
            grids, data = _make_2d_test_data()
            itp = cubic_interp(grids, data)
            val_before = itp(Q2D)
            data .= 0.0
            val_after = itp(Q2D)
            @test val_before == val_after
        end

        @testset "ConstantInterpolantND" begin
            grids, data = _make_2d_test_data()
            itp = constant_interp(grids, data)
            val_before = itp(Q2D)
            data .= 0.0
            val_after = itp(Q2D)
            @test val_before == val_after
        end
    end

    # ============================================================
    # §6  ND Interpolant — data mutation (3D)
    # ============================================================

    @testset "ND data mutation (3D)" begin
        @testset "LinearInterpolantND" begin
            grids, data = _make_3d_test_data()
            itp = linear_interp(grids, data)
            val_before = itp(Q3D)
            data .= 0.0
            val_after = itp(Q3D)
            @test val_before == val_after
        end

        @testset "CubicInterpolantND" begin
            grids, data = _make_3d_test_data()
            itp = cubic_interp(grids, data)
            val_before = itp(Q3D)
            data .= 0.0
            val_after = itp(Q3D)
            @test val_before == val_after
        end

        @testset "ConstantInterpolantND" begin
            grids, data = _make_3d_test_data()
            itp = constant_interp(grids, data)
            val_before = itp(Q3D)
            data .= 0.0
            val_after = itp(Q3D)
            @test val_before == val_after
        end

        @testset "QuadraticInterpolantND" begin
            grids, data = _make_3d_test_data()
            itp = quadratic_interp(grids, data)
            val_before = itp(Q3D)
            data .= 0.0
            val_after = itp(Q3D)
            @test val_before == val_after
        end
    end

    # ============================================================
    # §7  ND Interpolant — grid mutation (2D)
    # ============================================================
    # Q2D = (1.5, 0.8), x = range(0, 2π, 10) step ≈ 0.698
    # Q2D[1]=1.5 → index ≈ 3.15, uses x[3] and x[4]
    # Mutate x[3] to expose the bug.

    @testset "ND grid mutation (2D)" begin
        @testset "LinearInterpolantND" begin
            grids, data = _make_2d_test_data()
            itp = linear_interp(grids, data)
            val_before = itp(Q2D)
            grids[1][3] = 100.0  # mutate near query point
            val_after = itp(Q2D)
            @test val_before == val_after
        end

        @testset "QuadraticInterpolantND" begin
            grids, data = _make_2d_test_data()
            itp = quadratic_interp(grids, data)
            val_before = itp(Q2D)
            grids[1][3] = 100.0
            val_after = itp(Q2D)
            @test val_before == val_after
        end

        @testset "CubicInterpolantND" begin
            grids, data = _make_2d_test_data()
            itp = cubic_interp(grids, data)
            val_before = itp(Q2D)
            grids[1][3] = 100.0
            val_after = itp(Q2D)
            @test val_before == val_after
        end

        @testset "ConstantInterpolantND" begin
            grids, data = _make_2d_test_data()
            itp = constant_interp(grids, data)
            val_before = itp(Q2D)
            grids[1][3] = 100.0
            val_after = itp(Q2D)
            @test val_before == val_after
        end
    end

    # ============================================================
    # §8  ND Interpolant — data mutation with PeriodicBC (2D)
    # ============================================================

    @testset "ND data mutation (2D, PeriodicBC)" begin
        @testset "CubicInterpolantND" begin
            nx, ny = 10, 12
            x = collect(range(0.0, 2π, nx))
            y = collect(range(0.0, 2π, ny))
            # Create data that satisfies periodicity on both axes
            data = [sin(xi) * sin(yj) for xi in x, yj in y]
            # Enforce exact periodicity: data[end,:] = data[1,:], data[:,end] = data[:,1]
            data[end, :] .= data[1, :]
            data[:, end] .= data[:, 1]
            itp = cubic_interp((x, y), data; bc = PeriodicBC())
            val_before = itp(Q2D)
            data .= 0.0
            val_after = itp(Q2D)
            @test val_before == val_after
        end
    end

    # ============================================================
    # §9  1D Adjoint — grid mutation
    # ============================================================

    @testset "1D Adjoint grid mutation" begin
        @testset "LinearAdjoint" begin
            x = collect(range(0.0, 1.0, 50))
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]
            y_bar = [1.0, 2.0, 3.0, 4.0, 5.0]
            adj = linear_adjoint(x, xq)
            result_before = adj(y_bar)
            x[25] = 100.0
            result_after = adj(y_bar)
            @test result_before == result_after
        end

        @testset "CubicAdjoint" begin
            x = collect(range(0.0, 1.0, 50))
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]
            y_bar = [1.0, 2.0, 3.0, 4.0, 5.0]
            adj = cubic_adjoint(x, xq)
            result_before = adj(y_bar)
            x[25] = 100.0
            result_after = adj(y_bar)
            @test result_before == result_after
        end

        @testset "ConstantAdjoint" begin
            x = collect(range(0.0, 1.0, 50))
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]
            y_bar = [1.0, 2.0, 3.0, 4.0, 5.0]
            adj = constant_adjoint(x, xq)
            result_before = adj(y_bar)
            x[25] = 100.0
            result_after = adj(y_bar)
            @test result_before == result_after
        end

        @testset "CubicAdjoint PeriodicBC" begin
            x = collect(range(0.0, 2π, 50))
            xq = [0.5, 1.0, 2.0, 3.0, 5.0]
            y_bar = [1.0, 2.0, 3.0, 4.0, 5.0]
            adj = cubic_adjoint(x, xq; bc = PeriodicBC())
            result_before = adj(y_bar)
            x[25] = 100.0
            result_after = adj(y_bar)
            @test result_before == result_after
        end
    end

    # ============================================================
    # §10  ND Adjoint — grid mutation (2D)
    # ============================================================

    @testset "ND Adjoint grid mutation (2D)" begin
        @testset "LinearAdjointND" begin
            x = collect(range(0.0, 1.0, 10))
            y = collect(range(0.0, 2.0, 12))
            xq = [0.2, 0.5, 0.8]
            yq = [0.3, 1.0, 1.5]
            y_bar = [1.0, 2.0, 3.0]
            adj = linear_adjoint((x, y), (xq, yq))
            result_before = adj(y_bar)
            x[5] = 100.0
            result_after = adj(y_bar)
            @test result_before == result_after
        end

        @testset "CubicAdjointND" begin
            x = collect(range(0.0, 1.0, 10))
            y = collect(range(0.0, 2.0, 12))
            xq = [0.2, 0.5, 0.8]
            yq = [0.3, 1.0, 1.5]
            y_bar = [1.0, 2.0, 3.0]
            adj = cubic_adjoint((x, y), (xq, yq))
            result_before = adj(y_bar)
            x[5] = 100.0
            result_after = adj(y_bar)
            @test result_before == result_after
        end

        @testset "ConstantAdjointND" begin
            x = collect(range(0.0, 1.0, 10))
            y = collect(range(0.0, 2.0, 12))
            xq = [0.2, 0.5, 0.8]
            yq = [0.3, 1.0, 1.5]
            y_bar = [1.0, 2.0, 3.0]
            adj = constant_adjoint((x, y), (xq, yq))
            result_before = adj(y_bar)
            x[5] = 100.0
            result_after = adj(y_bar)
            @test result_before == result_after
        end
    end

    # ============================================================
    # §11  1D Adjoint — derivative mode grid mutation
    # ============================================================

    @testset "1D Adjoint deriv grid mutation" begin
        @testset "CubicAdjoint deriv=1" begin
            x = collect(range(0.0, 1.0, 50))
            xq = [0.1, 0.3, 0.5, 0.7, 0.9]
            y_bar = [1.0, 2.0, 3.0, 4.0, 5.0]
            adj = cubic_adjoint(x, xq)
            result_before = adj(y_bar; deriv = DerivOp(1))
            x[25] = 100.0
            result_after = adj(y_bar; deriv = DerivOp(1))
            @test result_before == result_after
        end
    end

    # ============================================================
    # §12  Issue #80 exact MWE reproduction
    # ============================================================

    @testset "Issue #80 MWE" begin
        p = 2.4

        @testset "constant_interp" begin
            xx = collect(range(0, 10, 50))
            yy = sin.(xx)
            itp = constant_interp(xx, yy)
            val_before = itp(p)
            yy .= zero.(xx)
            val_after = itp(p)
            @test val_before == val_after
            @test val_before != 0.0
        end

        @testset "linear_interp" begin
            xx = collect(range(0, 10, 50))
            yy = sin.(xx)
            itp = linear_interp(xx, yy)
            val_before = itp(p)
            yy .= zero.(xx)
            val_after = itp(p)
            @test val_before == val_after
            @test val_before != 0.0
        end

        @testset "quadratic_interp" begin
            xx = collect(range(0, 10, 50))
            yy = sin.(xx)
            itp = quadratic_interp(xx, yy)
            val_before = itp(p)
            yy .= zero.(xx)
            val_after = itp(p)
            @test val_before == val_after
            @test val_before != 0.0
        end

        @testset "cubic_interp (should pass)" begin
            xx = collect(range(0, 10, 50))
            yy = sin.(xx)
            itp = cubic_interp(xx, yy)
            val_before = itp(p)
            yy .= zero.(xx)
            val_after = itp(p)
            @test val_before == val_after
            @test val_before != 0.0
        end
    end

    # ============================================================
    # §13  Mutable container pattern (issue #80)
    # ============================================================

    @testset "Mutable container pattern" begin
        @testset "LinearInterpolant in mutable struct" begin
            xs = collect(range(0, 10, 50))
            ys = sin.(xs)
            interp = linear_interp(xs, ys)
            f = _MutTestFooLinear(xs, ys, interp)
            val_original = f.interp(2.5)
            f.ys .= 0.0
            val_after_mutation = f.interp(2.5)
            @test val_original == val_after_mutation
        end

        @testset "ConstantInterpolant in mutable struct" begin
            xs = collect(range(0, 10, 50))
            ys = sin.(xs)
            interp = constant_interp(xs, ys)
            f = _MutTestFooConstant(xs, ys, interp)
            val_original = f.interp(2.5)
            f.ys .= 0.0
            val_after_mutation = f.interp(2.5)
            @test val_original == val_after_mutation
        end
    end

    # ============================================================
    # §14  Derivative evaluation after y-mutation
    # ============================================================

    @testset "1D derivative after y-mutation" begin
        @testset "QuadraticInterpolant deriv=1" begin
            x, y = _make_1d_test_data()
            itp = quadratic_interp(x, y)
            val_before = itp(Q1D; deriv = DerivOp(1))
            y .= 0.0
            val_after = itp(Q1D; deriv = DerivOp(1))
            @test val_before == val_after
        end

        @testset "CubicInterpolant deriv=1" begin
            x, y = _make_1d_test_data()
            itp = cubic_interp(x, y)
            val_before = itp(Q1D; deriv = DerivOp(1))
            y .= 0.0
            val_after = itp(Q1D; deriv = DerivOp(1))
            @test val_before == val_after
        end

        @testset "CubicInterpolant deriv=2" begin
            x, y = _make_1d_test_data()
            itp = cubic_interp(x, y)
            val_before = itp(Q1D; deriv = DerivOp(2))
            y .= 0.0
            val_after = itp(Q1D; deriv = DerivOp(2))
            @test val_before == val_after
        end
    end

    # ============================================================
    # §15  ND derivative evaluation after data mutation
    # ============================================================

    @testset "ND derivative after data mutation (2D)" begin
        @testset "LinearInterpolantND deriv=(1,0)" begin
            grids, data = _make_2d_test_data()
            itp = linear_interp(grids, data)
            val_before = itp(Q2D; deriv = DerivOp(1, 0))
            data .= 0.0
            val_after = itp(Q2D; deriv = DerivOp(1, 0))
            @test val_before == val_after
        end

        @testset "CubicInterpolantND deriv=(1,0)" begin
            grids, data = _make_2d_test_data()
            itp = cubic_interp(grids, data)
            val_before = itp(Q2D; deriv = DerivOp(1, 0))
            data .= 0.0
            val_after = itp(Q2D; deriv = DerivOp(1, 0))
            @test val_before == val_after
        end

        @testset "CubicInterpolantND deriv=(0,1)" begin
            grids, data = _make_2d_test_data()
            itp = cubic_interp(grids, data)
            val_before = itp(Q2D; deriv = DerivOp(0, 1))
            data .= 0.0
            val_after = itp(Q2D; deriv = DerivOp(0, 1))
            @test val_before == val_after
        end
    end

    # ============================================================
    # §16  Single-element mutation (subtle case)
    # ============================================================

    @testset "Single-element y mutation" begin
        @testset "LinearInterpolant" begin
            x, y = _make_1d_test_data()
            itp = linear_interp(x, y)
            val_before = itp(Q1D)
            y[12] = 999.0  # near query point
            val_after = itp(Q1D)
            @test val_before == val_after
        end

        @testset "LinearInterpolantND" begin
            grids, data = _make_2d_test_data()
            itp = linear_interp(grids, data)
            val_before = itp(Q2D)
            data[3, 4] = 999.0
            val_after = itp(Q2D)
            @test val_before == val_after
        end
    end

    # ============================================================
    # §17  Float32 mutation safety
    # ============================================================

    @testset "Float32 mutation safety" begin
        @testset "LinearInterpolant Float32" begin
            x = collect(range(0.0f0, 10.0f0, 50))
            y = sin.(x)
            itp = linear_interp(x, y)
            val_before = itp(2.4f0)
            y .= 0.0f0
            val_after = itp(2.4f0)
            @test val_before == val_after
        end

        @testset "CubicInterpolant Float32" begin
            x = collect(range(0.0f0, 10.0f0, 50))
            y = sin.(x)
            itp = cubic_interp(x, y)
            val_before = itp(2.4f0)
            y .= 0.0f0
            val_after = itp(2.4f0)
            @test val_before == val_after
        end

        @testset "LinearInterpolantND Float32 (2D)" begin
            x = collect(range(0.0f0, Float32(2π), 10))
            y = collect(range(0.0f0, Float32(π), 12))
            data = Float32[sin(xi) * cos(yj) for xi in x, yj in y]
            itp = linear_interp((x, y), data)
            val_before = itp((1.5f0, 0.8f0))
            data .= 0.0f0
            val_after = itp((1.5f0, 0.8f0))
            @test val_before == val_after
        end
    end

    # ============================================================
    # §18  View-backed input construction (regression guard)
    # ============================================================
    # copy() in inner constructors must not break when the input
    # is a SubArray/view, since copy(view) → Vector while the
    # struct's type parameter was bound to the original view type.

    @testset "1D view-backed construction" begin
        @testset "ConstantInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y = sin.(x)
            itp = constant_interp(@view(x[:]), @view(y[:]))
            @test itp(Q1D) ≈ constant_interp(x, y)(Q1D)
        end

        @testset "LinearInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y = sin.(x)
            itp = linear_interp(@view(x[:]), @view(y[:]))
            @test itp(Q1D) ≈ linear_interp(x, y)(Q1D)
        end

        @testset "QuadraticInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y = sin.(x)
            itp = quadratic_interp(@view(x[:]), @view(y[:]))
            @test itp(Q1D) ≈ quadratic_interp(x, y)(Q1D)
        end

        @testset "CubicInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y = sin.(x)
            itp = cubic_interp(@view(x[:]), @view(y[:]))
            @test itp(Q1D) ≈ cubic_interp(x, y)(Q1D)
        end
    end

    @testset "ND view-backed construction (2D)" begin
        @testset "LinearInterpolantND" begin
            x = collect(range(0.0, 2π, 10))
            y = collect(range(0.0, π, 12))
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = linear_interp((@view(x[:]), @view(y[:])), @view(data[:, :]))
            @test itp(Q2D) ≈ linear_interp((x, y), data)(Q2D)
        end

        @testset "ConstantInterpolantND" begin
            x = collect(range(0.0, 2π, 10))
            y = collect(range(0.0, π, 12))
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = constant_interp((@view(x[:]), @view(y[:])), @view(data[:, :]))
            @test itp(Q2D) ≈ constant_interp((x, y), data)(Q2D)
        end

        @testset "QuadraticInterpolantND" begin
            x = collect(range(0.0, 2π, 10))
            y = collect(range(0.0, π, 12))
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = quadratic_interp((@view(x[:]), @view(y[:])), @view(data[:, :]))
            @test itp(Q2D) ≈ quadratic_interp((x, y), data)(Q2D)
        end

        @testset "CubicInterpolantND" begin
            x = collect(range(0.0, 2π, 10))
            y = collect(range(0.0, π, 12))
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = cubic_interp((@view(x[:]), @view(y[:])), @view(data[:, :]))
            @test itp(Q2D) ≈ cubic_interp((x, y), data)(Q2D)
        end
    end

    # ============================================================
    # §18b  ND view-backed mutation safety (grids + data)
    # ============================================================

    @testset "ND view mutation safety (2D)" begin
        @testset "ConstantInterpolantND" begin
            x = collect(range(0.0, 2π, 10))
            y = collect(range(0.0, π, 12))
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = constant_interp((@view(x[:]), @view(y[:])), @view(data[:, :]))
            val_before = itp(Q2D)
            data .= 0.0; x[3] = 100.0; y[4] = 100.0
            @test itp(Q2D) == val_before
        end

        @testset "LinearInterpolantND" begin
            x = collect(range(0.0, 2π, 10))
            y = collect(range(0.0, π, 12))
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = linear_interp((@view(x[:]), @view(y[:])), @view(data[:, :]))
            val_before = itp(Q2D)
            data .= 0.0; x[3] = 100.0; y[4] = 100.0
            @test itp(Q2D) == val_before
        end

        @testset "QuadraticInterpolantND" begin
            x = collect(range(0.0, 2π, 10))
            y = collect(range(0.0, π, 12))
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = quadratic_interp((@view(x[:]), @view(y[:])), @view(data[:, :]))
            val_before = itp(Q2D)
            data .= 0.0; x[3] = 100.0; y[4] = 100.0
            @test itp(Q2D) == val_before
        end

        @testset "CubicInterpolantND" begin
            x = collect(range(0.0, 2π, 10))
            y = collect(range(0.0, π, 12))
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = cubic_interp((@view(x[:]), @view(y[:])), @view(data[:, :]))
            val_before = itp(Q2D)
            data .= 0.0; x[3] = 100.0; y[4] = 100.0
            @test itp(Q2D) == val_before
        end
    end

    # ============================================================
    # §19  View-backed input + mutation (copy must decouple)
    # ============================================================

    @testset "1D view mutation safety" begin
        @testset "ConstantInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y = sin.(x)
            itp = constant_interp(@view(x[:]), @view(y[:]))
            val_before = itp(Q1D)
            y .= 0.0; x[12] = 100.0
            @test itp(Q1D) == val_before
        end

        @testset "LinearInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y = sin.(x)
            itp = linear_interp(@view(x[:]), @view(y[:]))
            val_before = itp(Q1D)
            y .= 0.0; x[12] = 100.0
            @test itp(Q1D) == val_before
        end

        @testset "QuadraticInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y = sin.(x)
            itp = quadratic_interp(@view(x[:]), @view(y[:]))
            val_before = itp(Q1D)
            y .= 0.0; x[12] = 100.0
            @test itp(Q1D) == val_before
        end

        @testset "CubicInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y = sin.(x)
            itp = cubic_interp(@view(x[:]), @view(y[:]))
            val_before = itp(Q1D)
            y .= 0.0; x[12] = 100.0
            @test itp(Q1D) == val_before
        end
    end

    # ============================================================
    # §20  Series view-backed construction (regression guard)
    # ============================================================
    # Series constructors must rebind X type after copy(x),
    # same as the 1D/ND fix in c38cf043.

    @testset "Series view-backed construction" begin
        x = collect(range(0.0, 10.0, 50))
        y1 = sin.(x)
        y2 = cos.(x)

        @testset "ConstantSeriesInterpolant" begin
            sitp = constant_interp(@view(x[:]), Series(y1, y2))
            ref = constant_interp(x, Series(y1, y2))
            @test sitp(Q1D) ≈ ref(Q1D)
        end

        @testset "LinearSeriesInterpolant" begin
            sitp = linear_interp(@view(x[:]), Series(y1, y2))
            ref = linear_interp(x, Series(y1, y2))
            @test sitp(Q1D) ≈ ref(Q1D)
        end

        @testset "QuadraticSeriesInterpolant" begin
            sitp = quadratic_interp(@view(x[:]), Series(y1, y2))
            ref = quadratic_interp(x, Series(y1, y2))
            @test sitp(Q1D) ≈ ref(Q1D)
        end

        @testset "CubicSeriesInterpolant" begin
            sitp = cubic_interp(@view(x[:]), Series(y1, y2))
            ref = cubic_interp(x, Series(y1, y2))
            @test sitp(Q1D) ≈ ref(Q1D)
        end
    end

    # ============================================================
    # §21  Series view-backed mutation safety
    # ============================================================

    @testset "Series view mutation safety" begin
        @testset "ConstantSeriesInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y1 = sin.(x); y2 = cos.(x)
            sitp = constant_interp(@view(x[:]), Series(y1, y2))
            val_before = sitp(Q1D)
            x[12] = 100.0
            @test sitp(Q1D) == val_before
        end

        @testset "LinearSeriesInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y1 = sin.(x); y2 = cos.(x)
            sitp = linear_interp(@view(x[:]), Series(y1, y2))
            val_before = sitp(Q1D)
            x[12] = 100.0
            @test sitp(Q1D) == val_before
        end

        @testset "QuadraticSeriesInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y1 = sin.(x); y2 = cos.(x)
            sitp = quadratic_interp(@view(x[:]), Series(y1, y2))
            val_before = sitp(Q1D)
            x[12] = 100.0
            @test sitp(Q1D) == val_before
        end

        @testset "CubicSeriesInterpolant" begin
            x = collect(range(0.0, 10.0, 50))
            y1 = sin.(x); y2 = cos.(x)
            sitp = cubic_interp(@view(x[:]), Series(y1, y2))
            val_before = sitp(Q1D)
            x[12] = 100.0
            @test sitp(Q1D) == val_before
        end
    end

end  # top-level @testset
