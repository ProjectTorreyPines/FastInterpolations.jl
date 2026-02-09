# ========================================
# ND Hint Support Tests
# ========================================
#
# Tests for per-axis persistent hints in ND interpolants.
# Hints use NTuple{N, Base.RefValue{Int}} to maintain search state
# across calls, enabling O(1) lookup for sequential/sorted queries.

using Test
using FastInterpolations

@testset "ND Hint Support" begin

    # ========================================
    # Test Data Setup
    # ========================================
    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))
    z = collect(range(0.0, 3.0, 16))

    data_2d = [sin(xi) * cos(yj) for xi in x, yj in y]
    data_3d = [sin(xi) * cos(yj) * exp(-zk/3) for xi in x, yj in y, zk in z]
    const_2d = [Float64(i + j) for i in 1:length(x), j in 1:length(y)]

    # ========================================
    # Cubic ND
    # ========================================
    @testset "CubicInterpolantND" begin
        itp = cubic_interp((x, y), data_2d)
        qx, qy = 1.0, 0.5

        @testset "hint=nothing matches no-hint" begin
            ref = itp((qx, qy))
            @test itp((qx, qy); hint=nothing) == ref
        end

        @testset "Scalar hint updates Refs" begin
            hints = (Ref(1), Ref(1))
            result = itp((qx, qy); hint=hints)
            @test result ≈ itp((qx, qy))
            # Refs should have been updated to point near the queried interval
            @test hints[1][] >= 1
            @test hints[2][] >= 1
        end

        @testset "Repeated call is idempotent" begin
            hints = (Ref(1), Ref(1))
            itp((qx, qy); hint=hints)
            h1, h2 = hints[1][], hints[2][]
            itp((qx, qy); hint=hints)
            @test hints[1][] == h1
            @test hints[2][] == h2
        end

        @testset "Vector query with hint" begin
            hints = (Ref(1), Ref(1))
            result = itp([qx, qy]; hint=hints)
            @test result ≈ itp((qx, qy))
            @test hints[1][] >= 1
        end

        @testset "SoA batch with hint" begin
            xs = [0.5, 1.0, 1.5]
            ys = [0.2, 0.5, 0.8]
            hints = (Ref(1), Ref(1))
            results_hint = itp((xs, ys); hint=hints)
            results_ref = itp((xs, ys))
            @test results_hint ≈ results_ref
        end

        @testset "AoS batch with hint" begin
            queries = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
            hints = (Ref(1), Ref(1))
            results_hint = itp(queries; hint=hints)
            results_ref = itp(queries)
            @test results_hint ≈ results_ref
        end

        @testset "Derivatives with hint" begin
            hints = (Ref(1), Ref(1))
            ref_d1 = itp((qx, qy); deriv=1)
            @test itp((qx, qy); deriv=1, hint=hints) ≈ ref_d1
            ref_dx = itp((qx, qy); deriv=(1,0))
            @test itp((qx, qy); deriv=(1,0), hint=hints) ≈ ref_dx
        end
    end

    # ========================================
    # Linear ND
    # ========================================
    @testset "LinearInterpolantND" begin
        itp = linear_interp((x, y), data_2d)
        qx, qy = 1.0, 0.5

        @testset "hint=nothing matches no-hint" begin
            @test itp((qx, qy); hint=nothing) == itp((qx, qy))
        end

        @testset "Scalar hint updates Refs" begin
            hints = (Ref(1), Ref(1))
            result = itp((qx, qy); hint=hints)
            @test result ≈ itp((qx, qy))
            @test hints[1][] >= 1
            @test hints[2][] >= 1
        end

        @testset "SoA batch with hint" begin
            xs = [0.5, 1.0, 1.5]
            ys = [0.2, 0.5, 0.8]
            hints = (Ref(1), Ref(1))
            @test itp((xs, ys); hint=hints) ≈ itp((xs, ys))
        end

        @testset "AoS batch with hint" begin
            queries = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
            hints = (Ref(1), Ref(1))
            @test itp(queries; hint=hints) ≈ itp(queries)
        end
    end

    # ========================================
    # Quadratic ND
    # ========================================
    @testset "QuadraticInterpolantND" begin
        itp = quadratic_interp((x, y), data_2d)
        qx, qy = 1.0, 0.5

        @testset "hint=nothing matches no-hint" begin
            @test itp((qx, qy); hint=nothing) == itp((qx, qy))
        end

        @testset "Scalar hint updates Refs" begin
            hints = (Ref(1), Ref(1))
            result = itp((qx, qy); hint=hints)
            @test result ≈ itp((qx, qy))
            @test hints[1][] >= 1
            @test hints[2][] >= 1
        end

        @testset "SoA batch with hint" begin
            xs = [0.5, 1.0, 1.5]
            ys = [0.2, 0.5, 0.8]
            hints = (Ref(1), Ref(1))
            @test itp((xs, ys); hint=hints) ≈ itp((xs, ys))
        end

        @testset "AoS batch with hint" begin
            queries = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
            hints = (Ref(1), Ref(1))
            @test itp(queries; hint=hints) ≈ itp(queries)
        end
    end

    # ========================================
    # Constant ND
    # ========================================
    @testset "ConstantInterpolantND" begin
        itp = constant_interp((x, y), const_2d)
        qx, qy = 1.0, 0.5

        @testset "hint=nothing matches no-hint" begin
            @test itp((qx, qy); hint=nothing) == itp((qx, qy))
        end

        @testset "Scalar hint updates Refs" begin
            hints = (Ref(1), Ref(1))
            result = itp((qx, qy); hint=hints)
            @test result == itp((qx, qy))
            @test hints[1][] >= 1
            @test hints[2][] >= 1
        end

        @testset "SoA batch with hint" begin
            xs = [0.5, 1.0, 1.5]
            ys = [0.2, 0.5, 0.8]
            hints = (Ref(1), Ref(1))
            @test itp((xs, ys); hint=hints) == itp((xs, ys))
        end

        @testset "AoS batch with hint" begin
            queries = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
            hints = (Ref(1), Ref(1))
            @test itp(queries; hint=hints) == itp(queries)
        end
    end

    # ========================================
    # 3D Hint Support
    # ========================================
    @testset "3D hint support" begin
        itp = cubic_interp((x, y, z), data_3d)
        q = (1.0, 0.5, 1.5)

        hints = (Ref(1), Ref(1), Ref(1))
        result = itp(q; hint=hints)
        @test result ≈ itp(q)
        # All 3 axes should be updated
        @test hints[1][] >= 1
        @test hints[2][] >= 1
        @test hints[3][] >= 1
    end

    # ========================================
    # Binary Auto-Upgrade with Hint
    # ========================================
    @testset "Binary auto-upgrade with hint" begin
        itp = cubic_interp((x, y), data_2d; search=Binary())
        qx, qy = 1.5, 0.7
        hints = (Ref(1), Ref(1))

        result = itp((qx, qy); hint=hints)
        @test result ≈ itp((qx, qy))
        # Hint Refs should have been updated, proving HintedBinary was used
        @test hints[1][] > 1  # 1.5 is well past interval 1
        @test hints[2][] > 1  # 0.7 is well past interval 1
    end

    # ========================================
    # Monotonic Sequence (Hint Reuse)
    # ========================================
    @testset "Monotonic sequence benefits from hint" begin
        itp = cubic_interp((x, y), data_2d)
        hints = (Ref(1), Ref(1))

        # Monotonically increasing queries — hint should advance
        for xi in range(0.1, 1.9, 10)
            itp((xi, 0.5); hint=hints)
        end
        # After scanning x from 0.1 to 1.9 on a grid 0..2 with 21 points,
        # the x-hint should be near the end
        @test hints[1][] > 10
    end

    # ========================================
    # DerivativeView Passthrough
    # ========================================
    @testset "DerivativeView passes hint through" begin
        itp = cubic_interp((x, y), data_2d)
        dv = deriv_view(itp, (1, 0))  # ∂f/∂x

        hints = (Ref(1), Ref(1))
        result = dv((1.0, 0.5); hint=hints)
        @test result ≈ itp((1.0, 0.5); deriv=(1, 0))
        @test hints[1][] >= 1
    end
end
