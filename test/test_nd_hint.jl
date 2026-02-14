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

    # Helper: compute expected interval index for a query on a grid
    # Mirrors what search_interval does: searchsortedlast clamped to [1, n-1]
    expected_interval(grid, q) = clamp(searchsortedlast(grid, q), 1, length(grid) - 1)

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
    # Parameterized tests for all 4 ND types
    # ========================================
    interp_configs = [
        ("CubicInterpolantND",    cubic_interp,     data_2d),
        ("LinearInterpolantND",   linear_interp,    data_2d),
        ("QuadraticInterpolantND", quadratic_interp, data_2d),
        ("ConstantInterpolantND", constant_interp,  const_2d),
    ]

    @testset "$name" for (name, interp_fn, data) in interp_configs
        itp = interp_fn((x, y), data)
        qx, qy = 1.0, 0.5

        @testset "hint=nothing matches no-hint" begin
            @test itp((qx, qy); hint=nothing) == itp((qx, qy))
        end

        @testset "Scalar hint updates Refs" begin
            hints = (Ref(1), Ref(1))
            result = itp((qx, qy); hint=hints)
            @test result ≈ itp((qx, qy))
            @test hints[1][] == expected_interval(x, qx)
            @test hints[2][] == expected_interval(y, qy)
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
    # Cubic-specific extras
    # ========================================
    @testset "CubicInterpolantND extras" begin
        itp = cubic_interp((x, y), data_2d)
        qx, qy = 1.0, 0.5

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
            @test hints[1][] == expected_interval(x, qx)
            @test hints[2][] == expected_interval(y, qy)
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
    # 3D Hint Support
    # ========================================
    @testset "3D hint support" begin
        itp = cubic_interp((x, y, z), data_3d)
        q = (1.0, 0.5, 1.5)

        hints = (Ref(1), Ref(1), Ref(1))
        result = itp(q; hint=hints)
        @test result ≈ itp(q)
        # Verify exact interval for all 3 axes
        @test hints[1][] == expected_interval(x, q[1])
        @test hints[2][] == expected_interval(y, q[2])
        @test hints[3][] == expected_interval(z, q[3])
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
        # Verify exact interval, proving HintedBinary was used
        @test hints[1][] == expected_interval(x, qx)
        @test hints[2][] == expected_interval(y, qy)
    end

    # ========================================
    # Heterogeneous Grid + Hint (Range × Vector)
    # ========================================
    @testset "Heterogeneous grid hint support" begin
        # Range (ScalarSpacing) × Vector (IrregularSpacing)
        # Both Range and Vector axes update hint Refs.
        x_range = range(0.0, 2.0, 21)
        y_vec   = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
        hdata_2d = [sin(xi) * cos(yj) for xi in x_range, yj in y_vec]

        hetero_configs = [
            ("CubicInterpolantND",     cubic_interp,     hdata_2d, (bc=CubicFit(),)),
            ("LinearInterpolantND",    linear_interp,    hdata_2d, NamedTuple()),
            ("QuadraticInterpolantND", quadratic_interp, hdata_2d, (bc=Right(QuadraticFit()),)),
            ("ConstantInterpolantND",  constant_interp,  hdata_2d, NamedTuple()),
        ]

        @testset "$name" for (name, interp_fn, data, kwargs) in hetero_configs
            itp = interp_fn((x_range, y_vec), data; kwargs...)
            qx, qy = 1.0, 0.5

            @testset "scalar with hint" begin
                hints = (Ref(1), Ref(1))
                ref = itp((qx, qy))
                @test itp((qx, qy); hint=hints) ≈ ref
                # Both axes update hints
                @test hints[1][] == expected_interval(x_range, qx)
                @test hints[2][] == expected_interval(y_vec, qy)
            end

            @testset "SoA batch with hint" begin
                xs = [0.5, 1.0, 1.5]
                ys = [0.2, 0.5, 0.8]
                hints = (Ref(1), Ref(1))
                @test itp((xs, ys); hint=hints) ≈ itp((xs, ys))
                # After batch, hints point to last query's intervals
                @test hints[1][] == expected_interval(x_range, xs[end])
                @test hints[2][] == expected_interval(y_vec, ys[end])
            end

            @testset "AoS batch with hint" begin
                queries = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
                hints = (Ref(1), Ref(1))
                @test itp(queries; hint=hints) ≈ itp(queries)
                @test hints[1][] == expected_interval(x_range, queries[end][1])
                @test hints[2][] == expected_interval(y_vec, queries[end][2])
            end
        end

        # 3D heterogeneous: Range × Vector × Range
        @testset "3D heterogeneous hint" begin
            z_range = range(0.0, 1.5, 9)
            hdata_3d = [sin(xi) * cos(yj) * exp(-zk/3)
                        for xi in x_range, yj in y_vec, zk in z_range]
            itp = cubic_interp((x_range, y_vec, z_range), hdata_3d; bc=CubicFit())
            q = (1.0, 0.5, 0.7)
            hints = (Ref(1), Ref(1), Ref(1))
            @test itp(q; hint=hints) ≈ itp(q)
            # All axes update hints (Range and Vector alike)
            @test hints[1][] == expected_interval(x_range, q[1])
            @test hints[2][] == expected_interval(y_vec, q[2])
            @test hints[3][] == expected_interval(z_range, q[3])
        end

        # Gradient/Hessian with heterogeneous grid + hint
        @testset "gradient with heterogeneous grid + hint" begin
            itp = cubic_interp((x_range, y_vec), hdata_2d; bc=CubicFit())
            q = (1.0, 0.5)
            hints = (Ref(1), Ref(1))
            ref = gradient(itp, q)
            @test gradient(itp, q; hint=hints) == ref
            @test hints[1][] == expected_interval(x_range, q[1])
            @test hints[2][] == expected_interval(y_vec, q[2])
        end
    end

    # ========================================
    # All-Range Grid Hint Support
    # ========================================
    @testset "All-Range grid hint support" begin
        x_range = range(0.0, 2.0, 21)
        y_range = range(0.0, 1.0, 11)
        rdata_2d = [sin(xi) * cos(yj) for xi in x_range, yj in y_range]

        range_configs = [
            ("CubicInterpolantND",     cubic_interp,     rdata_2d, (bc=CubicFit(),)),
            ("LinearInterpolantND",    linear_interp,    rdata_2d, NamedTuple()),
            ("QuadraticInterpolantND", quadratic_interp, rdata_2d, (bc=Right(QuadraticFit()),)),
            ("ConstantInterpolantND",  constant_interp,  rdata_2d, NamedTuple()),
        ]

        @testset "$name" for (name, interp_fn, data, kwargs) in range_configs
            itp = interp_fn((x_range, y_range), data; kwargs...)
            qx, qy = 1.0, 0.5

            @testset "scalar with hint" begin
                hints = (Ref(1), Ref(1))
                ref = itp((qx, qy))
                @test itp((qx, qy); hint=hints) ≈ ref
                @test hints[1][] == expected_interval(x_range, qx)
                @test hints[2][] == expected_interval(y_range, qy)
            end

            @testset "hint cache hit returns same result" begin
                hints = (Ref(1), Ref(1))
                itp((qx, qy); hint=hints)  # first call sets hint
                ix, iy = hints[1][], hints[2][]
                @test itp((qx, qy); hint=hints) ≈ itp((qx, qy))
                # hint stays the same (cache hit)
                @test hints[1][] == ix
                @test hints[2][] == iy
            end

            @testset "SoA batch with hint" begin
                xs = [0.5, 1.0, 1.5]
                ys = [0.2, 0.5, 0.8]
                hints = (Ref(1), Ref(1))
                @test itp((xs, ys); hint=hints) ≈ itp((xs, ys))
                @test hints[1][] == expected_interval(x_range, xs[end])
                @test hints[2][] == expected_interval(y_range, ys[end])
            end

            @testset "AoS batch with hint" begin
                queries = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
                hints = (Ref(1), Ref(1))
                @test itp(queries; hint=hints) ≈ itp(queries)
                @test hints[1][] == expected_interval(x_range, queries[end][1])
                @test hints[2][] == expected_interval(y_range, queries[end][2])
            end
        end
    end

    # ========================================
    # Monotonic Sequence (Hint Reuse)
    # ========================================
    @testset "Monotonic sequence benefits from hint" begin
        itp = cubic_interp((x, y), data_2d)
        hints = (Ref(1), Ref(1))

        # Monotonically increasing queries — hint should advance
        xs_mono = collect(range(0.1, 1.9, 10))
        for xi in xs_mono
            itp((xi, 0.5); hint=hints)
        end
        # After the full scan, hint should be at the interval of the last query
        @test hints[1][] == expected_interval(x, xs_mono[end])
        @test hints[2][] == expected_interval(y, 0.5)
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
        @test hints[1][] == expected_interval(x, 1.0)
        @test hints[2][] == expected_interval(y, 0.5)
    end

    # ========================================
    # Vector Calculus with Hint
    # ========================================
    @testset "gradient with hint" begin
        itp = cubic_interp((x, y), data_2d)
        q = (1.0, 0.5)
        hints = (Ref(1), Ref(1))

        ref = gradient(itp, q)
        @test gradient(itp, q; hint=hints) == ref
        @test hints[1][] == expected_interval(x, q[1])
        @test hints[2][] == expected_interval(y, q[2])

        # Vector API
        hints2 = (Ref(1), Ref(1))
        @test gradient(itp, [1.0, 0.5]; hint=hints2) ≈ collect(ref)
    end

    @testset "gradient! with hint" begin
        itp = cubic_interp((x, y), data_2d)
        q = (1.0, 0.5)
        hints = (Ref(1), Ref(1))

        G_ref = zeros(2)
        G_hint = zeros(2)
        gradient!(G_ref, itp, q)
        gradient!(G_hint, itp, q; hint=hints)
        @test G_hint ≈ G_ref
        @test hints[1][] == expected_interval(x, q[1])
        @test hints[2][] == expected_interval(y, q[2])
    end

    @testset "hessian with hint" begin
        itp = cubic_interp((x, y), data_2d)
        q = (1.0, 0.5)
        hints = (Ref(1), Ref(1))

        ref = hessian(itp, q)
        @test hessian(itp, q; hint=hints) ≈ ref
        @test hints[1][] == expected_interval(x, q[1])
        @test hints[2][] == expected_interval(y, q[2])

        # Vector API
        hints2 = (Ref(1), Ref(1))
        @test hessian(itp, [1.0, 0.5]; hint=hints2) ≈ ref
    end

    @testset "hessian! with hint" begin
        itp = cubic_interp((x, y), data_2d)
        q = (1.0, 0.5)
        hints = (Ref(1), Ref(1))

        H_ref = zeros(2, 2)
        H_hint = zeros(2, 2)
        hessian!(H_ref, itp, q)
        hessian!(H_hint, itp, q; hint=hints)
        @test H_hint ≈ H_ref
        @test hints[1][] == expected_interval(x, q[1])
        @test hints[2][] == expected_interval(y, q[2])
    end

    @testset "laplacian with hint" begin
        itp = cubic_interp((x, y), data_2d)
        q = (1.0, 0.5)
        hints = (Ref(1), Ref(1))

        ref = laplacian(itp, q)
        @test laplacian(itp, q; hint=hints) ≈ ref
        @test hints[1][] == expected_interval(x, q[1])
        @test hints[2][] == expected_interval(y, q[2])

        # Vector API
        hints2 = (Ref(1), Ref(1))
        @test laplacian(itp, [1.0, 0.5]; hint=hints2) ≈ ref
    end
end
