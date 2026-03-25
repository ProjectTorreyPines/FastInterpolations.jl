using Test
using FastInterpolations

@testset "nodal_partials" begin
    # ========================================
    # Test Data Setup
    # ========================================
    x = range(0.0, 1.0; length = 11)
    y = range(0.0, 1.0; length = 9)
    z = range(0.0, 1.0; length = 7)
    data_2d = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
    data_3d = [sin(2π * xi) * cos(2π * yj) * exp(zk) for xi in x, yj in y, zk in z]

    # ========================================
    # CubicInterpolantND — 2D
    # ========================================
    @testset "CubicInterpolantND 2D" begin
        itp = interp((x, y), data_2d; method = CubicInterp())

        # All 4 derivative combinations match internal storage
        for (order, p) in [((0, 0), 1), ((1, 0), 2), ((0, 1), 3), ((1, 1), 4)]
            v = nodal_partials(itp, order)
            @test v == itp.nodal_derivs.partials[p, :, :]
            @test size(v) == (length(x), length(y))
        end

        # Zero-copy: view shares memory with internal storage
        v = nodal_partials(itp, (0, 0))
        @test pointer(v) == pointer(itp.nodal_derivs.partials)

        # Type stability
        @test @inferred(nodal_partials(itp, (0, 0))) isa AbstractArray{Float64, 2}
        @test @inferred(nodal_partials(itp, (1, 1))) isa AbstractArray{Float64, 2}
    end

    # ========================================
    # CubicInterpolantND — 3D
    # ========================================
    @testset "CubicInterpolantND 3D" begin
        itp = interp((x, y, z), data_3d; method = CubicInterp())

        # All 8 derivative combinations
        for b1 in 0:1, b2 in 0:1, b3 in 0:1
            order = (b1, b2, b3)
            p = 1 + b1 + 2 * b2 + 4 * b3
            v = nodal_partials(itp, order)
            @test v == itp.nodal_derivs.partials[p, :, :, :]
            @test size(v) == (length(x), length(y), length(z))
        end

        # Type stability
        @test @inferred(nodal_partials(itp, (0, 0, 0))) isa AbstractArray{Float64, 3}
    end

    # ========================================
    # QuadraticInterpolantND — 2D
    # ========================================
    @testset "QuadraticInterpolantND 2D" begin
        itp = interp((x, y), data_2d; method = QuadraticInterp())

        for (order, p) in [((0, 0), 1), ((1, 0), 2), ((0, 1), 3), ((1, 1), 4)]
            v = nodal_partials(itp, order)
            @test v == itp.nodal_derivs.partials[p, :, :]
            @test size(v) == (length(x), length(y))
        end

        # Type stability
        @test @inferred(nodal_partials(itp, (0, 0))) isa AbstractArray{Float64, 2}
    end

    # ========================================
    # HeteroInterpolantND — PreCompute
    # ========================================
    @testset "HeteroInterpolantND PreCompute (Cubic × Linear)" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()),
            coeffs = PreCompute(),
        )

        # (0, 0) — function values: always valid
        v = nodal_partials(itp, (0, 0))
        @test size(v) == (length(x), length(y))
        @test v == itp.data.partials[1, :, :]

        # (1, 0) — cubic axis derivative: valid
        v = nodal_partials(itp, (1, 0))
        @test size(v) == (length(x), length(y))
        # sizes = (2, 1), so p = 1 + 1*1 = 2
        @test v == itp.data.partials[2, :, :]

        # (0, 1) — linear axis derivative: NOT stored → error
        @test_throws ArgumentError nodal_partials(itp, (0, 1))

        # (1, 1) — mixed with linear axis: NOT stored → error
        @test_throws ArgumentError nodal_partials(itp, (1, 1))

        # Type stability for valid calls
        @test @inferred(nodal_partials(itp, (0, 0))) isa AbstractArray{Float64, 2}
        @test @inferred(nodal_partials(itp, (1, 0))) isa AbstractArray{Float64, 2}
    end

    @testset "HeteroInterpolantND PreCompute (Cubic × Linear) — value correctness" begin
        # Use polynomial f(x,y) = x^3 + 2xy — cubic spline reproduces cubics exactly
        xp = range(0.0, 1.0; length = 11)
        yp = range(0.0, 2.0; length = 9)
        poly_2d = [xi^3 + 2 * xi * yj for xi in xp, yj in yp]

        itp_hetero = interp(
            (xp, yp), poly_2d;
            method = (CubicInterp(), LinearInterp()),
            coeffs = PreCompute(),
        )

        # (0, 0) — function values must match original data exactly
        f_vals = nodal_partials(itp_hetero, (0, 0))
        @test f_vals ≈ poly_2d

        # (1, 0) — ∂f/∂x = 3x² + 2y at grid nodes
        # Compare with a pure cubic 2D interp on the same data
        itp_cubic = interp((xp, yp), poly_2d; method = CubicInterp())
        dfdx_hetero = nodal_partials(itp_hetero, (1, 0))
        dfdx_cubic = nodal_partials(itp_cubic, (1, 0))
        @test dfdx_hetero ≈ dfdx_cubic

        # Also verify against analytic: ∂f/∂x = 3x² + 2y
        # Cubic spline reproduces cubic polynomials exactly with CubicFit BC
        dfdx_analytic = [3 * xi^2 + 2 * yj for xi in xp, yj in yp]
        @test dfdx_hetero ≈ dfdx_analytic atol = 1e-10
    end

    @testset "HeteroInterpolantND PreCompute (Cubic × Quadratic) — value correctness" begin
        # f(x,y) = x^3 + y^2 — cubic exact on x, quadratic exact on y
        xp = range(0.0, 1.0; length = 11)
        yp = range(0.0, 1.0; length = 9)
        poly_2d = [xi^3 + yj^2 for xi in xp, yj in yp]

        itp = interp(
            (xp, yp), poly_2d;
            method = (CubicInterp(), QuadraticInterp()),
            coeffs = PreCompute(),
        )

        # (0, 0) — function values
        @test nodal_partials(itp, (0, 0)) ≈ poly_2d

        # (1, 0) — ∂f/∂x = 3x² at nodes (independent of y)
        dfdx = nodal_partials(itp, (1, 0))
        dfdx_analytic = [3 * xi^2 for xi in xp, yj in yp]
        @test dfdx ≈ dfdx_analytic atol = 1e-10

        # (0, 1) — ∂f/∂y = 2y at nodes (independent of x)
        dfdy = nodal_partials(itp, (0, 1))
        dfdy_analytic = [2 * yj for xi in xp, yj in yp]
        @test dfdy ≈ dfdy_analytic atol = 1e-10

        # (1, 1) — ∂²f/∂x∂y = 0 for f = x³ + y²
        dfdxy = nodal_partials(itp, (1, 1))
        @test all(abs.(dfdxy) .< 1e-10)
    end

    @testset "HeteroInterpolantND PreCompute (Cubic × Quadratic × Linear) 3D" begin
        itp = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), QuadraticInterp(), LinearInterp()),
            coeffs = PreCompute(),
        )
        # sizes = (2, 2, 1) → 4 partials per grid point

        # Valid: (0,0,0), (1,0,0), (0,1,0), (1,1,0)
        @test size(nodal_partials(itp, (0, 0, 0))) == (length(x), length(y), length(z))
        @test size(nodal_partials(itp, (1, 0, 0))) == (length(x), length(y), length(z))
        @test size(nodal_partials(itp, (0, 1, 0))) == (length(x), length(y), length(z))
        @test size(nodal_partials(itp, (1, 1, 0))) == (length(x), length(y), length(z))

        # Invalid: any combo with z-derivative (Linear axis)
        @test_throws ArgumentError nodal_partials(itp, (0, 0, 1))
        @test_throws ArgumentError nodal_partials(itp, (1, 0, 1))
        @test_throws ArgumentError nodal_partials(itp, (0, 1, 1))
        @test_throws ArgumentError nodal_partials(itp, (1, 1, 1))
    end

    @testset "HeteroInterpolantND PreCompute 3D — value correctness" begin
        # f(x,y,z) = x^3 + y^2 + z  (cubic on x, quadratic on y, linear on z)
        xp = range(0.0, 1.0; length = 11)
        yp = range(0.0, 1.0; length = 9)
        zp = range(0.0, 1.0; length = 7)
        poly_3d = [xi^3 + yj^2 + zk for xi in xp, yj in yp, zk in zp]

        itp = interp(
            (xp, yp, zp), poly_3d;
            method = (CubicInterp(), QuadraticInterp(), LinearInterp()),
            coeffs = PreCompute(),
        )

        # (0,0,0) — function values
        @test nodal_partials(itp, (0, 0, 0)) ≈ poly_3d

        # (1,0,0) — ∂f/∂x = 3x²
        dfdx = nodal_partials(itp, (1, 0, 0))
        dfdx_analytic = [3 * xi^2 for xi in xp, yj in yp, zk in zp]
        @test dfdx ≈ dfdx_analytic atol = 1e-10

        # (0,1,0) — ∂f/∂y = 2y
        dfdy = nodal_partials(itp, (0, 1, 0))
        dfdy_analytic = [2 * yj for xi in xp, yj in yp, zk in zp]
        @test dfdy ≈ dfdy_analytic atol = 1e-10

        # (1,1,0) — ∂²f/∂x∂y = 0
        dfdxy = nodal_partials(itp, (1, 1, 0))
        @test all(abs.(dfdxy) .< 1e-10)
    end

    # ========================================
    # Error Cases
    # ========================================
    @testset "Error cases" begin
        itp_cubic_2d = interp((x, y), data_2d; method = CubicInterp())

        # DimensionMismatch: wrong tuple length
        @test_throws DimensionMismatch nodal_partials(itp_cubic_2d, (1, 0, 0))
        @test_throws DimensionMismatch nodal_partials(itp_cubic_2d, (1,))

        # ArgumentError: invalid order values
        @test_throws ArgumentError nodal_partials(itp_cubic_2d, (2, 0))
        @test_throws ArgumentError nodal_partials(itp_cubic_2d, (-1, 0))

        # LinearInterpolantND: no stored partials
        itp_linear = interp((x, y), data_2d; method = LinearInterp())
        @test_throws ArgumentError nodal_partials(itp_linear, (0, 0))

        # ConstantInterpolantND: no stored partials
        itp_const = interp((x, y), data_2d; method = ConstantInterp())
        @test_throws ArgumentError nodal_partials(itp_const, (0, 0))

        # HeteroInterpolantND OnTheFly: no precomputed partials
        itp_onthefly = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()),
            coeffs = OnTheFly(),
        )
        @test_throws ArgumentError nodal_partials(itp_onthefly, (0, 0))
    end

    # ========================================
    # Error Message Content
    # ========================================
    @testset "Error message content" begin
        # Hetero: axis-specific error message
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()),
            coeffs = PreCompute(),
        )
        err = try
            nodal_partials(itp, (0, 1))
        catch e
            e
        end
        @test err isa ArgumentError
        msg = err.msg
        @test occursin("axis 2", msg)
        @test occursin("LinearInterp", msg)
        @test occursin("does not store", msg)

        # DimensionMismatch message
        itp_2d = interp((x, y), data_2d; method = CubicInterp())
        err2 = try
            nodal_partials(itp_2d, (1, 0, 0))
        catch e
            e
        end
        @test err2 isa DimensionMismatch
        @test occursin("3", err2.msg) && occursin("2", err2.msg)
    end
end
