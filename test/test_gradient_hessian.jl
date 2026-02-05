using Test
using FastInterpolations

@testset "Vector Calculus (gradient, hessian, laplacian)" begin

    @testset "2D Gradient" begin
        # Test function: f(x,y) = sin(x) * cos(y)
        # ∂f/∂x = cos(x) * cos(y)
        # ∂f/∂y = -sin(x) * sin(y)
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        # Test at a point away from grid
        xq, yq = 1.7, 0.9
        grad = gradient(itp, (xq, yq))

        expected_dx = cos(xq) * cos(yq)
        expected_dy = -sin(xq) * sin(yq)

        @test grad isa NTuple{2}
        @test grad[1] ≈ expected_dx atol=1e-3
        @test grad[2] ≈ expected_dy atol=1e-3

        # Test vector API
        grad_vec = gradient(itp, [xq, yq])
        @test grad_vec isa Vector
        @test grad_vec[1] ≈ expected_dx atol=1e-3
        @test grad_vec[2] ≈ expected_dy atol=1e-3
    end

    @testset "2D Hessian" begin
        # Test function: f(x,y) = sin(x) * cos(y)
        # ∂²f/∂x² = -sin(x) * cos(y)
        # ∂²f/∂y² = -sin(x) * cos(y)
        # ∂²f/∂x∂y = -cos(x) * sin(y)
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        xq, yq = 1.7, 0.9
        H = hessian(itp, (xq, yq))

        expected_dxx = -sin(xq) * cos(yq)
        expected_dyy = -sin(xq) * cos(yq)
        expected_dxy = -cos(xq) * sin(yq)

        @test H isa Matrix
        @test size(H) == (2, 2)
        @test H[1, 1] ≈ expected_dxx atol=1e-2
        @test H[2, 2] ≈ expected_dyy atol=1e-2
        @test H[1, 2] ≈ expected_dxy atol=1e-2
        @test H[2, 1] ≈ expected_dxy atol=1e-2  # Symmetry

        # Test vector API
        H_vec = hessian(itp, [xq, yq])
        @test H_vec ≈ H
    end

    @testset "3D Gradient" begin
        # Test function: f(x,y,z) = x² + y² + z²
        # ∂f/∂x = 2x, ∂f/∂y = 2y, ∂f/∂z = 2z
        x = range(-1.0, 1.0, 21)
        y = range(-1.0, 1.0, 21)
        z = range(-1.0, 1.0, 21)
        data = [xi^2 + yj^2 + zk^2 for xi in x, yj in y, zk in z]
        itp = cubic_interp((x, y, z), data)

        xq, yq, zq = 0.3, -0.4, 0.5
        grad = gradient(itp, (xq, yq, zq))

        @test grad isa NTuple{3}
        @test grad[1] ≈ 2xq atol=1e-3
        @test grad[2] ≈ 2yq atol=1e-3
        @test grad[3] ≈ 2zq atol=1e-3
    end

    @testset "3D Hessian" begin
        # Test function: f(x,y,z) = x² + y² + z²
        # Hessian = diag(2, 2, 2), off-diagonal = 0
        x = range(-1.0, 1.0, 21)
        y = range(-1.0, 1.0, 21)
        z = range(-1.0, 1.0, 21)
        data = [xi^2 + yj^2 + zk^2 for xi in x, yj in y, zk in z]
        itp = cubic_interp((x, y, z), data)

        xq, yq, zq = 0.3, -0.4, 0.5
        H = hessian(itp, (xq, yq, zq))

        @test size(H) == (3, 3)
        # Diagonal elements
        @test H[1, 1] ≈ 2.0 atol=1e-2
        @test H[2, 2] ≈ 2.0 atol=1e-2
        @test H[3, 3] ≈ 2.0 atol=1e-2
        # Off-diagonal elements (should be zero for x² + y² + z²)
        @test abs(H[1, 2]) < 1e-2
        @test abs(H[1, 3]) < 1e-2
        @test abs(H[2, 3]) < 1e-2
        # Symmetry
        @test H[1, 2] ≈ H[2, 1]
        @test H[1, 3] ≈ H[3, 1]
        @test H[2, 3] ≈ H[3, 2]
    end

    @testset "Dimension Mismatch Error" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [xi * yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        # Wrong number of elements in vector
        @test_throws DimensionMismatch gradient(itp, [0.5])
        @test_throws DimensionMismatch gradient(itp, [0.5, 0.5, 0.5])
        @test_throws DimensionMismatch hessian(itp, [0.5])
        @test_throws DimensionMismatch hessian(itp, [0.5, 0.5, 0.5])
    end

    @testset "Complex-valued Data" begin
        # Complex data: f(x,y) = sin(x) + i*cos(y)
        x = range(0.0, 2π, 31)
        y = range(0.0, π, 21)
        data = [sin(xi) + im * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        xq, yq = 1.5, 0.8
        grad = gradient(itp, (xq, yq))

        # ∂f/∂x = cos(x), ∂f/∂y = -i*sin(y)
        @test grad[1] ≈ cos(xq) atol=1e-2
        @test grad[2] ≈ -im * sin(yq) atol=1e-2
    end

    @testset "2D Laplacian" begin
        # Test function: f(x,y) = x² + y²
        # ∇²f = ∂²f/∂x² + ∂²f/∂y² = 2 + 2 = 4
        x = range(-1.0, 1.0, 21)
        y = range(-1.0, 1.0, 21)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        xq, yq = 0.3, -0.4
        lap = laplacian(itp, (xq, yq))

        @test lap ≈ 4.0 atol=1e-2

        # Vector API
        lap_vec = laplacian(itp, [xq, yq])
        @test lap_vec ≈ 4.0 atol=1e-2
    end

    @testset "3D Laplacian" begin
        # Test function: f(x,y,z) = x² + y² + z²
        # ∇²f = 2 + 2 + 2 = 6
        x = range(-1.0, 1.0, 15)
        y = range(-1.0, 1.0, 15)
        z = range(-1.0, 1.0, 15)
        data = [xi^2 + yj^2 + zk^2 for xi in x, yj in y, zk in z]
        itp = cubic_interp((x, y, z), data)

        xq, yq, zq = 0.3, -0.4, 0.5
        lap = laplacian(itp, (xq, yq, zq))

        @test lap ≈ 6.0 atol=1e-2
    end

    @testset "Laplacian vs Hessian trace" begin
        # Verify laplacian == tr(hessian)
        x = range(0.0, 2π, 31)
        y = range(0.0, π, 21)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        xq, yq = 1.5, 0.8
        lap = laplacian(itp, (xq, yq))
        H = hessian(itp, (xq, yq))

        # trace = sum of diagonal elements
        @test lap ≈ H[1,1] + H[2,2] atol=1e-10
    end

end
