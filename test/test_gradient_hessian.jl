@testitem "Vector Calculus (gradient, hessian, laplacian)" setup = [AllocConstants] begin

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
        @test grad[1] ≈ expected_dx atol = 1.0e-3
        @test grad[2] ≈ expected_dy atol = 1.0e-3

        # Test vector API
        grad_vec = gradient(itp, [xq, yq])
        @test grad_vec isa Vector
        @test grad_vec[1] ≈ expected_dx atol = 1.0e-3
        @test grad_vec[2] ≈ expected_dy atol = 1.0e-3
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
        @test H[1, 1] ≈ expected_dxx atol = 1.0e-2
        @test H[2, 2] ≈ expected_dyy atol = 1.0e-2
        @test H[1, 2] ≈ expected_dxy atol = 1.0e-2
        @test H[2, 1] ≈ expected_dxy atol = 1.0e-2  # Symmetry

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
        @test grad[1] ≈ 2xq atol = 1.0e-3
        @test grad[2] ≈ 2yq atol = 1.0e-3
        @test grad[3] ≈ 2zq atol = 1.0e-3
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
        @test H[1, 1] ≈ 2.0 atol = 1.0e-2
        @test H[2, 2] ≈ 2.0 atol = 1.0e-2
        @test H[3, 3] ≈ 2.0 atol = 1.0e-2
        # Off-diagonal elements (should be zero for x² + y² + z²)
        @test abs(H[1, 2]) < 1.0e-2
        @test abs(H[1, 3]) < 1.0e-2
        @test abs(H[2, 3]) < 1.0e-2
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
        @test grad[1] ≈ cos(xq) atol = 1.0e-2
        @test grad[2] ≈ -im * sin(yq) atol = 1.0e-2
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

        @test lap ≈ 4.0 atol = 1.0e-2

        # Vector API
        lap_vec = laplacian(itp, [xq, yq])
        @test lap_vec ≈ 4.0 atol = 1.0e-2
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

        @test lap ≈ 6.0 atol = 1.0e-2
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
        @test lap ≈ H[1, 1] + H[2, 2] atol = 1.0e-10
    end

    # ========================================
    # Zero-allocation checks
    # ========================================

    @testset "Zero allocation - gradient!" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = cubic_interp((x, y), data)
        G = zeros(2)
        query = (0.5, 0.5)

        # Warmup
        gradient!(G, itp, query)

        # Correctness: in-place must match out-of-place
        G_ref = gradient(itp, query)
        @test G[1] ≈ G_ref[1]
        @test G[2] ≈ G_ref[2]

        allocs = @allocated gradient!(G, itp, query)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation - hessian!" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = cubic_interp((x, y), data)
        H = zeros(2, 2)
        query = (0.5, 0.5)

        # Warmup
        hessian!(H, itp, query)

        # Correctness: in-place must match out-of-place
        H_ref = hessian(itp, query)
        @test H ≈ H_ref

        allocs = @allocated hessian!(H, itp, query)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # Vector-grid tests (binary search path)
    # ========================================

    @testset "Vector grid - gradient matches range grid" begin
        x_range = range(0.0, 2π, 51)
        y_range = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x_range, yj in y_range]

        itp_range = cubic_interp((x_range, y_range), data)
        itp_vec = cubic_interp((collect(x_range), collect(y_range)), data)

        xq, yq = 1.7, 0.9
        grad_range = gradient(itp_range, (xq, yq))
        grad_vec = gradient(itp_vec, (xq, yq))

        @test grad_range[1] ≈ grad_vec[1] atol = 1.0e-12
        @test grad_range[2] ≈ grad_vec[2] atol = 1.0e-12
    end

    @testset "Vector grid - hessian matches range grid" begin
        x_range = range(0.0, 2π, 51)
        y_range = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x_range, yj in y_range]

        itp_range = cubic_interp((x_range, y_range), data)
        itp_vec = cubic_interp((collect(x_range), collect(y_range)), data)

        xq, yq = 1.7, 0.9
        H_range = hessian(itp_range, (xq, yq))
        H_vec = hessian(itp_vec, (xq, yq))

        @test H_range ≈ H_vec atol = 1.0e-12
    end

    # ========================================
    # In-place vector query dispatch
    # ========================================

    @testset "gradient! with vector query" begin
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)
        G = zeros(2)
        query = [1.7, 0.9]

        ret = gradient!(G, itp, query)
        G_ref = gradient(itp, (1.7, 0.9))

        @test G[1] ≈ G_ref[1]
        @test G[2] ≈ G_ref[2]
        @test ret === G  # returns the same buffer
    end

    @testset "hessian! with vector query" begin
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)
        H = zeros(2, 2)
        query = [1.7, 0.9]

        ret = hessian!(H, itp, query)
        H_ref = hessian(itp, (1.7, 0.9))

        @test H ≈ H_ref
        @test ret === H  # returns the same buffer
    end

    # ========================================
    # DimensionMismatch errors for in-place APIs
    # ========================================

    @testset "DimensionMismatch - in-place APIs" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [xi * yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        # gradient!: buffer too small
        G_small = zeros(1)
        @test_throws DimensionMismatch gradient!(G_small, itp, (0.5, 0.5))

        # gradient!: wrong query length (vector dispatch)
        G = zeros(2)
        @test_throws DimensionMismatch gradient!(G, itp, [0.5])
        @test_throws DimensionMismatch gradient!(G, itp, [0.5, 0.5, 0.5])

        # hessian!: wrong matrix size
        H_bad = zeros(3, 3)
        @test_throws DimensionMismatch hessian!(H_bad, itp, (0.5, 0.5))

        # hessian!: wrong query length (vector dispatch)
        H = zeros(2, 2)
        @test_throws DimensionMismatch hessian!(H, itp, [0.5])
        @test_throws DimensionMismatch hessian!(H, itp, [0.5, 0.5, 0.5])

        # laplacian: wrong query length (vector dispatch)
        @test_throws DimensionMismatch laplacian(itp, [0.5])
        @test_throws DimensionMismatch laplacian(itp, [0.5, 0.5, 0.5])
    end

    # ========================================
    # Non-cubic interpolant types
    # ========================================

    @testset "Linear interpolant - gradient" begin
        # f(x,y) = 3x + 2y → ∂f/∂x = 3, ∂f/∂y = 2 (exact for linear)
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [3xi + 2yj for xi in x, yj in y]
        itp = linear_interp((x, y), data)

        grad = gradient(itp, (0.5, 0.5))
        @test grad[1] ≈ 3.0 atol = 1.0e-10
        @test grad[2] ≈ 2.0 atol = 1.0e-10
    end

    @testset "Quadratic interpolant - gradient" begin
        # f(x,y) = x² + y² → ∂f/∂x = 2x, ∂f/∂y = 2y
        x = range(-1.0, 1.0, 21)
        y = range(-1.0, 1.0, 21)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = quadratic_interp((x, y), data)

        xq, yq = 0.3, -0.4
        grad = gradient(itp, (xq, yq))
        @test grad[1] ≈ 2xq atol = 1.0e-2
        @test grad[2] ≈ 2yq atol = 1.0e-2
    end

    @testset "Constant interpolant - gradient is zero" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [5.0 for xi in x, yj in y]
        itp = constant_interp((x, y), data)

        grad = gradient(itp, (0.5, 0.5))
        @test grad[1] ≈ 0.0 atol = 1.0e-15
        @test grad[2] ≈ 0.0 atol = 1.0e-15
    end

    @testset "Linear interpolant - hessian" begin
        # f(x,y) = 3x + 2y → all second derivatives = 0
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [3xi + 2yj for xi in x, yj in y]
        itp = linear_interp((x, y), data)

        H = hessian(itp, (0.5, 0.5))
        @test all(abs.(H) .< 1.0e-10)
    end

    # ========================================
    # VALUE_GRADIENT
    # ========================================

    @testset "2D value_gradient - cubic" begin
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        xq, yq = 1.7, 0.9
        val, grad = value_gradient(itp, (xq, yq))

        @test val ≈ itp((xq, yq)) atol = 1.0e-14
        @test grad isa NTuple{2}
        @test grad[1] ≈ gradient(itp, (xq, yq))[1] atol = 1.0e-14
        @test grad[2] ≈ gradient(itp, (xq, yq))[2] atol = 1.0e-14

        # Vector API
        val_v, grad_v = value_gradient(itp, [xq, yq])
        @test val_v ≈ val atol = 1.0e-14
        @test grad_v isa Vector
        @test grad_v ≈ collect(grad) atol = 1.0e-14
    end

    @testset "3D value_gradient - cubic" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        z = range(0.0, 1.0, 11)
        data = [xi^2 + yj^2 + zk^2 for xi in x, yj in y, zk in z]
        itp = cubic_interp((x, y, z), data)

        q = (0.3, 0.5, 0.7)
        val, grad = value_gradient(itp, q)

        @test val ≈ itp(q) atol = 1.0e-14
        g_ref = gradient(itp, q)
        @test grad[1] ≈ g_ref[1] atol = 1.0e-14
        @test grad[2] ≈ g_ref[2] atol = 1.0e-14
        @test grad[3] ≈ g_ref[3] atol = 1.0e-14
    end

    @testset "value_gradient - quadratic" begin
        x = range(-1.0, 1.0, 21)
        y = range(-1.0, 1.0, 21)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = quadratic_interp((x, y), data)

        q = (0.3, -0.4)
        val, grad = value_gradient(itp, q)
        @test val ≈ itp(q) atol = 1.0e-14
        @test collect(grad) ≈ collect(gradient(itp, q)) atol = 1.0e-14
    end

    @testset "value_gradient - linear" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [3xi + 2yj for xi in x, yj in y]
        itp = linear_interp((x, y), data)

        q = (0.5, 0.5)
        val, grad = value_gradient(itp, q)
        @test val ≈ itp(q) atol = 1.0e-14
        @test collect(grad) ≈ collect(gradient(itp, q)) atol = 1.0e-14
    end

    @testset "value_gradient - DimensionMismatch" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [xi + yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @test_throws DimensionMismatch value_gradient(itp, [0.5, 0.5, 0.5])
        @test_throws DimensionMismatch value_gradient(itp, [0.5])
    end

    @testset "value_gradient - complex data" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [(1.0 + 2.0im) * xi + (3.0 - 1.0im) * yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        q = (0.5, 0.5)
        val, grad = value_gradient(itp, q)
        @test val ≈ itp(q) atol = 1.0e-14
        @test collect(grad) ≈ collect(gradient(itp, q)) atol = 1.0e-14
    end

    @testset "value_gradient - OOB FillExtrap" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [xi + yj for xi in x, yj in y]

        # Test with non-zero fill value to catch value vs zero bugs
        fill_value = NaN
        itp = cubic_interp((x, y), data; extrap = FillExtrap(fill_value))

        val, grad = value_gradient(itp, (2.0, 0.5))
        @test isnan(val)
        @test all(g -> g == 0.0, grad)

        # Also test with finite non-zero fill value
        itp2 = cubic_interp((x, y), data; extrap = FillExtrap(-999.0))
        val2, grad2 = value_gradient(itp2, (2.0, 0.5))
        @test val2 == -999.0
        @test all(g -> g == 0.0, grad2)
    end

    @testset "Zero allocation - value_gradient tuple query" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = cubic_interp((x, y), data)
        query = (0.5, 0.5)

        function _test_vg_alloc(itp, q)
            value_gradient(itp, q)
            return nothing
        end
        _test_vg_alloc(itp, query)  # warmup
        allocs = @allocated _test_vg_alloc(itp, query)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Type stability - value_gradient" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        result = @inferred value_gradient(itp, (0.5, 0.5))
        @test result isa Tuple{Float64, NTuple{2, Float64}}
    end

    @testset "Vector grid - value_gradient matches range grid" begin
        x_range = range(0.0, 2π, 51)
        y_range = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x_range, yj in y_range]

        itp_range = cubic_interp((x_range, y_range), data)
        itp_vec = cubic_interp((collect(x_range), collect(y_range)), data)

        xq, yq = 1.7, 0.9
        val_r, grad_r = value_gradient(itp_range, (xq, yq))
        val_v, grad_v = value_gradient(itp_vec, (xq, yq))

        @test val_r ≈ val_v atol = 1.0e-12
        @test grad_r[1] ≈ grad_v[1] atol = 1.0e-12
        @test grad_r[2] ≈ grad_v[2] atol = 1.0e-12
    end

    @testset "PeriodicBC - value_gradient" begin
        # Exclusive periodic: data does NOT repeat endpoint
        nx, ny = 50, 50
        x = range(0.0, 2π, nx + 1)[1:nx]  # exclude endpoint
        y = range(0.0, 2π, ny + 1)[1:ny]
        data = [cos(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data; bc = PeriodicBC(endpoint = :exclusive, period = 2π))

        q = (1.5, 2.0)
        val, grad = value_gradient(itp, q)
        @test val ≈ itp(q) atol = 1.0e-14
        @test collect(grad) ≈ collect(gradient(itp, q)) atol = 1.0e-14
    end

    @testset "Quadratic interpolant - laplacian" begin
        # f(x,y) = x² + y² → ∇²f = 2 + 2 = 4
        x = range(-1.0, 1.0, 21)
        y = range(-1.0, 1.0, 21)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = quadratic_interp((x, y), data)

        lap = laplacian(itp, (0.3, -0.4))
        @test lap ≈ 4.0 atol = 1.0e-1
    end

    # ========================================
    # GridIdx support for non-HeteroInterpolantND interpolants
    # ========================================
    # GridIdx(k) on axis d → evaluate at grids[d][k], treat axis as discrete (deriv=0).
    # This works generically for CubicInterpolantND, LinearInterpolantND, etc.

    @testset "GridIdx gradient: CubicInterpolantND" begin
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        k = 15
        g = gradient(itp, (1.7, GridIdx(k)))
        g_ref = gradient(itp, (1.7, y[k]))
        # GridIdx is just index-based query — ALL derivatives are real
        @test g[1] ≈ g_ref[1] rtol = 1.0e-14
        @test g[2] ≈ g_ref[2] rtol = 1.0e-14
    end

    @testset "GridIdx gradient: LinearInterpolantND" begin
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = linear_interp((x, y), data)

        g = gradient(itp, (1.7, GridIdx(15)))
        g_ref = gradient(itp, (1.7, y[15]))
        @test g[1] ≈ g_ref[1] rtol = 1.0e-14
        @test g[2] ≈ g_ref[2] rtol = 1.0e-14
    end

    @testset "GridIdx hessian: CubicInterpolantND" begin
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        H = hessian(itp, (1.7, GridIdx(15)))
        H_ref = hessian(itp, (1.7, y[15]))
        @test size(H) == (2, 2)
        # Full hessian matches standard eval at grid point
        @test H ≈ H_ref rtol = 1.0e-14
    end

    @testset "GridIdx laplacian: CubicInterpolantND" begin
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        L = laplacian(itp, (1.7, GridIdx(15)))
        L_ref = laplacian(itp, (1.7, y[15]))
        # Full laplacian (all axes) matches standard
        @test L ≈ L_ref rtol = 1.0e-14
    end

    @testset "GridIdx OOB: CubicInterpolantND" begin
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @test_throws ArgumentError gradient(itp, (1.0, GridIdx(32)))   # y has 31 points
        @test_throws ArgumentError hessian(itp, (1.0, GridIdx(32)))
        @test_throws ArgumentError laplacian(itp, (1.0, GridIdx(32)))
    end

    @testset "GridIdx gradient: QuadraticInterpolantND" begin
        x = range(0.0, 2π, 51)
        y = range(0.0, π, 31)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = quadratic_interp((x, y), data)

        g = gradient(itp, (1.7, GridIdx(15)))
        g_ref = gradient(itp, (1.7, y[15]))
        @test g[1] ≈ g_ref[1] rtol = 1.0e-14
        @test g[2] ≈ g_ref[2] rtol = 1.0e-14
    end

    # ========================================
    # Phase 4: Cell-local windowed Hermite ND vector calculus
    # ========================================
    # The OnTheFly path now uses cell-local stencil windows when at least one axis
    # is a local-Hermite method. `_locate_cell` caches the windows in the cell tuple
    # so gradient/hessian/laplacian (which call `_eval_at_cell` N or N² times) only
    # pay the windowing cost once per query. These tests verify correctness against
    # ForwardDiff (which goes through the same forward path via Dual numbers, so it
    # exercises the same windowed kernel — but with a query type promotion).
    @testset "Hermite ND windowed gradient/hessian (Phase 4)" begin
        using ForwardDiff

        x = collect(range(0.0, 2π, 30))
        y = collect(range(-1.0, 1.0, 25))
        data = [sin(2xi) * exp(-yj^2) for xi in x, yj in y]

        # Sample queries away from boundaries (interior cells) where the Hermite ND
        # value should match an analytical reference within Hermite truncation error.
        sample_pts = [(1.0, 0.3), (3.5, -0.4), (5.5, 0.7), (2.0, 0.0)]

        for methods in (
                (PchipInterp(), PchipInterp()),
                (CardinalInterp(), CardinalInterp()),
                (AkimaInterp(), AkimaInterp()),
                # Mixed local × global — Cubic axis stays full, Cardinal is windowed
                (CardinalInterp(), CubicInterp()),
                (CubicInterp(), PchipInterp()),
            )
            itp = interp((x, y), data; method = methods, coeffs = OnTheFly())

            for q in sample_pts
                # gradient via vector_calculus = should equal ForwardDiff of the same itp
                g = gradient(itp, q)
                g_fd = ForwardDiff.gradient(p -> itp((p[1], p[2])), [q[1], q[2]])
                @test g[1] ≈ g_fd[1] rtol = 1.0e-10 atol = 1.0e-12
                @test g[2] ≈ g_fd[2] rtol = 1.0e-10 atol = 1.0e-12

                # value_gradient: value + gradient via the same locate-once cell tuple
                v, vg = value_gradient(itp, q)
                @test v == itp(q)
                @test vg[1] ≈ g_fd[1] rtol = 1.0e-10 atol = 1.0e-12
                @test vg[2] ≈ g_fd[2] rtol = 1.0e-10 atol = 1.0e-12
            end
        end

        # Hessian (N² eval_at_cell calls per query) — covers locate-once amortization
        @testset "Hessian: $(string(typeof(methods).name))" for methods in (
                (CardinalInterp(), CardinalInterp()),
                (PchipInterp(), CubicInterp()),
            )
            itp = interp((x, y), data; method = methods, coeffs = OnTheFly())
            for q in sample_pts
                H = hessian(itp, q)
                H_fd = ForwardDiff.hessian(p -> itp((p[1], p[2])), [q[1], q[2]])
                @test H[1, 1] ≈ H_fd[1, 1] rtol = 1.0e-9 atol = 1.0e-11
                @test H[1, 2] ≈ H_fd[1, 2] rtol = 1.0e-9 atol = 1.0e-11
                @test H[2, 1] ≈ H_fd[2, 1] rtol = 1.0e-9 atol = 1.0e-11
                @test H[2, 2] ≈ H_fd[2, 2] rtol = 1.0e-9 atol = 1.0e-11
            end
        end

        # Laplacian = trace of hessian
        let methods = (CardinalInterp(), CardinalInterp())
            itp = interp((x, y), data; method = methods, coeffs = OnTheFly())
            for q in sample_pts
                lap = laplacian(itp, q)
                H = hessian(itp, q)
                @test lap ≈ (H[1, 1] + H[2, 2]) rtol = 1.0e-12
            end
        end
    end

end
