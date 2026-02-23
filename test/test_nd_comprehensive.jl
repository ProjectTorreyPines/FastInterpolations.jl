# ========================================
# Comprehensive ND Interpolation Tests
# ========================================
#
# Coverage-focused tests for:
# - nd_types.jl: Type constructors, accessors, helpers
# - nd_eval.jl: Batch evaluation (SoA, AoS), vector API
# - nd_build.jl: Periodic data validation, differentiation
# - nd_utils.jl: Resolution helpers, error paths
# - nd_math.jl: Third derivatives, BC applications
# - nd_api.jl: Oneshot API, grid conversion
# - vector_calculus.jl: gradient, hessian, laplacian
# - show.jl: ND display methods
# - derivative_view.jl: ND-specific deriv_view

using Test
using FastInterpolations

@testset "ND Comprehensive Tests" begin

    # ========================================
    # VECTOR CALCULUS (gradient, hessian, laplacian)
    # ========================================
    @testset "Vector Calculus Operations" begin
        # Test function: f(x,y) = x²y + y³ → known analytical derivatives
        # ∂f/∂x = 2xy
        # ∂f/∂y = x² + 3y²
        # ∂²f/∂x² = 2y
        # ∂²f/∂y² = 6y
        # ∂²f/∂x∂y = 2x
        # ∇²f = 2y + 6y = 8y

        x = range(0.0, 2.0, 21)
        y = range(0.0, 1.0, 11)
        data = [xi^2 * yj + yj^3 for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @testset "gradient - tuple API" begin
            xq, yq = 1.0, 0.5
            g = gradient(itp, (xq, yq))

            @test g isa Tuple{Float64, Float64}
            @test length(g) == 2

            # ∂f/∂x = 2xy = 2 * 1.0 * 0.5 = 1.0
            @test g[1] ≈ 2 * xq * yq atol=1e-3
            # ∂f/∂y = x² + 3y² = 1.0 + 0.75 = 1.75
            @test g[2] ≈ xq^2 + 3*yq^2 atol=1e-3
        end

        @testset "gradient - vector API" begin
            xq, yq = 1.0, 0.5
            g = gradient(itp, [xq, yq])

            @test g isa Vector{Float64}
            @test length(g) == 2
            @test g[1] ≈ 2 * xq * yq atol=1e-3
            @test g[2] ≈ xq^2 + 3*yq^2 atol=1e-3

            # Wrong dimension should throw
            @test_throws DimensionMismatch gradient(itp, [1.0])
            @test_throws DimensionMismatch gradient(itp, [1.0, 0.5, 0.3])
        end

        @testset "hessian - tuple API" begin
            xq, yq = 1.0, 0.5
            H = hessian(itp, (xq, yq))

            @test H isa Matrix{Float64}
            @test size(H) == (2, 2)

            # ∂²f/∂x² = 2y = 1.0
            @test H[1, 1] ≈ 2 * yq atol=1e-2
            # ∂²f/∂y² = 6y = 3.0
            @test H[2, 2] ≈ 6 * yq atol=1e-2
            # ∂²f/∂x∂y = 2x = 2.0
            @test H[1, 2] ≈ 2 * xq atol=1e-2
            # Symmetric
            @test H[1, 2] ≈ H[2, 1] atol=1e-12
        end

        @testset "hessian - vector API" begin
            xq, yq = 1.0, 0.5
            H = hessian(itp, [xq, yq])

            @test H isa Matrix{Float64}
            @test size(H) == (2, 2)
            @test H[1, 1] ≈ 2 * yq atol=1e-2

            # Wrong dimension should throw
            @test_throws DimensionMismatch hessian(itp, [1.0])
        end

        @testset "laplacian - tuple API" begin
            xq, yq = 1.0, 0.5
            lap = laplacian(itp, (xq, yq))

            @test lap isa Float64
            # ∇²f = ∂²f/∂x² + ∂²f/∂y² = 2y + 6y = 8y
            @test lap ≈ 8 * yq atol=1e-2

            # Should equal trace of Hessian
            H = hessian(itp, (xq, yq))
            @test lap ≈ H[1,1] + H[2,2] atol=1e-10
        end

        @testset "laplacian - vector API" begin
            xq, yq = 1.0, 0.5
            lap = laplacian(itp, [xq, yq])

            @test lap isa Float64
            @test lap ≈ 8 * yq atol=1e-2

            # Wrong dimension should throw
            @test_throws DimensionMismatch laplacian(itp, [1.0])
        end

        @testset "3D vector calculus" begin
            # f(x,y,z) = x²y + yz² → more complex test
            x = range(0.0, 2.0, 11)
            y = range(0.0, 1.0, 6)
            z = range(0.0, 1.5, 8)
            data = [xi^2 * yj + yj * zk^2 for xi in x, yj in y, zk in z]
            itp3d = cubic_interp((x, y, z), data)

            xq, yq, zq = 1.0, 0.5, 0.75

            # gradient
            g = gradient(itp3d, (xq, yq, zq))
            @test length(g) == 3
            # ∂f/∂x = 2xy
            @test g[1] ≈ 2 * xq * yq atol=1e-2
            # ∂f/∂y = x² + z²
            @test g[2] ≈ xq^2 + zq^2 atol=1e-2
            # ∂f/∂z = 2yz
            @test g[3] ≈ 2 * yq * zq atol=1e-2

            # hessian
            H = hessian(itp3d, (xq, yq, zq))
            @test size(H) == (3, 3)
            # Symmetric
            @test H[1, 2] ≈ H[2, 1] atol=1e-10
            @test H[1, 3] ≈ H[3, 1] atol=1e-10
            @test H[2, 3] ≈ H[3, 2] atol=1e-10

            # laplacian
            lap = laplacian(itp3d, (xq, yq, zq))
            @test lap isa Float64
            # Laplacian should equal H[1,1] + H[2,2] + H[3,3] (trace)
            @test lap ≈ H[1,1] + H[2,2] + H[3,3] atol=1e-10
        end
    end

    # ========================================
    # BATCH EVALUATION (SoA, AoS)
    # ========================================
    @testset "Batch Evaluation APIs" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @testset "SoA (Tuple of Vectors) Batch" begin
            xqs = collect(range(0.5, 2.0, 10))
            yqs = collect(range(0.2, 1.5, 10))

            vals = itp((xqs, yqs))

            @test vals isa Vector{Float64}
            @test length(vals) == 10

            # Verify results match individual evaluations
            for k in 1:10
                expected = itp((xqs[k], yqs[k]))
                @test vals[k] ≈ expected atol=1e-12
            end

            # Wrong lengths should error
            @test_throws DimensionMismatch itp((xqs, yqs[1:5]))
        end

        @testset "SoA with derivatives" begin
            xqs = collect(range(0.5, 2.0, 5))
            yqs = collect(range(0.2, 1.5, 5))

            # First derivative ∂f/∂x
            vals_dx = itp((xqs, yqs); deriv=DerivOp(1, 0))
            @test length(vals_dx) == 5

            # DerivOp derivative spec
            vals_dx_val = itp((xqs, yqs); deriv=DerivOp(1, 0))
            @test vals_dx ≈ vals_dx_val

            # Integer derivative (all axes same order)
            vals_d1 = itp((xqs, yqs); deriv=DerivOp(1, 1))
            @test length(vals_d1) == 5
        end

        @testset "AoS (Vector of Tuples) Batch" begin
            points = [(0.5, 0.3), (1.0, 0.5), (1.5, 0.8), (2.0, 1.0)]

            vals = itp(points)

            @test vals isa Vector{Float64}
            @test length(vals) == 4

            # Verify results
            for (k, pt) in enumerate(points)
                @test vals[k] ≈ itp(pt) atol=1e-12
            end
        end

        @testset "AoS with derivatives" begin
            points = [(0.5, 0.3), (1.0, 0.5), (1.5, 0.8)]

            vals_dx = itp(points; deriv=DerivOp(1, 0))
            @test length(vals_dx) == 3

            vals_dx_val = itp(points; deriv=DerivOp(1, 0))
            @test vals_dx ≈ vals_dx_val
        end

        @testset "Vector Input API" begin
            # Vector input (for ForwardDiff compatibility)
            result = itp([1.0, 0.5])
            @test result isa Float64
            @test result ≈ itp((1.0, 0.5))

            # Wrong dimension
            @test_throws DimensionMismatch itp([1.0])
            @test_throws DimensionMismatch itp([1.0, 0.5, 0.3])
        end
    end

    # ========================================
    # EXTRAPOLATION MODES
    # ========================================
    @testset "Extrapolation Modes" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi + yj for xi in x, yj in y]  # Simple linear function

        @testset "ConstExtrap() extrapolation" begin
            itp = cubic_interp((x, y), data; extrap=ConstExtrap())

            # Left boundary clamp
            @test itp((-0.5, 0.5)) ≈ itp((0.0, 0.5))
            # Right boundary clamp
            @test itp((2.5, 0.5)) ≈ itp((2.0, 0.5))
            # Top boundary clamp
            @test itp((1.0, 1.5)) ≈ itp((1.0, 1.0))
            # Bottom boundary clamp
            @test itp((1.0, -0.5)) ≈ itp((1.0, 0.0))
            # Corner clamp
            @test itp((-0.5, -0.5)) ≈ itp((0.0, 0.0))
        end

        @testset "Per-axis extrapolation" begin
            # :constant on x, :none on y
            itp = cubic_interp((x, y), data; extrap=(ConstExtrap(), NoExtrap()))

            # x outside domain should clamp
            @test itp((-0.5, 0.5)) ≈ itp((0.0, 0.5))
            # y outside domain should throw
            @test_throws DomainError itp((1.0, 1.5))
        end

        @testset "WrapExtrap() extrapolation (with PeriodicBC)" begin
            # Periodic data that wraps correctly
            x = range(0.0, 2π, 21)  # First and last are same modulo 2π
            y = range(0.0, 2π, 21)

            # sin and cos are periodic
            data = [sin(xi) * cos(yj) for xi in x, yj in y]

            itp = cubic_interp((x, y), data; bc=PeriodicBC(), extrap=WrapExtrap())

            # Should wrap around
            xq = 2π + 0.5  # Past domain
            result_wrapped = itp((xq, 0.5))
            result_base = itp((0.5, 0.5))  # Same as wrapped position
            @test result_wrapped ≈ result_base atol=1e-2
        end
    end

    # ========================================
    # BOUNDARY CONDITIONS (EDGE CASES)
    # ========================================
    @testset "Boundary Condition Edge Cases" begin
        x = range(0.0, 2.0, 21)
        y = range(0.0, 1.0, 11)
        data = [xi^2 * yj for xi in x, yj in y]

        @testset "NTuple BC resolution" begin
            # Per-axis BCs
            itp = cubic_interp((x, y), data; bc=(ZeroCurvBC(), ZeroSlopeBC()))
            @test itp((1.0, 0.5)) isa Float64

            # Per-axis with different PolyFit
            itp2 = cubic_interp((x, y), data; bc=(LinearFit(), CubicFit()))
            @test itp2((1.0, 0.5)) isa Float64
        end

        @testset "BC resolution error paths" begin
            # Wrong number of BCs - rejected by keyword type assertion
            @test_throws TypeError cubic_interp((x, y), data; bc=(ZeroCurvBC(),))
            @test_throws TypeError cubic_interp((x, y), data; bc=(ZeroCurvBC(), ZeroCurvBC(), ZeroCurvBC()))
        end

        @testset "Extrap resolution error paths" begin
            # Wrong number of extrap modes - rejected by keyword type assertion
            @test_throws TypeError cubic_interp((x, y), data; extrap=(NoExtrap(),))
            @test_throws TypeError cubic_interp((x, y), data; extrap=(NoExtrap(), NoExtrap(), NoExtrap()))
        end

        @testset "Search resolution error paths" begin
            # Wrong number of search policies - rejected by keyword type assertion
            @test_throws TypeError cubic_interp((x, y), data; search=(Binary(),))
            @test_throws TypeError cubic_interp((x, y), data; search=(Binary(), Binary(), Binary()))
        end
    end

    # ========================================
    # THIRD DERIVATIVE
    # ========================================
    @testset "Third Derivative Evaluation" begin
        x = range(0.0, 2.0, 21)
        y = range(0.0, 1.0, 11)
        # f(x,y) = x³y + xy³ → ∂³f/∂x³ = 6y, ∂³f/∂y³ = 6x
        data = [xi^3 * yj + xi * yj^3 for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        xq, yq = 1.0, 0.5

        # ∂³f/∂x³ = 6y
        d3x = itp((xq, yq); deriv=DerivOp(3, 0))
        @test d3x ≈ 6 * yq atol=0.5  # Third derivative has lower accuracy

        # ∂³f/∂y³ = 6x
        d3y = itp((xq, yq); deriv=DerivOp(0, 3))
        @test d3y ≈ 6 * xq atol=0.5

        # Using Val
        d3x_val = itp((xq, yq); deriv=DerivOp(3, 0))
        @test d3x_val ≈ d3x atol=1e-12

        # All axes deriv=DerivOp(3) via Int
        d3_all = itp((xq, yq); deriv=DerivOp(3, 3))
        @test d3_all isa Float64
    end

    # ========================================
    # DERIVATIVE VIEW (ND SPECIFIC)
    # ========================================
    @testset "DerivativeView for ND" begin
        x = range(0.0, 2.0, 21)
        y = range(0.0, 1.0, 11)
        data = [xi^2 * yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @testset "deriv_view with tuple order" begin
            # ∂f/∂x
            dx = deriv_view(itp, (1, 0))
            @test dx isa FastInterpolations.DerivativeView

            result = dx((1.0, 0.5))
            expected = itp((1.0, 0.5); deriv=DerivOp(1, 0))
            @test result ≈ expected

            # ∂f/∂y
            dy = deriv_view(itp, (0, 1))
            result_y = dy((1.0, 0.5))
            expected_y = itp((1.0, 0.5); deriv=DerivOp(0, 1))
            @test result_y ≈ expected_y

            # Mixed partial ∂²f/∂x∂y
            dxy = deriv_view(itp, (1, 1))
            result_xy = dxy((1.0, 0.5))
            expected_xy = itp((1.0, 0.5); deriv=DerivOp(1, 1))
            @test result_xy ≈ expected_xy
        end

        @testset "deriv_view with Int order (ND broadcast)" begin
            # order=1 for ND → (1, 1) all axes
            d_all = deriv_view(itp, 1)
            result = d_all((1.0, 0.5))
            expected = itp((1.0, 0.5); deriv=DerivOp(1, 1))
            @test result ≈ expected
        end

        @testset "deriv1/deriv2/deriv3 error on ND" begin
            @test_throws ArgumentError deriv1(itp)
            @test_throws ArgumentError deriv2(itp)
            @test_throws ArgumentError deriv3(itp)
        end

        @testset "DerivativeView broadcast" begin
            dx = deriv_view(itp, (1, 0))
            points = [(0.5, 0.3), (1.0, 0.5), (1.5, 0.8)]

            results = dx.(points)
            @test length(results) == 3

            for (k, pt) in enumerate(points)
                @test results[k] ≈ dx(pt) atol=1e-12
            end
        end

        @testset "DerivativeView deriv kwarg rejection" begin
            dx = deriv_view(itp, (1, 0))
            # Attempting to override deriv should throw
            @test_throws ArgumentError dx((1.0, 0.5); deriv=DerivOp(0, 1))
        end
    end

    # ========================================
    # SHOW METHODS (ND SPECIFIC)
    # ========================================
    @testset "Show Methods for CubicInterpolantND" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi * yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @testset "Compact show" begin
            buf = IOBuffer()
            show(buf, itp)
            str = String(take!(buf))

            @test contains(str, "CubicInterpolantND")
            @test contains(str, "11×6") || contains(str, "11")
        end

        @testset "Verbose show (text/plain)" begin
            buf = IOBuffer()
            show(buf, MIME("text/plain"), itp)
            str = String(take!(buf))

            @test contains(str, "CubicInterpolantND")
            @test contains(str, "Grid") || contains(str, "2D")
        end

        @testset "Show with color" begin
            buf = IOBuffer()
            ctx = IOContext(buf, :color => true)
            show(ctx, MIME("text/plain"), itp)
            str = String(take!(buf))

            # Should still contain type info
            @test contains(str, "CubicInterpolantND")
        end

        @testset "Show with mixed BCs" begin
            itp_mixed = cubic_interp((x, y), data; bc=(ZeroCurvBC(), ZeroSlopeBC()))

            buf = IOBuffer()
            show(buf, MIME("text/plain"), itp_mixed)
            str = String(take!(buf))

            @test contains(str, "CubicInterpolantND")
        end

        @testset "Show with non-uniform grids" begin
            x_vec = [0.0, 0.1, 0.3, 0.6, 1.0]
            y_vec = [0.0, 0.5, 1.0]
            data_nu = [xi * yj for xi in x_vec, yj in y_vec]
            itp_nu = cubic_interp((x_vec, y_vec), data_nu)

            buf = IOBuffer()
            show(buf, MIME("text/plain"), itp_nu)
            str = String(take!(buf))

            @test contains(str, "CubicInterpolantND")
            # Should show search info since we have Vector grids
            @test contains(str, "Search") || contains(str, "Vector")
        end

        @testset "3D show" begin
            x = range(0.0, 1.0, 5)
            y = range(0.0, 1.0, 4)
            z = range(0.0, 1.0, 3)
            data3d = [xi + yj + zk for xi in x, yj in y, zk in z]
            itp3d = cubic_interp((x, y, z), data3d)

            buf = IOBuffer()
            show(buf, MIME("text/plain"), itp3d)
            str = String(take!(buf))

            @test contains(str, "CubicInterpolantND")
            @test contains(str, "3") || contains(str, "5×4×3")
        end
    end

    # ========================================
    # DERIVATIVE VIEW SHOW (ND)
    # ========================================
    @testset "DerivativeView Show for ND" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi * yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        dx = deriv_view(itp, (1, 0))

        @testset "Compact show" begin
            buf = IOBuffer()
            show(buf, dx)
            str = String(take!(buf))

            @test contains(str, "DerivativeView")
        end

        @testset "Verbose show" begin
            buf = IOBuffer()
            show(buf, MIME("text/plain"), dx)
            str = String(take!(buf))

            @test contains(str, "DerivativeView")
            @test contains(str, "Parent") || contains(str, "CubicInterpolantND")
        end
    end

    # ========================================
    # TYPE INTROSPECTION AND ACCESSORS
    # ========================================
    @testset "Type Accessors and Introspection" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi * yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @testset "Basic type queries" begin
            @test ndims(itp) == 2
            @test size(itp) == (11, 6)
            @test axes(itp) == (x, y)
            @test grid_type(itp) == Float64
            @test value_type(itp) == Float64
        end

        @testset "num_partials" begin
            @test FastInterpolations.num_partials(itp) == 4  # 2^2 for 2D
            @test FastInterpolations.num_partials(typeof(itp)) == 4

            # 3D should have 8 partials
            x3 = range(0.0, 1.0, 5)
            y3 = range(0.0, 1.0, 4)
            z3 = range(0.0, 1.0, 3)
            data3d = [xi + yj + zk for xi in x3, yj in y3, zk in z3]
            itp3d = cubic_interp((x3, y3, z3), data3d)
            @test FastInterpolations.num_partials(itp3d) == 8  # 2^3
        end

        @testset "Val-based accessors" begin
            # These use Val{D} dispatch
            @test FastInterpolations._grid(itp, Val(1)) === x
            @test FastInterpolations._grid(itp, Val(2)) === y
            @test FastInterpolations._bc(itp, Val(1)) isa FastInterpolations.AbstractBC
            @test FastInterpolations._extrap(itp, Val(1)) === FastInterpolations.NoExtrap()
            @test FastInterpolations._search(itp, Val(1)) isa FastInterpolations.AbstractSearchPolicy
        end
    end

    # ========================================
    # COMPLEX VALUES
    # ========================================
    @testset "Complex Value Support (Extended)" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)

        # Complex function: f(x,y) = exp(i*x) * y
        data = [exp(im * xi) * yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @test value_type(itp) == ComplexF64

        @testset "Complex batch evaluation" begin
            xqs = collect(range(0.5, 2.0, 5))
            yqs = collect(range(0.2, 1.5, 5))

            vals = itp((xqs, yqs))
            @test eltype(vals) == ComplexF64
            @test length(vals) == 5
        end

        @testset "Complex vector calculus" begin
            g = gradient(itp, (1.0, 0.5))
            @test all(v -> v isa Complex, g)

            H = hessian(itp, (1.0, 0.5))
            @test eltype(H) <: Complex

            lap = laplacian(itp, (1.0, 0.5))
            @test lap isa Complex
        end
    end

    # ========================================
    # FLOAT32 SUPPORT
    # ========================================
    @testset "Float32 Support (Extended)" begin
        x = range(0.0f0, 2.0f0, 11)
        y = range(0.0f0, 1.0f0, 6)
        data = Float32[sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @test grid_type(itp) == Float32
        @test value_type(itp) == Float32

        @testset "Float32 batch evaluation" begin
            xqs = Float32[0.5, 1.0, 1.5]
            yqs = Float32[0.2, 0.4, 0.6]

            vals = itp((xqs, yqs))
            @test eltype(vals) == Float32
        end

        @testset "Float32 vector calculus" begin
            g = gradient(itp, (0.5f0, 0.3f0))
            @test all(v -> v isa Float32, g)

            lap = laplacian(itp, (0.5f0, 0.3f0))
            @test lap isa Float32
        end
    end

    # ========================================
    # QUADRATIC ND: SHOW, DERIV VIEW, BATCH
    # ========================================
    @testset "QuadraticND Show Methods" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @testset "compact show" begin
            str = sprint(show, itp)
            @test contains(str, "QuadraticInterpolantND")
        end

        @testset "verbose show" begin
            str = sprint(show, MIME("text/plain"), itp)
            @test contains(str, "QuadraticInterpolantND")
        end

        @testset "show with color" begin
            buf = IOBuffer()
            ctx = IOContext(buf, :color => true)
            show(ctx, MIME("text/plain"), itp)
            str = String(take!(buf))
            @test contains(str, "QuadraticInterpolantND")
        end

        @testset "show with non-uniform grids" begin
            x_vec = [0.0, 0.1, 0.3, 0.6, 1.0]
            y_vec = [0.0, 0.5, 1.0]
            data_nu = [xi^2 + yj^2 for xi in x_vec, yj in y_vec]
            itp_nu = quadratic_interp((x_vec, y_vec), data_nu; bc=Right(QuadraticFit()))

            buf = IOBuffer()
            show(buf, MIME("text/plain"), itp_nu)
            str = String(take!(buf))
            @test contains(str, "QuadraticInterpolantND")
        end
    end

    @testset "QuadraticND DerivativeView" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 11)
        data = [xi^2 + yi^2 for xi in x, yi in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @testset "deriv_view with tuple order" begin
            dx = deriv_view(itp, (1, 0))
            @test dx isa FastInterpolations.DerivativeView

            result = dx((1.0, 0.5))
            expected = itp((1.0, 0.5); deriv=DerivOp(1, 0))
            @test result ≈ expected

            dy = deriv_view(itp, (0, 1))
            @test dy((1.0, 0.5)) ≈ itp((1.0, 0.5); deriv=DerivOp(0, 1))
        end

        @testset "deriv_view broadcast" begin
            dx = deriv_view(itp, (1, 0))
            points = [(0.5, 0.3), (1.0, 0.5), (1.5, 0.8)]
            results = dx.(points)
            @test length(results) == 3
            for (k, pt) in enumerate(points)
                @test results[k] ≈ dx(pt) atol=1e-12
            end
        end

        @testset "deriv1/deriv2/deriv3 error on ND" begin
            @test_throws ArgumentError deriv1(itp)
            @test_throws ArgumentError deriv2(itp)
            @test_throws ArgumentError deriv3(itp)
        end
    end

    @testset "QuadraticND Batch APIs" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + yi^2
        data = [f(xi, yi) for xi in x, yi in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @testset "SoA batch with derivatives" begin
            xqs = collect(range(0.2, 1.8, 5))
            yqs = collect(range(0.1, 0.9, 5))

            vals = itp((xqs, yqs))
            for k in eachindex(xqs)
                @test vals[k] ≈ itp((xqs[k], yqs[k])) atol=1e-12
            end

            vals_dx = itp((xqs, yqs); deriv=DerivOp(1, 0))
            vals_dx_val = itp((xqs, yqs); deriv=DerivOp(1, 0))
            @test vals_dx ≈ vals_dx_val

            vals_d1 = itp((xqs, yqs); deriv=DerivOp(1, 1))
            @test length(vals_d1) == 5
        end

        @testset "AoS batch with derivatives" begin
            points = [(0.5, 0.3), (1.0, 0.5), (1.5, 0.8)]
            vals = itp(points)
            @test length(vals) == 3

            vals_dx = itp(points; deriv=DerivOp(1, 0))
            vals_dx_val = itp(points; deriv=DerivOp(1, 0))
            @test vals_dx ≈ vals_dx_val
        end

        @testset "SoA length mismatch" begin
            xqs = collect(range(0.2, 1.8, 5))
            yqs = collect(range(0.1, 0.9, 3))
            @test_throws DimensionMismatch itp((xqs, yqs))
        end
    end

    @testset "QuadraticND Integer Grid Conversion" begin
        x = 1:10
        y = 1:5
        data = Float64[Float64(xi^2 + yj^2) for xi in x, yj in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))
        @test grid_type(itp) == Float64
    end

    # ========================================
    # GRID CONVERSION EDGE CASES
    # ========================================
    @testset "Grid Conversion" begin
        @testset "Mixed grid types (Range + Vector)" begin
            x_range = range(0.0, 1.0, 11)  # Range
            y_vec = [0.0, 0.2, 0.5, 0.8, 1.0]  # Vector
            data = [xi * yj for xi in x_range, yj in y_vec]

            itp = cubic_interp((x_range, y_vec), data)
            @test itp((0.5, 0.3)) isa Float64
        end

        @testset "Integer grid conversion" begin
            # Integer grids should be converted to Float64
            x = 1:10
            y = 1:5
            data = Float64[xi * yj for xi in x, yj in y]

            itp = cubic_interp((x, y), data)
            @test grid_type(itp) == Float64
        end
    end

    # ========================================
    # ONESHOT API
    # ========================================
    @testset "Oneshot API Variations" begin
        x = range(0.0, 2.0, 21)
        y = range(0.0, 1.0, 11)
        data = [xi * yj for xi in x, yj in y]

        @testset "Single point oneshot" begin
            val = cubic_interp((x, y), data, (1.0, 0.5))
            @test val ≈ 1.0 * 0.5 atol=1e-3
        end

        @testset "Single point oneshot with derivative" begin
            # ∂f/∂x = y
            val_dx = cubic_interp((x, y), data, (1.0, 0.5); deriv=DerivOp(1, 0))
            @test val_dx ≈ 0.5 atol=1e-3
        end

        @testset "Batch oneshot" begin
            xqs = [0.5, 1.0, 1.5]
            yqs = [0.2, 0.4, 0.6]
            vals = cubic_interp((x, y), data, (xqs, yqs))

            @test length(vals) == 3
            for k in 1:3
                @test vals[k] ≈ xqs[k] * yqs[k] atol=1e-3
            end
        end
    end

    # ========================================
    # ERROR HANDLING
    # ========================================
    @testset "Error Handling" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 6)
        data = rand(11, 6)
        itp = cubic_interp((x, y), data)

        @testset "Domain errors with NoExtrap() extrap" begin
            @test_throws DomainError itp((-0.1, 0.5))
            @test_throws DomainError itp((1.1, 0.5))
            @test_throws DomainError itp((0.5, -0.1))
            @test_throws DomainError itp((0.5, 1.1))
        end

        @testset "Dimension mismatch in construction" begin
            bad_data = rand(10, 6)  # Wrong x dimension
            @test_throws DimensionMismatch cubic_interp((x, y), bad_data)

            bad_data2 = rand(11, 5)  # Wrong y dimension
            @test_throws DimensionMismatch cubic_interp((x, y), bad_data2)
        end

        @testset "OnTheFly strategy not implemented" begin
            @test_throws ArgumentError cubic_interp((x, y), data; coeffs=OnTheFly())
        end
    end

end
