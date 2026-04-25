@testitem "QuadraticAdjoint" setup = [AllocConstants] begin
    using LinearAlgebra: dot

    # ========================================
    # Helper: Dot-product test for quadratic adjoint correctness
    # ========================================
    # Gold standard: ⟨W·f, ȳ⟩ = ⟨f, Wᵀ·ȳ⟩
    # W is affine (non-zero BC values add a constant offset), so we
    # subtract the zero-data response to isolate the linear part.

    function quadratic_dot_test(
            x, xq, f, y_bar;
            bc = Left(QuadraticFit()), extrap = NoExtrap(), deriv = EvalValue(),
            atol = 0, rtol = sqrt(eps(eltype(x)))
        )
        itp = quadratic_interp(x, f; bc = bc, extrap = extrap)
        adj = quadratic_adjoint(x, xq; bc = bc, extrap = extrap)

        # Subtract constant offset from non-zero BC values (e.g. Deriv1(0.5))
        f_zero = zeros(eltype(f), length(f))
        itp_zero = quadratic_interp(x, f_zero; bc = bc, extrap = extrap)
        Wf = itp.(xq; deriv = deriv) .- itp_zero.(xq; deriv = deriv)

        WTy = adj(y_bar; deriv = deriv)

        lhs = dot(Wf, y_bar)
        rhs = dot(f, WTy)
        return lhs, rhs, isapprox(lhs, rhs; atol = atol, rtol = rtol)
    end


    # ========================================
    # Test data setup
    # ========================================
    n_grid = 50
    n_query = 30

    x_uniform = collect(range(0.0, 1.0, n_grid))
    x_nonuniform = cumsum(0.5 .+ rand(n_grid))
    x_nonuniform .= (x_nonuniform .- x_nonuniform[1]) ./ (x_nonuniform[end] - x_nonuniform[1])

    xq = sort(rand(n_query)) .* 0.98 .+ 0.01  # inside domain with margin
    f = randn(n_grid)
    y_bar = randn(n_query)

    # ========================================
    # Dot-product tests: all BC types
    # ========================================
    @testset "Dot-product — $bc_name" for (bc_name, bc) in [
            ("Left(QuadraticFit) default", Left(QuadraticFit())),
            ("Right(QuadraticFit)", Right(QuadraticFit())),
            ("Left(Deriv1(0.5))", Left(Deriv1(0.5))),
            ("Right(Deriv1(-0.3))", Right(Deriv1(-0.3))),
            ("Left(Deriv2(0.0))", Left(Deriv2(0.0))),
            ("Right(Deriv2(0.0))", Right(Deriv2(0.0))),
            ("Left(Deriv2(1.0))", Left(Deriv2(1.0))),
            ("Right(Deriv2(-0.5))", Right(Deriv2(-0.5))),
            ("Left(CubicFit)", Left(CubicFit())),
            ("Right(CubicFit)", Right(CubicFit())),
            ("Left(LinearFit)", Left(LinearFit())),
            ("Right(LinearFit)", Right(LinearFit())),
            ("MinCurvFit", MinCurvFit()),
        ]
        @testset "Uniform grid" begin
            _, _, ok = quadratic_dot_test(x_uniform, xq, f, y_bar; bc = bc)
            @test ok
        end
        @testset "Non-uniform grid" begin
            _, _, ok = quadratic_dot_test(x_nonuniform, xq, f, y_bar; bc = bc)
            @test ok
        end
    end

    # ========================================
    # Dot-product tests: extrap modes with OOB queries
    # ========================================
    xq_oob = vcat(-0.3, -0.1, sort(rand(n_query)) .* 0.98 .+ 0.01, 1.1, 1.3)
    y_bar_oob = randn(length(xq_oob))

    @testset "Dot-product — OOB queries — ExtendExtrap" begin
        @testset "Uniform" begin
            _, _, ok = quadratic_dot_test(x_uniform, xq_oob, f, y_bar_oob; extrap = ExtendExtrap())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = quadratic_dot_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = ExtendExtrap())
            @test ok
        end
    end

    @testset "Dot-product — OOB queries — ClampExtrap" begin
        @testset "Uniform" begin
            _, _, ok = quadratic_dot_test(x_uniform, xq_oob, f, y_bar_oob; extrap = ClampExtrap())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = quadratic_dot_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = ClampExtrap())
            @test ok
        end
    end

    @testset "Dot-product — OOB queries — WrapExtrap" begin
        @testset "Uniform" begin
            _, _, ok = quadratic_dot_test(x_uniform, xq_oob, f, y_bar_oob; extrap = WrapExtrap())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = quadratic_dot_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = WrapExtrap())
            @test ok
        end
    end

    @testset "Dot-product — OOB queries — FillExtrap" begin
        @testset "Uniform" begin
            _, _, ok = quadratic_dot_test(x_uniform, xq_oob, f, y_bar_oob; extrap = FillExtrap(0.0))
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = quadratic_dot_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = FillExtrap(0.0))
            @test ok
        end
    end

    # ========================================
    # In-place vs allocating
    # ========================================
    @testset "In-place == allocating" begin
        adj = quadratic_adjoint(x_uniform, xq; bc = Left(QuadraticFit()))
        f_bar_oop = adj(y_bar)
        f_bar_ip = zeros(n_grid)
        adj(f_bar_ip, y_bar)
        @test f_bar_oop ≈ f_bar_ip
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = quadratic_adjoint(x_uniform, xq)
        @test size(adj) == (n_grid, n_query)
        @test size(adj, 1) == n_grid
        @test size(adj, 2) == n_query
    end

    # ========================================
    # Type stability — constructor @inferred
    # ========================================
    @testset "Type stability — constructor" begin
        @test @inferred(quadratic_adjoint(x_uniform, xq)) isa QuadraticAdjoint
        @test @inferred(quadratic_adjoint(x_uniform, xq; bc = Left(QuadraticFit()))) isa QuadraticAdjoint
        @test @inferred(quadratic_adjoint(x_uniform, xq; bc = Right(QuadraticFit()))) isa QuadraticAdjoint
        @test @inferred(quadratic_adjoint(x_uniform, xq; bc = Left(Deriv1(0.0)))) isa QuadraticAdjoint
        @test @inferred(quadratic_adjoint(x_uniform, xq; bc = Right(Deriv2(0.0)))) isa QuadraticAdjoint
        @test @inferred(quadratic_adjoint(x_uniform, xq; bc = MinCurvFit())) isa QuadraticAdjoint

        # Extrap variations
        @test @inferred(quadratic_adjoint(x_uniform, xq; extrap = NoExtrap())) isa QuadraticAdjoint
        @test @inferred(quadratic_adjoint(x_uniform, xq; extrap = ExtendExtrap())) isa QuadraticAdjoint
        @test @inferred(quadratic_adjoint(x_uniform, xq; extrap = ClampExtrap())) isa QuadraticAdjoint
        @test @inferred(quadratic_adjoint(x_uniform, xq; extrap = WrapExtrap())) isa QuadraticAdjoint
        @test @inferred(quadratic_adjoint(x_uniform, xq; extrap = FillExtrap(0.0))) isa QuadraticAdjoint
    end

    # ========================================
    # Type stability — apply @inferred
    # ========================================
    @testset "Type stability — apply Float64" begin
        adj = quadratic_adjoint(x_uniform, xq; bc = Left(QuadraticFit()))
        @test @inferred(adj(y_bar)) isa Vector{Float64}
    end

    @testset "Type stability — apply Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        yb32 = randn(Float32, n_query)
        adj32 = quadratic_adjoint(x32, xq32; bc = Left(QuadraticFit()))
        @test @inferred(adj32(yb32)) isa Vector{Float32}
    end

    # ========================================
    # Float32 support
    # ========================================
    @testset "Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        f32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)

        _, _, ok = quadratic_dot_test(x32, xq32, f32, yb32; bc = Left(QuadraticFit()), rtol = sqrt(eps(Float32)))
        @test ok
    end

    # ========================================
    # Derivative adjoint: deriv keyword
    # ========================================
    @testset "Derivative adjoint — $d_name, $bc_name" for
        (d_name, deriv_op) in [
                ("deriv=1", EvalDeriv1()),
                ("deriv=2", EvalDeriv2()),
                ("deriv=3", EvalDeriv3()),
            ],
            (bc_name, bc) in [
                ("Left(QuadraticFit)", Left(QuadraticFit())),
                ("Right(QuadraticFit)", Right(QuadraticFit())),
                ("Left(Deriv1(0.5))", Left(Deriv1(0.5))),
                ("Right(Deriv2(0.0))", Right(Deriv2(0.0))),
                ("MinCurvFit", MinCurvFit()),
            ]
        @testset "Uniform grid" begin
            _, _, ok = quadratic_dot_test(x_uniform, xq, f, y_bar; bc = bc, deriv = deriv_op)
            @test ok
        end
        @testset "Non-uniform grid" begin
            _, _, ok = quadratic_dot_test(x_nonuniform, xq, f, y_bar; bc = bc, deriv = deriv_op)
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: in-place == allocating
    # ========================================
    @testset "In-place == allocating — deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        adj = quadratic_adjoint(x_uniform, xq; bc = Left(QuadraticFit()))
        fb_oop = adj(y_bar; deriv = op)
        fb_ip = zeros(n_grid)
        adj(fb_ip, y_bar; deriv = op)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # Float32 derivative adjoint
    # ========================================
    @testset "Float32 — deriv=1" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        f32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        _, _, ok = quadratic_dot_test(x32, xq32, f32, yb32; bc = Left(QuadraticFit()), deriv = EvalDeriv1(), rtol = sqrt(eps(Float32)))
        @test ok
    end

    # ========================================
    # Complex data support
    # ========================================
    @testset "Complex — dot-product — $bc_name" for (bc_name, bc) in [
            ("Left(QuadraticFit)", Left(QuadraticFit())),
            ("MinCurvFit", MinCurvFit()),
            ("Left(Deriv1(0.5))", Left(Deriv1(0.5))),
        ]
        f_c = randn(ComplexF64, n_grid)
        yb_c = randn(ComplexF64, n_query)
        @testset "Uniform" begin
            _, _, ok = quadratic_dot_test(x_uniform, xq, f_c, yb_c; bc = bc)
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = quadratic_dot_test(x_nonuniform, xq, f_c, yb_c; bc = bc)
            @test ok
        end
    end

    @testset "Complex — in-place == allocating" begin
        adj = quadratic_adjoint(x_uniform, xq; bc = Left(QuadraticFit()))
        yb_c = randn(ComplexF64, n_query)
        fb_oop = adj(yb_c)
        fb_ip = zeros(ComplexF64, n_grid)
        adj(fb_ip, yb_c)
        @test fb_oop ≈ fb_ip
    end

    @testset "Complex — derivative adjoint — deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        f_c = randn(ComplexF64, n_grid)
        yb_c = randn(ComplexF64, n_query)
        _, _, ok = quadratic_dot_test(x_uniform, xq, f_c, yb_c; bc = Left(QuadraticFit()), deriv = op)
        @test ok
    end

    # ========================================
    # Edge cases: minimum grid sizes
    # ========================================
    @testset "Minimum grid — QuadraticFit (3 points)" begin
        x_min = collect(range(0.0, 1.0, 3))
        xq_min = [0.25, 0.5, 0.75]
        f_min = randn(3)
        yb_min = randn(3)
        _, _, ok = quadratic_dot_test(x_min, xq_min, f_min, yb_min; bc = Left(QuadraticFit()))
        @test ok
    end

    @testset "Minimum grid — Deriv1 (2 points)" begin
        x_min = [0.0, 1.0]
        xq_min = [0.5]
        f_min = randn(2)
        yb_min = randn(1)
        _, _, ok = quadratic_dot_test(x_min, xq_min, f_min, yb_min; bc = Left(Deriv1(0.0)))
        @test ok
    end

    @testset "Minimum grid — MinCurvFit (2 points)" begin
        x_min = [0.0, 1.0]
        xq_min = [0.5]
        f_min = randn(2)
        yb_min = randn(1)
        _, _, ok = quadratic_dot_test(x_min, xq_min, f_min, yb_min; bc = MinCurvFit())
        @test ok
    end

    # ========================================
    # Matrix materialization
    # ========================================
    @testset "Matrix materialization" begin
        adj = quadratic_adjoint(x_uniform, xq; bc = Left(QuadraticFit()))
        M = Matrix(adj)
        @test size(M) == (n_grid, n_query)
        @test M * y_bar ≈ adj(y_bar)
    end

    @testset "Matrix — deriv=1" begin
        adj = quadratic_adjoint(x_uniform, xq; bc = Left(QuadraticFit()))
        M1 = Matrix(adj; deriv = DerivOp(1))
        @test M1 * y_bar ≈ adj(y_bar; deriv = DerivOp(1))
    end

    # ========================================
    # Matrix(itp, xq) — convenience forward matrix
    # ========================================
    @testset "Matrix(itp, xq) — forward matrix" begin
        itp = quadratic_interp(x_uniform, f; bc = Left(QuadraticFit()))
        W = Matrix(itp, xq)
        @test size(W) == (n_query, n_grid)
        @test W * f ≈ itp.(xq)
    end

    @testset "Matrix(itp, xq) — deriv=1" begin
        itp = quadratic_interp(x_uniform, f; bc = Left(QuadraticFit()))
        W1 = Matrix(itp, xq; deriv = DerivOp(1))
        @test W1 * f ≈ itp.(xq; deriv = DerivOp(1))
    end

    @testset "Matrix(itp, xq) — non-zero BC (Deriv1)" begin
        bc = Left(Deriv1(0.5))
        itp = quadratic_interp(x_uniform, f; bc = bc)
        W = Matrix(itp, xq)
        @test size(W) == (n_query, n_grid)
        # For non-zero BC, W*f matches only the linear part; verify via itp.bc round-trip
        @test itp.bc === bc
    end

    @testset "Matrix(itp, xq) — MinCurvFit" begin
        itp = quadratic_interp(x_uniform, f; bc = MinCurvFit())
        W = Matrix(itp, xq)
        @test W * f ≈ itp.(xq)
    end

    # ========================================
    # Scalar query convenience
    # ========================================
    @testset "Scalar query" begin
        adj_scalar = quadratic_adjoint(x_uniform, 0.5)
        @test size(adj_scalar) == (n_grid, 1)
        fb = adj_scalar([1.0])
        @test length(fb) == n_grid
    end

    # ========================================
    # NoExtrap domain validation
    # ========================================
    @testset "NoExtrap — DomainError for OOB" begin
        @test_throws DomainError quadratic_adjoint(x_uniform, [-0.1, 0.5])
        @test_throws DomainError quadratic_adjoint(x_uniform, [0.5, 1.1])
    end

    # ========================================
    # EvalDeriv3 is zero
    # ========================================
    @testset "EvalDeriv3 → zero output" begin
        adj = quadratic_adjoint(x_uniform, xq; bc = Left(QuadraticFit()))
        fb = adj(y_bar; deriv = DerivOp(3))
        @test all(iszero, fb)
    end

    # ========================================
    # DimensionMismatch error tests (T1)
    # ========================================
    @testset "DimensionMismatch errors" begin
        adj = quadratic_adjoint(x_uniform, xq; bc = Left(QuadraticFit()))

        # y_bar wrong length (allocating)
        @test_throws DimensionMismatch adj(randn(n_query + 1))

        # f_bar wrong length (in-place)
        @test_throws DimensionMismatch adj(zeros(n_grid + 1), y_bar)

        # Scalar y_bar when multiple queries
        @test_throws DimensionMismatch adj(1.5)

        # 1-tuple y_bar when multiple queries
        @test_throws DimensionMismatch adj((1.0,))
    end

    # ========================================
    # Scalar / Tuple y_bar (T2)
    # ========================================
    @testset "Scalar y_bar (1 query point)" begin
        xq_single = [0.3]
        adj = quadratic_adjoint(x_uniform, xq_single; bc = Left(QuadraticFit()))
        ref = adj([1.5])

        # Scalar y_bar
        @test adj(1.5) ≈ ref

        # In-place scalar
        f_bar = zeros(n_grid)
        adj(f_bar, 1.5)
        @test f_bar ≈ ref
    end

    @testset "Scalar y_bar with deriv" begin
        xq_single = [0.4]
        adj = quadratic_adjoint(x_uniform, xq_single; bc = Left(QuadraticFit()))
        ref = adj([1.0]; deriv = DerivOp(1))
        @test adj(1.0; deriv = DerivOp(1)) ≈ ref
    end

    @testset "Tuple y_bar (multiple query points)" begin
        xq_tup = [0.2, 0.5, 0.8]
        adj = quadratic_adjoint(x_uniform, xq_tup; bc = Left(QuadraticFit()))
        ref = adj([1.0, 2.0, 3.0])

        # Tuple y_bar
        @test adj((1.0, 2.0, 3.0)) ≈ ref

        # In-place tuple
        f_bar = zeros(n_grid)
        adj(f_bar, (1.0, 2.0, 3.0))
        @test f_bar ≈ ref
    end

    # ========================================
    # Mutation safety — grid mutation after construction
    # ========================================
    @testset "Mutation safety — grid mutation" begin
        x = collect(range(0.0, 1.0, n_grid))
        adj = quadratic_adjoint(x, xq; bc = Left(QuadraticFit()))
        result_before = adj(y_bar)
        x[5] = 100.0
        result_after = adj(y_bar)
        @test result_before == result_after
    end

    # ========================================
    # Allocation tests (zero-alloc in-place)
    # ========================================

    # Function barriers: @testset wraps body in try/catch → type-unstable locals.
    # All setup + warmup + @allocated must be inside ONE function for accurate results.

    function _test_quadratic_adjoint_alloc_inplace(
            x, xq, f_bar, y_bar; bc = Left(QuadraticFit()), deriv = EvalValue()
        )
        adj = quadratic_adjoint(x, xq; bc = bc)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place QuadraticFit" begin
        fb = zeros(n_grid)
        allocs = _test_quadratic_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place non-uniform" begin
        fb = zeros(n_grid)
        allocs = _test_quadratic_adjoint_alloc_inplace(x_nonuniform, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Deriv1(0.5)" begin
        fb = zeros(n_grid)
        allocs = _test_quadratic_adjoint_alloc_inplace(
            x_uniform, xq, fb, y_bar; bc = Left(Deriv1(0.5))
        )
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Deriv2(0.0)" begin
        fb = zeros(n_grid)
        allocs = _test_quadratic_adjoint_alloc_inplace(
            x_uniform, xq, fb, y_bar; bc = Left(Deriv2(0.0))
        )
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place MinCurvFit" begin
        fb = zeros(n_grid)
        allocs = _test_quadratic_adjoint_alloc_inplace(
            x_uniform, xq, fb, y_bar; bc = MinCurvFit()
        )
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        fb = zeros(n_grid)
        allocs = _test_quadratic_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar; deriv = op)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        fb32 = zeros(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        allocs = _test_quadratic_adjoint_alloc_inplace(x32, xq32, fb32, yb32)
        @test allocs <= ALLOC_THRESHOLD
    end
end
