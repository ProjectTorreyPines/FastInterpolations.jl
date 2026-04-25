@testitem "HermiteAdjoint1D" setup = [AllocConstants] begin
    using LinearAlgebra: dot

    # ========================================
    # Helper: Dot-product test for Hermite adjoint correctness
    # ========================================
    # With dy=0, the forward is purely W_y * y, so: dot(itp.(xq), y_bar) = dot(y, adj(y_bar))
    # With non-zero dy, use Matrix materialization: dot(W_T' * y, y_bar) = dot(y, W_T * y_bar)

    function hermite_dot_product_test(
            x, xq, y, dy, y_bar;
            extrap = NoExtrap(), deriv = EvalValue(),
            atol = 0, rtol = sqrt(eps(eltype(x)))
        )
        itp = hermite_interp(x, y, dy; extrap = extrap)
        adj = hermite_adjoint(x, xq; extrap = extrap)

        Wf = itp.(xq; deriv = deriv)      # forward eval
        WTy = adj(y_bar; deriv = deriv)    # adjoint

        lhs = dot(Wf, y_bar)
        rhs = dot(y, WTy)
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
    y = randn(n_grid)
    dy_zero = zeros(n_grid)
    y_bar = randn(n_query)

    # ========================================
    # Analytical — single midpoint query (dy=0)
    # ========================================
    @testset "Analytical — single midpoint (dy=0)" begin
        x3 = [0.0, 1.0, 2.0]
        adj = hermite_adjoint(x3, [0.5])

        # At t=0.5: h00(0.5) = 2(0.125) - 3(0.25) + 1 = 0.5
        #           h01(0.5) = -2(0.125) + 3(0.25) = 0.5
        # So adj([1.0]) should scatter 0.5 to idx=1 and 0.5 to idx=2, zero elsewhere
        result = adj([1.0])
        @test result[1] ≈ 0.5
        @test result[2] ≈ 0.5
        @test result[3] ≈ 0.0
    end

    # ========================================
    # Analytical — two queries accumulation (dy=0)
    # ========================================
    @testset "Analytical — two-query accumulation (dy=0)" begin
        x3 = [0.0, 1.0, 2.0]
        # Two queries: 0.5 (in [0,1]) and 1.5 (in [1,2])
        adj = hermite_adjoint(x3, [0.5, 1.5])

        # Query 0.5: t=0.5, idx=1 → 0.5*yb[1] to node 1, 0.5*yb[1] to node 2
        # Query 1.5: t=0.5, idx=2 → 0.5*yb[2] to node 2, 0.5*yb[2] to node 3
        # With y_bar = [1.0, 1.0]:
        result = adj([1.0, 1.0])
        @test result[1] ≈ 0.5         # from query 1 only
        @test result[2] ≈ 0.5 + 0.5   # from both queries
        @test result[3] ≈ 0.5         # from query 2 only
    end

    # ========================================
    # Type stability — constructor
    # ========================================
    @testset "Type stability — constructor @inferred" begin
        @test @inferred(hermite_adjoint(x_uniform, xq)) isa HermiteAdjoint1D
        @test @inferred(hermite_adjoint(x_uniform, xq; extrap = NoExtrap())) isa HermiteAdjoint1D
        @test @inferred(hermite_adjoint(x_uniform, xq; extrap = ExtendExtrap())) isa HermiteAdjoint1D
        @test @inferred(hermite_adjoint(x_uniform, xq; extrap = ClampExtrap())) isa HermiteAdjoint1D
        @test @inferred(hermite_adjoint(x_uniform, xq; extrap = FillExtrap(0.0))) isa HermiteAdjoint1D
        @test @inferred(hermite_adjoint(x_uniform, xq; extrap = WrapExtrap())) isa HermiteAdjoint1D
    end

    # ========================================
    # Type stability — apply
    # ========================================
    @testset "Type stability — apply Float64" begin
        adj = hermite_adjoint(x_uniform, xq)
        @test @inferred(adj(y_bar)) isa Vector{Float64}
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = hermite_adjoint(x_uniform, xq)
        @test size(adj) == (n_grid, n_query)
        @test size(adj, 1) == n_grid
        @test size(adj, 2) == n_query
    end

    # ========================================
    # Dot-product tests: dy=0, NoExtrap (uniform + non-uniform)
    # ========================================
    @testset "Dot-product — dy=0 — NoExtrap" begin
        @testset "Uniform grid" begin
            _, _, ok = hermite_dot_product_test(x_uniform, xq, y, dy_zero, y_bar)
            @test ok
        end
        @testset "Non-uniform grid" begin
            _, _, ok = hermite_dot_product_test(x_nonuniform, xq, y, dy_zero, y_bar)
            @test ok
        end
    end

    # ========================================
    # Dot-product tests: dy=0, extrap modes with OOB queries
    # ========================================
    xq_oob = vcat(-0.3, -0.1, sort(rand(n_query)) .* 0.98 .+ 0.01, 1.1, 1.3)
    y_bar_oob = randn(length(xq_oob))

    @testset "Dot-product — dy=0 — $ep_name" for (ep_name, ep) in [
            ("ExtendExtrap", ExtendExtrap()),
            ("ClampExtrap", ClampExtrap()),
            ("FillExtrap", FillExtrap(0.0)),
        ]
        @testset "Uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_uniform, xq_oob, y, dy_zero, y_bar_oob; extrap = ep
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_nonuniform, xq_oob, y, dy_zero, y_bar_oob; extrap = ep
            )
            @test ok
        end
    end

    # ========================================
    # Dot-product: WrapExtrap (dy=0)
    # ========================================
    @testset "Dot-product — dy=0 — WrapExtrap" begin
        @testset "Uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_uniform, xq_oob, y, dy_zero, y_bar_oob; extrap = WrapExtrap()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_nonuniform, xq_oob, y, dy_zero, y_bar_oob; extrap = WrapExtrap()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv1 (dy=0)
    # ========================================
    @testset "Dot-product — dy=0 — EvalDeriv1" begin
        @testset "Uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_uniform, xq, y, dy_zero, y_bar; deriv = EvalDeriv1()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_nonuniform, xq, y, dy_zero, y_bar; deriv = EvalDeriv1()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv2 (dy=0)
    # ========================================
    @testset "Dot-product — dy=0 — EvalDeriv2" begin
        @testset "Uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_uniform, xq, y, dy_zero, y_bar; deriv = EvalDeriv2()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_nonuniform, xq, y, dy_zero, y_bar; deriv = EvalDeriv2()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv3 (dy=0)
    # ========================================
    @testset "Dot-product — dy=0 — EvalDeriv3" begin
        @testset "Uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_uniform, xq, y, dy_zero, y_bar; deriv = EvalDeriv3()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = hermite_dot_product_test(
                x_nonuniform, xq, y, dy_zero, y_bar; deriv = EvalDeriv3()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: DerivOp(4) — zero (cubic)
    # ========================================
    @testset "Dot-product — dy=0 — DerivOp(4) is zero" begin
        adj = hermite_adjoint(x_uniform, xq)
        result = adj(y_bar; deriv = DerivOp(4))
        @test all(iszero, result)
    end

    # ========================================
    # Matrix materialization
    # ========================================
    @testset "Matrix(adj) -- W_T * y_bar == adj(y_bar)" begin
        adj = hermite_adjoint(x_uniform, xq)
        W_T = Matrix(adj)
        @test size(W_T) == (n_grid, n_query)
        @test W_T * y_bar ≈ adj(y_bar)
    end

    @testset "Matrix(adj)' -- W * y == itp.(xq) with dy=0" begin
        itp = hermite_interp(x_uniform, y, dy_zero; extrap = NoExtrap())
        adj = hermite_adjoint(x_uniform, xq)
        W = Matrix(adj)'
        @test W * y ≈ itp.(xq)
    end

    @testset "Matrix(adj) -- with random dy" begin
        # Matrix materialization is always correct (linear algebra identity)
        dy_rand = randn(n_grid)
        adj = hermite_adjoint(x_uniform, xq)
        W_T = Matrix(adj)
        @test W_T * y_bar ≈ adj(y_bar)
    end

    @testset "Matrix -- deriv keyword" begin
        adj = hermite_adjoint(x_uniform, xq)
        for (name, op) in [("d1", DerivOp(1)), ("d2", DerivOp(2)), ("d3", DerivOp(3))]
            W_T = Matrix(adj; deriv = op)
            @test W_T * y_bar ≈ adj(y_bar; deriv = op)
        end
    end

    @testset "Matrix -- non-uniform grid" begin
        adj = hermite_adjoint(x_nonuniform, xq)
        itp = hermite_interp(x_nonuniform, y, dy_zero; extrap = NoExtrap())
        W_T = Matrix(adj)
        @test W_T * y_bar ≈ adj(y_bar)
        @test Matrix(adj)' * y ≈ itp.(xq)
    end

    # ========================================
    # In-place == allocating
    # ========================================
    @testset "In-place == allocating" begin
        adj = hermite_adjoint(x_uniform, xq)
        f_bar_oop = adj(y_bar)
        f_bar_ip = zeros(n_grid)
        adj(f_bar_ip, y_bar)
        @test f_bar_oop ≈ f_bar_ip
    end

    @testset "In-place == allocating -- deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        adj = hermite_adjoint(x_uniform, xq)
        fb_oop = adj(y_bar; deriv = op)
        fb_ip = zeros(n_grid)
        adj(fb_ip, y_bar; deriv = op)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # Float32 support
    # ========================================
    @testset "Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        y32 = randn(Float32, n_grid)
        dy32 = zeros(Float32, n_grid)
        yb32 = randn(Float32, n_query)

        _, _, ok = hermite_dot_product_test(x32, xq32, y32, dy32, yb32; rtol = sqrt(eps(Float32)))
        @test ok
    end

    @testset "Float32 — type stability" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        yb32 = randn(Float32, n_query)
        adj32 = hermite_adjoint(x32, xq32)
        @test @inferred(adj32(yb32)) isa Vector{Float32}
    end

    @testset "Float32 — Matrix eltype" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        adj32 = hermite_adjoint(x32, xq32)
        W_T = Matrix(adj32)
        @test eltype(W_T) == Float32
        @test size(W_T) == (n_grid, n_query)
    end

    # ========================================
    # Scalar / Tuple y_bar + scalar query constructor
    # ========================================
    @testset "Scalar y_bar (1 query point)" begin
        xq_single = [0.3]
        adj = hermite_adjoint(x_uniform, xq_single)
        ref = adj([1.5])
        @test adj(1.5) ≈ ref
        @test adj((1.5,)) ≈ ref

        # In-place
        f_bar = zeros(n_grid)
        adj(f_bar, 1.5)
        @test f_bar ≈ ref

        fill!(f_bar, 0.0)
        adj(f_bar, (1.5,))
        @test f_bar ≈ ref
    end

    @testset "Tuple y_bar (multiple query points)" begin
        adj = hermite_adjoint(x_uniform, xq)
        ref = adj(collect(y_bar))
        @test adj(Tuple(y_bar)) ≈ ref

        # In-place
        f_bar = zeros(n_grid)
        adj(f_bar, Tuple(y_bar))
        @test f_bar ≈ ref
    end

    @testset "Scalar query constructor" begin
        adj_vec = hermite_adjoint(x_uniform, [0.5])
        adj_scalar = hermite_adjoint(x_uniform, 0.5)
        @test adj_vec(1.0) ≈ adj_scalar(1.0)
    end

    @testset "Dimension mismatch errors" begin
        adj = hermite_adjoint(x_uniform, xq)  # n_query queries
        @test_throws DimensionMismatch adj(1.5)            # scalar but n_query queries
        @test_throws DimensionMismatch adj((1.0,))          # 1-tuple but n_query queries
    end

    # ========================================
    # Edge cases: minimum grid size
    # ========================================
    @testset "Minimum grid (2 points)" begin
        x_min = [0.0, 1.0]
        xq_min = [0.5]
        y_min = randn(2)
        dy_min = zeros(2)
        yb_min = randn(1)
        _, _, ok = hermite_dot_product_test(x_min, xq_min, y_min, dy_min, yb_min)
        @test ok
    end

    @testset "Grid too small error" begin
        @test_throws ArgumentError hermite_adjoint([0.0], [0.5])
    end

    @testset "NoExtrap domain error" begin
        @test_throws DomainError hermite_adjoint(x_uniform, [-0.1])
        @test_throws DomainError hermite_adjoint(x_uniform, [1.1])
    end

    # ========================================
    # Non-zero dy: Matrix-based verification
    # ========================================
    @testset "Non-zero dy -- Matrix identity holds" begin
        # Even with non-zero dy, the adjoint operator W_T satisfies:
        # W_T * y_bar == adj(y_bar)  (by construction)
        # W_T' * y == y-contribution to forward (not the full itp.(xq))
        dy_rand = randn(n_grid)
        adj = hermite_adjoint(x_uniform, xq)
        W_T = Matrix(adj)

        # Matrix-vector identity always holds
        @test W_T * y_bar ≈ adj(y_bar)

        # The forward with dy=0 isolates W_y:
        itp_dy0 = hermite_interp(x_uniform, y, dy_zero)
        itp_full = hermite_interp(x_uniform, y, dy_rand)
        @test W_T' * y ≈ itp_dy0.(xq)  # W_y * y = forward with dy=0
        # Full forward includes dy contribution:
        # itp_full.(xq) = W_y * y + W_dy * dy != W_T' * y
        @test !(W_T' * y ≈ itp_full.(xq))
    end

    # ========================================
    # Allocation tests (zero-alloc in-place)
    # ========================================
    function _test_hermite_adjoint_alloc_inplace(x, xq, f_bar, y_bar; extrap = NoExtrap(), deriv = EvalValue())
        adj = hermite_adjoint(x, xq; extrap = extrap)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place uniform" begin
        fb = zeros(n_grid)
        allocs = _test_hermite_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place non-uniform" begin
        fb = zeros(n_grid)
        allocs = _test_hermite_adjoint_alloc_inplace(x_nonuniform, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        fb = zeros(n_grid)
        allocs = _test_hermite_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar; deriv = op)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        fb32 = zeros(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        allocs = _test_hermite_adjoint_alloc_inplace(x32, xq32, fb32, yb32)
        @test allocs <= ALLOC_THRESHOLD
    end
end

# ========================================
# CardinalAdjoint1D
# ========================================

@testitem "CardinalAdjoint1D" setup = [AllocConstants] begin
    using LinearAlgebra: dot

    function cardinal_dot_product_test(
            x, xq, y, y_bar;
            tension = 0.0, extrap = NoExtrap(), deriv = EvalValue(),
            atol = 0, rtol = sqrt(eps(eltype(x)))
        )
        itp = cardinal_interp(x, y; tension = tension, extrap = extrap)
        adj = cardinal_adjoint(x, xq; tension = tension, extrap = extrap)

        Wf = itp.(xq; deriv = deriv)      # forward eval
        WTy = adj(y_bar; deriv = deriv)    # adjoint

        lhs = dot(Wf, y_bar)
        rhs = dot(y, WTy)
        return lhs, rhs, isapprox(lhs, rhs; atol = atol, rtol = rtol)
    end

    # ========================================
    # Test data setup
    # ========================================
    n_grid = 20
    n_query = 15

    x_uniform = collect(range(0.0, 1.0, n_grid))
    x_nonuniform = cumsum(0.5 .+ rand(n_grid))
    x_nonuniform .= (x_nonuniform .- x_nonuniform[1]) ./ (x_nonuniform[end] - x_nonuniform[1])

    xq = sort(rand(n_query)) .* 0.98 .+ 0.01  # inside domain with margin
    y = randn(n_grid)
    y_bar = randn(n_query)

    # ========================================
    # Type stability — constructor
    # ========================================
    @testset "Type stability — constructor @inferred" begin
        @test @inferred(cardinal_adjoint(x_uniform, xq)) isa CardinalAdjoint1D
        @test @inferred(cardinal_adjoint(x_uniform, xq; extrap = ExtendExtrap())) isa CardinalAdjoint1D
        @test @inferred(cardinal_adjoint(x_uniform, xq; tension = 0.5)) isa CardinalAdjoint1D
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = cardinal_adjoint(x_uniform, xq)
        @test size(adj) == (n_grid, n_query)
        @test size(adj, 1) == n_grid
        @test size(adj, 2) == n_query
    end

    # ========================================
    # Dot-product: NoExtrap (uniform + non-uniform)
    # ========================================
    @testset "Dot-product — NoExtrap" begin
        @testset "Uniform grid" begin
            _, _, ok = cardinal_dot_product_test(x_uniform, xq, y, y_bar)
            @test ok
        end
        @testset "Non-uniform grid" begin
            _, _, ok = cardinal_dot_product_test(x_nonuniform, xq, y, y_bar)
            @test ok
        end
    end

    # ========================================
    # Dot-product: various tension values
    # ========================================
    @testset "Dot-product — tension=$t" for t in [0.0, 0.25, 0.5, 0.75, 1.0]
        @testset "Uniform" begin
            _, _, ok = cardinal_dot_product_test(x_uniform, xq, y, y_bar; tension = t)
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = cardinal_dot_product_test(x_nonuniform, xq, y, y_bar; tension = t)
            @test ok
        end
    end

    # ========================================
    # Dot-product: extrap modes with OOB queries
    # ========================================
    xq_oob = vcat(-0.3, -0.1, sort(rand(n_query)) .* 0.98 .+ 0.01, 1.1, 1.3)
    y_bar_oob = randn(length(xq_oob))

    @testset "Dot-product — $ep_name" for (ep_name, ep) in [
            ("ExtendExtrap", ExtendExtrap()),
            ("ClampExtrap", ClampExtrap()),
            ("FillExtrap", FillExtrap(0.0)),
        ]
        @testset "Uniform" begin
            _, _, ok = cardinal_dot_product_test(
                x_uniform, xq_oob, y, y_bar_oob; extrap = ep
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = cardinal_dot_product_test(
                x_nonuniform, xq_oob, y, y_bar_oob; extrap = ep
            )
            @test ok
        end
    end

    # ========================================
    # Dot-product: WrapExtrap
    # ========================================
    @testset "Dot-product — WrapExtrap" begin
        @testset "Uniform" begin
            _, _, ok = cardinal_dot_product_test(
                x_uniform, xq_oob, y, y_bar_oob; extrap = WrapExtrap()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = cardinal_dot_product_test(
                x_nonuniform, xq_oob, y, y_bar_oob; extrap = WrapExtrap()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv1
    # ========================================
    @testset "Dot-product — EvalDeriv1" begin
        @testset "Uniform" begin
            _, _, ok = cardinal_dot_product_test(
                x_uniform, xq, y, y_bar; deriv = EvalDeriv1()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = cardinal_dot_product_test(
                x_nonuniform, xq, y, y_bar; deriv = EvalDeriv1()
            )
            @test ok
        end
    end

    # ========================================
    # Matrix materialization
    # ========================================
    @testset "Matrix(adj) -- W_T * y_bar == adj(y_bar)" begin
        adj = cardinal_adjoint(x_uniform, xq)
        W_T = Matrix(adj)
        @test size(W_T) == (n_grid, n_query)
        @test W_T * y_bar ≈ adj(y_bar)
    end

    @testset "Matrix(adj)' -- W * y == itp.(xq)" begin
        itp = cardinal_interp(x_uniform, y)
        adj = cardinal_adjoint(x_uniform, xq)
        W = Matrix(adj)'
        @test W * y ≈ itp.(xq)
    end

    @testset "Matrix -- deriv keyword" begin
        adj = cardinal_adjoint(x_uniform, xq)
        W_T = Matrix(adj; deriv = DerivOp(1))
        @test W_T * y_bar ≈ adj(y_bar; deriv = DerivOp(1))
    end

    # ========================================
    # In-place == allocating
    # ========================================
    @testset "In-place == allocating" begin
        adj = cardinal_adjoint(x_uniform, xq)
        f_bar_oop = adj(y_bar)
        f_bar_ip = zeros(n_grid)
        adj(f_bar_ip, y_bar)
        @test f_bar_oop ≈ f_bar_ip
    end

    @testset "In-place == allocating -- deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)),
        ]
        adj = cardinal_adjoint(x_uniform, xq)
        fb_oop = adj(y_bar; deriv = op)
        fb_ip = zeros(n_grid)
        adj(fb_ip, y_bar; deriv = op)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # Scalar query constructor
    # ========================================
    @testset "Scalar query constructor" begin
        adj_vec = cardinal_adjoint(x_uniform, [0.5])
        adj_scalar = cardinal_adjoint(x_uniform, 0.5)
        @test adj_vec(1.0) ≈ adj_scalar(1.0)
    end

    # ========================================
    # Edge case: minimum grid (2 points)
    # ========================================
    @testset "Minimum grid (2 points)" begin
        x_min = [0.0, 1.0]
        xq_min = [0.5]
        y_min = randn(2)
        yb_min = randn(1)
        _, _, ok = cardinal_dot_product_test(x_min, xq_min, y_min, yb_min)
        @test ok
    end

    # ========================================
    # Float32
    # ========================================
    @testset "Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        y32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        _, _, ok = cardinal_dot_product_test(x32, xq32, y32, yb32; rtol = sqrt(eps(Float32)))
        @test ok
    end

    # ========================================
    # Allocation tests (zero-alloc in-place)
    # ========================================
    function _test_cardinal_adjoint_alloc_inplace(x, xq, f_bar, y_bar; tension = 0.0, extrap = NoExtrap(), deriv = EvalValue())
        adj = cardinal_adjoint(x, xq; tension = tension, extrap = extrap)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place uniform" begin
        fb = zeros(n_grid)
        allocs = _test_cardinal_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place non-uniform" begin
        fb = zeros(n_grid)
        allocs = _test_cardinal_adjoint_alloc_inplace(x_nonuniform, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end
end

# ========================================
# PchipAdjoint1D
# ========================================

@testitem "PchipAdjoint1D" setup = [AllocConstants] begin
    using LinearAlgebra: dot

    function pchip_dot_product_test(
            x, xq, y, y_bar;
            extrap = NoExtrap(), deriv = EvalValue(),
            atol = 0, rtol = sqrt(eps(eltype(x)))
        )
        itp = pchip_interp(x, y; extrap = extrap)
        adj = pchip_adjoint(x, y, xq; extrap = extrap)

        Wf = itp.(xq; deriv = deriv)      # forward eval
        WTy = adj(y_bar; deriv = deriv)    # adjoint

        lhs = dot(Wf, y_bar)
        rhs = dot(y, WTy)
        return lhs, rhs, isapprox(lhs, rhs; atol = atol, rtol = rtol)
    end

    # ========================================
    # Test data setup
    # ========================================
    n_grid = 20
    n_query = 15

    x_uniform = collect(range(0.0, 1.0, n_grid))
    x_nonuniform = cumsum(0.5 .+ rand(n_grid))
    x_nonuniform .= (x_nonuniform .- x_nonuniform[1]) ./ (x_nonuniform[end] - x_nonuniform[1])

    xq = sort(rand(n_query)) .* 0.98 .+ 0.01  # inside domain with margin
    y = randn(n_grid)
    y_bar = randn(n_query)

    # Monotone data (ensures non-clamped interior)
    y_mono = collect(range(0.0, 3.0, n_grid)) .+ 0.1 .* randn(n_grid)
    sort!(y_mono)  # guarantee monotone

    # ========================================
    # Type stability — constructor
    # ========================================
    @testset "Type stability — constructor @inferred" begin
        @test @inferred(pchip_adjoint(x_uniform, y, xq)) isa PchipAdjoint1D
        @test @inferred(pchip_adjoint(x_uniform, y, xq; extrap = NoExtrap())) isa PchipAdjoint1D
        @test @inferred(pchip_adjoint(x_uniform, y, xq; extrap = ExtendExtrap())) isa PchipAdjoint1D
        @test @inferred(pchip_adjoint(x_uniform, y, xq; extrap = ClampExtrap())) isa PchipAdjoint1D
        @test @inferred(pchip_adjoint(x_uniform, y, xq; extrap = FillExtrap(0.0))) isa PchipAdjoint1D
        @test @inferred(pchip_adjoint(x_uniform, y, xq; extrap = WrapExtrap())) isa PchipAdjoint1D
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = pchip_adjoint(x_uniform, y, xq)
        @test size(adj) == (n_grid, n_query)
        @test size(adj, 1) == n_grid
        @test size(adj, 2) == n_query
    end

    # ========================================
    # Dot-product: NoExtrap (uniform + non-uniform)
    # ========================================
    @testset "Dot-product — NoExtrap" begin
        @testset "Uniform grid — random data" begin
            _, _, ok = pchip_dot_product_test(x_uniform, xq, y, y_bar)
            @test ok
        end
        @testset "Non-uniform grid — random data" begin
            _, _, ok = pchip_dot_product_test(x_nonuniform, xq, y, y_bar)
            @test ok
        end
        @testset "Uniform grid — monotone data" begin
            _, _, ok = pchip_dot_product_test(x_uniform, xq, y_mono, y_bar)
            @test ok
        end
        @testset "Non-uniform grid — monotone data" begin
            _, _, ok = pchip_dot_product_test(x_nonuniform, xq, y_mono, y_bar)
            @test ok
        end
    end

    # ========================================
    # Dot-product: extrap modes with OOB queries
    # ========================================
    xq_oob = vcat(-0.3, -0.1, sort(rand(n_query)) .* 0.98 .+ 0.01, 1.1, 1.3)
    y_bar_oob = randn(length(xq_oob))

    @testset "Dot-product — $ep_name" for (ep_name, ep) in [
            ("ExtendExtrap", ExtendExtrap()),
            ("ClampExtrap", ClampExtrap()),
            ("FillExtrap", FillExtrap(0.0)),
        ]
        @testset "Uniform" begin
            _, _, ok = pchip_dot_product_test(
                x_uniform, xq_oob, y, y_bar_oob; extrap = ep
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = pchip_dot_product_test(
                x_nonuniform, xq_oob, y, y_bar_oob; extrap = ep
            )
            @test ok
        end
    end

    # ========================================
    # Dot-product: WrapExtrap
    # ========================================
    @testset "Dot-product — WrapExtrap" begin
        @testset "Uniform" begin
            _, _, ok = pchip_dot_product_test(
                x_uniform, xq_oob, y, y_bar_oob; extrap = WrapExtrap()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = pchip_dot_product_test(
                x_nonuniform, xq_oob, y, y_bar_oob; extrap = WrapExtrap()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv1
    # ========================================
    @testset "Dot-product — EvalDeriv1" begin
        @testset "Uniform" begin
            _, _, ok = pchip_dot_product_test(
                x_uniform, xq, y, y_bar; deriv = EvalDeriv1()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = pchip_dot_product_test(
                x_nonuniform, xq, y, y_bar; deriv = EvalDeriv1()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv2
    # ========================================
    @testset "Dot-product — EvalDeriv2" begin
        @testset "Uniform" begin
            _, _, ok = pchip_dot_product_test(
                x_uniform, xq, y, y_bar; deriv = EvalDeriv2()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv3
    # ========================================
    @testset "Dot-product — EvalDeriv3" begin
        @testset "Uniform" begin
            _, _, ok = pchip_dot_product_test(
                x_uniform, xq, y, y_bar; deriv = EvalDeriv3()
            )
            @test ok
        end
    end

    # ========================================
    # DerivOp(4) — zero (cubic)
    # ========================================
    @testset "DerivOp(4) is zero" begin
        adj = pchip_adjoint(x_uniform, y, xq)
        result = adj(y_bar; deriv = DerivOp(4))
        @test all(iszero, result)
    end

    # ========================================
    # Matrix materialization
    # ========================================
    @testset "Matrix(adj) -- W_T * y_bar == adj(y_bar)" begin
        adj = pchip_adjoint(x_uniform, y, xq)
        W_T = Matrix(adj)
        @test size(W_T) == (n_grid, n_query)
        @test W_T * y_bar ≈ adj(y_bar)
    end

    @testset "Matrix(adj)' -- W * y == itp.(xq)" begin
        itp = pchip_interp(x_uniform, y)
        adj = pchip_adjoint(x_uniform, y, xq)
        W = Matrix(adj)'
        @test W * y ≈ itp.(xq)
    end

    @testset "Matrix -- deriv keyword" begin
        adj = pchip_adjoint(x_uniform, y, xq)
        W_T = Matrix(adj; deriv = DerivOp(1))
        @test W_T * y_bar ≈ adj(y_bar; deriv = DerivOp(1))
    end

    # ========================================
    # In-place == allocating
    # ========================================
    @testset "In-place == allocating" begin
        adj = pchip_adjoint(x_uniform, y, xq)
        f_bar_oop = adj(y_bar)
        f_bar_ip = zeros(n_grid)
        adj(f_bar_ip, y_bar)
        @test f_bar_oop ≈ f_bar_ip
    end

    @testset "In-place == allocating -- deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)),
        ]
        adj = pchip_adjoint(x_uniform, y, xq)
        fb_oop = adj(y_bar; deriv = op)
        fb_ip = zeros(n_grid)
        adj(fb_ip, y_bar; deriv = op)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # Scalar query constructor
    # ========================================
    @testset "Scalar query constructor" begin
        adj_vec = pchip_adjoint(x_uniform, y, [0.5])
        adj_scalar = pchip_adjoint(x_uniform, y, 0.5)
        @test adj_vec(1.0) ≈ adj_scalar(1.0)
    end

    # ========================================
    # Edge case: minimum grid (2 points)
    # ========================================
    @testset "Minimum grid (2 points)" begin
        x_min = [0.0, 1.0]
        y_min = randn(2)
        xq_min = [0.5]
        yb_min = randn(1)
        _, _, ok = pchip_dot_product_test(x_min, xq_min, y_min, yb_min)
        @test ok
    end

    @testset "Grid too small error" begin
        @test_throws ArgumentError pchip_adjoint([0.0], [1.0], [0.5])
    end

    @testset "NoExtrap domain error" begin
        @test_throws DomainError pchip_adjoint(x_uniform, y, [-0.1])
        @test_throws DomainError pchip_adjoint(x_uniform, y, [1.1])
    end

    # ========================================
    # Float32
    # ========================================
    @testset "Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        y32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        _, _, ok = pchip_dot_product_test(x32, xq32, y32, yb32; rtol = sqrt(eps(Float32)))
        @test ok
    end

    @testset "Float32 — type stability" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        y32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        adj32 = pchip_adjoint(x32, y32, xq32)
        @test @inferred(adj32(yb32)) isa Vector{Float32}
    end

    # ========================================
    # Allocation tests (zero-alloc in-place)
    # ========================================
    function _test_pchip_adjoint_alloc_inplace(x, y, xq, f_bar, y_bar; extrap = NoExtrap(), deriv = EvalValue())
        adj = pchip_adjoint(x, y, xq; extrap = extrap)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place uniform" begin
        fb = zeros(n_grid)
        allocs = _test_pchip_adjoint_alloc_inplace(x_uniform, y, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place non-uniform" begin
        fb = zeros(n_grid)
        allocs = _test_pchip_adjoint_alloc_inplace(x_nonuniform, y, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # Stress test: many random datasets (dot-product identity)
    # ========================================
    @testset "Stress: 10 random datasets" begin
        for _ in 1:10
            n = rand(5:30)
            nq = rand(3:20)
            xr = sort(vcat(0.0, rand(n - 2), 1.0))
            xqr = sort(rand(nq)) .* 0.98 .+ 0.01
            yr = randn(n)
            ybr = randn(nq)
            _, _, ok = pchip_dot_product_test(xr, xqr, yr, ybr)
            @test ok
        end
    end
end

# ========================================
# AkimaAdjoint1D
# ========================================

@testitem "AkimaAdjoint1D" setup = [AllocConstants] begin
    using LinearAlgebra: dot

    function akima_dot_product_test(
            x, xq, y, y_bar;
            extrap = NoExtrap(), deriv = EvalValue(),
            atol = 0, rtol = sqrt(eps(eltype(x)))
        )
        itp = akima_interp(x, y; extrap = extrap)
        adj = akima_adjoint(x, y, xq; extrap = extrap)

        Wf = itp.(xq; deriv = deriv)      # forward eval
        WTy = adj(y_bar; deriv = deriv)    # adjoint

        lhs = dot(Wf, y_bar)
        rhs = dot(y, WTy)
        return lhs, rhs, isapprox(lhs, rhs; atol = atol, rtol = rtol)
    end

    # ========================================
    # Test data setup
    # ========================================
    n_grid = 20
    n_query = 15

    x_uniform = collect(range(0.0, 1.0, n_grid))
    x_nonuniform = cumsum(0.5 .+ rand(n_grid))
    x_nonuniform .= (x_nonuniform .- x_nonuniform[1]) ./ (x_nonuniform[end] - x_nonuniform[1])

    xq = sort(rand(n_query)) .* 0.98 .+ 0.01  # inside domain with margin
    y = randn(n_grid)
    y_bar = randn(n_query)

    # ========================================
    # Type stability — constructor
    # ========================================
    @testset "Type stability — constructor @inferred" begin
        @test @inferred(akima_adjoint(x_uniform, y, xq)) isa AkimaAdjoint1D
        @test @inferred(akima_adjoint(x_uniform, y, xq; extrap = NoExtrap())) isa AkimaAdjoint1D
        @test @inferred(akima_adjoint(x_uniform, y, xq; extrap = ExtendExtrap())) isa AkimaAdjoint1D
        @test @inferred(akima_adjoint(x_uniform, y, xq; extrap = ClampExtrap())) isa AkimaAdjoint1D
        @test @inferred(akima_adjoint(x_uniform, y, xq; extrap = FillExtrap(0.0))) isa AkimaAdjoint1D
        @test @inferred(akima_adjoint(x_uniform, y, xq; extrap = WrapExtrap())) isa AkimaAdjoint1D
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = akima_adjoint(x_uniform, y, xq)
        @test size(adj) == (n_grid, n_query)
        @test size(adj, 1) == n_grid
        @test size(adj, 2) == n_query
    end

    # ========================================
    # Dot-product: NoExtrap (uniform + non-uniform)
    # ========================================
    @testset "Dot-product — NoExtrap" begin
        @testset "Uniform grid" begin
            _, _, ok = akima_dot_product_test(x_uniform, xq, y, y_bar)
            @test ok
        end
        @testset "Non-uniform grid" begin
            _, _, ok = akima_dot_product_test(x_nonuniform, xq, y, y_bar)
            @test ok
        end
    end

    # ========================================
    # Dot-product: extrap modes with OOB queries
    # ========================================
    xq_oob = vcat(-0.3, -0.1, sort(rand(n_query)) .* 0.98 .+ 0.01, 1.1, 1.3)
    y_bar_oob = randn(length(xq_oob))

    @testset "Dot-product — $ep_name" for (ep_name, ep) in [
            ("ExtendExtrap", ExtendExtrap()),
            ("ClampExtrap", ClampExtrap()),
            ("FillExtrap", FillExtrap(0.0)),
        ]
        @testset "Uniform" begin
            _, _, ok = akima_dot_product_test(
                x_uniform, xq_oob, y, y_bar_oob; extrap = ep
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = akima_dot_product_test(
                x_nonuniform, xq_oob, y, y_bar_oob; extrap = ep
            )
            @test ok
        end
    end

    # ========================================
    # Dot-product: WrapExtrap
    # ========================================
    @testset "Dot-product — WrapExtrap" begin
        @testset "Uniform" begin
            _, _, ok = akima_dot_product_test(
                x_uniform, xq_oob, y, y_bar_oob; extrap = WrapExtrap()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = akima_dot_product_test(
                x_nonuniform, xq_oob, y, y_bar_oob; extrap = WrapExtrap()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv1
    # ========================================
    @testset "Dot-product — EvalDeriv1" begin
        @testset "Uniform" begin
            _, _, ok = akima_dot_product_test(
                x_uniform, xq, y, y_bar; deriv = EvalDeriv1()
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = akima_dot_product_test(
                x_nonuniform, xq, y, y_bar; deriv = EvalDeriv1()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv2
    # ========================================
    @testset "Dot-product — EvalDeriv2" begin
        @testset "Uniform" begin
            _, _, ok = akima_dot_product_test(
                x_uniform, xq, y, y_bar; deriv = EvalDeriv2()
            )
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint: EvalDeriv3
    # ========================================
    @testset "Dot-product — EvalDeriv3" begin
        @testset "Uniform" begin
            _, _, ok = akima_dot_product_test(
                x_uniform, xq, y, y_bar; deriv = EvalDeriv3()
            )
            @test ok
        end
    end

    # ========================================
    # DerivOp(4) — zero (cubic)
    # ========================================
    @testset "DerivOp(4) is zero" begin
        adj = akima_adjoint(x_uniform, y, xq)
        result = adj(y_bar; deriv = DerivOp(4))
        @test all(iszero, result)
    end

    # ========================================
    # Matrix materialization
    # ========================================
    @testset "Matrix(adj) -- W_T * y_bar == adj(y_bar)" begin
        adj = akima_adjoint(x_uniform, y, xq)
        W_T = Matrix(adj)
        @test size(W_T) == (n_grid, n_query)
        @test W_T * y_bar ≈ adj(y_bar)
    end

    @testset "Matrix(adj)' -- W * y == itp.(xq)" begin
        itp = akima_interp(x_uniform, y)
        adj = akima_adjoint(x_uniform, y, xq)
        W = Matrix(adj)'
        @test W * y ≈ itp.(xq)
    end

    @testset "Matrix -- deriv keyword" begin
        adj = akima_adjoint(x_uniform, y, xq)
        W_T = Matrix(adj; deriv = DerivOp(1))
        @test W_T * y_bar ≈ adj(y_bar; deriv = DerivOp(1))
    end

    # ========================================
    # In-place == allocating
    # ========================================
    @testset "In-place == allocating" begin
        adj = akima_adjoint(x_uniform, y, xq)
        f_bar_oop = adj(y_bar)
        f_bar_ip = zeros(n_grid)
        adj(f_bar_ip, y_bar)
        @test f_bar_oop ≈ f_bar_ip
    end

    @testset "In-place == allocating -- deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)),
        ]
        adj = akima_adjoint(x_uniform, y, xq)
        fb_oop = adj(y_bar; deriv = op)
        fb_ip = zeros(n_grid)
        adj(fb_ip, y_bar; deriv = op)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # Scalar query constructor
    # ========================================
    @testset "Scalar query constructor" begin
        adj_vec = akima_adjoint(x_uniform, y, [0.5])
        adj_scalar = akima_adjoint(x_uniform, y, 0.5)
        @test adj_vec(1.0) ≈ adj_scalar(1.0)
    end

    # ========================================
    # Edge case: minimum grid (2 points)
    # ========================================
    @testset "Minimum grid (2 points)" begin
        x_min = [0.0, 1.0]
        y_min = randn(2)
        xq_min = [0.5]
        yb_min = randn(1)
        _, _, ok = akima_dot_product_test(x_min, xq_min, y_min, yb_min)
        @test ok
    end

    # ========================================
    # Edge case: 3 points (special branch)
    # ========================================
    @testset "Minimum grid (3 points)" begin
        x3 = [0.0, 0.5, 1.0]
        y3 = randn(3)
        xq3 = [0.25, 0.75]
        yb3 = randn(2)
        _, _, ok = akima_dot_product_test(x3, xq3, y3, yb3)
        @test ok
    end

    # ========================================
    # Edge case: 4 points (boundary of general case)
    # ========================================
    @testset "4 points" begin
        x4 = [0.0, 0.3, 0.7, 1.0]
        y4 = randn(4)
        xq4 = [0.15, 0.5, 0.85]
        yb4 = randn(3)
        _, _, ok = akima_dot_product_test(x4, xq4, y4, yb4)
        @test ok
    end

    @testset "Grid too small error" begin
        @test_throws ArgumentError akima_adjoint([0.0], [1.0], [0.5])
    end

    @testset "NoExtrap domain error" begin
        @test_throws DomainError akima_adjoint(x_uniform, y, [-0.1])
        @test_throws DomainError akima_adjoint(x_uniform, y, [1.1])
    end

    # ========================================
    # Float32
    # ========================================
    @testset "Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        y32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        _, _, ok = akima_dot_product_test(x32, xq32, y32, yb32; rtol = sqrt(eps(Float32)))
        @test ok
    end

    @testset "Float32 — type stability" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        y32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        adj32 = akima_adjoint(x32, y32, xq32)
        @test @inferred(adj32(yb32)) isa Vector{Float32}
    end

    # ========================================
    # Allocation tests (zero-alloc in-place)
    # ========================================
    function _test_akima_adjoint_alloc_inplace(x, y, xq, f_bar, y_bar; extrap = NoExtrap(), deriv = EvalValue())
        adj = akima_adjoint(x, y, xq; extrap = extrap)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place uniform" begin
        fb = zeros(n_grid)
        allocs = _test_akima_adjoint_alloc_inplace(x_uniform, y, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place non-uniform" begin
        fb = zeros(n_grid)
        allocs = _test_akima_adjoint_alloc_inplace(x_nonuniform, y, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # Stress test: many random datasets (dot-product identity)
    # ========================================
    @testset "Stress: 10 random datasets" begin
        for _ in 1:10
            n = rand(5:30)
            nq = rand(3:20)
            xr = sort(vcat(0.0, rand(n - 2), 1.0))
            xqr = sort(rand(nq)) .* 0.98 .+ 0.01
            yr = randn(n)
            ybr = randn(nq)
            _, _, ok = akima_dot_product_test(xr, xqr, yr, ybr)
            @test ok
        end
    end
end

# ========================================
# Hermite Family — integrate
# ========================================

@testitem "Hermite Family — integrate" setup = [AllocConstants] begin
    using LinearAlgebra: dot


    # Known integral: Hermite is exact for cubics with exact slopes
    @testset "Exact cubic polynomial — Hermite" begin
        x = collect(range(0.0, 1.0, 20))
        y = x .^ 3
        dy = 3 .* x .^ 2
        itp = hermite_interp(x, y, dy)
        @test integrate(itp, 0.0, 1.0) ≈ 0.25 rtol = 1.0e-12
    end

    # Auto-slope methods approximate cubics
    @testset "Approximate cubic — $method" for (method, make_itp) in [
            ("PCHIP", (x, y) -> pchip_interp(x, y)),
            ("Cardinal", (x, y) -> cardinal_interp(x, y)),
            ("Akima", (x, y) -> akima_interp(x, y)),
        ]
        x = collect(range(0.0, 1.0, 20))
        y = x .^ 3
        itp = make_itp(x, y)
        @test integrate(itp, 0.0, 1.0) ≈ 0.25 rtol = 1.0e-2
    end

    # Sign reversal: ∫_a^b = -∫_b^a
    @testset "Sign reversal — $T" for (T, make_itp) in [
            ("Hermite", (x, y, dy) -> hermite_interp(x, y, dy)),
            ("PCHIP", (x, y, _) -> pchip_interp(x, y)),
            ("Cardinal", (x, y, _) -> cardinal_interp(x, y)),
            ("Akima", (x, y, _) -> akima_interp(x, y)),
        ]
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)
        itp = make_itp(x, y, dy)
        @test integrate(itp, 0.2, 0.8) ≈ -integrate(itp, 0.8, 0.2) rtol = eps()
    end

    # Additivity: ∫_a^b + ∫_b^c = ∫_a^c
    @testset "Additivity — $T" for (T, make_itp) in [
            ("Hermite", (x, y, dy) -> hermite_interp(x, y, dy)),
            ("PCHIP", (x, y, _) -> pchip_interp(x, y)),
            ("Cardinal", (x, y, _) -> cardinal_interp(x, y)),
            ("Akima", (x, y, _) -> akima_interp(x, y)),
        ]
        x = collect(range(0.0, 1.0, 15))
        y = sin.(x)
        dy = cos.(x)
        itp = make_itp(x, y, dy)
        I_ab = integrate(itp, 0.1, 0.5)
        I_bc = integrate(itp, 0.5, 0.9)
        I_ac = integrate(itp, 0.1, 0.9)
        @test I_ab + I_bc ≈ I_ac rtol = sqrt(eps())
    end

    # Partial cell
    @testset "Partial cell — $T" for (T, make_itp) in [
            ("Hermite", (x, y, dy) -> hermite_interp(x, y, dy)),
            ("PCHIP", (x, y, _) -> pchip_interp(x, y)),
        ]
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)
        itp = make_itp(x, y, dy)
        I_full = integrate(itp, 0.0, 1.0)
        I_part = integrate(itp, 0.3, 0.7)
        @test abs(I_part) < abs(I_full)
    end

    # Compare with known: ∫₀¹ sin(x) dx = 1 - cos(1) ≈ 0.4597
    @testset "sin(x) integral accuracy — $T" for (T, make_itp) in [
            ("Hermite", (x, y, dy) -> hermite_interp(x, y, dy)),
            ("PCHIP", (x, y, _) -> pchip_interp(x, y)),
            ("Cardinal", (x, y, _) -> cardinal_interp(x, y)),
            ("Akima", (x, y, _) -> akima_interp(x, y)),
        ]
        x = collect(range(0.0, 1.0, 50))
        y = sin.(x)
        dy = cos.(x)
        itp = make_itp(x, y, dy)
        exact = 1 - cos(1.0)
        @test integrate(itp, 0.0, 1.0) ≈ exact rtol = 1.0e-6
    end

    # Float32 inputs — kernel promotes to Float64 via _IH* literal constants
    @testset "Float32 inputs — Hermite" begin
        x = Float32.(collect(range(0.0f0, 1.0f0, 10)))
        y = sin.(x)
        dy = cos.(x)
        itp = hermite_interp(x, y, dy)
        result = integrate(itp, 0.0f0, 1.0f0)
        @test result ≈ (1 - cos(1.0)) rtol = 1.0e-3
    end

    # ── Akima equal-weight fallback coverage ──────────────────────────
    # Constant secants → all Δm = 0 → wsum = 0 → fallback to 50/50 average.
    # This exercises the otherwise-uncovered equal-weight branch in _akima_slope_adjoint!

    @testset "Akima adjoint — equal-weight fallback (wsum=0)" begin
        # Linear data: constant secants → wsum = 0 at every point
        x = collect(range(0.0, 5.0, 10))
        y = 2.0 .* x .+ 1.0   # perfectly linear → all secants equal
        xq = sort(rand(8) .* 4.0 .+ 0.5)
        y_bar = randn(length(xq))

        adj = akima_adjoint(x, y, xq)
        itp = akima_interp(x, y)

        Wf = [itp(q) for q in xq]
        WTy = adj(y_bar)

        @test dot(Wf, y_bar) ≈ dot(y, WTy) rtol = sqrt(eps())
    end

    @testset "Akima adjoint — equal-weight fallback deriv=1" begin
        x = collect(range(0.0, 5.0, 10))
        y = 2.0 .* x .+ 1.0
        xq = sort(rand(8) .* 4.0 .+ 0.5)
        y_bar = randn(length(xq))

        adj = akima_adjoint(x, y, xq)
        itp = akima_interp(x, y)

        Wf = [itp(q; deriv = DerivOp(1)) for q in xq]
        WTy = adj(y_bar; deriv = DerivOp(1))

        @test dot(Wf, y_bar) ≈ dot(y, WTy) rtol = sqrt(eps())
    end
end
