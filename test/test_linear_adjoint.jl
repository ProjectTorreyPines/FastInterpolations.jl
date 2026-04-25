@testitem "LinearAdjoint" setup=[AllocConstants] begin
    using LinearAlgebra: dot

    # ========================================
    # Helper: Dot-product test for adjoint correctness
    # ========================================
    # The gold standard: ⟨W·f, ȳ⟩ = ⟨f, Wᵀ·ȳ⟩
    # Linear interp is purely linear in f (no BC constant offset), so no subtraction needed.

    function linear_dot_product_test(
            x, xq, f, y_bar;
            extrap = NoExtrap(), deriv = EvalValue(),
            atol = 0, rtol = sqrt(eps(eltype(x)))
        )
        itp = linear_interp(x, f; extrap = extrap)
        adj = linear_adjoint(x, xq; extrap = extrap)

        Wf = itp.(xq; deriv = deriv)
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
    # Dot-product tests: basic
    # ========================================
    @testset "Dot-product — $grid_name" for (grid_name, grid) in [
            ("Uniform", x_uniform),
            ("Non-uniform", x_nonuniform),
        ]
        lhs, rhs, ok = linear_dot_product_test(grid, xq, f, y_bar)
        @test ok
    end

    # ========================================
    # Dot-product tests: extrap modes with OOB queries
    # ========================================
    xq_oob = vcat(-0.3, -0.1, sort(rand(n_query)) .* 0.98 .+ 0.01, 1.1, 1.3)
    y_bar_oob = randn(length(xq_oob))

    @testset "Dot-product — OOB queries — ExtendExtrap" begin
        @testset "Uniform" begin
            _, _, ok = linear_dot_product_test(x_uniform, xq_oob, f, y_bar_oob; extrap = ExtendExtrap())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = linear_dot_product_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = ExtendExtrap())
            @test ok
        end
    end

    @testset "Dot-product — OOB queries — ClampExtrap" begin
        @testset "Uniform" begin
            _, _, ok = linear_dot_product_test(x_uniform, xq_oob, f, y_bar_oob; extrap = ClampExtrap())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = linear_dot_product_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = ClampExtrap())
            @test ok
        end
    end

    @testset "Dot-product — OOB queries — WrapExtrap" begin
        @testset "Uniform" begin
            _, _, ok = linear_dot_product_test(x_uniform, xq_oob, f, y_bar_oob; extrap = WrapExtrap())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = linear_dot_product_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = WrapExtrap())
            @test ok
        end
    end

    @testset "Dot-product — OOB queries — FillExtrap(0.0)" begin
        @testset "Uniform" begin
            _, _, ok = linear_dot_product_test(x_uniform, xq_oob, f, y_bar_oob; extrap = FillExtrap(0.0))
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = linear_dot_product_test(x_nonuniform, xq_oob, f, y_bar_oob; extrap = FillExtrap(0.0))
            @test ok
        end
    end

    # ========================================
    # Derivative adjoint
    # ========================================
    @testset "Derivative adjoint — deriv=1" begin
        @testset "Uniform" begin
            _, _, ok = linear_dot_product_test(x_uniform, xq, f, y_bar; deriv = EvalDeriv1())
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = linear_dot_product_test(x_nonuniform, xq, f, y_bar; deriv = EvalDeriv1())
            @test ok
        end
    end

    @testset "Derivative adjoint — deriv=2 returns zero" begin
        adj = linear_adjoint(x_uniform, xq)
        f_bar = adj(y_bar; deriv = EvalDeriv2())
        @test all(iszero, f_bar)
    end

    @testset "Derivative adjoint — deriv=3 returns zero" begin
        adj = linear_adjoint(x_uniform, xq)
        f_bar = adj(y_bar; deriv = EvalDeriv3())
        @test all(iszero, f_bar)
    end

    @testset "Derivative adjoint — deriv=1 with OOB" begin
        @testset "ExtendExtrap" begin
            _, _, ok = linear_dot_product_test(
                x_uniform, xq_oob, f, y_bar_oob;
                extrap = ExtendExtrap(), deriv = EvalDeriv1()
            )
            @test ok
        end
        @testset "ClampExtrap" begin
            _, _, ok = linear_dot_product_test(
                x_uniform, xq_oob, f, y_bar_oob;
                extrap = ClampExtrap(), deriv = EvalDeriv1()
            )
            @test ok
        end
        @testset "FillExtrap" begin
            _, _, ok = linear_dot_product_test(
                x_uniform, xq_oob, f, y_bar_oob;
                extrap = FillExtrap(0.0), deriv = EvalDeriv1()
            )
            @test ok
        end
    end

    # ========================================
    # In-place vs allocating
    # ========================================
    @testset "In-place == allocating" begin
        adj = linear_adjoint(x_uniform, xq)
        f_bar_oop = adj(y_bar)
        f_bar_ip = zeros(n_grid)
        adj(f_bar_ip, y_bar)
        @test f_bar_oop ≈ f_bar_ip
    end

    @testset "In-place == allocating — deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        adj = linear_adjoint(x_uniform, xq)
        fb_oop = adj(y_bar; deriv = op)
        fb_ip = zeros(n_grid)
        adj(fb_ip, y_bar; deriv = op)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = linear_adjoint(x_uniform, xq)
        @test size(adj) == (n_grid, n_query)
        @test size(adj, 1) == n_grid
        @test size(adj, 2) == n_query
    end

    # ========================================
    # Type stability
    # ========================================
    @testset "Type stability — constructor @inferred" begin
        @test @inferred(linear_adjoint(x_uniform, xq)) isa LinearAdjoint
        @test @inferred(linear_adjoint(x_uniform, xq; extrap = NoExtrap())) isa LinearAdjoint
        @test @inferred(linear_adjoint(x_uniform, xq; extrap = ExtendExtrap())) isa LinearAdjoint
        @test @inferred(linear_adjoint(x_uniform, xq; extrap = ClampExtrap())) isa LinearAdjoint
        @test @inferred(linear_adjoint(x_uniform, xq; extrap = WrapExtrap())) isa LinearAdjoint
        @test @inferred(linear_adjoint(x_uniform, xq; extrap = FillExtrap(0.0))) isa LinearAdjoint
    end

    @testset "Type stability — apply Float64" begin
        adj = linear_adjoint(x_uniform, xq)
        @test @inferred(adj(y_bar)) isa Vector{Float64}
    end

    @testset "Type stability — apply Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        yb32 = randn(Float32, n_query)
        adj32 = linear_adjoint(x32, xq32)
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
        lhs, rhs, ok = linear_dot_product_test(x32, xq32, f32, yb32; rtol = sqrt(eps(Float32)))
        @test ok
    end

    @testset "Float32 — deriv=1" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        f32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        _, _, ok = linear_dot_product_test(
            x32, xq32, f32, yb32;
            deriv = EvalDeriv1(), rtol = sqrt(eps(Float32))
        )
        @test ok
    end

    # ========================================
    # Complex data support
    # ========================================
    @testset "Complex — dot-product" begin
        f_c = randn(ComplexF64, n_grid)
        yb_c = randn(ComplexF64, n_query)
        @testset "Uniform" begin
            lhs, rhs, ok = linear_dot_product_test(x_uniform, xq, f_c, yb_c)
            @test ok
        end
        @testset "Non-uniform" begin
            lhs, rhs, ok = linear_dot_product_test(x_nonuniform, xq, f_c, yb_c)
            @test ok
        end
    end

    @testset "Complex — in-place == allocating" begin
        adj = linear_adjoint(x_uniform, xq)
        yb_c = randn(ComplexF64, n_query)
        fb_oop = adj(yb_c)
        fb_ip = zeros(ComplexF64, n_grid)
        adj(fb_ip, yb_c)
        @test fb_oop ≈ fb_ip
    end

    @testset "Complex — derivative adjoint" begin
        f_c = randn(ComplexF64, n_grid)
        yb_c = randn(ComplexF64, n_query)
        _, _, ok = linear_dot_product_test(x_uniform, xq, f_c, yb_c; deriv = EvalDeriv1())
        @test ok
    end

    # ========================================
    # Edge cases: minimum grid
    # ========================================
    @testset "Minimum grid (2 points)" begin
        x_min = [0.0, 1.0]
        xq_min = [0.5]
        f_min = randn(2)
        yb_min = randn(1)
        lhs, rhs, ok = linear_dot_product_test(x_min, xq_min, f_min, yb_min)
        @test ok
    end

    # ========================================
    # Scalar / Tuple y_bar + scalar query constructor
    # ========================================
    @testset "Scalar y_bar (1 query point)" begin
        xq_single = [0.3]
        adj = linear_adjoint(x_uniform, xq_single)
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
        xq3 = [0.2, 0.5, 0.8]
        adj = linear_adjoint(x_uniform, xq3)
        ref = adj([1.0, 2.0, 3.0])
        @test adj((1.0, 2.0, 3.0)) ≈ ref

        # In-place
        f_bar = zeros(n_grid)
        adj(f_bar, (1.0, 2.0, 3.0))
        @test f_bar ≈ ref
    end

    @testset "Scalar y_bar with deriv" begin
        xq_single = [0.4]
        adj = linear_adjoint(x_uniform, xq_single)
        ref = adj([1.0]; deriv = DerivOp(1))
        @test adj(1.0; deriv = DerivOp(1)) ≈ ref
    end

    @testset "Scalar query constructor" begin
        adj_vec = linear_adjoint(x_uniform, [0.5])
        adj_scalar = linear_adjoint(x_uniform, 0.5)
        @test adj_vec(1.0) ≈ adj_scalar(1.0)
    end

    @testset "Dimension mismatch errors" begin
        adj = linear_adjoint(x_uniform, [0.2, 0.5])  # 2 queries
        @test_throws DimensionMismatch adj(1.5)         # scalar but 2 queries
        @test_throws DimensionMismatch adj((1.0,))       # 1-tuple but 2 queries
    end

    # ========================================
    # Matrix materialization
    # ========================================
    @testset "Matrix(adj) — Wᵀ·ȳ == adj(ȳ)" begin
        adj = linear_adjoint(x_uniform, xq)
        W_T = Matrix(adj)
        @test size(W_T) == (n_grid, n_query)
        @test W_T * y_bar ≈ adj(y_bar)
    end

    @testset "Matrix(adj)' — W·f == itp.(xq)" begin
        itp = linear_interp(x_uniform, f)
        adj = linear_adjoint(x_uniform, xq)
        W = Matrix(adj)'
        @test W * f ≈ itp.(xq)
    end

    @testset "Matrix — deriv keyword" begin
        adj = linear_adjoint(x_uniform, xq)
        itp = linear_interp(x_uniform, f)
        W_T = Matrix(adj; deriv = EvalDeriv1())
        @test W_T * y_bar ≈ adj(y_bar; deriv = EvalDeriv1())
        @test Matrix(adj; deriv = EvalDeriv1())' * f ≈ itp.(xq; deriv = EvalDeriv1())
    end

    @testset "Matrix — non-uniform grid" begin
        adj = linear_adjoint(x_nonuniform, xq)
        itp = linear_interp(x_nonuniform, f)
        W_T = Matrix(adj)
        @test W_T * y_bar ≈ adj(y_bar)
        @test Matrix(adj)' * f ≈ itp.(xq)
    end

    @testset "Matrix — Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        adj32 = linear_adjoint(x32, xq32)
        W_T = Matrix(adj32)
        @test eltype(W_T) == Float32
        @test size(W_T) == (n_grid, n_query)
    end

    # ========================================
    # Allocation tests (zero-alloc in-place)
    # ========================================

    function _test_linear_adjoint_alloc_inplace(x, xq, f_bar, y_bar; extrap = NoExtrap(), deriv = EvalValue())
        adj = linear_adjoint(x, xq; extrap = extrap)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place uniform" begin
        fb = zeros(n_grid)
        allocs = _test_linear_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place non-uniform" begin
        fb = zeros(n_grid)
        allocs = _test_linear_adjoint_alloc_inplace(x_nonuniform, xq, fb, y_bar)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        fb = zeros(n_grid)
        allocs = _test_linear_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar; deriv = op)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        fb32 = zeros(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        allocs = _test_linear_adjoint_alloc_inplace(x32, xq32, fb32, yb32)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # NoExtrap throws for OOB queries
    # ========================================
    @testset "NoExtrap — OOB throws DomainError" begin
        @test_throws DomainError linear_adjoint(x_uniform, [-0.1, 0.5])
        @test_throws DomainError linear_adjoint(x_uniform, [0.5, 1.1])
    end

    # ========================================
    # Singleton grid rejection
    # ========================================
    @testset "Singleton grid throws ArgumentError" begin
        @test_throws ArgumentError linear_adjoint([0.0], [0.0])
    end

    # ========================================
    # Analytical value tests (hand-computed expected values)
    # ========================================
    # Linear adjoint weights are trivial to compute by hand.
    # These tests verify exact values, not just the dot-product identity.

    @testset "Analytical — single midpoint query" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [0.5])  # α = 0.5 in interval [0,1]
        @test adj([1.0]) ≈ [0.5, 0.5, 0.0]
        @test adj([1.0]; deriv = EvalDeriv1()) ≈ [-1.0, 1.0, 0.0]
    end

    @testset "Analytical — two queries, accumulation" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [0.5, 1.5])
        # q1: α=0.5 in [0,1] → f_bar[1]+=0.5*2, f_bar[2]+=0.5*2
        # q2: α=0.5 in [1,2] → f_bar[2]+=0.5*3, f_bar[3]+=0.5*3
        @test adj([2.0, 3.0]) ≈ [1.0, 2.5, 1.5]
    end

    @testset "Analytical — query at grid node" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [1.0])
        # At grid node x[2]: all weight on node 2 regardless of which interval
        @test adj([1.0]) ≈ [0.0, 1.0, 0.0]
    end

    @testset "Analytical — non-uniform grid" begin
        x = [0.0, 0.3, 1.0]  # h1=0.3, h2=0.7
        adj = linear_adjoint(x, [0.15])  # α = 0.15/0.3 = 0.5
        @test adj([1.0]) ≈ [0.5, 0.5, 0.0]
        # Deriv: inv_h = 1/0.3
        @test adj([1.0]; deriv = EvalDeriv1()) ≈ [-1.0 / 0.3, 1.0 / 0.3, 0.0]
    end

    @testset "Analytical — non-uniform deriv in second interval" begin
        x = [0.0, 0.5, 2.0]  # h2 = 1.5, inv_h2 = 1/1.5
        adj = linear_adjoint(x, [1.0])  # in interval [0.5, 2.0], α = 0.5/1.5 = 1/3
        @test adj([1.0]) ≈ [0.0, 2 / 3, 1 / 3]
        @test adj([1.0]; deriv = EvalDeriv1()) ≈ [0.0, -1.0 / 1.5, 1.0 / 1.5]
    end

    @testset "Analytical — EvalDeriv2/3 always zero" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [0.5])
        @test adj([1.0]; deriv = EvalDeriv2()) == [0.0, 0.0, 0.0]
        @test adj([1.0]; deriv = EvalDeriv3()) == [0.0, 0.0, 0.0]
    end

    # ── Extrap-specific analytical tests ──

    @testset "Analytical — ExtendExtrap (OOB below)" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [-0.5]; extrap = ExtendExtrap())
        # Uses first interval [0,1], α = (-0.5 - 0)/1 = -0.5
        # f_bar[1] += (1-(-0.5))*1 = 1.5, f_bar[2] += (-0.5)*1 = -0.5
        @test adj([1.0]) ≈ [1.5, -0.5, 0.0]
    end

    @testset "Analytical — ExtendExtrap (OOB above)" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [2.7]; extrap = ExtendExtrap())
        # Uses last interval [1,2], α = (2.7 - 1)/1 = 1.7
        # f_bar[2] += (1-1.7)*1 = -0.7, f_bar[3] += 1.7*1 = 1.7
        @test adj([1.0]) ≈ [0.0, -0.7, 1.7]
    end

    @testset "Analytical — FillExtrap (OOB queries contribute nothing)" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [-0.5, 0.5, 2.5]; extrap = FillExtrap(0.0))
        # Only q2 (0.5) contributes. q1 and q3 are OOB → skipped.
        # q2: α=0.5 in [0,1], y_bar=2.0 → f_bar[1]+=1.0, f_bar[2]+=1.0
        @test adj([1.0, 2.0, 3.0]) ≈ [1.0, 1.0, 0.0]
    end

    @testset "Analytical — FillExtrap deriv (OOB queries contribute nothing)" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [-0.5, 0.5, 2.5]; extrap = FillExtrap(0.0))
        # Only q2 contributes deriv. inv_h=1.
        # f_bar[1] += -1*2 = -2, f_bar[2] += 1*2 = 2
        @test adj([1.0, 2.0, 3.0]; deriv = EvalDeriv1()) ≈ [-2.0, 2.0, 0.0]
    end

    @testset "Analytical — ClampExtrap value (OOB at boundary)" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [-0.5, 0.5, 2.5]; extrap = ClampExtrap())
        # q1: clamped to 0.0, α=0 → f_bar[1] += 1*1.0 = 1.0
        # q2: α=0.5 → f_bar[1] += 0.5*2 = 1.0, f_bar[2] += 0.5*2 = 1.0
        # q3: clamped to 2.0, α=1 in [1,2] → f_bar[3] += 1*3.0 = 3.0
        @test adj([1.0, 2.0, 3.0]) ≈ [2.0, 1.0, 3.0]
    end

    @testset "Analytical — ClampExtrap deriv (OOB → zero derivative)" begin
        x = [0.0, 1.0, 2.0]
        adj = linear_adjoint(x, [-0.5, 0.5, 2.5]; extrap = ClampExtrap())
        # OOB deriv → skipped for ClampExtrap. Only q2 contributes.
        # q2: inv_h=1, f_bar[1] += -1*2 = -2, f_bar[2] += 1*2 = 2
        @test adj([1.0, 2.0, 3.0]; deriv = EvalDeriv1()) ≈ [-2.0, 2.0, 0.0]
    end

    @testset "Analytical — WrapExtrap (wraps to domain)" begin
        x = [0.0, 1.0, 2.0]  # domain [0, 2], period = 2
        adj = linear_adjoint(x, [2.5]; extrap = WrapExtrap())
        # 2.5 wraps to 0.5 → α=0.5 in [0,1]
        @test adj([1.0]) ≈ [0.5, 0.5, 0.0]
    end

    @testset "Analytical — WrapExtrap (negative wrap)" begin
        x = [0.0, 1.0, 2.0]  # domain [0, 2], period = 2
        adj = linear_adjoint(x, [-0.5]; extrap = WrapExtrap())
        # -0.5 wraps to 1.5 → α=0.5 in [1,2]
        @test adj([1.0]) ≈ [0.0, 0.5, 0.5]
    end
end
