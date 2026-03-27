using Test
using LinearAlgebra: dot
using FastInterpolations

# ========================================
# Helper: Dot-product test for adjoint correctness
# ========================================
# The gold standard: ⟨W·f, ȳ⟩ = ⟨f, Wᵀ·ȳ⟩
# Constant interp is purely linear in f (no BC constant offset), so no subtraction needed.

function constant_dot_product_test(
        x, xq, f, y_bar;
        side = NearestSide(), extrap = NoExtrap(), deriv = EvalValue(),
        atol = 0, rtol = sqrt(eps(eltype(x)))
    )
    itp = constant_interp(x, f; side = side, extrap = extrap)
    adj = constant_adjoint(x, xq; side = side, extrap = extrap)

    Wf = itp.(xq; deriv = deriv)
    WTy = adj(y_bar; deriv = deriv)

    lhs = dot(Wf, y_bar)
    rhs = dot(f, WTy)
    return lhs, rhs, isapprox(lhs, rhs; atol = atol, rtol = rtol)
end

@testset "ConstantAdjoint" begin
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
    # Dot-product tests: basic (all side modes)
    # ========================================
    @testset "Dot-product — $grid_name — $side_name" for (grid_name, grid) in [
                ("Uniform", x_uniform),
                ("Non-uniform", x_nonuniform),
            ],
            (side_name, side_mode) in [
                ("NearestSide", NearestSide()),
                ("LeftSide", LeftSide()),
                ("RightSide", RightSide()),
            ]

        _, _, ok = constant_dot_product_test(grid, xq, f, y_bar; side = side_mode)
        @test ok
    end

    # ========================================
    # Dot-product tests: extrap modes with OOB queries
    # ========================================
    xq_oob = vcat(-0.3, -0.1, sort(rand(n_query)) .* 0.98 .+ 0.01, 1.1, 1.3)
    y_bar_oob = randn(length(xq_oob))

    @testset "Dot-product — OOB — $ext_name — $side_name" for (ext_name, ext) in [
                ("ExtendExtrap", ExtendExtrap()),
                ("ClampExtrap", ClampExtrap()),
                ("WrapExtrap", WrapExtrap()),
                ("FillExtrap", FillExtrap(0.0)),
            ],
            (side_name, side_mode) in [
                ("NearestSide", NearestSide()),
                ("LeftSide", LeftSide()),
                ("RightSide", RightSide()),
            ]

        _, _, ok = constant_dot_product_test(
            x_uniform, xq_oob, f, y_bar_oob;
            side = side_mode, extrap = ext
        )
        @test ok
    end

    # ========================================
    # Derivative adjoint — all derivatives are zero
    # ========================================
    @testset "Derivative adjoint — deriv=$d returns zero" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        adj = constant_adjoint(x_uniform, xq)
        f_bar = adj(y_bar; deriv = op)
        @test all(iszero, f_bar)
    end

    @testset "Derivative dot-product — deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        # All derivatives of constant interp are zero, so both sides should be zero
        _, _, ok = constant_dot_product_test(x_uniform, xq, f, y_bar; deriv = op, atol = 1.0e-15)
        @test ok
    end

    # ========================================
    # In-place vs allocating
    # ========================================
    @testset "In-place == allocating — $side_name" for (side_name, side_mode) in [
            ("NearestSide", NearestSide()),
            ("LeftSide", LeftSide()),
            ("RightSide", RightSide()),
        ]
        adj = constant_adjoint(x_uniform, xq; side = side_mode)
        f_bar_oop = adj(y_bar)
        f_bar_ip = zeros(n_grid)
        adj(f_bar_ip, y_bar)
        @test f_bar_oop ≈ f_bar_ip
    end

    @testset "In-place == allocating — deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        adj = constant_adjoint(x_uniform, xq)
        fb_oop = adj(y_bar; deriv = op)
        fb_ip = zeros(n_grid)
        adj(fb_ip, y_bar; deriv = op)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = constant_adjoint(x_uniform, xq)
        @test size(adj) == (n_grid, n_query)
        @test size(adj, 1) == n_grid
        @test size(adj, 2) == n_query
    end

    # ========================================
    # Type stability
    # ========================================
    @testset "Type stability — constructor @inferred" begin
        @test @inferred(constant_adjoint(x_uniform, xq)) isa ConstantAdjoint
        @test @inferred(constant_adjoint(x_uniform, xq; side = NearestSide())) isa ConstantAdjoint
        @test @inferred(constant_adjoint(x_uniform, xq; side = LeftSide())) isa ConstantAdjoint
        @test @inferred(constant_adjoint(x_uniform, xq; side = RightSide())) isa ConstantAdjoint
        @test @inferred(constant_adjoint(x_uniform, xq; extrap = ExtendExtrap())) isa ConstantAdjoint
        @test @inferred(constant_adjoint(x_uniform, xq; extrap = ClampExtrap())) isa ConstantAdjoint
        @test @inferred(constant_adjoint(x_uniform, xq; extrap = WrapExtrap())) isa ConstantAdjoint
        @test @inferred(constant_adjoint(x_uniform, xq; extrap = FillExtrap(0.0))) isa ConstantAdjoint
    end

    @testset "Type stability — apply Float64" begin
        adj = constant_adjoint(x_uniform, xq)
        @test @inferred(adj(y_bar)) isa Vector{Float64}
    end

    @testset "Type stability — apply Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        yb32 = randn(Float32, n_query)
        adj32 = constant_adjoint(x32, xq32)
        @test @inferred(adj32(yb32)) isa Vector{Float32}
    end

    # ========================================
    # Float32 support
    # ========================================
    @testset "Float32 — $side_name" for (side_name, side_mode) in [
            ("NearestSide", NearestSide()),
            ("LeftSide", LeftSide()),
            ("RightSide", RightSide()),
        ]
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        f32 = randn(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        _, _, ok = constant_dot_product_test(
            x32, xq32, f32, yb32;
            side = side_mode, rtol = sqrt(eps(Float32))
        )
        @test ok
    end

    # ========================================
    # Complex data support
    # ========================================
    @testset "Complex — dot-product" begin
        f_c = randn(ComplexF64, n_grid)
        yb_c = randn(ComplexF64, n_query)
        @testset "$grid_name" for (grid_name, grid) in [
                ("Uniform", x_uniform),
                ("Non-uniform", x_nonuniform),
            ]
            _, _, ok = constant_dot_product_test(grid, xq, f_c, yb_c)
            @test ok
        end
    end

    @testset "Complex — in-place == allocating" begin
        adj = constant_adjoint(x_uniform, xq)
        yb_c = randn(ComplexF64, n_query)
        fb_oop = adj(yb_c)
        fb_ip = zeros(ComplexF64, n_grid)
        adj(fb_ip, yb_c)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # Edge cases: minimum grid
    # ========================================
    @testset "Minimum grid (2 points)" begin
        x_min = [0.0, 1.0]
        xq_min = [0.5]
        f_min = randn(2)
        yb_min = randn(1)
        _, _, ok = constant_dot_product_test(x_min, xq_min, f_min, yb_min)
        @test ok
    end

    # ========================================
    # Scalar / Tuple y_bar + scalar query constructor
    # ========================================
    @testset "Scalar y_bar (1 query point)" begin
        xq_single = [0.3]
        adj = constant_adjoint(x_uniform, xq_single)
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
        adj = constant_adjoint(x_uniform, xq3)
        ref = adj([1.0, 2.0, 3.0])
        @test adj((1.0, 2.0, 3.0)) ≈ ref

        # In-place
        f_bar = zeros(n_grid)
        adj(f_bar, (1.0, 2.0, 3.0))
        @test f_bar ≈ ref
    end

    @testset "Scalar query constructor" begin
        adj_vec = constant_adjoint(x_uniform, [0.5])
        adj_scalar = constant_adjoint(x_uniform, 0.5)
        @test adj_vec(1.0) ≈ adj_scalar(1.0)
    end

    @testset "Dimension mismatch errors" begin
        adj = constant_adjoint(x_uniform, [0.2, 0.5])  # 2 queries
        @test_throws DimensionMismatch adj(1.5)         # scalar but 2 queries
        @test_throws DimensionMismatch adj((1.0,))       # 1-tuple but 2 queries
    end

    # ========================================
    # Matrix materialization
    # ========================================
    @testset "Matrix(adj) — Wᵀ·ȳ == adj(ȳ) — $side_name" for (side_name, side_mode) in [
            ("NearestSide", NearestSide()),
            ("LeftSide", LeftSide()),
            ("RightSide", RightSide()),
        ]
        adj = constant_adjoint(x_uniform, xq; side = side_mode)
        W_T = Matrix(adj)
        @test size(W_T) == (n_grid, n_query)
        @test W_T * y_bar ≈ adj(y_bar)
    end

    @testset "Matrix(adj)' — W·f == itp.(xq) — $side_name" for (side_name, side_mode) in [
            ("NearestSide", NearestSide()),
            ("LeftSide", LeftSide()),
            ("RightSide", RightSide()),
        ]
        itp = constant_interp(x_uniform, f; side = side_mode)
        adj = constant_adjoint(x_uniform, xq; side = side_mode)
        W = Matrix(adj)'
        @test W * f ≈ itp.(xq)
    end

    @testset "Matrix — Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        adj32 = constant_adjoint(x32, xq32)
        W_T = Matrix(adj32)
        @test eltype(W_T) == Float32
        @test size(W_T) == (n_grid, n_query)
    end

    # ========================================
    # Allocation tests (zero-alloc in-place)
    # ========================================
    function _test_constant_adjoint_alloc_inplace(
            x, xq, f_bar, y_bar;
            side = NearestSide(), extrap = NoExtrap(), deriv = EvalValue()
        )
        adj = constant_adjoint(x, xq; side = side, extrap = extrap)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place — $side_name" for (side_name, side_mode) in [
            ("NearestSide", NearestSide()),
            ("LeftSide", LeftSide()),
            ("RightSide", RightSide()),
        ]
        fb = zeros(n_grid)
        allocs = _test_constant_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar; side = side_mode)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place deriv=$d" for (d, op) in [
            (1, DerivOp(1)), (2, DerivOp(2)), (3, DerivOp(3)), (4, DerivOp(4)),
        ]
        fb = zeros(n_grid)
        allocs = _test_constant_adjoint_alloc_inplace(x_uniform, xq, fb, y_bar; deriv = op)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Float32" begin
        x32 = Float32.(x_uniform)
        xq32 = Float32.(xq)
        fb32 = zeros(Float32, n_grid)
        yb32 = randn(Float32, n_query)
        allocs = _test_constant_adjoint_alloc_inplace(x32, xq32, fb32, yb32)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # NoExtrap throws for OOB queries
    # ========================================
    @testset "NoExtrap — OOB throws DomainError" begin
        @test_throws DomainError constant_adjoint(x_uniform, [-0.1, 0.5])
        @test_throws DomainError constant_adjoint(x_uniform, [0.5, 1.1])
    end

    # ========================================
    # Singleton grid rejection
    # ========================================
    @testset "Singleton grid throws ArgumentError" begin
        @test_throws ArgumentError constant_adjoint([0.0], [0.0])
    end

    # ========================================
    # Analytical value tests (hand-computed expected values)
    # ========================================

    @testset "Analytical — LeftSide (always selects left value)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [0.7]; side = LeftSide())
        # LeftSide: offset=0 → scatter to idx=1 (left boundary of interval [0,1])
        @test adj([1.0]) ≈ [1.0, 0.0, 0.0]
    end

    @testset "Analytical — RightSide (dL != 0 → right value)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [0.7]; side = RightSide())
        # RightSide: dL=0.7 != 0 → offset=1 → scatter to idx=2
        @test adj([1.0]) ≈ [0.0, 1.0, 0.0]
    end

    @testset "Analytical — RightSide at grid node (dL == 0 → left value)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [1.0]; side = RightSide())
        # At grid node x[2]=1.0: idx=1, dL=0 → offset=0 → scatter to f_bar[1]?
        # Actually at grid node 1.0: idx=1, dL=1.0-0.0=1.0? No, search gives idx=1 with xL=1.0
        # search_interval(1.0) in [0,1,2] → idx=1, xL=1.0... this depends on search behavior
        # At grid node, forward returns y[2] (the value AT node 1.0).
        # We verify via dot-product instead:
        f_test = [10.0, 20.0, 30.0]
        itp = constant_interp(x, f_test; side = RightSide())
        adj2 = constant_adjoint(x, [1.0]; side = RightSide())
        @test dot(itp.([1.0]), [1.0]) ≈ dot(f_test, adj2([1.0]))
    end

    @testset "Analytical — NearestSide (dL > h/2 → right value)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [0.7]; side = NearestSide())
        # NearestSide: dL=0.7, h=1.0, dL > h/2=0.5 → offset=1 → scatter to idx=2
        @test adj([1.0]) ≈ [0.0, 1.0, 0.0]
    end

    @testset "Analytical — NearestSide at midpoint (dL == h/2 → left tie-break)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [0.5]; side = NearestSide())
        # NearestSide: dL=0.5, h=1.0, dL <= h/2 → offset=0 → scatter to idx=1
        @test adj([1.0]) ≈ [1.0, 0.0, 0.0]
    end

    @testset "Analytical — NearestSide (dL < h/2 → left value)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [0.3]; side = NearestSide())
        # NearestSide: dL=0.3, h=1.0, dL <= h/2=0.5 → offset=0 → scatter to idx=1
        @test adj([1.0]) ≈ [1.0, 0.0, 0.0]
    end

    @testset "Analytical — right boundary (xq == x_max)" begin
        x = [0.0, 1.0, 2.0]
        @testset "$side_name" for (side_name, side_mode) in [
                ("NearestSide", NearestSide()),
                ("LeftSide", LeftSide()),
                ("RightSide", RightSide()),
            ]
            adj = constant_adjoint(x, [2.0]; side = side_mode)
            # At right boundary, all side modes should scatter to f_bar[end]
            @test adj([1.0]) ≈ [0.0, 0.0, 1.0]
        end
    end

    @testset "Analytical — two queries, accumulation (NearestSide)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [0.3, 1.7]; side = NearestSide())
        # q1: dL=0.3 <= 0.5 → offset=0 → f_bar[1] += 2.0
        # q2: dL=0.7 > 0.5 → offset=1 → f_bar[3] += 3.0
        @test adj([2.0, 3.0]) ≈ [2.0, 0.0, 3.0]
    end

    @testset "Analytical — all derivatives are zero" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [0.5])
        @test adj([1.0]; deriv = EvalDeriv1()) == [0.0, 0.0, 0.0]
        @test adj([1.0]; deriv = EvalDeriv2()) == [0.0, 0.0, 0.0]
        @test adj([1.0]; deriv = EvalDeriv3()) == [0.0, 0.0, 0.0]
    end

    # ── Extrap-specific analytical tests ──

    @testset "Analytical — ExtendExtrap (OOB below)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [-0.5]; side = NearestSide(), extrap = ExtendExtrap())
        # OOB below: forward returns y[1] (extrap delegates to ClampExtrap)
        @test adj([1.0]) ≈ [1.0, 0.0, 0.0]
    end

    @testset "Analytical — ExtendExtrap (OOB above)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [2.5]; side = LeftSide(), extrap = ExtendExtrap())
        # OOB above: forward returns y[end] regardless of side → f_bar[3]
        @test adj([1.0]) ≈ [0.0, 0.0, 1.0]
    end

    @testset "Analytical — FillExtrap (OOB queries contribute nothing)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [-0.5, 0.7, 2.5]; side = NearestSide(), extrap = FillExtrap(0.0))
        # Only q2 (0.7) contributes. q1 and q3 are OOB → skipped.
        # q2: dL=0.7 > 0.5 → offset=1 → f_bar[2] += 2.0
        @test adj([1.0, 2.0, 3.0]) ≈ [0.0, 2.0, 0.0]
    end

    @testset "Analytical — ClampExtrap value (OOB at boundary)" begin
        x = [0.0, 1.0, 2.0]
        adj = constant_adjoint(x, [-0.5, 0.3, 2.5]; side = NearestSide(), extrap = ClampExtrap())
        # q1: clamped to 0.0, dL=0 → offset=0 → f_bar[1] += 1.0
        # q2: dL=0.3 <= 0.5 → offset=0 → f_bar[1] += 2.0
        # q3: clamped to 2.0 → right boundary → f_bar[3] += 3.0
        @test adj([1.0, 2.0, 3.0]) ≈ [3.0, 0.0, 3.0]
    end

    @testset "Analytical — WrapExtrap (wraps to domain)" begin
        x = [0.0, 1.0, 2.0]  # domain [0, 2], period = 2
        adj = constant_adjoint(x, [2.3]; side = NearestSide(), extrap = WrapExtrap())
        # 2.3 wraps to 0.3 → dL=0.3 <= 0.5 → offset=0 → f_bar[1] += 1.0
        @test adj([1.0]) ≈ [1.0, 0.0, 0.0]
    end

    @testset "Analytical — WrapExtrap (negative wrap)" begin
        x = [0.0, 1.0, 2.0]  # domain [0, 2], period = 2
        adj = constant_adjoint(x, [-0.3]; side = NearestSide(), extrap = WrapExtrap())
        # -0.3 wraps to 1.7 → in interval [1,2], dL=0.7 > 0.5 → offset=1 → f_bar[3] += 1.0
        @test adj([1.0]) ≈ [0.0, 0.0, 1.0]
    end
end
