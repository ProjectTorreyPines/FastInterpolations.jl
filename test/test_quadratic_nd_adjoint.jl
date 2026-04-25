@testitem "QuadraticAdjointND (N=2)" setup = [AllocConstants] begin
    using LinearAlgebra: dot
    using StaticArrays: SVector

    # ========================================
    # Helper: ND Dot-product test for quadratic adjoint
    # ========================================
    # Gold standard: ⟨W·f, ȳ⟩ = ⟨f, Wᵀ·ȳ⟩
    # W is affine (non-zero BC values add a constant offset), so we
    # subtract the zero-data response to isolate the linear part.

    function quadratic_nd_dot_product_test(
            grids, xqs, f, y_bar;
            bc = Left(QuadraticFit()),
            extrap = NoExtrap(),
            deriv = EvalValue(),
            rtol = sqrt(eps(eltype(grids[1])))
        )
        itp = quadratic_interp(grids, f; bc = bc, extrap = extrap)
        adj = quadratic_adjoint(grids, xqs; bc = bc, extrap = extrap)

        n_queries = length(xqs[1])

        # Forward: W·f (subtract constant offset for affine BCs)
        f_zero = zeros(eltype(f), size(f))
        itp_zero = quadratic_interp(grids, f_zero; bc = bc, extrap = extrap)
        Wf = Vector{eltype(f)}(undef, n_queries)
        Wf_zero = Vector{eltype(f)}(undef, n_queries)
        itp(Wf, xqs; deriv = deriv)
        itp_zero(Wf_zero, xqs; deriv = deriv)
        Wf .-= Wf_zero  # linear part only

        # Adjoint: Wᵀ·ȳ
        WTy = adj(y_bar; deriv = deriv)

        lhs = dot(Wf, y_bar)
        rhs = dot(vec(f), vec(WTy))
        return lhs, rhs, isapprox(lhs, rhs; rtol = rtol)
    end


    # ========================================
    # Test data setup
    # ========================================
    nx, ny = 15, 12
    n_query = 40

    x_uniform = range(0.0, 1.0, nx)
    y_uniform = range(0.0, 2.0, ny)

    x_nonuniform = cumsum(0.5 .+ rand(nx))
    x_nonuniform .= (x_nonuniform .- x_nonuniform[1]) ./ (x_nonuniform[end] - x_nonuniform[1])
    y_nonuniform = cumsum(0.5 .+ rand(ny))
    y_nonuniform .= (y_nonuniform .- y_nonuniform[1]) ./ (y_nonuniform[end] - y_nonuniform[1]) .* 2.0

    # Query points inside domain
    xq = sort(rand(n_query)) .* 0.96 .+ 0.02
    yq = sort(rand(n_query)) .* 1.92 .+ 0.04

    f = randn(nx, ny)
    y_bar = randn(n_query)

    # ========================================
    # Dot-product identity tests — grid types
    # ========================================
    @testset "Dot-product — QuadraticFit — $grid_name" for (grid_name, grids) in [
            ("Uniform (Range × Range)", (x_uniform, y_uniform)),
            ("Non-uniform (Vector × Vector)", (collect(x_nonuniform), collect(y_nonuniform))),
            ("Mixed (Range × Vector)", (x_uniform, collect(y_nonuniform))),
        ]

        f_g = randn(nx, ny)
        _, _, ok = quadratic_nd_dot_product_test(grids, (xq, yq), f_g, y_bar)
        @test ok
    end

    # ========================================
    # Various grid sizes
    # ========================================
    @testset "Dot-product — grid sizes" begin
        for (gx, gy, label) in [
                (5, 5, "5x5"),
                (10, 15, "10x15"),
                (20, 20, "20x20"),
            ]
            @testset "$label" begin
                x_g = range(0.0, 1.0, gx)
                y_g = range(0.0, 1.0, gy)
                f_g = randn(gx, gy)
                xq_g = sort(rand(20)) .* 0.96 .+ 0.02
                yq_g = sort(rand(20)) .* 0.96 .+ 0.02
                yb_g = randn(20)
                _, _, ok = quadratic_nd_dot_product_test((x_g, y_g), (xq_g, yq_g), f_g, yb_g)
                @test ok
            end
        end
    end

    # ========================================
    # Dot-product — all BC types
    # ========================================
    @testset "Dot-product — $bc_name" for (bc_name, bc) in [
            ("Left(QuadraticFit)", Left(QuadraticFit())),
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

        f_bc = randn(nx, ny)
        @testset "Uniform" begin
            _, _, ok = quadratic_nd_dot_product_test(
                (x_uniform, y_uniform), (xq, yq), f_bc, y_bar; bc = bc
            )
            @test ok
        end
        @testset "Non-uniform" begin
            _, _, ok = quadratic_nd_dot_product_test(
                (collect(x_nonuniform), collect(y_nonuniform)), (xq, yq), f_bc, y_bar; bc = bc
            )
            @test ok
        end
    end

    # ========================================
    # Per-axis BC
    # ========================================
    @testset "Dot-product — per-axis BC" begin
        f_pa = randn(nx, ny)
        _, _, ok = quadratic_nd_dot_product_test(
            (x_uniform, y_uniform), (xq, yq), f_pa, y_bar;
            bc = (Left(QuadraticFit()), Right(Deriv2(0.0)))
        )
        @test ok
    end

    @testset "Dot-product — per-axis BC (MinCurvFit × Deriv1)" begin
        f_pa2 = randn(nx, ny)
        _, _, ok = quadratic_nd_dot_product_test(
            (x_uniform, y_uniform), (xq, yq), f_pa2, y_bar;
            bc = (MinCurvFit(), Left(Deriv1(0.0)))
        )
        @test ok
    end

    # ========================================
    # OOB extrap dot-product tests
    # ========================================
    xq_oob = vcat(-0.3 .+ 0.1 .* rand(10), xq, 1.0 .+ 0.1 .* rand(10))
    yq_oob = vcat(-0.5 .+ 0.2 .* rand(10), yq, 2.0 .+ 0.2 .* rand(10))
    y_bar_oob = randn(length(xq_oob))
    f_oob = randn(nx, ny)

    @testset "Dot-product — OOB — $ext_name" for (ext_name, ext) in [
            ("ExtendExtrap", ExtendExtrap()),
            ("ClampExtrap", ClampExtrap()),
            ("FillExtrap", FillExtrap(0.0)),
            ("WrapExtrap", WrapExtrap()),
        ]

        _, _, ok = quadratic_nd_dot_product_test(
            (x_uniform, y_uniform), (xq_oob, yq_oob), f_oob, y_bar_oob;
            extrap = ext
        )
        @test ok
    end

    # ========================================
    # Derivative adjoint — per-axis deriv
    # ========================================
    @testset "Derivative adjoint — $d_name" for (d_name, deriv_op) in [
            ("deriv=(1,0)", (EvalDeriv1(), EvalValue())),
            ("deriv=(0,1)", (EvalValue(), EvalDeriv1())),
            ("deriv=(1,1)", (EvalDeriv1(), EvalDeriv1())),
            ("deriv=(2,0)", (EvalDeriv2(), EvalValue())),
            ("deriv=(0,2)", (EvalValue(), EvalDeriv2())),
        ]

        f_d = randn(nx, ny)
        _, _, ok = quadratic_nd_dot_product_test(
            (x_uniform, y_uniform), (xq, yq), f_d, y_bar; deriv = deriv_op
        )
        @test ok
    end

    @testset "Derivative adjoint — non-uniform — deriv=(1,0)" begin
        f_d = randn(nx, ny)
        _, _, ok = quadratic_nd_dot_product_test(
            (collect(x_nonuniform), collect(y_nonuniform)), (xq, yq), f_d, y_bar;
            deriv = (EvalDeriv1(), EvalValue())
        )
        @test ok
    end

    @testset "Derivative adjoint — deriv=(1,0) — $bc_name" for (bc_name, bc) in [
            ("Left(QuadraticFit)", Left(QuadraticFit())),
            ("Right(Deriv2(0.0))", Right(Deriv2(0.0))),
            ("MinCurvFit", MinCurvFit()),
        ]

        f_d = randn(nx, ny)
        _, _, ok = quadratic_nd_dot_product_test(
            (x_uniform, y_uniform), (xq, yq), f_d, y_bar;
            bc = bc, deriv = (EvalDeriv1(), EvalValue())
        )
        @test ok
    end

    # ========================================
    # EvalDeriv3 → zero (quadratic has no 3rd derivative)
    # ========================================
    @testset "Derivative adjoint — deriv with EvalDeriv3 → zero" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))

        @testset "deriv=(3,0)" begin
            f_bar = adj(y_bar; deriv = (EvalDeriv3(), EvalValue()))
            @test all(iszero, f_bar)
        end
        @testset "deriv=(0,3)" begin
            f_bar = adj(y_bar; deriv = (EvalValue(), EvalDeriv3()))
            @test all(iszero, f_bar)
        end
        @testset "deriv=(3,3)" begin
            f_bar = adj(y_bar; deriv = (EvalDeriv3(), EvalDeriv3()))
            @test all(iszero, f_bar)
        end
    end

    # ========================================
    # In-place vs allocating
    # ========================================
    @testset "In-place vs allocating" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))
        f_bar_alloc = adj(y_bar)
        f_bar_inplace = zeros(nx, ny)
        adj(f_bar_inplace, y_bar)
        @test f_bar_alloc ≈ f_bar_inplace
    end

    @testset "In-place vs allocating — deriv=(1,0)" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))
        f_bar_alloc = adj(y_bar; deriv = (EvalDeriv1(), EvalValue()))
        f_bar_inplace = zeros(nx, ny)
        adj(f_bar_inplace, y_bar; deriv = (EvalDeriv1(), EvalValue()))
        @test f_bar_alloc ≈ f_bar_inplace
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))
        @test size(adj) == (nx, ny, n_query)
        @test size(adj, 1) == nx
        @test size(adj, 2) == ny
        @test size(adj, 3) == n_query
    end

    # ========================================
    # Type stability
    # ========================================
    @testset "Type stability — constructor @inferred" begin
        @test @inferred(quadratic_adjoint((x_uniform, y_uniform), (xq, yq))) isa QuadraticAdjointND
        @test @inferred(
            quadratic_adjoint(
                (x_uniform, y_uniform), (xq, yq);
                bc = Left(QuadraticFit())
            )
        ) isa QuadraticAdjointND
        @test @inferred(
            quadratic_adjoint(
                (x_uniform, y_uniform), (xq, yq);
                bc = MinCurvFit()
            )
        ) isa QuadraticAdjointND
    end

    @testset "Type stability — apply Float64" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))
        @test @inferred(adj(y_bar)) isa Matrix{Float64}
    end

    @testset "Type stability — apply Float32" begin
        x32 = range(0.0f0, 1.0f0, nx)
        y32 = range(0.0f0, 2.0f0, ny)
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        yb32 = randn(Float32, n_query)
        adj32 = quadratic_adjoint((x32, y32), (xq32, yq32))
        @test @inferred(adj32(yb32)) isa Matrix{Float32}
    end

    # ========================================
    # Float32
    # ========================================
    @testset "Dot-product — Float32" begin
        x32 = range(0.0f0, 1.0f0, nx)
        y32 = range(0.0f0, 2.0f0, ny)
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        f32 = randn(Float32, nx, ny)
        yb32 = randn(Float32, n_query)
        _, _, ok = quadratic_nd_dot_product_test(
            (x32, y32), (xq32, yq32), f32, yb32;
            rtol = sqrt(eps(Float32))
        )
        @test ok
    end

    # ========================================
    # Matrix materialization
    # ========================================
    @testset "Matrix materialization" begin
        sx, sy = 5, 4
        x_s = range(0.0, 1.0, sx)
        y_s = range(0.0, 1.0, sy)
        n_q = 8
        xq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        yq_s = sort(rand(n_q)) .* 0.96 .+ 0.02

        adj = quadratic_adjoint((x_s, y_s), (xq_s, yq_s))

        # Build Wᵀ by probing adjoint with unit vectors
        WT = zeros(sx * sy, n_q)
        e_q = zeros(n_q)
        for q in 1:n_q
            e_q[q] = 1.0
            WT[:, q] = vec(adj(e_q))
            e_q[q] = 0.0
        end

        # Test Matrix() function directly
        WT_mat = Matrix(adj)
        @test WT_mat ≈ WT rtol = 1.0e-12
    end

    # ========================================
    # Matrix materialization — with deriv (T4)
    # ========================================
    @testset "Matrix materialization — deriv=(1,0)" begin
        sx, sy = 5, 4
        x_s = range(0.0, 1.0, sx)
        y_s = range(0.0, 1.0, sy)
        n_q = 8
        xq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        yq_s = sort(rand(n_q)) .* 0.96 .+ 0.02

        adj = quadratic_adjoint((x_s, y_s), (xq_s, yq_s))
        deriv_ops = (EvalDeriv1(), EvalValue())

        # Build Wᵀ by probing adjoint with unit vectors
        WT = zeros(sx * sy, n_q)
        e_q = zeros(n_q)
        for q in 1:n_q
            e_q[q] = 1.0
            WT[:, q] = vec(adj(e_q; deriv = deriv_ops))
            e_q[q] = 0.0
        end

        WT_mat = Matrix(adj; deriv = deriv_ops)
        @test WT_mat ≈ WT rtol = 1.0e-12
    end

    @testset "Matrix materialization — deriv=(0,2)" begin
        sx, sy = 5, 4
        x_s = range(0.0, 1.0, sx)
        y_s = range(0.0, 1.0, sy)
        n_q = 8
        xq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        yq_s = sort(rand(n_q)) .* 0.96 .+ 0.02

        adj = quadratic_adjoint((x_s, y_s), (xq_s, yq_s))
        deriv_ops = (EvalValue(), EvalDeriv2())

        WT = zeros(sx * sy, n_q)
        e_q = zeros(n_q)
        for q in 1:n_q
            e_q[q] = 1.0
            WT[:, q] = vec(adj(e_q; deriv = deriv_ops))
            e_q[q] = 0.0
        end

        WT_mat = Matrix(adj; deriv = deriv_ops)
        @test WT_mat ≈ WT rtol = 1.0e-12
    end

    # ========================================
    # Scalar / Tuple y_bar — ND (T2)
    # ========================================
    @testset "Scalar y_bar (single query)" begin
        adj_1 = quadratic_adjoint((x_uniform, y_uniform), ([0.5], [1.0]))
        ref = adj_1([1.0])
        @test adj_1(1.0) ≈ ref

        # In-place scalar
        fb = zeros(nx, ny)
        adj_1(fb, 1.0)
        @test fb ≈ ref
    end

    @testset "DimensionMismatch — scalar y_bar with multiple queries" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))
        @test_throws DimensionMismatch adj(1.0)
    end

    @testset "Mismatched query-length constructor" begin
        @test_throws DimensionMismatch quadratic_adjoint(
            (x_uniform, y_uniform), (xq, randn(n_query + 1))
        )
    end

    # ========================================
    # Edge cases
    # ========================================
    @testset "Single query point" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), ([0.5], [1.0]))
        f_bar = adj([1.0])
        @test size(f_bar) == (nx, ny)

        # Scalar y_bar
        f_bar_scalar = adj(1.0)
        @test f_bar_scalar ≈ f_bar

        # In-place scalar
        f_bar_ip = zeros(nx, ny)
        adj(f_bar_ip, 1.0)
        @test f_bar_ip ≈ f_bar
    end

    @testset "Query at grid nodes — dot-product" begin
        xq_nodes = collect(x_uniform)[2:(end - 1)]
        yq_nodes = collect(y_uniform)[2:min(end - 1, length(xq_nodes) + 1)]
        n_both = min(length(xq_nodes), length(yq_nodes))
        xq_n = xq_nodes[1:n_both]
        yq_n = yq_nodes[1:n_both]
        yb_n = randn(n_both)
        f_n = randn(nx, ny)
        _, _, ok = quadratic_nd_dot_product_test(
            (x_uniform, y_uniform), (xq_n, yq_n), f_n, yb_n
        )
        @test ok
    end

    # ========================================
    # SVector queries
    # ========================================
    @testset "SVector query — scalar" begin
        sv = SVector(0.5, 1.0)
        adj = quadratic_adjoint((x_uniform, y_uniform), sv)
        f_bar = adj(1.0)
        @test size(f_bar) == (nx, ny)

        adj2 = quadratic_adjoint((x_uniform, y_uniform), ([0.5], [1.0]))
        @test f_bar ≈ adj2(1.0)
    end

    @testset "SVector query — vector of SVector" begin
        svecs = [SVector(xq[i], yq[i]) for i in 1:n_query]
        adj_sv = quadratic_adjoint((x_uniform, y_uniform), svecs)
        adj_soa = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))
        @test adj_sv(y_bar) ≈ adj_soa(y_bar)
    end

    # ========================================
    # Single-tuple query constructor
    # ========================================
    @testset "Single-tuple query constructor" begin
        adj_soa = quadratic_adjoint((x_uniform, y_uniform), ([0.5], [1.0]))
        adj_tup = quadratic_adjoint((x_uniform, y_uniform), (0.5, 1.0))
        @test adj_soa(1.0) ≈ adj_tup(1.0)
        @test adj_soa([1.0]) ≈ adj_tup([1.0])
    end

    # ========================================
    # Singleton grid rejection
    # ========================================
    @testset "Singleton grid throws" begin
        @test_throws Exception quadratic_adjoint(([0.0], [0.0, 1.0, 2.0]), ([0.0], [0.5]))
        @test_throws Exception quadratic_adjoint(([0.0, 1.0, 2.0], [0.0]), ([0.5], [0.0]))
    end

    # ========================================
    # Mutation safety — grid mutation after construction
    # ========================================
    @testset "Mutation safety — grid mutation" begin
        x = collect(range(0.0, 1.0, nx))
        y = collect(range(0.0, 2.0, ny))
        adj = quadratic_adjoint((x, y), (xq, yq))
        result_before = adj(y_bar)
        x[5] = 100.0
        y[4] = 100.0
        result_after = adj(y_bar)
        @test result_before == result_after
    end

    # ========================================
    # Complex data support
    # ========================================
    @testset "Complex — dot-product" begin
        f_c = randn(ComplexF64, nx, ny)
        yb_c = randn(ComplexF64, n_query)
        _, _, ok = quadratic_nd_dot_product_test(
            (x_uniform, y_uniform), (xq, yq), f_c, yb_c
        )
        @test ok
    end

    @testset "Complex — in-place == allocating" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))
        yb_c = randn(ComplexF64, n_query)
        fb_oop = adj(yb_c)
        fb_ip = zeros(ComplexF64, nx, ny)
        adj(fb_ip, yb_c)
        @test fb_oop ≈ fb_ip
    end

    # ========================================
    # Mixed-precision y_bar (T6)
    # ========================================
    @testset "Mixed-precision — Float32 y_bar with Float64 adjoint" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))
        yb32 = randn(Float32, n_query)

        # Allocating
        f_bar = adj(yb32)
        @test eltype(f_bar) == Float64
        @test size(f_bar) == (nx, ny)

        # Dot-product correctness
        f_test = randn(nx, ny)
        _, _, ok = quadratic_nd_dot_product_test(
            (x_uniform, y_uniform), (xq, yq), f_test, yb32
        )
        @test ok
    end

    # ========================================
    # Dimension validation
    # ========================================
    @testset "Dimension mismatch errors" begin
        adj = quadratic_adjoint((x_uniform, y_uniform), (xq, yq))
        @test_throws DimensionMismatch adj(randn(n_query + 1))
        @test_throws DimensionMismatch adj(zeros(nx + 1, ny), y_bar)
    end

    # ========================================
    # NoExtrap — OOB throws DomainError
    # ========================================
    @testset "NoExtrap — OOB throws DomainError" begin
        @test_throws DomainError quadratic_adjoint(
            (x_uniform, y_uniform), ([-0.1, 0.5], [0.5, 0.5])
        )
        @test_throws DomainError quadratic_adjoint(
            (x_uniform, y_uniform), ([0.5, 1.1], [0.5, 0.5])
        )
    end

    # ========================================
    # Zero-allocation tests
    # ========================================
    function _test_quadratic_nd_adjoint_alloc_inplace(
            grids, queries, f_bar, y_bar;
            bc = Left(QuadraticFit()), extrap = NoExtrap(), deriv = EvalValue()
        )
        adj = quadratic_adjoint(grids, queries; bc = bc, extrap = extrap)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place — $bc_name" for (bc_name, bc) in [
            ("Left(QuadraticFit)", Left(QuadraticFit())),
            ("Right(QuadraticFit)", Right(QuadraticFit())),
            ("MinCurvFit", MinCurvFit()),
        ]
        fb = zeros(nx, ny)
        allocs = _test_quadratic_nd_adjoint_alloc_inplace(
            (x_uniform, y_uniform), (xq, yq), fb, y_bar; bc = bc
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end

    # ========================================
    # Zero-alloc: derivative operations (T3)
    # ========================================
    function _test_quadratic_nd_adjoint_alloc_deriv(grids, queries, f_bar, y_bar, deriv)
        adj = quadratic_adjoint(grids, queries)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place deriv=1" begin
        fb = zeros(nx, ny)
        allocs = _test_quadratic_nd_adjoint_alloc_deriv(
            (x_uniform, y_uniform), (xq, yq), fb, y_bar, DerivOp(1)
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place mixed deriv" begin
        fb = zeros(nx, ny)
        allocs = _test_quadratic_nd_adjoint_alloc_deriv(
            (x_uniform, y_uniform), (xq, yq), fb, y_bar, (DerivOp(1), EvalValue())
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Float32" begin
        x32 = Float32.(collect(x_uniform))
        y32 = Float32.(collect(y_uniform))
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        fb32 = zeros(Float32, nx, ny)
        yb32 = randn(Float32, n_query)
        allocs = _test_quadratic_nd_adjoint_alloc_inplace(
            (x32, y32), (xq32, yq32), fb32, yb32
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end
end

# ========================================
# N=3 Tests (generic ntuple path)
# ========================================
@testitem "QuadraticAdjointND (N=3)" setup = [AllocConstants] begin
    using LinearAlgebra: dot
    using StaticArrays: SVector

    # Helper: ND Dot-product test for quadratic adjoint (subtracts BC offset).
    function quadratic_nd_dot_product_test(
            grids, xqs, f, y_bar;
            bc = Left(QuadraticFit()),
            extrap = NoExtrap(),
            deriv = EvalValue(),
            rtol = sqrt(eps(eltype(grids[1])))
        )
        itp = quadratic_interp(grids, f; bc = bc, extrap = extrap)
        adj = quadratic_adjoint(grids, xqs; bc = bc, extrap = extrap)

        n_queries = length(xqs[1])

        f_zero = zeros(eltype(f), size(f))
        itp_zero = quadratic_interp(grids, f_zero; bc = bc, extrap = extrap)
        Wf = Vector{eltype(f)}(undef, n_queries)
        Wf_zero = Vector{eltype(f)}(undef, n_queries)
        itp(Wf, xqs; deriv = deriv)
        itp_zero(Wf_zero, xqs; deriv = deriv)
        Wf .-= Wf_zero

        WTy = adj(y_bar; deriv = deriv)

        lhs = dot(Wf, y_bar)
        rhs = dot(vec(f), vec(WTy))
        return lhs, rhs, isapprox(lhs, rhs; rtol = rtol)
    end


    nx3, ny3, nz3 = 8, 6, 5
    nq3 = 20

    x3 = range(0.0, 1.0, nx3)
    y3 = range(0.0, 2.0, ny3)
    z3 = range(0.0, 0.5, nz3)

    xq3 = sort(rand(nq3)) .* 0.96 .+ 0.02
    yq3 = sort(rand(nq3)) .* 1.92 .+ 0.04
    zq3 = sort(rand(nq3)) .* 0.46 .+ 0.02

    f3 = randn(nx3, ny3, nz3)
    yb3 = randn(nq3)

    @testset "Dot-product — uniform grids" begin
        _, _, ok = quadratic_nd_dot_product_test(
            (x3, y3, z3), (xq3, yq3, zq3), f3, yb3
        )
        @test ok
    end

    @testset "Dot-product — non-uniform grids" begin
        x_nu = cumsum(0.5 .+ rand(nx3))
        x_nu .= (x_nu .- x_nu[1]) ./ (x_nu[end] - x_nu[1])
        y_nu = cumsum(0.5 .+ rand(ny3))
        y_nu .= (y_nu .- y_nu[1]) ./ (y_nu[end] - y_nu[1]) .* 2.0
        z_nu = cumsum(0.5 .+ rand(nz3))
        z_nu .= (z_nu .- z_nu[1]) ./ (z_nu[end] - z_nu[1]) .* 0.5
        f_nu = randn(nx3, ny3, nz3)
        _, _, ok = quadratic_nd_dot_product_test(
            (collect(x_nu), collect(y_nu), collect(z_nu)), (xq3, yq3, zq3), f_nu, yb3
        )
        @test ok
    end

    @testset "Dot-product — $bc_name" for (bc_name, bc) in [
            ("Left(QuadraticFit)", Left(QuadraticFit())),
            ("Right(QuadraticFit)", Right(QuadraticFit())),
            ("Left(Deriv1(0.0))", Left(Deriv1(0.0))),
            ("Right(Deriv2(0.0))", Right(Deriv2(0.0))),
            ("Left(Deriv2(1.0))", Left(Deriv2(1.0))),
            ("MinCurvFit", MinCurvFit()),
        ]

        f_bc = randn(nx3, ny3, nz3)
        _, _, ok = quadratic_nd_dot_product_test(
            (x3, y3, z3), (xq3, yq3, zq3), f_bc, yb3; bc = bc
        )
        @test ok
    end

    @testset "Dot-product — per-axis BC" begin
        f_pa = randn(nx3, ny3, nz3)
        _, _, ok = quadratic_nd_dot_product_test(
            (x3, y3, z3), (xq3, yq3, zq3), f_pa, yb3;
            bc = (Left(QuadraticFit()), Right(Deriv2(0.0)), MinCurvFit())
        )
        @test ok
    end

    @testset "Derivative adjoint — deriv=(1,0,0)" begin
        f_d = randn(nx3, ny3, nz3)
        _, _, ok = quadratic_nd_dot_product_test(
            (x3, y3, z3), (xq3, yq3, zq3), f_d, yb3;
            deriv = (EvalDeriv1(), EvalValue(), EvalValue())
        )
        @test ok
    end

    @testset "Derivative adjoint — deriv=(0,1,0)" begin
        f_d = randn(nx3, ny3, nz3)
        _, _, ok = quadratic_nd_dot_product_test(
            (x3, y3, z3), (xq3, yq3, zq3), f_d, yb3;
            deriv = (EvalValue(), EvalDeriv1(), EvalValue())
        )
        @test ok
    end

    @testset "Derivative adjoint — deriv=(0,0,2)" begin
        f_d = randn(nx3, ny3, nz3)
        _, _, ok = quadratic_nd_dot_product_test(
            (x3, y3, z3), (xq3, yq3, zq3), f_d, yb3;
            deriv = (EvalValue(), EvalValue(), EvalDeriv2())
        )
        @test ok
    end

    @testset "EvalDeriv3 → zero" begin
        adj = quadratic_adjoint((x3, y3, z3), (xq3, yq3, zq3))
        fb = adj(yb3; deriv = (EvalDeriv3(), EvalValue(), EvalValue()))
        @test all(iszero, fb)
    end

    @testset "In-place vs allocating" begin
        adj = quadratic_adjoint((x3, y3, z3), (xq3, yq3, zq3))
        fb_alloc = adj(yb3)
        fb_ip = zeros(nx3, ny3, nz3)
        adj(fb_ip, yb3)
        @test fb_alloc ≈ fb_ip
    end

    @testset "Size" begin
        adj = quadratic_adjoint((x3, y3, z3), (xq3, yq3, zq3))
        @test size(adj) == (nx3, ny3, nz3, nq3)
    end

    @testset "Type stability — constructor" begin
        @test @inferred(
            quadratic_adjoint((x3, y3, z3), (xq3, yq3, zq3))
        ) isa QuadraticAdjointND
    end

    @testset "Type stability — apply" begin
        adj = quadratic_adjoint((x3, y3, z3), (xq3, yq3, zq3))
        @test @inferred(adj(yb3)) isa Array{Float64, 3}
    end

    @testset "SVector query (N=3)" begin
        pt3 = SVector(0.5, 1.0, 0.25)
        adj_sv3 = quadratic_adjoint((x3, y3, z3), pt3)
        adj_tu3 = quadratic_adjoint((x3, y3, z3), (0.5, 1.0, 0.25))
        @test adj_sv3(1.0) ≈ adj_tu3(1.0)
    end

    @testset "Vector{SVector} batch (N=3)" begin
        queries_sv3 = [SVector(xq3[k], yq3[k], zq3[k]) for k in 1:nq3]
        adj_aos3 = quadratic_adjoint((x3, y3, z3), queries_sv3)
        adj_soa3 = quadratic_adjoint((x3, y3, z3), (xq3, yq3, zq3))
        @test adj_aos3(yb3) ≈ adj_soa3(yb3)
    end

    xq_oob3 = vcat(-0.1, xq3[1:5], 1.1)
    yq_oob3 = vcat(-0.3, yq3[1:5], 2.1)
    zq_oob3 = vcat(-0.05, zq3[1:5], 0.55)
    yb_oob3 = randn(length(xq_oob3))
    f_oob3 = randn(nx3, ny3, nz3)

    @testset "OOB — $ext_name (N=3)" for (ext_name, ext) in [
            ("ExtendExtrap", ExtendExtrap()),
            ("ClampExtrap", ClampExtrap()),
            ("FillExtrap", FillExtrap(0.0)),
        ]

        _, _, ok = quadratic_nd_dot_product_test(
            (x3, y3, z3), (xq_oob3, yq_oob3, zq_oob3), f_oob3, yb_oob3;
            extrap = ext
        )
        @test ok
    end

    @testset "Mutation safety — grid mutation" begin
        x = collect(range(0.0, 1.0, nx3))
        y = collect(range(0.0, 2.0, ny3))
        z = collect(range(0.0, 0.5, nz3))
        adj = quadratic_adjoint((x, y, z), (xq3, yq3, zq3))
        result_before = adj(yb3)
        x[3] = 100.0; y[2] = 100.0; z[2] = 100.0
        result_after = adj(yb3)
        @test result_before == result_after
    end
end
