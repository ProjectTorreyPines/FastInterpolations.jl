@testitem "ConstantAdjointND (N=2)" setup = [AllocConstants] begin
    using LinearAlgebra: dot
    using StaticArrays: SVector

    # ========================================
    # Helper: ND Dot-product test
    # ========================================
    # Gold standard: ⟨W·f, ȳ⟩ = ⟨f, Wᵀ·ȳ⟩

    function constant_nd_dot_product_test(
            grids, xqs, f, y_bar;
            side = NearestSide(),
            extrap = NoExtrap(),
            deriv = EvalValue(),
            rtol = sqrt(eps(eltype(grids[1])))
        )
        itp = constant_interp(grids, f; side = side, extrap = extrap)
        adj = constant_adjoint(grids, xqs; side = side, extrap = extrap)

        n_queries = length(xqs[1])
        Wf = Vector{eltype(f)}(undef, n_queries)
        itp(Wf, xqs; deriv = deriv)

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

    xq = sort(rand(n_query)) .* 0.96 .+ 0.02
    yq = sort(rand(n_query)) .* 1.92 .+ 0.04

    f = randn(nx, ny)
    y_bar = randn(n_query)

    # ========================================
    # Dot-product identity tests (all side modes × grid types)
    # ========================================
    @testset "Dot-product — $grid_name — $side_name" for (grid_name, grids) in [
                ("Uniform", (x_uniform, y_uniform)),
                ("Non-uniform", (collect(x_nonuniform), collect(y_nonuniform))),
                ("Mixed", (x_uniform, collect(y_nonuniform))),
            ],
            (side_name, side_mode) in [
                ("NearestSide", NearestSide()),
                ("LeftSide", LeftSide()),
                ("RightSide", RightSide()),
            ]

        f_g = randn(nx, ny)
        _, _, ok = constant_nd_dot_product_test(grids, (xq, yq), f_g, y_bar; side = side_mode)
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
                _, _, ok = constant_nd_dot_product_test((x_g, y_g), (xq_g, yq_g), f_g, yb_g)
                @test ok
            end
        end
    end

    # ========================================
    # OOB extrap dot-product tests
    # ========================================
    xq_oob = vcat(-0.3 .+ 0.1 .* rand(10), xq, 1.0 .+ 0.1 .* rand(10))
    yq_oob = vcat(-0.5 .+ 0.2 .* rand(10), yq, 2.0 .+ 0.2 .* rand(10))
    y_bar_oob = randn(length(xq_oob))
    f_oob = randn(nx, ny)

    @testset "Dot-product — OOB — $ext_name — $side_name" for (ext_name, ext) in [
                ("ExtendExtrap", ExtendExtrap()),
                ("ClampExtrap", ClampExtrap()),
                ("FillExtrap", FillExtrap(0.0)),
                ("WrapExtrap", WrapExtrap()),
            ],
            (side_name, side_mode) in [
                ("NearestSide", NearestSide()),
                ("LeftSide", LeftSide()),
                ("RightSide", RightSide()),
            ]

        _, _, ok = constant_nd_dot_product_test(
            (x_uniform, y_uniform), (xq_oob, yq_oob), f_oob, y_bar_oob;
            side = side_mode, extrap = ext
        )
        @test ok
    end

    # ========================================
    # Derivative adjoint — all derivatives return zero
    # ========================================
    @testset "Derivative adjoint — any derivative → zero" begin
        adj = constant_adjoint((x_uniform, y_uniform), (xq, yq))

        @testset "deriv=(1,0)" begin
            f_bar = adj(y_bar; deriv = (EvalDeriv1(), EvalValue()))
            @test all(iszero, f_bar)
        end
        @testset "deriv=(0,1)" begin
            f_bar = adj(y_bar; deriv = (EvalValue(), EvalDeriv1()))
            @test all(iszero, f_bar)
        end
        @testset "deriv=(1,1)" begin
            f_bar = adj(y_bar; deriv = (EvalDeriv1(), EvalDeriv1()))
            @test all(iszero, f_bar)
        end
        @testset "deriv=(2,0)" begin
            f_bar = adj(y_bar; deriv = (EvalDeriv2(), EvalValue()))
            @test all(iszero, f_bar)
        end
        @testset "deriv=(0,3)" begin
            f_bar = adj(y_bar; deriv = (EvalValue(), EvalDeriv3()))
            @test all(iszero, f_bar)
        end
    end

    @testset "Derivative dot-product — all zero" begin
        @testset "deriv=(1,0)" begin
            adj = constant_adjoint((x_uniform, y_uniform), (xq, yq))
            @test all(iszero, adj(y_bar; deriv = (EvalDeriv1(), EvalValue())))
        end
    end

    # ========================================
    # In-place vs allocating
    # ========================================
    @testset "In-place vs allocating — $side_name" for (side_name, side_mode) in [
            ("NearestSide", NearestSide()),
            ("LeftSide", LeftSide()),
            ("RightSide", RightSide()),
        ]
        adj = constant_adjoint((x_uniform, y_uniform), (xq, yq); side = side_mode)
        f_bar_alloc = adj(y_bar)
        f_bar_inplace = zeros(nx, ny)
        adj(f_bar_inplace, y_bar)
        @test f_bar_alloc ≈ f_bar_inplace
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = constant_adjoint((x_uniform, y_uniform), (xq, yq))
        @test size(adj) == (nx, ny, n_query)
        @test size(adj, 1) == nx
        @test size(adj, 2) == ny
        @test size(adj, 3) == n_query
    end

    # ========================================
    # Type stability
    # ========================================
    @testset "Type stability — constructor @inferred" begin
        @test @inferred(constant_adjoint((x_uniform, y_uniform), (xq, yq))) isa ConstantAdjointND
        @test @inferred(
            constant_adjoint(
                (x_uniform, y_uniform), (xq, yq);
                side = LeftSide()
            )
        ) isa ConstantAdjointND
    end

    @testset "Type stability — apply" begin
        adj = constant_adjoint((x_uniform, y_uniform), (xq, yq))
        @test @inferred(adj(y_bar)) isa Matrix{Float64}
    end

    # ========================================
    # Float32
    # ========================================
    @testset "Float32" begin
        x32 = Float32.(collect(x_uniform))
        y32 = Float32.(collect(y_uniform))
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        f32 = randn(Float32, nx, ny)
        yb32 = randn(Float32, n_query)
        _, _, ok = constant_nd_dot_product_test(
            (x32, y32), (xq32, yq32), f32, yb32;
            rtol = sqrt(eps(Float32))
        )
        @test ok
    end

    # ========================================
    # Matrix materialization
    # ========================================
    @testset "Matrix materialization — $side_name" for (side_name, side_mode) in [
            ("NearestSide", NearestSide()),
            ("LeftSide", LeftSide()),
            ("RightSide", RightSide()),
        ]
        sx, sy = 5, 4
        x_s = range(0.0, 1.0, sx)
        y_s = range(0.0, 1.0, sy)
        n_q = 8
        xq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        yq_s = sort(rand(n_q)) .* 0.96 .+ 0.02

        adj = constant_adjoint((x_s, y_s), (xq_s, yq_s); side = side_mode)

        # Build Wᵀ by probing adjoint with unit vectors
        WT = zeros(sx * sy, n_q)
        e_q = zeros(n_q)
        for q in 1:n_q
            e_q[q] = 1.0
            WT[:, q] = vec(adj(e_q))
            e_q[q] = 0.0
        end

        # Build W by probing forward with unit vectors
        W = zeros(n_q, sx * sy)
        output = zeros(n_q)
        for k in 1:(sx * sy)
            e_f = zeros(sx, sy)
            e_f[k] = 1.0
            itp = constant_interp((x_s, y_s), e_f; side = side_mode)
            itp(output, (xq_s, yq_s))
            W[:, k] = output
        end

        @test W' ≈ WT rtol = 1.0e-12

        # Test Matrix() function directly
        WT_mat = Matrix(adj)
        @test WT_mat ≈ WT rtol = 1.0e-12
    end

    # ========================================
    # Edge cases
    # ========================================
    @testset "Single query point" begin
        adj = constant_adjoint((x_uniform, y_uniform), ([0.5], [1.0]))
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
        _, _, ok = constant_nd_dot_product_test(
            (x_uniform, y_uniform), (xq_n, yq_n), f_n, yb_n
        )
        @test ok
    end

    # ========================================
    # SVector queries
    # ========================================
    @testset "SVector query — scalar" begin
        sv = SVector(0.5, 1.0)
        adj = constant_adjoint((x_uniform, y_uniform), sv)
        f_bar = adj(1.0)
        @test size(f_bar) == (nx, ny)

        adj2 = constant_adjoint((x_uniform, y_uniform), ([0.5], [1.0]))
        @test f_bar ≈ adj2(1.0)
    end

    @testset "SVector query — vector of SVector" begin
        svecs = [SVector(xq[i], yq[i]) for i in 1:n_query]
        adj_sv = constant_adjoint((x_uniform, y_uniform), svecs)
        adj_soa = constant_adjoint((x_uniform, y_uniform), (xq, yq))
        @test adj_sv(y_bar) ≈ adj_soa(y_bar)
    end

    # ========================================
    # Per-axis side modes
    # ========================================
    @testset "Per-axis side modes" begin
        adj = constant_adjoint(
            (x_uniform, y_uniform), (xq, yq);
            side = (LeftSide(), RightSide())
        )
        itp = constant_interp(
            (x_uniform, y_uniform), f;
            side = (LeftSide(), RightSide())
        )

        Wf = Vector{Float64}(undef, n_query)
        itp(Wf, (xq, yq))
        WTy = adj(y_bar)
        @test dot(Wf, y_bar) ≈ dot(vec(f), vec(WTy))
    end

    # ========================================
    # Analytical value tests (N=2)
    # ========================================
    @testset "Analytical — NearestSide single scatter" begin
        gx = [0.0, 1.0, 2.0]
        gy = [0.0, 1.0]

        # Query (0.7, 0.3): x-axis dL=0.7 > 0.5 → offset=1, y-axis dL=0.3 <= 0.5 → offset=0
        # Target: (1+1, 1+0) = (2, 1)
        adj = constant_adjoint((gx, gy), ([0.7], [0.3]); side = NearestSide())
        expected = zeros(3, 2)
        expected[2, 1] = 1.0
        @test adj([1.0]) ≈ expected
    end

    @testset "Analytical — LeftSide single scatter" begin
        gx = [0.0, 1.0, 2.0]
        gy = [0.0, 1.0]

        # LeftSide: offset always 0 → target = (1, 1)
        adj = constant_adjoint((gx, gy), ([0.7], [0.3]); side = LeftSide())
        expected = zeros(3, 2)
        expected[1, 1] = 1.0
        @test adj([1.0]) ≈ expected
    end

    @testset "Analytical — RightSide single scatter" begin
        gx = [0.0, 1.0, 2.0]
        gy = [0.0, 1.0]

        # RightSide: dL != 0 → offset=1 for both axes → target = (2, 2)
        adj = constant_adjoint((gx, gy), ([0.7], [0.3]); side = RightSide())
        expected = zeros(3, 2)
        expected[2, 2] = 1.0
        @test adj([1.0]) ≈ expected
    end

    @testset "Analytical — two queries accumulation" begin
        gx = [0.0, 1.0, 2.0]
        gy = [0.0, 1.0]

        adj = constant_adjoint((gx, gy), ([0.3, 0.7], [0.3, 0.7]); side = NearestSide())
        # q1: x dL=0.3<=0.5→off=0, y dL=0.3<=0.5→off=0 → target (1,1) += 2.0
        # q2: x dL=0.7>0.5→off=1, y dL=0.7>0.5→off=1 → target (2,2) += 3.0
        expected = zeros(3, 2)
        expected[1, 1] = 2.0
        expected[2, 2] = 3.0
        @test adj([2.0, 3.0]) ≈ expected
    end

    # ── Extrap analytical tests (ND) ──

    @testset "Analytical — FillExtrap (OOB axis → skip)" begin
        gx = [0.0, 1.0, 2.0]
        gy = [0.0, 1.0]
        adj = constant_adjoint((gx, gy), ([-0.5], [0.5]); side = NearestSide(), extrap = FillExtrap(0.0))
        # x OOB with FillExtrap → entire query skipped (fill value independent of f)
        @test adj([1.0]) ≈ zeros(3, 2)
    end

    @testset "Analytical — ClampExtrap + NearestSide (OOB clamped)" begin
        gx = [0.0, 1.0, 2.0]
        gy = [0.0, 1.0]
        adj = constant_adjoint((gx, gy), ([2.5], [0.5]); side = NearestSide(), extrap = ClampExtrap())
        # x: clamped to 2.0, idx=2, dL=1.0>0.5 → off=1 → x=3
        # y: idx=1, dL=0.5<=0.5 → off=0 → y=1
        expected = zeros(3, 2)
        expected[3, 1] = 1.0
        @test adj([1.0]) ≈ expected
    end

    @testset "Analytical — ClampExtrap + LeftSide (OOB at right boundary)" begin
        gx = [0.0, 1.0, 2.0]
        gy = [0.0, 1.0]
        adj = constant_adjoint((gx, gy), ([2.5], [0.5]); side = LeftSide(), extrap = ClampExtrap())
        # x: clamped to 2.0, idx=2, dL=1.0. LeftSide: off=0 → x=2
        # y: idx=1, dL=0.5. LeftSide: off=0 → y=1
        expected = zeros(3, 2)
        expected[2, 1] = 1.0
        @test adj([1.0]) ≈ expected
    end

    @testset "Analytical — ExtendExtrap (OOB below)" begin
        gx = [0.0, 1.0, 2.0]
        gy = [0.0, 1.0]
        adj = constant_adjoint((gx, gy), ([-0.5], [0.5]); side = NearestSide(), extrap = ExtendExtrap())
        # x: -0.5 passed through, idx=1, dL=-0.5<=0.5 → off=0 → x=1
        # y: idx=1, dL=0.5<=0.5 → off=0 → y=1
        expected = zeros(3, 2)
        expected[1, 1] = 1.0
        @test adj([1.0]) ≈ expected
    end

    @testset "Analytical — WrapExtrap" begin
        gx = [0.0, 1.0, 2.0]  # period = 2
        gy = [0.0, 1.0]
        adj = constant_adjoint((gx, gy), ([2.3], [0.5]); side = NearestSide(), extrap = WrapExtrap())
        # x: 2.3 wraps to 0.3, idx=1, dL=0.3<=0.5 → off=0 → x=1
        # y: idx=1, dL=0.5<=0.5 → off=0 → y=1
        expected = zeros(3, 2)
        expected[1, 1] = 1.0
        @test adj([1.0]) ≈ expected
    end

    # ========================================
    # Zero-allocation tests
    # ========================================
    function _test_constant_nd_adjoint_alloc_inplace(
            grids, queries, f_bar, y_bar;
            side = NearestSide(), extrap = NoExtrap(), deriv = EvalValue()
        )
        adj = constant_adjoint(grids, queries; side = side, extrap = extrap)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place — $side_name" for (side_name, side_mode) in [
            ("NearestSide", NearestSide()),
            ("LeftSide", LeftSide()),
            ("RightSide", RightSide()),
        ]
        fb = zeros(nx, ny)
        allocs = _test_constant_nd_adjoint_alloc_inplace(
            (x_uniform, y_uniform), (xq, yq), fb, y_bar; side = side_mode
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
        allocs = _test_constant_nd_adjoint_alloc_inplace(
            (x32, y32), (xq32, yq32), fb32, yb32
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end

    # ========================================
    # Singleton grid rejection
    # ========================================
    @testset "Singleton grid throws ArgumentError" begin
        @test_throws ArgumentError constant_adjoint(([0.0], [0.0, 1.0]), ([0.0], [0.5]))
        @test_throws ArgumentError constant_adjoint(([0.0, 1.0], [0.0]), ([0.5], [0.0]))
    end

    # ========================================
    # Mutation safety — grid mutation after construction
    # ========================================
    @testset "Mutation safety — grid mutation" begin
        x = collect(range(0.0, 1.0, nx))
        y = collect(range(0.0, 2.0, ny))
        adj = constant_adjoint((x, y), (xq, yq))
        result_before = adj(y_bar)
        x[5] = 100.0
        y[4] = 100.0
        result_after = adj(y_bar)
        @test result_before == result_after
    end
end

# ========================================
# N=3 Tests (generic ntuple path)
# ========================================
@testitem "ConstantAdjointND (N=3)" setup = [AllocConstants] begin
    using LinearAlgebra: dot

    # ========================================
    # Helper: ND Dot-product test
    # ========================================
    # Gold standard: ⟨W·f, ȳ⟩ = ⟨f, Wᵀ·ȳ⟩

    function constant_nd_dot_product_test(
            grids, xqs, f, y_bar;
            side = NearestSide(),
            extrap = NoExtrap(),
            deriv = EvalValue(),
            rtol = sqrt(eps(eltype(grids[1])))
        )
        itp = constant_interp(grids, f; side = side, extrap = extrap)
        adj = constant_adjoint(grids, xqs; side = side, extrap = extrap)

        n_queries = length(xqs[1])
        Wf = Vector{eltype(f)}(undef, n_queries)
        itp(Wf, xqs; deriv = deriv)

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

    @testset "Dot-product — $side_name" for (side_name, side_mode) in [
            ("NearestSide", NearestSide()),
            ("LeftSide", LeftSide()),
            ("RightSide", RightSide()),
        ]
        itp = constant_interp((x3, y3, z3), f3; side = side_mode)
        adj = constant_adjoint((x3, y3, z3), (xq3, yq3, zq3); side = side_mode)

        Wf = Vector{Float64}(undef, nq3)
        itp(Wf, (xq3, yq3, zq3))
        WTy = adj(yb3)

        @test dot(Wf, yb3) ≈ dot(vec(f3), vec(WTy))
    end

    @testset "In-place vs allocating" begin
        adj = constant_adjoint((x3, y3, z3), (xq3, yq3, zq3))
        fb_alloc = adj(yb3)
        fb_ip = zeros(nx3, ny3, nz3)
        adj(fb_ip, yb3)
        @test fb_alloc ≈ fb_ip
    end

    @testset "Size" begin
        adj = constant_adjoint((x3, y3, z3), (xq3, yq3, zq3))
        @test size(adj) == (nx3, ny3, nz3, nq3)
    end

    @testset "Derivative → zero" begin
        adj = constant_adjoint((x3, y3, z3), (xq3, yq3, zq3))
        fb = adj(yb3; deriv = (EvalDeriv1(), EvalValue(), EvalValue()))
        @test all(iszero, fb)
    end

    @testset "Mutation safety — grid mutation" begin
        x = collect(range(0.0, 1.0, nx3))
        y = collect(range(0.0, 2.0, ny3))
        z = collect(range(0.0, 0.5, nz3))
        adj = constant_adjoint((x, y, z), (xq3, yq3, zq3))
        result_before = adj(yb3)
        x[3] = 100.0; y[2] = 100.0; z[2] = 100.0
        result_after = adj(yb3)
        @test result_before == result_after
    end
end
