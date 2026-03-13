using Test
using LinearAlgebra: dot
using StaticArrays: SVector
using FastInterpolations

# ========================================
# Helper: ND Dot-product test
# ========================================
# Gold standard: ⟨W·f, ȳ⟩ = ⟨f, Wᵀ·ȳ⟩
# Linear interp is purely linear in f (no BC constant offset).

function linear_nd_dot_product_test(
        grids, xqs, f, y_bar;
        extrap = NoExtrap(),
        deriv = EvalValue(),
        rtol = sqrt(eps(eltype(grids[1])))
    )
    itp = linear_interp(grids, f; extrap = extrap)
    adj = linear_adjoint(grids, xqs; extrap = extrap)

    n_queries = length(xqs[1])
    Wf = Vector{eltype(f)}(undef, n_queries)
    itp(Wf, xqs; deriv = deriv)

    WTy = adj(y_bar; deriv = deriv)

    lhs = dot(Wf, y_bar)
    rhs = dot(vec(f), vec(WTy))
    return lhs, rhs, isapprox(lhs, rhs; rtol = rtol)
end

@testset "LinearAdjointND (N=2)" begin
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
    # Dot-product identity tests
    # ========================================
    @testset "Dot-product — grid types" begin
        @testset "Uniform grids (Range × Range)" begin
            _, _, ok = linear_nd_dot_product_test(
                (x_uniform, y_uniform), (xq, yq), f, y_bar
            )
            @test ok
        end

        @testset "Non-uniform grids (Vector × Vector)" begin
            f_nu = randn(nx, ny)
            _, _, ok = linear_nd_dot_product_test(
                (collect(x_nonuniform), collect(y_nonuniform)), (xq, yq), f_nu, y_bar
            )
            @test ok
        end

        @testset "Mixed grids (Range × Vector)" begin
            f_mix = randn(nx, ny)
            _, _, ok = linear_nd_dot_product_test(
                (x_uniform, collect(y_nonuniform)), (xq, yq), f_mix, y_bar
            )
            @test ok
        end
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
                _, _, ok = linear_nd_dot_product_test((x_g, y_g), (xq_g, yq_g), f_g, yb_g)
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

    @testset "Dot-product — OOB queries — ExtendExtrap" begin
        _, _, ok = linear_nd_dot_product_test(
            (x_uniform, y_uniform), (xq_oob, yq_oob), f_oob, y_bar_oob;
            extrap = ExtendExtrap()
        )
        @test ok
    end

    @testset "Dot-product — OOB queries — ClampExtrap" begin
        _, _, ok = linear_nd_dot_product_test(
            (x_uniform, y_uniform), (xq_oob, yq_oob), f_oob, y_bar_oob;
            extrap = ClampExtrap()
        )
        @test ok
    end

    @testset "Dot-product — OOB queries — FillExtrap(0.0)" begin
        _, _, ok = linear_nd_dot_product_test(
            (x_uniform, y_uniform), (xq_oob, yq_oob), f_oob, y_bar_oob;
            extrap = FillExtrap(0.0)
        )
        @test ok
    end

    # ========================================
    # Derivative adjoint
    # ========================================
    @testset "Derivative adjoint — per-axis deriv" begin
        @testset "deriv=(1,0)" begin
            _, _, ok = linear_nd_dot_product_test(
                (x_uniform, y_uniform), (xq, yq), f, y_bar;
                deriv = (EvalDeriv1(), EvalValue())
            )
            @test ok
        end
        @testset "deriv=(0,1)" begin
            _, _, ok = linear_nd_dot_product_test(
                (x_uniform, y_uniform), (xq, yq), f, y_bar;
                deriv = (EvalValue(), EvalDeriv1())
            )
            @test ok
        end
        @testset "deriv=(1,1)" begin
            _, _, ok = linear_nd_dot_product_test(
                (x_uniform, y_uniform), (xq, yq), f, y_bar;
                deriv = (EvalDeriv1(), EvalDeriv1())
            )
            @test ok
        end
    end

    @testset "Derivative adjoint — 2nd+ derivative returns zero" begin
        adj = linear_adjoint((x_uniform, y_uniform), (xq, yq))
        f_bar = adj(y_bar; deriv = (EvalDeriv2(), EvalValue()))
        @test all(iszero, f_bar)
        f_bar2 = adj(y_bar; deriv = (EvalValue(), EvalDeriv3()))
        @test all(iszero, f_bar2)
    end

    # ========================================
    # In-place vs allocating
    # ========================================
    @testset "In-place vs allocating" begin
        adj = linear_adjoint((x_uniform, y_uniform), (xq, yq))
        f_bar_alloc = adj(y_bar)
        f_bar_inplace = zeros(nx, ny)
        adj(f_bar_inplace, y_bar)
        @test f_bar_alloc ≈ f_bar_inplace
    end

    # ========================================
    # size()
    # ========================================
    @testset "size" begin
        adj = linear_adjoint((x_uniform, y_uniform), (xq, yq))
        @test size(adj) == (nx, ny, n_query)
        @test size(adj, 1) == nx
        @test size(adj, 2) == ny
        @test size(adj, 3) == n_query
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

        adj = linear_adjoint((x_s, y_s), (xq_s, yq_s))

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
            itp = linear_interp((x_s, y_s), e_f)
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
        adj = linear_adjoint((x_uniform, y_uniform), ([0.5], [1.0]))
        f_bar = adj([1.0])
        @test size(f_bar) == (nx, ny)
    end

    @testset "Query at grid nodes" begin
        xq_nodes = collect(x_uniform)[2:(end - 1)]
        yq_nodes = collect(y_uniform)[2:min(end - 1, length(xq_nodes) + 1)]
        n_both = min(length(xq_nodes), length(yq_nodes))
        xq_n = xq_nodes[1:n_both]
        yq_n = yq_nodes[1:n_both]
        yb_n = randn(n_both)
        f_n = randn(nx, ny)
        _, _, ok = linear_nd_dot_product_test(
            (x_uniform, y_uniform), (xq_n, yq_n), f_n, yb_n
        )
        @test ok
    end

    # ========================================
    # Dimension validation
    # ========================================
    @testset "Dimension mismatch errors" begin
        adj = linear_adjoint((x_uniform, y_uniform), (xq, yq))
        @test_throws DimensionMismatch adj(randn(n_query + 1))
        @test_throws DimensionMismatch adj(zeros(nx + 1, ny), y_bar)
    end

    # ========================================
    # Type stability
    # ========================================
    @testset "Type stability — constructor" begin
        @test @inferred(linear_adjoint((x_uniform, y_uniform), (xq, yq))) isa LinearAdjointND
        @test @inferred(
            linear_adjoint(
                (collect(x_nonuniform), collect(y_nonuniform)), (xq, yq)
            )
        ) isa LinearAdjointND
        @test @inferred(
            linear_adjoint(
                (x_uniform, collect(y_nonuniform)), (xq, yq)
            )
        ) isa LinearAdjointND
    end

    @testset "Type stability — apply Float64" begin
        adj = linear_adjoint((x_uniform, y_uniform), (xq, yq))
        @test @inferred(adj(y_bar)) isa Matrix{Float64}
    end

    @testset "Type stability — apply Float32" begin
        x32 = range(0.0f0, 1.0f0, nx)
        y32 = range(0.0f0, 2.0f0, ny)
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        yb32 = randn(Float32, n_query)
        adj32 = linear_adjoint((x32, y32), (xq32, yq32))
        @test @inferred(adj32(yb32)) isa Matrix{Float32}
    end

    @testset "Dot-product — Float32" begin
        x32 = range(0.0f0, 1.0f0, nx)
        y32 = range(0.0f0, 2.0f0, ny)
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        f32 = randn(Float32, nx, ny)
        yb32 = randn(Float32, n_query)
        _, _, ok = linear_nd_dot_product_test((x32, y32), (xq32, yq32), f32, yb32)
        @test ok
    end

    # ========================================
    # SVector query
    # ========================================
    @testset "SVector query" begin
        adj_soa = linear_adjoint((x_uniform, y_uniform), (xq, yq))
        adj_sv = linear_adjoint((x_uniform, y_uniform), SVector(0.5, 1.0))
        f_bar = adj_sv([1.0])
        @test size(f_bar) == (nx, ny)
    end

    # ========================================
    # Zero-allocation (in-place)
    # ========================================
    function _test_linear_nd_adjoint_alloc(grids, queries, f_bar, y_bar)
        adj = linear_adjoint(grids, queries)
        adj(f_bar, y_bar)  # warmup
        adj(f_bar, y_bar)  # warmup
        return @allocated adj(f_bar, y_bar)
    end

    @testset "Zero-alloc: in-place uniform" begin
        fb = zeros(nx, ny)
        allocs = _test_linear_nd_adjoint_alloc((x_uniform, y_uniform), (xq, yq), fb, y_bar)
        @test allocs <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place non-uniform" begin
        fb = zeros(nx, ny)
        allocs = _test_linear_nd_adjoint_alloc(
            (collect(x_nonuniform), collect(y_nonuniform)), (xq, yq), fb, y_bar
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place Float32" begin
        x32 = range(0.0f0, 1.0f0, nx)
        y32 = range(0.0f0, 2.0f0, ny)
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        fb32 = zeros(Float32, nx, ny)
        yb32 = randn(Float32, n_query)
        allocs = _test_linear_nd_adjoint_alloc((x32, y32), (xq32, yq32), fb32, yb32)
        @test allocs <= ND_ALLOC_THRESHOLD
    end
end

# ========================================
# N=3 Tests
# ========================================

@testset "LinearAdjointND (N=3)" begin
    nx, ny, nz = 10, 8, 6
    n_query = 30

    x_uniform = range(0.0, 1.0, nx)
    y_uniform = range(0.0, 2.0, ny)
    z_uniform = range(0.0, 1.5, nz)

    xq = sort(rand(n_query)) .* 0.96 .+ 0.02
    yq = sort(rand(n_query)) .* 1.92 .+ 0.04
    zq = sort(rand(n_query)) .* 1.44 .+ 0.03

    f = randn(nx, ny, nz)
    y_bar = randn(n_query)

    @testset "Dot-product — uniform grids" begin
        _, _, ok = linear_nd_dot_product_test(
            (x_uniform, y_uniform, z_uniform), (xq, yq, zq), f, y_bar
        )
        @test ok
    end

    @testset "Dot-product — non-uniform grids" begin
        x_nu = cumsum(0.5 .+ rand(nx))
        x_nu .= (x_nu .- x_nu[1]) ./ (x_nu[end] - x_nu[1])
        y_nu = cumsum(0.5 .+ rand(ny))
        y_nu .= (y_nu .- y_nu[1]) ./ (y_nu[end] - y_nu[1]) .* 2.0
        z_nu = cumsum(0.5 .+ rand(nz))
        z_nu .= (z_nu .- z_nu[1]) ./ (z_nu[end] - z_nu[1]) .* 1.5
        f_nu = randn(nx, ny, nz)
        _, _, ok = linear_nd_dot_product_test(
            (collect(x_nu), collect(y_nu), collect(z_nu)), (xq, yq, zq), f_nu, y_bar
        )
        @test ok
    end

    @testset "In-place vs allocating" begin
        adj = linear_adjoint((x_uniform, y_uniform, z_uniform), (xq, yq, zq))
        f_bar_alloc = adj(y_bar)
        f_bar_inplace = zeros(nx, ny, nz)
        adj(f_bar_inplace, y_bar)
        @test f_bar_alloc ≈ f_bar_inplace
    end

    @testset "Single query point" begin
        adj = linear_adjoint(
            (x_uniform, y_uniform, z_uniform), ([0.5], [1.0], [0.75])
        )
        f_bar = adj([1.0])
        @test size(f_bar) == (nx, ny, nz)
    end

    @testset "Type stability — constructor" begin
        @test @inferred(
            linear_adjoint(
                (x_uniform, y_uniform, z_uniform), (xq, yq, zq)
            )
        ) isa LinearAdjointND
    end

    @testset "Type stability — apply" begin
        adj = linear_adjoint((x_uniform, y_uniform, z_uniform), (xq, yq, zq))
        @test @inferred(adj(y_bar)) isa Array{Float64, 3}
    end

    function _test_linear_nd_adjoint_alloc_3d(grids, queries, f_bar, y_bar)
        adj = linear_adjoint(grids, queries)
        adj(f_bar, y_bar)  # warmup
        adj(f_bar, y_bar)  # warmup
        return @allocated adj(f_bar, y_bar)
    end

    @testset "Zero-alloc: in-place uniform" begin
        fb = zeros(nx, ny, nz)
        allocs = _test_linear_nd_adjoint_alloc_3d(
            (x_uniform, y_uniform, z_uniform), (xq, yq, zq), fb, y_bar
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end
end
