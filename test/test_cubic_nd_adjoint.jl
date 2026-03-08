using Test
using LinearAlgebra: dot
using FastInterpolations

# ========================================
# Helper: ND Dot-product test
# ========================================
# Gold standard: ⟨W·f, ȳ⟩ = ⟨f, Wᵀ·ȳ⟩

function dot_product_test_nd(
        grids, xqs, f, y_bar;
        bc = CubicFit(),
        rtol = sqrt(eps(eltype(grids[1])))
    )
    itp = cubic_interp(grids, f; bc = bc)
    adj = cubic_adjoint(grids, xqs; bc = bc)

    n_queries = length(xqs[1])

    # Forward: W·f (subtract constant offset for affine BCs)
    f_zero = zeros(eltype(f), size(f))
    itp_zero = cubic_interp(grids, f_zero; bc = bc)
    Wf = Vector{eltype(f)}(undef, n_queries)
    Wf_zero = Vector{eltype(f)}(undef, n_queries)
    itp(Wf, xqs)
    itp_zero(Wf_zero, xqs)
    Wf .-= Wf_zero  # linear part only

    # Adjoint: Wᵀ·ȳ
    WTy = adj(y_bar)

    lhs = dot(Wf, y_bar)
    rhs = dot(vec(f), vec(WTy))
    return lhs, rhs, isapprox(lhs, rhs; rtol = rtol)
end

@testset "CubicAdjointND (N=2)" begin
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
    # Dot-product identity tests
    # ========================================
    @testset "Dot-product — CubicFit" begin
        @testset "Uniform grids (Range × Range)" begin
            _, _, ok = dot_product_test_nd(
                (x_uniform, y_uniform), (xq, yq), f, y_bar
            )
            @test ok
        end

        @testset "Non-uniform grids (Vector × Vector)" begin
            f_nu = randn(nx, ny)
            _, _, ok = dot_product_test_nd(
                (collect(x_nonuniform), collect(y_nonuniform)), (xq, yq), f_nu, y_bar
            )
            @test ok
        end

        @testset "Mixed grids (Range × Vector)" begin
            f_mix = randn(nx, ny)
            _, _, ok = dot_product_test_nd(
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
                _, _, ok = dot_product_test_nd((x_g, y_g), (xq_g, yq_g), f_g, yb_g)
                @test ok
            end
        end
    end

    # ========================================
    # In-place vs allocating consistency
    # ========================================
    @testset "In-place vs allocating" begin
        adj = cubic_adjoint((x_uniform, y_uniform), (xq, yq))

        f_bar_alloc = adj(y_bar)
        f_bar_inplace = zeros(nx, ny)
        adj(f_bar_inplace, y_bar)

        @test f_bar_alloc ≈ f_bar_inplace
    end

    # ========================================
    # Matrix materialization (small grid)
    # ========================================
    @testset "Matrix materialization" begin
        sx, sy = 5, 4
        x_s = range(0.0, 1.0, sx)
        y_s = range(0.0, 1.0, sy)
        n_q = 8
        xq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        yq_s = sort(rand(n_q)) .* 0.96 .+ 0.02

        adj = cubic_adjoint((x_s, y_s), (xq_s, yq_s))

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
            itp = cubic_interp((x_s, y_s), e_f)
            itp(output, (xq_s, yq_s))
            W[:, k] = output
        end

        @test W' ≈ WT rtol = 1e-12
    end

    # ========================================
    # Edge cases
    # ========================================
    @testset "Edge cases" begin
        @testset "Single query point" begin
            adj = cubic_adjoint((x_uniform, y_uniform), ([0.5], [1.0]))
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
            _, _, ok = dot_product_test_nd(
                (x_uniform, y_uniform), (xq_n, yq_n), f_n, yb_n
            )
            @test ok
        end
    end

    # ========================================
    # Dimension validation
    # ========================================
    @testset "Dimension mismatch errors" begin
        adj = cubic_adjoint((x_uniform, y_uniform), (xq, yq))

        @test_throws DimensionMismatch adj(randn(n_query + 1))
        @test_throws DimensionMismatch adj(zeros(nx + 1, ny), y_bar)
        @test_throws DimensionMismatch cubic_adjoint(
            (x_uniform, y_uniform), (xq, randn(n_query + 1))
        )
    end

    # ========================================
    # Type stability — constructor @inferred
    # ========================================
    @testset "Type stability — constructor" begin
        @test @inferred(cubic_adjoint((x_uniform, y_uniform), (xq, yq))) isa CubicAdjointND
        @test @inferred(cubic_adjoint(
            (x_uniform, y_uniform), (xq, yq); bc = CubicFit()
        )) isa CubicAdjointND

        # Non-uniform grids
        @test @inferred(cubic_adjoint(
            (collect(x_nonuniform), collect(y_nonuniform)), (xq, yq)
        )) isa CubicAdjointND

        # Mixed grids (Range × Vector)
        @test @inferred(cubic_adjoint(
            (x_uniform, collect(y_nonuniform)), (xq, yq)
        )) isa CubicAdjointND
    end

    # ========================================
    # Type stability — apply @inferred
    # ========================================
    @testset "Type stability — apply Float64" begin
        adj = cubic_adjoint((x_uniform, y_uniform), (xq, yq))
        @test @inferred(adj(y_bar)) isa Matrix{Float64}
    end

    @testset "Type stability — apply Float32" begin
        x32 = range(0.0f0, 1.0f0, nx)
        y32 = range(0.0f0, 2.0f0, ny)
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        yb32 = randn(Float32, n_query)
        adj32 = cubic_adjoint((x32, y32), (xq32, yq32))
        @test @inferred(adj32(yb32)) isa Matrix{Float32}
    end

    # ========================================
    # Zero-allocation (in-place)
    # ========================================
    # Function barrier: @testset wraps body in try/catch → type-unstable locals.
    # All setup + warmup + @allocated must be inside ONE function.

    function _test_nd_adjoint_alloc(grids, queries, f_bar, y_bar)
        adj = cubic_adjoint(grids, queries)
        adj(f_bar, y_bar)  # warmup
        adj(f_bar, y_bar)  # warmup
        return @allocated adj(f_bar, y_bar)
    end

    @testset "Zero-alloc: in-place uniform" begin
        fb = zeros(nx, ny)
        allocs = _test_nd_adjoint_alloc((x_uniform, y_uniform), (xq, yq), fb, y_bar)
        @test allocs <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place non-uniform" begin
        fb = zeros(nx, ny)
        allocs = _test_nd_adjoint_alloc(
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
        allocs = _test_nd_adjoint_alloc((x32, y32), (xq32, yq32), fb32, yb32)
        @test allocs <= ND_ALLOC_THRESHOLD
    end
end
