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
        extrap = NoExtrap(),
        deriv = EvalValue(),
        rtol = sqrt(eps(eltype(grids[1])))
    )
    itp = cubic_interp(grids, f; bc = bc, extrap = extrap)
    adj = cubic_adjoint(grids, xqs; bc = bc, extrap = extrap)

    n_queries = length(xqs[1])

    # Forward: W·f (subtract constant offset for affine BCs)
    f_zero = zeros(eltype(f), size(f))
    itp_zero = cubic_interp(grids, f_zero; bc = bc, extrap = extrap)
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
    # OOB extrap dot-product tests
    # ========================================
    # OOB queries that extend beyond the grid domain in each axis.
    xq_oob = vcat(-0.3 .+ 0.1 .* rand(10), xq, 1.0 .+ 0.1 .* rand(10))
    yq_oob = vcat(-0.5 .+ 0.2 .* rand(10), yq, 2.0 .+ 0.2 .* rand(10))
    y_bar_oob = randn(length(xq_oob))
    f_oob = randn(nx, ny)

    @testset "Dot-product — OOB queries — ExtendExtrap" begin
        _, _, ok = dot_product_test_nd(
            (x_uniform, y_uniform), (xq_oob, yq_oob), f_oob, y_bar_oob;
            extrap = ExtendExtrap()
        )
        @test ok
    end

    @testset "Dot-product — OOB queries — ClampExtrap" begin
        _, _, ok = dot_product_test_nd(
            (x_uniform, y_uniform), (xq_oob, yq_oob), f_oob, y_bar_oob;
            extrap = ClampExtrap()
        )
        @test ok
    end

    @testset "Dot-product — OOB queries — FillExtrap" begin
        _, _, ok = dot_product_test_nd(
            (x_uniform, y_uniform), (xq_oob, yq_oob), f_oob, y_bar_oob;
            extrap = FillExtrap(0.0)
        )
        @test ok
    end

    @testset "Dot-product — OOB queries — WrapExtrap" begin
        _, _, ok = dot_product_test_nd(
            (x_uniform, y_uniform), (xq_oob, yq_oob), f_oob, y_bar_oob;
            extrap = WrapExtrap()
        )
        @test ok
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

        @test W' ≈ WT rtol = 1.0e-12

        # Test Matrix() function directly
        WT_mat = Matrix(adj)
        @test WT_mat ≈ WT rtol = 1.0e-12
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
        @test @inferred(
            cubic_adjoint(
                (x_uniform, y_uniform), (xq, yq); bc = CubicFit()
            )
        ) isa CubicAdjointND

        # Non-uniform grids
        @test @inferred(
            cubic_adjoint(
                (collect(x_nonuniform), collect(y_nonuniform)), (xq, yq)
            )
        ) isa CubicAdjointND

        # Mixed grids (Range × Vector)
        @test @inferred(
            cubic_adjoint(
                (x_uniform, collect(y_nonuniform)), (xq, yq)
            )
        ) isa CubicAdjointND
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

    @testset "Dot-product — Float32" begin
        x32 = range(0.0f0, 1.0f0, nx)
        y32 = range(0.0f0, 2.0f0, ny)
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        f32 = randn(Float32, nx, ny)
        yb32 = randn(Float32, n_query)
        _, _, ok = dot_product_test_nd((x32, y32), (xq32, yq32), f32, yb32)
        @test ok
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

# ========================================
# N=3 Tests
# ========================================

@testset "CubicAdjointND (N=3)" begin
    nx, ny, nz = 10, 8, 6
    n_query = 30

    x_uniform = range(0.0, 1.0, nx)
    y_uniform = range(0.0, 2.0, ny)
    z_uniform = range(0.0, 1.5, nz)

    x_nonuniform = cumsum(0.5 .+ rand(nx))
    x_nonuniform .= (x_nonuniform .- x_nonuniform[1]) ./ (x_nonuniform[end] - x_nonuniform[1])

    # Query points inside domain
    xq = sort(rand(n_query)) .* 0.96 .+ 0.02
    yq = sort(rand(n_query)) .* 1.92 .+ 0.04
    zq = sort(rand(n_query)) .* 1.44 .+ 0.03

    f = randn(nx, ny, nz)
    y_bar = randn(n_query)

    # ========================================
    # Dot-product identity
    # ========================================
    @testset "Dot-product — uniform grids" begin
        _, _, ok = dot_product_test_nd(
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
        _, _, ok = dot_product_test_nd(
            (collect(x_nu), collect(y_nu), collect(z_nu)), (xq, yq, zq), f_nu, y_bar
        )
        @test ok
    end

    # ========================================
    # Non-CubicFit BCs (N=3)
    # ========================================
    @testset "Dot-product — N=3 $bc_name" for (bc_name, bc) in [
            ("ZeroCurvBC", ZeroCurvBC()),
            ("ZeroSlopeBC", ZeroSlopeBC()),
            ("QuadraticFit", QuadraticFit()),
        ]
        _, _, ok = dot_product_test_nd(
            (x_uniform, y_uniform, z_uniform), (xq, yq, zq), f, y_bar; bc = bc
        )
        @test ok
    end

    @testset "Dot-product — N=3 mixed BCs (CubicFit × ZeroCurvBC × ZeroSlopeBC)" begin
        _, _, ok = dot_product_test_nd(
            (x_uniform, y_uniform, z_uniform), (xq, yq, zq), f, y_bar;
            bc = (CubicFit(), ZeroCurvBC(), ZeroSlopeBC())
        )
        @test ok
    end

    # ========================================
    # In-place vs allocating
    # ========================================
    @testset "In-place vs allocating" begin
        adj = cubic_adjoint((x_uniform, y_uniform, z_uniform), (xq, yq, zq))
        f_bar_alloc = adj(y_bar)
        f_bar_inplace = zeros(nx, ny, nz)
        adj(f_bar_inplace, y_bar)
        @test f_bar_alloc ≈ f_bar_inplace
    end

    # ========================================
    # Matrix materialization (small grid)
    # ========================================
    @testset "Matrix materialization" begin
        sx, sy, sz = 5, 4, 4
        x_s = range(0.0, 1.0, sx)
        y_s = range(0.0, 1.0, sy)
        z_s = range(0.0, 1.0, sz)
        n_q = 8
        xq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        yq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        zq_s = sort(rand(n_q)) .* 0.96 .+ 0.02

        adj = cubic_adjoint((x_s, y_s, z_s), (xq_s, yq_s, zq_s))
        n_grid = sx * sy * sz

        # Build Wᵀ by probing adjoint with unit vectors
        WT = zeros(n_grid, n_q)
        e_q = zeros(n_q)
        for q in 1:n_q
            e_q[q] = 1.0
            WT[:, q] = vec(adj(e_q))
            e_q[q] = 0.0
        end

        # Build W by probing forward with unit vectors
        W = zeros(n_q, n_grid)
        output = zeros(n_q)
        for k in 1:n_grid
            e_f = zeros(sx, sy, sz)
            e_f[k] = 1.0
            itp = cubic_interp((x_s, y_s, z_s), e_f)
            itp(output, (xq_s, yq_s, zq_s))
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
        adj = cubic_adjoint(
            (x_uniform, y_uniform, z_uniform), ([0.5], [1.0], [0.75])
        )
        f_bar = adj([1.0])
        @test size(f_bar) == (nx, ny, nz)
    end

    # ========================================
    # Type stability
    # ========================================
    @testset "Type stability — constructor" begin
        @test @inferred(
            cubic_adjoint(
                (x_uniform, y_uniform, z_uniform), (xq, yq, zq)
            )
        ) isa CubicAdjointND
    end

    @testset "Type stability — apply" begin
        adj = cubic_adjoint((x_uniform, y_uniform, z_uniform), (xq, yq, zq))
        @test @inferred(adj(y_bar)) isa Array{Float64, 3}
    end

    # ========================================
    # Zero-allocation (in-place)
    # ========================================
    function _test_nd_adjoint_alloc_3d(grids, queries, f_bar, y_bar)
        adj = cubic_adjoint(grids, queries)
        adj(f_bar, y_bar)  # warmup
        adj(f_bar, y_bar)  # warmup
        return @allocated adj(f_bar, y_bar)
    end

    @testset "Zero-alloc: in-place uniform" begin
        fb = zeros(nx, ny, nz)
        allocs = _test_nd_adjoint_alloc_3d(
            (x_uniform, y_uniform, z_uniform), (xq, yq, zq), fb, y_bar
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end
end

# ========================================
# Boundary Condition Tests (N=2)
# ========================================

@testset "CubicAdjointND — Boundary Conditions" begin
    nx, ny = 12, 10
    n_query = 25

    x = range(0.0, 1.0, nx)
    y = range(0.0, 1.0, ny)
    xq = sort(rand(n_query)) .* 0.96 .+ 0.02
    yq = sort(rand(n_query)) .* 0.96 .+ 0.02
    f = randn(nx, ny)
    y_bar = randn(n_query)

    @testset "BC: $bc_name" for (bc_name, bc) in [
            ("Deriv2(0) / Natural", Deriv2(0.0)),
            ("ZeroCurvBC", ZeroCurvBC()),
            ("ZeroSlopeBC", ZeroSlopeBC()),
            ("QuadraticFit", QuadraticFit()),
        ]
        _, _, ok = dot_product_test_nd((x, y), (xq, yq), f, y_bar; bc = bc)
        @test ok
    end

    @testset "Mixed BCs (CubicFit × ZeroCurvBC)" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar; bc = (CubicFit(), ZeroCurvBC())
        )
        @test ok
    end

    @testset "Mixed BCs (ZeroSlopeBC × QuadraticFit)" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar; bc = (ZeroSlopeBC(), QuadraticFit())
        )
        @test ok
    end

    # Non-zero BC values — adjoint should be independent of the affine constant
    @testset "BC: Deriv1(0.5)" begin
        _, _, ok = dot_product_test_nd((x, y), (xq, yq), f, y_bar; bc = Deriv1(0.5))
        @test ok
    end

    @testset "BC: Deriv3(1.0)" begin
        _, _, ok = dot_product_test_nd((x, y), (xq, yq), f, y_bar; bc = Deriv3(1.0))
        @test ok
    end
end

# ========================================
# PeriodicBC Tests
# ========================================

# Helper: generate periodic-compatible data for inclusive grids.
# Uses sin/cos basis functions, then explicitly enforces f[end,...] = f[1,...]
# to avoid floating-point mismatch (sin(2π) ≈ -2.45e-16, not exactly 0).
function _make_periodic_data_inclusive(grids)
    N = length(grids)
    sizes = ntuple(d -> length(grids[d]), Val(N))
    f = zeros(sizes)
    for I in CartesianIndices(f)
        val = 0.0
        for d in 1:N
            x_d = grids[d][I[d]]
            x_min = first(grids[d])
            period = last(grids[d]) - first(grids[d])
            val += sin(2π * (x_d - x_min) / period) + 0.5 * cos(4π * (x_d - x_min) / period)
        end
        f[I] = val
    end
    # Enforce strict periodicity: f[end,...] = f[1,...] for each axis
    for d in 1:N
        selectdim(f, d, sizes[d]) .= selectdim(f, d, 1)
    end
    return f
end

@testset "CubicAdjointND — PeriodicBC" begin
    # ========================================
    # Inclusive PeriodicBC — N=2
    # ========================================
    @testset "PeriodicBC (inclusive) — N=2" begin
        nx, ny = 15, 12
        n_query = 30

        x = range(0.0, 1.0, nx)
        y = range(0.0, 2.0, ny)
        f = _make_periodic_data_inclusive((x, y))
        xq = sort(rand(n_query)) .* 0.96 .+ 0.02
        yq = sort(rand(n_query)) .* 1.92 .+ 0.04
        y_bar = randn(n_query)

        @testset "Dot-product — uniform" begin
            _, _, ok = dot_product_test_nd(
                (x, y), (xq, yq), f, y_bar; bc = PeriodicBC()
            )
            @test ok
        end

        @testset "Dot-product — non-uniform" begin
            x_nu = cumsum(0.5 .+ rand(nx))
            x_nu .= (x_nu .- x_nu[1]) ./ (x_nu[end] - x_nu[1])
            y_nu = cumsum(0.5 .+ rand(ny))
            y_nu .= (y_nu .- y_nu[1]) ./ (y_nu[end] - y_nu[1]) .* 2.0
            grids_nu = (collect(x_nu), collect(y_nu))
            f_nu = _make_periodic_data_inclusive(grids_nu)
            _, _, ok = dot_product_test_nd(
                grids_nu, (xq, yq), f_nu, y_bar; bc = PeriodicBC()
            )
            @test ok
        end

        @testset "In-place vs allocating" begin
            adj = cubic_adjoint((x, y), (xq, yq); bc = PeriodicBC())
            f_bar_alloc = adj(y_bar)
            f_bar_inplace = zeros(nx, ny)
            adj(f_bar_inplace, y_bar)
            @test f_bar_alloc ≈ f_bar_inplace
        end
    end

    # ========================================
    # Exclusive PeriodicBC — N=2
    # ========================================
    @testset "PeriodicBC (exclusive) — N=2" begin
        nx, ny = 14, 11  # exclusive: no repeated endpoint
        n_query = 25

        x = range(0.0, 1.0, nx + 1)[1:nx]  # [0, 1) with nx points
        y = range(0.0, 2.0, ny + 1)[1:ny]  # [0, 2) with ny points
        bc = PeriodicBC(; endpoint = :exclusive)

        # Periodic function on exclusive grid
        f = zeros(nx, ny)
        for j in 1:ny, i in 1:nx
            f[i, j] = sin(2π * x[i]) + cos(2π * y[j] / 2.0)
        end

        xq = sort(rand(n_query)) .* 0.96 .+ 0.02
        yq = sort(rand(n_query)) .* 1.92 .+ 0.04
        y_bar = randn(n_query)

        @testset "Dot-product" begin
            _, _, ok = dot_product_test_nd(
                (x, y), (xq, yq), f, y_bar; bc = bc
            )
            @test ok
        end

        @testset "Output size" begin
            adj = cubic_adjoint((x, y), (xq, yq); bc = bc)
            f_bar = adj(y_bar)
            @test size(f_bar) == (nx, ny)
        end

        @testset "In-place vs allocating" begin
            adj = cubic_adjoint((x, y), (xq, yq); bc = bc)
            f_bar_alloc = adj(y_bar)
            f_bar_inplace = zeros(nx, ny)
            adj(f_bar_inplace, y_bar)
            @test f_bar_alloc ≈ f_bar_inplace
        end
    end

    # ========================================
    # Inclusive PeriodicBC — N=3
    # ========================================
    @testset "PeriodicBC (inclusive) — N=3" begin
        nx, ny, nz = 10, 8, 6
        n_query = 20

        x = range(0.0, 1.0, nx)
        y = range(0.0, 1.0, ny)
        z = range(0.0, 1.0, nz)
        f = _make_periodic_data_inclusive((x, y, z))
        xq = sort(rand(n_query)) .* 0.96 .+ 0.02
        yq = sort(rand(n_query)) .* 0.96 .+ 0.02
        zq = sort(rand(n_query)) .* 0.96 .+ 0.02
        y_bar = randn(n_query)

        @testset "Dot-product" begin
            _, _, ok = dot_product_test_nd(
                (x, y, z), (xq, yq, zq), f, y_bar; bc = PeriodicBC()
            )
            @test ok
        end
    end

    # ========================================
    # Mixed BCs — PeriodicBC × non-periodic
    # ========================================
    @testset "Mixed BC (PeriodicBC × CubicFit) — N=2" begin
        nx, ny = 12, 10
        n_query = 25

        x = range(0.0, 1.0, nx)
        y = range(0.0, 1.0, ny)

        # Only x-axis is periodic: f[1,j] == f[end,j] required
        f = zeros(nx, ny)
        for j in 1:ny, i in 1:nx
            f[i, j] = sin(2π * x[i]) + y[j]^2
        end
        f[end, :] .= f[1, :]  # enforce strict periodicity on x-axis

        xq = sort(rand(n_query)) .* 0.96 .+ 0.02
        yq = sort(rand(n_query)) .* 0.96 .+ 0.02
        y_bar = randn(n_query)

        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar; bc = (PeriodicBC(), CubicFit())
        )
        @test ok
    end

    @testset "Mixed BC (CubicFit × PeriodicBC) — N=2" begin
        nx, ny = 12, 10
        n_query = 25

        x = range(0.0, 1.0, nx)
        y = range(0.0, 1.0, ny)

        # Only y-axis is periodic: f[i,1] == f[i,end] required
        f = zeros(nx, ny)
        for j in 1:ny, i in 1:nx
            f[i, j] = x[i]^2 + cos(2π * y[j])
        end
        f[:, end] .= f[:, 1]  # enforce strict periodicity on y-axis

        xq = sort(rand(n_query)) .* 0.96 .+ 0.02
        yq = sort(rand(n_query)) .* 0.96 .+ 0.02
        y_bar = randn(n_query)

        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar; bc = (CubicFit(), PeriodicBC())
        )
        @test ok
    end

    # ========================================
    # Matrix materialization — periodic (small grid, exclusive)
    # ========================================
    # Uses exclusive periodic: n-point input space with no endpoint constraints,
    # so unit vector probing works cleanly (no f[1]==f[end] requirement).
    @testset "Matrix materialization — periodic exclusive" begin
        sx, sy = 6, 5
        bc_excl = PeriodicBC(; endpoint = :exclusive)
        x_s = range(0.0, 1.0, sx + 1)[1:sx]
        y_s = range(0.0, 1.0, sy + 1)[1:sy]
        n_q = 8
        xq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        yq_s = sort(rand(n_q)) .* 0.96 .+ 0.02

        adj = cubic_adjoint((x_s, y_s), (xq_s, yq_s); bc = bc_excl)
        n_grid = sx * sy

        # Build Wᵀ by probing adjoint
        WT = zeros(n_grid, n_q)
        e_q = zeros(n_q)
        for q in 1:n_q
            e_q[q] = 1.0
            WT[:, q] = vec(adj(e_q))
            e_q[q] = 0.0
        end

        # Build W by probing forward
        W = zeros(n_q, n_grid)
        output = zeros(n_q)
        for k in 1:n_grid
            e_f = zeros(sx, sy)
            e_f[k] = 1.0
            itp = cubic_interp((x_s, y_s), e_f; bc = bc_excl)
            itp(output, (xq_s, yq_s))
            W[:, k] = output
        end

        @test W' ≈ WT rtol = 1.0e-12

        # Test Matrix() function directly
        WT_mat = Matrix(adj)
        @test WT_mat ≈ WT rtol = 1.0e-12
    end

    # ========================================
    # Type stability — periodic
    # ========================================
    @testset "Type stability — constructor (periodic)" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        xq = sort(rand(15)) .* 0.96 .+ 0.02
        yq = sort(rand(15)) .* 0.96 .+ 0.02

        @test @inferred(
            cubic_adjoint(
                (x, y), (xq, yq); bc = PeriodicBC()
            )
        ) isa CubicAdjointND
    end

    @testset "Type stability — apply (periodic)" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 1.0, 8)
        xq = sort(rand(15)) .* 0.96 .+ 0.02
        yq = sort(rand(15)) .* 0.96 .+ 0.02
        adj = cubic_adjoint((x, y), (xq, yq); bc = PeriodicBC())
        yb = randn(15)
        @test @inferred(adj(yb)) isa Matrix{Float64}
    end

    # ========================================
    # Zero-allocation — periodic inclusive in-place
    # ========================================
    function _test_nd_adjoint_alloc_periodic(grids, queries, f_bar, y_bar)
        adj = cubic_adjoint(grids, queries; bc = PeriodicBC())
        adj(f_bar, y_bar)  # warmup
        adj(f_bar, y_bar)  # warmup
        return @allocated adj(f_bar, y_bar)
    end

    @testset "Zero-alloc: periodic inclusive in-place" begin
        nx, ny = 12, 10
        n_query = 20
        x = range(0.0, 1.0, nx)
        y = range(0.0, 1.0, ny)
        xq = sort(rand(n_query)) .* 0.96 .+ 0.02
        yq = sort(rand(n_query)) .* 0.96 .+ 0.02
        fb = zeros(nx, ny)
        yb = randn(n_query)
        allocs = _test_nd_adjoint_alloc_periodic(
            (x, y), (xq, yq), fb, yb
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end
end

# ========================================
# Derivative Adjoint Tests (Phase 4)
# ========================================
# Verifies adjoint of derivative evaluation: adj(ȳ; deriv=DerivOp(k))
# Core identity: ⟨W_d·f, ȳ⟩ = ⟨f, W_dᵀ·ȳ⟩

@testset "CubicAdjointND — Derivative Adjoint" begin
    nx, ny = 12, 10
    n_query = 25
    x = range(0.0, 1.0, nx)
    y = range(0.0, 1.0, ny)
    xq = sort(rand(n_query)) .* 0.96 .+ 0.02
    yq = sort(rand(n_query)) .* 0.96 .+ 0.02
    f = randn(nx, ny)
    y_bar = randn(n_query)

    # ========================================
    # Dot-product identity — N=2
    # ========================================
    @testset "Dot-product — ∂f/∂x" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            deriv = (DerivOp(1), EvalValue())
        )
        @test ok
    end

    @testset "Dot-product — ∂f/∂y" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            deriv = (EvalValue(), DerivOp(1))
        )
        @test ok
    end

    @testset "Dot-product — broadcast DerivOp(1)" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            deriv = DerivOp(1)
        )
        @test ok
    end

    @testset "Dot-product — ∂²f/∂x²" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            deriv = (DerivOp(2), EvalValue())
        )
        @test ok
    end

    @testset "Dot-product — ∂²f/∂x∂y" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            deriv = (DerivOp(1), DerivOp(1))
        )
        @test ok
    end

    @testset "Dot-product — broadcast DerivOp(2)" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            deriv = DerivOp(2)
        )
        @test ok
    end

    @testset "Dot-product — broadcast DerivOp(3)" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            deriv = DerivOp(3)
        )
        @test ok
    end

    @testset "Dot-product — per-axis (DerivOp(3), EvalValue())" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            deriv = (DerivOp(3), EvalValue())
        )
        @test ok
    end

    # ========================================
    # BCs × derivative adjoint
    # ========================================
    @testset "Dot-product — deriv=1 + $bc_name" for (bc_name, bc) in [
            ("ZeroCurvBC", ZeroCurvBC()),
            ("ZeroSlopeBC", ZeroSlopeBC()),
            ("QuadraticFit", QuadraticFit()),
        ]
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            bc = bc, deriv = DerivOp(1)
        )
        @test ok
    end

    @testset "Dot-product — deriv=1 + mixed BCs (CubicFit × ZeroCurvBC)" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            bc = (CubicFit(), ZeroCurvBC()), deriv = DerivOp(1)
        )
        @test ok
    end

    # ========================================
    # In-place vs allocating consistency
    # ========================================
    @testset "In-place vs allocating — deriv=1" begin
        adj = cubic_adjoint((x, y), (xq, yq))
        f_bar_alloc = adj(y_bar; deriv = DerivOp(1))
        f_bar_inplace = zeros(nx, ny)
        adj(f_bar_inplace, y_bar; deriv = DerivOp(1))
        @test f_bar_alloc ≈ f_bar_inplace
    end

    @testset "In-place vs allocating — mixed deriv" begin
        adj = cubic_adjoint((x, y), (xq, yq))
        deriv_mixed = (DerivOp(1), EvalValue())
        f_bar_alloc = adj(y_bar; deriv = deriv_mixed)
        f_bar_inplace = zeros(nx, ny)
        adj(f_bar_inplace, y_bar; deriv = deriv_mixed)
        @test f_bar_alloc ≈ f_bar_inplace
    end

    # ========================================
    # Matrix materialization with deriv
    # ========================================
    @testset "Matrix materialization — deriv=1" begin
        sx, sy = 5, 4
        x_s = range(0.0, 1.0, sx)
        y_s = range(0.0, 1.0, sy)
        n_q = 8
        xq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        yq_s = sort(rand(n_q)) .* 0.96 .+ 0.02
        deriv_ops = (DerivOp(1), EvalValue())

        adj = cubic_adjoint((x_s, y_s), (xq_s, yq_s))

        # Build Wᵀ by probing adjoint with unit vectors
        WT = zeros(sx * sy, n_q)
        e_q = zeros(n_q)
        for q in 1:n_q
            e_q[q] = 1.0
            WT[:, q] = vec(adj(e_q; deriv = deriv_ops))
            e_q[q] = 0.0
        end

        # Build W by probing forward with unit vectors
        W = zeros(n_q, sx * sy)
        output = zeros(n_q)
        for k in 1:(sx * sy)
            e_f = zeros(sx, sy)
            e_f[k] = 1.0
            itp = cubic_interp((x_s, y_s), e_f)
            itp(output, (xq_s, yq_s); deriv = deriv_ops)
            W[:, k] = output
        end

        @test W' ≈ WT rtol = 1.0e-12

        # Test Matrix() function directly with deriv
        WT_mat = Matrix(adj; deriv = deriv_ops)
        @test WT_mat ≈ WT rtol = 1.0e-12
    end

    # ========================================
    # Type stability
    # ========================================
    @testset "Type stability — deriv=1" begin
        adj = cubic_adjoint((x, y), (xq, yq))
        @test @inferred(adj(y_bar; deriv = DerivOp(1))) isa Matrix{Float64}
    end

    @testset "Type stability — mixed deriv" begin
        adj = cubic_adjoint((x, y), (xq, yq))
        @test @inferred(adj(y_bar; deriv = (DerivOp(1), EvalValue()))) isa Matrix{Float64}
    end

    @testset "Type stability — deriv=1 Float32" begin
        x32 = range(0.0f0, 1.0f0, nx)
        y32 = range(0.0f0, 1.0f0, ny)
        xq32 = Float32.(xq)
        yq32 = Float32.(yq)
        yb32 = randn(Float32, n_query)
        adj32 = cubic_adjoint((x32, y32), (xq32, yq32))
        @test @inferred(adj32(yb32; deriv = DerivOp(1))) isa Matrix{Float32}
    end

    # ========================================
    # Dot-product — non-uniform grids + deriv
    # ========================================
    @testset "Dot-product — deriv=1 + non-uniform grids" begin
        x_nu = cumsum(0.5 .+ rand(nx))
        x_nu .= (x_nu .- x_nu[1]) ./ (x_nu[end] - x_nu[1])
        y_nu = cumsum(0.5 .+ rand(ny))
        y_nu .= (y_nu .- y_nu[1]) ./ (y_nu[end] - y_nu[1])
        f_nu = randn(nx, ny)
        _, _, ok = dot_product_test_nd(
            (collect(x_nu), collect(y_nu)), (xq, yq), f_nu, y_bar;
            deriv = DerivOp(1)
        )
        @test ok
    end

    # ========================================
    # Zero-allocation (in-place)
    # ========================================
    function _test_nd_adjoint_alloc_deriv(grids, queries, f_bar, y_bar, deriv)
        adj = cubic_adjoint(grids, queries)
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: in-place deriv=1" begin
        fb = zeros(nx, ny)
        allocs = _test_nd_adjoint_alloc_deriv(
            (x, y), (xq, yq), fb, y_bar, DerivOp(1)
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc: in-place mixed deriv" begin
        fb = zeros(nx, ny)
        allocs = _test_nd_adjoint_alloc_deriv(
            (x, y), (xq, yq), fb, y_bar, (DerivOp(1), EvalValue())
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end
end

# ========================================
# Derivative Adjoint — N=3
# ========================================

@testset "CubicAdjointND — Derivative Adjoint (N=3)" begin
    nx, ny, nz = 8, 7, 6
    n_query = 15
    x = range(0.0, 1.0, nx)
    y = range(0.0, 1.0, ny)
    z = range(0.0, 1.0, nz)
    xq = sort(rand(n_query)) .* 0.96 .+ 0.02
    yq = sort(rand(n_query)) .* 0.96 .+ 0.02
    zq = sort(rand(n_query)) .* 0.96 .+ 0.02
    f = randn(nx, ny, nz)
    y_bar = randn(n_query)

    @testset "Dot-product — broadcast DerivOp(1)" begin
        _, _, ok = dot_product_test_nd(
            (x, y, z), (xq, yq, zq), f, y_bar;
            deriv = DerivOp(1)
        )
        @test ok
    end

    @testset "Dot-product — mixed (DerivOp(1), EvalValue, DerivOp(2))" begin
        _, _, ok = dot_product_test_nd(
            (x, y, z), (xq, yq, zq), f, y_bar;
            deriv = (DerivOp(1), EvalValue(), DerivOp(2))
        )
        @test ok
    end

    @testset "Type stability — deriv=1" begin
        adj = cubic_adjoint((x, y, z), (xq, yq, zq))
        @test @inferred(adj(y_bar; deriv = DerivOp(1))) isa Array{Float64, 3}
    end
end

# ========================================
# Derivative Adjoint — Periodic BCs
# ========================================

@testset "CubicAdjointND — Derivative Adjoint + Periodic" begin
    nx, ny = 12, 10
    n_query = 20
    x = range(0.0, 2π, nx)
    y = range(0.0, 2π, ny)
    xq = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
    yq = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
    f = _make_periodic_data_inclusive((x, y))
    y_bar = randn(n_query)

    @testset "Dot-product — ∂f/∂x + PeriodicBC" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            bc = PeriodicBC(), deriv = (DerivOp(1), EvalValue())
        )
        @test ok
    end

    @testset "Dot-product — ∂f/∂y + PeriodicBC" begin
        _, _, ok = dot_product_test_nd(
            (x, y), (xq, yq), f, y_bar;
            bc = PeriodicBC(), deriv = (EvalValue(), DerivOp(1))
        )
        @test ok
    end

    @testset "Dot-product — mixed deriv + mixed BC (PeriodicBC × CubicFit)" begin
        x_cf = range(0.0, 1.0, nx)
        y_cf = range(0.0, 1.0, ny)
        xq_cf = sort(rand(n_query)) .* 0.96 .+ 0.02
        yq_cf = sort(rand(n_query)) .* 0.96 .+ 0.02
        f_mix = randn(nx, ny)
        for j in 1:ny
            f_mix[end, j] = f_mix[1, j]
        end
        y_bar_m = randn(n_query)
        _, _, ok = dot_product_test_nd(
            (x, y_cf), (xq, yq_cf), f_mix, y_bar_m;
            bc = (PeriodicBC(), CubicFit()),
            deriv = (DerivOp(1), EvalValue())
        )
        @test ok
    end

    # ========================================
    # Exclusive periodic + deriv
    # ========================================
    @testset "Dot-product — ∂f/∂x + PeriodicBC (exclusive)" begin
        nx_e, ny_e = 14, 11
        x_e = range(0.0, 2π, nx_e + 1)[1:nx_e]
        y_e = range(0.0, 2π, ny_e + 1)[1:ny_e]
        bc_excl = PeriodicBC(; endpoint = :exclusive)
        f_e = zeros(nx_e, ny_e)
        for j in 1:ny_e, i in 1:nx_e
            f_e[i, j] = sin(x_e[i]) + cos(y_e[j])
        end
        xq_e = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
        yq_e = sort(rand(n_query)) .* (2π * 0.96) .+ (2π * 0.02)
        yb_e = randn(n_query)
        _, _, ok = dot_product_test_nd(
            (x_e, y_e), (xq_e, yq_e), f_e, yb_e;
            bc = bc_excl, deriv = (DerivOp(1), EvalValue())
        )
        @test ok
    end

    # ========================================
    # Zero-allocation — periodic + deriv
    # ========================================
    function _test_nd_adjoint_alloc_periodic_deriv(grids, queries, f_bar, y_bar, deriv)
        adj = cubic_adjoint(grids, queries; bc = PeriodicBC())
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        adj(f_bar, y_bar; deriv = deriv)  # warmup
        return @allocated adj(f_bar, y_bar; deriv = deriv)
    end

    @testset "Zero-alloc: periodic inclusive + deriv=1" begin
        fb = zeros(nx, ny)
        allocs = _test_nd_adjoint_alloc_periodic_deriv(
            (x, y), (xq, yq), fb, y_bar, DerivOp(1)
        )
        @test allocs <= ND_ALLOC_THRESHOLD
    end
end
