@testitem "ConstantAdjointND PeriodicBC dot-product identity" begin
    using LinearAlgebra: dot

    # Constant interp is single-point (no blending), so the dot-product
    # identity ⟨W·f, ȳ⟩ == ⟨f, Wᵀ·ȳ⟩ is not just within rtol but exact —
    # both sides are sums over a 1-1 mapping query → grid index.
    function dot_id_test(grids, xqs, f, y_bar; bc, side = NearestSide(), extrap = NoExtrap())
        itp = constant_interp(grids, f; bc = bc, side = side, extrap = extrap)
        adj = constant_adjoint(grids, xqs; bc = bc, side = side, extrap = extrap)

        n_queries = length(xqs[1])
        Wf = Vector{eltype(f)}(undef, n_queries)
        itp(Wf, xqs)

        WTy = adj(y_bar)
        @test size(WTy) == size(f)

        return isapprox(dot(Wf, y_bar), dot(vec(f), vec(WTy)); rtol = 1.0e-12)
    end

    n_query = 40

    @testset "PeriodicBC{:inclusive} — Range × Range" begin
        nx, ny = 12, 10
        x = range(0.0, 1.0, nx)
        y = range(0.0, 2.0, ny)
        f = randn(nx, ny)
        f[end, :] .= f[1, :]
        f[:, end] .= f[:, 1]
        xqs = (rand(n_query), rand(n_query) .* 2)
        y_bar = randn(n_query)
        bc = (PeriodicBC(), PeriodicBC())
        @test dot_id_test((x, y), xqs, f, y_bar; bc = bc)
    end

    @testset "PeriodicBC{:exclusive} — Range × Range" begin
        nx, ny = 12, 10
        x = collect(range(0.0, step = 1.0 / nx, length = nx))
        y = collect(range(0.0, step = 2.0 / ny, length = ny))
        f = randn(nx, ny)
        xqs = (rand(n_query), rand(n_query) .* 2)
        y_bar = randn(n_query)
        bc = (
            PeriodicBC(endpoint = :exclusive, period = 1.0),
            PeriodicBC(endpoint = :exclusive, period = 2.0),
        )
        @test dot_id_test((x, y), xqs, f, y_bar; bc = bc)
    end

    @testset "PeriodicBC{:exclusive} — Vector × Vector" begin
        nx, ny = 11, 9
        x = sort(rand(nx)) .* 0.95
        y = sort(rand(ny)) .* 1.95
        f = randn(nx, ny)
        xqs = (rand(n_query) .* x[end], rand(n_query) .* y[end])
        y_bar = randn(n_query)
        bc = (
            PeriodicBC(endpoint = :exclusive, period = 1.0),
            PeriodicBC(endpoint = :exclusive, period = 2.0),
        )
        @test dot_id_test((x, y), xqs, f, y_bar; bc = bc)
    end

    @testset "Mixed — :exclusive × NoBC (Range × Range)" begin
        nx, ny = 14, 11
        x = collect(range(0.0, step = 1.0 / nx, length = nx))
        y = range(0.0, 1.0, ny)
        f = randn(nx, ny)
        xqs = (rand(n_query), rand(n_query))
        y_bar = randn(n_query)
        bc = (PeriodicBC(endpoint = :exclusive, period = 1.0), NoBC())
        @test dot_id_test((x, y), xqs, f, y_bar; bc = bc)
    end

    @testset "3D mixed BC — :exclusive × :inclusive × NoBC" begin
        nx, ny, nz = 8, 7, 6
        x = collect(range(0.0, step = 1.0 / nx, length = nx))
        y = range(0.0, 2.0, ny)
        z = range(0.0, 1.0, nz)
        f = randn(nx, ny, nz)
        f[:, end, :] .= f[:, 1, :]
        n_q3 = 30
        xqs = (rand(n_q3), rand(n_q3) .* 2, rand(n_q3))
        y_bar = randn(n_q3)
        bc = (
            PeriodicBC(endpoint = :exclusive, period = 1.0),
            PeriodicBC(),
            NoBC(),
        )
        @test dot_id_test((x, y, z), xqs, f, y_bar; bc = bc)
    end

    @testset "Side modes — LeftSide × RightSide on :exclusive" begin
        nx, ny = 10, 8
        x = collect(range(0.0, step = 1.0 / nx, length = nx))
        y = collect(range(0.0, step = 2.0 / ny, length = ny))
        f = randn(nx, ny)
        xqs = (rand(n_query), rand(n_query) .* 2)
        y_bar = randn(n_query)
        bc = (
            PeriodicBC(endpoint = :exclusive, period = 1.0),
            PeriodicBC(endpoint = :exclusive, period = 2.0),
        )
        @test dot_id_test(
            (x, y), xqs, f, y_bar;
            bc = bc, side = (LeftSide(), RightSide())
        )
    end
end
