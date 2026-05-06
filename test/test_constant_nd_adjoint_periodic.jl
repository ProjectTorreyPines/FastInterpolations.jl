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

    # ─────────────────────────────────────────────────────────────────────────
    # Type stability — pins the closure-over-Tg regression precedent.
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Type stability — @inferred(constant_adjoint(...; bc=PeriodicBC))" begin
        x_r_inc = range(0.0, 1.0, 10)
        y_r_inc = range(0.0, 2.0, 8)
        x_r_exc = collect(range(0.0, step = 1.0 / 10, length = 10))
        xqs = (rand(5), rand(5) .* 2)

        bc_excl = (PeriodicBC(endpoint = :exclusive, period = 1.0), NoBC())
        bc_incl = (PeriodicBC(), PeriodicBC())
        @test @inferred(constant_adjoint((x_r_exc, y_r_inc), xqs; bc = bc_excl)) isa ConstantAdjointND
        @test @inferred(constant_adjoint((x_r_inc, y_r_inc), xqs; bc = bc_incl)) isa ConstantAdjointND

        xv = sort(rand(10)) .* 0.95
        yv = sort(rand(8)) .* 1.95
        xqs_v = (rand(5) .* xv[end], rand(5) .* yv[end])
        bc_v = (
            PeriodicBC(endpoint = :exclusive, period = 1.0),
            PeriodicBC(endpoint = :exclusive, period = 2.0),
        )
        @test @inferred(constant_adjoint((xv, yv), xqs_v; bc = bc_v)) isa ConstantAdjointND
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Allocation regression — `adj(f_bar, y_bar)` zero-alloc on periodic path.
    # `:exclusive` finalize uses compile-time-unrolled seam fold + view trim,
    # both specialized via `Val{d}` for static-dim dispatch.
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Allocation — periodic adj in-place is zero-alloc" begin
        function alloc_test(adj, f_bar, y_bar)
            adj(f_bar, y_bar)
            adj(f_bar, y_bar)
            return @allocated adj(f_bar, y_bar)
        end

        n_q = 32
        let x = collect(range(0.0, step = 1.0 / 12, length = 12)),
                y = collect(range(0.0, step = 2.0 / 10, length = 10)),
                xqs = (rand(n_q), rand(n_q) .* 2),
                bc = (
                PeriodicBC(endpoint = :exclusive, period = 1.0),
                PeriodicBC(endpoint = :exclusive, period = 2.0),
            )
            adj = constant_adjoint((x, y), xqs; bc = bc)
            f_bar = zeros(12, 10)
            y_bar = ones(n_q)
            @test alloc_test(adj, f_bar, y_bar) == 0
        end

        let x = collect(range(0.0, step = 1.0 / 14, length = 14)),
                y = range(0.0, 1.0, 11),
                xqs = (rand(n_q), rand(n_q)),
                bc = (PeriodicBC(endpoint = :exclusive, period = 1.0), NoBC())
            adj = constant_adjoint((x, y), xqs; bc = bc)
            f_bar = zeros(14, 11)
            y_bar = ones(n_q)
            @test alloc_test(adj, f_bar, y_bar) == 0
        end

        let x = range(0.0, 1.0, 12),
                y = range(0.0, 2.0, 10),
                xqs = (rand(n_q), rand(n_q) .* 2),
                bc = (PeriodicBC(), PeriodicBC())
            adj = constant_adjoint((x, y), xqs; bc = bc)
            f_bar = zeros(12, 10)
            y_bar = ones(n_q)
            @test alloc_test(adj, f_bar, y_bar) == 0
        end
    end
end
