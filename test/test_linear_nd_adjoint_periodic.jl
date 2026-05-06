@testitem "LinearAdjointND PeriodicBC dot-product identity" begin
    using LinearAlgebra: dot

    # ─────────────────────────────────────────────────────────────────────────
    # Gold-standard adjoint test: ⟨W·f, ȳ⟩ == ⟨f, Wᵀ·ȳ⟩.
    #
    # Linear interpolation is purely linear in `f` (no BC constant offset),
    # so this identity is exact up to floating-point round-off — including
    # for periodic BCs. If the adjoint correctly mirrors the forward weights
    # and seam fold, the identity must hold. Any drift indicates a bug in
    # the periodic-aware anchor baking, scatter, or post-apply finalize.
    # ─────────────────────────────────────────────────────────────────────────

    function dot_id_test(grids, xqs, f, y_bar; bc, extrap = NoExtrap(), rtol = 1.0e-9)
        itp = linear_interp(grids, f; bc = bc, extrap = extrap)
        adj = linear_adjoint(grids, xqs; bc = bc, extrap = extrap)

        n_queries = length(xqs[1])
        Wf = Vector{eltype(f)}(undef, n_queries)
        itp(Wf, xqs)

        WTy = adj(y_bar)
        # Output shape must match the user-supplied grid shape (post-trim).
        @test size(WTy) == size(f)

        return isapprox(dot(Wf, y_bar), dot(vec(f), vec(WTy)); rtol = rtol)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 2D — both axes periodic
    # ─────────────────────────────────────────────────────────────────────────
    n_query = 40

    @testset "PeriodicBC{:inclusive} — Range × Range" begin
        nx, ny = 12, 10
        x = range(0.0, 1.0, nx)
        y = range(0.0, 2.0, ny)
        # `:inclusive` requires data[1, :] == data[end, :] and data[:, 1] == data[:, end]
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
        # `:exclusive` user-grid has length n (no closing point); period from kwarg
        x = collect(range(0.0, step = 1.0 / nx, length = nx))
        y = collect(range(0.0, step = 2.0 / ny, length = ny))
        f = randn(nx, ny)
        # Queries strictly inside the closed period [first, first+period)
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
        # Non-uniform Vector grids over [0, 1) × [0, 2)
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
        x = collect(range(0.0, step = 1.0 / nx, length = nx))  # exclusive periodic
        y = range(0.0, 1.0, ny)                                 # NoBC
        f = randn(nx, ny)
        xqs = (rand(n_query), rand(n_query))
        y_bar = randn(n_query)
        bc = (PeriodicBC(endpoint = :exclusive, period = 1.0), NoBC())
        @test dot_id_test((x, y), xqs, f, y_bar; bc = bc)
    end

    @testset "Mixed — :inclusive × :exclusive (Range × Vector)" begin
        nx, ny = 9, 8
        x = range(0.0, 1.0, nx)                       # inclusive
        y_pts = sort(rand(ny)) .* 1.95                 # exclusive Vector
        f = randn(nx, ny)
        f[end, :] .= f[1, :]                           # `:inclusive` constraint
        xqs = (rand(n_query), rand(n_query) .* y_pts[end])
        y_bar = randn(n_query)
        bc = (PeriodicBC(), PeriodicBC(endpoint = :exclusive, period = 2.0))
        @test dot_id_test((x, y_pts), xqs, f, y_bar; bc = bc)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 3D — :exclusive × :inclusive × NoBC mixed
    # ─────────────────────────────────────────────────────────────────────────
    @testset "3D mixed BC — :exclusive × :inclusive × NoBC" begin
        nx, ny, nz = 8, 7, 6
        x = collect(range(0.0, step = 1.0 / nx, length = nx))
        y = range(0.0, 2.0, ny)                                 # inclusive
        z = range(0.0, 1.0, nz)                                 # NoBC
        f = randn(nx, ny, nz)
        f[:, end, :] .= f[:, 1, :]                              # axis-2 inclusive constraint
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

    # ─────────────────────────────────────────────────────────────────────────
    # Seam-cell coverage: query lands EXACTLY in the seam cell on a periodic axis
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Seam-cell queries (axis 1 :exclusive, query > x[end])" begin
        nx, ny = 10, 8
        x = collect(range(0.0, step = 1.0 / nx, length = nx))   # x[end] = 0.9
        y = range(0.0, 1.0, ny)
        f = randn(nx, ny)
        # Half of queries inside the seam cell [x[end], first+period)
        xqs = (
            vcat(rand(20) .* (1.0 - 1.0 / nx), rand(20) .* (1.0 / nx) .+ x[end]),
            rand(40),
        )
        y_bar = randn(40)
        bc = (PeriodicBC(endpoint = :exclusive, period = 1.0), NoBC())
        @test dot_id_test((x, y), xqs, f, y_bar; bc = bc)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Type stability — `@inferred` over the periodic outer ctor.
    # Pins the closure-over-Tg regression that previously broke ND inferrability;
    # see MEMORY.md "ND Constructor Inferrability Pattern".
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Type stability — @inferred(linear_adjoint(...; bc=PeriodicBC))" begin
        # Range grid (`:inclusive` requires closed-cycle [0, 1]/[0, 2])
        x_r_inc = range(0.0, 1.0, 10)
        y_r_inc = range(0.0, 2.0, 8)
        # Range grid (`:exclusive` half-open [0, 1)/[0, 2))
        x_r_exc = collect(range(0.0, step = 1.0 / 10, length = 10))
        y_r_exc = collect(range(0.0, step = 2.0 / 8, length = 8))
        # Queries strictly inside the closed period
        xqs = (rand(5), rand(5) .* 2)

        bc_excl = (PeriodicBC(endpoint = :exclusive, period = 1.0), NoBC())
        bc_incl = (PeriodicBC(), PeriodicBC())
        @test @inferred(linear_adjoint((x_r_exc, y_r_inc), xqs; bc = bc_excl)) isa LinearAdjointND
        @test @inferred(linear_adjoint((x_r_inc, y_r_inc), xqs; bc = bc_incl)) isa LinearAdjointND

        # Vector grid, both axes :exclusive
        xv = sort(rand(10)) .* 0.95
        yv = sort(rand(8)) .* 1.95
        xqs_v = (rand(5) .* xv[end], rand(5) .* yv[end])
        bc_v = (
            PeriodicBC(endpoint = :exclusive, period = 1.0),
            PeriodicBC(endpoint = :exclusive, period = 2.0),
        )
        @test @inferred(linear_adjoint((xv, yv), xqs_v; bc = bc_v)) isa LinearAdjointND
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Allocation regression — `adj(f_bar, y_bar)` must remain zero-alloc on
    # the in-place call across all periodic BC combinations. The `:exclusive`
    # path runs `_adjoint_apply_exclusive_nd!` (pool-acquired work buffer +
    # compile-time-unrolled seam fold + view trim); any non-zero allocation
    # indicates a regression in static-dim specialization.
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Allocation — periodic adj in-place is zero-alloc" begin
        function alloc_test(adj, f_bar, y_bar)
            adj(f_bar, y_bar)            # warm-up
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
            adj = linear_adjoint((x, y), xqs; bc = bc)
            f_bar = zeros(12, 10)
            y_bar = ones(n_q)
            @test alloc_test(adj, f_bar, y_bar) == 0
        end

        let x = collect(range(0.0, step = 1.0 / 14, length = 14)),
                y = range(0.0, 1.0, 11),
                xqs = (rand(n_q), rand(n_q)),
                bc = (PeriodicBC(endpoint = :exclusive, period = 1.0), NoBC())
            adj = linear_adjoint((x, y), xqs; bc = bc)
            f_bar = zeros(14, 11)
            y_bar = ones(n_q)
            @test alloc_test(adj, f_bar, y_bar) == 0
        end

        let x = range(0.0, 1.0, 12),
                y = range(0.0, 2.0, 10),
                xqs = (rand(n_q), rand(n_q) .* 2),
                bc = (PeriodicBC(), PeriodicBC())
            adj = linear_adjoint((x, y), xqs; bc = bc)
            f_bar = zeros(12, 10)
            y_bar = ones(n_q)
            @test alloc_test(adj, f_bar, y_bar) == 0
        end
    end
end
