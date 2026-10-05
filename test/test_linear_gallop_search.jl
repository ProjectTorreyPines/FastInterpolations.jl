@testitem "LinearGallopSearch" setup = [Basic, AllocConstants] begin
    using FastInterpolations: search_interval, _search_binary, Searcher, RefHint, _to_searcher,
        AbstractSearchPolicy, InBounds
    using Random

    # Queries whose cell differs from a stateless binary search (empty = all agree).
    function mismatches(s, x, qs, ext...)
        bad = eltype(qs)[]
        for q in qs
            idx, _, _, _ = search_interval(s, x, q, ext...)
            (idx == first(_search_binary(x, q)) && s.hint.idx[] == idx) || push!(bad, q)
        end
        return bad
    end

    rng = MersenneTwister(7)
    xg = cumsum(rand(rng, 1000) .+ 0.01)
    lo, hi = xg[1], xg[end]
    span(n) = lo .+ (hi - lo) .* rand(rng, n)

    @testset "Construction" begin
        @test LinearGallopSearch{8} <: AbstractSearchPolicy
        @test LinearGallopSearch() === LinearGallopSearch{8}()
        for w in (0, 1, 2, 4, 8, 16, 32, 64, 128)
            @test LinearGallopSearch(linear_window = w) === LinearGallopSearch{w}()
            @test LinearGallopSearch(w) === LinearGallopSearch{w}()
        end
        @test_throws ArgumentError LinearGallopSearch(linear_window = 3)
        @test_throws ArgumentError LinearGallopSearch(256)
    end

    @testset "Searcher" begin
        @test _to_searcher(LinearGallopSearch()) isa Searcher{LinearGallopSearch{8}, RefHint}
        h = Ref(5)
        s = _to_searcher(LinearGallopSearch{2}(), h)
        @test s isa Searcher{LinearGallopSearch{2}, RefHint}
        @test s.hint.idx === h
    end

    @testset "Same cell as binary search for any jump" begin
        # Random jumps both ways, grid points, endpoints, out-of-domain, long ascending and
        # descending runs (gallop right / left repeatedly).
        qs = [span(300); rand(rng, xg, 50); lo; hi; lo - 1; hi + 1; sort(span(300)); sort(span(100); rev = true)]
        for MAX in (0, 2, 8, 32)
            s = Searcher{LinearGallopSearch{MAX}, RefHint}(RefHint(Ref(1)))
            @test isempty(mismatches(s, xg, qs))
            s.hint.idx[] = 1
            @test isempty(mismatches(s, xg, filter(q -> lo <= q <= hi, qs), InBounds()))
        end
        # Int grid with Float queries (promoting comparisons)
        xi = cumsum(rand(rng, 1:5, 300))
        qi = sort(xi[1] .+ (xi[end] - xi[1]) .* rand(rng, 40))
        @test isempty(mismatches(Searcher{LinearGallopSearch{8}, RefHint}(RefHint(Ref(1))), xi, qi))
    end

    # API level: every family must agree with BinarySearch. Same maths on two code paths,
    # so values are compared in ULP (setup.jl `PATH_ULPS`), not bits.
    x = cumsum(rand(rng, 2000) .+ 0.05)
    y = sin.(x ./ 20)
    a, b = x[1], x[end]
    q_dense = sort(a .+ (b - a) .* rand(rng, 500))   # ≈ 4 cells apart: walk + some gallop
    q_sparse = sort(a .+ (b - a) .* rand(rng, 40))   # ≈ 50 cells apart: gallop
    q_rand = a .+ (b - a) .* rand(rng, 300)
    G = LinearGallopSearch()
    B = BinarySearch()

    @testset "API parity with BinarySearch: $(nameof(build))" for build in (
            linear_interp, constant_interp, quadratic_interp, cubic_interp, pchip_interp,
        )
        itp = build(x, y)
        for q in (q_dense, q_sparse, q_rand)
            @test isclose(itp(q; search = G), itp(q; search = B); nulps = PATH_ULPS)
            out_g, out_b = similar(q), similar(q)
            itp(out_g, q; search = G)
            itp(out_b, q; search = B)
            @test isclose(out_g, out_b; nulps = PATH_ULPS)
        end
        @test isclose(build(x, y; search = G)(q_sparse), itp(q_sparse; search = B); nulps = PATH_ULPS)
    end

    @testset "One-shot, Series, ND" begin
        @test isclose(linear_interp(x, y, q_sparse; search = G), linear_interp(x, y, q_sparse; search = B); nulps = PATH_ULPS)
        @test isclose(cubic_interp(x, y, q_sparse; search = G), cubic_interp(x, y, q_sparse; search = B); nulps = SOLVE_ULPS)
        Ys = Series(hcat(y, 2 .* y))
        @test all(isclose.(linear_interp(x, Ys, q_sparse; search = G), linear_interp(x, Ys, q_sparse; search = B); nulps = PATH_ULPS))

        gx = cumsum(rand(rng, 300) .+ 0.05)
        gy = cumsum(rand(rng, 200) .+ 0.05)
        Z = [sin(u / 10) * cos(v / 10) for u in gx, v in gy]
        qx = sort(gx[1] .+ (gx[end] - gx[1]) .* rand(rng, 60))
        qy = sort(gy[1] .+ (gy[end] - gy[1]) .* rand(rng, 60))
        ref = linear_interp((gx, gy), Z; search = B)
        @test isclose(linear_interp((gx, gy), Z; search = (G, B))((qx, qy)), ref((qx, qy)); nulps = PATH_ULPS)
        @test isclose(linear_interp((gx, gy), Z; search = G)((qx[7], qy[7])), ref((qx[7], qy[7])); nulps = PATH_ULPS)
    end

    @testset "Hint write-back (scalar)" begin
        itp = linear_interp(x, y)
        h = Ref(1)
        for q in q_sparse
            itp(q; search = G, hint = h)
            @test h[] == first(_search_binary(x, q))
        end
    end

    @testset "Out-of-domain with extrapolation" begin
        q_oob = sort([a - 5; a .+ (b - a) .* rand(rng, 50); b + 5])
        for ext in (ExtendExtrap(), ClampExtrap())
            itp = linear_interp(x, y; extrap = ext)
            @test isclose(itp(q_oob; search = G), itp(q_oob; search = B); nulps = PATH_ULPS)
        end
    end

    @testset "In-place eval allocates nothing" begin
        itp = linear_interp(x, y)
        out = similar(q_sparse)
        run!(itp, out, q) = itp(out, q; search = LinearGallopSearch())
        run!(itp, out, q_sparse)
        run!(itp, out, q_sparse)
        @test (@allocated run!(itp, out, q_sparse)) <= ALLOC_THRESHOLD
    end
end
