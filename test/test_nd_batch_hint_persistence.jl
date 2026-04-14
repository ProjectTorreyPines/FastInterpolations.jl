# ═══════════════════════════════════════════════════════════════
# ND batch hint persistence — zero-alloc + hint mutation contract
# ═══════════════════════════════════════════════════════════════
#
# Verifies that the generic ND batch callable in `interpolant_protocol.jl`
# auto-creates persistent hint Refs at batch entry (via `_ensure_hint_nd`),
# eliminating:
#   1. Union boxing from `_resolve_search_nd` (Finding #1)
#   2. Per-query `RefHint()` heap allocation (Finding #2)
#
# Uses VECTOR grids (not Range) — Range grids use O(1) DirectSearch and
# bypass the hint/search machinery entirely.
#
# Function-barrier pattern is mandatory: @testset wraps body in try/catch
# → makes locals type-unstable → @allocated shows artifacts.

@testset "ND batch hint persistence" begin

    # ── Vector grid fixtures ─────────────────────────────────────
    # Slightly perturbed uniform grids → Vector{Float64}, not Range

    function _make_vector_grid(n, lo, hi)
        g = collect(range(lo, hi, length = n))
        if n > 2
            g[2:(end - 1)] .+= 1.0e-12 .* (1:(n - 2))  # deterministic perturbation
        end
        return g
    end

    # ── Allocation tests ─────────────────────────────────────────
    # All methods × 2D/3D must be zero-alloc on Vector grids.

    @testset "Zero-alloc — $label $nd" for (label, builder) in [
                ("Constant", (g, d) -> constant_interp(g, d)),
                ("Linear", (g, d) -> linear_interp(g, d)),
                ("Quadratic", (g, d) -> quadratic_interp(g, d)),
                ("Cubic", (g, d) -> cubic_interp(g, d)),
                ("PCHIP", (g, d) -> pchip_interp(g, d)),
                ("Cardinal", (g, d) -> cardinal_interp(g, d)),
                ("Akima", (g, d) -> akima_interp(g, d)),
            ], (nd, alloc_fn) in [
                (
                    "2D", function (builder)
                        x = _make_vector_grid(21, 0.0, 2π)
                        y = _make_vector_grid(11, 0.0, π)
                        data = [sin(xi) * cos(yj) for xi in x, yj in y]
                        itp = builder((x, y), data)
                        xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
                        yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
                        out = Vector{Float64}(undef, 5)
                        itp(out, (xqs, yqs)); itp(out, (xqs, yqs))
                        return @allocated itp(out, (xqs, yqs))
                end
                ),
                (
                    "3D", function (builder)
                        x = _make_vector_grid(21, 0.0, 2π)
                        y = _make_vector_grid(11, 0.0, π)
                        z = _make_vector_grid(7, 0.0, 1.0)
                        data = [sin(xi) * cos(yj) * (1 + zk) for xi in x, yj in y, zk in z]
                        itp = builder((x, y, z), data)
                        xqs = [0.5, 1.0, 1.5, 2.0, 3.0]
                        yqs = [0.2, 0.4, 0.6, 0.8, 1.0]
                        zqs = [0.1, 0.3, 0.5, 0.7, 0.9]
                        out = Vector{Float64}(undef, 5)
                        itp(out, (xqs, yqs, zqs)); itp(out, (xqs, yqs, zqs))
                        return @allocated itp(out, (xqs, yqs, zqs))
                end
                ),
            ]
        @test alloc_fn(builder) <= ND_ALLOC_THRESHOLD
    end

    # ── Heterogeneous method tuples (zero-alloc) ─────────────────

    @testset "Zero-alloc — Hetero $label" for (label, methods, ndims) in [
            ("Cubic×Linear 2D", (CubicInterp(), LinearInterp()), 2),
            ("Cardinal×Cubic 2D", (CardinalInterp(), CubicInterp()), 2),
            ("PCHIP×Linear 2D", (PchipInterp(), LinearInterp()), 2),
            ("Cubic×Linear×PCHIP 3D", (CubicInterp(), LinearInterp(), PchipInterp()), 3),
            ("Cardinal×Akima×Cubic 3D", (CardinalInterp(), AkimaInterp(), CubicInterp()), 3),
        ]
        function _alloc_hetero(methods, ndims)
            x = _make_vector_grid(21, 0.0, 2π)
            y = _make_vector_grid(11, 0.0, π)
            grids = ndims == 2 ? (x, y) : (x, y, _make_vector_grid(7, 0.0, 1.0))
            data = ndims == 2 ?
                [sin(xi) * cos(yj) for xi in grids[1], yj in grids[2]] :
                [sin(xi) * cos(yj) * (1 + zk) for xi in grids[1], yj in grids[2], zk in grids[3]]
            itp = interp(grids, data; method = methods)
            qs = ndims == 2 ?
                ([0.5, 1.0, 1.5, 2.0, 3.0], [0.2, 0.4, 0.6, 0.8, 1.0]) :
                ([0.5, 1.0, 1.5, 2.0, 3.0], [0.2, 0.4, 0.6, 0.8, 1.0], [0.1, 0.3, 0.5, 0.7, 0.9])
            out = Vector{Float64}(undef, 5)
            itp(out, qs); itp(out, qs)
            return @allocated itp(out, qs)
        end
        @test _alloc_hetero(methods, ndims) <= ND_ALLOC_THRESHOLD
    end

    # ── Mixed search policies (zero-alloc + correctness) ─────────

    @testset "Zero-alloc — Mixed search policies" begin
        function _test_mixed_policy(policy)
            x = _make_vector_grid(21, 0.0, 2π)
            y = _make_vector_grid(11, 0.0, π)
            z = _make_vector_grid(7, 0.0, 1.0)
            data = [sin(xi) * cos(yj) * (1 + zk) for xi in x, yj in y, zk in z]
            itp = cubic_interp((x, y, z), data)
            qs = (
                [0.5, 1.0, 1.5, 2.0, 3.0],
                [0.2, 0.4, 0.6, 0.8, 1.0],
                [0.1, 0.3, 0.5, 0.7, 0.9],
            )
            out = Vector{Float64}(undef, 5)
            ref = Vector{Float64}(undef, 5)

            # Reference: default search
            itp(ref, qs)

            # Mixed policy: correctness
            itp(out, qs; search = policy)
            @test out == ref

            # Mixed policy: allocation
            itp(out, qs; search = policy); itp(out, qs; search = policy)
            allocs = @allocated itp(out, qs; search = policy)
            @test allocs <= ND_ALLOC_THRESHOLD
        end

        _test_mixed_policy((BinarySearch(), AutoSearch(), BinarySearch()))
        _test_mixed_policy((AutoSearch(), BinarySearch(), AutoSearch()))
        _test_mixed_policy((LinearBinarySearch(), LinearBinarySearch(), BinarySearch()))
    end

    # ── Hint mutation tests ──────────────────────────────────────
    # Verify that auto-generated hints are updated across queries in batch.

    @testset "Hint mutation — $label" for (label, builder) in [
            ("Cubic", (g, d) -> cubic_interp(g, d)),
            ("Linear", (g, d) -> linear_interp(g, d)),
            ("Cardinal", (g, d) -> cardinal_interp(g, d)),
        ]

        # 2D: sorted queries should leave hints near end of grid
        x = _make_vector_grid(100, 0.0, 1.0)
        y = _make_vector_grid(50, 0.0, 1.0)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
        itp = builder((x, y), data)

        nq = 50
        qs_sorted = (
            collect(range(0.01, 0.99, length = nq)),
            collect(range(0.01, 0.99, length = nq)),
        )
        out = Vector{Float64}(undef, nq)

        # Auto-generated hints (hint=nothing): should still update internally
        # We can't observe auto hints directly, but we can verify no regression.
        itp(out, qs_sorted)  # should not error

        # User-provided hints: must be mutated
        hints = (Ref(1), Ref(1))
        itp(out, qs_sorted; hint = hints)

        # After sorted queries ending near 0.99, hints should be near end of grid
        @test hints[1][] > length(x) ÷ 2   # x hint advanced past midpoint
        @test hints[2][] > length(y) ÷ 2   # y hint advanced past midpoint
    end

    # ── User-provided hints preserved ────────────────────────────

    @testset "User hints not overwritten" begin
        x = _make_vector_grid(100, 0.0, 1.0)
        y = _make_vector_grid(50, 0.0, 1.0)
        data = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        user_hints = (Ref(42), Ref(21))
        qs = ([0.5, 0.6], [0.3, 0.4])
        out = Vector{Float64}(undef, 2)

        itp(out, qs; hint = user_hints)

        # Hints should be updated (not stuck at original values)
        # and should NOT be replaced by auto-generated Ref(1)
        @test user_hints[1][] != 42 || user_hints[2][] != 21  # at least one updated
    end

    # ── Correctness: batch vs scalar reference ───────────────────

    @testset "Correctness — batch vs scalar — $label" for (label, builder) in [
            ("Cubic", (g, d) -> cubic_interp(g, d)),
            ("Linear", (g, d) -> linear_interp(g, d)),
            ("Quadratic", (g, d) -> quadratic_interp(g, d)),
            ("Cardinal", (g, d) -> cardinal_interp(g, d)),
            ("PCHIP", (g, d) -> pchip_interp(g, d)),
        ]
        x = _make_vector_grid(21, 0.0, 2π)
        y = _make_vector_grid(11, 0.0, π)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = builder((x, y), data)

        nq = 20
        xqs = collect(range(0.1, 6.0, length = nq))
        yqs = collect(range(0.1, 3.0, length = nq))
        out_batch = Vector{Float64}(undef, nq)
        itp(out_batch, (xqs, yqs))

        # Scalar reference
        out_scalar = [itp((xqs[i], yqs[i])) for i in 1:nq]

        @test out_batch == out_scalar  # bitwise identical
    end

    # ── Zero-alloc with search override + hint ───────────────────

    @testset "Zero-alloc — search override + hint" begin
        function _alloc_override(search, hint_mode)
            x = _make_vector_grid(101, 0.0, 2π)
            y = _make_vector_grid(51, 0.0, π)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = cubic_interp((x, y), data)
            nq = 20
            qs = (collect(range(0.1, 6.0, length = nq)),
                  collect(range(0.1, 3.0, length = nq)))
            out = Vector{Float64}(undef, nq)
            if hint_mode == :hint
                h = (Ref(1), Ref(1))
                itp(out, qs; search = search, hint = h)
                itp(out, qs; search = search, hint = h)
                return @allocated itp(out, qs; search = search, hint = h)
            else
                itp(out, qs; search = search)
                itp(out, qs; search = search)
                return @allocated itp(out, qs; search = search)
            end
        end

        @test _alloc_override(BinarySearch(), :nohint)             <= ND_ALLOC_THRESHOLD
        @test _alloc_override(BinarySearch(), :hint)               <= ND_ALLOC_THRESHOLD
        @test _alloc_override(LinearBinarySearch(), :nohint)       <= ND_ALLOC_THRESHOLD
        @test _alloc_override(LinearBinarySearch(), :hint)         <= ND_ALLOC_THRESHOLD
        @test _alloc_override(AutoSearch(), :nohint)               <= ND_ALLOC_THRESHOLD
        @test _alloc_override(AutoSearch(), :hint)                 <= ND_ALLOC_THRESHOLD
    end

    # ── Range grid — zero-alloc + correctness ───────────────────
    # Range grids dispatch to DirectSearch O(1). Verify they still work
    # through the (policies, hints, mono) infrastructure without allocating.

    @testset "Zero-alloc — Range grid $label" for (label, builder) in [
            ("Cubic", (g, d) -> cubic_interp(g, d)),
            ("Linear", (g, d) -> linear_interp(g, d)),
        ]
        function _alloc_range(builder)
            x = range(0.0, 2π, 101)
            y = range(0.0, π, 51)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = builder((x, y), data)
            nq = 20
            qs = (collect(range(0.1, 6.0, length = nq)),
                  collect(range(0.1, 3.0, length = nq)))
            out = Vector{Float64}(undef, nq)
            itp(out, qs); itp(out, qs)
            return @allocated itp(out, qs)
        end
        @test _alloc_range(builder) <= ND_ALLOC_THRESHOLD
    end

    @testset "Correctness — Range vs Vector — batch" begin
        xr = range(0.0, 2π, 101)
        yr = range(0.0, π, 51)
        xv = collect(xr)
        yv = collect(yr)
        data = [sin(xi) * cos(yj) for xi in xr, yj in yr]
        itp_r = cubic_interp((xr, yr), data)
        itp_v = cubic_interp((xv, yv), data)

        nq = 20
        qs = (collect(range(0.1, 6.0, length = nq)),
              collect(range(0.1, 3.0, length = nq)))
        out_r = Vector{Float64}(undef, nq)
        out_v = Vector{Float64}(undef, nq)
        itp_r(out_r, qs)
        itp_v(out_v, qs)
        @test out_r ≈ out_v  atol = 1e-12  # Range vs Vector may differ at ULP level
    end

    @testset "Hint mutation — Range grid" begin
        xr = range(0.0, 1.0, 100)
        yr = range(0.0, 1.0, 50)
        data = [sin(2π * xi) * cos(2π * yj) for xi in xr, yj in yr]
        itp = cubic_interp((xr, yr), data)

        nq = 50
        qs = (collect(range(0.01, 0.99, length = nq)),
              collect(range(0.01, 0.99, length = nq)))
        out = Vector{Float64}(undef, nq)
        hints = (Ref(1), Ref(1))
        itp(out, qs; hint = hints)

        @test hints[1][] > length(xr) ÷ 2
        @test hints[2][] > length(yr) ÷ 2
    end

    # ── Mixed grid (Range × Vector) — zero-alloc + correctness ──

    @testset "Zero-alloc — Mixed grid Range×Vector" begin
        function _alloc_mixed_grid()
            xr = range(0.0, 2π, 101)
            yv = _make_vector_grid(51, 0.0, π)
            data = [sin(xi) * cos(yj) for xi in xr, yj in yv]
            itp = cubic_interp((xr, yv), data)
            nq = 20
            qs = (collect(range(0.1, 6.0, length = nq)),
                  collect(range(0.1, 3.0, length = nq)))
            out = Vector{Float64}(undef, nq)
            itp(out, qs); itp(out, qs)
            return @allocated itp(out, qs)
        end
        @test _alloc_mixed_grid() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-alloc — Mixed grid 3D Range×Vector×Range" begin
        function _alloc_mixed_3d()
            xr = range(0.0, 2π, 21)
            yv = _make_vector_grid(11, 0.0, π)
            zr = range(0.0, 1.0, 7)
            data = [sin(xi) * cos(yj) * (1 + zk) for xi in xr, yj in yv, zk in zr]
            itp = cubic_interp((xr, yv, zr), data)
            nq = 10
            qs = (collect(range(0.1, 6.0, length = nq)),
                  collect(range(0.1, 3.0, length = nq)),
                  collect(range(0.1, 0.9, length = nq)))
            out = Vector{Float64}(undef, nq)
            itp(out, qs); itp(out, qs)
            return @allocated itp(out, qs)
        end
        @test _alloc_mixed_3d() <= ND_ALLOC_THRESHOLD
    end

    # ── AoS batch queries — correctness ─────────────────────────
    # Array-of-Structs queries: Vector{Tuple}. _check_mono_nd uses protocol-based
    # _is_axis_likely_monotone for per-axis monotonicity check (no allocation).

    @testset "Correctness — AoS batch queries" begin
        x = _make_vector_grid(21, 0.0, 2π)
        y = _make_vector_grid(11, 0.0, π)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        nq = 10
        xqs = collect(range(0.1, 6.0, length = nq))
        yqs = collect(range(0.1, 3.0, length = nq))

        # SoA reference
        out_soa = Vector{Float64}(undef, nq)
        itp(out_soa, (xqs, yqs))

        # AoS query
        qs_aos = [(xqs[i], yqs[i]) for i in 1:nq]
        out_aos = Vector{Float64}(undef, nq)
        itp(out_aos, qs_aos)

        @test out_aos == out_soa  # bitwise identical
    end

    @testset "Zero-alloc — AoS batch queries" begin
        function _alloc_aos()
            x = _make_vector_grid(101, 0.0, 2π)
            y = _make_vector_grid(51, 0.0, π)
            data = [sin(xi) * cos(yj) for xi in x, yj in y]
            itp = cubic_interp((x, y), data)
            nq = 20
            xqs = collect(range(0.1, 6.0, length = nq))
            yqs = collect(range(0.1, 3.0, length = nq))
            qs = [(xqs[i], yqs[i]) for i in 1:nq]
            out = Vector{Float64}(undef, nq)
            itp(out, qs); itp(out, qs)
            return @allocated itp(out, qs)
        end
        @test _alloc_aos() <= ND_ALLOC_THRESHOLD
    end
end
