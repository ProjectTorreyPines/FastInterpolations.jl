# Tests for PeriodicBC on PCHIP ND forward path (OnTheFly only).
#
# Coverage:
#   - NoBC default is no-op (regression guard).
#   - Homogeneous PCHIP×PCHIP with `bc` broadcast and `bcs` per-axis API.
#   - Heterogeneous combinations (PchipInterp(bc) × LinearInterp()).
#   - C¹-at-seam continuity per axis on smooth periodic 2D data.
#   - Wrap query past seam.
#   - Both endpoints (`:inclusive`, `:exclusive`).
#   - Persistent path delegation (same shape as homogeneous oneshot).
#   - Type stability via @inferred (function-barrier).
#   - Zero-alloc oneshot scalar (function-barrier).

@testitem "PCHIP PeriodicBC ND forward" setup = [AllocConstants] begin
    f(x, y) = sin(2π * x) * cos(2π * y)
    fpx(x, y) = 2π * cos(2π * x) * cos(2π * y)

    @testset "NoBC default is no-op" begin
        n = 11
        x = collect(range(0.0, 1.0, length = n))
        y = collect(range(0.0, 1.0, length = n))
        data = [f(xi, yj) for xi in x, yj in y]

        v_default = pchip_interp((x, y), data, (0.4, 0.6))
        v_explicit_nobc = pchip_interp((x, y), data, (0.4, 0.6); bc = NoBC())
        @test v_default === v_explicit_nobc
    end

    @testset "Homogeneous PCHIP×PCHIP — bc broadcast (exclusive)" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        # Sanity: value reasonably close to true f
        v = pchip_interp((x, y), data, (0.5, 0.5); bc = bc)
        @test isapprox(v, f(0.5, 0.5); atol = 1.0e-3)

        # C¹ at seam (x axis crossing)
        ε = 1.0e-7
        v_left = pchip_interp((x, y), data, (1.0 - ε, 0.5); bc = bc)
        v_right = pchip_interp((x, y), data, (0.0 + ε, 0.5); bc = bc)
        @test abs(v_left - v_right) < 1.0e-3

        # Wrap query past seam
        @test pchip_interp((x, y), data, (1.25, 0.5); bc = bc) ≈
              pchip_interp((x, y), data, (0.25, 0.5); bc = bc) atol = 1.0e-12
    end

    @testset "Homogeneous PCHIP×PCHIP — bcs per-axis tuple" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        # bcs (per-axis) and bc (broadcast) must agree when both axes use the same BC
        v_bcs = pchip_interp((x, y), data, (0.5, 0.5); bcs = (bc, bc))
        v_bc = pchip_interp((x, y), data, (0.5, 0.5); bc = bc)
        @test v_bcs === v_bc

        # Mixed via bcs: PeriodicBC × NoBC
        v_mixed = pchip_interp((x, y), data, (0.5, 0.5); bcs = (bc, NoBC()))
        @test isfinite(v_mixed)
    end

    @testset "Inclusive endpoint" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n))
        y = collect(range(0.0, 1.0, length = n))
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :inclusive)

        @test isapprox(pchip_interp((x, y), data, (0.5, 0.5); bc = bc), f(0.5, 0.5); atol = 1.0e-3)

        # C¹ at seam
        ε = 1.0e-7
        v_left = pchip_interp((x, y), data, (1.0 - ε, 0.5); bc = bc)
        v_right = pchip_interp((x, y), data, (0.0 + ε, 0.5); bc = bc)
        @test abs(v_left - v_right) < 1.0e-3
    end

    @testset "Heterogeneous: PchipInterp(bc) × LinearInterp()" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        # Periodic on axis 1 only
        v = interp((x, y), data, (0.5, 0.5); method = (PchipInterp(bc), LinearInterp()))
        @test isapprox(v, f(0.5, 0.5); atol = 1.0e-2)

        # x seam continuity (axis 1 periodic, axis 2 not)
        ε = 1.0e-7
        v_left = interp((x, y), data, (1.0 - ε, 0.5); method = (PchipInterp(bc), LinearInterp()))
        v_right = interp((x, y), data, (0.0 + ε, 0.5); method = (PchipInterp(bc), LinearInterp()))
        @test abs(v_left - v_right) < 1.0e-3
    end

    @testset "Persistent ND interpolant" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        itp = pchip_interp((x, y), data; bc = bc)
        @test isapprox(itp((0.5, 0.5)), f(0.5, 0.5); atol = 1.0e-3)
        ε = 1.0e-7
        @test abs(itp((1.0 - ε, 0.5)) - itp((0.0 + ε, 0.5))) < 1.0e-3
    end

    @testset "Vector batch (in-place, allocating)" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        queries = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.85)]

        v_alloc = pchip_interp((x, y), data, queries; bc = bc)
        out = similar(v_alloc)
        pchip_interp!(out, (x, y), data, queries; bc = bc)
        @test out == v_alloc
    end

    @testset "Type stability (@inferred)" begin
        # Function barrier — keeps body's locals concrete for inference.
        function _inferred_pchip_periodic_nd()
            n = 21
            x = collect(range(0.0, 1.0, length = n + 1))[1:n]
            y = collect(range(0.0, 1.0, length = n + 1))[1:n]
            data = [f(xi, yj) for xi in x, yj in y]
            bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
            return @inferred pchip_interp((x, y), data, (0.5, 0.5); bc = bc)
        end
        v = _inferred_pchip_periodic_nd()
        @test v isa Float64
    end

    @testset "Zero-alloc scalar oneshot (function barrier)" begin
        # Function barrier required: @testset wraps body in try/catch which makes
        # locals type-unstable. Inside this function, the call is concrete.
        function _alloc_check()
            n = 21
            x = collect(range(0.0, 1.0, length = n + 1))[1:n]
            y = collect(range(0.0, 1.0, length = n + 1))[1:n]
            data = [f(xi, yj) for xi in x, yj in y]
            bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
            # Warmup
            pchip_interp((x, y), data, (0.5, 0.5); bc = bc)
            # Measured call
            return @allocated pchip_interp((x, y), data, (0.5, 0.5); bc = bc)
        end
        # Pool-managed buffers ⇒ zero allocation after warmup. Allow a tiny budget
        # for any framework-level box (rare, but avoids brittleness).
        @test _alloc_check() <= ALLOC_THRESHOLD
    end

    # Regression: persistent OnTheFly path must use a BC-aware search so that
    # exclusive-periodic seam queries pick the same cell as the oneshot path.
    # Without it, `_search_all_intervals` clamps `q ≥ x[n]` to interval n-1
    # (last *real* cell) instead of returning the seam cell n (= virtual wrap),
    # and the windowed evaluation diverges from the oneshot result.
    @testset "Persistent ↔ oneshot consistency at exclusive seam" begin
        n = 10   # small n amplifies the per-cell mismatch
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = collect(range(0.0, 1.0, length = n + 1))[1:n]
        data = [f(xi, yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        queries = [
            (0.95, 0.3),
            (0.99, 0.5),
            (x[n], 0.3),
            (0.5, x[n]),
            (x[n], x[n]),
        ]
        for q in queries
            v_persistent = pchip_interp((x, y), data; bc = bc)(q)
            v_oneshot = pchip_interp((x, y), data, q; bc = bc)
            @test isapprox(v_persistent, v_oneshot; atol = 1.0e-10)
        end
    end

    # Regression: inclusive PeriodicBC ND must validate `data[1, :] ≈ data[end, :]`
    # per axis at construction / oneshot entry, mirroring 1D and existing
    # CubicInterpolantND. Without this, mismatched data is silently accepted.
    @testset "Inclusive ND validation rejects mismatched endpoints" begin
        n = 11
        x = collect(range(0.0, 1.0, length = n))
        y = collect(range(0.0, 1.0, length = n))
        data = rand(n, n)
        data[end, :] .= data[1, :] .+ 0.5   # axis-1 endpoint mismatch
        bc = PeriodicBC(endpoint = :inclusive)

        @test_throws ArgumentError pchip_interp((x, y), data, (0.5, 0.5); bc = bc)
        @test_throws ArgumentError pchip_interp((x, y), data; bc = bc)

        # `check=false` bypass — must not throw.
        bc_unchecked = PeriodicBC(endpoint = :inclusive, check = false)
        v = pchip_interp((x, y), data, (0.5, 0.5); bc = bc_unchecked)
        @test isfinite(v)
        itp = pchip_interp((x, y), data; bc = bc_unchecked)
        @test isfinite(itp((0.5, 0.5)))
    end
end
