# ─────────────────────────────────────────────────────────────────────────
# 1D adjoint PeriodicBC coverage for linear / constant / pchip / cardinal / akima.
#
# Identity tested (uniform across linear-in-y and nonlinear-in-y forwards):
#
#     adj(y_bar) == ∂(dot(forward(y), y_bar)) / ∂y
#
# Computed via ForwardDiff on the forward (which already respects PeriodicBC).
# For Linear/Constant/Cardinal the forward is linear in `y`, so this reduces
# to the dot-product identity. For PCHIP/Akima the forward is nonlinear in
# `y` (slope limiter); ForwardDiff still gives the exact JVP-transpose at
# the linearization point.
#
# Each method/BC combo is exercised at both random interior queries and at
# the seam cell `[x[n], x[1]+period]` — boundary-touching queries are
# required to drive the wrap-aware stencil and the seam-fold finalize path.
# ─────────────────────────────────────────────────────────────────────────

@testitem "1D linear_adjoint — PeriodicBC TDD" begin
    using LinearAlgebra: dot
    using ForwardDiff

    # `:inclusive` — domain [x[1], x[n]], BC enforces f[1]==f[n].
    @testset "PeriodicBC{:inclusive}" begin
        nx = 12
        x = collect(range(0.0, 1.0, nx))
        f = randn(nx); f[end] = f[1]
        # OOB queries that wrap around (need PeriodicBC semantics for forward).
        xq = vcat(0.05 .+ 0.9 .* rand(10), 1.0 .+ 0.4 .* rand(10) .- 0.2)
        y_bar = randn(length(xq))
        bc = PeriodicBC()

        forward(ftest) = linear_interp(x, ftest, xq; bc = bc)
        f_bar_ref = ForwardDiff.gradient(ftest -> dot(forward(ftest), y_bar), f)

        adj = linear_adjoint(x, xq; bc = bc)
        f_bar = adj(y_bar)

        @test size(f_bar) == size(f)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-10
    end

    # `:exclusive` — domain [x[1], x[1]+period), seam cell [x[n], x[1]+period].
    # The CRITICAL test — exposes the bc-absorption bug unambiguously.
    @testset "PeriodicBC{:exclusive}" begin
        nx = 11
        period = 1.0
        x = collect(range(0.0, step = period / nx, length = nx))
        f = randn(nx)
        xq = period .* rand(20)  # spans full period including seam cell
        y_bar = randn(length(xq))
        bc = PeriodicBC(endpoint = :exclusive, period = period)

        forward(ftest) = linear_interp(x, ftest, xq; bc = bc)
        f_bar_ref = ForwardDiff.gradient(ftest -> dot(forward(ftest), y_bar), f)

        adj = linear_adjoint(x, xq; bc = bc)
        f_bar = adj(y_bar)

        @test size(f_bar) == size(f)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-10
    end

    # Vector grid (non-uniform) `:exclusive`.
    @testset "PeriodicBC{:exclusive} — Vector grid" begin
        nx = 9
        period = 2.0
        x = sort(rand(nx)) .* (0.95 * period)
        f = randn(nx)
        xq = period .* rand(20)
        y_bar = randn(length(xq))
        bc = PeriodicBC(endpoint = :exclusive, period = period)

        forward(ftest) = linear_interp(x, ftest, xq; bc = bc)
        f_bar_ref = ForwardDiff.gradient(ftest -> dot(forward(ftest), y_bar), f)

        adj = linear_adjoint(x, xq; bc = bc)
        f_bar = adj(y_bar)

        @test size(f_bar) == size(f)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-10
    end
end


@testitem "1D constant_adjoint — PeriodicBC TDD" begin
    using LinearAlgebra: dot
    using ForwardDiff

    @testset "PeriodicBC{:inclusive}" begin
        nx = 12
        x = collect(range(0.0, 1.0, nx))
        f = randn(nx); f[end] = f[1]
        xq = vcat(0.05 .+ 0.9 .* rand(10), 1.0 .+ 0.4 .* rand(10) .- 0.2)
        y_bar = randn(length(xq))
        bc = PeriodicBC()

        forward(ftest) = constant_interp(x, ftest, xq; bc = bc, side = NearestSide())
        f_bar_ref = ForwardDiff.gradient(ftest -> dot(forward(ftest), y_bar), f)

        adj = constant_adjoint(x, xq; bc = bc, side = NearestSide())
        f_bar = adj(y_bar)

        @test size(f_bar) == size(f)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-10
    end

    @testset "PeriodicBC{:exclusive}" begin
        nx = 11
        period = 1.0
        x = collect(range(0.0, step = period / nx, length = nx))
        f = randn(nx)
        xq = period .* rand(20)
        y_bar = randn(length(xq))
        bc = PeriodicBC(endpoint = :exclusive, period = period)

        forward(ftest) = constant_interp(x, ftest, xq; bc = bc, side = NearestSide())
        f_bar_ref = ForwardDiff.gradient(ftest -> dot(forward(ftest), y_bar), f)

        adj = constant_adjoint(x, xq; bc = bc, side = NearestSide())
        f_bar = adj(y_bar)

        @test size(f_bar) == size(f)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-10
    end
end


@testitem "1D pchip_adjoint — PeriodicBC TDD" begin
    using LinearAlgebra: dot
    using ForwardDiff

    @testset "PeriodicBC{:inclusive}" begin
        nx = 12
        x = collect(range(0.0, 1.0, nx))
        # Smooth monotone-ish data; PCHIP slope limiter still nonlinear in y.
        y = sin.(2π .* x); y[end] = y[1]
        xq = vcat(0.05 .+ 0.9 .* rand(8), 1.0 .+ 0.4 .* rand(8) .- 0.2)
        y_bar = randn(length(xq))
        bc = PeriodicBC()

        forward(ytest) = pchip_interp(x, ytest, xq; bc = bc)
        f_bar_ref = ForwardDiff.gradient(ytest -> dot(forward(ytest), y_bar), y)

        adj = pchip_adjoint(x, y, xq; bc = bc)
        f_bar = adj(y_bar)

        @test size(f_bar) == size(y)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-9
    end

    @testset "PeriodicBC{:exclusive}" begin
        nx = 11
        period = 1.0
        x = collect(range(0.0, step = period / nx, length = nx))
        y = sin.(2π .* x ./ period)
        xq = period .* rand(16)
        y_bar = randn(length(xq))
        bc = PeriodicBC(endpoint = :exclusive, period = period)

        forward(ytest) = pchip_interp(x, ytest, xq; bc = bc)
        f_bar_ref = ForwardDiff.gradient(ytest -> dot(forward(ytest), y_bar), y)

        adj = pchip_adjoint(x, y, xq; bc = bc)
        f_bar = adj(y_bar)

        @test size(f_bar) == size(y)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-9
    end
end


@testitem "1D cardinal_adjoint — PeriodicBC TDD" begin
    using LinearAlgebra: dot
    using ForwardDiff

    # cardinal_adjoint is data-free; build f_bar_ref via FD on the forward.
    @testset "PeriodicBC{:inclusive}" begin
        nx = 12
        x = collect(range(0.0, 1.0, nx))
        f = randn(nx); f[end] = f[1]
        xq = vcat(0.05 .+ 0.9 .* rand(10), 1.0 .+ 0.4 .* rand(10) .- 0.2)
        y_bar = randn(length(xq))
        bc = PeriodicBC()

        forward(ftest) = cardinal_interp(x, ftest, xq; bc = bc)
        f_bar_ref = ForwardDiff.gradient(ftest -> dot(forward(ftest), y_bar), f)

        adj = cardinal_adjoint(x, xq; bc = bc)
        f_bar = adj(y_bar)

        @test size(f_bar) == size(f)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-10
    end

    @testset "PeriodicBC{:exclusive}" begin
        nx = 11
        period = 1.0
        x = collect(range(0.0, step = period / nx, length = nx))
        f = randn(nx)
        xq = period .* rand(20)
        y_bar = randn(length(xq))
        bc = PeriodicBC(endpoint = :exclusive, period = period)

        forward(ftest) = cardinal_interp(x, ftest, xq; bc = bc)
        f_bar_ref = ForwardDiff.gradient(ftest -> dot(forward(ftest), y_bar), f)

        adj = cardinal_adjoint(x, xq; bc = bc)
        f_bar = adj(y_bar)

        @test size(f_bar) == size(f)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-10
    end
end


@testitem "1D akima_adjoint — PeriodicBC TDD" begin
    using LinearAlgebra: dot
    using ForwardDiff

    @testset "PeriodicBC{:inclusive}" begin
        nx = 12
        x = collect(range(0.0, 1.0, nx))
        y = sin.(2π .* x); y[end] = y[1]
        xq = vcat(0.05 .+ 0.9 .* rand(8), 1.0 .+ 0.4 .* rand(8) .- 0.2)
        y_bar = randn(length(xq))
        bc = PeriodicBC()

        forward(ytest) = akima_interp(x, ytest, xq; bc = bc)
        f_bar_ref = ForwardDiff.gradient(ytest -> dot(forward(ytest), y_bar), y)

        adj = akima_adjoint(x, y, xq; bc = bc)
        f_bar = adj(y_bar)

        @test size(f_bar) == size(y)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-9
    end

    @testset "PeriodicBC{:exclusive}" begin
        nx = 11
        period = 1.0
        x = collect(range(0.0, step = period / nx, length = nx))
        y = sin.(2π .* x ./ period)
        xq = period .* rand(16)
        y_bar = randn(length(xq))
        bc = PeriodicBC(endpoint = :exclusive, period = period)

        forward(ytest) = akima_interp(x, ytest, xq; bc = bc)
        f_bar_ref = ForwardDiff.gradient(ytest -> dot(forward(ytest), y_bar), y)

        adj = akima_adjoint(x, y, xq; bc = bc)
        f_bar = adj(y_bar)

        @test size(f_bar) == size(y)
        @test f_bar ≈ f_bar_ref rtol = 1.0e-9
    end
end


# ─────────────────────────────────────────────────────────────────────────
# Seam-cell coverage: queries placed STRICTLY inside the seam cell
# `[x[n], x[1]+period]` for `:exclusive`. This is the cell that requires
# the wrapper / extension + seam-fold path; failure here would indicate
# the periodic plumbing missed the edge case.
# ─────────────────────────────────────────────────────────────────────────

@testitem "1D adjoint — seam-cell PeriodicBC{:exclusive}" begin
    using LinearAlgebra: dot
    using ForwardDiff

    nx = 8
    period = 1.0
    x = collect(range(0.0, step = period / nx, length = nx))    # x[end] ≈ 0.875
    bc = PeriodicBC(endpoint = :exclusive, period = period)
    # Queries strictly inside the seam cell (between x[end] and x[1]+period).
    seam_lo = x[end] + 1.0e-3
    seam_hi = x[1] + period - 1.0e-3
    xq = collect(range(seam_lo, seam_hi, length = 6))
    y_bar = randn(length(xq))

    @testset "linear_adjoint seam" begin
        f = randn(nx)
        adj = linear_adjoint(x, xq; bc = bc)
        f_bar = adj(y_bar)
        ref = ForwardDiff.gradient(ftest -> dot(linear_interp(x, ftest, xq; bc = bc), y_bar), f)
        @test size(f_bar) == size(f)
        @test f_bar ≈ ref rtol = 1.0e-10
    end

    @testset "constant_adjoint seam" begin
        f = randn(nx)
        adj = constant_adjoint(x, xq; bc = bc, side = NearestSide())
        f_bar = adj(y_bar)
        ref = ForwardDiff.gradient(
            ftest -> dot(constant_interp(x, ftest, xq; bc = bc, side = NearestSide()), y_bar),
            f,
        )
        @test size(f_bar) == size(f)
        @test f_bar ≈ ref rtol = 1.0e-10
    end

    @testset "pchip_adjoint seam" begin
        y = sin.(2π .* x ./ period)
        adj = pchip_adjoint(x, y, xq; bc = bc)
        f_bar = adj(y_bar)
        ref = ForwardDiff.gradient(ytest -> dot(pchip_interp(x, ytest, xq; bc = bc), y_bar), y)
        @test size(f_bar) == size(y)
        @test f_bar ≈ ref rtol = 1.0e-9
    end

    @testset "cardinal_adjoint seam" begin
        f = randn(nx)
        adj = cardinal_adjoint(x, xq; bc = bc)
        f_bar = adj(y_bar)
        ref = ForwardDiff.gradient(ftest -> dot(cardinal_interp(x, ftest, xq; bc = bc), y_bar), f)
        @test size(f_bar) == size(f)
        @test f_bar ≈ ref rtol = 1.0e-10
    end

    @testset "akima_adjoint seam" begin
        y = sin.(2π .* x ./ period)
        adj = akima_adjoint(x, y, xq; bc = bc)
        f_bar = adj(y_bar)
        ref = ForwardDiff.gradient(ytest -> dot(akima_interp(x, ytest, xq; bc = bc), y_bar), y)
        @test size(f_bar) == size(y)
        @test f_bar ≈ ref rtol = 1.0e-9
    end
end


# ─────────────────────────────────────────────────────────────────────────
# Boundary-cell coverage for `:inclusive` Hermite (PCHIP/Cardinal/Akima):
# stencil-aware loop split treats k ∈ {1, n} (PCHIP/Cardinal radius 1) and
# k ∈ {1, 2, n-1, n} (Akima radius 2) as boundary, wrapping via `mod1`.
# Random queries do not deterministically hit these cells, so we place
# queries explicitly inside each boundary cell.
# ─────────────────────────────────────────────────────────────────────────

@testitem "1D Hermite adjoint — PeriodicBC{:inclusive} boundary cells" begin
    using LinearAlgebra: dot
    using ForwardDiff

    nx = 12
    x = collect(range(0.0, 1.0, nx))
    bc = PeriodicBC()

    # Explicit boundary cells.
    # Radius-1 cells: [x[1], x[2]] and [x[n-1], x[n]].
    # Radius-2 cells (Akima only): [x[2], x[3]] and [x[n-2], x[n-1]].
    cell(a, b) = collect(range(a + 1.0e-3, b - 1.0e-3, length = 4))
    xq_r1 = vcat(cell(x[1], x[2]), cell(x[end - 1], x[end]))
    xq_r2 = vcat(cell(x[2], x[3]), cell(x[end - 2], x[end - 1]))
    xq = vcat(xq_r1, xq_r2)
    y_bar = randn(length(xq))

    @testset "pchip_adjoint :inclusive boundary" begin
        y = sin.(2π .* x); y[end] = y[1]
        adj = pchip_adjoint(x, y, xq; bc = bc)
        f_bar = adj(y_bar)
        ref = ForwardDiff.gradient(ytest -> dot(pchip_interp(x, ytest, xq; bc = bc), y_bar), y)
        @test size(f_bar) == size(y)
        @test f_bar ≈ ref rtol = 1.0e-9
    end

    @testset "cardinal_adjoint :inclusive boundary" begin
        f = randn(nx); f[end] = f[1]
        adj = cardinal_adjoint(x, xq; bc = bc)
        f_bar = adj(y_bar)
        ref = ForwardDiff.gradient(ftest -> dot(cardinal_interp(x, ftest, xq; bc = bc), y_bar), f)
        @test size(f_bar) == size(f)
        @test f_bar ≈ ref rtol = 1.0e-10
    end

    @testset "akima_adjoint :inclusive boundary (radius 2)" begin
        y = sin.(2π .* x); y[end] = y[1]
        adj = akima_adjoint(x, y, xq; bc = bc)
        f_bar = adj(y_bar)
        ref = ForwardDiff.gradient(ytest -> dot(akima_interp(x, ytest, xq; bc = bc), y_bar), y)
        @test size(f_bar) == size(y)
        @test f_bar ≈ ref rtol = 1.0e-9
    end
end


# ─────────────────────────────────────────────────────────────────────────
# Tiny-grid coverage (n ∈ {2, 3}). The Hermite-family periodic slope kernels
# have explicit small-`n` fallback branches that avoid the boundary/interior
# loop split (PCHIP/Cardinal n=2; Akima n ∈ {2, 3}). Random testitems above
# all use nx ≥ 11, never reaching these paths.
# ─────────────────────────────────────────────────────────────────────────

@testitem "1D Hermite adjoint — PeriodicBC tiny grids (n ∈ {2, 3})" begin
    using LinearAlgebra: dot
    using ForwardDiff

    # Skip `:inclusive n=2` because the closing constraint `y[end]=y[1]` forces
    # constant data (PCHIP/Akima 0/0 in slope formulas — undefined, not a bug).
    @testset "n=$nx, :$endpoint" for (nx, endpoint) in [
            (3, :inclusive), (2, :exclusive), (3, :exclusive),
        ]
        period = 1.0
        if endpoint === :inclusive
            x = collect(range(0.0, period, nx))
            y = randn(nx); y[end] = y[1]
            bc = PeriodicBC()
        else
            x = collect(range(0.0, step = period / nx, length = nx))
            y = randn(nx)
            bc = PeriodicBC(endpoint = :exclusive, period = period)
        end
        xq = [0.1 * period, 0.5 * period, 0.9 * period]
        y_bar = randn(length(xq))

        for (label, fwd, adj_ctor, data_arg) in [
                ("PCHIP", pchip_interp, pchip_adjoint, y),
                ("Cardinal", cardinal_interp, cardinal_adjoint, nothing),
                ("Akima", akima_interp, akima_adjoint, y),
            ]
            @testset "$label" begin
                forward(ytest) = fwd(x, ytest, xq; bc = bc)
                ref = ForwardDiff.gradient(ytest -> dot(forward(ytest), y_bar), y)
                adj = data_arg === nothing ?
                    adj_ctor(x, xq; bc = bc) :
                    adj_ctor(x, data_arg, xq; bc = bc)
                f_bar = adj(y_bar)
                @test size(f_bar) == size(y)
                @test f_bar ≈ ref rtol = 1.0e-9
            end
        end
    end
end


# ─────────────────────────────────────────────────────────────────────────
# Zero-allocation regression: in-place `adj(f_bar, y_bar)` must not
# allocate beyond the pool buffer (which is acquired/returned per-call
# inside `@with_pool` and counted as 0 bytes after warmup).
#
# Pattern matches MEMORY.md `feedback_test_alloc_function_barrier`: setup,
# warmup, and `@allocated` all live inside ONE function — `@testset`
# wraps its body in try/catch which makes locals type-unstable and shows
# allocation artifacts.
# ─────────────────────────────────────────────────────────────────────────

@testitem "1D adjoint — PeriodicBC zero-alloc in-place" begin
    nx = 16
    n_q = 32
    period = 1.0

    function alloc_linear(bc, x, xq, y_bar, f_bar)
        adj = linear_adjoint(x, xq; bc = bc)
        adj(f_bar, y_bar)            # warmup
        return @allocated adj(f_bar, y_bar)
    end

    function alloc_constant(bc, x, xq, y_bar, f_bar)
        adj = constant_adjoint(x, xq; bc = bc)
        adj(f_bar, y_bar)
        return @allocated adj(f_bar, y_bar)
    end

    function alloc_pchip(bc, x, y, xq, y_bar, f_bar)
        adj = pchip_adjoint(x, y, xq; bc = bc)
        adj(f_bar, y_bar)
        return @allocated adj(f_bar, y_bar)
    end

    function alloc_cardinal(bc, x, xq, y_bar, f_bar)
        adj = cardinal_adjoint(x, xq; bc = bc)
        adj(f_bar, y_bar)
        return @allocated adj(f_bar, y_bar)
    end

    function alloc_akima(bc, x, y, xq, y_bar, f_bar)
        adj = akima_adjoint(x, y, xq; bc = bc)
        adj(f_bar, y_bar)
        return @allocated adj(f_bar, y_bar)
    end

    # ── Inclusive: closed at n, no extension/seam-fold ──────────────────
    @testset "PeriodicBC{:inclusive}" begin
        x_inc = collect(range(0.0, period, nx))
        y_inc = sin.(2π .* x_inc ./ period); y_inc[end] = y_inc[1]
        f_inc = copy(y_inc)
        xq = 0.1 .+ 0.8 * period .* rand(n_q)
        y_bar = randn(n_q)
        f_bar = zeros(nx)
        bc = PeriodicBC()

        @test alloc_linear(bc, x_inc, xq, y_bar, f_bar) == 0
        @test alloc_constant(bc, x_inc, xq, y_bar, f_bar) == 0
        @test alloc_pchip(bc, x_inc, y_inc, xq, y_bar, f_bar) == 0
        @test alloc_cardinal(bc, x_inc, xq, y_bar, f_bar) == 0
        @test alloc_akima(bc, x_inc, y_inc, xq, y_bar, f_bar) == 0
    end

    # ── Exclusive: extension to n+1 + seam fold via pool buffer ─────────
    @testset "PeriodicBC{:exclusive}" begin
        x_exc = collect(range(0.0, step = period / nx, length = nx))
        y_exc = sin.(2π .* x_exc ./ period)
        xq = period .* rand(n_q)
        y_bar = randn(n_q)
        f_bar = zeros(nx)
        bc = PeriodicBC(endpoint = :exclusive, period = period)

        @test alloc_linear(bc, x_exc, xq, y_bar, f_bar) == 0
        @test alloc_constant(bc, x_exc, xq, y_bar, f_bar) == 0
        @test alloc_pchip(bc, x_exc, y_exc, xq, y_bar, f_bar) == 0
        @test alloc_cardinal(bc, x_exc, xq, y_bar, f_bar) == 0
        @test alloc_akima(bc, x_exc, y_exc, xq, y_bar, f_bar) == 0
    end
end
