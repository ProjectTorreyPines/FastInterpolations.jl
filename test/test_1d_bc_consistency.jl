# ─────────────────────────────────────────────────────────────────────────
# 1D forward/adjoint cross-BC consistency tests.
#
# "Logically same grid" setup (n_excl-cell uniform grid on [0, period]):
#
#   x_exc = [0, h, 2h, ..., (n_exc-1)*h]    ← n_exc points,   period via kwarg
#   x_inc = [x_exc; n_exc*h]                ← n_exc+1 points, last == first+period
#   f_exc = randn(n_exc)
#   f_inc = [f_exc; f_exc[1]]               ← `:inclusive` constraint
#
# All three (NoBC on x_inc/f_inc, inclusive on x_inc/f_inc, exclusive on
# x_exc/f_exc) describe the SAME logical periodic function on [0, period].
#
# Expectations:
#
# Linear / Constant — eval is purely local (no slope kernel that "sees"
# beyond the immediate cell). For queries strictly in `[0, period)` the
# three representations produce **bit-equivalent** forward results, and
# their adjoints are equivalent up to the seam-fold (exclusive folds the
# n+1-th internal entry back into index 1).
#
# PCHIP / Cardinal / Akima — local-Hermite slopes use a stencil that
# crosses the join at boundary indices. NoBC uses one-sided FD (or
# virtual-secant) at i=1 and i=n, while PeriodicBC uses the closed-cycle
# stencil. Therefore:
#   - NoBC ≠ Periodic for queries in cells touching the boundary
#     (different boundary slopes propagate to interior eval).
#   - inclusive == exclusive (both closed-cycle, same logical setup).
# ─────────────────────────────────────────────────────────────────────────

@testitem "1D Linear forward — NoBC == inclusive == exclusive (same logical grid)" begin
    nx_exc = 16
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)             # closing point
    f_exc = randn(nx_exc)
    f_inc = vcat(f_exc, f_exc[1])                       # f[end] = f[1]

    # Queries strictly inside [0, period) — in-domain for ALL three setups.
    n_q = 30
    xq = (0.02 + 0.96 * rand()) * period .* rand(n_q)   # in (0, 0.96·period)
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    y_nobc = linear_interp(x_inc, f_inc, xq)
    y_inc  = linear_interp(x_inc, f_inc, xq; bc = bc_inc)
    y_exc  = linear_interp(x_exc, f_exc, xq; bc = bc_exc)

    # Linear is purely local — three representations bit-equivalent.
    # `:exclusive` synthesizes its seam endpoint via `inner[1] + period`, but
    # cell lookup + lerp arithmetic is identical to NoBC over the closed grid,
    # so the result must match bit-for-bit (no rounding budget needed).
    @test y_nobc == y_inc
    @test y_nobc == y_exc
end


@testitem "1D Constant forward — NoBC == inclusive == exclusive (same logical grid)" begin
    nx_exc = 16
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    f_exc = randn(nx_exc)
    f_inc = vcat(f_exc, f_exc[1])

    n_q = 30
    xq = 0.02 * period .+ 0.94 * period .* rand(n_q)
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    y_nobc = constant_interp(x_inc, f_inc, xq; side = NearestSide())
    y_inc  = constant_interp(x_inc, f_inc, xq; side = NearestSide(), bc = bc_inc)
    y_exc  = constant_interp(x_exc, f_exc, xq; side = NearestSide(), bc = bc_exc)

    # Constant is single-point lookup — bit-equivalent across BCs.
    @test y_nobc == y_inc
    @test y_nobc == y_exc
end


@testitem "1D Linear adjoint — NoBC == inclusive ≡ exclusive(after seam fold)" begin
    using LinearAlgebra: dot

    nx_exc = 16
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    n_q = 30
    xq = 0.05 * period .+ 0.9 * period .* rand(n_q)
    y_bar = randn(n_q)
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    f_bar_nobc = linear_adjoint(x_inc, xq)(y_bar)              # length nx_exc+1
    f_bar_inc  = linear_adjoint(x_inc, xq; bc = bc_inc)(y_bar) # length nx_exc+1
    f_bar_exc  = linear_adjoint(x_exc, xq; bc = bc_exc)(y_bar) # length nx_exc (post seam fold)

    # NoBC == inclusive on the closed n+1-grid (no extension/fold either way).
    @test f_bar_nobc ≈ f_bar_inc atol = 1.0e-13

    # Exclusive == NoBC after seam-fold + trim.
    folded = copy(f_bar_nobc)
    folded[1] += folded[end]
    @test f_bar_exc ≈ folded[1:nx_exc] atol = 1.0e-13
end


@testitem "1D Constant adjoint — NoBC == inclusive ≡ exclusive(after seam fold)" begin
    nx_exc = 16
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    n_q = 30
    xq = 0.05 * period .+ 0.9 * period .* rand(n_q)
    y_bar = randn(n_q)
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    f_bar_nobc = constant_adjoint(x_inc, xq; side = NearestSide())(y_bar)
    f_bar_inc  = constant_adjoint(x_inc, xq; side = NearestSide(), bc = bc_inc)(y_bar)
    f_bar_exc  = constant_adjoint(x_exc, xq; side = NearestSide(), bc = bc_exc)(y_bar)

    @test f_bar_nobc ≈ f_bar_inc atol = 1.0e-13

    folded = copy(f_bar_nobc)
    folded[1] += folded[end]
    @test f_bar_exc ≈ folded[1:nx_exc] atol = 1.0e-13
end


# ─────────────────────────────────────────────────────────────────────────
# Local-Hermite family — NoBC ≠ Periodic, but inclusive == exclusive
#
# To guarantee an *observable* NoBC vs Periodic discrepancy, we use
# monotonically increasing `f_exc` (closed via `vcat(f_exc, f_exc[1])`) —
# the resulting `f_inc` has a sharp drop across the seam so:
#   - NoBC boundary slope: one-sided FD over the smoothly-rising stencil
#     (sees positive slopes everywhere near i=1).
#   - Periodic boundary slope: closed-cycle stencil that includes the
#     large-negative seam secant `(f[1] - f[end]) / h_seam`. With sign
#     mismatch this triggers PCHIP's zero-clamp branch (`dy[1] = 0`) or
#     a sharply different value for Cardinal/Akima.
# This breaks the sin/cos symmetry coincidence (where m_seam happens to
# equal m_1 by sin's analytic relation).
#
# The 6 testitems below use `xq = period .* rand(n_q)` plus a deterministic
# boundary-cell query prepended via `vcat`. Pure random alone has a non-zero
# probability of missing all radius-1 boundary cells on a 12-cell grid
# (P((10/12)^30) ≈ 0.4%) — a real CI flake risk. The deterministic prepend
# guarantees the `!isapprox(y_nobc, y_inc)` assertion always observes the
# boundary-stencil difference.
# ─────────────────────────────────────────────────────────────────────────


# Regression guard: with seed 579 the 30 random uniforms all land in
# [1/12, 11/12] (no boundary-cell sample), so NoBC and Periodic agree at every
# query and the bare-random pattern `xq = period .* rand(n_q)` would falsely
# silence the difference. The vcat-boundary prepend used in the testitems
# below restores deterministic detection.
@testitem "PCHIP NoBC≠Periodic — deterministic boundary required (flake guard)" begin
    using Random
    nx_exc = 12
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    f_exc = collect(range(-1.0, 1.0, length = nx_exc))
    f_inc = vcat(f_exc, f_exc[1])
    n_q = 30
    bc_inc = PeriodicBC()

    Random.seed!(579)   # produces 30 uniforms entirely in [1/12, 11/12]
    xq_random = period .* rand(n_q)
    @test all(q -> q >= h && q <= period - h, xq_random)   # precondition: misses boundaries

    # Bare-random pattern (current existing test pattern) — under this seed,
    # NoBC and Periodic agree at every interior query → `!isapprox` would
    # silently fail to detect the boundary-cell semantic difference.
    y_nobc_r = pchip_interp(x_inc, f_inc, xq_random)
    y_inc_r  = pchip_interp(x_inc, f_inc, xq_random; bc = bc_inc)
    @test isapprox(y_nobc_r, y_inc_r; atol = 1.0e-12)  # documents the gap

    # Deterministic boundary prepend (the fix applied to all 6 testitems below).
    xq_pinned = vcat(0.5 * h, xq_random)
    y_nobc_p = pchip_interp(x_inc, f_inc, xq_pinned)
    y_inc_p  = pchip_interp(x_inc, f_inc, xq_pinned; bc = bc_inc)
    @test !isapprox(y_nobc_p, y_inc_p; rtol = 1.0e-3)
end

@testitem "1D PCHIP forward — NoBC ≠ Periodic, inclusive ≈ exclusive" begin
    nx_exc = 12
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    # Use a function whose monotonicity changes near the seam — exposes the
    # boundary slope difference (one-sided NoBC vs cyclic Periodic).
    # Monotonically increasing data, closed by vcat(f_exc, f_exc[1]) — gives
    # a sharp seam drop that GUARANTEES NoBC's one-sided FD disagrees with
    # Periodic's closed-cycle stencil at boundary indices. Avoids the
    # sin/cos-on-uniform-grid coincidence where m_seam = m_1 analytically.
    f_exc = collect(range(-1.0, 1.0, length = nx_exc))
    f_inc = vcat(f_exc, f_exc[1])  # `:inclusive` constraint (sharp seam drop)

    n_q = 30
    xq = vcat(0.5 * h, period - 0.5 * h, period .* rand(n_q - 2))   # full period, includes near-boundary queries
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    y_nobc = pchip_interp(x_inc, f_inc, xq)
    y_inc  = pchip_interp(x_inc, f_inc, xq; bc = bc_inc)
    y_exc  = pchip_interp(x_exc, f_exc, xq; bc = bc_exc)

    # Inclusive vs exclusive: same closed-cycle slope formula, equivalent up
    # to floating-point rounding from different stencil arithmetic.
    @test y_inc ≈ y_exc atol = 1.0e-12

    # NoBC differs from Periodic at boundary cells — assertable when there
    # exists ≥1 query in a boundary cell.
    @test !isapprox(y_nobc, y_inc; rtol = 1.0e-3)
end


@testitem "1D Cardinal forward — NoBC ≠ Periodic, inclusive ≈ exclusive" begin
    nx_exc = 12
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    # Monotonically increasing data, closed by vcat(f_exc, f_exc[1]) — gives
    # a sharp seam drop that GUARANTEES NoBC's one-sided FD disagrees with
    # Periodic's closed-cycle stencil at boundary indices. Avoids the
    # sin/cos-on-uniform-grid coincidence where m_seam = m_1 analytically.
    f_exc = collect(range(-1.0, 1.0, length = nx_exc))
    f_inc = vcat(f_exc, f_exc[1])  # `:inclusive` constraint (sharp seam drop)

    n_q = 30
    xq = vcat(0.5 * h, period - 0.5 * h, period .* rand(n_q - 2))
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    y_nobc = cardinal_interp(x_inc, f_inc, xq)
    y_inc  = cardinal_interp(x_inc, f_inc, xq; bc = bc_inc)
    y_exc  = cardinal_interp(x_exc, f_exc, xq; bc = bc_exc)

    @test y_inc ≈ y_exc atol = 1.0e-12
    @test !isapprox(y_nobc, y_inc; rtol = 1.0e-3)
end


@testitem "1D Akima forward — NoBC ≠ Periodic, inclusive ≈ exclusive" begin
    nx_exc = 12
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    # Monotonically increasing data, closed by vcat(f_exc, f_exc[1]) — gives
    # a sharp seam drop that GUARANTEES NoBC's one-sided FD disagrees with
    # Periodic's closed-cycle stencil at boundary indices. Avoids the
    # sin/cos-on-uniform-grid coincidence where m_seam = m_1 analytically.
    f_exc = collect(range(-1.0, 1.0, length = nx_exc))
    f_inc = vcat(f_exc, f_exc[1])  # `:inclusive` constraint (sharp seam drop)

    n_q = 30
    xq = vcat(0.5 * h, period - 0.5 * h, period .* rand(n_q - 2))
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    y_nobc = akima_interp(x_inc, f_inc, xq)
    y_inc  = akima_interp(x_inc, f_inc, xq; bc = bc_inc)
    y_exc  = akima_interp(x_exc, f_exc, xq; bc = bc_exc)

    @test y_inc ≈ y_exc atol = 1.0e-12
    @test !isapprox(y_nobc, y_inc; rtol = 1.0e-3)
end


@testitem "1D PCHIP adjoint — NoBC ≠ Periodic, inclusive ≡ exclusive(seam fold)" begin
    nx_exc = 12
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    # Monotonically increasing data, closed by vcat(f_exc, f_exc[1]) — gives
    # a sharp seam drop that GUARANTEES NoBC's one-sided FD disagrees with
    # Periodic's closed-cycle stencil at boundary indices. Avoids the
    # sin/cos-on-uniform-grid coincidence where m_seam = m_1 analytically.
    f_exc = collect(range(-1.0, 1.0, length = nx_exc))
    f_inc = vcat(f_exc, f_exc[1])  # `:inclusive` constraint (sharp seam drop)
    n_q = 30
    xq = vcat(0.5 * h, period - 0.5 * h, period .* rand(n_q - 2))
    y_bar = randn(n_q)
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    f_bar_nobc = pchip_adjoint(x_inc, f_inc, xq)(y_bar)              # length nx_exc+1
    f_bar_inc  = pchip_adjoint(x_inc, f_inc, xq; bc = bc_inc)(y_bar) # length nx_exc+1
    f_bar_exc  = pchip_adjoint(x_exc, f_exc, xq; bc = bc_exc)(y_bar) # length nx_exc

    # Inclusive ≡ exclusive (after fold).
    folded = copy(f_bar_inc)
    folded[1] += folded[end]
    @test f_bar_exc ≈ folded[1:nx_exc] atol = 1.0e-11

    # NoBC ≠ Periodic (different boundary slope adjoint chains).
    @test !isapprox(f_bar_nobc, f_bar_inc; rtol = 1.0e-3)
end


@testitem "1D Cardinal adjoint — NoBC ≠ Periodic, inclusive ≡ exclusive(seam fold)" begin
    nx_exc = 12
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    n_q = 30
    xq = vcat(0.5 * h, period - 0.5 * h, period .* rand(n_q - 2))
    y_bar = randn(n_q)
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    f_bar_nobc = cardinal_adjoint(x_inc, xq)(y_bar)
    f_bar_inc  = cardinal_adjoint(x_inc, xq; bc = bc_inc)(y_bar)
    f_bar_exc  = cardinal_adjoint(x_exc, xq; bc = bc_exc)(y_bar)

    folded = copy(f_bar_inc)
    folded[1] += folded[end]
    @test f_bar_exc ≈ folded[1:nx_exc] atol = 1.0e-12

    @test !isapprox(f_bar_nobc, f_bar_inc; rtol = 1.0e-3)
end


@testitem "1D Akima adjoint — NoBC ≠ Periodic, inclusive ≡ exclusive(seam fold)" begin
    nx_exc = 12
    period = 1.0
    h = period / nx_exc
    x_exc = collect(range(0.0, step = h, length = nx_exc))
    x_inc = vcat(x_exc, x_exc[1] + period)
    # Monotonically increasing data, closed by vcat(f_exc, f_exc[1]) — gives
    # a sharp seam drop that GUARANTEES NoBC's one-sided FD disagrees with
    # Periodic's closed-cycle stencil at boundary indices. Avoids the
    # sin/cos-on-uniform-grid coincidence where m_seam = m_1 analytically.
    f_exc = collect(range(-1.0, 1.0, length = nx_exc))
    f_inc = vcat(f_exc, f_exc[1])  # `:inclusive` constraint (sharp seam drop)
    n_q = 30
    xq = vcat(0.5 * h, period - 0.5 * h, period .* rand(n_q - 2))
    y_bar = randn(n_q)
    bc_inc = PeriodicBC()
    bc_exc = PeriodicBC(endpoint = :exclusive, period = period)

    f_bar_nobc = akima_adjoint(x_inc, f_inc, xq)(y_bar)
    f_bar_inc  = akima_adjoint(x_inc, f_inc, xq; bc = bc_inc)(y_bar)
    f_bar_exc  = akima_adjoint(x_exc, f_exc, xq; bc = bc_exc)(y_bar)

    folded = copy(f_bar_inc)
    folded[1] += folded[end]
    @test f_bar_exc ≈ folded[1:nx_exc] atol = 1.0e-11

    @test !isapprox(f_bar_nobc, f_bar_inc; rtol = 1.0e-3)
end
