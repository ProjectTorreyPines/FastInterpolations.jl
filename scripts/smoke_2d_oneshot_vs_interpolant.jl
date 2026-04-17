#!/usr/bin/env julia
#
# Smoke test: 2D Range grid Linear interpolation
#   - One-shot (`linear_interp(grids, data, q; bc, extrap)`)
#   - vs persistent interpolant (`itp(q)`)
#
# Verifies:
#   1. Zero-allocation on warmed one-shot (function-barrier pattern)
#   2. One-shot perf parity with persistent interpolant (within ~2×)
#   3. Both NoBC and PeriodicBC(:exclusive) paths
#
# Run: julia --project scripts/smoke_2d_oneshot_vs_interpolant.jl

using FastInterpolations
using Printf

const NWARMUP = 3
const NREPS = 10_000

# ─────────────────────────────────────────────────────────────
# Function-barriers (forces compile with concrete types, no captured boxes)
# ─────────────────────────────────────────────────────────────

@inline function oneshot_nobc_2d(grids, data, q)
    return linear_interp(grids, data, q)
end

@inline function oneshot_periodic_excl_2d(grids, data, q, bc1, bc2)
    return linear_interp(grids, data, q; bc=(bc1, bc2))
end

@inline function persistent_call(itp, q)
    return itp(q)
end

# ─────────────────────────────────────────────────────────────
# Timing helper (min over reps after warmup; returns ns/call)
# ─────────────────────────────────────────────────────────────

function time_call(f, args...)
    for _ in 1:NWARMUP
        f(args...)
    end
    best = Inf
    for _ in 1:5
        t = @elapsed for _ in 1:NREPS
            f(args...)
        end
        best = min(best, t)
    end
    return best * 1e9 / NREPS  # ns/call
end

# ─────────────────────────────────────────────────────────────
# Allocation check (function-barrier @allocated, post-warmup)
# ─────────────────────────────────────────────────────────────

function check_alloc(f, args...)
    for _ in 1:NWARMUP
        f(args...)
    end
    return @allocated f(args...)
end

# ─────────────────────────────────────────────────────────────
# Main smoke test
# ─────────────────────────────────────────────────────────────

function run_smoke()
    println("=" ^ 70)
    println("2D Range Linear Interpolation Smoke Test")
    println("  One-shot vs Persistent Interpolant, NoBC / PeriodicBC(:excl)")
    println("=" ^ 70)

    # 2D Range grid
    nx, ny = 100, 100
    x = range(0.0, 2π, length=nx)
    y = range(-1.0, 1.0, length=ny)
    grids = (x, y)
    data = [sin(xi) * exp(-yj^2) for xi in x, yj in y]
    q = (1.5, 0.3)

    # For exclusive periodic (drop last column to avoid matched endpoint; period=2π)
    x_excl = range(0.0, step=2π/nx, length=nx)
    y_excl = y  # non-periodic axis
    grids_excl = (x_excl, y_excl)
    data_excl = [sin(xi) * exp(-yj^2) for xi in x_excl, yj in y_excl]
    q_excl = (1.5, 0.3)

    bc_nobc = (NoBC(), NoBC())
    bc_excl = (PeriodicBC(endpoint=:exclusive, period=2π), NoBC())

    println()
    println("── Build persistent interpolants (NoBC + PeriodicBC excl)")
    itp_nobc = linear_interp(grids, data)
    itp_excl = linear_interp(grids_excl, data_excl;
                              bc=bc_excl)
    println("  itp_nobc  = ", typeof(itp_nobc).name.wrapper, "{...}")
    println("  itp_excl  = ", typeof(itp_excl).name.wrapper, "{...}")

    println()
    println("── Correctness spot-check")
    val_os_nobc = oneshot_nobc_2d(grids, data, q)
    val_itp_nobc = persistent_call(itp_nobc, q)
    println(@sprintf("  NoBC       : oneshot=%.15e  itp=%.15e  diff=%.2e",
        val_os_nobc, val_itp_nobc, abs(val_os_nobc - val_itp_nobc)))
    val_os_excl = oneshot_periodic_excl_2d(grids_excl, data_excl, q_excl, bc_excl[1], bc_excl[2])
    val_itp_excl = persistent_call(itp_excl, q_excl)
    println(@sprintf("  PeriodicBC : oneshot=%.15e  itp=%.15e  diff=%.2e",
        val_os_excl, val_itp_excl, abs(val_os_excl - val_itp_excl)))

    println()
    println("── Zero-alloc verification (warmed, function-barrier)")
    alloc_os_nobc = check_alloc(oneshot_nobc_2d, grids, data, q)
    alloc_os_excl = check_alloc(oneshot_periodic_excl_2d, grids_excl, data_excl, q_excl, bc_excl[1], bc_excl[2])
    alloc_itp_nobc = check_alloc(persistent_call, itp_nobc, q)
    alloc_itp_excl = check_alloc(persistent_call, itp_excl, q_excl)
    println(@sprintf("  oneshot  NoBC     : %d bytes  %s", alloc_os_nobc, alloc_os_nobc == 0 ? "✓" : "✗"))
    println(@sprintf("  oneshot  Periodic : %d bytes  %s", alloc_os_excl, alloc_os_excl == 0 ? "✓" : "✗"))
    println(@sprintf("  itp      NoBC     : %d bytes  %s", alloc_itp_nobc, alloc_itp_nobc == 0 ? "✓" : "✗"))
    println(@sprintf("  itp      Periodic : %d bytes  %s", alloc_itp_excl, alloc_itp_excl == 0 ? "✓" : "✗"))

    println()
    println("── Timing (ns/call, min over 5×10k reps after warmup)")
    t_os_nobc = time_call(oneshot_nobc_2d, grids, data, q)
    t_os_excl = time_call(oneshot_periodic_excl_2d, grids_excl, data_excl, q_excl, bc_excl[1], bc_excl[2])
    t_itp_nobc = time_call(persistent_call, itp_nobc, q)
    t_itp_excl = time_call(persistent_call, itp_excl, q_excl)

    ratio_nobc = t_os_nobc / t_itp_nobc
    ratio_excl = t_os_excl / t_itp_excl
    println(@sprintf("  oneshot  NoBC     : %7.2f ns   itp NoBC     : %7.2f ns   ratio: %.2fx",
        t_os_nobc, t_itp_nobc, ratio_nobc))
    println(@sprintf("  oneshot  Periodic : %7.2f ns   itp Periodic : %7.2f ns   ratio: %.2fx",
        t_os_excl, t_itp_excl, ratio_excl))

    println()
    println("── Summary")
    all_pass = true
    zero_alloc_oneshot = (alloc_os_nobc == 0) && (alloc_os_excl == 0)
    if !zero_alloc_oneshot
        println("  ✗ Zero-alloc oneshot FAILED")
        all_pass = false
    else
        println("  ✓ Zero-alloc oneshot (NoBC + PeriodicBC excl)")
    end
    # Parity target: oneshot within 2× of persistent (good), within 3× (acceptable for 2D)
    if ratio_nobc <= 2.0 && ratio_excl <= 2.0
        println(@sprintf("  ✓ Oneshot ≤ 2× persistent (NoBC: %.2fx, Excl: %.2fx)", ratio_nobc, ratio_excl))
    elseif ratio_nobc <= 3.0 && ratio_excl <= 3.0
        println(@sprintf("  ⚠ Oneshot 2-3× persistent (NoBC: %.2fx, Excl: %.2fx) — acceptable for 2D", ratio_nobc, ratio_excl))
    else
        println(@sprintf("  ✗ Oneshot > 3× persistent (NoBC: %.2fx, Excl: %.2fx)", ratio_nobc, ratio_excl))
        all_pass = false
    end
    println()
    return all_pass
end

run_smoke()
