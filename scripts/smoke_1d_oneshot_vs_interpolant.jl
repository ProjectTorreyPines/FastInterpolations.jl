#!/usr/bin/env julia
#
# Smoke test: 1D Range grid Linear interpolation
#   - One-shot (`linear_interp(x, y, xq; bc, extrap)`) — Phase 3 refactor target
#   - vs persistent interpolant (`itp(xq)`)
#
# Verifies:
#   1. Zero-allocation on warmed one-shot (function-barrier pattern)
#   2. One-shot perf parity with persistent interpolant (target ≤ 2×)
#   3. Both NoBC and PeriodicBC(:exclusive) paths (NEW: Phase 3 no-pool path)
#
# Run: julia --project scripts/smoke_1d_oneshot_vs_interpolant.jl

using FastInterpolations
using Printf

const NWARMUP = 3
const NREPS = 100_000

@inline oneshot_nobc_1d(x, y, xq) = linear_interp(x, y, xq)
@inline oneshot_periodic_excl_1d(x, y, xq, bc) = linear_interp(x, y, xq; bc=bc)
@inline persistent_call_1d(itp, xq) = itp(xq)

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
    return best * 1e9 / NREPS
end

function check_alloc(f, args...)
    for _ in 1:NWARMUP
        f(args...)
    end
    return @allocated f(args...)
end

function run_smoke_1d()
    println("=" ^ 70)
    println("1D Range Linear Interpolation Smoke Test  (Phase 3 target)")
    println("  One-shot vs Persistent Interpolant, NoBC / PeriodicBC(:excl)")
    println("=" ^ 70)

    n = 100
    x = range(0.0, 2π, length=n)
    y = sin.(x)

    x_excl = range(0.0, step=2π/n, length=n)
    y_excl = sin.(x_excl)

    xq = 1.5
    bc_excl = PeriodicBC(endpoint=:exclusive, period=2π)

    println()
    println("── Build persistent interpolants")
    itp_nobc = linear_interp(x, y)
    itp_excl = linear_interp(x_excl, y_excl; bc=bc_excl)
    println("  itp_nobc  = ", typeof(itp_nobc).name.wrapper, "{...}")
    println("  itp_excl  = ", typeof(itp_excl).name.wrapper, "{...}")

    println()
    println("── Correctness spot-check")
    val_os_nobc = oneshot_nobc_1d(x, y, xq)
    val_itp_nobc = persistent_call_1d(itp_nobc, xq)
    println(@sprintf("  NoBC       : oneshot=%.15e  itp=%.15e  diff=%.2e",
        val_os_nobc, val_itp_nobc, abs(val_os_nobc - val_itp_nobc)))
    val_os_excl = oneshot_periodic_excl_1d(x_excl, y_excl, xq, bc_excl)
    val_itp_excl = persistent_call_1d(itp_excl, xq)
    println(@sprintf("  PeriodicBC : oneshot=%.15e  itp=%.15e  diff=%.2e",
        val_os_excl, val_itp_excl, abs(val_os_excl - val_itp_excl)))

    # Seam query
    xq_seam = x_excl[n] + 0.01
    val_os_seam = oneshot_periodic_excl_1d(x_excl, y_excl, xq_seam, bc_excl)
    val_itp_seam = persistent_call_1d(itp_excl, xq_seam)
    println(@sprintf("  Seam xq    : oneshot=%.15e  itp=%.15e  diff=%.2e (expected ≈0)",
        val_os_seam, val_itp_seam, abs(val_os_seam - val_itp_seam)))

    println()
    println("── Zero-alloc verification (warmed, function-barrier)")
    alloc_os_nobc = check_alloc(oneshot_nobc_1d, x, y, xq)
    alloc_os_excl = check_alloc(oneshot_periodic_excl_1d, x_excl, y_excl, xq, bc_excl)
    alloc_itp_nobc = check_alloc(persistent_call_1d, itp_nobc, xq)
    alloc_itp_excl = check_alloc(persistent_call_1d, itp_excl, xq)
    println(@sprintf("  oneshot  NoBC     : %d bytes  %s", alloc_os_nobc, alloc_os_nobc == 0 ? "✓" : "✗"))
    println(@sprintf("  oneshot  Periodic : %d bytes  %s", alloc_os_excl, alloc_os_excl == 0 ? "✓" : "✗"))
    println(@sprintf("  itp      NoBC     : %d bytes  %s", alloc_itp_nobc, alloc_itp_nobc == 0 ? "✓" : "✗"))
    println(@sprintf("  itp      Periodic : %d bytes  %s", alloc_itp_excl, alloc_itp_excl == 0 ? "✓" : "✗"))

    println()
    println("── Timing (ns/call, min over 5×100k reps after warmup)")
    t_os_nobc = time_call(oneshot_nobc_1d, x, y, xq)
    t_os_excl = time_call(oneshot_periodic_excl_1d, x_excl, y_excl, xq, bc_excl)
    t_itp_nobc = time_call(persistent_call_1d, itp_nobc, xq)
    t_itp_excl = time_call(persistent_call_1d, itp_excl, xq)

    ratio_nobc = t_os_nobc / t_itp_nobc
    ratio_excl = t_os_excl / t_itp_excl
    println(@sprintf("  oneshot  NoBC     : %7.2f ns   itp NoBC     : %7.2f ns   ratio: %.2fx",
        t_os_nobc, t_itp_nobc, ratio_nobc))
    println(@sprintf("  oneshot  Periodic : %7.2f ns   itp Periodic : %7.2f ns   ratio: %.2fx",
        t_os_excl, t_itp_excl, ratio_excl))

    println()
    println("── Summary")
    zero_alloc_ok = (alloc_os_nobc == 0) && (alloc_os_excl == 0)
    println(zero_alloc_ok ? "  ✓ Zero-alloc oneshot" : "  ✗ Zero-alloc oneshot FAILED")
    parity_ok = ratio_nobc <= 2.0 && ratio_excl <= 2.0
    if parity_ok
        println(@sprintf("  ✓ Oneshot ≤ 2× persistent (NoBC: %.2fx, Excl: %.2fx)", ratio_nobc, ratio_excl))
    elseif ratio_nobc <= 3.0 && ratio_excl <= 3.0
        println(@sprintf("  ⚠ Oneshot 2-3× persistent (NoBC: %.2fx, Excl: %.2fx)", ratio_nobc, ratio_excl))
    else
        println(@sprintf("  ✗ Oneshot > 3× persistent (NoBC: %.2fx, Excl: %.2fx)", ratio_nobc, ratio_excl))
    end
    println()
end

run_smoke_1d()
