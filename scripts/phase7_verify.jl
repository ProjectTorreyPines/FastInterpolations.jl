# Phase 7 verification: type stability, allocation, real-grid speedup
using FastInterpolations
using BenchmarkTools
using InteractiveUtils  # for code_warntype

println("="^70)
println("Phase 7: Type stability, allocation, and speedup verification")
println("="^70)

# ── 7a: SubArray fast path / type stability ──
println("\n[7a] @code_warntype on Cardinal × Cardinal 2D itp(query)")
let
    x = collect(range(0.0, 2π, 30))
    y = collect(range(-1.0, 1.0, 25))
    data = [sin(2xi) * cos(yj) for xi in x, yj in y]
    itp = interp((x, y), data; method = (CardinalInterp(), CardinalInterp()), coeffs = OnTheFly())
    # Warmup
    itp((1.5, 0.4))
    println("\nType-stable check (no Union/Any in return path):")
    code_warntype(itp, Tuple{Tuple{Float64, Float64}})
end

# ── 7c: Allocation check ──
println("\n" * "="^70)
println("[7c] Allocation check — must be 0 after warmup")
println("="^70)

function alloc_check_pure_local(N)
    x = collect(range(0.0, 2π, N))
    y = collect(range(-1.0, 1.0, N))
    data = [sin(2xi) * cos(yj) for xi in x, yj in y]
    itp = interp((x, y), data; method = (CardinalInterp(), CardinalInterp()), coeffs = OnTheFly())
    itp((1.5, 0.4))
    itp((1.5, 0.4))
    return @allocated itp((1.5, 0.4))
end

function alloc_check_mixed(N)
    x = collect(range(0.0, 2π, N))
    y = collect(range(-1.0, 1.0, N))
    data = [sin(2xi) * cos(yj) for xi in x, yj in y]
    itp = interp((x, y), data; method = (CubicInterp(), CardinalInterp()), coeffs = OnTheFly())
    itp((1.5, 0.4))
    itp((1.5, 0.4))
    return @allocated itp((1.5, 0.4))
end

function alloc_check_3d_pchip(N)
    x = collect(range(0.0, 2π, N))
    y = collect(range(-1.0, 1.0, N))
    z = collect(range(0.0, 1.0, N))
    data = [sin(2xi) * cos(yj) * (1 + zk) for xi in x, yj in y, zk in z]
    itp = interp((x, y, z), data; method = (PchipInterp(), PchipInterp(), PchipInterp()), coeffs = OnTheFly())
    itp((1.5, 0.4, 0.5))
    itp((1.5, 0.4, 0.5))
    return @allocated itp((1.5, 0.4, 0.5))
end

println("\nCardinal × Cardinal 2D (N=30):  ", alloc_check_pure_local(30), " bytes")
println("Cardinal × Cardinal 2D (N=100): ", alloc_check_pure_local(100), " bytes")
println("Cubic × Cardinal 2D (N=100):    ", alloc_check_mixed(100), " bytes")
println("Pchip × Pchip × Pchip 3D (N=20):", alloc_check_3d_pchip(20), " bytes")

# ── 7b: Real-grid speedup vs full-grid traversal ──
# Compare cell-local windowed path against the full-grid global-solve path with
# the same data — they're functionally different but the former should be much
# faster on local Hermite tuples.
println("\n" * "="^70)
println("[7b] Per-query timing on a 100×100 grid")
println("="^70)

let
    x = collect(range(0.0, 2π, 100))
    y = collect(range(-1.0, 1.0, 100))
    data = [sin(2xi) * cos(yj) for xi in x, yj in y]
    q = (1.5, 0.4)

    # Cell-local windowed path
    itp_card = interp((x, y), data; method = (CardinalInterp(), CardinalInterp()), coeffs = OnTheFly())
    itp_card(q)
    b_card = @benchmark $itp_card($q) samples = 5000 evals = 1
    println("Cardinal × Cardinal 2D (cell-local windowed): ", round(minimum(b_card).time, digits = 1), " ns")

    # Pchip
    itp_pchip = interp((x, y), data; method = (PchipInterp(), PchipInterp()), coeffs = OnTheFly())
    itp_pchip(q)
    b_pchip = @benchmark $itp_pchip($q) samples = 5000 evals = 1
    println("Pchip × Pchip 2D (cell-local windowed):       ", round(minimum(b_pchip).time, digits = 1), " ns")

    # Akima (6-point window)
    itp_akima = interp((x, y), data; method = (AkimaInterp(), AkimaInterp()), coeffs = OnTheFly())
    itp_akima(q)
    b_akima = @benchmark $itp_akima($q) samples = 5000 evals = 1
    println("Akima × Akima 2D (cell-local windowed):       ", round(minimum(b_akima).time, digits = 1), " ns")

    # Cubic (full path — global solve, not windowed)
    itp_cubic = interp((x, y), data; method = (CubicInterp(), CubicInterp()), coeffs = OnTheFly())
    itp_cubic(q)
    b_cubic = @benchmark $itp_cubic($q) samples = 5000 evals = 1
    println("Cubic × Cubic 2D (full global solve):         ", round(minimum(b_cubic).time, digits = 1), " ns")
end

println("\nDone.")
