# ============================================================================
# Phase 6d: Hint vs No-Hint Performance Comparison (Cell-Local OnTheFly ND)
# ============================================================================
#
# Quantifies the value of user-provided hints when the OnTheFly path uses
# cell-local windowing. After Phase 3, the inner kernel only sees a 4-6 point
# stencil per axis — so the *inner* search is essentially free regardless of
# hint usage. The hint payoff comes from the *outer* `_search_all_intervals`
# call at the windowing entry point: with a good hint, that pre-search
# short-circuits in ~1 comparison; without it, the binary search costs ~log2(n).
#
# Expected outcome: hints still help, but the speedup is *less dramatic* than
# the 5–50× of the global-solve path, because the inner stencil search is
# already O(log 4) ≈ 2 comparisons.
#
# Coverage:
#   - 100×100 OnTheFly Cardinal/Pchip/Akima ND
#   - Sorted (best case for hint), random (worst case), clustered queries
#   - Hint vs no-hint timings + ratio
#
# Run: julia --project benchmark/onthefly_nd_hint_comparison.jl

using FastInterpolations
using BenchmarkTools
using Random
using Printf

# ----- test data -----
const N = 100
const x = collect(range(0.0, 2π, N))
const y = collect(range(-1.0, 1.0, N))
const data2d = [sin(2xi) * exp(-yj^2) for xi in x, yj in y]

# ----- query patterns -----
const NQ = 1000

function sorted_queries(seed::Int)
    rng = MersenneTwister(seed)
    qx = sort!(0.5 .+ 5.5 .* rand(rng, NQ))
    qy = sort!(-0.9 .+ 1.8 .* rand(rng, NQ))
    return collect(zip(qx, qy))
end

function random_queries(seed::Int)
    rng = MersenneTwister(seed)
    qx = 0.5 .+ 5.5 .* rand(rng, NQ)
    qy = -0.9 .+ 1.8 .* rand(rng, NQ)
    return collect(zip(qx, qy))
end

function clustered_queries(seed::Int)
    rng = MersenneTwister(seed)
    # 10 clusters of 100 queries each, each cluster sorted
    clusters = []
    for _ in 1:10
        cx = 0.5 + 5.5 * rand(rng)
        cy = -0.9 + 1.8 * rand(rng)
        qs = [(cx + 0.05 * randn(rng), cy + 0.05 * randn(rng)) for _ in 1:100]
        sort!(qs, by = q -> q[1])
        append!(clusters, qs)
    end
    return clusters
end

# ----- driver -----
function run_queries_no_hint(itp, qs)
    s = 0.0
    @inbounds for q in qs
        s += itp(q)
    end
    return s
end

function run_queries_with_hint(itp, qs, hint)
    s = 0.0
    @inbounds for q in qs
        s += itp(q; hint = hint)
    end
    return s
end

function bench_pattern(label::String, methods, qs)
    itp = interp((x, y), data2d; method = methods, coeffs = OnTheFly())
    # Warmup
    run_queries_no_hint(itp, qs[1:5])
    hint = (Ref(1), Ref(1))
    run_queries_with_hint(itp, qs[1:5], hint)

    bn = @benchmark run_queries_no_hint($itp, $qs)               samples = 100 evals = 1
    bh = @benchmark run_queries_with_hint($itp, $qs, $hint)       samples = 100 evals = 1

    tn = minimum(bn).time / NQ      # ns per query
    th = minimum(bh).time / NQ
    speedup = tn / th
    @info "$label" no_hint_ns_per_q = round(tn, digits = 1) hint_ns_per_q = round(th, digits = 1) speedup = round(speedup, digits = 2)
    return (label, tn, th, speedup)
end

println("="^78)
println("Phase 6d: Hint vs No-Hint OnTheFly ND (cell-local windowing)")
println("Grid: $N × $N, Queries: $NQ per pattern")
println("="^78)

results = []
for (mname, methods) in (
        ("Cardinal × Cardinal", (CardinalInterp(), CardinalInterp())),
        ("Pchip × Pchip", (PchipInterp(), PchipInterp())),
        ("Akima × Akima", (AkimaInterp(), AkimaInterp())),
        ("Cubic × Cardinal", (CubicInterp(), CardinalInterp())),
        ("Linear × Pchip", (LinearInterp(), PchipInterp())),
    )
    println("\n--- $mname ---")
    push!(results, bench_pattern("$mname  / sorted", methods, sorted_queries(1)))
    push!(results, bench_pattern("$mname  / random", methods, random_queries(1)))
    push!(results, bench_pattern("$mname  / clustered", methods, clustered_queries(1)))
end

println("\n" * "="^78)
println("Summary (per-query ns):")
println("="^78)
@printf "%-40s  %12s  %12s  %10s\n" "Method × Pattern" "no_hint" "with_hint" "speedup"
for (label, tn, th, sp) in results
    @printf "%-40s  %10.1f ns  %10.1f ns  %9.2fx\n" label tn th sp
end
