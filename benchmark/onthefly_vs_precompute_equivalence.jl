# ============================================================================
# OnTheFly vs PreCompute (Cubic global-solve) + OnTheFly one-shot vs
# OnTheFly persistent interpolant (Hermite local methods)
# ============================================================================
#
# Two orthogonal comparisons:
#
# (A) Cubic × Cubic (pure global-solve)
#     Build both an `coeffs=OnTheFly()` and a `coeffs=PreCompute()` interpolant
#     over the same data and evaluate the same query stream. Verifies:
#       - numerical equivalence (should match within a few ULPs; CubicInterp
#         solves the same tridiagonal system both ways, only *when* it solves
#         differs)
#       - performance ordering (PreCompute amortizes the solve, so persistent
#         evaluation should be ~10²-10⁴× faster per query than OnTheFly)
#
# (B) Pchip / Cardinal / Akima / Linear (local Hermite methods)
#     Compare three OnTheFly evaluation entry points on the same data + query:
#       1. `interp(grids, data, q; method, coeffs=OnTheFly())`       (scalar one-shot)
#       2. `itp = interp(grids, data; method, coeffs=OnTheFly()); itp(q)`  (persistent)
#       3. `interp(grids, data, [q...]; method, coeffs=OnTheFly())`  (batch one-shot)
#
#     Verifies:
#       - bit-exact agreement across all three paths (value queries)
#       - performance ordering: persistent interpolant should match scalar
#         one-shot closely (they share `_collapse_dims` post-PR), and batch
#         one-shot amortizes the spacings precompute
#
# Grid: 100×100, evaluated at interior + boundary sweeps
# Run: julia --project=benchmark benchmark/onthefly_vs_precompute_equivalence.jl

using FastInterpolations
using BenchmarkTools
using Printf
using Random

# ---------- test data ----------
const N = 100
const x = collect(range(0.0, 2π, N))
const y = collect(range(-1.0, 1.0, N))
const data2d = [sin(2xi) * exp(-yj^2) for xi in x, yj in y]

# Query set: a mix of interior + near-boundary + deep-interior queries,
# deterministically generated. NQ is small-ish so per-query timing stays
# clean of cache-eviction noise; we use minimum() over many samples.
const NQ = 500
function make_queries(seed::Int = 1)
    rng = MersenneTwister(seed)
    qx = 0.2 .+ (2π - 0.4) .* rand(rng, NQ)
    qy = -0.9 .+ 1.8 .* rand(rng, NQ)
    return collect(zip(qx, qy))
end
const queries = make_queries(1)
const q1 = queries[1]  # single-query probe

# ---------- equivalence helper ----------
# Return (max_abs_diff, max_ulp_diff) between two vectors.
#
# ULP metric: we floor the scale at 1.0 so that near-zero values are measured
# in ULPs-of-1.0 (≈ eps(Float64) ≈ 2.2e-16). Without the floor, `eps(x)` shrinks
# as |x| → 0 and a diff of 1e-17 near 1e-10 would report as ~50 000 ULPs, which
# is noise, not signal.
function diff_stats(a::AbstractVector, b::AbstractVector)
    @assert length(a) == length(b)
    max_abs = 0.0
    max_ulp = 0
    for i in eachindex(a)
        d = abs(a[i] - b[i])
        if d > max_abs
            max_abs = d
        end
        scale = max(abs(a[i]), abs(b[i]), 1.0)   # floor at 1.0 → absolute ULP below unity
        u = round(Int, d / eps(scale))
        if u > max_ulp
            max_ulp = u
        end
    end
    return (max_abs, max_ulp)
end

# ---------- drivers (closures are hot-loop-friendly since everything is const) ----------
function eval_loop_itp(itp, qs)
    s = 0.0
    @inbounds for q in qs
        s += itp(q)
    end
    return s
end

function eval_loop_oneshot(grids, data, qs, method)
    s = 0.0
    @inbounds for q in qs
        s += interp(grids, data, q; method = method, coeffs = OnTheFly())
    end
    return s
end

# Collect the full result vector so we can check equivalence element-wise.
function collect_itp(itp, qs)
    out = Vector{Float64}(undef, length(qs))
    @inbounds for i in eachindex(qs)
        out[i] = itp(qs[i])
    end
    return out
end

function collect_oneshot(grids, data, qs, method)
    out = Vector{Float64}(undef, length(qs))
    @inbounds for i in eachindex(qs)
        out[i] = interp(grids, data, qs[i]; method = method, coeffs = OnTheFly())
    end
    return out
end

# ============================================================================
# (A) Cubic × Cubic: OnTheFly vs PreCompute persistent interpolant
# ============================================================================
println("="^78)
println("(A) Cubic × Cubic — OnTheFly vs PreCompute persistent interpolant")
println("    Grid: $N × $N, Queries: $NQ")
println("="^78)

let method = (CubicInterp(), CubicInterp())
    itp_otf = interp((x, y), data2d; method = method, coeffs = OnTheFly())
    itp_pc = interp((x, y), data2d; method = method, coeffs = PreCompute())

    # Warmup
    itp_otf(q1); itp_pc(q1)

    # Equivalence check
    r_otf = collect_itp(itp_otf, queries)
    r_pc = collect_itp(itp_pc, queries)
    (dabs, dulp) = diff_stats(r_otf, r_pc)
    @printf "  value equivalence: max|Δ| = %.3e   max ULP = %d\n" dabs dulp

    # Per-query timing
    b_otf = @benchmark eval_loop_itp($itp_otf, $queries) samples = 50 evals = 1
    b_pc = @benchmark eval_loop_itp($itp_pc, $queries) samples = 50 evals = 1
    t_otf = minimum(b_otf).time / NQ
    t_pc = minimum(b_pc).time / NQ
    @printf "  OnTheFly  persistent: %8.1f ns/query   (%d alloc/loop)\n" t_otf b_otf.allocs
    @printf "  PreCompute persistent: %8.1f ns/query   (%d alloc/loop)\n" t_pc  b_pc.allocs
    @printf "  speedup (PreCompute / OnTheFly): %.1f×\n" (t_otf / t_pc)
end

# ============================================================================
# (B) Hermite local methods: one-shot vs persistent interpolant (both OnTheFly)
# ============================================================================
println("\n" * "="^78)
println("(B) Local Hermite — one-shot vs persistent interpolant (both OnTheFly)")
println("    All paths share `_collapse_dims` with cell-local windowing.")
println("="^78)

const hermite_configs = (
    ("Cardinal × Cardinal", (CardinalInterp(), CardinalInterp())),
    ("Pchip × Pchip", (PchipInterp(), PchipInterp())),
    ("Akima × Akima", (AkimaInterp(), AkimaInterp())),
    ("Linear × Linear", (LinearInterp(), LinearInterp())),
    ("Cubic × Cardinal", (CubicInterp(), CardinalInterp())),  # mixed: partial fast path
    # Asymmetric windowing demos: persistent path uses _has_any_windowable_method,
    # so Linear/Constant axes trigger the pre-slice + fast-path too, and when
    # paired with a Cubic axis the windowing indirectly shrinks the Cubic
    # tridiagonal solve from N fibers to 2.
    ("Cubic × Linear", (CubicInterp(), LinearInterp())),
    ("Linear × Cubic", (LinearInterp(), CubicInterp())),
    ("Constant × Linear", (ConstantInterp(), LinearInterp())),
)

# Pretty header
# Canonical equivalence check is persistent-vs-batch (both share windowed
# `_collapse_dims`). Scalar one-shot may route through a specialized path for
# pure-homogeneous tuples (e.g. linear_interp_nd_oneshot), so small cross-path
# ULP drift vs. one-shot is informational, not a correctness signal for this PR.
@printf "\n%-22s  %10s  %10s  %10s  %14s  %14s\n" "config" "one-shot" "persistent" "batch" "persist=batch" "vs one-shot"
println("-"^94)

for (label, method) in hermite_configs
    # Build persistent interpolant
    itp = interp((x, y), data2d; method = method, coeffs = OnTheFly())

    # Warmup all three paths
    interp((x, y), data2d, q1; method = method, coeffs = OnTheFly())
    itp(q1)
    v3_tmp = Vector{Float64}(undef, 1)
    interp!(v3_tmp, (x, y), data2d, [q1]; method = method, coeffs = OnTheFly())

    # Element-wise results for every path
    r_oneshot = collect_oneshot((x, y), data2d, queries, method)
    r_persist = collect_itp(itp, queries)
    r_batch = interp((x, y), data2d, queries; method = method, coeffs = OnTheFly())

    # (1) Canonical: persistent vs batch — must be bit-exact for this PR
    (dpb_abs, dpb_ulp) = diff_stats(r_persist, r_batch)
    pb_verdict = dpb_abs == 0.0 ? "BIT-EXACT" : "ULP=$(dpb_ulp)"

    # (2) Informational: persistent vs one-shot — may differ if one-shot routes
    #     through a specialized homogeneous kernel (e.g. pure Linear)
    (dpo_abs, dpo_ulp) = diff_stats(r_persist, r_oneshot)
    po_verdict = dpo_abs == 0.0 ? "BIT-EXACT" : "ULP=$(dpo_ulp)"

    # Timing — minimum over samples, normalized per query
    b_oneshot = @benchmark eval_loop_oneshot($((x, y)), $data2d, $queries, $method) samples = 50 evals = 1
    b_persist = @benchmark eval_loop_itp($itp, $queries) samples = 100 evals = 1
    b_batch = @benchmark interp($((x, y)), $data2d, $queries; method = $method, coeffs = OnTheFly()) samples = 50 evals = 1

    t_oneshot = minimum(b_oneshot).time / NQ
    t_persist = minimum(b_persist).time / NQ
    t_batch = minimum(b_batch).time / NQ

    @printf "%-22s  %7.1f ns  %7.1f ns  %7.1f ns  %14s  %14s\n" label t_oneshot t_persist t_batch pb_verdict po_verdict
end

# ============================================================================
# (C) Sanity: allocation check for persistent OnTheFly interpolant
# ============================================================================
println("\n" * "="^78)
println("(C) Persistent OnTheFly allocation check (must be 0 after warmup)")
println("="^78)

function alloc_persist(method)
    itp = interp((x, y), data2d; method = method, coeffs = OnTheFly())
    itp(q1); itp(q1)
    return @allocated itp(q1)
end

for (label, method) in hermite_configs
    a = alloc_persist(method)
    @printf "  %-22s : %d bytes/query\n" label a
end

# Cubic persistent interpolant (both strategies) — check that OnTheFly global
# solve path is truly alloc-free on repeated query, matching pre-PR behavior.
let method = (CubicInterp(), CubicInterp())
    function alloc_cubic(strategy)
        itp = interp((x, y), data2d; method = method, coeffs = strategy)
        itp(q1); itp(q1)
        return @allocated itp(q1)
    end
    @printf "  %-22s : %d bytes/query  (OnTheFly)\n"   "Cubic × Cubic"      alloc_cubic(OnTheFly())
    @printf "  %-22s : %d bytes/query  (PreCompute)\n" "Cubic × Cubic"      alloc_cubic(PreCompute())
end

println("\nDone.")
