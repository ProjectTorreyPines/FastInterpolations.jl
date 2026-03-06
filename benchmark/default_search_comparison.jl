# ═══════════════════════════════════════════════════════════════════════════════
# Default Search Policy Comparison: Binary() vs LinearBinary(2)
# ═══════════════════════════════════════════════════════════════════════════════
#
# End-to-end benchmark measuring actual interpolation performance
# with the old default (Binary) vs new default (LinearBinary, window=2).
#
# Tests all 4 interpolation types × 5 query patterns × 2 grid sizes
# in both scalar-loop and vector-call modes.
#
# Run: julia --project benchmark/default_search_comparison.jl
# ═══════════════════════════════════════════════════════════════════════════════

using FastInterpolations
using BenchmarkTools
using Printf
using Random

# ═══════════════════════════════════════════════════════════════════════════════
# Query Pattern Generators
# ═══════════════════════════════════════════════════════════════════════════════
# Each returns a Float64 vector of `n` queries within [lo, hi].
# These simulate real-world usage patterns.

const PATTERNS = [
    # Random: worst case for LinearBinary — no locality at all
    ("Random", (n, lo, hi) -> rand(n) .* (hi - lo) .+ lo),

    # Sorted: typical ODE/PDE integration pattern — monotone increasing
    ("Sorted", (n, lo, hi) -> sort(rand(n) .* (hi - lo) .+ lo)),

    # Clustered: physics simulation — queries bunch around a few hotspots
    (
        "Clustered", (n, lo, hi) -> begin
            ctrs = range(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), length = 5)
            w = (hi - lo) * 0.03
            clamp!(vcat([sort(randn(n ÷ 5) .* w .+ c) for c in ctrs]...), lo, hi)
        end,
    ),

    # Reverse: sorted descending — tests backward walk in LinearBinary
    ("Reverse", (n, lo, hi) -> reverse(sort(rand(n) .* (hi - lo) .+ lo))),

    # Dense local: extreme locality — ODE with tiny timestep
    (
        "DenseLocal", (n, lo, hi) -> begin
            center = (lo + hi) / 2
            span = (hi - lo) * 0.02  # 2% of domain
            sort(rand(n) .* span .+ (center - span / 2))
        end,
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Harness: Scalar Loop (DCE-safe)
# ═══════════════════════════════════════════════════════════════════════════════
# Accumulates results to prevent dead-code elimination.

function bench_scalar_loop(itp, queries::Vector{Float64})
    acc = 0.0
    @inbounds for i in eachindex(queries)
        acc += itp(queries[i])
    end
    return acc
end

# With external hint — simulates persistent-hint pattern
function bench_scalar_loop_hint(itp, queries::Vector{Float64}, hint::Base.RefValue{Int})
    acc = 0.0
    @inbounds for i in eachindex(queries)
        acc += itp(queries[i]; hint = hint)
    end
    return acc
end

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Harness: Vector Call (in-place)
# ═══════════════════════════════════════════════════════════════════════════════

function bench_vector_call!(itp, out::Vector{Float64}, queries::Vector{Float64})
    itp(out, queries)
    return out[1]  # prevent DCE
end

# ═══════════════════════════════════════════════════════════════════════════════
# Interpolant Factory
# ═══════════════════════════════════════════════════════════════════════════════
# Creates each interpolation type with the given search policy.

function make_interpolants(x, y, search)
    return (
        Linear = linear_interp(x, y; search),
        Cubic = cubic_interp(x, y; search, autocache = false),
        Quadratic = quadratic_interp(x, y; search),
        Constant = constant_interp(x, y; search),
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    Random.seed!(42)

    println("="^78)
    println("  End-to-End Search Policy Benchmark: Binary() vs LinearBinary(2)")
    println("  Measuring actual interpolation time (search + kernel), ns per query")
    println("="^78)

    grid_sizes = [500, 2000]

    for n_grid in grid_sizes
        x = collect(range(0.0, 10.0, length = n_grid))
        y = sin.(x) .+ 0.1 .* cos.(3.0 .* x)  # non-trivial function
        nq = n_grid * 2

        lo = x[2]      # avoid exact first grid point
        hi = x[end - 1]  # avoid exact last grid point

        # Build interpolants once per grid size
        itps_bin = make_interpolants(x, y, Binary())
        itps_lb = make_interpolants(x, y, LinearBinary())

        # ── Section Header ──
        println()
        println("━"^78)
        @printf("  Grid = %d points, Queries = %d\n", n_grid, nq)
        println("━"^78)

        # ────────────────────────────────────────────
        # Part 1: Vector call (itp(out, queries))
        # ────────────────────────────────────────────
        println()
        println("  ▸ Vector Call: itp(out, queries)")
        println("  " * "─"^74)
        @printf(
            "  %-11s  %-10s  %9s  %9s  %8s\n",
            "Pattern", "InterpType", "Binary", "LB{2}", "Speedup"
        )
        println("  " * "─"^74)

        for (pname, pfn) in PATTERNS
            queries = pfn(nq, lo, hi)
            out = Vector{Float64}(undef, nq)

            for (itp_name, itp_b, itp_l) in [
                    ("Linear", itps_bin.Linear, itps_lb.Linear),
                    ("Cubic", itps_bin.Cubic, itps_lb.Cubic),
                    ("Quadratic", itps_bin.Quadratic, itps_lb.Quadratic),
                    ("Constant", itps_bin.Constant, itps_lb.Constant),
                ]
                # Warmup
                bench_vector_call!(itp_b, out, queries)
                bench_vector_call!(itp_l, out, queries)

                t_b = median(@benchmark bench_vector_call!($itp_b, $out, $queries) samples = 60 evals = 10).time
                t_l = median(@benchmark bench_vector_call!($itp_l, $out, $queries) samples = 60 evals = 10).time

                speedup = t_b / t_l
                marker = speedup > 1.05 ? "▲" : speedup < 0.95 ? "▼" : "="

                @printf(
                    "  %-11s  %-10s  %7.1f ns  %7.1f ns  %5.2fx %s\n",
                    itp_name == "Linear" ? pname : "",
                    itp_name,
                    t_b / nq, t_l / nq,
                    speedup, marker
                )
            end
        end

        # ────────────────────────────────────────────
        # Part 2: Scalar loop with persistent hint
        # ────────────────────────────────────────────
        println()
        println("  ▸ Scalar Loop with Persistent Hint: for q in queries; itp(q; hint=hint); end")
        println("  " * "─"^74)
        @printf(
            "  %-11s  %-10s  %9s  %9s  %8s\n",
            "Pattern", "InterpType", "Binary", "LB{2}", "Speedup"
        )
        println("  " * "─"^74)

        for (pname, pfn) in PATTERNS
            queries = pfn(nq, lo, hi)

            # Only test Linear and Cubic for scalar loop (representative)
            for (itp_name, itp_b, itp_l) in [
                    ("Linear", itps_bin.Linear, itps_lb.Linear),
                    ("Cubic", itps_bin.Cubic, itps_lb.Cubic),
                ]
                hint_b = Ref(1)
                hint_l = Ref(1)

                # Warmup
                bench_scalar_loop_hint(itp_b, queries, hint_b)
                bench_scalar_loop_hint(itp_l, queries, hint_l)

                t_b = median(@benchmark bench_scalar_loop_hint($itp_b, $queries, $hint_b) samples = 60 evals = 10).time
                t_l = median(@benchmark bench_scalar_loop_hint($itp_l, $queries, $hint_l) samples = 60 evals = 10).time

                speedup = t_b / t_l
                marker = speedup > 1.05 ? "▲" : speedup < 0.95 ? "▼" : "="

                @printf(
                    "  %-11s  %-10s  %7.1f ns  %7.1f ns  %5.2fx %s\n",
                    itp_name == "Linear" ? pname : "",
                    itp_name,
                    t_b / nq, t_l / nq,
                    speedup, marker
                )
            end
        end

        # ────────────────────────────────────────────
        # Part 3: Scalar loop WITHOUT hint (broadcast-like)
        # ────────────────────────────────────────────
        # This is the itp(q) call — no external hint.
        # For Binary(), there's no hint at all.
        # For LinearBinary(), a fresh internal hint is created per call.
        println()
        println("  ▸ Scalar Loop (no hint): for q in queries; itp(q); end")
        println("  " * "─"^74)
        @printf(
            "  %-11s  %-10s  %9s  %9s  %8s\n",
            "Pattern", "InterpType", "Binary", "LB{2}", "Speedup"
        )
        println("  " * "─"^74)

        for (pname, pfn) in PATTERNS
            queries = pfn(nq, lo, hi)

            for (itp_name, itp_b, itp_l) in [
                    ("Linear", itps_bin.Linear, itps_lb.Linear),
                    ("Cubic", itps_bin.Cubic, itps_lb.Cubic),
                ]
                # Warmup
                bench_scalar_loop(itp_b, queries)
                bench_scalar_loop(itp_l, queries)

                t_b = median(@benchmark bench_scalar_loop($itp_b, $queries) samples = 60 evals = 10).time
                t_l = median(@benchmark bench_scalar_loop($itp_l, $queries) samples = 60 evals = 10).time

                speedup = t_b / t_l
                marker = speedup > 1.05 ? "▲" : speedup < 0.95 ? "▼" : "="

                @printf(
                    "  %-11s  %-10s  %7.1f ns  %7.1f ns  %5.2fx %s\n",
                    itp_name == "Linear" ? pname : "",
                    itp_name,
                    t_b / nq, t_l / nq,
                    speedup, marker
                )
            end
        end
    end

    # ── Legend ──
    println()
    println("="^78)
    println("  Legend:")
    println("    ▲ = LinearBinary faster   ▼ = Binary faster   = = within 5%")
    println("    Speedup = Binary_time / LB_time (>1 means LB wins)")
    println()
    println("  Expected behavior:")
    println("    Random:     LB ≈ 0.5x (2x slower) — structural penalty from hint walk")
    println("    Sorted:     LB ≈ 5-10x faster — hint almost always hits")
    println("    Clustered:  LB ≈ 3-8x faster — local queries exploit hint")
    println("    Reverse:    LB ≈ 5-10x faster — backward walk exploits hint")
    println("    DenseLocal: LB ≈ 10-50x faster — nearly all queries hit hint")
    return println("="^78)
end

main()
