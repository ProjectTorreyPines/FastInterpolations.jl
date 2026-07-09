"""
    ci_benchmark.jl

Benchmark script for GitHub Actions CI.
Outputs JSON compatible with github-action-benchmark.

Usage:
    julia --project=benchmark benchmark/ci_benchmark.jl                              # all groups (CI mode)
    julia --project=benchmark benchmark/ci_benchmark.jl 12                           # group 12 only
    julia --project=benchmark benchmark/ci_benchmark.jl 3 12                         # groups 3 and 12
    julia --project=benchmark benchmark/ci_benchmark.jl --list                       # show available groups
    julia --project=benchmark benchmark/ci_benchmark.jl --baseline baseline_data.js  # CI mode with regression verification
"""

using BenchmarkTools
using FastInterpolations
using JSON
using OrderedCollections
using Random

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Representative sizes for benchmarking
const QUERY_SIZES = [1, 100, 10_000]      # queries: single, medium, large batch
const GRID_SIZES = [10, 100, 1000]        # grids: small, medium, large

# Benchmark parameters for noise reduction in CI
# BenchmarkTools stops when EITHER limit is reached (whichever comes first)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10_000

# Fixed evals by speed category (skip tuning for faster CI)
# Higher evals = more stable ns-level measurements
const EVALS_FAST = 100      # ~10-50ns benchmarks
const EVALS_MED = 50        # ~500ns-2μs benchmarks (50 evals still < 1% timer overhead)
const EVALS_SLOW = 10       # ~30-100μs benchmarks

# ══════════════════════════════════════════════════════════════════════════════
# Hardware fingerprint
# ══════════════════════════════════════════════════════════════════════════════
#
# GitHub's shared runner fleet mixes CPU generations, so a run can land on a
# noticeably faster/slower box than the last. We record a per-run fingerprint and
# derive a machine key from it (see bench_machine.jl); the key gates the
# min-merge and the master store so we only ever compare like-with-like.

include(joinpath(@__DIR__, "bench_machine.jl"))

let hw = hardware_fingerprint()
    println("Runner hardware: $(hw["cpu_name"]) | $(hw["model"]) | $(hw["ncores"]) cores | julia $(hw["julia"]) | key=$(machine_key(hw))")
    open("hardware.json", "w") do io
        JSON.print(io, hw)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

suite = BenchmarkGroup()

# Use medium grid for oneshot and eval benchmarks
const N_GRID = 100
x = range(0.0, 10.0, N_GRID)
y = sin.(x) .+ 0.1 .* collect(x)

# Pre-build interpolants for evaluation benchmarks
clear_cubic_cache!()
const itp_linear = linear_interp(x, y)
const itp_cubic = cubic_interp(x, y)

# Also create vector-based grid version for dispatch comparison (cubic only)
const x_vec = collect(x)
const y_vec = collect(y)
clear_cubic_cache!()
const itp_cubic_vec = cubic_interp(x_vec, y_vec)

# Scalar query for scalar vs vec1 comparison
const xq_scalar = 5.0

# ══════════════════════════════════════════════════════════════════════════════
# Cubic Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

println("Setting up cubic benchmarks...")

# 1. Cubic One-Shot (construct + evaluate)
for nq in (1, 10_000)  # scalar + large batch (skip q100)
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    clear_cubic_cache!()
    cubic_interp(x, y, xi)  # prime cache
    label = lpad(nq, 5, '0')  # 00001, 00100, 10000
    b = @benchmarkable cubic_interp($x, $y, $xi)
    b.params.evals = nq >= 10_000 ? EVALS_SLOW : EVALS_MED
    suite["1_cubic_oneshot"]["q$label"] = b
end

# 2. Cubic Construction (varying grid size)
for ng in (100, 1000)  # medium + large (skip trivial g=10)
    x_grid = range(0.0, 10.0, ng)
    y_grid = sin.(x_grid) .+ 0.1 .* collect(x_grid)
    clear_cubic_cache!()
    label = lpad(ng, 4, '0')  # 0010, 0100, 1000
    b = @benchmarkable cubic_interp($x_grid, $y_grid; autocache = false)
    b.params.evals = ng >= 1000 ? EVALS_SLOW : EVALS_MED
    suite["2_cubic_construct"]["g$label"] = b
end

# 3. Cubic Evaluation (reuse interpolant)
# Use in-place API for vector queries to avoid GC noise in benchmarks
for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    label = lpad(nq, 5, '0')
    # In-place: pre-allocate output to measure pure computation
    out = Vector{Float64}(undef, nq)
    b = @benchmarkable $itp_cubic($out, $xi)
    b.params.evals = nq == 1 ? EVALS_FAST : nq == 100 ? EVALS_MED : EVALS_SLOW
    suite["3_cubic_eval"]["q$label"] = b
end

# ══════════════════════════════════════════════════════════════════════════════
# Linear Benchmarks (shown second)
# ══════════════════════════════════════════════════════════════════════════════

println("Setting up linear benchmarks...")

# 4. Linear One-Shot (construct + evaluate)
for nq in (1, 10_000)  # scalar + large batch (skip q100)
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    label = lpad(nq, 5, '0')
    b = @benchmarkable linear_interp($x, $y, $xi)
    b.params.evals = nq == 1 ? EVALS_FAST : nq == 100 ? EVALS_MED : EVALS_SLOW
    suite["4_linear_oneshot"]["q$label"] = b
end

# 5. Linear Construction (varying grid size) - nearly instant, use high evals
for ng in (100, 1000)  # medium + large (skip trivial g=10)
    x_grid = range(0.0, 10.0, ng)
    y_grid = sin.(x_grid) .+ 0.1 .* collect(x_grid)
    label = lpad(ng, 4, '0')
    b = @benchmarkable linear_interp($x_grid, $y_grid)
    b.params.evals = EVALS_FAST
    suite["5_linear_construct"]["g$label"] = b
end

# 6. Linear Evaluation (reuse interpolant)
# Use in-place API for vector queries to avoid GC noise in benchmarks
for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    label = lpad(nq, 5, '0')
    # In-place: pre-allocate output to measure pure computation
    out = Vector{Float64}(undef, nq)
    b = @benchmarkable $itp_linear($out, $xi)
    b.params.evals = nq == 1 ? EVALS_FAST : nq == 100 ? EVALS_MED : EVALS_SLOW
    suite["6_linear_eval"]["q$label"] = b
end

# ══════════════════════════════════════════════════════════════════════════════
# Cubic Scalar Dispatch Comparison
# ══════════════════════════════════════════════════════════════════════════════

println("Setting up cubic scalar dispatch benchmarks...")

# 7. Cubic: Range grid vs Vector grid scalar dispatch
let b = @benchmarkable $itp_cubic($xq_scalar)
    b.params.evals = EVALS_FAST
    suite["7_cubic_range"]["scalar_query"] = b
end

let b = @benchmarkable $itp_cubic_vec($xq_scalar)
    b.params.evals = EVALS_FAST
    suite["7_cubic_vec"]["scalar_query"] = b
end

# ══════════════════════════════════════════════════════════════════════════════
# Cubic Multi-Interpolant Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

println("Setting up cubic multi-interpolant benchmarks...")

const MULTI_SERIES = [1, 10, 100]  # single series (to test overhead), medium batch
const N_QUERY_MULTI = 100

for ns in MULTI_SERIES
    ys = [sin.(x .+ 0.1 * i) for i in 1:ns]
    slabel = lpad(ns, 3, '0')
    qlabel = lpad(N_QUERY_MULTI, 3, '0')

    # Construction benchmark
    clear_cubic_cache!()
    cubic_interp(x, Series(ys))  # prime cache
    let b = @benchmarkable cubic_interp($x, Series($ys))
        b.params.evals = ns >= 50 ? EVALS_SLOW : EVALS_MED
        suite["8_cubic_multi"]["construct_s$(slabel)_q$(qlabel)"] = b
    end

    # Vector evaluation benchmark (batch query) - in-place for zero-alloc measurement
    clear_cubic_cache!()
    mitp = cubic_interp(x, Series(ys))
    xq_multi = collect(range(0.1, 9.9, N_QUERY_MULTI))
    # Pre-allocate outputs: one vector per series, each of length N_QUERY_MULTI
    outputs_multi = [Vector{Float64}(undef, N_QUERY_MULTI) for _ in 1:ns]
    let b = @benchmarkable $mitp($outputs_multi, $xq_multi)
        b.params.evals = ns >= 50 ? EVALS_SLOW : EVALS_MED
        suite["8_cubic_multi"]["eval_s$(slabel)_q$(qlabel)"] = b
    end

    # Scalar loop evaluation benchmark (tests SIMD scalar kernel in realistic usage)
    # Pattern: for xq in queries; mitp(out, xq); end - like ODE solver callbacks
    if ns > 1  # skip s=1 (equivalent to normal cubic eval)
        out_scalar = zeros(ns)
        let b = @benchmarkable begin
                for xq in $xq_multi
                    $mitp($out_scalar, xq)
                end
            end
            b.params.evals = ns >= 50 ? EVALS_SLOW : EVALS_MED
            suite["8_cubic_multi"]["eval_s$(slabel)_q$(qlabel)_scalar_loop"] = b
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# ND Interpolation Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

println("Setting up ND benchmarks...")

# --- 2D Setup (50×50 = 2,500 grid points) ---
const x2d = range(0.0, 10.0, 50)
const y2d = range(0.0, 6.0, 50)
const data2d = [sin(xi) * cos(yj) for xi in x2d, yj in y2d]

const itp_linear_2d = linear_interp((x2d, y2d), data2d)
clear_cubic_cache!()
const itp_cubic_2d = cubic_interp((x2d, y2d), data2d)

# --- 3D Setup (20×20×20 = 8,000 grid points) ---
const x3d = range(0.0, 10.0, 20)
const y3d = range(0.0, 6.0, 20)
const z3d = range(0.0, 4.0, 20)
const data3d = [sin(xi) * cos(yj) + zk for xi in x3d, yj in y3d, zk in z3d]

const itp_linear_3d = linear_interp((x3d, y3d, z3d), data3d)
clear_cubic_cache!()
const itp_cubic_3d = cubic_interp((x3d, y3d, z3d), data3d)

# --- ND Query Points ---
const N_ND_QUERY = 100
const xqs_2d = collect(range(0.1, 9.9, N_ND_QUERY))
const yqs_2d = collect(range(0.1, 5.9, N_ND_QUERY))
const xqs_3d = collect(range(0.1, 9.9, N_ND_QUERY))
const yqs_3d = collect(range(0.1, 5.9, N_ND_QUERY))
const zqs_3d = collect(range(0.1, 3.9, N_ND_QUERY))
const pt_2d = (5.0, 3.0)
const pt_3d = (5.0, 3.0, 2.0)
const out_nd = Vector{Float64}(undef, N_ND_QUERY)

# 9. ND One-Shot (construct + evaluate, separate code path)
let b = @benchmarkable linear_interp(($x2d, $y2d), $data2d, ($xqs_2d, $yqs_2d))
    b.params.evals = EVALS_MED
    suite["9_nd_oneshot"]["bilinear_2d"] = b
end

let b = @benchmarkable linear_interp(($x3d, $y3d, $z3d), $data3d, ($xqs_3d, $yqs_3d, $zqs_3d))
    b.params.evals = EVALS_SLOW
    suite["9_nd_oneshot"]["trilinear_3d"] = b
end

clear_cubic_cache!()
cubic_interp((x2d, y2d), data2d, (xqs_2d, yqs_2d))  # prime cache
let b = @benchmarkable cubic_interp(($x2d, $y2d), $data2d, ($xqs_2d, $yqs_2d))
    b.params.evals = EVALS_SLOW
    suite["9_nd_oneshot"]["bicubic_2d"] = b
end

clear_cubic_cache!()
cubic_interp((x3d, y3d, z3d), data3d, (xqs_3d, yqs_3d, zqs_3d))  # prime cache
let b = @benchmarkable cubic_interp(($x3d, $y3d, $z3d), $data3d, ($xqs_3d, $yqs_3d, $zqs_3d))
    b.params.evals = EVALS_SLOW
    suite["9_nd_oneshot"]["tricubic_3d"] = b
end

# 10. ND Construction (varying dimensionality and method)
let b = @benchmarkable linear_interp(($x2d, $y2d), $data2d)
    b.params.evals = EVALS_MED
    suite["10_nd_construct"]["bilinear_2d"] = b
end

let b = @benchmarkable linear_interp(($x3d, $y3d, $z3d), $data3d)
    b.params.evals = EVALS_MED
    suite["10_nd_construct"]["trilinear_3d"] = b
end

# Cubic ND construction: clear cache in setup + evals=1 to measure full construction
# (ND API lacks autocache=false, so we emulate it with per-sample cache clearing)
let b = @benchmarkable cubic_interp(($x2d, $y2d), $data2d) setup = (clear_cubic_cache!())
    b.params.evals = 1
    suite["10_nd_construct"]["bicubic_2d"] = b
end

let b = @benchmarkable cubic_interp(($x3d, $y3d, $z3d), $data3d) setup = (clear_cubic_cache!())
    b.params.evals = 1
    suite["10_nd_construct"]["tricubic_3d"] = b
end

# 11. ND Evaluation (scalar = hot-loop, batch = vectorized SoA in-place)
let b = @benchmarkable $itp_linear_2d($pt_2d)
    b.params.evals = EVALS_FAST
    suite["11_nd_eval"]["bilinear_2d_scalar"] = b
end

let b = @benchmarkable $itp_linear_3d($pt_3d)
    b.params.evals = EVALS_FAST
    suite["11_nd_eval"]["trilinear_3d_scalar"] = b
end

let b = @benchmarkable $itp_cubic_2d($pt_2d)
    b.params.evals = EVALS_FAST
    suite["11_nd_eval"]["bicubic_2d_scalar"] = b
end

let b = @benchmarkable $itp_cubic_3d($pt_3d)
    b.params.evals = EVALS_FAST
    suite["11_nd_eval"]["tricubic_3d_scalar"] = b
end

let b = @benchmarkable $itp_cubic_2d($out_nd, ($xqs_2d, $yqs_2d))
    b.params.evals = EVALS_SLOW
    suite["11_nd_eval"]["bicubic_2d_batch"] = b
end

let b = @benchmarkable $itp_cubic_3d($out_nd, ($xqs_3d, $yqs_3d, $zqs_3d))
    b.params.evals = EVALS_SLOW
    suite["11_nd_eval"]["tricubic_3d_batch"] = b
end

# ══════════════════════════════════════════════════════════════════════════════
# Cubic Grid Type × Query Pattern Benchmarks (Range vs Vector × Sorted vs Random)
# ══════════════════════════════════════════════════════════════════════════════

println("Setting up cubic grid×query pattern benchmarks...")

# Fixed seed: ensures identical random query sequence across CI runs
const BENCH_RNG_SEED = 12345

# 12. Cubic Eval: Grid type × Query pattern (reuse interpolant)
# Range grid → LinearBinary (O(1) arithmetic), Vector grid → binary search
const N_QUERY_GQ = 1000
const xq_sorted_gq = collect(range(0.1, 9.9, N_QUERY_GQ))
const xq_random_gq = shuffle(MersenneTwister(BENCH_RNG_SEED), copy(xq_sorted_gq))
const out_gq = Vector{Float64}(undef, N_QUERY_GQ)

for (glabel, itp) in [("range", itp_cubic), ("vec", itp_cubic_vec)]
    for (qlbl, xq) in [("sorted", xq_sorted_gq), ("random", xq_random_gq)]
        let b = @benchmarkable $itp($out_gq, $xq)
            b.params.evals = EVALS_MED
            suite["12_cubic_eval_gridquery"]["$(glabel)_$(qlbl)"] = b
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# ND One-Shot: Vector grid × Query pattern (per-axis adaptive search)
# ══════════════════════════════════════════════════════════════════════════════
#
# Exercises the per-axis adaptive search dispatch inside ND oneshot batch
# (sort axis → LinearBinary + persistent hint, rand axis → Binary). Vector
# grids force the runtime search path; Range grids would use O(1) arithmetic
# regardless of query pattern and hide the dispatch behavior. Includes a
# `sort_rand` mixed case so a regression that disables per-axis selection
# (collapsing both axes to a single policy) shows up here.

println("Setting up ND one-shot grid×query pattern benchmarks...")

const N_ND_GQ = 51
const NQ_ND_GQ = 1000
const x2d_vec_gq = collect(range(0.0, 10.0, N_ND_GQ))
const y2d_vec_gq = collect(range(0.0, 6.0, N_ND_GQ))
const data2d_gq = [sin(xi) * cos(yj) for xi in x2d_vec_gq, yj in y2d_vec_gq]
const qx_sort_gq = collect(range(0.1, 9.9, NQ_ND_GQ))
const qy_sort_gq = collect(range(0.1, 5.9, NQ_ND_GQ))
const qx_rand_gq = shuffle(MersenneTwister(BENCH_RNG_SEED), copy(qx_sort_gq))
const qy_rand_gq = shuffle(MersenneTwister(BENCH_RNG_SEED + 1), copy(qy_sort_gq))

const _ND_GQ_PATTERNS = (
    ("sort_sort", qx_sort_gq, qy_sort_gq),
    ("rand_rand", qx_rand_gq, qy_rand_gq),
    ("sort_rand", qx_sort_gq, qy_rand_gq),
)

# 13. ND One-Shot: Vector × Vector grid × query pattern
for (lbl, qx, qy) in _ND_GQ_PATTERNS
    let b = @benchmarkable linear_interp(($x2d_vec_gq, $y2d_vec_gq), $data2d_gq, ($qx, $qy))
        b.params.evals = EVALS_MED
        suite["13_nd_oneshot_gridquery"]["bilinear_2d_$(lbl)"] = b
    end
end

for (lbl, qx, qy) in _ND_GQ_PATTERNS
    clear_cubic_cache!()
    cubic_interp((x2d_vec_gq, y2d_vec_gq), data2d_gq, (qx, qy))  # prime cache
    let b = @benchmarkable cubic_interp(($x2d_vec_gq, $y2d_vec_gq), $data2d_gq, ($qx, $qy))
        b.params.evals = EVALS_SLOW
        suite["13_nd_oneshot_gridquery"]["bicubic_2d_$(lbl)"] = b
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Linear / Constant Series One-Shot Batch (in-place)
# ══════════════════════════════════════════════════════════════════════════════
#
# Detects regressions in the Series oneshot vector-batch code path
# (`{linear,constant}_interp!(outs, x, ::Series, xqs)`). Vector grid + many
# random queries is the configuration where the loop ordering
# (Q outer × K inner vs K outer × Q inner) materially changes per-query cache
# behavior on the `outputs[k][j]` write pattern. The in-place API is used so
# the timing reflects pure computation — no per-call output allocation, no GC
# pressure. (`8_cubic_multi` already covers Cubic series in-place.)

println("Setting up Linear/Constant Series one-shot batch benchmarks...")

const N_SER = 100
const K_SER = 8
const NQ_SER = 1000
const x_ser = collect(range(0.0, 2π, N_SER + 1))
const Y_ser = hcat([sin.(j .* x_ser) for j in 1:K_SER]...)
const Ys_ser = Series(Y_ser)
const q_ser_rand = rand(MersenneTwister(BENCH_RNG_SEED), NQ_SER) .* 2π
const outs_ser = [Vector{Float64}(undef, NQ_SER) for _ in 1:K_SER]

# 14. Linear / Constant Series one-shot batch (in-place)
let b = @benchmarkable linear_interp!($outs_ser, $x_ser, $Ys_ser, $q_ser_rand)
    b.params.evals = EVALS_SLOW
    suite["14_series_oneshot_batch"]["linear_inplace_vec_k$(K_SER)_q$(NQ_SER)_rand"] = b
end

let b = @benchmarkable constant_interp!($outs_ser, $x_ser, $Ys_ser, $q_ser_rand)
    b.params.evals = EVALS_SLOW
    suite["14_series_oneshot_batch"]["constant_inplace_vec_k$(K_SER)_q$(NQ_SER)_rand"] = b
end

# ══════════════════════════════════════════════════════════════════════════════
# GriddedQuery Benchmarks (shaped in-place tensor-product query)
# ══════════════════════════════════════════════════════════════════════════════
#
# Catches regressions in the GriddedQuery fast paths before they become visible
# in downstream resize-style workloads. The shaped in-place API avoids timing
# output allocation; persistent and one-shot entries are both included because
# they route through different public surfaces.

println("Setting up GriddedQuery benchmarks...")

const gq2d_bench = GriddedQuery((range(0.05, 9.95, 40), range(0.05, 5.95, 32)))
const out_gq2d = Matrix{Float64}(undef, size(gq2d_bench))

const gq3d_bench = GriddedQuery((range(0.1, 9.9, 12), range(0.1, 5.9, 10), range(0.1, 3.9, 8)))
const out_gq3d = Array{Float64, 3}(undef, size(gq3d_bench))
const GQ_CUBIC_METHOD = CubicInterp()

# 15. GriddedQuery: persistent and one-shot shaped in-place paths
let b = @benchmarkable $itp_linear_2d($out_gq2d, $gq2d_bench)
    b.params.evals = EVALS_MED
    suite["15_gridded_query"]["linear_persistent_2d_40x32"] = b
end

let b = @benchmarkable linear_interp!($out_gq2d, ($x2d, $y2d), $data2d, $gq2d_bench)
    b.params.evals = EVALS_MED
    suite["15_gridded_query"]["linear_oneshot_2d_40x32"] = b
end

let b = @benchmarkable $itp_linear_3d($out_gq3d, $gq3d_bench)
    b.params.evals = EVALS_MED
    suite["15_gridded_query"]["linear_persistent_3d_12x10x8"] = b
end

let b = @benchmarkable linear_interp!($out_gq3d, ($x3d, $y3d, $z3d), $data3d, $gq3d_bench)
    b.params.evals = EVALS_MED
    suite["15_gridded_query"]["linear_oneshot_3d_12x10x8"] = b
end

# Cubic's named ND batch API writes a flat vector by contract; the shaped
# GriddedQuery one-shot fast path is exposed through unified `interp!`.
interp!(out_gq2d, (x2d, y2d), data2d, gq2d_bench; method = GQ_CUBIC_METHOD)
let b = @benchmarkable $itp_cubic_2d($out_gq2d, $gq2d_bench)
    b.params.evals = EVALS_SLOW
    suite["15_gridded_query"]["cubic_persistent_2d_40x32"] = b
end

let b = @benchmarkable interp!($out_gq2d, ($x2d, $y2d), $data2d, $gq2d_bench; method = $GQ_CUBIC_METHOD)
    b.params.evals = EVALS_SLOW
    suite["15_gridded_query"]["cubic_oneshot_2d_40x32"] = b
end

# ══════════════════════════════════════════════════════════════════════════════
# CLI Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════

# Value-carrying flags consumed before positional (group-number) parsing.
# --baseline <path>    gh-pages baseline data.js for regression verification
# --prev-best <path>   JSON array [{name,value}] of prior best times for this
#                      commit (from the existing PR comment); enables cross-run
#                      min-merge so re-running a flagged commit only lowers times
# --only <names>       comma-separated benchmark full-names ("group/bench") to
#                      run in isolation (flagged-only subset re-run)
function _extract_flag_value(args, flag)
    idx = findfirst(==(flag), args)
    isnothing(idx) && return ""
    idx < length(args) || error("$flag requires an argument")
    return args[idx + 1]
end

# --master-sha <sha>   store mode: min-merge/regression-check the master commit's
#                      point vs the *previous* master and emit master_benches.json
const _VALUE_FLAGS = ("--baseline", "--prev-best", "--only", "--master-sha")

const BASELINE_PATH = _extract_flag_value(ARGS, "--baseline")
const PREVBEST_PATH = _extract_flag_value(ARGS, "--prev-best")
const MASTER_SHA = _extract_flag_value(ARGS, "--master-sha")
const ONLY_NAMES = let raw = _extract_flag_value(ARGS, "--only")
    isempty(raw) ? Set{String}() : Set(String.(filter(!isempty, split(raw, ','))))
end

# Strip value flags (and their arguments) from ARGS for group-number parsing
const _POSITIONAL_ARGS = let filtered = String[]
    skip_next = false
    for arg in ARGS
        if skip_next
            skip_next = false
            continue
        end
        if arg in _VALUE_FLAGS
            skip_next = true
            continue
        end
        push!(filtered, arg)
    end
    filtered
end

# --list: show available groups and exit
if "--list" in _POSITIONAL_ARGS
    println("Available benchmark groups:")
    for key in sort(collect(keys(suite)))
        n = length(suite[key])
        println("  $key  ($n benchmarks)")
    end
    exit(0)
end

# Parse group numbers from positional args to filter suite
const FILTER_GROUPS = let nums = Int[]
    for arg in _POSITIONAL_ARGS
        arg == "--list" && continue
        n = tryparse(Int, arg)
        isnothing(n) && error("Unknown argument: $arg (expected group number, --list, or --baseline <path>)")
        push!(nums, n)
    end
    nums
end
const IS_FILTERED = !isempty(FILTER_GROUPS)

if IS_FILTERED
    for key in collect(keys(suite))
        group_num = tryparse(Int, split(key, '_')[1])
        if isnothing(group_num) || group_num ∉ FILTER_GROUPS
            delete!(suite, key)
        end
    end
    println("\nFiltered to groups: $(join(FILTER_GROUPS, ", ")) → $(length(suite)) group(s)")
end

# --only: keep only the named benchmarks ("group/bench"). Used for flagged-only
# subset re-runs. Non-run benchmarks are filled from --prev-best at report time,
# so the emitted report stays complete. If no name matches (e.g. stale names),
# the suite empties and the run degrades to producing a report purely from
# prev_best — never a crash.
const IS_ONLY = !isempty(ONLY_NAMES)
if IS_ONLY
    for gkey in collect(keys(suite))
        for bkey in collect(keys(suite[gkey]))
            "$gkey/$bkey" ∉ ONLY_NAMES && delete!(suite[gkey], bkey)
        end
        isempty(suite[gkey]) && delete!(suite, gkey)
    end
    println("\nRestricted to $(length(ONLY_NAMES)) named benchmark(s) → $(sum(length, values(suite); init = 0)) kept")
end

# ══════════════════════════════════════════════════════════════════════════════
# Run and Save
# ══════════════════════════════════════════════════════════════════════════════

# Skip tuning - we set evals manually for consistent CI results
# Run each group separately with GC.gc() between groups (not per-sample)
# to avoid GC overhead consuming the time budget (~100ms/sample → only ~30 samples)
println("\nRunning benchmarks (evals preset, no tuning)...")
results = BenchmarkGroup()
for group_key in sort(collect(keys(suite)))
    GC.gc()
    println("  Running [$group_key]...")
    results[group_key] = run(suite[group_key], verbose = true)
end

# ══════════════════════════════════════════════════════════════════════════════
# Regression Verification (when --baseline is provided)
# ══════════════════════════════════════════════════════════════════════════════

const _HAS_BASELINE = !isempty(BASELINE_PATH) && isfile(BASELINE_PATH) && filesize(BASELINE_PATH) > 0
const _HAS_PREVBEST = !isempty(PREVBEST_PATH) && isfile(PREVBEST_PATH) && filesize(PREVBEST_PATH) > 0

# Enter whenever there is something to do: master store (always — bootstraps the
# first point), a baseline to compare against, OR a prior-best to min-merge. The
# last case matters when the gh-pages baseline fetch failed transiently on a PR
# re-run: we still apply the floor and emit the report so the BENCH_DATA blob
# keeps the cross-run minimum instead of resetting to this run's raw values.
if !IS_FILTERED && (!isempty(MASTER_SHA) || _HAS_BASELINE || _HAS_PREVBEST)
    include(joinpath(@__DIR__, "regression_check.jl"))

    println("\n" * "="^70)
    println("REGRESSION VERIFICATION")
    println("="^70)

    if !isempty(MASTER_SHA)
        # ── Master store mode ──────────────────────────────────────────────
        # prev_best = same-(commit,machine) floor (re-run only lowers the point);
        # latest/window_avg = the *previous* master on THIS machine, for detection.
        prev_best, latest, window_avg = load_master_baseline(BASELINE_PATH, MASTER_SHA, machine_key())
        if !isempty(prev_best)
            println("Loaded $(length(prev_best)) same-commit prior value(s) for SHA $(MASTER_SHA[1:min(8, lastindex(MASTER_SHA))]) (floor)")
        end

        effective = compute_effective(results, prev_best)
        flagged = detect_regressions(effective, latest, window_avg)
        confirmed = FlaggedBench[]
        if !isempty(flagged)
            println("Flagged $(length(flagged)) benchmark(s) vs previous master; re-running $(RERUN_N)×...")
            rerun_and_merge!(suite, results, effective, flagged, RERUN_N, prev_best, latest, window_avg)
            confirmed = detect_regressions(effective, latest, window_avg)
            println("$(length(confirmed)) still above threshold after re-run (stored as measured; the graph shows the trend)")
        else
            println("No regressions vs previous master")
        end

        write_master_benches("master_benches.json", effective, results)
        println("Wrote master_benches.json ($(length(effective)) benches)")

        # Also emit the rich report so the push workflow can post/refresh a commit
        # comment (same table as a PR): the master min-merge floor lives in
        # gh-pages, so re-running a commit only lowers these numbers.
        write_regression_report(
            "regression_report.json", effective, latest, window_avg, flagged, confirmed,
            machine_key(), latest_master_machine(BASELINE_PATH),
        )
        println("Wrote regression_report.json (for the commit comment)")
    else
        # ── PR mode ────────────────────────────────────────────────────────
        # Prior best times for this commit (from the existing PR comment). Empty
        # on the first run / when the stored SHA didn't match.
        prev_best = Dict{String, Float64}()
        if _HAS_PREVBEST
            for e in JSON.parsefile(PREVBEST_PATH)
                prev_best[String(e["name"])] = Float64(e["value"])
            end
            println("Loaded $(length(prev_best)) prior-best value(s) for cross-run min-merge")
        end

        # Min-merge is applied UNCONDITIONALLY (not gated on the baseline): a
        # re-run only ever lowers values, and the report we emit — hence the
        # BENCH_DATA blob — preserves the cross-run minimum even when the
        # baseline is missing. Only the regression *comparison* needs a baseline.
        effective = compute_effective(results, prev_best)

        latest = Dict{String, Float64}()
        window_avg = Dict{String, Float64}()
        flagged = FlaggedBench[]
        confirmed = FlaggedBench[]

        if _HAS_BASELINE
            latest, window_avg = load_baseline(BASELINE_PATH, machine_key())
            flagged = detect_regressions(effective, latest, window_avg)

            if !isempty(flagged)
                println("Flagged $(length(flagged)) benchmark(s) for re-verification:")
                for fb in flagged
                    tier_str = fb.tier == :both ? "immediate+gradual" : string(fb.tier)
                    r_imm = isnothing(fb.ratio_immediate) ? "-" : string(round(fb.ratio_immediate, digits = 3))
                    r_grad = isnothing(fb.ratio_gradual) ? "-" : string(round(fb.ratio_gradual, digits = 3))
                    println("  [$tier_str] $(fb.full_name)  imm=$(r_imm) grad=$(r_grad)")
                end

                println("\nRe-running flagged benchmarks $(RERUN_N) time(s)...")
                rerun_and_merge!(suite, results, effective, flagged, RERUN_N, prev_best, latest, window_avg)

                # Re-evaluate after merge
                confirmed = detect_regressions(effective, latest, window_avg)

                if !isempty(confirmed)
                    println("\nConfirmed $(length(confirmed)) regression(s) after re-verification:")
                    for fb in confirmed
                        r_imm = isnothing(fb.ratio_immediate) ? "-" : string(round(fb.ratio_immediate, digits = 3))
                        r_grad = isnothing(fb.ratio_gradual) ? "-" : string(round(fb.ratio_gradual, digits = 3))
                        println("  $(fb.full_name)  imm=$(r_imm) grad=$(r_grad)")
                    end
                else
                    println("\nAll flagged benchmarks verified as noise after re-run")
                end
            else
                println("No regressions detected")
            end
        else
            println("No baseline available — applied prev-best min-merge, skipped regression comparison")
        end

        # Record which CPU this run measured on vs the CPU of master's most-recent
        # commit (the natural baseline) so the comment can warn on a cross-CPU
        # comparison — the more dangerous mismatch than a mere runner change.
        cur_machine = machine_key()
        base_machine = _HAS_BASELINE ? latest_master_machine(BASELINE_PATH) : ""
        write_regression_report(
            "regression_report.json", effective, latest, window_avg, flagged, confirmed,
            cur_machine, base_machine,
        )
        println("Wrote regression_report.json (runner=$cur_machine, master-baseline=$(isempty(base_machine) ? "none" : base_machine))")
    end   # master-store vs PR mode
end

# ══════════════════════════════════════════════════════════════════════════════
# Save Results (CI mode only)
# ══════════════════════════════════════════════════════════════════════════════

function sort_keys_recursive(obj)
    if obj isa AbstractDict
        sorted = OrderedDict{String, Any}()
        for k in sort(collect(keys(obj)); by = string)
            sorted[string(k)] = sort_keys_recursive(obj[k])
        end
        return sorted
    elseif obj isa AbstractVector
        return [sort_keys_recursive(item) for item in obj]
    else
        return obj
    end
end

if !IS_FILTERED
    println("\nSaving results to output.json...")
    BenchmarkTools.save("output.json", minimum(results))

    println("Sorting JSON keys for dashboard display...")
    json_data = JSON.parsefile("output.json")
    sorted_data = sort_keys_recursive(json_data)
    open("output.json", "w") do io
        JSON.print(io, sorted_data)
    end
    println("Saved $(length(collect(BenchmarkTools.leaves(minimum(results))))) benchmarks (sorted)")
end

# ══════════════════════════════════════════════════════════════════════════════
# Print Summary
# ══════════════════════════════════════════════════════════════════════════════

function format_time(ns::Float64)
    if ns < 1000
        return "$(round(ns, digits = 1)) ns"
    elseif ns < 1_000_000
        return "$(round(ns / 1000, digits = 2)) μs"
    else
        return "$(round(ns / 1_000_000, digits = 2)) ms"
    end
end

println("\n" * "="^70)
if IS_FILTERED
    println("BENCHMARK RESULTS (groups: $(join(FILTER_GROUPS, ", ")))")
else
    println("BENCHMARK SUMMARY")
end
println("="^70)

for group_name in sort(collect(keys(results)))
    group = results[group_name]
    println("\n[$group_name]")
    for bench_name in sort(collect(keys(group)))
        trial = group[bench_name]

        t_min = minimum(trial).time
        t_med = median(trial).time
        t_mean = mean(trial).time
        n_samples = length(trial.times)
        n_evals = trial.params.evals

        total_time = sum(trial.times)
        total_gc = sum(trial.gctimes)
        gc_pct = total_time > 0 ? round(100 * total_gc / total_time, digits = 1) : 0.0

        if IS_FILTERED
            # Detailed output for local runs
            t_std = std(trial).time
            cv = t_med > 0 ? round(100 * t_std / t_med, digits = 1) : 0.0
            println("  $(rpad(bench_name, 30)) min: $(rpad(format_time(t_min), 10)) med: $(rpad(format_time(t_med), 10)) mean: $(rpad(format_time(t_mean), 10)) std: $(rpad(format_time(t_std), 10)) cv: $(lpad(string(cv), 4))%")
            println("  $(rpad("", 30)) gc: $(lpad(string(gc_pct), 4))%  mem: $(trial.memory) B  allocs: $(trial.allocs)  samples: $(n_samples)  evals: $(n_evals)")
        else
            # Compact output for CI
            println("  $(rpad(bench_name, 20)) min: $(rpad(format_time(t_min), 9)) | med: $(rpad(format_time(t_med), 9)) | gc: $(lpad(string(gc_pct), 4))% | mem: $(trial.memory) B | samples: $(n_samples) | evals: $(n_evals)")
        end
    end
end

println("\nBenchmarking Done!")
