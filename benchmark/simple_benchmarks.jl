"""
    simple_benchmarks.jl

Simple, modular benchmark functions for generating README figures.
Each function returns a NamedTuple with timing results in nanoseconds.

# Usage
```julia
include("simple_benchmarks.jl")

# Construction benchmark
constr = benchmark_construction(n_grid=100, use_range=true)
# => (FastInterp=1234.0, Interpolations=5678.0, DataInterp=9012.0)

# Evaluation benchmark (reusing interpolant, n_query points)
eval_result = benchmark_evaluation(n_grid=100, n_query=1000, use_range=true)
# => (FastInterp=1234.0, Interpolations=5678.0, DataInterp=9012.0)

# One-shot benchmark (construct + evaluate n_query points)
oneshot = benchmark_oneshot(n_grid=100, n_query=1000, use_range=true)
# => (FastInterp=(time_ns=1234.0, alloc_bytes=0), ...)

# All-in-one for README
result = benchmark_for_readme(n_grid=100, n_query=1000, use_range=true)
```
"""

using BenchmarkTools
using FastInterpolations
import Interpolations
import DataInterpolations
using DataInterpolations: ExtrapolationType
using DataFrames

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const DEFAULT_BENCH_SECONDS = 0.5

function _setup_benchmark()
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = DEFAULT_BENCH_SECONDS
end

# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

"""Generate test data for benchmarks."""
function _generate_data(n_grid::Int, n_query::Int; use_range::Bool=false)
    if use_range
        x = range(0.0, 10.0, n_grid)
    else
        x = collect(range(0.0, 10.0, n_grid))
    end
    y = sin.(x) .+ 0.1 .* x

    # Query points within the grid range
    xi_vec = n_query == 1 ? [5.0] : collect(range(0.1, 9.9, n_query))
    xi_scalar = 5.0

    return (; x, y, xi_vec, xi_scalar)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Construction Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

"""
    benchmark_construction(; n_grid=100, use_range=false, interp_type=:cubic)

Benchmark interpolant construction time.

# Arguments
- `n_grid::Int=100`: Number of grid points
- `use_range::Bool=false`: Use StepRangeLen instead of Vector for x-grid
- `interp_type::Symbol=:cubic`: `:cubic` or `:linear`

# Returns
NamedTuple with median times in nanoseconds
"""
function benchmark_construction(; n_grid::Int=100, use_range::Bool=false, interp_type::Symbol=:cubic)
    _setup_benchmark()
    data = _generate_data(n_grid, 100; use_range)
    x, y = data.x, data.y

    results = Dict{Symbol, Float64}()

    if interp_type == :cubic
        clear_cubic_cache!()
        b = @benchmark cubic_interp($x, $y; autocache=false)
        results[:FastInterp] = median(b.times)

        if use_range
            b = @benchmark Interpolations.cubic_spline_interpolation($x, $y)
            results[:Interpolations] = median(b.times)
        else
            results[:Interpolations] = NaN
        end

        b = @benchmark DataInterpolations.CubicSpline($y, $x)
        results[:DataInterp] = median(b.times)
    else
        b = @benchmark linear_interp($x, $y)
        results[:FastInterp] = median(b.times)

        if use_range
            b = @benchmark Interpolations.linear_interpolation($x, $y)
        else
            b = @benchmark Interpolations.LinearInterpolation(($x,), $y)
        end
        results[:Interpolations] = median(b.times)

        b = @benchmark DataInterpolations.LinearInterpolation($y, $x)
        results[:DataInterp] = median(b.times)
    end

    return (
        FastInterp = results[:FastInterp],
        Interpolations = results[:Interpolations],
        DataInterp = results[:DataInterp]
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Benchmark (Reusing Interpolant)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    benchmark_evaluation(; n_grid=100, n_query=1000, use_range=false, interp_type=:cubic)

Benchmark interpolation evaluation time (reusing pre-built interpolant).

# Arguments
- `n_grid::Int=100`: Number of grid points
- `n_query::Int=1000`: Number of query points
- `use_range::Bool=false`: Use StepRangeLen instead of Vector for x-grid
- `interp_type::Symbol=:cubic`: `:cubic` or `:linear`

# Returns
NamedTuple with median times in nanoseconds (for n_query points)
"""
function benchmark_evaluation(; n_grid::Int=100, n_query::Int=1000, use_range::Bool=false,
                               interp_type::Symbol=:cubic)
    _setup_benchmark()
    data = _generate_data(n_grid, n_query; use_range)
    x, y, xi_vec = data.x, data.y, data.xi_vec

    results = Dict{Symbol, Float64}()

    if interp_type == :cubic
        clear_cubic_cache!()
        itp_fast = cubic_interp(x, y; autocache=false)
        itp_data = DataInterpolations.CubicSpline(y, x)

        b = @benchmark $itp_fast($xi_vec)
        results[:FastInterp] = median(b.times)

        if use_range
            itp_interp = Interpolations.cubic_spline_interpolation(x, y)
            b = @benchmark $itp_interp($xi_vec)
            results[:Interpolations] = median(b.times)
        else
            results[:Interpolations] = NaN
        end

        b = @benchmark $itp_data($xi_vec)
        results[:DataInterp] = median(b.times)
    else
        itp_fast = linear_interp(x, y)
        itp_data = DataInterpolations.LinearInterpolation(y, x)

        if use_range
            itp_interp = Interpolations.linear_interpolation(x, y)
        else
            itp_interp = Interpolations.LinearInterpolation((x,), y)
        end

        b = @benchmark $itp_fast($xi_vec)
        results[:FastInterp] = median(b.times)

        b = @benchmark $itp_interp($xi_vec)
        results[:Interpolations] = median(b.times)

        b = @benchmark $itp_data($xi_vec)
        results[:DataInterp] = median(b.times)
    end

    return (
        FastInterp = results[:FastInterp],
        Interpolations = results[:Interpolations],
        DataInterp = results[:DataInterp]
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# One-Shot Benchmark (Construction + Evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    benchmark_oneshot(; n_grid=100, n_query=1000, use_range=false, interp_type=:cubic)

Benchmark one-shot interpolation (construct + evaluate in single call).
This is the typical use case for users who interpolate once per dataset.

# Arguments
- `n_grid::Int=100`: Number of grid points
- `n_query::Int=1000`: Number of query points
- `use_range::Bool=false`: Use StepRangeLen instead of Vector for x-grid
- `interp_type::Symbol=:cubic`: `:cubic` or `:linear`

# Returns
NamedTuple with (time_ns, alloc_bytes) for each package
"""
function benchmark_oneshot(; n_grid::Int=100, n_query::Int=1000, use_range::Bool=false, interp_type::Symbol=:cubic)
    _setup_benchmark()
    data = _generate_data(n_grid, n_query; use_range)
    x, y, xi_vec = data.x, data.y, data.xi_vec

    results = Dict{Symbol, NamedTuple{(:time_ns, :alloc_bytes), Tuple{Float64, Int}}}()

    if interp_type == :cubic
        clear_cubic_cache!()
        cubic_interp(x, y, xi_vec)  # warmup to populate cache
        b = @benchmark cubic_interp($x, $y, $xi_vec)
        results[:FastInterp] = (time_ns=median(b.times), alloc_bytes=b.allocs > 0 ? Int(b.memory) : 0)

        if use_range
            b = @benchmark Interpolations.cubic_spline_interpolation($x, $y)($xi_vec)
            results[:Interpolations] = (time_ns=median(b.times), alloc_bytes=Int(b.memory))
        else
            results[:Interpolations] = (time_ns=NaN, alloc_bytes=0)
        end

        b = @benchmark DataInterpolations.CubicSpline($y, $x)($xi_vec)
        results[:DataInterp] = (time_ns=median(b.times), alloc_bytes=Int(b.memory))
    else
        b = @benchmark linear_interp($x, $y, $xi_vec)
        results[:FastInterp] = (time_ns=median(b.times), alloc_bytes=b.allocs > 0 ? Int(b.memory) : 0)

        if use_range
            b = @benchmark Interpolations.linear_interpolation($x, $y)($xi_vec)
        else
            b = @benchmark Interpolations.LinearInterpolation(($x,), $y)($xi_vec)
        end
        results[:Interpolations] = (time_ns=median(b.times), alloc_bytes=Int(b.memory))

        b = @benchmark DataInterpolations.LinearInterpolation($y, $x)($xi_vec)
        results[:DataInterp] = (time_ns=median(b.times), alloc_bytes=Int(b.memory))
    end

    return (
        FastInterp = results[:FastInterp],
        Interpolations = results[:Interpolations],
        DataInterp = results[:DataInterp]
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Allocation Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

"""
    benchmark_allocation(; n_grid=100, use_range=false)

Measure memory allocations for different evaluation methods.

# Returns
NamedTuple with allocation in bytes for each method
"""
function benchmark_allocation(; n_grid::Int=100, use_range::Bool=false)
    data = _generate_data(n_grid, 100; use_range)
    x, y, xi_vec, xi_scalar = data.x, data.y, data.xi_vec, data.xi_scalar

    # FastInterpolations - Callable
    clear_cubic_cache!()
    itp = cubic_interp(x, y; autocache=false)
    itp(xi_scalar)  # warmup
    alloc_callable_scalar = @allocated itp(xi_scalar)
    itp(xi_vec)  # warmup
    alloc_callable_vector = @allocated itp(xi_vec)

    # FastInterpolations - Autocache hit
    clear_cubic_cache!()
    cubic_interp(x, y, xi_scalar)  # warmup (cache miss)
    cubic_interp(x, y, xi_scalar)  # warmup (cache hit)
    alloc_autocache = @allocated cubic_interp(x, y, xi_scalar)

    # FastInterpolations - In-place
    out = similar(xi_vec)
    cache = CubicSplineCache(x)
    cubic_interp!(out, cache, y, xi_vec)  # warmup
    alloc_inplace = @allocated cubic_interp!(out, cache, y, xi_vec)

    # Interpolations.jl
    if use_range
        itp_interp = Interpolations.cubic_spline_interpolation(x, y)
        itp_interp(xi_scalar)
        alloc_interp_scalar = @allocated itp_interp(xi_scalar)
        itp_interp(xi_vec)
        alloc_interp_vector = @allocated itp_interp(xi_vec)
    else
        alloc_interp_scalar = -1
        alloc_interp_vector = -1
    end

    # DataInterpolations.jl
    itp_data = DataInterpolations.CubicSpline(y, x)
    itp_data(xi_scalar)
    alloc_data_scalar = @allocated itp_data(xi_scalar)
    itp_data(xi_vec)
    alloc_data_vector = @allocated itp_data(xi_vec)

    return (
        FastInterp_callable_scalar = alloc_callable_scalar,
        FastInterp_callable_vector = alloc_callable_vector,
        FastInterp_autocache = alloc_autocache,
        FastInterp_inplace = alloc_inplace,
        Interpolations_scalar = alloc_interp_scalar,
        Interpolations_vector = alloc_interp_vector,
        DataInterp_scalar = alloc_data_scalar,
        DataInterp_vector = alloc_data_vector
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: Run All for README
# ═══════════════════════════════════════════════════════════════════════════════

"""
    benchmark_for_readme(; n_grid=100, n_query=1000, use_range=true)

Run benchmarks suitable for README figure generation.

# Returns
NamedTuple with all benchmark results
"""
function benchmark_for_readme(; n_grid::Int=100, n_query::Int=1000, use_range::Bool=true)
    println("Running benchmarks for README (n_grid=$n_grid, n_query=$n_query, use_range=$use_range)")
    println("This may take a few minutes...")

    print("  Construction... ")
    constr = benchmark_construction(; n_grid, use_range)
    println("done")

    print("  Evaluation ($n_query points)... ")
    eval_result = benchmark_evaluation(; n_grid, n_query, use_range)
    println("done")

    print("  One-shot ($n_query points)... ")
    oneshot = benchmark_oneshot(; n_grid, n_query, use_range)
    println("done")

    print("  Allocations... ")
    allocs = benchmark_allocation(; n_grid, use_range)
    println("done")

    println("\n" * "="^60)
    println("Results Summary (times in μs, n_query=$n_query)")
    println("="^60)

    println("\nConstruction:")
    println("  FastInterp:     $(round(constr.FastInterp/1000, digits=1)) μs")
    println("  Interpolations: $(isnan(constr.Interpolations) ? "N/A" : "$(round(constr.Interpolations/1000, digits=1)) μs")")
    println("  DataInterp:     $(round(constr.DataInterp/1000, digits=1)) μs")

    println("\nEvaluation (reuse interpolant, $n_query points):")
    println("  FastInterp:     $(round(eval_result.FastInterp/1000, digits=1)) μs")
    println("  Interpolations: $(isnan(eval_result.Interpolations) ? "N/A" : "$(round(eval_result.Interpolations/1000, digits=1)) μs")")
    println("  DataInterp:     $(round(eval_result.DataInterp/1000, digits=1)) μs")

    println("\nOne-shot (construct + eval, $n_query points):")
    println("  FastInterp:     $(round(oneshot.FastInterp.time_ns/1000, digits=1)) μs ($(oneshot.FastInterp.alloc_bytes) bytes)")
    println("  Interpolations: $(isnan(oneshot.Interpolations.time_ns) ? "N/A" : "$(round(oneshot.Interpolations.time_ns/1000, digits=1)) μs ($(oneshot.Interpolations.alloc_bytes) bytes)")")
    println("  DataInterp:     $(round(oneshot.DataInterp.time_ns/1000, digits=1)) μs ($(oneshot.DataInterp.alloc_bytes) bytes)")

    println("\nAllocations (scalar eval with cache hit):")
    println("  FastInterp callable: $(allocs.FastInterp_callable_scalar) bytes")
    println("  FastInterp autocache: $(allocs.FastInterp_autocache) bytes")
    println("  Interpolations: $(allocs.Interpolations_scalar == -1 ? "N/A" : "$(allocs.Interpolations_scalar) bytes")")
    println("  DataInterp: $(allocs.DataInterp_scalar) bytes")

    return (
        construction = constr,
        evaluation = eval_result,
        oneshot = oneshot,
        allocations = allocs,
        params = (n_grid=n_grid, n_query=n_query, use_range=use_range)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Format for plotting
# ═══════════════════════════════════════════════════════════════════════════════

"""
    to_bar_data(result; metric=:time_ns)

Convert benchmark result to format suitable for bar charts.

# Returns
Vector of (label, value) pairs
"""
function to_bar_data(result::NamedTuple; divide_by::Float64=1000.0)
    labels = ["FastInterp", "Interpolations", "DataInterp"]
    values = [
        result.FastInterp / divide_by,
        isnan(result.Interpolations) ? 0.0 : result.Interpolations / divide_by,
        result.DataInterp / divide_by
    ]
    return collect(zip(labels, values))
end

"""Print a simple ASCII bar chart."""
function print_bar_chart(title::String, data::Vector{Tuple{String, Float64}}; unit::String="μs", width::Int=40)
    println("\n$title")
    println("-"^60)

    max_val = maximum(d[2] for d in data if d[2] > 0)

    for (label, val) in data
        if val <= 0
            println("  $(rpad(label, 15)) N/A")
        else
            bar_len = round(Int, (val / max_val) * width)
            bar = "█"^bar_len
            println("  $(rpad(label, 15)) $bar $(round(val, digits=1)) $unit")
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Scaling Benchmark (for README plots)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    benchmark_scaling() -> NamedTuple{(:construction, :evaluation, :oneshot)}

Run scaling benchmarks for cubic interpolation with Range x-grid.

Returns three DataFrames:
- `construction`: Construction time vs n_grid (10, 20, 50, ..., 1000)
- `evaluation`: Evaluation time vs n_query (1, 2, 5, ..., 100000) with fixed n_grid=100
- `oneshot`: One-shot time vs n_query (same as evaluation)

All times are in seconds. Columns: `n`, `FastInterp`, `Interpolations`, `DataInterp`

# Example
```julia
result = benchmark_scaling()
result.construction  # DataFrame with construction scaling
result.evaluation    # DataFrame with evaluation scaling
result.oneshot       # DataFrame with one-shot scaling
```
"""
function benchmark_scaling(; verbose::Bool=true)
    _setup_benchmark()

    # Grid sizes for construction: 10, 20, 50, 100, ..., 10000
    grid_sizes = [5, 10, 20, 50, 100, 200, 500, 1000]
    # grid_sizes = [10, 100, 1000]
    # grid_sizes = [10]

    # Query sizes for evaluation/oneshot: log scale from 1 to 100000
    query_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    ns_to_sec(ns) = ns / 1e9

    # ─────────────────────────────────────────────────────────────────────────
    # Construction Benchmark (varying n_grid)
    # ─────────────────────────────────────────────────────────────────────────
    verbose && println("Construction benchmark (varying n_grid)...")

    constr_rows = []
    for n in grid_sizes
        verbose && print("  n_grid=$n... ")

        x = range(0.0, 10.0, n)
        y = sin.(x) .+ 0.1 .* collect(x)

        # FastInterpolations
        clear_cubic_cache!()
        b = @benchmark cubic_interp($x, $y; autocache=false)
        t_fast = ns_to_sec(median(b.times))
        alloc_fast = Int(b.memory)

        # Interpolations.jl
        b = @benchmark Interpolations.cubic_spline_interpolation($x, $y)
        t_itp = ns_to_sec(median(b.times))
        alloc_itp = Int(b.memory)

        # DataInterpolations.jl
        b = @benchmark DataInterpolations.CubicSpline($y, $x)
        t_di = ns_to_sec(median(b.times))
        alloc_di = Int(b.memory)

        push!(constr_rows, (
            n=n,
            FastInterp=t_fast,
            Interpolations=t_itp,
            DataInterp=t_di,
            alloc_FastInterp=alloc_fast,
            alloc_Interpolations=alloc_itp,
            alloc_DataInterp=alloc_di
        ))
        verbose && println("done")
    end
    df_construction = DataFrame(constr_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation Benchmark (fixed n_grid=100, varying n_query)
    # ─────────────────────────────────────────────────────────────────────────
    verbose && println("\nEvaluation benchmark (n_grid=100, varying n_query)...")

    n_grid = 100
    x = range(0.0, 10.0, n_grid)
    y = sin.(x) .+ 0.1 .* collect(x)

    # Pre-build interpolants
    clear_cubic_cache!()
    itp_fast = cubic_interp(x, y; autocache=false)
    itp_itp = Interpolations.cubic_spline_interpolation(x, y)
    itp_di = DataInterpolations.CubicSpline(y, x)

    eval_rows = []
    for nq in query_sizes
        verbose && print("  n_query=$nq... ")

        xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))

        b = @benchmark $itp_fast($xi)
        t_fast = ns_to_sec(median(b.times))

        b = @benchmark $itp_itp($xi)
        t_itp = ns_to_sec(median(b.times))

        b = @benchmark $itp_di($xi)
        t_di = ns_to_sec(median(b.times))

        push!(eval_rows, (n=nq, FastInterp=t_fast, Interpolations=t_itp, DataInterp=t_di))
        verbose && println("done")
    end
    df_evaluation = DataFrame(eval_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # One-Shot Benchmark (fixed n_grid=100, varying n_query)
    # ─────────────────────────────────────────────────────────────────────────
    verbose && println("\nOne-shot benchmark (n_grid=100, varying n_query)...")

    oneshot_rows = []
    for nq in query_sizes
        verbose && print("  n_query=$nq... ")

        xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))

        # FastInterpolations - autocache OFF (fresh construction each time)
        b = @benchmark cubic_interp($x, $y, $xi; autocache=false)
        t_fast_nocache = ns_to_sec(median(b.times))
        alloc_fast_nocache = Int(b.memory)

        # FastInterpolations - autocache ON (cache hit after prime)
        clear_cubic_cache!()
        cubic_interp(x, y, xi)  # prime cache
        b = @benchmark cubic_interp($x, $y, $xi)
        t_fast_cache = ns_to_sec(median(b.times))
        alloc_fast_cache = Int(b.memory)

        # Interpolations.jl (construct + evaluate each time)
        b = @benchmark Interpolations.cubic_spline_interpolation($x, $y)($xi)
        t_itp = ns_to_sec(median(b.times))
        alloc_itp = Int(b.memory)

        # DataInterpolations.jl (construct + evaluate each time)
        b = @benchmark DataInterpolations.CubicSpline($y, $x)($xi)
        t_di = ns_to_sec(median(b.times))
        alloc_di = Int(b.memory)

        push!(oneshot_rows, (
            n=nq,
            FastInterp_nocache=t_fast_nocache,
            FastInterp_cached=t_fast_cache,
            Interpolations=t_itp,
            DataInterp=t_di,
            alloc_FastInterp_nocache=alloc_fast_nocache,
            alloc_FastInterp_cached=alloc_fast_cache,
            alloc_Interpolations=alloc_itp,
            alloc_DataInterp=alloc_di
        ))
        verbose && println("done")
    end
    df_oneshot = DataFrame(oneshot_rows)

    verbose && println("\nDone! Results returned as (construction=DataFrame, evaluation=DataFrame, oneshot=DataFrame)")

    return (construction=df_construction, evaluation=df_evaluation, oneshot=df_oneshot)
end
