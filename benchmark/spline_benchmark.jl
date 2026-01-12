#!/usr/bin/env julia
#
# Cubic Spline Benchmark: Performance Comparison
#
# Compares cubic spline performance across three Julia interpolation packages:
#   - Interpolations.jl    
#   - DataInterpolations.jl 
#   - FastInterpolations.jl 
#
# Benchmark Structure:
#   1. Initialization: Time to construct mpert×mpert spline objects
#   2. Scalar API: Point-by-point evaluation 
#   3. Vector API: Batch evaluation with in-place output (where supported)
#
# Use Case:
#   Simulates JPEC equilibrium matrix interpolation where npsi×mpert×mpert
#   data requires mpert² independent splines along the psi direction.
#
# Usage:
#   julia --project=benchmark benchmark/spline_benchmark.jl (from repository root)
#   julia --project=. spline_benchmark.jl (inside /benchmark folder)

using BenchmarkTools
using Interpolations
using DataInterpolations
using FastInterpolations
using Random
using Printf
using Statistics

# =============================================================================
# Configuration
# =============================================================================

const NPSI = 64            # Number of grid points
const MPERT = 200          # Matrix dimension (mpert × mpert splines)
const N_EVAL_POINTS = 4000 # Evaluation points (query points) 

# =============================================================================
# Test Data Generation
# =============================================================================

"""
    generate_test_data(npsi, mpert; seed=42) -> (psi_grid, data)

Generate uniform grid and random 3D data matrix for benchmarking.
Returns a range for psi_grid (uniform spacing) and npsi×mpert×mpert data array.
"""
function generate_test_data(npsi::Int, mpert::Int; seed::Int=42)
    Random.seed!(seed)
    psi_grid = range(0.0, 1.0, length=npsi)
    data = rand(npsi, mpert, mpert)
    return psi_grid, data
end

"""
    generate_evaluation_points(n_points) -> Vector{Float64}

Generate evaluation points with cubic spacing (clusters near psi=0).
This mimics typical ODE solver behavior near singular surfaces.
"""
function generate_evaluation_points(n_points::Int)
    return collect(range(0.0, 1.0, length=n_points)) .^ 3
end

# =============================================================================
# Interpolations.jl
# =============================================================================

"""
Initialize Interpolations.jl cubic splines for all (m, m') pairs.
Uses `cubic_spline_interpolation` which wraps BSpline(Cubic) with natural BC.
"""
function init_interpolations_splines(psi_grid::AbstractRange, data::Array{Float64,3})
    _, mpert, _ = size(data)
    first_itp = cubic_spline_interpolation(psi_grid, data[:, 1, 1])
    splines = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    splines[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        splines[m1, m2] = cubic_spline_interpolation(psi_grid, data[:, m1, m2])
    end
    return splines
end

"""Evaluate all splines at a single point (scalar API)."""
function eval_interpolations_splines!(A::Matrix{Float64}, splines::Matrix, psi::Float64)
    for m1 in axes(splines, 1), m2 in axes(splines, 2)
        A[m1, m2] = splines[m1, m2](psi)
    end
    return A
end

"""Sequential evaluation loop simulating ODE integration."""
function run_interpolations_eval_loop!(A::Matrix{Float64}, splines::Matrix, psi_values::Vector{Float64})
    for psi in psi_values
        eval_interpolations_splines!(A, splines, psi)
    end
    return A
end

"""Batch evaluation using broadcasting."""
function run_interpolations_broadcast!(A::Array{Float64,3}, splines::Matrix, psi_values::Vector{Float64})
    for m1 in axes(splines, 1), m2 in axes(splines, 2)
        @. A[:, m1, m2] = splines[m1, m2](psi_values)
    end
    return A
end

# =============================================================================
# FastInterpolations.jl
# =============================================================================

"""
Initialize FastInterpolations.jl cubic splines for all (m, m') pairs.
Uses `cubic_interp` which provides C² continuous with Natural boundary conditions.
"""
function init_fast_interpolations_splines(psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    first_itp = FastInterpolations.cubic_interp(psi_grid, data[:, 1, 1])
    splines = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    splines[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        splines[m1, m2] = FastInterpolations.cubic_interp(psi_grid, data[:, m1, m2])
    end
    return splines
end

"""Evaluate all splines at a single point (scalar API)."""
function eval_fast_interpolations_splines!(A::Matrix{Float64}, splines::Matrix, psi::Float64)
    for m1 in axes(splines, 1), m2 in axes(splines, 2)
        A[m1, m2] = splines[m1, m2](psi)
    end
    return A
end

"""Sequential evaluation loop using scalar API."""
function run_fast_interpolations_eval_loop!(A::Matrix{Float64}, splines::Matrix, psi_values::Vector{Float64})
    for psi in psi_values
        eval_fast_interpolations_splines!(A, splines, psi)
    end
    return A
end

"""
Batch evaluation using in-place vector API.
Syntax: spline(output, input_vector).
"""
function run_fast_interpolations_vector_API!(A::Array{Float64,3}, splines::Matrix, psi_values::Vector{Float64})
    for m1 in axes(splines, 1), m2 in axes(splines, 2)
        @views splines[m1, m2](A[:, m1, m2], psi_values)
    end
    return A
end

# =============================================================================
# MultiCubicInterpolant Helpers
# =============================================================================

"""Initialize MultiCubicInterpolant from 3D data array."""
function init_multi_cubic_interp(psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    # Create vector of y-series in COLUMN-MAJOR order (m1 varies fastest)
    # This matches reshape() behavior: reshape(vec, m, n)[i,j] = vec[(j-1)*m + i]
    ys = [data[:, m1, m2] for m2 in 1:mpert for m1 in 1:mpert]
    return FastInterpolations.cubic_interp(psi_grid, ys)
end

"""Sequential evaluation loop using scalar API with anchor reuse."""
function run_multi_cubic_eval_loop!(A::Vector{Float64}, mitp, psi_values::Vector{Float64})
    for psi in psi_values
        mitp(A, psi)  # In-place scalar evaluation
    end
    return A
end

"""Batch evaluation using in-place vector API."""
function run_multi_cubic_vector_API!(A_all::Vector{Vector{Float64}}, mitp, psi_values::Vector{Float64})
    mitp(A_all, psi_values)
    return A_all
end

# =============================================================================
# DataInterpolations.jl
# =============================================================================

"""
Initialize DataInterpolations.jl cubic splines for all (m, m') pairs.
Uses `CubicSpline(u, t)` - note argument order: values first, then grid.
"""
function init_data_interpolations_splines(psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    t = collect(psi_grid)  # DataInterpolations requires Vector
    first_itp = DataInterpolations.CubicSpline(data[:, 1, 1], t)
    splines = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    splines[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        splines[m1, m2] = DataInterpolations.CubicSpline(data[:, m1, m2], t)
    end
    return splines
end

"""Evaluate all splines at a single point (scalar API)."""
function eval_data_interpolations_splines!(A::Matrix{Float64}, splines::Matrix, psi::Float64)
    for m1 in axes(splines, 1), m2 in axes(splines, 2)
        A[m1, m2] = splines[m1, m2](psi)
    end
    return A
end

"""Sequential evaluation loop using scalar API."""
function run_data_interpolations_eval_loop!(A::Matrix{Float64}, splines::Matrix, psi_values::Vector{Float64})
    for psi in psi_values
        eval_data_interpolations_splines!(A, splines, psi)
    end
    return A
end

"""
Batch evaluation using in-place vector API.
Syntax: spline(output, input_vector) - same as FastInterpolations.jl.
"""
function run_data_interpolations_vector_API!(A::Array{Float64,3}, splines::Matrix, psi_values::Vector{Float64})
    for m1 in axes(splines, 1), m2 in axes(splines, 2)
        @views splines[m1, m2](A[:, m1, m2], psi_values)
    end
    return A
end

# =============================================================================
# Benchmark Utilities
# =============================================================================

"""Print section header with divider lines."""
function print_section_header(title::String)
    println("-"^70)
    println(title)
    println("-"^70)
end

"""Print detailed benchmark statistics."""
function print_benchmark_stats(b, label::String)
    med = median(b)
    @printf("    @benchmark details (%s):\n", label)
    @printf("       ├─ Time: %.4f ± %.4f s\n", med.time / 1e9, std(b.times) / 1e9)
    @printf("       ├─ GC time: %.2f ms (%.1f%%)\n", med.gctime / 1e6, 100 * med.gctime / med.time)
    @printf("       ├─ Allocations: %d (%.2f MiB)\n", med.allocs, med.memory / 1024^2)
    @printf("       └─ Samples: %d, Evals/sample: %d\n", length(b.times), b.params.evals)
end

"""
Benchmark initialization and return (time, std, allocs, memory, splines).
The `init_func` should be a zero-argument closure that returns the splines.
"""
function benchmark_init(init_func; n_splines::Int, verbose::Bool=false)
    println(" Benchmarking initialization...")
    init_func()  # warm-up
    GC.gc()
    b = @benchmark $init_func() samples = 5 evals = 2 seconds = 120
    init_time = median(b).time / 1e9
    init_std = std(b.times) / 1e9
    init_allocs = median(b).allocs
    init_memory = median(b).memory
    verbose && print_benchmark_stats(b, "Init")
    @printf("   Time per spline: %.2f μs\n", init_time / n_splines * 1e6)
    println()
    return (time=init_time, std=init_std, allocs=init_allocs, memory=init_memory, splines=init_func())
end

"""
Benchmark evaluation and store results.
The `eval_func` should be a zero-argument closure that performs evaluation.
`get_matrix` extracts the final matrix for consistency check.
"""
function benchmark_eval!(
    results::Dict, final_matrices::Dict,
    name::String, api_label::String,
    eval_func, get_matrix;
    init_time::Float64, init_std::Float64,
    init_allocs::Int, init_memory::Int,
    total_evals::Int, verbose::Bool=false
)
    println(" Benchmarking $api_label...")
    eval_func()  # warm-up
    GC.gc()
    b = @benchmark $eval_func() samples = 5 evals = 2 seconds = 120
    eval_time = median(b).time / 1e9
    eval_std = std(b.times) / 1e9
    verbose && print_benchmark_stats(b, "Eval")
    @printf("   Evals/sec: %.2e\n", total_evals / eval_time)
    @printf("   Time per eval: %.2f ns\n", eval_time / total_evals * 1e9)
    results[name] = (
        init_time=init_time, init_std=init_std,
        init_allocs=init_allocs, init_memory=init_memory,
        eval_time=eval_time, eval_std=eval_std,
        eval_allocs=median(b).allocs, eval_memory=median(b).memory
    )
    final_matrices[name] = get_matrix()
    println()
end

# =============================================================================
# Main Benchmark
# =============================================================================

function run_benchmark(; verbose::Bool=false)
    println("="^70)
    println("Cubic Spline Benchmark")
    println("="^70)
    println()
    @printf("Configuration:\n")
    @printf("  Grid points (npsi)     = %d\n", NPSI)
    @printf("  Matrix dimension       = %d × %d\n", MPERT, MPERT)
    @printf("  Total splines          = %d\n", MPERT * MPERT)
    @printf("  Evaluation points      = %d\n", N_EVAL_POINTS)
    println()

    # Generate test data: psi_grid (uniform), data[psi, m1, m2]
    psi_grid, data = generate_test_data(NPSI, MPERT)
    psi_values = generate_evaluation_points(N_EVAL_POINTS)
    # Total individual spline evaluations: N_EVAL_POINTS × (MPERT × MPERT splines)
    n_splines = MPERT * MPERT
    total_spline_evals = N_EVAL_POINTS * n_splines

    @printf("Grid spacing: %.4e (uniform)\n", step(psi_grid))
    println()

    # Pre-allocate output arrays
    # A: Single evaluation output - stores mpert×mpert matrix for one psi value (scalar API)
    A = Matrix{Float64}(undef, MPERT, MPERT)
    # A_all: Batch evaluation output - stores all evaluations: A_all[i,m1,m2] = spline[m1,m2](psi_values[i])
    A_all = Array{Float64,3}(undef, N_EVAL_POINTS, MPERT, MPERT)

    # Results storage
    results = Dict{String,Any}()                    # Benchmark timing results per package
    final_matrices = Dict{String,Matrix{Float64}}() # Final output for numerical consistency check

    # -------------------------------------------------------------------------
    # Interpolations.jl
    # -------------------------------------------------------------------------
    print_section_header("Interpolations.jl (BSpline Cubic)")

    init_result = benchmark_init(() -> init_interpolations_splines(psi_grid, data); n_splines=n_splines, verbose=verbose)
    interp_splines = init_result.splines

    benchmark_eval!(results, final_matrices,
        "Interpolations.jl (scalar)", "scalar API",
        () -> run_interpolations_eval_loop!(A, interp_splines, psi_values),
        () -> copy(A);
        init_time=init_result.time, init_std=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_spline_evals, verbose=verbose)

    benchmark_eval!(results, final_matrices,
        "Interpolations.jl (broadcast)", "broadcast API",
        () -> run_interpolations_broadcast!(A_all, interp_splines, psi_values),
        () -> copy(A_all[end, :, :]);
        init_time=init_result.time, init_std=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_spline_evals, verbose=verbose)

    # -------------------------------------------------------------------------
    # FastInterpolations.jl
    # -------------------------------------------------------------------------
    print_section_header("FastInterpolations.jl (cubic_interp)")

    init_result = benchmark_init(() -> init_fast_interpolations_splines(psi_grid, data); n_splines=n_splines, verbose=verbose)
    fast_splines = init_result.splines

    benchmark_eval!(results, final_matrices,
        "FastInterpolations.jl (scalar)", "scalar API",
        () -> run_fast_interpolations_eval_loop!(A, fast_splines, psi_values),
        () -> copy(A);
        init_time=init_result.time, init_std=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_spline_evals, verbose=verbose)

    benchmark_eval!(results, final_matrices,
        "FastInterpolations.jl (vector)", "vector API (in-place)",
        () -> run_fast_interpolations_vector_API!(A_all, fast_splines, psi_values),
        () -> copy(A_all[end, :, :]);
        init_time=init_result.time, init_std=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_spline_evals, verbose=verbose)

    # -------------------------------------------------------------------------
    # FastInterpolations.jl (MultiCubicInterpolant API)
    # -------------------------------------------------------------------------
    print_section_header("FastInterpolations.jl (MultiCubicInterpolant)")
    println()
    println(" Note: MultiCubicInterpolant computes anchor ONCE per query point,")
    println("       then reuses it for all $(n_splines) y-series. This should")
    println("       significantly outperform independent splines on scalar API.")
    println()

    init_result = benchmark_init(() -> init_multi_cubic_interp(psi_grid, data); n_splines=n_splines, verbose=verbose)
    mitp = init_result.splines

    # Pre-allocate outputs for Multi API
    A_multi = Vector{Float64}(undef, n_splines)
    A_multi_all = [Vector{Float64}(undef, N_EVAL_POINTS) for _ in 1:n_splines]

    benchmark_eval!(results, final_matrices,
        "FastInterpolations.jl (Multi+scalar)", "scalar API (anchor reuse)",
        () -> run_multi_cubic_eval_loop!(A_multi, mitp, psi_values),
        () -> reshape(copy(A_multi), MPERT, MPERT);
        init_time=init_result.time, init_std=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_spline_evals, verbose=verbose)

    benchmark_eval!(results, final_matrices,
        "FastInterpolations.jl (Multi+vector)", "vector API (in-place, zero-alloc)",
        () -> run_multi_cubic_vector_API!(A_multi_all, mitp, psi_values),
        () -> reshape([buf[end] for buf in A_multi_all], MPERT, MPERT);
        init_time=init_result.time, init_std=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_spline_evals, verbose=verbose)

    # -------------------------------------------------------------------------
    # DataInterpolations.jl
    # -------------------------------------------------------------------------
    print_section_header("DataInterpolations.jl (CubicSpline)")

    init_result = benchmark_init(() -> init_data_interpolations_splines(psi_grid, data); n_splines=n_splines, verbose=verbose)
    data_splines = init_result.splines

    benchmark_eval!(results, final_matrices,
        "DataInterpolations.jl (scalar)", "scalar API",
        () -> run_data_interpolations_eval_loop!(A, data_splines, psi_values),
        () -> copy(A);
        init_time=init_result.time, init_std=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_spline_evals, verbose=verbose)

    benchmark_eval!(results, final_matrices,
        "DataInterpolations.jl (vector)", "vector API (in-place)",
        () -> run_data_interpolations_vector_API!(A_all, data_splines, psi_values),
        () -> copy(A_all[end, :, :]);
        init_time=init_result.time, init_std=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_spline_evals, verbose=verbose)


    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    # Color mapping for packages
    pkg_colors = Dict(
        "Interpolations.jl" => :blue,
        "FastInterpolations.jl" => :green,
        "DataInterpolations.jl" => :yellow,
    )
    get_color(name) = begin
        for (prefix, c) in pkg_colors
            startswith(name, prefix) && return c
        end
        :default
    end

    # --- Part 1: One-shot (Total = Init + Eval) ---
    println("="^100)
    println("Summary: Total (Init + Eval)")
    println("="^100)
    @printf("%-38s %16s %10s %12s\n", "Package", "Total (s)", "Allocs", "Memory")
    println("-"^100)

    for (name, r) in sort(collect(results), by=x -> x[2].init_time + x[2].eval_time)
        total_time = r.init_time + r.eval_time
        total_std = sqrt(r.init_std^2 + r.eval_std^2)
        total_allocs = r.init_allocs + r.eval_allocs
        total_memory = r.init_memory + r.eval_memory
        main_part = @sprintf("%-38s %7.4f ± %6.4f", name, total_time, total_std)
        printstyled(main_part; color=get_color(name))
        alloc_part = @sprintf(" %10d %10.2f MiB", total_allocs, total_memory / 1024^2)
        print(alloc_part)
        println()
    end
    println()

    # --- Part 2: Initialization ---
    println("="^100)
    println("Summary: Initialization (vs DataInterpolations.jl scalar)")
    println("="^100)
    @printf("%-38s %16s %14s %8s %10s %12s\n", "Package", "Init (s)", "Splines/sec", "Speedup", "Allocs", "Memory")
    println("-"^100)

    baseline_init = results["DataInterpolations.jl (scalar)"].init_time
    for (name, r) in sort(collect(results), by=x -> x[2].init_time)
        splines_per_sec = n_splines / r.init_time
        speedup = baseline_init / r.init_time
        main_part = @sprintf("%-38s %7.4f ± %6.4f %14.2e", name, r.init_time, r.init_std, splines_per_sec)
        printstyled(main_part; color=get_color(name))
        speedup_part = @sprintf(" %7.2fx", speedup)
        print(speedup_part)
        alloc_part = @sprintf(" %10d %10.2f MiB", r.init_allocs, r.init_memory / 1024^2)
        print(alloc_part)
        println()
    end
    println()

    # --- Part 3: Evaluation ---
    println("="^100)
    println("Summary: Evaluation (vs DataInterpolations.jl scalar)")
    println("="^100)
    @printf("%-38s %16s %14s %8s %10s %12s\n", "Package", "Eval (s)", "Evals/sec", "Speedup", "Allocs", "Memory")
    println("-"^100)

    baseline_eval = results["DataInterpolations.jl (scalar)"].eval_time
    for (name, r) in sort(collect(results), by=x -> x[2].eval_time)
        evals_per_sec = total_spline_evals / r.eval_time
        speedup = baseline_eval / r.eval_time
        main_part = @sprintf("%-38s %7.4f ± %6.4f %14.2e", name, r.eval_time, r.eval_std, evals_per_sec)
        printstyled(main_part; color=get_color(name))
        speedup_part = @sprintf(" %7.2fx", speedup)
        print(speedup_part)
        alloc_part = @sprintf(" %10d %10.2f MiB", r.eval_allocs, r.eval_memory / 1024^2)
        print(alloc_part)
        println()
    end
    println()

    # Verify numerical consistency (baseline: DataInterpolations.jl scalar)
    println("="^100)
    println("Numerical Consistency Check (vs DataInterpolations.jl scalar)")
    println("="^100)
    ref_name = "DataInterpolations.jl (scalar)"
    if haskey(final_matrices, ref_name)
        ref_matrix = final_matrices[ref_name]
        @printf("%-50s %14s %10s\n", "Package", "Max Diff", "Status")
        println("-"^100)
        for (pkg, matrix) in sort(collect(final_matrices), by=x -> x[1])
            pkg == ref_name && continue
            max_diff = maximum(abs.(ref_matrix .- matrix))
            status = max_diff <= 1e-14 ? "✓" : (max_diff <= 1e-10 ? "⚠️ WARN" : "❌ FAIL")
            @printf("%-50s %14.2e %10s\n", pkg, max_diff, status)
        end
    else
        println("  Reference package not found!")
    end
    println()

    println("Benchmark complete!")
    return results
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
