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

"""Print detailed benchmark statistics."""
function print_benchmark_stats(b, label::String)
    med = median(b)
    @printf("    @benchmark details (%s):\n", label)
    @printf("       ├─ Time: %.4f ± %.4f s\n", med.time / 1e9, std(b.times) / 1e9)
    @printf("       ├─ GC time: %.2f ms (%.1f%%)\n", med.gctime / 1e6, 100 * med.gctime / med.time)
    @printf("       ├─ Allocations: %d (%.2f MiB)\n", med.allocs, med.memory / 1024^2)
    @printf("       └─ Samples: %d, Evals/sample: %d\n", length(b.times), b.params.evals)
end

# =============================================================================
# Main Benchmark
# =============================================================================

function run_benchmark()
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

    # Generate test data
    psi_grid, data = generate_test_data(NPSI, MPERT)
    psi_values = generate_evaluation_points(N_EVAL_POINTS)
    total_spline_evals = N_EVAL_POINTS * MPERT * MPERT

    @printf("Grid spacing: %.4e (uniform)\n", step(psi_grid))
    println()

    # Pre-allocate output arrays
    A = Matrix{Float64}(undef, MPERT, MPERT)
    A_all = Array{Float64,3}(undef, N_EVAL_POINTS, MPERT, MPERT)

    # Results storage
    results = Dict{String,Any}()
    final_matrices = Dict{String,Matrix{Float64}}()

    # -------------------------------------------------------------------------
    # Interpolations.jl
    # -------------------------------------------------------------------------
    println("-"^70)
    println("Interpolations.jl (BSpline Cubic)")
    println("-"^70)

    println(" Benchmarking initialization...")
    init_interpolations_splines(psi_grid, data) # warm-up
    GC.gc()
    b_init = @benchmark $init_interpolations_splines($psi_grid, $data) samples = 5 evals = 2 seconds = 120
    init_time = median(b_init).time / 1e9
    init_std = std(b_init.times) / 1e9
    print_benchmark_stats(b_init, "Init")
    @printf("   Time per spline: %.2f μs\n", init_time / (MPERT * MPERT) * 1e6)
    println()

    interp_splines = init_interpolations_splines(psi_grid, data)

    println(" Benchmarking scalar API...")
    run_interpolations_eval_loop!(A, interp_splines, psi_values) # warm-up
    GC.gc()
    b_eval = @benchmark $run_interpolations_eval_loop!($A, $interp_splines, $psi_values) samples = 5 evals = 2 seconds = 120
    eval_time = median(b_eval).time / 1e9
    eval_std = std(b_eval.times) / 1e9
    print_benchmark_stats(b_eval, "Eval")
    @printf("   Evals/sec: %.2e\n", total_spline_evals / eval_time)
    @printf("   Time per eval: %.2f ns\n", eval_time / total_spline_evals * 1e9)
    results["Interpolations.jl (scalar)"] = (init_time=init_time, init_std=init_std,
        eval_time=eval_time, eval_std=eval_std, allocs=median(b_eval).allocs, memory=median(b_eval).memory)
    final_matrices["Interpolations.jl (scalar)"] = copy(A)
    println()

    println(" Benchmarking broadcast API...")
    run_interpolations_broadcast!(A_all, interp_splines, psi_values) # warm-up
    GC.gc()
    b_eval = @benchmark $run_interpolations_broadcast!($A_all, $interp_splines, $psi_values) samples = 5 evals = 2 seconds = 120
    eval_time = median(b_eval).time / 1e9
    eval_std = std(b_eval.times) / 1e9
    print_benchmark_stats(b_eval, "Eval")
    @printf("   Evals/sec: %.2e\n", total_spline_evals / eval_time)
    @printf("   Time per eval: %.2f ns\n", eval_time / total_spline_evals * 1e9)
    results["Interpolations.jl (broadcast)"] = (init_time=init_time, init_std=init_std,
        eval_time=eval_time, eval_std=eval_std, allocs=median(b_eval).allocs, memory=median(b_eval).memory)
    final_matrices["Interpolations.jl (broadcast)"] = copy(A_all[end, :, :])
    println()

    # -------------------------------------------------------------------------
    # DataInterpolations.jl
    # -------------------------------------------------------------------------
    println("-"^70)
    println("DataInterpolations.jl (CubicSpline)")
    println("-"^70)

    println(" Benchmarking initialization...")
    init_data_interpolations_splines(psi_grid, data) # warm-up
    GC.gc()
    b_init = @benchmark $init_data_interpolations_splines($psi_grid, $data) samples = 5 evals = 2 seconds = 120
    init_time = median(b_init).time / 1e9
    init_std = std(b_init.times) / 1e9
    print_benchmark_stats(b_init, "Init")
    @printf("   Time per spline: %.2f μs\n", init_time / (MPERT * MPERT) * 1e6)
    println()

    data_splines = init_data_interpolations_splines(psi_grid, data)

    println(" Benchmarking scalar API...")
    run_data_interpolations_eval_loop!(A, data_splines, psi_values)
    GC.gc()
    b_eval = @benchmark $run_data_interpolations_eval_loop!($A, $data_splines, $psi_values) samples = 5 evals = 2 seconds = 120
    eval_time = median(b_eval).time / 1e9
    eval_std = std(b_eval.times) / 1e9
    print_benchmark_stats(b_eval, "Eval")
    @printf("   Evals/sec: %.2e\n", total_spline_evals / eval_time)
    @printf("   Time per eval: %.2f ns\n", eval_time / total_spline_evals * 1e9)
    results["DataInterpolations.jl (scalar)"] = (init_time=init_time, init_std=init_std,
        eval_time=eval_time, eval_std=eval_std, allocs=median(b_eval).allocs, memory=median(b_eval).memory)
    final_matrices["DataInterpolations.jl (scalar)"] = copy(A)
    println()

    println(" Benchmarking vector API (in-place)...")
    run_data_interpolations_vector_API!(A_all, data_splines, psi_values)
    GC.gc()
    b_eval = @benchmark $run_data_interpolations_vector_API!($A_all, $data_splines, $psi_values) samples = 5 evals = 2 seconds = 120
    eval_time = median(b_eval).time / 1e9
    eval_std = std(b_eval.times) / 1e9
    print_benchmark_stats(b_eval, "Eval")
    @printf("   Evals/sec: %.2e\n", total_spline_evals / eval_time)
    @printf("   Time per eval: %.2f ns\n", eval_time / total_spline_evals * 1e9)
    results["DataInterpolations.jl (vector)"] = (init_time=init_time, init_std=init_std,
        eval_time=eval_time, eval_std=eval_std, allocs=median(b_eval).allocs, memory=median(b_eval).memory)
    final_matrices["DataInterpolations.jl (vector)"] = copy(A_all[end, :, :])
    println()

    # -------------------------------------------------------------------------
    # FastInterpolations.jl
    # -------------------------------------------------------------------------
    println("-"^70)
    println("FastInterpolations.jl (cubic_interp)")
    println("-"^70)

    println(" Benchmarking initialization...")
    init_fast_interpolations_splines(psi_grid, data) # warm-up
    GC.gc()
    b_init = @benchmark $init_fast_interpolations_splines($psi_grid, $data) samples = 5 evals = 2 seconds = 120
    init_time = median(b_init).time / 1e9
    init_std = std(b_init.times) / 1e9
    print_benchmark_stats(b_init, "Init")
    @printf("   Time per spline: %.2f μs\n", init_time / (MPERT * MPERT) * 1e6)
    println()

    fast_splines = init_fast_interpolations_splines(psi_grid, data)

    println(" Benchmarking scalar API...")
    run_fast_interpolations_eval_loop!(A, fast_splines, psi_values) # warm-up
    GC.gc()
    b_eval = @benchmark $run_fast_interpolations_eval_loop!($A, $fast_splines, $psi_values) samples = 5 evals = 2 seconds = 120
    eval_time = median(b_eval).time / 1e9
    eval_std = std(b_eval.times) / 1e9
    print_benchmark_stats(b_eval, "Eval")
    @printf("   Evals/sec: %.2e\n", total_spline_evals / eval_time)
    @printf("   Time per eval: %.2f ns\n", eval_time / total_spline_evals * 1e9)
    results["FastInterpolations.jl (scalar)"] = (init_time=init_time, init_std=init_std,
        eval_time=eval_time, eval_std=eval_std, allocs=median(b_eval).allocs, memory=median(b_eval).memory)
    final_matrices["FastInterpolations.jl (scalar)"] = copy(A)
    println()

    println(" Benchmarking vector API (in-place)...")
    run_fast_interpolations_vector_API!(A_all, fast_splines, psi_values) # warm-up
    GC.gc()
    b_eval = @benchmark $run_fast_interpolations_vector_API!($A_all, $fast_splines, $psi_values) samples = 5 evals = 2 seconds = 120
    eval_time = median(b_eval).time / 1e9
    eval_std = std(b_eval.times) / 1e9
    print_benchmark_stats(b_eval, "Eval")
    @printf("   Evals/sec: %.2e\n", total_spline_evals / eval_time)
    @printf("   Time per eval: %.2f ns\n", eval_time / total_spline_evals * 1e9)
    results["FastInterpolations.jl (vector)"] = (init_time=init_time, init_std=init_std,
        eval_time=eval_time, eval_std=eval_std, allocs=median(b_eval).allocs, memory=median(b_eval).memory)
    final_matrices["FastInterpolations.jl (vector)"] = copy(A_all[end, :, :])
    println()


    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    println("="^100)
    println("Summary (sorted by evaluation time)")
    println("="^100)
    println()
    @printf("%-35s %16s %16s %12s %10s %12s\n",
        "Package", "Init (s)", "Eval (s)", "Evals/sec", "Allocs", "Memory")
    println("-"^100)
    for (name, r) in sort(collect(results), by=x -> x[2].eval_time)
        evals_per_sec = total_spline_evals / r.eval_time
        @printf("%-35s %7.4f ± %6.4f %7.4f ± %6.4f %12.2e %10d %10.2f MiB\n",
            name, r.init_time, r.init_std, r.eval_time, r.eval_std,
            evals_per_sec, r.allocs, r.memory / 1024^2)
    end
    println()

    # Verify numerical consistency
    if length(final_matrices) > 1
        println("Numerical Consistency Check:")
        packages = collect(keys(final_matrices))
        ref_name = first(packages)
        ref_matrix = final_matrices[ref_name]
        for pkg in packages[2:end]
            max_diff = maximum(abs.(ref_matrix .- final_matrices[pkg]))
            @printf("  %s vs %s: max diff = %.2e\n", ref_name, pkg, max_diff)
        end
        println()
    end

    println("Benchmark complete!")
    return results
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
