#!/usr/bin/env julia
#
# Generalized Interpolation Benchmark: Performance Comparison
#
# Compares interpolation performance across three Julia packages:
#   - Interpolations.jl
#   - DataInterpolations.jl
#   - FastInterpolations.jl
#
# Supported Methods:
#   - constant: Piecewise constant (nearest-neighbor) interpolation
#   - linear:   Piecewise linear interpolation
#   - quadratic: Quadratic spline interpolation
#   - cubic:    Cubic spline interpolation (C² continuous)
#
# Benchmark Structure:
#   1. Initialization: Time to construct mpert×mpert interpolant objects
#   2. Scalar API: Point-by-point evaluation
#   3. Vector API: Batch evaluation with in-place output (where supported)
#
# FastInterpolations.jl Series APIs:
#   - *SeriesInterpolant: Adaptive layout (series-contiguous + lazy transpose)
#
# Use Case:
#   Simulates JPEC equilibrium matrix interpolation where npsi×mpert×mpert
#   data requires mpert² independent interpolants along the psi direction.
#
# Usage:
#   julia --project=benchmark benchmark/interpolation_benchmark.jl [METHOD] [SIZE]
#   julia --project=. interpolation_benchmark.jl [METHOD] [SIZE]
#
# METHOD options:
#   --constant  Piecewise constant interpolation
#   --linear    Piecewise linear interpolation
#   --quadratic Quadratic spline interpolation
#   --cubic     Cubic spline interpolation [default]
#
# SIZE options:
#   --tiny     Quick smoke test (mpert=10, eval=50)
#   --small    Fast iteration (mpert=20, eval=100)
#   --default  Standard benchmark (mpert=100, eval=100)  [default]
#   --large    Production-like (mpert=200, eval=4000)
#   --huge     Stress test (mpert=400, eval=10000)

using BenchmarkTools
using Interpolations
using DataInterpolations
using FastInterpolations
using Dierckx
using Random
using Printf
using Statistics

# =============================================================================
# Configuration
# =============================================================================

# Problem size presets: (npsi, mpert, n_eval_points)
const SIZE_PRESETS = Dict(
    :large_grid    => (1000,  2,    100),
    :tiny    => (16,  2,    5),
    :small   => (64,  5,   100),
    :default => (64, 100,   1000),
    :large   => (64, 200,  4000),
    :huge    => (64, 400, 10000),
)

# Interpolation method options
const METHOD_OPTIONS = [:constant, :linear, :quadratic, :cubic]

function print_help()
    println("""
Generalized Interpolation Benchmark

Compare interpolation performance across Julia packages:
  - Interpolations.jl
  - DataInterpolations.jl
  - FastInterpolations.jl

USAGE:
    julia --project=benchmark benchmark/interpolation_benchmark.jl [OPTIONS]

METHOD OPTIONS (choose one):
    --constant    Piecewise constant (nearest-neighbor) interpolation
    --linear      Piecewise linear interpolation
    --quadratic   Quadratic spline interpolation
    --cubic       Cubic spline interpolation [DEFAULT]

SIZE OPTIONS (choose one):
    --tiny        Quick smoke test (mpert=2, eval=5)
    --small       Fast iteration (mpert=5, eval=100)
    --default     Standard benchmark (mpert=100, eval=1000) [DEFAULT]
    --large       Production-like (mpert=200, eval=4000)
    --huge        Stress test (mpert=400, eval=10000)

OTHER OPTIONS:
    --help, -h    Show this help message

EXAMPLES:
    # Run cubic spline benchmark with default size
    julia --project=benchmark benchmark/interpolation_benchmark.jl

    # Run linear interpolation with small size
    julia --project=benchmark benchmark/interpolation_benchmark.jl --linear --small

    # Quick smoke test for all methods
    julia --project=benchmark benchmark/interpolation_benchmark.jl --quadratic --tiny
""")
end

function parse_args(args)
    # Check for help flag first
    if "--help" in args || "-h" in args
        print_help()
        exit(0)
    end

    size_key = :default
    method_key = :cubic

    for arg in args
        if startswith(arg, "--")
            key = Symbol(arg[3:end])
            if haskey(SIZE_PRESETS, key)
                size_key = key
            elseif key in METHOD_OPTIONS
                method_key = key
            else
                @warn "Unknown argument: $arg. Use --help to see available options."
            end
        end
    end
    return size_key, method_key
end

const (SIZE_KEY, METHOD_KEY) = parse_args(ARGS)
const (NPSI, MPERT, N_EVAL_POINTS) = SIZE_PRESETS[SIZE_KEY]

@info "Benchmark configuration" method=METHOD_KEY size=SIZE_KEY npsi=NPSI mpert=MPERT n_eval=N_EVAL_POINTS

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
# Method Names for Display
# =============================================================================

const METHOD_NAMES = Dict(
    :constant => "Constant (Piecewise)",
    :linear => "Linear",
    :quadratic => "Quadratic Spline",
    :cubic => "Cubic Spline",
)

# =============================================================================
# Interpolations.jl - Generic Functions
# =============================================================================

"""Initialize Interpolations.jl interpolants for all (m, m') pairs."""
function init_interpolations(::Val{:constant}, psi_grid::AbstractRange, data::Array{Float64,3})
    _, mpert, _ = size(data)
    first_itp = constant_interpolation(psi_grid, data[:, 1, 1])
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = constant_interpolation(psi_grid, data[:, m1, m2])
    end
    return interps
end

function init_interpolations(::Val{:linear}, psi_grid::AbstractRange, data::Array{Float64,3})
    _, mpert, _ = size(data)
    first_itp = linear_interpolation(psi_grid, data[:, 1, 1])
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = linear_interpolation(psi_grid, data[:, m1, m2])
    end
    return interps
end

function init_interpolations(::Val{:quadratic}, psi_grid::AbstractRange, data::Array{Float64,3})
    _, mpert, _ = size(data)
    # Interpolations.jl requires explicit BSpline construction for quadratic
    # Using Reflect boundary condition which is common for quadratic splines
    knots = (psi_grid,)
    first_itp = extrapolate(scale(interpolate(data[:, 1, 1], BSpline(Quadratic(Reflect(OnCell())))), knots), Throw())
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = extrapolate(scale(interpolate(data[:, m1, m2], BSpline(Quadratic(Reflect(OnCell())))), knots), Throw())
    end
    return interps
end

function init_interpolations(::Val{:cubic}, psi_grid::AbstractRange, data::Array{Float64,3})
    _, mpert, _ = size(data)
    first_itp = cubic_spline_interpolation(psi_grid, data[:, 1, 1])
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = cubic_spline_interpolation(psi_grid, data[:, m1, m2])
    end
    return interps
end

"""Evaluate all interpolants at a single point (scalar API)."""
function eval_interpolations!(A::Matrix{Float64}, interps::Matrix, psi::Float64)
    for m2 in axes(interps, 2), m1 in axes(interps, 1)
        A[m1, m2] = interps[m1, m2](psi)
    end
    return A
end

"""Sequential evaluation loop simulating ODE integration."""
function run_interpolations_eval_loop!(A::Matrix{Float64}, interps::Matrix, psi_values::Vector{Float64})
    for psi in psi_values
        eval_interpolations!(A, interps, psi)
    end
    return A
end

"""Batch evaluation using broadcasting."""
function run_interpolations_broadcast!(A::Array{Float64,3}, interps::Matrix, psi_values::Vector{Float64})
    for m2 in axes(interps, 2), m1 in axes(interps, 1)
        @. A[:, m1, m2] = interps[m1, m2](psi_values)
    end
    return A
end

# =============================================================================
# FastInterpolations.jl - Generic Functions
# =============================================================================

"""Initialize FastInterpolations.jl interpolants for all (m, m') pairs."""
function init_fast_interpolations(::Val{:constant}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    first_itp = FastInterpolations.constant_interp(psi_grid, data[:, 1, 1])
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = FastInterpolations.constant_interp(psi_grid, data[:, m1, m2])
    end
    return interps
end

function init_fast_interpolations(::Val{:linear}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    first_itp = FastInterpolations.linear_interp(psi_grid, data[:, 1, 1])
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = FastInterpolations.linear_interp(psi_grid, data[:, m1, m2])
    end
    return interps
end

function init_fast_interpolations(::Val{:quadratic}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    first_itp = FastInterpolations.quadratic_interp(psi_grid, data[:, 1, 1])
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = FastInterpolations.quadratic_interp(psi_grid, data[:, m1, m2])
    end
    return interps
end

function init_fast_interpolations(::Val{:cubic}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    first_itp = FastInterpolations.cubic_interp(psi_grid, data[:, 1, 1])
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = FastInterpolations.cubic_interp(psi_grid, data[:, m1, m2])
    end
    return interps
end

"""Evaluate all interpolants at a single point (scalar API)."""
function eval_fast_interpolations!(A::Matrix{Float64}, interps::Matrix, psi::Float64)
    for m2 in axes(interps, 2), m1 in axes(interps, 1)
        A[m1, m2] = interps[m1, m2](psi)
    end
    return A
end

"""Sequential evaluation loop using scalar API."""
function run_fast_interpolations_eval_loop!(A::Matrix{Float64}, interps::Matrix, psi_values::Vector{Float64})
    for psi in psi_values
        eval_fast_interpolations!(A, interps, psi)
    end
    return A
end

"""
Batch evaluation using in-place vector API.
Syntax: interpolant(output, input_vector).
"""
function run_fast_interpolations_vector_API!(A::Array{Float64,3}, interps::Matrix, psi_values::Vector{Float64})
    for m2 in axes(interps, 2), m1 in axes(interps, 1)
        @views interps[m1, m2](A[:, m1, m2], psi_values)
    end
    return A
end

# =============================================================================
# FastInterpolations.jl Series API - Generic Functions
# =============================================================================

"""Initialize Series interpolant from 3D data array."""
function init_series_interp(::Val{:constant}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    # Create vector of y-series in COLUMN-MAJOR order (m1 varies fastest)
    ys = [data[:, m1, m2] for m2 in 1:mpert for m1 in 1:mpert]
    return FastInterpolations.constant_interp(psi_grid, ys)
end

function init_series_interp(::Val{:linear}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    ys = [data[:, m1, m2] for m2 in 1:mpert for m1 in 1:mpert]
    return FastInterpolations.linear_interp(psi_grid, ys)
end

function init_series_interp(::Val{:quadratic}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    ys = [data[:, m1, m2] for m2 in 1:mpert for m1 in 1:mpert]
    return FastInterpolations.quadratic_interp(psi_grid, ys)
end

function init_series_interp(::Val{:cubic}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    ys = [data[:, m1, m2] for m2 in 1:mpert for m1 in 1:mpert]
    return FastInterpolations.cubic_interp(psi_grid, ys)
end

"""Sequential evaluation loop using scalar API with anchor reuse."""
function run_series_eval_loop!(A::Vector{Float64}, sitp, psi_values::Vector{Float64})
    for psi in psi_values
        sitp(A, psi)  # In-place scalar evaluation
    end
    return A
end

"""Batch evaluation using in-place vector API."""
function run_series_vector_API!(A_all::Vector{Vector{Float64}}, sitp, psi_values::Vector{Float64})
    sitp(A_all, psi_values)
    return A_all
end

# =============================================================================
# DataInterpolations.jl - Generic Functions
# =============================================================================

"""Initialize DataInterpolations.jl interpolants for all (m, m') pairs."""
function init_data_interpolations(::Val{:constant}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    t = collect(psi_grid)  # DataInterpolations requires Vector
    first_itp = DataInterpolations.ConstantInterpolation(data[:, 1, 1], t)
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = DataInterpolations.ConstantInterpolation(data[:, m1, m2], t)
    end
    return interps
end

function init_data_interpolations(::Val{:linear}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    t = collect(psi_grid)
    first_itp = DataInterpolations.LinearInterpolation(data[:, 1, 1], t)
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = DataInterpolations.LinearInterpolation(data[:, m1, m2], t)
    end
    return interps
end

function init_data_interpolations(::Val{:quadratic}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    t = collect(psi_grid)
    first_itp = DataInterpolations.QuadraticInterpolation(data[:, 1, 1], t)
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = DataInterpolations.QuadraticInterpolation(data[:, m1, m2], t)
    end
    return interps
end

function init_data_interpolations(::Val{:cubic}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    t = collect(psi_grid)
    first_itp = DataInterpolations.CubicSpline(data[:, 1, 1], t)
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = DataInterpolations.CubicSpline(data[:, m1, m2], t)
    end
    return interps
end

"""Evaluate all interpolants at a single point (scalar API)."""
function eval_data_interpolations!(A::Matrix{Float64}, interps::Matrix, psi::Float64)
    for m2 in axes(interps, 2), m1 in axes(interps, 1)
        A[m1, m2] = interps[m1, m2](psi)
    end
    return A
end

"""Sequential evaluation loop using scalar API."""
function run_data_interpolations_eval_loop!(A::Matrix{Float64}, interps::Matrix, psi_values::Vector{Float64})
    for psi in psi_values
        eval_data_interpolations!(A, interps, psi)
    end
    return A
end

"""
Batch evaluation using in-place vector API.
Syntax: interpolant(output, input_vector) - same as FastInterpolations.jl.
"""
function run_data_interpolations_vector_API!(A::Array{Float64,3}, interps::Matrix, psi_values::Vector{Float64})
    for m2 in axes(interps, 2), m1 in axes(interps, 1)
        @views interps[m1, m2](A[:, m1, m2], psi_values)
    end
    return A
end

# =============================================================================
# Dierckx.jl - Cubic Spline Only (FITPACK wrapper)
# =============================================================================

"""Initialize Dierckx.jl interpolants for all (m, m') pairs. Cubic only."""
function init_dierckx(::Val{:cubic}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    t = collect(psi_grid)  # Dierckx requires Vector
    first_itp = Dierckx.Spline1D(t, data[:, 1, 1]; k=3, s=0.0)
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = Dierckx.Spline1D(t, data[:, m1, m2]; k=3, s=0.0)
    end
    return interps
end

# Dierckx also supports k=1 (linear) and k=2 (quadratic), but we focus on cubic
function init_dierckx(::Val{:linear}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    t = collect(psi_grid)
    first_itp = Dierckx.Spline1D(t, data[:, 1, 1]; k=1, s=0.0)
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = Dierckx.Spline1D(t, data[:, m1, m2]; k=1, s=0.0)
    end
    return interps
end

function init_dierckx(::Val{:quadratic}, psi_grid, data::Array{Float64,3})
    _, mpert, _ = size(data)
    t = collect(psi_grid)
    first_itp = Dierckx.Spline1D(t, data[:, 1, 1]; k=2, s=0.0)
    interps = Matrix{typeof(first_itp)}(undef, mpert, mpert)
    interps[1, 1] = first_itp
    for m1 in 1:mpert, m2 in 1:mpert
        (m1 == 1 && m2 == 1) && continue
        interps[m1, m2] = Dierckx.Spline1D(t, data[:, m1, m2]; k=2, s=0.0)
    end
    return interps
end

# Dierckx doesn't support k=0 (constant), skip for :constant
function init_dierckx(::Val{:constant}, psi_grid, data::Array{Float64,3})
    return nothing  # Not supported
end

"""Evaluate all Dierckx interpolants at a single point (scalar API)."""
function eval_dierckx!(A::Matrix{Float64}, interps::Matrix, psi::Float64)
    for m2 in axes(interps, 2), m1 in axes(interps, 1)
        A[m1, m2] = interps[m1, m2](psi)
    end
    return A
end

"""Sequential evaluation loop using scalar API."""
function run_dierckx_eval_loop!(A::Matrix{Float64}, interps::Matrix, psi_values::Vector{Float64})
    for psi in psi_values
        eval_dierckx!(A, interps, psi)
    end
    return A
end

"""Batch evaluation using Dierckx's vector evaluation."""
function run_dierckx_vector_API!(A::Array{Float64,3}, interps::Matrix, psi_values::Vector{Float64})
    for m2 in axes(interps, 2), m1 in axes(interps, 1)
        # Dierckx Spline1D supports vector evaluation directly
        @views A[:, m1, m2] .= interps[m1, m2](psi_values)
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
    @printf("       ├─ Time: %.3f ± %.3f ms\n", med.time / 1e6, std(b.times) / 1e6)
    @printf("       ├─ GC time: %.2f ms (%.1f%%)\n", med.gctime / 1e6, 100 * med.gctime / med.time)
    @printf("       ├─ Allocations: %d (%.2f MiB)\n", med.allocs, med.memory / 1024^2)
    @printf("       └─ Samples: %d, Evals/sample: %d\n", length(b.times), b.params.evals)
end

"""
Benchmark initialization and return (time, std, allocs, memory, interps).
The `init_func` should be a zero-argument closure that returns the interpolants.
"""
function benchmark_init(init_func; n_interps::Int, verbose::Bool=false)
    println(" Benchmarking initialization...")
    init_func()  # warm-up
    GC.gc()
    b = @benchmark $init_func() samples = 5 evals = 2 seconds = 120
    init_time_ms = median(b).time / 1e6
    init_std_ms = std(b.times) / 1e6
    init_allocs = median(b).allocs
    init_memory = median(b).memory
    verbose && print_benchmark_stats(b, "Init")
    @printf("   Time per interpolant: %.2f μs\n", init_time_ms / n_interps * 1e3)
    println()
    return (time=init_time_ms, std=init_std_ms, allocs=init_allocs, memory=init_memory, interps=init_func())
end

"""
Benchmark evaluation and store results.
The `eval_func` should be a zero-argument closure that performs evaluation.
`get_matrix` extracts the final matrix for consistency check.
Note: All times are in milliseconds (ms).
"""
function benchmark_eval!(
    results::Dict, final_matrices::Dict,
    name::String, api_label::String,
    eval_func, get_matrix;
    init_time_ms::Float64, init_std_ms::Float64,
    init_allocs::Int, init_memory::Int,
    total_evals::Int, verbose::Bool=false
)
    println(" Benchmarking $api_label...")
    eval_func()  # warm-up
    GC.gc()
    b = @benchmark $eval_func() samples = 5 evals = 2 seconds = 120
    eval_time_ms = median(b).time / 1e6
    eval_std_ms = std(b.times) / 1e6
    verbose && print_benchmark_stats(b, "Eval")
    @printf("   Evals/sec: %.2e\n", total_evals / (eval_time_ms / 1e3))
    @printf("   Time per eval: %.2f ns\n", eval_time_ms / total_evals * 1e6)
    results[name] = (
        init_time=init_time_ms, init_std=init_std_ms,
        init_allocs=init_allocs, init_memory=init_memory,
        eval_time=eval_time_ms, eval_std=eval_std_ms,
        eval_allocs=median(b).allocs, eval_memory=median(b).memory
    )
    final_matrices[name] = get_matrix()
    println()
end

# =============================================================================
# Main Benchmark
# =============================================================================

function run_benchmark(method::Symbol; verbose::Bool=false)
    method_name = METHOD_NAMES[method]
    method_val = Val(method)

    println("="^70)
    println("$(method_name) Interpolation Benchmark")
    println("="^70)
    println()
    @printf("Configuration:\n")
    @printf("  Method               = %s\n", method_name)
    @printf("  Grid points (npsi)   = %d\n", NPSI)
    @printf("  Matrix dimension     = %d × %d\n", MPERT, MPERT)
    @printf("  Total interpolants   = %d\n", MPERT * MPERT)
    @printf("  Evaluation points    = %d\n", N_EVAL_POINTS)
    println()

    # Generate test data: psi_grid (uniform), data[psi, m1, m2]
    psi_grid, data = generate_test_data(NPSI, MPERT)
    psi_values = generate_evaluation_points(N_EVAL_POINTS)
    # Total individual interpolant evaluations: N_EVAL_POINTS × (MPERT × MPERT interpolants)
    n_interps = MPERT * MPERT
    total_interp_evals = N_EVAL_POINTS * n_interps

    @printf("Grid spacing: %.4e (uniform)\n", step(psi_grid))
    println()

    # Pre-allocate output arrays
    # A: Single evaluation output - stores mpert×mpert matrix for one psi value (scalar API)
    A = Matrix{Float64}(undef, MPERT, MPERT)
    # A_all: Batch evaluation output - stores all evaluations: A_all[i,m1,m2] = interp[m1,m2](psi_values[i])
    A_all = Array{Float64,3}(undef, N_EVAL_POINTS, MPERT, MPERT)

    # Results storage
    results = Dict{String,Any}()                    # Benchmark timing results per package
    final_matrices = Dict{String,Matrix{Float64}}() # Final output for numerical consistency check

    # -------------------------------------------------------------------------
    # Interpolations.jl
    # -------------------------------------------------------------------------
    print_section_header("Interpolations.jl ($(method_name))")

    init_result = benchmark_init(() -> init_interpolations(method_val, psi_grid, data); n_interps=n_interps, verbose=verbose)
    interp_interps = init_result.interps

    benchmark_eval!(results, final_matrices,
        "Interpolations.jl (scalar)", "scalar API",
        () -> run_interpolations_eval_loop!(A, interp_interps, psi_values),
        () -> copy(A);
        init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)

    benchmark_eval!(results, final_matrices,
        "Interpolations.jl (broadcast)", "broadcast API",
        () -> run_interpolations_broadcast!(A_all, interp_interps, psi_values),
        () -> copy(A_all[end, :, :]);
        init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)

    # -------------------------------------------------------------------------
    # FastInterpolations.jl
    # -------------------------------------------------------------------------
    print_section_header("FastInterpolations.jl ($(method_name))")

    init_result = benchmark_init(() -> init_fast_interpolations(method_val, psi_grid, data); n_interps=n_interps, verbose=verbose)
    fast_interps = init_result.interps

    benchmark_eval!(results, final_matrices,
        "FastInterpolations.jl (scalar)", "scalar API",
        () -> run_fast_interpolations_eval_loop!(A, fast_interps, psi_values),
        () -> copy(A);
        init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)

    benchmark_eval!(results, final_matrices,
        "FastInterpolations.jl (vector)", "vector API (in-place)",
        () -> run_fast_interpolations_vector_API!(A_all, fast_interps, psi_values),
        () -> copy(A_all[end, :, :]);
        init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)

    # -------------------------------------------------------------------------
    # FastInterpolations.jl (Series API)
    # -------------------------------------------------------------------------
    series_type_name = uppercase(string(method)[1]) * string(method)[2:end] * "SeriesInterpolant"
    print_section_header("FastInterpolations.jl ($(series_type_name))")
    println()
    println(" Note: $(series_type_name) computes anchor ONCE per query point,")
    println("       then reuses it for all $(n_interps) y-series. This should")
    println("       significantly outperform independent interpolants on scalar API.")
    println()

    init_result = benchmark_init(() -> init_series_interp(method_val, psi_grid, data); n_interps=n_interps, verbose=verbose)
    sitp = init_result.interps

    # Pre-allocate outputs for Series API
    A_series = Vector{Float64}(undef, n_interps)
    A_series_all = [Vector{Float64}(undef, N_EVAL_POINTS) for _ in 1:n_interps]

    benchmark_eval!(results, final_matrices,
        "FastInterpolations.jl (Series+scalar)", "scalar API (anchor reuse)",
        () -> run_series_eval_loop!(A_series, sitp, psi_values),
        () -> reshape(copy(A_series), MPERT, MPERT);
        init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)

    benchmark_eval!(results, final_matrices,
        "FastInterpolations.jl (Series+vector)", "vector API (in-place, zero-alloc)",
        () -> run_series_vector_API!(A_series_all, sitp, psi_values),
        () -> reshape([buf[end] for buf in A_series_all], MPERT, MPERT);
        init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)

    # -------------------------------------------------------------------------
    # DataInterpolations.jl
    # -------------------------------------------------------------------------
    print_section_header("DataInterpolations.jl ($(method_name))")

    init_result = benchmark_init(() -> init_data_interpolations(method_val, psi_grid, data); n_interps=n_interps, verbose=verbose)
    data_interps = init_result.interps

    benchmark_eval!(results, final_matrices,
        "DataInterpolations.jl (scalar)", "scalar API",
        () -> run_data_interpolations_eval_loop!(A, data_interps, psi_values),
        () -> copy(A);
        init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)

    benchmark_eval!(results, final_matrices,
        "DataInterpolations.jl (vector)", "vector API (in-place)",
        () -> run_data_interpolations_vector_API!(A_all, data_interps, psi_values),
        () -> copy(A_all[end, :, :]);
        init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)

    # -------------------------------------------------------------------------
    # Dierckx.jl (FITPACK wrapper - cubic/quadratic/linear only, no constant)
    # -------------------------------------------------------------------------
    dierckx_interps = init_dierckx(method_val, psi_grid, data)
    if dierckx_interps !== nothing
        print_section_header("Dierckx.jl ($(method_name))")

        init_result = benchmark_init(() -> init_dierckx(method_val, psi_grid, data); n_interps=n_interps, verbose=verbose)
        dierckx_interps = init_result.interps

        benchmark_eval!(results, final_matrices,
            "Dierckx.jl (scalar)", "scalar API",
            () -> run_dierckx_eval_loop!(A, dierckx_interps, psi_values),
            () -> copy(A);
            init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)

        benchmark_eval!(results, final_matrices,
            "Dierckx.jl (vector)", "vector API",
            () -> run_dierckx_vector_API!(A_all, dierckx_interps, psi_values),
            () -> copy(A_all[end, :, :]);
            init_time_ms=init_result.time, init_std_ms=init_result.std, init_allocs=init_result.allocs, init_memory=init_result.memory, total_evals=total_interp_evals, verbose=verbose)
    else
        println()
        println("Skipping Dierckx.jl: $(method_name) interpolation not supported (constant requires k≥1)")
        println()
    end

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    # Color mapping for packages
    pkg_colors = Dict(
        "Interpolations.jl" => :blue,
        "FastInterpolations.jl" => :green,
        "DataInterpolations.jl" => :yellow,
        "Dierckx.jl" => :magenta,
    )
    get_color(name) = begin
        for (prefix, c) in pkg_colors
            startswith(name, prefix) && return c
        end
        :default
    end

    # --- Part 1: One-shot (Total = Init + Eval) ---
    println("="^100)
    println("Summary: Total (Init + Eval) vs DataInterpolations.jl scalar")
    println("="^100)
    @printf("%-38s %18s %8s %10s %12s\n", "Package", "Total (ms)", "Speedup", "Allocs", "Memory")
    println("-"^100)

    baseline_total = results["DataInterpolations.jl (scalar)"].init_time + results["DataInterpolations.jl (scalar)"].eval_time
    for (name, r) in sort(collect(results), by=x -> x[2].init_time + x[2].eval_time)
        total_time = r.init_time + r.eval_time
        total_std = sqrt(r.init_std^2 + r.eval_std^2)
        total_allocs = r.init_allocs + r.eval_allocs
        total_memory = r.init_memory + r.eval_memory
        speedup = baseline_total / total_time
        main_part = @sprintf("%-38s %8.3f ± %6.3f", name, total_time, total_std)
        printstyled(main_part; color=get_color(name))
        speedup_part = @sprintf(" %7.2fx", speedup)
        print(speedup_part)
        alloc_part = @sprintf(" %10d %10.2f MiB", total_allocs, total_memory / 1024^2)
        print(alloc_part)
        println()
    end
    println()

    # --- Part 2: Initialization ---
    println("="^100)
    println("Summary: Initialization (vs DataInterpolations.jl scalar)")
    println("="^100)
    @printf("%-38s %18s %14s %8s %10s %12s\n", "Package", "Init (ms)", "Interps/sec", "Speedup", "Allocs", "Memory")
    println("-"^100)

    baseline_init = results["DataInterpolations.jl (scalar)"].init_time
    for (name, r) in sort(collect(results), by=x -> x[2].init_time)
        interps_per_sec = n_interps / (r.init_time / 1e3)  # ms → s for rate
        speedup = baseline_init / r.init_time
        main_part = @sprintf("%-38s %8.3f ± %6.3f %14.2e", name, r.init_time, r.init_std, interps_per_sec)
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
    @printf("%-38s %18s %14s %8s %10s %12s\n", "Package", "Eval (ms)", "Evals/sec", "Speedup", "Allocs", "Memory")
    println("-"^100)

    baseline_eval = results["DataInterpolations.jl (scalar)"].eval_time
    for (name, r) in sort(collect(results), by=x -> x[2].eval_time)
        evals_per_sec = total_interp_evals / (r.eval_time / 1e3)  # ms → s for rate
        speedup = baseline_eval / r.eval_time
        main_part = @sprintf("%-38s %8.3f ± %6.3f %14.2e", name, r.eval_time, r.eval_std, evals_per_sec)
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
    run_benchmark(METHOD_KEY)
end
