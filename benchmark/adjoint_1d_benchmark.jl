#!/usr/bin/env julia
#
# Cubic Spline Data-Adjoint Benchmark: Native API vs AD Backends
#
# Compares full ∂f/∂y gradient computation for cubic_interp across:
#   - Native API:   cubic_interp! (forward) + cubic_adjoint (pullback)
#   - ForwardDiff:  Dual number propagation through tridiagonal solve
#   - Zygote:       rrule → CubicAdjoint (analytical pullback)
#   - Enzyme:       EnzymeRules → CubicAdjoint (analytical pullback)
#
# All backends compute the same L2 loss gradient:
#   loss(f) = sum(|cubic_interp(x, f, xq) - y_obs|^2)
#   ∇f = ∂loss/∂f
#
# Correctness is verified via isapprox against ForwardDiff (ground truth).
#
# Usage:
#   julia --project adjoint_benchmark.jl [SIZE]
#
# SIZE options:
#   --small    N=20,  Nq=10   (quick smoke test)
#   --default  N=50,  Nq=30   (standard benchmark)
#   --medium   N=100, Nq=50
#   --large    N=200, Nq=100
#   --huge     N=500, Nq=300

using BenchmarkTools
using FastInterpolations
using ForwardDiff
using Zygote
using Enzyme
using LinearAlgebra: norm
using Printf
using Random

# =============================================================================
# Configuration
# =============================================================================

const SIZE_PRESETS = Dict(
    :small => (20, 10),
    :default => (50, 30),
    :medium => (100, 50),
    :large => (200, 100),
    :huge => (500, 300),
)

function parse_args(args)
    if "--help" in args || "-h" in args
        println(
            """
            Cubic Spline Data-Adjoint Benchmark: Native API vs AD Backends

            Compares full ∂f/∂y gradient pipeline:
              loss(f) = sum(|cubic_interp(x, f, xq) - y_obs|^2)

            Backends: Native (in-place/allocating), ForwardDiff, Zygote, Enzyme

            SIZE OPTIONS:
                --small    N=20,  Nq=10   (quick smoke test)
                --default  N=50,  Nq=30   [DEFAULT]
                --medium   N=100, Nq=50
                --large    N=200, Nq=100
                --huge     N=500, Nq=300

                --help, -h  Show this message
            """
        )
        exit(0)
    end

    size_key = :default
    for arg in args
        if startswith(arg, "--")
            key = Symbol(arg[3:end])
            haskey(SIZE_PRESETS, key) && (size_key = key)
        end
    end
    return size_key
end

const SIZE_KEY = parse_args(ARGS)
const (N_GRID, N_QUERY) = SIZE_PRESETS[SIZE_KEY]

# =============================================================================
# Test Data
# =============================================================================

Random.seed!(42)

const x = collect(range(0.0, 1.0, N_GRID))
const f = sin.(range(0, 2π, N_GRID)) .+ 0.1 .* randn(N_GRID)
const xq = sort(rand(N_QUERY)) .* 0.98 .+ 0.01  # inside domain
const y_obs = cos.(xq) .+ 0.05 .* randn(N_QUERY)

# =============================================================================
# Gradient Implementations
# =============================================================================

# --- Loss function (shared by ForwardDiff / Zygote) ---
function loss(y)
    return sum(abs2, cubic_interp(x, y, xq) .- y_obs)
end

# --- Loss function for Enzyme (explicit args → static activity analysis) ---
# Enzyme's static activity analysis can't resolve captured constants mixed
# with active variables in broadcasts (e.g. `y .- y_obs` where y is active
# and y_obs is a captured constant). Passing y_obs as an explicit Const arg
# lets Enzyme resolve activity at compile time — no runtime overhead.
function loss_enz(y, y_obs, x, xq)
    return sum(abs2, cubic_interp(x, y, xq) .- y_obs)
end

# --- Native: allocating (forward + manual chain rule + adj) ---
const adj = cubic_adjoint(x, xq; bc = CubicFit())

function native_alloc_grad(f)
    y = cubic_interp(x, f, xq)
    ȳ = 2 .* (y .- y_obs)
    return adj(ȳ)
end

# --- Native: in-place (zero-alloc target) ---
function native_inplace_grad!(out, f, y_buf, ȳ_buf)
    cubic_interp!(y_buf, x, f, xq)
    @. ȳ_buf = 2 * (y_buf - y_obs)
    adj(out, ȳ_buf)
    return out
end

# =============================================================================
# Correctness Verification
# =============================================================================

function verify_correctness()
    println("="^70)
    println("  CORRECTNESS")
    println("  loss(f) = sum(|cubic_interp(x, f, xq) - y_obs|^2)")
    println("  Reference: ForwardDiff (full Dual propagation)")
    println("="^70)

    # ForwardDiff — ground truth
    g_fd = ForwardDiff.gradient(loss, f)

    # Native (allocating)
    g_native = native_alloc_grad(f)

    # Native (in-place)
    g_native_ip = zeros(N_GRID)
    native_inplace_grad!(g_native_ip, f, zeros(N_QUERY), zeros(N_QUERY))

    # Zygote
    g_zy = Zygote.gradient(loss, f)[1]

    # Enzyme
    g_enz = zeros(N_GRID)
    Enzyme.autodiff(
        Enzyme.Reverse, loss_enz, Active,
        Duplicated(f, g_enz), Const(y_obs), Const(x), Const(xq)
    )

    println()
    @printf("  %-22s  %12s  %12s  %s\n", "Backend", "max|err|", "rel err", "isapprox?")
    println("  " * "-"^58)

    all_ok = true
    for (name, g) in [
            ("Native (alloc)", g_native),
            ("Native (in-place)", g_native_ip),
            ("Zygote", g_zy),
            ("Enzyme", g_enz),
        ]
        abs_err = maximum(abs, g .- g_fd)
        rel_err = norm(g .- g_fd) / norm(g_fd)
        ok = isapprox(g, g_fd; rtol = sqrt(eps()))
        all_ok &= ok
        @printf("  %-22s  %12.2e  %12.2e  %s\n", name, abs_err, rel_err, ok ? "YES" : "NO")
    end
    println()

    if !all_ok
        @error "Correctness check FAILED — results diverge from ForwardDiff reference"
    end

    return all_ok
end

# =============================================================================
# Benchmark
# =============================================================================

function run_benchmark()
    println("="^70)
    println("  PERFORMANCE — full gradient pipeline")
    println("  N=$N_GRID grid points, Nq=$N_QUERY query points (size=$SIZE_KEY)")
    println("="^70)

    # Pre-allocate buffers
    g_buf = zeros(N_GRID)
    y_buf = zeros(N_QUERY)
    ȳ_buf = zeros(N_QUERY)

    # Warmup all paths
    native_alloc_grad(f)
    native_inplace_grad!(g_buf, f, y_buf, ȳ_buf)
    ForwardDiff.gradient(loss, f)
    Zygote.gradient(loss, f)
    g_buf .= 0
    Enzyme.autodiff(
        Enzyme.Reverse, loss_enz, Active,
        Duplicated(f, g_buf), Const(y_obs), Const(x), Const(xq)
    )

    # --- Benchmarks ---
    println()
    println("  Native (allocating):")
    b_nat = @benchmark native_alloc_grad($f)
    display(b_nat); println()

    println("  Native (in-place):")
    b_nat_ip = @benchmark native_inplace_grad!($g_buf, $f, $y_buf, $ȳ_buf)
    display(b_nat_ip); println()

    println("  ForwardDiff:")
    b_fd = @benchmark ForwardDiff.gradient($loss, $f)
    display(b_fd); println()

    println("  Zygote:")
    b_zy = @benchmark Zygote.gradient($loss, $f)
    display(b_zy); println()

    println("  Enzyme:")
    b_enz = @benchmark begin
        $g_buf .= 0
        Enzyme.autodiff(
            Enzyme.Reverse, loss_enz, Active,
            Duplicated($f, $g_buf), Const($y_obs), Const($x), Const($xq)
        )
    end
    display(b_enz); println()

    # --- Summary ---
    med_us(b) = round(median(b).time / 1.0e3; digits = 1)
    med_alloc(b) = Int(median(b).allocs)
    med_mem(b) = Int(round(median(b).memory))

    fd_t = median(b_fd).time

    println()
    println("="^70)
    println("  SUMMARY — ∇f of L2 loss (N=$N_GRID, Nq=$N_QUERY)")
    println("="^70)
    println()
    @printf(
        "  %-22s  %8s  %8s  %8s  %8s\n",
        "Backend", "time(us)", "speedup", "allocs", "mem(B)"
    )
    println("  " * "-"^56)

    for (name, b) in [
            ("Native (alloc)", b_nat),
            ("Native (in-place)", b_nat_ip),
            ("ForwardDiff", b_fd),
            ("Zygote", b_zy),
            ("Enzyme", b_enz),
        ]
        t = med_us(b)
        sp = round(fd_t / median(b).time; digits = 1)
        @printf(
            "  %-22s  %8.1f  %7.1fx  %8d  %8d\n",
            name, t, sp, med_alloc(b), med_mem(b)
        )
    end
    println()

    return nothing
end

# =============================================================================
# Main
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Adjoint benchmark" size = SIZE_KEY N = N_GRID Nq = N_QUERY

    ok = verify_correctness()
    ok && run_benchmark()
end
