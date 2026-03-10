#!/usr/bin/env julia
#
# 2D Cubic Spline Data-Adjoint Benchmark: Native API vs AD Backends
#
# Compares full ∂loss/∂data gradient computation for 2D cubic_interp across:
#   - Native API:     cubic_interp (forward) + cubic_adjoint (pullback)
#   - ForwardDiff:    Dual number propagation through tridiagonal solve
#   - Zygote:         rrule → CubicAdjointND (analytical pullback)
#   - Enzyme:         EnzymeRules → CubicAdjointND (analytical pullback)
#   - Enzyme simple:  Enzyme.gradient(set_runtime_activity(Reverse), loss, data)
#
# All backends compute the same L2 loss gradient:
#   loss(data) = sum(|cubic_interp(grids, data, queries) - y_obs|²)
#   ∇data = ∂loss/∂data
#
# Correctness is verified against ForwardDiff (ground truth).
#
# Usage:
#   julia --project benchmark/adjoint_nd_benchmark.jl [SIZE] [OPTIONS]
#
# SIZE options:
#   --small    Grid 20×20,   Queries 100    (quick smoke test)
#   --default  Grid 100×100, Queries 2500   [DEFAULT]
#   --large    Grid 200×200, Queries 10000
#
# OPTIONS:
#   --no-fd     Skip ForwardDiff entirely
#   --force-fd  Force ForwardDiff @benchmark even for large grids
#   --help      Show this message

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
    :small   => (20, 10),       # 20×20 grid, 10×10 = 100 queries
    :default => (100, 50),      # 100×100 grid, 50×50 = 2500 queries
    :large   => (200, 100),     # 200×200 grid, 100×100 = 10000 queries
)

# ForwardDiff full @benchmark only below this grid element count
# (above this, uses single @elapsed to avoid minutes-long compile + eval)
const FD_BENCH_THRESHOLD = 2500

function parse_args(args)
    if "--help" in args || "-h" in args
        println(
            """
            2D Cubic Spline Data-Adjoint Benchmark: Native API vs AD Backends

            Compares full ∂loss/∂data gradient pipeline:
              loss(data) = sum(|cubic_interp(grids, data, queries) - y_obs|²)

            Backends: Native (in-place/allocating), ForwardDiff, Zygote, Enzyme

            SIZE OPTIONS:
                --small    Grid 20×20,   Queries 100    (quick smoke test)
                --default  Grid 100×100, Queries 2500   [DEFAULT]
                --large    Grid 200×200, Queries 10000

            OPTIONS:
                --no-fd      Skip ForwardDiff entirely
                --force-fd   Force ForwardDiff @benchmark for large grids
                --help, -h   Show this message
            """
        )
        exit(0)
    end

    size_key = :default
    skip_fd = "--no-fd" in args
    force_fd = "--force-fd" in args

    for arg in args
        if startswith(arg, "--")
            key = Symbol(arg[3:end])
            haskey(SIZE_PRESETS, key) && (size_key = key)
        end
    end
    return size_key, skip_fd, force_fd
end

const (SIZE_KEY, SKIP_FD, FORCE_FD) = parse_args(ARGS)
const (NG, NQ_AX) = SIZE_PRESETS[SIZE_KEY]
const NQ = NQ_AX^2                 # total query points (SoA)
const N_DATA = NG^2                # total grid data points

# ForwardDiff strategy:
#   small grid  → full @benchmark
#   large grid  → single @elapsed (warmup + timed eval)
#   --no-fd     → skip entirely
#   --force-fd  → full @benchmark regardless of size
const FD_MODE = if SKIP_FD
    :skip
elseif FORCE_FD || N_DATA <= FD_BENCH_THRESHOLD
    :benchmark
else
    :single_eval
end

# =============================================================================
# Test Data
# =============================================================================

Random.seed!(42)

const xg = collect(range(0.0, 2π, NG))
const yg = collect(range(0.0, 2π, NG))
const grids = (xg, yg)

# f(x,y) = sin(x)cos(y) + noise
const data = [sin(x) * cos(y) for x in xg, y in yg] .+ 0.05 .* randn(NG, NG)

# Scattered query points inside domain (2% margin from boundaries)
const margin = 2π * 0.02
const span   = 2π * 0.96
const xq = sort(rand(NQ)) .* span .+ margin
const yq = sort(rand(NQ)) .* span .+ margin
const queries = (xq, yq)

const y_obs = sin.(xq) .* cos.(yq) .+ 0.05 .* randn(NQ)

# =============================================================================
# Gradient Implementations
# =============================================================================

# --- Loss function (Zygote / ForwardDiff / Enzyme.gradient) ---
function loss(d)
    return sum(abs2, cubic_interp(grids, d, queries) .- y_obs)
end

# --- Loss function for Enzyme.autodiff (explicit Const args) ---
# Enzyme's static activity analysis can't resolve captured constants mixed
# with active variables in broadcasts (e.g. `d .- y_obs`).
# Passing y_obs/grids/queries as explicit Const args lets Enzyme resolve
# activity at compile time — no runtime overhead.
function loss_enz(d, y_obs_arg, grids_arg, queries_arg)
    return sum(abs2, cubic_interp(grids_arg, d, queries_arg) .- y_obs_arg)
end

# --- Native: pre-built adjoint operator (amortized construction) ---
const adj_nd = cubic_adjoint(grids, queries; bc = CubicFit())

function native_alloc_grad(d)
    y = cubic_interp(grids, d, queries)
    ȳ = 2 .* (y .- y_obs)
    return adj_nd(ȳ)
end

# --- Native: in-place (zero-alloc target on steady state) ---
function native_inplace_grad!(out, d, y_buf, ȳ_buf)
    cubic_interp!(y_buf, grids, d, queries)
    @. ȳ_buf = 2 * (y_buf - y_obs)
    adj_nd(out, ȳ_buf)
    return out
end

# =============================================================================
# Correctness Verification
# =============================================================================

function verify_correctness()
    println("=" ^ 70)
    println("  CORRECTNESS")
    println("  loss(data) = sum(|cubic_interp(grids, data, queries) - y_obs|²)")
    println("  Grid: $(NG)×$(NG) = $N_DATA pts, Queries: $NQ pts")
    println("=" ^ 70)

    # --- Compute reference (ForwardDiff) ---
    # For large grids, use a reduced 20×20 problem for ForwardDiff reference
    if FD_MODE == :skip
        # Skip ForwardDiff entirely — verify only mutual consistency
        println()
        println("  ForwardDiff: SKIPPED (--no-fd), checking mutual consistency only:")
        g_nat_full = native_alloc_grad(data)
        g_zy_full  = Zygote.gradient(loss, data)[1]
        g_enz_full = Enzyme.gradient(set_runtime_activity(Reverse), loss, data)[1]

        @printf("  %-22s  %12s  %12s  %s\n", "Backend vs Native", "max|err|", "rel err", "isapprox?")
        println("  " * "-" ^ 58)
        all_ok = true
        for (name, g) in [("Zygote", g_zy_full), ("Enzyme", g_enz_full)]
            abs_err = maximum(abs, g .- g_nat_full)
            rel_err = norm(g .- g_nat_full) / max(norm(g_nat_full), eps())
            ok = isapprox(g, g_nat_full; rtol = sqrt(eps()))
            all_ok &= ok
            @printf("  %-22s  %12.2e  %12.2e  %s\n", name, abs_err, rel_err, ok ? "YES" : "NO")
        end
    elseif N_DATA > FD_BENCH_THRESHOLD
        print("  ForwardDiff reference on reduced 20×20 grid... ")
        _rng = Random.MersenneTwister(99)
        _xg = collect(range(0.0, 2π, 20))
        _yg = collect(range(0.0, 2π, 20))
        _g  = (_xg, _yg)
        _d  = [sin(x) * cos(y) for x in _xg, y in _yg] .+ 0.05 .* randn(_rng, 20, 20)
        _nq = min(NQ, 100)
        _xq = sort(rand(_rng, _nq)) .* span .+ margin
        _yq = sort(rand(_rng, _nq)) .* span .+ margin
        _q  = (_xq, _yq)
        _yo = sin.(_xq) .* cos.(_yq) .+ 0.05 .* randn(_rng, _nq)
        _loss(dd) = sum(abs2, cubic_interp(_g, dd, _q) .- _yo)

        g_ref_small = ForwardDiff.gradient(_loss, _d)
        println("done")

        # Verify reverse-mode backends on the SAME reduced problem
        _adj = cubic_adjoint(_g, _q; bc = CubicFit())
        _y = cubic_interp(_g, _d, _q)
        _ȳ = 2 .* (_y .- _yo)
        g_native_small = _adj(_ȳ)

        g_zy_small = Zygote.gradient(_loss, _d)[1]

        _loss_e(dd, yo, gg, qq) = sum(abs2, cubic_interp(gg, dd, qq) .- yo)
        g_enz_small = zeros(20, 20)
        Enzyme.autodiff(
            Enzyme.Reverse, _loss_e, Active,
            Duplicated(copy(_d), g_enz_small), Const(_yo), Const(_g), Const(_q)
        )

        println()
        @printf("  Reference: ForwardDiff (20×20 reduced grid)\n")
        @printf("  %-22s  %12s  %12s  %s\n", "Backend", "max|err|", "rel err", "isapprox?")
        println("  " * "-" ^ 58)

        all_ok = true
        for (name, g) in [
                ("Native (alloc)", g_native_small),
                ("Zygote", g_zy_small),
                ("Enzyme", g_enz_small),
            ]
            abs_err = maximum(abs, g .- g_ref_small)
            rel_err = norm(g .- g_ref_small) / max(norm(g_ref_small), eps())
            ok = isapprox(g, g_ref_small; rtol = sqrt(eps()))
            all_ok &= ok
            @printf("  %-22s  %12.2e  %12.2e  %s\n",
                    name, abs_err, rel_err, ok ? "YES" : "NO")
        end

        # Also verify mutual consistency on the full grid
        println()
        println("  Mutual consistency on full $(NG)×$(NG) grid:")
        g_nat_full = native_alloc_grad(data)
        g_zy_full  = Zygote.gradient(loss, data)[1]
        g_enz_full = zeros(NG, NG)
        Enzyme.autodiff(
            Enzyme.Reverse, loss_enz, Active,
            Duplicated(copy(data), g_enz_full), Const(y_obs), Const(grids), Const(queries)
        )

        @printf("  %-22s  %12s  %12s  %s\n", "Backend", "max|err|", "rel err", "isapprox?")
        println("  " * "-" ^ 58)
        for (name, g) in [("Zygote", g_zy_full), ("Enzyme", g_enz_full)]
            abs_err = maximum(abs, g .- g_nat_full)
            rel_err = norm(g .- g_nat_full) / max(norm(g_nat_full), eps())
            ok = isapprox(g, g_nat_full; rtol = sqrt(eps()))
            all_ok &= ok
            @printf("  %-22s  %12.2e  %12.2e  %s\n",
                    name, abs_err, rel_err, ok ? "YES" : "NO")
        end
    else
        # Small grid: ForwardDiff on full problem
        print("  Computing ForwardDiff reference ($(NG)×$(NG))... ")
        g_ref = ForwardDiff.gradient(loss, data)
        println("done")

        g_native    = native_alloc_grad(data)
        g_native_ip = zeros(NG, NG)
        native_inplace_grad!(g_native_ip, data, zeros(NQ), zeros(NQ))
        g_zy  = Zygote.gradient(loss, data)[1]
        g_enz = zeros(NG, NG)
        Enzyme.autodiff(
            Enzyme.Reverse, loss_enz, Active,
            Duplicated(copy(data), g_enz), Const(y_obs), Const(grids), Const(queries)
        )

        println()
        @printf("  Reference: ForwardDiff\n")
        @printf("  %-22s  %12s  %12s  %s\n", "Backend", "max|err|", "rel err", "isapprox?")
        println("  " * "-" ^ 58)

        all_ok = true
        for (name, g) in [
                ("Native (alloc)", g_native),
                ("Native (in-place)", g_native_ip),
                ("Zygote", g_zy),
                ("Enzyme", g_enz),
            ]
            abs_err = maximum(abs, g .- g_ref)
            rel_err = norm(g .- g_ref) / max(norm(g_ref), eps())
            ok = isapprox(g, g_ref; rtol = sqrt(eps()))
            all_ok &= ok
            @printf("  %-22s  %12.2e  %12.2e  %s\n",
                    name, abs_err, rel_err, ok ? "YES" : "NO")
        end
    end

    println()
    if !(all_ok)
        @error "Correctness check FAILED — results diverge"
    end
    return all_ok
end

# =============================================================================
# Benchmark
# =============================================================================

function run_benchmark()
    println("=" ^ 70)
    println("  PERFORMANCE — full ∇data pipeline")
    println("  Grid: $(NG)×$(NG) = $N_DATA, Queries: $NQ (size=$SIZE_KEY)")
    println("=" ^ 70)

    # Pre-allocate buffers
    g_buf = zeros(NG, NG)
    y_buf = zeros(NQ)
    ȳ_buf = zeros(NQ)
    g_enz_buf = zeros(NG, NG)

    # ---- Warmup all paths ----
    native_alloc_grad(data)
    native_inplace_grad!(g_buf, data, y_buf, ȳ_buf)
    Zygote.gradient(loss, data)
    g_enz_buf .= 0
    Enzyme.autodiff(
        Enzyme.Reverse, loss_enz, Active,
        Duplicated(copy(data), g_enz_buf), Const(y_obs), Const(grids), Const(queries)
    )
    Enzyme.gradient(set_runtime_activity(Reverse), loss, data)
    if FD_MODE != :skip
        ForwardDiff.gradient(loss, data)
    end

    # ---- Benchmarks ----
    println()
    println("  Native (allocating):")
    b_nat = @benchmark native_alloc_grad($data)
    display(b_nat); println()

    println("  Native (in-place):")
    b_nat_ip = @benchmark native_inplace_grad!($g_buf, $data, $y_buf, $ȳ_buf)
    display(b_nat_ip); println()

    # ForwardDiff
    b_fd = nothing
    t_fd_single = nothing
    if FD_MODE == :benchmark
        println("  ForwardDiff:")
        b_fd = @benchmark ForwardDiff.gradient($loss, $data) seconds = 30
        display(b_fd); println()
    elseif FD_MODE == :single_eval
        println("  ForwardDiff (single eval — grid too large for full @benchmark):")
        n_chunks = cld(N_DATA, ForwardDiff.DEFAULT_CHUNK_THRESHOLD)
        @printf("    %d parameters, chunk=%d → %d forward passes\n",
                N_DATA, ForwardDiff.DEFAULT_CHUNK_THRESHOLD, n_chunks)
        t_fd_single = @elapsed ForwardDiff.gradient(loss, data)
        @printf("    Time: %.3f s\n\n", t_fd_single)
    else
        println("  ForwardDiff: SKIPPED (--no-fd)")
        println()
    end

    println("  Zygote:")
    b_zy = @benchmark Zygote.gradient($loss, $data)
    display(b_zy); println()

    # Enzyme — explicit Const args (optimal, no runtime activity)
    println("  Enzyme (autodiff + Const):")
    b_enz = @benchmark begin
        $g_enz_buf .= 0
        Enzyme.autodiff(
            Enzyme.Reverse, loss_enz, Active,
            Duplicated($data, $g_enz_buf), Const($y_obs), Const($grids), Const($queries)
        )
    end
    display(b_enz); println()

    # Enzyme — simple API (Zygote-like, but needs set_runtime_activity for closures)
    println("  Enzyme (gradient, simple API):")
    b_enz_simple = @benchmark Enzyme.gradient(set_runtime_activity(Reverse), $loss, $data)
    display(b_enz_simple); println()

    # ---- Summary Table ----
    med_ms(b)     = round(median(b).time / 1.0e6; digits = 3)
    med_alloc(b)  = Int(median(b).allocs)
    med_mem_kb(b) = round(median(b).memory / 1024; digits = 1)

    println()
    println("=" ^ 70)
    println("  SUMMARY — ∇data of L2 loss")
    println("  Grid: $(NG)×$(NG) = $N_DATA, Queries: $NQ")

    # Determine reference time for speedup column
    if b_fd !== nothing
        ref_t = median(b_fd).time
        ref_name = "ForwardDiff"
    elseif t_fd_single !== nothing
        ref_t = t_fd_single * 1.0e9   # convert seconds → nanoseconds
        ref_name = "ForwardDiff (single)"
    else
        ref_t = median(b_nat).time
        ref_name = "Native (alloc)"
    end
    println("  Speedup relative to $ref_name")
    println("=" ^ 70)

    println()
    @printf("  %-28s  %12s  %8s  %8s  %10s\n",
            "Backend", "time(ms)", "speedup", "allocs", "mem(KB)")
    println("  " * "-" ^ 70)

    function print_row(name, b)
        t = med_ms(b)
        sp = round(ref_t / median(b).time; digits = 1)
        @printf("  %-28s  %12.3f  %7.1fx  %8d  %10.1f\n",
                name, t, sp, med_alloc(b), med_mem_kb(b))
    end

    function print_single_row(name, t_sec)
        sp = round(ref_t / (t_sec * 1e9); digits = 1)
        @printf("  %-28s  %12.3f  %7.1fx  %8s  %10s\n",
                name, t_sec * 1e3, sp, "N/A", "N/A")
    end

    print_row("Native (alloc)", b_nat)
    print_row("Native (in-place)", b_nat_ip)

    if b_fd !== nothing
        print_row("ForwardDiff", b_fd)
    elseif t_fd_single !== nothing
        print_single_row("ForwardDiff*", t_fd_single)
    end

    print_row("Zygote", b_zy)
    print_row("Enzyme (autodiff+Const)", b_enz)
    print_row("Enzyme (gradient,simple)", b_enz_simple)

    println()
    if t_fd_single !== nothing
        println("  * single @elapsed (not full @benchmark).")
        println("    Use --force-fd for full @benchmark, --small for smaller grid.")
    end
    println()
    println("  Note: Native API pre-builds CubicAdjointND (amortized construction).")
    println("  Zygote/Enzyme rebuild the adjoint operator each gradient call.")
    println("  Enzyme (gradient,simple) uses set_runtime_activity due to closure.")
    println()

    return nothing
end

# =============================================================================
# Main
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    fd_str = FD_MODE == :benchmark ? "benchmark" :
             FD_MODE == :single_eval ? "single eval" : "skipped"
    @info "2D Adjoint Benchmark" size = SIZE_KEY grid = "$(NG)×$(NG)" queries = NQ ForwardDiff = fd_str

    ok = verify_correctness()
    ok && run_benchmark()
end
