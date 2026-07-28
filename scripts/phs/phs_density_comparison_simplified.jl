# ============================================================
# PHS Density Analysis — Simplified Profiling & Benchmark Script
# ============================================================
#
# A streamlined version of phs_density_comparison.jl focused solely on:
#   1. Building and evaluating the PHS interpolant (and only the PHS).
#   2. Validating that computed PHS Laplacian values match the reference.
#   3. Optionally plotting the PHS vs. Analytical Laplacian magnitude
#      along with its relative error (highly customizable, skip to speed up).
#   4. Enabling easy profiling and benchmarking of the PHS evaluation
#      hot-path using high-repetition timing loops.
#
# Run this script with:
#   julia --project=scripts scripts/phs/phs_density_comparison_simplified.jl
# Or for tracking allocations:
#   julia --project=scripts --track-allocation=user scripts/phs/phs_density_comparison_simplified.jl
# Or with standard Julia profiling:
#   julia --project=scripts -e 'using Profile; include("scripts/phs/phs_density_comparison_simplified.jl")'
#

# ============================================================
# Configuration Options
# ============================================================
const PLOT = false            # Enable/disable generating the sanity-check plot
const BENCHMARK = true        # Enable/disable the high-repetition benchmark loops
const BENCHMARK_REPS = 5     # Number of evaluation repetitions for profiling (e.g. 1000 reps)
const PROFILE = true          # Enable/disable the CPU profiling run

# ============================================================
# Dependencies
# ============================================================
using FastInterpolations
using DelimitedFiles
using LinearAlgebra
using Pickle
using Printf
using Statistics
using Profile

if PLOT
    using Plots
end

# ============================================================
# Patch Pickle.jl to support numpy._core (NumPy >= 2.0)
# ============================================================
Pickle.np_methods!(mt) = begin
    mt["numpy.core.multiarray._reconstruct"] = Pickle.np_multiarray_reconstruct
    mt["numpy._core.multiarray._reconstruct"] = Pickle.np_multiarray_reconstruct
    mt["numpy.dtype"] = Pickle.np_dtype
    mt["numpy.core.multiarray.scalar"] = Pickle.np_scalar
    mt["numpy._core.multiarray.scalar"] = Pickle.np_scalar
    mt["__build__.Pickle.NpyDtype"] = Pickle.build_npydtype
    mt["__build__.Pickle.NpyArrayPlaceholder"] = Pickle.build_nparray
    return mt
end

# ============================================================
# Configuration — paths resolved relative to this script (@__DIR__)
# ============================================================
# wfc/ auto-downloads from critic2 (ensure_wfc_files); .pkl grid and .csv line cut
# are committed under dat/ (no public download source).
const PKL_PATH = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl")
const CSV_PATH = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_line_O7_H21_N1000.csv")
const XYZ_PATH = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_atoms.xyz")
const WFC_DIR = joinpath(@__DIR__, "dat", "wfc")
const OUT_PATH = normpath(joinpath(@__DIR__, "..", "..", "docs", "images", "phs_laplacian_comparison_simplified.png"))

const BOHR2ANG = 0.529177210903   # 1 Bohr → Angstrom

# ── Element symbol → atomic number (full periodic table) ──────────────────────
const ELEMENT_Z = Dict(
    "H" => 1, "He" => 2, "Li" => 3, "Be" => 4, "B" => 5, "C" => 6, "N" => 7, "O" => 8,
    "F" => 9, "Ne" => 10, "Na" => 11, "Mg" => 12, "Al" => 13, "Si" => 14, "P" => 15, "S" => 16,
    "Cl" => 17, "Ar" => 18, "K" => 19, "Ca" => 20, "Sc" => 21, "Ti" => 22, "V" => 23, "Cr" => 24,
    "Mn" => 25, "Fe" => 26, "Co" => 27, "Ni" => 28, "Cu" => 29, "Zn" => 30, "Ga" => 31, "Ge" => 32,
    "As" => 33, "Se" => 34, "Br" => 35, "Kr" => 36, "Rb" => 37, "Sr" => 38, "Y" => 39, "Zr" => 40,
    "Nb" => 41, "Mo" => 42, "Tc" => 43, "Ru" => 44, "Rh" => 45, "Pd" => 46, "Ag" => 47, "Cd" => 48,
    "In" => 49, "Sn" => 50, "Sb" => 51, "Te" => 52, "I" => 53, "Xe" => 54, "Cs" => 55, "Ba" => 56,
    "La" => 57, "Ce" => 58, "Pr" => 59, "Nd" => 60, "Pm" => 61, "Sm" => 62, "Eu" => 63, "Gd" => 64,
    "Tb" => 65, "Dy" => 66, "Ho" => 67, "Er" => 68, "Tm" => 69, "Yb" => 70, "Lu" => 71, "Hf" => 72,
    "Ta" => 73, "W" => 74, "Re" => 75, "Os" => 76, "Ir" => 77, "Pt" => 78, "Au" => 79, "Hg" => 80,
    "Tl" => 81, "Pb" => 82, "Bi" => 83, "Po" => 84, "At" => 85, "Rn" => 86, "Fr" => 87, "Ra" => 88,
    "Ac" => 89, "Th" => 90, "Pa" => 91, "U" => 92, "Np" => 93, "Pu" => 94, "Am" => 95, "Cm" => 96,
    "Bk" => 97, "Cf" => 98, "Es" => 99, "Fm" => 100, "Md" => 101, "No" => 102, "Lr" => 103,
    "Rf" => 104, "Db" => 105, "Sg" => 106, "Bh" => 107, "Hs" => 108, "Mt" => 109, "Ds" => 110,
    "Rg" => 111, "Cn" => 112, "Nh" => 113, "Fl" => 114, "Mc" => 115, "Lv" => 116, "Ts" => 117, "Og" => 118,
)

# ============================================================
# Auto-download PBE wavefunction files if missing
# ============================================================
function ensure_wfc_files()
    mkpath(WFC_DIR)

    all_symbols = [rpad(lowercase(sym), 2, '_') for sym in keys(ELEMENT_Z) if any(lowercase(sym) == s for s in ("h", "c", "o"))]
    existing = filter(f -> endswith(f, ".wfc"), readdir(WFC_DIR))
    if length(existing) >= length(all_symbols)
        println("✓ All wavefunction files already present")
        return
    end

    println("Downloading PBE wavefunction files from critic2 (GitHub)...")
    base_url = "https://raw.githubusercontent.com/aoterodelaroza/critic2/master/dat/wfc"

    download_count = 0
    for sym in all_symbols
        fname = sym * "_pbe.wfc"
        fpath = joinpath(WFC_DIR, fname)
        if isfile(fpath) && filesize(fpath) > 1000
            continue
        end
        url = "$base_url/$fname"
        try
            run(`curl -s -o $fpath $url`)
            if isfile(fpath) && filesize(fpath) > 1000
                download_count += 1
                print(".")
                if download_count % 30 == 0
                    println("")
                end
            else
                isfile(fpath) && rm(fpath)
            end
        catch e
            isfile(fpath) && rm(fpath)
        end
    end
    return println("\n  Downloaded $download_count new wavefunction files")
end

ensure_wfc_files()

# ============================================================
# 1. Load 3D grid and 1D analytical path
# ============================================================
println("\nLoading files...")
pkl = Pickle.npyload(PKL_PATH)
x_grid = Float64.(pkl["x"])
y_grid = Float64.(pkl["y"])
z_grid = Float64.(pkl["z"])
rho_3d = Float64.(pkl["variables"]["density_scf"])

@printf "  Grid: %d×%d×%d, ρ ∈ [%.2e, %.2e] a.u.\n" length(x_grid) length(y_grid) length(z_grid) minimum(rho_3d) maximum(rho_3d)

raw = readdlm(CSV_PATH, ',', skipstart = 1)
qx = Float64.(raw[:, 2])               # x_bohr
qy = Float64.(raw[:, 3])               # y_bohr
qz = Float64.(raw[:, 4])               # z_bohr
s_ang = Float64.(raw[:, 5]) .* BOHR2ANG   # arclength in Å
ρ_ref = Float64.(raw[:, 6])               # density_scf [a.u.]
∇²ρ_ref = abs.(Float64.(raw[:, 9]))         # |laplacian_scf| [a.u./Bohr²]
N_path = length(qx)

@printf "  Path: %d points, s ∈ [%.4f, %.4f] Å\n" N_path s_ang[1] s_ang[end]

# ============================================================
# 2. Promolecular reference density (PromolecularRef)
# ============================================================
wfc_filename(sym::String) = rpad(lowercase(sym), 2, '_') * "_pbe.wfc"

function parse_wfc(filepath::String)
    return open(filepath) do io
        norb = parse(Int, readline(io))
        readline(io)
        occ = parse.(Float64, split(readline(io)))
        readline(io)
        ngrid = parse(Int, readline(io))

        r_vals = Vector{Float64}(undef, ngrid)
        rho_vals = Vector{Float64}(undef, ngrid)
        pi4 = 4π

        for i in 1:ngrid
            row = parse.(Float64, split(readline(io)))
            r = row[1]
            psi = @view row[2:end]
            rr0 = dot(occ, psi .^ 2)
            r_vals[i] = r
            rho_vals[i] = rr0 / (pi4 * r^2)
        end
        return r_vals, rho_vals
    end
end

const _wfc_cache = Dict{Int, Any}()

function get_rho_itp(Z::Int)
    haskey(_wfc_cache, Z) && return _wfc_cache[Z]
    sym = findfirst(==(Z), ELEMENT_Z)
    sym === nothing && error("Unknown atomic number Z=$Z")
    fname = joinpath(WFC_DIR, wfc_filename(sym))
    isfile(fname) || error("wfc file not found: $fname")
    r_grid, rho_vals = parse_wfc(fname)
    itp = cubic_interp(r_grid, rho_vals; extrap = FillExtrap(0.0))
    _wfc_cache[Z] = itp
    return itp
end

const ANG2BOHR = 1.0 / BOHR2ANG

function load_xyz(filepath::String)
    lines = readlines(filepath)
    n = parse(Int, strip(lines[1]))
    atoms = Vector{Tuple{Int, NTuple{3, Float64}}}(undef, n)
    for i in 1:n
        parts = split(strip(lines[i + 2]))
        sym = String(parts[1])
        Z = ELEMENT_Z[sym]
        x, y, z = parse(Float64, parts[2]) * ANG2BOHR,
            parse(Float64, parts[3]) * ANG2BOHR,
            parse(Float64, parts[4]) * ANG2BOHR
        atoms[i] = (Z, (x, y, z))
    end
    return atoms
end

const ATOMS = load_xyz(XYZ_PATH)

struct PromolecularRef{I}
    atoms::Vector{Tuple{Int, NTuple{3, Float64}}}
    cache_array::Vector{I}
    cache::Dict{Int, I}
end

function PromolecularRef(atoms::Vector{Tuple{Int, NTuple{3, Float64}}})
    for (Z, _) in atoms
        get_rho_itp(Z)
    end
    I = typeof(first(values(_wfc_cache)))
    cache = Dict{Int, I}(k => v for (k, v) in _wfc_cache)

    max_z = maximum(Z for (Z, _) in atoms)
    cache_array = Vector{I}(undef, max_z)
    for (Z, itp) in cache
        if Z <= max_z
            cache_array[Z] = itp
        end
    end
    return PromolecularRef{I}(atoms, cache_array, cache)
end

@inline function _pmr_get_deriv_info(::Type{O}) where {O <: Tuple}
    orders = (deriv_order(fieldtype(O, 1)), deriv_order(fieldtype(O, 2)), deriv_order(fieldtype(O, 3)))
    total = sum(orders)
    if total == 1
        ax = findfirst(o -> o == 1, orders)::Int
        return (total, ax, 0)
    elseif total == 2
        ax1 = findfirst(o -> o > 0, orders)::Int
        ax2 = ax1 < 3 ? findnext(o -> o > 0, orders, ax1 + 1) : nothing
        ax2 = ax2 !== nothing ? ax2 : ax1
        return (total, ax1, ax2)
    else
        return (total, 0, 0)
    end
end

@inline function _pmr_eval_val_internal(pmr::PromolecularRef, q::NTuple{3, <:Real})
    # Value evaluation
    f = 0.0
    @inbounds for i in 1:length(pmr.atoms)
        Z, R = pmr.atoms[i]
        xx1 = q[1] - R[1]
        xx2 = q[2] - R[2]
        xx3 = q[3] - R[3]
        r = sqrt(xx1 * xx1 + xx2 * xx2 + xx3 * xx3)
        r < 1.0e-14 && continue
        f += max(pmr.cache_array[Z](r), 0.0)
    end
    return f
end

@inline function (pmr::PromolecularRef)(q::NTuple{3, <:Real})
    return _pmr_eval_val_internal(pmr, q)
end

@inline function (pmr::PromolecularRef)(q::NTuple{3, <:Real}, ops::O) where {O}
    info = _pmr_get_deriv_info(O)
    total = info[1]
    if total == 0
        return _pmr_eval_val_internal(pmr, q)
    elseif total == 1
        ax = info[2]
        fp = 0.0
        D1 = DerivOp{1}()
        @inbounds for i in 1:length(pmr.atoms)
            Z, R = pmr.atoms[i]
            xx1 = q[1] - R[1]
            xx2 = q[2] - R[2]
            xx3 = q[3] - R[3]
            r = sqrt(xx1 * xx1 + xx2 * xx2 + xx3 * xx3)
            r < 1.0e-14 && continue
            dx = ax == 1 ? xx1 : (ax == 2 ? xx2 : xx3)
            fp += pmr.cache_array[Z](r; deriv = D1) * dx / r
        end
        return fp
    elseif total == 2
        ax1 = info[2]
        ax2 = info[3]
        fpp = 0.0
        D1 = DerivOp{1}()
        D2 = DerivOp{2}()
        @inbounds for i in 1:length(pmr.atoms)
            Z, R = pmr.atoms[i]
            xx1 = q[1] - R[1]
            xx2 = q[2] - R[2]
            xx3 = q[3] - R[3]
            r = sqrt(xx1 * xx1 + xx2 * xx2 + xx3 * xx3)
            r < 1.0e-14 && continue
            rho_itp = pmr.cache_array[Z]
            rhop = rho_itp(r; deriv = D1)
            rhopp = rho_itp(r; deriv = D2)
            rfac = (rhopp - rhop / r) / (r * r)
            dx1 = ax1 == 1 ? xx1 : (ax1 == 2 ? xx2 : xx3)
            dx2 = ax2 == 1 ? xx1 : (ax2 == 2 ? xx2 : xx3)
            fpp += ax1 == ax2 ? rhop / r + rfac * (dx1 * dx1) : rfac * dx1 * dx2
        end
        return fpp
    else
        return 0.0
    end
end

@inline function (pmr::PromolecularRef)(q::NTuple{3, <:Real}; deriv = nothing)
    if deriv === nothing
        return _pmr_eval_val_internal(pmr, q)
    else
        return pmr(q, deriv)
    end
end

# In-place batch evaluation mirroring the PHS batch API
function (pmr::PromolecularRef)(
        out::AbstractVector{T},
        queries::Union{Tuple{Vararg{AbstractVector, 3}}, AbstractVector};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, 3}}} = EvalValue(),
    ) where {T}
    N = 3
    ops = FastInterpolations._resolve_deriv_nd(deriv, Val(N))
    nq = FastInterpolations._query_length(queries)
    length(out) == nq || throw(DimensionMismatch("Query and output sizes mismatch"))

    @inbounds for k in 1:nq
        q = FastInterpolations._extract_query_point(queries, k, Val(N))
        out[k] = pmr(q; deriv = ops)
    end
    return out
end

println("Building PromolecularRef reference interpolant...")
const ref_rho0 = PromolecularRef(ATOMS)

# ============================================================
# 3. Build only the PHS Interpolant
# ============================================================
grids = (x_grid, y_grid, z_grid)
println("\nBuilding Polyharmonic spline (PHS-3, stencil_size=8, log-density transform)...")
time_phs = @elapsed itp_phs = phs_interp(
    grids, rho_3d; stencil_size = 8, degree = 3, blend_factor = 2.0,
    reference_interp = ref_rho0
)
@printf "  Built in %.4f seconds\n" time_phs

# ============================================================
# 4. Correctness Sanity Check
# ============================================================
println("\nRunning Correctness Sanity Check...")

# Allocate result arrays for PHS evaluation
ρ_phs = zeros(N_path)
_gx = zeros(N_path)
_gy = zeros(N_path)
_gz = zeros(N_path)
∇²ρ_phs = zeros(N_path)

# Query coordinates SoA format
const queries = (qx, qy, qz)

# Derivative operators
const D0 = DerivOp{0}()
const D1 = DerivOp{1}()
const D2 = DerivOp{2}()

# Warm up PHS compilation & stencil caches
print("  Warming up PHS evaluation ... ")
flush(stdout)
itp_phs(ρ_phs, queries)
itp_phs(_gx, queries; deriv = (D1, D0, D0))
itp_phs(_gy, queries; deriv = (D0, D1, D0))
itp_phs(_gz, queries; deriv = (D0, D0, D1))
itp_phs(_gx, queries; deriv = (D2, D0, D0))
itp_phs(_gy, queries; deriv = (D0, D2, D0))
itp_phs(_gz, queries; deriv = (D0, D0, D2))
println("done.")

# Evaluate PHS Laplacian Magnitude
itp_phs(_gx, queries; deriv = (D2, D0, D0))
itp_phs(_gy, queries; deriv = (D0, D2, D0))
itp_phs(_gz, queries; deriv = (D0, D0, D2))
@. ∇²ρ_phs = abs(_gx + _gy + _gz)

# Helper function to compute relative error
compute_rel_error(computed, reference) = begin
    errors = similar(computed)
    for i in eachindex(computed)
        ref_val = abs(reference[i])
        if ref_val > 0.0
            errors[i] = abs(computed[i] - reference[i]) / ref_val
        else
            errors[i] = abs(computed[i])
        end
    end
    return errors
end

errors_lap = compute_rel_error(∇²ρ_phs, ∇²ρ_ref)

# Compute error statistics
phs_lap_stats = Dict(
    :min => minimum(errors_lap),
    :max => maximum(errors_lap),
    :mean => mean(errors_lap),
    :median => median(errors_lap)
)

println("\n" * "="^50)
println("PHS LAPLACIAN ERROR STATISTICS VS ANALYTICAL")
println("="^50)
@printf "  Min Error:    %.2e\n" phs_lap_stats[:min]
@printf "  Max Error:    %.2e\n" phs_lap_stats[:max]
@printf "  Mean Error:   %.2e\n" phs_lap_stats[:mean]
@printf "  Median Error: %.2e\n" phs_lap_stats[:median]
println("="^50)

# ============================================================
# 5. Hot-Path Benchmarking / Profiling
# ============================================================
if BENCHMARK
    println("\n" * "="^60)
    println("BENCHMARKING PHS INTERPOLANT EVALUATION")
    println("="^60)
    @printf "Running %d repetitions over %d query points...\n\n" BENCHMARK_REPS N_path

    # Standard Julia @time macro to capture allocations & GC overhead
    println("PHS Density (ρ) Evaluation:")
    @time itp_phs(ρ_phs, queries)

    println("\nPHS Laplacian Component xx Evaluation:")
    @time itp_phs(_gx, queries; deriv = (D2, D0, D0))

    println("\nBenchmarking over high loop counts to run profilers:")

    # 1. Benchmark value evaluation (ρ)
    print("  Evaluating values (ρ) ... ")
    flush(stdout)
    t_val = @elapsed begin
        for _ in 1:BENCHMARK_REPS
            itp_phs(ρ_phs, queries)
        end
    end
    @printf "%.4f seconds (%.2f μs per query point)\n" t_val (t_val * 1.0e6 / (BENCHMARK_REPS * N_path))

    # 2. Benchmark Laplacian evaluation (|∇²ρ|)
    print("  Evaluating Laplacian (|∇²ρ| components) ... ")
    flush(stdout)
    t_lap = @elapsed begin
        for _ in 1:BENCHMARK_REPS
            itp_phs(_gx, queries; deriv = (D2, D0, D0))
            itp_phs(_gy, queries; deriv = (D0, D2, D0))
            itp_phs(_gz, queries; deriv = (D0, D0, D2))
            @. ∇²ρ_phs = abs(_gx + _gy + _gz)
        end
    end
    @printf "%.4f seconds (%.2f μs per query point)\n" t_lap (t_lap * 1.0e6 / (BENCHMARK_REPS * N_path))

    if PROFILE
        # Increase profile buffer for more complete data collection
        Profile.init(n = 50_000_000, delay = 0.001)

        # 1. Profile Density (ρ) Evaluation
        println("\nRunning CPU Profiling for Density (ρ) Evaluation (100 repetitions)...")
        Profile.clear()
        @profile for _ in 1:100
            itp_phs(ρ_phs, queries)
        end
        println("\n" * "="^60)
        println("CPU PROFILING RESULTS: DENSITY (ρ) EVALUATION")
        println("="^60)
        Profile.print(format = :flat, mincount = 5, noisefloor = 2.0, groupby = [:task, :thread], maxdepth = 40)

        # 2. Profile Gradient (|∇ρ|) Evaluation
        println("\nRunning CPU Profiling for Gradient (|∇ρ|) Evaluation (100 repetitions)...")
        Profile.clear()
        @profile for _ in 1:100
            itp_phs(_gx, queries; deriv = (D1, D0, D0))
            itp_phs(_gy, queries; deriv = (D0, D1, D0))
            itp_phs(_gz, queries; deriv = (D0, D0, D1))
        end
        println("\n" * "="^60)
        println("CPU PROFILING RESULTS: GRADIENT (|∇ρ|) EVALUATION")
        println("="^60)
        Profile.print(format = :flat, mincount = 5, noisefloor = 2.0, groupby = [:task, :thread], maxdepth = 40)

        # 3. Profile Laplacian (|∇²ρ|) Evaluation
        println("\nRunning CPU Profiling for Laplacian (|∇²ρ|) Evaluation (100 repetitions)...")
        Profile.clear()
        @profile for _ in 1:100
            itp_phs(_gx, queries; deriv = (D2, D0, D0))
            itp_phs(_gy, queries; deriv = (D0, D2, D0))
            itp_phs(_gz, queries; deriv = (D0, D0, D2))
            @. ∇²ρ_phs = abs(_gx + _gy + _gz)
        end
        println("\n" * "="^60)
        println("CPU PROFILING RESULTS: LAPLACIAN (|∇²ρ|) EVALUATION")
        println("="^60)
        Profile.print(format = :flat, mincount = 5, noisefloor = 2.0, groupby = [:task, :thread], maxdepth = 40)
    end

    println("="^60)
    println("Profiling code can be run via: ")
    println("  julia --project=scripts --track-allocation=user scripts/phs/phs_density_comparison_simplified.jl")
end

# ============================================================
# 6. Plotting (Optional Sanity Check)
# ============================================================
if PLOT
    println("\nGenerating sanity-check plots...")

    # Replace non-positive values with NaN for safe log-scale rendering
    logclean(v) = [x > 0.0 ? x : NaN for x in v]

    # Style definitions matching premium aesthetics
    col_analytical = RGB(0.1, 0.1, 0.1)      # Sleek Charcoal Black
    col_phs = RGB(0.85, 0.15, 0.15)   # Vibrant Crimson Red
    col_error = RGB(0.0, 0.45, 0.7)    # Deep Ocean Blue

    yformatter = (y) -> begin
        if y > 10000
            @sprintf("%.1e", y)
        else
            s = @sprintf("%.6f", y)
            s = replace(s, r"0+$" => "")
            replace(s, r"\.$" => "")
        end
    end

    # Create the left plot (Laplacian Magnitude)
    p1 = plot(
        xaxis = :identity,
        yaxis = :log10,
        xlims = (s_ang[1], s_ang[end]),
        xlabel = "Distance along O···H hydrogen bond (Å)",
        ylabel = "|∇²ρ| (a.u.)",
        xticks = 0:0.2:2.0,
        yticks = [10.0^i for i in -4:6],
        yformatter = yformatter,
        legend = :topright,
        minorgrid = true,
        framestyle = :box,
        title = "Laplacian Magnitude Comparison",
        titlefontsize = 10,
        guidefontsize = 9,
        tickfontsize = 8,
        legendfontsize = 8,
    )
    plot!(p1, s_ang, logclean(∇²ρ_ref); label = "Analytical", color = col_analytical, linewidth = 2.0, linestyle = :solid)
    plot!(p1, s_ang, logclean(∇²ρ_phs); label = "Polyharmonic", color = col_phs, linewidth = 1.0, linestyle = :dash)

    # Create the right plot (PHS Laplacian Relative Error)
    p2 = plot(
        xaxis = :identity,
        yaxis = :log10,
        xlims = (s_ang[1], s_ang[end]),
        xlabel = "Distance along O···H hydrogen bond (Å)",
        ylabel = "Relative Error",
        xticks = 0:0.2:2.0,
        yticks = [10.0^i for i in -7:1],
        yformatter = yformatter,
        legend = false,
        minorgrid = true,
        framestyle = :box,
        title = "PHS Laplacian Relative Error",
        titlefontsize = 10,
        guidefontsize = 9,
        tickfontsize = 8,
    )
    plot!(p2, s_ang, logclean(errors_lap); color = col_error, linewidth = 1.2, linestyle = :solid)

    # Combine into a gorgeous premium layout
    fig = plot(
        p1, p2;
        layout = (1, 2),
        size = (1000, 450),
        dpi = 150,
        left_margin = 6Plots.mm,
        bottom_margin = 6Plots.mm,
        top_margin = 6Plots.mm,
        right_margin = 4Plots.mm,
    )

    savefig(fig, OUT_PATH)
    println("Saved sanity-check plot: $OUT_PATH")
end

println("\nFinished successfully!")
