# ============================================================
# PHS Charge Density Analysis — Comparison Script
# ============================================================
#
# Reproduces the 3×2 log-scale comparison plot comparing analytical DFT
# (reference) values against 3D interpolants for:
#   Row 1:  ρ        (charge density)
#   Row 2:  |∇ρ|     (gradient magnitude)
#   Row 3:  |∇²ρ|    (Laplacian magnitude)
# along the O7...H21 hydrogen-bond path in the phenol dimer.
#
# Left column  — all standard 3D methods vs DFT reference
# Right column — Polyharmonic spline (PHS) vs DFT reference
#
# For PHS, the reference density is the analytical promolecule density
# constructed from critic2 PBE wavefunction files, enabling log-density
# smoothing via the reference_interp interface.
#
# Data files (edit the paths below if needed):
#   3D grid : phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl
#             Python pickle: x (75,), y (113,), z (70,) [Bohr],
#             variables["density_scf"] (75,113,70) [a.u.] — DFT reference density
#   1D path : phenol-dimer_B3LYP_TZ2P_GO_line_O7_H21_N1000.csv
#             Columns: point, x_bohr, y_bohr, z_bohr, arclength_bohr,
#                      density_scf, density_frag, dengrad_mag, laplacian_scf
#
# Dependencies (add to your active Julia environment as needed):
#   Pkg.add("Plots")   — Plots.jl for visualisation
#   Pkg.add("Pickle")  — Pickle.jl for loading the .pkl grid file
#
# Implementation notes on PHS derivatives:
#   PHSInterpolantND does NOT implement _locate_cell/_eval_at_cell, so
#   gradient() / laplacian() from vector_calculus.jl are NOT available for it.
#   Instead use the `deriv` keyword on the callable:
#     itp_phs(q; deriv = (DerivOp{1}(), DerivOp{0}(), DerivOp{0}()))  # ∂ρ/∂x
#     itp_phs(q; deriv = (DerivOp{2}(), DerivOp{0}(), DerivOp{0}()))  # ∂²ρ/∂x²

using FastInterpolations
using DelimitedFiles
using LinearAlgebra
using Pickle
using Plots
using Printf
using Statistics

# ============================================================
# Patch Pickle.jl to support numpy._core (NumPy >= 2.0)
# ============================================================
Pickle.np_methods!(mt) = begin
    mt["numpy.core.multiarray._reconstruct"]  = Pickle.np_multiarray_reconstruct
    mt["numpy._core.multiarray._reconstruct"] = Pickle.np_multiarray_reconstruct
    mt["numpy.dtype"]                          = Pickle.np_dtype
    mt["numpy.core.multiarray.scalar"]         = Pickle.np_scalar
    mt["numpy._core.multiarray.scalar"]        = Pickle.np_scalar
    mt["__build__.Pickle.NpyDtype"]            = Pickle.build_npydtype
    mt["__build__.Pickle.NpyArrayPlaceholder"] = Pickle.build_nparray
    return mt
end

# ============================================================
# Configuration — edit paths here
# ============================================================
const PKL_PATH = "dat/phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl"
const CSV_PATH = "dat/phenol-dimer_B3LYP_TZ2P_GO_line_O7_H21_N1000.csv"
const XYZ_PATH = "dat/phenol-dimer_B3LYP_TZ2P_GO_atoms.xyz"
const WFC_DIR  = joinpath(@__DIR__, "dat", "wfc")
const OUT_PATH = "../docs/images/phs_density_comparison.png"

const BOHR2ANG = 0.529177210903   # 1 Bohr → Angstrom

# ── Element symbol → atomic number (full periodic table) ──────────────────────
const ELEMENT_Z = Dict(
    "H"=>1,  "He"=>2, "Li"=>3,  "Be"=>4,  "B"=>5,   "C"=>6,   "N"=>7,   "O"=>8,
    "F"=>9,  "Ne"=>10,"Na"=>11, "Mg"=>12, "Al"=>13, "Si"=>14, "P"=>15,  "S"=>16,
    "Cl"=>17,"Ar"=>18,"K"=>19,  "Ca"=>20, "Sc"=>21, "Ti"=>22, "V"=>23,  "Cr"=>24,
    "Mn"=>25,"Fe"=>26,"Co"=>27, "Ni"=>28, "Cu"=>29, "Zn"=>30, "Ga"=>31, "Ge"=>32,
    "As"=>33,"Se"=>34,"Br"=>35, "Kr"=>36, "Rb"=>37, "Sr"=>38, "Y"=>39,  "Zr"=>40,
    "Nb"=>41,"Mo"=>42,"Tc"=>43, "Ru"=>44, "Rh"=>45, "Pd"=>46, "Ag"=>47, "Cd"=>48,
    "In"=>49,"Sn"=>50,"Sb"=>51, "Te"=>52, "I"=>53,  "Xe"=>54, "Cs"=>55, "Ba"=>56,
    "La"=>57,"Ce"=>58,"Pr"=>59, "Nd"=>60, "Pm"=>61, "Sm"=>62, "Eu"=>63, "Gd"=>64,
    "Tb"=>65,"Dy"=>66,"Ho"=>67, "Er"=>68, "Tm"=>69, "Yb"=>70, "Lu"=>71, "Hf"=>72,
    "Ta"=>73,"W"=>74, "Re"=>75, "Os"=>76, "Ir"=>77, "Pt"=>78, "Au"=>79, "Hg"=>80,
    "Tl"=>81,"Pb"=>82,"Bi"=>83, "Po"=>84, "At"=>85, "Rn"=>86, "Fr"=>87, "Ra"=>88,
    "Ac"=>89,"Th"=>90,"Pa"=>91, "U"=>92,  "Np"=>93, "Pu"=>94, "Am"=>95, "Cm"=>96,
    "Bk"=>97,"Cf"=>98,"Es"=>99,"Fm"=>100,"Md"=>101,"No"=>102,"Lr"=>103,
    "Rf"=>104,"Db"=>105,"Sg"=>106,"Bh"=>107,"Hs"=>108,"Mt"=>109,"Ds"=>110,
    "Rg"=>111,"Cn"=>112,"Nh"=>113,"Fl"=>114,"Mc"=>115,"Lv"=>116,"Ts"=>117,"Og"=>118,
)

# ============================================================
# Auto-download PBE wavefunction files if missing
# ============================================================

"""
    ensure_wfc_files()

Download critic2 PBE wavefunction files if not already present.
Files are downloaded from GitHub and cached in `WFC_DIR`.
Uses regex to parse GitHub API response (simple approach, no JSON dependency).
"""
function ensure_wfc_files()
    mkpath(WFC_DIR)

    # List of all elements (Z=1 to Z=118) with their 2-letter symbols,
    # using the keys of ELEMENT_Z.  Single-letter symbols get an extra underscore for filename formatting.
    # Single letters get underscore padding (H→h_, C→c_, etc.)
    # We only need H, C, O for this system, but it's written in a way that's easy to extend if needed.
    all_symbols = [rpad(lowercase(sym), 2, '_') for sym in keys(ELEMENT_Z) if any(lowercase(sym) == s for s in ("h", "c", "o"))]
    
    # Check if files already exist
    existing = filter(f -> endswith(f, ".wfc"), readdir(WFC_DIR))
    existing_count = length(existing)
    
    if existing_count >= length(all_symbols)
        println("✓ All wavefunction files already present")
        return
    end
    
    println("Downloading PBE wavefunction files from critic2 (GitHub)...")
    
    # Try to download files using raw GitHub URLs with hardcoded filename list
    base_url = "https://raw.githubusercontent.com/aoterodelaroza/critic2/master/dat/wfc"
    
    download_count = 0
    for sym in all_symbols
        fname = sym * "_pbe.wfc"
        fpath = joinpath(WFC_DIR, fname)
        
        # Skip if already exists and is non-empty (>1KB suggests real data)
        if isfile(fpath) && filesize(fpath) > 1000
            continue
        end
        
        url = "$base_url/$fname"
        
        try
            run(`curl -s -o $fpath $url`)
            # Check if download was successful (file size > 1KB)
            if isfile(fpath) && filesize(fpath) > 1000
                download_count += 1
                print(".")
                if download_count % 30 == 0
                    println("")
                end
            else
                # Failed download, remove small file
                isfile(fpath) && rm(fpath)
            end
        catch e
            # Download error, clean up
            isfile(fpath) && rm(fpath)
        end
    end
    
    println("\n  Downloaded $download_count new wavefunction files")
    if download_count > 0
        println("  ✓ Wavefunction files ready")
    end
end

# Download wfc files if needed
ensure_wfc_files()

# ============================================================
# 1. Load 3D grid from pickle via Pickle.jl
# ============================================================
println("Loading 3D grid from pickle...")

pkl = Pickle.npyload(PKL_PATH)
x_grid = Float64.(pkl["x"])
y_grid = Float64.(pkl["y"])
z_grid = Float64.(pkl["z"])
rho_3d  = Float64.(pkl["variables"]["density_scf"])   # DFT reference density: (nx,ny,nz) via c2f

@printf "  Grid: %d×%d×%d, ρ ∈ [%.2e, %.2e] a.u.\n" length(x_grid) length(y_grid) length(z_grid) minimum(rho_3d) maximum(rho_3d)

# ============================================================
# 2. Load 1D analytical path from CSV
# ============================================================
println("Loading 1D analytical path from CSV...")

raw = readdlm(CSV_PATH, ',', skipstart = 1)
# column order: point, x_bohr, y_bohr, z_bohr, arclength_bohr,
#               density_scf, density_frag, dengrad_mag, laplacian_scf
qx      = Float64.(raw[:, 2])               # x_bohr
qy      = Float64.(raw[:, 3])               # y_bohr
qz      = Float64.(raw[:, 4])               # z_bohr
s_ang   = Float64.(raw[:, 5]) .* BOHR2ANG   # arclength in Å
ρ_ref   = Float64.(raw[:, 6])               # density_scf [a.u.]
∇ρ_ref  = Float64.(raw[:, 8])               # dengrad_mag [a.u./Bohr]
∇²ρ_ref = abs.(Float64.(raw[:, 9]))         # |laplacian_scf| [a.u./Bohr²]
N_path  = length(qx)

@printf "  Path: %d points, s ∈ [%.4f, %.4f] Å\n" N_path s_ang[1] s_ang[end]

# ============================================================
# 3. Promolecular reference density (PromolecularRef)
#    Replicates the Fortran crystalmod_promolecular approach:
#    ρ₀(x) = Σᵢ ρᵢ(|x - Rᵢ|)  using PBE all-electron atomic densities
#    from critic2 dat/wfc files, giving an accurate ρ₀ and exact
#    analytical derivatives everywhere — no cubic-spline Gibbs near cusps.
# ============================================================

# wfc filename: single-letter symbols get an extra trailing underscore
# e.g. "H" → "h__pbe.wfc", "He" → "he_pbe.wfc"
wfc_filename(sym::String) = rpad(lowercase(sym), 2, '_') * "_pbe.wfc"

"""
    parse_wfc(filepath) -> (r_grid, rho_values)

Parse a critic2 PBE all-electron wfc file and return the radial grid and
electron density ρ(r) = Σⱼ occⱼ ψⱼ(r)² / (4πr²).

File format:
  Line 1:  norb
  Line 2:  orbital labels
  Line 3:  occupations (integers)
  Line 4:  orbital energies
  Line 5:  ngrid
  Lines 6…: r  ψ₁(r)  ψ₂(r)  …  ψₙₒᵣb(r)
"""
function parse_wfc(filepath::String)
    open(filepath) do io
        norb = parse(Int, readline(io))
        readline(io)                              # labels — not needed
        occ  = parse.(Float64, split(readline(io)))
        readline(io)                              # energies — not needed
        ngrid = parse(Int, readline(io))

        r_vals   = Vector{Float64}(undef, ngrid)
        rho_vals = Vector{Float64}(undef, ngrid)
        pi4 = 4π

        for i in 1:ngrid
            row   = parse.(Float64, split(readline(io)))
            r     = row[1]
            psi   = @view row[2:end]
            rr0   = dot(occ, psi .^ 2)           # Σⱼ occⱼ ψⱼ²
            r_vals[i]   = r
            rho_vals[i] = rr0 / (pi4 * r^2)      # ρ(r)
        end
        return r_vals, rho_vals
    end
end

# Build a Dict: Z => 1D cubic spline of ρ(r) for each element present in system
# (lazy-loaded; only elements actually needed are parsed)
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

# ── XYZ loader — returns Vector of (Z, (x,y,z)) with positions in Bohr ────────
const ANG2BOHR = 1.0 / BOHR2ANG   # 1 Å → Bohr

"""
    load_xyz(filepath) -> Vector{Tuple{Int, NTuple{3,Float64}}}

Load an XYZ file (positions in Angstrom) and return atoms as (Z, (x,y,z)) in Bohr.
Skips the first two header lines.
"""
function load_xyz(filepath::String)
    lines = readlines(filepath)
    n = parse(Int, strip(lines[1]))
    atoms = Vector{Tuple{Int, NTuple{3,Float64}}}(undef, n)
    for i in 1:n
        parts = split(strip(lines[i + 2]))
        sym = String(parts[1])
        Z   = ELEMENT_Z[sym]
        x, y, z = parse(Float64, parts[2]) * ANG2BOHR,
                   parse(Float64, parts[3]) * ANG2BOHR,
                   parse(Float64, parts[4]) * ANG2BOHR
        atoms[i] = (Z, (x, y, z))
    end
    return atoms
end

println("Loading atomic geometry from XYZ...")
const ATOMS = load_xyz(XYZ_PATH)
@printf "  %d atoms loaded\n" length(ATOMS)

# ── Verify path endpoints are near atoms ──────────────────────────────────────
let
    p_start = (qx[1],   qy[1],   qz[1])
    p_end   = (qx[end], qy[end], qz[end])
    for (label, pt) in (("start", p_start), ("end", p_end))
        best_d = Inf; best_i = 0
        for (i, (Z, R)) in enumerate(ATOMS)
            d = sqrt(sum((pt[k] - R[k])^2 for k in 1:3))
            if d < best_d; best_d = d; best_i = i; end
        end
        Z_near, R_near = ATOMS[best_i]
        sym_near = findfirst(==(Z_near), ELEMENT_Z)
        @printf "  Path %-5s → nearest atom %2d (%s) at (%.4f, %.4f, %.4f) Bohr, dist = %.4f Bohr\n" label best_i sym_near R_near[1] R_near[2] R_near[3] best_d
    end
end

# ── PromolecularRef ────────────────────────────────────────────────────────────
# Callable with the same (q; deriv=nothing) interface as LogCubicRef.
# Uses the Fortran chain rule for exact gradient and Hessian:
#   ∂ρ₀/∂xd      = Σᵢ ρᵢ'(rᵢ) · xxd / rᵢ
#   ∂²ρ₀/∂xd²    = Σᵢ [ ρᵢ'(rᵢ)/rᵢ + (ρᵢ''(rᵢ) − ρᵢ'(rᵢ)/rᵢ) · xxd² / rᵢ² ]
#   ∂²ρ₀/∂xd1∂xd2 = Σᵢ [ (ρᵢ''(rᵢ) − ρᵢ'(rᵢ)/rᵢ) · xxd1 · xxd2 / rᵢ² ]
struct PromolecularRef
    atoms::Vector{Tuple{Int, NTuple{3,Float64}}}  # (Z, (x,y,z)) in Bohr
end

function (pmr::PromolecularRef)(q; deriv=nothing)
    # Determine derivative order along each axis
    total = deriv === nothing ? 0 : sum(deriv_order(op) for op in deriv)

    if total == 0
        f = 0.0
        for (Z, R) in pmr.atoms
            xx1 = q[1] - R[1]; xx2 = q[2] - R[2]; xx3 = q[3] - R[3]
            r = sqrt(xx1^2 + xx2^2 + xx3^2)
            r < 1e-14 && continue
            f += max(get_rho_itp(Z)(r), 0.0)
        end
        return f
    end

    if total == 1
        ax = findfirst(d -> deriv_order(deriv[d]) == 1, 1:3)::Int
        fp = 0.0
        for (Z, R) in pmr.atoms
            xx = (q[1] - R[1], q[2] - R[2], q[3] - R[3])
            r  = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            rhop = get_rho_itp(Z)(r; deriv = DerivOp(1))
            fp += rhop * xx[ax] / r
        end
        return fp
    end

    if total == 2
        nonzero = [d for d in 1:3 if deriv_order(deriv[d]) > 0]
        ax1 = nonzero[1]; ax2 = length(nonzero) >= 2 ? nonzero[2] : ax1
        fpp = 0.0
        for (Z, R) in pmr.atoms
            xx = (q[1] - R[1], q[2] - R[2], q[3] - R[3])
            r  = sqrt(xx[1]^2 + xx[2]^2 + xx[3]^2)
            r < 1e-14 && continue
            rho_itp = get_rho_itp(Z)
            rhop  = rho_itp(r; deriv = DerivOp(1))
            rhopp = rho_itp(r; deriv = DerivOp(2))
            rfac  = (rhopp - rhop / r) / r^2
            fpp += ax1 == ax2 ? rhop / r + rfac * xx[ax1]^2 :
                                 rfac * xx[ax1] * xx[ax2]
        end
        return fpp
    end

    return 0.0
end

println("Building PromolecularRef (loading wfc files for present elements)...")
# Pre-warm the wfc cache for all elements in the system
for (Z, _) in ATOMS; get_rho_itp(Z); end
const ref_rho0 = PromolecularRef(ATOMS)
@printf "  ρ₀ at path start: %.6e a.u.\n" ref_rho0((qx[1], qy[1], qz[1]))

# ============================================================
# 4. Build 3D interpolants — with timing capture
# ============================================================
grids = (x_grid, y_grid, z_grid)
println("\nBuilding interpolants on $(length(x_grid))×$(length(y_grid))×$(length(z_grid)) grid...")

# Storage for timing information
build_times = Dict{String, Float64}()
eval_times = Dict{String, Dict{String, Float64}}()

println("  [1/4] Nearest (constant)...")
time_nearest = @elapsed itp_nearest  = constant_interp(grids, rho_3d)
build_times["Nearest"] = time_nearest
@printf "    %.4f seconds\n" time_nearest

println("  [2/4] Trilinear (linear)...")
time_linear = @elapsed itp_linear   = linear_interp(grids, rho_3d)
build_times["Linear"] = time_linear
@printf "    %.4f seconds\n" time_linear

println("  [3/4] Trispline (global cubic spline)...")
time_cubic = @elapsed itp_cubic    = cubic_interp(grids, rho_3d)
build_times["Cubic"] = time_cubic
@printf "    %.4f seconds\n" time_cubic

println("  [4/4] Tricubic (Cardinal / Catmull-Rom)...")
time_cardinal = @elapsed itp_cardinal = interp(grids, rho_3d;
    method = (CardinalInterp(), CardinalInterp(), CardinalInterp()))
build_times["Cardinal"] = time_cardinal
@printf "    %.4f seconds\n" time_cardinal

println("  [PHS] Polyharmonic spline (PHS-3, stencil_size=8, log-density transform)...")
# Paper (Sec. III): N = 8³ = 512 stencil nodes; f = log(ρ_scf / ρ₀) is smooth
# across the whole grid.  PromolecularRef provides ρ₀ and exact derivatives
# from PBE all-electron atomic radial splines via the chain rule — matches the
# Fortran crystalmod_promolecular approach and avoids Gibbs-like errors from
# a cubic spline of log(ρ₀) near nuclear cusps.
time_phs = @elapsed itp_phs = phs_interp(grids, rho_3d; stencil_size = 8, degree = 3, blend_factor = 2.0,
    reference_interp = ref_rho0)
build_times["PHS"] = time_phs
@printf "    %.4f seconds\n" time_phs

println("All interpolants built.")

# ============================================================
# 5. Evaluate along the 1D path
# ============================================================
println("\nEvaluating along path ($N_path points)...")

# Allocate result arrays
ρ_nearest  = zeros(N_path);  ρ_linear   = zeros(N_path)
ρ_cubic    = zeros(N_path);  ρ_cardinal = zeros(N_path)
ρ_phs      = zeros(N_path)

∇ρ_linear   = zeros(N_path);  ∇ρ_cubic    = zeros(N_path)
∇ρ_cardinal = zeros(N_path);  ∇ρ_phs      = zeros(N_path)

∇²ρ_cubic    = zeros(N_path)
∇²ρ_cardinal = zeros(N_path)
∇²ρ_phs      = zeros(N_path)

# Derivative operator shortcuts:
#   DerivOp{n}() selects the n-th derivative along that axis (0 = value)
const D0 = DerivOp{0}()
const D1 = DerivOp{1}()
const D2 = DerivOp{2}()

# SoA query format — all ND interpolants accept (x_vec, y_vec, z_vec).
# PHSInterpolantND batch evaluation uses Threads.@threads internally.
const queries = (qx, qy, qz)

# Scratch buffers shared across all gradient / Laplacian batch calls
const _gx = zeros(N_path)
const _gy = zeros(N_path)
const _gz = zeros(N_path)

# ── density ρ ──────────────────────────────────────────────────────────────────
println("  Density (ρ):")
eval_times["ρ"] = Dict()

print("    Nearest ... ")
eval_times["ρ"]["Nearest"] = @elapsed @time itp_nearest(ρ_nearest,  queries)

print("    Linear ... ")
eval_times["ρ"]["Linear"] = @elapsed @time itp_linear(ρ_linear,    queries)

print("    Cubic ... ")
eval_times["ρ"]["Cubic"] = @elapsed @time itp_cubic(ρ_cubic,      queries)

print("    Cardinal ... ")
eval_times["ρ"]["Cardinal"] = @elapsed @time itp_cardinal(ρ_cardinal, queries)

print("    PHS ... ")
eval_times["ρ"]["PHS"] = @elapsed @time itp_phs(ρ_phs,          queries)

# ── gradient magnitude |∇ρ| ────────────────────────────────────────────────────
# All ND interpolants accept the batch form itp(out, queries; deriv=(...)).
# PHS does not implement _locate_cell/_eval_at_cell so gradient() is unavailable,
# but the same computation works via the deriv kwarg on the batch callable.
println("  Gradient Magnitude (|∇ρ|):")
eval_times["|∇ρ|"] = Dict()

print("    Linear ... ")
eval_times["|∇ρ|"]["Linear"] = @elapsed @time begin
    itp_linear(_gx, queries; deriv = (D1, D0, D0))
    itp_linear(_gy, queries; deriv = (D0, D1, D0))
    itp_linear(_gz, queries; deriv = (D0, D0, D1))
    @. ∇ρ_linear = sqrt(_gx^2 + _gy^2 + _gz^2)
end

print("    Cubic ... ")
eval_times["|∇ρ|"]["Cubic"] = @elapsed @time begin
    itp_cubic(_gx, queries; deriv = (D1, D0, D0))
    itp_cubic(_gy, queries; deriv = (D0, D1, D0))
    itp_cubic(_gz, queries; deriv = (D0, D0, D1))
    @. ∇ρ_cubic = sqrt(_gx^2 + _gy^2 + _gz^2)
end

print("    Cardinal ... ")
eval_times["|∇ρ|"]["Cardinal"] = @elapsed @time begin
    itp_cardinal(_gx, queries; deriv = (D1, D0, D0))
    itp_cardinal(_gy, queries; deriv = (D0, D1, D0))
    itp_cardinal(_gz, queries; deriv = (D0, D0, D1))
    @. ∇ρ_cardinal = sqrt(_gx^2 + _gy^2 + _gz^2)
end

print("    PHS ... ")
eval_times["|∇ρ|"]["PHS"] = @elapsed @time begin
    itp_phs(_gx, queries; deriv = (D1, D0, D0))
    itp_phs(_gy, queries; deriv = (D0, D1, D0))
    itp_phs(_gz, queries; deriv = (D0, D0, D1))
    @. ∇ρ_phs = sqrt(_gx^2 + _gy^2 + _gz^2)
end

# ── Laplacian magnitude |∇²ρ| ──────────────────────────────────────────────────
println("  Laplacian Magnitude (|∇²ρ|):")
eval_times["|∇²ρ|"] = Dict()

print("    Cubic ... ")
eval_times["|∇²ρ|"]["Cubic"] = @elapsed @time begin
    itp_cubic(_gx, queries; deriv = (D2, D0, D0))
    itp_cubic(_gy, queries; deriv = (D0, D2, D0))
    itp_cubic(_gz, queries; deriv = (D0, D0, D2))
    @. ∇²ρ_cubic = abs(_gx + _gy + _gz)
end

print("    Cardinal ... ")
eval_times["|∇²ρ|"]["Cardinal"] = @elapsed @time begin
    itp_cardinal(_gx, queries; deriv = (D2, D0, D0))
    itp_cardinal(_gy, queries; deriv = (D0, D2, D0))
    itp_cardinal(_gz, queries; deriv = (D0, D0, D2))
    @. ∇²ρ_cardinal = abs(_gx + _gy + _gz)
end

print("    PHS ... ")
eval_times["|∇²ρ|"]["PHS"] = @elapsed @time begin
    itp_phs(_gx, queries; deriv = (D2, D0, D0))
    itp_phs(_gy, queries; deriv = (D0, D2, D0))
    itp_phs(_gz, queries; deriv = (D0, D0, D2))
    @. ∇²ρ_phs = abs(_gx + _gy + _gz)
end

println("Evaluation complete.")

# ============================================================
# 5a. Compute errors and generate summary tables
# ============================================================

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

# Compute errors for each method
errors_rho = Dict()
errors_grad = Dict()
errors_lap = Dict()

errors_rho["Nearest"] = compute_rel_error(ρ_nearest, ρ_ref)
errors_rho["Linear"] = compute_rel_error(ρ_linear, ρ_ref)
errors_rho["Cubic"] = compute_rel_error(ρ_cubic, ρ_ref)
errors_rho["Cardinal"] = compute_rel_error(ρ_cardinal, ρ_ref)
errors_rho["PHS"] = compute_rel_error(ρ_phs, ρ_ref)

errors_grad["Linear"] = compute_rel_error(∇ρ_linear, ∇ρ_ref)
errors_grad["Cubic"] = compute_rel_error(∇ρ_cubic, ∇ρ_ref)
errors_grad["Cardinal"] = compute_rel_error(∇ρ_cardinal, ∇ρ_ref)
errors_grad["PHS"] = compute_rel_error(∇ρ_phs, ∇ρ_ref)

errors_lap["Cubic"] = compute_rel_error(∇²ρ_cubic, ∇²ρ_ref)
errors_lap["Cardinal"] = compute_rel_error(∇²ρ_cardinal, ∇²ρ_ref)
errors_lap["PHS"] = compute_rel_error(∇²ρ_phs, ∇²ρ_ref)

# Print performance summary tables
println("\n" * "="^80)
println("PERFORMANCE SUMMARY")
println("="^80)

# Combined Build and Evaluation Times Table
println("\n### Timing Summary (per method, 1000 query points)\n")
println("| Method | Build (s) | ρ Time (s) | \\|∇ρ\\| Time (s) | \\|∇²ρ\\| Time (s) |")
println("|--------|-----------|------------|--------------|----------------|")

for method in ["Nearest", "Linear", "Cubic", "Cardinal", "PHS"]
    build_time = build_times[method]
    rho_time = get(eval_times["ρ"], method, nothing)
    grad_time = get(eval_times["|∇ρ|"], method, nothing)
    lap_time = get(eval_times["|∇²ρ|"], method, nothing)
    
    rho_str = rho_time !== nothing ? @sprintf("%.6f", rho_time) : "—"
    grad_str = grad_time !== nothing ? @sprintf("%.6f", grad_time) : "—"
    lap_str = lap_time !== nothing ? @sprintf("%.6f", lap_time) : "—"
    
    @printf "| %-18s | %.6f | %10s | %12s | %14s |\n" method build_time rho_str grad_str lap_str
end

# Table 2: Density (ρ) Errors
println("\n### Charge Density (ρ) — Relative Error Statistics\n")
println("| Method | Min Error | Max Error | Mean Error | Median Error |")
println("|--------|-----------|-----------|------------|--------------|")
for method in ["Nearest", "Linear", "Cubic", "Cardinal", "PHS"]
    errs = errors_rho[method]
    min_err = minimum(errs)
    max_err = maximum(errs)
    mean_err = sum(errs) / length(errs)
    median_err = median(errs)
    @printf "| %-18s | %.2e | %.2e | %.2e | %.2e |\n" method min_err max_err mean_err median_err
end

# Table 3: Gradient Error
println("\n### Gradient Magnitude (|∇ρ|) — Relative Error Statistics\n")
println("| Method | Min Error | Max Error | Mean Error | Median Error |")
println("|--------|-----------|-----------|------------|--------------|")
for method in ["Linear", "Cubic", "Cardinal", "PHS"]
    errs = errors_grad[method]
    min_err = minimum(errs)
    max_err = maximum(errs)
    mean_err = sum(errs) / length(errs)
    median_err = median(errs)
    @printf "| %-18s | %.2e | %.2e | %.2e | %.2e |\n" method min_err max_err mean_err median_err
end

# Table 4: Laplacian Error
println("\n### Laplacian Magnitude (|∇²ρ|) — Relative Error Statistics\n")
println("| Method | Min Error | Max Error | Mean Error | Median Error |")
println("|--------|-----------|-----------|------------|--------------|")
for method in ["Cubic", "Cardinal", "PHS"]
    errs = errors_lap[method]
    min_err = minimum(errs)
    max_err = maximum(errs)
    mean_err = sum(errs) / length(errs)
    median_err = median(errs)
    @printf "| %-18s | %.2e | %.2e | %.2e | %.2e |\n" method min_err max_err mean_err median_err
end

# Table 5: PHS Error Comparison
println("\n### PHS Error Relative to Other Methods (mean relative error ratio)\n")

# Compute PHS mean errors
phs_rho_mean = sum(errors_rho["PHS"]) / length(errors_rho["PHS"])
phs_grad_mean = sum(errors_grad["PHS"]) / length(errors_grad["PHS"])
phs_lap_mean = sum(errors_lap["PHS"]) / length(errors_lap["PHS"])

# Density comparison
println("**Charge Density (ρ):**")
println("| Method | Error ratio to PHS | PHS Improvement Factor |")
println("|--------|--------------|-------------------|")
for method in ["Nearest", "Linear", "Cubic", "Cardinal"]
    method_mean = sum(errors_rho[method]) / length(errors_rho[method])
    ratio = method_mean / phs_rho_mean
    @printf "| %-18s | %.2f× | %.1f%% better |\n" method ratio (1.0 - 1.0/ratio) * 100
end

# Gradient comparison
println("\n**Gradient Magnitude (|∇ρ|):**")
println("| Method | Error ratio to PHS | PHS Improvement Factor |")
println("|--------|--------------|-------------------|")
for method in ["Linear", "Cubic", "Cardinal"]
    method_mean = sum(errors_grad[method]) / length(errors_grad[method])
    ratio = method_mean / phs_grad_mean
    @printf "| %-18s | %.2f× | %.1f%% better |\n" method ratio (1.0 - 1.0/ratio) * 100
end

# Laplacian comparison
println("\n**Laplacian Magnitude (|∇²ρ|):**")
println("| Method | Error ratio to PHS | PHS Improvement Factor |")
println("|--------|--------------|-------------------|")
for method in ["Cubic", "Cardinal"]
    method_mean = sum(errors_lap[method]) / length(errors_lap[method])
    ratio = method_mean / phs_lap_mean
    @printf "| %-18s | %.2f× | %.1f%% better |\n" method ratio (1.0 - 1.0/ratio) * 100
end

println("\n" * "="^80 * "\n")

# ============================================================
# 6. Plot — 3 rows × 2 columns, log-scale y-axis
# ============================================================
println("\nGenerating plot...")

# Colour scheme (approximately matching the reference figure)
col_analytical = :black
col_nearest    = :blue
col_linear     = :red
col_cubic      = :darkorange   # Trispline
col_cardinal   = :darkgreen  # Tricubic
col_phs        = :red

lw_ref = 2.0
lw_itp = 1.5
lw_phs = 0.8  # Thinner line for PHS

# Replace non-positive values with NaN for safe log-scale rendering
logclean(v) = [x > 0.0 ? x : NaN for x in v]

xlims_val  = (s_ang[1], s_ang[end])
xlabel_str = "Distance along O···H hydrogen bond (Å)"

# Custom Y-axis formatter for log scale
# Decimal notation for values ≤ 10000, scientific for values > 10000
yformatter = (y) -> begin
    if y > 10000
        # Use scientific notation for large values
        @sprintf("%.1e", y)
    else
        # Format with decimal notation, remove trailing zeros
        s = @sprintf("%.8f", y)
        s = replace(s, r"0+$" => "")
        replace(s, r"\.$" => "")
    end
end

# Y-axis ticks: Powers of 10 only, extended range to 10^6
# Range: 10^-4 to 10^6 to cover full data span without gaps
yticks_val = [10.0^i for i in -4:6]

kw_common = (
    xaxis      = :identity,
    yaxis      = :log10,
    xlims      = xlims_val,
    xlabel     = xlabel_str,
    xticks     = 0:0.2:2.0,
    yticks     = yticks_val,
    yformatter = yformatter,
    legend     = :topright,
    minorgrid  = true,
    framestyle = :box,
)

function ref_series!(p, s, ref; label = "Analytical")
    plot!(p, s, logclean(ref); label = label,
          color = col_analytical, linewidth = lw_ref, linestyle = :solid)
end

function add_series!(p, s, data, label, color; lw = lw_itp)
    plot!(p, s, logclean(data); label = label, color = color, linewidth = lw)
end

# ── Row 1: ρ ──────────────────────────────────────────────────────────────────
p11 = plot(; kw_common..., ylabel = "ρ (a.u.)")
ref_series!(p11, s_ang, ρ_ref)
add_series!(p11, s_ang, ρ_nearest,  "Nearest",    col_nearest)
add_series!(p11, s_ang, ρ_linear,   "Trilinear",  col_linear)
add_series!(p11, s_ang, ρ_cardinal, "Tricubic",   col_cardinal)
add_series!(p11, s_ang, ρ_cubic,    "Trispline",  col_cubic)

p12 = plot(; kw_common..., ylabel = "ρ (a.u.)")
ref_series!(p12, s_ang, ρ_ref)
add_series!(p12, s_ang, ρ_phs, "Polyharmonic", col_phs; lw = lw_phs)

# ── Row 2: |∇ρ| ───────────────────────────────────────────────────────────────
p21 = plot(; kw_common..., ylabel = "|∇ρ| (a.u.)")
ref_series!(p21, s_ang, ∇ρ_ref)
add_series!(p21, s_ang, ∇ρ_linear,   "Trilinear", col_linear)
add_series!(p21, s_ang, ∇ρ_cardinal, "Tricubic",  col_cardinal)
add_series!(p21, s_ang, ∇ρ_cubic,    "Trispline", col_cubic)

p22 = plot(; kw_common..., ylabel = "|∇ρ| (a.u.)")
ref_series!(p22, s_ang, ∇ρ_ref)
add_series!(p22, s_ang, ∇ρ_phs, "Polyharmonic", col_phs; lw = lw_phs)

# ── Row 3: |∇²ρ| ──────────────────────────────────────────────────────────────
p31 = plot(; kw_common..., ylabel = "|∇²ρ| (a.u.)")
ref_series!(p31, s_ang, ∇²ρ_ref)
add_series!(p31, s_ang, ∇²ρ_cardinal, "Tricubic",  col_cardinal)
add_series!(p31, s_ang, ∇²ρ_cubic,    "Trispline", col_cubic)

p32 = plot(; kw_common..., ylabel = "|∇²ρ| (a.u.)")
ref_series!(p32, s_ang, ∇²ρ_ref)
add_series!(p32, s_ang, ∇²ρ_phs, "Polyharmonic", col_phs; lw = lw_phs)

# ── Combine ───────────────────────────────────────────────────────────────────
fig = plot(p11, p12, p21, p22, p31, p32;
    layout        = (3, 2),
    size          = (900, 1050),
    dpi           = 150,
    left_margin   = 10Plots.mm,
    bottom_margin = 7Plots.mm,
    top_margin    = 4Plots.mm,
    right_margin  = 3Plots.mm)

savefig(fig, OUT_PATH)
println("Saved: $OUT_PATH")
display(fig)
