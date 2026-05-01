# ============================================================
# PHS Charge Density Analysis — Comparison Script
# ============================================================
#
# Reproduces the 3×2 log-scale comparison plot comparing analytical
# (DFT reference) values against 3D interpolants for:
#   Row 1:  ρ        (charge density)
#   Row 2:  |∇ρ|     (gradient magnitude)
#   Row 3:  |∇²ρ|    (Laplacian magnitude)
# along the O7...H21 hydrogen-bond path in the phenol dimer.
#
# Left column  — all standard 3D methods vs Analytical
# Right column — Polyharmonic spline (PHS) vs Analytical
#
# Data files (edit the paths below if needed):
#   3D grid : phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl
#             Python pickle: x (75,), y (113,), z (70,) [Bohr],
#             variables["density_scf"] (75,113,70) [a.u.]
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
const PKL_PATH = "/Users/haiiro/scratch/phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl"
const CSV_PATH = "/Users/haiiro/scratch/phenol-dimer_B3LYP_TZ2P_GO_line_O7_H21_N1000.csv"
const OUT_PATH = joinpath(@__DIR__, "phs_density_comparison.png")

const BOHR2ANG = 0.529177210903   # 1 Bohr → Angstrom

# ============================================================
# 1. Load 3D grid from pickle via Pickle.jl
# ============================================================
println("Loading 3D grid from pickle...")

pkl = Pickle.npyload(PKL_PATH)
x_grid = Float64.(pkl["x"])
y_grid = Float64.(pkl["y"])
z_grid = Float64.(pkl["z"])
rho_3d  = Float64.(pkl["variables"]["density_scf"])   # already (nx,ny,nz) via c2f
rho0_3d = Float64.(pkl["variables"]["density_frag"])

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
# 3. Build 3D interpolants
# ============================================================
grids = (x_grid, y_grid, z_grid)
println("\nBuilding interpolants on $(length(x_grid))×$(length(y_grid))×$(length(z_grid)) grid...")

println("  [1/5] Nearest (constant)...")
@time itp_nearest  = constant_interp(grids, rho_3d)

println("  [2/5] Trilinear (linear)...")
@time itp_linear   = linear_interp(grids, rho_3d)

println("  [3/5] Trispline (global cubic spline)...")
@time itp_cubic    = cubic_interp(grids, rho_3d)

println("  [4/5] Tricubic (Cardinal / Catmull-Rom)...")
@time itp_cardinal = interp(grids, rho_3d;
    method = (CardinalInterp(), CardinalInterp(), CardinalInterp()))

# ============================================================
# LogCubicRef — fast, smooth ρ₀ reference from a global cubic spline of log(ρ₀)
#
# Near nuclei, log(ρ₀) is approximately linear in r (exponential decay).
# A GLOBAL cubic spline of log(ρ₀) gives smoother, less oscillatory derivatives
# than a local PHS stencil that spans the nuclear cusp node.
# Evaluation is microsecond-fast (vs milliseconds for PHS).
#
# The chain rule recovers ρ₀ and its derivatives in the ρ₀ domain:
#   ρ₀         = exp(log_itp(q))
#   ∂ρ₀/∂xξ   = ρ₀ · ∂(log ρ₀)/∂xξ
#   ∂²ρ₀/∂xξ² = ρ₀ · [(∂ log ρ₀/∂xξ)² + ∂²(log ρ₀)/∂xξ²]
# ============================================================
struct LogCubicRef{T}
    log_itp::T  # cubic spline of log(ρ₀)
end
function (r::LogCubicRef{T})(q; deriv=nothing) where T
    N = length(q)
    rho0 = exp(r.log_itp(q))
    deriv === nothing && return rho0
    total = sum(o -> deriv_order(o), deriv)
    total == 0 && return rho0
    if total == 1
        return rho0 * r.log_itp(q; deriv=deriv)
    elseif total == 2
        nonzero = [d for d in 1:N if deriv_order(deriv[d]) > 0]
        ax1 = nonzero[1]; ax2 = length(nonzero) >= 2 ? nonzero[2] : ax1
        ops1 = ntuple(d -> d == ax1 ? DerivOp{1}() : EvalValue(), Val(N))
        ops2 = ax1 == ax2 ? ops1 : ntuple(d -> d == ax2 ? DerivOp{1}() : EvalValue(), Val(N))
        dl1  = r.log_itp(q; deriv=ops1)
        dl2  = ax1 == ax2 ? dl1 : r.log_itp(q; deriv=ops2)
        d2l  = r.log_itp(q; deriv=deriv)
        return rho0 * (dl1 * dl2 + d2l)
    end
    return zero(rho0)
end

println("  Building log-cubic reference for ρ₀ (cubic spline of log ρ₀)...")
@time log_rho0_itp = cubic_interp(grids, log.(rho0_3d))
ref_rho0 = LogCubicRef(log_rho0_itp)

println("  [5/5] Polyharmonic spline (PHS-3, stencil_size=8, log-density transform)...")
# Paper (Sec. III): N = 8³ = 512 stencil nodes; f = log(ρ_scf / ρ₀) is smooth
# across the whole grid, eliminating the nuclear-cusp oscillations that affect
# raw interpolation of ρ directly.  LogCubicRef provides ρ₀ and its derivatives
# via the chain rule on a global cubic spline of log(ρ₀) — much faster at eval
# time and smoother near nuclei than a local PHS stencil.
@time itp_phs = phs_interp(grids, rho_3d; stencil_size = 8, degree = 3, blend_factor = 3.0,
    reference_interp = ref_rho0, reference_data = rho0_3d)

println("All interpolants built.")

# ============================================================
# 4. Evaluate along the 1D path
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
print("  ρ ... ")
@time begin
    itp_nearest(ρ_nearest,  queries)
    itp_linear(ρ_linear,    queries)
    itp_cubic(ρ_cubic,      queries)
    itp_cardinal(ρ_cardinal, queries)
    itp_phs(ρ_phs,          queries)
end

# ── gradient magnitude |∇ρ| ────────────────────────────────────────────────────
# All ND interpolants accept the batch form itp(out, queries; deriv=(...)).
# PHS does not implement _locate_cell/_eval_at_cell so gradient() is unavailable,
# but the same computation works via the deriv kwarg on the batch callable.
print("  |∇ρ| ... ")
@time begin
    itp_linear(_gx, queries; deriv = (D1, D0, D0))
    itp_linear(_gy, queries; deriv = (D0, D1, D0))
    itp_linear(_gz, queries; deriv = (D0, D0, D1))
    @. ∇ρ_linear = sqrt(_gx^2 + _gy^2 + _gz^2)

    itp_cubic(_gx, queries; deriv = (D1, D0, D0))
    itp_cubic(_gy, queries; deriv = (D0, D1, D0))
    itp_cubic(_gz, queries; deriv = (D0, D0, D1))
    @. ∇ρ_cubic = sqrt(_gx^2 + _gy^2 + _gz^2)

    itp_cardinal(_gx, queries; deriv = (D1, D0, D0))
    itp_cardinal(_gy, queries; deriv = (D0, D1, D0))
    itp_cardinal(_gz, queries; deriv = (D0, D0, D1))
    @. ∇ρ_cardinal = sqrt(_gx^2 + _gy^2 + _gz^2)

    itp_phs(_gx, queries; deriv = (D1, D0, D0))
    itp_phs(_gy, queries; deriv = (D0, D1, D0))
    itp_phs(_gz, queries; deriv = (D0, D0, D1))
    @. ∇ρ_phs = sqrt(_gx^2 + _gy^2 + _gz^2)
end

# ── Laplacian magnitude |∇²ρ| ──────────────────────────────────────────────────
print("  |∇²ρ| ... ")
@time begin
    itp_cubic(_gx, queries; deriv = (D2, D0, D0))
    itp_cubic(_gy, queries; deriv = (D0, D2, D0))
    itp_cubic(_gz, queries; deriv = (D0, D0, D2))
    @. ∇²ρ_cubic = abs(_gx + _gy + _gz)

    itp_cardinal(_gx, queries; deriv = (D2, D0, D0))
    itp_cardinal(_gy, queries; deriv = (D0, D2, D0))
    itp_cardinal(_gz, queries; deriv = (D0, D0, D2))
    @. ∇²ρ_cardinal = abs(_gx + _gy + _gz)

    itp_phs(_gx, queries; deriv = (D2, D0, D0))
    itp_phs(_gy, queries; deriv = (D0, D2, D0))
    itp_phs(_gz, queries; deriv = (D0, D0, D2))
    @. ∇²ρ_phs = abs(_gx + _gy + _gz)
end

println("Evaluation complete.")

# ============================================================
# 5. Plot — 3 rows × 2 columns, log-scale y-axis
# ============================================================
println("\nGenerating plot...")

# Colour scheme (approximately matching the reference figure)
col_analytical = :black
col_nearest    = :blue
col_linear     = :red
col_cubic      = :darkgreen   # Trispline
col_cardinal   = :darkorange  # Tricubic
col_phs        = :red

lw_ref = 2.0
lw_itp = 1.5

# Replace non-positive values with NaN for safe log-scale rendering
logclean(v) = [x > 0.0 ? x : NaN for x in v]

xlims_val  = (s_ang[1], s_ang[end])
xlabel_str = "Distance along O···H hydrogen bond (Å)"

kw_common = (
    xaxis      = :identity,
    yaxis      = :log10,
    xlims      = xlims_val,
    xlabel     = xlabel_str,
    legend     = :topright,
    minorgrid  = true,
    framestyle = :box,
)

function ref_series!(p, s, ref; label = "Analytical")
    plot!(p, s, logclean(ref); label = label,
          color = col_analytical, linewidth = lw_ref, linestyle = :solid)
end

function add_series!(p, s, data, label, color)
    plot!(p, s, logclean(data); label = label, color = color, linewidth = lw_itp)
end

# ── Row 1: ρ ──────────────────────────────────────────────────────────────────
p11 = plot(; kw_common..., ylabel = "ρ (a.u.)")
ref_series!(p11, s_ang, ρ_ref)
add_series!(p11, s_ang, ρ_nearest,  "Nearest",    col_nearest)
add_series!(p11, s_ang, ρ_linear,   "Trilinear",  col_linear)
add_series!(p11, s_ang, ρ_cubic,    "Trispline",  col_cubic)
add_series!(p11, s_ang, ρ_cardinal, "Tricubic",   col_cardinal)

p12 = plot(; kw_common..., ylabel = "ρ (a.u.)")
ref_series!(p12, s_ang, ρ_ref)
add_series!(p12, s_ang, ρ_phs, "Polyharmonic", col_phs)

# ── Row 2: |∇ρ| ───────────────────────────────────────────────────────────────
p21 = plot(; kw_common..., ylabel = "|∇ρ| (a.u./Bohr)")
ref_series!(p21, s_ang, ∇ρ_ref)
add_series!(p21, s_ang, ∇ρ_linear,   "Trilinear", col_linear)
add_series!(p21, s_ang, ∇ρ_cubic,    "Trispline", col_cubic)
add_series!(p21, s_ang, ∇ρ_cardinal, "Tricubic",  col_cardinal)

p22 = plot(; kw_common..., ylabel = "|∇ρ| (a.u./Bohr)")
ref_series!(p22, s_ang, ∇ρ_ref)
add_series!(p22, s_ang, ∇ρ_phs, "Polyharmonic", col_phs)

# ── Row 3: |∇²ρ| ──────────────────────────────────────────────────────────────
p31 = plot(; kw_common..., ylabel = "|∇²ρ| (a.u./Bohr²)")
ref_series!(p31, s_ang, ∇²ρ_ref)
add_series!(p31, s_ang, ∇²ρ_cubic,    "Trispline", col_cubic)
add_series!(p31, s_ang, ∇²ρ_cardinal, "Tricubic",  col_cardinal)

p32 = plot(; kw_common..., ylabel = "|∇²ρ| (a.u./Bohr²)")
ref_series!(p32, s_ang, ∇²ρ_ref)
add_series!(p32, s_ang, ∇²ρ_phs, "Polyharmonic", col_phs)

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
