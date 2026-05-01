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
#   Python 3 + numpy   — for loading the .pkl grid file (no Julia PyCall needed)
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
using Plots
using Printf

# ============================================================
# Configuration — edit paths here
# ============================================================
const PKL_PATH = "/Users/haiiro/scratch/phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl"
const CSV_PATH = "/Users/haiiro/scratch/phenol-dimer_B3LYP_TZ2P_GO_line_O7_H21_N1000.csv"
const OUT_PATH = joinpath(@__DIR__, "phs_density_comparison.png")

const BOHR2ANG = 0.529177210903   # 1 Bohr → Angstrom

# ============================================================
# 1. Load 3D grid from pickle via Python subprocess
# ============================================================
println("Loading 3D grid from pickle...")

tmp_bin = tempname()
py_script = """
import pickle, struct, numpy as np
d = pickle.load(open(r'$(PKL_PATH)', 'rb'))
x = np.asarray(d['x'], dtype='<f8')
y = np.asarray(d['y'], dtype='<f8')
z = np.asarray(d['z'], dtype='<f8')
rho = np.asarray(d['variables']['density_scf'], dtype='<f8')
nx, ny, nz = rho.shape
with open(r'$(tmp_bin)', 'wb') as f:
    f.write(struct.pack('<3i', nx, ny, nz))
    f.write(x.tobytes())
    f.write(y.tobytes())
    f.write(z.tobytes())
    f.write(rho.tobytes(order='F'))
"""
run(`python3 -c $py_script`)

x_grid, y_grid, z_grid, rho_3d = open(tmp_bin, "r") do io
    nx, ny, nz = Int.(read!(io, Vector{Int32}(undef, 3)))
    xg   = read!(io, Vector{Float64}(undef, nx))
    yg   = read!(io, Vector{Float64}(undef, ny))
    zg   = read!(io, Vector{Float64}(undef, nz))
    flat = read!(io, Vector{Float64}(undef, nx * ny * nz))
    xg, yg, zg, reshape(copy(flat), nx, ny, nz)
end
rm(tmp_bin)

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

println("  [5/5] Polyharmonic spline (PHS-3, stencil_size=4)...")
# Note: reference_interp enables a log-density smoothing transform (log(ρ/ρ₀)).
# This is most effective when reference_interp is an analytic promolecular density.
# Using another grid interpolant as reference causes degeneracy (log-ratio≈0 at nodes),
# so we use raw PHS here.
@time itp_phs = phs_interp(grids, rho_3d; stencil_size = 4, degree = 3, blend_factor = 2.0)

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

# Derivative operator shortcuts for PHS:
#   DerivOp{n}() selects the n-th derivative along that axis (0 = value)
const D0 = DerivOp{0}()
const D1 = DerivOp{1}()
const D2 = DerivOp{2}()

# ── density ρ ──────────────────────────────────────────────────────────────────
print("  ρ ... ")
@time for i in 1:N_path
    q = (qx[i], qy[i], qz[i])
    ρ_nearest[i]  = itp_nearest(q)
    ρ_linear[i]   = itp_linear(q)
    ρ_cubic[i]    = itp_cubic(q)
    ρ_cardinal[i] = itp_cardinal(q)
    ρ_phs[i]      = itp_phs(q)
end

# ── gradient magnitude |∇ρ| ────────────────────────────────────────────────────
# gradient(itp, q) returns NTuple{3, Float64} for cubic / cardinal interpolants.
# PHSInterpolantND: must use the deriv kwarg (no _locate_cell/_eval_at_cell).
print("  |∇ρ| ... ")
@time for i in 1:N_path
    q = (qx[i], qy[i], qz[i])

    g = gradient(itp_linear, q)
    ∇ρ_linear[i] = sqrt(g[1]^2 + g[2]^2 + g[3]^2)

    g = gradient(itp_cubic, q)
    ∇ρ_cubic[i] = sqrt(g[1]^2 + g[2]^2 + g[3]^2)

    g = gradient(itp_cardinal, q)
    ∇ρ_cardinal[i] = sqrt(g[1]^2 + g[2]^2 + g[3]^2)

    gx = itp_phs(q; deriv = (D1, D0, D0))
    gy = itp_phs(q; deriv = (D0, D1, D0))
    gz = itp_phs(q; deriv = (D0, D0, D1))
    ∇ρ_phs[i] = sqrt(gx^2 + gy^2 + gz^2)
end

# ── Laplacian magnitude |∇²ρ| ──────────────────────────────────────────────────
# laplacian(itp, q) returns scalar ∇²f for cubic / cardinal.
# PHSInterpolantND: sum of the three diagonal second-order partials.
print("  |∇²ρ| ... ")
@time for i in 1:N_path
    q = (qx[i], qy[i], qz[i])

    ∇²ρ_cubic[i]    = abs(laplacian(itp_cubic,    q))
    ∇²ρ_cardinal[i] = abs(laplacian(itp_cardinal, q))

    d2x = itp_phs(q; deriv = (D2, D0, D0))
    d2y = itp_phs(q; deriv = (D0, D2, D0))
    d2z = itp_phs(q; deriv = (D0, D0, D2))
    ∇²ρ_phs[i] = abs(d2x + d2y + d2z)
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
