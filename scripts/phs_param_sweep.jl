# ============================================================
# PHS Parameter Sweep — gradient & laplacian only
# ============================================================
# Loads data once, builds PHS interpolants with different blend_factor
# combinations, evaluates all on the same path, and plots comparisons.
#
# Key question: which (ref_blend, main_blend) pairing eliminates:
#   - Spurious gradient minimum at ~0.3 Å    (from reference ρ₀ derivatives)
#   - Spurious laplacian feature at ~0.6 Å   (second-derivative of same)
#   - X-shifted laplacian minima             (from main PHS blending)
#   - Endpoint deviations at s=0, s=1.94 Å  (from both)

using FastInterpolations
using DelimitedFiles
using LinearAlgebra
using Pickle
using Plots
using Printf

# ── Pickle patch for NumPy ≥ 2.0 ──────────────────────────────────────────────
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

# ── Config ─────────────────────────────────────────────────────────────────────
const PKL_PATH = "/Users/haiiro/scratch/phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl"
const CSV_PATH = "/Users/haiiro/scratch/phenol-dimer_B3LYP_TZ2P_GO_line_O7_H21_N1000.csv"
const OUT_PATH = joinpath(@__DIR__, "phs_param_sweep.png")
const BOHR2ANG = 0.529177210903

# ── Load data ─────────────────────────────────────────────────────────────────
println("Loading data...")
pkl    = Pickle.npyload(PKL_PATH)
x_grid = Float64.(pkl["x"]); y_grid = Float64.(pkl["y"]); z_grid = Float64.(pkl["z"])
rho_3d  = Float64.(pkl["variables"]["density_scf"])
rho0_3d = Float64.(pkl["variables"]["density_frag"])
grids   = (x_grid, y_grid, z_grid)
@printf "  Grid: %d×%d×%d\n" length(x_grid) length(y_grid) length(z_grid)

raw    = readdlm(CSV_PATH, ',', skipstart = 1)
qx     = Float64.(raw[:, 2]); qy = Float64.(raw[:, 3]); qz = Float64.(raw[:, 4])
s_ang  = Float64.(raw[:, 5]) .* BOHR2ANG
∇ρ_ref  = Float64.(raw[:, 8])
∇²ρ_ref = abs.(Float64.(raw[:, 9]))
N_path  = length(qx)
@printf "  Path: %d points\n" N_path

# ── Parameter combinations to sweep ──────────────────────────────────────────
# (label, ref_blend_factor, main_blend_factor)
configs = [
    ("ref=2.0, main=2.0  [current]",  2.0, 2.0),
    ("ref=2.5, main=2.0",             2.5, 2.0),
    ("ref=3.0, main=2.0",             3.0, 2.0),
    ("ref=2.0, main=2.5",             2.0, 2.5),
    ("ref=2.0, main=3.0",             2.0, 3.0),
    ("ref=2.5, main=2.5",             2.5, 2.5),
    ("ref=3.0, main=3.0",             3.0, 3.0),
]

# ── Build all interpolants ────────────────────────────────────────────────────
println("\nBuilding interpolants...")
itps = []
for (label, ref_bf, main_bf) in configs
    print("  $label ...")
    t = @elapsed begin
        itp_ref = phs_interp(grids, rho0_3d; stencil_size=8, degree=3,
                             blend_factor=ref_bf, reference_interp=ConstantRef(1.0))
        itp = phs_interp(grids, rho_3d; stencil_size=8, degree=3,
                         blend_factor=main_bf, reference_interp=itp_ref,
                         reference_data=rho0_3d)
    end
    push!(itps, itp)
    @printf "  %.1fs\n" t
end

# ── Evaluate on path ──────────────────────────────────────────────────────────
println("\nEvaluating on path...")
D0 = DerivOp{0}(); D1 = DerivOp{1}(); D2 = DerivOp{2}()

n_configs = length(configs)
grad_results = [zeros(N_path) for _ in 1:n_configs]
lap_results  = [zeros(N_path) for _ in 1:n_configs]

for (ci, (label, _, _)) in enumerate(configs)
    itp = itps[ci]
    print("  grad $label ...")
    t = @elapsed for i in 1:N_path
        q = (qx[i], qy[i], qz[i])
        gx = itp(q; deriv=(D1,D0,D0))
        gy = itp(q; deriv=(D0,D1,D0))
        gz = itp(q; deriv=(D0,D0,D1))
        grad_results[ci][i] = sqrt(gx^2 + gy^2 + gz^2)
    end
    @printf " %.0fs\n" t
end

for (ci, (label, _, _)) in enumerate(configs)
    itp = itps[ci]
    print("  lap  $label ...")
    t = @elapsed for i in 1:N_path
        q = (qx[i], qy[i], qz[i])
        l = itp(q; deriv=(D2,D0,D0)) + itp(q; deriv=(D0,D2,D0)) + itp(q; deriv=(D0,D0,D2))
        lap_results[ci][i] = abs(l)
    end
    @printf " %.0fs\n" t
end

# ── Plot ──────────────────────────────────────────────────────────────────────
println("\nGenerating plot...")
colors = [:red, :blue, :green, :orange, :purple, :brown, :magenta]
n_cols = ceil(Int, n_configs / 2)
n_rows = 2

fig = plot(layout=(n_rows, n_cols), size=(380*n_cols, 340*n_rows), dpi=120,
           left_margin=5Plots.mm, bottom_margin=5Plots.mm)

for (ci, (label, _, _)) in enumerate(configs)
    row = (ci - 1) ÷ n_cols + 1
    col = (ci - 1) % n_cols + 1
    sp  = (row-1)*n_cols + col

    # gradient
    p = plot!(fig, subplot=sp, s_ang, ∇ρ_ref, color=:black, lw=1.2, label="Analytical",
              yscale=:log10, xlim=(0, s_ang[end]), ylim=(1e-4, 1e4),
              title=label, titlefontsize=7,
              xlabel="s (Å)", ylabel="|∇ρ|",
              legend=:topright, legendfontsize=6)
    plot!(fig, subplot=sp, s_ang, grad_results[ci], color=colors[ci], lw=1.0, ls=:dash,
          label="grad")
end

# laplacian as second row - overlay on existing subplots using twinx-like approach
# Instead, just do a 2-row layout: row 1 = gradient, row 2 = laplacian for each config
fig2 = plot(layout=(2, n_cols), size=(380*n_cols, 340*2), dpi=120,
            left_margin=5Plots.mm, bottom_margin=5Plots.mm)

for (ci, (label, _, _)) in enumerate(configs)
    col = (ci - 1) % n_cols + 1

    # row 1: gradient
    sp_g = col
    plot!(fig2, subplot=sp_g, s_ang, ∇ρ_ref, color=:black, lw=1.5, label="Analytical",
          yscale=:log10, xlim=(0, s_ang[end]), ylim=(1e-4, 1e4),
          title=label, titlefontsize=7,
          xlabel="s (Å)", ylabel="|∇ρ| (a.u./Bohr)",
          legend=:topright, legendfontsize=6)
    plot!(fig2, subplot=sp_g, s_ang, grad_results[ci], color=colors[ci], lw=1.0,
          label="PHS")

    # row 2: laplacian
    sp_l = n_cols + col
    plot!(fig2, subplot=sp_l, s_ang, ∇²ρ_ref, color=:black, lw=1.5, label="Analytical",
          yscale=:log10, xlim=(0, s_ang[end]), ylim=(1e-2, 1e8),
          xlabel="s (Å)", ylabel="|∇²ρ| (a.u./Bohr²)",
          legend=:topright, legendfontsize=6)
    plot!(fig2, subplot=sp_l, s_ang, lap_results[ci], color=colors[ci], lw=1.0,
          label="PHS")
end

savefig(fig2, OUT_PATH)
println("Saved: $OUT_PATH")
