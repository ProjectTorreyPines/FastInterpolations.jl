#!/usr/bin/env julia
"""
Enhanced Profiler: Compare Old (3 calls) vs New (fused) Hessian Evaluation

Shows CPU hotspots for:
1. Density evaluation (baseline)
2. Laplacian with OLD approach (3 separate D2 calls)
3. Laplacian with NEW approach (fused phs_itp_hessian_diag!)
"""

using FastInterpolations
using Profile
using Pickle
using Printf
using DelimitedFiles

# Patch Pickle.jl to support numpy._core (NumPy >= 2.0)
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

# Configuration
const PKL_PATH = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl")
const CSV_PATH = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_line_O7_H21_N1000.csv")
const PROFILE_REPS = 100  # Increased for better statistics

# Load electron density
println("Loading electron density...")
pkl = Pickle.npyload(PKL_PATH)
grids = (Float64.(pkl["x"]), Float64.(pkl["y"]), Float64.(pkl["z"]))
rho = Float64.(pkl["variables"]["density_scf"])
println("Grid: $(size(rho)), ρ ∈ [$(minimum(rho)), $(maximum(rho))]")

# Load CSV query points
raw = readdlm(CSV_PATH, ',', skipstart=1)
qx = Float64.(raw[:, 2])  # x_bohr
qy = Float64.(raw[:, 3])  # y_bohr
qz = Float64.(raw[:, 4])  # z_bohr

println("Building PHS interpolant...")
phs_itp = FastInterpolations.phs_interp(
    grids, rho;
    stencil_size=8,
    degree=3
)

D0 = FastInterpolations.DerivOp{0}()
D2 = FastInterpolations.DerivOp{2}()

println("\n" * "="^70)
println("PROFILING 1: DENSITY EVALUATION (baseline)")
println("="^70)

ρ_phs = zeros(length(qx))
phs_itp(ρ_phs, (qx, qy, qz))  # Warmup

Profile.clear()
@profile for _ in 1:PROFILE_REPS
    phs_itp(ρ_phs, (qx, qy, qz))
end

println("\nHotspots (top 15):")
Profile.print(format=:flat, maxdepth=20, mincount=10)

println("\n" * "="^70)
println("PROFILING 2: LAPLACIAN - OLD APPROACH (3 separate calls)")
println("="^70)

_gx_old = zeros(length(qx))
_gy_old = zeros(length(qx))
_gz_old = zeros(length(qx))

# Warmup
phs_itp(_gx_old, (qx, qy, qz); deriv=(D2, D0, D0))
phs_itp(_gy_old, (qx, qy, qz); deriv=(D0, D2, D0))
phs_itp(_gz_old, (qx, qy, qz); deriv=(D0, D0, D2))

Profile.clear()
@profile for _ in 1:PROFILE_REPS
    phs_itp(_gx_old, (qx, qy, qz); deriv=(D2, D0, D0))
    phs_itp(_gy_old, (qx, qy, qz); deriv=(D0, D2, D0))
    phs_itp(_gz_old, (qx, qy, qz); deriv=(D0, D0, D2))
end

println("\nHotspots (top 15):")
Profile.print(format=:flat, maxdepth=20, mincount=30)

println("\n" * "="^70)
println("PROFILING 3: LAPLACIAN - NEW APPROACH (fused single call)")
println("="^70)

_gx_new = zeros(length(qx))
_gy_new = zeros(length(qx))
_gz_new = zeros(length(qx))

# Warmup
phs_itp_hessian_diag!(phs_itp, _gx_new, _gy_new, _gz_new, (qx, qy, qz))

Profile.clear()
@profile for _ in 1:PROFILE_REPS
    phs_itp_hessian_diag!(phs_itp, _gx_new, _gy_new, _gz_new, (qx, qy, qz))
end

println("\nHotspots (top 15):")
Profile.print(format=:flat, maxdepth=20, mincount=30)

# Verify correctness
println("\n" * "="^70)
println("CORRECTNESS CHECK")
println("="^70)
max_err_xx = maximum(abs.(_gx_old .- _gx_new))
max_err_yy = maximum(abs.(_gy_old .- _gy_new))
max_err_zz = maximum(abs.(_gz_old .- _gz_new))
@printf "Max error Gxx: %.2e\n" max_err_xx
@printf "Max error Gyy: %.2e\n" max_err_yy
@printf "Max error Gzz: %.2e\n" max_err_zz

println("\nProfiler analysis complete.")
