#!/usr/bin/env julia

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
const BENCHMARK_REPS = 10
const PROFILE_REPS = 50

# Load electron density
println("Loading electron density...")
pkl = Pickle.npyload(PKL_PATH)
grids = (Float64.(pkl["x"]), Float64.(pkl["y"]), Float64.(pkl["z"]))
rho = Float64.(pkl["variables"]["density_scf"])
println("Grid: $(size(rho)), ρ ∈ [$(minimum(rho)), $(maximum(rho))]")

# Load CSV query points
# Load CSV query points
raw = readdlm(CSV_PATH, ',', skipstart = 1)
qx = Float64.(raw[:, 2])  # x_bohr
qy = Float64.(raw[:, 3])  # y_bohr
qz = Float64.(raw[:, 4])  # z_bohr
query_points = [(qx[i], qy[i], qz[i]) for i in 1:length(qx)]

println("Building PHS interpolant...")
phs_itp = FastInterpolations.phs_interp(
    grids, rho;
    stencil_size = 8,
    degree = 3
)

println("\n" * "="^60)
println("PROFILING: DENSITY (ρ) EVALUATION")
println("="^60)

# Pre-allocate result arrays
ρ_phs = zeros(length(qx))
D0 = FastInterpolations.DerivOp{0}()

# Warmup
phs_itp(ρ_phs, (qx, qy, qz))

Profile.clear()
@profile for _ in 1:PROFILE_REPS
    phs_itp(ρ_phs, (qx, qy, qz))
end

println("\nTop functions by sample count:")
Profile.print(format = :flat, maxdepth = 30, mincount = 20)

println("\n" * "="^60)
println("PROFILING: LAPLACIAN (|∇²ρ|) EVALUATION")
println("="^60)

# Pre-allocate result arrays
_gx = zeros(length(qx))
_gy = zeros(length(qx))
_gz = zeros(length(qx))
D1 = FastInterpolations.DerivOp{1}()
D2 = FastInterpolations.DerivOp{2}()

# Warmup
phs_itp(_gx, (qx, qy, qz); deriv = (D2, D0, D0))
phs_itp(_gy, (qx, qy, qz); deriv = (D0, D2, D0))
phs_itp(_gz, (qx, qy, qz); deriv = (D0, D0, D2))

Profile.clear()
@profile for _ in 1:PROFILE_REPS
    phs_itp(_gx, (qx, qy, qz); deriv = (D2, D0, D0))
    phs_itp(_gy, (qx, qy, qz); deriv = (D0, D2, D0))
    phs_itp(_gz, (qx, qy, qz); deriv = (D0, D0, D2))
end

println("\nTop functions by sample count:")
Profile.print(format = :flat, maxdepth = 30, mincount = 20)

println("\nProfiler run complete.")
