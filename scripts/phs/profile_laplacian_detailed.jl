# Analyze performance of different Laplacian evaluation strategies
using FastInterpolations
using Pickle, LinearAlgebra, Statistics
using Printf

function load_data()
    # Load phenol-dimer dataset
    raw_path = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl")
    if !isfile(raw_path)
        error("Dataset not found at $raw_path")
    end

    raw = Pickle.load(open(raw_path))
    return raw
end

# Load dataset
raw = load_data()
x_grid = Float64.(raw["x"])
y_grid = Float64.(raw["y"])
z_grid = Float64.(raw["z"])
rho_3d = Float64.(raw["variables"]["density_scf"])
grids = (x_grid, y_grid, z_grid)
@printf "Grid: %d × %d × %d\n" length(grids[1]) length(grids[2]) length(grids[3])

# Build PHS interpolant with log-density transform
using FastInterpolations: PromolecularRef
pmr = PromolecularRef()
itp_phs = phs_interp(
    grids, rho_3d;
    stencil_size = 8, degree = 3, blend_factor = 1.0,
    reference_interp = pmr
)

# Query points along a path (1000 points for realistic benchmark)
qx = range(grids[1][1], grids[1][end], 1000)
qy = range(grids[2][1], grids[2][end], 1000)
qz = range(grids[3][1], grids[3][end], 1000)
queries = (collect(qx), collect(qy), collect(qz))
N = length(qx)

# Output buffers
gxx = zeros(N)
gyy = zeros(N)
gzz = zeros(N)

const D2 = DerivOp{2}()
const D0 = DerivOp{0}()

# Warm up (allocates, JIT compiles, fills caches)
itp_phs(gxx, queries; deriv = (D2, D0, D0))
itp_phs(gyy, queries; deriv = (D0, D2, D0))
itp_phs(gzz, queries; deriv = (D0, D0, D2))

println("\n=== LAPLACIAN PERFORMANCE ANALYSIS ===\n")

# Benchmark individual components
println("Single component ∂²ρ/∂x² (1000 points, 5 runs):")
time_xx = @time for _ in 1:5
    itp_phs(gxx, queries; deriv = (D2, D0, D0))
end
println("  Per query: $(1000 * time_xx / 5 / N)μs\n")

println("Unified Laplacian (all 3 components, 5 runs):")
time_lap = @time for _ in 1:5
    itp_phs(gxx, queries; deriv = (D2, D0, D0))
    itp_phs(gyy, queries; deriv = (D0, D2, D0))
    itp_phs(gzz, queries; deriv = (D0, D0, D2))
end
println("  Per query (avg): $(1000 * time_lap / 5 / (3 * N))μs\n")

# Compute speedup
speedup_theoretical = 3 * time_xx / time_lap
println("=== SUMMARY ===")
@printf "∂²ρ/∂x² time:    %.6f ms/run (%.4f μs/query)\n" 1000 * time_xx / 5 1000 * time_xx / 5 / N
@printf "Laplacian time:  %.6f ms/run (%.4f μs/query avg)\n" 1000 * time_lap / 5 1000 * time_lap / 5 / 1000
@printf "Theoretical 3× speedup factor: %.2f× (needs %.4f ms)\n" speedup_theoretical 1000 * speedup_theoretical * time_lap / 5 / 3
@printf "Current overhead: %.1f%%\n" 100 * (1 - speedup_theoretical / 3)
