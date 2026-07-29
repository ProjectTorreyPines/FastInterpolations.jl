# Quick analysis of where optimization opportunities exist
using FastInterpolations
using Pickle, LinearAlgebra, Statistics
using Printf

# Use same data loading as phs_density_comparison.jl
const PKL_PATH = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl")
pkl = Pickle.npyload(PKL_PATH)
x_grid = Float64.(pkl["x"])
y_grid = Float64.(pkl["y"])
z_grid = Float64.(pkl["z"])
rho_3d = Float64.(pkl["variables"]["density_scf"])
grids = (x_grid, y_grid, z_grid)

@printf "Grid: %d × %d × %d\n" length(x_grid) length(y_grid) length(z_grid)

# Build PHS with log-density transform
using FastInterpolations: PromolecularRef
pmr = PromolecularRef()
itp_phs = phs_interp(grids, rho_3d;
    stencil_size=8, degree=3, blend_factor=1.0,
    reference_interp=pmr)

# Generate test queries (subset for faster analysis)
qx = range(x_grid[1], x_grid[end], 100)
qy = range(y_grid[1], y_grid[end], 100)
qz = range(z_grid[1], z_grid[end], 100)
queries = (collect(qx), collect(qy), collect(qz))
N = length(qx)

gxx = zeros(N)
gyy = zeros(N)
gzz = zeros(N)

const D2 = DerivOp{2}()
const D0 = DerivOp{0}()

# Warm up
itp_phs(gxx, queries; deriv = (D2, D0, D0))

println("\n=== PERFORMANCE ANALYSIS ===\n")

# Baseline single component
println("Single component (∂²ρ/∂x²), 10 runs:")
t1 = @time for _ in 1:10; itp_phs(gxx, queries; deriv = (D2, D0, D0)); end
time_per_component = t1 / 10

# Three components (Laplacian)
println("\nThree components (Laplacian), 10 runs:")
t3 = @time for _ in 1:10
    itp_phs(gxx, queries; deriv = (D2, D0, D0))
    itp_phs(gyy, queries; deriv = (D0, D2, D0))
    itp_phs(gzz, queries; deriv = (D0, D0, D2))
end
time_total_lap = t3 / 10

println("\n=== SUMMARY ===")
@printf "Per-component time: %.6f ms (%d queries)\n" 1000*time_per_component N
@printf "Total Laplacian time: %.6f ms\n" 1000*time_total_lap
@printf "Effective 3× speedup target: %.6f ms\n" 1000*time_per_component

overhead = (time_total_lap - time_per_component) / time_per_component
@printf "Overhead between calls: %.1f%% (%.6f ms)\n" 100*overhead 1000*(time_total_lap-time_per_component)

# Analyze speedup potential
speedup_if_combined = time_total_lap / time_per_component
@printf "\nIf combined into single call: %.2f× speedup needed to save %.6f ms\n" 3.0 1000*(time_total_lap - time_per_component/3)
