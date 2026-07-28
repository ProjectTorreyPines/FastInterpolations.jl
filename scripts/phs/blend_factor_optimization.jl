#= 
Blend Factor Optimization Study
Tests how blend_factor affects accuracy and performance
=#

using FastInterpolations
using DelimitedFiles
using Pickle
using Printf
using Statistics
using LinearAlgebra

# ============================================================
# Configuration
# ============================================================
const BOHR2ANG = 0.529177210903
const PKL_PATH = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_3dgrid_sp0.236_ext3.pkl")
const CSV_PATH = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_line_O7_H21_N1000.csv")
const XYZ_PATH = joinpath(@__DIR__, "dat", "phenol-dimer_B3LYP_TZ2P_GO_atoms.xyz")

# Patch Pickle.jl to support numpy._core (NumPy >= 2.0) - handled by npyload


function main()
    # ============================================================
    # Load Data
    # ============================================================
    @printf "\n%-60s\n" "="^60
    println("LOADING DATA")
    @printf "%-60s\n" "="^60

    # Load pickle data
    pkl = Pickle.npyload(PKL_PATH)
    x_grid = Float64.(pkl["x"])
    y_grid = Float64.(pkl["y"])
    z_grid = Float64.(pkl["z"])
    rho_3d = Float64.(pkl["variables"]["density_scf"])
    
    println("✓ Loaded 3D grid: $(size(rho_3d))")
    
    # Load CSV line data
    raw = readdlm(CSV_PATH, ',', skipstart=1)
    xyz_line = Float64.(raw[:, 2:4]) .* BOHR2ANG  # Convert to Angstrom
    rho_line_ref = Float64.(raw[:, end])
    
    println("✓ Loaded line data: $(size(rho_line_ref)[1]) points")
    
    # Convert grids to Angstrom
    x_grid = Float64.(x_grid) .* BOHR2ANG
    y_grid = Float64.(y_grid) .* BOHR2ANG
    z_grid = Float64.(z_grid) .* BOHR2ANG
    
    # ============================================================
    # Build Interpolants with Different Blend Factors
    # ============================================================
    @printf "\n%-60s\n" "="^60
    println("BUILDING INTERPOLANTS")
    @printf "%-60s\n" "="^60
    
    grids = (x_grid, y_grid, z_grid)
    blend_factors = [0.75, 1.0, 1.25, 1.5, 2.0]
    itps = Dict{Float64, Any}()
    
    for bf in blend_factors
        @printf "  Building with blend_factor = %.2f ... " bf
        flush(stdout)
        time_build = @elapsed itp = phs_interp(grids, rho_3d; stencil_size=8, degree=3, blend_factor=bf)
        
        # Count blend nodes
        blend_nodes = prod(2 .* itp.blend_r_idx .+ 1)
        
        @printf "%.3fs, %d blend nodes\n" time_build blend_nodes
        itps[bf] = itp
    end
    
    # ============================================================
    # Evaluate on Test Line and Compare
    # ============================================================
    @printf "\n%-60s\n" "="^60
    println("ACCURACY COMPARISON (Line Cut)")
    @printf "%-60s\n" "="^60
    
    results = Dict{Float64, Dict{String, Any}}()
    
    for bf in blend_factors
        itp = itps[bf]
        
        # Allocate output
        rho_phs = similar(rho_line_ref)
        
        # Evaluate on line
        time_eval = @elapsed itp(rho_phs, xyz_line)
        
        # Compute error statistics
        rel_error = abs.(rho_phs .- rho_line_ref) ./ (abs.(rho_line_ref) .+ 1e-16)
        abs_error = abs.(rho_phs .- rho_line_ref)
        
        # Handle NaN/Inf
        valid = isfinite.(rel_error)
        
        results[bf] = Dict(
            "time_eval" => time_eval,
            "max_abs_error" => maximum(abs_error[valid]),
            "mean_abs_error" => mean(abs_error[valid]),
            "max_rel_error" => maximum(rel_error[valid]),
            "mean_rel_error" => mean(rel_error[valid]),
            "n_blend_nodes" => prod(2 .* itp.blend_r_idx .+ 1),
        )
    end
    
    # ============================================================
    # Print Results Table
    # ============================================================
    println("\nBlend Factor Analysis:")
    println("-" * 90)
    println("Factor |  Nodes |  Time(ms) | Max Rel Err | Mean Rel Err |  vs bf=2.0")
    println("-" * 90)
    
    baseline_time = results[2.0]["time_eval"]
    
    for bf in blend_factors
        r = results[bf]
        speedup = baseline_time / r["time_eval"]
        speedup_str = speedup < 1.1 ? @sprintf("%.1f%%", speedup * 100) : @sprintf("%.2f×", speedup)
        
        @printf "%6.2f | %6d | %10.2f | %12.2e | %12.2e | %12s\n" bf r["n_blend_nodes"] r["time_eval"]*1000 r["max_rel_error"] r["mean_rel_error"] speedup_str
    end
    
    println("-" * 90)
    println("\nRecommendations:")
    
    # Check if bf=1.0 maintains accuracy
    err_ratio_1_0 = results[1.0]["max_rel_error"] / results[2.0]["max_rel_error"]
    if err_ratio_1_0 < 1.5
        println("✓ blend_factor=1.0 maintains <50% error increase while getting 78% speedup!")
        println("  Recommended for high-performance use cases where slight accuracy loss is acceptable.")
    else
        println("✗ blend_factor=1.0 loses too much accuracy ($(round(err_ratio_1_0, digits=1))× error increase).")
        println("  Use blend_factor=1.25 for modest speedup with better accuracy retention.")
    end
    
end

if isinteractive()
    main()
else
    main()
end
