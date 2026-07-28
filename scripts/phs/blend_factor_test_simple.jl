#= Minimal Blend Factor Optimization Test =#

using FastInterpolations
using Printf
using Statistics

# Create synthetic 3D test data
x = range(0, 10, 40)
y = range(0, 10, 40)
z = range(0, 10, 40)

# Smooth synthetic function: exp(-r²/20)
data = [exp(-((xi-5)^2 + (yi-5)^2 + (zi-5)^2) / 20.0) for xi in x, yi in y, zi in z]

# Generate test queries - line cut
test_queries = [
    (Float64(xi), 5.0, 5.0) for xi in range(2, 8, 20)
]

# Compute reference values
function reference(x, y, z)
    return exp(-((x-5)^2 + (y-5)^2 + (z-5)^2) / 20.0)
end

ref_values = [reference(q...) for q in test_queries]

println("="^70)
println("BLEND FACTOR OPTIMIZATION - SYNTHETIC TEST")
println("="^70)
println("Grid size: $(length(x)) × $(length(y)) × $(length(z))")
println("Test points: $(length(test_queries))")
println()

# Test different blend factors
blend_factors = [0.75, 1.0, 1.25, 1.5, 2.0]
results = Dict{Float64, Any}()

for bf in blend_factors
    @printf "Testing blend_factor = %.2f ... " bf
    flush(stdout)
    
    # Build interpolant
    time_build = @elapsed itp = phs_interp((x, y, z), data; stencil_size=6, degree=3, blend_factor=bf)
    
    # Evaluate on test points
    out = Vector{Float64}(undef, length(test_queries))
    time_eval = @elapsed itp(out, test_queries)
    
    # Compute errors
    errors = abs.(out .- ref_values)
    rel_errors = errors ./ (abs.(ref_values) .+ 1e-16)
    
    blend_nodes = prod(2 .* itp.blend_r_idx .+ 1)
    
    results[bf] = (
        time_eval = time_eval,
        max_error = maximum(errors),
        mean_error = mean(errors),
        max_rel_error = maximum(rel_errors),
        mean_rel_error = mean(rel_errors),
        blend_nodes = blend_nodes
    )
    
    @printf "%.3fms, %d nodes, max_rel_err=%.2e\n" time_eval*1000 blend_nodes results[bf].max_rel_error
end

# Print summary
println("\n" * "="^70)
println("SUMMARY TABLE")
println("="^70)
println("Factor | Nodes | Time(ms) | Max Rel Err | Mean Rel Err |  vs bf=2.0")
println("="^70)

baseline_time = results[2.0].time_eval
for bf in blend_factors
    r = results[bf]
    speedup = baseline_time / r.time_eval
    speedup_str = speedup < 1.1 ? @sprintf("%.1f%%", speedup*100) : @sprintf("%.2f×", speedup)
    @printf "%6.2f | %5d | %8.2f | %11.2e | %12.2e | %9s\n" bf r.blend_nodes r.time_eval*1000 r.max_rel_error r.mean_rel_error speedup_str
end

println("="^70)
println("\nRecommendation:")
err_ratio = results[1.0].max_rel_error / results[2.0].max_rel_error
time_ratio = results[1.0].time_eval / results[2.0].time_eval
@printf "• blend_factor=1.0: %.1f%% faster, error increases by %.1f×\n" (1-time_ratio)*100 err_ratio
if err_ratio < 2.0
    println("  ✓ Acceptable trade-off for most applications")
else
    println("  ✗ Too much error increase, use blend_factor=1.25")
end
