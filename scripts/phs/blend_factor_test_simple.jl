#= Minimal Blend Factor Optimization Test =#

using FastInterpolations
using Printf
using Statistics

# Create synthetic 3D test data
x = range(0, 10, 40)
y = range(0, 10, 40)
z = range(0, 10, 40)

# Smooth synthetic function: exp(-r²/20)
data = [exp(-((xi - 5)^2 + (yi - 5)^2 + (zi - 5)^2) / 20.0) for xi in x, yi in y, zi in z]

# Generate test queries - line cut
test_queries = [
    (Float64(xi), 5.0, 5.0) for xi in range(2, 8, 20)
]

# Compute reference values
function reference(x, y, z)
    return exp(-((x - 5)^2 + (y - 5)^2 + (z - 5)^2) / 20.0)
end

ref_values = [reference(q...) for q in test_queries]

println("="^70)
println("BLEND FACTOR OPTIMIZATION - SYNTHETIC TEST")
println("="^70)
println("Grid size: $(length(x)) × $(length(y)) × $(length(z))")
println("Test points: $(length(test_queries))")
println()

# Test different blend factors (using stencil_size=8 for consistency with production)
blend_factors = [0.5, 1.0, 1.5, 2.0]
results = Dict{Float64, Any}()

# Warm up
itp = phs_interp((x, y, z), data; stencil_size = 8, degree = 3, blend_factor = 1.0)
out = Vector{Float64}(undef, length(test_queries))
itp(out, test_queries)

for bf in blend_factors
    @printf "Testing blend_factor = %.1f ... " bf
    flush(stdout)

    # Build interpolant
    time_build = @elapsed itp = phs_interp((x, y, z), data; stencil_size = 8, degree = 3, blend_factor = bf)

    # Evaluate on test points
    out = Vector{Float64}(undef, length(test_queries))
    time_eval = @elapsed itp(out, test_queries)

    # Compute errors
    errors = abs.(out .- ref_values)
    rel_errors = errors ./ (abs.(ref_values) .+ 1.0e-16)

    blend_nodes = prod(2 .* itp.blend_r_idx .+ 1)

    results[bf] = (
        time_build = time_build,
        time_eval = time_eval,
        max_error = maximum(errors),
        mean_error = mean(errors),
        max_rel_error = maximum(rel_errors),
        mean_rel_error = mean(rel_errors),
        blend_nodes = blend_nodes,
    )

    @printf "%.3fms eval, %d nodes, max_rel_err=%.2e\n" time_eval * 1000 blend_nodes results[bf].max_rel_error
end

# Print summary
println("\n" * "="^80)
println("SUMMARY TABLE")
println("="^80)

# ASCII table (for terminal viewing)
println("\nFactor | Nodes | Build(ms) | Eval(ms) | Max Rel Err | Speedup | Rel.Err")
println("-"^80)

baseline_time = results[2.0].time_eval
baseline_err = results[2.0].max_rel_error

for bf in blend_factors
    r = results[bf]
    speedup = baseline_time / r.time_eval
    if speedup >= 1.1
        speedup_str = @sprintf("%.2f×", speedup)
    elseif speedup < 1.0
        speedup_str = @sprintf("%.2f×↓", 1 / speedup)
    else
        speedup_str = "baseline"
    end
    err_ratio = r.max_rel_error / baseline_err
    @printf "%6.1f | %5d | %9.2f | %8.3f | %11.2e | %7s | %8.2f×\n" bf r.blend_nodes r.time_build * 1000 r.time_eval * 1000 r.max_rel_error speedup_str err_ratio
end

# Markdown table
println("\n" * "="^80)
println("MARKDOWN TABLE")
println("="^80)
println()
println("| blend_factor | Blend Nodes | Build (ms) | Eval (ms) | Max Rel Err | Speedup | Rel.Err |")
println("|---|---|---|---|---|---|---|")
for bf in blend_factors
    r = results[bf]
    speedup = baseline_time / r.time_eval
    if speedup >= 1.1
        speedup_str = @sprintf("%.2f×", speedup)
    elseif speedup < 1.0
        speedup_str = @sprintf("%.2f×↓", 1 / speedup)
    else
        speedup_str = "baseline"
    end
    err_ratio = r.max_rel_error / baseline_err
    @printf "| %.1f | %d | %.2f | %.3f | %.2e | %s | %.2f× |\n" bf r.blend_nodes r.time_build * 1000 r.time_eval * 1000 r.max_rel_error speedup_str err_ratio
end
println()
println("\nRecommendation:")
if haskey(results, 1.0)
    err_ratio = results[1.0].max_rel_error / results[2.0].max_rel_error
    time_ratio = results[1.0].time_eval / results[2.0].time_eval
    speedup = (1 - time_ratio) * 100
    @printf "• blend_factor=1.0: %.1f%% faster than 2.0, error increases by %.1f×\n" speedup err_ratio
    if err_ratio < 2.0
        println("  ✓ Excellent trade-off: recommended for most applications")
    else
        println("  ⚠ Noticeable error increase; consider 1.5 for balance")
    end
end
