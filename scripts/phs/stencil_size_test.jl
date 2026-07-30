#= Stencil Size Optimization Study =#

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
println("STENCIL SIZE OPTIMIZATION - SYNTHETIC TEST")
println("="^70)
println("Grid size: $(length(x)) × $(length(y)) × $(length(z))")
println("Test points: $(length(test_queries))")
println()

# Test different stencil sizes
stencil_sizes = [3, 4, 5, 6, 7, 8, 10]
results = Dict{Int, Any}()

for ss in stencil_sizes
    @printf "Testing stencil_size = %d ... " ss
    flush(stdout)

    # Build interpolant
    time_build = @elapsed itp = phs_interp((x, y, z), data; stencil_size = ss, degree = 3, blend_factor = 1.0)

    # Evaluate on test points
    out = Vector{Float64}(undef, length(test_queries))
    time_eval = @elapsed itp(out, test_queries)

    # Compute errors
    errors = abs.(out .- ref_values)
    rel_errors = errors ./ (abs.(ref_values) .+ 1.0e-16)

    # Get stencil info
    stencil_size_total = size(itp.phi_inv, 1)

    results[ss] = (
        time_eval = time_eval,
        max_error = maximum(errors),
        mean_error = mean(errors),
        max_rel_error = maximum(rel_errors),
        mean_rel_error = mean(rel_errors),
        stencil_size_total = stencil_size_total,
    )

    @printf "%.3fms, %d total coeff, max_rel_err=%.2e\n" time_eval * 1000 stencil_size_total results[ss].max_rel_error
end

# Print summary
println("\n" * "="^80)
println("SUMMARY TABLE")
println("="^80)

# ASCII table (for terminal viewing)
println("\nSize | Total Coeff | Time(ms) | Max Rel Err | Speedup | Error Ratio")
println("-"^80)

baseline_time = results[8].time_eval
baseline_err = results[8].max_rel_error

for ss in stencil_sizes
    r = results[ss]
    speedup = baseline_time / r.time_eval
    if speedup >= 1.1
        speedup_str = @sprintf("%.2f×", speedup)
    elseif speedup < 1.0
        speedup_str = @sprintf("%.2f×↓", 1 / speedup)
    else
        speedup_str = "baseline"
    end
    err_ratio = r.max_rel_error / baseline_err
    @printf "%4d | %11d | %8.2f | %11.2e | %8s | %12.2f×\n" ss r.stencil_size_total r.time_eval * 1000 r.max_rel_error speedup_str err_ratio
end

# Markdown table
println("\n" * "="^80)
println("MARKDOWN TABLE")
println("="^80)
println()
println("| stencil_size | Total Coeff | Time(ms) | Max Rel Err | Speedup | Error Ratio |")
println("|---|---|---|---|---|---|")
for ss in stencil_sizes
    r = results[ss]
    speedup = baseline_time / r.time_eval
    if speedup >= 1.1
        speedup_str = @sprintf("%.2f×", speedup)
    elseif speedup < 1.0
        speedup_str = @sprintf("%.2f×↓", 1 / speedup)
    else
        speedup_str = "baseline"
    end
    err_ratio = r.max_rel_error / baseline_err
    @printf "| %d | %d | %.2f | %.2e | %s | %.2f× |\n" ss r.stencil_size_total r.time_eval * 1000 r.max_rel_error speedup_str err_ratio
end
println()
println("\nAnalysis:")

# Find optimal stencil_size (best speed/accuracy trade-off)
min_ss_for_accuracy = nothing

for ss in stencil_sizes[1:(end - 1)]
    r = results[ss]
    speedup = baseline_time / r.time_eval
    err_ratio = r.max_rel_error / baseline_err

    if err_ratio < 1.5 && speedup > 1.3
        speedup_pct = (1 - 1 / speedup) * 100
        err_increase = (err_ratio - 1) * 100
        println("✓ stencil_size=$(ss): $(round(speedup_pct, digits = 1))% faster, max error $(round(err_increase, digits = 0))% larger ($(round(err_ratio, digits = 2))× relative)")
        println("  → RECOMMENDED for high-performance use cases")
        break
    end
end

# Check if smaller stencils can match accuracy
for ss in stencil_sizes
    r = results[ss]
    if r.max_rel_error < baseline_err * 1.1  # Within 10% of ss=8
        min_ss_for_accuracy = ss
        break
    end
end

if min_ss_for_accuracy !== nothing && min_ss_for_accuracy < 8
    speedup = baseline_time / results[min_ss_for_accuracy].time_eval
    err_ratio = results[min_ss_for_accuracy].max_rel_error / baseline_err
    err_increase = (err_ratio - 1) * 100
    speedup_pct = (1 - 1 / speedup) * 100
    println("\n✓ stencil_size=$(min_ss_for_accuracy): achieves similar accuracy ($(round(err_increase, digits = 0))% error increase) while being $(round(speedup_pct, digits = 1))% faster")
    println("  → GOOD BALANCE for production")
end

println("\n✓ Default stencil_size=8 provides excellent accuracy. Smaller sizes trade significant accuracy for moderate speedup.")
