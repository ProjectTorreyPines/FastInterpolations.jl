# Zero-Overhead Validation for SearchPolicy
#
# This script validates that the new SearchPolicy system introduces
# NO performance regression for the default Binary() path.
#
# Run: julia --project benchmark/validate_zero_overhead.jl

using FastInterpolations
using BenchmarkTools

println("=" ^ 60)
println("SearchPolicy Zero-Overhead Validation")
println("=" ^ 60)
println()

# Test data
const x_vec = collect(range(0.0, 1.0, 1001))
const x_range = range(0.0, 1.0, 1001)
const xi = 0.5

# ========================================
# Benchmark 1: Vector Path
# ========================================

println("## Vector Path (O(log n) binary search)")
println()

# Baseline: internal alias (should be identical to old _find_interval)
t_baseline_vec = @belapsed FastInterpolations._search_interval($x_vec, $xi)
println("  _search_interval(x_vec, xi):     $(round(t_baseline_vec * 1e9, digits=2)) ns")

# Policy path: new dispatcher
policy = FastInterpolations.DEFAULT_SEARCH_POLICY
t_policy_vec = @belapsed FastInterpolations.search_interval($policy, $x_vec, $xi)
println("  search_interval(policy, x_vec, xi): $(round(t_policy_vec * 1e9, digits=2)) ns")

# Deprecated alias
t_deprecated_vec = @belapsed FastInterpolations._find_interval($x_vec, $xi)
println("  _find_interval(x_vec, xi):       $(round(t_deprecated_vec * 1e9, digits=2)) ns")

# Calculate overhead
overhead_vec = (t_policy_vec - t_baseline_vec) / t_baseline_vec * 100
println()
println("  Overhead: $(round(overhead_vec, digits=2))%")

# Assertion: must be within ±5% (noise)
if abs(overhead_vec) > 5.0
    @error "VALIDATION FAILED: Vector path overhead $(round(overhead_vec, digits=2))% exceeds ±5% threshold"
    exit(1)
else
    println("  ✅ PASS: Within ±5% noise threshold")
end

println()

# ========================================
# Benchmark 2: Range Path
# ========================================

println("## Range Path (O(1) direct calculation)")
println()

t_baseline_range = @belapsed FastInterpolations._search_interval($x_range, $xi)
println("  _search_interval(x_range, xi):     $(round(t_baseline_range * 1e9, digits=2)) ns")

t_policy_range = @belapsed FastInterpolations.search_interval($policy, $x_range, $xi)
println("  search_interval(policy, x_range, xi): $(round(t_policy_range * 1e9, digits=2)) ns")

overhead_range = (t_policy_range - t_baseline_range) / t_baseline_range * 100
println()
println("  Overhead: $(round(overhead_range, digits=2))%")

if abs(overhead_range) > 5.0
    @error "VALIDATION FAILED: Range path overhead $(round(overhead_range, digits=2))% exceeds ±5% threshold"
    exit(1)
else
    println("  ✅ PASS: Within ±5% noise threshold")
end

println()

# ========================================
# Benchmark 3: Spacing-aware Path
# ========================================

println("## Spacing-aware Path (O(1) with inv_h)")
println()

spacing = FastInterpolations._create_spacing(x_range)

t_baseline_spacing = @belapsed FastInterpolations._search_interval($x_range, $spacing, $xi)
println("  _search_interval(x, spacing, xi):     $(round(t_baseline_spacing * 1e9, digits=2)) ns")

t_policy_spacing = @belapsed FastInterpolations.search_interval($policy, $x_range, $spacing, $xi)
println("  search_interval(policy, x, spacing, xi): $(round(t_policy_spacing * 1e9, digits=2)) ns")

overhead_spacing = (t_policy_spacing - t_baseline_spacing) / t_baseline_spacing * 100
println()
println("  Overhead: $(round(overhead_spacing, digits=2))%")

if abs(overhead_spacing) > 5.0
    @error "VALIDATION FAILED: Spacing path overhead $(round(overhead_spacing, digits=2))% exceeds ±5% threshold"
    exit(1)
else
    println("  ✅ PASS: Within ±5% noise threshold")
end

println()

# ========================================
# Type Stability Validation
# ========================================

println("## Type Stability")
println()

using InteractiveUtils

# Check return type inference
function check_type_stability()
    x = x_vec
    policy = FastInterpolations.DEFAULT_SEARCH_POLICY

    # Get inferred return type
    ret_type = Base.return_types(FastInterpolations.search_interval, (typeof(policy), typeof(x), Float64))[1]

    expected = Tuple{Int64, Float64, Float64}
    if ret_type == expected
        println("  ✅ Return type: $ret_type (expected)")
        return true
    else
        @error "VALIDATION FAILED: Return type is $ret_type, expected $expected"
        return false
    end
end

if !check_type_stability()
    exit(1)
end

println()

# ========================================
# Correctness Validation
# ========================================

println("## Correctness (result equality)")
println()

function check_correctness()
    test_points = [0.0, 0.001, 0.1, 0.5, 0.999, 1.0]
    policy = FastInterpolations.DEFAULT_SEARCH_POLICY

    for xi in test_points
        r1 = FastInterpolations._search_interval(x_vec, xi)
        r2 = FastInterpolations.search_interval(policy, x_vec, xi)
        r3 = FastInterpolations._find_interval(x_vec, xi)

        if r1 != r2 || r2 != r3
            @error "VALIDATION FAILED: Results differ at xi=$xi" r1 r2 r3
            return false
        end
    end

    println("  ✅ All paths return identical results")
    return true
end

if !check_correctness()
    exit(1)
end

println()

# ========================================
# Summary
# ========================================

println("=" ^ 60)
println("🎉 ALL VALIDATIONS PASSED")
println("=" ^ 60)
println()
println("Zero-overhead confirmed for SearchPolicy default path.")
println("Safe to proceed to Phase 3.")
