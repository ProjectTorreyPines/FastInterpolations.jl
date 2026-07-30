#!/usr/bin/env julia
# Profile blend weight computation cost vs. other operations
using Profile, Statistics
using Printf

# Simulate typical blend weight computations
function benchmark_blend_weight_exp_based()
    # Reference implementation with exp
    a = 1.0
    a3 = a^3
    times = Float64[]

    for trial in 1:5
        t = @elapsed begin
            for _ in 1:1_000_000
                d = rand() * a * 0.99  # Random d in [0, 0.99*a)
                d3 = d * d * d
                w = exp(d3 / (d3 - a3))
            end
        end
        push!(times, t)
    end
    return median(times)
end

function benchmark_blend_weight_derivs()
    # Full derivative computation
    a = 1.0
    a3 = a^3
    times = Float64[]

    for trial in 1:5
        t = @elapsed begin
            for _ in 1:1_000_000
                d = rand() * a * 0.99
                d2 = d * d
                d3 = d2 * d
                denom = d3 - a3
                inv_denom = 1.0 / denom
                inv_denom2 = inv_denom * inv_denom
                inv_denom4 = inv_denom2 * inv_denom2
                w = exp(d3 * inv_denom)
                wp = -3 * a3 * d2 * w * inv_denom2
                wpp = 3 * a3 * d * (4 * d3 * d3 + a3 * d3 - 2 * a3 * a3) * w * inv_denom4
            end
        end
        push!(times, t)
    end
    return median(times)
end

function benchmark_polynomial_approx()
    # Hypothetical cheap polynomial (without exp)
    # Just multiply-add operations
    a = 1.0
    times = Float64[]

    for trial in 1:5
        t = @elapsed begin
            for _ in 1:1_000_000
                d = rand() * a * 0.99
                ξ = d / a
                ξ2 = ξ * ξ
                ξ3 = ξ2 * ξ

                # Pure polynomial: c0 + c1*ξ + c2*ξ² + c3*ξ³ + c4*ξ⁴ + c5*ξ⁵
                w = 0.979 + 0.852 * ξ - 7.55 * ξ2 + 23.66 * ξ3 - 33.17 * ξ2 * ξ2 + 15.2 * ξ2 * ξ3
            end
        end
        push!(times, t)
    end
    return median(times)
end

function benchmark_distance_calc()
    # What we're comparing against - distance calculation in blend loop
    times = Float64[]

    for trial in 1:5
        t = @elapsed begin
            for _ in 1:1_000_000
                Δx, Δy, Δz = rand(), rand(), rand()
                d2 = Δx * Δx + Δy * Δy + Δz * Δz
                d = sqrt(d2)
            end
        end
        push!(times, t)
    end
    return median(times)
end

println("="^80)
println("BLEND WEIGHT COMPUTATION COST ANALYSIS")
println("="^80)
println()

t_exp = benchmark_blend_weight_exp_based()
t_derivs = benchmark_blend_weight_derivs()
t_poly = benchmark_polynomial_approx()
t_dist = benchmark_distance_calc()

@printf "Time per operation (1M iterations):\n"
@printf "  Distance calculation (3D):           %.6f ms\n" 1000 * t_dist
@printf "  Blend weight (exp only):             %.6f ms\n" 1000 * t_exp
@printf "  Blend weight + derivatives (exp):    %.6f ms\n" 1000 * t_derivs
@printf "  Polynomial approximation (no exp):   %.6f ms\n" 1000 * t_poly
@printf "\n"

speedup = t_derivs / t_poly
savings = t_derivs - t_poly

@printf "Potential speedup from polynomial:\n"
@printf "  Factor: %.2f× \n" speedup
@printf "  Time saved per M iterations: %.6f ms\n" 1000 * savings
@printf "\n"

# Estimate impact on overall Laplacian evaluation
# Typical: ~10-20 blend neighbors per query
neighbors_per_query = 15
queries = 1000
total_blend_calls = neighbors_per_query * queries
time_saved_total = (t_derivs - t_poly) * total_blend_calls

@printf "Impact on Laplacian evaluation:\n"
@printf "  Queries: %d, Neighbors/query: %d\n" queries neighbors_per_query
@printf "  Total blend calls: %d\n" total_blend_calls
@printf "  Total time saved: %.6f ms\n" 1000 * time_saved_total
@printf "  Current Laplacian time: ~8.0 ms\n"
@printf "  Speedup: %.1f%%\n" 100 * time_saved_total / 0.008
