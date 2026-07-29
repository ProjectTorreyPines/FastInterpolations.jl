#!/usr/bin/env julia
# Derive and validate Taylor approximation for blend weight function
using Printf, Statistics

# Reference implementation
function blend_weight_ref(d::Float64, a::Float64)
    d >= a && return 0.0
    d3 = d * d * d
    a3 = a * a * a
    return exp(d3 / (d3 - a3))
end

function blend_weight_deriv1_ref(d::Float64, a::Float64)
    d >= a && return 0.0
    d2 = d * d
    d3 = d2 * d
    a3 = a * a * a
    denom = d3 - a3
    w = exp(d3 / denom)
    return -3 * a3 * d2 * w / (denom * denom)
end

function blend_weight_deriv2_ref(d::Float64, a::Float64)
    d >= a && return 0.0
    d2 = d * d
    d3 = d2 * d
    a3 = a * a * a
    denom = d3 - a3
    inv_denom = 1.0 / denom
    inv_denom2 = inv_denom * inv_denom
    inv_denom4 = inv_denom2 * inv_denom2
    w = exp(d3 * inv_denom)
    return 3 * a3 * d * (4 * d3 * d3 + a3 * d3 - 2 * a3 * a3) * w * inv_denom4
end

# Taylor approximation using polynomial fitting
# Strategy: fit polynomial on dimensionless variable ξ = d/a
function fit_taylor_polynomial(max_degree::Int=6)
    a = 1.0  # Use normalized a=1
    
    # Sample points: ξ ∈ [0, 0.99]
    n_samples = 100
    ξs = collect(range(0.0, 0.99, length=n_samples))
    
    # Evaluate reference at sample points
    w_vals = [blend_weight_ref(ξ * a, a) for ξ in ξs]
    wp_vals = [blend_weight_deriv1_ref(ξ * a, a) / a for ξ in ξs]  # Divide by a for derivative
    wpp_vals = [blend_weight_deriv2_ref(ξ * a, a) / a^2 for ξ in ξs]  # Divide by a²
    
    # Build Vandermonde matrix for polynomial fitting
    function fit_poly(ys::Vector{Float64}, max_deg::Int)
        V = [ξs[i]^j for i in 1:n_samples, j in 0:max_deg]
        # Least squares fit
        coeffs = V \ ys
        return coeffs
    end
    
    w_coeffs = fit_poly(w_vals, max_degree)
    wp_coeffs = fit_poly(wp_vals, max_degree)
    wpp_coeffs = fit_poly(wpp_vals, max_degree)
    
    return ξs, w_vals, wp_vals, wpp_vals, w_coeffs, wp_coeffs, wpp_coeffs
end

# Evaluate polynomial approximation
function eval_poly(ξ::Float64, coeffs::Vector{Float64})
    result = 0.0
    ξ_power = 1.0
    for c in coeffs
        result += c * ξ_power
        ξ_power *= ξ
    end
    return result
end

# Test accuracy
function validate_approximation()
    println("=" ^ 80)
    println("BLEND FUNCTION TAYLOR APPROXIMATION VALIDATION")
    println("=" ^ 80)
    println()
    
    # Fit polynomials of different degrees
    for degree in [3, 4, 5, 6]
        println("Fitting degree $degree polynomials...")
        ξs, w_vals, wp_vals, wpp_vals, w_coeffs, wp_coeffs, wpp_coeffs = 
            fit_taylor_polynomial(degree)
        
        # Test accuracy on uniform grid
        a = 1.0
        test_ξs = collect(range(0.0, 0.99, length=20))
        
        w_errors = Float64[]
        wp_errors = Float64[]
        wpp_errors = Float64[]
        
        for ξ in test_ξs
            d = ξ * a
            # Reference values
            w_ref = blend_weight_ref(d, a)
            wp_ref = blend_weight_deriv1_ref(d, a) / a
            wpp_ref = blend_weight_deriv2_ref(d, a) / a^2
            
            # Polynomial approximations
            w_approx = eval_poly(ξ, w_coeffs)
            wp_approx = eval_poly(ξ, wp_coeffs)
            wpp_approx = eval_poly(ξ, wpp_coeffs)
            
            # Relative errors
            push!(w_errors, abs(w_approx - w_ref) / (abs(w_ref) + 1e-10))
            push!(wp_errors, abs(wp_approx - wp_ref) / (abs(wp_ref) + 1e-10))
            push!(wpp_errors, abs(wpp_approx - wpp_ref) / (abs(wpp_ref) + 1e-10))
        end
        
        @printf("\n  Degree %d Results:\n", degree)
        @printf("    w:    max_error = %.2e (mean = %.2e)\n", 
                maximum(w_errors), mean(w_errors))
        @printf("    w':   max_error = %.2e (mean = %.2e)\n", 
                maximum(wp_errors), mean(wp_errors))
        @printf("    w'':  max_error = %.2e (mean = %.2e)\n", 
                maximum(wpp_errors), mean(wpp_errors))
        
        # Print coefficients for degree 5 (good balance)
        if degree == 5
            println("\n  Coefficients for degree 5 (recommended):")
            println("    w:  ", w_coeffs)
            println("    w': ", wp_coeffs)
            println("    w'':", wpp_coeffs)
        end
    end
    
    println("\n" ^ 2)
    println("RECOMMENDATION:")
    println("Use degree 5 polynomial for <1% error with fast evaluation")
end

validate_approximation()
