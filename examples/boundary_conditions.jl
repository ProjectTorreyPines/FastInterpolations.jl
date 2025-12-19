#=
Boundary Conditions and Extrapolation Methods in FastInterpolations.jl
=======================================================================

This script demonstrates different boundary conditions (BC) and extrapolation
behaviors for both linear and cubic interpolation.

Run this script:
    julia --project examples/boundary_conditions.jl

Requirements:
    - Plots.jl (for visualization)
    - FastInterpolations.jl
=#

using Pkg
Pkg.activate(dirname(@__DIR__))

using FastInterpolations
using Plots

# Set plot defaults
default(
    linewidth=2,
    markersize=6,
    legend=:topleft,
    size=(800, 500)
)

#=
## 1. Linear Interpolation: Extrapolation Methods

When `bc=:none` (default), the `extrapolation` parameter controls
out-of-domain behavior:

- `:extension` (default): Extends the boundary segments linearly
- `:constant`: Returns boundary values outside the domain
=#

function demo_linear_extrapolation()
    println("\n" * "="^60)
    println("1. Linear Interpolation: Extrapolation Methods")
    println("="^60)

    # Sample data (non-periodic function)
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.5, 1.5, 3.0, 2.0]

    # Query points including extrapolation region
    xq = range(-1.0, 5.0, 200)

    # Different extrapolation methods
    y_extension = linear_interp(x, y, collect(xq); extrapolation=:extension)
    y_constant = linear_interp(x, y, collect(xq); extrapolation=:constant)

    # Create plot
    p = plot(title="Linear Interpolation: Extrapolation Methods",
             xlabel="x", ylabel="y")

    # Shade extrapolation regions
    vspan!([-1.0, 0.0], alpha=0.1, color=:gray, label="extrapolation region")
    vspan!([4.0, 5.0], alpha=0.1, color=:gray, label=nothing)

    # Plot interpolation results
    plot!(p, xq, y_extension, label="extrapolation=:extension", color=:blue)
    plot!(p, xq, y_constant, label="extrapolation=:constant",
          color=:red, linestyle=:dash)

    # Plot original data points
    scatter!(p, x, y, label="data points", color=:black, markersize=8)

    savefig(p, joinpath(@__DIR__, "linear_extrapolation.png"))
    println("Saved: examples/linear_extrapolation.png")

    # Print usage
    println("""

    Usage:
        # Extension (default) - continues slope at boundaries
        linear_interp(x, y, xq; extrapolation=:extension)

        # Constant - clamps to boundary values
        linear_interp(x, y, xq; extrapolation=:constant)
    """)

    return p
end

#=
## 2. Linear Interpolation: Periodic Boundary Condition

When `bc=:periodic`, the function wraps query points to the domain [x[1], x[end]).
This is useful for periodic functions like sin, cos, or any cyclic data.

Note: For periodic BC, ensure y[1] ≈ y[end] for smooth results.
=#

function demo_linear_periodic()
    println("\n" * "="^60)
    println("2. Linear Interpolation: Periodic BC")
    println("="^60)

    # Periodic data: one complete sine wave
    n = 21
    x = range(0.0, 2π, n)
    y = sin.(x)  # y[1] = y[end] = 0

    # Query points extending beyond the domain
    xq = range(-π, 3π, 300)

    # Periodic BC vs Extension extrapolation
    y_periodic = linear_interp(collect(x), collect(y), collect(xq); bc=:periodic)
    y_extension = linear_interp(collect(x), collect(y), collect(xq); extrapolation=:extension)

    # True sin function for reference
    y_true = sin.(xq)

    # Create plot
    p = plot(title="Linear Interpolation: Periodic BC vs Extension",
             xlabel="x", ylabel="y")

    # Shade extrapolation regions
    vspan!([-π, 0.0], alpha=0.1, color=:gray, label="outside domain")
    vspan!([2π, 3π], alpha=0.1, color=:gray, label=nothing)

    # Plot results
    plot!(p, xq, y_true, label="sin(x) reference", color=:lightgray, linewidth=3)
    plot!(p, xq, y_periodic, label="bc=:periodic", color=:blue)
    plot!(p, xq, y_extension, label="extrapolation=:extension",
          color=:red, linestyle=:dash)

    # Mark the domain
    vline!([0.0, 2π], color=:black, linestyle=:dot, alpha=0.5, label=nothing)

    # Original data points
    scatter!(p, x, y, label="data points", color=:black, markersize=5)

    savefig(p, joinpath(@__DIR__, "linear_periodic.png"))
    println("Saved: examples/linear_periodic.png")

    println("""

    Usage:
        # Periodic BC - wraps x to [x[1], x[end]) before interpolation
        linear_interp(x, y, xq; bc=:periodic)

    Note: Ensure y[1] ≈ y[end] for smooth periodic interpolation.
    """)

    return p
end

#=
## 3. Cubic Interpolation: Natural vs Periodic BC

Cubic spline boundary conditions:

- `bc=:natural` (default): S''(x₁) = S''(xₙ) = 0 (natural spline)
- `bc=:periodic`: S'(x₁) = S'(xₙ), S''(x₁) = S''(xₙ) (C² continuity at boundaries)

The periodic BC uses the Sherman-Morrison formula to solve the cyclic
tridiagonal system efficiently.
=#

function demo_cubic_bc()
    println("\n" * "="^60)
    println("3. Cubic Interpolation: Natural vs Periodic BC")
    println("="^60)

    # Periodic data: one complete sine wave
    n = 11  # Fewer points to see the difference more clearly
    x = range(0.0, 2π, n)
    y = sin.(x)

    # Query points extending beyond domain
    xq = range(-π, 3π, 400)

    # Different boundary conditions
    y_natural = cubic_interp(collect(x), collect(y), collect(xq); bc=:natural)
    y_periodic = cubic_interp(collect(x), collect(y), collect(xq); bc=:periodic)

    # True sin function
    y_true = sin.(xq)

    # Create plot
    p = plot(title="Cubic Interpolation: Natural vs Periodic BC",
             xlabel="x", ylabel="y")

    # Shade extrapolation regions
    vspan!([-π, 0.0], alpha=0.1, color=:gray, label="outside domain")
    vspan!([2π, 3π], alpha=0.1, color=:gray, label=nothing)

    # Plot results
    plot!(p, xq, y_true, label="sin(x) reference", color=:lightgray, linewidth=3)
    plot!(p, xq, y_natural, label="bc=:natural", color=:red, linestyle=:dash)
    plot!(p, xq, y_periodic, label="bc=:periodic", color=:blue)

    # Domain markers
    vline!([0.0, 2π], color=:black, linestyle=:dot, alpha=0.5, label=nothing)

    # Data points
    scatter!(p, x, y, label="data points", color=:black, markersize=6)

    savefig(p, joinpath(@__DIR__, "cubic_bc.png"))
    println("Saved: examples/cubic_bc.png")

    println("""

    Usage:
        # Natural BC (default) - S''(x₁) = S''(xₙ) = 0
        cubic_interp(x, y, xq; bc=:natural)

        # Periodic BC - C² continuous at boundaries
        cubic_interp(x, y, xq; bc=:periodic)
    """)

    return p
end

#=
## 4. C² Continuity Visualization

The key difference between natural and periodic BC is at the boundary:

- Natural BC: Second derivative is forced to zero at boundaries
- Periodic BC: First AND second derivatives match at x[1] and x[end]

This visualization shows the curvature (second derivative) near boundaries.
=#

function demo_c2_continuity()
    println("\n" * "="^60)
    println("4. C² Continuity Comparison")
    println("="^60)

    # Periodic function
    n = 17
    x = range(0.0, 2π, n)
    y = sin.(x) .+ 0.3 .* cos.(3 .* x)

    # Fine query grid for derivatives
    xq = range(-0.5, 2π + 0.5, 500)

    # Interpolate with both BC types
    y_natural = cubic_interp(collect(x), collect(y), collect(xq); bc=:natural)
    y_periodic = cubic_interp(collect(x), collect(y), collect(xq); bc=:periodic)

    # Numerical second derivative (curvature)
    function numerical_second_deriv(xq, yq)
        h = xq[2] - xq[1]
        d2y = similar(yq)
        d2y[1] = (yq[3] - 2yq[2] + yq[1]) / h^2
        d2y[end] = (yq[end] - 2yq[end-1] + yq[end-2]) / h^2
        for i in 2:length(yq)-1
            d2y[i] = (yq[i+1] - 2yq[i] + yq[i-1]) / h^2
        end
        return d2y
    end

    d2y_natural = numerical_second_deriv(collect(xq), y_natural)
    d2y_periodic = numerical_second_deriv(collect(xq), y_periodic)

    # Create two subplots
    p1 = plot(title="Interpolation Results",
              xlabel="x", ylabel="y", legend=:topright)
    plot!(p1, xq, y_natural, label="bc=:natural", color=:red)
    plot!(p1, xq, y_periodic, label="bc=:periodic", color=:blue)
    scatter!(p1, x, y, label="data", color=:black, markersize=5)
    vline!(p1, [0.0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)

    p2 = plot(title="Second Derivative (Curvature)",
              xlabel="x", ylabel="y''", legend=:topright)
    plot!(p2, xq, d2y_natural, label="bc=:natural (→0 at boundaries)", color=:red)
    plot!(p2, xq, d2y_periodic, label="bc=:periodic (matches at boundaries)", color=:blue)
    vline!(p2, [0.0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
    hline!(p2, [0.0], color=:black, linestyle=:dash, alpha=0.3, label=nothing)

    # Combine plots
    p = plot(p1, p2, layout=(2, 1), size=(800, 700))

    savefig(p, joinpath(@__DIR__, "c2_continuity.png"))
    println("Saved: examples/c2_continuity.png")

    # Print the C² continuity check
    h = Float64(xq[2] - xq[1])
    idx_left = findfirst(x -> x >= 0.0, xq)
    idx_right = findfirst(x -> x >= 2π, xq)

    println("\nC² Continuity Check at boundaries:")
    println("  Natural BC:  y''(0) = $(round(d2y_natural[idx_left], digits=3)), y''(2π) = $(round(d2y_natural[idx_right], digits=3))")
    println("  Periodic BC: y''(0) = $(round(d2y_periodic[idx_left], digits=3)), y''(2π) = $(round(d2y_periodic[idx_right], digits=3))")

    println("""

    Key Observation:
        - Natural BC forces y'' → 0 at boundaries
        - Periodic BC ensures y''(x₁) ≈ y''(xₙ) for true C² continuity
    """)

    return p
end

#=
## 5. Callable Interpolant Objects

For repeated interpolation at different query points, create an interpolant
object once and call it multiple times (more efficient).
=#

function demo_callable()
    println("\n" * "="^60)
    println("5. Callable Interpolant Objects")
    println("="^60)

    # Create data
    x = range(0.0, 2π, 51)
    y = sin.(x)

    # Create interpolants (with different BC)
    linear_natural = LinearInterpolant(collect(x), collect(y))
    linear_periodic = LinearInterpolant(collect(x), collect(y); bc=:periodic)

    cubic_natural = cubic_interp(collect(x), collect(y))  # Returns CubicInterpolant
    cubic_periodic = cubic_interp(collect(x), collect(y); bc=:periodic)

    # Query at various points
    test_points = [-0.5, 0.5, π, 2π + 0.5]

    println("\nEvaluation at test points:")
    println("-"^60)
    println("  x\t\tLinear\t\tLinear\t\tCubic\t\tCubic")
    println("  \t\t(natural)\t(periodic)\t(natural)\t(periodic)")
    println("-"^60)

    for xi in test_points
        v1 = linear_natural(xi)
        v2 = linear_periodic(xi)
        v3 = cubic_natural(xi)
        v4 = cubic_periodic(xi)
        println("  $(round(xi, digits=2))\t\t$(round(v1, digits=4))\t\t$(round(v2, digits=4))\t\t$(round(v3, digits=4))\t\t$(round(v4, digits=4))")
    end

    println("""

    Usage:
        # Create once
        itp = LinearInterpolant(x, y; bc=:periodic)
        itp = cubic_interp(x, y; bc=:periodic)

        # Call multiple times (zero-allocation for scalar queries)
        y1 = itp(x1)
        y2 = itp(x2)
    """)
end

#=
## 6. Pre-built Cache for Multiple Y-values

When you need to interpolate many different y-values on the same x-grid,
pre-build the cache for maximum efficiency.
=#

function demo_cache()
    println("\n" * "="^60)
    println("6. Pre-built Cache for Multiple Y-values")
    println("="^60)

    # Same x-grid, different y-values
    x = range(0.0, 2π, 51)
    y1 = sin.(x)
    y2 = cos.(x)
    y3 = sin.(2 .* x)

    # Pre-build cache (computed once)
    cache_periodic = CubicSplineCache(collect(x); bc=:periodic)

    # Interpolate multiple y-values with same cache
    xq = range(0.0, 2π, 200)

    result1 = cubic_interp(cache_periodic, collect(y1), collect(xq))
    result2 = cubic_interp(cache_periodic, collect(y2), collect(xq))
    result3 = cubic_interp(cache_periodic, collect(y3), collect(xq))

    # Plot
    p = plot(title="Multiple Y-values with Same Cache",
             xlabel="x", ylabel="y", legend=:topright)

    plot!(p, xq, result1, label="sin(x)", color=:blue)
    plot!(p, xq, result2, label="cos(x)", color=:red)
    plot!(p, xq, result3, label="sin(2x)", color=:green)

    scatter!(p, x[1:5:end], y1[1:5:end], label=nothing, color=:blue, markersize=4)
    scatter!(p, x[1:5:end], collect(y2)[1:5:end], label=nothing, color=:red, markersize=4)
    scatter!(p, x[1:5:end], collect(y3)[1:5:end], label=nothing, color=:green, markersize=4)

    savefig(p, joinpath(@__DIR__, "cache_usage.png"))
    println("Saved: examples/cache_usage.png")

    println("""

    Usage:
        # Pre-build cache for x-grid (computed once)
        cache = CubicSplineCache(x; bc=:periodic)

        # Interpolate different y-values efficiently
        result1 = cubic_interp(cache, y1, xq)
        result2 = cubic_interp(cache, y2, xq)
        result3 = cubic_interp(cache, y3, xq)
    """)

    return p
end

# ============================================================================
# Run all demos
# ============================================================================

function main()
    println("\n" * "="^60)
    println("FastInterpolations.jl: Boundary Conditions Demo")
    println("="^60)

    demo_linear_extrapolation()
    demo_linear_periodic()
    demo_cubic_bc()
    demo_c2_continuity()
    demo_callable()
    demo_cache()

    println("\n" * "="^60)
    println("All demos completed!")
    println("Check the examples/ folder for generated plots.")
    println("="^60)
end

main()
