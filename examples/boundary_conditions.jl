# # Boundary Conditions and Extrapolation
#
# This tutorial demonstrates different boundary conditions and extrapolation
# behaviors for both linear and cubic interpolation in FastInterpolations.jl.

using FastInterpolations
using Plots

# Set plot defaults for consistent visualization
default(linewidth=2, markersize=6, legend=:topleft, size=(700, 400))

# ## Linear Interpolation: Extrapolation Methods
#
# The `extrap` parameter controls out-of-domain behavior:
#
# | Option | Behavior |
# |--------|----------|
# | `:none` | Throws DomainError (default) |
# | `:extension` | Extends boundary segments linearly |
# | `:constant` | Returns boundary values |
# | `:wrap` | Wraps coordinates (for periodic data) |

# Sample data
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0, 2.5, 1.5, 3.0, 2.0]

# Query points including extrapolation region
xq = range(-1.0, 5.0, 200)

# Different extrapolation methods
y_extension = linear_interp(x, y, collect(xq); extrap=:extension)
y_constant = linear_interp(x, y, collect(xq); extrap=:constant)

# Visualize the results
p = plot(title="Linear Interpolation: Extrapolation Methods",
         xlabel="x", ylabel="y")
vspan!([-1.0, 0.0], alpha=0.1, color=:gray, label="extrapolation region")
vspan!([4.0, 5.0], alpha=0.1, color=:gray, label=nothing)
plot!(xq, y_extension, label="extrap=:extension", color=:blue)
plot!(xq, y_constant, label="extrap=:constant", color=:red, linestyle=:dash)
scatter!(x, y, label="data points", color=:black, markersize=8)

# ## Linear Interpolation: Wrap Extrapolation
#
# When `extrap=:wrap`, the function wraps query points to the domain `[x[1], x[end])`.
# This is useful for periodic functions like sin, cos, or any cyclic data.
#
# **Note**: For smooth wrapping, ensure `y[1] ≈ y[end]`.

# Periodic data: one complete sine wave
n = 21
x_periodic = collect(range(0.0, 2π, n))
y_periodic = sin.(x_periodic)  # y[1] = y[end] = 0

# Query points extending beyond the domain
xq_extended = range(-π, 3π, 300)

# Compare wrap vs extension
y_wrap = linear_interp(x_periodic, y_periodic, collect(xq_extended); extrap=:wrap)
y_ext = linear_interp(x_periodic, y_periodic, collect(xq_extended); extrap=:extension)
y_true = sin.(xq_extended)

# Visualize
p2 = plot(title="Linear Interpolation: Wrap vs Extension", xlabel="x", ylabel="y")
vspan!([-π, 0.0], alpha=0.1, color=:gray, label="outside domain")
vspan!([2π, 3π], alpha=0.1, color=:gray, label=nothing)
plot!(xq_extended, y_true, label="sin(x) reference", color=:lightgray, linewidth=3)
plot!(xq_extended, y_wrap, label="extrap=:wrap", color=:blue)
plot!(xq_extended, y_ext, label="extrap=:extension", color=:red, linestyle=:dash)
vline!([0.0, 2π], color=:black, linestyle=:dot, alpha=0.5, label=nothing)
scatter!(x_periodic, y_periodic, label="data points", color=:black, markersize=5)

# ## Cubic Interpolation: Natural vs Periodic BC
#
# Cubic spline boundary conditions control the behavior at domain edges:
#
# | BC | Description |
# |----|-------------|
# | `:natural` | S''(x₁) = S''(xₙ) = 0 (default) |
# | `:periodic` | S'(x₁) = S'(xₙ), S''(x₁) = S''(xₙ) (C² continuity) |
#
# The periodic BC uses the Sherman-Morrison formula to solve the cyclic
# tridiagonal system efficiently.

# Fewer points to see the difference more clearly
n_cubic = 11
x_cubic = collect(range(0.0, 2π, n_cubic))
y_cubic = sin.(x_cubic)

# Query points
xq_cubic = range(-π, 3π, 400)

# Different boundary conditions
y_natural = cubic_interp(x_cubic, y_cubic, xq_cubic; bc=:natural, extrap=:extension)
y_periodic_bc = cubic_interp(x_cubic, y_cubic, xq_cubic; bc=:periodic, extrap=:extension)
y_sin = sin.(xq_cubic)

# Visualize
p3 = plot(title="Cubic Interpolation: Natural vs Periodic BC", xlabel="x", ylabel="y")
vspan!([-π, 0.0], alpha=0.1, color=:gray, label="outside domain")
vspan!([2π, 3π], alpha=0.1, color=:gray, label=nothing)
plot!(xq_cubic, y_sin, label="sin(x) reference", color=:lightgray, linewidth=3)
plot!(xq_cubic, y_natural, label="bc=:natural", color=:red, linestyle=:dash)
plot!(xq_cubic, y_periodic_bc, label="bc=:periodic", color=:blue)
vline!([0.0, 2π], color=:black, linestyle=:dot, alpha=0.5, label=nothing)
scatter!(x_cubic, y_cubic, label="data points", color=:black, markersize=6)

# ## C² Continuity Visualization
#
# The key difference between natural and periodic BC is at the boundary:
# - **Natural BC**: Second derivative is forced to zero at boundaries
# - **Periodic BC**: First AND second derivatives match at boundaries

# More complex periodic function
n_c2 = 17
x_c2 = collect(range(0.0, 2π, n_c2))
y_c2 = sin.(x_c2) .+ 0.3 .* cos.(3 .* x_c2)

# Fine query grid
xq_c2 = collect(range(-0.5, 2π + 0.5, 500))

# Interpolate with both BC types
y_nat = cubic_interp(x_c2, y_c2, xq_c2; bc=:natural, extrap=:extension)
y_per = cubic_interp(x_c2, y_c2, xq_c2; bc=:periodic)

# Numerical second derivative (curvature)
function numerical_d2(xq, yq)
    h = xq[2] - xq[1]
    d2y = similar(yq)
    d2y[1] = (yq[3] - 2yq[2] + yq[1]) / h^2
    d2y[end] = (yq[end] - 2yq[end-1] + yq[end-2]) / h^2
    for i in 2:length(yq)-1
        d2y[i] = (yq[i+1] - 2yq[i] + yq[i-1]) / h^2
    end
    return d2y
end

d2y_nat = numerical_d2(xq_c2, y_nat)
d2y_per = numerical_d2(xq_c2, y_per)

# Two subplots
p4a = plot(title="Interpolation Results", xlabel="x", ylabel="y", legend=:topright)
plot!(p4a, xq_c2, y_nat, label="bc=:natural", color=:red)
plot!(p4a, xq_c2, y_per, label="bc=:periodic", color=:blue)
scatter!(p4a, x_c2, y_c2, label="data", color=:black, markersize=5)
vline!(p4a, [0.0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)

p4b = plot(title="Second Derivative (Curvature)", xlabel="x", ylabel="y''", legend=:topright)
plot!(p4b, xq_c2, d2y_nat, label="bc=:natural (→0 at boundaries)", color=:red)
plot!(p4b, xq_c2, d2y_per, label="bc=:periodic (matches)", color=:blue)
vline!(p4b, [0.0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
hline!(p4b, [0.0], color=:black, linestyle=:dash, alpha=0.3, label=nothing)

plot(p4a, p4b, layout=(2, 1), size=(700, 600))

# ## Callable Interpolant Objects
#
# For repeated interpolation at different query points, create an interpolant
# object once and call it multiple times. This is more efficient for repeated queries.

x_call = collect(range(0.0, 2π, 51))
y_call = sin.(x_call)

# Create interpolants with different extrapolation modes
linear_ext = LinearInterpolant(x_call, y_call; extrap=:extension)
linear_wrap_itp = LinearInterpolant(x_call, y_call; extrap=:wrap)
cubic_nat = cubic_interp(x_call, y_call; extrap=:extension)
cubic_per = cubic_interp(x_call, y_call; bc=:periodic)

# Evaluate at test points
test_points = [-0.5, 0.5, π, 2π + 0.5]

println("Evaluation at test points:")
println("| x | Linear (ext) | Linear (wrap) | Cubic (natural) | Cubic (periodic) |")
println("|---|--------------|---------------|-----------------|------------------|")
for xi in test_points
    v1 = round(linear_ext(xi), digits=4)
    v2 = round(linear_wrap_itp(xi), digits=4)
    v3 = round(cubic_nat(xi), digits=4)
    v4 = round(cubic_per(xi), digits=4)
    println("| $xi | $v1 | $v2 | $v3 | $v4 |")
end

# ## Pre-built Cache for Multiple Y-values
#
# When you need to interpolate many different y-values on the same x-grid,
# pre-build the cache for maximum efficiency.

x_cache = collect(range(0.0, 2π, 51))
y1_cache = sin.(x_cache)
y2_cache = cos.(x_cache)
y3_cache = sin.(2 .* x_cache)

# Pre-build cache (computed once)
cache = CubicSplineCache(x_cache; bc=:periodic)

# Interpolate multiple y-values with the same cache
xq_cache = collect(range(0.0, 2π, 200))
result1 = cubic_interp(cache, y1_cache, xq_cache)
result2 = cubic_interp(cache, y2_cache, xq_cache)
result3 = cubic_interp(cache, y3_cache, xq_cache)

# Visualize
p5 = plot(title="Multiple Y-values with Same Cache", xlabel="x", ylabel="y", legend=:topright)
plot!(p5, xq_cache, result1, label="sin(x)", color=:blue)
plot!(p5, xq_cache, result2, label="cos(x)", color=:red)
plot!(p5, xq_cache, result3, label="sin(2x)", color=:green)
scatter!(p5, x_cache[1:5:end], y1_cache[1:5:end], label=nothing, color=:blue, markersize=4)
scatter!(p5, x_cache[1:5:end], y2_cache[1:5:end], label=nothing, color=:red, markersize=4)
scatter!(p5, x_cache[1:5:end], y3_cache[1:5:end], label=nothing, color=:green, markersize=4)
