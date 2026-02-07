# [Optimization with Optim.jl](@id optimization_guide)

FastInterpolations.jl integrates seamlessly with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) for optimization problems over interpolated surfaces.

This guide demonstrates three approaches to provide derivatives, ranging from simplest to fastest.

## Setup: Rosenbrock on an Interpolated Surface

The classic Rosenbrock function ``f(x,y) = (1-x)^2 + 100(y-x^2)^2`` is a standard optimization benchmark. Here we build a cubic interpolant over a 2D grid and optimize it:

```julia
using FastInterpolations, Optim

rosenbrock(x, y) = (1.0 - x)^2 + 100.0 * (y - x^2)^2

# Build the interpolated surface
xg = range(-0.7, 1.5, length=101)
yg = range(-0.7, 1.5, length=101)
zg = [rosenbrock(xi, yi) for xi in xg, yi in yg]

itp = cubic_interp((xg, yg), zg; extrap=:extension, bc=CubicFit())
```

!!! tip "Why `extrap=:extension`?"
    Trust-region and line-search methods may evaluate points outside the grid during
    intermediate steps. `:extension` extrapolates smoothly from the boundary, preventing
    errors without distorting the objective landscape near the boundary.

## Three Ways to Run Optimization

```julia
using Optim: ADTypes

x0 = [-0.15, 0.6] # Initial guess

# ── Method 1: Default (Optim estimates derivatives via finite differences) ──
result = optimize(itp, x0, NewtonTrustRegion())

# ── Method 2: AD packages (machine-precision gradients) ──
# FastInterpolations.jl supports all major AD packages
using ForwardDiff, Zygote, Enzyme 
result = optimize(itp, x0, NewtonTrustRegion(), autodiff=ADTypes.AutoForwardDiff())
result = optimize(itp, x0, NewtonTrustRegion(), autodiff=ADTypes.AutoZygote())
result = optimize(itp, x0, NewtonTrustRegion(), autodiff=ADTypes.AutoEnzyme())

# ── Method 3: Analytical derivatives (fastest & accurate) ──
grad!(G, x) = FastInterpolations.gradient!(G, itp, x)
hess!(H, x) = FastInterpolations.hessian!(H, itp, x)
result = optimize(itp, grad!, hess!, x0, NewtonTrustRegion())
```

| Method | Derivatives | Accuracy |
|--------|-------------|-------|
| Default | Numerical (Finite Difference) | ~O(h²) |
| AD ([ForwardDiff, Zygote, Enzyme](autodiff_nd.md)) | Automatic Differentiation | Accurate |
| Analytical [`gradient!`](@ref)/[`hessian!`](@ref) | Direct Analytic Estimation | Accurate |

Method 1 requires zero setup — Optim.jl internally approximates derivatives via finite differences when no `autodiff` or gradient function is provided, but numerical approximation can be inaccurate or unstable depending on the surface. Methods 2 and 3 both provide exact derivatives of the interpolant (identical to machine precision); Method 3 is recommended for performance-critical loops.

!!! tip "In-place convention"
    [`gradient!`](@ref) and [`hessian!`](@ref) follow Optim.jl's `optimize(f, g!, h!, x0, method)` convention — they mutate the first argument in-place, avoiding allocations in the optimization loop.

## Results

![FDM vs Analytical Newton on Rosenbrock](../images/rosenbrock_optim.png)

The plot compares Method 1 (finite differences), Method 2 (AD; ForwardDiff) and Method 3 (analytical derivatives) on the Rosenbrock surface. Method 2 & 3 converge faster and more stably. Note that all methods converge to the minimum of the *interpolated* surface, which may differ slightly from the true analytic minimum at (1, 1) — this is inherent to optimizing over a discrete grid representation.

The full runnable script is available at [`examples/rosenbrock_optim.jl`](https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master/examples/rosenbrock_optim.jl).

## See Also

- [AD Support for ND Interpolants](autodiff_nd.md) — ForwardDiff, Zygote, Enzyme backend details
- [`gradient`](@ref), [`gradient!`](@ref) — analytical gradient API
- [`hessian`](@ref), [`hessian!`](@ref) — analytical Hessian API
