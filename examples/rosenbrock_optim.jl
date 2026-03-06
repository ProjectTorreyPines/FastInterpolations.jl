###############################################################################
# Example: Rosenbrock Optimization with Optim.jl
###############################################################################
#
# Demonstrates three ways to run optimization over an interpolated surface:
#   Method 1 — Default (Optim uses finite differences internally)
#   Method 2 — AD packages (ForwardDiff, Zygote, Enzyme)
#   Method 3 — Analytical derivatives via gradient!/hessian! (fastest)
#
# The plot at the end compares all three methods' convergence trajectories.
# All converge to the minimum of the *interpolated* surface, which differs
# slightly from the true analytic minimum at (1, 1).
###############################################################################

using FastInterpolations, LinearAlgebra
using Optim, Optim.ADTypes
using Plots
using Printf

# ── Build interpolated Rosenbrock surface ─────────────────────────────────

rosenbrock(x, y) = (1.0 - x)^2 + 100.0 * (y - x^2)^2

xg = range(-0.7, 1.5, length = 101)
yg = range(-0.7, 1.5, length = 101)
zg = [rosenbrock(xi, yi) for xi in xg, yi in yg]

# ExtendExtrap() allows trust-region steps outside the grid without error
itp = cubic_interp((xg, yg), zg; extrap = ExtendExtrap(), bc = CubicFit())

x0 = [-0.15, 0.6]

# store_trace/extended_trace are only needed for the visualization below
opts = Optim.Options(store_trace = true, extended_trace = true, iterations = 50)

# ── Method 1: Default (Optim estimates derivatives via finite differences) ─

result_fdm = optimize(itp, x0, NewtonTrustRegion(), opts)

# ── Method 2: AD packages (machine-precision derivatives) ─────────────────
# FastInterpolations.jl supports all major AD backends.
# Uncomment the one you want to use:

result_ad = optimize(itp, x0, NewtonTrustRegion(), opts, autodiff = ADTypes.AutoForwardDiff())
# result_ad = optimize(itp, x0, NewtonTrustRegion(), opts, autodiff=ADTypes.AutoZygote())
# result_ad = optimize(itp, x0, NewtonTrustRegion(), opts, autodiff=ADTypes.AutoEnzyme())

# ── Method 3: Analytical derivatives (fastest) ───────────────────────────
# gradient! and hessian! match Optim's in-place convention directly.
# Results are identical to Method 2 in machine precision, but faster.

grad!(G, x) = FastInterpolations.gradient!(G, itp, x)
hess!(H, x) = FastInterpolations.hessian!(H, itp, x)

result_ana = optimize(itp, grad!, hess!, x0, NewtonTrustRegion(), opts)

# ── Extract traces ────────────────────────────────────────────────────────

function extract_path(result)
    trace = Optim.trace(result)
    path = [haskey(t.metadata, "x") ? collect(t.metadata["x"]) : nothing for t in trace]
    return filter(!isnothing, path)
end

path_fdm = extract_path(result_fdm)
path_ad = extract_path(result_ad)
path_ana = extract_path(result_ana)

xstar = [1.0, 1.0]
dist_fdm = [norm(p .- xstar) for p in path_fdm]
dist_ad = [norm(p .- xstar) for p in path_ad]
dist_ana = [norm(p .- xstar) for p in path_ana]

# ── Summary ──────────────────────────────────────────────────────────────

println("="^80)
println("  Rosenbrock Optimization Comparison")
println("="^80)
@printf("  %-20s %18s %18s %18s\n", "", "FDM", "AD (ForwardDiff)", "Analytical")
println("-"^80)
@printf(
    "  %-20s %18d %18d %18d\n", "Iterations",
    Optim.iterations(result_fdm), Optim.iterations(result_ad), Optim.iterations(result_ana)
)
@printf(
    "  %-20s %18d %18d %18d\n", "f calls",
    Optim.f_calls(result_fdm), Optim.f_calls(result_ad), Optim.f_calls(result_ana)
)
@printf(
    "  %-20s %18.2e %18.2e %18.2e\n", "f(x*)",
    Optim.minimum(result_fdm), Optim.minimum(result_ad), Optim.minimum(result_ana)
)
@printf(
    "  %-20s %18.2e %18.2e %18.2e\n", "‖x* - (1,1)‖",
    dist_fdm[end], dist_ad[end], dist_ana[end]
)
println("="^80)

# ── Visualization ────────────────────────────────────────────────────────

xvis = range(first(xg), last(xg), length = 300)
yvis = range(first(yg), last(yg), length = 300)
zvis = [log10(max(1.0e-6, rosenbrock(xi, yi))) for yi in yvis, xi in xvis]

# Panel 1: contour + trajectories
p1 = contourf(
    xvis, yvis, zvis; levels = 30, color = :inferno,
    xlabel = "x", ylabel = "y", title = "Optimization Trajectories", linewidth = 0.2
)

pxf, pyf = [p[1] for p in path_fdm], [p[2] for p in path_fdm]
pxa, pya = [p[1] for p in path_ad], [p[2] for p in path_ad]
pxn, pyn = [p[1] for p in path_ana], [p[2] for p in path_ana]

plot!(p1, pxf, pyf; lw = 2, color = :cyan, alpha = 0.4, label = "FDM")
scatter!(p1, pxf, pyf; ms = 4, marker = :circle, color = :cyan, alpha = 0.4, markerstrokewidth = 0, label = false)
plot!(p1, pxa, pya; lw = 2, color = :orange, alpha = 0.4, label = "AD (ForwardDiff)")
scatter!(p1, pxa, pya; ms = 4, marker = :square, color = :orange, alpha = 0.4, markerstrokewidth = 0, label = false)
plot!(p1, pxn, pyn; lw = 2, color = :lime, alpha = 0.4, label = "Analytical")
scatter!(p1, pxn, pyn; ms = 4, marker = :diamond, color = :lime, alpha = 0.4, markerstrokewidth = 0, label = false)

scatter!(p1, [x0[1]], [x0[2]]; ms = 10, marker = :dtriangle, color = :red, label = "start")
scatter!(p1, [1.0], [1.0]; ms = 10, marker = :star5, color = :white, label = "true min")
plot!(p1; xlims = extrema(xvis), ylims = extrema(yvis), aspect_ratio = :equal, legend = :topleft)

# Panel 2: convergence distance
p2 = plot(
    0:(length(dist_fdm) - 1), dist_fdm; alpha = 0.6, lw = 2.5, color = :cyan, marker = :circle, ms = 6,
    yscale = :log10, label = "FDM",
    xlabel = "Iteration", ylabel = "‖x - x*‖", title = "Distance to True Minimum", legend = :bottomleft
)
plot!(
    p2, 0:(length(dist_ad) - 1), dist_ad; alpha = 0.6, lw = 2.5, color = :orange, marker = :square, ms = 6,
    label = "AD (ForwardDiff)"
)
plot!(
    p2, 0:(length(dist_ana) - 1), dist_ana; alpha = 0.6, lw = 2.5, color = :lime, marker = :diamond, ms = 6,
    label = "Analytical"
)

plt = plot(
    p1, p2; layout = (1, 2), size = (1050, 420), margin = 4Plots.mm, dpi = 150,
    plot_title = "FastInterpolations.jl — Rosenbrock Optimization",
    plot_titlevspan = 0.1
)
display(plt)
savefig(plt, joinpath(@__DIR__, "rosenbrock_optim.png"))
println("Saved to examples/rosenbrock_optim.png")
