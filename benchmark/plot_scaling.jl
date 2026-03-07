"""
    plot_scaling.jl

Plot benchmark_scaling() results on log-log scale using Plots.jl.

# Usage
```julia
include("simple_benchmarks.jl")
include("plot_scaling.jl")

# Run benchmarks and plot
result = benchmark_scaling()
plot_scaling_results(result)

# Or save to file
plot_scaling_results(result; save_path="benchmark_plots.png")
```
"""

using Plots
using DataFrames

"""
    plot_scaling_results(result; save_path=nothing, dpi=150)

Plot benchmark scaling results on log-log scale.

# Arguments
- `result`: Output from `benchmark_scaling()` containing :construction, :evaluation, :oneshot DataFrames
- `save_path`: Optional path to save the figure (e.g., "plots.png")
- `dpi`: Resolution for saved figure (default: 150)

# Returns
Combined plot with 3 subplots
"""
function plot_scaling_results(result; save_path::Union{String, Nothing} = nothing, dpi::Int = 300, shared_ylim::Bool = true)
    df_constr = result.construction
    df_eval = result.evaluation
    df_oneshot = result.oneshot

    # Color scheme: Interpolations → DataInterp → FastInterp (FastInterp last = most visible)
    colors = [:orange, :green, :blue]
    labels = ["Interpolations.jl" "DataInterpolations.jl" "FastInterpolations.jl"]

    # Calculate shared y-axis limits across all datasets
    all_times = vcat(
        df_constr.FastInterp, df_constr.Interpolations, df_constr.DataInterp,
        df_eval.FastInterp, df_eval.Interpolations, df_eval.DataInterp,
        df_oneshot.FastInterp_cached, df_oneshot.Interpolations, df_oneshot.DataInterp
    )
    ymin = minimum(all_times) * 0.5  # add some padding
    ymax = maximum(all_times) * 2.0
    ylims_shared = shared_ylim ? (ymin, ymax) : :auto

    args = (;
        # shared plot arguments
        label = labels,
        ylabel = "Time (s)",
        xscale = :log10,
        yscale = :log10,
        ylims = ylims_shared,
        marker = :circle,
        markersize = 6,
        markerstrokecolor = :auto,
        linewidth = 2,
        color = permutedims(colors),
        legend = :topleft,
        grid = true,
        minorgrid = true,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 14,
        legendfontsize = 10,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1: Construction time vs n_grid
    # ─────────────────────────────────────────────────────────────────────────
    p1 = plot(
        df_constr.n,
        [df_constr.Interpolations df_constr.DataInterp df_constr.FastInterp];
        title = "Construction Time",
        xlabel = "Grid size (n)",
        args...
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2: Evaluation time vs n_query
    # ─────────────────────────────────────────────────────────────────────────
    p2 = plot(
        df_eval.n,
        [df_eval.Interpolations df_eval.DataInterp df_eval.FastInterp];
        title = "Evaluation Time (n_grid=100, reuse interpolant)",
        xlabel = "Query points (n)",
        args...
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3: One-shot time vs n_query (cached only for consistency)
    # ─────────────────────────────────────────────────────────────────────────
    p3 = plot(
        df_oneshot.n,
        [df_oneshot.Interpolations df_oneshot.DataInterp df_oneshot.FastInterp_cached];
        title = "One-Shot Time (n_grid=100, construct+eval)",
        xlabel = "Query points (n)",
        args...
    )

    # Combine plots
    combined = plot(p1, p2, p3, layout = (1, 3), size = (1600, 400), margin = 10Plots.mm)

    # Save if path provided
    if save_path !== nothing
        savefig(combined, save_path, dpi = dpi)
        println("Saved plot to: $save_path (dpi=$dpi)")
    end

    return combined
end

"""
    plot_scaling_separate(result; save_dir="../docs/images", dpi=250)

Save each plot as a separate file.

# Arguments
- `result`: Output from `benchmark_scaling()`
- `save_dir`: Directory to save plots (default: ../docs/images for Documenter.jl compatibility)
- `dpi`: Resolution for saved figures (default: 250)
"""
function plot_scaling_separate(result; save_dir::String = "../docs/images", prefix::String = "benchmark", dpi::Int = 250)
    # Create directory if it doesn't exist
    mkpath(save_dir)

    df_constr = result.construction
    df_eval = result.evaluation
    df_oneshot = result.oneshot

    # Interpolations → DataInterp → FastInterp (FastInterp last = most visible)
    colors = [:orange, :green, :blue]
    labels = ["Interpolations.jl" "DataInterpolations.jl" "FastInterpolations.jl"]

    # Construction plot
    p1 = plot(
        df_constr.n, [df_constr.Interpolations df_constr.DataInterp df_constr.FastInterp],
        label = labels,
        xlabel = "Grid size",
        ylabel = "Time (s)",
        title = "Construction",
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 6,
        linewidth = 2,
        color = permutedims(colors),
        legend = :topleft,
        grid = true,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 16,
        legendfontsize = 10,
        minorgrid = true,
        size = (600, 450),
        dpi = dpi
    )
    savefig(p1, joinpath(save_dir, "$(prefix)_construction.png"))

    # Evaluation plot
    p2 = plot(
        df_eval.n, [df_eval.Interpolations df_eval.DataInterp df_eval.FastInterp],
        label = labels,
        xlabel = "Query points",
        ylabel = "Time (s)",
        title = "Evaluation",
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 6,
        linewidth = 2,
        color = permutedims(colors),
        legend = :topleft,
        grid = true,
        minorgrid = true,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 16,
        legendfontsize = 10,
        size = (600, 450),
        dpi = dpi
    )
    savefig(p2, joinpath(save_dir, "$(prefix)_evaluation.png"))

    # One-shot plot
    p3 = plot(
        df_oneshot.n, [df_oneshot.Interpolations df_oneshot.DataInterp df_oneshot.FastInterp_cached],
        label = labels,
        xlabel = "Query points",
        ylabel = "Time (s)",
        title = "One-Shot (Construction + Evaluation)",
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 6,
        linewidth = 2,
        color = permutedims(colors),
        legend = :topleft,
        grid = true,
        minorgrid = true,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 16,
        legendfontsize = 10,
        size = (600, 450),
        dpi = dpi
    )
    savefig(p3, joinpath(save_dir, "$(prefix)_oneshot.png"))

    # One-shot detailed plot (with cached vs uncached FastInterpolations)
    p3_detail = plot(
        df_oneshot.n, [df_oneshot.Interpolations df_oneshot.DataInterp df_oneshot.FastInterp_cached],
        label = ["Interpolations.jl" "DataInterpolations.jl" "FastInterpolations.jl (cache-hit)"],
        xlabel = "Query points",
        ylabel = "Time (s)",
        title = "One-Shot (Construction + Evaluation)",
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 6,
        linewidth = 2,
        color = permutedims(colors),
        legend = :topleft,
        grid = true,
        minorgrid = true,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 16,
        legendfontsize = 10,
        size = (600, 450),
        dpi = dpi
    )
    # Add uncached FastInterpolations as dashed line (same blue color)
    plot!(
        p3_detail, df_oneshot.n, df_oneshot.FastInterp_nocache,
        label = "FastInterpolations.jl (cache-miss)",
        linestyle = :dash,
        linewidth = 2,
        color = :blue,
        marker = :none
    )
    savefig(p3_detail, joinpath(save_dir, "$(prefix)_oneshot_detail.png"))

    # One-shot allocation plot
    p4 = plot(
        df_oneshot.n, [df_oneshot.alloc_Interpolations df_oneshot.alloc_DataInterp df_oneshot.alloc_FastInterp_cached],
        label = labels,
        xlabel = "Query points",
        ylabel = "Allocation (bytes)",
        title = "One-Shot Allocation",
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 6,
        linewidth = 2,
        color = permutedims(colors),
        legend = :topleft,
        grid = true,
        minorgrid = true,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 16,
        legendfontsize = 10,
        size = (600, 450),
        dpi = dpi
    )
    savefig(p4, joinpath(save_dir, "$(prefix)_oneshot_allocation.png"))

    # Construction allocation plot
    p5 = plot(
        df_constr.n, [df_constr.alloc_Interpolations df_constr.alloc_DataInterp df_constr.alloc_FastInterp],
        label = labels,
        xlabel = "Grid size",
        ylabel = "Allocation (bytes)",
        title = "Construction Allocation",
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        markersize = 6,
        linewidth = 2,
        color = permutedims(colors),
        legend = :topleft,
        grid = true,
        minorgrid = true,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 16,
        legendfontsize = 10,
        size = (600, 450),
        dpi = dpi
    )
    savefig(p5, joinpath(save_dir, "$(prefix)_construction_allocation.png"))

    println("Saved plots (dpi=$dpi) to:")
    println("  - $(joinpath(save_dir, "$(prefix)_construction.png"))")
    println("  - $(joinpath(save_dir, "$(prefix)_evaluation.png"))")
    println("  - $(joinpath(save_dir, "$(prefix)_oneshot.png"))")
    println("  - $(joinpath(save_dir, "$(prefix)_oneshot_detail.png"))")
    println("  - $(joinpath(save_dir, "$(prefix)_oneshot_allocation.png"))")
    println("  - $(joinpath(save_dir, "$(prefix)_construction_allocation.png"))")

    return (construction = p1, evaluation = p2, oneshot = p3, oneshot_detail = p3_detail, oneshot_allocation = p4, construction_allocation = p5)
end

"""
    plot_speedup(result; save_path=nothing)

Plot speedup ratio (competitor time / FastInterp time) on log scale.
Shows how much faster FastInterp is compared to other packages.
"""
function plot_speedup(result; save_path::Union{String, Nothing} = nothing)
    df_constr = result.construction
    df_eval = result.evaluation
    df_oneshot = result.oneshot

    colors = [:orange, :green]
    labels = ["Interpolations.jl / FastInterpolations.jl" "DataInterpolations.jl / FastInterpolations.jl"]

    # Construction speedup
    speedup_itp_constr = df_constr.Interpolations ./ df_constr.FastInterp
    speedup_di_constr = df_constr.DataInterp ./ df_constr.FastInterp

    args = (;
        label = labels,
        ylabel = "Speedup (x times faster)",
        xscale = :log10,
        marker = :circle,
        markersize = 6,
        markerstrokecolor = :auto,
        linewidth = 2,
        color = permutedims(colors),
        legend = :topright,
        grid = true,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 16,
        legendfontsize = 10,
    )

    p1 = plot(
        df_constr.n, [speedup_itp_constr speedup_di_constr];
        xlabel = "Grid size (n)",
        title = "Construction Speedup",
        args...,
    )
    hline!(p1, [1.0], linestyle = :dash, color = :gray, label = "")

    # Evaluation speedup
    speedup_itp_eval = df_eval.Interpolations ./ df_eval.FastInterp
    speedup_di_eval = df_eval.DataInterp ./ df_eval.FastInterp

    p2 = plot(
        df_eval.n, [speedup_itp_eval speedup_di_eval];
        xlabel = "Query points (n)",
        title = "Evaluation Speedup",
        args...,
    )
    hline!(p2, [1.0], linestyle = :dash, color = :gray, label = "")

    # One-shot speedup (vs cached FastInterp)
    speedup_itp_oneshot = df_oneshot.Interpolations ./ df_oneshot.FastInterp_cached
    speedup_di_oneshot = df_oneshot.DataInterp ./ df_oneshot.FastInterp_cached

    p3 = plot(
        df_oneshot.n, [speedup_itp_oneshot speedup_di_oneshot];
        xlabel = "Query points (n)",
        title = "One-Shot Speedup",
        args...,
    )
    hline!(p3, [1.0], linestyle = :dash, color = :gray, label = "")

    combined = plot(p1, p2, p3, layout = (1, 3), size = (1500, 400), margin = 9Plots.mm)

    if save_path !== nothing
        savefig(combined, save_path)
        println("Saved speedup plot to: $save_path")
    end

    return combined
end
