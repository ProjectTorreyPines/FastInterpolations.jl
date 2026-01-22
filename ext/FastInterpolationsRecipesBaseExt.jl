# ========================================
# Plots.jl Recipe Extension for FastInterpolations.jl
# ========================================
# This extension provides automatic visualization for interpolants
# when Plots.jl (or RecipesBase) is loaded.
#
# Usage:
#   using FastInterpolations, Plots
#   itp = cubic_interp(x, y; extrap=:constant)
#   plot(itp)  # automatic documentation-style visualization

module FastInterpolationsRecipesBaseExt

using FastInterpolations
using RecipesBase: @recipe, @series

# Import types for dispatch
import FastInterpolations:
    AbstractInterpolant, AbstractSeriesInterpolant, AbstractDerivativeView,
    LinearInterpolant, ConstantInterpolant, QuadraticInterpolant, CubicInterpolant,
    LinearSeriesInterpolant, ConstantSeriesInterpolant,
    QuadraticSeriesInterpolant, CubicSeriesInterpolant,
    DerivativeView

# ========================================
# Helper Functions
# ========================================

"""
    _get_recipe_data(itp) -> (x::AbstractVector, y::AbstractVector, extrap::Val)

Extract grid, values, and extrapolation mode from any single-series interpolant.
Unified interface handles field name differences across types.
"""
_get_recipe_data(itp::LinearInterpolant) = (itp.x, itp.y, itp.extrap)
_get_recipe_data(itp::ConstantInterpolant) = (itp.x, itp.y, itp.extrap)
_get_recipe_data(itp::QuadraticInterpolant) = (itp.x, itp.y, itp.extrap)
_get_recipe_data(itp::CubicInterpolant) = (itp.cache.x, itp.y, itp.extrap)

"""
    compute_marker_size(n; max_size=7.0, min_size=3.0, max_n=100) -> Float64

Compute marker size based on number of data points.
More points → smaller markers to avoid visual clutter.
"""
function compute_marker_size(n::Integer; max_size::Float64=7.0, min_size::Float64=3.0, max_n::Integer=100)
    s = max_size - (max_size - min_size) / max_n * n
    return clamp(s, min_size, max_size)
end

"""
    compute_marker_alpha(n; max_α=0.75, min_α=0.3, max_n=100) -> Float64

Compute marker alpha (opacity) based on number of data points.
More points → more transparent markers to reduce visual density.
"""
function compute_marker_alpha(n::Integer; max_α::Float64=0.75, min_α::Float64=0.3, max_n::Integer=100)
    a = max_α - (max_α - min_α) / max_n * n
    return clamp(a, min_α, max_α)
end

"""
Threshold for automatic scatter visibility.
If data count >= this value, scatter is hidden by default.
User can override with `show_data=true`.
"""
const SCATTER_THRESHOLD = 200

"""
    _get_series_recipe_data(sitp) -> (x::Vector, Y::Matrix, extrap::Val, n_series::Int)

Extract data from multi-series interpolants.
"""
function _get_series_recipe_data(sitp::LinearSeriesInterpolant)
    (sitp.x, sitp.y, sitp.extrap, size(sitp.y, 2))
end
function _get_series_recipe_data(sitp::ConstantSeriesInterpolant)
    (sitp.x, sitp.y, sitp.extrap, size(sitp.y, 2))
end
function _get_series_recipe_data(sitp::QuadraticSeriesInterpolant)
    (sitp.x, sitp.y, sitp.extrap, size(sitp.y, 2))
end
function _get_series_recipe_data(sitp::CubicSeriesInterpolant)
    (sitp.cache.x, sitp.y, sitp.extrap, size(sitp.y, 2))
end

"""
    _interpolant_label(itp) -> String

Return human-readable label for interpolant type (method name only).
"""
_interpolant_label(::LinearInterpolant) = "linear"
_interpolant_label(::ConstantInterpolant) = "constant"
_interpolant_label(::QuadraticInterpolant) = "quadratic"
_interpolant_label(::CubicInterpolant) = "cubic"
_interpolant_label(::LinearSeriesInterpolant) = "linear"
_interpolant_label(::ConstantSeriesInterpolant) = "constant"
_interpolant_label(::QuadraticSeriesInterpolant) = "quadratic"
_interpolant_label(::CubicSeriesInterpolant) = "cubic"

"""
    _default_samples(x) -> Int

Compute default sample count: clamp(50*(n-1), 200, 2000)
"""
_default_samples(x::AbstractVector) = clamp(50 * (length(x) - 1), 200, 2000)

"""
    _default_margin(x) -> eltype(x)

Compute default domain margin: 0.25 * (x[end] - x[1])
"""
_default_margin(x::AbstractVector{T}) where {T} = T(0.25) * (last(x) - first(x))

# ========================================
# Single-Series Interpolant Recipe
# ========================================

"""
Recipe for single-series interpolants.

Generates multiple series:
1. Out-of-domain shading (vspan)
2. Boundary vertical lines
3. Original data scatter
4. Interpolation curve

# Keyword Arguments
- `show_data::Union{Bool,Nothing} = nothing`: Show original data points.
  If `nothing` (default), auto-determines: hidden when n ≥ $(SCATTER_THRESHOLD), shown otherwise.
  Set `true`/`false` to override.
- `show_bounds::Bool = true`: Show boundary lines
- `show_outside::Bool = true`: Show out-of-domain shading
- `samples::Int`: Curve sample count (default: auto)
- `domain_margin`: Domain extension for visualization (default: auto)
"""
@recipe function f(itp::AbstractInterpolant{T}) where {T}

    # Skip if SeriesInterpolant (handled by separate recipe)
    if itp isa AbstractSeriesInterpolant
        return
    end

    # Extract custom options from plotattributes
    show_data_opt = pop!(plotattributes, :show_data, nothing)
    show_bounds = pop!(plotattributes, :show_bounds, true)
    show_outside = pop!(plotattributes, :show_outside, true)
    samples = pop!(plotattributes, :samples, nothing)
    domain_margin = pop!(plotattributes, :domain_margin, nothing)
    user_xlims = pop!(plotattributes, :xlims, nothing)

    # Extract data
    x, y, extrap = _get_recipe_data(itp)
    x_vec = collect(x)
    y_vec = collect(y)

    # Compute defaults
    n_samples = isnothing(samples) ? _default_samples(x_vec) : samples
    margin = isnothing(domain_margin) ? _default_margin(x_vec) : T(domain_margin)

    x_min, x_max = first(x_vec), last(x_vec)
    label_str = _interpolant_label(itp)

    # Small visual margin for edge point visibility (2% of domain)
    visual_margin = T(0.02) * (x_max - x_min)

    # Compute curve evaluation range (xq) and display range (xlims) separately
    if extrap === Val(:none)
        # Curve stays within domain, but xlims has small margin for visibility
        xq_min, xq_max = x_min, x_max
        default_xlim_min = x_min - visual_margin
        default_xlim_max = x_max + visual_margin
    else
        # Curve extends beyond domain
        xq_min, xq_max = x_min - margin, x_max + margin
        default_xlim_min, default_xlim_max = xq_min, xq_max
    end

    # If user provides xlims and extrap is enabled, use wider range for curve evaluation
    if !isnothing(user_xlims) && extrap !== Val(:none)
        xq_min = min(T(first(user_xlims)), xq_min)
        xq_max = max(T(last(user_xlims)), xq_max)
    end

    # Generate query range
    xq = range(xq_min, xq_max; length=n_samples)

    # Final xlims for plot (user override or default)
    final_xlims = isnothing(user_xlims) ? (default_xlim_min, default_xlim_max) : user_xlims

    # Evaluate interpolant
    yq = itp.(collect(xq))

    # Set plot defaults
    legend --> :best
    legendfontsize --> 11
    tickfontsize --> 12

    # Compute data-dependent marker properties
    n_data = length(x_vec)

    # Determine show_data: auto-hide for large datasets, user can override
    show_data = isnothing(show_data_opt) ? (n_data < SCATTER_THRESHOLD) : show_data_opt

    # Compute ylims based on extrapolation mode:
    # - extrap=:none → use only original data (y_vec)
    # - extrap enabled → use both data and extrapolated curve (yq) for balanced view
    if extrap === Val(:none)
        y_for_lims = y_vec
    else
        y_for_lims = vcat(y_vec, yq)
    end
    y_range = maximum(y_for_lims) - minimum(y_for_lims)
    y_margin = max(y_range * 0.05, eps(T))  # 5% margin
    y_lim_min = minimum(y_for_lims) - y_margin
    y_lim_max = maximum(y_for_lims) + y_margin

    # Series 1 & 2: Out-of-domain shading (only when extrapolation is enabled)
    # Shading extends from data domain boundary to xq_min/xq_max (respects user xlims)
    if show_outside && extrap !== Val(:none)
        @series begin
            seriestype := :shape
            fillcolor --> :gray
            fillalpha --> 0.1
            linewidth := 0
            label := nothing
            [xq_min, x_min, x_min, xq_min],
            [y_lim_min, y_lim_min, y_lim_max, y_lim_max]
        end
        @series begin
            seriestype := :shape
            fillcolor --> :gray
            fillalpha --> 0.1
            linewidth := 0
            label --> "out of domain"
            [x_max, xq_max, xq_max, x_max],
            [y_lim_min, y_lim_min, y_lim_max, y_lim_max]
        end
    end

    # Series 3: Boundary lines
    if show_bounds
        @series begin
            seriestype := :vline
            color --> :gray
            linestyle --> :dot
            alpha --> 0.5
            linewidth --> 1
            label := nothing
            [x_min, x_max]
        end
    end

    # Series 4: Data scatter (user-overridable styling with data-dependent defaults)
    if show_data
        @series begin
            seriestype := :scatter
            color --> :blue
            markersize --> compute_marker_size(n_data)
            markeralpha --> compute_marker_alpha(n_data)
            markerstrokewidth --> 0
            label --> "data"
            x_vec, y_vec
        end
    end

    # Series 5: Interpolation curve (primary, user-overridable styling)
    @series begin
        seriestype := :path
        color --> :blue
        linewidth --> 2
        label --> "$(label_str) (S)"
        xlims --> final_xlims
        ylims --> (y_lim_min, y_lim_max)
        collect(xq), yq
    end
end

# ========================================
# Multi-Series Interpolant Recipe
# ========================================

@recipe function f(sitp::AbstractSeriesInterpolant{T}) where {T}

    # Extract custom options from plotattributes
    show_data_opt = pop!(plotattributes, :show_data, nothing)
    show_bounds = pop!(plotattributes, :show_bounds, true)
    show_outside = pop!(plotattributes, :show_outside, true)
    samples = pop!(plotattributes, :samples, nothing)
    domain_margin = pop!(plotattributes, :domain_margin, nothing)
    series_idx = pop!(plotattributes, :series_idx, :all)
    user_xlims = pop!(plotattributes, :xlims, nothing)

    # Extract data
    x, Y, extrap, n_ser = _get_series_recipe_data(sitp)
    x_vec = collect(x)

    # Compute defaults
    n_samples = isnothing(samples) ? _default_samples(x_vec) : samples
    margin = isnothing(domain_margin) ? _default_margin(x_vec) : T(domain_margin)

    x_min, x_max = first(x_vec), last(x_vec)
    label_base = _interpolant_label(sitp)

    # Determine which series to plot
    series_indices = if series_idx === :all
        1:n_ser
    elseif series_idx === :first
        1:1
    elseif series_idx isa Int
        series_idx:series_idx
    else
        series_idx
    end

    # Small visual margin for edge point visibility (2% of domain)
    visual_margin = T(0.02) * (x_max - x_min)

    # Compute curve evaluation range (xq) and display range (xlims) separately
    if extrap === Val(:none)
        # Curve stays within domain, but xlims has small margin for visibility
        xq_min, xq_max = x_min, x_max
        default_xlim_min = x_min - visual_margin
        default_xlim_max = x_max + visual_margin
    else
        # Curve extends beyond domain
        xq_min, xq_max = x_min - margin, x_max + margin
        default_xlim_min, default_xlim_max = xq_min, xq_max
    end

    # If user provides xlims and extrap is enabled, use wider range for curve evaluation
    if !isnothing(user_xlims) && extrap !== Val(:none)
        xq_min = min(T(first(user_xlims)), xq_min)
        xq_max = max(T(last(user_xlims)), xq_max)
    end

    # Generate query range
    xq = range(xq_min, xq_max; length=n_samples)
    xq_vec = collect(xq)

    # Final xlims for plot (user override or default)
    final_xlims = isnothing(user_xlims) ? (default_xlim_min, default_xlim_max) : user_xlims

    # Evaluate all series
    yq_all = [sitp(xi) for xi in xq_vec]
    yq_matrix = reduce(hcat, yq_all)'  # n_samples x n_series

    # Compute ylims based on extrapolation mode:
    # - extrap=:none → use only original data (Y)
    # - extrap enabled → use both data and extrapolated curve for balanced view
    if extrap === Val(:none)
        y_for_lims = vec(Y)
    else
        y_for_lims = vcat(vec(Y), vec(yq_matrix))
    end
    y_range = maximum(y_for_lims) - minimum(y_for_lims)
    y_margin = max(y_range * 0.05, eps(T))  # 5% margin
    y_lim_min = minimum(y_for_lims) - y_margin
    y_lim_max = maximum(y_for_lims) + y_margin

    # Compute data-dependent marker properties
    n_data = length(x_vec)

    # Determine show_data: auto-hide for large datasets, user can override
    show_data = isnothing(show_data_opt) ? (n_data < SCATTER_THRESHOLD) : show_data_opt

    legend --> :best
    legendfontsize --> 11
    tickfontsize --> 12

    # Out-of-domain shading (once, internal series - use := for type, --> for colors)
    if show_outside && length(series_indices) > 0
        if extrap !== Val(:none)
            @series begin
                seriestype := :shape
                fillcolor --> :gray
                fillalpha --> 0.1
                linewidth := 0
                label := nothing
                [xq_min, x_min, x_min, xq_min],
                [y_lim_min, y_lim_min, y_lim_max, y_lim_max]
            end
            @series begin
                seriestype := :shape
                fillcolor --> :gray
                fillalpha --> 0.1
                linewidth := 0
                label --> "out of domain"
                [x_max, xq_max, xq_max, x_max],
                [y_lim_min, y_lim_min, y_lim_max, y_lim_max]
            end
        end
    end

    # Boundary lines (once)
    if show_bounds
        @series begin
            seriestype := :vline
            color --> :gray
            linestyle --> :dot
            alpha --> 0.5
            linewidth --> 1
            label := nothing
            [x_min, x_max]
        end
    end

    # Default color palette for multi-series (user can override with seriescolor)
    default_palette = [:blue, :red, :green, :orange, :purple, :cyan, :magenta, :brown]

    # Data and curves for each series
    for (plot_idx, k) in enumerate(series_indices)
        # Use consistent color for both scatter and line within each series
        series_color = default_palette[mod1(plot_idx, length(default_palette))]

        if show_data
            @series begin
                seriestype := :scatter
                color --> series_color
                markersize --> compute_marker_size(n_data)
                markeralpha --> compute_marker_alpha(n_data)
                markerstrokewidth --> 0
                label --> (plot_idx == 1 ? "data" : nothing)
                x_vec, Y[:, k]
            end
        end

        @series begin
            seriestype := :path
            color --> series_color
            linewidth --> 2
            label --> (n_ser > 1 ? "$label_base (S) [$k]" : "$label_base (S)")
            xlims --> final_xlims
            ylims --> (y_lim_min, y_lim_max)
            xq_vec, yq_matrix[:, k]
        end
    end
end

# ========================================
# DerivativeView Recipe
# ========================================

@recipe function f(dv::DerivativeView{Order, ITP}) where {Order, ITP}

    # Extract custom options from plotattributes
    show_data = pop!(plotattributes, :show_data, false)  # Default false for derivatives
    show_bounds = pop!(plotattributes, :show_bounds, true)
    show_outside = pop!(plotattributes, :show_outside, true)
    samples = pop!(plotattributes, :samples, nothing)
    domain_margin = pop!(plotattributes, :domain_margin, nothing)
    user_xlims = pop!(plotattributes, :xlims, nothing)

    parent = dv.parent

    # Get data from parent
    if parent isa AbstractSeriesInterpolant
        x, Y, extrap, _ = _get_series_recipe_data(parent)
    else
        x, _, extrap = _get_recipe_data(parent)
    end

    x_vec = collect(x)
    ElType = eltype(x_vec)

    # Compute defaults
    n_samples = isnothing(samples) ? _default_samples(x_vec) : samples
    margin = isnothing(domain_margin) ? _default_margin(x_vec) : ElType(domain_margin)

    x_min, x_max = first(x_vec), last(x_vec)

    # Small visual margin for edge point visibility (2% of domain)
    visual_margin = ElType(0.02) * (x_max - x_min)

    # Compute curve evaluation range (xq) and display range (xlims) separately
    if extrap === Val(:none)
        # Curve stays within domain, but xlims has small margin for visibility
        xq_min, xq_max = x_min, x_max
        default_xlim_min = x_min - visual_margin
        default_xlim_max = x_max + visual_margin
    else
        # Curve extends beyond domain
        xq_min, xq_max = x_min - margin, x_max + margin
        default_xlim_min, default_xlim_max = xq_min, xq_max
    end

    # If user provides xlims and extrap is enabled, use wider range for curve evaluation
    if !isnothing(user_xlims) && extrap !== Val(:none)
        xq_min = min(ElType(first(user_xlims)), xq_min)
        xq_max = max(ElType(last(user_xlims)), xq_max)
    end

    # Generate query range
    xq = range(xq_min, xq_max; length=n_samples)
    xq_vec = collect(xq)

    # Final xlims for plot (user override or default)
    final_xlims = isnothing(user_xlims) ? (default_xlim_min, default_xlim_max) : user_xlims

    # Evaluate derivative
    yq = dv.(xq_vec)

    # Determine label with explicit derivative notation (S', S'', S''')
    deriv_notation = Order == 1 ? "S'" : Order == 2 ? "S''" : "S'''"
    base_label = _interpolant_label(parent)
    final_label = "$(base_label) ($(deriv_notation))"

    legend --> :best
    legendfontsize --> 11
    tickfontsize --> 12

    # Compute ylims from evaluated derivative (yq already respects extrap mode via xq range)
    y_range = maximum(yq) - minimum(yq)
    y_margin = max(y_range * 0.05, eps(ElType))  # 5% margin
    y_lim_min = minimum(yq) - y_margin
    y_lim_max = maximum(yq) + y_margin

    # Out-of-domain shading (respects user xlims)
    if show_outside && extrap !== Val(:none)
        @series begin
            seriestype := :shape
            fillcolor --> :gray
            fillalpha --> 0.1
            linewidth := 0
            label := nothing
            [xq_min, x_min, x_min, xq_min],
            [y_lim_min, y_lim_min, y_lim_max, y_lim_max]
        end
        @series begin
            seriestype := :shape
            fillcolor --> :gray
            fillalpha --> 0.1
            linewidth := 0
            label --> "out of domain"
            [x_max, xq_max, xq_max, x_max],
            [y_lim_min, y_lim_min, y_lim_max, y_lim_max]
        end
    end

    # Boundary lines
    if show_bounds
        @series begin
            seriestype := :vline
            color --> :gray
            linestyle --> :dot
            alpha --> 0.5
            linewidth --> 1
            label := nothing
            [x_min, x_max]
        end
    end

    # Derivative curve (user-overridable styling)
    @series begin
        seriestype := :path
        color --> :steelblue
        linewidth --> 2
        linestyle --> :dash
        label --> final_label
        xlims --> final_xlims
        ylims --> (y_lim_min, y_lim_max)
        xq_vec, yq
    end
end

end # module
