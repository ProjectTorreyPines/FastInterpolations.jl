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
import HelpPlots: assert_type_and_record_argument, recipe_dispatch

# Import types for dispatch
import FastInterpolations:
    AbstractInterpolant, AbstractSeriesInterpolant, AbstractDerivativeView,
    AbstractInterpolantND, CubicInterpolantND,
    LinearInterpolantND, ConstantInterpolantND, QuadraticInterpolantND,
    LinearInterpolant, ConstantInterpolant, QuadraticInterpolant, CubicInterpolant,
    LinearSeriesInterpolant, ConstantSeriesInterpolant,
    QuadraticSeriesInterpolant, CubicSeriesInterpolant,
    DerivativeView, NoExtrap

# ========================================
# Helper Functions
# ========================================

"""
    _get_recipe_data(itp) -> (x::AbstractVector, y::AbstractVector, extrap::AbstractExtrap)

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
# HelpPlots Integration (recipe_dispatch)
# ========================================

# Define dispatch names for help_plot discovery
recipe_dispatch(::AbstractInterpolant) = "AbstractInterpolant"
recipe_dispatch(::AbstractSeriesInterpolant) = "AbstractSeriesInterpolant"
recipe_dispatch(::DerivativeView) = "DerivativeView"

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
    user_ylims = pop!(plotattributes, :ylims, nothing)

    # Record arguments for help_plot discovery
    dispatch_name = recipe_dispatch(itp)
    assert_type_and_record_argument(dispatch_name, Union{Bool,Nothing},
        "Show original data points (default: auto, hidden when n ≥ $(SCATTER_THRESHOLD))"; show_data=show_data_opt)
    assert_type_and_record_argument(dispatch_name, Bool,
        "Show domain boundary lines (default: true)"; show_bounds)
    assert_type_and_record_argument(dispatch_name, Bool,
        "Show out-of-domain shading (default: true)"; show_outside)
    assert_type_and_record_argument(dispatch_name, Union{Integer,Nothing},
        "Number of curve samples (nothing = auto-computed from grid size)"; samples)
    assert_type_and_record_argument(dispatch_name, Union{Real,Nothing},
        "Domain extension for extrapolation visualization (nothing = 25% of domain)"; domain_margin)

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
    if extrap isa NoExtrap
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
    if !isnothing(user_xlims) && !(extrap isa NoExtrap)
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
    # User-provided ylims override auto-computed limits
    if !isnothing(user_ylims)
        y_lim_min, y_lim_max = T(first(user_ylims)), T(last(user_ylims))
    else
        if extrap isa NoExtrap
            y_for_lims = y_vec
        else
            y_for_lims = vcat(y_vec, yq)
        end
        y_range = maximum(y_for_lims) - minimum(y_for_lims)
        y_margin = max(y_range * 0.05, eps(T))  # 5% margin
        y_lim_min = minimum(y_for_lims) - y_margin
        y_lim_max = maximum(y_for_lims) + y_margin
    end

    # Series 1 & 2: Out-of-domain shading (only when extrapolation is enabled)
    # Use large values so shading auto-clips to actual xlims/ylims
    shade_min, shade_max = T(-1e10), T(1e10)
    if show_outside && !(extrap isa NoExtrap)
        @series begin
            seriestype := :shape
            fillcolor --> :gray
            fillalpha --> 0.1
            linewidth := 0
            label := nothing
            [shade_min, x_min, x_min, shade_min],
            [shade_min, shade_min, shade_max, shade_max]
        end
        @series begin
            seriestype := :shape
            fillcolor --> :gray
            fillalpha --> 0.1
            linewidth := 0
            label --> "out of domain"
            [x_max, shade_max, shade_max, x_max],
            [shade_min, shade_min, shade_max, shade_max]
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
    user_ylims = pop!(plotattributes, :ylims, nothing)

    # Record arguments for help_plot discovery
    dispatch_name = recipe_dispatch(sitp)
    assert_type_and_record_argument(dispatch_name, Union{Bool,Nothing},
        "Show original data points (default: auto, hidden when n ≥ $(SCATTER_THRESHOLD))"; show_data=show_data_opt)
    assert_type_and_record_argument(dispatch_name, Bool,
        "Show domain boundary lines (default: true)"; show_bounds)
    assert_type_and_record_argument(dispatch_name, Bool,
        "Show out-of-domain shading (default: true)"; show_outside)
    assert_type_and_record_argument(dispatch_name, Union{Integer,Nothing},
        "Number of curve samples (nothing = auto-computed from grid size)"; samples)
    assert_type_and_record_argument(dispatch_name, Union{Real,Nothing},
        "Domain extension for extrapolation visualization (nothing = 25% of domain)"; domain_margin)
    assert_type_and_record_argument(dispatch_name, Union{Symbol,Integer,AbstractRange,AbstractVector{<:Integer}},
        "Which series to plot: :all, :first, Int, range, or vector of indices (default: :all)"; series_idx)

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
    if extrap isa NoExtrap
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
    if !isnothing(user_xlims) && !(extrap isa NoExtrap)
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
    # User-provided ylims override auto-computed limits
    if !isnothing(user_ylims)
        y_lim_min, y_lim_max = T(first(user_ylims)), T(last(user_ylims))
    else
        if extrap isa NoExtrap
            y_for_lims = vec(Y)
        else
            y_for_lims = vcat(vec(Y), vec(yq_matrix))
        end
        y_range = maximum(y_for_lims) - minimum(y_for_lims)
        y_margin = max(y_range * 0.05, eps(T))  # 5% margin
        y_lim_min = minimum(y_for_lims) - y_margin
        y_lim_max = maximum(y_for_lims) + y_margin
    end

    # Compute data-dependent marker properties
    n_data = length(x_vec)

    # Determine show_data: auto-hide for large datasets, user can override
    show_data = isnothing(show_data_opt) ? (n_data < SCATTER_THRESHOLD) : show_data_opt

    legend --> :best
    legendfontsize --> 11
    tickfontsize --> 12

    # Out-of-domain shading (once, internal series - use := for type, --> for colors)
    # Use large values so shading auto-clips to actual xlims/ylims
    shade_min, shade_max = T(-1e10), T(1e10)
    if show_outside && length(series_indices) > 0
        if !(extrap isa NoExtrap)
            @series begin
                seriestype := :shape
                fillcolor --> :gray
                fillalpha --> 0.1
                linewidth := 0
                label := nothing
                [shade_min, x_min, x_min, shade_min],
                [shade_min, shade_min, shade_max, shade_max]
            end
            @series begin
                seriestype := :shape
                fillcolor --> :gray
                fillalpha --> 0.1
                linewidth := 0
                label --> "out of domain"
                [x_max, shade_max, shade_max, x_max],
                [shade_min, shade_min, shade_max, shade_max]
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

    # ND derivative views are handled by the 2D ND recipe (via type recipe forwarding)
    if dv.parent isa AbstractInterpolantND
        return
    end

    # Extract custom options from plotattributes
    show_data = pop!(plotattributes, :show_data, false)  # Default false for derivatives
    show_bounds = pop!(plotattributes, :show_bounds, true)
    show_outside = pop!(plotattributes, :show_outside, true)
    samples = pop!(plotattributes, :samples, nothing)
    domain_margin = pop!(plotattributes, :domain_margin, nothing)
    user_xlims = pop!(plotattributes, :xlims, nothing)
    user_ylims = pop!(plotattributes, :ylims, nothing)

    # Record arguments for help_plot discovery
    dispatch_name = recipe_dispatch(dv)
    assert_type_and_record_argument(dispatch_name, Bool,
        "Show original data points (default: false for derivatives)"; show_data)
    assert_type_and_record_argument(dispatch_name, Bool,
        "Show domain boundary lines (default: true)"; show_bounds)
    assert_type_and_record_argument(dispatch_name, Bool,
        "Show out-of-domain shading (default: true)"; show_outside)
    assert_type_and_record_argument(dispatch_name, Union{Integer,Nothing},
        "Number of curve samples (nothing = auto-computed from grid size)"; samples)
    assert_type_and_record_argument(dispatch_name, Union{Real,Nothing},
        "Domain extension for extrapolation visualization (nothing = 25% of domain)"; domain_margin)

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
    if extrap isa NoExtrap
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
    if !isnothing(user_xlims) && !(extrap isa NoExtrap)
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
    # User-provided ylims override auto-computed limits
    if !isnothing(user_ylims)
        y_lim_min, y_lim_max = ElType(first(user_ylims)), ElType(last(user_ylims))
    else
        y_range = maximum(yq) - minimum(yq)
        y_margin = max(y_range * 0.05, eps(ElType))  # 5% margin
        y_lim_min = minimum(yq) - y_margin
        y_lim_max = maximum(yq) + y_margin
    end

    # Out-of-domain shading - use large values so shading auto-clips to actual xlims/ylims
    shade_min, shade_max = ElType(-1e10), ElType(1e10)
    if show_outside && !(extrap isa NoExtrap)
        @series begin
            seriestype := :shape
            fillcolor --> :gray
            fillalpha --> 0.1
            linewidth := 0
            label := nothing
            [shade_min, x_min, x_min, shade_min],
            [shade_min, shade_min, shade_max, shade_max]
        end
        @series begin
            seriestype := :shape
            fillcolor --> :gray
            fillalpha --> 0.1
            linewidth := 0
            label --> "out of domain"
            [x_max, shade_max, shade_max, x_max],
            [shade_min, shade_min, shade_max, shade_max]
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

# ========================================
# 2D N-Dimensional Interpolant Recipe (AbstractInterpolantND with N=2)
# ========================================

# HelpPlots dispatch for ND interpolants
recipe_dispatch(::AbstractInterpolantND) = "AbstractInterpolantND"

# Helper: interpolant label for ND types
_interpolant_label(::LinearInterpolantND) = "linear"
_interpolant_label(::ConstantInterpolantND) = "constant"
_interpolant_label(::QuadraticInterpolantND) = "quadratic"
_interpolant_label(::CubicInterpolantND) = "cubic"

# Helper: default high-resolution samples for 2D visualization
_default_2d_samples(nx::Integer, ny::Integer) = (
    clamp(10 * nx, 100, 500),
    clamp(10 * ny, 100, 500)
)

"""
Threshold for automatic node/gridline visibility in 2D plots.
If total node count (nx × ny) >= this value, nodes/gridlines are hidden by default.
User can override with `show_nodes=true` or `show_gridlines=true`.
"""
const SCATTER_THRESHOLD_2D = 400  # 20×20 grid

"""
    compute_marker_size_2d(n; max_size=6.0, min_size=2.0, max_n=400) -> Float64

Compute marker size for 2D plots based on total node count.
More nodes → smaller markers to avoid visual clutter.
"""
function compute_marker_size_2d(n::Integer; max_size::Float64=6.0, min_size::Float64=2.0, max_n::Integer=400)
    s = max_size - (max_size - min_size) / max_n * n
    return clamp(s, min_size, max_size)
end

"""
    compute_marker_alpha_2d(n; max_α=0.85, min_α=0.4, max_n=400) -> Float64

Compute marker alpha (opacity) for 2D plots based on total node count.
More nodes → more transparent markers to reduce visual density.
"""
function compute_marker_alpha_2d(n::Integer; max_α::Float64=0.85, min_α::Float64=0.4, max_n::Integer=400)
    a = max_α - (max_α - min_α) / max_n * n
    return clamp(a, min_α, max_α)
end

"""
    compute_gridline_alpha_2d(n_lines; max_α=0.4, min_α=0.1, max_n=50) -> Float64

Compute gridline alpha based on number of grid lines.
More lines → more transparent to reduce visual clutter.
"""
function compute_gridline_alpha_2d(n_lines::Integer; max_α::Float64=0.4, min_α::Float64=0.1, max_n::Integer=50)
    a = max_α - (max_α - min_α) / max_n * n_lines
    return clamp(a, min_α, max_α)
end

"""
    _default_2d_margin(grid) -> eltype(grid)

Compute default domain margin for 2D extrapolation visualization: 15% of span.
Smaller than 1D (25%) since 2D visual space is more constrained.
"""
_default_2d_margin(grid) = eltype(grid)(0.15) * (last(grid) - first(grid))

"""
    _has_extrap(itp::AbstractInterpolantND) -> Bool

Check if any axis of an ND interpolant has extrapolation enabled (not `:none`).
"""
_has_extrap(itp::AbstractInterpolantND) = any(e -> !(e isa NoExtrap), itp.extraps)

"""
    _axis_has_extrap(itp::AbstractInterpolantND, d::Int) -> Bool

Check if axis `d` has extrapolation enabled.
"""
_axis_has_extrap(itp::AbstractInterpolantND, d::Int) = !(itp.extraps[d] isa NoExtrap)

"""
Recipe for 2D N-dimensional interpolants (AbstractInterpolantND with N=2).

Generates a visualization with:
1. High-resolution heatmap of interpolated values (extended when extrapolation enabled)
2. Domain boundary rectangle (dashed, when extrapolation enabled)
3. Scatter plot of original grid nodes (auto-hidden for large grids)
4. Dashed grid lines connecting nodes (auto-hidden for large grids)

# Keyword Arguments
- `show_nodes::Union{Bool,Nothing} = nothing`: Show grid node markers.
  If `nothing` (default), auto-determines: hidden when nx×ny ≥ $(SCATTER_THRESHOLD_2D).
- `show_gridlines::Union{Bool,Nothing} = nothing`: Show grid lines.
  If `nothing` (default), auto-determines: hidden when nx×ny ≥ $(SCATTER_THRESHOLD_2D).
- `show_boundary::Bool = true`: Show domain boundary rectangle when extrapolation is enabled.
- `domain_margin::Union{Real,Nothing} = nothing`: Extension margin for extrapolation (per axis).
  Default: 15% of each axis span.
- `resolution::Union{Tuple{Int,Int}, Nothing} = nothing`: Heatmap resolution (nx, ny)
- `equal_aspect::Bool = false`: Use equal aspect ratio (default: false, figure is already near-square)
- `clims_padding::Real = 0.02`: Padding for color limits (fraction of data range)
- `node_color = :white`: Color for grid node markers
- `node_size::Union{Real, Nothing} = nothing`: Marker size (nothing = auto based on grid size)
- `node_alpha::Union{Real, Nothing} = nothing`: Marker transparency (nothing = auto)
- `gridline_color = :white`: Color for grid lines
- `gridline_alpha::Union{Real, Nothing} = nothing`: Grid line transparency (nothing = auto)
- `gridline_style = :dash`: Line style for grid lines (:dash, :dot, :solid)
- `boundary_color = :white`: Color for domain boundary lines
- `boundary_width::Real = 2.5`: Line width for domain boundary
- `boundary_style = :solid`: Line style for domain boundary
- `boundary_alpha::Real = 0.9`: Alpha for domain boundary lines
"""
@recipe function f(itp::AbstractInterpolantND{Tg, Tv, 2}) where {Tg, Tv}

    # Extract custom options from plotattributes
    show_nodes_opt = pop!(plotattributes, :show_nodes, nothing)
    show_gridlines_opt = pop!(plotattributes, :show_gridlines, nothing)
    show_boundary_opt = pop!(plotattributes, :show_boundary, true)
    domain_margin_opt = pop!(plotattributes, :domain_margin, nothing)
    user_xlims = get(plotattributes, :xlims, nothing)
    user_ylims = get(plotattributes, :ylims, nothing)
    resolution = pop!(plotattributes, :resolution, nothing)
    equal_aspect = pop!(plotattributes, :equal_aspect, false)
    clims_padding = pop!(plotattributes, :clims_padding, 0.02)
    user_clims = pop!(plotattributes, :clims, nothing)
    node_color = pop!(plotattributes, :node_color, :white)
    node_size_opt = pop!(plotattributes, :node_size, nothing)
    node_alpha_opt = pop!(plotattributes, :node_alpha, 0.6)
    gridline_color = pop!(plotattributes, :gridline_color, :white)
    gridline_alpha_opt = pop!(plotattributes, :gridline_alpha, 0.6)
    gridline_style = pop!(plotattributes, :gridline_style, :dot)
    boundary_color = pop!(plotattributes, :boundary_color, :white)
    boundary_width = pop!(plotattributes, :boundary_width, 2.5)
    boundary_style = pop!(plotattributes, :boundary_style, :solid)
    boundary_alpha = pop!(plotattributes, :boundary_alpha, 0.9)

    # Record arguments for help_plot discovery
    dispatch_name = recipe_dispatch(itp)
    assert_type_and_record_argument(dispatch_name, Union{Bool, Nothing},
        "Show grid node markers (default: auto, hidden when nx×ny ≥ $(SCATTER_THRESHOLD_2D))"; show_nodes=show_nodes_opt)
    assert_type_and_record_argument(dispatch_name, Union{Bool, Nothing},
        "Show grid lines (default: auto, hidden when nx×ny ≥ $(SCATTER_THRESHOLD_2D))"; show_gridlines=show_gridlines_opt)
    assert_type_and_record_argument(dispatch_name, Bool,
        "Show domain boundary when extrapolation enabled (default: true)"; show_boundary=show_boundary_opt)
    assert_type_and_record_argument(dispatch_name, Union{Real, Nothing},
        "Extension margin for extrapolation visualization (nothing = 15% of axis span)"; domain_margin=domain_margin_opt)
    assert_type_and_record_argument(dispatch_name, Union{Tuple{Int,Int}, Nothing},
        "Heatmap resolution (nx, ny), nothing for auto"; resolution)
    assert_type_and_record_argument(dispatch_name, Bool,
        "Use equal aspect ratio (default: false)"; equal_aspect)
    assert_type_and_record_argument(dispatch_name, Real,
        "Padding for color limits as fraction of data range (default: 0.02)"; clims_padding)
    assert_type_and_record_argument(dispatch_name, Any,
        "Color for grid node markers (default: :white)"; node_color)
    assert_type_and_record_argument(dispatch_name, Union{Real, Nothing},
        "Marker size (nothing = auto based on grid size)"; node_size=node_size_opt)
    assert_type_and_record_argument(dispatch_name, Union{Real, Nothing},
        "Marker transparency (nothing = auto based on grid size)"; node_alpha=node_alpha_opt)
    assert_type_and_record_argument(dispatch_name, Any,
        "Color for grid lines (default: :white)"; gridline_color)
    assert_type_and_record_argument(dispatch_name, Union{Real, Nothing},
        "Grid line transparency (nothing = auto based on grid size)"; gridline_alpha=gridline_alpha_opt)
    assert_type_and_record_argument(dispatch_name, Symbol,
        "Line style for grid lines: :dash, :dot, :solid (default: :dash)"; gridline_style)
    assert_type_and_record_argument(dispatch_name, Any,
        "Color for domain boundary lines (default: :white)"; boundary_color)
    assert_type_and_record_argument(dispatch_name, Real,
        "Line width for domain boundary (default: 2.5)"; boundary_width)
    assert_type_and_record_argument(dispatch_name, Symbol,
        "Line style for domain boundary: :solid, :dash, :dot (default: :solid)"; boundary_style)
    assert_type_and_record_argument(dispatch_name, Real,
        "Alpha for domain boundary lines (default: 0.9)"; boundary_alpha)

    # Extract grids from interpolant
    x_grid = collect(itp.grids[1])
    y_grid = collect(itp.grids[2])
    nx, ny = length(x_grid), length(y_grid)
    n_total = nx * ny  # Total number of grid nodes

    # Domain boundaries
    x_min, x_max = first(x_grid), last(x_grid)
    y_min, y_max = first(y_grid), last(y_grid)

    # Check per-axis extrapolation
    has_extrap_x = _axis_has_extrap(itp, 1)
    has_extrap_y = _axis_has_extrap(itp, 2)
    has_any_extrap = has_extrap_x || has_extrap_y

    # Compute per-axis margins for extrapolation extension
    margin_x = has_extrap_x ? (isnothing(domain_margin_opt) ? _default_2d_margin(x_grid) : Tg(domain_margin_opt)) : zero(Tg)
    margin_y = has_extrap_y ? (isnothing(domain_margin_opt) ? _default_2d_margin(y_grid) : Tg(domain_margin_opt)) : zero(Tg)

    # Evaluation bounds: default margin, extended by user xlims/ylims when extrap is enabled
    eval_x_min = x_min - margin_x
    eval_x_max = x_max + margin_x
    eval_y_min = y_min - margin_y
    eval_y_max = y_max + margin_y

    if !isnothing(user_xlims) && has_extrap_x
        eval_x_min = min(eval_x_min, Tg(first(user_xlims)))
        eval_x_max = max(eval_x_max, Tg(last(user_xlims)))
    end
    if !isnothing(user_ylims) && has_extrap_y
        eval_y_min = min(eval_y_min, Tg(first(user_ylims)))
        eval_y_max = max(eval_y_max, Tg(last(user_ylims)))
    end

    # Auto-determine visibility based on grid size
    show_nodes = isnothing(show_nodes_opt) ? (n_total < SCATTER_THRESHOLD_2D) : show_nodes_opt
    show_gridlines = isnothing(show_gridlines_opt) ? (n_total < SCATTER_THRESHOLD_2D) : show_gridlines_opt

    # Compute dynamic marker properties based on grid size
    node_size = isnothing(node_size_opt) ? compute_marker_size_2d(n_total) : node_size_opt
    node_alpha = isnothing(node_alpha_opt) ? compute_marker_alpha_2d(n_total) : node_alpha_opt

    # Compute dynamic gridline alpha based on number of lines
    n_lines = nx + ny
    gridline_alpha = isnothing(gridline_alpha_opt) ? compute_gridline_alpha_2d(n_lines) : gridline_alpha_opt

    # Compute high-resolution sampling grid (extended if extrapolation enabled)
    nx_hr, ny_hr = isnothing(resolution) ? _default_2d_samples(nx, ny) : resolution
    x_hr = range(eval_x_min, eval_x_max; length=nx_hr)
    y_hr = range(eval_y_min, eval_y_max; length=ny_hr)

    # Evaluate on high-resolution grid (use _evaluator if set by DerivativeView recipe)
    # Use real() for complex values to make heatmap work
    evaluator = pop!(plotattributes, :_evaluator, itp)
    z_hr = [real(evaluator((xi, yj))) for xi in x_hr, yj in y_hr]

    # Compute color limits with padding
    z_min, z_max = extrema(z_hr)
    z_range = z_max - z_min
    if z_range < eps(typeof(z_min))
        # Constant data - add small range for visibility
        z_range = max(abs(z_min), one(typeof(z_min))) * 0.1
    end
    padding = z_range * clims_padding
    computed_clims = isnothing(user_clims) ? (z_min - padding, z_max + padding) : user_clims

    # Plot defaults — slightly wider than tall to account for colorbar
    size --> (550, 450)
    xlabel --> "x₁"
    ylabel --> "x₂"
    title --> "$(typeof(itp).name.name) ($(nx)×$(ny) grid)"

    # Aspect ratio: optional equal (default: off, size already near-square)
    if equal_aspect
        aspect_ratio --> :equal
    end

    # Set clims at plot level (not series level) to ensure it persists
    if !isnothing(computed_clims)
        clims --> computed_clims
    end

    # Series 1: Heatmap of interpolated values
    @series begin
        seriestype := :heatmap
        seriescolor --> :viridis
        colorbar --> true
        label := nothing
        collect(x_hr), collect(y_hr), z_hr'
    end

    # Series 2: Domain boundary rectangle (only when extrapolation is enabled)
    if show_boundary_opt && has_any_extrap
        @series begin
            seriestype := :path
            color --> boundary_color
            alpha --> boundary_alpha
            linestyle --> boundary_style
            linewidth --> boundary_width
            label --> "domain"
            # Closed rectangle: bottom → right → top → left → close
            Tg[x_min, x_max, x_max, x_min, x_min],
            Tg[y_min, y_min, y_max, y_max, y_min]
        end
    end

    # Series 3: Grid lines (horizontal lines at each y_grid value)
    if show_gridlines
        for yj in y_grid
            @series begin
                seriestype := :path
                color --> gridline_color
                alpha --> gridline_alpha
                linestyle --> gridline_style
                linewidth --> 1
                label := nothing
                [first(x_grid), last(x_grid)], [yj, yj]
            end
        end

        # Vertical lines at each x_grid value
        for xi in x_grid
            @series begin
                seriestype := :path
                color --> gridline_color
                alpha --> gridline_alpha
                linestyle --> gridline_style
                linewidth --> 1
                label := nothing
                [xi, xi], [first(y_grid), last(y_grid)]
            end
        end
    end

    # Series 4: Grid node scatter
    if show_nodes
        # Create mesh grid coordinates for scatter
        node_x = [xi for xi in x_grid for _ in y_grid]
        node_y = [yj for _ in x_grid for yj in y_grid]

        @series begin
            seriestype := :scatter
            markercolor --> node_color
            markersize --> node_size
            markeralpha --> node_alpha
            markerstrokewidth --> 0.5
            markerstrokecolor --> :black
            markerstrokealpha --> node_alpha
            label --> "nodes ($(nx)×$(ny))"
            node_x, node_y
        end
    end
end

# ========================================
# 2D DerivativeView Recipe (type recipe → forwards to AbstractInterpolantND recipe)
# ========================================

# Helper: format ND derivative order tuple as readable label
# e.g. (1,0) → "∂f/∂x₁", (0,1) → "∂f/∂x₂", (1,1) → "∂²f/∂x₁∂x₂", (2,0) → "∂²f/∂x₁²"
const _SUBSCRIPT_CHARS = ('₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')
const _SUPERSCRIPT_CHARS = ('¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')

_superscript(n::Int) = n in 1:length(_SUPERSCRIPT_CHARS) ? string(_SUPERSCRIPT_CHARS[n]) : "^$n"

function _deriv_label_nd(order::NTuple{N, Int}) where {N}
    total = sum(order)
    total == 0 && return "f"
    parts = String[]
    for (i, d) in enumerate(order)
        d == 0 && continue
        sub = i <= length(_SUBSCRIPT_CHARS) ? string(_SUBSCRIPT_CHARS[i]) : "$i"
        if d == 1
            push!(parts, "∂x$sub")
        else
            push!(parts, "∂x$sub$(_superscript(d))")
        end
    end
    denom = join(parts, "")
    sup_total = total == 1 ? "" : _superscript(total)
    return "∂$(sup_total)f/$denom"
end

"""
Thin type recipe for 2D DerivativeView.

Sets derivative-specific title and `_evaluator` in plotattributes, then forwards
to the `AbstractInterpolantND{Tg, Tv, 2}` recipe which handles all visualization.
"""
@recipe function f(dv::DerivativeView{Order, <:AbstractInterpolantND{Tg, Tv, 2}}) where {Order, Tg, Tv}
    itp = dv.parent
    nx, ny = length(itp.grids[1]), length(itp.grids[2])
    deriv_str = _deriv_label_nd(Order)
    base_label = _interpolant_label(itp)
    title --> "$base_label $(nx)×$(ny): $deriv_str"
    plotattributes[:_evaluator] = dv
    itp
end

end # module
