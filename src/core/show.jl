# ========================================
# Custom Show Methods for FastInterpolations.jl
# ========================================
#
# This file provides visually appealing REPL output using:
# - Unicode box-drawing characters (├─, └─)
# - Colored output via printstyled
# - Compact and verbose display modes
#
# Color Scheme:
# - Type name: cyan + bold
# - Type param (Float64): light_blue
# - Labels (Grid, Extrap): light_black
# - Numeric values: yellow
# - Symbols (:none, etc.): magenta
# - BC names: green
# - Box chars: light_black
# - Series count: light_cyan

# ========================================
# Helper Functions
# ========================================

"""Check if IO context supports color output."""
@inline _show_has_color(io::IO) = get(io, :color, false)::Bool

"""
    _show_print(io, text, color; bold=false)

Print text with color if supported, otherwise plain text.
"""
@inline function _show_print(io::IO, text, color::Symbol; bold::Bool=false)
    if _show_has_color(io)
        printstyled(io, text; color=color, bold=bold)
    else
        print(io, text)
    end
end

"""
    _show_type_header(io, typename, T; suffix="")

Print type name in cyan bold with type parameter in light_blue.
Example: `CubicInterpolant{Float64}`
"""
function _show_type_header(io::IO, typename::String, ::Type{T}; suffix::String="") where {T}
    _show_print(io, typename, :cyan; bold=true)
    _show_print(io, "{", :light_black)
    _show_print(io, string(T), :light_blue)
    _show_print(io, "}", :light_black)
    if !isempty(suffix)
        _show_print(io, suffix, :light_cyan)
    end
end

"""
    _show_row(io, is_last, label, value; value_color=:yellow)

Print a box-drawing row with label and value.
"""
function _show_row(io::IO, is_last::Bool, label::String, value::String; value_color::Symbol=:yellow)
    prefix = is_last ? "└─ " : "├─ "
    _show_print(io, prefix, :light_black)
    _show_print(io, label, :light_black)
    print(io, " ")
    _show_print(io, value, value_color)
end

# ========================================
# Formatting Functions
# ========================================

"""Format grid information: N points, Type [min, max]"""
function _format_grid_info(x::AbstractVector{T}) where {T}
    n = length(x)
    x_min, x_max = first(x), last(x)
    grid_type = x isa AbstractRange ? "Range" : "Vector"
    return "$n points, $grid_type [$x_min, $x_max]"
end

"""Format extrapolation mode from ExtrapVal."""
function _format_extrap(mode::ExtrapVal)
    mode === Val(:none) && return ":none"
    mode === Val(:constant) && return ":constant"
    mode === Val(:extension) && return ":extension"
    mode === Val(:wrap) && return ":wrap"
    return "unknown"
end

"""Format side selection from SideVal."""
function _format_side(side::SideVal)
    side === Val(:nearest) && return ":nearest"
    side === Val(:left) && return ":left"
    side === Val(:right) && return ":right"
    return "unknown"
end

"""Format search policy name."""
_format_search(::Binary) = "Binary"
_format_search(::HintedBinary) = "HintedBinary"
_format_search(::Linear) = "Linear"
_format_search(::LinearBinary{MAX}) where {MAX} = "LinearBinary{$MAX}"

"""Format boundary condition for display."""
_format_bc(::NaturalBC) = "Natural (S''=0 at ends)"
_format_bc(::ClampedBC) = "Clamped (S'=0 at ends)"
_format_bc(::PeriodicBC) = "Periodic"
_format_bc(::PeriodicData) = "Periodic"
_format_bc(::MinCurvFit) = "MinCurvFit"
_format_bc(::ParabolaFit) = "ParabolaFit"
_format_bc(bc::Deriv1) = "Deriv1($(bc.val))"
_format_bc(bc::Deriv2) = "Deriv2($(bc.val))"
_format_bc(bc::Deriv3) = "Deriv3($(bc.val))"
_format_bc(bc::Left) = "Left($(nameof(typeof(bc.bc))))"
_format_bc(bc::Right) = "Right($(nameof(typeof(bc.bc))))"

function _format_bc(bc::BCPair)
    left_str = _format_bc_point(bc.left)
    right_str = _format_bc_point(bc.right)
    # Check for common named patterns
    if bc.left isa Deriv2 && bc.right isa Deriv2 && bc.left.val == 0 && bc.right.val == 0
        return "Natural (S''=0 at ends)"
    elseif bc.left isa Deriv1 && bc.right isa Deriv1 && bc.left.val == 0 && bc.right.val == 0
        return "Clamped (S'=0 at ends)"
    else
        return "$left_str | $right_str"
    end
end

# Helper for BCPair formatting
_format_bc_point(bc::Deriv1) = "Deriv1($(bc.val))"
_format_bc_point(bc::Deriv2) = "Deriv2($(bc.val))"
_format_bc_point(bc::Deriv3) = "Deriv3($(bc.val))"
_format_bc_point(::ParabolaFit) = "ParabolaFit"
_format_bc_point(bc) = string(nameof(typeof(bc)))

"""Format derivative order as ordinal."""
function _format_deriv_order(order::Int)
    order == 1 && return "1st"
    order == 2 && return "2nd"
    order == 3 && return "3rd"
    return "$(order)th"
end

# ========================================
# Show Methods: Single-Series Interpolants
# ========================================

# --- LinearInterpolant ---

function Base.show(io::IO, itp::LinearInterpolant{T}) where {T}
    n = length(itp.x)
    _show_type_header(io, "LinearInterpolant", T)
    print(io, "($n pts)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::LinearInterpolant{T}) where {T}
    _show_type_header(io, "LinearInterpolant", T)
    println(io)
    _show_row(io, false, "Grid:  ", _format_grid_info(itp.x))
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(itp.mode); value_color=:magenta)
    println(io)
    _show_row(io, true, "Search:", _format_search(itp.search_policy); value_color=:green)
end

# --- ConstantInterpolant ---

function Base.show(io::IO, itp::ConstantInterpolant{T}) where {T}
    n = length(itp.x)
    _show_type_header(io, "ConstantInterpolant", T)
    print(io, "($n pts)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::ConstantInterpolant{T}) where {T}
    _show_type_header(io, "ConstantInterpolant", T)
    println(io)
    _show_row(io, false, "Grid:  ", _format_grid_info(itp.x))
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(itp.mode); value_color=:magenta)
    println(io)
    _show_row(io, false, "Side:  ", _format_side(itp.side); value_color=:magenta)
    println(io)
    _show_row(io, true, "Search:", _format_search(itp.search_policy); value_color=:green)
end

# --- QuadraticInterpolant ---

function Base.show(io::IO, itp::QuadraticInterpolant{T}) where {T}
    n = length(itp.x)
    _show_type_header(io, "QuadraticInterpolant", T)
    print(io, "($n pts)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::QuadraticInterpolant{T}) where {T}
    _show_type_header(io, "QuadraticInterpolant", T)
    println(io)
    _show_row(io, false, "Grid:  ", _format_grid_info(itp.x))
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(itp.mode); value_color=:magenta)
    println(io)
    _show_row(io, true, "Search:", _format_search(itp.search_policy); value_color=:green)
end

# --- CubicInterpolant ---

function Base.show(io::IO, itp::CubicInterpolant{T}) where {T}
    n = length(itp.cache.x)
    bc_name = _short_bc_name(itp.cache.bc_config)
    _show_type_header(io, "CubicInterpolant", T)
    print(io, "($n pts, $bc_name)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::CubicInterpolant{T}) where {T}
    _show_type_header(io, "CubicInterpolant", T)
    println(io)
    _show_row(io, false, "Grid:  ", _format_grid_info(itp.cache.x))
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(itp.extrap); value_color=:magenta)
    println(io)
    _show_row(io, false, "Search:", _format_search(itp.search_policy); value_color=:green)
    println(io)
    _show_row(io, true, "BC:    ", _format_bc(itp.cache.bc_config); value_color=:green)
end

# Short BC name for compact display
_short_bc_name(::BCPair{T,L,R}) where {T, L<:Deriv2, R<:Deriv2} = "Natural"
_short_bc_name(::BCPair{T,L,R}) where {T, L<:Deriv1, R<:Deriv1} = "Clamped"
_short_bc_name(::PeriodicData) = "Periodic"
_short_bc_name(bc::BCPair) = "Custom"

# ========================================
# Show Methods: Multi-Series Interpolants
# ========================================

# --- LinearSeriesInterpolant ---

function Base.show(io::IO, sitp::LinearSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    _show_type_header(io, "LinearSeriesInterpolant", T)
    print(io, "($np × $ns)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::LinearSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    _show_type_header(io, "LinearSeriesInterpolant", T; suffix=" with $ns series")
    println(io)
    _show_row(io, false, "Grid:  ", _format_grid_info(sitp.x))
    println(io)
    _show_row(io, false, "Matrix:", "$np × $ns (n_points × n_series)"; value_color=:yellow)
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(sitp.extrap); value_color=:magenta)
    println(io)
    _show_row(io, true, "Search:", _format_search(sitp.search_policy); value_color=:green)
end

# --- ConstantSeriesInterpolant ---

function Base.show(io::IO, sitp::ConstantSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    _show_type_header(io, "ConstantSeriesInterpolant", T)
    print(io, "($np × $ns)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::ConstantSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    _show_type_header(io, "ConstantSeriesInterpolant", T; suffix=" with $ns series")
    println(io)
    _show_row(io, false, "Grid:  ", _format_grid_info(sitp.x))
    println(io)
    _show_row(io, false, "Matrix:", "$np × $ns (n_points × n_series)"; value_color=:yellow)
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(sitp.extrap); value_color=:magenta)
    println(io)
    _show_row(io, false, "Side:  ", _format_side(sitp.side); value_color=:magenta)
    println(io)
    _show_row(io, true, "Search:", _format_search(sitp.search_policy); value_color=:green)
end

# --- QuadraticSeriesInterpolant ---

function Base.show(io::IO, sitp::QuadraticSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    _show_type_header(io, "QuadraticSeriesInterpolant", T)
    print(io, "($np × $ns)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::QuadraticSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    _show_type_header(io, "QuadraticSeriesInterpolant", T; suffix=" with $ns series")
    println(io)
    _show_row(io, false, "Grid:  ", _format_grid_info(sitp.x))
    println(io)
    _show_row(io, false, "Matrix:", "$np × $ns (n_points × n_series)"; value_color=:yellow)
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(sitp.extrap); value_color=:magenta)
    println(io)
    _show_row(io, true, "Search:", _format_search(sitp.search_policy); value_color=:green)
end

# --- CubicSeriesInterpolant ---

function Base.show(io::IO, sitp::CubicSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    bc_name = _short_bc_name(sitp.bc_for_solve)
    _show_type_header(io, "CubicSeriesInterpolant", T)
    print(io, "($np × $ns, $bc_name)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::CubicSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    _show_type_header(io, "CubicSeriesInterpolant", T; suffix=" with $ns series")
    println(io)
    _show_row(io, false, "Grid:  ", _format_grid_info(sitp.cache.x))
    println(io)
    _show_row(io, false, "Matrix:", "$np × $ns (n_points × n_series)"; value_color=:yellow)
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(sitp.extrap); value_color=:magenta)
    println(io)
    _show_row(io, false, "Search:", _format_search(sitp.search_policy); value_color=:green)
    println(io)
    _show_row(io, true, "BC:    ", _format_bc(sitp.bc_for_solve); value_color=:green)
end

# ========================================
# Show Methods: DerivativeView
# ========================================

function Base.show(io::IO, d::DerivativeView{Order, ITP}) where {Order, ITP}
    _show_print(io, "DerivativeView", :cyan; bold=true)
    _show_print(io, "{$Order}", :light_blue)
    print(io, "(")
    # Compact parent display
    show(io, d.parent)
    print(io, ")")
end

"""Extract float type T from AbstractInterpolant{T}."""
_interpolant_float_type(::AbstractInterpolant{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", d::DerivativeView{Order, ITP}) where {Order, ITP}
    ord_str = _format_deriv_order(Order)
    _show_print(io, "DerivativeView", :cyan; bold=true)
    _show_print(io, " ($ord_str derivative)", :light_black)
    println(io)

    # Get parent info
    parent = d.parent
    parent_type = nameof(typeof(parent))
    T = _interpolant_float_type(parent)

    # Determine number of points
    n_pts = if hasproperty(parent, :x)
        length(parent.x)
    elseif hasproperty(parent, :cache) && hasproperty(parent.cache, :x)
        length(parent.cache.x)
    else
        "?"
    end

    _show_row(io, true, "Parent:", "$parent_type{$T}, $n_pts points"; value_color=:yellow)
end
