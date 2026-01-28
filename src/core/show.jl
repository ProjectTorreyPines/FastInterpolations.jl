# ========================================
# Custom Show Methods for FastInterpolations.jl
# ========================================
#
# This file provides visually appealing REPL output using:
# - Unicode box-drawing characters (├─, └─)
# - Colored output via printstyled
# - Compact and verbose display modes
#
# Color Scheme ("Less is More"):
# - Type name: cyan + bold
# - Type param (Float64): light_blue
# - Labels & box chars: light_black
# - Values (Grid, Search, BC, etc.): default (terminal text color)
# - Symbols only (:none, :wrap, etc.): magenta

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
        _show_print(io, suffix, :cyan)
    end
end

"""
    _show_row(io, is_last, label, value; value_color=:normal)

Print a box-drawing row with label and value.
Default value_color is :normal (terminal default text color).
"""
function _show_row(io::IO, is_last::Bool, label::String, value::String; value_color::Symbol=:normal)
    prefix = is_last ? "└─ " : "├─ "
    _show_print(io, prefix, :light_black)
    _show_print(io, label, :light_black)
    print(io, " ")
    if value_color === :normal
        print(io, value)
    else
        _show_print(io, value, value_color)
    end
end

# ========================================
# Formatting Functions
# ========================================

"""Format a number compactly: use %.3g for floats, string for others."""
function _format_num(x::Real)
    # Use %.3g for compact display (3 significant digits)
    return Printf.@sprintf("%.3g", x)
end

"""
    _show_grid_row(io, is_last, x)

Print grid info with type highlighted: `Grid:   Range, 100 points ∈ [0.0, 1.0]`
"""
function _show_grid_row(io::IO, is_last::Bool, x::AbstractVector)
    n = length(x)
    x_min_str = _format_num(first(x))
    x_max_str = _format_num(last(x))
    grid_type = x isa AbstractRange ? "Range" : "Vector"

    prefix = is_last ? "└─ " : "├─ "
    _show_print(io, prefix, :light_black)
    _show_print(io, "Grid:  ", :light_black)
    _show_print(io, grid_type, :magenta)
    print(io, ", $n points ∈ [$x_min_str, $x_max_str]")
end

"""Format extrapolation mode from ExtrapVal."""
function _format_extrap(mode)
    mode === Val(:none) && return ":none"
    mode === Val(:constant) && return ":constant"
    mode === Val(:extension) && return ":extension"
    mode === Val(:wrap) && return ":wrap"
    return "unknown"
end

"""Format side selection from SideVal."""
function _format_side(side)
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
_format_bc(bc::Deriv1) = "Deriv1($(bc.val))"
_format_bc(bc::Deriv2) = "Deriv2($(bc.val))"
_format_bc(bc::Deriv3) = "Deriv3($(bc.val))"
_format_bc(::MinCurvFit) = "MinCurvFit"
_format_bc(::PolyFit{1}) = "LinearFit"
_format_bc(::PolyFit{2}) = "QuadraticFit"
_format_bc(::PolyFit{3}) = "CubicFit"
_format_bc(::PolyFit{D}) where {D} = "PolyFit{$D}"  # Fallback for other degrees
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
_format_bc_point(::PolyFit{1}) = "LinearFit"
_format_bc_point(::PolyFit{2}) = "QuadraticFit"
_format_bc_point(::PolyFit{3}) = "CubicFit"
_format_bc_point(::PolyFit{D}) where {D} = "PolyFit{$D}"  # Fallback for other degrees
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
    is_range = itp.x isa AbstractRange
    _show_grid_row(io, false, itp.x)
    println(io)
    _show_row(io, is_range, "Extrap:", _format_extrap(itp.extrap); value_color=:magenta)
    if !is_range
        println(io)
        _show_row(io, true, "Search:", _format_search(itp.search_policy))
    end
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
    is_range = itp.x isa AbstractRange
    _show_grid_row(io, false, itp.x)
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(itp.extrap); value_color=:magenta)
    println(io)
    _show_row(io, is_range, "Side:  ", _format_side(itp.side); value_color=:magenta)
    if !is_range
        println(io)
        _show_row(io, true, "Search:", _format_search(itp.search_policy))
    end
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
    is_range = itp.x isa AbstractRange
    _show_grid_row(io, false, itp.x)
    println(io)
    _show_row(io, is_range, "Extrap:", _format_extrap(itp.extrap); value_color=:magenta)
    if !is_range
        println(io)
        _show_row(io, true, "Search:", _format_search(itp.search_policy))
    end
end

# --- CubicInterpolant ---

function Base.show(io::IO, itp::CubicInterpolant{T}) where {T}
    n = length(itp.cache.x)
    bc_name = _short_bc_name(itp.bc)
    _show_type_header(io, "CubicInterpolant", T)
    print(io, "($n pts, $bc_name)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::CubicInterpolant{T}) where {T}
    _show_type_header(io, "CubicInterpolant", T)
    println(io)
    is_range = itp.cache.x isa AbstractRange
    _show_grid_row(io, false, itp.cache.x)
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(itp.extrap); value_color=:magenta)
    if !is_range
        println(io)
        _show_row(io, false, "Search:", _format_search(itp.search_policy))
    end
    println(io)
    _show_row(io, true, "BC:    ", _format_bc(itp.bc))
end

# Short BC name for compact display
_short_bc_name(::PeriodicBC) = "Periodic"
function _short_bc_name(bc::BCPair)
    # Check for Natural: both ends have Deriv2 with val=0
    if bc.left isa Deriv2 && bc.right isa Deriv2 && bc.left.val == 0 && bc.right.val == 0
        return "Natural"
    # Check for Clamped: both ends have Deriv1 with val=0
    elseif bc.left isa Deriv1 && bc.right isa Deriv1 && bc.left.val == 0 && bc.right.val == 0
        return "Clamped"
    else
        return "Custom"
    end
end

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
    is_range = sitp.x isa AbstractRange
    _show_type_header(io, "LinearSeriesInterpolant", T; suffix=" with $ns series")
    println(io)
    _show_grid_row(io, false, sitp.x)
    println(io)
    _show_row(io, false, "Matrix:", "$np × $ns (n_points × n_series)")
    println(io)
    _show_row(io, is_range, "Extrap:", _format_extrap(sitp.extrap); value_color=:magenta)
    if !is_range
        println(io)
        _show_row(io, true, "Search:", _format_search(sitp.search_policy))
    end
end

# --- ConstantSeriesInterpolant ---

function Base.show(io::IO, sitp::ConstantSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    _show_type_header(io, "ConstantSeriesInterpolant", T)
    print(io, "($np × $ns)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::ConstantSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    is_range = sitp.x isa AbstractRange
    _show_type_header(io, "ConstantSeriesInterpolant", T; suffix=" with $ns series")
    println(io)
    _show_grid_row(io, false, sitp.x)
    println(io)
    _show_row(io, false, "Matrix:", "$np × $ns (n_points × n_series)")
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(sitp.extrap); value_color=:magenta)
    println(io)
    _show_row(io, is_range, "Side:  ", _format_side(sitp.side); value_color=:magenta)
    if !is_range
        println(io)
        _show_row(io, true, "Search:", _format_search(sitp.search_policy))
    end
end

# --- QuadraticSeriesInterpolant ---

function Base.show(io::IO, sitp::QuadraticSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    _show_type_header(io, "QuadraticSeriesInterpolant", T)
    print(io, "($np × $ns)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::QuadraticSeriesInterpolant{T}) where {T}
    np, ns = size(sitp.y)
    is_range = sitp.x isa AbstractRange
    _show_type_header(io, "QuadraticSeriesInterpolant", T; suffix=" with $ns series")
    println(io)
    _show_grid_row(io, false, sitp.x)
    println(io)
    _show_row(io, false, "Matrix:", "$np × $ns (n_points × n_series)")
    println(io)
    _show_row(io, is_range, "Extrap:", _format_extrap(sitp.extrap); value_color=:magenta)
    if !is_range
        println(io)
        _show_row(io, true, "Search:", _format_search(sitp.search_policy))
    end
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
    is_range = sitp.cache.x isa AbstractRange
    _show_type_header(io, "CubicSeriesInterpolant", T; suffix=" with $ns series")
    println(io)
    _show_grid_row(io, false, sitp.cache.x)
    println(io)
    _show_row(io, false, "Matrix:", "$np × $ns (n_points × n_series)")
    println(io)
    _show_row(io, false, "Extrap:", _format_extrap(sitp.extrap); value_color=:magenta)
    if !is_range
        println(io)
        _show_row(io, false, "Search:", _format_search(sitp.search_policy))
    end
    println(io)
    _show_row(io, true, "BC:    ", _format_bc(sitp.bc_for_solve))
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

    _show_row(io, true, "Parent:", "$parent_type{$T}, $n_pts points")
end
