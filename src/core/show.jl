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
    _show_type_header_2params(io, typename, Tg, Tv; suffix="")

Print type name with two type parameters for {Tg, Tv} interpolants.
Shows `LinearInterpolant{Float64, ComplexF64}` for complex values,
or `LinearInterpolant{Float64}` if Tv == Tg (backward compatible display).
"""
function _show_type_header_2params(io::IO, typename::String, ::Type{Tg}, ::Type{Tv}; suffix::String="") where {Tg, Tv}
    _show_print(io, typename, :cyan; bold=true)
    _show_print(io, "{", :light_black)
    _show_print(io, string(Tg), :light_blue)
    # Only show Tv if different from Tg (Complex case)
    if Tv !== Tg
        _show_print(io, ", ", :light_black)
        _show_print(io, string(Tv), :light_blue)
    end
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
_format_bc(::PeriodicBC{:inclusive}) = "Periodic"
_format_bc(bc::PeriodicBC{:exclusive}) = "Periodic (exclusive, T=$(bc.period))"
_format_bc(::PeriodicData) = "Periodic"
_format_bc(bc::Deriv1) = "Deriv1($(bc.val))"
_format_bc(bc::Deriv2) = "Deriv2($(bc.val))"
_format_bc(bc::Deriv3) = "Deriv3($(bc.val))"
_format_bc(::MinCurvFit) = "MinCurvFit"
_format_bc(::PolyFit{1}) = "LinearFit"
_format_bc(::PolyFit{2}) = "QuadraticFit"
_format_bc(::PolyFit{3}) = "CubicFit"
_format_bc(::PolyFit{D}) where {D} = "PolyFit{$D}"  # Fallback for other degrees
_format_bc(bc::Left) = "Left($(_format_bc_point(bc.bc)))"
_format_bc(bc::Right) = "Right($(_format_bc_point(bc.bc)))"

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

function _format_deriv_order(order::Tuple)
    N = length(order)
    all_zero = all(o -> o == 0, order)
    all_zero && return string(order, " value ≡ f")

    parts = String[]
    for i in 1:N
        ord = order[i]
        ord == 0 && continue
        # Use normal 'x' instead of subscript 'ₓ' for better visibility
        # Example: ∂x₁ instead of ∂ₓ₁
        if ord == 1
            push!(parts, "∂x" * _subscript_digit(i))
        else
            push!(parts, "∂" * _superscript_int(ord) * "x" * _subscript_digit(i))
        end
    end
    # Wrap in parentheses to group the operator
    return string(order, " partial derivatives ≡ (", join(parts, ""), ")f")
end

@inline function _superscript_digit(d::Int)
    d == 0 && return "⁰"
    d == 1 && return "¹"
    d == 2 && return "²"
    d == 3 && return "³"
    d == 4 && return "⁴"
    d == 5 && return "⁵"
    d == 6 && return "⁶"
    d == 7 && return "⁷"
    d == 8 && return "⁸"
    d == 9 && return "⁹"
    return string(d)
end

@inline function _superscript_int(n::Int)
    n < 0 && return "^$(n)"
    n < 10 && return _superscript_digit(n)
    digits = reverse(digits(n))
    return join((_superscript_digit(d) for d in digits))
end

# ========================================
# Show Methods: Single-Series Interpolants
# ========================================

# --- LinearInterpolant ---

function Base.show(io::IO, itp::LinearInterpolant{Tg, Tv}) where {Tg, Tv}
    n = length(itp.x)
    _show_type_header_2params(io, "LinearInterpolant", Tg, Tv)
    print(io, "($n pts)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::LinearInterpolant{Tg, Tv}) where {Tg, Tv}
    _show_type_header_2params(io, "LinearInterpolant", Tg, Tv)
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

function Base.show(io::IO, itp::ConstantInterpolant{Tg, Tv}) where {Tg, Tv}
    n = length(itp.x)
    _show_type_header_2params(io, "ConstantInterpolant", Tg, Tv)
    print(io, "($n pts)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::ConstantInterpolant{Tg, Tv}) where {Tg, Tv}
    _show_type_header_2params(io, "ConstantInterpolant", Tg, Tv)
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

function Base.show(io::IO, itp::QuadraticInterpolant{Tg, Tv}) where {Tg, Tv}
    n = length(itp.x)
    _show_type_header_2params(io, "QuadraticInterpolant", Tg, Tv)
    print(io, "($n pts)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::QuadraticInterpolant{Tg, Tv}) where {Tg, Tv}
    _show_type_header_2params(io, "QuadraticInterpolant", Tg, Tv)
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

function Base.show(io::IO, itp::CubicInterpolant{Tg, Tv}) where {Tg, Tv}
    n = length(itp.cache.x)
    bc_name = _short_bc_name(itp.bc)
    _show_type_header_2params(io, "CubicInterpolant", Tg, Tv)
    print(io, "($n pts, $bc_name)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::CubicInterpolant{Tg, Tv}) where {Tg, Tv}
    _show_type_header_2params(io, "CubicInterpolant", Tg, Tv)
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
_short_bc_name(::PeriodicBC{:inclusive}) = "Periodic"
_short_bc_name(::PeriodicBC{:exclusive}) = "Periodic(excl)"
_short_bc_name(::PeriodicData) = "Periodic"
_short_bc_name(::MinCurvFit) = "MinCurvFit"
_short_bc_name(bc::Left) = "Left($(_format_bc_point(bc.bc)))"
_short_bc_name(bc::Right) = "Right($(_format_bc_point(bc.bc)))"
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

function Base.show(io::IO, sitp::LinearSeriesInterpolant{Tg, Tv}) where {Tg, Tv}
    np, ns = size(sitp.y)
    _show_type_header_2params(io, "LinearSeriesInterpolant", Tg, Tv)
    print(io, "($np × $ns)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::LinearSeriesInterpolant{Tg, Tv}) where {Tg, Tv}
    np, ns = size(sitp.y)
    is_range = sitp.x isa AbstractRange
    _show_type_header_2params(io, "LinearSeriesInterpolant", Tg, Tv; suffix=" with $ns series")
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

function Base.show(io::IO, sitp::ConstantSeriesInterpolant{Tg, Tv}) where {Tg, Tv}
    np, ns = size(sitp.y)
    _show_type_header_2params(io, "ConstantSeriesInterpolant", Tg, Tv)
    print(io, "($np × $ns)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::ConstantSeriesInterpolant{Tg, Tv}) where {Tg, Tv}
    np, ns = size(sitp.y)
    is_range = sitp.x isa AbstractRange
    _show_type_header_2params(io, "ConstantSeriesInterpolant", Tg, Tv; suffix=" with $ns series")
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

function Base.show(io::IO, sitp::QuadraticSeriesInterpolant{Tg, Tv}) where {Tg, Tv}
    np, ns = size(sitp.y)
    _show_type_header_2params(io, "QuadraticSeriesInterpolant", Tg, Tv)
    print(io, "($np × $ns)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::QuadraticSeriesInterpolant{Tg, Tv}) where {Tg, Tv}
    np, ns = size(sitp.y)
    is_range = sitp.x isa AbstractRange
    _show_type_header_2params(io, "QuadraticSeriesInterpolant", Tg, Tv; suffix=" with $ns series")
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

function Base.show(io::IO, sitp::CubicSeriesInterpolant{Tg, Tv}) where {Tg, Tv}
    np, ns = size(sitp.y)
    bc_name = _short_bc_name(sitp.bc_for_solve)
    _show_type_header_2params(io, "CubicSeriesInterpolant", Tg, Tv)
    print(io, "($np × $ns, $bc_name)")
end

function Base.show(io::IO, ::MIME"text/plain", sitp::CubicSeriesInterpolant{Tg, Tv}) where {Tg, Tv}
    np, ns = size(sitp.y)
    is_range = sitp.cache.x isa AbstractRange
    _show_type_header_2params(io, "CubicSeriesInterpolant", Tg, Tv; suffix=" with $ns series")
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

"""Extract grid type Tg from AbstractInterpolant{Tg, Tv}."""
_interpolant_float_type(::AbstractInterpolant{Tg, Tv}) where {Tg, Tv} = Tg

function Base.show(io::IO, ::MIME"text/plain", d::DerivativeView{Order, ITP}) where {Order, ITP}
    ord_str = _format_deriv_order(Order)
    _show_print(io, "DerivativeView", :cyan; bold=true)
    if Order isa Tuple
        _show_print(io, " $ord_str", :light_black)
    else
        _show_print(io, " ($ord_str derivative)", :light_black)
    end
    println(io)

    # Get parent info
    parent = d.parent
    parent_type = nameof(typeof(parent))
    T = _interpolant_float_type(parent)

    if parent isa AbstractInterpolantND
        # ND interpolants: show dimensionality and grid sizes
        N = ndims(parent)
        sizes = if hasproperty(parent, :grids)
            join([string(length(g)) for g in parent.grids], "×")
        else
            "?"
        end
        _show_row(io, true, "Parent:", "$parent_type{$T}, $(N)D, sizes: $sizes")
    else
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
end

# ========================================
# Show Methods: ND Interpolants (Helpers)
# ========================================
#
# Reusable helper functions for N-dimensional interpolant display.
# Designed for future ND types: LinearInterpolantND, ConstantInterpolantND, etc.

"""
    _show_type_header_nd(io, typename, Tg, Tv, N; suffix="")

Print type name with ND type parameters: `CubicInterpolantND{Float64, Float64, 3}`.
If Tv == Tg, shows only Tg for cleaner display: `CubicInterpolantND{Float64, 3}`.
"""
function _show_type_header_nd(
    io::IO, typename::String, ::Type{Tg}, ::Type{Tv}, N::Int; suffix::String=""
) where {Tg, Tv}
    _show_print(io, typename, :cyan; bold=true)
    _show_print(io, "{", :light_black)
    _show_print(io, string(Tg), :light_blue)
    # Only show Tv if different from Tg (Complex case)
    if Tv !== Tg
        _show_print(io, ", ", :light_black)
        _show_print(io, string(Tv), :light_blue)
    end
    _show_print(io, ", ", :light_black)
    _show_print(io, string(N), :light_blue)
    _show_print(io, "}", :light_black)
    if !isempty(suffix)
        _show_print(io, suffix, :cyan)
    end
end

"""
    _show_nd_grids_summary(io, is_last, grids::Tuple)

Print ND grid summary with per-axis details.
Example output:
  ├─ Grids:  3D, 50×30×20 points
  │  ├─ x₁: Range ∈ [0.0, 6.28]
  │  ├─ x₂: Range ∈ [0.0, 3.14]
  │  └─ x₃: Vector ∈ [0.0, 1.0]

Note: grids may be heterogeneous Tuple (Vector + Range mixed).
"""
function _show_nd_grids_summary(io::IO, is_last::Bool, grids::Tuple)
    N = length(grids)
    # Main grid row
    sizes = join([string(length(g)) for g in grids], "×")
    prefix = is_last ? "└─ " : "├─ "
    _show_print(io, prefix, :light_black)
    _show_print(io, "Grids: ", :light_black)
    print(io, "$(N)D, $sizes points")
    println(io)

    # Per-axis details
    tree_prefix = is_last ? "   " : "│  "
    for d in 1:N
        g = grids[d]
        is_last_axis = (d == N)
        axis_prefix = is_last_axis ? "└─ " : "├─ "
        grid_type = g isa AbstractRange ? "Range" : "Vector"
        x_min_str = _format_num(first(g))
        x_max_str = _format_num(last(g))

        _show_print(io, tree_prefix, :light_black)
        _show_print(io, axis_prefix, :light_black)
        _show_print(io, "x", :light_black)
        _show_print(io, _subscript_digit(d), :light_black)
        print(io, ": ")
        _show_print(io, grid_type, :magenta)
        print(io, " ∈ [$x_min_str, $x_max_str]")
        if d < N
            println(io)
        end
    end
end

"""Convert digit to Unicode subscript for axis labels (x₁, x₂, etc.)."""
function _subscript_digit(d::Int)
    subscripts = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    if 0 ≤ d ≤ 9
        return string(subscripts[d + 1])
    else
        # For d >= 10, just use parentheses
        return "($d)"
    end
end

"""
    _show_nd_config_row(io, is_last, label, configs::Tuple, format_fn)

Print per-axis configuration. If all axes have same config, show once with "(all axes)".
Otherwise show tuple format.

# Arguments
- `format_fn`: Function to format each config element (e.g., `_format_extrap`, `_format_bc`)

Note: configs may be heterogeneous Tuple (not NTuple) when configs differ per axis.
"""
function _show_nd_config_row(
    io::IO, is_last::Bool, label::String, configs::Tuple, format_fn::Function;
    value_color::Symbol=:normal
)
    N = length(configs)
    formatted = ntuple(d -> format_fn(configs[d]), Val(N))
    prefix = is_last ? "└─ " : "├─ "

    _show_print(io, prefix, :light_black)
    _show_print(io, label, :light_black)
    print(io, " ")

    # Check if all configs are the same
    if all(f -> f == formatted[1], formatted)
        # All same: show single value
        if value_color === :normal
            print(io, formatted[1])
        else
            _show_print(io, formatted[1], value_color)
        end
        _show_print(io, " (all axes)", :light_black)
    else
        # Different: show tuple
        print(io, "(")
        for (i, f) in enumerate(formatted)
            if value_color === :normal
                print(io, f)
            else
                _show_print(io, f, value_color)
            end
            i < N && print(io, ", ")
        end
        print(io, ")")
    end
end

# ========================================
# Show Methods: CubicInterpolantND
# ========================================

# Short BC name for ND compact display (reuses 1D helper where possible)
# Note: bcs may be heterogeneous Tuple (not NTuple) when BCs differ per axis
function _short_bc_name_nd(bcs::Tuple)
    N = length(bcs)
    names = ntuple(d -> _short_bc_name(bcs[d]), Val(N))
    if all(n -> n == names[1], names)
        return names[1]
    else
        return "Mixed"
    end
end

function Base.show(io::IO, itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    sizes = join([string(length(g)) for g in itp.grids], "×")
    bc_name = _short_bc_name_nd(itp.bcs)
    _show_type_header_nd(io, "CubicInterpolantND", Tg, Tv, N)
    print(io, "($sizes, $bc_name)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    _show_type_header_nd(io, "CubicInterpolantND", Tg, Tv, N)
    println(io)

    # Grid info with per-axis details
    _show_nd_grids_summary(io, false, itp.grids)
    println(io)

    # Extrapolation modes
    _show_nd_config_row(io, false, "Extrap:", itp.extraps, _format_extrap; value_color=:magenta)
    println(io)

    # Search policies (only if any axis has non-Range grid)
    has_vector_grid = any(g -> !(g isa AbstractRange), itp.grids)
    if has_vector_grid
        _show_nd_config_row(io, false, "Search:", itp.searches, _format_search)
        println(io)
    end

    # Boundary conditions (per-axis hierarchical display)
    _show_nd_bc_summary(io, true, itp.bcs)
end

"""
    _show_nd_bc_summary(io, is_last, bcs::Tuple)

Print BC summary. If all axes have the same BC, show single line with "(all axes)".
Otherwise show per-axis details in hierarchical format.

Example output (all same):
  └─ BC: Natural (S''=0 at ends) (all axes)

Example output (different):
  └─ BC:
     ├─ x₁: CubicFit | CubicFit
     └─ x₂: LinearFit | LinearFit
"""
function _show_nd_bc_summary(io::IO, is_last::Bool, bcs::Tuple)
    N = length(bcs)
    prefix = is_last ? "└─ " : "├─ "

    # Check if all BCs are the same by comparing formatted strings
    formatted = ntuple(d -> _format_bc(bcs[d]), Val(N))
    all_same = all(f -> f == formatted[1], formatted)

    if all_same
        # Single line display
        _show_print(io, prefix, :light_black)
        _show_print(io, "BC:", :light_black)
        print(io, " $(formatted[1])")
        _show_print(io, " (all axes)", :light_black)
    else
        # Hierarchical per-axis display
        _show_print(io, prefix, :light_black)
        _show_print(io, "BC:", :light_black)
        println(io)

        tree_prefix = is_last ? "   " : "│  "
        for d in 1:N
            is_last_axis = (d == N)
            axis_prefix = is_last_axis ? "└─ " : "├─ "

            _show_print(io, tree_prefix, :light_black)
            _show_print(io, axis_prefix, :light_black)
            _show_print(io, "x", :light_black)
            _show_print(io, _subscript_digit(d), :light_black)
            print(io, ": $(formatted[d])")
            if d < N
                println(io)
            end
        end
    end
end

# ========================================
# ConstantInterpolantND Show Methods
# ========================================

function Base.show(io::IO, itp::ConstantInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    sizes = join([string(length(g)) for g in itp.grids], "×")
    side_str = _short_side_name_nd(itp.sides)
    _show_type_header_nd(io, "ConstantInterpolantND", Tg, Tv, N)
    print(io, "($sizes, $side_str)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::ConstantInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    _show_type_header_nd(io, "ConstantInterpolantND", Tg, Tv, N)
    println(io)

    # Grid info with per-axis details
    _show_nd_grids_summary(io, false, itp.grids)
    println(io)

    # Extrapolation modes
    _show_nd_config_row(io, false, "Extrap:", itp.extraps, _format_extrap; value_color=:magenta)
    println(io)

    # Side selection
    _show_nd_config_row(io, false, "Side:  ", itp.sides, _format_side; value_color=:magenta)
    println(io)

    # Search policies (only if any axis has non-Range grid)
    has_vector_grid = any(g -> !(g isa AbstractRange), itp.grids)
    if has_vector_grid
        _show_nd_config_row(io, true, "Search:", itp.searches, _format_search)
    end
end

"""
    _short_side_name_nd(sides::Tuple) -> String

Get compact side name for ND constant interpolant display.
"""
function _short_side_name_nd(sides::Tuple)
    N = length(sides)
    formatted = ntuple(d -> _format_side(sides[d]), Val(N))
    all_same = all(f -> f == formatted[1], formatted)
    if all_same
        return formatted[1]
    else
        return join(formatted, ",")
    end
end

# ========================================
# LinearInterpolantND Show Methods
# ========================================

function Base.show(io::IO, itp::LinearInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    sizes = join([string(length(g)) for g in itp.grids], "×")
    _show_type_header_nd(io, "LinearInterpolantND", Tg, Tv, N)
    print(io, "($sizes)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::LinearInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    _show_type_header_nd(io, "LinearInterpolantND", Tg, Tv, N)
    println(io)

    # Grid info with per-axis details
    _show_nd_grids_summary(io, false, itp.grids)
    println(io)

    # Extrapolation modes
    _show_nd_config_row(io, false, "Extrap:", itp.extraps, _format_extrap; value_color=:magenta)
    println(io)

    # Search policies (only if any axis has non-Range grid)
    has_vector_grid = any(g -> !(g isa AbstractRange), itp.grids)
    if has_vector_grid
        _show_nd_config_row(io, true, "Search:", itp.searches, _format_search)
    end
end

# ========================================
# QuadraticInterpolantND Show Methods
# ========================================

function Base.show(io::IO, itp::QuadraticInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    sizes = join([string(length(g)) for g in itp.grids], "×")
    bc_name = _short_bc_name_nd(itp.bcs)
    _show_type_header_nd(io, "QuadraticInterpolantND", Tg, Tv, N)
    print(io, "($sizes, $bc_name)")
end

function Base.show(io::IO, ::MIME"text/plain", itp::QuadraticInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    _show_type_header_nd(io, "QuadraticInterpolantND", Tg, Tv, N)
    println(io)

    # Grid info with per-axis details
    _show_nd_grids_summary(io, false, itp.grids)
    println(io)

    # Extrapolation modes
    _show_nd_config_row(io, false, "Extrap:", itp.extraps, _format_extrap; value_color=:magenta)
    println(io)

    # Search policies (only if any axis has non-Range grid)
    has_vector_grid = any(g -> !(g isa AbstractRange), itp.grids)
    if has_vector_grid
        _show_nd_config_row(io, false, "Search:", itp.searches, _format_search)
        println(io)
    end

    # Boundary conditions
    _show_nd_bc_summary(io, true, itp.bcs)
end
