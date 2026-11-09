# ========================================
# PHS — Input Gates
# ========================================
#
# Inputs the implementation cannot evaluate correctly are rejected here with an
# ArgumentError instead of producing wrong values. Every gate runs in the
# constructor or at a callable's entry, before any pool buffer is acquired: an
# exception inside the pooled evaluator would leave those buffers un-rewound.

# Mean node spacing per axis, as the stencil build and the blend radius use it.
@inline _phs_gate_spacings(grids::NTuple{N, AbstractVector}, ::Type{Tg}) where {N, Tg} =
    ntuple(d -> Tg((last(grids[d]) - first(grids[d])) / (length(grids[d]) - 1)), Val(N))

# ----------------------------------------
# Element types
# ----------------------------------------
# Coefficient caches and pool buffers are typed by the grid float type, so only
# real floating-point grids and data can be evaluated. Unit-carrying axes are
# rejected before promotion, which takes a separate per-axis path for them.

@inline function _phs_check_raw_eltypes(grids::Tuple, data::AbstractArray)
    Tg_raw = _promote_grid_eltype(grids)
    (Tg_raw <: Real && eltype(data) <: Real) || _throw_phs_eltype(Tg_raw, eltype(data))
    return nothing
end

@inline _phs_check_float_types(::Type{<:AbstractFloat}, ::Type{<:AbstractFloat}) = nothing
@inline _phs_check_float_types(::Type{Tg}, ::Type{Tv}) where {Tg, Tv} = _throw_phs_eltype(Tg, Tv)

@noinline _throw_phs_eltype(Tg, Tv) = throw(
    ArgumentError(
        "phs_interp supports real floating-point grids and data only (grid eltype $Tg, " *
            "data eltype $Tv); complex, unit-carrying and dual-number values are not supported"
    )
)

# ----------------------------------------
# Uniform axes
# ----------------------------------------
# One stencil geometry (and one Φ⁻¹) is built per axis from the mean spacing, so
# every axis must be equally spaced. Ranges are uniform by construction; other
# vectors may deviate from the mean-spacing lattice by rounding only.

@inline _phs_check_uniform_axes(::Tuple{}, ::Int = 1) = nothing
@inline function _phs_check_uniform_axes(grids::Tuple, d::Int = 1)
    _phs_check_uniform_axis(first(grids), d)
    return _phs_check_uniform_axes(Base.tail(grids), d + 1)
end

@inline _phs_check_uniform_axis(::AbstractRange, ::Int) = nothing
function _phs_check_uniform_axis(g::AbstractVector{T}, d::Int) where {T}
    x1, xn = first(g), last(g)
    h = (xn - x1) / (length(g) - 1)
    tol = sqrt(eps(T)) * abs(h) + 4 * eps(T) * max(abs(x1), abs(xn))
    dev = zero(T)
    for (k, x) in enumerate(g)
        dev = max(dev, abs(x - (x1 + (k - 1) * h)))
    end
    dev <= tol || _throw_phs_nonuniform(d, dev, h)
    return nothing
end

@noinline _throw_phs_nonuniform(d, dev, h) = throw(
    ArgumentError(
        "phs_interp requires uniformly spaced grid axes, but axis $d deviates from its mean " *
            "spacing $h by up to $dev. PHS builds one stencil per axis from that spacing, so a " *
            "non-uniform axis would be evaluated at the wrong node positions. If the axis is " *
            "uniform up to rounding, pass `range(first(x), last(x), length(x))` instead."
    )
)

# ----------------------------------------
# Stencil size
# ----------------------------------------
# The stencil is the stencil_size^N nodes nearest the base node in physical
# distance, so on anisotropic grids it stretches along the finer axes. A stencil
# spanning more nodes than an axis has cannot be shifted inside the grid, and
# boundary nodes then lose polynomial reproduction. Below the minimum size, the
# boundary-shifted local systems become singular (1-D/2-D need 3 nodes per axis).

@inline _phs_min_stencil_size(::Val{N}) where {N} = N <= 2 ? 3 : 2

function _phs_check_stencil_size(s::Int, grids::NTuple{N, AbstractVector}, ::Type{Tg}) where {N, Tg}
    s_min = _phs_min_stencil_size(Val(N))
    s >= s_min || _throw_phs_small_stencil(s, s_min, N)
    offsets = _phs_stencil_offsets(N, s, _phs_gate_spacings(grids, Tg))
    for d in 1:N
        e = maximum(off -> off[d], offsets) - minimum(off -> off[d], offsets) + 1
        e <= length(grids[d]) || _throw_phs_large_stencil(s, d, e, length(grids[d]))
    end
    return nothing
end

@noinline _throw_phs_small_stencil(s, s_min, N) = throw(
    ArgumentError("stencil_size must be ≥ $s_min in $N-D (smaller stencils give singular local systems), got $s")
)
@noinline _throw_phs_large_stencil(s, d, e, n) = throw(
    ArgumentError(
        "stencil_size = $s spans $e nodes along axis $d, which has only $n (the stencil is the " *
            "stencil_size^N nodes nearest the base node, so it stretches along finer axes); " *
            "pass a smaller stencil_size"
    )
)

# ----------------------------------------
# Blend factor
# ----------------------------------------
# A query is blended from base nodes within a = blend_factor · h_max. The cell
# centre is the point farthest from every node, at d_c = ½‖h‖₂; if the blend
# weight there is below eps, the evaluator skips every node and returns zero.
# The check uses the evaluator's own weight function and threshold. The blend
# radius in index units must also fit an Int.

function _phs_check_blend_factor(bf::Real, grids::NTuple{N, AbstractVector}, ::Type{Tg}) where {N, Tg}
    hs = _phs_gate_spacings(grids, Tg)
    h_max = maximum(hs)
    a = Tg(bf) * h_max
    (isfinite(a) && a / minimum(hs) < typemax(Int)) || _throw_phs_blend_range(bf)
    d_c = sqrt(sum(abs2, hs)) / 2
    _phs_blend_weight(d_c, a) >= eps(Tg) || _throw_phs_blend_coverage(bf, d_c / h_max)
    return nothing
end

@noinline _throw_phs_blend_range(bf) = throw(
    ArgumentError("blend_factor must be finite and of moderate size, got $bf")
)
@noinline _throw_phs_blend_coverage(bf, ratio) = throw(
    ArgumentError(
        "blend_factor = $bf does not cover the grid: cell centres lie $(round(ratio; sigdigits = 4)) " *
            "× the largest spacing from their nearest nodes, so blend_factor must be larger than " *
            "that (with margin), or queries near cell centres would evaluate to zero"
    )
)

# ----------------------------------------
# Extrapolation
# ----------------------------------------
# Only NoExtrap, FillExtrap and InBounds are implemented. Clamp/Wrap need query
# coordinate transforms, and Extend would have to reach past the compact-support
# blend; all three used to return 0.0 out of domain.

@inline _phs_supported_extrap(::Union{NoExtrap, FillExtrap, InBounds}) = true
@inline _phs_supported_extrap(::AbstractExtrap) = false

function _phs_check_extraps(extraps::Tuple)
    for (d, e) in enumerate(extraps)
        _phs_supported_extrap(e) || _throw_phs_extrap(d, e)
    end
    return nothing
end

@noinline _throw_phs_extrap(d, e) = throw(
    ArgumentError(
        "phs_interp supports NoExtrap, FillExtrap and InBounds only; axis $d got $(nameof(typeof(e)))"
    )
)

# ----------------------------------------
# Derivative order
# ----------------------------------------
# Derivatives up to total order 2 are implemented. A degree-3 PHS blended with the
# exponential bump has nonzero derivatives of every order, so higher orders are
# rejected rather than returned as zero. The order is a constant of the `DerivOp`
# types, so the check folds away for value queries.

@inline function _phs_check_deriv_order(ops::Tuple)
    sum(deriv_order, ops) <= 2 || _throw_phs_deriv_order(ops)
    return nothing
end

@noinline _throw_phs_deriv_order(ops) = throw(
    ArgumentError(
        "PHS supports derivatives up to total order 2, got order $(sum(deriv_order, ops)) for deriv = $ops"
    )
)

# ----------------------------------------
# Log transform with second derivatives
# ----------------------------------------
# `_phs_eval_blended_G_with_hess` collects the blend nodes into a fixed 27-slot
# buffer, the candidate box of a 3-D blend with blend_factor ≤ 1. A larger
# candidate box could overflow it (a BoundsError, or an out-of-bounds write with
# bounds checks off), so that combination is rejected.

@inline _phs_check_log_hessian(::Nothing, ::Tuple, ::Tuple) = nothing
@inline function _phs_check_log_hessian(::_PHSLogTransform, r_idx::NTuple{N, Int}, ops::Tuple) where {N}
    (sum(deriv_order, ops) == 2 && prod(2 .* r_idx .+ 1) > 27) && _throw_phs_log_hessian(r_idx)
    return nothing
end

@noinline _throw_phs_log_hessian(r_idx) = throw(
    ArgumentError(
        "second derivatives with `log_reference` support blend neighbourhoods of up to 27 " *
            "candidate nodes (e.g. blend_factor ≤ 1 in 3-D); this interpolant's blend spans " *
            "$(prod(2 .* r_idx .+ 1)). Use a smaller blend_factor"
    )
)
