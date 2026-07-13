# ============================================================================
# _AxisAnchor — per-axis anchor backbone TYPES (core layer)
# ============================================================================
#
# The struct + virtual properties + interval-type selector live in core (before
# any method module loads) so 1D method files can define their own payloads and
# consume `_AxisAnchor` directly. The gridded-specific resolution machinery
# (`_axis_anchors_loop!` and friends) stays in gridded/axis_anchor.jl, which is
# included after all method modules.
#
# `interval::I` is the physical search cell (shared `_AbstractIndices{2}` layer,
# same as `_AnchorLoc`): ordinary axes store one index (`_ContiguousIndices{2}`,
# right tap derived), exclusive-periodic axes store both (`_ExplicitIndices{2}`,
# the seam wraps). `payload::P` is a concrete named `_AbstractAnchorPayload`
# subtype — its identity is the method/op tag, so no phantom method parameter is
# needed.
#
# This file also owns `_AbstractAnchorPayload`: the internal nominal root every
# payload occupying the `P` slot of `_AxisAnchor` subtypes. It is a marker only
# (no shared behavior/trait fallback) and stays unexported. `P` remains a
# concrete subtype at every instantiation, so the bound adds a type invariant
# without dynamic dispatch or layout change.

abstract type _AbstractAnchorPayload end

struct _AxisAnchor{I <: _AbstractIndices{2}, P <: _AbstractAnchorPayload}
    interval::I
    payload::P
end

# Virtual properties: `idxL`/`idxR` read through the interval; every other
# symbol forwards to the named payload's field (`alpha`, `inv_h`, `dL`, `h`,
# `select_right`, `state`, `inner`, …). Val-dispatch keeps each access a single
# folded `getfield`.
@inline Base.getproperty(a::_AxisAnchor, s::Symbol) = _get_axis_anchor_property(a, Val(s))
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{:interval}) = getfield(a, :interval)
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{:payload}) = getfield(a, :payload)
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{:idxL}) = getfield(a, :interval)[Val(1)]
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{:idxR}) = getfield(a, :interval)[Val(2)]
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{s}) where {s} = getproperty(getfield(a, :payload), s)

# Interval representation per axis type — reuses the `_AnchorLoc` selector's
# invariant (contiguous ordinary / explicit exclusive-periodic).
@inline _interval_type(::AbstractVector) = _ContiguousIndices{2}
@inline _interval_type(::_ExclusivePeriodicAxis) = _ExplicitIndices{2}

# Generic extrap wrapper — not method-specific: wraps ANY payload; the inner
# payload keeps every op/type detail, the wrapper only adds the OOB
# classification. Selected at anchor-build time for the flat extraps
# (Clamp/Fill) whose OOB handling needs an eval-time state branch; all other
# extraps use the bare payload and a branch-free kernel.
struct _StatefulPayload{P <: _AbstractAnchorPayload} <: _AbstractAnchorPayload
    inner::P
    state::UInt8      # IN_DOMAIN / OOB_LEFT / OOB_RIGHT
end
