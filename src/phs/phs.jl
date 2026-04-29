# ========================================
# PHS — Include Aggregator
# ========================================
#
# Load order:
#   kernels   — pure math, no deps
#   stencil   — Φ matrix construction + unique-stencil precomputation
#   types     — PHSInterpolantND + PHSLogTransform struct definitions
#   eval      — evaluation engine (base-node lookup, stencil eval, blending)
#   interp    — constructor (phs_interp) + callable overloads
#   oneshot   — one-shot public API (phs_interp with 3-arg + phs_interp!)

include("phs_kernels.jl")
include("phs_stencil.jl")
include("phs_types.jl")
include("phs_eval.jl")
include("phs_interpolant.jl")
include("phs_oneshot.jl")
